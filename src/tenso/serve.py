"""
tenso.serve: Quick-start utilities for spinning up Tenso-optimized servers.

Provides helpers that wrap FastAPI/gRPC with sensible defaults for
high-throughput tensor serving: correct content types, worker pool sizing
based on CPU/GPU resources, and built-in Tenso request/response handling.

Example::

    from tenso.serve import create_app, run

    app = create_app()

    @app.tenso_endpoint("/infer")
    def infer(tensor):
        return tensor * 2.0

    run(app, workers=4)
"""

import asyncio
import inspect
import logging
import os
from typing import Any, Callable, Optional

logger = logging.getLogger("tenso.serve")


def _detect_workers() -> int:
    """Determine optimal worker count based on available resources."""
    cpu_count = os.cpu_count() or 1
    # For tensor serving, use fewer workers than CPUs to leave room
    # for numpy/torch compute threads within each worker.
    return max(1, min(cpu_count // 2, 8))


class TensoApp:
    """
    A lightweight wrapper around FastAPI pre-configured for Tenso serving.

    Automatically handles Tenso binary request parsing and response
    serialization for registered endpoints.
    """

    def __init__(
        self,
        title: str = "Tenso Server",
        check_integrity: bool = False,
        compress_response: bool = False,
    ):
        from fastapi import FastAPI
        self._app = FastAPI(title=title)
        self._check_integrity = check_integrity
        self._compress_response = compress_response
        self._endpoints = []

    @property
    def app(self):
        """The underlying FastAPI application instance."""
        return self._app

    def tenso_endpoint(
        self,
        path: str,
        method: str = "POST",
        check_integrity: Optional[bool] = None,
    ) -> Callable:
        """
        Decorator to register a function as a Tenso-powered endpoint.

        The decorated function receives a deserialized tensor/bundle
        and should return a numpy array or dict. The return value is
        automatically serialized as a TensoResponse.

        Parameters
        ----------
        path : str
            The URL path for the endpoint.
        method : str
            HTTP method (default: POST).
        check_integrity : bool, optional
            Override per-endpoint integrity checking.
        """
        integrity = check_integrity if check_integrity is not None else self._check_integrity

        def decorator(func: Callable) -> Callable:
            from fastapi import Request
            from .fastapi import TensoResponse, get_tenso_data

            is_async = asyncio.iscoroutinefunction(func)

            async def endpoint_handler(request: Request) -> TensoResponse:
                tensor = await get_tenso_data(request)
                if is_async:
                    result = await func(tensor)
                else:
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(None, func, tensor)
                return TensoResponse(
                    result,
                    check_integrity=integrity,
                )

            self._app.add_api_route(
                path,
                endpoint_handler,
                methods=[method.upper()],
            )
            self._endpoints.append(path)
            return func

        return decorator

    def add_health_check(self, path: str = "/health"):
        """Add a simple health check endpoint."""
        @self._app.get(path)
        def health():
            return {"status": "ok", "endpoints": self._endpoints}


def create_app(
    title: str = "Tenso Server",
    check_integrity: bool = False,
    health_check: bool = True,
) -> TensoApp:
    """
    Create a new TensoApp with sensible defaults.

    Parameters
    ----------
    title : str
        Server title.
    check_integrity : bool
        Enable XXH3 integrity checking on all endpoints.
    health_check : bool
        Add a /health endpoint.

    Returns
    -------
    TensoApp
        The configured application.
    """
    app = TensoApp(title=title, check_integrity=check_integrity)
    if health_check:
        app.add_health_check()
    return app


def run(
    app: TensoApp,
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: Optional[int] = None,
    log_level: str = "info",
):
    """
    Run a TensoApp using uvicorn with optimized settings.

    Parameters
    ----------
    app : TensoApp
        The TensoApp to serve.
    host : str
        Bind address.
    port : int
        Bind port.
    workers : int, optional
        Number of worker processes. Auto-detected if None.
    log_level : str
        Uvicorn log level.
    """
    import uvicorn

    if workers is None:
        workers = _detect_workers()

    uvicorn.run(
        app.app,
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        limit_concurrency=1000,
        timeout_keep_alive=30,
    )


def run_grpc(
    servicer_class: Any,
    port: int = 50051,
    max_workers: Optional[int] = None,
    max_message_length: int = 256 * 1024 * 1024,
):
    """
    Run a gRPC server with Tenso-optimized settings.

    Parameters
    ----------
    servicer_class : class
        A TensorInferenceServicer subclass instance.
    port : int
        gRPC listen port.
    max_workers : int, optional
        Thread pool size. Auto-detected if None.
    max_message_length : int
        Maximum message size (default 256MB for large tensors).
    """
    import grpc
    from concurrent import futures
    from .grpc import tenso_msg_pb2_grpc

    if max_workers is None:
        max_workers = _detect_workers() * 2

    options = [
        ("grpc.max_send_message_length", max_message_length),
        ("grpc.max_receive_message_length", max_message_length),
    ]

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=options,
    )
    tenso_msg_pb2_grpc.add_TensorInferenceServicer_to_server(
        servicer_class, server
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info("Tenso gRPC server started on port %d with %d workers", port, max_workers)
    server.wait_for_termination()
