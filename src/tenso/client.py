"""
High-level HTTP client for Tenso-powered FastAPI endpoints.

Provides TensoFastAPIClient that natively streams and unpacks
TensoResponse chunks, handling all protocol details under the hood.

Requires ``httpx`` (``pip install httpx``).
"""

from typing import Any, Dict, Optional, Union

import numpy as np

from .core import dumps, loads


class TensoFastAPIClient:
    """
    Client for communicating with FastAPI endpoints that use TensoResponse
    and get_tenso_data.

    Example::

        client = TensoFastAPIClient("http://localhost:8000")
        result = client.predict("/infer", np.random.randn(1, 224, 224, 3).astype(np.float32))
        print(result.shape)

        # Async usage
        result = await client.apredict("/infer", tensor)
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        check_integrity: bool = False,
        headers: Optional[Dict[str, str]] = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._check_integrity = check_integrity
        self._extra_headers = headers or {}
        self._sync_client = None
        self._async_client = None

    def _get_sync_client(self):
        if self._sync_client is None:
            import httpx
            self._sync_client = httpx.Client(
                base_url=self._base_url,
                timeout=self._timeout,
                headers=self._extra_headers,
            )
        return self._sync_client

    def _get_async_client(self):
        if self._async_client is None:
            import httpx
            self._async_client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                headers=self._extra_headers,
            )
        return self._async_client

    def _make_headers(self) -> dict:
        return {"content-type": "application/octet-stream"}

    def predict(
        self,
        endpoint: str,
        tensor: Union[np.ndarray, dict],
        compress: bool = False,
    ) -> Any:
        """
        Send a tensor to a Tenso-powered endpoint and return the deserialized response.

        Parameters
        ----------
        endpoint : str
            The API path (e.g. "/infer").
        tensor : np.ndarray or dict
            The input tensor or bundle.
        compress : bool
            Whether to LZ4-compress the request body.

        Returns
        -------
        Any
            The deserialized response (np.ndarray, dict, or sparse matrix).
        """
        # dumps() returns a memoryview; httpx's `content=` accepts any
        # bytes-like object, so pass the view directly to avoid an extra copy.
        packet = dumps(
            tensor,
            check_integrity=self._check_integrity,
            compress=compress,
        )
        client = self._get_sync_client()
        response = client.post(
            endpoint,
            content=packet,
            headers=self._make_headers(),
        )
        response.raise_for_status()
        return loads(response.content)

    async def apredict(
        self,
        endpoint: str,
        tensor: Union[np.ndarray, dict],
        compress: bool = False,
    ) -> Any:
        """Async version of predict()."""
        # dumps() returns a memoryview; httpx accepts bytes-like content
        # directly, avoiding an extra copy on the request side.
        packet = dumps(
            tensor,
            check_integrity=self._check_integrity,
            compress=compress,
        )
        client = self._get_async_client()
        response = await client.post(
            endpoint,
            content=packet,
            headers=self._make_headers(),
        )
        response.raise_for_status()
        return loads(response.content)

    def stream_predict(
        self,
        endpoint: str,
        tensor: Union[np.ndarray, dict],
    ) -> Any:
        """
        Send a tensor and stream the response, reassembling chunks
        into the final deserialized object.
        """
        # dumps() returns a memoryview; httpx accepts it as request content
        # without an extra copy.
        packet = dumps(tensor, check_integrity=self._check_integrity)
        client = self._get_sync_client()

        chunks = []
        with client.stream(
            "POST", endpoint, content=packet, headers=self._make_headers()
        ) as response:
            response.raise_for_status()
            for chunk in response.iter_bytes():
                chunks.append(chunk)

        # loads() needs a single contiguous buffer, so the streamed chunks must
        # be concatenated here; this join is required, not an avoidable copy.
        return loads(b"".join(chunks))

    def close(self):
        """Close sync client. For async client, use ``async with`` or ``await client.aclose()``."""
        if self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None

    async def aclose(self):
        """Close both sync and async clients."""
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()
