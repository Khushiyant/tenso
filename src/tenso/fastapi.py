"""
FastAPI Integration for Tenso.

Allows zero-copy streaming of tensors from API endpoints and
high-performance ingestion of incoming Tenso packets.
"""

from fastapi import Request, HTTPException
from fastapi.responses import StreamingResponse
import numpy as np
from typing import Any
from .core import iter_dumps, loads


class TensoResponse(StreamingResponse):
    """
    FastAPI Response for zero-copy tensor streaming.

    Parameters
    ----------
    tensor : np.ndarray
        The tensor to stream.
    filename : str, optional
        Filename for Content-Disposition header.
    strict : bool, default False
        Strict contiguous check.
    check_integrity : bool, default False
        Include checksum.
    **kwargs
        Passed to StreamingResponse.
    """

    def __init__(
        self,
        tensor: np.ndarray,
        filename: str = None,
        strict: bool = False,
        check_integrity: bool = False,
        **kwargs,
    ):
        stream = iter_dumps(tensor, strict=strict, check_integrity=check_integrity)
        super().__init__(stream, media_type="application/octet-stream", **kwargs)
        if not hasattr(self, "background"):
            self.background = kwargs.get("background")
        self.headers["X-Tenso-Version"] = "2"
        self.headers["X-Tenso-Shape"] = str(tensor.shape)
        self.headers["X-Tenso-Dtype"] = str(tensor.dtype)
        if filename:
            self.headers["Content-Disposition"] = f'attachment; filename="{filename}"'


async def get_tenso_data(request: Request) -> Any:
    """
    Dependency to extract a Tenso object from an incoming FastAPI Request.

    Parameters
    ----------
    request : Request
        The FastAPI request object.

    Returns
    -------
    Any
        The deserialized array, bundle, or sparse matrix.

    Raises
    ------
    HTTPException
        If the payload is invalid or headers are missing.
    """
    if request.headers.get("content-type") != "application/octet-stream":
        raise HTTPException(
            status_code=400, detail="Expected application/octet-stream content type."
        )
    body = await request.body()
    try:
        return loads(body)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid Tenso packet: {str(e)}")
