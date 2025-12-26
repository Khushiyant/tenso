"""
FastAPI Integration for Tenso.
"""

from typing import Any

import numpy as np
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse

from .core import iter_dumps, loads


class TensoResponse(StreamingResponse):
    """FastAPI Response class for high-performance tensor streaming."""

    def __init__(
        self,
        tensor: np.ndarray,
        filename: str = None,
        strict: bool = False,
        check_integrity: bool = False,
        **kwargs,
    ):
        """
        Initialize the TensoResponse.

        Parameters
        ----------
        tensor : np.ndarray
            The tensor to stream.
        filename : str, optional
            Filename for the response.
        strict : bool, default False
            Whether to enforce strict C-contiguous arrays.
        check_integrity : bool, default False
            Whether to include integrity check.
        **kwargs
            Additional arguments passed to StreamingResponse.
        """
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
    """Dependency to extract a Tenso tensor from an incoming FastAPI Request.

    Parameters
    ----------
    request : Request
        The FastAPI request object.

    Returns
    -------
    Any
        The deserialized tensor.

    Raises
    ------
    HTTPException
        If content type is wrong or invalid packet.
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
