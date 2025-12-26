"""
Async I/O Support for Tenso.
"""

import asyncio
import struct
from typing import Optional, Union

import numpy as np
import xxhash

from .config import MAX_ELEMENTS, MAX_NDIM
from .core import _ALIGNMENT, _REV_DTYPE_MAP, FLAG_INTEGRITY, iter_dumps


async def aread_stream(reader: asyncio.StreamReader) -> Optional[np.ndarray]:
    """Asynchronously read a Tenso packet from a StreamReader.

    Parameters
    ----------
    reader : asyncio.StreamReader
        The stream reader to read from.

    Returns
    -------
    Optional[np.ndarray]
        The deserialized tensor, or None if stream is empty.
    """
    try:
        header = await reader.readexactly(8)
    except asyncio.IncompleteReadError as e:
        if len(e.partial) == 0:
            return None
        raise
    _, _, flags, dtype_code, ndim = struct.unpack("<4sBBBB", header)
    if ndim > MAX_NDIM:
        raise ValueError(f"Packet exceeds maximum dimensions ({ndim})")

    shape_bytes = await reader.readexactly(ndim * 4)
    shape = struct.unpack(f"<{ndim}I", shape_bytes)
    if int(np.prod(shape)) > MAX_ELEMENTS:
        raise ValueError(f"Packet exceeds maximum elements ({int(np.prod(shape))})")

    pad_len = (_ALIGNMENT - ((8 + (ndim * 4)) % _ALIGNMENT)) % _ALIGNMENT
    if pad_len > 0:
        await reader.readexactly(pad_len)

    dtype = _REV_DTYPE_MAP.get(dtype_code)
    body_data = await reader.readexactly(int(np.prod(shape)) * dtype.itemsize)

    if flags & FLAG_INTEGRITY:
        footer = await reader.readexactly(8)
        if xxhash.xxh3_64_intdigest(body_data) != struct.unpack("<Q", footer)[0]:
            raise ValueError("Integrity check failed: XXH3 mismatch")

    arr = np.frombuffer(body_data, dtype=dtype).reshape(shape)
    arr.flags.writeable = False
    return arr


async def awrite_stream(
    tensor: np.ndarray,
    writer: asyncio.StreamWriter,
    strict: bool = False,
    check_integrity: bool = False,
) -> None:
    """Asynchronously write a tensor to a StreamWriter.

    Parameters
    ----------
    tensor : np.ndarray
        The tensor to serialize.
    writer : asyncio.StreamWriter
        The stream writer to write to.
    strict : bool, optional
        Whether to use strict mode. Default is False.
    check_integrity : bool, optional
        Whether to check integrity. Default is False.

    Returns
    -------
    None
    """
    for chunk in iter_dumps(tensor, strict=strict, check_integrity=check_integrity):
        writer.write(chunk)
        await writer.drain()
