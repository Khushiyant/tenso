"""
Async I/O Support for Tenso.

Provides coroutines for reading and writing Tenso packets using
asyncio stream readers and writers.
"""

import asyncio
import numpy as np
import struct
import xxhash
from typing import Optional
from .core import _REV_DTYPE_MAP, _ALIGNMENT, FLAG_INTEGRITY, iter_dumps
from .config import MAX_NDIM, MAX_ELEMENTS


async def aread_stream(reader: asyncio.StreamReader) -> Optional[np.ndarray]:
    """
    Asynchronously read a Tenso packet from a StreamReader.

    Parameters
    ----------
    reader : asyncio.StreamReader
        The stream reader source.

    Returns
    -------
    Optional[np.ndarray]
        The deserialized array.
    """
    try:
        header = await reader.readexactly(8)
    except asyncio.IncompleteReadError as e:
        if len(e.partial) == 0:
            return None
        raise
    magic, _, flags, dtype_code, ndim = struct.unpack("<4sBBBB", header)
    if ndim > MAX_NDIM:
        raise ValueError(f"Packet exceeds maximum dimensions ({ndim} > {MAX_NDIM})")

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
    """
    Asynchronously write a tensor to a StreamWriter.

    Parameters
    ----------
    tensor : np.ndarray
        The array to write.
    writer : asyncio.StreamWriter
        The stream writer destination.
    strict : bool, default False
        Strict contiguous check.
    check_integrity : bool, default False
        Include checksum.
    """
    for chunk in iter_dumps(tensor, strict=strict, check_integrity=check_integrity):
        writer.write(chunk)
        await writer.drain()
