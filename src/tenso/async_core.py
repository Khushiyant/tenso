import asyncio
import numpy as np
import struct
import xxhash
from typing import Union
from .core import _REV_DTYPE_MAP, _ALIGNMENT, FLAG_INTEGRITY

async def aread_stream(reader: asyncio.StreamReader) -> Union[np.ndarray, None]:
    """
    Asynchronously read a Tenso packet from an asyncio StreamReader.

    This function reads a complete Tenso packet from an asynchronous stream,
    verifies its integrity if enabled, and returns the deserialized numpy array.

    Args:
        reader: An asyncio.StreamReader object to read the packet from.

    Returns:
        np.ndarray or None: The deserialized numpy array, or None if the stream
        is at EOF before any data is read.

    Raises:
        asyncio.IncompleteReadError: If the stream ends unexpectedly during reading.
        ValueError: If the packet is invalid or integrity check fails.
    """
    try:
        header = await reader.readexactly(8)
    except asyncio.IncompleteReadError as e:
        if len(e.partial) == 0: return None
        raise

    magic, ver, flags, dtype_code, ndim = struct.unpack('<4sBBBB', header)
    
    shape_bytes = await reader.readexactly(ndim * 4)
    shape = struct.unpack(f'<{ndim}I', shape_bytes)
    
    current_pos = 8 + (ndim * 4)
    remainder = current_pos % _ALIGNMENT
    padding_len = 0 if remainder == 0 else (_ALIGNMENT - remainder)
    if padding_len > 0: await reader.readexactly(padding_len)

    dtype = _REV_DTYPE_MAP[dtype_code]
    body_len = int(np.prod(shape) * dtype.itemsize)
    body_data = await reader.readexactly(body_len)
    
    if flags & FLAG_INTEGRITY:
        footer_bytes = await reader.readexactly(8)
        expected_hash = struct.unpack('<Q', footer_bytes)[0]
        actual_hash = xxhash.xxh3_64_intdigest(body_data)
        if actual_hash != expected_hash:
            raise ValueError("Integrity check failed: XXH3 mismatch")

    arr = np.frombuffer(body_data, dtype=dtype).reshape(shape)
    arr.flags.writeable = False 
    return arr