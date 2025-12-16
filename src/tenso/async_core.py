import asyncio
import numpy as np
import struct
import math
from .core import loads, _REV_DTYPE_MAP, _MAGIC, _ALIGNMENT

async def aread_stream(reader: asyncio.StreamReader) -> np.ndarray:
    """
    Asynchronously read a Tenso packet from an asyncio StreamReader using zero-copy buffering.

    Designed for FastAPI/asyncio servers. Reads directly from the stream into a pre-allocated numpy buffer for efficiency.

    Args:
        reader: An asyncio.StreamReader to read from.

    Returns:
        np.ndarray: The deserialized numpy array, or None if EOF at start.

    Raises:
        asyncio.IncompleteReadError: If the stream ends unexpectedly during read.
        ValueError: If the packet is invalid or dtype is unknown.
    """
    # 1. Read Header
    try:
        header = await reader.readexactly(8)
    except asyncio.IncompleteReadError as e:
        if len(e.partial) == 0:
            return None
        raise e

    magic, ver, flags, dtype_code, ndim = struct.unpack('<4sBBBB', header)
    if magic != _MAGIC: raise ValueError("Invalid tenso packet")

    # 2. Read Shape
    shape_len = ndim * 4
    shape_bytes = await reader.readexactly(shape_len)
    shape = struct.unpack(f'<{ndim}I', shape_bytes)

    # 3. Calculate Layout
    dtype = _REV_DTYPE_MAP.get(dtype_code)
    if dtype is None: raise ValueError(f"Unknown dtype: {dtype_code}")

    current_pos = 8 + shape_len
    remainder = current_pos % _ALIGNMENT
    padding_len = 0 if remainder == 0 else (_ALIGNMENT - remainder)
    
    body_len = int(math.prod(shape) * dtype.itemsize)
    total_len = current_pos + padding_len + body_len

    # 4. Allocate Buffer (Uninitialized)
    full_buffer = np.empty(total_len, dtype=np.uint8)
    
    # Fill Header/Shape
    full_buffer[0:8] = list(header)
    full_buffer[8:8+shape_len] = list(shape_bytes)
    
    # 5. Read Body (Consume padding + Read Data)
    if padding_len > 0:
        await reader.readexactly(padding_len)

    body_data = await reader.readexactly(body_len)
    
    body_start = current_pos + padding_len
    full_buffer[body_start:body_start+body_len] = np.frombuffer(body_data, dtype=np.uint8)

    return loads(full_buffer)