import asyncio
import numpy as np
import struct
import xxhash
from typing import Union
from .core import _REV_DTYPE_MAP, _ALIGNMENT, FLAG_INTEGRITY, iter_dumps
from .config import MAX_NDIM, MAX_ELEMENTS

async def aread_stream(reader: asyncio.StreamReader) -> Union[np.ndarray, None]:
    """
    Asynchronously read a Tenso packet from an asyncio StreamReader.
    
    This coroutine reads a complete Tenso packet from an asyncio stream
    and deserializes it into a numpy array. Includes DoS protection and
    integrity checking if the packet includes it.
    
    Args:
        reader: An asyncio.StreamReader instance to read the packet from.
    
    Returns:
        np.ndarray or None: The deserialized tensor with writeable=False,
                           or None if the stream ended before reading any data.
    
    Raises:
        ValueError: If the packet is invalid, exceeds security limits, or fails
                   integrity checks.
        asyncio.IncompleteReadError: If the stream ends prematurely.
    """
    try:
        header = await reader.readexactly(8)
    except asyncio.IncompleteReadError as e:
        if len(e.partial) == 0: return None
        raise

    magic, ver, flags, dtype_code, ndim = struct.unpack('<4sBBBB', header)
    
    # [SECURITY] DoS Protection
    if ndim > MAX_NDIM:
        raise ValueError(f"Packet exceeds maximum dimensions ({ndim})")
    
    shape_bytes = await reader.readexactly(ndim * 4)
    shape = struct.unpack(f'<{ndim}I', shape_bytes)
    
    # [SECURITY] DoS Protection
    num_elements = int(np.prod(shape))
    if num_elements > MAX_ELEMENTS:
        raise ValueError(f"Packet exceeds maximum elements ({num_elements})")
    
    current_pos = 8 + (ndim * 4)
    remainder = current_pos % _ALIGNMENT
    padding_len = 0 if remainder == 0 else (_ALIGNMENT - remainder)
    if padding_len > 0: await reader.readexactly(padding_len)

    dtype = _REV_DTYPE_MAP.get(dtype_code)
    if dtype is None: raise ValueError(f"Unsupported dtype code: {dtype_code}")

    body_len = num_elements * dtype.itemsize
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

async def awrite_stream(tensor: np.ndarray, writer: asyncio.StreamWriter, strict: bool = False, check_integrity: bool = False) -> None:
    """
    Asynchronously write a tensor to an asyncio StreamWriter.
    
    Serializes a tensor and writes it to an asyncio stream using vectored I/O.
    Uses iter_dumps internally to avoid large memory copies and yields control
    to the event loop between chunks to prevent blocking.
    
    Args:
        tensor: The numpy array to serialize.
        writer: An asyncio.StreamWriter instance to write the packet to.
        strict: If True, raises error for non-contiguous arrays. Default False.
        check_integrity: If True, includes integrity hash. Default False.
    
    Raises:
        ValueError: If serialization fails.
        OSError: If writing to the stream fails.
    """
    for chunk in iter_dumps(tensor, strict=strict, check_integrity=check_integrity):
        writer.write(chunk)
        # Yield control to event loop to allow concurrent processing
        await writer.drain()