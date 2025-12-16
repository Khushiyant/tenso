from .config import _MAGIC, _ALIGNMENT, _DTYPE_MAP, _REV_DTYPE_MAP
import numpy as np
import struct
import ctypes

# Utility functions
def is_aligned(data: bytes, alignment: int = 64) -> bool:
    """
    Check if a data buffer is aligned to a specified byte boundary.

    Args:
        data: The data buffer (bytes-like object).
        alignment: The alignment boundary in bytes (default: 64).

    Returns:
        bool: True if the buffer is aligned, False otherwise.
    """
    return (ctypes.addressof(ctypes.c_char.from_buffer(bytearray(data))) % alignment) == 0


def get_packet_info(data: bytes) -> dict:
    """
    Extract metadata from a Tenso packet without full deserialization.

    Args:
        data: The Tenso packet bytes.

    Returns:
        dict: Dictionary with packet metadata, including version, dtype, shape, ndim, flags, alignment, total elements, and data size in bytes.

    Raises:
        ValueError: If the packet is too short or invalid.
    """
    if len(data) < 8:
        raise ValueError("Packet too short")
    
    magic, ver, flags, dtype_code, ndim = struct.unpack('<4sBBBB', data[:8])
    
    if magic != _MAGIC:
        raise ValueError("Invalid tenso packet")
    
    shape_end = 8 + (ndim * 4)
    if len(data) < shape_end:
        raise ValueError("Packet too short to contain shape")
    
    shape = struct.unpack(f'<{ndim}I', data[8:shape_end])
    dtype = _REV_DTYPE_MAP.get(dtype_code, None)
    
    return {
        'version': ver,
        'dtype': dtype,
        'shape': shape,
        'ndim': ndim,
        'flags': flags,
        'aligned': bool(flags & 1),
        'total_elements': int(np.prod(shape)),
        'data_size_bytes': int(np.prod(shape)) * (dtype.itemsize if dtype else 0)
    }