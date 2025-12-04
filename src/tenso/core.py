import struct
import numpy as np
from typing import BinaryIO
import math
from .config import _MAGIC, _VERSION, _ALIGNMENT, _DTYPE_MAP, _REV_DTYPE_MAP

def dumps(tensor: np.ndarray) -> bytes:
    """
    Serialize a numpy array into bytes with 64-byte alignment.
    
    Args:
        tensor: NumPy array to serialize
        
    Returns:
        Serialized bytes in Tenso format
        
    Raises:
        ValueError: If dtype is unsupported
    """
    # 1. Validation & Preparation
    if tensor.dtype not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {tensor.dtype}")
    
    # Force C-Contiguous layout
    if not tensor.flags['C_CONTIGUOUS']:
        tensor = np.ascontiguousarray(tensor)
    
    dtype_code = _DTYPE_MAP[tensor.dtype]
    shape = tensor.shape
    ndim = len(shape)
    
    # Validate ndim fits in uint8
    if ndim > 255:
        raise ValueError(f"Too many dimensions: {ndim} (max 255)")
    
    # 2. Calculate Sizes for Alignment
    header_size = 8
    shape_size = ndim * 4
    current_offset = header_size + shape_size
    
    # Calculate padding needed to reach next 64-byte boundary
    remainder = current_offset % _ALIGNMENT
    padding_size = 0 if remainder == 0 else (_ALIGNMENT - remainder)
    
    # 3. Construct Parts
    # Flags=1 indicates "Aligned 64" logic is active
    header = struct.pack('<4sBBBB', _MAGIC, _VERSION, 1, dtype_code, ndim)
    shape_block = struct.pack(f'<{ndim}I', *shape)
    padding = b'\x00' * padding_size
    
    # 4. Assemble packet
    return header + shape_block + padding + tensor.tobytes()


def loads(data: bytes, copy: bool = False) -> np.ndarray:
    """
    Deserialize bytes back into a numpy array.
    
    Args:
        data: The bytes object containing the Tenso packet
        copy: If True, forces a memory copy (safer for temporary buffers).
              If False (default), returns a zero-copy view (fastest)
              
    Returns:
        Deserialized NumPy array
        
    Raises:
        ValueError: If packet is invalid or corrupted
    """
    # 1. Basic Size Check
    if len(data) < 8:
        raise ValueError("Packet too short to contain header")
    
    # 2. Parse Header
    magic, ver, flags, dtype_code, ndim = struct.unpack('<4sBBBB', data[:8])
    
    if magic != _MAGIC:
        raise ValueError("Invalid tenso packet (magic bytes mismatch)")
    
    # Version Compatibility Check
    if ver > _VERSION:
        raise ValueError(f"Unsupported version: {ver} (library supports v{_VERSION})")
    
    # Validate dtype code
    if dtype_code not in _REV_DTYPE_MAP:
        raise ValueError(f"Unknown dtype code: {dtype_code}")
    
    # 3. Parse Shape
    shape_start = 8
    shape_end = 8 + (ndim * 4)
    
    if len(data) < shape_end:
        raise ValueError("Packet too short to contain shape data")
    
    shape = struct.unpack(f'<{ndim}I', data[shape_start:shape_end])
    
    # 4. Handle Alignment / Padding
    body_start = shape_end
    if ver >= 2 and flags & 1:  # Check alignment flag
        remainder = shape_end % _ALIGNMENT
        padding_size = 0 if remainder == 0 else (_ALIGNMENT - remainder)
        body_start += padding_size
    
    # 5. Validate Body Size
    dtype = _REV_DTYPE_MAP[dtype_code]

    total_elements = math.prod(shape) 
    expected_body_size = total_elements * dtype.itemsize
    
    if len(data) < body_start + expected_body_size:
        raise ValueError(
            f"Packet too short (expected {body_start + expected_body_size} bytes, "
            f"got {len(data)})"
        )
    
    # 6. Create Array (Zero-Copy)
    arr = np.frombuffer(
        data,
        dtype=dtype,
        offset=body_start,
        count=int(np.prod(shape))
    )
    arr = arr.reshape(shape)
    
    # 7. Handle Safety Copy
    if copy:
        return arr.copy()
    
    # Make array read-only to prevent accidental modification of buffer
    arr.flags.writeable = False
    return arr


def dump(tensor: np.ndarray, fp: BinaryIO) -> None:
    """
    Serialize a numpy array to a file-like object.
    
    Args:
        tensor: NumPy array to serialize
        fp: File-like object opened in binary write mode
    """
    fp.write(dumps(tensor))


def load(fp: BinaryIO, copy: bool = False) -> np.ndarray:
    """
    Deserialize a numpy array from a file-like object.
    
    Args:
        fp: File-like object opened in binary read mode
        copy: If True, forces a memory copy
        
    Returns:
        Deserialized NumPy array
    """
    return loads(fp.read(), copy=copy)


