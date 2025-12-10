import struct
import numpy as np
from typing import BinaryIO, Union
import math
import mmap
import sys
from .config import _MAGIC, _VERSION, _ALIGNMENT, _DTYPE_MAP, _REV_DTYPE_MAP

def dumps(tensor: np.ndarray, strict: bool = False) -> bytes:
    """
    Serialize a numpy array into bytes with 64-byte alignment.
    """
    # 1. Validation & Preparation
    if tensor.dtype not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {tensor.dtype}")
    
    # 2. Handle Memory Layout (Strict Mode)
    if not tensor.flags['C_CONTIGUOUS']:
        if strict:
            raise ValueError("Tensor is not C-Contiguous and strict=True. "
                             "Reshape or copy array before serializing.")
        tensor = np.ascontiguousarray(tensor)

    # 3. Handle Endianness (Portable Safety)
    if sys.byteorder == 'big' or tensor.dtype.byteorder == '>':
        tensor = tensor.astype(tensor.dtype.newbyteorder('<'))

    dtype_code = _DTYPE_MAP[tensor.dtype]
    shape = tensor.shape
    ndim = len(shape)
    
    if ndim > 255:
        raise ValueError(f"Too many dimensions: {ndim} (max 255)")
    
    # 4. Calculate Sizes for Alignment
    header_size = 8
    shape_size = ndim * 4
    current_offset = header_size + shape_size
    
    remainder = current_offset % _ALIGNMENT
    padding_size = 0 if remainder == 0 else (_ALIGNMENT - remainder)
    
    # 5. Construct Parts
    header = struct.pack('<4sBBBB', _MAGIC, _VERSION, 1, dtype_code, ndim)
    shape_block = struct.pack(f'<{ndim}I', *shape)
    padding = b'\x00' * padding_size
    
    # 6. Assemble packet
    return header + shape_block + padding + tensor.tobytes()


def loads(data: Union[bytes, mmap.mmap], copy: bool = False) -> np.ndarray:
    """
    Deserialize bytes back into a numpy array.
    """
    if len(data) < 8:
        raise ValueError("Packet too short to contain header")
    
    magic, ver, flags, dtype_code, ndim = struct.unpack('<4sBBBB', data[:8])
    
    if magic != _MAGIC:
        raise ValueError("Invalid tenso packet (magic bytes mismatch)")
    
    if ver > _VERSION:
        raise ValueError(f"Unsupported version: {ver} (library supports v{_VERSION})")
    
    if dtype_code not in _REV_DTYPE_MAP:
        raise ValueError(f"Unknown dtype code: {dtype_code}")
    
    shape_start = 8
    shape_end = 8 + (ndim * 4)
    
    if len(data) < shape_end:
        raise ValueError("Packet too short to contain shape data")
    
    shape = struct.unpack(f'<{ndim}I', data[shape_start:shape_end])
    
    body_start = shape_end
    if ver >= 2 and flags & 1:  # Check alignment flag
        remainder = shape_end % _ALIGNMENT
        padding_size = 0 if remainder == 0 else (_ALIGNMENT - remainder)
        body_start += padding_size
    
    dtype = _REV_DTYPE_MAP[dtype_code]

    total_elements = math.prod(shape) 
    expected_body_size = total_elements * dtype.itemsize
    
    if len(data) < body_start + expected_body_size:
        raise ValueError(
            f"Packet too short (expected {body_start + expected_body_size} bytes, "
            f"got {len(data)})"
        )
    
    arr = np.frombuffer(
        data,
        dtype=dtype,
        offset=body_start,
        count=int(np.prod(shape))
    )
    arr = arr.reshape(shape)
    
    if copy:
        return arr.copy()
    
    arr.flags.writeable = False
    return arr


def dump(tensor: np.ndarray, fp: BinaryIO, strict: bool = False) -> None:
    """
    Serialize a numpy array to a file-like object using Streaming Write.
    Avoids creating a massive bytes object in RAM.
    """
    # 1. Validation (Same as dumps)
    if tensor.dtype not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {tensor.dtype}")
    
    if not tensor.flags['C_CONTIGUOUS']:
        if strict:
            raise ValueError("Tensor is not C-Contiguous and strict=True.")
        tensor = np.ascontiguousarray(tensor)

    # Endianness
    if sys.byteorder == 'big' or tensor.dtype.byteorder == '>':
        tensor = tensor.astype(tensor.dtype.newbyteorder('<'))

    dtype_code = _DTYPE_MAP[tensor.dtype]
    shape = tensor.shape
    ndim = len(shape)
    
    if ndim > 255:
        raise ValueError(f"Too many dimensions: {ndim} (max 255)")
    
    header_size = 8
    shape_size = ndim * 4
    current_offset = header_size + shape_size
    
    remainder = current_offset % _ALIGNMENT
    padding_size = 0 if remainder == 0 else (_ALIGNMENT - remainder)
    
    # Write Parts directly to file stream
    header = struct.pack('<4sBBBB', _MAGIC, _VERSION, 1, dtype_code, ndim)
    shape_block = struct.pack(f'<{ndim}I', *shape)
    padding = b'\x00' * padding_size
    
    fp.write(header)
    fp.write(shape_block)
    fp.write(padding)
    
    # Optimization: Write directly from memoryview, avoiding copies
    fp.write(tensor.data)


def load(fp: BinaryIO, mmap_mode: bool = False, copy: bool = False) -> np.ndarray:
    """
    Deserialize a numpy array from a file-like object.
    """
    if mmap_mode:
        mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
        return loads(mm, copy=copy)
    else:
        return loads(fp.read(), copy=copy)