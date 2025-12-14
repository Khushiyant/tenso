import struct
import numpy as np
from typing import BinaryIO, Union, Any
import math
import mmap
from .config import _MAGIC, _VERSION, _ALIGNMENT, _DTYPE_MAP, _REV_DTYPE_MAP, IS_LITTLE_ENDIAN

# --- Stream Helper ---
def _read_exact(source: Any, n: int) -> Union[bytes, None]:
    """Helper to read exactly n bytes from a socket or file-like object."""
    if n == 0:
        return b''
        
    data = b''
    while len(data) < n:
        # Support both socket.recv and file.read
        if hasattr(source, 'recv'):
            chunk = source.recv(n - len(data))
        else:
            chunk = source.read(n - len(data))
            
        if not chunk:
            # If we read nothing at the very start, it's a clean EOF (if expected)
            if len(data) == 0:
                return None
            # If we read partial data then EOF, it's an error
            raise EOFError(f"Stream closed unexpectedly. Expected {n} bytes, got {len(data)}")
        data += chunk
    return data

def read_stream(source: Any) -> Union[np.ndarray, None]:
    """
    Reads a complete Tenso packet directly from a socket or file stream.
    
    Args:
        source: A socket object or file-like object (must support recv or read).
        
    Returns:
        numpy.ndarray or None if stream is closed (EOF).
    """
    # 1. Read Fixed Header
    header = _read_exact(source, 8)
    if header is None:
        return None
        
    magic, ver, flags, dtype_code, ndim = struct.unpack('<4sBBBB', header)
    
    if magic != _MAGIC:
        raise ValueError("Invalid tenso packet (magic bytes mismatch)")

    # 2. Read Shape Block
    shape_len = ndim * 4
    shape_bytes = _read_exact(source, shape_len)
    if shape_bytes is None: raise EOFError("Stream ended during shape read")
    
    shape = struct.unpack(f'<{ndim}I', shape_bytes)
    
    # 3. Calculate Padding
    current_offset = 8 + shape_len
    remainder = current_offset % _ALIGNMENT
    padding_len = 0 if remainder == 0 else (_ALIGNMENT - remainder)
    
    # 4. Read Padding
    padding = _read_exact(source, padding_len)
    if padding is None and padding_len > 0: raise EOFError("Stream ended during padding read")
    if padding is None: padding = b''

    # 5. Calculate Body Size
    dtype = _REV_DTYPE_MAP.get(dtype_code)
    if dtype is None:
        raise ValueError(f"Unknown dtype code: {dtype_code}")
        
    total_elements = math.prod(shape)
    body_len = int(total_elements * dtype.itemsize)
    
    # 6. Read Body
    body = _read_exact(source, body_len)
    if body is None and body_len > 0: raise EOFError("Stream ended during body read")
    if body is None: body = b''

    # 7. Reconstruct
    # We combine parts to use the robust loads() logic for final object creation
    return loads(header + shape_bytes + padding + body)


# --- Core Functions ---

def dumps(tensor: np.ndarray, strict: bool = False) -> bytes:
    """
    Serialize a numpy array into bytes with 64-byte alignment.
    """
    # 1. Validation & Preparation
    if tensor.dtype not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {tensor.dtype}")
    
    # 2. Handle Memory Layout
    if not tensor.flags['C_CONTIGUOUS']:
        if strict:
            raise ValueError("Tensor is not C-Contiguous and strict=True. "
                             "Reshape or copy array before serializing.")
        tensor = np.ascontiguousarray(tensor)

    # 3. Handle Endianness (Optimized)
    if not IS_LITTLE_ENDIAN or tensor.dtype.byteorder == '>':
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
    
    # Safe Creation
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
    Serialize a numpy array to a file-like object using Coalesced Writes.
    """
    # 1. Validation (Same as dumps)
    if tensor.dtype not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {tensor.dtype}")
    
    if not tensor.flags['C_CONTIGUOUS']:
        if strict:
            raise ValueError("Tensor is not C-Contiguous and strict=True.")
        tensor = np.ascontiguousarray(tensor)

    # Endianness
    if not IS_LITTLE_ENDIAN or tensor.dtype.byteorder == '>':
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
    
    # Optimization: Coalesce metadata writes
    # Reduces syscalls from 4 to 2
    header = struct.pack('<4sBBBB', _MAGIC, _VERSION, 1, dtype_code, ndim)
    shape_block = struct.pack(f'<{ndim}I', *shape)
    padding = b'\x00' * padding_size
    
    fp.write(header + shape_block + padding)
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