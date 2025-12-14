import struct
import numpy as np
from typing import BinaryIO, Union, Any
import math
import mmap
import sys
import os
from .config import _MAGIC, _VERSION, _ALIGNMENT, _DTYPE_MAP, _REV_DTYPE_MAP

IS_LITTLE_ENDIAN = (sys.byteorder == 'little')

# --- Stream Helper (Read) ---

def _read_exact(source: Any, n: int) -> Union[bytes, None]:
    """Optimized reader: Allocates memory once, reads directly into it."""
    if n == 0:
        return b''
        
    buf = bytearray(n)
    view = memoryview(buf)
    pos = 0
    
    while pos < n:
        bytes_read = 0
        if hasattr(source, 'recv_into'): # Socket
            try:
                bytes_read = source.recv_into(view[pos:])
            except BlockingIOError:
                continue 
        elif hasattr(source, 'readinto'): # File
            bytes_read = source.readinto(view[pos:])
        else: # Fallback
            if hasattr(source, 'recv'):
                chunk = source.recv(n - pos)
            else:
                chunk = source.read(n - pos)
            if chunk:
                view[pos:pos+len(chunk)] = chunk
                bytes_read = len(chunk)
            else:
                bytes_read = 0

        if bytes_read == 0:
            if pos == 0: return None
            raise EOFError(f"Stream closed. Expected {n} bytes, got {pos}")
            
        pos += bytes_read
        
    return bytes(buf)

def read_stream(source: Any) -> Union[np.ndarray, None]:
    """Reads a tensor from a socket/file using Zero-Copy buffering."""
    try:
        header = _read_exact(source, 8)
    except EOFError as e:
        raise EOFError("Stream ended during header read") from e
        
    if header is None: return None
        
    magic, ver, flags, dtype_code, ndim = struct.unpack('<4sBBBB', header)
    if magic != _MAGIC: raise ValueError("Invalid tenso packet")

    shape_len = ndim * 4
    try:
        shape_bytes = _read_exact(source, shape_len)
    except EOFError as e:
        raise EOFError("Stream ended during shape read") from e
    
    shape = struct.unpack(f'<{ndim}I', shape_bytes)
    
    current_offset = 8 + shape_len
    remainder = current_offset % _ALIGNMENT
    padding_len = 0 if remainder == 0 else (_ALIGNMENT - remainder)
    
    try:
        padding = _read_exact(source, padding_len)
    except EOFError as e:
        raise EOFError("Stream ended during padding read") from e
    if padding is None: padding = b''

    dtype = _REV_DTYPE_MAP.get(dtype_code)
    if dtype is None: raise ValueError(f"Unknown dtype: {dtype_code}")
        
    body_len = int(math.prod(shape) * dtype.itemsize)
    
    try:
        body = _read_exact(source, body_len)
    except EOFError as e:
        raise EOFError("Stream ended during body read") from e
    if body is None: body = b''

    return loads(header + shape_bytes + padding + body)


# --- Stream Helper (Write) ---

def write_stream(tensor: np.ndarray, dest: Any, strict: bool = False) -> int:
    """
    Writes a tensor to a socket/file using Vectored I/O (os.writev).
    Sends Header + Shape + Body in ONE system call (Atomic Send).
    """
    # 1. Prepare Metadata
    if tensor.dtype not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {tensor.dtype}")
    
    # [FIXED] Restored strict mode check
    if not tensor.flags['C_CONTIGUOUS']:
        if strict:
            raise ValueError("Tensor is not C-Contiguous and strict=True. "
                             "Reshape or copy array before serializing.")
        tensor = np.ascontiguousarray(tensor)

    if not IS_LITTLE_ENDIAN or tensor.dtype.byteorder == '>':
        tensor = tensor.astype(tensor.dtype.newbyteorder('<'))

    dtype_code = _DTYPE_MAP[tensor.dtype]
    shape = tensor.shape
    ndim = len(shape)
    
    header_size = 8
    shape_size = ndim * 4
    padding_size = (64 - (header_size + shape_size) % 64) % 64
    
    header = struct.pack('<4sBBBB', _MAGIC, _VERSION, 1, dtype_code, ndim)
    shape_block = struct.pack(f'<{ndim}I', *shape)
    padding = b'\x00' * padding_size
    
    # 2. Try Atomic Vectored Write (Best for Sockets)
    if hasattr(dest, 'fileno'):
        try:
            fd = dest.fileno()
            if hasattr(os, 'writev'):
                return os.writev(fd, [header, shape_block, padding, tensor.data])
        except (AttributeError, OSError):
            pass 
            
    # 3. Fallback (Coalesced Write)
    dest.write(header + shape_block + padding)
    dest.write(tensor.data)
    return len(header) + len(shape_block) + len(padding) + tensor.nbytes


# --- Core Functions ---

def dumps(tensor: np.ndarray, strict: bool = False) -> bytes:
    """Serialize to bytes."""
    if tensor.dtype not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {tensor.dtype}")
    
    # [FIXED] Restored strict mode check
    if not tensor.flags['C_CONTIGUOUS']:
        if strict:
            raise ValueError("Tensor is not C-Contiguous and strict=True. "
                             "Reshape or copy array before serializing.")
        tensor = np.ascontiguousarray(tensor)

    if not IS_LITTLE_ENDIAN or tensor.dtype.byteorder == '>':
        tensor = tensor.astype(tensor.dtype.newbyteorder('<'))

    dtype_code = _DTYPE_MAP[tensor.dtype]
    shape = tensor.shape
    ndim = len(shape)
    
    padding_size = (64 - (8 + ndim*4) % 64) % 64
    
    parts = [
        struct.pack('<4sBBBB', _MAGIC, _VERSION, 1, dtype_code, ndim),
        struct.pack(f'<{ndim}I', *shape),
        b'\x00' * padding_size,
        tensor.tobytes()
    ]
    return b''.join(parts)


def loads(data: Union[bytes, mmap.mmap], copy: bool = False) -> np.ndarray:
    """Deserialize from bytes."""
    if len(data) < 8: raise ValueError("Packet too short")
    magic, ver, flags, dtype_code, ndim = struct.unpack('<4sBBBB', data[:8])
    if magic != _MAGIC: raise ValueError("Invalid tenso packet")
    
    if ver > _VERSION:
        raise ValueError(f"Unsupported version: {ver}")

    if dtype_code not in _REV_DTYPE_MAP:
        raise ValueError(f"Unknown dtype code: {dtype_code}")

    shape_start = 8
    shape_end = 8 + (ndim * 4)
    shape = struct.unpack(f'<{ndim}I', data[shape_start:shape_end])
    
    body_start = shape_end
    if ver >= 2 and flags & 1:
        padding_size = (64 - shape_end % 64) % 64
        body_start += padding_size
        
    dtype = _REV_DTYPE_MAP[dtype_code]
    
    arr = np.frombuffer(
        data,
        dtype=dtype,
        offset=body_start,
        count=int(math.prod(shape))
    )
    arr = arr.reshape(shape)
    
    if copy: return arr.copy()
    arr.flags.writeable = False
    return arr


def dump(tensor: np.ndarray, fp: BinaryIO, strict: bool = False) -> None:
    """Alias for write_stream for backward compatibility."""
    write_stream(tensor, fp, strict=strict)

def load(fp: BinaryIO, mmap_mode: bool = False, copy: bool = False) -> np.ndarray:
    """Alias for read_stream logic or mmap."""
    if mmap_mode:
        mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
        return loads(mm, copy=copy)
    return read_stream(fp)