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

def _read_into_buffer(source: Any, buf: Union[bytearray, memoryview, np.ndarray]) -> bool:
    """
    Fill a buffer from a source (socket or file-like object).

    Args:
        source: The data source supporting 'readinto', 'recv_into', 'read', or 'recv'.
        buf: The buffer to fill (bytearray, memoryview, or numpy array).

    Returns:
        bool: True if the buffer was filled completely, False if EOF at start.

    Raises:
        EOFError: If the stream ends before the buffer is fully filled.
    """
    view = memoryview(buf)
    n = view.nbytes
    if n == 0:
        return True
        
    pos = 0
    while pos < n:
        read = 0
        if hasattr(source, 'readinto'): # File-like
            read = source.readinto(view[pos:])
        elif hasattr(source, 'recv_into'): # Socket-like
            try:
                read = source.recv_into(view[pos:])
            except BlockingIOError:
                continue
        else: # Fallback (Mock objects, etc)
            chunk = None
            remaining = n - pos
            if hasattr(source, 'recv'):
                chunk = source.recv(remaining)
            elif hasattr(source, 'read'):
                chunk = source.read(remaining)
            
            if chunk:
                view[pos:pos+len(chunk)] = chunk
                read = len(chunk)
            else:
                read = 0

        if read == 0:
            if pos == 0:
                return False # Clean EOF
            raise EOFError(f"Stream ended unexpectedly. Expected {n} bytes, got {pos}")
            
        pos += read
        
    return True

def read_stream(source: Any) -> Union[np.ndarray, None]:
    """
    Read a tensor from a socket or file using zero-copy buffering.

    Allocates a single uninitialized buffer and reads directly into it for efficiency.

    Args:
        source: The data source (socket or file-like object).

    Returns:
        np.ndarray or None: The deserialized tensor, or None if EOF at start.

    Raises:
        EOFError: If the stream ends unexpectedly during read.
        ValueError: If the packet is invalid or dtype is unknown.
    """
    # 1. Read Header (8 bytes)
    header = bytearray(8)
    try:
        if not _read_into_buffer(source, header):
            return None
    except EOFError as e:
        raise EOFError("Stream ended during header read") from e
        
    magic, ver, flags, dtype_code, ndim = struct.unpack('<4sBBBB', header)
    if magic != _MAGIC: raise ValueError("Invalid tenso packet")

    # 2. Read Shape
    shape_len = ndim * 4
    shape_bytes = bytearray(shape_len)
    try:
        if not _read_into_buffer(source, shape_bytes):
            raise EOFError("Stream ended during shape read")
    except EOFError as e:
        raise EOFError("Stream ended during shape read") from e
    
    shape = struct.unpack(f'<{ndim}I', shape_bytes)
    
    # 3. Calculate Layout
    dtype = _REV_DTYPE_MAP.get(dtype_code)
    if dtype is None: raise ValueError(f"Unknown dtype: {dtype_code}")
    
    # Calculate padding required to align the body
    current_pos = 8 + shape_len
    remainder = current_pos % _ALIGNMENT
    padding_len = 0 if remainder == 0 else (_ALIGNMENT - remainder)
    
    body_len = int(math.prod(shape) * dtype.itemsize)
    total_len = current_pos + padding_len + body_len
    
    # 4. Allocate ONE Uninitialized Buffer (Fastest)
    # We use uint8 to manipulate the raw bytes first
    full_buffer = np.empty(total_len, dtype=np.uint8)
    
    # Fill Header/Shape into the buffer (for loads compatibility)
    full_buffer[0:8] = list(header)
    full_buffer[8:8+shape_len] = list(shape_bytes)
    
    # 5. Read Padding (Consumption)
    if padding_len > 0:
        pad_view = full_buffer[current_pos : current_pos+padding_len]
        try:
            if not _read_into_buffer(source, pad_view):
                raise EOFError("Stream ended during padding read")
        except EOFError as e:
            raise EOFError("Stream ended during padding read") from e

    # 6. Read Body DIRECTLY into buffer
    body_view = full_buffer[current_pos+padding_len:]
    try:
        if not _read_into_buffer(source, body_view):
            raise EOFError("Stream ended during body read")
    except EOFError as e:
        raise EOFError("Stream ended during body read") from e

    # 7. Zero-Copy Load
    return loads(full_buffer)


# --- Stream Helper (Write) ---

def write_stream(tensor: np.ndarray, dest: Any, strict: bool = False) -> int:
    """
    Write a tensor to a socket or file using vectored I/O (os.writev if available).

    Sends header, shape, and body in a single system call for atomicity and performance.

    Args:
        tensor: The numpy array to serialize and send.
        dest: The destination (socket or file-like object).
        strict: If True, require tensor to be C-contiguous.

    Returns:
        int: Number of bytes written.

    Raises:
        ValueError: If dtype is unsupported or tensor is not C-contiguous in strict mode.
    """
    if tensor.dtype not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {tensor.dtype}")
    
    if not tensor.flags['C_CONTIGUOUS']:
        if strict:
            raise ValueError("Tensor is not C-Contiguous")
        tensor = np.ascontiguousarray(tensor)

    if not IS_LITTLE_ENDIAN or tensor.dtype.byteorder == '>':
        tensor = tensor.astype(tensor.dtype.newbyteorder('<'))

    dtype_code = _DTYPE_MAP[tensor.dtype]
    shape = tensor.shape
    ndim = len(shape)
    
    current_len = 8 + ndim * 4
    remainder = current_len % _ALIGNMENT
    padding_size = 0 if remainder == 0 else (_ALIGNMENT - remainder)
    
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

def dumps(tensor: np.ndarray, strict: bool = False) -> memoryview:
    """
    Serialize a numpy array to a Tenso packet (zero-copy, uninitialized buffer).

    Args:
        tensor: The numpy array to serialize.
        strict: If True, require tensor to be C-contiguous.

    Returns:
        memoryview: A memoryview of the serialized packet (no copy).

    Raises:
        ValueError: If dtype is unsupported or tensor is not C-contiguous in strict mode.
    """
    if tensor.dtype not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {tensor.dtype}")
    
    # 1. Ensure C-Contiguous
    if not tensor.flags['C_CONTIGUOUS']:
        if strict:
            raise ValueError("Tensor is not C-Contiguous")
        tensor = np.ascontiguousarray(tensor)

    if not IS_LITTLE_ENDIAN or tensor.dtype.byteorder == '>':
        tensor = tensor.astype(tensor.dtype.newbyteorder('<'))

    # 2. Calculate Layout
    dtype_code = _DTYPE_MAP[tensor.dtype]
    shape = tensor.shape
    ndim = len(shape)
    
    header_len = 8
    shape_len = ndim * 4
    current_len = header_len + shape_len
    remainder = current_len % _ALIGNMENT
    padding_len = 0 if remainder == 0 else (_ALIGNMENT - remainder)
    
    total_len = current_len + padding_len + tensor.nbytes
    
    # 3. Allocate UNINITIALIZED memory (Fastest allocation in Python)
    buffer = np.empty(total_len, dtype=np.uint8)
    
    # 4. Write Metadata
    struct.pack_into('<4sBBBB', buffer, 0, _MAGIC, _VERSION, 1, dtype_code, ndim)
    struct.pack_into(f'<{ndim}I', buffer, 8, *shape)
    
    # 5. Zero out padding
    if padding_len > 0:
        pad_start = current_len
        buffer[pad_start : pad_start+padding_len] = 0
    
    # 6. Copy Data 
    body_start = current_len + padding_len
    dest_view = buffer[body_start:].view(dtype=tensor.dtype).reshape(shape)
    np.copyto(dest_view, tensor, casting='no')
    
    # 7. Return View (No Copy)
    return memoryview(buffer)


def loads(data: Union[bytes, bytearray, memoryview, np.ndarray, mmap.mmap], copy: bool = False) -> np.ndarray:
    """
    Deserialize a Tenso packet from a bytes-like object.

    Args:
        data: The bytes-like object containing the Tenso packet.
        copy: If True, return a copy of the data; otherwise, return a zero-copy view.

    Returns:
        np.ndarray: The deserialized numpy array.

    Raises:
        ValueError: If the packet is invalid, version is unsupported, or dtype is unknown.
    """
    mv = memoryview(data)
    
    if len(mv) < 8: raise ValueError("Packet too short")
    
    magic, ver, flags, dtype_code, ndim = struct.unpack('<4sBBBB', mv[:8])
    if magic != _MAGIC: raise ValueError("Invalid tenso packet")
    
    if ver > _VERSION:
        raise ValueError(f"Unsupported version: {ver}")

    if dtype_code not in _REV_DTYPE_MAP:
        raise ValueError(f"Unknown dtype code: {dtype_code}")

    shape_start = 8
    shape_end = 8 + (ndim * 4)
    shape = struct.unpack(f'<{ndim}I', mv[shape_start:shape_end])
    
    body_start = shape_end
    if ver >= 2 and flags & 1:
        remainder = shape_end % _ALIGNMENT
        padding_size = 0 if remainder == 0 else (_ALIGNMENT - remainder)
        body_start += padding_size
        
    dtype = _REV_DTYPE_MAP[dtype_code]
    
    arr = np.frombuffer(
        mv,
        dtype=dtype,
        offset=body_start,
        count=int(math.prod(shape))
    )
    arr = arr.reshape(shape)
    
    if copy: return arr.copy()
    
    # SAFETY: Enforce read-only for zero-copy views, even if underlying buffer is mutable.
    # This prevents users from accidentally corrupting the receive buffer.
    arr.flags.writeable = False
        
    return arr


def dump(tensor: np.ndarray, fp: BinaryIO, strict: bool = False) -> None:
    """
    Serialize and write a tensor to a file-like object.

    Alias for write_stream().

    Args:
        tensor: The numpy array to serialize.
        fp: The file-like object to write to.
        strict: If True, require tensor to be C-contiguous.
    """
    write_stream(tensor, fp, strict=strict)

def load(fp: BinaryIO, mmap_mode: bool = False, copy: bool = False) -> np.ndarray:
    """
    Load a tensor from a file-like object, optionally using memory-mapping.

    Args:
        fp: The file-like object to read from.
        mmap_mode: If True, use memory-mapped file access.
        copy: If True, return a copy of the data.

    Returns:
        np.ndarray: The loaded numpy array.
    """
    if mmap_mode:
        mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
        return loads(mm, copy=copy)
    return read_stream(fp)