import struct
import numpy as np
import xxhash
from typing import BinaryIO, Union, Any, Generator
import math
import mmap
import sys
import os
from .config import (
    _MAGIC, _VERSION, _ALIGNMENT, _DTYPE_MAP, _REV_DTYPE_MAP, 
    FLAG_ALIGNED, FLAG_INTEGRITY, MAX_NDIM, MAX_ELEMENTS
)

IS_LITTLE_ENDIAN = (sys.byteorder == 'little')

# --- Helper Functions ---

def _read_into_buffer(source: Any, buf: Union[bytearray, memoryview, np.ndarray]) -> bool:
    """
    Fill a buffer from a source, handling various I/O types.
    
    This internal helper function reads data from different types of sources
    (files, sockets, streams) into a buffer, handling partial reads and
    different I/O interfaces gracefully.
    
    Args:
        source: The data source to read from. Can be any object with read(),
                readinto(), recv(), or recv_into() methods.
        buf: The buffer to fill with data. Can be bytearray, memoryview, or np.ndarray.
    
    Returns:
        bool: True if the buffer was filled completely, False if EOF was reached
              before filling the buffer (only possible on first read).
    
    Raises:
        EOFError: If the source ends prematurely after some data has been read.
    """
    view = memoryview(buf)
    n = view.nbytes
    if n == 0: return True
        
    pos = 0
    while pos < n:
        read = 0
        if hasattr(source, 'readinto'): 
            read = source.readinto(view[pos:])
        elif hasattr(source, 'recv_into'): 
            try:
                read = source.recv_into(view[pos:])
            except BlockingIOError: continue
        else: 
            chunk = None
            remaining = n - pos
            if hasattr(source, 'recv'): chunk = source.recv(remaining)
            elif hasattr(source, 'read'): chunk = source.read(remaining)
            
            if chunk:
                view[pos:pos+len(chunk)] = chunk
                read = len(chunk)
            else: read = 0

        if read == 0:
            if pos == 0: return False 
            raise EOFError(f"Expected {n} bytes, got {pos}")
            
        pos += read
    return True

# --- Stream Readers ---

def read_stream(source: Any) -> Union[np.ndarray, None]:
    """
    Read and deserialize a tensor from a stream source with DoS protection.
    
    This function reads a complete Tenso packet from any stream-like source
    (file, socket, etc.) and deserializes it into a numpy array. Includes
    built-in protection against denial-of-service attacks by limiting maximum
    dimensions and element counts.
    
    Args:
        source: Stream source to read from. Must support read(), readinto(),
                recv(), or recv_into() methods.
    
    Returns:
        np.ndarray or None: The deserialized tensor with writeable=False flag set,
                           or None if the stream ended before any data was read.
    
    Raises:
        ValueError: If the packet is invalid, exceeds security limits, or fails
                   integrity checks.
        EOFError: If the stream ends prematurely during reading.
    """
    # 1. Read Header
    header = bytearray(8)
    try:
        if not _read_into_buffer(source, header): return None
    except EOFError as e:
        raise EOFError(f"Stream ended during header read. {e}") from None
        
    magic, ver, flags, dtype_code, ndim = struct.unpack('<4sBBBB', header)
    if magic != _MAGIC: raise ValueError("Invalid tenso packet")
    
    # [SECURITY] DoS Protection: Check Dimensions
    if ndim > MAX_NDIM:
        raise ValueError(f"Packet exceeds maximum dimensions ({ndim} > {MAX_NDIM})")

    # 2. Read Shape
    shape_len = ndim * 4
    shape_bytes = bytearray(shape_len)
    try:
        if not _read_into_buffer(source, shape_bytes):
            raise EOFError("Stream ended during shape read")
    except EOFError as e:
        raise EOFError(f"Stream ended during shape read. {e}") from None

    shape = struct.unpack(f'<{ndim}I', shape_bytes)
    
    # [SECURITY] DoS Protection: Check Element Count
    num_elements = int(np.prod(shape))
    if num_elements > MAX_ELEMENTS:
        raise ValueError(f"Packet exceeds maximum elements ({num_elements} > {MAX_ELEMENTS})")

    dtype = _REV_DTYPE_MAP.get(dtype_code)
    if dtype is None:
        raise ValueError(f"Unsupported dtype code: {dtype_code}")
    
    # 3. Calculate Layout
    current_pos = 8 + shape_len
    remainder = current_pos % _ALIGNMENT
    padding_len = 0 if remainder == 0 else (_ALIGNMENT - remainder)
    body_len = num_elements * dtype.itemsize
    footer_len = 8 if (flags & FLAG_INTEGRITY) else 0
    
    # 4. Consolidated Read (Padding + Body + Footer)
    data_buffer = np.empty(padding_len + body_len + footer_len, dtype=np.uint8)
    try:
        if not _read_into_buffer(source, data_buffer):
            raise EOFError("Stream ended during body read")
    except EOFError as e:
        raise EOFError(f"Stream ended during body read. {e}") from None

    # 5. Verify Integrity
    if footer_len > 0:
        body_slice = data_buffer[padding_len : padding_len + body_len]
        actual_hash = xxhash.xxh3_64_intdigest(body_slice)
        expected_hash = struct.unpack('<Q', data_buffer[padding_len + body_len :])[0]
        if actual_hash != expected_hash:
            raise ValueError("Integrity check failed: XXH3 mismatch")

    arr = np.frombuffer(data_buffer, dtype=dtype, offset=padding_len, count=num_elements).reshape(shape)
    arr.flags.writeable = False 
    return arr

# --- Stream Writers ---

def iter_dumps(tensor: np.ndarray, strict: bool = False, check_integrity: bool = False) -> Generator[Union[bytes, memoryview], None, None]:
    """
    Vectored serialization: Yields packet parts to avoid memory copies.
    
    This generator function serializes a tensor into Tenso format by yielding
    individual chunks of the packet. This allows for zero-copy streaming and
    efficient I/O operations, especially useful for large tensors.
    
    Args:
        tensor: The numpy array to serialize. Must have a supported dtype.
        strict: If True, raises an error for non-contiguous arrays instead of
                making them contiguous. Default False.
        check_integrity: If True, includes an XXH3 hash for integrity verification.
                        Default False.
    
    Yields:
        bytes or memoryview: Sequential chunks of the Tenso packet (header,
                           shape, padding, body, optional integrity hash).
    
    Raises:
        ValueError: If the tensor dtype is unsupported or (if strict=True) if
                   the tensor is not C-contiguous.
    """
    if tensor.dtype not in _DTYPE_MAP: raise ValueError(f"Unsupported dtype: {tensor.dtype}")
    
    if not tensor.flags['C_CONTIGUOUS']:
        if strict: raise ValueError("Tensor is not C-Contiguous")
        tensor = np.ascontiguousarray(tensor)

    if not IS_LITTLE_ENDIAN or tensor.dtype.byteorder == '>':
        tensor = tensor.astype(tensor.dtype.newbyteorder('<'))

    dtype_code = _DTYPE_MAP[tensor.dtype]
    shape = tensor.shape
    ndim = len(shape)
    
    flags = FLAG_ALIGNED | (FLAG_INTEGRITY if check_integrity else 0)
    header = struct.pack('<4sBBBB', _MAGIC, _VERSION, flags, dtype_code, ndim)
    shape_block = struct.pack(f'<{ndim}I', *shape)
    yield header
    yield shape_block
    
    current_len = 8 + (ndim * 4)
    remainder = current_len % _ALIGNMENT
    padding_len = 0 if remainder == 0 else (_ALIGNMENT - remainder)
    if padding_len > 0: yield b'\x00' * padding_len
        
    yield tensor.data

    if check_integrity:
        yield struct.pack('<Q', xxhash.xxh3_64_intdigest(tensor.data))

def write_stream(tensor: np.ndarray, dest: Any, strict: bool = False, check_integrity: bool = False) -> int:
    """
    Write a tensor to a destination using vectored I/O.
    
    Serializes a tensor and writes it to any destination that supports write()
    or has a fileno() method (for direct system calls). Uses iter_dumps internally
    for memory-efficient streaming.
    
    Args:
        tensor: The numpy array to serialize.
        dest: Destination to write to. Must support write() method, or have
              fileno() for direct system calls.
        strict: If True, raises error for non-contiguous arrays. Default False.
        check_integrity: If True, includes integrity hash. Default False.
    
    Returns:
        int: Number of bytes written.
    
    Raises:
        ValueError: If tensor dtype is unsupported or other serialization errors.
        OSError: If writing to the destination fails.
    """
    chunks = list(iter_dumps(tensor, strict=strict, check_integrity=check_integrity))
    
    # Removed os.writev optimization as it conflicts with Python's buffered I/O
    # (e.g., tempfile, BytesIO) causing data corruption or lost writes.
    # Python's write() handles memoryviews efficiently enough.
    written = 0
    for chunk in chunks:
        dest.write(chunk)
        written += len(chunk)
    return written

# --- Core Functions ---

def dumps(tensor: np.ndarray, strict: bool = False, check_integrity: bool = False) -> memoryview:
    """
    Serialize a numpy array to a Tenso packet.
    
    Creates a complete Tenso packet in memory containing the serialized tensor.
    The returned memoryview provides zero-copy access to the packet data.
    
    Args:
        tensor: The numpy array to serialize. Must have a supported dtype.
        strict: If True, raises error for non-contiguous arrays instead of
                making them contiguous. Default False.
        check_integrity: If True, includes XXH3 hash for integrity verification.
                        Default False.
    
    Returns:
        memoryview: A view of the complete Tenso packet bytes.
    
    Raises:
        ValueError: If tensor dtype is unsupported or other serialization errors.
    """
    if tensor.dtype not in _DTYPE_MAP: raise ValueError(f"Unsupported dtype: {tensor.dtype}")
    
    if not tensor.flags['C_CONTIGUOUS']:
        if strict: raise ValueError("Tensor is not C-Contiguous")
        tensor = np.ascontiguousarray(tensor)

    if not IS_LITTLE_ENDIAN or tensor.dtype.byteorder == '>':
        tensor = tensor.astype(tensor.dtype.newbyteorder('<'))

    dtype_code = _DTYPE_MAP[tensor.dtype]
    shape = tensor.shape
    ndim = len(shape)
    
    current_len = 8 + (ndim * 4)
    remainder = current_len % _ALIGNMENT
    padding_len = 0 if remainder == 0 else (_ALIGNMENT - remainder)
    footer_len = 8 if check_integrity else 0
    total_len = current_len + padding_len + tensor.nbytes + footer_len
    
    buffer = np.empty(total_len, dtype=np.uint8)
    flags = FLAG_ALIGNED | (FLAG_INTEGRITY if check_integrity else 0)
    
    struct.pack_into('<4sBBBB', buffer, 0, _MAGIC, _VERSION, flags, dtype_code, ndim)
    struct.pack_into(f'<{ndim}I', buffer, 8, *shape)
    
    if padding_len > 0: buffer[current_len : current_len + padding_len] = 0
    
    body_start = current_len + padding_len
    body_end = body_start + tensor.nbytes
    
    # Efficient copy
    dest_view = buffer[body_start:body_end].view(dtype=tensor.dtype).reshape(shape)
    np.copyto(dest_view, tensor, casting='no')
    
    if check_integrity:
        digest = xxhash.xxh3_64_intdigest(buffer[body_start:body_end])
        struct.pack_into('<Q', buffer, body_end, digest)
    
    return memoryview(buffer)

def loads(data: Union[bytes, bytearray, memoryview, np.ndarray, mmap.mmap], copy: bool = False) -> np.ndarray:
    """
    Deserialize a Tenso packet from a bytes-like object with DoS protection.
    
    Parses a complete Tenso packet from memory and reconstructs the original
    numpy array. Includes security checks to prevent denial-of-service attacks.
    
    Args:
        data: The raw Tenso packet data as bytes, bytearray, memoryview,
              numpy array, or memory-mapped file.
        copy: If True, returns a copy of the data instead of a read-only view.
              Default False.
    
    Returns:
        np.ndarray: The deserialized tensor. If copy=False, the array has
                   writeable=False to prevent accidental modification.
    
    Raises:
        ValueError: If the packet is invalid, corrupted, or exceeds security limits.
    """
    mv = memoryview(data)
    if len(mv) < 8: raise ValueError("Packet too short")
    magic, ver, flags, dtype_code, ndim = struct.unpack('<4sBBBB', mv[:8])
    if magic != _MAGIC: raise ValueError("Invalid tenso packet")
    
    if ndim > MAX_NDIM:
        raise ValueError(f"Packet exceeds maximum dimensions ({ndim})")
    
    shape_end = 8 + (ndim * 4)
    shape = struct.unpack(f'<{ndim}I', mv[8:shape_end])
    
    # [SECURITY] DoS Protection
    if np.prod(shape) > MAX_ELEMENTS:
        raise ValueError("Packet exceeds maximum elements")

    dtype = _REV_DTYPE_MAP.get(dtype_code)
    if dtype is None: raise ValueError(f"Unsupported dtype code: {dtype_code}")
    
    body_start = shape_end
    if flags & FLAG_ALIGNED:
        remainder = shape_end % _ALIGNMENT
        body_start += (0 if remainder == 0 else (_ALIGNMENT - remainder))
        
    num_elements = int(np.prod(shape))
    body_len = num_elements * dtype.itemsize
    body_end = body_start + body_len

    if flags & FLAG_INTEGRITY:
        actual_hash = xxhash.xxh3_64_intdigest(mv[body_start:body_end])
        expected_hash = struct.unpack('<Q', mv[body_end : body_end + 8])[0]
        if actual_hash != expected_hash:
            raise ValueError("Integrity check failed: XXH3 mismatch")

    arr = np.frombuffer(mv, dtype=dtype, offset=body_start, count=num_elements).reshape(shape)
    if copy: return arr.copy()
    arr.flags.writeable = False
    return arr

def dump(tensor: np.ndarray, fp: BinaryIO, strict: bool = False, check_integrity: bool = False) -> None:
    """
    Serialize a tensor and write it to a binary file.
    
    Convenience function that serializes a tensor and writes the complete
    Tenso packet to an open binary file.
    
    Args:
        tensor: The numpy array to serialize.
        fp: Open binary file object to write to (must be opened in binary mode).
        strict: If True, raises error for non-contiguous arrays. Default False.
        check_integrity: If True, includes integrity hash. Default False.
    
    Raises:
        ValueError: If serialization fails.
        OSError: If writing to the file fails.
    """
    write_stream(tensor, fp, strict=strict, check_integrity=check_integrity)

def load(fp: BinaryIO, mmap_mode: bool = False, copy: bool = False) -> np.ndarray:
    """
    Deserialize a tensor from a binary file.
    
    Reads a complete Tenso packet from an open binary file and deserializes it.
    Optionally uses memory mapping for efficient loading of large files.
    
    Args:
        fp: Open binary file object to read from.
        mmap_mode: If True, uses memory mapping for potentially better performance
                  with large files. Default False.
        copy: If True, returns a copy instead of a read-only view. Default False.
    
    Returns:
        np.ndarray: The deserialized tensor.
    
    Raises:
        ValueError: If the packet is invalid or corrupted.
        OSError: If reading from the file fails.
        EOFError: If the file is empty or the stream ends prematurely.
    """
    if mmap_mode:
        mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
        return loads(mm, copy=copy)
    
    result = read_stream(fp)
    if result is None:
        raise EOFError("Empty file or stream")
    return result