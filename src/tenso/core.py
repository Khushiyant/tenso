import struct
import numpy as np
import xxhash
from typing import BinaryIO, Union, Any, Generator
import math
import mmap
import sys
import os
from .config import _MAGIC, _VERSION, _ALIGNMENT, _DTYPE_MAP, _REV_DTYPE_MAP, FLAG_ALIGNED, FLAG_INTEGRITY

IS_LITTLE_ENDIAN = (sys.byteorder == 'little')

# --- Helper Functions ---

def _read_into_buffer(source: Any, buf: Union[bytearray, memoryview, np.ndarray]) -> bool:
    """
    Fill a buffer from a source (socket or file-like object).

    This is a low-level function that handles reading data into a buffer from
    various source types, including sockets and file-like objects. It ensures
    the entire buffer is filled or returns False on EOF at the start.

    Args:
        source: The data source, which can be a socket, file-like object, or
            any object with 'readinto', 'recv_into', 'read', or 'recv' methods.
        buf: The buffer to fill, which can be a bytearray, memoryview, or numpy array.

    Returns:
        bool: True if the buffer was fully filled, False if EOF was encountered
        before reading any data.

    Raises:
        EOFError: If the stream ends unexpectedly after partial reading.
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
            raise EOFError(f"Stream ended unexpectedly. Expected {n} bytes, got {pos}")
            
        pos += read
    return True

# --- Stream Readers ---

def read_stream(source: Any) -> Union[np.ndarray, None]:
    """
    Optimized stream reader: Consolidates data reads and verifies integrity.

    Read a complete Tenso packet from a stream source and deserialize it into
    a numpy array. This function handles the full packet parsing, including
    header, shape, padding, body, and optional integrity footer.

    Args:
        source: The data source, which can be a socket, file-like object, or
            any object supported by _read_into_buffer.

    Returns:
        np.ndarray or None: The deserialized numpy array with writeable=False,
        or None if EOF is encountered before any data.

    Raises:
        ValueError: If the packet magic number is invalid or integrity check fails.
        EOFError: If the stream ends unexpectedly during reading.
    """
    header = bytearray(8)
    if not _read_into_buffer(source, header): return None
        
    magic, ver, flags, dtype_code, ndim = struct.unpack('<4sBBBB', header)
    if magic != _MAGIC: raise ValueError("Invalid tenso packet")

    shape_len = ndim * 4
    shape_bytes = bytearray(shape_len)
    if not _read_into_buffer(source, shape_bytes): raise EOFError("Stream ended during shape read")
    shape = struct.unpack(f'<{ndim}I', shape_bytes)
    
    dtype = _REV_DTYPE_MAP.get(dtype_code)
    current_pos = 8 + shape_len
    remainder = current_pos % _ALIGNMENT
    padding_len = 0 if remainder == 0 else (_ALIGNMENT - remainder)
    body_len = int(np.prod(shape) * dtype.itemsize)
    
    # Read Padding + Body + optional Footer in ONE buffer
    footer_len = 8 if (flags & FLAG_INTEGRITY) else 0
    data_buffer = np.empty(padding_len + body_len + footer_len, dtype=np.uint8)
    if not _read_into_buffer(source, data_buffer): raise EOFError("Stream ended during data read")

    # Verify XXH3 Integrity
    if footer_len > 0:
        body_slice = data_buffer[padding_len : padding_len + body_len]
        actual_hash = xxhash.xxh3_64_intdigest(body_slice)
        expected_hash = struct.unpack('<Q', data_buffer[padding_len + body_len :])[0]
        if actual_hash != expected_hash:
            raise ValueError("Integrity check failed: XXH3 mismatch")

    arr = np.frombuffer(data_buffer, dtype=dtype, offset=padding_len, count=int(np.prod(shape))).reshape(shape)
    arr.flags.writeable = False 
    return arr

# --- Stream Writers ---

def iter_dumps(tensor: np.ndarray, strict: bool = False, check_integrity: bool = False) -> Generator[Union[bytes, memoryview], None, None]:
    """
    Vectored serialization: Yields packet parts to avoid memory copies.

    Serialize a numpy array into a Tenso packet by yielding individual parts
    (header, shape, padding, body, footer) as separate chunks. This allows
    for efficient I/O operations without creating a contiguous copy of the
    entire packet in memory.

    Args:
        tensor: The numpy array to serialize.
        strict: If True, raise an error if the tensor is not C-contiguous.
            If False, make a contiguous copy if necessary.
        check_integrity: If True, include an XXH3 checksum in the packet footer.

    Yields:
        bytes or memoryview: Sequential parts of the Tenso packet.

    Raises:
        ValueError: If the tensor's dtype is unsupported or (if strict=True)
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

    Serialize a numpy array and write it to a destination using efficient
    vectored I/O operations. If the destination supports writev (like files),
    it uses that for atomic writes; otherwise falls back to sequential writes.

    Args:
        tensor: The numpy array to serialize and write.
        dest: The destination object, which can be a file-like object with
            a write method or a file descriptor with fileno().
        strict: If True, raise an error if the tensor is not C-contiguous.
            If False, make a contiguous copy if necessary.
        check_integrity: If True, include an XXH3 checksum in the packet.

    Returns:
        int: The total number of bytes written.

    Raises:
        ValueError: If the tensor's dtype is unsupported or (if strict=True)
        the tensor is not C-contiguous.
        OSError: If writing to the destination fails.
    """
    chunks = list(iter_dumps(tensor, strict=strict, check_integrity=check_integrity))
    
    if hasattr(dest, 'fileno'):
        try:
            fd = dest.fileno()
            if hasattr(os, 'writev'):
                return os.writev(fd, chunks)
        except (AttributeError, OSError): pass 
            
    written = 0
    for chunk in chunks:
        dest.write(chunk)
        written += len(chunk)
    return written

# --- Core Functions ---

def dumps(tensor: np.ndarray, strict: bool = False, check_integrity: bool = False) -> memoryview:
    """
    Serialize a numpy array to a Tenso packet.

    Convert a numpy array into a complete Tenso packet in memory, returning
    a memoryview for efficient access without copying.

    Args:
        tensor: The numpy array to serialize.
        strict: If True, raise an error if the tensor is not C-contiguous.
            If False, make a contiguous copy if necessary.
        check_integrity: If True, include an XXH3 checksum in the packet.

    Returns:
        memoryview: A memoryview of the complete Tenso packet.

    Raises:
        ValueError: If the tensor's dtype is unsupported or (if strict=True)
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
    dest_view = buffer[body_start:body_end].view(dtype=tensor.dtype).reshape(shape)
    np.copyto(dest_view, tensor, casting='no')
    
    if check_integrity:
        digest = xxhash.xxh3_64_intdigest(buffer[body_start:body_end])
        struct.pack_into('<Q', buffer, body_end, digest)
    
    return memoryview(buffer)

def loads(data: Union[bytes, bytearray, memoryview, np.ndarray, mmap.mmap], copy: bool = False) -> np.ndarray:
    """
    Deserialize a Tenso packet from a bytes-like object.

    Parse a complete Tenso packet from memory and reconstruct the original
    numpy array. Supports various buffer types and optional copying.

    Args:
        data: The raw Tenso packet data as bytes, bytearray, memoryview,
            numpy array, or memory-mapped file.
        copy: If True, return a copy of the array data. If False, return
            a read-only view into the original buffer.

    Returns:
        np.ndarray: The deserialized numpy array. If copy=False, the array
        has writeable=False to prevent accidental modification.

    Raises:
        ValueError: If the packet is invalid, too short, or integrity check fails.
    """
    mv = memoryview(data)
    if len(mv) < 8: raise ValueError("Packet too short")
    magic, ver, flags, dtype_code, ndim = struct.unpack('<4sBBBB', mv[:8])
    if magic != _MAGIC: raise ValueError("Invalid tenso packet")
    
    shape_end = 8 + (ndim * 4)
    shape = struct.unpack(f'<{ndim}I', mv[8:shape_end])
    dtype = _REV_DTYPE_MAP[dtype_code]
    
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
    Serialize and write a tensor to a file using the optimized path.

    This is a convenience function that combines serialization and file writing
    using the efficient write_stream function.

    Args:
        tensor: The numpy array to serialize and write.
        fp: A binary file-like object open for writing.
        strict: If True, raise an error if the tensor is not C-contiguous.
            If False, make a contiguous copy if necessary.
        check_integrity: If True, include an XXH3 checksum in the packet.

    Raises:
        ValueError: If the tensor's dtype is unsupported or (if strict=True)
        the tensor is not C-contiguous.
        OSError: If writing to the file fails.
    """
    write_stream(tensor, fp, strict=strict, check_integrity=check_integrity)

def load(fp: BinaryIO, mmap_mode: bool = False, copy: bool = False) -> np.ndarray:
    """
    Load a tensor from a file, optionally using memory-mapping.

    Read a Tenso packet from a file and deserialize it. Supports memory-mapping
    for large files to avoid loading the entire file into memory.

    Args:
        fp: A binary file-like object open for reading.
        mmap_mode: If True, use memory-mapping to read the file. This is
            efficient for large files but requires the file to be seekable.
        copy: If True and mmap_mode=True, return a copy of the array data.
            If False, return a read-only view.

    Returns:
        np.ndarray: The deserialized numpy array.

    Raises:
        ValueError: If the packet is invalid or integrity check fails.
        EOFError: If the file ends unexpectedly.
    """
    if mmap_mode:
        mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
        return loads(mm, copy=copy)
    return read_stream(fp)