"""
Core Serialization Engine for Tenso.
"""

import mmap
import struct
import sys
from typing import Any, BinaryIO, Generator, Optional, Union

import numpy as np
import xxhash

from .config import (
    _ALIGNMENT,
    _DTYPE_MAP,
    _MAGIC,
    _REV_DTYPE_MAP,
    _VERSION,
    FLAG_ALIGNED,
    FLAG_COMPRESSION,
    FLAG_INTEGRITY,
    FLAG_SPARSE,
    MAX_ELEMENTS,
    MAX_NDIM,
)

try:
    import lz4.frame

    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False

IS_LITTLE_ENDIAN = sys.byteorder == "little"


def _read_into_buffer(
    source: Any, buf: Union[bytearray, memoryview, np.ndarray]
) -> bool:
    """Fill a buffer from a source, handling various I/O types."""
    view = memoryview(buf)
    n = view.nbytes
    if n == 0:
        return True
    pos = 0
    while pos < n:
        read = 0
        if hasattr(source, "readinto"):
            read = source.readinto(view[pos:])
        elif hasattr(source, "recv_into"):
            try:
                read = source.recv_into(view[pos:])
            except BlockingIOError:
                continue
        else:
            remaining = n - pos
            chunk = (
                source.recv(remaining)
                if hasattr(source, "recv")
                else source.read(remaining)
            )
            if chunk:
                view[pos : pos + len(chunk)] = chunk
                read = len(chunk)
            else:
                read = 0
        if read == 0:
            if pos == 0:
                return False
            raise EOFError(f"Expected {n} bytes, got {pos}")
        pos += read
    return True


def read_stream(source: Any) -> Optional[np.ndarray]:
    """Read and deserialize a tensor from a stream source with DoS protection."""
    header = bytearray(8)
    try:
        if not _read_into_buffer(source, header):
            return None
    except EOFError as e:
        raise EOFError(f"Stream ended during header read. {e}") from None

    magic, ver, flags, dtype_code, ndim = struct.unpack("<4sBBBB", header)
    if magic != _MAGIC:
        raise ValueError("Invalid tenso packet")
    if ndim > MAX_NDIM:
        raise ValueError(f"Packet exceeds maximum dimensions ({ndim} > {MAX_NDIM})")

    shape_len = ndim * 4
    shape_bytes = bytearray(shape_len)
    try:
        if not _read_into_buffer(source, shape_bytes):
            raise EOFError("Stream ended during shape read")
    except EOFError as e:
        raise EOFError(f"Stream ended during shape read. {e}") from None

    shape = struct.unpack(f"<{ndim}I", shape_bytes)
    num_elements = int(np.prod(shape))
    if num_elements > MAX_ELEMENTS:
        raise ValueError(
            f"Packet exceeds maximum elements ({num_elements} > {MAX_ELEMENTS})"
        )

    dtype = _REV_DTYPE_MAP.get(dtype_code)
    if dtype is None:
        raise ValueError(f"Unsupported dtype code: {dtype_code}")

    current_pos = 8 + shape_len
    remainder = current_pos % _ALIGNMENT
    padding_len = 0 if remainder == 0 else (_ALIGNMENT - remainder)
    body_len = num_elements * dtype.itemsize
    footer_len = 8 if (flags & FLAG_INTEGRITY) else 0

    data_buffer = np.empty(padding_len + body_len + footer_len, dtype=np.uint8)
    try:
        if not _read_into_buffer(source, data_buffer):
            raise EOFError("Stream ended during body read")
    except EOFError as e:
        raise EOFError(f"Stream ended during body read. {e}") from None

    if footer_len > 0:
        body_slice = data_buffer[padding_len : padding_len + body_len]
        if (
            xxhash.xxh3_64_intdigest(body_slice)
            != struct.unpack("<Q", data_buffer[padding_len + body_len :])[0]
        ):
            raise ValueError("Integrity check failed: XXH3 mismatch")

    arr = np.frombuffer(
        data_buffer, dtype=dtype, offset=padding_len, count=num_elements
    ).reshape(shape)
    arr.flags.writeable = False
    return arr


def iter_dumps(
    tensor: np.ndarray, strict: bool = False, check_integrity: bool = False
) -> Generator[Union[bytes, memoryview], None, None]:
    """
    Vectored serialization: Yields packet parts to avoid memory copies.

    Parameters
    ----------
    tensor : np.ndarray
        The tensor to serialize.
    strict : bool, default False
        Whether to enforce C-contiguous arrays.
    check_integrity : bool, default False
        Whether to include integrity check.

    Yields
    ------
    bytes or memoryview
        Packet parts for serialization.
    """
    if tensor.dtype not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {tensor.dtype}")
    if not tensor.flags["C_CONTIGUOUS"]:
        if strict:
            raise ValueError("Tensor is not C-Contiguous")
        tensor = np.ascontiguousarray(tensor)
    if not IS_LITTLE_ENDIAN or tensor.dtype.byteorder == ">":
        tensor = tensor.astype(tensor.dtype.newbyteorder("<"))

    dtype_code = _DTYPE_MAP[tensor.dtype]
    shape = tensor.shape
    ndim = len(shape)
    flags = FLAG_ALIGNED | (FLAG_INTEGRITY if check_integrity else 0)

    yield struct.pack("<4sBBBB", _MAGIC, _VERSION, flags, dtype_code, ndim)
    yield struct.pack(f"<{ndim}I", *shape)

    current_len = 8 + (ndim * 4)
    remainder = current_len % _ALIGNMENT
    padding_len = 0 if remainder == 0 else (_ALIGNMENT - remainder)
    if padding_len > 0:
        yield b"\x00" * padding_len

    yield tensor.data
    if check_integrity:
        yield struct.pack("<Q", xxhash.xxh3_64_intdigest(tensor.data))


def write_stream(
    tensor: np.ndarray, dest: Any, strict: bool = False, check_integrity: bool = False
) -> int:
    """
    Write a tensor to a destination using vectored I/O.

    Parameters
    ----------
    tensor : np.ndarray
        The tensor to write.
    dest : file-like object
        The destination to write to.
    strict : bool, default False
        Whether to enforce C-contiguous arrays.
    check_integrity : bool, default False
        Whether to include integrity check.

    Returns
    -------
    int
        Number of bytes written.
    """
    chunks = list(iter_dumps(tensor, strict=strict, check_integrity=check_integrity))
    written = 0
    for chunk in chunks:
        dest.write(chunk)
        written += len(chunk)
    return written


def dumps(
    tensor: Any,
    strict: bool = False,
    check_integrity: bool = False,
    compress: bool = False,
) -> memoryview:
    """
    Serialize a numpy or sparse (COO) array to a Tenso packet.

    Parameters
    ----------
    tensor : array_like or sparse matrix
        The tensor to serialize.
    strict : bool, default False
        Whether to enforce C-contiguous arrays.
    check_integrity : bool, default False
        Whether to include integrity check.
    compress : bool, default False
        Whether to compress the data.

    Returns
    -------
    memoryview
        The serialized packet.
    """
    if hasattr(tensor, "tocoo") and not isinstance(tensor, np.ndarray):
        coo = tensor.tocoo()
        data_p = dumps(coo.data, strict, False, False)
        row_p = dumps(coo.row, strict, False, False)
        col_p = dumps(coo.col, strict, False, False)
        header = struct.pack(
            "<4sBBBB", _MAGIC, _VERSION, FLAG_SPARSE, 0, len(coo.shape)
        )
        shape_block = struct.pack(f"<{len(coo.shape)}I", *coo.shape)
        return memoryview(b"".join([header, shape_block, data_p, row_p, col_p]))

    if tensor.dtype not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {tensor.dtype}")
    if not tensor.flags["C_CONTIGUOUS"]:
        if strict:
            raise ValueError("Tensor is not C-Contiguous")
        tensor = np.ascontiguousarray(tensor)
    if not IS_LITTLE_ENDIAN or tensor.dtype.byteorder == ">":
        tensor = tensor.astype(tensor.dtype.newbyteorder("<"))

    dtype_code = _DTYPE_MAP[tensor.dtype]
    shape = tensor.shape
    ndim = len(shape)
    body = tensor.tobytes()
    flags = FLAG_ALIGNED | (FLAG_INTEGRITY if check_integrity else 0)

    if compress:
        if not HAS_LZ4:
            raise ImportError("Compression requires 'lz4'")
        body = lz4.frame.compress(body)
        flags |= FLAG_COMPRESSION

    current_len = 8 + (ndim * 4)
    padding_len = (_ALIGNMENT - (current_len % _ALIGNMENT)) % _ALIGNMENT
    total_len = current_len + padding_len + len(body) + (8 if check_integrity else 0)

    buffer = bytearray(total_len)
    struct.pack_into("<4sBBBB", buffer, 0, _MAGIC, _VERSION, flags, dtype_code, ndim)
    struct.pack_into(f"<{ndim}I", buffer, 8, *shape)

    body_start = current_len + padding_len
    buffer[body_start : body_start + len(body)] = body
    if check_integrity:
        digest = xxhash.xxh3_64_intdigest(body)
        struct.pack_into("<Q", buffer, body_start + len(body), digest)
    return memoryview(buffer)


def loads(
    data: Union[bytes, bytearray, memoryview, np.ndarray, mmap.mmap], copy: bool = False
) -> Any:
    """Deserialize a Tenso packet from a bytes-like object with DoS protection.

    Parameters
    ----------
    data : bytes, bytearray, memoryview, np.ndarray, or mmap.mmap
        The serialized Tenso packet data.
    copy : bool, optional
        Whether to copy the data. Default is False.

    Returns
    -------
    Any
        The deserialized tensor or sparse matrix.
    """
    mv = memoryview(data)
    if len(mv) < 8:
        raise ValueError("Packet too short")
    magic, ver, flags, dtype_code, ndim = struct.unpack("<4sBBBB", mv[:8])
    if magic != _MAGIC:
        raise ValueError("Invalid tenso packet")
    if ndim > MAX_NDIM:
        raise ValueError(f"Packet exceeds maximum dimensions ({ndim})")

    shape_end = 8 + (ndim * 4)
    shape = struct.unpack(f"<{ndim}I", mv[8:shape_end])
    if np.prod(shape) > MAX_ELEMENTS:
        raise ValueError("Packet exceeds maximum elements")

    if flags & FLAG_SPARSE:
        from scipy.sparse import coo_matrix

        offset = shape_end
        d_arr = loads(mv[offset:], copy)
        offset += len(dumps(d_arr))
        r_arr = loads(mv[offset:], copy)
        offset += len(dumps(r_arr))
        c_arr = loads(mv[offset:], copy)
        return coo_matrix((d_arr, (r_arr, c_arr)), shape=shape)

    dtype = _REV_DTYPE_MAP.get(dtype_code)
    if dtype is None:
        raise ValueError(f"Unsupported dtype code: {dtype_code}")

    body_start = shape_end
    if flags & FLAG_ALIGNED:
        body_start += (_ALIGNMENT - (shape_end % _ALIGNMENT)) % _ALIGNMENT

    body_len = (
        (int(np.prod(shape)) * dtype.itemsize)
        if not (flags & FLAG_COMPRESSION)
        else (len(mv) - body_start - (8 if flags & FLAG_INTEGRITY else 0))
    )
    body_data = mv[body_start : body_start + body_len]

    if flags & FLAG_INTEGRITY:
        expected = struct.unpack(
            "<Q", mv[body_start + body_len : body_start + body_len + 8]
        )[0]
        if xxhash.xxh3_64_intdigest(body_data) != expected:
            raise ValueError("Integrity check failed: XXH3 mismatch")

    if flags & FLAG_COMPRESSION:
        body_data = lz4.frame.decompress(body_data)

    arr = np.frombuffer(body_data, dtype=dtype, count=int(np.prod(shape))).reshape(
        shape
    )
    if copy:
        return arr.copy()
    arr.flags.writeable = False
    return arr


def dump(
    tensor: np.ndarray,
    fp: BinaryIO,
    strict: bool = False,
    check_integrity: bool = False,
) -> None:
    """Serialize a tensor and write it to a binary file.

    Parameters
    ----------
    tensor : np.ndarray
        The tensor to serialize.
    fp : BinaryIO
        The binary file pointer to write to.
    strict : bool, optional
        Whether to use strict mode. Default is False.
    check_integrity : bool, optional
        Whether to check integrity. Default is False.

    Returns
    -------
    None
    """
    write_stream(tensor, fp, strict=strict, check_integrity=check_integrity)


def load(fp: BinaryIO, mmap_mode: bool = False, copy: bool = False) -> Any:
    """Deserialize a tensor from a binary file.

    Parameters
    ----------
    fp : BinaryIO
        The binary file pointer to read from.
    mmap_mode : bool, optional
        Whether to use memory mapping. Default is False.
    copy : bool, optional
        Whether to copy the data. Default is False.

    Returns
    -------
    Any
        The deserialized tensor.
    """
    if mmap_mode:
        mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
        return loads(mm, copy=copy)
    result = read_stream(fp)
    if result is None:
        raise EOFError("Empty file or stream")
    return result
