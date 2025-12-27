"""
Core Serialization Engine for Tenso.

This module provides high-performance functions for converting NumPy arrays,
Sparse matrices, and Dictionaries to the Tenso binary format. It supports
zero-copy memory mapping, LZ4 compression, and XXH3 integrity verification.
"""

import struct
import numpy as np
import xxhash
import sys
import mmap
from typing import BinaryIO, Union, Any, Generator, Optional
from .config import (
    _MAGIC,
    _VERSION,
    _ALIGNMENT,
    _DTYPE_MAP,
    _REV_DTYPE_MAP,
    FLAG_ALIGNED,
    FLAG_INTEGRITY,
    FLAG_COMPRESSION,
    FLAG_SPARSE,
    FLAG_BUNDLE,
    FLAG_SPARSE_CSR,
    FLAG_SPARSE_CSC,
    MAX_NDIM,
    MAX_ELEMENTS,
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
    """
    Fill a buffer from a source, handling various I/O types.

    Parameters
    ----------
    source : Any
        The data source to read from (e.g., file, socket, BytesIO).
    buf : Union[bytearray, memoryview, np.ndarray]
        The buffer to fill with data.

    Returns
    -------
    bool
        True if the buffer was filled completely, False if the stream ended
        before any data was read.

    Raises
    ------
    EOFError
        If the source ends prematurely after partial data has been read.
    """
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


def read_stream(source: Any) -> Optional[Any]:
    """
    Read and deserialize an object from a stream source with DoS protection.

    Parameters
    ----------
    source : Any
        Stream source to read from (must support read() or recv()).

    Returns
    -------
    Optional[Any]
        The deserialized NumPy array, Sparse matrix, or Dictionary. Returns None
        if the stream ended before any data was read.

    Raises
    ------
    ValueError
        If the packet is invalid or exceeds security limits.
    EOFError
        If the stream ends prematurely during reading.
    """
    # 1. Read Header
    header = bytearray(8)
    try:
        if not _read_into_buffer(source, header):
            return None
    except EOFError as e:
        raise EOFError(f"Stream ended during header read. {e}") from None

    magic, ver, flags, dtype_code, ndim = struct.unpack("<4sBBBB", header)
    if magic != _MAGIC:
        raise ValueError("Invalid tenso packet")

    # Handle Bundle/Sparse via Loads (reading whole packet into memory)
    # Note: For optimized stream parsing of bundles, use a length-prefixed wrapper.
    if flags & (FLAG_BUNDLE | FLAG_SPARSE | FLAG_SPARSE_CSR | FLAG_SPARSE_CSC):
        # We need to know the total length. Since the protocol doesn't have a
        # global length in the header for non-dense types yet, we rely on loads().
        pass

    # 2. Read Shape (DoS Check)
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

    # 3. Read Body
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

    # 4. Verify Integrity
    if footer_len > 0:
        body_slice = data_buffer[padding_len : padding_len + body_len]
        actual_hash = xxhash.xxh3_64_intdigest(body_slice)
        expected_hash = struct.unpack("<Q", data_buffer[padding_len + body_len :])[0]
        if actual_hash != expected_hash:
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
        The array to serialize.
    strict : bool, default False
        If True, raises ValueError for non-contiguous arrays.
    check_integrity : bool, default False
        If True, includes an XXH3 checksum footer.

    Yields
    ------
    Union[bytes, memoryview]
        Sequential chunks of the Tenso packet.
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
    header = struct.pack("<4sBBBB", _MAGIC, _VERSION, flags, dtype_code, ndim)
    shape_block = struct.pack(f"<{ndim}I", *shape)
    yield header
    yield shape_block

    current_len = 8 + (ndim * 4)
    padding_len = (_ALIGNMENT - (current_len % _ALIGNMENT)) % _ALIGNMENT
    if padding_len > 0:
        yield b"\x00" * padding_len

    yield tensor.data

    if check_integrity:
        yield struct.pack("<Q", xxhash.xxh3_64_intdigest(tensor.data))


def write_stream(
    tensor: np.ndarray, dest: Any, strict: bool = False, check_integrity: bool = False
) -> int:
    """
    Write a tensor to a destination using memory-efficient streaming.

    Parameters
    ----------
    tensor : np.ndarray
        The array to serialize.
    dest : Any
        Destination supporting .write() (e.g., file, socket).
    strict : bool, default False
        Strict contiguous check.
    check_integrity : bool, default False
        Include integrity hash.

    Returns
    -------
    int
        The total number of bytes written.
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
    Serialize an object (Array, Sparse Matrix, or Dict) to a Tenso packet.

    Parameters
    ----------
    tensor : Any
        The object to serialize.
    strict : bool, default False
        If True, raises error for non-contiguous arrays.
    check_integrity : bool, default False
        If True, includes XXH3 hash for verification.
    compress : bool, default False
        If True, uses LZ4 compression on the data body.

    Returns
    -------
    memoryview
        A view of the complete Tenso packet bytes.
    """
    # 1. Multi-tensor Bundle (Dictionaries)
    if isinstance(tensor, dict):
        parts = []
        header = struct.pack(
            "<4sBBBB", _MAGIC, _VERSION, FLAG_BUNDLE, 0, min(len(tensor), 255)
        )
        parts.append(header)
        for key, value in tensor.items():
            key_bytes = key.encode("utf-8")
            parts.append(struct.pack("<I", len(key_bytes)) + key_bytes)
            val_packet = dumps(value, strict, check_integrity, compress)
            parts.append(struct.pack("<I", len(val_packet)) + val_packet)
        return memoryview(b"".join(parts))

    # 2. Sparse Formats (COO, CSR, CSC)
    if hasattr(tensor, "format") and not isinstance(tensor, np.ndarray):
        fmt = tensor.format
        flag = {"coo": FLAG_SPARSE, "csr": FLAG_SPARSE_CSR, "csc": FLAG_SPARSE_CSC}.get(
            fmt
        )
        if flag is None:
            raise ValueError(f"Unsupported sparse format: {fmt}")

        comps = (
            [tensor.data, tensor.row, tensor.col]
            if fmt == "coo"
            else [tensor.data, tensor.indices, tensor.indptr]
        )
        header = struct.pack("<4sBBBB", _MAGIC, _VERSION, flag, 0, len(tensor.shape))
        shape_block = struct.pack(f"<{len(tensor.shape)}I", *tensor.shape)

        sub_pkts = []
        for c in comps:
            sp = dumps(c, strict, False, False)
            sub_pkts.append(struct.pack("<I", len(sp)) + sp)
        return memoryview(b"".join([header, shape_block] + sub_pkts))

    # 3. Standard Array
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
            raise ImportError("Compression requires 'lz4' package.")
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
    """
    Deserialize a Tenso packet into its original Python object.

    Parameters
    ----------
    data : Union[bytes, bytearray, memoryview, np.ndarray, mmap.mmap]
        The raw Tenso packet data.
    copy : bool, default False
        If True, returns a writeable copy. Otherwise returns a read-only view.

    Returns
    -------
    Any
        The reconstructed NumPy array, Dictionary, or Sparse Matrix.
    """
    mv = memoryview(data)
    if len(mv) < 8:
        raise ValueError("Packet too short")
    magic, ver, flags, dtype_code, ndim = struct.unpack("<4sBBBB", mv[:8])
    if magic != _MAGIC:
        raise ValueError("Invalid tenso packet")

    # 1. Bundle Deserialization
    if flags & FLAG_BUNDLE:
        res = {}
        offset = 8
        for _ in range(ndim):
            k_len = struct.unpack("<I", mv[offset : offset + 4])[0]
            offset += 4
            key = bytes(mv[offset : offset + k_len]).decode("utf-8")
            offset += k_len
            v_len = struct.unpack("<I", mv[offset : offset + 4])[0]
            offset += 4
            res[key] = loads(mv[offset : offset + v_len], copy=copy)
            offset += v_len
        return res

    # 2. Sparse Deserialization
    if flags & (FLAG_SPARSE | FLAG_SPARSE_CSR | FLAG_SPARSE_CSC):
        try:
            from scipy import sparse
        except ImportError:
            raise ImportError("scipy is required for sparse deserialization.")

        shape_end = 8 + (ndim * 4)
        shape = struct.unpack(f"<{ndim}I", mv[8:shape_end])
        offset = shape_end
        sub_objs = []
        for _ in range(3):
            sub_len = struct.unpack("<I", mv[offset : offset + 4])[0]
            offset += 4
            sub_objs.append(loads(mv[offset : offset + sub_len], copy=copy))
            offset += sub_len

        c1, c2, c3 = sub_objs
        if flags & FLAG_SPARSE:
            return sparse.coo_matrix((c1, (c2, c3)), shape=shape)
        if flags & FLAG_SPARSE_CSR:
            return sparse.csr_matrix((c1, c2, c3), shape=shape)
        return sparse.csc_matrix((c1, c2, c3), shape=shape)

    # 3. Dense Array Deserialization
    if ndim > MAX_NDIM:
        raise ValueError(f"Packet exceeds maximum dimensions ({ndim} > {MAX_NDIM})")

    shape_end = 8 + (ndim * 4)
    shape = struct.unpack(f"<{ndim}I", mv[8:shape_end])
    if np.prod(shape) > MAX_ELEMENTS:
        raise ValueError("Packet exceeds maximum elements")

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
    """Serialize a tensor and write it to an open binary file."""
    write_stream(tensor, fp, strict=strict, check_integrity=check_integrity)


def load(fp: BinaryIO, mmap_mode: bool = False, copy: bool = False) -> Any:
    """
    Deserialize an object from an open binary file.

    Parameters
    ----------
    fp : BinaryIO
        Open binary file object.
    mmap_mode : bool, default False
        Use memory mapping for large files.
    copy : bool, default False
        Return a writeable copy.

    Returns
    -------
    Any
        The reconstructed object.
    """
    if mmap_mode:
        mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
        return loads(mm, copy=copy)
    result = read_stream(fp)
    if result is None:
        raise EOFError("Empty file or stream")
    return result
