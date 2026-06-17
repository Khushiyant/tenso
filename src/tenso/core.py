"""
Core Serialization Engine for Tenso.

This module provides high-performance functions for converting NumPy arrays,
Sparse matrices, and Dictionaries to the Tenso binary format. It supports
zero-copy memory mapping, LZ4 compression, and XXH3 integrity verification.
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
    _HEADER_BASE_V3,
    _HEADER_BASE_V4,
    _MAGIC,
    _REV_DTYPE_MAP,
    _VERSION,
    FLAG_ALIGNED,
    FLAG_BUNDLE,
    FLAG_COMPRESSION,
    FLAG_CUST_ALIGN,
    FLAG_INTEGRITY,
    FLAG_SPARSE,
    FLAG_SPARSE_CSC,
    FLAG_SPARSE_CSR,
    MAX_ELEMENTS,
    MAX_NDIM,
    QDTYPE_QINT4,
    QDTYPE_QINT8,
    QDTYPE_QUINT4,
    QDTYPE_QUINT8,
)
from .quantize import QuantizedTensor

# --- OPTIONAL DEPENDENCIES ---
try:
    import lz4.frame

    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False

# --- RUST CORE INTEGRATION ---
try:
    # Attempt to import the optimized Rust backend
    from .tenso_rs import dumps_rs, loads_rs, dump_to_fd_rs

    HAS_RUST = True
except ImportError:
    # Fallback to pure Python if the extension isn't compiled
    HAS_RUST = False

IS_LITTLE_ENDIAN = sys.byteorder == "little"

_QUANTIZED_CODES = (QDTYPE_QINT8, QDTYPE_QUINT8, QDTYPE_QINT4, QDTYPE_QUINT4)
_4BIT_CODES = (QDTYPE_QINT4, QDTYPE_QUINT4)

# The wire format encodes each dimension as a 32-bit integer.
_MAX_DIM = 0xFFFFFFFF


def _validate_dims(shape) -> None:
    """Reject shapes whose dimensions overflow the wire format's u32 dim slots.

    Without this guard each dim was silently truncated to 32 bits, corrupting
    the array on reserialization (issue #4).
    """
    for d in shape:
        if d > _MAX_DIM:
            raise ValueError(
                f"Dimension {d} exceeds the wire-format limit of {_MAX_DIM} "
                "(each dimension is stored as a 32-bit integer)"
            )


def _aligned_empty(shape, dtype, alignment: int) -> np.ndarray:
    """Allocate an array whose data pointer is a multiple of ``alignment``.

    NumPy gives no alignment guarantee stronger than the element size, so we
    over-allocate a raw byte buffer and carve out an aligned, correctly typed
    view of it.
    """
    count = int(np.prod(shape)) if len(shape) else 1
    nbytes = count * dtype.itemsize
    buf = np.empty(nbytes + alignment, dtype=np.uint8)
    offset = (-buf.ctypes.data) % alignment
    return buf[offset : offset + nbytes].view(dtype).reshape(shape)


def _ensure_aligned(obj, alignment: int, copy: bool):
    """Guarantee ``obj``'s arrays are ``alignment``-aligned in memory.

    A zero-copy ``np.frombuffer`` view inherits the alignment of the caller's
    transport buffer, which Tenso does not control, so the "aligned" promise did
    not hold for the returned array (issue #5). Here we keep the zero-copy view
    whenever it already happens to be aligned and only fall back to an aligned
    copy otherwise. ``copy=True`` always yields a writeable aligned copy.
    """
    if isinstance(obj, dict):
        return {k: _ensure_aligned(v, alignment, copy) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        if alignment <= 1 or obj.nbytes == 0:
            return obj.copy() if copy else obj
        if not copy and obj.ctypes.data % alignment == 0:
            return obj
        out = _aligned_empty(obj.shape, obj.dtype, alignment)
        out[...] = obj
        out.flags.writeable = copy
        return out
    # sparse matrices, QuantizedTensor, etc. handle their own copy semantics.
    return obj


def _target_alignment(mv, flags: int, header_base: int, ndim: int) -> int:
    """Resolve the alignment a packet's arrays should be guaranteed to have."""
    if flags & FLAG_CUST_ALIGN:
        pos = header_base + ndim * 4
        if len(mv) > pos:
            return 1 << mv[pos]
        return _ALIGNMENT
    if flags & FLAG_ALIGNED:
        return _ALIGNMENT
    # Bundle sub-packets carry their own alignment; default dumps uses 64.
    if flags & FLAG_BUNDLE:
        return _ALIGNMENT
    return 1


def _quantized_body_size(dtype_code: int, num_elements: int) -> int:
    if dtype_code in _4BIT_CODES:
        return (num_elements + 1) // 2
    return num_elements


def _serialize_quantized(
    qt: "QuantizedTensor", check_integrity: bool, alignment: int
) -> memoryview:
    """Serialize a QuantizedTensor to a Tenso packet."""
    import math

    shape = qt.shape
    ndim = len(shape)
    num_elements = int(np.prod(shape))

    use_custom_align = alignment != 64
    flags = 0
    if use_custom_align:
        flags |= FLAG_CUST_ALIGN
    else:
        flags |= FLAG_ALIGNED
    if check_integrity:
        flags |= FLAG_INTEGRITY

    # Quant metadata: 1 byte scheme + 1 byte axis + 4 bytes group_size
    #                  + 4 bytes num_scales + num_scales*4 scales
    #                  + num_scales*4 zero_points
    num_scales = len(qt.scales)
    quant_meta_len = 1 + 1 + 4 + 4 + num_scales * 4 + num_scales * 4

    header_len = _HEADER_BASE_V4 + ndim * 4 + quant_meta_len
    if use_custom_align:
        header_len += 1

    remainder = header_len % alignment
    padding_len = 0 if remainder == 0 else (alignment - remainder)

    body_len = _quantized_body_size(qt.dtype_code, num_elements)
    footer_len = 8 if check_integrity else 0
    total_len = header_len + padding_len + body_len + footer_len

    buffer = bytearray(total_len)

    # v4 Header: magic(4) + ver(1) + flags_u16(2) + dtype(1) + ndim(1) + reserved(1)
    struct.pack_into("<4sBHBBB", buffer, 0, _MAGIC, _VERSION, flags, qt.dtype_code, ndim, 0)
    struct.pack_into(f"<{ndim}I", buffer, _HEADER_BASE_V4, *shape)

    cursor = _HEADER_BASE_V4 + ndim * 4
    meta_start = cursor

    # Quant metadata
    buffer[cursor] = qt.quant_scheme
    cursor += 1
    buffer[cursor] = qt.axis & 0xFF
    cursor += 1
    struct.pack_into("<I", buffer, cursor, qt.group_size)
    cursor += 4
    struct.pack_into("<I", buffer, cursor, num_scales)
    cursor += 4
    # Scales
    scales_bytes = qt.scales.tobytes()
    buffer[cursor : cursor + len(scales_bytes)] = scales_bytes
    cursor += len(scales_bytes)
    # Zero points
    zp_bytes = qt.zero_points.tobytes()
    buffer[cursor : cursor + len(zp_bytes)] = zp_bytes
    cursor += len(zp_bytes)

    # Alignment exponent
    if use_custom_align:
        buffer[cursor] = alignment.bit_length() - 1
        cursor += 1

    # Body
    body_start = header_len + padding_len
    buffer[body_start : body_start + body_len] = bytes(qt.data[:body_len])

    # Integrity footer — covers the quant metadata (scheme/axis/group_size/
    # scales/zero_points) as well as the packed body, so corruption of the
    # scales or zero points is also detected.
    if check_integrity:
        protected = bytes(buffer[meta_start : meta_start + quant_meta_len]) + bytes(
            buffer[body_start : body_start + body_len]
        )
        digest = xxhash.xxh3_64_intdigest(protected)
        struct.pack_into("<Q", buffer, body_start + body_len, digest)

    return memoryview(buffer)


def _deserialize_quantized(
    mv: memoryview, flags: int, dtype_code: int, ndim: int, copy: bool
) -> "QuantizedTensor":
    """Deserialize a QuantizedTensor from a Tenso packet memoryview."""
    base = _HEADER_BASE_V4 if mv[4] >= 4 else _HEADER_BASE_V3
    shape_end = base + ndim * 4
    shape = struct.unpack(f"<{ndim}I", mv[base:shape_end])
    num_elements = int(np.prod(shape))

    cursor = shape_end
    meta_start = cursor

    # Quant metadata
    quant_scheme = mv[cursor]
    cursor += 1
    axis = mv[cursor]
    cursor += 1
    group_size = struct.unpack("<I", mv[cursor : cursor + 4])[0]
    cursor += 4
    num_scales = struct.unpack("<I", mv[cursor : cursor + 4])[0]
    cursor += 4
    scales = np.frombuffer(
        mv[cursor : cursor + num_scales * 4], dtype=np.float32
    ).copy()
    cursor += num_scales * 4
    zero_points = np.frombuffer(
        mv[cursor : cursor + num_scales * 4], dtype=np.float32
    ).copy()
    cursor += num_scales * 4
    meta_len = cursor - meta_start

    # Alignment
    use_custom_align = (flags & FLAG_CUST_ALIGN) != 0
    if use_custom_align:
        exponent = mv[cursor]
        alignment = 1 << exponent
        cursor += 1
    elif flags & FLAG_ALIGNED:
        alignment = _ALIGNMENT
    else:
        alignment = 1

    header_len = cursor
    remainder = header_len % alignment
    padding_len = 0 if remainder == 0 else (alignment - remainder)
    body_start = header_len + padding_len

    body_len = _quantized_body_size(dtype_code, num_elements)
    footer_len = 8 if (flags & FLAG_INTEGRITY) else 0

    # Integrity check — covers quant metadata + packed body.
    if flags & FLAG_INTEGRITY:
        protected = bytes(mv[meta_start : meta_start + meta_len]) + bytes(
            mv[body_start : body_start + body_len]
        )
        expected = struct.unpack(
            "<Q", mv[body_start + body_len : body_start + body_len + 8]
        )[0]
        if xxhash.xxh3_64_intdigest(protected) != expected:
            raise ValueError("Integrity check failed: XXH3 mismatch")

    data = np.frombuffer(mv[body_start : body_start + body_len], dtype=np.uint8)
    if copy:
        data = data.copy()
    else:
        data = np.array(data)

    return QuantizedTensor(
        data=data,
        scales=scales,
        zero_points=zero_points,
        shape=shape,
        dtype_code=dtype_code,
        quant_scheme=quant_scheme,
        group_size=group_size,
        axis=axis,
    )


def _read_stream_quantized(
    source, flags: int, dtype_code: int, ndim: int, header_size: int = _HEADER_BASE_V4
) -> "QuantizedTensor":
    """Read a quantized tensor from a stream."""
    # Read shape
    shape_len = ndim * 4
    shape_bytes = bytearray(shape_len)
    try:
        if not _read_into_buffer(source, shape_bytes):
            raise EOFError("Stream ended during quantized shape read")
    except EOFError as e:
        raise EOFError(f"Stream ended during quantized shape read. {e}") from None
    shape = struct.unpack(f"<{ndim}I", shape_bytes)
    num_elements = int(np.prod(shape))

    # Read quant metadata header (scheme 1 + axis 1 + group_size 4 + num_scales 4 = 10 bytes)
    meta_header = bytearray(10)
    try:
        if not _read_into_buffer(source, meta_header):
            raise EOFError("Stream ended during quant metadata read")
    except EOFError as e:
        raise EOFError(f"Stream ended during quant metadata read. {e}") from None

    quant_scheme = meta_header[0]
    axis = meta_header[1]
    group_size = struct.unpack("<I", meta_header[2:6])[0]
    num_scales = struct.unpack("<I", meta_header[6:10])[0]

    # Read scales and zero_points
    sz_len = num_scales * 4 * 2  # scales + zero_points
    sz_bytes = bytearray(sz_len)
    try:
        if not _read_into_buffer(source, sz_bytes):
            raise EOFError("Stream ended during scales/zero_points read")
    except EOFError as e:
        raise EOFError(f"Stream ended during scales/zero_points read. {e}") from None

    scales = np.frombuffer(sz_bytes[: num_scales * 4], dtype=np.float32).copy()
    zero_points = np.frombuffer(sz_bytes[num_scales * 4 :], dtype=np.float32).copy()

    # Current position for alignment calculation
    current_pos = header_size + shape_len + 10 + sz_len

    # Alignment
    use_custom_align = (flags & FLAG_CUST_ALIGN) != 0
    if use_custom_align:
        exp_buf = bytearray(1)
        try:
            if not _read_into_buffer(source, exp_buf):
                raise EOFError("Stream ended during alignment byte read")
        except EOFError as e:
            raise EOFError(f"Stream ended during alignment byte read. {e}") from None
        alignment = 1 << exp_buf[0]
        current_pos += 1
    elif flags & FLAG_ALIGNED:
        alignment = _ALIGNMENT
    else:
        alignment = 1

    remainder = current_pos % alignment
    padding_len = 0 if remainder == 0 else (alignment - remainder)
    body_len = _quantized_body_size(dtype_code, num_elements)
    footer_len = 8 if (flags & FLAG_INTEGRITY) else 0

    # Read padding + body + footer
    total_remaining = padding_len + body_len + footer_len
    data_buffer = bytearray(total_remaining)
    try:
        if not _read_into_buffer(source, data_buffer):
            raise EOFError("Stream ended during quantized body read")
    except EOFError as e:
        raise EOFError(f"Stream ended during quantized body read. {e}") from None

    # Verify integrity — covers quant metadata + packed body (must match the
    # in-memory serializer's coverage).
    if footer_len > 0:
        body_slice = data_buffer[padding_len : padding_len + body_len]
        protected = bytes(meta_header) + bytes(sz_bytes) + bytes(body_slice)
        actual_hash = xxhash.xxh3_64_intdigest(protected)
        expected_hash = struct.unpack(
            "<Q", data_buffer[padding_len + body_len : padding_len + body_len + 8]
        )[0]
        if actual_hash != expected_hash:
            raise ValueError("Integrity check failed: XXH3 mismatch")

    data = np.frombuffer(
        data_buffer, dtype=np.uint8, offset=padding_len, count=body_len
    ).copy()

    return QuantizedTensor(
        data=data,
        scales=scales,
        zero_points=zero_points,
        shape=shape,
        dtype_code=dtype_code,
        quant_scheme=quant_scheme,
        group_size=group_size,
        axis=axis,
    )


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

    # Cache method lookups outside the loop for performance
    readinto = getattr(source, "readinto", None)
    recv_into = getattr(source, "recv_into", None)
    recv = getattr(source, "recv", None)
    read = getattr(source, "read", None)

    pos = 0
    while pos < n:
        bytes_read = 0
        if readinto is not None:
            bytes_read = readinto(view[pos:])
        elif recv_into is not None:
            try:
                bytes_read = recv_into(view[pos:])
            except BlockingIOError:
                continue
        else:
            remaining = n - pos
            chunk = recv(remaining) if recv is not None else read(remaining)

            if chunk:
                view[pos : pos + len(chunk)] = chunk
                bytes_read = len(chunk)
            else:
                bytes_read = 0

        if bytes_read == 0:
            if pos == 0:
                return False
            raise EOFError(f"Expected {n} bytes, got {pos}")

        pos += bytes_read
    return True


def _parse_header(mv: memoryview) -> "tuple[int, int, int, int, int]":
    """
    Parse a Tenso packet header from a memoryview.

    Returns ``(version, flags, dtype_code, ndim, header_size)``. Supports
    v3 (8-byte header, ``flags_u8``) and v4 (10-byte header, ``flags_u16``).

    Raises
    ------
    ValueError
        If the buffer is too short, the magic is wrong, or the version is
        unsupported.
    """
    if len(mv) < _HEADER_BASE_V3:
        raise ValueError("Packet too short")
    if bytes(mv[:4]) != _MAGIC:
        raise ValueError("Invalid tenso packet")

    ver = mv[4]
    if ver == 4:
        if len(mv) < _HEADER_BASE_V4:
            raise ValueError("Packet too short for v4 header")
        flags = struct.unpack_from("<H", mv, 5)[0]
        return ver, flags, mv[7], mv[8], _HEADER_BASE_V4
    if ver == 3:
        return ver, mv[5], mv[6], mv[7], _HEADER_BASE_V3
    raise ValueError(f"Unsupported protocol version: {ver}")


def _read_packet_header(source: Any) -> Optional["tuple[int, int, int, int, int]"]:
    """
    Read a Tenso packet header from a synchronous stream.

    Returns ``(version, flags, dtype_code, ndim, header_size)``, or ``None``
    if the stream is at EOF before any byte is read. Reads the additional
    2 bytes for v4 packets transparently.
    """
    header = bytearray(_HEADER_BASE_V3)
    try:
        if not _read_into_buffer(source, header):
            return None
    except EOFError as e:
        raise EOFError(f"Stream ended during header read. {e}") from None

    if header[:4] != _MAGIC:
        raise ValueError("Invalid tenso packet")

    ver = header[4]
    if ver == 4:
        extra = bytearray(_HEADER_BASE_V4 - _HEADER_BASE_V3)
        try:
            if not _read_into_buffer(source, extra):
                raise EOFError("Stream ended during v4 header read")
        except EOFError as e:
            raise EOFError(f"Stream ended during v4 header read. {e}") from None
        full = bytes(header) + bytes(extra)
        return ver, struct.unpack_from("<H", full, 5)[0], full[7], full[8], _HEADER_BASE_V4
    if ver == 3:
        return ver, header[5], header[6], header[7], _HEADER_BASE_V3
    raise ValueError(f"Unsupported protocol version: {ver}")


async def _aread_packet_header(reader) -> Optional["tuple[int, int, int, int, int]"]:
    """
    Async counterpart to :func:`_read_packet_header`.

    Returns ``None`` on clean stream end (no bytes available); otherwise
    returns ``(version, flags, dtype_code, ndim, header_size)``.
    """
    import asyncio
    try:
        header = await reader.readexactly(_HEADER_BASE_V3)
    except asyncio.IncompleteReadError as e:
        if len(e.partial) == 0:
            return None
        raise

    if header[:4] != _MAGIC:
        raise ValueError("Invalid tenso packet")

    ver = header[4]
    if ver == 4:
        extra = await reader.readexactly(_HEADER_BASE_V4 - _HEADER_BASE_V3)
        full = header + extra
        return ver, struct.unpack_from("<H", full, 5)[0], full[7], full[8], _HEADER_BASE_V4
    if ver == 3:
        return ver, header[5], header[6], header[7], _HEADER_BASE_V3
    raise ValueError(f"Unsupported protocol version: {ver}")


def read_stream(source: Any) -> Optional[Any]:
    """
    Read and deserialize an object from a stream source with DoS protection.

    This function supports streaming deserialization for dense NumPy arrays,
    multi-tensor bundles (dictionaries), and sparse matrices (COO, CSR, CSC).
    It avoids loading the entire packet into memory before parsing, making it
    suitable for large-scale data ingestion.

    Parameters
    ----------
    source : Any
        Stream source to read from (must support .read() or .recv()).

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
    ImportError
        If scipy is missing during sparse matrix deserialization.
    """
    parsed = _read_packet_header(source)
    if parsed is None:
        return None
    _ver, flags, dtype_code, ndim, header_size = parsed

    # 2. Handle Bundle (Dictionaries)
    if flags & FLAG_BUNDLE:
        res = {}
        # ndim stores the number of items for bundles (up to 255)
        for _ in range(ndim):
            # Read Key Length
            k_len_buf = bytearray(4)
            try:
                if not _read_into_buffer(source, k_len_buf):
                    raise EOFError("Stream ended during bundle key length read")
            except EOFError as e:
                raise EOFError(
                    f"Stream ended during bundle key length read. {e}"
                ) from None
            k_len = struct.unpack("<I", k_len_buf)[0]

            # Read Key
            key_buf = bytearray(k_len)
            try:
                if not _read_into_buffer(source, key_buf):
                    raise EOFError("Stream ended during bundle key read")
            except EOFError as e:
                raise EOFError(f"Stream ended during bundle key read. {e}") from None
            key = key_buf.decode("utf-8")

            # Read Value Packet Length prefix (4 bytes)
            v_len_buf = bytearray(4)
            try:
                if not _read_into_buffer(source, v_len_buf):
                    raise EOFError("Stream ended during bundle value length read")
            except EOFError as e:
                raise EOFError(
                    f"Stream ended during bundle value length read. {e}"
                ) from None

            # Recursively read the nested Tenso packet
            res[key] = read_stream(source)
        return res

    # 3. Handle Sparse Formats (COO, CSR, CSC)
    if flags & (FLAG_SPARSE | FLAG_SPARSE_CSR | FLAG_SPARSE_CSC):
        try:
            from scipy import sparse
        except ImportError:
            raise ImportError("scipy is required for sparse deserialization.")

        # Read Shape
        shape_len = ndim * 4
        shape_bytes = bytearray(shape_len)
        try:
            if not _read_into_buffer(source, shape_bytes):
                raise EOFError("Stream ended during sparse shape read")
        except EOFError as e:
            raise EOFError(f"Stream ended during sparse shape read. {e}") from None
        shape = struct.unpack(f"<{ndim}I", shape_bytes)

        # Read 3 sub-packets (data, indices/row, indptr/col)
        sub_objs = []
        for i, label in enumerate(["data", "indices/row", "indptr/col"]):
            v_len_buf = bytearray(4)
            try:
                if not _read_into_buffer(source, v_len_buf):
                    raise EOFError(f"Stream ended during sparse {label} length read")
            except EOFError as e:
                raise EOFError(
                    f"Stream ended during sparse {label} length read. {e}"
                ) from None
            sub_objs.append(read_stream(source))

        c1, c2, c3 = sub_objs
        if flags & FLAG_SPARSE:
            return sparse.coo_matrix((c1, (c2, c3)), shape=shape)
        if flags & FLAG_SPARSE_CSR:
            return sparse.csr_matrix((c1, c2, c3), shape=shape)
        return sparse.csc_matrix((c1, c2, c3), shape=shape)

    # 3.5 Handle Quantized Types
    if dtype_code in _QUANTIZED_CODES:
        return _read_stream_quantized(source, flags, dtype_code, ndim, header_size)

    # 4. Dense Array Logic (DoS Protection & Buffer Allocation)
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

    if flags & FLAG_COMPRESSION:
        # The streaming dense path computes body_len from the UNCOMPRESSED element
        # count and has no length prefix for a compressed body, so it would over-
        # read / mis-hash. Reject explicitly, matching the async and GPU readers.
        raise ValueError(
            "Cannot stream a compressed dense packet: the streaming format "
            "carries no length prefix for compressed bodies. Read the full "
            "packet and use loads() instead."
        )

    # Read Body & Padding
    current_pos = header_size + shape_len
    alignment = _ALIGNMENT
    use_custom_align = (flags & FLAG_CUST_ALIGN) != 0

    if use_custom_align:
        # Read Exponent Byte
        exp_buf = bytearray(1)
        try:
            if not _read_into_buffer(source, exp_buf):
                raise EOFError("Stream ended during alignment byte read")
        except EOFError as e:
            raise EOFError(f"Stream ended during alignment byte read. {e}") from None
        exponent = exp_buf[0]
        alignment = 1 << exponent
        current_pos += 1

    remainder = current_pos % alignment
    padding_len = 0 if remainder == 0 else (alignment - remainder)
    body_len = num_elements * dtype.itemsize
    footer_len = 8 if (flags & FLAG_INTEGRITY) else 0

    data_buffer = np.empty(padding_len + body_len + footer_len, dtype=np.uint8)
    try:
        if not _read_into_buffer(source, data_buffer):
            raise EOFError("Stream ended during body read")
    except EOFError as e:
        raise EOFError(f"Stream ended during body read. {e}") from None

    # Verify Integrity
    if footer_len > 0:
        body_slice = data_buffer[padding_len : padding_len + body_len]
        actual_hash = xxhash.xxh3_64_intdigest(body_slice)
        expected_hash = struct.unpack("<Q", data_buffer[padding_len + body_len :])[0]
        if actual_hash != expected_hash:
            raise ValueError("Integrity check failed: XXH3 mismatch")

    arr = np.frombuffer(
        data_buffer, dtype=dtype, offset=padding_len, count=num_elements
    ).reshape(shape)
    arr = _ensure_aligned(arr, alignment, copy=False)
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
    # Handle QuantizedTensor by delegating to dumps
    if isinstance(tensor, QuantizedTensor):
        yield bytes(dumps(tensor, check_integrity=check_integrity))
        return

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
    header = struct.pack("<4sBHBBB", _MAGIC, _VERSION, flags, dtype_code, ndim, 0)
    shape_block = struct.pack(f"<{ndim}I", *shape)
    yield header
    yield shape_block

    current_len = _HEADER_BASE_V4 + (ndim * 4)
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
    Supports both file-like objects (.write) and sockets (.sendall).

    Parameters
    ----------
    tensor : np.ndarray
        The array to serialize.
    dest : Any
        Destination supporting .write() or .sendall().
    strict : bool, default False
        Strict contiguous check.
    check_integrity : bool, default False
        Include integrity hash.

    Returns
    -------
    int
        The total number of bytes written.
    """
    # Determine the correct method for writing (cache lookup before iteration)
    write_method = getattr(dest, "sendall", None) or getattr(dest, "write", None)
    if write_method is None:
        raise AttributeError(
            f"Destination {type(dest)} has no '.write' or '.sendall' method."
        )

    # Stream directly from generator without materializing to list
    written = 0
    for chunk in iter_dumps(tensor, strict=strict, check_integrity=check_integrity):
        write_method(chunk)
        written += len(chunk)
    return written


def _bundle_contains_quantized(d: dict) -> bool:
    """True if a bundle dict holds a QuantizedTensor at any nesting depth.

    The Rust bundle encoder recurses into nested dicts, so a QuantizedTensor below
    the top level is just as unsupported there as one at the top level.
    """
    for v in d.values():
        if isinstance(v, QuantizedTensor):
            return True
        if isinstance(v, dict) and _bundle_contains_quantized(v):
            return True
    return False


def _bundle_make_contiguous(d: dict, strict: bool) -> dict:
    """Return a copy of a bundle dict with every nested numpy array made
    C-contiguous, recursing into nested dicts.

    The Rust dense path reads each array's bytes at ``ctypes.data`` for ``nbytes``
    assuming C-order; a non-contiguous value (transpose/slice/Fortran-order) would
    otherwise be serialized as corrupt data. Sparse and other values pass through
    unchanged (the Rust sparse path coerces its own components).
    """
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            if not v.flags["C_CONTIGUOUS"]:
                if strict:
                    raise ValueError("Tensor is not C-Contiguous")
                v = np.ascontiguousarray(v)
        elif isinstance(v, dict):
            v = _bundle_make_contiguous(v, strict)
        out[k] = v
    return out


def dumps(
    tensor: Any,
    strict: bool = False,
    check_integrity: bool = False,
    compress: bool = False,
    alignment: int = 64,
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
    alignment : int, default 64
        Memory alignment boundary (must be power of 2).

    Returns
    -------
    memoryview
        A view of the complete Tenso packet bytes.
    """
    if not (alignment > 0 and (alignment & (alignment - 1) == 0)):
        raise ValueError("Alignment must be a power of two")

    # Reject oversized dims up front so the error is a clean ValueError on both
    # the Rust and Python paths instead of a silent truncation (issue #4).
    if isinstance(tensor, np.ndarray):
        _validate_dims(tensor.shape)
    elif hasattr(tensor, "shape") and not isinstance(tensor, dict):
        _validate_dims(tensor.shape)

    # 1. FAST PATH: RUST ACCELERATION
    if HAS_RUST and not compress:
        try:
            # Check if this type is handled by Rust
            is_numpy = isinstance(tensor, np.ndarray)
            is_sparse = hasattr(tensor, "format")
            is_dict = isinstance(tensor, dict)

            # Skip Rust for any bundle that holds a QuantizedTensor at ANY depth:
            # the Rust dense path cannot introspect a QuantizedTensor (it has no
            # `dtype`/`ctypes` surface) and would raise; only the Python path
            # serializes quantized values. A top-level-only check missed nested
            # dicts like {"a": {"b": qt}}.
            if is_dict and _bundle_contains_quantized(tensor):
                is_dict = False
                is_numpy = False
                is_sparse = False

            if is_numpy or is_sparse or is_dict:
                if is_numpy and not tensor.flags["C_CONTIGUOUS"]:
                    if strict:
                        raise ValueError("Tensor is not C-Contiguous")
                    tensor = np.ascontiguousarray(tensor)

                # For a bundle, the Rust path reads each numpy value's raw bytes at
                # ctypes.data assuming C-contiguity; a non-contiguous nested value
                # (transposed/sliced/Fortran-order) would otherwise be serialized as
                # corrupt data. Normalize every nested array to C-order first.
                if is_dict:
                    tensor = _bundle_make_contiguous(tensor, strict)

                # dumps_rs handles Dense, Sparse, and Bundles
                return memoryview(dumps_rs(tensor, check_integrity=check_integrity, alignment=alignment))
        # AttributeError: the Rust path tried to introspect a value it does not
        # support (e.g. a nested custom type) — fall back to the Python path
        # rather than crashing.
        except (TypeError, ValueError, AttributeError):
            pass

    # 1.5. Quantized Tensor (always Python path)
    if isinstance(tensor, QuantizedTensor):
        return _serialize_quantized(tensor, check_integrity, alignment)

    # 2. Multi-tensor Bundle (Dictionaries) - Python Fallback
    if isinstance(tensor, dict):
        if len(tensor) > 255:
            raise ValueError(
                f"Bundle has {len(tensor)} entries; the wire format encodes the "
                "entry count in a single byte, so at most 255 entries are supported"
            )
        parts = []
        header = struct.pack(
            "<4sBHBBB", _MAGIC, _VERSION, FLAG_BUNDLE, 0, len(tensor), 0
        )
        parts.append(header)
        for key, value in tensor.items():
            key_bytes = key.encode("utf-8")
            parts.append(struct.pack("<I", len(key_bytes)) + key_bytes)
            val_packet = dumps(value, strict, check_integrity, compress, alignment)
            parts.append(struct.pack("<I", len(val_packet)) + val_packet)
        return memoryview(b"".join(parts))

    # 3. Sparse Formats (COO, CSR, CSC) - Python Fallback
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
        header = struct.pack("<4sBHBBB", _MAGIC, _VERSION, flag, 0, len(tensor.shape), 0)
        shape_block = struct.pack(f"<{len(tensor.shape)}I", *tensor.shape)

        sub_pkts = []
        for c in comps:
            sp = dumps(c, strict, False, False, alignment)
            sub_pkts.append(struct.pack("<I", len(sp)) + sp)
        return memoryview(b"".join([header, shape_block] + sub_pkts))

    # 4. Standard Python Implementation (Fallback for Dense)
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
    
    flags = 0
    use_custom_align = alignment != 64
    if use_custom_align:
        flags |= FLAG_CUST_ALIGN
    else:
        flags |= FLAG_ALIGNED
    
    if check_integrity:
        flags |= FLAG_INTEGRITY

    if compress:
        if not HAS_LZ4:
            raise ImportError("Compression requires 'lz4' package.")
        body = lz4.frame.compress(body)
        flags |= FLAG_COMPRESSION

    current_len = _HEADER_BASE_V4 + (ndim * 4)
    if use_custom_align:
        current_len += 1

    padding_len = 0
    remainder = current_len % alignment
    if remainder != 0:
        padding_len = alignment - remainder

    total_len = current_len + padding_len + len(body) + (8 if check_integrity else 0)

    buffer = bytearray(total_len)
    struct.pack_into("<4sBHBBB", buffer, 0, _MAGIC, _VERSION, flags, dtype_code, ndim, 0)
    struct.pack_into(f"<{ndim}I", buffer, _HEADER_BASE_V4, *shape)

    cursor = _HEADER_BASE_V4 + (ndim * 4)
    if use_custom_align:
        buffer[cursor] = alignment.bit_length() - 1
        cursor += 1
    
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

    # Resolve the alignment the packet promises so it can be guaranteed on the
    # returned arrays regardless of which decode path runs (issue #5). A
    # zero-copy view only inherits the transport buffer's alignment, which Tenso
    # does not control, so the guarantee is enforced after decoding.
    try:
        _hdr = _parse_header(mv)
        alignment = _target_alignment(mv, _hdr[1], _hdr[4], _hdr[3])
    except Exception:
        _hdr = None
        alignment = _ALIGNMENT

    # 0. FAST PATH: RUST ACCELERATION
    if HAS_RUST and (hasattr(data, "__buffer__") or isinstance(data, (bytes, bytearray, memoryview, np.ndarray, mmap.mmap))):
        # loads_rs returns None if flags are unsupported (e.g. Bundle/Sparse/Compression)
        # It raises ValueError if the packet is malformed or integrity check fails.
        try:
            res = loads_rs(data)
            if res is not None:
                return _ensure_aligned(res, alignment, copy)
        except (ValueError, TypeError):
             # Fallback to Python if Rust fails for any reason
             pass

    _ver, flags, dtype_code, ndim, header_base = (
        _hdr if _hdr is not None else _parse_header(mv)
    )

    # 1. Bundle Deserialization
    if flags & FLAG_BUNDLE:
        res = {}
        offset = header_base
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

        shape_end = header_base + (ndim * 4)
        shape = struct.unpack(f"<{ndim}I", mv[header_base:shape_end])
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

    # 2.5. Quantized Types
    if dtype_code in _QUANTIZED_CODES:
        return _deserialize_quantized(mv, flags, dtype_code, ndim, copy)

    # 3. Dense Array Deserialization
    if ndim > MAX_NDIM:
        raise ValueError(f"Packet exceeds maximum dimensions ({ndim} > {MAX_NDIM})")

    shape_end = header_base + (ndim * 4)
    shape = struct.unpack(f"<{ndim}I", mv[header_base:shape_end])
    num_elements = int(np.prod(shape))
    if num_elements > MAX_ELEMENTS:
        raise ValueError("Packet exceeds maximum elements")

    dtype = _REV_DTYPE_MAP.get(dtype_code)
    if dtype is None:
        raise ValueError(f"Unsupported dtype code: {dtype_code}")

    body_start = shape_end

    use_custom_align = (flags & FLAG_CUST_ALIGN) != 0
    alignment = 1

    if use_custom_align:
        if len(mv) < body_start + 1:
            raise ValueError("Packet too short (alignment byte)")
        exponent = mv[body_start]
        alignment = 1 << exponent
        body_start += 1
    elif flags & FLAG_ALIGNED:
        alignment = _ALIGNMENT

    remainder = body_start % alignment
    if remainder != 0:
        body_start += (alignment - remainder)

    body_len = (
        (num_elements * dtype.itemsize)
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

    arr = np.frombuffer(body_data, dtype=dtype, count=num_elements).reshape(shape)
    arr = _ensure_aligned(arr, alignment, copy)
    if not copy:
        arr.flags.writeable = False
    return arr


def dump(
    tensor: np.ndarray,
    fp: BinaryIO,
    strict: bool = False,
    check_integrity: bool = False,
) -> None:
    """
    Serialize a tensor and write it to an open binary file.
    
    Optimized for large arrays by writing the complete packet in a single
    system call instead of multiple small writes.
    
    Parameters
    ----------
    tensor : np.ndarray
        The array to serialize.
    fp : BinaryIO
        Open binary file object.
    strict : bool, default False
        If True, raises error for non-contiguous arrays.
    check_integrity : bool, default False
        If True, includes XXH3 hash for verification.
    
    Returns
    -------
    None
    """
    # 0. FAST PATH: ZERO-ALLOCATION STREAMING (RUST)
    if HAS_RUST and hasattr(fp, "fileno"):
        try:
            fd = fp.fileno()
            # Ensure contiguous for Rust
            if not tensor.flags["C_CONTIGUOUS"]:
                if strict:
                    raise ValueError("Tensor is not C-Contiguous")
                # Note: ascontiguousarray makes a copy, but it's unavoidable if data is not contiguous
                tensor = np.ascontiguousarray(tensor)
            
            # Write directly to FD without allocating PyBytes
            dump_to_fd_rs(tensor, fd, check_integrity=check_integrity)
            return
        except (ValueError, TypeError, AttributeError, OSError):
            # Fallback if fileno() is unavailable/invalid (e.g. BytesIO) or Rust fails
            pass

    # Use dumps() to create complete packet, then single write
    # This is 6x faster than chunked writes for large arrays
    packet = dumps(tensor, strict=strict, check_integrity=check_integrity)
    fp.write(packet)


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