"""
Core Serialization Engine for Tenso.

This module provides high-performance functions for converting NumPy arrays,
Sparse matrices, and Dictionaries to the Tenso binary format. It supports
zero-copy memory mapping, LZ4 compression, and XXH3 integrity verification.
"""

import io
import math
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

# Rust extension is REQUIRED: single source of truth for the wire format.
try:
    from .tenso_rs import (
        dumps_rs,
        loads_rs,
        dump_to_fd_rs,
        dumps_quantized_rs,
        dumps_string_rs,
        encode_bundle_rs,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "tenso requires its compiled Rust extension (tenso.tenso_rs), which could "
        "not be imported. Install a prebuilt wheel (`pip install tenso`) or build "
        "it with maturin (`maturin develop`). The pure-Python codec fallback was "
        "removed in favor of a single Rust core."
    ) from exc

# Always True; kept so call sites that branch on it keep working.
HAS_RUST = True

IS_LITTLE_ENDIAN = sys.byteorder == "little"

_QUANTIZED_CODES = (QDTYPE_QINT8, QDTYPE_QUINT8, QDTYPE_QINT4, QDTYPE_QUINT4)
_4BIT_CODES = (QDTYPE_QINT4, QDTYPE_QUINT4)

# Wire format encodes each dimension as u32.
_MAX_DIM = 0xFFFFFFFF


def _validate_byteorder(arr: np.ndarray) -> None:
    """Reject arrays whose dtype is not in the wire format's byte order.

    The format stores bodies little-endian, and the encoder copies raw bytes from
    ``arr.ctypes.data`` without inspecting byte order. ``_DTYPE_MAP`` is keyed on
    native dtype objects, so a byteswapped dtype like ``'>f4'`` never matched and
    fell through unchecked -- the bytes were written big-endian and read back
    little-endian, silently producing different numbers with no error raised.

    Rejecting is deliberate: converting on the caller's behalf would hide an
    allocation and a copy inside what is documented as a zero-copy write.
    """
    if arr.dtype.isnative:
        return
    raise ValueError(
        f"Cannot serialize dtype {arr.dtype!r}: tenso writes the array's raw "
        f"bytes, so the dtype must be in this platform's native byte order "
        f"({sys.byteorder}-endian). Convert first with "
        f"arr.astype(arr.dtype.newbyteorder('='))."
    )


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

    Only reached when the caller opts in (``loads(..., align=True)``) or asks for
    a writeable copy, because guaranteeing an *absolute* address is not free: a
    decoded view inherits the transport buffer's address, which Tenso does not
    own, so the guarantee costs a full copy whenever that address does not
    already line up. See :func:`loads` for why that is opt-in rather than default.
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
    # sparse, QuantizedTensor, etc. handle their own copy semantics.
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

    # Cache method lookups outside the loop.
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


def _read_full_packet(source: Any) -> Optional[bytearray]:
    """Read exactly one complete Tenso packet's bytes from a synchronous stream.

    This performs only *framing* (using the wire-format length fields); the actual
    decode is delegated to the Rust core via :func:`loads` in :func:`read_stream`.
    Mirrors the async ``_aread_full_packet``. Returns ``None`` on a clean stream
    end (no bytes available); raises ``EOFError`` on a truncated packet.
    """
    head = bytearray(_HEADER_BASE_V3)
    try:
        if not _read_into_buffer(source, head):
            return None
    except EOFError as e:
        raise EOFError(f"Stream ended during header read. {e}") from None

    if head[:4] != _MAGIC:
        raise ValueError("Invalid tenso packet")

    ver = head[4]
    if ver == 4:
        extra = bytearray(_HEADER_BASE_V4 - _HEADER_BASE_V3)
        if not _read_into_buffer(source, extra):
            raise EOFError("Stream ended during v4 header read")
        head += extra
        flags = struct.unpack_from("<H", head, 5)[0]
        dtype_code, ndim = head[7], head[8]
    elif ver == 3:
        flags, dtype_code, ndim = head[5], head[6], head[7]
    else:
        raise ValueError(f"Unsupported protocol version: {ver}")

    buf = bytearray(head)

    def _take(n: int, what: str = "packet read") -> bytes:
        b = bytearray(n)
        if n:
            try:
                ok = _read_into_buffer(source, b)
            except EOFError as e:
                raise EOFError(f"Stream ended during {what}. {e}") from None
            if not ok:
                raise EOFError(f"Stream ended during {what}")
        return bytes(b)

    # Bundle: ndim is the entry count; entries are length-prefixed.
    if flags & FLAG_BUNDLE:
        for _ in range(ndim):
            k_len_b = _take(4)
            buf += k_len_b
            buf += _take(struct.unpack("<I", k_len_b)[0])
            v_len_b = _take(4)
            buf += v_len_b
            buf += _take(struct.unpack("<I", v_len_b)[0])
        return buf

    if ndim > MAX_NDIM:
        raise ValueError(f"Packet exceeds maximum dimensions ({ndim} > {MAX_NDIM})")

    shape_b = _take(ndim * 4)
    buf += shape_b
    shape = struct.unpack(f"<{ndim}I", shape_b)
    # math.prod, not np.prod: these dims come off an untrusted header, and np.prod
    # multiplies in int64, which wraps silently and lets a hostile shape slip past
    # the MAX_ELEMENTS guard below. Python ints have no such ceiling.
    num_elements = math.prod(shape)
    if num_elements > MAX_ELEMENTS:
        raise ValueError(f"Packet exceeds maximum elements ({num_elements})")

    # Sparse: three length-prefixed sub-packets.
    if flags & (FLAG_SPARSE | FLAG_SPARSE_CSR | FLAG_SPARSE_CSC):
        for _ in range(3):
            sz_b = _take(4)
            buf += sz_b
            buf += _take(struct.unpack("<I", sz_b)[0])
        return buf

    footer_len = 8 if (flags & FLAG_INTEGRITY) else 0

    if dtype_code in _QUANTIZED_CODES:
        # Quant metadata: scheme(1)+axis(1)+group_size(4)+num_scales(4), then scales+zero_points.
        meta = _take(10)
        buf += meta
        num_scales = struct.unpack("<I", meta[6:10])[0]
        buf += _take(num_scales * 4 * 2)
        body_len = _quantized_body_size(dtype_code, num_elements)
    else:
        dtype = _REV_DTYPE_MAP.get(dtype_code)
        if dtype is None:
            raise ValueError(f"Unsupported dtype code: {dtype_code}")
        if flags & FLAG_COMPRESSION:
            raise ValueError(
                "Cannot stream a compressed dense packet: the streaming format "
                "carries no length prefix for compressed bodies. Read the full "
                "packet and use loads() instead."
            )
        body_len = num_elements * dtype.itemsize

    cursor = len(buf)
    if flags & FLAG_CUST_ALIGN:
        ab = _take(1)
        buf += ab
        alignment = 1 << ab[0]
        cursor += 1
    elif flags & FLAG_ALIGNED:
        alignment = _ALIGNMENT
    else:
        alignment = 1

    pad_len = (alignment - (cursor % alignment)) % alignment
    if pad_len:
        buf += _take(pad_len, "body read")

    buf += _take(body_len + footer_len, "body read")
    return buf


def read_stream(source: Any) -> Optional[Any]:
    """Read and deserialize one Tenso packet from a synchronous stream.

    Frames the complete packet from the stream (header + length-prefixed payload)
    and decodes it through the Rust core via :func:`loads`. Supports dense,
    bundle, sparse, and quantized packets (v3/v4, custom alignment, integrity).
    Returns ``None`` if the stream ends before any byte is read.
    """
    packet = _read_full_packet(source)
    if packet is None:
        return None
    # toreadonly() rather than bytes(): both give loads() an immutable buffer, but
    # bytes() copies the whole packet first, which on a 100 MB frame cost ~4 ms and
    # defeated the point of a zero-copy read. The returned view keeps this
    # bytearray alive, and an exported buffer blocks resizing it.
    return loads(memoryview(packet).toreadonly())


def iter_dumps(
    tensor: Any, strict: bool = False, check_integrity: bool = False
) -> Generator[Union[bytes, memoryview], None, None]:
    """Yield the Tenso packet for ``tensor`` (encoded by the Rust core).

    Retained for the streaming-write API (``write_stream`` / ``awrite_stream``);
    it now yields the single packet produced by :func:`dumps` rather than
    re-encoding in Python.
    """
    yield bytes(dumps(tensor, strict=strict, check_integrity=check_integrity))


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
            _validate_byteorder(v)
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

    # Reject oversized dims up front: clean ValueError instead of u32 truncation (issue #4).
    if isinstance(tensor, np.ndarray):
        _validate_byteorder(tensor)
        _validate_dims(tensor.shape)
        # Reject oversized element COUNT before contiguity coercion, else a huge
        # strided view forces a multi-GB ascontiguousarray just to be rejected (OOM).
        num_elements = math.prod(tensor.shape)
        if num_elements > MAX_ELEMENTS:
            raise ValueError(
                f"Packet exceeds maximum elements ({num_elements} > {MAX_ELEMENTS})"
            )
    elif hasattr(tensor, "shape") and not isinstance(tensor, dict):
        _validate_dims(tensor.shape)

    # All wire-format encoding goes through the Rust core.
    if isinstance(tensor, QuantizedTensor):
        return memoryview(
            dumps_quantized_rs(tensor, check_integrity=check_integrity, alignment=alignment)
        )

    is_numpy = isinstance(tensor, np.ndarray)
    is_sparse = hasattr(tensor, "format") and not is_numpy
    is_dict = isinstance(tensor, dict)

    # dumps_rs's dict path only handles numpy/sparse, so encode each value here
    # (routes quantized/string to Rust) and frame the bundle via encode_bundle_rs.
    if is_dict and _bundle_contains_quantized(tensor):
        if len(tensor) > 255:
            raise ValueError(
                f"Bundle has {len(tensor)} entries; the wire format encodes the "
                "entry count in a single byte, so at most 255 are supported"
            )
        entries = [
            (key, bytes(dumps(value, strict, check_integrity, compress, alignment)))
            for key, value in tensor.items()
        ]
        return memoryview(encode_bundle_rs(entries))

    if is_numpy or is_sparse or is_dict:
        if is_numpy and not tensor.flags["C_CONTIGUOUS"]:
            if strict:
                raise ValueError("Tensor is not C-Contiguous")
            tensor = np.ascontiguousarray(tensor)
        # Normalize nested bundle arrays to C-order (Rust dense path copies from
        # ctypes.data assuming C-contiguity).
        if is_dict:
            tensor = _bundle_make_contiguous(tensor, strict)
        # dumps_rs handles Dense, Sparse, and Bundles (compressed or not).
        return memoryview(
            dumps_rs(
                tensor,
                check_integrity=check_integrity,
                compress=compress,
                alignment=alignment,
            )
        )

    raise TypeError(f"Cannot serialize object of type {type(tensor).__name__!r}")




def loads(
    data: Union[bytes, bytearray, memoryview, np.ndarray, mmap.mmap],
    copy: bool = False,
    align: bool = False,
) -> Any:
    """
    Deserialize a Tenso packet into its original Python object.

    By default this is genuinely zero-copy: the returned array is a view onto
    ``data``, so nothing is allocated and nothing is copied regardless of packet
    size. The view keeps ``data`` alive, so it stays valid after you drop your own
    reference.

    Parameters
    ----------
    data : Union[bytes, bytearray, memoryview, np.ndarray, mmap.mmap]
        The raw Tenso packet data.
    copy : bool, default False
        If True, returns a writeable, alignment-guaranteed copy. Otherwise
        returns a read-only view.
    align : bool, default False
        If True, guarantee the returned array's *base address* is a multiple of
        the packet's declared alignment, copying if the transport buffer does not
        already line up. Off by default; see Notes.

    Returns
    -------
    Any
        The reconstructed NumPy array, Dictionary, or Sparse Matrix.

    Notes
    -----
    The wire format guarantees the body starts at a 64-byte *offset* from the
    packet start, so an array is aligned in memory exactly when the buffer holding
    the packet is. That holds for ``mmap`` (page-aligned), shared memory, and any
    aligned allocator — but not for a plain ``bytes`` object, whose payload sits a
    fixed 32 bytes past its allocation and is therefore never 64-byte aligned.

    Guaranteeing an absolute address there costs a full copy of the body, which is
    the one thing this format exists to avoid, and buys nothing for ordinary NumPy
    work: NumPy issues unaligned SIMD loads and imposes no alignment requirement
    beyond the element size. So the guarantee is opt-in via ``align=True``, for
    callers who need it (explicit aligned AVX-512 load/store, DMA, pinned host
    buffers for GPU transfer) rather than charged to everyone.
    """
    # All decoding goes through the Rust core; it raises ValueError on a
    # malformed/integrity-failed packet.
    res = loads_rs(data)
    if res is None:
        # Core does not surface GPU IpcRef packets (decoded on device).
        raise ValueError(
            "Unsupported tenso packet (e.g. a GPU IPC reference, which is "
            "decoded on the device rather than via loads())"
        )

    if not (align or copy):
        # Hot path: hand back the core's view untouched. Note this also skips the
        # Python-side header reparse below, which was pure overhead per call.
        return res

    # Resolve the alignment the packet declares, then enforce it (issue #5).
    try:
        mv = memoryview(data)
        hdr = _parse_header(mv)
        alignment = _target_alignment(mv, hdr[1], hdr[4], hdr[3])
    except Exception:
        alignment = _ALIGNMENT
    return _ensure_aligned(res, alignment, copy)



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
    # Fast path: zero-allocation streaming to the FD via Rust.
    if HAS_RUST and hasattr(fp, "fileno"):
        try:
            fd = fp.fileno()
            # The Rust writer goes straight to the fd, behind fp's buffer. Anything
            # the caller wrote earlier may still be sitting in that buffer, and
            # would be flushed *after* our packet -- reordering the file. Drain it
            # first so the packet lands where the caller's file position says.
            fp.flush()
            if not tensor.flags["C_CONTIGUOUS"]:
                if strict:
                    raise ValueError("Tensor is not C-Contiguous")
                tensor = np.ascontiguousarray(tensor)

            dump_to_fd_rs(tensor, fd, check_integrity=check_integrity)
            return
        except (AttributeError, io.UnsupportedOperation):
            # No usable fileno() (e.g. BytesIO); fall through to the buffered
            # write. Deliberately narrow: a failure *after* dump_to_fd_rs has
            # written must not fall through and append a second packet.
            pass

    # Single write of the complete packet (~6x faster than chunked for large arrays).
    packet = dumps(tensor, strict=strict, check_integrity=check_integrity)
    fp.write(packet)


class _MmapArray(np.ndarray):
    """A view that keeps its backing mmap alive (see ``load(mmap_mode=True)``).

    A plain ndarray has no writable ``base``, so a zero-copy view returned from
    ``loads(mm)`` would not keep ``mm`` referenced; once ``mm`` is collected the
    mapping is unmapped and touching the view segfaults. Holding ``mm`` as an
    attribute here ties the mapping's lifetime to the array's.
    """

    _tenso_mmap = None


def load(
    fp: BinaryIO, mmap_mode: bool = False, copy: bool = False, align: bool = False
) -> Any:
    """
    Deserialize an object from an open binary file.

    Parameters
    ----------
    fp : BinaryIO
        Open binary file object.
    mmap_mode : bool, default False
        Use memory mapping for large files. This is the fully zero-copy read path:
        the mapping is page-aligned, so the array is 64-byte aligned for free.
    copy : bool, default False
        Return a writeable copy.
    align : bool, default False
        Guarantee the returned array's base address is aligned. See :func:`loads`.

    Returns
    -------
    Any
        The reconstructed object.
    """
    if mmap_mode:
        mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
        result = loads(mm, copy=copy, align=align)
        if copy or not isinstance(result, np.ndarray):
            # A copy owns its data; non-array results are materialized too, so
            # the mapping can be released when this frame returns.
            return result
        # Zero-copy view into the mapping: anchor mm so it outlives the view
        # (and any view derived from it via numpy's base chain).
        view = result.view(_MmapArray)
        view._tenso_mmap = mm
        return view
    result = read_stream(fp)
    if result is None:
        raise EOFError("Empty file or stream")
    return result