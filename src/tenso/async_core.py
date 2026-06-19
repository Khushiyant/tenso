"""
Async I/O Support for Tenso.

Provides coroutines for reading and writing Tenso packets using
asyncio stream readers and writers.
"""

import asyncio
import struct
from typing import Optional

import numpy as np

from .config import (
    _HEADER_BASE_V3,
    _HEADER_BASE_V4,
    _MAGIC,
    _ALIGNMENT,
    MAX_ELEMENTS,
    MAX_NDIM,
    FLAG_ALIGNED,
    FLAG_BUNDLE,
    FLAG_COMPRESSION,
    FLAG_CUST_ALIGN,
    FLAG_INTEGRITY,
    FLAG_SPARSE,
    FLAG_SPARSE_CSC,
    FLAG_SPARSE_CSR,
)
from .core import (
    _QUANTIZED_CODES,
    _REV_DTYPE_MAP,
    _quantized_body_size,
    iter_dumps,
    loads,
)

_SPARSE_MASK = FLAG_SPARSE | FLAG_SPARSE_CSR | FLAG_SPARSE_CSC


async def _aread_full_packet(reader: asyncio.StreamReader) -> Optional[bytes]:
    """
    Read exactly one complete Tenso packet from an asyncio stream.

    The full packet (header + payload) is reassembled into a ``bytes`` object
    using the wire-format length fields, so every packet kind the in-memory
    :func:`loads` understands (dense, bundle, sparse, quantized; v3 or v4;
    custom alignment; integrity) round-trips identically over a stream.

    Returns ``None`` on a clean stream end (no bytes available).
    """
    try:
        head = await reader.readexactly(_HEADER_BASE_V3)
    except asyncio.IncompleteReadError as e:
        if len(e.partial) == 0:
            return None
        raise

    if head[:4] != _MAGIC:
        raise ValueError("Invalid tenso packet")

    ver = head[4]
    if ver == 4:
        head += await reader.readexactly(_HEADER_BASE_V4 - _HEADER_BASE_V3)
        flags = struct.unpack_from("<H", head, 5)[0]
        dtype_code, ndim, header_size = head[7], head[8], _HEADER_BASE_V4
    elif ver == 3:
        flags, dtype_code, ndim, header_size = head[5], head[6], head[7], _HEADER_BASE_V3
    else:
        raise ValueError(f"Unsupported protocol version: {ver}")

    buf = bytearray(head)

    # Bundle: ndim repurposed as the entry count; entries are length-prefixed.
    if flags & FLAG_BUNDLE:
        for _ in range(ndim):
            k_len_b = await reader.readexactly(4)
            buf += k_len_b
            buf += await reader.readexactly(struct.unpack("<I", k_len_b)[0])
            v_len_b = await reader.readexactly(4)
            buf += v_len_b
            buf += await reader.readexactly(struct.unpack("<I", v_len_b)[0])
        return bytes(buf)

    if ndim > MAX_NDIM:
        raise ValueError(f"Packet exceeds maximum dimensions ({ndim} > {MAX_NDIM})")

    shape_b = await reader.readexactly(ndim * 4)
    buf += shape_b
    shape = struct.unpack(f"<{ndim}I", shape_b)
    # np.prod(()) == 1, so a 0-dim scalar correctly counts as one element
    # (matching core.loads); do not special-case ndim == 0 to zero.
    num_elements = int(np.prod(shape))
    if num_elements > MAX_ELEMENTS:
        raise ValueError(f"Packet exceeds maximum elements ({num_elements})")

    # Sparse: three length-prefixed sub-packets (data + index arrays).
    if flags & _SPARSE_MASK:
        for _ in range(3):
            sz_b = await reader.readexactly(4)
            buf += sz_b
            buf += await reader.readexactly(struct.unpack("<I", sz_b)[0])
        return bytes(buf)

    footer_len = 8 if (flags & FLAG_INTEGRITY) else 0

    if dtype_code in _QUANTIZED_CODES:
        # Quant metadata precedes the alignment byte: scheme(1) + axis(1)
        # + group_size(4) + num_scales(4), then scales + zero_points.
        meta = await reader.readexactly(10)
        buf += meta
        num_scales = struct.unpack("<I", meta[6:10])[0]
        buf += await reader.readexactly(num_scales * 4 * 2)
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

    # Alignment exponent byte (custom alignment only), then padding.
    cursor = len(buf)
    if flags & FLAG_CUST_ALIGN:
        ab = await reader.readexactly(1)
        buf += ab
        alignment = 1 << ab[0]
        cursor += 1
    elif flags & FLAG_ALIGNED:
        alignment = _ALIGNMENT
    else:
        alignment = 1

    pad_len = (alignment - (cursor % alignment)) % alignment
    if pad_len:
        buf += await reader.readexactly(pad_len)

    buf += await reader.readexactly(body_len + footer_len)
    return bytes(buf)


async def aread_stream(reader: asyncio.StreamReader):
    """
    Asynchronously read and deserialize a Tenso packet from a StreamReader.

    Supports dense arrays, multi-tensor bundles, sparse matrices, and quantized
    tensors (v3 and v4 packets, custom alignment, and integrity footers) — full
    parity with the synchronous :func:`tenso.read_stream`.

    Parameters
    ----------
    reader : asyncio.StreamReader
        The stream reader source.

    Returns
    -------
    Optional[Any]
        The deserialized object, or ``None`` if the stream ended before any
        bytes were read.
    """
    packet = await _aread_full_packet(reader)
    if packet is None:
        return None
    return loads(packet)


async def awrite_stream(
    tensor: np.ndarray,
    writer: asyncio.StreamWriter,
    strict: bool = False,
    check_integrity: bool = False,
) -> None:
    """
    Asynchronously write a tensor to a StreamWriter.

    Parameters
    ----------
    tensor : np.ndarray
        The array to write.
    writer : asyncio.StreamWriter
        The stream writer destination.
    strict : bool, default False
        Strict contiguous check.
    check_integrity : bool, default False
        Include checksum.
    """
    for chunk in iter_dumps(tensor, strict=strict, check_integrity=check_integrity):
        writer.write(chunk)
    await writer.drain()
