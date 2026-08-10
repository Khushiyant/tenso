"""
Regression tests for the cross-implementation and correctness fixes.

Each test pins a specific bug that was fixed so it can't silently regress.
"""

import asyncio
import gc
import io
import struct

import numpy as np
import pytest

import tenso
from tenso import QuantizedTensor, StringTensor
from tenso.config import _HEADER_BASE_V4


# --------------------------------------------------------------------------
# Bundle >255 entries: must hard-error instead of silently truncating.
# --------------------------------------------------------------------------

def test_bundle_over_255_raises():
    big = {f"k{i}": np.array([i], dtype=np.int32) for i in range(300)}
    with pytest.raises(ValueError, match="255"):
        tenso.dumps(big)


def test_bundle_exactly_255_roundtrips():
    ok = {f"k{i}": np.array([i], dtype=np.int32) for i in range(255)}
    restored = tenso.loads(tenso.dumps(ok))
    assert len(restored) == 255
    assert restored["k254"][0] == 254


# --------------------------------------------------------------------------
# Quantization: off-zero-center data and per_channel axis != 0.
# --------------------------------------------------------------------------

def test_quantize_offset_data_accuracy():
    # Data far from zero used to saturate to a single level (zero-point clamp).
    rng = np.random.default_rng(0)
    data = rng.uniform(100, 110, size=(64, 64)).astype(np.float32)
    qt = QuantizedTensor.quantize(data, dtype="qint8", scheme="per_tensor")
    recon = qt.dequantize()
    # 8-bit over a span of ~10 => step ~0.04; comfortably under 0.1.
    assert np.max(np.abs(recon - data)) < 0.1


def test_quantize_per_channel_axis1_roundtrip():
    rng = np.random.default_rng(1)
    cols = np.arange(1, 33).astype(np.float32)
    data = (rng.random((8, 32)).astype(np.float32)) * cols  # column j spans [0, j)
    qt = QuantizedTensor.quantize(data, dtype="qint8", scheme="per_channel", axis=1)

    restored = tenso.loads(tenso.dumps(qt))
    assert restored.axis == 1
    recon = restored.dequantize()
    assert recon.shape == data.shape

    # Each column is quantized with its own ~col/255 step. This only holds if
    # the channel axis is tracked correctly through (de)serialization.
    per_col_step = cols / 255.0
    assert np.all(np.abs(recon - data) <= per_col_step[None, :] * 1.5 + 1e-4)


def test_quantized_integrity_covers_scales():
    rng = np.random.default_rng(2)
    data = rng.standard_normal((16, 16)).astype(np.float32)
    qt = QuantizedTensor.quantize(data, dtype="qint8", scheme="per_tensor")
    packet = bytearray(tenso.dumps(qt, check_integrity=True))

    # Corrupt the first scale float (lives in the quant metadata, which the
    # integrity footer must now cover).
    ndim = 2
    scales_off = _HEADER_BASE_V4 + ndim * 4 + 1 + 1 + 4 + 4
    packet[scales_off] ^= 0xFF
    with pytest.raises(ValueError, match="Integrity"):
        tenso.loads(bytes(packet))


# --------------------------------------------------------------------------
# Compression: Rust write path no longer leaves trailing slack that would
# misplace the integrity footer on read.
# --------------------------------------------------------------------------

def test_rust_dumps_compress_roundtrip():
    rs = pytest.importorskip("tenso.tenso_rs")
    data = np.tile(np.arange(50, dtype=np.float32), 200)  # compressible
    packet = bytes(rs.dumps_rs(data, check_integrity=True, compress=True))
    out = tenso.loads(packet)
    assert np.array_equal(out, data)


def test_compress_integrity_roundtrip():
    data = np.tile(np.arange(100, dtype=np.float64), 50)
    out = tenso.loads(tenso.dumps(data, compress=True, check_integrity=True))
    assert np.array_equal(out, data)


# --------------------------------------------------------------------------
# float16 file dump (Rust dump_to_fd_rs previously omitted float16).
# --------------------------------------------------------------------------

def test_float16_dump_to_fd_rust(tmp_path):
    # Exercise the Rust fd writer directly: pre-fix, dump_to_fd_rs had no
    # float16 arm and raised "Unsupported dtype: float16". (tenso.dump would
    # silently fall back to Python, so go through the Rust function to pin it.)
    rs = pytest.importorskip("tenso.tenso_rs")
    rng = np.random.default_rng(3)
    data = np.ascontiguousarray((rng.random((50, 50)) * 10).astype(np.float16))
    path = tmp_path / "f16.tenso"
    with open(path, "wb") as f:
        rs.dump_to_fd_rs(data, f.fileno())
    with open(path, "rb") as f:
        restored = tenso.load(f)
    assert restored.dtype == np.float16
    assert np.array_equal(restored, data)


# --------------------------------------------------------------------------
# is_aligned: must reflect the caller's actual buffer, not a throwaway copy.
# --------------------------------------------------------------------------

def test_is_aligned_checks_real_buffer():
    base = np.zeros(256, dtype=np.uint8)
    assert base.ctypes.data % 2 == 0  # numpy base is at least 2-aligned
    assert tenso.is_aligned(memoryview(base), alignment=2) is True
    # A 1-byte-offset view cannot be 2-aligned.
    assert tenso.is_aligned(memoryview(base)[1:], alignment=2) is False
    assert tenso.is_aligned(b"", alignment=64) is True
    assert tenso.is_aligned(b"\x00" * 8, alignment=1) is True


# --------------------------------------------------------------------------
# StringTensor: reject corrupt (non-monotonic) offsets.
# --------------------------------------------------------------------------

def test_stringtensor_rejects_nonmonotonic_offsets():
    st = StringTensor(["hello", "world", "abc"])
    packet = bytearray(st.dumps())
    # offsets array begins at the 64-byte-aligned body start; offset[1] @ +8.
    body_start = 64
    struct.pack_into("<Q", packet, body_start + 8, 100)  # > offset[2]=10
    with pytest.raises(ValueError):
        StringTensor.loads(bytes(packet))


# --------------------------------------------------------------------------
# Async reader parity: custom alignment, bundles, quantized.
# --------------------------------------------------------------------------

def _aread(packet: bytes):
    async def go():
        reader = asyncio.StreamReader()
        reader.feed_data(packet)
        reader.feed_eof()
        return await tenso.aread_stream(reader)

    return asyncio.run(go())


@pytest.mark.skipif(tenso.aread_stream is None, reason="async support unavailable")
def test_aread_stream_custom_alignment():
    data = np.arange(20, dtype=np.float32).reshape(4, 5)
    out = _aread(bytes(tenso.dumps(data, alignment=128)))
    assert np.array_equal(out, data)


@pytest.mark.skipif(tenso.aread_stream is None, reason="async support unavailable")
def test_aread_stream_bundle():
    bundle = {"a": np.ones((2, 3), dtype=np.float32), "b": np.arange(5, dtype=np.int64)}
    out = _aread(bytes(tenso.dumps(bundle)))
    assert set(out) == {"a", "b"}
    assert np.array_equal(out["b"], np.arange(5))


@pytest.mark.skipif(tenso.aread_stream is None, reason="async support unavailable")
def test_aread_stream_quantized():
    rng = np.random.default_rng(4)
    qt = QuantizedTensor.quantize(
        rng.standard_normal((8, 8)).astype(np.float32), dtype="qint8"
    )
    out = _aread(bytes(tenso.dumps(qt)))
    assert isinstance(out, QuantizedTensor)
    assert out.shape == (8, 8)


@pytest.mark.skipif(tenso.aread_stream is None, reason="async support unavailable")
def test_aread_stream_quantized_per_channel():
    # Pins the async reader's multi-scale + axis metadata-length math.
    rng = np.random.default_rng(5)
    cols = np.arange(1, 17).astype(np.float32)
    data = rng.random((4, 16)).astype(np.float32) * cols
    qt = QuantizedTensor.quantize(data, dtype="qint8", scheme="per_channel", axis=1)
    out = _aread(bytes(tenso.dumps(qt)))
    assert isinstance(out, QuantizedTensor)
    assert out.axis == 1
    recon = out.dequantize()
    assert np.all(np.abs(recon - data) <= (cols / 255.0)[None, :] * 1.5 + 1e-4)


@pytest.mark.skipif(tenso.aread_stream is None, reason="async support unavailable")
def test_aread_stream_integrity():
    data = np.arange(64, dtype=np.float64)
    out = _aread(bytes(tenso.dumps(data, check_integrity=True)))
    assert np.array_equal(out, data)


@pytest.mark.skipif(tenso.aread_stream is None, reason="async support unavailable")
def test_aread_stream_scalar_then_array():
    # 0-dim scalars must read exactly one element and not corrupt the next
    # packet on the stream (regression: ndim==0 was under-read to 0 bytes).
    p1 = bytes(tenso.dumps(np.array(1.5, dtype=np.float32)))
    p2 = bytes(tenso.dumps(np.arange(3, dtype=np.int32)))

    async def go():
        reader = asyncio.StreamReader()
        reader.feed_data(p1 + p2)
        reader.feed_eof()
        return await tenso.aread_stream(reader), await tenso.aread_stream(reader)

    a, b = asyncio.run(go())
    assert a.shape == () and float(a) == 1.5
    assert np.array_equal(b, np.arange(3))


def test_scalar_roundtrip():
    for val, dt in [(7, np.int64), (3.5, np.float32), (True, np.bool_)]:
        out = tenso.loads(tenso.dumps(np.array(val, dtype=dt)))
        assert out.shape == ()
        assert out.item() == val


# --------------------------------------------------------------------------
# Bundle with a non-C-contiguous numpy value: the Rust fast path read raw
# ctypes.data bytes assuming C-order and silently corrupted transposed /
# Fortran-order / sliced arrays. dumps() must normalize nested values first.
# --------------------------------------------------------------------------

def test_bundle_noncontiguous_value_roundtrips():
    base = np.arange(12, dtype=np.float32).reshape(3, 4)

    transposed = base.T  # shape (4, 3), not C-contiguous
    assert not transposed.flags["C_CONTIGUOUS"]
    restored = tenso.loads(tenso.dumps({"w": transposed}))
    assert np.array_equal(restored["w"], transposed)

    fortran = np.asfortranarray(base)
    assert not fortran.flags["C_CONTIGUOUS"]
    restored_f = tenso.loads(tenso.dumps({"x": fortran}))
    assert np.array_equal(restored_f["x"], fortran)

    sliced = base[:, ::2]  # strided view, not contiguous
    assert not sliced.flags["C_CONTIGUOUS"]
    restored_s = tenso.loads(tenso.dumps({"y": sliced}))
    assert np.array_equal(restored_s["y"], sliced)


def test_bundle_nested_noncontiguous_value_roundtrips():
    t = np.arange(12, dtype=np.float32).reshape(3, 4).T
    restored = tenso.loads(tenso.dumps({"outer": {"w": t}}))
    assert isinstance(restored["outer"], dict)
    assert np.array_equal(restored["outer"]["w"], t)


# --------------------------------------------------------------------------
# Bundle with a nested QuantizedTensor must fall back to the Python path
# (the Rust fast path raised AttributeError introspecting the quantized value).
# --------------------------------------------------------------------------

def test_bundle_with_nested_quantized_roundtrips():
    data = np.linspace(-2.0, 2.0, 16, dtype=np.float32).reshape(4, 4)
    qt = QuantizedTensor.quantize(data, dtype="qint8", scheme="per_tensor")
    restored = tenso.loads(tenso.dumps({"layer": {"weight": qt}}))
    assert isinstance(restored["layer"], dict)
    recon = restored["layer"]["weight"].dequantize()
    assert recon.shape == data.shape
    np.testing.assert_allclose(recon, data, atol=0.05)


# --------------------------------------------------------------------------
# read_stream must reject a compressed dense packet instead of over-reading
# (the streaming format has no length prefix for a compressed body).
# --------------------------------------------------------------------------

def test_read_stream_rejects_compressed_dense():
    import io

    packet = bytes(tenso.dumps(np.arange(1000, dtype=np.float64), compress=True))
    with pytest.raises(ValueError, match="compressed"):
        tenso.read_stream(io.BytesIO(packet))


# --------------------------------------------------------------------------
# Issue #4: dimensions > u32 must raise instead of silently truncating.
# --------------------------------------------------------------------------

def test_oversized_dim_raises_not_truncates():
    # A real 4.29 GB allocation is unnecessary: a stride-0 broadcast view
    # reports the oversized shape while owning a single byte. The dim guard
    # fires before any data is touched.
    huge = np.broadcast_to(np.zeros(1, dtype=np.uint8), (0x100000003,))
    assert huge.shape == (0x100000003,)
    with pytest.raises(ValueError, match="32-bit"):
        tenso.dumps(huge)


def test_max_u32_dim_boundary():
    # u32::MAX itself must be rejected only when exceeded; a dim at the limit is
    # allowed by the guard (allocation aside, the wire format can encode it).
    at_limit = np.broadcast_to(np.zeros(1, dtype=np.uint8), (0xFFFFFFFF,))
    # The guard must not raise for a dim that fits in u32. It may still fail
    # later on the (correct) max-elements cap, but never with the truncation
    # error.
    try:
        tenso.dumps(at_limit)
    except ValueError as e:
        assert "32-bit" not in str(e)


# --------------------------------------------------------------------------
# Issue #5: loads(align=True) must return arrays aligned to the packet's
# boundary, regardless of the transport buffer's own alignment.
#
# These assert the OPT-IN guarantee. They must not be relaxed to the default
# path: alignment on an arbitrary buffer costs a full copy of the body, so
# charging it to every caller is the regression fixed in
# test_loads_is_zero_copy_by_default below.
# --------------------------------------------------------------------------

def _misaligned_packet_bytes(packet) -> bytes:
    # Force a deliberately misaligned backing buffer: a bytes object at an
    # arbitrary address plus a 1-byte prefix shift, then strip the prefix so the
    # packet starts at an odd offset within an owned buffer.
    raw = bytes(packet)
    shifted = (b"\x00" + raw)
    return memoryview(shifted)[1:]


@pytest.mark.parametrize("dtype", [np.uint64, np.float32, np.int16, np.float64])
def test_loads_returns_aligned_array(dtype):
    x = np.array([[1, 5], [9, 24]], dtype=dtype)
    for _ in range(64):
        packet = _misaligned_packet_bytes(tenso.dumps(x))
        out = tenso.loads(packet, align=True)
        assert out.ctypes.data % 64 == 0, (
            f"deserialized array not 64-byte aligned: {out.ctypes.data % 64}"
        )
        assert np.array_equal(out, x)


def test_loads_aligned_with_copy():
    x = np.arange(37, dtype=np.float64)
    packet = _misaligned_packet_bytes(tenso.dumps(x))
    out = tenso.loads(packet, copy=True)
    assert out.ctypes.data % 64 == 0
    assert out.flags.writeable is True
    assert np.array_equal(out, x)


def test_loads_custom_alignment_honored():
    x = np.arange(50, dtype=np.float32)
    packet = _misaligned_packet_bytes(tenso.dumps(x, alignment=128))
    out = tenso.loads(packet, align=True)
    assert out.ctypes.data % 128 == 0
    assert np.array_equal(out, x)


def test_loads_bundle_arrays_aligned():
    bundle = {
        "a": np.arange(10, dtype=np.float32),
        "b": np.array([[1, 2], [3, 4]], dtype=np.uint64),
    }
    packet = _misaligned_packet_bytes(tenso.dumps(bundle))
    out = tenso.loads(packet, align=True)
    for k, v in out.items():
        assert v.ctypes.data % 64 == 0, f"bundle member {k} not aligned"


def test_loads_zero_copy_preserved_when_input_aligned():
    # When the caller's buffer is already aligned, no copy should occur: the
    # result must share memory with the input (true zero-copy path).
    x = np.arange(64, dtype=np.float64)
    src = bytes(tenso.dumps(x))
    n = len(src)
    buf = np.empty(n + 64, dtype=np.uint8)
    off = (-buf.ctypes.data) % 64
    aligned = buf[off : off + n]
    aligned[:] = np.frombuffer(src, dtype=np.uint8)
    assert aligned.ctypes.data % 64 == 0

    out = tenso.loads(memoryview(aligned), align=True)
    assert np.array_equal(out, x)
    # body lives at an aligned offset within an aligned buffer -> aliased view.
    assert np.shares_memory(out, aligned)


# --------------------------------------------------------------------------
# Zero-copy regression guard.
#
# 4b4dbb2 ("guarantee array alignment (#5)") made loads() copy the entire body
# on every call. Its own message claimed "zero-copy preserved otherwise", but
# the otherwise never happens: a CPython bytes payload sits exactly 32 bytes
# past its allocation (refcount + type + size + hash) and allocators align to
# 16, so a packet body is always 32 mod 64 and the alignment check can never
# pass. That silently turned a ~0.003 ms 64 MB read into ~2.4 ms and invalidated
# every published benchmark for two releases with no test failing.
#
# These assert the property directly rather than a proxy for it.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [np.uint8, np.float32, np.float64, np.int64])
def test_loads_is_zero_copy_by_default(dtype):
    x = np.arange(1024, dtype=dtype)
    packet = bytes(tenso.dumps(x))
    out = tenso.loads(packet)
    assert np.shares_memory(out, np.frombuffer(packet, dtype=np.uint8)), (
        "loads() allocated instead of returning a view onto the packet"
    )
    assert np.array_equal(out, x)
    assert out.flags.writeable is False


def test_loads_zero_copy_across_repeated_calls():
    # One lucky address is not evidence. Every call must alias, on fresh packets
    # at fresh addresses, or the guarantee is really a coin flip.
    x = np.arange(97, dtype=np.float32)
    for i in range(64):
        packet = bytes(tenso.dumps(x))
        assert np.shares_memory(
            tenso.loads(packet), np.frombuffer(packet, dtype=np.uint8)
        ), f"loads() copied on iteration {i}"


def test_loads_bundle_is_zero_copy_by_default():
    bundle = {"a": np.arange(64, dtype=np.float32), "b": np.arange(8, dtype=np.uint64)}
    packet = bytes(tenso.dumps(bundle))
    pv = np.frombuffer(packet, dtype=np.uint8)
    out = tenso.loads(packet)
    for k, v in out.items():
        assert np.shares_memory(v, pv), f"bundle member {k} was copied"


def test_zero_copy_view_outlives_the_packet_reference():
    # The view aliases the packet, so it must keep the packet alive. If it did
    # not, zero-copy-by-default would be a use-after-free rather than a speedup.
    x = np.arange(4096, dtype=np.float64)
    packet = bytes(tenso.dumps(x))
    out = tenso.loads(packet)
    del packet
    gc.collect()
    assert np.array_equal(out, x)


def test_read_stream_does_not_copy_the_packet():
    # read_stream used to hand loads() a bytes() of its framing buffer, copying the
    # whole packet before decoding it. The result must alias the framing buffer.
    x = np.arange(65536, dtype=np.float64)
    out = tenso.read_stream(io.BytesIO(bytes(tenso.dumps(x))))
    assert np.array_equal(out, x)
    # A copy would own its data outright; a view is backed by the framing buffer.
    assert out.base is not None, "read_stream materialized instead of viewing"
    assert out.flags.writeable is False, "read_stream leaked a writable view"


def test_align_true_still_returns_correct_data_and_is_read_only():
    # The opt-in path allocates, so it must not accidentally hand back a
    # writeable buffer (that is copy=True's contract, not align's).
    x = np.arange(129, dtype=np.float32)
    out = tenso.loads(bytes(tenso.dumps(x)), align=True)
    assert out.ctypes.data % 64 == 0
    assert out.flags.writeable is False
    assert np.array_equal(out, x)


def test_load_mmap_mode_is_zero_copy_and_aligned(tmp_path):
    # mmap is the path where alignment is free: the mapping is page-aligned, so
    # the body's 64-byte offset lands on a 64-byte address with no copy.
    x = np.arange(8192, dtype=np.float32)
    p = tmp_path / "t.tenso"
    p.write_bytes(bytes(tenso.dumps(x)))
    with open(p, "rb") as f:
        out = tenso.load(f, mmap_mode=True)
    assert out.ctypes.data % 64 == 0
    assert np.array_equal(out, x)


# --------------------------------------------------------------------------
# Non-native byte order: dumps() wrote the raw bytes and loads() read them back
# in the opposite order, silently returning different numbers. _DTYPE_MAP is
# keyed on native dtype objects, so '>f4' never matched and fell through
# unvalidated. Reject rather than convert: converting would hide an allocation
# and a copy inside a documented zero-copy write.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dt", [">f4", ">f8", ">i4", ">u8", ">i2"])
def test_dumps_rejects_non_native_byteorder(dt):
    x = np.arange(4, dtype=np.dtype(dt))
    with pytest.raises(ValueError, match="native byte order"):
        tenso.dumps(x)


def test_dumps_rejects_non_native_byteorder_in_bundle():
    # The bundle path recurses separately, so it needs its own guard.
    bundle = {"ok": np.arange(4, dtype="<f4"), "bad": np.arange(4, dtype=">f4")}
    with pytest.raises(ValueError, match="native byte order"):
        tenso.dumps(bundle)


def test_dumps_accepts_explicit_native_byteorder():
    # '<f4' on a little-endian host is native; only a genuine mismatch is an error.
    native = "<f4" if np.little_endian else ">f4"
    x = np.arange(4, dtype=np.dtype(native))
    assert np.array_equal(tenso.loads(tenso.dumps(x)), x)


def test_dumps_accepts_single_byte_dtypes():
    # uint8/int8 have byteorder '|' (not applicable) and must never be rejected.
    for dt in (np.uint8, np.int8, np.bool_):
        x = np.arange(4).astype(dt)
        assert np.array_equal(tenso.loads(tenso.dumps(x)), x)


# --------------------------------------------------------------------------
# dump() took an fd fast path via fileno() and wrote behind the file object's
# buffer, so anything the caller had already written was flushed AFTER the
# packet -- reordering the file with no error. The broad
# `except (ValueError, TypeError, AttributeError, OSError)` then appended a
# second packet to the already-corrupt file.
# --------------------------------------------------------------------------

def test_dump_preserves_prior_buffered_writes(tmp_path):
    x = np.arange(4, dtype=np.float32)
    p = tmp_path / "out.bin"
    with open(p, "wb") as f:
        f.write(b"HEADER")
        tenso.dump(x, f)
    raw = p.read_bytes()
    assert raw[:6] == b"HEADER", (
        f"prior write was reordered; HEADER landed at offset {raw.find(b'HEADER')}"
    )
    assert np.array_equal(tenso.loads(raw[6:]), x)


def test_dump_writes_exactly_one_packet(tmp_path):
    # Guards the fall-through: a failure after the fd write must not silently
    # append a second copy of the packet.
    x = np.arange(64, dtype=np.float64)
    p = tmp_path / "one.bin"
    with open(p, "wb") as f:
        tenso.dump(x, f)
    expected = len(bytes(tenso.dumps(x)))
    assert p.stat().st_size == expected, "dump() wrote more than one packet"


def test_dump_still_works_without_a_fileno():
    # BytesIO has no usable fileno(); the buffered fallback must still fire.
    buf = io.BytesIO()
    x = np.arange(10, dtype=np.int32)
    buf.write(b"PRE")
    tenso.dump(x, buf)
    raw = buf.getvalue()
    assert raw[:3] == b"PRE"
    assert np.array_equal(tenso.loads(raw[3:]), x)


# --------------------------------------------------------------------------
# Untrusted shape dims were multiplied with np.prod (int64), which wraps
# silently and could carry a hostile element count past the MAX_ELEMENTS guard.
# --------------------------------------------------------------------------

def _forge_v4_header(ndim: int, dims: list[int], dtype_code: int = 1) -> bytes:
    """A v4 header plus shape block: magic(4) ver(1) flags(2) dtype(1) ndim(1) rsvd(1)."""
    from tenso.config import _MAGIC, FLAG_ALIGNED

    hdr = _MAGIC + bytes([4]) + struct.pack("<H", FLAG_ALIGNED) + bytes(
        [dtype_code, ndim, 0]
    )
    assert len(hdr) == _HEADER_BASE_V4
    return hdr + struct.pack(f"<{len(dims)}I", *dims)


def test_stream_framing_rejects_int64_overflowing_shape():
    # Eight dims of u32::MAX. The true element count is ~2**256; np.prod computes
    # it in int64, which wraps to a small number and slips past MAX_ELEMENTS.
    ndim = 8
    forged = _forge_v4_header(ndim, [0xFFFFFFFF] * ndim)
    with pytest.raises(ValueError, match="maximum elements"):
        tenso.read_stream(io.BytesIO(forged))


def test_stream_framing_still_accepts_an_honest_shape():
    # The overflow guard must not reject legitimate multi-dimensional packets.
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    assert np.array_equal(
        tenso.read_stream(io.BytesIO(bytes(tenso.dumps(x)))), x
    )
