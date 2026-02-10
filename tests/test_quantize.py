import io
import struct

import numpy as np
import pytest

import tenso
from tenso import QuantizedTensor, dumps, loads, read_stream, write_stream
from tenso.config import (
    QDTYPE_QINT4,
    QDTYPE_QINT8,
    QDTYPE_QUINT4,
    QDTYPE_QUINT8,
    QUANT_PER_CHANNEL,
    QUANT_PER_GROUP,
    QUANT_PER_TENSOR,
    _QDTYPE_NAMES,
)
from tenso.quantize import _pack_int4, _unpack_int4
from tenso.utils import get_packet_info


# --- Parametrized round-trip tests ---

QDTYPES = ["qint8", "quint8", "qint4", "quint4"]
SCHEMES = [
    ("per_tensor", 0),
    ("per_channel", 0),
    ("per_group", 32),
]


@pytest.mark.parametrize("dtype_name", QDTYPES)
@pytest.mark.parametrize("scheme,group_size", SCHEMES)
def test_roundtrip(dtype_name, scheme, group_size):
    """Verify serialize -> deserialize round-trip for all quantized dtypes and schemes."""
    arr = np.random.randn(4, 32).astype(np.float32)
    qt = QuantizedTensor.quantize(arr, dtype_name, scheme=scheme, group_size=group_size)

    packet = dumps(qt)
    result = loads(packet)

    assert isinstance(result, QuantizedTensor)
    assert result.shape == qt.shape
    assert result.dtype_code == qt.dtype_code
    assert result.quant_scheme == qt.quant_scheme
    assert result.group_size == qt.group_size
    np.testing.assert_array_equal(result.data, qt.data)
    np.testing.assert_array_almost_equal(result.scales, qt.scales)
    np.testing.assert_array_almost_equal(result.zero_points, qt.zero_points)


@pytest.mark.parametrize("dtype_name", QDTYPES)
def test_roundtrip_with_integrity(dtype_name):
    """Verify round-trip with integrity checking enabled."""
    arr = np.random.randn(8, 16).astype(np.float32)
    qt = QuantizedTensor.quantize(arr, dtype_name)

    packet = dumps(qt, check_integrity=True)
    result = loads(packet)

    assert isinstance(result, QuantizedTensor)
    np.testing.assert_array_equal(result.data, qt.data)


@pytest.mark.parametrize("dtype_name", QDTYPES)
def test_integrity_corruption_detected(dtype_name):
    """Verify that corrupted packets are detected with integrity enabled."""
    arr = np.random.randn(4, 8).astype(np.float32)
    qt = QuantizedTensor.quantize(arr, dtype_name)

    packet = bytes(dumps(qt, check_integrity=True))
    # Corrupt a byte in the body (near the end, before footer)
    corrupted = bytearray(packet)
    corrupted[-10] ^= 0xFF
    corrupted = bytes(corrupted)

    with pytest.raises(ValueError, match="Integrity check failed"):
        loads(corrupted)


# --- int4 packing edge cases ---

def test_pack_int4_even_count():
    """Pack an even number of 4-bit values."""
    values = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint8)
    packed = _pack_int4(values)
    unpacked = _unpack_int4(packed, 6, signed=False)
    np.testing.assert_array_equal(unpacked, values)


def test_pack_int4_odd_count():
    """Pack an odd number of 4-bit values."""
    values = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
    packed = _pack_int4(values)
    assert len(packed) == 3  # ceil(5/2)
    unpacked = _unpack_int4(packed, 5, signed=False)
    np.testing.assert_array_equal(unpacked, values)


def test_pack_int4_single_element():
    """Pack a single element."""
    values = np.array([7], dtype=np.uint8)
    packed = _pack_int4(values)
    assert len(packed) == 1
    unpacked = _unpack_int4(packed, 1, signed=False)
    np.testing.assert_array_equal(unpacked, values)


def test_pack_int4_signed():
    """Pack signed 4-bit values with sign extension."""
    values = np.array([-8, -1, 0, 7], dtype=np.int8)
    # Convert to uint8 representation for packing
    uint_vals = values.view(np.uint8)
    packed = _pack_int4(uint_vals)
    unpacked = _unpack_int4(packed, 4, signed=True)
    np.testing.assert_array_equal(unpacked, values)


def test_pack_int4_empty():
    """Pack empty array."""
    values = np.array([], dtype=np.uint8)
    packed = _pack_int4(values)
    assert len(packed) == 0


# --- Dequantize accuracy ---

def test_dequantize_int8_accuracy():
    """qint8 dequantization should be close to original (atol=0.1 for small range)."""
    arr = np.random.uniform(-1, 1, (16, 16)).astype(np.float32)
    qt = QuantizedTensor.quantize(arr, "qint8")
    reconstructed = qt.dequantize()
    np.testing.assert_allclose(reconstructed, arr, atol=0.01)


def test_dequantize_uint8_accuracy():
    """quint8 dequantization should be close to original."""
    arr = np.random.uniform(0, 1, (16, 16)).astype(np.float32)
    qt = QuantizedTensor.quantize(arr, "quint8")
    reconstructed = qt.dequantize()
    np.testing.assert_allclose(reconstructed, arr, atol=0.01)


def test_dequantize_int4_accuracy():
    """qint4 dequantization has lower precision."""
    arr = np.random.uniform(-1, 1, (8, 8)).astype(np.float32)
    qt = QuantizedTensor.quantize(arr, "qint4")
    reconstructed = qt.dequantize()
    np.testing.assert_allclose(reconstructed, arr, atol=0.5)


def test_dequantize_uint4_accuracy():
    """quint4 dequantization has lower precision."""
    arr = np.random.uniform(0, 1, (8, 8)).astype(np.float32)
    qt = QuantizedTensor.quantize(arr, "quint4")
    reconstructed = qt.dequantize()
    np.testing.assert_allclose(reconstructed, arr, atol=0.5)


def test_dequantize_per_channel():
    """Per-channel dequantization accuracy."""
    arr = np.random.randn(4, 64).astype(np.float32)
    qt = QuantizedTensor.quantize(arr, "qint8", scheme="per_channel", axis=0)
    reconstructed = qt.dequantize()
    np.testing.assert_allclose(reconstructed, arr, atol=0.1)


def test_dequantize_per_group():
    """Per-group dequantization accuracy."""
    arr = np.random.randn(128).astype(np.float32)
    qt = QuantizedTensor.quantize(arr, "qint8", scheme="per_group", group_size=32)
    reconstructed = qt.dequantize()
    np.testing.assert_allclose(reconstructed, arr, atol=0.1)


# --- Bundle interop ---

def test_bundle_mixed_quantized_and_dense():
    """Bundle containing both quantized and dense tensors."""
    dense = np.random.randn(10, 10).astype(np.float32)
    qt = QuantizedTensor.quantize(
        np.random.randn(8, 16).astype(np.float32), "qint8"
    )

    bundle = {"dense": dense, "quantized": qt}
    packet = dumps(bundle)
    result = loads(packet)

    assert isinstance(result, dict)
    assert set(result.keys()) == {"dense", "quantized"}
    np.testing.assert_array_equal(result["dense"], dense)
    assert isinstance(result["quantized"], QuantizedTensor)
    np.testing.assert_array_equal(result["quantized"].data, qt.data)


def test_bundle_multiple_quantized():
    """Bundle with multiple quantized tensors of different types."""
    qt8 = QuantizedTensor.quantize(
        np.random.randn(4, 8).astype(np.float32), "qint8"
    )
    qt4 = QuantizedTensor.quantize(
        np.random.randn(4, 8).astype(np.float32), "quint4"
    )

    bundle = {"int8": qt8, "uint4": qt4}
    packet = dumps(bundle)
    result = loads(packet)

    assert isinstance(result["int8"], QuantizedTensor)
    assert isinstance(result["uint4"], QuantizedTensor)
    assert result["int8"].dtype_code == QDTYPE_QINT8
    assert result["uint4"].dtype_code == QDTYPE_QUINT4


# --- Streaming tests ---

@pytest.mark.parametrize("dtype_name", QDTYPES)
def test_write_stream_read_stream(dtype_name):
    """Verify streaming round-trip for quantized tensors."""
    arr = np.random.randn(4, 16).astype(np.float32)
    qt = QuantizedTensor.quantize(arr, dtype_name)

    buf = io.BytesIO()
    write_stream(qt, buf)

    buf.seek(0)
    result = read_stream(buf)

    assert isinstance(result, QuantizedTensor)
    assert result.shape == qt.shape
    assert result.dtype_code == qt.dtype_code
    np.testing.assert_array_equal(result.data, qt.data)


def test_stream_with_integrity():
    """Streaming round-trip with integrity checking."""
    arr = np.random.randn(4, 16).astype(np.float32)
    qt = QuantizedTensor.quantize(arr, "qint8")

    buf = io.BytesIO()
    write_stream(qt, buf, check_integrity=True)

    buf.seek(0)
    result = read_stream(buf)

    assert isinstance(result, QuantizedTensor)
    np.testing.assert_array_equal(result.data, qt.data)


# --- Custom alignment ---

@pytest.mark.parametrize("alignment", [32, 128, 256])
def test_custom_alignment(alignment):
    """Quantized tensors with custom alignment."""
    arr = np.random.randn(4, 8).astype(np.float32)
    qt = QuantizedTensor.quantize(arr, "qint8")

    packet = dumps(qt, alignment=alignment)
    result = loads(packet)

    assert isinstance(result, QuantizedTensor)
    np.testing.assert_array_equal(result.data, qt.data)


# --- Shape edge cases ---

def test_1d_tensor():
    """Quantize a 1D tensor."""
    arr = np.random.randn(100).astype(np.float32)
    qt = QuantizedTensor.quantize(arr, "qint8")
    assert qt.shape == (100,)

    result = loads(dumps(qt))
    assert result.shape == (100,)


def test_high_dimensional():
    """Quantize a high-dimensional tensor."""
    arr = np.random.randn(2, 3, 4, 5).astype(np.float32)
    qt = QuantizedTensor.quantize(arr, "quint8")
    assert qt.shape == (2, 3, 4, 5)

    result = loads(dumps(qt))
    assert result.shape == (2, 3, 4, 5)


def test_large_shape():
    """Quantize a larger tensor."""
    arr = np.random.randn(256, 256).astype(np.float32)
    qt = QuantizedTensor.quantize(arr, "qint8")

    packet = dumps(qt)
    result = loads(packet)

    assert isinstance(result, QuantizedTensor)
    assert result.shape == (256, 256)
    np.testing.assert_array_equal(result.data, qt.data)


def test_scalar_like():
    """Quantize a shape-(1,) tensor."""
    arr = np.array([3.14], dtype=np.float32)
    qt = QuantizedTensor.quantize(arr, "qint8")
    result = loads(dumps(qt))
    assert result.shape == (1,)


# --- Properties ---

def test_properties():
    """Test QuantizedTensor property accessors."""
    arr = np.random.randn(4, 8).astype(np.float32)

    qt8 = QuantizedTensor.quantize(arr, "qint8")
    assert qt8.dtype_name == "qint8"
    assert not qt8.is_4bit
    assert qt8.is_signed
    assert qt8.nbytes == 32

    qt4 = QuantizedTensor.quantize(arr, "quint4")
    assert qt4.dtype_name == "quint4"
    assert qt4.is_4bit
    assert not qt4.is_signed
    assert qt4.nbytes == 16  # ceil(32/2)


def test_repr():
    """Test string representation."""
    arr = np.random.randn(4, 8).astype(np.float32)
    qt = QuantizedTensor.quantize(arr, "qint8")
    r = repr(qt)
    assert "qint8" in r
    assert "(4, 8)" in r


# --- get_packet_info ---

@pytest.mark.parametrize("dtype_name", QDTYPES)
def test_packet_info(dtype_name):
    """get_packet_info correctly reports quantized types."""
    arr = np.random.randn(4, 8).astype(np.float32)
    qt = QuantizedTensor.quantize(arr, dtype_name)
    packet = bytes(dumps(qt))

    info = get_packet_info(packet)
    assert info["dtype"] == dtype_name
    assert info["quantized"] is True
    assert info["shape"] == (4, 8)
    assert info["total_elements"] == 32


# --- Error cases ---

def test_invalid_scheme():
    """Invalid quantization scheme raises KeyError."""
    arr = np.random.randn(4, 8).astype(np.float32)
    with pytest.raises(KeyError):
        QuantizedTensor.quantize(arr, "qint8", scheme="invalid")


def test_invalid_dtype():
    """Invalid quantized dtype name raises KeyError."""
    arr = np.random.randn(4, 8).astype(np.float32)
    with pytest.raises(KeyError):
        QuantizedTensor.quantize(arr, "qint3")


def test_per_group_requires_group_size():
    """per_group scheme requires group_size > 0."""
    arr = np.random.randn(64).astype(np.float32)
    with pytest.raises(ValueError, match="group_size must be > 0"):
        QuantizedTensor.quantize(arr, "qint8", scheme="per_group", group_size=0)


# --- Constant tensor (all same values) ---

def test_constant_tensor():
    """Quantize a tensor where all values are the same."""
    arr = np.full((4, 8), 5.0, dtype=np.float32)
    qt = QuantizedTensor.quantize(arr, "qint8")
    result = loads(dumps(qt))
    reconstructed = result.dequantize()
    np.testing.assert_allclose(reconstructed, arr, atol=0.01)
