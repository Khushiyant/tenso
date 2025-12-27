import pytest
import numpy as np
import tenso
import tempfile
import os
import struct
from tenso.config import MAX_NDIM, MAX_ELEMENTS, _MAGIC, _VERSION, FLAG_ALIGNED

# Try to import bfloat16 for testing
try:
    from ml_dtypes import bfloat16

    HAS_BF16 = True
except ImportError:
    try:
        # NumPy 2.0+ might have it
        np.dtype("bfloat16")
        bfloat16 = np.float16  # Placeholder if needed
        HAS_BF16 = True
    except TypeError:
        HAS_BF16 = False

# --- Core Functionality Tests ---

DTYPES_TO_TEST = [
    np.float32,
    np.int32,
    np.float64,
    np.int64,
    np.uint8,
    np.uint16,
    np.bool_,
    np.float16,
    np.int8,
    np.int16,
    np.uint32,
    np.uint64,
    np.complex64,
    np.complex128,
]

if HAS_BF16:
    DTYPES_TO_TEST.append(np.dtype("bfloat16"))


@pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
def test_all_dtypes(dtype):
    """Verify all supported dtypes serialize and deserialize correctly."""
    shape = (10, 10)

    # [FIX] Normalize to dtype instance to access .name safely
    dt_instance = np.dtype(dtype)

    if dtype == np.bool_:
        original = np.random.randint(0, 2, size=shape).astype(dtype)
    elif np.issubdtype(dtype, np.integer):
        original = np.random.randint(0, 10, size=shape).astype(dtype)
    elif np.issubdtype(dtype, np.complexfloating):
        real = np.random.randn(*shape)
        imag = np.random.randn(*shape)
        original = (real + 1j * imag).astype(dtype)
    elif dt_instance.name == "bfloat16":
        # Create float32 and cast
        original = np.random.randn(*shape).astype(np.float32).astype(dtype)
    else:
        original = np.random.randn(*shape).astype(dtype)

    packet = tenso.dumps(original)
    restored = tenso.loads(packet)

    assert restored.dtype == original.dtype
    assert restored.shape == original.shape
    # Use array_equal for exact match
    assert np.array_equal(original, restored)


def test_large_dimensions():
    """Verify handling of high-dimensional arrays (within limits)."""
    shape = (2, 2, 2, 2, 2, 2)
    original = np.zeros(shape, dtype=np.float32)

    packet = tenso.dumps(original)
    restored = tenso.loads(packet)

    assert restored.shape == shape

    info = tenso.get_packet_info(packet)
    assert info["ndim"] == 6


# --- [NEW] DoS Protection Tests ---


def test_dos_max_ndim():
    """Verify that exceeding MAX_NDIM raises a ValueError."""
    malicious_ndim = MAX_NDIM + 1
    header = struct.pack("<4sBBBB", _MAGIC, _VERSION, FLAG_ALIGNED, 1, malicious_ndim)
    packet = header + b"\x00" * 100

    with pytest.raises(ValueError, match="Packet exceeds maximum dimensions"):
        tenso.loads(packet)


def test_dos_max_elements():
    """Verify that exceeding MAX_ELEMENTS raises a ValueError."""
    ndim = 1
    header = struct.pack("<4sBBBB", _MAGIC, _VERSION, FLAG_ALIGNED, 1, ndim)
    malicious_shape = struct.pack("<I", MAX_ELEMENTS + 1)
    packet = header + malicious_shape + b"\x00" * 100

    with pytest.raises(ValueError, match="Packet exceeds maximum elements"):
        tenso.loads(packet)


# --- Strict Mode Tests ---


def test_strict_mode_success():
    """Strict mode should pass for C-Contiguous arrays."""
    data = np.random.rand(10, 10).astype(np.float32)
    packet = tenso.dumps(data, strict=True)
    assert len(packet) > 0


def test_strict_mode_failure():
    """Strict mode should raise ValueError for non-contiguous arrays."""
    data = np.random.rand(10, 10).astype(np.float32)
    data_f = np.asfortranarray(data)

    with pytest.raises(ValueError, match="Tensor is not C-Contiguous"):
        tenso.dumps(data_f, strict=True)

    packet = tenso.dumps(data_f, strict=False)
    restored = tenso.loads(packet)
    assert np.array_equal(data, restored)


# --- Memory Mapping Tests ---


def test_mmap_loading():
    """Verify that we can load files using mmap_mode."""
    data = np.random.rand(50, 50).astype(np.float32)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tenso.dump(data, tmp)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            loaded = tenso.load(f, mmap_mode=True)
            assert np.array_equal(data, loaded)
            assert loaded.base is not None
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_copy_flag():
    """Verify copy=True behavior."""
    data = np.zeros((10,), dtype=np.float32)
    packet = tenso.dumps(data)

    arr_view = tenso.loads(packet, copy=False)
    assert arr_view.flags.writeable is False

    arr_copy = tenso.loads(packet, copy=True)
    assert arr_copy.flags.writeable is True
    assert arr_copy.base is None


def test_dumps_return_type():
    """Verify dumps returns a memoryview."""
    data = np.zeros((10,), dtype=np.float32)
    packet = tenso.dumps(data)
    assert isinstance(packet, memoryview)
