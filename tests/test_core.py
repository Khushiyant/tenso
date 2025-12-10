import pytest
import numpy as np
import tenso
import tempfile
import os

# --- Core Functionality Tests ---

@pytest.mark.parametrize("dtype", [
    np.float32, np.int32, np.float64, np.int64,
    np.uint8, np.uint16, np.bool_, np.float16,
    np.int8, np.int16, np.uint32, np.uint64,
    # [NEW] Added Complex Support
    np.complex64, np.complex128
])
def test_all_dtypes(dtype):
    """Verify all supported dtypes serialize and deserialize correctly."""
    shape = (10, 10)
    if dtype == np.bool_:
        original = np.random.randint(0, 2, size=shape).astype(dtype)
    elif np.issubdtype(dtype, np.integer):
        original = np.random.randint(0, 10, size=shape).astype(dtype)
    elif np.issubdtype(dtype, np.complexfloating):
        # Generate random complex numbers
        real = np.random.randn(*shape)
        imag = np.random.randn(*shape)
        original = (real + 1j * imag).astype(dtype)
    else:
        original = np.random.randn(*shape).astype(dtype)
        
    packet = tenso.dumps(original)
    restored = tenso.loads(packet)
    
    assert restored.dtype == original.dtype
    assert restored.shape == original.shape
    assert np.array_equal(original, restored)

def test_large_dimensions():
    """Verify handling of high-dimensional arrays."""
    # Create a 6D array (max ndim is 255 in protocol v2)
    shape = (2, 2, 2, 2, 2, 2)
    original = np.zeros(shape, dtype=np.float32)
    
    packet = tenso.dumps(original)
    restored = tenso.loads(packet)
    
    assert restored.shape == shape
    
    # Test packet inspection
    info = tenso.get_packet_info(packet)
    assert info['ndim'] == 6
    assert info['shape'] == shape

# --- [NEW] Strict Mode Tests ---

def test_strict_mode_success():
    """Strict mode should pass for C-Contiguous arrays."""
    data = np.random.rand(10, 10).astype(np.float32)
    # Default numpy arrays are C-Contiguous
    packet = tenso.dumps(data, strict=True)
    assert len(packet) > 0

def test_strict_mode_failure():
    """Strict mode should raise ValueError for non-contiguous arrays."""
    # Create a non-contiguous array using slicing
    data = np.random.rand(10, 10).astype(np.float32)
    data_f = np.asfortranarray(data)  # Force Fortran (column-major) layout
    
    # 1. verify it fails with strict=True
    with pytest.raises(ValueError, match="Tensor is not C-Contiguous"):
        tenso.dumps(data_f, strict=True)
        
    # 2. verify it succeeds (by copying) with strict=False
    packet = tenso.dumps(data_f, strict=False)
    restored = tenso.loads(packet)
    assert np.array_equal(data, restored)

# --- [NEW] Memory Mapping Tests ---

def test_mmap_loading():
    """Verify that we can load files using mmap_mode."""
    data = np.random.rand(50, 50).astype(np.float32)
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tenso.dump(data, tmp)
        tmp_path = tmp.name
        
    try:
        with open(tmp_path, "rb") as f:
            # Load with memory mapping
            loaded = tenso.load(f, mmap_mode=True)
            
            # Verify data
            assert np.array_equal(data, loaded)
            
            # Verify it is actually using mmap (base should be mmap)
            # Depending on numpy version, base might be the mmap object directly or indirect
            assert loaded.base is not None
            
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# --- Alignment & Safety Tests ---

def test_is_aligned_utility():
    """Test the is_aligned utility function."""
    data = b'\x00' * 100
    aligned = tenso.is_aligned(data)
    assert isinstance(aligned, bool)

def test_padding_logic():
    """Verify that packets contain correct padding for 64-byte alignment."""
    data = np.zeros((10,), dtype=np.float32)
    packet = tenso.dumps(data)
    
    header_shape_len = 8 + 4
    padding_len = 64 - header_shape_len
    
    # Extract padding area directly
    padding = packet[header_shape_len : header_shape_len + padding_len]
    assert len(padding) == padding_len
    assert padding == b'\x00' * padding_len

def test_copy_flag():
    """Verify copy=True behavior."""
    data = np.zeros((10,), dtype=np.float32)
    packet = tenso.dumps(data)
    
    # Default (Zero Copy)
    arr_view = tenso.loads(packet, copy=False)
    # Writeable flag should be False for zero-copy views to prevent corruption
    assert arr_view.flags.writeable is False
    
    # Copy (Safe)
    arr_copy = tenso.loads(packet, copy=True)
    assert arr_copy.flags.writeable is True
    assert arr_copy.base is None  # Owns its memory