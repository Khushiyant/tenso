import pytest
import numpy as np
import tenso

@pytest.mark.parametrize("shape", [
    (10,),           # 1D Vector
    (32, 128),       # 2D Matrix
    (3, 256, 256),   # 3D Image
    (8, 10, 10, 5)   # 4D Tensor
])
@pytest.mark.parametrize("dtype", [
    np.float32, np.int32, np.float64, np.int64
])
def test_round_trip(shape, dtype):
    original = np.random.rand(*shape).astype(dtype)
    if np.issubdtype(dtype, np.integer):
        original = np.random.randint(0, 100, size=shape).astype(dtype)
        
    packet = tenso.dumps(original)
    restored = tenso.loads(packet)
    
    assert restored.shape == original.shape
    assert restored.dtype == original.dtype
    assert np.array_equal(original, restored)

def test_file_io(tmp_path):
    data = np.array([1, 2, 3], dtype=np.float32)
    file_path = tmp_path / "test.tenso"
    
    with open(file_path, "wb") as f:
        tenso.dump(data, f)
        
    with open(file_path, "rb") as f:
        restored = tenso.load(f)
        
    assert np.array_equal(data, restored)

def test_invalid_magic():
    with pytest.raises(ValueError, match="Invalid tenso packet"):
        tenso.loads(b'BAD_DATA_12345')

def test_uint8_image():
    """Test standard image format."""
    # Random 256x256 RGB image
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    restored = tenso.loads(tenso.dumps(img))
    assert np.array_equal(img, restored)