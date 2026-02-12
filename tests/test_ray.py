"""Tests for Tenso Ray integration."""

import pytest
import numpy as np

# Skip entire module if ray is not installed
ray = pytest.importorskip("ray")

from tenso.ray import (
    register,
    unregister,
    _serialize_ndarray,
    _deserialize_ndarray,
    _serialize_dict,
    _deserialize_dict,
    _serialize_quantized,
    _deserialize_quantized,
)
from tenso.quantize import QuantizedTensor


# --- Unit tests for serializer functions (no Ray needed) ---


class TestSerializerFunctions:
    """Test the raw serialize/deserialize functions without Ray."""

    @pytest.mark.parametrize(
        "dtype",
        [np.float32, np.float64, np.int32, np.int64, np.uint8, np.float16],
    )
    def test_ndarray_roundtrip(self, dtype):
        arr = np.random.randn(100, 50).astype(dtype)
        data = _serialize_ndarray(arr)
        result = _deserialize_ndarray(data)
        np.testing.assert_array_equal(arr, result)

    def test_ndarray_scalar(self):
        arr = np.array(42.0, dtype=np.float32)
        data = _serialize_ndarray(arr)
        result = _deserialize_ndarray(data)
        np.testing.assert_array_equal(arr, result)

    def test_ndarray_empty(self):
        arr = np.array([], dtype=np.float32)
        data = _serialize_ndarray(arr)
        result = _deserialize_ndarray(data)
        np.testing.assert_array_equal(arr, result)

    def test_ndarray_high_dimensional(self):
        arr = np.random.randn(2, 3, 4, 5).astype(np.float32)
        data = _serialize_ndarray(arr)
        result = _deserialize_ndarray(data)
        np.testing.assert_array_equal(arr, result)

    def test_ndarray_non_contiguous(self):
        arr = np.random.randn(100, 100).astype(np.float32)
        sliced = arr[::2, ::3]  # Non-contiguous view
        assert not sliced.flags["C_CONTIGUOUS"]
        data = _serialize_ndarray(sliced)
        result = _deserialize_ndarray(data)
        np.testing.assert_array_equal(sliced, result)

    def test_dict_bundle_roundtrip(self):
        bundle = {
            "weights": np.random.randn(10, 10).astype(np.float32),
            "bias": np.random.randn(10).astype(np.float32),
        }
        data = _serialize_dict(bundle)
        result = _deserialize_dict(data)
        for key in bundle:
            np.testing.assert_array_equal(bundle[key], result[key])

    def test_quantized_roundtrip(self):
        qt = QuantizedTensor.quantize(
            np.random.randn(64).astype(np.float32), bits=8
        )
        data = _serialize_quantized(qt)
        result = _deserialize_quantized(data)
        assert isinstance(result, QuantizedTensor)
        assert result.shape == qt.shape
        np.testing.assert_array_equal(qt.data, result.data)

    def test_deserialized_is_writeable(self):
        """Deserialized arrays should be writeable copies (not read-only views)."""
        arr = np.ones((10, 10), dtype=np.float32)
        data = _serialize_ndarray(arr)
        result = _deserialize_ndarray(data)
        result[0, 0] = 999.0  # Should not raise
        assert result[0, 0] == 999.0


# --- Integration tests with Ray runtime ---


@pytest.fixture(scope="module")
def ray_env():
    """Start a local Ray instance for tests, shut down after."""
    ray.init(num_cpus=2, log_to_driver=False)
    register()
    yield
    unregister()
    ray.shutdown()


class TestRayIntegration:
    """Test Tenso serialization through Ray's object store."""

    def test_put_get_basic(self, ray_env):
        arr = np.random.randn(100, 100).astype(np.float32)
        ref = ray.put(arr)
        result = ray.get(ref)
        np.testing.assert_array_equal(arr, result)

    def test_put_get_large(self, ray_env):
        arr = np.random.randn(1000, 1000).astype(np.float64)
        ref = ray.put(arr)
        result = ray.get(ref)
        np.testing.assert_array_equal(arr, result)

    def test_remote_function(self, ray_env):
        @ray.remote
        def double(tensor):
            return tensor * 2

        arr = np.arange(100, dtype=np.float32)
        result = ray.get(double.remote(arr))
        np.testing.assert_array_equal(arr * 2, result)

    def test_remote_function_multiple_args(self, ray_env):
        @ray.remote
        def add(a, b):
            return a + b

        a = np.ones((50, 50), dtype=np.float32)
        b = np.ones((50, 50), dtype=np.float32) * 2
        result = ray.get(add.remote(a, b))
        np.testing.assert_array_equal(a + b, result)

    def test_actor(self, ray_env):
        @ray.remote
        class Accumulator:
            def __init__(self):
                self.total = np.zeros(10, dtype=np.float32)

            def add(self, tensor):
                self.total = self.total + tensor
                return self.total

        actor = Accumulator.remote()
        ones = np.ones(10, dtype=np.float32)
        result = ray.get(actor.add.remote(ones))
        np.testing.assert_array_equal(result, ones)

        result2 = ray.get(actor.add.remote(ones))
        np.testing.assert_array_equal(result2, ones * 2)

    @pytest.mark.parametrize(
        "dtype",
        [np.float32, np.float64, np.int32, np.uint8],
    )
    def test_put_get_dtypes(self, ray_env, dtype):
        arr = np.arange(100).astype(dtype)
        ref = ray.put(arr)
        result = ray.get(ref)
        np.testing.assert_array_equal(arr, result)

    def test_multiple_refs(self, ray_env):
        """Test that multiple object refs work correctly."""
        arrays = [np.random.randn(50, 50).astype(np.float32) for _ in range(5)]
        refs = [ray.put(a) for a in arrays]
        results = ray.get(refs)
        for original, result in zip(arrays, results):
            np.testing.assert_array_equal(original, result)
