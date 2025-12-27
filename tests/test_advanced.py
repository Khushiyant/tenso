"""
Advanced Feature Tests for Tenso.
Verifies Compression, Sparse Formats, and Multi-tensor Bundling.
"""

import pytest
import numpy as np
import tenso

try:
    from scipy import sparse

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def test_compression_roundtrip():
    """Verify LZ4 compression reduces size and preserves data."""
    # Highly compressible data
    data = np.zeros((100, 100), dtype=np.float32)
    packet_compressed = tenso.dumps(data, compress=True)
    packet_raw = tenso.dumps(data, compress=False)

    assert len(packet_compressed) < len(packet_raw)

    restored = tenso.loads(packet_compressed)
    assert np.array_equal(data, restored)
    assert restored.dtype == data.dtype


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
def test_sparse_coo_roundtrip():
    """Verify Sparse COO matrix serialization."""
    row = np.array([0, 3, 1, 0])
    col = np.array([0, 3, 1, 2])
    data = np.array([4, 5, 7, 9], dtype=np.int32)
    mtx = sparse.coo_matrix((data, (row, col)), shape=(4, 4))

    packet = tenso.dumps(mtx)
    restored = tenso.loads(packet)

    assert np.array_equal(mtx.toarray(), restored.toarray())
    assert restored.format == "coo"


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
@pytest.mark.parametrize("fmt", ["csr", "csc"])
def test_sparse_advanced_roundtrip(fmt):
    """Verify CSR and CSC format support."""
    mtx = sparse.random(10, 10, density=0.1, format=fmt, dtype=np.float32)
    packet = tenso.dumps(mtx)
    restored = tenso.loads(packet)

    assert np.array_equal(mtx.toarray(), restored.toarray())
    assert restored.format == fmt


def test_bundle_roundtrip():
    """Verify dictionary bundling (multi-tensor packets)."""
    bundle = {
        "weights": np.random.rand(10, 10).astype(np.float32),
        "bias": np.random.rand(10).astype(np.float32),
        "id": np.array([123], dtype=np.int64),
    }

    packet = tenso.dumps(bundle)
    restored = tenso.loads(packet)

    assert isinstance(restored, dict)
    assert set(restored.keys()) == set(bundle.keys())
    for k in bundle:
        assert np.array_equal(bundle[k], restored[k])


def test_nested_bundle():
    """Verify recursive bundling support."""
    nested = {
        "layer1": {
            "w": np.ones((2, 2), dtype=np.float32),
            "b": np.zeros((2,), dtype=np.float32),
        },
        "step": np.array([1], dtype=np.int32),
    }

    packet = tenso.dumps(nested)
    restored = tenso.loads(packet)

    assert np.array_equal(nested["layer1"]["w"], restored["layer1"]["w"])
    assert np.array_equal(nested["step"], restored["step"])


def test_empty_bundle():
    """Verify handling of empty dictionaries."""
    packet = tenso.dumps({})
    restored = tenso.loads(packet)
    assert restored == {}
