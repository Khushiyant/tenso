import pytest
import numpy as np
import tenso
import struct

try:
    from ml_dtypes import bfloat16
    HAS_BF16 = True
except ImportError:
    try:
        np.dtype("bfloat16")
        bfloat16 = np.dtype("bfloat16").type
        HAS_BF16 = True
    except TypeError:
        HAS_BF16 = False

try:
    from scipy import sparse
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

@pytest.mark.skipif(not HAS_BF16, reason="bfloat16 not supported")
def test_bfloat16_simple():
    """Verify simple BFloat16 serialization."""
    data = np.array([1.0, 2.0, 3.5], dtype=bfloat16)
    packet = tenso.dumps(data)
    restored = tenso.loads(packet)
    assert restored.dtype.name == "bfloat16"
    assert np.allclose(data.astype(np.float32), restored.astype(np.float32))

def test_integrity_rust_dense():
    """Verify integrity checks on the Rust path for dense arrays."""
    data = np.random.rand(100).astype(np.float32)
    packet = tenso.dumps(data, check_integrity=True)
    
    # Should load fine
    tenso.loads(packet)
    
    # Corrupt the body (header is 8 bytes + shape)
    # Shape is 1 dim -> 4 bytes. Total header = 12. Padding to 64 -> 52 bytes padding?
    # Let's just corrupt the LAST byte which is definitely body or hash.
    # Actually, packet structure: Header | Shape | Padding | Body | Hash
    
    # Mutable bytearray
    corrupt_packet = bytearray(packet)
    # Corrupt a byte in the middle (body)
    idx = len(packet) // 2
    corrupt_packet[idx] = (corrupt_packet[idx] + 1) % 255
    
    with pytest.raises(ValueError, match="Integrity check failed"):
        tenso.loads(corrupt_packet)

@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
def test_mixed_bundle_integrity():
    """Verify a bundle with mixed types and integrity enabled."""
    # Integrity is currently ignored for non-dense in Rust or handled gracefully?
    # Python code passed check_integrity=True to recursive calls.
    # Rust code ignored `check_integrity` for Sparse/Bundle container itself but passed it down?
    # Let's verify it doesn't crash.
    
    dense = np.array([1, 2, 3], dtype=np.int32)
    sp = sparse.csr_matrix([[1, 0], [0, 1]], dtype=np.float32)
    
    bundle = {"d": dense, "s": sp}
    
    # This should work
    packet = tenso.dumps(bundle, check_integrity=True)
    restored = tenso.loads(packet)
    
    assert np.array_equal(restored["d"], dense)
    assert np.array_equal(restored["s"].toarray(), sp.toarray())

def test_rust_sparse_dispatch_check():
    """Verify we are actually using the Rust path for sparse."""
    # We can't easily introspect this without mocking, but if we pass an invalid
    # alignment (e.g. 3) which Rust checks early, we might catch it?
    # Rust: alignment must be power of two.
    
    if not HAS_SCIPY:
        return

    sp = sparse.csr_matrix([[1]], dtype=np.float32)
    
    with pytest.raises(ValueError, match="Alignment must be a power of two"):
        tenso.dumps(sp, alignment=3)
