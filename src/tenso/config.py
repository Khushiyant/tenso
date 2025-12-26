"""
Configuration and Protocol Constants for Tenso.
"""

import numpy as np

_MAGIC = b"TNSO"  #: Magic number for Tenso packet header
_VERSION = 2  #: Protocol version
_ALIGNMENT = 64  #: SIMD alignment boundary

# --- Security Limits (DoS Protection) ---
MAX_NDIM = 32  #: Maximum dimensions to prevent allocation attacks
MAX_ELEMENTS = 10**9  #: Maximum elements per tensor

# --- Flags ---
FLAG_ALIGNED = 1  #: Packet uses 64-byte alignment
FLAG_INTEGRITY = 2  #: Packet includes an 8-byte XXH3 checksum footer
FLAG_COMPRESSION = 4  #: Packet body is compressed using LZ4
FLAG_SPARSE = 8  #: Packet contains a Sparse COO tensor

# --- Dtype Mapping ---
_DTYPE_MAP = {
    np.dtype("float32"): 1,
    np.dtype("int32"): 2,
    np.dtype("float64"): 3,
    np.dtype("int64"): 4,
    np.dtype("uint8"): 5,
    np.dtype("uint16"): 6,
    np.dtype("bool"): 7,
    np.dtype("float16"): 8,
    np.dtype("int8"): 9,
    np.dtype("int16"): 10,
    np.dtype("uint32"): 11,
    np.dtype("uint64"): 12,
    np.dtype("complex64"): 13,
    np.dtype("complex128"): 14,
}

try:
    _bf16 = np.dtype("bfloat16")
    _DTYPE_MAP[_bf16] = 15
except (TypeError, Exception):
    try:
        import ml_dtypes

        _DTYPE_MAP[np.dtype("bfloat16")] = 15
    except ImportError:
        pass

_REV_DTYPE_MAP = {v: k for k, v in _DTYPE_MAP.items()}
