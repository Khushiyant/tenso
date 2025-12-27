"""
Configuration and Protocol Constants for Tenso.

This module defines the binary protocol version, magic numbers,
memory alignment requirements, and feature flags used across the library.
"""

import numpy as np

_MAGIC = b"TNSO"  #: Magic number for Tenso packet header (bytes)
_VERSION = 2  #: Protocol version (int)
_ALIGNMENT = 64  #: Align body to 64-byte boundaries for AVX-512/SIMD (int)

# --- Security Limits (DoS Protection) ---
MAX_NDIM = 32  #: Maximum number of dimensions (int)
MAX_ELEMENTS = 10**9  #: Maximum elements per tensor (int)

# --- Protocol Flags ---
FLAG_ALIGNED = 1  #: Packet uses 64-byte alignment (int)
FLAG_INTEGRITY = 2  #: Packet includes an 8-byte XXH3 checksum footer (int)
FLAG_COMPRESSION = 4  #: Packet body is compressed using LZ4 (int)
FLAG_SPARSE = 8  #: Packet contains a Sparse COO tensor (int)
FLAG_BUNDLE = 16  #: Packet contains a collection (dict) of tensors (int)
FLAG_SPARSE_CSR = 32  #: Packet contains a Sparse CSR tensor (int)
FLAG_SPARSE_CSC = 64  #: Packet contains a Sparse CSC tensor (int)

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

# Try to register bfloat16 (Code 15)
try:
    _bf16 = np.dtype("bfloat16")
    _DTYPE_MAP[_bf16] = 15
except (TypeError, Exception):
    try:
        from ml_dtypes import bfloat16

        _DTYPE_MAP[np.dtype(bfloat16)] = 15
    except ImportError:
        pass

_REV_DTYPE_MAP = {v: k for k, v in _DTYPE_MAP.items()}
