"""
Tenso configuration constants and dtype mappings.

This module defines magic numbers, versioning, alignment, and dtype mappings for the Tenso serialization format.
"""

import numpy as np
import sys

_MAGIC = b'TNSO'  #: Magic number for Tenso packet header (bytes)
_VERSION = 2      #: Protocol version (int)
_ALIGNMENT = 64   #: Align body to 64-byte boundaries for AVX-512/SIMD (int)

# Dtype Mapping
_DTYPE_MAP = {
    np.dtype('float32'): 1,      # 32-bit float
    np.dtype('int32'): 2,        # 32-bit int
    np.dtype('float64'): 3,      # 64-bit float
    np.dtype('int64'): 4,        # 64-bit int
    np.dtype('uint8'): 5,        # 8-bit unsigned int
    np.dtype('uint16'): 6,       # 16-bit unsigned int
    np.dtype('bool'): 7,         # Boolean
    np.dtype('float16'): 8,      # 16-bit float
    np.dtype('int8'): 9,         # 8-bit int
    np.dtype('int16'): 10,       # 16-bit int
    np.dtype('uint32'): 11,      # 32-bit unsigned int
    np.dtype('uint64'): 12,      # 64-bit unsigned int
    # New additions
    np.dtype('complex64'): 13,   # 64-bit complex
    np.dtype('complex128'): 14,  # 128-bit complex
}
_REV_DTYPE_MAP = {v: k for k, v in _DTYPE_MAP.items()}  #: Reverse mapping from code to numpy dtype