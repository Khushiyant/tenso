import numpy as np
import sys

_MAGIC = b'TNSO'
_VERSION = 2
_ALIGNMENT = 64  # Align body to 64-byte boundaries for AVX-512/SIMD

# Dtype Mapping
_DTYPE_MAP = {
    np.dtype('float32'): 1,
    np.dtype('int32'): 2,
    np.dtype('float64'): 3,
    np.dtype('int64'): 4,
    np.dtype('uint8'): 5,
    np.dtype('uint16'): 6,
    np.dtype('bool'): 7,
    np.dtype('float16'): 8,
    np.dtype('int8'): 9,
    np.dtype('int16'): 10,
    np.dtype('uint32'): 11,
    np.dtype('uint64'): 12,
    # New additions
    np.dtype('complex64'): 13,
    np.dtype('complex128'): 14,
}
_REV_DTYPE_MAP = {v: k for k, v in _DTYPE_MAP.items()}