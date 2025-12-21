import numpy as np
import sys

_MAGIC = b'TNSO'  #: Magic number for Tenso packet header (bytes)
_VERSION = 2      #: Protocol version (int)
_ALIGNMENT = 64   #: Align body to 64-byte boundaries for AVX-512/SIMD (int)

# Flags
FLAG_ALIGNED = 1    #: Packet uses 64-byte alignment
FLAG_INTEGRITY = 2  #: Packet includes an 8-byte XXH3 checksum footer

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
    np.dtype('complex64'): 13,
    np.dtype('complex128'): 14,
}
_REV_DTYPE_MAP = {v: k for k, v in _DTYPE_MAP.items()}