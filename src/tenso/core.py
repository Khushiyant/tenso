import struct
import numpy as np

# --- The Tenso Protocol ---
_MAGIC = b'TNSO'  # Magic bytes for file identification
_VERSION = 1

_DTYPE_MAP = {
    np.dtype('float32'): 1,
    np.dtype('int32'): 2,
    np.dtype('float64'): 3,
    np.dtype('int64'): 4,

    np.dtype('uint8'): 5,   # Images
    np.dtype('uint16'): 6,  # High-depth images
    np.dtype('bool'): 7,    # Masks / Logic
    np.dtype('float16'): 8, # LLMs / Mixed Precision
}
_REV_DTYPE_MAP = {v: k for k, v in _DTYPE_MAP.items()}

def dumps(tensor: np.ndarray) -> bytes:
    """Serialize a numpy array into bytes (Zero-Copy)."""
    # 1. Validation
    if tensor.dtype not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {tensor.dtype}")
    
    dtype_code = _DTYPE_MAP[tensor.dtype]
    shape = tensor.shape
    ndim = len(shape)
    
    # 2. Header (8 Bytes): Magic + Ver + Flags + Dtype + Ndim
    header = struct.pack('<4sBBBB', _MAGIC, _VERSION, 0, dtype_code, ndim)
    
    # 3. Shape Block (Variable): Ndim * uint32
    shape_block = struct.pack(f'<{ndim}I', *shape)
    
    # 4. Body (Zero-Copy): Raw memory dump
    return header + shape_block + tensor.tobytes()

def loads(data: bytes) -> np.ndarray:
    """Deserialize bytes back into a numpy array."""
    # 1. Validation: Minimum size (Header is 8 bytes)
    if len(data) < 8:
        raise ValueError("Packet too short to contain header")

    # 2. Parse Header
    magic, ver, flags, dtype_code, ndim = struct.unpack('<4sBBBB', data[:8])
    
    if magic != _MAGIC:
        raise ValueError("Invalid tenso packet (Magic bytes mismatch)")
    
    if ver > _VERSION:
        raise ValueError(f"Unsupported version: {ver} (Library supports v{_VERSION})")

    if dtype_code not in _REV_DTYPE_MAP:
        raise ValueError(f"Unknown dtype code: {dtype_code}")

    # 3. Parse Shape
    shape_start = 8
    shape_end = 8 + (ndim * 4)
    
    if len(data) < shape_end:
        raise ValueError("Packet too short to contain shape data")
        
    shape = struct.unpack(f'<{ndim}I', data[shape_start:shape_end])
    
    # 4. Validate Body Size (The "Corrupted Data" Check)
    dtype = _REV_DTYPE_MAP[dtype_code]
    itemsize = dtype.itemsize
    expected_body_size = 1
    for dim in shape:
        expected_body_size *= dim
    expected_body_size *= itemsize
    
    expected_total_size = shape_end + expected_body_size
    
    if len(data) != expected_total_size:
        raise ValueError(f"Corrupted packet: Expected {expected_total_size} bytes, got {len(data)}")
    
    # 5. Parse Body
    return np.frombuffer(data, dtype=dtype, offset=shape_end).reshape(shape)

# File I/O Helpers
def dump(tensor: np.ndarray, fp) -> None:
    fp.write(dumps(tensor))

def load(fp) -> np.ndarray:
    return loads(fp.read())