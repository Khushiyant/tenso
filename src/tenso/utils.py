import struct
import numpy as np
from .config import _QDTYPE_NAMES, _REV_DTYPE_MAP, FLAG_BUNDLE, FLAG_INTEGRITY
from .core import _parse_header

# --- RUST INTEGRATION ---
try:
    from .tenso_rs import get_packet_info_rs

    HAS_RUST = True
except ImportError:
    HAS_RUST = False


def is_aligned(data: bytes, alignment: int = 64) -> bool:
    """
    Check if the given bytes data is aligned to the specified boundary.

    Parameters
    ----------
    data : bytes
        The bytes object to check alignment for.
    alignment : int, optional
        The alignment boundary in bytes. Default is 64.

    Returns
    -------
    bool
        True if the data is aligned, False otherwise.
    """
    mv = memoryview(data)
    if mv.nbytes == 0:
        return True
    # np.frombuffer is zero-copy, so .ctypes.data is the caller's own buffer address.
    addr = np.frombuffer(mv, dtype=np.uint8).ctypes.data
    return (addr % alignment) == 0


def get_packet_info(data: bytes) -> dict:
    """
    Extract metadata from a Tenso packet without deserializing the full tensor.

    This function parses the header of a Tenso packet to provide information
    about the tensor's properties, such as dtype, shape, and flags.

    Parameters
    ----------
    data : bytes
        The raw bytes of the Tenso packet.

    Returns
    -------
    dict
        A dictionary containing packet information with keys:
        - 'version': Protocol version
        - 'dtype': NumPy dtype of the tensor
        - 'shape': Tuple representing tensor shape
        - 'ndim': Number of dimensions
        - 'flags': Raw flags byte
        - 'aligned': Boolean indicating if packet uses alignment
        - 'integrity_protected': Boolean indicating if integrity check is enabled
        - 'total_elements': Total number of elements in the tensor
        - 'data_size_bytes': Size of the tensor data in bytes

    Raises
    ------
    ValueError
        If the packet is too short or invalid.

    Uses Rust implementation for performance if available, otherwise falls back to Python.
    """
    # Fast path (Rust).
    if HAS_RUST:
        try:
            info = get_packet_info_rs(data)

            # Map Rust's 'dtype_code' (int) to the 'dtype' (np.dtype) Python expects.
            if "dtype" not in info and "dtype_code" in info:
                dc = info["dtype_code"]
                if dc in _QDTYPE_NAMES:
                    info["dtype"] = _QDTYPE_NAMES[dc]
                    info["quantized"] = True
                    if "total_elements" in info:
                        total = info["total_elements"]
                        if dc in (18, 19):
                            info["data_size_bytes"] = (total + 1) // 2
                        else:
                            info["data_size_bytes"] = total
                else:
                    dtype = _REV_DTYPE_MAP.get(dc)
                    info["dtype"] = dtype
                    if "data_size_bytes" not in info and "total_elements" in info:
                        itemsize = dtype.itemsize if dtype else 0
                        info["data_size_bytes"] = info["total_elements"] * itemsize

            return info
        except ValueError as e:
            # Rust raises ValueError on bad packets; propagate.
            raise e

    # Slow path (Python fallback).
    mv = memoryview(data)
    ver, flags, dtype_code, ndim, header_base = _parse_header(mv)

    if flags & FLAG_BUNDLE:
        # Bundle: `ndim` is the entry count; post-header bytes are key-length prefixes.
        return {
            "version": ver,
            "dtype": None,
            "shape": (),
            "ndim": ndim,
            "entry_count": ndim,
            "flags": flags,
            "aligned": bool(flags & 1),
            "integrity_protected": bool(flags & FLAG_INTEGRITY),
            "total_elements": 0,
            "data_size_bytes": 0,
        }

    shape_end = header_base + (ndim * 4)
    if len(mv) < shape_end:
        raise ValueError("Packet too short to contain shape")
    shape = struct.unpack(f"<{ndim}I", mv[header_base:shape_end])
    dtype = _REV_DTYPE_MAP.get(dtype_code, None)

    info = {
        "version": ver,
        "dtype": dtype,
        "shape": shape,
        "ndim": ndim,
        "flags": flags,
        "aligned": bool(flags & 1),
        "integrity_protected": bool(flags & FLAG_INTEGRITY),
        "total_elements": int(np.prod(shape)),
        "data_size_bytes": int(np.prod(shape)) * (dtype.itemsize if dtype else 0),
    }

    # Quantized dtype resolution
    if dtype_code in _QDTYPE_NAMES:
        info["dtype"] = _QDTYPE_NAMES[dtype_code]
        info["quantized"] = True
        total = int(np.prod(shape))
        if dtype_code in (18, 19):  # 4-bit types
            info["data_size_bytes"] = (total + 1) // 2
        else:
            info["data_size_bytes"] = total

    return info
