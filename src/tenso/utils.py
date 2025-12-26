import ctypes
import struct

import numpy as np

from .config import _MAGIC, _REV_DTYPE_MAP, FLAG_INTEGRITY


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
    return (
        ctypes.addressof(ctypes.c_char.from_buffer(bytearray(data))) % alignment
    ) == 0


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
    """
    if len(data) < 8:
        raise ValueError("Packet too short")
    magic, ver, flags, dtype_code, ndim = struct.unpack("<4sBBBB", data[:8])
    if magic != _MAGIC:
        raise ValueError("Invalid tenso packet")

    shape_end = 8 + (ndim * 4)
    if len(data) < shape_end:
        raise ValueError("Packet too short to contain shape")
    shape = struct.unpack(f"<{ndim}I", data[8:shape_end])
    dtype = _REV_DTYPE_MAP.get(dtype_code, None)

    return {
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
