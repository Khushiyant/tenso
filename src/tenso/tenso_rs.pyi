"""
Type stubs for the Rust extension module.

This file helps IDEs provide autocomplete and type checking
for functions implemented in Rust via PyO3.
"""

import numpy as np
from numpy.typing import NDArray

def dumps_rs(
    tensor: NDArray,
    check_integrity: bool = False,
    alignment: int = 64
) -> bytes:
    """
    Rust-accelerated serialization (internal use only).
    
    Use tenso.dumps() instead of calling this directly.
    
    Parameters
    ----------
    tensor : NDArray
        NumPy array to serialize (must be C-contiguous).
    check_integrity : bool, default False
        Include XXH3 checksum footer.
    alignment : int, default 64
        Memory alignment boundary (power of 2).
    
    Returns
    -------
    bytes
        Tenso packet bytes.
    """
    ...

def loads_rs(packet: bytes) -> NDArray:
    """
    Rust-accelerated deserialization (internal use only).
    
    Use tenso.loads() instead of calling this directly.
    
    Parameters
    ----------
    packet : bytes
        Tenso packet to deserialize.
    
    Returns
    -------
    NDArray
        Reconstructed NumPy array.
    """
    ...

def dump_to_fd_rs(
    fd: int,
    tensor: NDArray,
    check_integrity: bool = False
) -> int:
    """
    Write tensor to file descriptor (Unix only, internal use).
    
    Parameters
    ----------
    fd : int
        Raw file descriptor.
    tensor : NDArray
        NumPy array to write.
    check_integrity : bool, default False
        Include integrity hash.
    
    Returns
    -------
    int
        Number of bytes written.
    """
    ...

def get_packet_info_rs(packet: bytes) -> tuple:
    """
    Extract metadata from Tenso packet header (internal use).
    
    Returns
    -------
    tuple
        (dtype_code, shape, flags, version)
    """
    ...
