"""
Type stubs for the Rust extension module.

This file helps IDEs provide autocomplete and type checking
for functions implemented in Rust via PyO3.

Every signature here must match the `#[pyo3(signature = ...)]` attribute on the
corresponding function in ``src/lib.rs``. These are internal entry points; use the
``tenso.*`` wrappers instead of calling them directly.
"""

from typing import Any, Sequence

from numpy.typing import NDArray

# --- Encode ---------------------------------------------------------------

def dumps_rs(
    array: Any,
    check_integrity: bool = False,
    compress: bool = False,
    alignment: int = 64,
) -> bytes:
    """Serialize a dense array, sparse matrix, or dict bundle to a packet.

    Parameters
    ----------
    array : Any
        NumPy array (must be C-contiguous), scipy sparse matrix, or dict.
    check_integrity : bool, default False
        Append an 8-byte XXH3 checksum footer.
    compress : bool, default False
        LZ4-frame the body.
    alignment : int, default 64
        Body alignment boundary (power of two).
    """
    ...

def dumps_quantized_rs(
    qt: Any,
    check_integrity: bool = False,
    alignment: int = 64,
) -> bytes:
    """Serialize a :class:`tenso.QuantizedTensor`."""
    ...

def dumps_string_rs(
    offsets: Any,
    payload: Any,
    count: int,
    check_integrity: bool = False,
) -> bytes:
    """Serialize a packed string tensor from ``(count + 1)`` u64 LE offsets."""
    ...

def encode_bundle_rs(entries: Sequence[tuple[str, bytes]]) -> bytes:
    """Frame already-encoded ``(key, value_packet)`` pairs into a bundle packet.

    The wire format encodes the entry count in a single byte, so at most 255
    entries are supported.
    """
    ...

# --- Encode straight into a destination -----------------------------------

def dump_to_buffer_rs(
    array: Any,
    buffer: Any,
    check_integrity: bool = False,
    compress: bool = False,
    alignment: int = 64,
) -> int:
    """Encode into a caller-provided writable buffer. Returns bytes written."""
    ...

def dump_to_fd_rs(
    array: Any,
    fd: int,
    check_integrity: bool = False,
    compress: bool = False,
    alignment: int = 64,
) -> int:
    """Encode straight to a raw file descriptor. Returns bytes written.

    Writes behind any buffering on the Python file object, so callers must flush
    it first or the packet will be reordered against pending writes.
    """
    ...

# --- Decode ---------------------------------------------------------------

def loads_rs(data: Any) -> Any | None:
    """Decode a packet into an array, dict, sparse matrix, or quantized tensor.

    Returns a zero-copy view onto ``data`` for dense packets; the view holds a
    reference to ``data``, so it stays valid after the caller drops theirs.
    Returns ``None`` for a GPU IPC-reference packet, which is decoded on device
    rather than here.
    """
    ...

def get_packet_info_rs(data: Any) -> dict[str, Any]:
    """Parse a packet header without decoding the body."""
    ...

# --- POSIX process-shared mutex (Unix only) -------------------------------
#
# Each of these validates that ``offset + shm_mutex_size() <= len(buffer)`` and
# raises ValueError otherwise; they write through the resulting pointer.

def shm_mutex_size() -> int:
    """Bytes required for one process-shared mutex."""
    ...

def shm_mutex_init(buffer: Any, offset: int) -> None:
    """Initialize a process-shared mutex at ``offset`` in a writable buffer."""
    ...

def shm_mutex_lock(buffer: Any, offset: int, timeout_secs: float = 5.0) -> bool:
    """Lock the mutex. Returns True if it was recovered from a dead owner."""
    ...

def shm_mutex_unlock(buffer: Any, offset: int) -> None:
    """Unlock the mutex."""
    ...

def shm_mutex_destroy(buffer: Any, offset: int) -> None:
    """Destroy the mutex."""
    ...

__all__ = [
    "dumps_rs",
    "dumps_quantized_rs",
    "dumps_string_rs",
    "encode_bundle_rs",
    "dump_to_buffer_rs",
    "dump_to_fd_rs",
    "loads_rs",
    "get_packet_info_rs",
    "shm_mutex_size",
    "shm_mutex_init",
    "shm_mutex_lock",
    "shm_mutex_unlock",
    "shm_mutex_destroy",
]
