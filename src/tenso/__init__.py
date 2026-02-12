"""
Tenso: High-performance tensor serialization and streaming.

This package provides efficient serialization, deserialization, and streaming of numpy arrays (tensors),
with optional support for asynchronous and GPU-accelerated workflows.

Main API:
    - dumps, loads, dump, load: Core serialization/deserialization functions.
    - read_stream, write_stream: Stream-based I/O.
    - aread_stream: Async stream reader (if available).
    - read_to_device: GPU direct transfer (if available).
    - get_packet_info, is_aligned: Utilities for packet inspection and alignment.
"""

from .core import dumps, loads, dump, load, read_stream, write_stream, iter_dumps
from .quantize import QuantizedTensor
from .utils import get_packet_info, is_aligned


# Optional Async support
try:
    from .async_core import aread_stream
except ImportError:
    aread_stream = None

# Optional GPU support
try:
    from .gpu import read_to_device
except ImportError:
    read_to_device = None

try:
    from importlib.metadata import version as _version

    __version__ = _version("tenso")
except Exception:
    __version__ = "0.6.1"  # Fallback to pyproject.toml value

__all__ = [
    "dumps",
    "loads",
    "dump",
    "load",
    "read_stream",
    "write_stream",
    "aread_stream",
    "read_to_device",
    "get_packet_info",
    "is_aligned",
    "iter_dumps",
    "QuantizedTensor",
    "TensoShm",
    "TensoCache",
]

# Optional SHM support
try:
    from .shm import TensoShm
except ImportError:
    TensoShm = None

# Optional Cache support
try:
    from .cache import TensoCache
except ImportError:
    TensoCache = None

# Ray integration is imported explicitly via tenso.ray
# Usage: from tenso.ray import register
