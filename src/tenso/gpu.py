"""
GPU Acceleration for Tenso.

Implements fast transfers between device memory (CuPy/PyTorch/JAX)
and Tenso streams using pinned host memory.
"""

import struct
import numpy as np
from typing import Any, Tuple
from .config import _MAGIC, _ALIGNMENT, _REV_DTYPE_MAP
from .core import _read_into_buffer, dumps

# --- BACKEND DETECTION ---
BACKEND = None
try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    import jax

    HAS_JAX = True
except ImportError:
    jax = None
    HAS_JAX = False

if HAS_CUPY:
    BACKEND = "cupy"
elif HAS_TORCH:
    BACKEND = "torch"
elif HAS_JAX:
    BACKEND = "jax"


def _get_allocator(size: int) -> Tuple[np.ndarray, Any]:
    """Allocate pinned host memory for fast GPU transfer."""
    if BACKEND == "cupy":
        mem = cp.cuda.alloc_pinned_memory(size)
        return np.frombuffer(mem, dtype=np.uint8, count=size), mem
    elif BACKEND == "torch":
        tensor = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        return tensor.numpy(), tensor
    return np.empty(size, dtype=np.uint8), None


def write_from_device(tensor: Any, dest: Any, check_integrity: bool = False) -> int:
    """
    Serialize a GPU tensor directly to an I/O stream.

    Parameters
    ----------
    tensor : Any
        A GPU-resident array (CuPy, PyTorch, or JAX).
    dest : Any
        Destination with .write() method.
    check_integrity : bool, default False
        Include XXH3 checksum.

    Returns
    -------
    int
        Number of bytes written.
    """
    if HAS_CUPY and isinstance(tensor, cp.ndarray):
        host_arr = cp.asnumpy(tensor)
    elif HAS_TORCH and isinstance(tensor, torch.Tensor):
        host_arr = tensor.detach().cpu().numpy()
    elif HAS_JAX and hasattr(tensor, "device"):
        host_arr = np.asarray(tensor)
    else:
        host_arr = np.asarray(tensor)

    packet = dumps(host_arr, check_integrity=check_integrity)
    dest.write(packet)
    return len(packet)


def read_to_device(source: Any, device_id: int = 0) -> Any:
    """
    Read a Tenso packet from a stream directly into GPU memory.

    Parameters
    ----------
    source : Any
        Stream-like object (file, socket).
    device_id : int, default 0
        The target GPU device ID.

    Returns
    -------
    Any
        The GPU tensor.
    """
    header = bytearray(8)
    if not _read_into_buffer(source, header):
        return None
    magic, _, _, dtype_code, ndim = struct.unpack("<4sBBBB", header)
    if magic != _MAGIC:
        raise ValueError("Invalid tenso packet")

    shape_bytes = bytearray(ndim * 4)
    if not _read_into_buffer(source, shape_bytes):
        raise EOFError("Stream ended during shape read")
    shape = struct.unpack(f"<{ndim}I", shape_bytes)
    dtype_np = _REV_DTYPE_MAP.get(dtype_code)

    current_pos = 8 + (ndim * 4)
    padding_len = (_ALIGNMENT - (current_pos % _ALIGNMENT)) % _ALIGNMENT
    body_len = int(np.prod(shape) * dtype_np.itemsize)

    host_view, _ = _get_allocator(padding_len + body_len)
    try:
        if not _read_into_buffer(source, host_view):
            raise EOFError("Stream ended during body read")
    except EOFError as e:
        raise EOFError(f"Stream ended during body read. {e}") from None

    body_view = host_view[padding_len:].view(dtype=dtype_np).reshape(shape)

    if BACKEND == "cupy":
        with cp.cuda.Device(device_id):
            return cp.array(body_view)
    elif BACKEND == "torch":
        return torch.from_numpy(body_view).to(
            device=f"cuda:{device_id}", non_blocking=True
        )
    elif BACKEND == "jax":
        return jax.device_put(body_view, device=jax.devices()[device_id])
    return body_view
