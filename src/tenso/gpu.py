"""
GPU Acceleration for Tenso.

Fast transfers between device memory and Tenso streams using pinned buffers.
Supports CuPy, PyTorch, and JAX.
"""

import struct
from typing import Any, Optional, Tuple

import numpy as np

from .config import _ALIGNMENT, _MAGIC, _REV_DTYPE_MAP
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
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    jax = None
    HAS_JAX = False

# Preference: CuPy > PyTorch > JAX
if HAS_CUPY:
    BACKEND = "cupy"
elif HAS_TORCH:
    BACKEND = "torch"
elif HAS_JAX:
    BACKEND = "jax"
else:
    BACKEND = None


def _get_allocator(size: int) -> Tuple[np.ndarray, Any]:
    """Allocate pinned host memory for fast GPU transfer.

    Parameters
    ----------
    size : int
        The size in bytes to allocate.

    Returns
    -------
    Tuple[np.ndarray, Any]
        The allocated array and backend-specific memory object.
    """
    if BACKEND == "cupy":
        mem = cp.cuda.alloc_pinned_memory(size)
        return np.frombuffer(mem, dtype=np.uint8, count=size), mem
    elif BACKEND == "torch":
        tensor = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        return tensor.numpy(), tensor
    elif BACKEND == "jax":
        # JAX doesn't expose a direct pinned allocator; fallback to standard numpy
        arr = np.empty(size, dtype=np.uint8)
        return arr, None
    else:
        raise ImportError(
            "Tenso GPU support requires 'cupy', 'torch', or 'jax' installed."
        )


def write_from_device(tensor: Any, dest: Any, check_integrity: bool = False) -> int:
    """
    Serialize a GPU tensor directly to an I/O stream.

    Parameters
    ----------
    tensor : Any
        A GPU-resident array (CuPy ndarray, PyTorch Tensor, or JAX Array).
    dest : Any
        A file-like object with a .write() method.
    check_integrity : bool, optional
        Include XXH3 hash for verification. Default is False.

    Returns
    -------
    int
        Number of bytes written.
    """
    if HAS_CUPY and isinstance(tensor, cp.ndarray):
        host_arr = cp.asnumpy(tensor)
    elif HAS_TORCH and isinstance(tensor, torch.Tensor):
        host_arr = tensor.detach().cpu().numpy()
    elif HAS_JAX and isinstance(tensor, (jax.Array, np.ndarray)):
        host_arr = np.asarray(tensor)
    else:
        host_arr = np.asarray(tensor)

    packet = dumps(host_arr, check_integrity=check_integrity)
    dest.write(packet)
    return len(packet)


def read_to_device(source: Any, device_id: int = 0) -> Any:
    """
    Read a Tenso packet directly into pinned memory and transfer to GPU.

    Parameters
    ----------
    source : Any
        Stream source to read the packet from (file, socket, etc.).
    device_id : int, optional
        GPU device ID to transfer to. Default is 0.

    Returns
    -------
    Any
        GPU tensor object: The tensor in GPU memory (CuPy, PyTorch, or JAX).
    """
    header = bytearray(8)
    if not _read_into_buffer(source, header):
        return None

    magic, ver, flags, dtype_code, ndim = struct.unpack("<4sBBBB", header)
    if magic != _MAGIC:
        raise ValueError("Invalid tenso packet")

    shape_bytes = bytearray(ndim * 4)
    if not _read_into_buffer(source, shape_bytes):
        raise EOFError("Stream ended during shape read")

    shape = struct.unpack(f"<{ndim}I", shape_bytes)
    dtype_np = _REV_DTYPE_MAP.get(dtype_code)
    if dtype_np is None:
        raise ValueError(f"Unknown dtype: {dtype_code}")

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
        try:
            device = jax.devices()[device_id]
        except IndexError:
            device = jax.devices()[0]
        return jax.device_put(body_view, device=device)

    return body_view
