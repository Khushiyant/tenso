import struct
import numpy as np
from typing import Union, Optional, Tuple, Any
from .config import _MAGIC, _ALIGNMENT, _REV_DTYPE_MAP
from .core import _read_into_buffer

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

# Preference: CuPy > PyTorch > JAX (Arbitrary, can be user-defined later)
if HAS_CUPY:
    BACKEND = 'cupy'
elif HAS_TORCH:
    BACKEND = 'torch'
elif HAS_JAX:
    BACKEND = 'jax'
else:
    BACKEND = None

def _get_allocator(size: int) -> Tuple[np.ndarray, Any]:
    """
    Allocate a pinned (page-locked) memory buffer for fast GPU transfer.
    
    This internal function allocates host memory that is pinned (page-locked)
    to enable fast, asynchronous DMA transfers to GPU devices. The allocation
    method depends on the available GPU backend (CuPy, PyTorch, or JAX).
    
    Args:
        size: Number of bytes to allocate.
    
    Returns:
        Tuple[np.ndarray, Any]: A tuple containing:
            - A numpy array view of the allocated pinned memory
            - The backend-specific handle/object for the allocation
    
    Raises:
        ImportError: If no supported GPU backend is available.
    """
    if BACKEND == 'cupy':
        mem = cp.cuda.alloc_pinned_memory(size)
        return np.frombuffer(mem, dtype=np.uint8, count=size), mem
        
    elif BACKEND == 'torch':
        tensor = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        return tensor.numpy(), tensor
    
    elif BACKEND == 'jax':
        # JAX doesn't expose a direct "pinned memory allocator" easily via Python API 
        # that returns a numpy view without copies. 
        # Fallback: Standard numpy array (OS may optimize) or use PyTorch/CuPy if available.
        # For pure JAX env, we stick to standard numpy allocation.
        arr = np.empty(size, dtype=np.uint8)
        return arr, None
        
    else:
        raise ImportError("Tenso GPU support requires 'cupy', 'torch', or 'jax' installed.")

def read_to_device(source: Any, device_id: int = 0) -> Any:
    """
    Read a Tenso packet directly into pinned memory and transfer it to GPU.
    
    This function reads a Tenso packet from a stream source, allocates pinned
    host memory, and transfers the tensor directly to GPU memory using the
    available backend (CuPy, PyTorch, or JAX). This minimizes CPU usage and
    maximizes transfer speed.
    
    Args:
        source: Stream source to read the packet from (file, socket, etc.).
        device_id: GPU device ID to transfer to. Default 0.
    
    Returns:
        GPU tensor object: The tensor in GPU memory. Type depends on backend:
            - CuPy: cupy.ndarray
            - PyTorch: torch.Tensor
            - JAX: jax.Array
    
    Raises:
        ValueError: If the packet is invalid or unsupported.
        EOFError: If the stream ends prematurely.
        ImportError: If no GPU backend is available.
    """
    # 1. Read Header
    header = bytearray(8)
    if not _read_into_buffer(source, header):
        return None
    
    magic, ver, flags, dtype_code, ndim = struct.unpack('<4sBBBB', header)
    if magic != _MAGIC: raise ValueError("Invalid tenso packet")

    # 2. Read Shape
    shape_len = ndim * 4
    shape_bytes = bytearray(shape_len)
    if not _read_into_buffer(source, shape_bytes):
        raise EOFError("Stream ended during shape read")
        
    shape = struct.unpack(f'<{ndim}I', shape_bytes)
    
    # 3. Calculate Layout
    dtype_np = _REV_DTYPE_MAP.get(dtype_code)
    if dtype_np is None: raise ValueError(f"Unknown dtype: {dtype_code}")

    current_pos = 8 + shape_len
    remainder = current_pos % _ALIGNMENT
    padding_len = 0 if remainder == 0 else (_ALIGNMENT - remainder)
    
    body_len = int(np.prod(shape) * dtype_np.itemsize)
    total_read_len = padding_len + body_len
    
    # 4. Allocate Memory (Pinned if possible)
    host_view, host_handle = _get_allocator(total_read_len)
    
    # 5. Read
    pinned_view = memoryview(host_view)
    if not _read_into_buffer(source, pinned_view):
        raise EOFError("Stream ended during body read")
        
    # 6. Transfer to GPU
    body_view = host_view[padding_len:]
    
    if BACKEND == 'cupy':
        with cp.cuda.Device(device_id):
            host_typed = body_view.view(dtype=dtype_np).reshape(shape)
            return cp.array(host_typed) 
            
    elif BACKEND == 'torch':
        host_typed = torch.from_numpy(body_view.view(dtype=dtype_np).reshape(shape))
        return host_typed.to(device=f'cuda:{device_id}', non_blocking=True)
        
    elif BACKEND == 'jax':
        # JAX Device Put (Async dispatch)
        host_typed = body_view.view(dtype=dtype_np).reshape(shape)
        try:
            device = jax.devices()[device_id]
        except IndexError:
            device = jax.devices()[0] # Fallback
        return jax.device_put(host_typed, device=device)