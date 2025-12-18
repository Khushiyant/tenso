import struct
import numpy as np
from typing import Union, Optional, Tuple, Any
from .config import _MAGIC, _ALIGNMENT, _REV_DTYPE_MAP
from .core import _read_into_buffer

# --- BACKEND DETECTION ---
BACKEND = None

# --- BACKEND DETECTION ---
# We try to import both to ensure type hints work (especially for Sphinx)
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

# Determine active backend (prefer CuPy if available)
if HAS_CUPY:
    BACKEND = 'cupy'
elif HAS_TORCH:
    BACKEND = 'torch'
else:
    BACKEND = None

def _get_allocator(size: int) -> Tuple[np.ndarray, Any]:
    """
    Allocate a pinned (page-locked) memory buffer for fast GPU transfer.

    Pinned memory allows the GPU to read directly from RAM via DMA, bypassing the CPU cache.

    Args:
        size: Number of bytes to allocate.

    Returns:
        tuple: (numpy view of pinned memory, backend-specific pinned memory handle)

    Raises:
        ImportError: If neither 'cupy' nor 'torch' is installed.
    """
    if BACKEND == 'cupy':
        # CuPy Pinned Memory
        mem = cp.cuda.alloc_pinned_memory(size)
        # Create a numpy view over that pinned memory
        return np.frombuffer(mem, dtype=np.uint8, count=size), mem
        
    elif BACKEND == 'torch':
        # PyTorch Pinned Memory
        # allocate a byte tensor in pinned memory
        tensor = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        # return numpy view (shares memory)
        return tensor.numpy(), tensor
        
    else:
        raise ImportError("Tenso GPU support requires 'cupy' or 'torch' installed.")

def read_to_device(source: Any, device_id: int = 0) -> Union['cp.ndarray', 'torch.Tensor', None]:
    """
    Read a Tenso packet directly into pinned memory and transfer it to GPU.

    This is the fastest way to move network data onto a GPU, using pinned memory and async transfer.

    Args:
        source: The data source (socket or file-like object).
        device_id: GPU device index to transfer to.

    Returns:
        cupy.ndarray or torch.Tensor or None: The tensor on GPU, or None if EOF at start.

    Raises:
        EOFError: If the stream ends unexpectedly during read.
        ValueError: If the packet is invalid or dtype is unknown.
        ImportError: If neither 'cupy' nor 'torch' is installed.
    """
    # 1. Read Header (reuse core logic)
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
    
    # 4. Allocate PINNED Memory (The Secret Sauce)
    # We allocate enough for padding + body. 
    # host_handle is kept to prevent GC of the pinned memory during read.
    host_view, host_handle = _get_allocator(total_read_len)
    
    # 5. Read directly into Pinned Memory
    # _read_into_buffer handles recv_into, readinto, and partial reads automatically.
    # We create a memoryview of the pinned buffer.
    pinned_view = memoryview(host_view)
    if not _read_into_buffer(source, pinned_view):
        raise EOFError("Stream ended during body read")
        
    # 6. ASYNC Transfer to GPU (DMA)
    # The CPU job is done. The DMA engine takes over.
    body_view = host_view[padding_len:] # Skip padding
    
    if BACKEND == 'cupy':
        with cp.cuda.Device(device_id):
            # View as correct dtype
            host_typed = body_view.view(dtype=dtype_np).reshape(shape)
            # cp.array(host_typed) triggers the D2D copy (or H2D from pinned)
            return cp.array(host_typed) 
            
    elif BACKEND == 'torch':
        # View as correct dtype
        host_typed = torch.from_numpy(body_view.view(dtype=dtype_np).reshape(shape))
        
        # non_blocking=True is crucial here to allow CPU to return immediately
        return host_typed.to(device=f'cuda:{device_id}', non_blocking=True)