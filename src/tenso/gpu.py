"""
GPU Acceleration for Tenso.

Implements fast transfers between device memory (CuPy/PyTorch/JAX)
and Tenso streams using pinned host memory.
"""

import struct
import numpy as np
import xxhash
from typing import Any, Tuple
from .config import (
    _ALIGNMENT,
    _REV_DTYPE_MAP,
    FLAG_ALIGNED,
    FLAG_BUNDLE,
    FLAG_COMPRESSION,
    FLAG_CUST_ALIGN,
    FLAG_INTEGRITY,
    FLAG_SPARSE,
    FLAG_SPARSE_CSC,
    FLAG_SPARSE_CSR,
    MAX_ELEMENTS,
    MAX_NDIM,
)
from .core import _QUANTIZED_CODES, _read_into_buffer, _read_packet_header, dumps

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
    Serialize a GPU tensor directly to an I/O stream using pinned memory staging.

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
        # Use pinned memory for fast transfer
        host_view, _ = _get_allocator(tensor.nbytes)
        # Reshape view to match tensor for copy
        target = host_view.view(tensor.dtype).reshape(tensor.shape)
        tensor.get(out=target)
        host_arr = target
    elif HAS_TORCH and isinstance(tensor, torch.Tensor):
        # Torch can move directly to pinned memory
        host_arr = tensor.detach().to("cpu", pin_memory=True).numpy()
    elif HAS_JAX and hasattr(tensor, "device"):
        # JAX currently doesn't have a simple pinned memory API like CuPy
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

    Raises
    ------
    ValueError
        If packet is invalid or integrity check fails.
    EOFError
        If stream ends prematurely.
    """
    parsed = _read_packet_header(source)
    if parsed is None:
        return None
    _ver, flags, dtype_code, ndim, header_size = parsed

    # GPU-direct transfer targets a single dense, uncompressed tensor. Bundles,
    # sparse, compressed, and quantized payloads must go through loads().
    if flags & (FLAG_BUNDLE | FLAG_SPARSE | FLAG_SPARSE_CSR | FLAG_SPARSE_CSC):
        raise ValueError(
            "read_to_device supports only dense tensors; read bundles/sparse via loads()"
        )
    if flags & FLAG_COMPRESSION:
        raise ValueError(
            "read_to_device does not support compressed packets; use loads()"
        )
    if dtype_code in _QUANTIZED_CODES:
        raise ValueError(
            "read_to_device does not support quantized packets; use loads()"
        )

    if ndim > MAX_NDIM:
        raise ValueError(f"Packet exceeds maximum dimensions ({ndim} > {MAX_NDIM})")

    shape_bytes = bytearray(ndim * 4)
    if not _read_into_buffer(source, shape_bytes):
        raise EOFError("Stream ended during shape read")
    shape = struct.unpack(f"<{ndim}I", shape_bytes)
    # np.prod(()) == 1, so a 0-dim scalar counts as one element (matching loads).
    num_elements = int(np.prod(shape))
    if num_elements > MAX_ELEMENTS:
        raise ValueError(f"Packet exceeds maximum elements ({num_elements})")

    dtype_np = _REV_DTYPE_MAP.get(dtype_code)
    if dtype_np is None:
        raise ValueError(f"Unsupported dtype code: {dtype_code}")

    # Resolve alignment from flags (custom-align exponent byte, 64-byte default,
    # or unaligned) instead of assuming 64.
    current_pos = header_size + (ndim * 4)
    if flags & FLAG_CUST_ALIGN:
        exp_buf = bytearray(1)
        if not _read_into_buffer(source, exp_buf):
            raise EOFError("Stream ended during alignment byte read")
        alignment = 1 << exp_buf[0]
        current_pos += 1
    elif flags & FLAG_ALIGNED:
        alignment = _ALIGNMENT
    else:
        alignment = 1

    padding_len = (alignment - (current_pos % alignment)) % alignment
    body_len = num_elements * dtype_np.itemsize
    has_integrity = (flags & FLAG_INTEGRITY) != 0
    footer_len = 8 if has_integrity else 0

    host_view, _owner = _get_allocator(padding_len + body_len + footer_len)
    try:
        if not _read_into_buffer(source, host_view):
            raise EOFError("Stream ended during body read")
    except EOFError as e:
        raise EOFError(f"Stream ended during body read. {e}") from None

    # Verify integrity if flag is set
    if has_integrity:
        body_data = host_view[padding_len : padding_len + body_len]
        actual_hash = xxhash.xxh3_64_intdigest(body_data)
        expected_hash = struct.unpack(
            "<Q", host_view[padding_len + body_len : padding_len + body_len + 8]
        )[0]
        if actual_hash != expected_hash:
            raise ValueError("Integrity check failed: XXH3 mismatch")

    body_view = host_view[padding_len : padding_len + body_len].view(dtype=dtype_np).reshape(shape)

    if BACKEND == "cupy":
        with cp.cuda.Device(device_id):
            return cp.array(body_view)  # synchronous H2D copy
    elif BACKEND == "torch":
        result = torch.from_numpy(body_view).to(
            device=f"cuda:{device_id}", non_blocking=True
        )
        # The pinned staging buffer (_owner) backs the async copy; block until
        # it completes so the buffer can be released without a use-after-free.
        torch.cuda.synchronize(device_id)
        return result
    elif BACKEND == "jax":
        return jax.device_put(body_view, device=jax.devices()[device_id])
    return body_view


# ---------------------------------------------------------------------------
# GPU Direct Storage / RDMA Abstraction
# ---------------------------------------------------------------------------

class GPUDirectTransfer:
    """
    Abstraction layer for GPU-Direct Storage and RDMA transfers.

    When available, uses NVIDIA GPUDirect Storage (GDS) to bypass CPU
    staging entirely — network/storage data lands directly in GPU memory.
    Falls back gracefully to pinned-memory staging when GDS is unavailable.

    Example::

        gdt = GPUDirectTransfer(device_id=0)

        # From a file descriptor (e.g., NVMe SSD or network socket)
        tensor = gdt.read_from_fd(fd, shape=(1, 3, 224, 224), dtype=np.float32)

        # From a Tenso packet buffer already in host memory
        tensor = gdt.read_packet(packet_bytes)
    """

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self._gds_available = False
        self._backend = BACKEND

        # Detect GDS (cuFile) availability
        if HAS_CUPY:
            try:
                import kvikio
                self._gds_available = True
                self._kvikio = kvikio
            except ImportError:
                pass

    @property
    def gds_available(self) -> bool:
        return self._gds_available

    def read_from_fd(
        self,
        fd: int,
        shape: tuple,
        dtype: np.dtype,
        offset: int = 0,
    ) -> Any:
        """
        Read raw tensor data from a file descriptor directly into GPU memory.

        Uses GPUDirect Storage (kvikio/cuFile) when available, bypassing
        CPU RAM entirely. Falls back to pinned memory staging otherwise.

        Parameters
        ----------
        fd : int
            OS-level file descriptor.
        shape : tuple
            Tensor shape.
        dtype : np.dtype
            Element data type.
        offset : int
            Byte offset into the file where data begins.

        Returns
        -------
        GPU tensor (CuPy, PyTorch, or JAX array).
        """
        dtype = np.dtype(dtype)
        nbytes = int(np.prod(shape)) * dtype.itemsize

        if self._gds_available and self._backend == "cupy":
            return self._gds_read(fd, shape, dtype, nbytes, offset)

        return self._staged_read(fd, shape, dtype, nbytes, offset)

    def _gds_read(self, fd, shape, dtype, nbytes, offset):
        """GPU-Direct Storage path via kvikio."""
        import os

        with cp.cuda.Device(self.device_id):
            gpu_buf = cp.empty(int(np.prod(shape)), dtype=dtype)

            fobj = os.fdopen(fd, 'rb', closefd=False)
            try:
                f = self._kvikio.CuFile(fobj)
                try:
                    bytes_read = f.pread(gpu_buf, nbytes, file_offset=offset)
                finally:
                    f.close()
            finally:
                fobj.close()

            if bytes_read != nbytes:
                raise IOError(
                    f"GDS short read: expected {nbytes}, got {bytes_read}"
                )

            return gpu_buf.reshape(shape)

    def _staged_read(self, fd, shape, dtype, nbytes, offset):
        """Fallback: pinned memory staging."""
        import os

        host_view, _ = _get_allocator(nbytes)
        f = os.fdopen(fd, 'rb', closefd=False)
        f.seek(offset)

        pos = 0
        mv = memoryview(host_view)
        while pos < nbytes:
            chunk = f.read(nbytes - pos)
            if not chunk:
                raise EOFError(f"Short read: expected {nbytes}, got {pos}")
            mv[pos:pos + len(chunk)] = chunk
            pos += len(chunk)

        arr = host_view[:nbytes].view(dtype).reshape(shape)

        if self._backend == "cupy":
            with cp.cuda.Device(self.device_id):
                return cp.array(arr)  # synchronous H2D copy
        elif self._backend == "torch":
            result = torch.from_numpy(arr).to(
                device=f"cuda:{self.device_id}", non_blocking=True
            )
            # Block until the async copy finishes so the pinned staging buffer
            # can be freed without a use-after-free.
            torch.cuda.synchronize(self.device_id)
            return result
        elif self._backend == "jax":
            return jax.device_put(arr, device=jax.devices()[self.device_id])
        return arr

    def read_packet(self, source: Any, device_id: int = None) -> Any:
        """
        Read a Tenso packet and transfer to GPU.

        Delegates to read_to_device() but can be extended with GDS
        for direct-from-network transfers.
        """
        did = device_id if device_id is not None else self.device_id
        return read_to_device(source, device_id=did)

    def __repr__(self) -> str:
        gds = "GDS" if self._gds_available else "staged"
        return f"GPUDirectTransfer(device={self.device_id}, mode={gds}, backend={self._backend})"
