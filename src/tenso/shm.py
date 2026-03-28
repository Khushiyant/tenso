"""
Shared Memory Transport for Tenso.

This module provides high-performance Inter-Process Communication (IPC)
capabilities using POSIX Shared Memory. It allows zero-copy transfer of
tensors between local processes.
"""

from typing import Any, Optional, Self, Union
import numpy as np

from .core import dumps, loads as core_loads

try:
    from multiprocessing import shared_memory
    HAS_SHM = True
except ImportError:
    HAS_SHM = False

try:
    from .tenso_rs import dump_to_buffer_rs, loads_rs
except ImportError:
    # If compiled extension is missing, we can't do the optimized buffer write
    # We could implement a fallback but the whole point is speed.
    dump_to_buffer_rs = None
    loads_rs = None


class TensoShm:
    """
    A wrapper around SharedMemory for Tenso-protocol objects.

    Example::

        # Writer
        ary = np.random.rand(100, 100)
        with TensoShm.create_from("my_tensor", ary) as shm:
            print("Wrote to SHM")
            input("Press enter to cleanup...")

        # Reader
        with TensoShm("my_tensor") as shm:
            ary = shm.get()
            print(ary.shape)
    """

    def __init__(self, name: str, create: bool = False, size: int = 0):
        if not HAS_SHM:
            raise ImportError("multiprocessing.shared_memory is not available")
        
        self._shm = shared_memory.SharedMemory(name=name, create=create, size=size)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def size(self) -> int:
        return self._shm.size

    @property
    def buffer(self) -> memoryview:
        return self._shm.buf

    def close(self):
        """Close access to the shared memory."""
        try:
            self._shm.close()
        except BufferError:
            # Zero-copy numpy views still reference the mmap.
            # Disarm __del__ so it won't re-raise the same error.
            self._shm._buf = None
            self._shm._mmap = None

    def unlink(self):
        """Request that the shared memory be destroyed."""
        self._shm.unlink()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        # Note: We do NOT unlink automatically on exit, as the other process might need it.
        # User must call unlink() explicitly or use a specific cleanup strategy.

    def put(
        self,
        obj: Any,
        check_integrity: bool = False,
        compress: bool = False,
        alignment: int = 64
    ) -> int:
        """
        Serialize an object directly into the shared memory.

        Supports NumPy arrays, sparse matrices, and dicts via the Rust fast-path.
        Falls back to Python serialization + copy if the Rust extension is unavailable.

        Returns
        -------
        int
            Number of bytes written.
        """
        if dump_to_buffer_rs is not None:
            try:
                return dump_to_buffer_rs(
                    obj,
                    self._shm.buf,
                    check_integrity=check_integrity,
                    compress=compress,
                    alignment=alignment,
                )
            except (NotImplementedError, TypeError):
                pass

        # Fallback: serialize to bytes then copy into SHM
        packet = bytes(dumps(obj, check_integrity=check_integrity, compress=compress, alignment=alignment))
        buf = self._shm.buf
        if len(packet) > len(buf):
            raise MemoryError(f"Packet ({len(packet)} bytes) exceeds SHM size ({len(buf)} bytes)")
        buf[:len(packet)] = packet
        return len(packet)

    def get(self) -> Optional[Union[np.ndarray, dict]]:
        """
        Deserialize the object currently in shared memory.

        Returns a zero-copy view into the shared memory buffer.
        The view remains valid as long as the underlying SHM segment
        has not been unlinked.

        Returns
        -------
        Optional[Union[np.ndarray, dict]]
            The reconstructed object (zero-copy view).
        """
        if loads_rs is not None:
            result = loads_rs(self._shm.buf)
            if result is not None:
                return result

        # Fallback to Python deserialization
        return core_loads(self._shm.buf)

    @classmethod
    def create_from(
        cls,
        name: str,
        obj: Any,
        check_integrity: bool = False,
        compress: bool = False,
        alignment: int = 64
    ) -> Self:
        """
        Create a new SharedMemory segment sized to fit the object and write it.

        Supports NumPy arrays, sparse matrices (COO/CSR/CSC), and dicts of arrays.
        """
        estimated_size = cls._estimate_size(obj, check_integrity, alignment)

        shm = cls(name, create=True, size=estimated_size)
        try:
            shm.put(obj, check_integrity=check_integrity, compress=compress, alignment=alignment)
            return shm
        except Exception:
            shm.close()
            shm.unlink()
            raise

    @staticmethod
    def _estimate_size(obj: Any, check_integrity: bool, alignment: int) -> int:
        """Estimate SHM size needed for an object."""
        footer_size = 8 if check_integrity else 0
        safety = 256

        if isinstance(obj, np.ndarray):
            ndim = obj.ndim
            header_size = 8 + (ndim * 4) + (1 if alignment != 64 else 0)
            return header_size + (alignment - 1) + obj.nbytes + footer_size + safety

        if isinstance(obj, dict):
            total = 8  # bundle header
            for key, value in obj.items():
                key_bytes = key.encode('utf-8') if isinstance(key, str) else str(key).encode('utf-8')
                total += 4 + len(key_bytes) + 4
                total += TensoShm._estimate_size(value, check_integrity, alignment)
            return total + safety

        if hasattr(obj, 'format') and hasattr(obj, 'data'):
            fmt = getattr(obj, 'format', '')
            shape = obj.shape
            ndim = len(shape)
            total = 8 + (ndim * 4)  # main header
            comps = []
            if fmt == 'coo':
                comps = [obj.data, obj.row, obj.col]
            elif fmt in ('csr', 'csc'):
                comps = [obj.data, obj.indices, obj.indptr]
            for c in comps:
                c = np.asarray(c)
                h = 8 + (c.ndim * 4) + (1 if alignment != 64 else 0)
                total += 4 + h + (alignment - 1) + c.nbytes + footer_size
            return total + safety

        raise TypeError(f"Unsupported type for SHM auto-sizing: {type(obj)}")
