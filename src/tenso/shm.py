"""
Shared Memory Transport for Tenso.

This module provides high-performance Inter-Process Communication (IPC)
capabilities using POSIX Shared Memory. It allows zero-copy transfer of
tensors between local processes.
"""

from typing import Any, Optional, Self, Union
import numpy as np

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
    
    Example:
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
        self._shm.close()

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
        
        Returns
        -------
        int
            Number of bytes written.
        """
        if dump_to_buffer_rs is None:
            raise NotImplementedError("Tenso Rust extension is required for SHM operations")

        return dump_to_buffer_rs(
            obj, 
            self._shm.buf, 
            check_integrity=check_integrity, 
            compress=compress, 
            alignment=alignment
        )

    def get(self) -> Optional[Union[np.ndarray, dict]]:
        """
        Deserialize the object currently in shared memory.
        
        Returns
        -------
        Optional[Union[np.ndarray, dict]]
            The reconstructed object (zero-copy view if possible).
        """
        if loads_rs is None:
             raise NotImplementedError("Tenso Rust extension is required for SHM operations")
        
        # loads_rs takes a buffer protocol object
        return loads_rs(self._shm.buf)

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
        """
        if not isinstance(obj, np.ndarray):
            raise NotImplementedError("Only NumPy arrays supported for SHM auto-sizing currently.")

        # Calculate exact overhead:
        # Header: 8 bytes + (ndim * 4) for shape + optional alignment byte
        # Padding: at most (alignment - 1) bytes
        # Footer: 8 bytes if integrity check enabled
        ndim = obj.ndim
        header_size = 8 + (ndim * 4) + (1 if alignment != 64 else 0)
        max_padding = alignment - 1
        footer_size = 8 if check_integrity else 0

        # Total overhead with small safety margin (256 bytes for edge cases)
        overhead = header_size + max_padding + footer_size + 256
        estimated_size = obj.nbytes + overhead

        shm = cls(name, create=True, size=estimated_size)
        try:
            shm.put(obj, check_integrity=check_integrity, compress=compress, alignment=alignment)
            return shm
        except Exception:
            shm.close()
            shm.unlink()
            raise
