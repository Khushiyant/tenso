"""
TensoCache: In-process tensor cache backed by shared memory.

Provides an embeddable, tensor-aware cache with mutable entries, zero-copy
reads, LRU eviction, TTL, in-place updates, and metadata inspection without
deserialization.

Memory Layout (single SHM segment):
    ┌──────────────────────────────────────────┐
    │  POOL HEADER (4096 bytes)                │
    │  magic | version | pool_size | max_entries│
    │  active_entries | free_bytes | watermark  │
    │  lru_head | lru_tail | free_list_head    │
    │  lock_word | generation | hits | misses   │
    ├──────────────────────────────────────────┤
    │  ENTRY INDEX TABLE (max_entries × 256 b) │
    ├──────────────────────────────────────────┤
    │  DATA REGION (rest of pool)              │
    │  64-byte aligned Tenso packets + free    │
    └──────────────────────────────────────────┘
"""

import struct
import sys
import threading
import time
from contextlib import contextmanager
from multiprocessing import shared_memory
from typing import Optional

import numpy as np

from .core import dumps, loads
from .utils import get_packet_info

try:
    from .tenso_rs import dump_to_buffer_rs, loads_rs
    _HAS_RUST = True
except ImportError:
    dump_to_buffer_rs = None
    loads_rs = None
    _HAS_RUST = False

try:
    from .quantize import QuantizedTensor
    _HAS_QUANTIZE = True
except ImportError:
    _HAS_QUANTIZE = False

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False

try:
    import jax
    import jax.numpy as jnp
    _HAS_JAX = True
except ImportError:
    jax = None
    jnp = None
    _HAS_JAX = False

try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    cp = None
    _HAS_CUPY = False

# -- Constants --

_CACHE_MAGIC = b"TCSH"
_CACHE_VERSION = 2
_POOL_HEADER_SIZE = 4096
_ENTRY_SIZE = 256
_DATA_ALIGNMENT = 64
_MAX_KEY_LEN = 128
_SENTINEL = 0xFFFFFFFF  # null pointer for LRU linked list

# Pool header field offsets (all little-endian)
_H_MAGIC = 0          # 4 bytes
_H_VERSION = 4        # 4 bytes
_H_POOL_SIZE = 8      # 8 bytes
_H_MAX_ENTRIES = 16   # 4 bytes
_H_ACTIVE = 20        # 4 bytes
_H_FREE_BYTES = 24    # 8 bytes
_H_WATERMARK = 32     # 8 bytes  (bump allocator offset into data region)
_H_LRU_HEAD = 40      # 4 bytes
_H_LRU_TAIL = 44      # 4 bytes
_H_FREE_LIST = 48     # 8 bytes  (offset of first free chunk in data region, 0=none)
_H_LOCK = 56          # 4 bytes (0=unlocked, 1=locked)
_H_LOCK_TIME = 60     # 8 bytes (monotonic timestamp of acquisition)
_H_GENERATION = 68    # 8 bytes
_H_HITS = 76          # 8 bytes
_H_MISSES = 84        # 8 bytes

# Entry slot field offsets (within each 256-byte slot)
_E_STATUS = 0         # 1 byte  (0=free, 1=active)
_E_KEY_LEN = 1        # 1 byte
_E_KEY = 2            # 128 bytes (padded)
_E_DATA_OFF = 130     # 8 bytes  (offset from start of SHM)
_E_ALLOC_SIZE = 138   # 8 bytes  (allocated size including alignment padding)
_E_PACKET_SIZE = 146  # 8 bytes  (actual tenso packet size)
_E_DTYPE = 154        # 1 byte
_E_NDIM = 155         # 1 byte
_E_SHAPE = 156        # 32*4 = 128 bytes (up to 32 dims)
_E_ACCESS_TIME = 284  # 8 bytes (double, seconds since epoch, actually fits: 156+128=284 but we have 256. Let me recalculate)
# Recalculate: 0+1+1+128+8+8+8+1+1+128 = 284 > 256. Need to compress.
# Let's use a more compact layout:
# status(1) + key_len(1) + key(128) + data_off(8) + alloc_size(4) + packet_size(4)
# + dtype(1) + ndim(1) + shape(8*4=32) + access_time(8) + create_time(8)
# + ttl(8) + lru_prev(4) + lru_next(4) = 1+1+128+8+4+4+1+1+32+8+8+8+4+4 = 212
# That fits in 256 with room to spare.

# Revised compact entry layout:
_E_STATUS = 0         # 1 byte  (0=free, 1=active)
_E_KEY_LEN = 1        # 1 byte
_E_KEY = 2            # 128 bytes
_E_DATA_OFF = 130     # 8 bytes  (absolute offset from SHM start)
_E_ALLOC_SIZE = 138   # 4 bytes  (allocated chunk size)
_E_PACKET_SIZE = 142  # 4 bytes  (actual tenso packet size)
_E_DTYPE = 146        # 1 byte
_E_NDIM = 147         # 1 byte
_E_SHAPE = 148        # 32 bytes (up to 8 dims as uint32, sufficient for most cases)
_E_ACCESS_TIME = 180  # 8 bytes (double)
_E_CREATE_TIME = 188  # 8 bytes (double)
_E_TTL = 196          # 8 bytes (double, 0 = no expiry)
_E_LRU_PREV = 204     # 4 bytes (slot index, _SENTINEL = null)
_E_LRU_NEXT = 208     # 4 bytes (slot index, _SENTINEL = null)
# Total: 212 bytes, 44 bytes spare

_STATUS_FREE = 0
_STATUS_ACTIVE = 1

# Free chunk header in data region: size(8 bytes) + next_offset(8 bytes)
_FREE_CHUNK_HEADER = 16


def _to_numpy(tensor) -> np.ndarray:
    """Convert a framework tensor (PyTorch, JAX, CuPy) to numpy.

    Falls back to np.asarray for unknown types.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    if _HAS_TORCH and isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    if _HAS_JAX and isinstance(tensor, jax.Array):
        return np.asarray(tensor)
    if _HAS_CUPY and isinstance(tensor, cp.ndarray):
        return tensor.get()
    return np.asarray(tensor)


def _numpy_to_device(arr: np.ndarray, device: str):
    """Convert a numpy array to a framework tensor on the specified device.

    Device spec format: "framework" or "framework:device_spec".
    Examples: "torch", "torch:cuda:0", "jax", "cupy", "cupy:0".
    """
    parts = device.split(":", 1)
    framework = parts[0].lower()

    if framework == "torch":
        if not _HAS_TORCH:
            raise ValueError("PyTorch is not installed")
        t = torch.from_numpy(arr)
        if len(parts) > 1:
            t = t.to(parts[1])
        return t

    if framework == "jax":
        if not _HAS_JAX:
            raise ValueError("JAX is not installed")
        result = jnp.array(arr)
        if len(parts) > 1:
            device_id = int(parts[1])
            devices = jax.devices()
            if device_id < len(devices):
                result = jax.device_put(result, devices[device_id])
        return result

    if framework == "cupy":
        if not _HAS_CUPY:
            raise ValueError("CuPy is not installed")
        if len(parts) > 1:
            with cp.cuda.Device(int(parts[1])):
                return cp.array(arr)
        return cp.array(arr)

    raise ValueError(
        f"Unsupported device framework: {framework!r}. "
        f"Supported: 'torch', 'jax', 'cupy'."
    )


def _parse_size(size) -> int:
    """Parse a human-readable size string (e.g. '256MB', '4GB') to bytes."""
    if isinstance(size, int):
        return size
    s = str(size).strip().upper()
    multipliers = {
        'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3, 'TB': 1024**4,
        'K': 1024, 'M': 1024**2, 'G': 1024**3, 'T': 1024**4,
    }
    for suffix, mult in sorted(multipliers.items(), key=lambda x: -len(x[0])):
        if s.endswith(suffix):
            return int(float(s[:-len(suffix)]) * mult)
    return int(s)


def _align_up(value: int, alignment: int) -> int:
    """Round up value to the next multiple of alignment."""
    return (value + alignment - 1) & ~(alignment - 1)


class TensoCache:
    """
    In-process tensor cache backed by a single shared memory pool.

    Supports mutable entries, zero-copy reads, LRU eviction, TTL,
    in-place updates, and metadata inspection without deserialization.

    Example::

        import numpy as np
        from tenso import TensoCache

        with TensoCache("64MB") as cache:
            cache.put("weights", np.random.randn(1000, 1000).astype(np.float32))
            arr = cache.get("weights")        # zero-copy view into SHM
            print(cache.info("weights"))       # metadata without deserialization
            print(cache.stats)                 # hit/miss counts, memory usage
    """

    def __init__(self, max_memory="256MB", name=None, create=True):
        """
        Create or attach to a TensoCache pool.

        Parameters
        ----------
        max_memory : str or int
            Pool size as human-readable string ('256MB', '1GB') or bytes.
            Ignored when ``create=False``.
        name : str, optional
            Shared memory segment name. Auto-generated if None.
        create : bool
            If True, create a new pool. If False, attach to existing.
        """
        self._lock = threading.RLock()
        self._closed = False
        self._owns = create
        self._key_to_slot: dict[str, int] = {}
        self._local_generation = -1

        if create:
            pool_size = _parse_size(max_memory)
            if pool_size < _POOL_HEADER_SIZE + _ENTRY_SIZE * 4:
                raise ValueError("Pool size too small (minimum ~5KB)")

            max_entries = min(
                (pool_size - _POOL_HEADER_SIZE) // (_ENTRY_SIZE + _DATA_ALIGNMENT * 16),
                65536
            )
            max_entries = max(max_entries, 1)

            entry_table_size = max_entries * _ENTRY_SIZE
            data_region_offset = _align_up(
                _POOL_HEADER_SIZE + entry_table_size, _DATA_ALIGNMENT
            )

            if data_region_offset >= pool_size:
                raise ValueError("Pool too small for any data after header + entry table")

            data_region_size = pool_size - data_region_offset

            self._shm = shared_memory.SharedMemory(
                name=name, create=True, size=pool_size
            )
            self._name = self._shm.name
            buf = self._shm.buf

            # Zero the entire pool
            buf[:pool_size] = b'\x00' * pool_size

            # Write pool header
            struct.pack_into('<4s', buf, _H_MAGIC, _CACHE_MAGIC)
            struct.pack_into('<I', buf, _H_VERSION, _CACHE_VERSION)
            struct.pack_into('<Q', buf, _H_POOL_SIZE, pool_size)
            struct.pack_into('<I', buf, _H_MAX_ENTRIES, max_entries)
            struct.pack_into('<I', buf, _H_ACTIVE, 0)
            struct.pack_into('<Q', buf, _H_FREE_BYTES, data_region_size)
            struct.pack_into('<Q', buf, _H_WATERMARK, data_region_offset)
            struct.pack_into('<I', buf, _H_LRU_HEAD, _SENTINEL)
            struct.pack_into('<I', buf, _H_LRU_TAIL, _SENTINEL)
            struct.pack_into('<Q', buf, _H_FREE_LIST, 0)
            struct.pack_into('<I', buf, _H_LOCK, 0)
            struct.pack_into('<d', buf, _H_LOCK_TIME, 0.0)
            struct.pack_into('<Q', buf, _H_GENERATION, 0)
            struct.pack_into('<Q', buf, _H_HITS, 0)
            struct.pack_into('<Q', buf, _H_MISSES, 0)

            # Initialize all entry slots as free with sentinel LRU pointers
            for i in range(max_entries):
                slot_off = _POOL_HEADER_SIZE + i * _ENTRY_SIZE
                buf[slot_off + _E_STATUS] = _STATUS_FREE
                struct.pack_into('<I', buf, slot_off + _E_LRU_PREV, _SENTINEL)
                struct.pack_into('<I', buf, slot_off + _E_LRU_NEXT, _SENTINEL)

            self._pool_size = pool_size
            self._max_entries = max_entries
            self._data_region_offset = data_region_offset
            self._data_region_end = pool_size

        else:
            if name is None:
                raise ValueError("name is required when create=False")
            self._shm = shared_memory.SharedMemory(name=name, create=False)
            self._name = self._shm.name
            buf = self._shm.buf

            magic = bytes(buf[_H_MAGIC:_H_MAGIC + 4])
            if magic != _CACHE_MAGIC:
                self._shm.close()
                raise ValueError(f"Invalid TensoCache magic: {magic!r}")
            version = struct.unpack_from('<I', buf, _H_VERSION)[0]
            if version != _CACHE_VERSION:
                self._shm.close()
                raise ValueError(f"Unsupported TensoCache version: {version}")

            self._pool_size = struct.unpack_from('<Q', buf, _H_POOL_SIZE)[0]
            self._max_entries = struct.unpack_from('<I', buf, _H_MAX_ENTRIES)[0]

            entry_table_size = self._max_entries * _ENTRY_SIZE
            self._data_region_offset = _align_up(
                _POOL_HEADER_SIZE + entry_table_size, _DATA_ALIGNMENT
            )
            self._data_region_end = self._pool_size

    @property
    def name(self) -> str:
        return self._name

    # -- Header accessors --

    def _read_header_u32(self, offset: int) -> int:
        return struct.unpack_from('<I', self._shm.buf, offset)[0]

    def _write_header_u32(self, offset: int, value: int):
        struct.pack_into('<I', self._shm.buf, offset, value)

    def _read_header_u64(self, offset: int) -> int:
        return struct.unpack_from('<Q', self._shm.buf, offset)[0]

    def _write_header_u64(self, offset: int, value: int):
        struct.pack_into('<Q', self._shm.buf, offset, value)

    # -- SHM spinlock (cross-process safety) --

    _STALE_LOCK_THRESHOLD = 30.0  # seconds before force-acquiring a stale lock

    def _shm_lock_acquire(self, timeout: float = 5.0):
        """
        Acquire the SHM spinlock for cross-process mutual exclusion.

        Uses a 4-byte lock word at _H_LOCK. Spins with 100us sleep.
        Stale locks (held > 30s) are force-acquired with a stderr warning
        to recover from crashed processes.

        Note: struct.pack_into on a 4-byte-aligned field is atomic at the
        hardware level on x86/ARM. For extreme multi-writer contention,
        an external multiprocessing.Lock is recommended.
        """
        buf = self._shm.buf
        deadline = time.monotonic() + timeout

        while True:
            lock_val = struct.unpack_from('<I', buf, _H_LOCK)[0]
            if lock_val == 0:
                # Unlocked — acquire
                struct.pack_into('<I', buf, _H_LOCK, 1)
                struct.pack_into('<d', buf, _H_LOCK_TIME, time.monotonic())
                return

            # Lock is held — check for staleness
            lock_time = struct.unpack_from('<d', buf, _H_LOCK_TIME)[0]
            if lock_time > 0 and (time.monotonic() - lock_time) > self._STALE_LOCK_THRESHOLD:
                print(
                    f"TensoCache: force-acquiring stale SHM lock "
                    f"(held for {time.monotonic() - lock_time:.1f}s, "
                    f"likely crashed process)",
                    file=sys.stderr,
                )
                struct.pack_into('<I', buf, _H_LOCK, 1)
                struct.pack_into('<d', buf, _H_LOCK_TIME, time.monotonic())
                return

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"TensoCache: SHM lock acquisition timed out after {timeout}s"
                )

            time.sleep(0.0001)  # 100μs spin

    def _shm_lock_release(self):
        """Release the SHM spinlock."""
        buf = self._shm.buf
        struct.pack_into('<d', buf, _H_LOCK_TIME, 0.0)
        struct.pack_into('<I', buf, _H_LOCK, 0)

    @contextmanager
    def _shm_locked(self):
        """Context manager wrapping SHM lock acquire/release."""
        self._shm_lock_acquire()
        try:
            yield
        finally:
            self._shm_lock_release()

    # -- Generation-based local index invalidation --

    def _sync_index(self):
        """Rebuild local key->slot dict if SHM generation has advanced."""
        gen = self._read_header_u64(_H_GENERATION)
        if gen != self._local_generation:
            self._rebuild_index()
            self._local_generation = gen

    def _bump_generation(self):
        gen = self._read_header_u64(_H_GENERATION) + 1
        self._write_header_u64(_H_GENERATION, gen)
        self._local_generation = gen

    def _rebuild_index(self):
        self._key_to_slot.clear()
        buf = self._shm.buf
        for i in range(self._max_entries):
            slot_off = _POOL_HEADER_SIZE + i * _ENTRY_SIZE
            if buf[slot_off + _E_STATUS] == _STATUS_ACTIVE:
                key_len = buf[slot_off + _E_KEY_LEN]
                key = bytes(buf[slot_off + _E_KEY: slot_off + _E_KEY + key_len]).decode('utf-8')
                self._key_to_slot[key] = i

    # -- Entry table operations --

    def _slot_offset(self, slot: int) -> int:
        return _POOL_HEADER_SIZE + slot * _ENTRY_SIZE

    def _read_entry_key(self, slot: int) -> str:
        buf = self._shm.buf
        off = self._slot_offset(slot)
        key_len = buf[off + _E_KEY_LEN]
        return bytes(buf[off + _E_KEY: off + _E_KEY + key_len]).decode('utf-8')

    def _read_entry(self, slot: int) -> dict:
        buf = self._shm.buf
        off = self._slot_offset(slot)
        key_len = buf[off + _E_KEY_LEN]
        key = bytes(buf[off + _E_KEY: off + _E_KEY + key_len]).decode('utf-8')
        data_off = struct.unpack_from('<Q', buf, off + _E_DATA_OFF)[0]
        alloc_size = struct.unpack_from('<I', buf, off + _E_ALLOC_SIZE)[0]
        packet_size = struct.unpack_from('<I', buf, off + _E_PACKET_SIZE)[0]
        dtype_code = buf[off + _E_DTYPE]
        ndim = buf[off + _E_NDIM]
        shape = struct.unpack_from(f'<{ndim}I', buf, off + _E_SHAPE) if ndim > 0 else ()
        access_time = struct.unpack_from('<d', buf, off + _E_ACCESS_TIME)[0]
        create_time = struct.unpack_from('<d', buf, off + _E_CREATE_TIME)[0]
        ttl = struct.unpack_from('<d', buf, off + _E_TTL)[0]
        lru_prev = struct.unpack_from('<I', buf, off + _E_LRU_PREV)[0]
        lru_next = struct.unpack_from('<I', buf, off + _E_LRU_NEXT)[0]
        return {
            'key': key, 'data_offset': data_off, 'alloc_size': alloc_size,
            'packet_size': packet_size, 'dtype_code': dtype_code, 'ndim': ndim,
            'shape': shape, 'access_time': access_time, 'create_time': create_time,
            'ttl': ttl, 'lru_prev': lru_prev, 'lru_next': lru_next,
        }

    def _write_entry(self, slot: int, key: str, data_off: int, alloc_size: int,
                     packet_size: int, dtype_code: int, ndim: int, shape: tuple,
                     ttl: float, now: float):
        buf = self._shm.buf
        off = self._slot_offset(slot)
        key_bytes = key.encode('utf-8')

        # Write barrier: mark FREE so concurrent readers skip this entry
        buf[off + _E_STATUS] = _STATUS_FREE

        # Write all metadata fields
        buf[off + _E_KEY_LEN] = len(key_bytes)
        buf[off + _E_KEY: off + _E_KEY + len(key_bytes)] = key_bytes
        # Zero remaining key space
        if len(key_bytes) < _MAX_KEY_LEN:
            buf[off + _E_KEY + len(key_bytes): off + _E_KEY + _MAX_KEY_LEN] = (
                b'\x00' * (_MAX_KEY_LEN - len(key_bytes))
            )
        struct.pack_into('<Q', buf, off + _E_DATA_OFF, data_off)
        struct.pack_into('<I', buf, off + _E_ALLOC_SIZE, alloc_size)
        struct.pack_into('<I', buf, off + _E_PACKET_SIZE, packet_size)
        buf[off + _E_DTYPE] = dtype_code
        buf[off + _E_NDIM] = ndim
        # Write shape (up to 8 dims)
        dims_to_write = min(ndim, 8)
        if dims_to_write > 0:
            struct.pack_into(f'<{dims_to_write}I', buf, off + _E_SHAPE, *shape[:dims_to_write])
        struct.pack_into('<d', buf, off + _E_ACCESS_TIME, now)
        struct.pack_into('<d', buf, off + _E_CREATE_TIME, now)
        struct.pack_into('<d', buf, off + _E_TTL, ttl or 0.0)
        struct.pack_into('<I', buf, off + _E_LRU_PREV, _SENTINEL)
        struct.pack_into('<I', buf, off + _E_LRU_NEXT, _SENTINEL)

        # Release barrier: metadata fully written, now visible to readers
        buf[off + _E_STATUS] = _STATUS_ACTIVE

    def _clear_entry(self, slot: int):
        buf = self._shm.buf
        off = self._slot_offset(slot)
        buf[off + _E_STATUS] = _STATUS_FREE
        struct.pack_into('<I', buf, off + _E_LRU_PREV, _SENTINEL)
        struct.pack_into('<I', buf, off + _E_LRU_NEXT, _SENTINEL)

    def _find_free_slot(self) -> int:
        """Find first free slot in entry table. Returns -1 if none."""
        buf = self._shm.buf
        for i in range(self._max_entries):
            off = _POOL_HEADER_SIZE + i * _ENTRY_SIZE
            if buf[off + _E_STATUS] == _STATUS_FREE:
                return i
        return -1

    # -- Data region allocator --

    def _allocate(self, size: int) -> int:
        """
        Allocate ``size`` bytes in the data region, 64-byte aligned.

        Returns the absolute offset within SHM. Raises MemoryError if
        allocation fails even after eviction.
        """
        alloc_size = _align_up(size, _DATA_ALIGNMENT)
        # Minimum allocation to hold a free chunk header when freed
        alloc_size = max(alloc_size, _FREE_CHUNK_HEADER)

        buf = self._shm.buf

        # 1. Try free list (first-fit)
        offset = self._alloc_from_free_list(alloc_size)
        if offset != 0:
            return offset

        # 2. Bump allocator
        watermark = self._read_header_u64(_H_WATERMARK)
        if watermark + alloc_size <= self._data_region_end:
            self._write_header_u64(_H_WATERMARK, watermark + alloc_size)
            free_bytes = self._read_header_u64(_H_FREE_BYTES)
            self._write_header_u64(_H_FREE_BYTES, free_bytes - alloc_size)
            return watermark

        # 3. Not enough space
        return 0

    def _alloc_from_free_list(self, alloc_size: int) -> int:
        """First-fit allocation from the free list. Returns 0 if nothing fits."""
        buf = self._shm.buf
        prev_ptr_loc = _H_FREE_LIST  # location storing pointer to current chunk
        current_off = self._read_header_u64(_H_FREE_LIST)

        while current_off != 0:
            chunk_size = struct.unpack_from('<Q', buf, current_off)[0]
            next_off = struct.unpack_from('<Q', buf, current_off + 8)[0]

            if chunk_size >= alloc_size:
                remainder = chunk_size - alloc_size
                if remainder >= _FREE_CHUNK_HEADER + _DATA_ALIGNMENT:
                    # Split: keep remainder as a smaller free chunk
                    new_free_off = current_off + alloc_size
                    struct.pack_into('<Q', buf, new_free_off, remainder)
                    struct.pack_into('<Q', buf, new_free_off + 8, next_off)
                    # Update previous pointer to new free chunk
                    struct.pack_into('<Q', buf, prev_ptr_loc, new_free_off)
                else:
                    # Use entire chunk (avoid tiny fragments)
                    alloc_size = chunk_size  # absorb remainder
                    struct.pack_into('<Q', buf, prev_ptr_loc, next_off)

                free_bytes = self._read_header_u64(_H_FREE_BYTES)
                self._write_header_u64(_H_FREE_BYTES, free_bytes - alloc_size)
                return current_off

            prev_ptr_loc = current_off + 8
            current_off = next_off

        return 0

    def _free_data(self, offset: int, alloc_size: int):
        """Return a data region chunk to the free list with coalescing."""
        buf = self._shm.buf
        original_size = alloc_size  # only these bytes are newly freed

        # Write free chunk header
        struct.pack_into('<Q', buf, offset, alloc_size)
        struct.pack_into('<Q', buf, offset + 8, 0)

        # Insert into free list sorted by offset
        prev_ptr_loc = _H_FREE_LIST
        current_off = self._read_header_u64(_H_FREE_LIST)

        while current_off != 0 and current_off < offset:
            prev_ptr_loc = current_off + 8
            current_off = struct.unpack_from('<Q', buf, current_off + 8)[0]

        # Link: prev -> new -> current
        struct.pack_into('<Q', buf, offset + 8, current_off)
        struct.pack_into('<Q', buf, prev_ptr_loc, offset)

        # Coalesce with next chunk
        if current_off != 0 and offset + alloc_size == current_off:
            next_size = struct.unpack_from('<Q', buf, current_off)[0]
            next_next = struct.unpack_from('<Q', buf, current_off + 8)[0]
            new_size = alloc_size + next_size
            struct.pack_into('<Q', buf, offset, new_size)
            struct.pack_into('<Q', buf, offset + 8, next_next)
            alloc_size = new_size

        # Coalesce with previous chunk
        if prev_ptr_loc != _H_FREE_LIST:
            prev_off = prev_ptr_loc - 8  # back up to start of prev chunk
            prev_size = struct.unpack_from('<Q', buf, prev_off)[0]
            if prev_off + prev_size == offset:
                new_size = prev_size + alloc_size
                struct.pack_into('<Q', buf, prev_off, new_size)
                next_off = struct.unpack_from('<Q', buf, offset + 8)[0]
                struct.pack_into('<Q', buf, prev_off + 8, next_off)

        # Only add the originally-freed bytes (coalesced chunks were already free)
        free_bytes = self._read_header_u64(_H_FREE_BYTES)
        self._write_header_u64(_H_FREE_BYTES, free_bytes + original_size)

    # -- LRU linked list --

    def _lru_insert_head(self, slot: int):
        """Insert slot at the head (MRU) of the LRU list."""
        buf = self._shm.buf
        off = self._slot_offset(slot)
        head = self._read_header_u32(_H_LRU_HEAD)

        struct.pack_into('<I', buf, off + _E_LRU_PREV, _SENTINEL)
        struct.pack_into('<I', buf, off + _E_LRU_NEXT, head)

        if head != _SENTINEL:
            head_off = self._slot_offset(head)
            struct.pack_into('<I', buf, head_off + _E_LRU_PREV, slot)

        self._write_header_u32(_H_LRU_HEAD, slot)

        if self._read_header_u32(_H_LRU_TAIL) == _SENTINEL:
            self._write_header_u32(_H_LRU_TAIL, slot)

    def _lru_remove(self, slot: int):
        """Remove slot from the LRU list."""
        buf = self._shm.buf
        off = self._slot_offset(slot)
        prev_slot = struct.unpack_from('<I', buf, off + _E_LRU_PREV)[0]
        next_slot = struct.unpack_from('<I', buf, off + _E_LRU_NEXT)[0]

        if prev_slot != _SENTINEL:
            prev_off = self._slot_offset(prev_slot)
            struct.pack_into('<I', buf, prev_off + _E_LRU_NEXT, next_slot)
        else:
            self._write_header_u32(_H_LRU_HEAD, next_slot)

        if next_slot != _SENTINEL:
            next_off = self._slot_offset(next_slot)
            struct.pack_into('<I', buf, next_off + _E_LRU_PREV, prev_slot)
        else:
            self._write_header_u32(_H_LRU_TAIL, prev_slot)

        struct.pack_into('<I', buf, off + _E_LRU_PREV, _SENTINEL)
        struct.pack_into('<I', buf, off + _E_LRU_NEXT, _SENTINEL)

    def _lru_move_to_head(self, slot: int):
        """Move slot to the head of the LRU list."""
        if self._read_header_u32(_H_LRU_HEAD) == slot:
            return  # already at head
        self._lru_remove(slot)
        self._lru_insert_head(slot)

    # -- Eviction --

    def _evict_expired(self) -> int:
        """Evict all TTL-expired entries. Returns bytes freed."""
        freed = 0
        now = time.monotonic()
        buf = self._shm.buf

        for i in range(self._max_entries):
            off = _POOL_HEADER_SIZE + i * _ENTRY_SIZE
            if buf[off + _E_STATUS] != _STATUS_ACTIVE:
                continue
            ttl = struct.unpack_from('<d', buf, off + _E_TTL)[0]
            if ttl <= 0:
                continue
            create_time = struct.unpack_from('<d', buf, off + _E_CREATE_TIME)[0]
            if now - create_time >= ttl:
                freed += self._delete_slot(i)
        return freed

    def _evict_for_space(self, needed: int) -> bool:
        """Evict entries until ``needed`` bytes are available. Returns True on success."""
        # First try evicting expired entries
        self._evict_expired()
        if self._can_allocate(needed):
            return True

        # Evict from LRU tail
        while not self._can_allocate(needed):
            tail = self._read_header_u32(_H_LRU_TAIL)
            if tail == _SENTINEL:
                return False  # nothing left to evict
            self._delete_slot(tail)
        return True

    def _can_allocate(self, size: int) -> bool:
        """Check if we can allocate ``size`` bytes (free list or bump)."""
        alloc_size = _align_up(size, _DATA_ALIGNMENT)
        alloc_size = max(alloc_size, _FREE_CHUNK_HEADER)

        # Check free list
        buf = self._shm.buf
        current_off = self._read_header_u64(_H_FREE_LIST)
        while current_off != 0:
            chunk_size = struct.unpack_from('<Q', buf, current_off)[0]
            if chunk_size >= alloc_size:
                return True
            current_off = struct.unpack_from('<Q', buf, current_off + 8)[0]

        # Check bump allocator
        watermark = self._read_header_u64(_H_WATERMARK)
        return watermark + alloc_size <= self._data_region_end

    def _delete_slot(self, slot: int) -> int:
        """Delete entry at slot, free its data, return bytes freed."""
        buf = self._shm.buf
        off = self._slot_offset(slot)

        data_off = struct.unpack_from('<Q', buf, off + _E_DATA_OFF)[0]
        alloc_size = struct.unpack_from('<I', buf, off + _E_ALLOC_SIZE)[0]
        key_len = buf[off + _E_KEY_LEN]
        key = bytes(buf[off + _E_KEY: off + _E_KEY + key_len]).decode('utf-8')

        self._lru_remove(slot)
        self._free_data(data_off, alloc_size)
        self._clear_entry(slot)

        # Update active count
        active = self._read_header_u32(_H_ACTIVE)
        self._write_header_u32(_H_ACTIVE, active - 1)

        # Update local index
        self._key_to_slot.pop(key, None)

        return alloc_size

    # -- Serialization helpers --

    def _serialize_tensor(self, tensor, buf_slice, quantize_dtype=None):
        """
        Serialize tensor into a buffer slice. Returns packet size.

        Tries Rust fast path first, falls back to Python dumps().
        """
        if not isinstance(tensor, np.ndarray) and not (_HAS_QUANTIZE and isinstance(tensor, QuantizedTensor)):
            tensor = _to_numpy(tensor)
        if quantize_dtype is not None and _HAS_QUANTIZE:
            tensor = QuantizedTensor.quantize(tensor, quantize_dtype)

        if _HAS_RUST and dump_to_buffer_rs is not None and isinstance(tensor, np.ndarray):
            try:
                return dump_to_buffer_rs(tensor, buf_slice)
            except (TypeError, NotImplementedError):
                pass

        # Fallback: serialize to bytes, then copy into buffer
        packet = bytes(dumps(tensor))
        if len(packet) > len(buf_slice):
            raise MemoryError("Serialized packet exceeds allocated space")
        buf_slice[:len(packet)] = packet
        return len(packet)

    def _estimate_packet_size(self, tensor, quantize_dtype=None) -> int:
        """Estimate the serialized packet size for allocation."""
        if not isinstance(tensor, np.ndarray) and not (_HAS_QUANTIZE and isinstance(tensor, QuantizedTensor)):
            tensor = _to_numpy(tensor)
        if quantize_dtype is not None and _HAS_QUANTIZE:
            tensor = QuantizedTensor.quantize(tensor, quantize_dtype)

        if isinstance(tensor, np.ndarray):
            ndim = tensor.ndim
            # Header(8) + shape(ndim*4) + alignment padding(up to 63) + data + safety margin
            header_size = 8 + ndim * 4
            padding = (_DATA_ALIGNMENT - (header_size % _DATA_ALIGNMENT)) % _DATA_ALIGNMENT
            return header_size + padding + tensor.nbytes + 256
        elif _HAS_QUANTIZE and isinstance(tensor, QuantizedTensor):
            packet = bytes(dumps(tensor))
            return len(packet) + 64
        else:
            packet = bytes(dumps(tensor))
            return len(packet) + 64

    # -- Public API --

    def put(self, key: str, tensor, ttl: float = None, quantize: str = None) -> int:
        """
        Store a tensor in the cache.

        Parameters
        ----------
        key : str
            Cache key (max 128 bytes UTF-8).
        tensor : np.ndarray, QuantizedTensor, or framework tensor
            Tensor to store. PyTorch, JAX, and CuPy tensors are
            automatically converted to numpy before caching.
        ttl : float, optional
            Time-to-live in seconds. None means no expiry.
        quantize : str, optional
            Quantization dtype ('qint8', 'quint8', 'qint4', 'quint4').

        Returns
        -------
        int
            Number of bytes written.

        Raises
        ------
        ValueError
            If key exceeds 128 bytes.
        MemoryError
            If pool is exhausted after eviction attempts.
        """
        key_bytes = key.encode('utf-8')
        if len(key_bytes) > _MAX_KEY_LEN:
            raise ValueError(f"Key too long: {len(key_bytes)} bytes (max {_MAX_KEY_LEN})")

        # Convert framework tensors outside the lock (GPU→CPU can be expensive)
        if not isinstance(tensor, np.ndarray) and not (_HAS_QUANTIZE and isinstance(tensor, QuantizedTensor)):
            tensor = _to_numpy(tensor)

        with self._lock:
            self._check_closed()
            with self._shm_locked():
                self._sync_index()

                # Pre-quantize if requested
                obj_to_store = tensor
                if quantize is not None and _HAS_QUANTIZE and isinstance(tensor, np.ndarray):
                    obj_to_store = QuantizedTensor.quantize(tensor, quantize)

                # Estimate size needed
                estimated = self._estimate_packet_size(obj_to_store)
                alloc_size = _align_up(estimated, _DATA_ALIGNMENT)
                alloc_size = max(alloc_size, _FREE_CHUNK_HEADER)

                # Check for in-place update
                existing_slot = self._key_to_slot.get(key)
                if existing_slot is not None:
                    entry = self._read_entry(existing_slot)
                    if entry['alloc_size'] >= alloc_size:
                        # Reuse existing allocation
                        data_off = entry['data_offset']
                        actual_alloc = entry['alloc_size']
                        buf_slice = self._shm.buf[data_off:data_off + actual_alloc]
                        packet_size = self._serialize_tensor(obj_to_store, buf_slice)

                        # Extract metadata from packet
                        dtype_code, ndim, shape = self._extract_metadata(
                            self._shm.buf[data_off:data_off + packet_size]
                        )

                        now = time.monotonic()
                        self._write_entry(
                            existing_slot, key, data_off, actual_alloc, packet_size,
                            dtype_code, ndim, shape, ttl or 0.0, now
                        )
                        self._lru_move_to_head(existing_slot)
                        self._bump_generation()
                        return packet_size
                    else:
                        # Free old allocation and re-allocate
                        self._delete_slot(existing_slot)

                # Allocate space
                data_off = self._allocate(alloc_size)
                if data_off == 0:
                    # Try eviction
                    if not self._evict_for_space(alloc_size):
                        raise MemoryError(
                            f"Cannot allocate {alloc_size} bytes in TensoCache pool "
                            f"(free: {self._read_header_u64(_H_FREE_BYTES)} bytes)"
                        )
                    data_off = self._allocate(alloc_size)
                    if data_off == 0:
                        raise MemoryError("Allocation failed after eviction")

                actual_alloc = _align_up(estimated, _DATA_ALIGNMENT)
                actual_alloc = max(actual_alloc, _FREE_CHUNK_HEADER)

                # Find a free entry slot
                slot = self._find_free_slot()
                if slot < 0:
                    # Evict LRU tail to free a slot
                    tail = self._read_header_u32(_H_LRU_TAIL)
                    if tail == _SENTINEL:
                        self._free_data(data_off, alloc_size)
                        raise MemoryError("No free entry slots in TensoCache")
                    self._delete_slot(tail)
                    slot = self._find_free_slot()
                    if slot < 0:
                        self._free_data(data_off, alloc_size)
                        raise MemoryError("No free entry slots after eviction")

                # Serialize into allocated region
                buf_slice = self._shm.buf[data_off:data_off + alloc_size]
                packet_size = self._serialize_tensor(obj_to_store, buf_slice)

                # Extract metadata from the serialized packet
                dtype_code, ndim, shape = self._extract_metadata(
                    self._shm.buf[data_off:data_off + packet_size]
                )

                now = time.monotonic()
                self._write_entry(
                    slot, key, data_off, alloc_size, packet_size,
                    dtype_code, ndim, shape, ttl or 0.0, now
                )
                self._lru_insert_head(slot)

                # Update active count and local index
                active = self._read_header_u32(_H_ACTIVE)
                self._write_header_u32(_H_ACTIVE, active + 1)
                self._key_to_slot[key] = slot
                self._bump_generation()

                return packet_size

    def _extract_metadata(self, packet_bytes) -> tuple:
        """Extract dtype_code, ndim, shape from a tenso packet header."""
        try:
            info = get_packet_info(bytes(packet_bytes))
            ndim = info.get('ndim', 0)
            shape = info.get('shape', ())
            dtype = info.get('dtype')
            # Map dtype back to code
            from .config import _DTYPE_MAP, _QDTYPE_FROM_NAME
            dtype_code = 0
            if isinstance(dtype, np.dtype):
                dtype_code = _DTYPE_MAP.get(dtype, 0)
            elif isinstance(dtype, str):
                dtype_code = _QDTYPE_FROM_NAME.get(dtype, 0)
            elif isinstance(dtype, int):
                dtype_code = dtype
            return dtype_code, ndim, shape
        except Exception:
            return 0, 0, ()

    def get(self, key: str, copy: bool = False, device: str = None):
        """
        Retrieve a tensor from the cache.

        Parameters
        ----------
        key : str
            Cache key.
        copy : bool
            If True, return a writeable copy. Default returns zero-copy view.
        device : str, optional
            Target device for the result. Format: "framework" or
            "framework:device_spec" (e.g. "torch", "torch:cuda:0", "jax",
            "cupy:0"). When set, the result is converted from numpy.
            Implies copy (SHM buffer cannot be shared with frameworks).

        Returns
        -------
        Optional[np.ndarray or framework tensor]
            The tensor, or None if not found or expired.
        """
        with self._lock:
            self._check_closed()
            self._sync_index()

            slot = self._key_to_slot.get(key)
            if slot is None:
                self._inc_misses()
                return None

            buf = self._shm.buf
            off = self._slot_offset(slot)

            if buf[off + _E_STATUS] != _STATUS_ACTIVE:
                self._key_to_slot.pop(key, None)
                self._inc_misses()
                return None

            # Check TTL
            ttl = struct.unpack_from('<d', buf, off + _E_TTL)[0]
            if ttl > 0:
                create_time = struct.unpack_from('<d', buf, off + _E_CREATE_TIME)[0]
                if time.monotonic() - create_time >= ttl:
                    self._delete_slot(slot)
                    self._bump_generation()
                    self._inc_misses()
                    return None

            # Read data
            data_off = struct.unpack_from('<Q', buf, off + _E_DATA_OFF)[0]
            packet_size = struct.unpack_from('<I', buf, off + _E_PACKET_SIZE)[0]
            packet_view = buf[data_off:data_off + packet_size]

            # Deserialize
            result = None
            if _HAS_RUST and loads_rs is not None:
                try:
                    result = loads_rs(packet_view)
                except (ValueError, TypeError):
                    pass

            if result is None:
                result = loads(bytes(packet_view), copy=copy)
            elif copy:
                if hasattr(result, 'copy'):
                    result = result.copy()

            # Update access time and LRU
            struct.pack_into('<d', buf, off + _E_ACCESS_TIME, time.monotonic())
            self._lru_move_to_head(slot)
            self._inc_hits()

            # Convert to target device if requested
            if device is not None:
                if not copy and isinstance(result, np.ndarray):
                    result = result.copy()  # can't share SHM buffer with frameworks
                result = _numpy_to_device(result, device)

            return result

    def delete(self, key: str) -> bool:
        """
        Delete an entry from the cache.

        Parameters
        ----------
        key : str
            Cache key.

        Returns
        -------
        bool
            True if the key was found and deleted.
        """
        with self._lock:
            self._check_closed()
            with self._shm_locked():
                self._sync_index()

                slot = self._key_to_slot.get(key)
                if slot is None:
                    return False

                self._delete_slot(slot)
                self._bump_generation()
                return True

    def info(self, key: str) -> Optional[dict]:
        """
        Get metadata about a cached entry without deserializing.

        Parameters
        ----------
        key : str
            Cache key.

        Returns
        -------
        Optional[dict]
            Dictionary with 'shape', 'dtype', 'ndim', 'size_bytes',
            'ttl', 'age', or None if key not found.
        """
        with self._lock:
            self._check_closed()
            self._sync_index()

            slot = self._key_to_slot.get(key)
            if slot is None:
                return None

            buf = self._shm.buf
            off = self._slot_offset(slot)

            if buf[off + _E_STATUS] != _STATUS_ACTIVE:
                return None

            # Check TTL
            ttl = struct.unpack_from('<d', buf, off + _E_TTL)[0]
            create_time = struct.unpack_from('<d', buf, off + _E_CREATE_TIME)[0]
            now = time.monotonic()

            if ttl > 0 and now - create_time >= ttl:
                self._delete_slot(slot)
                self._bump_generation()
                return None

            data_off = struct.unpack_from('<Q', buf, off + _E_DATA_OFF)[0]
            packet_size = struct.unpack_from('<I', buf, off + _E_PACKET_SIZE)[0]

            # Use get_packet_info for accurate metadata
            try:
                packet_data = bytes(buf[data_off:data_off + min(packet_size, 512)])
                pinfo = get_packet_info(packet_data)
                return {
                    'shape': pinfo.get('shape', ()),
                    'dtype': pinfo.get('dtype'),
                    'ndim': pinfo.get('ndim', 0),
                    'size_bytes': packet_size,
                    'alloc_bytes': struct.unpack_from('<I', buf, off + _E_ALLOC_SIZE)[0],
                    'ttl': ttl if ttl > 0 else None,
                    'age': now - create_time,
                }
            except Exception:
                return {
                    'shape': (),
                    'dtype': None,
                    'ndim': 0,
                    'size_bytes': packet_size,
                    'alloc_bytes': struct.unpack_from('<I', buf, off + _E_ALLOC_SIZE)[0],
                    'ttl': ttl if ttl > 0 else None,
                    'age': now - create_time,
                }

    def keys(self) -> list:
        """Return a list of all active (non-expired) keys."""
        with self._lock:
            self._check_closed()
            self._sync_index()
            now = time.monotonic()
            result = []
            buf = self._shm.buf

            for key, slot in list(self._key_to_slot.items()):
                off = self._slot_offset(slot)
                if buf[off + _E_STATUS] != _STATUS_ACTIVE:
                    continue
                ttl = struct.unpack_from('<d', buf, off + _E_TTL)[0]
                if ttl > 0:
                    create_time = struct.unpack_from('<d', buf, off + _E_CREATE_TIME)[0]
                    if now - create_time >= ttl:
                        continue
                result.append(key)
            return result

    def clear(self):
        """Remove all entries from the cache."""
        with self._lock:
            self._check_closed()
            with self._shm_locked():
                buf = self._shm.buf

                # Clear all active entries
                for i in range(self._max_entries):
                    off = _POOL_HEADER_SIZE + i * _ENTRY_SIZE
                    buf[off + _E_STATUS] = _STATUS_FREE
                    struct.pack_into('<I', buf, off + _E_LRU_PREV, _SENTINEL)
                    struct.pack_into('<I', buf, off + _E_LRU_NEXT, _SENTINEL)

                # Reset pool header
                self._write_header_u32(_H_ACTIVE, 0)
                self._write_header_u64(_H_FREE_BYTES,
                                       self._data_region_end - self._data_region_offset)
                self._write_header_u64(_H_WATERMARK, self._data_region_offset)
                self._write_header_u32(_H_LRU_HEAD, _SENTINEL)
                self._write_header_u32(_H_LRU_TAIL, _SENTINEL)
                self._write_header_u64(_H_FREE_LIST, 0)

                self._key_to_slot.clear()
                self._bump_generation()

    def __len__(self) -> int:
        with self._lock:
            self._check_closed()
            return self._read_header_u32(_H_ACTIVE)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            self._check_closed()
            self._sync_index()

            slot = self._key_to_slot.get(key)
            if slot is None:
                return False

            buf = self._shm.buf
            off = self._slot_offset(slot)
            if buf[off + _E_STATUS] != _STATUS_ACTIVE:
                return False

            # Check TTL
            ttl = struct.unpack_from('<d', buf, off + _E_TTL)[0]
            if ttl > 0:
                create_time = struct.unpack_from('<d', buf, off + _E_CREATE_TIME)[0]
                if time.monotonic() - create_time >= ttl:
                    return False
            return True

    @property
    def stats(self) -> dict:
        """
        Cache statistics.

        Returns
        -------
        dict
            Keys: entries, max_entries, pool_size, used_bytes, free_bytes,
            data_region_size, hits, misses, hit_rate
        """
        with self._lock:
            self._check_closed()
            active = self._read_header_u32(_H_ACTIVE)
            free_bytes = self._read_header_u64(_H_FREE_BYTES)
            pool_size = self._read_header_u64(_H_POOL_SIZE)
            data_region_size = self._data_region_end - self._data_region_offset
            used_bytes = data_region_size - free_bytes
            hits = self._read_header_u64(_H_HITS)
            misses = self._read_header_u64(_H_MISSES)
            total = hits + misses
            return {
                'entries': active,
                'max_entries': self._max_entries,
                'pool_size': pool_size,
                'used_bytes': used_bytes,
                'free_bytes': free_bytes,
                'data_region_size': data_region_size,
                'hits': hits,
                'misses': misses,
                'hit_rate': hits / total if total > 0 else 0.0,
            }

    def _inc_hits(self):
        hits = self._read_header_u64(_H_HITS)
        self._write_header_u64(_H_HITS, hits + 1)

    def _inc_misses(self):
        misses = self._read_header_u64(_H_MISSES)
        self._write_header_u64(_H_MISSES, misses + 1)

    # -- Lifecycle --

    def _check_closed(self):
        if self._closed:
            raise RuntimeError("TensoCache is closed")

    def close(self):
        """Close access to the shared memory pool."""
        if not self._closed:
            self._closed = True
            try:
                lock_val = struct.unpack_from('<I', self._shm.buf, _H_LOCK)[0]
                if lock_val == 1:
                    self._shm_lock_release()
            except Exception:
                pass
            try:
                self._shm.close()
            except BufferError:
                # Zero-copy numpy views still reference the mmap.
                # Disarm __del__ so it won't re-raise the same error.
                self._shm._buf = None
                self._shm._mmap = None
            except Exception:
                pass

    def unlink(self):
        """Request destruction of the shared memory segment."""
        try:
            self._shm.unlink()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        if self._owns:
            self.unlink()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        status = "closed" if self._closed else "open"
        return (
            f"TensoCache(name={self._name!r}, status={status}, "
            f"pool_size={self._pool_size}, max_entries={self._max_entries})"
        )
