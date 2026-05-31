Architecture
============

Tenso uses a **hybrid Python-Rust architecture** to achieve maximum performance while maintaining an intuitive Python API.

Overview
--------

.. code-block:: text

    ┌─────────────────────────────────────────┐
    │  Python API Layer (tenso.*)             │
    │  - High-level functions                 │
    │  - Type validation                      │
    │  - Feature routing (GPU, async, etc.)   │
    └──────────┬──────────────────────────────┘
               │
               ├─── Fast Path: Rust Core (tenso_rs)
               │    └─→ dumps_rs(), loads_rs(), dump_to_fd_rs()
               │        • Zero-copy serialization
               │        • SIMD-optimized operations
               │        • ~35x faster deserialization
               │
               └─── Fallback: Pure Python
                    └─→ Used for compression, sparse matrices, bundles

Performance Strategy
--------------------

Tenso automatically selects the optimal implementation:

1. **Rust Fast Path** (Primary)
   
   - Used for standard NumPy arrays
   - Requirements: C-contiguous, supported dtype, no compression
   - Implementation: ``tenso_rs`` Rust extension module
   - Performance: 0.004ms deserialize time for 64MB

2. **Python Fallback** (Automatic)
   
   - Used when Rust requirements aren't met
   - Handles: LZ4 compression, sparse matrices, bundles, complex dtypes
   - Still optimized with NumPy/xxhash

3. **Shared Memory IPC**
   
   - Used for local inter-process communication
   - Implementation: ``TensoShm`` class backed by ``dump_to_buffer_rs``
   - Performance: Zero-copy transfer via memory mapping

.. code-block:: python

    import numpy as np
    import tenso

    # Uses Rust fast path automatically
    data = np.random.rand(1000, 1000)
    packet = tenso.dumps(data)  # → calls dumps_rs() internally
    
    # Falls back to Python for compression
    packet_compressed = tenso.dumps(data, compress=True)

Wire Protocol
-------------

Every Tenso packet starts with a fixed-size header followed by a shape block,
optional padding, the body, and an optional 8-byte XXH3 footer.

**v4 header (current, 10 bytes):**

.. code-block:: text

    offset  size  field
    ------  ----  ----------------------------------------
       0     4    magic  = b"TNSO"
       4     1    version = 4
       5     2    flags (u16, little-endian)
       7     1    dtype_code
       8     1    ndim
       9     1    reserved (must be 0; ignored on read)

**v3 header (legacy, 8 bytes):**

.. code-block:: text

    offset  size  field
    ------  ----  ----------------------------------------
       0     4    magic  = b"TNSO"
       4     1    version = 3
       5     1    flags (u8)
       6     1    dtype_code
       7     1    ndim

The version bump from 3 to 4 widens ``flags`` from 8 to 16 bits to leave room
for future feature flags. All other field semantics are unchanged.

**Compatibility:**

- Tenso ≥ 0.21 emits v4 packets and reads both v3 and v4.
- Tenso ≤ 0.20 emits v3 packets and only reads v3 — it cannot read v4.
- Older clients reading a v4 packet will fail at the magic+version check or
  parse the wrong fields. If you need to interop with old clients across the
  upgrade, hold readers ahead of writers.

Rust Components
---------------

The Rust extension (``tenso_rs``) provides core functions exposed to Python via PyO3:

``dumps_rs(tensor, check_integrity=False, alignment=64) -> bytes``
    Serialize a NumPy array with zero-copy efficiency.

``dump_to_buffer_rs(array, buffer, check_integrity=False) -> int``
    Serialize directly into a pre-allocated writable buffer (e.g., SharedMemory).

``loads_rs(packet) -> numpy.ndarray``
    Deserialize a Tenso packet with minimal memory copying.

``dump_to_fd_rs(fd, tensor, check_integrity=False) -> int``
    Write directly to a file descriptor (Unix systems).

These are **not meant to be called directly**—use the Python API functions instead.

Building the Extension
----------------------

The Rust extension is built automatically via Maturin during package installation:

.. code-block:: bash

    # Development build
    pip install -e .
    
    # Or explicitly rebuild the Rust extension
    maturin develop --release

For contributors working on the Rust code:

.. code-block:: bash

    # Install Rust toolchain
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    
    # Edit Rust source
    vim src/lib.rs
    
    # Rebuild and test
    maturin develop && pytest

Source Files
------------

- ``src/lib.rs`` - Rust implementation (serialization, deserialization, dtypes)
- ``src/tenso/core.py`` - Python wrapper that calls Rust or falls back
- ``Cargo.toml`` - Rust dependencies (PyO3, numpy, xxhash, lz4_flex, rayon)
- ``pyproject.toml`` - Python package config and Maturin build settings

Why Rust?
---------

1. **Zero-Copy Memory Access**: Direct pointer manipulation without Python GIL
2. **SIMD Optimization**: Compiler auto-vectorization for data alignment
3. **Type Safety**: Compile-time guarantees prevent segfaults
4. **Parallelism**: Rayon for parallel processing without GIL limitations

The overhead of calling Rust from Python is ~100ns, which is negligible compared to the microseconds saved during (de)serialization.

Future Extensions
-----------------

Planned Rust optimizations:

- LZ4 compression integration (currently Python-only)
- GPU-direct deserialization (CUDA/ROCm interop)
- WebAssembly compilation for browser use
