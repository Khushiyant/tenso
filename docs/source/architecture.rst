Architecture
============

Tenso has a **single Rust core** that implements the wire format once, wrapped by
thin language bindings. Python, C/C++, GPU, and shared-memory transports all
encode and decode through that one core — there is no separate per-language
re-implementation and no pure-Python fallback.

Overview
--------

.. code-block:: text

    Python (pip)        C / C++ (binaries)     GPU            local IPC
    ┌────────────┐      ┌────────────────┐   ┌──────────┐   ┌──────────┐
    │  tenso_rs  │      │   tenso-ffi    │   │tenso-cuda│   │ tenso-bus│
    │  (PyO3)    │      │   (C ABI)      │   │          │   │          │
    └─────┬──────┘      └───────┬────────┘   └────┬─────┘   └────┬─────┘
          │                     │                 │              │
          └─────────────────────┴────────┬────────┴──────────────┘
                                          │
                                  ┌───────▼────────┐
                                  │     tenso      │   the single core
                                  │  (Rust, no_std │   - all encode/decode
                                  │   + alloc)     │   - zero-copy + SIMD layout
                                  └────────────────┘   - no external runtime deps

The core (the ``tenso`` crate) is ``no_std + alloc`` and depends on nothing
language- or OS-specific, so the *same* codec runs behind every binding.

How (de)serialization works
---------------------------

Every path funnels through the core:

1. **Python** — ``tenso.dumps`` / ``tenso.loads`` call the PyO3 binding
   (``tenso_rs``), which calls the ``tenso`` core. The compiled extension is
   **required**; there is no Python-side codec fallback.
2. **C / C++** — ``tenso-ffi`` exposes a stable C ABI (``tenso_encode_dense_into``,
   ``tenso_decode``, …) over the same core.
3. **Shared-memory IPC** — ``TensoShm`` (Python) and ``tenso-bus`` (Rust) write
   core-encoded packets directly into a shared buffer for zero-copy transfer.

.. code-block:: python

    import numpy as np
    import tenso

    data = np.random.rand(1000, 1000)
    packet = tenso.dumps(data)              # encoded by the Rust core
    packet_z = tenso.dumps(data, compress=True)   # LZ4 also runs in the core
    out = tenso.loads(packet)               # zero-copy view back

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

Workspace crates
----------------

The Rust side is a Cargo workspace. The core is published as ``tenso``; the
others are thin skins over it:

- ``tenso`` — the core codec (``no_std + alloc``); the single source of truth.
  Rust users add it with ``cargo add tenso`` and call ``encode_dense_into`` /
  ``decode``.
- ``tenso-ffi`` — the C ABI (``extern "C"``, ``tenso_``-prefixed). Generates
  ``include/tenso.h``; ships as prebuilt per-OS binaries.
- ``tenso-device`` — ``DeviceBackend`` trait + CPU/Mock backends and the
  GPU codec orchestration (IPC framing) over the core.
- ``tenso-cuda`` — CUDA backend; ``libcudart`` is ``dlopen``'d at runtime and
  gated behind the ``cuda`` feature (no link-time toolkit dependency).
- ``tenso-bus`` — shared-memory tensor bus (seqlock latest-value buffer + SPMC
  ring) carrying core-encoded packets.
- ``tenso-rs`` (repo root) — the PyO3 binding that produces the Python
  extension module ``tenso_rs``. Not published to crates.io.

Building
--------

The Python extension builds automatically via Maturin during install:

.. code-block:: bash

    # Development build (rebuilds the Rust extension)
    pip install -e .
    maturin develop --release

Working on the core or other crates directly:

.. code-block:: bash

    # Install the Rust toolchain
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

    # Build / test the whole workspace
    cargo build --workspace
    cargo test -p tenso -p tenso-device -p tenso-bus -p tenso-ffi

Source layout
-------------

- ``crates/tenso/`` - the core codec (encode/decode, dtypes, framing)
- ``crates/tenso-ffi/`` - C ABI + generated ``include/tenso.h``
- ``crates/tenso-device/``, ``crates/tenso-cuda/``, ``crates/tenso-bus/`` - device, CUDA, and IPC crates
- ``src/lib.rs`` - PyO3 binding (Python-facing glue only; calls the core)
- ``src/tenso/`` - Python package (high-level API, async, GPU, integrations)
- ``Cargo.toml`` / ``pyproject.toml`` - workspace + Python build config

Why Rust?
---------

1. **Zero-Copy Memory Access**: direct pointer manipulation without holding the GIL.
2. **SIMD-friendly layout**: 64-byte alignment enables compiler auto-vectorization.
3. **Type safety**: compile-time guarantees prevent whole classes of memory bugs.
4. **Portable core**: ``no_std`` means the same codec runs in Python, C/C++, and
   eventually embedded/WASM targets.

The overhead of calling Rust from Python is ~100ns, negligible against the
microseconds saved during (de)serialization.

Future Extensions
-----------------

- GPU-direct deserialization (broader CUDA/ROCm interop)
- WebAssembly build of the core for browser use
- A ``tenso`` umbrella crate exposing the device/cuda/bus features behind flags
