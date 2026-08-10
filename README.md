<img width="2439" height="966" alt="Tenso Banner" src="https://github.com/user-attachments/assets/5ec9b225-3615-4225-82ca-68e15b7045ce" />

# Tenso

**Up to 35x faster than Apache Arrow on deserialization. 46x less CPU than SafeTensors.**

Zero-copy, SIMD-aligned tensor protocol for high-performance ML infrastructure.

[![PyPI version](https://img.shields.io/pypi/v/tenso)](https://pypi.org/project/tenso/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

---

## Why Tenso?

Most serialization formats are designed for general data or disk storage. Tenso is **focused on network tensor transmission** where every microsecond matters.

### The Problem

Traditional formats waste CPU cycles during deserialization:
- **SafeTensors**: 37.1% CPU usage (great for disk, overkill for network)
- **Pickle**: 40.9% CPU usage + security vulnerabilities
- **Arrow**: Faster on serialization, but up to 32x slower on deserialization for large tensors

### The Solution

Tenso achieves **true zero-copy** with:
- **Minimalist Header**: Fixed 10-byte header eliminates JSON parsing overhead.
- **64-byte Alignment**: SIMD-ready padding ensures the data body is cache-line aligned.
- **Direct Memory Mapping**: The CPU points directly to existing buffers without copying.

**Result**: 0.8% CPU usage vs >40% for SafeTensors/Pickle.



---

## Benchmarks

For native, no-Python comparisons against the tools people actually use to move
tensors — CBOR, MessagePack, postcard, Apache Arrow, and ROS 2 / CDR — plus
binary footprint and verified embedded targets, see
[`benchmarks/`](https://github.com/Khushiyant/tenso/tree/main/benchmarks).

**System**: Python 3.12.9, NumPy 2.3.5, 12 CPU cores, M4 Pro

**Reproduce**: `pip install -e ".[dev]" && maturin develop --release && python benchmark.py`.
Timed sections take the fastest of 7 runs after 2 warmups; disk writes are fsynced and
disk reads fault in every page, so every row in a table measures the same work.

### 1. In-Memory Serialization (LLM Layer - 64MB)

| Format       | Size       | Serialize | Deserialize | Speedup (Deser) |
|--------------|------------|-----------|-------------|-----------------|
| **Tenso**    | 64.00 MB   | 4.00 ms   | **0.004 ms**| **1x**          |
| Arrow        | 64.00 MB   | 7.31 ms   | 0.008 ms    | 2x slower       |
| SafeTensors  | 64.00 MB   | 8.88 ms   | 2.98 ms     | 745x slower     |
| Pickle       | 64.00 MB   | 3.17 ms   | 3.00 ms     | 749x slower     |
| MsgPack      | 64.00 MB   | 11.51 ms  | 2.98 ms     | 744x slower     |

> **Reading the table**: Tenso and Arrow return zero-copy *views*; SafeTensors, Pickle
> and MsgPack *materialize* the array. The deserialize column therefore compares a view
> against a full reconstruction — which is exactly the "serialize once, read many"
> trade-off Tenso targets, not an identical-work decode. Against Arrow, which also
> returns a view, the margin at this size is ~2x; see table 5 for how it scales.

### 2. Disk I/O (256 MB Matrix)

| Format | Write (fsynced) | Read (all pages faulted) |
|--------|-----------------|--------------------------|
| **Tenso** | 47.05 ms | 31.21 ms |
| NumPy .npy | 31.13 ms | 29.53 ms |
| Pickle | 46.22 ms | 29.16 ms |

Disk I/O is bound by the device, not the format: with writes fsynced and reads actually
faulting in every page, all three land within noise of each other. Tenso's advantage is
not on this axis — it is that the mapped bytes need no decode step before use.

### 3. Stream Reading (95 MB Packet)

| Method | Time | Throughput | Speedup |
|--------|------|------------|---------|
| **Tenso read_stream** | **26.28 ms** | **3,628 MB/s** | **1x** |
| Naive chunk-and-join loop | 32.79 ms | 2,909 MB/s | 1.2x slower |

Framing dominates at this size, so the margin over a hand-rolled read loop is modest;
the value is that `read_stream` hands off to the decoder without a second copy.

### 4. CPU Usage (Efficiency)

| Format      | Serialize CPU% | Deserialize CPU% |
|-------------|----------------|------------------|
| **Tenso**   | 47.6%          | **0.2%**         |
| Arrow       | 59.6%          | 1.8%             |
| SafeTensors | 70.0%          | 41.7%            |
| Pickle      | 47.3%          | 42.9%            |

### 5. Arrow vs Tenso (Comparison)

| Size    | Tenso Ser | Arrow Ser | Tenso Des | Arrow Des | Speedup |
|---------|-----------|-----------|-----------|-----------|---------|
| Small   | 0.067ms   | 0.055ms   | 0.003ms   | 0.035ms   | 13.7x   |
| Medium  | 1.084ms   | 1.195ms   | 0.012ms   | 0.038ms   | 3.2x    |
| Large   | 6.145ms   | 8.053ms   | 0.033ms   | 0.538ms   | 16.2x   |
| XLarge  | 37.540ms  | 36.253ms  | 0.060ms   | 2.823ms   | **46.8x** |

Both return views, so this measures how each library's decode path scales rather than
memory bandwidth: Arrow's grows with size, Tenso's stays near-flat.

### 6. Network Performance

- **Packet Throughput**: 104,982 packets/sec (over localhost TCP, 1 KB tensors)
- **Latency**: 9.5 µs/packet (send-side cost; not a round trip)
- **Async serialization ceiling**: 6,242 MB/s (99,871 tensors/sec at 64 KB each) —
  encode plus framing into a null sink, with no socket or disk write. Real end-to-end
  throughput is bounded by your transport, not by this number.

---

## When should I use Tenso?

Tenso is built for one pattern: **serialize once, deserialize many times, over the network or between processes.** Reading data back is effectively free (a zero-copy memory view), so the more often your tensors are *read*, the more Tenso wins.

### Reach for Tenso when

- **You serve models over a network or RPC.** Inference nodes spend ~0.8% CPU decoding tensors instead of ~40%, leaving the cores for actual compute.
- **You pass tensors between machines or processes** — gradients/activations in distributed training (native Ray integration), or zero-copy hand-offs via shared memory (`TensoShm`).
- **Latency is critical** — real-time inference, robotics, sensor fusion (single-digit-µs reads).
- **You stream many tensors** — high-throughput pipelines, with `write_stream` sending straight from the array's own memory (no body copy).
- **You feed GPUs** from the network or disk with minimal host involvement.
- **You want integrity without overhead** — optional 64-bit XXH3 checksums.

### Consider an alternative when

- **You store model weights on disk and rarely reload them** → [SafeTensors](https://github.com/huggingface/safetensors) is purpose-built for that.
- **You need columnar analytics or a data ecosystem** (filters, Parquet, query engines) → [Apache Arrow](https://arrow.apache.org/).
- **You're serializing arbitrary Python objects**, not tensors → `pickle` (but never on untrusted input).
- **You only ever write and never read back** — Tenso's edge is on the read side, so the payoff is smaller.

> **Note on writes:** `dumps()` returns a fresh `bytes` object, so it pays a one-time buffer-allocation cost like any format that does. For write-heavy or networked paths, prefer `write_stream` / `iter_dumps`, which send the tensor's own memory with **zero body copy**.

### How it compares

| vs. | Built for | Where Tenso pulls ahead |
|-----|-----------|--------------------------|
| **Pickle** | Arbitrary Python objects | ~600x faster reads, ~50x less CPU, and **no arbitrary-code-execution risk** |
| **SafeTensors** | Safe on-disk weight storage | ~600x faster reads — **0.8% vs 37% CPU**, built for the wire, not the disk |
| **Apache Arrow** | Columnar data ecosystems | Up to **32x faster** deserialization on large tensors, with a tiny header instead of a full columnar runtime |
| **NumPy `.npy`** | Saving arrays to disk | Faster reads/writes **plus** streaming, integrity checks, and multi-tensor bundles |

---


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Khushiyant/tenso&type=timeline&legend=bottom-right)](https://www.star-history.com/#Khushiyant/tenso&type=timeline&legend=bottom-right)

---

## Installation

Tenso ships from one codebase to three ecosystems — all backed by the same Rust core.

### Python

```bash
pip install tenso
```

Prebuilt wheels (with the compiled core inside, no toolchain needed) cover Linux
x86_64/aarch64, macOS Apple Silicon, and Windows x64. Optional extras:

```bash
pip install tenso[api]    # gRPC, FastAPI, Ray integration
pip install tenso[gpu]    # GPU acceleration (CuPy/PyTorch/JAX)
```

### Rust

```bash
cargo add tenso
```

```rust
use tenso::{ArraySpec, Dtype, EncodeOpts, dense_required_size, encode_dense_into, decode};

let spec = ArraySpec { data: &bytes, dtype: Dtype::F32, shape: &[2, 3] };
let opts = EncodeOpts::default();

let mut buf = vec![0u8; dense_required_size(&spec, &opts)?];
encode_dense_into(&spec, &mut buf, &opts)?;
let decoded = decode(&buf)?;   // borrows buf, zero-copy
```

The C ABI (`tenso-ffi`), CUDA backend (`tenso-cuda`), and shared-memory bus
(`tenso-bus`) are published alongside it.

### C / C++

Download the `tenso-ffi-<platform>.tar.gz` archive from the
[latest GitHub Release](https://github.com/Khushiyant/tenso/releases) — it
contains `lib/` (shared + static) and `include/` (`tenso.h`, `tenso.hpp`). No
Rust toolchain required.

```c
#include "tenso.h"

uintptr_t need = 0, written = 0;
tenso_dense_required_size(data, data_len, dtype_code, shape, ndim,
                          /*check_integrity=*/false, /*compress=*/false,
                          /*alignment=*/64, &need);
uint8_t *out = malloc(need);
tenso_encode_dense_into(data, data_len, dtype_code, shape, ndim,
                        false, false, 64, out, need, &written);

TensoView *v = tenso_decode(packet, packet_len);
const uint8_t *body = tenso_view_body_ptr(v);
tenso_view_free(v);
```

---

## Quick Start

### Basic Serialization

```python
import numpy as np
import tenso

# Create tensor
data = np.random.rand(1024, 1024).astype(np.float32)

# Serialize
packet = tenso.dumps(data)

# Deserialize (Zero-copy view)
restored = tenso.loads(packet)
```

### Async I/O

```python
import asyncio
import tenso

async def handle_client(reader, writer):
    # Asynchronously read a tensor from the stream
    data = await tenso.aread_stream(reader)
    
    # Process and write back
    await tenso.awrite_stream(data * 2, writer)
```

### FastAPI Integration

```python
from fastapi import FastAPI
import numpy as np
from tenso.fastapi import TensoResponse

app = FastAPI()

@app.get("/tensor")
async def get_tensor():
    data = np.ones((1024, 1024), dtype=np.float32)
    return TensoResponse(data) # Zero-copy streaming response
```

---

## Advanced Features

### Ray Integration (Distributed Computing)

Replace pickle-based serialization in Ray with Tenso to cut encode cost on tensor
operations. Works transparently with `ray.put()`, `ray.get()`, remote functions, and
actors.

> Note: this path deserializes with `copy=True`, because a view into Ray's object store
> would not outlive the buffer Ray owns. So it avoids pickle's encode overhead but does
> allocate on read — the zero-copy deserialize figures above do not apply here.

```python
import ray
import numpy as np
from tenso.ray import register

ray.init()
register()  # Register Tenso as the serializer for numpy arrays

# All ray.put/get operations now use Tenso
ref = ray.put(np.zeros((1000, 1000)))
arr = ray.get(ref)  # Deserialized via Tenso

# Works transparently with remote functions
@ray.remote
def process(tensor):
    return tensor.mean()

ray.get(process.remote(np.random.randn(1000, 1000)))
```

Optional support for PyTorch and JAX tensors:

```python
register(include_torch=True, include_jax=True)
```

### Quantized Tensors (4-bit & 8-bit)

Native support for quantized representations to reduce memory footprint with minimal accuracy loss.

```python
from tenso.quantize import QuantizedTensor
import numpy as np

data = np.random.randn(1024, 1024).astype(np.float32)

# Quantize to 8-bit (per-tensor scheme)
qt = QuantizedTensor.quantize(data, dtype="qint8", scheme="per_tensor")
print(qt)  # QuantizedTensor(dtype=qint8, shape=(1024, 1024), ...)

# Serialize/deserialize with Tenso
import tenso
packet = tenso.dumps(qt)
restored = tenso.loads(packet)

# Dequantize back to float32
result = restored.dequantize()
```

Supported dtypes: `qint8`, `quint8`, `qint4`, `quint4`
Supported schemes: `per_tensor`, `per_channel`, `per_group`

### Inter-Process Communication (Shared Memory)

Transfer tensors between local processes with **single-digit microsecond latency** using Shared Memory. This avoids socket overhead entirely by passing memory handles.

```python
from tenso import TensoShm
import numpy as np

# Process A: Write to Shared Memory
data = np.random.randn(1024, 1024).astype(np.float32)
# Automatically sizes and creates the SHM segment
with TensoShm.create_from("shared_tensor_01", data) as shm:
    print("Tensor is in SHM. Waiting for reader...")
    input() # Keep process alive

# Process B: Read from Shared Memory (Zero-Copy)
with TensoShm("shared_tensor_01") as shm:
    # Instant view of the data without copying
    array = shm.get()
    print(f"Received: {array.shape}")
```

### GPU Acceleration (Direct Transfer)

Supports fast transfers between Tenso streams and device memory for **CuPy**, **PyTorch**, and **JAX** using pinned host memory.

```python
import tenso.gpu as tgpu

# Read directly from a stream into a GPU tensor
torch_tensor = tgpu.read_to_device(stream, device_id=0)
```

### bfloat16 Support

Native support for `bfloat16` dtype, commonly used in ML training. Works with NumPy 2.1+ natively or falls back to `ml_dtypes`.

```python
import numpy as np
import tenso

# Serialize bfloat16 tensors directly
data = np.ones((512, 512), dtype=np.float32)  # or bfloat16 if available
packet = tenso.dumps(data)
```

### Sparse Formats & Bundling

Tenso natively supports complex data structures beyond simple dense arrays:

* **Sparse Matrices**: Direct serialization for COO, CSR, and CSC formats.
* **Dictionary Bundling**: Pack multiple tensors into a single nested dictionary packet.
* **LZ4 Compression**: Optional high-speed compression for sparse or redundant data.

### Data Integrity (XXH3)

Protect your tensors against network corruption with ultra-fast 64-bit checksums:

```python
# Serialize with 64-bit checksum footer
packet = tenso.dumps(data, check_integrity=True)

# Verification is automatic during loads()
restored = tenso.loads(packet)
```

### gRPC Integration

Tenso provides built-in support for gRPC, allowing you to pass tensors between services with minimal overhead.

```python
from tenso.grpc import tenso_msg_pb2, tenso_msg_pb2_grpc
import tenso

# In your Servicer
def Predict(self, request, context):
    data = tenso.loads(request.tensor_packet)
    result = data * 2
    return tenso_msg_pb2.PredictResponse(
        result_packet=bytes(tenso.dumps(result))
    )
```

---

## Protocol Design

Tenso uses a minimalist structure designed for direct memory access:

```
┌─────────────┬──────────────┬──────────────┬────────────────────────┬──────────────┐
│   HEADER    │    SHAPE     │   PADDING    │    BODY (Raw Data)     │    FOOTER    │
│   10 bytes  │  Variable    │   0-63 bytes │   C-Contiguous Array   │   8 bytes*   │
└─────────────┴──────────────┴──────────────┴────────────────────────┴──────────────┘
                                                                        (*Optional)
```

The v4 header is 10 bytes: magic (`TNSO`, 4) + version (1) + flags (`u16`, 2) + dtype (1) + ndim (1) + reserved (1). Tenso writes v4 packets and still reads legacy v3 packets (which use an 8-byte header with a 1-byte flags field), and the widened `u16` flags field enables the StringTensor and RaggedArray formats.

The padding ensures the body starts at a **64-byte boundary**, enabling AVX-512 vectorization and zero-copy memory mapping.

---

## Use Cases

* **Model Serving APIs**: Up to 46x faster deserialization than Arrow, with far less CPU than SafeTensors, saves real overhead on inference nodes.
* **Distributed Training**: Efficiently pass gradients or activations between nodes with native Ray integration.
* **GPU-Direct Pipelines**: Stream data from network cards to GPU memory with minimal host intervention.
* **Real-time Robotics**: single-digit-µs decode, and a `no_std` core that builds for
  Cortex-M — see [`benchmarks/`](https://github.com/Khushiyant/tenso/tree/main/benchmarks)
  for the ROS 2 / CDR comparison.
* **High-Throughput Streaming**: ~105K packets/sec over localhost TCP for real-time data pipelines.

---

## Contributing

Contributions are welcome! A C ABI (`tenso-ffi`, usable from C/C++) already
ships; we are currently looking for help with:

* **More language clients**: JavaScript/WASM, Go, and other ecosystems on top of the C ABI.
* **Rust ergonomics**: a `tenso` umbrella crate exposing the device/cuda/bus features behind flags.

---

## License

Apache License 2.0 - see [LICENSE](LICENSE) file.

## Citation

```bibtex
@software{tenso2025,
  author = {Khushiyant},
  title = {Tenso: High-Performance Zero-Copy Tensor Protocol},
  year = {2025},
  url = {https://github.com/Khushiyant/tenso}
}
```
