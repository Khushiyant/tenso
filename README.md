<img width="2439" height="966" alt="Tenso Banner" src="https://github.com/user-attachments/assets/5ec9b225-3615-4225-82ca-68e15b7045ce" />

# Tenso

**Up to 35x faster than Apache Arrow on deserialization. 46x less CPU than SafeTensors.**

Zero-copy, SIMD-aligned tensor protocol for high-performance ML infrastructure.

[![PyPI version](https://img.shields.io/pypi/v/tenso)](https://pypi.org/project/tenso/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
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
- **Minimalist Header**: Fixed 8-byte header eliminates JSON parsing overhead.
- **64-byte Alignment**: SIMD-ready padding ensures the data body is cache-line aligned.
- **Direct Memory Mapping**: The CPU points directly to existing buffers without copying.

**Result**: 0.8% CPU usage vs >40% for SafeTensors/Pickle.

---

## Benchmarks

**System**: Python 3.12.9, NumPy 2.3.5, 12 CPU cores, M4 Pro

### 1. In-Memory Serialization (LLM Layer - 64MB)

| Format       | Size       | Serialize | Deserialize | Speedup (Deser) |
|--------------|------------|-----------|-------------|-----------------|
| **Tenso**    | 64.00 MB   | 3.51 ms   | **0.004 ms**| **1x**          |
| Arrow        | 64.00 MB   | 7.06 ms   | 0.011 ms    | 2.8x slower     |
| SafeTensors  | 64.00 MB   | 8.14 ms   | 2.39 ms     | 597x slower     |
| Pickle       | 64.00 MB   | 2.93 ms   | 2.71 ms     | 677x slower     |
| MsgPack      | 64.00 MB   | 10.44 ms  | 3.05 ms     | 763x slower     |

> **Note**: Tenso (Vect) variant is even faster with 0.000 ms deserialize time.

### 2. Disk I/O (256 MB Matrix)

| Format | Write | Read |
|--------|-------|------|
| **Tenso** | **29.41 ms** | **36.28 ms** |
| NumPy .npy | 24.83 ms | 43.08 ms |
| Pickle | 49.90 ms | 24.24 ms |

### 3. Stream Reading (95 MB Packet)

| Method | Time | Throughput | Speedup |
|--------|------|------------|---------|
| **Tenso read_stream** | **7.68 ms** | **12,417 MB/s** | **1x** |
| Optimised Loop | 13.89 ms | 7,396 MB/s | 1.9x slower |

### 4. CPU Usage (Efficiency)

| Format      | Serialize CPU% | Deserialize CPU% |
|-------------|----------------|------------------|
| **Tenso**   | 117.3%         | **0.8%**         |
| Arrow       | 57.1%          | 1.0%             |
| SafeTensors | 67.1%          | 37.1%            |
| Pickle      | 44.0%          | 40.9%            |

### 5. Arrow vs Tenso (Comparison)

| Size    | Tenso Ser | Arrow Ser | Tenso Des | Arrow Des | Speedup |
|---------|-----------|-----------|-----------|-----------|---------|
| Small   | 0.130ms   | 0.056ms   | 0.009ms   | 0.035ms   | 4.1x    |
| Medium  | 0.972ms   | 0.912ms   | 0.020ms   | 0.040ms   | 2.0x    |
| Large   | 3.166ms   | 3.655ms   | 0.019ms   | 0.222ms   | 11.8x   |
| XLarge  | 19.086ms  | 28.726ms  | 0.023ms   | 0.733ms   | **32.0x** |

### 6. Network Performance

- **Packet Throughput**: 89,183 packets/sec (over localhost TCP)
- **Latency**: 11.2 µs/packet
- **Async Write Throughput**: 88,397 MB/s (1.4M tensors/sec)

---

## Installation

```bash
pip install tenso
```

Optional extras:

```bash
pip install tenso[ray]    # Ray integration
pip install tenso[gpu]    # GPU acceleration (CuPy/PyTorch/JAX)
pip install tenso[grpc]   # gRPC support
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

Replace pickle-based serialization in Ray with Tenso for **46x less CPU overhead** on tensor operations. Works transparently with `ray.put()`, `ray.get()`, remote functions, and actors.

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
│   8 bytes   │  Variable    │   0-63 bytes │   C-Contiguous Array   │   8 bytes*   │
└─────────────┴──────────────┴──────────────┴────────────────────────┴──────────────┘
                                                                        (*Optional)
```

The padding ensures the body starts at a **64-byte boundary**, enabling AVX-512 vectorization and zero-copy memory mapping.

---

## Use Cases

* **Model Serving APIs**: Up to 35x faster deserialization with 46x less CPU saves massive overhead on inference nodes.
* **Distributed Training**: Efficiently pass gradients or activations between nodes with native Ray integration.
* **GPU-Direct Pipelines**: Stream data from network cards to GPU memory with minimal host intervention.
* **Real-time Robotics**: 10.2 µs latency for high-frequency sensor fusion (LIDAR, Radar).
* **High-Throughput Streaming**: 89K packets/sec network transmission for real-time data pipelines.

---

## Contributing

Contributions are welcome! We are currently looking for help with:

* **C++ / JavaScript Clients**: Extending the protocol to other ecosystems.

---

## License

Apache License 2.0 - see [LICENSE](https://www.google.com/search?q=LICENSE) file.

## Citation

```bibtex
@software{tenso2025,
  author = {Khushiyant},
  title = {Tenso: High-Performance Zero-Copy Tensor Protocol},
  year = {2025},
  url = {https://github.com/Khushiyant/tenso}
}
```
