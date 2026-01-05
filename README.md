
<img width="2439" height="966" alt="Tenso Banner" src="https://github.com/user-attachments/assets/5ec9b225-3615-4225-82ca-68e15b7045ce" />

# Tenso

<<<<<<< HEAD
<<<<<<< HEAD
**Up to 22x faster than Apache Arrow on deserialization. 55x less CPU than SafeTensors.**
=======
**Up to 140x faster than Apache Arrow. 42x less CPU than SafeTensors.**
>>>>>>> ddfc92b (fix: update performance metrics and benchmarks in README and benchmark.py)
=======
**Up to 22x faster than Apache Arrow on deserialization. 55x less CPU than SafeTensors.**
>>>>>>> aca8de7 (fix: update performance metrics in README and benchmark script)

Zero-copy, SIMD-aligned tensor protocol for high-performance ML infrastructure.

[![PyPI version](https://img.shields.io/pypi/v/tenso)](https://pypi.org/project/tenso/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

---

## Why Tenso?

Most serialization formats are designed for general data or disk storage. Tenso is **focused on network tensor transmission** where every microsecond matters.

### The Problem

Traditional formats waste CPU cycles during deserialization:
<<<<<<< HEAD
<<<<<<< HEAD
- **SafeTensors**: 38.8% CPU usage (great for disk, overkill for network)
- **Pickle**: 41.5% CPU usage + security vulnerabilities
- **Arrow**: Faster on serialization, but up to 22x slower on deserialization for large tensors
=======
- **SafeTensors**: 41.2% CPU usage (great for disk, overkill for network)
- **Pickle**: 41.3% CPU usage + security vulnerabilities
- **Arrow**: Fast, but up to 140x slower than Tenso for large tensors
>>>>>>> ddfc92b (fix: update performance metrics and benchmarks in README and benchmark.py)
=======
- **SafeTensors**: 38.8% CPU usage (great for disk, overkill for network)
- **Pickle**: 41.5% CPU usage + security vulnerabilities
- **Arrow**: Faster on serialization, but up to 22x slower on deserialization for large tensors
>>>>>>> aca8de7 (fix: update performance metrics in README and benchmark script)

### The Solution

Tenso achieves **true zero-copy** with:
- **Minimalist Header**: Fixed 8-byte header eliminates JSON parsing overhead.
- **64-byte Alignment**: SIMD-ready padding ensures the data body is cache-line aligned.
- **Direct Memory Mapping**: The CPU points directly to existing buffers without copying.

<<<<<<< HEAD
<<<<<<< HEAD
**Result**: 0.7% CPU usage vs >38% for SafeTensors/Pickle.
=======
**Result**: 1.1% CPU usage vs >41% for SafeTensors/Pickle.
>>>>>>> ddfc92b (fix: update performance metrics and benchmarks in README and benchmark.py)
=======
**Result**: 0.7% CPU usage vs >38% for SafeTensors/Pickle.
>>>>>>> aca8de7 (fix: update performance metrics in README and benchmark script)

---

## Benchmarks

**System**: Python 3.12.9, NumPy 2.3.5, 12 CPU cores, macOS

### Deserialization Speed (256 MB Matrix - 8192×8192 Float32)

<<<<<<< HEAD
<<<<<<< HEAD
| Format | Read Time | CPU Usage | Speedup |
|--------|-----------|-----------|---------|
=======
| Format | Read Time | CPU Usage | Speedup |
|--------|-----------|-----------|---------|---|
>>>>>>> aca8de7 (fix: update performance metrics in README and benchmark script)
| **Tenso** | **44.65ms** | **0.7%** | **1x** |
| NumPy .npy | 46.14ms | N/A | 1.03x slower |
| Pickle | 25.23ms* | 41.5% | 1.8x faster† |
| SafeTensors | ~3.42s | 38.8% | 77x slower |
| Arrow (zero-copy) | ~0.35s | 1.2% | 7.8x slower |
<<<<<<< HEAD
=======
| Format | Time (mean±std) | CPU Usage | Speedup |
|--------|-----------------|-----------|---------|------|
| **Tenso** | **0.03±0.01ms** | **1.1%** | **1x** |
| Arrow | 4.35±0.22ms | 0.9% | 140x slower |
| SafeTensors | 2.80±0.39ms | 41.2% | 93x slower |
| Pickle | 2.91±0.45ms | 41.3% | 97x slower |

**Note**: Benchmarks based on 15-20 iterations with warmup runs. Standard deviation shows measurement consistency.
>>>>>>> ddfc92b (fix: update performance metrics and benchmarks in README and benchmark.py)
=======

*Pickle faster on disk read but uses 59x more CPU and lacks security  
†Tenso optimized for network streaming, not disk I/O

### Large Tensor Performance (XLarge Dataset)

| Format | Serialization | Deserialization | Speedup (Deser) |
|--------|---------------|-----------------|---------|---|
| **Tenso** | 84.75ms | **0.059ms** | **1x** |
| Arrow (zero-copy) | 16.34ms | 1.306ms | 22.2x slower |
>>>>>>> aca8de7 (fix: update performance metrics in README and benchmark script)

*Pickle faster on disk read but uses 59x more CPU and lacks security  
†Tenso optimized for network streaming, not disk I/O

<<<<<<< HEAD
### Large Tensor Performance (XLarge Dataset)

| Format | Serialization | Deserialization | Speedup (Deser) |
|--------|---------------|-----------------|-----------------|
| **Tenso** | 84.75ms | **0.059ms** | **1x** |
| Arrow (zero-copy) | 16.34ms | 1.306ms | 22.2x slower |

### Stream Reading Performance (95 MB Packet)

<<<<<<< HEAD
| Method | Time | Throughput | Speedup |
|--------|------|------------|---------|
| **Tenso read_stream** | **6.43ms** | **14,830 MB/s** | **1x** |
| Naive loop | 14.50ms | 6,577 MB/s | 2.3x slower |
=======
| Method | Time (mean±std) | Throughput | Speedup |
|--------|-----------------|------------|---------|------|
| **Tenso read_stream** | **4.47±1.08ms** | **21,349 MB/s** | **1x** |
| Naive loop | 7,220.64±107.97ms | 13.21 MB/s | 1,616x slower |
=======
### Stream Reading Performance (95 MB Packet)

| Method | Time | Throughput | Speedup |
|--------|------|------------|---------|---|
| **Tenso read_stream** | **6.43ms** | **14,830 MB/s** | **1x** |
| Naive loop | 14.50ms | 6,577 MB/s | 2.3x slower |
>>>>>>> aca8de7 (fix: update performance metrics in README and benchmark script)

### Async I/O Performance (5,000 tensors × 64 KB)

| Method | Time | Throughput | Tensor Rate |
|--------|------|------------|-------------|---|
| **Async Write** | **4.3ms** | **72,021 MB/s** | **1.15M tensors/sec** |

### Network Transmission (10,000 packets × 1KB over TCP)

<<<<<<< HEAD
**

### Disk I/O Performance (256MB Matrix: 8192×8192 Float32)

| Format | Write (mean±std) | Read (mean±std) |
|--------|------------------|------------------|
| **Tenso** | **25.74±0.11ms** | **0.58±0.83ms** |
| NumPy .npy | 26.82±7.14ms | 25.63±63.56ms |
| Pickle | 52.41±2.03ms | 27.41±2.08ms |

**Note**: Low standard deviation in Tenso's write performance shows consistent, reliable behavior.

**

### Network Performance (Async I/O)

- **Throughput**: 1,060,867 tensors/sec (66,304 MB/s)
- **Latency**: 11.8 µs/packet (10,000 tensors over localhost TCP)
>>>>>>> ddfc92b (fix: update performance metrics and benchmarks in README and benchmark.py)
=======
| Metric | Performance |
|--------|-------------|
| **Throughput** | **88,491 packets/sec** |
| **Latency** | **11.3 µs/packet** |
>>>>>>> aca8de7 (fix: update performance metrics in README and benchmark script)

### Async I/O Performance (5,000 tensors × 64 KB)

| Method | Time | Throughput | Tensor Rate |
|--------|------|------------|-------------|
| **Async Write** | **4.3ms** | **72,021 MB/s** | **1.15M tensors/sec** |

### Network Transmission (10,000 packets × 1KB over TCP)

| Metric | Performance |
|--------|-------------|
| **Throughput** | **88,491 packets/sec** |
| **Latency** | **11.3 µs/packet** |

---

## Installation

```bash
pip install tenso

```

---

## Quick Start (v0.12.0)

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

**

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

**

---

## Advanced Features

### GPU Acceleration (Direct Transfer)

Supports fast transfers between Tenso streams and device memory for **CuPy**, **PyTorch**, and **JAX** using pinned host memory.

```python
import tenso.gpu as tgpu

# Read directly from a stream into a GPU tensor
torch_tensor = tgpu.read_to_device(stream, device_id=0) 

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

**

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

<<<<<<< HEAD
<<<<<<< HEAD
* **Model Serving APIs**: Up to 22x faster deserialization with 55x less CPU saves massive overhead on inference nodes.
* **Distributed Training**: Efficiently pass gradients or activations between nodes (Ray, Spark) at 72 GB/s.
=======
* **Model Serving APIs**: Up to 140x faster deserialization saves massive CPU overhead on inference nodes.
* **Distributed Training**: Efficiently pass gradients or activations between nodes (Ray, Spark).
>>>>>>> ddfc92b (fix: update performance metrics and benchmarks in README and benchmark.py)
* **GPU-Direct Pipelines**: Stream data from network cards to GPU memory with minimal host intervention.
* **Real-time Robotics**: 11.3 µs latency for high-frequency sensor fusion (LIDAR, Radar).
* **High-Throughput Streaming**: 88K packets/sec network transmission for real-time data pipelines.
=======
* **Model Serving APIs**: Up to 22x faster deserialization with 55x less CPU saves massive overhead on inference nodes.
* **Distributed Training**: Efficiently pass gradients or activations between nodes (Ray, Spark) at 72 GB/s.
* **GPU-Direct Pipelines**: Stream data from network cards to GPU memory with minimal host intervention.
* **Real-time Robotics**: 11.3 µs latency for high-frequency sensor fusion (LIDAR, Radar).
* **High-Throughput Streaming**: 88K packets/sec network transmission for real-time data pipelines.

---

## Contributing

Contributions are welcome! We are currently looking for help with:

* **Rust Core**: Porting serialization logic to Rust for even lower overhead.
* **C++ / JavaScript Clients**: Extending the protocol to other ecosystems.
>>>>>>> aca8de7 (fix: update performance metrics in README and benchmark script)

---

## License

Apache License 2.0 - see [LICENSE](https://www.google.com/search?q=LICENSE) file.

## Citation

```bibtex
@software{tenso2025,
  author = {Khushiyant},
  title = {Tenso: High-Performance Zero-Copy Tensor Protocol},
  year = {2025},
  url = {[https://github.com/Khushiyant/tenso](https://github.com/Khushiyant/tenso)}
}

```
