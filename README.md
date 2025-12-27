
<img width="2439" height="966" alt="Tenso Banner" src="https://github.com/user-attachments/assets/5ec9b225-3615-4225-82ca-68e15b7045ce" />

# Tenso

**Up to 12.6x faster than Apache Arrow. 32x less CPU than SafeTensors.**

Zero-copy, SIMD-aligned tensor protocol for high-performance ML infrastructure.

[![PyPI version](https://img.shields.io/pypi/v/tenso)](https://pypi.org/project/tenso/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

---

## Why Tenso?

Most serialization formats are designed for general data or disk storage. Tenso is **focused on network tensor transmission** where every microsecond matters.

### The Problem

Traditional formats waste CPU cycles during deserialization:
- **SafeTensors**: 41.3% CPU usage (great for disk, overkill for network)
- **Pickle**: 43.3% CPU usage + security vulnerabilities
- **Arrow**: Fast, but 12.6x slower than Tenso for large tensors

### The Solution

Tenso achieves **true zero-copy** with:
- **Minimalist Header**: Fixed 8-byte header eliminates JSON parsing overhead.
- **64-byte Alignment**: SIMD-ready padding ensures the data body is cache-line aligned.
- **Direct Memory Mapping**: The CPU points directly to existing buffers without copying.

**Result**: ~1.3% CPU usage vs >40% for SafeTensors/Pickle.

---

## Benchmarks

**System**: Python 3.12.9, NumPy 2.3.5, 12 CPU cores, macOS

### Deserialization Speed (8192×8192 Float32 Matrix)

| Format | Time | CPU Usage | Speedup |
|--------|------|-----------|---------|
| **Tenso** | **0.064ms** | **1.3%** | **1x** |
| Arrow | 0.810ms | 1.2% | 12.6x slower |
| SafeTensors | 2.792ms | 41.3% | 43x slower |
| Pickle | 3.031ms | 43.3% | 47x slower |

**

### Stream Reading Performance (95MB Packet)

| Method | Time | Throughput | Speedup |
|--------|------|------------|---------|
| **Tenso read_stream** | **7.05ms** | **13,534 MB/s** | **1x** |
| Naive loop | 7,399.7ms | 12.8 MB/s | 1,050x slower |

**

---

## Installation

```bash
pip install tenso

```

---

## Quick Start (v0.10.0)

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

* **Model Serving APIs**: 12.6x faster deserialization saves massive CPU overhead on inference nodes.
* **Distributed Training**: Efficiently pass gradients or activations between nodes (Ray, Spark).
* **GPU-Direct Pipelines**: Stream data from network cards to GPU memory with minimal host intervention.
* **Real-time Robotics**: Sub-millisecond latency for high-frequency sensor fusion (LIDAR, Radar).

---

## Contributing

Contributions are welcome! We are currently looking for help with:

* **Rust Core**: Porting serialization logic to Rust for even lower overhead.
* **gRPC Integration**: Native Tenso marshaling for gRPC services.
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
  version = {0.10.0},
  url = {[https://github.com/Khushiyant/tenso](https://github.com/Khushiyant/tenso)}
}

```
