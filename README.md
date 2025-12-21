<img width="2439" height="966" alt="Tenso Banner" src="https://github.com/user-attachments/assets/5ec9b225-3615-4225-82ca-68e15b7045ce" />

# Tenso

**Up to 23.8x faster than Apache Arrow. 61x less CPU than SafeTensors.**

Zero-copy, SIMD-aligned tensor protocol for high-performance ML infrastructure.

[![PyPI version](https://img.shields.io/pypi/v/tenso)](https://pypi.org/project/tenso/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Why Tenso?

Most serialization formats are designed for general data or disk storage. Tenso is **focused on network tensor transmission** where every microsecond matters.

### The Problem

Traditional formats waste CPU cycles:
- **SafeTensors**: 36.7% CPU usage per deserialization (great for disk, overkill for network)
- **Pickle**: 41.7% CPU usage + security vulnerabilities
- **Arrow**: Fast, but 23.8x slower than Tenso for large tensors

### The Solution

Tenso achieves **true zero-copy** with:
- Fixed 8-byte header (no JSON parsing overhead)
- 64-byte memory alignment (SIMD-ready)
- Direct memory mapping (CPU just points, never copies)

**Result**: 0.6% CPU usage vs 36.7% for SafeTensors

---

## Benchmarks

**System**: Python 3.12.9, NumPy 2.3.5, 12 CPU cores, macOS

### Deserialization Speed (8192×8192 Float32 Matrix)

| Format | Time | CPU Usage | Speedup |
|--------|------|-----------|---------|
| **Tenso** | **0.034ms** | **0.6%** | **1x** |
| Arrow | 0.805ms | 1.1% | 23.8x slower |
| SafeTensors | 2.621ms | 36.7% | 77x slower |
| Pickle | 3.293ms | 41.7% | 97x slower |

### Stream Reading Performance (95MB Packet)

| Method | Time | Throughput | Speedup |
|--------|------|------------|---------|
| **Tenso read_stream** | **21ms** | **4,500 MB/s** | **1x** |
| Naive loop | 7,870ms | 12 MB/s | 371x slower |

### Network Latency (1KB Tensor over TCP)

| Metric | Value |
|--------|-------|
| **Throughput** | **182,940 packets/sec** |
| **Latency** | **5.5 μs/packet** |

### Real-World Impact

**Scenario**: Inference API serving 10,000 req/sec with 64MB tensors

| Format | CPU Cores Used | Monthly Cost* | 
|--------|----------------|---------------|
| SafeTensors | 367 cores | ~$15,000 | 
| **Tenso** | **6 cores** | **~$245** |

*Based on typical cloud compute pricing

---

## Installation

```bash
pip install tenso
```

---

## Quick Start

### Basic Serialization

```python
import numpy as np
import tenso

# Create tensor
data = np.random.rand(1024, 1024).astype(np.float32)

# Serialize (8.5ms for 64MB)
packet = tenso.dumps(data)

# Deserialize (0.034ms for 64MB)
restored = tenso.loads(packet)
```

### Network Communication

```python
import socket
import tenso

# Server: Receive tensor
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('0.0.0.0', 9999))
server.listen(1)
conn, addr = server.accept()

# Zero-copy read with automatic buffering
tensor = tenso.read_stream(conn)  # Uses readinto() internally

# Client: Send tensor
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('localhost', 9999))

data = np.random.rand(256, 256).astype(np.float32)
tenso.write_stream(data, client)  # Atomic write with os.writev
```

### File I/O with Memory Mapping

```python
# Write to disk
with open("model_weights.tenso", "wb") as f:
    tenso.dump(large_tensor, f)

# Instant load (no matter the size)
with open("model_weights.tenso", "rb") as f:
    weights = tenso.load(f, mmap_mode=True)  # Memory-mapped, not loaded into RAM
```

---

## Use Cases

### Perfect For

- **Model Serving APIs** - 23.8x faster deserialization saves CPU cores
- **Distributed Training** - Efficient gradient/activation passing (Ray, Spark)
- **Real-time Robotics** - Sub-millisecond latency sensor fusion
- **High-Frequency Trading** - Microsecond-precision data exchange
- **Microservices** - Fast tensor exchange between services
- **Edge Devices** - Minimal dependencies, pure Python

### Consider Alternatives For

- **Long-term Model Storage** - Use SafeTensors (better ecosystem, HuggingFace integration)
- **Multi-column Dataframes** - Use Arrow (designed for tabular data)
- **Arbitrary Python Objects** - Use Pickle (if you trust the source)

---

## Protocol Design

Tenso uses a minimalist 4-part structure:

```
┌─────────────┬──────────────┬──────────────┬────────────────────────┬──────────────┐
│   HEADER    │    SHAPE     │   PADDING    │    BODY (Raw Data)     │    FOOTER    │
│   8 bytes   │  Variable    │   0-63 bytes │   C-Contiguous Array   │   8 bytes*   │
└─────────────┴──────────────┴──────────────┴────────────────────────┴──────────────┘
                                                                        (*Optional)
```

### Header (8 bytes)
```python
[4 bytes: Magic "TNSO"]
[1 byte:  Protocol Version (2)]
[1 byte:  Flags (Bit 0: Aligned, Bit 1: Integrity)]
[1 byte:  Dtype Code]
[1 byte:  Number of Dimensions]
```

### Why This Is Fast

**SafeTensors**: Uses JSON header - 3.67ms parsing overhead  
**Arrow**: Complex IPC format with schema validation - 0.805ms overhead  
**Tenso**: Fixed 8-byte struct - 0.034ms (just unpack and memory map)

The padding ensures the data body starts at a 64-byte boundary, enabling:
- AVX-512 vectorization
- Zero-copy memory mapping
- Cache-line alignment

---

## Advanced Features

### Data Integrity (XXH3)

Protect your tensors against network corruption with ultra-fast XXH3 hashing (adds <2% overhead):

```python
# Serialize with 64-bit checksum
packet = tenso.dumps(data, check_integrity=True)

# Verification is automatic during load
try:
    restored = tenso.loads(packet)
except ValueError:
    print("Detected corrupted data!")
```
### Strict Mode

Prevents accidental memory copies:

```python
# Force C-contiguous check
try:
    packet = tenso.dumps(fortran_array, strict=True)
except ValueError:
    print("Array must be C-contiguous!")
    fortran_array = np.ascontiguousarray(fortran_array)
```

### Packet Introspection

Inspect metadata without deserializing:

```python
info = tenso.get_packet_info(packet)
print(f"Shape: {info['shape']}")
print(f"Dtype: {info['dtype']}")
print(f"Size: {info['data_size_bytes']} bytes")
```

### Supported Dtypes

All NumPy numeric types including:
- Floats: `float16`, `float32`, `float64`
- Integers: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`
- Complex: `complex64`, `complex128`
- Boolean: `bool`

---

## Comparison Table

| Feature | Tenso | Arrow | SafeTensors | Pickle |
|---------|-------|-------|-------------|--------|
| **Deserialize Speed (64MB)** | 0.034ms | 0.805ms | 2.621ms | 3.293ms |
| **CPU Usage** | 0.6% | 1.1% | 36.7% | 41.7% |
| **Memory Overhead** | 0.00% | 0.00% | 0.00% | 0.00% |
| **Security** | Safe | Safe | Safe | RCE Risk |
| **Dependencies** | NumPy only | PyArrow (large) | Rust bindings | Python stdlib |
| **Best For** | Network/IPC | Dataframes | Disk storage | Python objects |
| **SIMD Aligned** | 64-byte | 64-byte | No | No |

---

## Performance Deep-Dive

Read the full story: [Breaking the Speed Limit: Optimizing Python Tensor Serialization to 5 GB/s](https://medium.com/@khushiyant/breaking-the-speed-limit-how-i-optimized-python-tensor-serialization-to-5-gb-s-f28df72ac598)

Key insights:
- Why JSON headers kill performance
- How memory alignment enables zero-copy
- Why Tenso beats Arrow for single tensors
- Real-world cost savings ($15k/month at scale)

---

## Development

```bash
# Clone repository
git clone https://github.com/Khushiyant/tenso.git
cd tenso

# Install with dev dependencies
pip install -e ".[dev]"

# Install with gpu dependencies
pip install -e ".[gpu]"

# Run tests
pytest

# Run comprehensive benchmarks
python benchmark.py all

# Quick benchmark (serialization + Arrow comparison)
python benchmark.py quick

# Benchmark with Integrity
python benchmark.py all --integrity # Require installation with [integrity] dependicies
```

---

## Requirements

- Python >= 3.10
- NumPy >= 1.20

**Optional** (for benchmarks):
- `pyarrow` - Compare with Apache Arrow
- `safetensors` - Compare with SafeTensors
- `msgpack` - Compare with MessagePack
- `psutil` - Monitor CPU/memory usage

- `xxhash` - Integrity Checks Implementation

---

## Contributing

Contributions welcome. Areas we'd love help with:

- Async support (`async def aread_stream()`)
- Compression integration (zstd, lz4)
- gRPC/FastAPI integration examples
- Rust bindings for even faster serialization
- JavaScript/WASM client for browser ML
- CUDA support for GPU-direct transfers


---

## License

MIT License - see [License](https://github.com/Khushiyant/tenso/blob/main/LICENSE) file.

---

## Citation

If you use Tenso in research, please cite:

```bibtex
@software{tenso2025,
  author = {Khushiyant},
  title = {Tenso: High-Performance Zero-Copy Tensor Protocol},
  year = {2025},
  url = {https://github.com/Khushiyant/tenso}
}
```

---

## Acknowledgments

Inspired by the need for faster ML inference infrastructure. Built with care for the ML community.

Star this repo if Tenso saved you CPU cycles.