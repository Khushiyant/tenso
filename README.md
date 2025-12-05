<!-- <img width="2816" height="1536" alt="Gemini_Generated_Image_v39t46v39t46v39t" src="https://github.com/user-attachments/assets/50378dc3-6165-4b79-831d-5ebf1303cada" /> -->
<img width="2439" height="966" alt="Gemini_Generated_Image_v39t46v39t46v39t" src="https://github.com/user-attachments/assets/5ec9b225-3615-4225-82ca-68e15b7045ce" />

# Tenso

High-Performance, Zero-Copy Tensor Protocol for Python.

## Overview

Tenso is a specialized binary protocol designed for one thing: moving NumPy arrays between backends instantly.

It avoids the massive CPU overhead of standard formats (JSON, Pickle, MsgPack) by using a strict Little-Endian, 64-byte aligned memory layout. This allows for Zero-Copy deserialization, meaning the CPU doesn't have to move data—it just points to it.

### The Zero-CPU Advantage

Tenso isn't just about speed; it's about resource efficiency.

- JSON/Pickle: Parsing large arrays consumes significant CPU cycles (100% usage during load). In a high-throughput cluster, this steals resources from your actual model inference.

- Tenso: Deserialization is effectively 0% CPU. The processor simply maps the existing memory address.

## Benchmark

**Scenario:** Reading a 64MB Float32 Matrix (Typical LLM Layer) from memory.

| Format | Read Time | Write Time | Status |
| :--- | :--- | :--- | :--- |
| **Tenso** | **0.006 ms** | **5.287 ms** | **Fastest & AVX-512 Safe** |
| **Arrow** | 0.007 ms | 7.368 ms | Heavy Dependency |
| **Pickle** | 2.670 ms | 2.773 ms | Unsafe (Security Risk) |
| **Safetensors** | 2.489 ms | 7.747 ms | - |
| **MsgPack** | 2.536 ms | 10.830 ms | - |



## Installation

```bash
pip install tenso
```

## Usage

### Network

```python
import numpy as np
import tenso

# Create a tensor
data = np.random.rand(100, 100).astype(np.float32)

# Serialize to bytes
packet = tenso.dumps(data)

# Deserialize back
restored = tenso.loads(packet)
```

### File I/O
```python
# Save to disk
with open("weights.tenso", "wb") as f:
    tenso.dump(data, f)

# Load from disk
with open("weights.tenso", "rb") as f:
    loaded_data = tenso.load(f)
```

## Protocol Specification

Tenso uses a Hybrid Fixed-Header format designed for SIMD safety.

- Header (8 bytes): TNSO Magic, Version, Flags, Dtype, NDim.

- Shape Block: Variable length (NDim * 4 bytes).

- Padding: 0-63 bytes to ensure the Body starts at a 64-byte aligned address.

- Body: Raw C-contiguous memory dump.

### Tenso vs. The World

| Feature | Tenso | Pickle | Arrow | Safetensors |
| :--- | :--- | :--- | :--- | :--- |
| **Speed (Read)** | **Instant** | Slow | Instant | Fast |
| **Safety** | **Secure** | Unsafe (RCE Risk) | Secure | Secure |
| **Alignment** | **64-byte** | None | 64-byte | None |
| **Dependencies** | **NumPy Only** | Python | PyArrow (Huge) | Rust/Bindings |
| **Best For** | **Network/IPC** | Python Objects | Dataframes | Disk Storage |

Why is Tenso 1500x faster than Pickle? Standard formats must copy data from the network buffer into a new NumPy array. Tenso uses Memory Mapping: it tells NumPy to point directly at the existing buffer. No copying, no CPU cycles.

## Development

```bash
# Clone the repository
git clone https://github.com/yourusername/tenso.git
cd tenso

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run benchmarks
python benchmark.py
```

## Requirements

- Python >= 3.10
- NumPy

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
