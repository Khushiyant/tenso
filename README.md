<!-- <img width="2816" height="1536" alt="Gemini_Generated_Image_v39t46v39t46v39t" src="https://github.com/user-attachments/assets/50378dc3-6165-4b79-831d-5ebf1303cada" /> -->
<img width="2439" height="966" alt="Gemini_Generated_Image_v39t46v39t46v39t" src="https://github.com/user-attachments/assets/5ec9b225-3615-4225-82ca-68e15b7045ce" />

# Tenso

High-performance zero-copy tensor serialization for NumPy arrays.

## Overview

Tenso is a lightweight Python library that provides fast, efficient serialization and deserialization of NumPy arrays. It uses a custom binary protocol designed for zero-copy operations, making it significantly faster and more space-efficient than JSON or pickle for numerical data.

## Features

- **Zero-copy serialization**: Direct memory dumps for maximum performance
- **Compact binary format**: Up to 90% smaller than JSON serialization
- **Simple API**: Four functions for all your needs: `dumps`, `loads`, `dump`, `load`
- **Type-safe**: Supports common NumPy dtypes (float32, float64, int32, int64)

## Installation

```bash
pip install tenso
```

## Quick Start

```python
import numpy as np
import tenso

# Create a tensor
data = np.random.rand(100, 100).astype(np.float32)

# Serialize to bytes
packet = tenso.dumps(data)

# Deserialize back
restored = tenso.loads(packet)

# Save to disk
with open("weights.tenso", "wb") as f:
    tenso.dump(data, f)

# Load from disk
with open("weights.tenso", "rb") as f:
    loaded_data = tenso.load(f)
```

## Protocol Specification

The Tenso binary format consists of three parts:

1. **Header (8 bytes)**:
   - Magic bytes: `TNSO` (4 bytes)
   - Version: 1 byte
   - Flags: 1 byte (reserved)
   - Dtype code: 1 byte
   - Number of dimensions: 1 byte

2. **Shape Block (variable)**:
   - N × uint32 values representing array dimensions

3. **Data Block (variable)**:
   - Raw memory dump of the array data


## Use Cases

- **Model checkpoint serialization**: Save and load neural network weights
- **Inter-process communication**: Share tensors between Python processes
- **Data pipelines**: Efficient storage and transfer of numerical data
- **Caching**: Store computed tensors for faster retrieval
- **API payloads**: Send tensors over network with minimal overhead

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
