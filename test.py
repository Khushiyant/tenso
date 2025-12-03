import numpy as np
import tenso

# Create a tensor
data = np.random.rand(100, 100).astype(np.float32)

# Serialize to bytes (Zero-Copy)
packet = tenso.dumps(data)

# Deserialize back
restored = tenso.loads(packet)

# Save to disk
with open("model_weights.tenso", "wb") as f:
    tenso.dump(data, f)

# Load from disk
with open("model_weights.tenso", "rb") as f:
    data = tenso.load(f)