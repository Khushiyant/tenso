import numpy as np
import tenso
import tempfile
import os

data = np.random.rand(100, 100).astype(np.float32)

with tempfile.NamedTemporaryFile(delete=False) as f:
    path = f.name
    # This should trigger the new dump_to_fd_rs path
    tenso.dump(data, f)

# Read back
with open(path, "rb") as f:
    restored = tenso.load(f)

assert np.allclose(data, restored)
print("Dump to FD test passed!")
os.remove(path)
