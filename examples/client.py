import socket
import numpy as np
import tenso
import time

# Connect
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('localhost', 9999))

# 1. Create a dummy image (Batch of 4)
data = np.random.rand(4, 256, 256, 3).astype(np.float32)

print(f"Sending Tensor: {data.shape} ({data.nbytes / 1024 / 1024:.2f} MB)")

t0 = time.time()
# 2. Serialize and Send
packet = tenso.dumps(data)
client.sendall(packet)

# 3. Receive Result
# (We reuse the helper logic, or just read 4096 for simplicity in this demo)
# In production, use the same recv_tenso logic here.
response_data = client.recv(len(packet) + 100) 
result = tenso.loads(response_data)

print(f"Got Result in {time.time() - t0:.4f}s")
print(f"Result Mean: {result.mean():.4f}")