import socket
import struct
import numpy as np
import tenso

def recv_tenso(sock):
    """Helper to read exactly one Tenso packet from a TCP stream."""
    # 1. Read the Fixed Header (8 bytes)
    header = b''
    while len(header) < 8:
        chunk = sock.recv(8 - len(header))
        if not chunk: return None
        header += chunk
        
    # 2. Parse Header to find Shape size
    magic, ver, flags, dtype_code, ndim = struct.unpack('<4sBBBB', header[0:8])
    shape_len = ndim * 4
    
    # 3. Read the Shape Block
    shape_bytes = b''
    while len(shape_bytes) < shape_len:
        chunk = sock.recv(shape_len - len(shape_bytes))
        if not chunk: raise ConnectionError("Socket closed during shape read")
        shape_bytes += chunk
        
    shape = struct.unpack(f'<{ndim}I', shape_bytes)
    
    # 4. Calculate Body Size
    # We need to map the dtype_code back to size (simple lookup)
    # For this demo, we can just use tenso.core._REV_DTYPE_MAP logic or hardcode standard sizes
    dtype_map = {1: 4, 2: 4, 3: 8, 4: 8, 5: 1, 6: 2, 7: 1, 8: 2} # approx
    item_size = dtype_map.get(dtype_code, 4)
    total_elements = np.prod(shape)
    body_len = int(total_elements * item_size)
    
    # 5. Read Body
    body = b''
    while len(body) < body_len:
        # Read in larger chunks for speed
        chunk = sock.recv(min(4096, body_len - len(body)))
        if not chunk: raise ConnectionError("Socket closed during body read")
        body += chunk
        
    # 6. Reconstruct
    return tenso.loads(header + shape_bytes + body)

# --- SERVER ---
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('0.0.0.0', 9999))
server.listen(1)

print("🚀 Tenso Inference Server Waiting...")

conn, addr = server.accept()
print(f"Connected by {addr}")

while True:
    # Wait for tensor
    try:
        tensor = recv_tenso(conn)
        if tensor is None: break
        
        print(f"Received Input: {tensor.shape} | Mean: {tensor.mean():.4f}")
        
        # Simulate Inference (dummy processing)
        result = tensor * 2 
        
        # Send result back
        print("Sending response...")
        conn.sendall(tenso.dumps(result))
        
    except Exception as e:
        print(e)
        break

conn.close()