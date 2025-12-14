import pytest
import numpy as np
import tenso
import io
import struct

class FragmentedStream:
    """Simulates a slow network connection that yields data in tiny chunks."""
    def __init__(self, data, chunk_size=10):
        self.data = data
        self.chunk_size = chunk_size
        self.pos = 0

    def read(self, n):
        # Simulate 'read' behavior for file-like objects
        if self.pos >= len(self.data):
            return b''
        end = min(self.pos + n, self.pos + self.chunk_size)
        chunk = self.data[self.pos:end]
        self.pos += len(chunk)
        return chunk

    def recv(self, n):
        # Simulate 'recv' behavior for sockets
        return self.read(n)

def test_read_stream_perfect():
    """Test reading from a perfect, non-fragmented stream (like a file)."""
    data = np.random.rand(10, 10).astype(np.float32)
    packet = tenso.dumps(data)
    
    stream = io.BytesIO(packet)
    result = tenso.read_stream(stream)
    
    assert np.array_equal(data, result)

def test_read_stream_fragmented():
    """Test reading from a stream that arrives in tiny pieces."""
    data = np.random.rand(50, 50).astype(np.float32)
    packet = tenso.dumps(data)
    
    # Feed data 1 byte at a time
    stream = FragmentedStream(packet, chunk_size=1)
    result = tenso.read_stream(stream)
    
    assert np.array_equal(data, result)

def test_read_stream_socket_simulation():
    """Test using the recv attribute specifically."""
    data = np.array([1, 2, 3], dtype=np.int32)
    packet = tenso.dumps(data)
    
    class MockSocket:
        def __init__(self, data):
            self.stream = io.BytesIO(data)
        def recv(self, n):
            return self.stream.read(n)
            
    sock = MockSocket(packet)
    result = tenso.read_stream(sock)
    assert np.array_equal(data, result)

def test_stream_disconnect_header():
    """Test graceful handling of disconnects before/during header."""
    stream = io.BytesIO(b'')
    # Empty stream should return None (graceful close)
    assert tenso.read_stream(stream) is None
    
    # Partial header should raise specific error now
    stream = io.BytesIO(b'TNS')
    with pytest.raises(EOFError, match="Stream ended during header read"):
        tenso.read_stream(stream)

def test_stream_disconnect_body():
    """Test disconnect in the middle of the body."""
    data = np.zeros((10, 10), dtype=np.float32) # 400 bytes body
    packet = tenso.dumps(data)
    
    # Cut off the last 10 bytes
    truncated = packet[:-10]
    stream = io.BytesIO(truncated)
    
    with pytest.raises(EOFError, match="Stream ended during body read"):
        tenso.read_stream(stream)