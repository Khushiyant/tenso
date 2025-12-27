import pytest
import numpy as np
import tenso
import io


class FragmentedStream:
    """Simulates a slow network connection that yields data in tiny chunks."""

    def __init__(self, data, chunk_size=10):
        self.data = data
        self.chunk_size = chunk_size
        self.pos = 0

    def read(self, n):
        if self.pos >= len(self.data):
            return b""
        end = min(self.pos + n, self.pos + self.chunk_size)
        chunk = self.data[self.pos : end]
        self.pos += len(chunk)
        return chunk

    def readinto(self, b):
        data = self.read(len(b))
        if not data:
            return 0
        b[: len(data)] = data
        return len(data)


class MockSocket:
    """Simulates a socket with recv_into (optimization path) and recv."""

    def __init__(self, data):
        self.stream = io.BytesIO(data)

    def recv(self, n):
        return self.stream.read(n)

    def recv_into(self, b):
        return self.stream.readinto(b)


def test_read_stream_perfect():
    """Test reading from a perfect, non-fragmented stream."""
    data = np.random.rand(10, 10).astype(np.float32)
    packet = tenso.dumps(data)

    stream = io.BytesIO(packet)
    result = tenso.read_stream(stream)

    assert np.array_equal(data, result)


def test_read_stream_fragmented():
    """Test reading from a stream that arrives in tiny pieces."""
    data = np.random.rand(50, 50).astype(np.float32)
    packet = tenso.dumps(data)

    stream = FragmentedStream(packet, chunk_size=1)
    result = tenso.read_stream(stream)

    assert np.array_equal(data, result)


def test_read_stream_socket_simulation():
    """Test using the recv/recv_into attributes."""
    data = np.array([1, 2, 3], dtype=np.int32)
    packet = tenso.dumps(data)

    sock = MockSocket(packet)
    result = tenso.read_stream(sock)
    assert np.array_equal(data, result)


def test_padding_consumption():
    """Verify that padding bytes are actually consumed from the stream."""
    data = np.array([1], dtype=np.int8)
    packet = tenso.dumps(data)

    # [FIX] Cast memoryview to bytes before concatenation
    stream = io.BytesIO(bytes(packet) + b"EXTRA_DATA")

    result = tenso.read_stream(stream)
    assert np.array_equal(data, result)

    remainder = stream.read()
    assert remainder == b"EXTRA_DATA"


def test_stream_disconnect_header():
    """Test graceful handling of disconnects."""
    stream = io.BytesIO(b"")
    assert tenso.read_stream(stream) is None

    stream = io.BytesIO(b"TNS")
    with pytest.raises(EOFError, match="Stream ended during header read"):
        tenso.read_stream(stream)


def test_stream_disconnect_body():
    """Test disconnect in the middle of the body."""
    data = np.zeros((10, 10), dtype=np.float32)
    packet = tenso.dumps(data)

    # Cut off the last byte
    truncated = packet[:-1]
    stream = io.BytesIO(truncated)

    with pytest.raises(EOFError, match="Stream ended during body read"):
        tenso.read_stream(stream)
