import pytest
import numpy as np
import tenso
import asyncio
from unittest.mock import MagicMock, AsyncMock

# Check if async_core is available
try:
    from tenso.async_core import aread_stream, awrite_stream
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False

@pytest.mark.skipif(not ASYNC_AVAILABLE, reason="Async core not implemented")
@pytest.mark.asyncio
async def test_aread_stream_success():
    """Test successful async read."""
    data = np.random.rand(20, 20).astype(np.float32)
    packet = tenso.dumps(data)
    
    reader = asyncio.StreamReader()
    reader.feed_data(packet)
    reader.feed_eof()
    
    result = await aread_stream(reader)
    assert np.array_equal(data, result)

@pytest.mark.skipif(not ASYNC_AVAILABLE, reason="Async core not implemented")
@pytest.mark.asyncio
async def test_awrite_stream_success():
    """Test successful async write."""
    data = np.random.rand(10, 10).astype(np.float32)
    
    # Mock StreamWriter
    transport = MagicMock()
    protocol = MagicMock()
    
    # [FIX] Get the running loop to satisfy StreamWriter's internal requirements
    loop = asyncio.get_running_loop()
    writer = asyncio.StreamWriter(transport, protocol, None, loop)
    
    # Mock drain to be awaitable
    writer.drain = AsyncMock()
    
    # Capture writes
    written_chunks = []
    def capture_write(chunk):
        written_chunks.append(chunk)
    writer.write = capture_write
    
    await awrite_stream(data, writer)
    
    # Reassemble and check
    full_packet = b''.join(written_chunks)
    restored = tenso.loads(full_packet)
    
    assert np.array_equal(data, restored)
    assert writer.drain.called

@pytest.mark.skipif(not ASYNC_AVAILABLE, reason="Async core not implemented")
@pytest.mark.asyncio
async def test_aread_stream_padding():
    """Test async read with padding."""
    data = np.array([123], dtype=np.uint8)
    packet = tenso.dumps(data)
    
    reader = asyncio.StreamReader()
    reader.feed_data(packet)
    reader.feed_data(b'NEXT')
    reader.feed_eof()
    
    result = await aread_stream(reader)
    assert np.array_equal(data, result)
    
    remainder = await reader.read()
    assert remainder == b'NEXT'

@pytest.mark.skipif(not ASYNC_AVAILABLE, reason="Async core not implemented")
@pytest.mark.asyncio
async def test_aread_stream_incomplete():
    """Test async read disconnects."""
    data = np.zeros((10, 10), dtype=np.float32)
    packet = tenso.dumps(data)
    
    # Incomplete Header
    reader = asyncio.StreamReader()
    reader.feed_data(packet[:4])
    reader.feed_eof()
    
    with pytest.raises(asyncio.IncompleteReadError):
        await aread_stream(reader)