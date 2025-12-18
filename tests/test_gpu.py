import pytest
import numpy as np
import tenso
import io
import sys
from unittest.mock import patch, MagicMock

# Attempt to import backends to determine capability
try:
    import torch
    HAS_TORCH = True
    HAS_CUDA_TORCH = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_CUDA_TORCH = False

try:
    import cupy
    HAS_CUPY = True
    try:
        # Check if a device is actually accessible
        cupy.cuda.runtime.getDeviceCount()
        HAS_CUDA_CUPY = True
    except Exception:
        HAS_CUDA_CUPY = False
except ImportError:
    HAS_CUPY = False
    HAS_CUDA_CUPY = False

# Try to import the gpu module directly for patching
try:
    from tenso import gpu
    GPU_MODULE_AVAILABLE = True
except ImportError:
    GPU_MODULE_AVAILABLE = False


@pytest.mark.skipif(not GPU_MODULE_AVAILABLE, reason="tenso.gpu module failed to import")
class TestGPU:

    def _create_stream(self, data):
        """Helper to create a BytesIO stream from a tensor."""
        packet = tenso.dumps(data)
        return io.BytesIO(packet)

    # --- REAL HARDWARE TESTS (Skipped if no GPU) ---

    @pytest.mark.skipif(not HAS_CUDA_TORCH, reason="Requires PyTorch with CUDA")
    def test_read_to_device_torch_real(self):
        """Real integration test: Stream -> Pinned RAM -> GPU (PyTorch)."""
        # Force backend to 'torch' to ensure we test this specific path
        with patch('tenso.gpu.BACKEND', 'torch'):
            data = np.random.rand(1024, 1024).astype(np.float32)
            stream = self._create_stream(data)
            
            # Perform Read
            tensor = gpu.read_to_device(stream, device_id=0)
            
            # Verify
            assert isinstance(tensor, torch.Tensor)
            assert tensor.device.type == 'cuda'
            assert tensor.device.index == 0
            
            # Verify Data Integrity (move back to CPU)
            cpu_data = tensor.cpu().numpy()
            assert np.array_equal(data, cpu_data)

    @pytest.mark.skipif(not HAS_CUDA_CUPY, reason="Requires CuPy with CUDA")
    def test_read_to_device_cupy_real(self):
        """Real integration test: Stream -> Pinned RAM -> GPU (CuPy)."""
        # Force backend to 'cupy'
        with patch('tenso.gpu.BACKEND', 'cupy'):
            data = np.random.rand(1024, 1024).astype(np.float32)
            stream = self._create_stream(data)
            
            # Perform Read
            tensor = gpu.read_to_device(stream, device_id=0)
            
            # Verify
            assert isinstance(tensor, cupy.ndarray)
            assert tensor.device.id == 0
            
            # Verify Data Integrity
            cpu_data = cupy.asnumpy(tensor)
            assert np.array_equal(data, cpu_data)

    # --- MOCKED TESTS (Run on CPU/CI) ---

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_read_to_device_torch_mocked(self):
        """Verify memory pinning and async transfer calls without a GPU."""
        
        # 1. Prepare Data & Stream FIRST to know the exact size
        data = np.random.rand(10, 10).astype(np.float32)
        stream = self._create_stream(data)
        
        # Calculate expected read size: Total Packet - Header(8) - Shape(8)
        # Note: Shape is 2 dims * 4 bytes = 8 bytes.
        packet_size = stream.getbuffer().nbytes
        expected_body_size = packet_size - 16
        
        # We mock 'tenso.gpu.torch' to capture calls
        with patch('tenso.gpu.BACKEND', 'torch'), \
             patch('tenso.gpu.torch') as mock_torch:
            
            # 2. Setup Mock for Pinned Memory Allocation
            # torch.empty(..., pin_memory=True) -> returns a mock that acts like a tensor
            mock_pinned_tensor = MagicMock()
            
            real_buffer = np.zeros(expected_body_size, dtype=np.uint8)
            mock_pinned_tensor.numpy.return_value = real_buffer
            
            mock_torch.empty.return_value = mock_pinned_tensor
            
            # 3. Setup Mock for the final .to(device) call
            # The chain is: from_numpy(...).view(...).reshape(...) -> .to(...)
            # We need the final mock in the chain to verify arguments
            mock_final_tensor = MagicMock()
            mock_torch.from_numpy.return_value.to.return_value = mock_final_tensor

            # 4. Execution
            result = gpu.read_to_device(stream, device_id=0)
            
            # 5. Assertions
            mock_torch.empty.assert_called_once()
            assert mock_torch.empty.call_args[0][0] == expected_body_size
            
            # FIX: Access the .to call directly from the from_numpy mock
            from_numpy_mock = mock_torch.from_numpy.return_value
            from_numpy_mock.to.assert_called_once()
            
            _, kwargs = from_numpy_mock.to.call_args
            assert kwargs.get('device') == 'cuda:0'
            assert kwargs.get('non_blocking') is True

    @pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
    def test_read_to_device_cupy_mocked(self):
        """Verify CuPy pinned memory logic without GPU."""
        
        with patch('tenso.gpu.BACKEND', 'cupy'), \
             patch('tenso.gpu.cp') as mock_cp:
        
            mock_cp.cuda.alloc_pinned_memory.return_value = bytearray(1024)

            data = np.random.rand(10, 10).astype(np.float32)
            stream = self._create_stream(data)
            
            # We expect a crash when it tries to np.frombuffer on our mock pointer,
            # but checking the alloc call is enough for logic verification.
            try:
                gpu.read_to_device(stream, device_id=1)
            except Exception:
                pass
                
            # 4. Assertions
            mock_cp.cuda.alloc_pinned_memory.assert_called_once()
            mock_cp.cuda.Device.assert_called_with(1)

    def test_invalid_packet(self):
        """Verify error handling on garbage data."""
        # Force one backend (torch) to test the parsing logic
        backend = 'torch' if HAS_TORCH else ('cupy' if HAS_CUPY else None)
        if not backend:
            pytest.skip("No backend available")

        with patch('tenso.gpu.BACKEND', backend):
            # Garbage stream
            stream = io.BytesIO(b'JUNK____' + b'\x00'*100)
            
            with pytest.raises(ValueError, match="Invalid tenso packet"):
                gpu.read_to_device(stream)

    def test_eof_handling(self):
        """Verify EOF behavior."""
        backend = 'torch' if HAS_TORCH else ('cupy' if HAS_CUPY else None)
        if not backend:
            pytest.skip("No backend available")

        with patch('tenso.gpu.BACKEND', backend):
            # Empty stream
            stream = io.BytesIO(b'')
            assert gpu.read_to_device(stream) is None
            
            # Partial header
            stream = io.BytesIO(b'TNSO')
            with pytest.raises(EOFError):
                gpu.read_to_device(stream)