"""
FastAPI Integration for Tenso.
Allows zero-copy streaming of tensors from API endpoints.
"""
try:
    from fastapi.responses import StreamingResponse
    import numpy as np
    from .core import iter_dumps
    
    class TensoResponse(StreamingResponse):
        """
        FastAPI Response class for high-performance tensor streaming.
        
        Usage:
            @app.get("/tensor")
            def get_tensor():
                data = np.random.rand(1000, 1000)
                return TensoResponse(data)
        """
        def __init__(self, tensor: np.ndarray, filename: str = None, 
                     strict: bool = False, check_integrity: bool = False, **kwargs):
            """
            Initialize a TensoResponse for streaming tensor data.
            
            Creates a FastAPI StreamingResponse that efficiently streams tensor
            data using Tenso's zero-copy serialization. The response includes
            custom headers to help clients identify and handle Tenso packets.
            
            Args:
                tensor: The numpy array to stream. Must have a supported dtype.
                filename: Optional filename for Content-Disposition header.
                strict: If True, raises error for non-contiguous arrays. Default False.
                check_integrity: If True, includes integrity hash. Default False.
                **kwargs: Additional arguments passed to StreamingResponse.
            
            Raises:
                ValueError: If tensor serialization fails.
            """
            
            # Use iter_dumps generator for memory-efficient streaming
            stream = iter_dumps(tensor, strict=strict, check_integrity=check_integrity)
            
            # Set correct MIME type
            media_type = "application/octet-stream"
            
            super().__init__(stream, media_type=media_type, **kwargs)
            
            # [FIX] Ensure background task attribute exists even if super init issues occur
            if not hasattr(self, "background"):
                self.background = kwargs.get("background")

            # Add Tenso header for clients to identify
            self.headers["X-Tenso-Version"] = "2"
            self.headers["X-Tenso-Shape"] = str(tensor.shape)
            self.headers["X-Tenso-Dtype"] = str(tensor.dtype)
            
            if filename:
                self.headers["Content-Disposition"] = f'attachment; filename="{filename}"'

except ImportError:
    class TensoResponse:
        def __init__(self, *args, **kwargs):
            raise ImportError("FastAPI is not installed. Run `pip install fastapi`.")