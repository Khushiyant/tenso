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
        
        This response class enables zero-copy streaming of tensors from FastAPI
        endpoints. It uses Tenso's iter_dumps generator to stream tensor data
        efficiently without loading the entire packet into memory.
        
        Usage:
            @app.get("/tensor")
            def get_tensor():
                data = np.random.rand(1000, 1000)
                return TensoResponse(data)
        
        Args:
            tensor: The numpy array to stream.
            filename: Optional filename for Content-Disposition header.
            strict: If True, raises error for non-contiguous arrays. Default False.
            check_integrity: If True, includes integrity hash. Default False.
            **kwargs: Additional arguments passed to StreamingResponse.
        """
        def __init__(self, tensor: np.ndarray, filename: str = None, 
                     strict: bool = False, check_integrity: bool = False, **kwargs):
            """
            Initialize a TensoResponse for streaming a tensor.
            
            Args:
                tensor: The numpy array to serialize and stream.
                filename: Optional filename for download headers.
                strict: If True, require C-contiguous tensors. Default False.
                check_integrity: If True, include integrity hash. Default False.
                **kwargs: Additional StreamingResponse arguments.
            """

except ImportError:
    class TensoResponse:
        def __init__(self, *args, **kwargs):
            raise ImportError("FastAPI is not installed. Run `pip install fastapi`.")