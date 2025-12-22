import pytest
import numpy as np
import tenso
try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from tenso.fastapi import TensoResponse
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
def test_tenso_response():
    """Verify FastAPI integration returns correct binary stream and headers."""
    app = FastAPI()

    @app.get("/tensor")
    def get_tensor():
        data = np.ones((10, 10), dtype=np.float32)
        return TensoResponse(data, filename="test.tenso")

    client = TestClient(app)
    response = client.get("/tensor")
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    assert response.headers["x-tenso-version"] == "2"
    assert "test.tenso" in response.headers["content-disposition"]
    
    # Verify content
    restored = tenso.loads(response.content)
    assert restored.shape == (10, 10)
    assert restored.dtype == np.float32
    assert np.all(restored == 1.0)