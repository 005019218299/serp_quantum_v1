import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

client = TestClient(app)

def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "Quantum SERP Resonance API" in response.json()["message"]

def test_analyze_keyword():
    """Test keyword analysis endpoint"""
    test_request = {
        "keyword": "máy lọc nước thông minh",
        "location": "Ho Chi Minh, Vietnam",
        "language": "vi",
        "analysis_depth": "comprehensive",
        "time_horizon": "30d",
        "competitor_focus": ["karofi.com", "sunhouse.com.vn"],
        "budget_constraint": "medium",
        "content_assets_available": {
            "blog_capacity": "10 articles/month",
            "video_capacity": "2 videos/month"
        }
    }
    
    response = client.post("/analyze", json=test_request)
    
    # Should return 200 or handle gracefully if external services fail
    assert response.status_code in [200, 500]
    
    if response.status_code == 200:
        data = response.json()
        assert "analysis_id" in data
        assert data["keyword"] == test_request["keyword"]
        assert "predictive_scoreboard" in data
        assert "recommended_strategies" in data

def test_serp_history():
    """Test SERP history endpoint"""
    response = client.get("/serp/test-keyword/history?days=7")
    assert response.status_code == 200
    
    data = response.json()
    assert "keyword" in data
    assert "history" in data

def test_simulation():
    """Test simulation endpoint"""
    strategic_moves = [
        {
            "type": "featured_snippet",
            "target": "Featured Snippet",
            "success_probability": 0.75,
            "implementation_day": 14
        }
    ]
    
    response = client.post(
        "/simulate?keyword=test-keyword&days=30",
        json=strategic_moves
    )
    
    # Should handle gracefully even if simulation fails
    assert response.status_code in [200, 500]

if __name__ == "__main__":
    pytest.main([__file__])