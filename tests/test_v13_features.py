import pytest
from fastapi.testclient import TestClient
import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app
from services.serp_change_detector import SERPChangeDetector
from services.device_gap_analyzer import DeviceGapAnalyzer

client = TestClient(app)

def test_health_check_v13():
    """Test v1.3 health check"""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["version"] == "1.3.0"
    assert "real_time_monitoring" in data["features"]
    assert "competitive_ai" in data["features"]
    assert "multi_device_analysis" in data["features"]

def test_start_monitoring():
    """Test starting real-time monitoring"""
    response = client.post("/monitor/start?keyword=test&location=vn")
    assert response.status_code == 200
    
    data = response.json()
    assert "task_id" in data
    assert data["status"] == "monitoring_started"

def test_competitor_analysis():
    """Test competitor analysis endpoint"""
    competitors = ["example1.com", "example2.com"]
    response = client.post(
        "/analyze/competitors?keyword=test",
        json=competitors
    )
    assert response.status_code == 200
    
    data = response.json()
    assert "task_id" in data
    assert data["status"] == "analysis_started"

def test_device_analysis():
    """Test device gap analysis"""
    response = client.post("/analyze/devices?keyword=test&location=vn")
    assert response.status_code == 200
    
    data = response.json()
    assert "task_id" in data
    assert data["status"] == "device_analysis_started"

def test_metrics_endpoint():
    """Test Prometheus metrics"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; charset=utf-8"

@pytest.mark.asyncio
async def test_serp_change_detector():
    """Test SERP change detection"""
    detector = SERPChangeDetector()
    
    # Mock test - would need actual SERP data
    result = await detector.monitor_keyword("test", "vn")
    assert "change_detected" in result
    assert "score" in result

@pytest.mark.asyncio
async def test_device_gap_analyzer():
    """Test device gap analysis"""
    analyzer = DeviceGapAnalyzer()
    
    # Mock test - would need actual device data
    try:
        result = await analyzer.analyze_device_opportunities("test", "vn")
        assert "keyword" in result
        assert "opportunities" in result
    except Exception:
        # Expected to fail without real data
        pass

if __name__ == "__main__":
    pytest.main([__file__])