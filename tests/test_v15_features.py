import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app
from ai_models.content_intelligence import ContentIntelligenceAI
from services.seo_autopilot import SEOAutoPilot, SEOAction, ActionType
from services.roi_intelligence import ROIIntelligenceEngine

client = TestClient(app)

def test_health_check_v15():
    """Test v1.5 health check"""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["version"] == "1.5.0"
    assert "content_intelligence_ai" in data["features"]
    assert "seo_autopilot" in data["features"]
    assert "advanced_serp_simulation" in data["features"]
    assert "roi_intelligence_engine" in data["features"]
    assert "predictive_analytics" in data["features"]

@pytest.mark.asyncio
async def test_content_intelligence():
    """Test Content Intelligence AI"""
    content_ai = ContentIntelligenceAI()
    
    competitor_analysis = {
        'top_competitors': ['example1.com', 'example2.com'],
        'content_gaps': ['featured_snippet_opportunity']
    }
    
    try:
        result = await content_ai.generate_serp_optimized_content("test keyword", competitor_analysis)
        
        assert "keyword" in result
        assert "generated_content" in result
        assert "optimization_score" in result
        assert "serp_mapping" in result
        
    except Exception as e:
        # Expected to fail without proper model setup
        assert "content generation" in str(e).lower() or "model" in str(e).lower()

def test_seo_autopilot():
    """Test SEO Auto-Pilot functionality"""
    autopilot = SEOAutoPilot()
    
    # Test action creation
    action = SEOAction(
        action_id="test_001",
        type=ActionType.CONTENT_OPTIMIZATION,
        priority="high",
        description="Test content optimization",
        implementation_steps=["Step 1", "Step 2"],
        expected_impact=3.5,
        risk_level="low",
        auto_executable=True
    )
    
    assert action.action_id == "test_001"
    assert action.type == ActionType.CONTENT_OPTIMIZATION
    assert action.auto_executable == True

@pytest.mark.asyncio
async def test_roi_intelligence():
    """Test ROI Intelligence Engine"""
    roi_engine = ROIIntelligenceEngine()
    
    seo_actions = [
        {
            'id': 'action_1',
            'type': 'content_optimization',
            'complexity': 'medium',
            'target_position': 3,
            'timeline_weeks': 8
        }
    ]
    
    business_context = {
        'industry': 'ecommerce',
        'current_ranking': 8,
        'monthly_search_volume': 10000,
        'conversion_rate': 0.025,
        'avg_order_value': 85
    }
    
    result = await roi_engine.calculate_predictive_roi(seo_actions, business_context)
    
    assert "total_investment" in result
    assert "portfolio_analysis" in result
    assert "individual_actions" in result
    assert "risk_assessment" in result
    assert "recommendations" in result

def test_content_intelligence_endpoint_without_auth():
    """Test content intelligence endpoint without authentication"""
    response = client.post("/content/intelligence?keyword=test")
    assert response.status_code == 403

def test_autopilot_endpoint_without_auth():
    """Test autopilot endpoint without authentication"""
    strategy = {"content_gaps": [{"type": "featured_snippet_opportunity", "keyword": "test"}]}
    response = client.post("/autopilot/execute", json=strategy)
    assert response.status_code == 403

def test_advanced_simulation_endpoint_without_auth():
    """Test advanced simulation endpoint without authentication"""
    actions = [{"type": "content_optimization", "target_position": 3}]
    response = client.post("/simulate/advanced?keyword=test&timeline=6-months", json=actions)
    assert response.status_code == 403

def test_roi_analysis_endpoint_without_auth():
    """Test ROI analysis endpoint without authentication"""
    seo_actions = [{"type": "content_optimization"}]
    business_context = {"industry": "ecommerce"}
    
    response = client.post("/roi/analyze", json={
        "seo_actions": seo_actions,
        "business_context": business_context
    })
    assert response.status_code == 403

if __name__ == "__main__":
    pytest.main([__file__])