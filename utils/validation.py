import re
from typing import Dict, List, Optional
from pydantic import BaseModel, validator, Field

class KeywordAnalysisRequest(BaseModel):
    keyword: str = Field(..., min_length=1, max_length=200)
    location: str = Field(default="Vietnam", max_length=100)
    time_horizon: str = Field(default="30d", regex=r"^\d+d$")
    budget_constraint: Optional[float] = Field(default=None, ge=0)
    competitor_focus: List[str] = Field(default_factory=list, max_items=10)
    content_assets_available: List[str] = Field(default_factory=list, max_items=20)
    
    @validator('keyword')
    def validate_keyword(cls, v):
        if not v.strip():
            raise ValueError('Keyword cannot be empty')
        # Remove potentially harmful characters
        cleaned = re.sub(r'[<>"\']', '', v.strip())
        return cleaned
    
    @validator('competitor_focus')
    def validate_competitors(cls, v):
        # Validate domain format
        domain_pattern = r'^[a-zA-Z0-9][a-zA-Z0-9-]{1,61}[a-zA-Z0-9]\.[a-zA-Z]{2,}$'
        for domain in v:
            if not re.match(domain_pattern, domain):
                raise ValueError(f'Invalid domain format: {domain}')
        return v

class ContentAnalysisRequest(BaseModel):
    content: str = Field(..., min_length=10, max_length=50000)
    keyword: str = Field(..., min_length=1, max_length=200)
    content_type: str = Field(default="article", regex=r"^(article|blog|product|landing)$")

class MonitoringRequest(BaseModel):
    keyword: str = Field(..., min_length=1, max_length=200)
    location: str = Field(default="vn", max_length=10)
    frequency: str = Field(default="hourly", regex=r"^(hourly|daily|weekly)$")

class SimulationRequest(BaseModel):
    keyword: str = Field(..., min_length=1, max_length=200)
    strategic_moves: List[Dict] = Field(..., max_items=10)
    days: int = Field(default=30, ge=1, le=365)

class DataValidator:
    """Validate and sanitize input data"""
    
    def validate_serp_data(self, data: Dict) -> Dict:
        """Validate SERP data structure"""
        if not isinstance(data, dict):
            return {'keyword': '', 'data': {}}
        
        # Ensure required fields exist
        validated = {
            'keyword': str(data.get('keyword', '')),
            'data': data.get('data', {}),
            'timestamp': data.get('timestamp', time.time())
        }
        
        # Validate data structure
        if isinstance(validated['data'], dict):
            validated['data'] = {
                'organic_results': validated['data'].get('organic_results', []),
                'featured_snippet': validated['data'].get('featured_snippet'),
                'video_results': validated['data'].get('video_results', []),
                'related_questions': validated['data'].get('related_questions', [])
            }
        
        return validated
    
    def validate_task_id(self, task_id: str) -> bool:
        """Validate Celery task ID format"""
        # UUID format validation
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        return bool(re.match(uuid_pattern, task_id, re.IGNORECASE))
    
    def sanitize_html(self, text: str) -> str:
        """Basic HTML sanitization"""
        # Remove script tags and their content
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
        # Remove other potentially harmful tags
        text = re.sub(r'<[^>]+>', '', text)
        return text.strip()

data_validator = DataValidator()