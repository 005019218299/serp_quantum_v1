"""
Security Configuration
Fixes: Missing security configurations, environment variables
"""

import os
from typing import Dict, List
from pydantic import BaseSettings

class SecuritySettings(BaseSettings):
    # JWT Configuration
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "change-this-in-production-very-long-secret-key")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24
    
    # API Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000
    RATE_LIMIT_PER_DAY: int = 10000
    
    # Redis Configuration
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    
    # CORS Configuration
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "https://yourdomain.com"
    ]
    
    # Security Headers
    SECURITY_HEADERS: Dict[str, str] = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'",
        "Referrer-Policy": "strict-origin-when-cross-origin"
    }
    
    # Input Validation
    MAX_REQUEST_SIZE: int = 10 * 1024 * 1024  # 10MB
    MAX_KEYWORD_LENGTH: int = 200
    MAX_CONTENT_LENGTH: int = 50000
    MAX_COMPETITORS: int = 10
    
    # Circuit Breaker Configuration
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 5
    CIRCUIT_BREAKER_TIMEOUT_SECONDS: int = 300
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_RETENTION_HOURS: int = 24
    
    class Config:
        env_file = ".env"

# Global settings instance
security_settings = SecuritySettings()

# Security middleware configuration
def get_security_middleware_config():
    return {
        "allowed_hosts": ["*"],  # Configure properly in production
        "allowed_origins": security_settings.ALLOWED_ORIGINS,
        "max_request_size": security_settings.MAX_REQUEST_SIZE,
        "security_headers": security_settings.SECURITY_HEADERS
    }

# Rate limiting configuration
def get_rate_limit_config():
    return {
        "per_minute": security_settings.RATE_LIMIT_PER_MINUTE,
        "per_hour": security_settings.RATE_LIMIT_PER_HOUR,
        "per_day": security_settings.RATE_LIMIT_PER_DAY
    }

# Validation rules
VALIDATION_RULES = {
    "keyword": {
        "min_length": 1,
        "max_length": security_settings.MAX_KEYWORD_LENGTH,
        "allowed_chars": r"^[a-zA-Z0-9\s\-_.,!?]+$"
    },
    "content": {
        "min_length": 10,
        "max_length": security_settings.MAX_CONTENT_LENGTH
    },
    "domain": {
        "pattern": r"^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    },
    "task_id": {
        "pattern": r"^[a-zA-Z0-9_-]+$",
        "max_length": 100
    }
}