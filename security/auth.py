"""
Security Authentication Module
Fixes: WebSocket authentication, API key validation, rate limiting
"""

import jwt
import hashlib
import time
from typing import Optional, Dict
from fastapi import HTTPException, status, WebSocket
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime, timedelta
import redis
import os

class SecurityManager:
    def __init__(self):
        self.secret_key = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        
    def create_api_key(self, tenant_id: str) -> str:
        """Create secure API key for tenant"""
        timestamp = str(int(time.time()))
        raw_key = f"{tenant_id}:{timestamp}:{self.secret_key}"
        return hashlib.sha256(raw_key.encode()).hexdigest()
    
    def verify_api_key(self, api_key: str) -> Optional[Dict]:
        """Verify API key and return tenant info"""
        try:
            # Check in database/cache
            tenant_data = self.redis_client.hgetall(f"api_key:{api_key}")
            if not tenant_data:
                return None
            
            # Check if key is active and not expired
            if tenant_data.get('status') != 'active':
                return None
                
            return {
                'tenant_id': tenant_data.get('tenant_id'),
                'plan': tenant_data.get('plan', 'basic'),
                'rate_limit': int(tenant_data.get('rate_limit', 100))
            }
        except Exception:
            return None
    
    def create_websocket_token(self, tenant_id: str) -> str:
        """Create JWT token for WebSocket authentication"""
        payload = {
            'tenant_id': tenant_id,
            'exp': datetime.utcnow() + timedelta(hours=1),
            'iat': datetime.utcnow(),
            'type': 'websocket'
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_websocket_token(self, token: str) -> Optional[Dict]:
        """Verify WebSocket JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            if payload.get('type') != 'websocket':
                return None
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def check_rate_limit(self, tenant_id: str, endpoint: str, limit: int = 100) -> bool:
        """Check rate limiting for tenant"""
        key = f"rate_limit:{tenant_id}:{endpoint}"
        current = self.redis_client.get(key)
        
        if current is None:
            self.redis_client.setex(key, 3600, 1)  # 1 hour window
            return True
        
        if int(current) >= limit:
            return False
        
        self.redis_client.incr(key)
        return True

security_manager = SecurityManager()

class AuthenticatedHTTPBearer(HTTPBearer):
    async def __call__(self, credentials: HTTPAuthorizationCredentials = None):
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing authentication credentials"
            )
        
        tenant = security_manager.verify_api_key(credentials.credentials)
        if not tenant:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        
        return tenant

async def authenticate_websocket(websocket: WebSocket) -> Optional[Dict]:
    """Authenticate WebSocket connection"""
    try:
        # Get token from query parameters
        token = websocket.query_params.get('token')
        if not token:
            await websocket.close(code=4001, reason="Missing authentication token")
            return None
        
        tenant = security_manager.verify_websocket_token(token)
        if not tenant:
            await websocket.close(code=4001, reason="Invalid authentication token")
            return None
        
        return tenant
    except Exception:
        await websocket.close(code=4000, reason="Authentication error")
        return None

def require_rate_limit(endpoint: str, limit: int = 100):
    """Decorator for rate limiting"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            tenant = kwargs.get('tenant')
            if tenant and not security_manager.check_rate_limit(
                tenant['tenant_id'], endpoint, limit
            ):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded"
                )
            return await func(*args, **kwargs)
        return wrapper
    return decorator