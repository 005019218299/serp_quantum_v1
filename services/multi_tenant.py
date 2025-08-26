from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime, timedelta
from typing import Dict, Optional
import uuid
import hashlib

Base = declarative_base()

class Tenant(Base):
    __tablename__ = 'tenants'
    
    id = Column(Integer, primary_key=True)
    tenant_id = Column(String(36), unique=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    api_key = Column(String(64), unique=True)
    plan = Column(String(50), default='basic')  # basic, pro, enterprise
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    settings = Column(JSON, default={})
    
    # Relationships
    usage_records = relationship("UsageRecord", back_populates="tenant")
    rate_limits = relationship("RateLimit", back_populates="tenant")

class UsageRecord(Base):
    __tablename__ = 'usage_records'
    
    id = Column(Integer, primary_key=True)
    tenant_id = Column(String(36), ForeignKey('tenants.tenant_id'))
    endpoint = Column(String(100))
    requests_count = Column(Integer, default=0)
    date = Column(DateTime, default=datetime.utcnow)
    
    tenant = relationship("Tenant", back_populates="usage_records")

class RateLimit(Base):
    __tablename__ = 'rate_limits'
    
    id = Column(Integer, primary_key=True)
    tenant_id = Column(String(36), ForeignKey('tenants.tenant_id'))
    endpoint = Column(String(100))
    requests_per_minute = Column(Integer)
    requests_per_day = Column(Integer)
    
    tenant = relationship("Tenant", back_populates="rate_limits")

class MultiTenantManager:
    def __init__(self, db_session):
        self.db = db_session
        self.plan_limits = {
            'basic': {'rpm': 10, 'rpd': 1000, 'keywords': 5},
            'pro': {'rpm': 60, 'rpd': 10000, 'keywords': 50},
            'enterprise': {'rpm': 300, 'rpd': 100000, 'keywords': 500}
        }
    
    def create_tenant(self, name: str, email: str, plan: str = 'basic') -> Dict:
        """Create new tenant"""
        api_key = self._generate_api_key(email)
        
        tenant = Tenant(
            name=name,
            email=email,
            api_key=api_key,
            plan=plan
        )
        
        self.db.add(tenant)
        self.db.commit()
        
        # Create default rate limits
        self._setup_rate_limits(tenant.tenant_id, plan)
        
        return {
            'tenant_id': tenant.tenant_id,
            'api_key': api_key,
            'plan': plan,
            'limits': self.plan_limits[plan]
        }
    
    def authenticate_tenant(self, api_key: str) -> Optional[Tenant]:
        """Authenticate tenant by API key"""
        return self.db.query(Tenant).filter(
            Tenant.api_key == api_key,
            Tenant.is_active == True
        ).first()
    
    def check_rate_limit(self, tenant_id: str, endpoint: str) -> Dict:
        """Check if tenant has exceeded rate limits"""
        tenant = self.db.query(Tenant).filter(Tenant.tenant_id == tenant_id).first()
        if not tenant:
            return {'allowed': False, 'reason': 'Invalid tenant'}
        
        plan_limits = self.plan_limits.get(tenant.plan, self.plan_limits['basic'])
        
        # Check per-minute limit
        minute_ago = datetime.utcnow() - timedelta(minutes=1)
        recent_requests = self.db.query(UsageRecord).filter(
            UsageRecord.tenant_id == tenant_id,
            UsageRecord.endpoint == endpoint,
            UsageRecord.date >= minute_ago
        ).count()
        
        if recent_requests >= plan_limits['rpm']:
            return {'allowed': False, 'reason': 'Rate limit exceeded (per minute)'}
        
        # Check per-day limit
        day_ago = datetime.utcnow() - timedelta(days=1)
        daily_requests = self.db.query(UsageRecord).filter(
            UsageRecord.tenant_id == tenant_id,
            UsageRecord.endpoint == endpoint,
            UsageRecord.date >= day_ago
        ).count()
        
        if daily_requests >= plan_limits['rpd']:
            return {'allowed': False, 'reason': 'Daily limit exceeded'}
        
        return {'allowed': True, 'remaining_rpm': plan_limits['rpm'] - recent_requests}
    
    def record_usage(self, tenant_id: str, endpoint: str):
        """Record API usage"""
        usage = UsageRecord(
            tenant_id=tenant_id,
            endpoint=endpoint,
            requests_count=1
        )
        self.db.add(usage)
        self.db.commit()
    
    def get_usage_analytics(self, tenant_id: str, days: int = 30) -> Dict:
        """Get usage analytics for tenant"""
        start_date = datetime.utcnow() - timedelta(days=days)
        
        usage_data = self.db.query(UsageRecord).filter(
            UsageRecord.tenant_id == tenant_id,
            UsageRecord.date >= start_date
        ).all()
        
        analytics = {
            'total_requests': len(usage_data),
            'endpoints_used': {},
            'daily_usage': {},
            'current_plan': self._get_tenant_plan(tenant_id)
        }
        
        for record in usage_data:
            # Endpoint usage
            endpoint = record.endpoint
            analytics['endpoints_used'][endpoint] = analytics['endpoints_used'].get(endpoint, 0) + 1
            
            # Daily usage
            date_key = record.date.strftime('%Y-%m-%d')
            analytics['daily_usage'][date_key] = analytics['daily_usage'].get(date_key, 0) + 1
        
        return analytics
    
    def upgrade_plan(self, tenant_id: str, new_plan: str) -> Dict:
        """Upgrade tenant plan"""
        if new_plan not in self.plan_limits:
            return {'success': False, 'error': 'Invalid plan'}
        
        tenant = self.db.query(Tenant).filter(Tenant.tenant_id == tenant_id).first()
        if not tenant:
            return {'success': False, 'error': 'Tenant not found'}
        
        old_plan = tenant.plan
        tenant.plan = new_plan
        self.db.commit()
        
        # Update rate limits
        self._setup_rate_limits(tenant_id, new_plan)
        
        return {
            'success': True,
            'old_plan': old_plan,
            'new_plan': new_plan,
            'new_limits': self.plan_limits[new_plan]
        }
    
    def _generate_api_key(self, email: str) -> str:
        """Generate unique API key"""
        timestamp = str(datetime.utcnow().timestamp())
        raw_key = f"{email}:{timestamp}:{uuid.uuid4()}"
        return hashlib.sha256(raw_key.encode()).hexdigest()
    
    def _setup_rate_limits(self, tenant_id: str, plan: str):
        """Setup rate limits for tenant"""
        limits = self.plan_limits[plan]
        
        # Remove existing limits
        self.db.query(RateLimit).filter(RateLimit.tenant_id == tenant_id).delete()
        
        # Add new limits for common endpoints
        endpoints = ['/analyze', '/monitor/start', '/analyze/competitors', '/analyze/devices']
        
        for endpoint in endpoints:
            rate_limit = RateLimit(
                tenant_id=tenant_id,
                endpoint=endpoint,
                requests_per_minute=limits['rpm'],
                requests_per_day=limits['rpd']
            )
            self.db.add(rate_limit)
        
        self.db.commit()
    
    def _get_tenant_plan(self, tenant_id: str) -> str:
        """Get tenant's current plan"""
        tenant = self.db.query(Tenant).filter(Tenant.tenant_id == tenant_id).first()
        return tenant.plan if tenant else 'basic'