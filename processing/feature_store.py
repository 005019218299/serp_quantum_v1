from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import Dict, List, Optional
from config.settings import settings

Base = declarative_base()

class SERPFeatures(Base):
    __tablename__ = 'serp_features'
    
    id = Column(Integer, primary_key=True)
    keyword = Column(String(500), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    features = Column(JSON)
    raw_data = Column(JSON)
    predictions = Column(JSON)
    volatility_score = Column(Float)

class CompetitorData(Base):
    __tablename__ = 'competitor_data'
    
    id = Column(Integer, primary_key=True)
    keyword = Column(String(500), index=True)
    domain = Column(String(255), index=True)
    position = Column(Integer)
    title = Column(Text)
    url = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

class FeatureStore:
    def __init__(self, db_url: str = None):
        self.db_url = db_url or settings.DATABASE_URL
        self.engine = create_engine(self.db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def store_features(self, features: Dict) -> int:
        """Lưu trữ features đã xử lý vào database"""
        session = self.Session()
        try:
            # Convert all datetime objects to ISO strings
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: convert_datetime(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_datetime(item) for item in obj]
                return obj
            
            features_copy = convert_datetime(features.copy())
            
            serp_feature = SERPFeatures(
                keyword=features['keyword'],
                timestamp=features['timestamp'] if isinstance(features['timestamp'], datetime) else datetime.utcnow(),
                features=features['serp_features'],
                raw_data=features_copy
            )
            session.add(serp_feature)
            session.commit()
            return serp_feature.id
        finally:
            session.close()
    
    def get_historical_data(self, keyword: str, days: int = 30) -> List[Dict]:
        """Lấy dữ liệu lịch sử cho từ khóa"""
        session = self.Session()
        try:
            from datetime import timedelta
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            results = session.query(SERPFeatures).filter(
                SERPFeatures.keyword == keyword,
                SERPFeatures.timestamp >= cutoff_date
            ).order_by(SERPFeatures.timestamp).all()
            
            return [
                {
                    'timestamp': r.timestamp,
                    'serp_features': r.features,
                    'raw_data': r.raw_data
                }
                for r in results
            ]
        finally:
            session.close()
    
    def store_competitor_data(self, keyword: str, organic_results: List[Dict]):
        """Lưu trữ dữ liệu đối thủ cạnh tranh"""
        session = self.Session()
        try:
            for i, result in enumerate(organic_results):
                competitor = CompetitorData(
                    keyword=keyword,
                    domain=self._extract_domain(result.get('url', '')),
                    position=i + 1,
                    title=result.get('title', ''),
                    url=result.get('url', '')
                )
                session.add(competitor)
            session.commit()
        finally:
            session.close()
    
    def get_competitor_trends(self, keyword: str, domain: str, days: int = 30) -> List[Dict]:
        """Lấy xu hướng của đối thủ cụ thể"""
        session = self.Session()
        try:
            from datetime import timedelta
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            results = session.query(CompetitorData).filter(
                CompetitorData.keyword == keyword,
                CompetitorData.domain == domain,
                CompetitorData.timestamp >= cutoff_date
            ).order_by(CompetitorData.timestamp).all()
            
            return [
                {
                    'timestamp': r.timestamp,
                    'position': r.position,
                    'title': r.title
                }
                for r in results
            ]
        finally:
            session.close()
    
    def _extract_domain(self, url: str) -> str:
        """Trích xuất domain từ URL"""
        from urllib.parse import urlparse
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower().replace('www.', '')
        except:
            return ''