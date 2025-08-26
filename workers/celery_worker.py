from celery import Celery
from config.settings import settings
import asyncio
from services.serp_change_detector import MasterSERPChangeDetector
from ai_models.competitor_analyzer import MasterCompetitorIntelligence
from services.device_gap_analyzer import DeviceGapAnalyzer

# Initialize Celery
celery_app = Celery(
    'quantum_serp',
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

@celery_app.task
def monitor_keyword_changes(keyword: str, location: str):
    """Background task to monitor keyword changes"""
    detector = MasterSERPChangeDetector()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            detector.monitor_keyword(keyword, location)
        )
        return result
    finally:
        loop.close()

@celery_app.task
def analyze_competitors(competitors: list, keyword: str):
    """Background task for competitor analysis"""
    analyzer = MasterCompetitorIntelligence()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            analyzer.analyze_competitor_patterns(competitors, keyword)
        )
        return result
    finally:
        loop.close()

@celery_app.task
def analyze_device_opportunities(keyword: str, location: str):
    """Background task for device gap analysis"""
    analyzer = DeviceGapAnalyzer()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            analyzer.analyze_device_opportunities(keyword, location)
        )
        return result
    finally:
        loop.close()

@celery_app.task
def periodic_serp_monitoring():
    """Periodic task to monitor all tracked keywords"""
    # This would fetch all tracked keywords from database
    # and schedule monitoring tasks for each
    tracked_keywords = [
        {"keyword": "máy lọc nước thông minh", "location": "vn"},
        # Add more from database
    ]
    
    for item in tracked_keywords:
        monitor_keyword_changes.delay(item["keyword"], item["location"])
    
    return f"Scheduled monitoring for {len(tracked_keywords)} keywords"

# Periodic task configuration
celery_app.conf.beat_schedule = {
    'monitor-serp-changes': {
        'task': 'workers.celery_worker.periodic_serp_monitoring',
        'schedule': 3600.0,  # Every hour
    },
}

if __name__ == '__main__':
    celery_app.start()