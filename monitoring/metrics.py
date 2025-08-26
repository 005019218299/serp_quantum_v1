from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Response
import time
from functools import wraps

# Metrics definitions
SERP_REQUESTS = Counter('serp_requests_total', 'Total SERP requests', ['keyword', 'status'])
REQUEST_TIME = Histogram('request_duration_seconds', 'Request latency in seconds', ['endpoint'])
ACTIVE_CONNECTIONS = Gauge('websocket_connections_active', 'Active WebSocket connections')
SERP_CHANGES_DETECTED = Counter('serp_changes_detected_total', 'Total SERP changes detected')
COMPETITOR_ANALYSES = Counter('competitor_analyses_total', 'Total competitor analyses performed')

class MonitoringMiddleware:
    def __init__(self, app):
        self.app = app
        
    async def __call__(self, scope, receive, send):
        if scope['type'] == 'http':
            start_time = time.time()
            path = scope['path']
            
            async def send_wrapper(message):
                if message['type'] == 'http.response.start':
                    duration = time.time() - start_time
                    REQUEST_TIME.labels(endpoint=path).observe(duration)
                    
                    status = message.get('status', 200)
                    if 'analyze' in path:
                        keyword = path.split('/')[-1] if '/' in path else 'unknown'
                        SERP_REQUESTS.labels(keyword=keyword, status=str(status)).inc()
                        
                await send(message)
                
            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)

def track_serp_change():
    """Increment SERP change counter"""
    SERP_CHANGES_DETECTED.inc()

def track_competitor_analysis():
    """Increment competitor analysis counter"""
    COMPETITOR_ANALYSES.inc()

def track_websocket_connection(delta: int):
    """Track WebSocket connection changes"""
    if delta > 0:
        ACTIVE_CONNECTIONS.inc()
    else:
        ACTIVE_CONNECTIONS.dec()

def get_metrics():
    """Get Prometheus metrics"""
    return Response(generate_latest(), media_type="text/plain")