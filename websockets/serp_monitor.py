from fastapi import WebSocket, WebSocketDisconnect
import json
import asyncio
from typing import List, Dict
from datetime import datetime
from services.serp_change_detector import SERPChangeDetector
from redis import Redis
from config.settings import settings

class SERPMonitorManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.detector = SERPChangeDetector()
        self.redis = Redis.from_url(settings.REDIS_URL)
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
    async def broadcast_change(self, keyword: str, change_data: Dict):
        """Broadcast SERP changes to all connected clients"""
        message = json.dumps({
            "type": "serp_change",
            "keyword": keyword,
            "data": change_data,
            "timestamp": datetime.now().isoformat()
        })
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn)
    
    async def start_monitoring(self):
        """Start Redis subscriber for real-time alerts"""
        pubsub = self.redis.pubsub()
        pubsub.subscribe("serp_changes")
        
        async for message in pubsub.listen():
            if message['type'] == 'message':
                data = json.loads(message['data'])
                await self.broadcast_change(data['keyword'], data)

monitor_manager = SERPMonitorManager()