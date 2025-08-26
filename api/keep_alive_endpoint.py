from fastapi import APIRouter
import os
from datetime import datetime

router = APIRouter()

@router.get("/metrics/system")
async def get_system_metrics():
    """System performance metrics"""
    import psutil
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "timestamp": datetime.now().isoformat()
    }

@router.get("/status/internal")
async def internal_status():
    """Internal service status check"""
    return {
        "status": "operational",
        "uptime": datetime.now().isoformat(),
        "version": "1.8.0"
    }