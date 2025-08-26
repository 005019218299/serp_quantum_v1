from fastapi import APIRouter, Depends
from typing import Dict
import psutil
import time
from datetime import datetime
import os

router = APIRouter()

@router.get("/system/status")
async def get_system_status():
    """Get comprehensive system status"""
    
    # System resources
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # Process information
    process = psutil.Process()
    process_memory = process.memory_info()
    
    # Check service health
    services_health = await check_services_health()
    
    # Model status
    model_status = await check_model_status()
    
    # API keys status
    api_keys_status = check_api_keys_status()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "system": {
            "cpu_usage_percent": cpu_percent,
            "memory": {
                "total_gb": round(memory.total / 1024**3, 2),
                "used_gb": round(memory.used / 1024**3, 2),
                "available_gb": round(memory.available / 1024**3, 2),
                "usage_percent": memory.percent
            },
            "disk": {
                "total_gb": round(disk.total / 1024**3, 2),
                "used_gb": round(disk.used / 1024**3, 2),
                "free_gb": round(disk.free / 1024**3, 2),
                "usage_percent": round((disk.used / disk.total) * 100, 2)
            },
            "process": {
                "memory_mb": round(process_memory.rss / 1024**2, 2),
                "cpu_percent": process.cpu_percent(),
                "threads": process.num_threads(),
                "uptime_hours": round((time.time() - process.create_time()) / 3600, 2)
            }
        },
        "services": services_health,
        "models": model_status,
        "api_keys": api_keys_status,
        "overall_health": calculate_overall_health(services_health, model_status, api_keys_status)
    }

async def check_services_health() -> Dict:
    """Check health of external services"""
    services = {
        "database": await check_database_health(),
        "redis": await check_redis_health(),
        "serp_apis": await check_serp_apis_health()
    }
    return services

async def check_database_health() -> Dict:
    """Check database connectivity and performance"""
    try:
        from processing.feature_store import FeatureStore
        feature_store = FeatureStore()
        
        start_time = time.time()
        # Simple query to test connection
        test_data = feature_store.get_historical_data("test", days=1)
        response_time = (time.time() - start_time) * 1000
        
        return {
            "status": "healthy",
            "response_time_ms": round(response_time, 2),
            "last_check": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "last_check": datetime.now().isoformat()
        }

async def check_redis_health() -> Dict:
    """Check Redis connectivity"""
    try:
        import redis
        from config.settings import settings
        
        r = redis.from_url(settings.REDIS_URL)
        start_time = time.time()
        r.ping()
        response_time = (time.time() - start_time) * 1000
        
        info = r.info()
        
        return {
            "status": "healthy",
            "response_time_ms": round(response_time, 2),
            "memory_usage_mb": round(info.get('used_memory', 0) / 1024**2, 2),
            "connected_clients": info.get('connected_clients', 0),
            "last_check": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "last_check": datetime.now().isoformat()
        }

async def check_serp_apis_health() -> Dict:
    """Check SERP API services health"""
    try:
        from data_collection.serp_harvester import SERPHarvester
        
        harvester = SERPHarvester()
        start_time = time.time()
        
        # Test with a simple query
        result = await harvester.fetch_serp("test keyword", location="vn")
        response_time = (time.time() - start_time) * 1000
        
        success = result and 'data' in result and not result.get('error')
        
        return {
            "status": "healthy" if success else "degraded",
            "response_time_ms": round(response_time, 2),
            "sources_available": result.get('sources_count', 0) if success else 0,
            "last_check": datetime.now().isoformat(),
            "error": result.get('error') if not success else None
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "last_check": datetime.now().isoformat()
        }

async def check_model_status() -> Dict:
    """Check ML models status"""
    try:
        from prediction.temporal_model import SERPPredictor
        from prediction.competitive_model import CompetitiveResponsePredictor
        
        temporal_model = SERPPredictor()
        competitive_model = CompetitiveResponsePredictor()
        
        return {
            "temporal_model": {
                "trained": temporal_model.is_trained,
                "info": temporal_model.get_model_info() if temporal_model.is_trained else None
            },
            "competitive_model": {
                "trained": competitive_model.is_trained
            },
            "last_check": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "error": str(e),
            "last_check": datetime.now().isoformat()
        }

def check_api_keys_status() -> Dict:
    """Check API keys status"""
    try:
        from services.api_key_manager import api_key_manager
        
        serpapi_stats = api_key_manager.get_key_stats("serpapi")
        serpstack_stats = api_key_manager.get_key_stats("serpstack")
        
        return {
            "serpapi": {
                "total_keys": serpapi_stats["total"],
                "active_keys": serpapi_stats["active"],
                "expired_keys": serpapi_stats["expired"]
            },
            "serpstack": {
                "total_keys": serpstack_stats["total"],
                "active_keys": serpstack_stats["active"],
                "expired_keys": serpstack_stats["expired"]
            },
            "last_check": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "error": str(e),
            "last_check": datetime.now().isoformat()
        }

def calculate_overall_health(services: Dict, models: Dict, api_keys: Dict) -> Dict:
    """Calculate overall system health score"""
    health_score = 100
    issues = []
    
    # Check services
    for service_name, service_data in services.items():
        if service_data.get("status") == "unhealthy":
            health_score -= 30
            issues.append(f"{service_name} is unhealthy")
        elif service_data.get("status") == "degraded":
            health_score -= 15
            issues.append(f"{service_name} is degraded")
    
    # Check models
    if "error" in models:
        health_score -= 20
        issues.append("Model status check failed")
    else:
        if not models.get("temporal_model", {}).get("trained"):
            health_score -= 10
            issues.append("Temporal model not trained")
        if not models.get("competitive_model", {}).get("trained"):
            health_score -= 10
            issues.append("Competitive model not trained")
    
    # Check API keys
    if "error" in api_keys:
        health_score -= 15
        issues.append("API keys status check failed")
    else:
        if api_keys.get("serpapi", {}).get("active_keys", 0) == 0:
            health_score -= 20
            issues.append("No active SerpAPI keys")
        if api_keys.get("serpstack", {}).get("active_keys", 0) == 0:
            health_score -= 10
            issues.append("No active SerpStack keys")
    
    # Determine overall status
    if health_score >= 90:
        status = "excellent"
    elif health_score >= 70:
        status = "good"
    elif health_score >= 50:
        status = "degraded"
    else:
        status = "critical"
    
    return {
        "status": status,
        "score": max(0, health_score),
        "issues": issues,
        "recommendations": generate_recommendations(issues)
    }

def generate_recommendations(issues: List[str]) -> List[str]:
    """Generate recommendations based on issues"""
    recommendations = []
    
    for issue in issues:
        if "database" in issue.lower():
            recommendations.append("Check database connection and restart if necessary")
        elif "redis" in issue.lower():
            recommendations.append("Verify Redis service is running and accessible")
        elif "serp" in issue.lower():
            recommendations.append("Check SERP API keys and network connectivity")
        elif "model" in issue.lower():
            recommendations.append("Run model training script to train missing models")
        elif "api key" in issue.lower():
            recommendations.append("Add valid API keys to configuration")
    
    if not recommendations:
        recommendations.append("System is operating normally")
    
    return recommendations

@router.get("/system/metrics")
async def get_system_metrics():
    """Get detailed system metrics for monitoring"""
    
    # CPU metrics
    cpu_times = psutil.cpu_times()
    cpu_count = psutil.cpu_count()
    
    # Memory metrics
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    # Disk metrics
    disk_usage = psutil.disk_usage('/')
    disk_io = psutil.disk_io_counters()
    
    # Network metrics
    network_io = psutil.net_io_counters()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "cpu": {
            "count": cpu_count,
            "usage_percent": psutil.cpu_percent(interval=1),
            "times": {
                "user": cpu_times.user,
                "system": cpu_times.system,
                "idle": cpu_times.idle
            }
        },
        "memory": {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "free": memory.free,
            "percent": memory.percent
        },
        "swap": {
            "total": swap.total,
            "used": swap.used,
            "free": swap.free,
            "percent": swap.percent
        },
        "disk": {
            "usage": {
                "total": disk_usage.total,
                "used": disk_usage.used,
                "free": disk_usage.free
            },
            "io": {
                "read_bytes": disk_io.read_bytes if disk_io else 0,
                "write_bytes": disk_io.write_bytes if disk_io else 0,
                "read_count": disk_io.read_count if disk_io else 0,
                "write_count": disk_io.write_count if disk_io else 0
            }
        },
        "network": {
            "bytes_sent": network_io.bytes_sent,
            "bytes_recv": network_io.bytes_recv,
            "packets_sent": network_io.packets_sent,
            "packets_recv": network_io.packets_recv
        }
    }