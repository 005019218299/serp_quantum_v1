#!/usr/bin/env python3
"""
Distributed Training API - REST API cho hệ thống distributed training
Endpoints để quản lý workers, jobs, và monitoring
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import asyncio
from datetime import datetime
import json

# Import services
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.master_coordinator import MasterCoordinator
from services.distributed_worker_manager import DistributedWorkerManager
from services.auto_keyword_discovery import AutoKeywordDiscovery

# Pydantic models
class TrainingRequest(BaseModel):
    user_id: str
    seed_keywords: Optional[List[str]] = None
    target_regions: Optional[List[str]] = None
    max_keywords_per_worker: Optional[int] = 50

class WorkerRegistration(BaseModel):
    region: str
    ip_address: str
    capabilities: List[str] = ["crawling", "keyword_discovery"]

class KeywordDiscoveryRequest(BaseModel):
    max_keywords: Optional[int] = 1000
    languages: Optional[List[str]] = None

# Initialize FastAPI app
app = FastAPI(
    title="Distributed Training API",
    description="API for managing distributed SERP training system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services
coordinator: Optional[MasterCoordinator] = None
worker_manager: Optional[DistributedWorkerManager] = None
keyword_discovery: Optional[AutoKeywordDiscovery] = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global coordinator, worker_manager, keyword_discovery
    
    print("🚀 Starting Distributed Training API...")
    
    # Initialize services
    coordinator = MasterCoordinator()
    await coordinator.initialize_system()
    
    worker_manager = coordinator.worker_manager
    keyword_discovery = coordinator.keyword_discovery
    
    print("✅ Distributed Training API ready!")

@app.get("/")
async def root():
    """API health check"""
    return {
        "message": "Distributed Training API",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

# Training Endpoints
@app.post("/api/training/start")
async def start_training_cycle(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start complete training cycle: Discovery → Crawling → Training"""
    
    if not coordinator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Start training cycle in background
        cycle_id = await coordinator.start_full_discovery_and_training_cycle(
            user_id=request.user_id,
            custom_seeds=request.seed_keywords
        )
        
        return {
            "success": True,
            "cycle_id": cycle_id,
            "message": "Training cycle started successfully",
            "estimated_duration_minutes": 45,
            "phases": [
                "Keyword Discovery",
                "Distributed Crawling", 
                "Data Aggregation",
                "Model Training"
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")

@app.get("/api/training/status/{cycle_id}")
async def get_training_status(cycle_id: str):
    """Get training cycle status"""
    
    if not coordinator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        status = await coordinator.get_cycle_status(cycle_id)
        
        if 'error' in status:
            raise HTTPException(status_code=404, detail=status['error'])
        
        return {
            "success": True,
            "cycle_id": cycle_id,
            "status": status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@app.get("/api/training/cycles")
async def list_training_cycles():
    """List all training cycles"""
    
    if not coordinator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        active_cycles = list(coordinator.active_jobs.keys())
        completed_cycles = list(coordinator.completed_jobs.keys())
        
        return {
            "success": True,
            "active_cycles": active_cycles,
            "completed_cycles": completed_cycles,
            "total_active": len(active_cycles),
            "total_completed": len(completed_cycles)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list cycles: {str(e)}")

# Worker Management Endpoints
@app.post("/api/workers/register")
async def register_worker(worker: WorkerRegistration):
    """Register new worker node"""
    
    if not worker_manager:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        worker_config = {
            'region': worker.region,
            'ip_address': worker.ip_address,
            'capabilities': worker.capabilities
        }
        
        worker_id = await worker_manager.register_worker(worker_config)
        
        return {
            "success": True,
            "worker_id": worker_id,
            "message": f"Worker registered successfully in region {worker.region}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register worker: {str(e)}")

@app.get("/api/workers/list")
async def list_workers():
    """List all registered workers"""
    
    if not worker_manager:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        workers_info = []
        
        for worker_id, worker in worker_manager.workers.items():
            workers_info.append({
                "worker_id": worker_id,
                "region": worker.region,
                "status": worker.status.value,
                "current_task": worker.current_task,
                "total_completed": worker.total_tasks_completed,
                "success_rate": worker.success_rate,
                "last_heartbeat": worker.last_heartbeat.isoformat()
            })
        
        return {
            "success": True,
            "workers": workers_info,
            "total_workers": len(workers_info)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list workers: {str(e)}")

@app.get("/api/workers/stats")
async def get_worker_stats():
    """Get worker system statistics"""
    
    if not worker_manager:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        stats = await worker_manager.get_system_stats()
        
        return {
            "success": True,
            "stats": stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get worker stats: {str(e)}")

# Keyword Discovery Endpoints
@app.post("/api/keywords/discover")
async def discover_keywords(request: KeywordDiscoveryRequest):
    """Discover trending keywords"""
    
    if not keyword_discovery:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        keywords = await keyword_discovery.discover_trending_keywords(
            max_keywords=request.max_keywords
        )
        
        # Filter by languages if specified
        if request.languages:
            filtered_keywords = {
                lang: kws for lang, kws in keywords.items() 
                if lang in request.languages
            }
            keywords = filtered_keywords
        
        total_keywords = sum(len(kws) for kws in keywords.values())
        
        return {
            "success": True,
            "keywords": keywords,
            "total_keywords": total_keywords,
            "languages": list(keywords.keys())
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to discover keywords: {str(e)}")

@app.get("/api/keywords/personalized/{user_id}")
async def get_personalized_keywords(user_id: str, 
                                  language: str = "en",
                                  interests: Optional[str] = None,
                                  industry: str = "general"):
    """Get personalized keywords for user"""
    
    if not keyword_discovery:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Parse interests
        user_interests = interests.split(",") if interests else []
        
        user_profile = {
            'language': language,
            'interests': user_interests,
            'industry': industry
        }
        
        personalized_keywords = await keyword_discovery.get_personalized_keywords(user_profile)
        
        return {
            "success": True,
            "user_id": user_id,
            "keywords": personalized_keywords,
            "total_keywords": len(personalized_keywords),
            "profile": user_profile
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get personalized keywords: {str(e)}")

# System Monitoring Endpoints
@app.get("/api/system/overview")
async def get_system_overview():
    """Get complete system overview"""
    
    if not coordinator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        overview = await coordinator.get_system_overview()
        
        return {
            "success": True,
            "overview": overview
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system overview: {str(e)}")

@app.get("/api/system/health")
async def system_health_check():
    """Comprehensive system health check"""
    
    health_status = {
        "api": "healthy",
        "coordinator": "unknown",
        "worker_manager": "unknown", 
        "keyword_discovery": "unknown",
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        # Check coordinator
        if coordinator:
            overview = await coordinator.get_system_overview()
            health_status["coordinator"] = "healthy"
            health_status["active_cycles"] = overview.get("active_cycles", 0)
        
        # Check worker manager
        if worker_manager:
            stats = await worker_manager.get_system_stats()
            health_status["worker_manager"] = "healthy"
            health_status["total_workers"] = stats.get("total_workers", 0)
            health_status["active_workers"] = stats.get("active_workers", 0)
        
        # Check keyword discovery
        if keyword_discovery:
            health_status["keyword_discovery"] = "healthy"
        
        # Overall health
        all_healthy = all(
            status == "healthy" 
            for key, status in health_status.items() 
            if key not in ["timestamp", "active_cycles", "total_workers", "active_workers"]
        )
        
        health_status["overall"] = "healthy" if all_healthy else "degraded"
        
        return {
            "success": True,
            "health": health_status
        }
        
    except Exception as e:
        health_status["overall"] = "unhealthy"
        health_status["error"] = str(e)
        
        return {
            "success": False,
            "health": health_status
        }

# Job Management Endpoints
@app.get("/api/jobs/crawl/{job_id}")
async def get_crawl_job_status(job_id: str):
    """Get crawl job status"""
    
    if not worker_manager:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        status = await worker_manager.get_job_status(job_id)
        
        if 'error' in status:
            raise HTTPException(status_code=404, detail=status['error'])
        
        return {
            "success": True,
            "job_id": job_id,
            "status": status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")

@app.delete("/api/jobs/cancel/{cycle_id}")
async def cancel_training_cycle(cycle_id: str):
    """Cancel training cycle"""
    
    if not coordinator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Check if cycle exists
        if cycle_id not in coordinator.active_jobs:
            raise HTTPException(status_code=404, detail="Cycle not found or already completed")
        
        # Move to completed with cancelled status
        cycle_info = coordinator.active_jobs[cycle_id]
        cycle_info['status'] = 'cancelled'
        cycle_info['cancelled_at'] = datetime.now().isoformat()
        
        coordinator.completed_jobs[cycle_id] = cycle_info
        del coordinator.active_jobs[cycle_id]
        
        return {
            "success": True,
            "message": f"Training cycle {cycle_id} cancelled successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel cycle: {str(e)}")

# Quick Start Endpoint
@app.post("/api/quick-start")
async def quick_start_demo():
    """Quick start demo with default settings"""
    
    if not coordinator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Start demo cycle
        demo_user_id = f"demo_user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        demo_seeds = ["AI tools", "web development", "digital marketing"]
        
        cycle_id = await coordinator.start_full_discovery_and_training_cycle(
            user_id=demo_user_id,
            custom_seeds=demo_seeds
        )
        
        return {
            "success": True,
            "message": "Demo training cycle started!",
            "cycle_id": cycle_id,
            "user_id": demo_user_id,
            "seed_keywords": demo_seeds,
            "estimated_completion": "30-45 minutes",
            "monitor_url": f"/api/training/status/{cycle_id}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start demo: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)