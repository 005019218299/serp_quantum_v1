#!/usr/bin/env python3
"""
Distributed System Launcher - Khởi chạy toàn bộ hệ thống distributed training
"""

import asyncio
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import API
from api.distributed_training_api import app as api_app

# Create main app
app = FastAPI(title="Distributed Training System")

# Mount API
app.mount("/api", api_app)

# Serve dashboard
@app.get("/")
async def dashboard():
    """Serve dashboard HTML"""
    dashboard_path = Path(__file__).parent / "templates" / "distributed_dashboard.html"
    return FileResponse(dashboard_path)

@app.get("/dashboard")
async def dashboard_redirect():
    """Redirect to dashboard"""
    dashboard_path = Path(__file__).parent / "templates" / "distributed_dashboard.html"
    return FileResponse(dashboard_path)

# System info endpoint
@app.get("/info")
async def system_info():
    """Get system information"""
    return {
        "system": "Distributed SERP Training System",
        "version": "1.0.0",
        "components": [
            "Master Coordinator",
            "Distributed Worker Manager", 
            "Auto Keyword Discovery",
            "Global Training Orchestrator"
        ],
        "endpoints": {
            "dashboard": "/",
            "api_docs": "/api/docs",
            "health": "/api/system/health",
            "start_training": "/api/training/start"
        }
    }

async def demo_system():
    """Run demo of the distributed system"""
    
    print("🌟 DISTRIBUTED TRAINING SYSTEM DEMO")
    print("=" * 50)
    
    # Import services
    from services.master_coordinator import MasterCoordinator
    
    # Initialize coordinator
    coordinator = MasterCoordinator()
    await coordinator.initialize_system()
    
    print("\n✅ System initialized successfully!")
    print("\n📊 System Overview:")
    
    # Get initial stats
    overview = await coordinator.get_system_overview()
    print(f"   - Total Workers: {overview['worker_stats']['total_workers']}")
    print(f"   - Active Workers: {overview['worker_stats']['active_workers']}")
    print(f"   - Target Regions: {', '.join(overview['target_regions'])}")
    
    print("\n🚀 Starting demo training cycle...")
    
    # Start demo cycle
    cycle_id = await coordinator.start_full_discovery_and_training_cycle(
        user_id="demo_user_001",
        custom_seeds=["AI tools", "machine learning", "web development"]
    )
    
    print(f"   - Cycle ID: {cycle_id}")
    print("   - Phases: Discovery → Crawling → Aggregation → Training")
    
    # Monitor for a bit
    print("\n👁️ Monitoring system (30 seconds)...")
    for i in range(6):
        await asyncio.sleep(5)
        
        # Get updated stats
        overview = await coordinator.get_system_overview()
        cycle_status = await coordinator.get_cycle_status(cycle_id)
        
        print(f"   [{i*5:2d}s] Workers: {overview['worker_stats']['busy_workers']}/{overview['worker_stats']['active_workers']} busy | Cycle: {cycle_status.get('status', 'unknown')}")
    
    print("\n🎉 Demo completed!")
    print("\n📋 Next Steps:")
    print("   1. Start the web server: python run_distributed_system.py")
    print("   2. Open dashboard: http://localhost:8000")
    print("   3. Use API endpoints: http://localhost:8000/api/docs")
    
    return cycle_id

def main():
    """Main entry point"""
    
    import argparse
    parser = argparse.ArgumentParser(description="Distributed Training System")
    parser.add_argument("--mode", choices=["server", "demo"], default="server",
                       help="Run mode: server (web server) or demo (demonstration)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        # Run demo
        asyncio.run(demo_system())
    else:
        # Run web server
        print("🚀 Starting Distributed Training System...")
        print(f"   - Dashboard: http://{args.host}:{args.port}")
        print(f"   - API Docs: http://{args.host}:{args.port}/api/docs")
        print(f"   - Health Check: http://{args.host}:{args.port}/api/system/health")
        
        uvicorn.run(
            "run_distributed_system:app",
            host=args.host,
            port=args.port,
            reload=True,
            log_level="info"
        )

if __name__ == "__main__":
    main()