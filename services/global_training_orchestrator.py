#!/usr/bin/env python3
"""
Global Training Orchestrator - Hệ thống train cho millions users worldwide
Distributed, Multi-language, Auto-scaling Training System
"""

import asyncio
import redis.asyncio as aioredis
import asyncpg
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import hashlib
from dataclasses import dataclass
from enum import Enum
import random

class TrainingPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class Region(Enum):
    US = "us"
    EU = "eu" 
    ASIA = "asia"
    GLOBAL = "global"

@dataclass
class TrainingJob:
    job_id: str
    user_id: str
    keywords: List[str]
    language: str
    region: Region
    priority: TrainingPriority
    created_at: datetime
    estimated_duration: int  # minutes

class GlobalTrainingOrchestrator:
    def __init__(self):
        self.redis_pool = None
        self.db_pool = None
        self.active_jobs = {}
        self.worker_nodes = []
        self.training_queue = asyncio.Queue()
        
        # Global stats
        self.stats = {
            'total_users': 0,
            'active_training_jobs': 0,
            'completed_jobs_today': 0,
            'global_accuracy': 0.0,
            'cost_per_user': 0.0
        }
    
    async def initialize_global_infrastructure(self):
        """Initialize distributed infrastructure"""
        print("🌍 Initializing Global Training Infrastructure...")
        
        # Redis cluster for job queue
        self.redis_pool = aioredis.from_url(
            'redis://default:FqCRUeurO3Y0PzOGhsJMnBah6VxOl7Am@redis-10481.c289.us-west-1-2.ec2.redns.redis-cloud.com:10481',
            encoding='utf-8',
            decode_responses=True
        )
        
        # PostgreSQL cluster for training data
        self.db_pool = await asyncpg.create_pool(
            "postgresql://admin:Anh12345@postgresql-201376-0.cloudclusters.net:19976/serp_global",
            min_size=10,
            max_size=50
        )
        
        # Initialize worker nodes
        await self._setup_worker_nodes()
        
        print("✅ Global infrastructure ready")
    
    async def _setup_worker_nodes(self):
        """Setup distributed worker nodes"""
        # Simulate worker nodes in different regions
        regions_config = {
            Region.US: {'nodes': 20, 'capacity': 1000},
            Region.EU: {'nodes': 15, 'capacity': 800}, 
            Region.ASIA: {'nodes': 25, 'capacity': 1200}
        }
        
        for region, config in regions_config.items():
            for i in range(config['nodes']):
                worker = {
                    'id': f"{region.value}-worker-{i+1}",
                    'region': region,
                    'capacity': config['capacity'] // config['nodes'],
                    'current_load': 0,
                    'status': 'ready'
                }
                self.worker_nodes.append(worker)
        
        print(f"🔧 Setup {len(self.worker_nodes)} worker nodes globally")
    
    async def submit_training_request(self, user_id: str, keywords: List[str], 
                                    language: str = 'en', region: Region = Region.GLOBAL) -> str:
        """Submit training request from user"""
        
        # Generate job ID
        job_id = hashlib.md5(f"{user_id}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        
        # Determine priority based on user tier
        priority = await self._get_user_priority(user_id)
        
        # Estimate training duration
        estimated_duration = self._estimate_training_time(len(keywords), language, region)
        
        # Create training job
        job = TrainingJob(
            job_id=job_id,
            user_id=user_id,
            keywords=keywords,
            language=language,
            region=region,
            priority=priority,
            created_at=datetime.now(),
            estimated_duration=estimated_duration
        )
        
        # Add to queue
        await self.training_queue.put(job)
        
        # Store in Redis for tracking
        await self.redis_pool.hset(
            f"training_job:{job_id}",
            mapping={
                'user_id': user_id,
                'status': 'queued',
                'keywords_count': len(keywords),
                'language': language,
                'region': region.value,
                'created_at': job.created_at.isoformat(),
                'estimated_duration': estimated_duration
            }
        )
        
        print(f"📝 Training job {job_id} queued for user {user_id}")
        return job_id
    
    async def process_training_queue(self):
        """Process training queue continuously"""
        print("🔄 Starting global training queue processor...")
        
        while True:
            try:
                # Get next job from queue
                job = await self.training_queue.get()
                
                # Find available worker
                worker = await self._find_optimal_worker(job.region, job.priority)
                
                if worker:
                    # Assign job to worker
                    await self._assign_job_to_worker(job, worker)
                else:
                    # No workers available, requeue with delay
                    await asyncio.sleep(30)
                    await self.training_queue.put(job)
                
            except Exception as e:
                print(f"❌ Queue processing error: {e}")
                await asyncio.sleep(10)
    
    async def _find_optimal_worker(self, preferred_region: Region, priority: TrainingPriority) -> Optional[Dict]:
        """Find optimal worker for job"""
        
        # Filter available workers
        available_workers = [
            w for w in self.worker_nodes 
            if w['status'] == 'ready' and w['current_load'] < w['capacity']
        ]
        
        if not available_workers:
            return None
        
        # Prefer workers in same region
        region_workers = [w for w in available_workers if w['region'] == preferred_region]
        if region_workers:
            # Sort by current load
            return min(region_workers, key=lambda x: x['current_load'])
        
        # Fallback to any available worker
        return min(available_workers, key=lambda x: x['current_load'])
    
    async def _assign_job_to_worker(self, job: TrainingJob, worker: Dict):
        """Assign training job to worker"""
        
        print(f"🎯 Assigning job {job.job_id} to worker {worker['id']}")
        
        # Update worker status
        worker['current_load'] += 1
        worker['status'] = 'busy'
        
        # Update job status
        await self.redis_pool.hset(
            f"training_job:{job.job_id}",
            mapping={
                'status': 'training',
                'worker_id': worker['id'],
                'started_at': datetime.now().isoformat()
            }
        )
        
        # Start training in background
        asyncio.create_task(self._execute_training_job(job, worker))
    
    async def _execute_training_job(self, job: TrainingJob, worker: Dict):
        """Execute training job on worker"""
        
        try:
            print(f"🧠 Training job {job.job_id} started on {worker['id']}")
            
            # Simulate distributed training
            training_result = await self._run_distributed_training(job, worker)
            
            # Update job status
            await self.redis_pool.hset(
                f"training_job:{job.job_id}",
                mapping={
                    'status': 'completed',
                    'completed_at': datetime.now().isoformat(),
                    'accuracy': training_result['accuracy'],
                    'model_size_mb': training_result['model_size_mb']
                }
            )
            
            # Store training results
            await self._store_training_results(job, training_result)
            
            print(f"✅ Job {job.job_id} completed with {training_result['accuracy']:.2%} accuracy")
            
        except Exception as e:
            print(f"❌ Training job {job.job_id} failed: {e}")
            
            await self.redis_pool.hset(
                f"training_job:{job.job_id}",
                mapping={
                    'status': 'failed',
                    'error': str(e),
                    'failed_at': datetime.now().isoformat()
                }
            )
        
        finally:
            # Release worker
            worker['current_load'] -= 1
            if worker['current_load'] == 0:
                worker['status'] = 'ready'
    
    async def _run_distributed_training(self, job: TrainingJob, worker: Dict) -> Dict:
        """Run distributed training simulation"""
        
        # Simulate training time based on keywords count and region
        base_time = len(job.keywords) * 0.1  # 0.1 seconds per keyword
        region_multiplier = {'us': 1.0, 'eu': 1.2, 'asia': 0.8}.get(job.region.value, 1.0)
        training_time = base_time * region_multiplier
        
        await asyncio.sleep(min(training_time, 5))  # Max 5 seconds for demo
        
        # Simulate training results
        accuracy = random.uniform(0.85, 0.95)  # 85-95% accuracy
        model_size = random.uniform(2.0, 8.0)  # 2-8 MB model size
        
        return {
            'accuracy': accuracy,
            'model_size_mb': model_size,
            'training_samples': len(job.keywords) * random.randint(10, 50),
            'worker_id': worker['id'],
            'region': job.region.value
        }
    
    async def _store_training_results(self, job: TrainingJob, result: Dict):
        """Store training results in database"""
        
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO global_training_results 
                (job_id, user_id, keywords_count, language, region, accuracy, 
                 model_size_mb, training_samples, completed_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """, 
            job.job_id, job.user_id, len(job.keywords), job.language, 
            job.region.value, result['accuracy'], result['model_size_mb'],
            result['training_samples'], datetime.now()
            )
    
    async def _get_user_priority(self, user_id: str) -> TrainingPriority:
        """Get user priority based on subscription tier"""
        
        # Simulate user tier lookup
        user_tiers = {
            'free': TrainingPriority.LOW,
            'pro': TrainingPriority.MEDIUM, 
            'enterprise': TrainingPriority.HIGH,
            'premium': TrainingPriority.CRITICAL
        }
        
        # Random tier for demo
        tier = random.choice(list(user_tiers.keys()))
        return user_tiers[tier]
    
    def _estimate_training_time(self, keywords_count: int, language: str, region: Region) -> int:
        """Estimate training time in minutes"""
        
        base_time = keywords_count * 0.5  # 0.5 minutes per keyword
        
        # Language complexity multiplier
        language_multipliers = {
            'en': 1.0, 'es': 1.1, 'fr': 1.1, 'de': 1.2,
            'zh': 1.5, 'ja': 1.4, 'ar': 1.3, 'vi': 1.2
        }
        
        # Region latency multiplier
        region_multipliers = {
            Region.US: 1.0,
            Region.EU: 1.1, 
            Region.ASIA: 0.9,
            Region.GLOBAL: 1.2
        }
        
        total_time = base_time * language_multipliers.get(language, 1.0) * region_multipliers.get(region, 1.0)
        
        return max(int(total_time), 5)  # Minimum 5 minutes
    
    async def get_global_stats(self) -> Dict:
        """Get global training statistics"""
        
        # Get stats from Redis
        active_jobs = await self.redis_pool.keys("training_job:*")
        
        # Calculate stats
        total_workers = len(self.worker_nodes)
        busy_workers = len([w for w in self.worker_nodes if w['status'] == 'busy'])
        
        return {
            'total_workers': total_workers,
            'busy_workers': busy_workers,
            'utilization_percent': (busy_workers / total_workers) * 100,
            'active_jobs': len(active_jobs),
            'queue_size': self.training_queue.qsize(),
            'regions': {
                'us_workers': len([w for w in self.worker_nodes if w['region'] == Region.US]),
                'eu_workers': len([w for w in self.worker_nodes if w['region'] == Region.EU]),
                'asia_workers': len([w for w in self.worker_nodes if w['region'] == Region.ASIA])
            }
        }
    
    async def get_user_training_status(self, user_id: str) -> List[Dict]:
        """Get training status for specific user"""
        
        jobs = []
        job_keys = await self.redis_pool.keys(f"training_job:*")
        
        for key in job_keys:
            job_data = await self.redis_pool.hgetall(key)
            if job_data.get('user_id') == user_id:
                jobs.append(job_data)
        
        return jobs
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.redis_pool:
            self.redis_pool.close()
            await self.redis_pool.wait_closed()
        
        if self.db_pool:
            await self.db_pool.close()

# Demo usage
async def demo_global_training():
    """Demo global training system"""
    
    orchestrator = GlobalTrainingOrchestrator()
    await orchestrator.initialize_global_infrastructure()
    
    # Start queue processor
    queue_task = asyncio.create_task(orchestrator.process_training_queue())
    
    # Simulate multiple users submitting training requests
    users = [
        {'id': 'user_us_1', 'keywords': ['AI tools', 'machine learning'], 'lang': 'en', 'region': Region.US},
        {'id': 'user_eu_1', 'keywords': ['IA outils', 'apprentissage automatique'], 'lang': 'fr', 'region': Region.EU},
        {'id': 'user_asia_1', 'keywords': ['AI ツール', '機械学習'], 'lang': 'ja', 'region': Region.ASIA},
        {'id': 'user_vn_1', 'keywords': ['công cụ AI', 'học máy'], 'lang': 'vi', 'region': Region.ASIA}
    ]
    
    # Submit training requests
    job_ids = []
    for user in users:
        job_id = await orchestrator.submit_training_request(
            user['id'], user['keywords'], user['lang'], user['region']
        )
        job_ids.append(job_id)
    
    # Wait for some training to complete
    await asyncio.sleep(10)
    
    # Get global stats
    stats = await orchestrator.get_global_stats()
    print(f"\n🌍 Global Training Stats:")
    print(f"   - Total Workers: {stats['total_workers']}")
    print(f"   - Busy Workers: {stats['busy_workers']}")
    print(f"   - Utilization: {stats['utilization_percent']:.1f}%")
    print(f"   - Active Jobs: {stats['active_jobs']}")
    
    # Check user training status
    for user in users[:2]:
        status = await orchestrator.get_user_training_status(user['id'])
        print(f"\n👤 User {user['id']} training status:")
        for job in status:
            print(f"   - Job: {job.get('status', 'unknown')}")
    
    queue_task.cancel()
    await orchestrator.cleanup()

if __name__ == "__main__":
    asyncio.run(demo_global_training())