#!/usr/bin/env python3
"""
Distributed Worker Manager - Quản lý workers crawl phân tán
Auto-scaling, Load balancing, Fault-tolerant Worker System
"""
import os
import asyncio
import aiohttp
import redis.asyncio as aioredis
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import json
import hashlib
import random
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path
 
# Import existing services
from .auto_keyword_discovery import AutoKeywordDiscovery
from .global_training_orchestrator import GlobalTrainingOrchestrator, Region

class WorkerStatus(Enum):
    IDLE = "idle"
    CRAWLING = "crawling"
    UPLOADING = "uploading"
    ERROR = "error"
    OFFLINE = "offline"

class TaskType(Enum):
    KEYWORD_DISCOVERY = "keyword_discovery"
    SERP_CRAWLING = "serp_crawling"
    DATA_PROCESSING = "data_processing"

@dataclass
class WorkerNode:
    worker_id: str
    region: str
    ip_address: str
    status: WorkerStatus
    current_task: Optional[str]
    capabilities: List[str]
    last_heartbeat: datetime
    total_tasks_completed: int
    success_rate: float
    avg_task_duration: float

@dataclass
class CrawlTask:
    task_id: str
    task_type: TaskType
    keywords: List[str]
    target_region: str
    language: str
    priority: int
    created_at: datetime
    assigned_worker: Optional[str]
    estimated_duration: int
    retry_count: int

class DistributedWorkerManager:
    def __init__(self):
        self.workers: Dict[str, WorkerNode] = {}
        self.task_queue = asyncio.Queue()
        self.completed_tasks: Dict[str, Dict] = {}
        self.redis_pool = None
        
        # Storage configuration
        self.storage_config = {
            'type': 'google_drive',  # google_drive, dropbox, s3, ftp
            'credentials': {
                'google_drive': {
                    'folder_id': 'Data_AI',
                    'service_account_key': self._get_safe_path('config/credentials/google_drive/fit-heaven-453117-m1-79222e8ee670.json')
                }
            }
        }
        
        # Auto-discovery service
        self.keyword_discovery = AutoKeywordDiscovery()
        
        # Training orchestrator
        self.training_orchestrator = GlobalTrainingOrchestrator()
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        
        
        
    def _get_safe_path(self, relative_path: str) -> str:
        """
        Chuyển đổi đường dẫn tương đối thành đường dẫn tuyệt đối an toàn
        ngăn chặn các lỗi path traversal
        """
        # Lấy thư mục gốc của project
        project_root = Path(__file__).parent.parent
        
        # Tạo path tuyệt đối
        absolute_path = (project_root / relative_path).resolve()
        
        # Kiểm tra path traversal - đảm bảo path vẫn nằm trong project root
        try:
            absolute_path.relative_to(project_root)
        except ValueError:
            raise ValueError(f"Path {relative_path} attempts to traverse outside project directory")
        
        # Tạo thư mục nếu chưa tồn tại
        absolute_path.parent.mkdir(parents=True, exist_ok=True)
        
        return str(absolute_path)
    async def initialize(self):
        """Initialize distributed worker system"""
        self.logger.info("🚀 Initializing Distributed Worker Manager...")
        
        # FIX: Sửa cấu hình Redis connection
        redis_url = os.environ.get('REDIS_URL', 'redis://default:FqCRUeurO3Y0PzOGhsJMnBah6VxOl7Am@redis-10481.c289.us-west-1-2.ec2.redns.redis-cloud.com:10481')
        
        try:
            self.redis_pool = aioredis.from_url(
                redis_url,
                encoding='utf-8',
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
            
            # Test connection
            await self.redis_pool.ping()
            self.logger.info("✅ Redis connection established")
            
        except Exception as e:
            self.logger.error(f"❌ Redis connection failed: {e}")
            self.logger.warning("⚠️  Using in-memory storage (Redis not available)")
            # Fallback to in-memory storage
            self._use_in_memory_storage = True
            self._memory_storage = {}
        
        # Initialize services
        try:
            await self.training_orchestrator.initialize_global_infrastructure()
        except Exception as e:
            self.logger.warning(f"⚠️  Training orchestrator init failed: {e}")
        
        # Start background tasks
        asyncio.create_task(self._worker_health_monitor())
        asyncio.create_task(self._task_dispatcher())
        asyncio.create_task(self._auto_scaling_manager())
        
        self.logger.info("✅ Distributed Worker Manager initialized")

    # Thêm các phương thức helper để xử lý khi Redis không available
    async def _redis_hset(self, key: str, mapping: Dict):
        """Safe Redis HSET with fallback"""
        if hasattr(self, '_use_in_memory_storage') and self._use_in_memory_storage:
            self._memory_storage[key] = mapping
        else:
            try:
                await self.redis_pool.hset(key, mapping=mapping)
            except Exception as e:
                self.logger.error(f"Redis HSET failed: {e}, using memory fallback")
                if not hasattr(self, '_memory_storage'):
                    self._memory_storage = {}
                self._memory_storage[key] = mapping

    async def _redis_hgetall(self, key: str) -> Dict:
        """Safe Redis HGETALL with fallback"""
        if hasattr(self, '_use_in_memory_storage') and self._use_in_memory_storage:
            return self._memory_storage.get(key, {})
        else:
            try:
                return await self.redis_pool.hgetall(key)
            except Exception as e:
                self.logger.error(f"Redis HGETALL failed: {e}, using memory fallback")
                return self._memory_storage.get(key, {})

    async def _redis_keys(self, pattern: str) -> List[str]:
        """Safe Redis KEYS with fallback"""
        if hasattr(self, '_use_in_memory_storage') and self._use_in_memory_storage:
            # Fallback for keys is not exact, just a simple filter
            return [k for k in self._memory_storage.keys() if k.startswith(pattern.split('*')[0])]
        else:
            try:
                return await self.redis_pool.keys(pattern)
            except Exception as e:
                self.logger.error(f"Redis KEYS failed: {e}, cannot use memory fallback for pattern matching")
                return []
    
    async def register_worker(self, worker_config: Dict) -> str:
        """Register new worker node"""
        
        # Validate worker_config to ensure no None values
        validated_config = {
            'region': worker_config.get('region', 'unknown') or 'unknown',
            'ip_address': worker_config.get('ip_address', '') or '',
            'capabilities': worker_config.get('capabilities', ['crawling', 'keyword_discovery']) or ['crawling', 'keyword_discovery']
        }
        
        worker_id = f"worker_{validated_config['region']}_{random.randint(1000, 9999)}"
        
        worker = WorkerNode(
            worker_id=worker_id,
            region=validated_config['region'],
            ip_address=validated_config['ip_address'],
            status=WorkerStatus.IDLE,
            current_task=None,
            capabilities=validated_config['capabilities'],
            last_heartbeat=datetime.now(),
            total_tasks_completed=0,
            success_rate=1.0,
            avg_task_duration=0.0
        )
        
        self.workers[worker_id] = worker

        # Convert worker to Redis-compatible dictionary
        worker_dict = {
            "worker_id": worker.worker_id,
            "region": worker.region,
            "ip_address": worker.ip_address,
            "status": worker.status.value,
            "current_task": worker.current_task or "",
            "capabilities": json.dumps(worker.capabilities),
            "last_heartbeat": worker.last_heartbeat.isoformat(),
            "total_tasks_completed": worker.total_tasks_completed,
            "success_rate": worker.success_rate,
            "avg_task_duration": worker.avg_task_duration
        }

        # Use safe Redis operation
        await self._redis_hset(f"worker:{worker_id}", worker_dict)
        
        self.logger.info(f"✅ Worker {worker_id} registered from {worker.region}")
        return worker_id

    async def submit_distributed_crawl_job(self, seed_keywords: List[str], 
                                         target_regions: List[str],
                                         max_keywords_per_worker: int = 50) -> str:
        """Submit distributed crawl job"""
        
        job_id = hashlib.md5(f"job_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        
        self.logger.info(f"📝 Creating distributed crawl job {job_id}")
        
        # Step 1: Auto-discover keywords from seeds
        all_discovered_keywords = {}
        
        for seed in seed_keywords:
            # Each worker will expand this seed
            discovered = await self.keyword_discovery.discover_trending_keywords(max_keywords=200)
            
            for lang, keywords in discovered.items():
                if lang not in all_discovered_keywords:
                    all_discovered_keywords[lang] = []
                all_discovered_keywords[lang].extend(keywords)
        
        # Step 2: Distribute tasks to workers
        tasks_created = []
        
        for region in target_regions:
            # Find available workers in region
            region_workers = [
                w for w in self.workers.values() 
                if w.region == region and w.status == WorkerStatus.IDLE
            ]
            
            if not region_workers:
                self.logger.warning(f"⚠️ No available workers in region {region}")
                continue
            
            # Distribute keywords among workers
            for lang, keywords in all_discovered_keywords.items():
                # Split keywords into chunks
                keyword_chunks = [
                    keywords[i:i + max_keywords_per_worker] 
                    for i in range(0, len(keywords), max_keywords_per_worker)
                ]
                
                for chunk in keyword_chunks:
                    task = CrawlTask(
                        task_id=f"{job_id}_{region}_{lang}_{len(tasks_created)}",
                        task_type=TaskType.SERP_CRAWLING,
                        keywords=chunk,
                        target_region=region,
                        language=lang,
                        priority=1,
                        created_at=datetime.now(),
                        assigned_worker=None,
                        estimated_duration=len(chunk) * 2,  # 2 minutes per keyword
                        retry_count=0
                    )
                    
                    await self.task_queue.put(task)
                    tasks_created.append(task.task_id)
        
        # Store job info
        job_info = {
            'job_id': job_id,
            'seed_keywords': json.dumps(seed_keywords),
            'target_regions': json.dumps(target_regions),
            'total_tasks': len(tasks_created),
            'created_at': datetime.now().isoformat(),
            'status': 'queued'
        }
        
        await self._redis_hset(f"job:{job_id}", job_info)
        
        self.logger.info(f"✅ Created {len(tasks_created)} tasks for job {job_id}")
        return job_id

    async def _task_dispatcher(self):
        """Dispatch tasks to available workers"""
        
        while True:
            try:
                # Get next task
                task = await self.task_queue.get()
                
                # Find best worker for task
                worker = await self._find_optimal_worker(task)
                
                if worker:
                    # Assign task to worker
                    await self._assign_task_to_worker(task, worker)
                else:
                    # No workers available, requeue
                    self.logger.warning(f"⚠️ No workers available for task {task.task_id}, re-queuing...")
                    await asyncio.sleep(30)
                    await self.task_queue.put(task)
                
            except Exception as e:
                self.logger.error(f"❌ Task dispatcher error: {e}")
                await asyncio.sleep(10)

    async def _find_optimal_worker(self, task: CrawlTask) -> Optional[WorkerNode]:
        """Find optimal worker for task"""
        
        # Filter available workers
        available_workers = [
            w for w in self.workers.values()
            if w.status == WorkerStatus.IDLE and 
               task.task_type.value in w.capabilities
        ]
        
        if not available_workers:
            return None
        
        # Prefer workers in same region
        region_workers = [w for w in available_workers if w.region == task.target_region]
        if region_workers:
            # Sort by success rate and avg duration
            return max(region_workers, key=lambda x: x.success_rate / (x.avg_task_duration + 1))
        
        # Fallback to any available worker
        return max(available_workers, key=lambda x: x.success_rate / (x.avg_task_duration + 1))

    async def _assign_task_to_worker(self, task: CrawlTask, worker: WorkerNode):
        """Assign task to worker"""
        
        self.logger.info(f"🎯 Assigning task {task.task_id} to worker {worker.worker_id}")
        
        # Update worker status
        worker.status = WorkerStatus.CRAWLING
        worker.current_task = task.task_id
        task.assigned_worker = worker.worker_id
        
        # Store task assignment
        await self._redis_hset(
            f"task:{task.task_id}",
            mapping={
                'status': 'assigned',
                'worker_id': worker.worker_id,
                'assigned_at': datetime.now().isoformat(),
                'keywords': json.dumps(task.keywords),
                'region': task.target_region
            }
        )
        
        # Send task to worker (simulate API call)
        asyncio.create_task(self._execute_worker_task(task, worker))

    async def _execute_worker_task(self, task: CrawlTask, worker: WorkerNode):
        """Execute task on worker (simulated)"""
        
        try:
            self.logger.info(f"🔄 Worker {worker.worker_id} executing task {task.task_id}")
            
            # Update task status
            await self._redis_hset(
                f"task:{task.task_id}",
                mapping={'status': 'running', 'started_at': datetime.now().isoformat()}
            )
            
            # Simulate crawling work
            crawl_results = await self._simulate_crawling_work(task, worker)
            
            # Upload results to shared storage
            upload_success = await self._upload_results_to_storage(task, crawl_results)
            
            if upload_success:
                # Mark task as completed
                await self._complete_task(task, worker, crawl_results)
            else:
                # Mark task as failed
                await self._fail_task(task, worker, "Upload failed")
                
        except Exception as e:
            self.logger.error(f"❌ Task execution failed: {e}")
            await self._fail_task(task, worker, str(e))

    async def _simulate_crawling_work(self, task: CrawlTask, worker: WorkerNode) -> Dict:
        """Simulate crawling work"""
        
        # Simulate crawling time
        crawl_time = len(task.keywords) * random.uniform(0.5, 2.0)  # 0.5-2 seconds per keyword
        await asyncio.sleep(min(crawl_time, 10))  # Max 10 seconds for demo
        
        # Generate mock SERP data
        results = {
            'task_id': task.task_id,
            'worker_id': worker.worker_id,
            'keywords_processed': len(task.keywords),
            'serp_data': [],
            'processing_time': crawl_time,
            'success_rate': random.uniform(0.85, 0.98)
        }
        
        # Mock SERP data for each keyword
        for keyword in task.keywords:
            serp_entry = {
                'keyword': keyword,
                'language': task.language,
                'region': task.target_region,
                'results': [
                    {
                        'title': f"Result for {keyword} #{i+1}",
                        'url': f"https://example{i+1}.com/{keyword.replace(' ', '-')}",
                        'description': f"Description for {keyword} result {i+1}",
                        'position': i+1
                    }
                    for i in range(10)  # Top 10 results
                ],
                'crawled_at': datetime.now().isoformat()
            }
            results['serp_data'].append(serp_entry)
        
        return results

    async def _upload_results_to_storage(self, task: CrawlTask, results: Dict) -> bool:
        """Upload results to shared storage"""
        
        try:
            # Generate filename
            filename = f"crawl_results_{task.task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Simulate upload to Google Drive/Dropbox/etc
            self.logger.info(f"📤 Uploading {filename} to shared storage...")
            
            # In real implementation, this would upload to actual storage
            # For now, simulate upload delay
            await asyncio.sleep(random.uniform(1, 3))
            
            # Store upload info
            upload_info = {
                'filename': filename,
                'storage_path': f"/shared/crawl_data/{filename}",
                'file_size_mb': len(json.dumps(results)) / (1024 * 1024),
                'uploaded_at': datetime.now().isoformat()
            }
            
            await self._redis_hset(
                f"upload:{task.task_id}",
                mapping=upload_info
            )
            
            self.logger.info(f"✅ Upload completed: {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Upload failed: {e}")
            return False

    async def _complete_task(self, task: CrawlTask, worker: WorkerNode, results: Dict):
        """Mark task as completed"""
        
        # Update task status
        await self._redis_hset(
            f"task:{task.task_id}",
            mapping={
                'status': 'completed',
                'completed_at': datetime.now().isoformat(),
                'keywords_processed': len(task.keywords),
                'success_rate': results['success_rate']
            }
        )
        
        # Update worker stats
        worker.status = WorkerStatus.IDLE
        worker.current_task = None
        worker.total_tasks_completed += 1
        worker.success_rate = (worker.success_rate * (worker.total_tasks_completed - 1) + results['success_rate']) / worker.total_tasks_completed
        worker.avg_task_duration = (worker.avg_task_duration * (worker.total_tasks_completed - 1) + results['processing_time']) / worker.total_tasks_completed
        
        # Store completed task
        self.completed_tasks[task.task_id] = results
        
        self.logger.info(f"✅ Task {task.task_id} completed by worker {worker.worker_id}")

    async def _fail_task(self, task: CrawlTask, worker: WorkerNode, error_message: str):
        """Mark task as failed"""
        
        # Update task status
        await self._redis_hset(
            f"task:{task.task_id}",
            mapping={
                'status': 'failed',
                'failed_at': datetime.now().isoformat(),
                'error_message': error_message,
                'retry_count': task.retry_count
            }
        )
        
        # Update worker status
        worker.status = WorkerStatus.IDLE
        worker.current_task = None
        
        # Retry logic
        if task.retry_count < 3:
            task.retry_count += 1
            await asyncio.sleep(60)  # Wait 1 minute before retry
            await self.task_queue.put(task)
            self.logger.info(f"🔄 Retrying task {task.task_id} (attempt {task.retry_count})")
        else:
            self.logger.error(f"❌ Task {task.task_id} failed permanently: {error_message}")

    async def _worker_health_monitor(self):
        """Monitor worker health and handle failures"""
        
        while True:
            try:
                current_time = datetime.now()
                
                for worker_id, worker in list(self.workers.items()):
                    # Check if worker is responsive
                    time_since_heartbeat = current_time - worker.last_heartbeat
                    
                    if time_since_heartbeat > timedelta(minutes=5):
                        self.logger.warning(f"⚠️ Worker {worker_id} appears offline")
                        worker.status = WorkerStatus.OFFLINE
                        
                        # Reassign current task if any
                        if worker.current_task:
                            await self._reassign_task(worker.current_task)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"❌ Health monitor error: {e}")
                await asyncio.sleep(60)

    async def _reassign_task(self, task_id: str):
        """Reassign a failed task to another worker"""
        self.logger.info(f"Reassigning task {task_id} due to worker failure.")
        task_data = await self._redis_hgetall(f"task:{task_id}")
        
        if not task_data:
            self.logger.warning(f"Task {task_id} not found, cannot reassign.")
            return

        try:
            # Recreate task object from stored data
            recreated_task = CrawlTask(
                task_id=task_id,
                task_type=TaskType.SERP_CRAWLING, # Assuming type for this demo
                keywords=json.loads(task_data.get('keywords', '[]')),
                target_region=task_data.get('region', 'unknown'),
                language='en', # Assuming language for this demo
                priority=1,
                created_at=datetime.fromisoformat(task_data.get('created_at', datetime.now().isoformat())),
                assigned_worker=None,
                estimated_duration=0,
                retry_count=int(task_data.get('retry_count', 0)) + 1
            )
            
            if recreated_task.retry_count < 3:
                await self.task_queue.put(recreated_task)
                self.logger.info(f"Task {task_id} re-queued for retry {recreated_task.retry_count}.")
            else:
                self.logger.error(f"Task {task_id} exceeded max retries, giving up.")

        except Exception as e:
            self.logger.error(f"Failed to reassign task {task_id}: {e}")
            
    async def _auto_scaling_manager(self):
        """Auto-scale workers based on queue size"""
        
        while True:
            try:
                queue_size = self.task_queue.qsize()
                active_workers = len([w for w in self.workers.values() if w.status != WorkerStatus.OFFLINE])
                
                # Scale up if queue is large
                if queue_size > active_workers * 2:
                    self.logger.info(f"📈 Queue size ({queue_size}) high, considering scale up")
                    # In real implementation, would spawn new workers
                
                # Scale down if too many idle workers
                idle_workers = len([w for w in self.workers.values() if w.status == WorkerStatus.IDLE])
                if idle_workers > queue_size + 5:
                    self.logger.info(f"📉 Too many idle workers ({idle_workers}), considering scale down")
                    # In real implementation, would terminate excess workers
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"❌ Auto-scaling error: {e}")
                await asyncio.sleep(300)

    async def get_job_status(self, job_id: str) -> Dict:
        """Get status of distributed crawl job"""
        
        job_info = await self._redis_hgetall(f"job:{job_id}")
        
        if not job_info:
            return {'error': 'Job not found'}
        
        # Get task statuses
        task_keys = await self._redis_keys(f"task:{job_id}_*")
        task_statuses = {}
        
        for key in task_keys:
            task_data = await self._redis_hgetall(key)
            task_id = key.split(':')[1]
            task_statuses[task_id] = task_data.get('status', 'unknown')
        
        # Calculate progress
        total_tasks = len(task_statuses)
        completed_tasks = len([s for s in task_statuses.values() if s == 'completed'])
        failed_tasks = len([s for s in task_statuses.values() if s == 'failed'])
        
        return {
            'job_id': job_id,
            'status': job_info.get('status'),
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks,
            'failed_tasks': failed_tasks,
            'progress_percent': (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
            'task_statuses': task_statuses
        }

    async def get_system_stats(self) -> Dict:
        """Get distributed system statistics"""
        
        total_workers = len(self.workers)
        active_workers = len([w for w in self.workers.values() if w.status != WorkerStatus.OFFLINE])
        busy_workers = len([w for w in self.workers.values() if w.status == WorkerStatus.CRAWLING])
        
        return {
            'total_workers': total_workers,
            'active_workers': active_workers,
            'busy_workers': busy_workers,
            'idle_workers': active_workers - busy_workers,
            'queue_size': self.task_queue.qsize(),
            'completed_tasks_total': sum(w.total_tasks_completed for w in self.workers.values()),
            'average_success_rate': sum(w.success_rate for w in self.workers.values()) / total_workers if total_workers > 0 else 0,
            'workers_by_region': {
                region: len([w for w in self.workers.values() if w.region == region])
                for region in set(w.region for w in self.workers.values())
            }
        }

# Demo usage
async def demo_distributed_system():
    """Demo distributed worker system"""
    
    manager = DistributedWorkerManager()
    await manager.initialize()
    
    # Register some workers
    regions = ['us-east', 'eu-west', 'asia-southeast']
    for region in regions:
        for i in range(3):  # 3 workers per region
            worker_config = {
                'region': region,
                'ip_address': f"192.168.{random.randint(1,254)}.{random.randint(1,254)}",
                'capabilities': ['crawling', 'keyword_discovery']
            }
            worker_id = await manager.register_worker(worker_config)
            print(f"✅ Registered worker: {worker_id}")
    
    # Submit distributed crawl job
    seed_keywords = ['AI tools', 'machine learning', 'web development']
    target_regions = ['us-east', 'eu-west']
    
    job_id = await manager.submit_distributed_crawl_job(
        seed_keywords=seed_keywords,
        target_regions=target_regions,
        max_keywords_per_worker=20
    )
    
    print(f"\n🚀 Submitted job: {job_id}")
    
    # Monitor progress
    for i in range(10):
        await asyncio.sleep(5)
        status = await manager.get_job_status(job_id)
        print(f"📊 Job progress: {status['progress_percent']:.1f}% ({status['completed_tasks']}/{status['total_tasks']})")
        
        if status['progress_percent'] >= 100:
            break
    
    # Get system stats
    stats = await manager.get_system_stats()
    print(f"\n📈 System Stats:")
    print(f"   - Total Workers: {stats['total_workers']}")
    print(f"   - Active Workers: {stats['active_workers']}")
    print(f"   - Queue Size: {stats['queue_size']}")
    print(f"   - Success Rate: {stats['average_success_rate']:.2%}")

if __name__ == "__main__":
    asyncio.run(demo_distributed_system())