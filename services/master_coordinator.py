#!/usr/bin/env python3
"""
Master Coordinator - Điều phối toàn bộ hệ thống distributed training
Tích hợp: Auto Keyword Discovery + Distributed Crawling + Central Training
"""

import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import os
from pathlib import Path

# Import existing services
from .distributed_worker_manager import DistributedWorkerManager
from .auto_keyword_discovery import AutoKeywordDiscovery
from .global_training_orchestrator import GlobalTrainingOrchestrator, Region

class MasterCoordinator:
    def __init__(self):
        # Core services
        self.worker_manager = DistributedWorkerManager()
        self.keyword_discovery = AutoKeywordDiscovery()
        self.training_orchestrator = GlobalTrainingOrchestrator()
        
        # Configuration
        self.config = {
            'seed_keywords': [
                'AI tools', 'machine learning', 'web development', 
                'digital marketing', 'e-commerce', 'mobile apps',
                'cloud computing', 'cybersecurity', 'data science'
            ],
            'target_regions': ['us-east', 'eu-west', 'asia-southeast'],
            'max_keywords_per_worker': 50,
            'training_batch_size': 100,
            'storage_path': './shared_data/',
            'auto_discovery_interval': 6  # hours
        }
        
        # Job tracking
        self.active_jobs = {}
        self.completed_jobs = {}
        
        # Create storage directory
        os.makedirs(self.config['storage_path'], exist_ok=True)

    async def initialize_system(self):
        """Initialize complete distributed system"""
        print("🌟 Initializing Master Coordinator System...")
        
        # Initialize all services
        await self.worker_manager.initialize()
        await self.training_orchestrator.initialize_global_infrastructure()
        
        # Register initial workers
        await self._setup_initial_workers()
        
        # Start background services
        asyncio.create_task(self._continuous_keyword_discovery())
        asyncio.create_task(self._auto_training_pipeline())
        
        print("✅ Master Coordinator System ready!")

    async def _setup_initial_workers(self):
        """Setup initial worker fleet"""
        print("🔧 Setting up initial worker fleet...")
        
        worker_configs = [
            # US East workers
            {'region': 'us-east', 'count': 3, 'ip_base': '10.1.1'},
            # EU West workers  
            {'region': 'eu-west', 'count': 2, 'ip_base': '10.2.1'},
            # Asia Southeast workers
            {'region': 'asia-southeast', 'count': 4, 'ip_base': '10.3.1'}
        ]
        
        total_workers = 0
        for config in worker_configs:
            for i in range(config['count']):
                worker_config = {
                    'region': config['region'],
                    'ip_address': f"{config['ip_base']}.{i+10}",
                    'capabilities': ['crawling', 'keyword_discovery', 'data_processing']
                }
                
                worker_id = await self.worker_manager.register_worker(worker_config)
                total_workers += 1
        
        print(f"✅ Registered {total_workers} workers across {len(worker_configs)} regions")

    async def start_full_discovery_and_training_cycle(self, user_id: str, 
                                                    custom_seeds: Optional[List[str]] = None) -> str:
        """Start complete cycle: Discovery → Crawling → Training"""
        
        cycle_id = f"cycle_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"🚀 Starting full cycle {cycle_id} for user {user_id}")
        
        # Use custom seeds or default
        seed_keywords = custom_seeds or self.config['seed_keywords']
        
        # Phase 1: Auto Keyword Discovery
        print("📍 Phase 1: Auto Keyword Discovery")
        discovered_keywords = await self._phase1_keyword_discovery(seed_keywords)
        
        # Phase 2: Distributed Crawling
        print("📍 Phase 2: Distributed Crawling")
        crawl_job_id = await self._phase2_distributed_crawling(discovered_keywords)
        
        # Phase 3: Data Aggregation
        print("📍 Phase 3: Data Aggregation")
        aggregated_data = await self._phase3_data_aggregation(crawl_job_id)
        
        # Phase 4: Model Training
        print("📍 Phase 4: Model Training")
        training_job_id = await self._phase4_model_training(user_id, aggregated_data)
        
        # Store cycle info
        cycle_info = {
            'cycle_id': cycle_id,
            'user_id': user_id,
            'seed_keywords': seed_keywords,
            'discovered_keywords_count': sum(len(kws) for kws in discovered_keywords.values()),
            'crawl_job_id': crawl_job_id,
            'training_job_id': training_job_id,
            'started_at': datetime.now().isoformat(),
            'status': 'running'
        }
        
        self.active_jobs[cycle_id] = cycle_info
        
        # Monitor cycle completion
        asyncio.create_task(self._monitor_cycle_completion(cycle_id))
        
        return cycle_id

    async def _phase1_keyword_discovery(self, seed_keywords: List[str]) -> Dict[str, List[str]]:
        """Phase 1: Auto discover keywords from seeds"""
        
        print(f"🔍 Discovering keywords from {len(seed_keywords)} seeds...")
        
        # Discover trending keywords
        discovered = await self.keyword_discovery.discover_trending_keywords(max_keywords=1000)
        
        # Add seed-specific discoveries
        for seed in seed_keywords:
            # Simulate seed expansion (in real implementation, this would use actual APIs)
            seed_expanded = [
                f"{seed} tutorial", f"{seed} guide", f"{seed} tips",
                f"best {seed}", f"{seed} 2024", f"free {seed}",
                f"{seed} course", f"{seed} tools", f"{seed} software"
            ]
            
            # Add to English keywords
            if 'en' not in discovered:
                discovered['en'] = []
            discovered['en'].extend(seed_expanded)
        
        # Remove duplicates
        for lang in discovered:
            discovered[lang] = list(set(discovered[lang]))
        
        total_keywords = sum(len(kws) for kws in discovered.values())
        print(f"✅ Discovered {total_keywords} keywords across {len(discovered)} languages")
        
        return discovered

    async def _phase2_distributed_crawling(self, discovered_keywords: Dict[str, List[str]]) -> str:
        """Phase 2: Distribute crawling across workers"""
        
        print(f"🕷️ Starting distributed crawling...")
        
        # Convert to seed format for worker manager
        all_keywords = []
        for lang, keywords in discovered_keywords.items():
            all_keywords.extend(keywords[:100])  # Limit per language
        
        # Submit distributed crawl job
        crawl_job_id = await self.worker_manager.submit_distributed_crawl_job(
            seed_keywords=all_keywords,
            target_regions=self.config['target_regions'],
            max_keywords_per_worker=self.config['max_keywords_per_worker']
        )
        
        print(f"✅ Distributed crawl job {crawl_job_id} submitted")
        return crawl_job_id

    async def _phase3_data_aggregation(self, crawl_job_id: str) -> Dict:
        """Phase 3: Wait for crawling completion and aggregate data"""
        
        print(f"📊 Waiting for crawl job {crawl_job_id} completion...")
        
        # Monitor crawl job progress
        while True:
            status = await self.worker_manager.get_job_status(crawl_job_id)
            
            if status.get('progress_percent', 0) >= 100:
                print(f"✅ Crawling completed: {status['completed_tasks']}/{status['total_tasks']} tasks")
                break
            elif status.get('progress_percent', 0) > 0:
                print(f"⏳ Crawling progress: {status['progress_percent']:.1f}%")
            
            await asyncio.sleep(30)  # Check every 30 seconds
        
        # Aggregate all crawled data
        aggregated_data = await self._collect_and_merge_crawl_results(crawl_job_id)
        
        return aggregated_data

    async def _collect_and_merge_crawl_results(self, crawl_job_id: str) -> Dict:
        """Collect and merge all crawl results from shared storage"""
        
        print("📥 Collecting crawl results from shared storage...")
        
        # In real implementation, this would download from Google Drive/Dropbox/S3
        # For demo, simulate aggregated data
        
        aggregated_data = {
            'job_id': crawl_job_id,
            'total_keywords_processed': 0,
            'total_serp_results': 0,
            'data_by_language': {},
            'data_by_region': {},
            'training_samples': [],
            'collected_at': datetime.now().isoformat()
        }
        
        # Simulate data collection from multiple workers
        languages = ['en', 'es', 'fr', 'de', 'vi', 'zh', 'ja']
        regions = self.config['target_regions']
        
        for lang in languages:
            lang_data = []
            for region in regions:
                # Simulate SERP data for this lang/region combination
                for i in range(50):  # 50 keywords per region/lang
                    sample = {
                        'keyword': f'sample keyword {i} {lang}',
                        'language': lang,
                        'region': region,
                        'serp_results': [
                            {
                                'title': f'Title {j+1} for keyword {i}',
                                'url': f'https://example{j+1}.com/page{i}',
                                'description': f'Description {j+1} for keyword {i}',
                                'position': j+1
                            }
                            for j in range(10)
                        ]
                    }
                    lang_data.append(sample)
            
            aggregated_data['data_by_language'][lang] = lang_data
            aggregated_data['training_samples'].extend(lang_data)
        
        aggregated_data['total_keywords_processed'] = len(aggregated_data['training_samples'])
        aggregated_data['total_serp_results'] = sum(
            len(sample['serp_results']) for sample in aggregated_data['training_samples']
        )
        
        # Save aggregated data locally
        output_file = f"{self.config['storage_path']}/aggregated_data_{crawl_job_id}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(aggregated_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Aggregated {aggregated_data['total_keywords_processed']} keywords with {aggregated_data['total_serp_results']} SERP results")
        print(f"💾 Data saved to: {output_file}")
        
        return aggregated_data

    async def _phase4_model_training(self, user_id: str, aggregated_data: Dict) -> str:
        """Phase 4: Train model with aggregated data"""
        
        print(f"🧠 Starting model training for user {user_id}...")
        
        # Prepare training data
        training_keywords = [
            sample['keyword'] for sample in aggregated_data['training_samples']
        ]
        
        # Submit training job to orchestrator
        training_job_id = await self.training_orchestrator.submit_training_request(
            user_id=user_id,
            keywords=training_keywords,
            language='en',  # Primary language
            region=Region.GLOBAL
        )
        
        print(f"✅ Training job {training_job_id} submitted")
        return training_job_id

    async def _monitor_cycle_completion(self, cycle_id: str):
        """Monitor complete cycle until finished"""
        
        cycle_info = self.active_jobs.get(cycle_id)
        if not cycle_info:
            return
        
        print(f"👁️ Monitoring cycle {cycle_id}...")
        
        while True:
            try:
                # Check training job status
                training_status = await self.training_orchestrator.get_user_training_status(
                    cycle_info['user_id']
                )
                
                # Find our training job
                our_job = None
                for job in training_status:
                    if job.get('job_id') == cycle_info.get('training_job_id'):
                        our_job = job
                        break
                
                if our_job and our_job.get('status') == 'completed':
                    # Cycle completed!
                    cycle_info['status'] = 'completed'
                    cycle_info['completed_at'] = datetime.now().isoformat()
                    
                    # Move to completed jobs
                    self.completed_jobs[cycle_id] = cycle_info
                    del self.active_jobs[cycle_id]
                    
                    print(f"🎉 Cycle {cycle_id} completed successfully!")
                    break
                
                elif our_job and our_job.get('status') == 'failed':
                    # Cycle failed
                    cycle_info['status'] = 'failed'
                    cycle_info['failed_at'] = datetime.now().isoformat()
                    cycle_info['error'] = our_job.get('error', 'Training failed')
                    
                    print(f"❌ Cycle {cycle_id} failed: {cycle_info['error']}")
                    break
                
                # Still running, wait and check again
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                print(f"❌ Error monitoring cycle {cycle_id}: {e}")
                await asyncio.sleep(60)

    async def _continuous_keyword_discovery(self):
        """Continuously discover new trending keywords"""
        
        print(f"🔄 Starting continuous keyword discovery (every {self.config['auto_discovery_interval']}h)")
        
        while True:
            try:
                # Discover new trending keywords
                new_keywords = await self.keyword_discovery.discover_trending_keywords(max_keywords=500)
                
                # Update seed keywords with trending ones
                if 'en' in new_keywords:
                    # Add top trending keywords to seeds
                    trending_seeds = new_keywords['en'][:10]
                    self.config['seed_keywords'].extend(trending_seeds)
                    
                    # Keep only unique seeds
                    self.config['seed_keywords'] = list(set(self.config['seed_keywords']))
                    
                    print(f"🔄 Updated seed keywords: {len(self.config['seed_keywords'])} total")
                
                # Wait for next discovery cycle
                await asyncio.sleep(self.config['auto_discovery_interval'] * 3600)
                
            except Exception as e:
                print(f"❌ Continuous discovery error: {e}")
                await asyncio.sleep(1800)  # Wait 30 minutes on error

    async def _auto_training_pipeline(self):
        """Auto-trigger training for users based on data freshness"""
        
        print("🤖 Starting auto-training pipeline...")
        
        while True:
            try:
                # Check if we have enough fresh data to trigger training
                # This is a simplified version - in production, you'd check user preferences
                
                if len(self.active_jobs) < 2:  # Don't overload system
                    # Simulate user requesting training
                    demo_user_id = f"auto_user_{datetime.now().strftime('%H%M')}"
                    
                    cycle_id = await self.start_full_discovery_and_training_cycle(
                        user_id=demo_user_id,
                        custom_seeds=['trending AI tools', 'latest tech news', 'popular apps']
                    )
                    
                    print(f"🤖 Auto-triggered training cycle: {cycle_id}")
                
                # Wait 2 hours before next auto-training
                await asyncio.sleep(7200)
                
            except Exception as e:
                print(f"❌ Auto-training pipeline error: {e}")
                await asyncio.sleep(3600)

    async def get_system_overview(self) -> Dict:
        """Get complete system overview"""
        
        # Get stats from all services
        worker_stats = await self.worker_manager.get_system_stats()
        training_stats = await self.training_orchestrator.get_global_stats()
        
        return {
            'system_status': 'running',
            'active_cycles': len(self.active_jobs),
            'completed_cycles': len(self.completed_jobs),
            'worker_stats': worker_stats,
            'training_stats': training_stats,
            'seed_keywords_count': len(self.config['seed_keywords']),
            'target_regions': self.config['target_regions'],
            'last_updated': datetime.now().isoformat()
        }

    async def get_cycle_status(self, cycle_id: str) -> Dict:
        """Get status of specific cycle"""
        
        if cycle_id in self.active_jobs:
            return {**self.active_jobs[cycle_id], 'status': 'active'}
        elif cycle_id in self.completed_jobs:
            return {**self.completed_jobs[cycle_id], 'status': 'completed'}
        else:
            return {'error': 'Cycle not found'}

# Demo usage
async def demo_master_coordinator():
    """Demo complete distributed system"""
    
    coordinator = MasterCoordinator()
    await coordinator.initialize_system()
    
    print("\n" + "="*60)
    print("🌟 DISTRIBUTED TRAINING SYSTEM DEMO")
    print("="*60)
    
    # Start a full cycle
    cycle_id = await coordinator.start_full_discovery_and_training_cycle(
        user_id="demo_user_001",
        custom_seeds=["AI chatbots", "machine learning tools", "web scraping"]
    )
    
    print(f"\n🚀 Started cycle: {cycle_id}")
    
    # Monitor system for a while
    for i in range(12):  # Monitor for 2 minutes
        await asyncio.sleep(10)
        
        # Get system overview
        overview = await coordinator.get_system_overview()
        cycle_status = await coordinator.get_cycle_status(cycle_id)
        
        print(f"\n📊 System Overview (t+{i*10}s):")
        print(f"   - Active Cycles: {overview['active_cycles']}")
        print(f"   - Worker Utilization: {overview['worker_stats']['busy_workers']}/{overview['worker_stats']['active_workers']}")
        print(f"   - Queue Size: {overview['worker_stats']['queue_size']}")
        print(f"   - Cycle Status: {cycle_status.get('status', 'unknown')}")
        
        if cycle_status.get('status') == 'completed':
            print("🎉 Demo cycle completed!")
            break
    
    print("\n✅ Demo completed!")

if __name__ == "__main__":
    asyncio.run(demo_master_coordinator())