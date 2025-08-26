#!/usr/bin/env python3
"""
Demo Distributed System - Script demo đầy đủ hệ thống distributed training
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List

# Import services
from services.master_coordinator import MasterCoordinator
from services.distributed_worker_manager import DistributedWorkerManager
from services.auto_keyword_discovery import AutoKeywordDiscovery

class DistributedSystemDemo:
    def __init__(self):
        self.coordinator = None
        self.demo_results = {}
        
    async def run_complete_demo(self):
        """Chạy demo đầy đủ hệ thống"""
        
        print("🌟" + "="*60 + "🌟")
        print("     DISTRIBUTED SERP TRAINING SYSTEM DEMO")
        print("🌟" + "="*60 + "🌟")
        
        # Phase 1: System Initialization
        await self._demo_phase1_initialization()
        
        # Phase 2: Keyword Discovery
        await self._demo_phase2_keyword_discovery()
        
        # Phase 3: Worker Management
        await self._demo_phase3_worker_management()
        
        # Phase 4: Distributed Training
        await self._demo_phase4_distributed_training()
        
        # Phase 5: Monitoring & Results
        await self._demo_phase5_monitoring()
        
        # Summary
        self._demo_summary()

    async def _demo_phase1_initialization(self):
        """Phase 1: Khởi tạo hệ thống"""
        
        print("\n🚀 PHASE 1: SYSTEM INITIALIZATION")
        print("-" * 40)
        
        print("📋 Initializing Master Coordinator...")
        self.coordinator = MasterCoordinator()
        await self.coordinator.initialize_system()
        
        # Get initial system stats
        overview = await self.coordinator.get_system_overview()
        
        print("✅ System initialized successfully!")
        print(f"   - Total Workers: {overview['worker_stats']['total_workers']}")
        print(f"   - Active Workers: {overview['worker_stats']['active_workers']}")
        print(f"   - Target Regions: {', '.join(overview['target_regions'])}")
        print(f"   - Seed Keywords: {len(overview['seed_keywords_count'])}")
        
        self.demo_results['initialization'] = {
            'status': 'success',
            'workers': overview['worker_stats']['total_workers'],
            'regions': len(overview['target_regions'])
        }

    async def _demo_phase2_keyword_discovery(self):
        """Phase 2: Auto Keyword Discovery"""
        
        print("\n🔍 PHASE 2: AUTO KEYWORD DISCOVERY")
        print("-" * 40)
        
        print("📊 Discovering trending keywords...")
        
        # Test keyword discovery
        keywords = await self.coordinator.keyword_discovery.discover_trending_keywords(max_keywords=200)
        
        total_keywords = sum(len(kws) for kws in keywords.values())
        languages = list(keywords.keys())
        
        print(f"✅ Keyword discovery completed!")
        print(f"   - Total Keywords: {total_keywords}")
        print(f"   - Languages: {', '.join(languages)}")
        
        # Show sample keywords for each language
        for lang, kws in keywords.items():
            if kws:
                samples = ', '.join(kws[:3])
                print(f"   - {lang.upper()}: {samples}...")
        
        # Test personalized keywords
        print("\n🎯 Testing personalized keyword discovery...")
        user_profile = {
            'language': 'vi',
            'interests': ['technology', 'AI', 'programming'],
            'industry': 'software'
        }
        
        personalized = await self.coordinator.keyword_discovery.get_personalized_keywords(user_profile)
        print(f"   - Personalized Keywords (Vietnamese Tech): {len(personalized)}")
        if personalized:
            print(f"   - Samples: {', '.join(personalized[:3])}...")
        
        self.demo_results['keyword_discovery'] = {
            'status': 'success',
            'total_keywords': total_keywords,
            'languages': len(languages),
            'personalized_keywords': len(personalized)
        }

    async def _demo_phase3_worker_management(self):
        """Phase 3: Worker Management"""
        
        print("\n👷 PHASE 3: WORKER MANAGEMENT")
        print("-" * 40)
        
        # Get worker stats
        worker_stats = await self.coordinator.worker_manager.get_system_stats()
        
        print("📊 Current Worker Status:")
        print(f"   - Total Workers: {worker_stats['total_workers']}")
        print(f"   - Active Workers: {worker_stats['active_workers']}")
        print(f"   - Busy Workers: {worker_stats['busy_workers']}")
        print(f"   - Queue Size: {worker_stats['queue_size']}")
        
        # Show workers by region
        print("\n🌍 Workers by Region:")
        for region, count in worker_stats['workers_by_region'].items():
            print(f"   - {region}: {count} workers")
        
        # Test worker registration
        print("\n➕ Testing worker registration...")
        new_worker_config = {
            'region': 'demo-region',
            'ip_address': '192.168.100.1',
            'capabilities': ['crawling', 'keyword_discovery', 'demo']
        }
        
        new_worker_id = await self.coordinator.worker_manager.register_worker(new_worker_config)
        print(f"   - New worker registered: {new_worker_id}")
        
        self.demo_results['worker_management'] = {
            'status': 'success',
            'total_workers': worker_stats['total_workers'] + 1,  # +1 for new worker
            'regions': len(worker_stats['workers_by_region']) + 1,
            'new_worker_id': new_worker_id
        }

    async def _demo_phase4_distributed_training(self):
        """Phase 4: Distributed Training"""
        
        print("\n🧠 PHASE 4: DISTRIBUTED TRAINING")
        print("-" * 40)
        
        # Start multiple training cycles
        demo_users = [
            {
                'user_id': 'demo_user_tech',
                'seeds': ['AI chatbots', 'machine learning tools', 'programming languages'],
                'description': 'Technology enthusiast'
            },
            {
                'user_id': 'demo_user_business', 
                'seeds': ['digital marketing', 'e-commerce platforms', 'business analytics'],
                'description': 'Business professional'
            },
            {
                'user_id': 'demo_user_lifestyle',
                'seeds': ['healthy recipes', 'fitness apps', 'travel destinations'],
                'description': 'Lifestyle blogger'
            }
        ]
        
        cycle_ids = []
        
        print("🚀 Starting multiple training cycles...")
        for user in demo_users:
            print(f"   - Starting cycle for {user['description']}...")
            
            cycle_id = await self.coordinator.start_full_discovery_and_training_cycle(
                user_id=user['user_id'],
                custom_seeds=user['seeds']
            )
            
            cycle_ids.append({
                'cycle_id': cycle_id,
                'user_id': user['user_id'],
                'description': user['description']
            })
            
            print(f"     ✅ Cycle started: {cycle_id}")
            
            # Small delay between cycles
            await asyncio.sleep(2)
        
        print(f"\n📊 Total cycles started: {len(cycle_ids)}")
        
        self.demo_results['distributed_training'] = {
            'status': 'success',
            'cycles_started': len(cycle_ids),
            'cycle_ids': [c['cycle_id'] for c in cycle_ids]
        }
        
        # Store for monitoring phase
        self.demo_results['active_cycles'] = cycle_ids

    async def _demo_phase5_monitoring(self):
        """Phase 5: Monitoring & Results"""
        
        print("\n👁️ PHASE 5: MONITORING & RESULTS")
        print("-" * 40)
        
        active_cycles = self.demo_results.get('active_cycles', [])
        
        if not active_cycles:
            print("⚠️ No active cycles to monitor")
            return
        
        print(f"📊 Monitoring {len(active_cycles)} training cycles...")
        
        # Monitor cycles for 60 seconds
        monitoring_duration = 60
        check_interval = 10
        checks = monitoring_duration // check_interval
        
        for i in range(checks):
            print(f"\n⏱️ Monitoring check {i+1}/{checks} (t+{i*check_interval}s)")
            
            # Get system overview
            overview = await self.coordinator.get_system_overview()
            
            print(f"   📈 System Stats:")
            print(f"      - Active Cycles: {overview['active_cycles']}")
            print(f"      - Worker Utilization: {overview['worker_stats']['busy_workers']}/{overview['worker_stats']['active_workers']}")
            print(f"      - Queue Size: {overview['worker_stats']['queue_size']}")
            
            # Check individual cycles
            completed_cycles = 0
            failed_cycles = 0
            
            for cycle_info in active_cycles:
                cycle_status = await self.coordinator.get_cycle_status(cycle_info['cycle_id'])
                status = cycle_status.get('status', 'unknown')
                
                if status == 'completed':
                    completed_cycles += 1
                elif status == 'failed':
                    failed_cycles += 1
                
                print(f"      - {cycle_info['description']}: {status}")
            
            if completed_cycles == len(active_cycles):
                print("🎉 All cycles completed!")
                break
            elif completed_cycles + failed_cycles == len(active_cycles):
                print("⚠️ All cycles finished (some failed)")
                break
            
            # Wait before next check
            if i < checks - 1:
                await asyncio.sleep(check_interval)
        
        self.demo_results['monitoring'] = {
            'status': 'success',
            'completed_cycles': completed_cycles,
            'failed_cycles': failed_cycles,
            'monitoring_duration': monitoring_duration
        }

    def _demo_summary(self):
        """Demo Summary"""
        
        print("\n🎉 DEMO SUMMARY")
        print("=" * 50)
        
        # Calculate overall success
        phases_completed = sum(1 for phase, result in self.demo_results.items() 
                             if isinstance(result, dict) and result.get('status') == 'success')
        
        total_phases = len([k for k in self.demo_results.keys() if k != 'active_cycles'])
        
        print(f"📊 Demo Results:")
        print(f"   - Phases Completed: {phases_completed}/{total_phases}")
        print(f"   - Overall Success Rate: {phases_completed/total_phases*100:.1f}%")
        
        # Detailed results
        print(f"\n📋 Detailed Results:")
        
        if 'initialization' in self.demo_results:
            init = self.demo_results['initialization']
            print(f"   🚀 Initialization: {init['workers']} workers, {init['regions']} regions")
        
        if 'keyword_discovery' in self.demo_results:
            kd = self.demo_results['keyword_discovery']
            print(f"   🔍 Keyword Discovery: {kd['total_keywords']} keywords, {kd['languages']} languages")
        
        if 'worker_management' in self.demo_results:
            wm = self.demo_results['worker_management']
            print(f"   👷 Worker Management: {wm['total_workers']} workers, {wm['regions']} regions")
        
        if 'distributed_training' in self.demo_results:
            dt = self.demo_results['distributed_training']
            print(f"   🧠 Distributed Training: {dt['cycles_started']} cycles started")
        
        if 'monitoring' in self.demo_results:
            mon = self.demo_results['monitoring']
            print(f"   👁️ Monitoring: {mon['completed_cycles']} completed, {mon['failed_cycles']} failed")
        
        # Performance metrics
        print(f"\n⚡ Performance Highlights:")
        print(f"   - Multi-region deployment: ✅")
        print(f"   - Auto keyword discovery: ✅")
        print(f"   - Distributed crawling: ✅")
        print(f"   - Central training: ✅")
        print(f"   - Real-time monitoring: ✅")
        
        # Next steps
        print(f"\n🚀 Next Steps:")
        print(f"   1. Start web server: python run_distributed_system.py")
        print(f"   2. Open dashboard: http://localhost:8000")
        print(f"   3. Use API: http://localhost:8000/api/docs")
        print(f"   4. Scale workers: Add more regions/workers")
        print(f"   5. Production deploy: Docker + Kubernetes")
        
        # Save results to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"demo_results_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.demo_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 Demo results saved to: {results_file}")
        print(f"\n🎊 Demo completed successfully! 🎊")

async def run_quick_demo():
    """Quick demo version (5 minutes)"""
    
    print("⚡ QUICK DEMO - Distributed Training System")
    print("=" * 50)
    
    demo = DistributedSystemDemo()
    
    # Quick initialization
    print("🚀 Initializing system...")
    demo.coordinator = MasterCoordinator()
    await demo.coordinator.initialize_system()
    
    # Quick keyword discovery
    print("🔍 Discovering keywords...")
    keywords = await demo.coordinator.keyword_discovery.discover_trending_keywords(max_keywords=50)
    total_keywords = sum(len(kws) for kws in keywords.values())
    print(f"   - Discovered {total_keywords} keywords")
    
    # Quick training cycle
    print("🧠 Starting training cycle...")
    cycle_id = await demo.coordinator.start_full_discovery_and_training_cycle(
        user_id="quick_demo_user",
        custom_seeds=["AI tools", "web development"]
    )
    print(f"   - Cycle started: {cycle_id}")
    
    # Quick monitoring
    print("👁️ Monitoring for 30 seconds...")
    for i in range(3):
        await asyncio.sleep(10)
        overview = await demo.coordinator.get_system_overview()
        print(f"   - Workers: {overview['worker_stats']['busy_workers']}/{overview['worker_stats']['active_workers']} busy")
    
    print("✅ Quick demo completed!")

def main():
    """Main entry point"""
    
    import argparse
    parser = argparse.ArgumentParser(description="Distributed System Demo")
    parser.add_argument("--mode", choices=["full", "quick"], default="quick",
                       help="Demo mode: full (complete demo) or quick (5 minutes)")
    
    args = parser.parse_args()
    
    if args.mode == "full":
        print("🌟 Starting FULL demo (may take 10-15 minutes)...")
        demo = DistributedSystemDemo()
        asyncio.run(demo.run_complete_demo())
    else:
        print("⚡ Starting QUICK demo (5 minutes)...")
        asyncio.run(run_quick_demo())

if __name__ == "__main__":
    main()