#!/usr/bin/env python3
"""
Quick Training Script - 5 phút setup
Chạy nhanh để test enhanced training system
"""

import asyncio
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.enhanced_train_models import EnhancedTrainer

async def quick_train():
    """Quick training với 3 keywords để test"""
    
    print("⚡ QUICK TRAINING - 5 MINUTE SETUP")
    print("=" * 50)
    
    # Quick keywords for testing
    quick_keywords = [
        "máy lọc nước",
        "điện thoại iPhone", 
        "laptop gaming"
    ]
    
    trainer = EnhancedTrainer()
    
    try:
        print("🚀 Starting quick training...")
        
        # Run quick training
        report = await trainer.enhanced_training_pipeline(quick_keywords)
        
        # Quick summary
        print("\n⚡ QUICK TRAINING RESULTS:")
        print("-" * 30)
        
        if report.get('performance_summary', {}).get('overall_success'):
            print("✅ Status: SUCCESS")
            print(f"📊 Samples: {report['data_collection']['training_samples']}")
            print(f"🎯 Success Rate: {report['data_collection']['success_rate_percent']:.1f}%")
            print(f"⏱️  Duration: {report['training_session']['duration_minutes']:.1f} min")
            print("🚀 Models ready for production!")
        else:
            print("❌ Status: FAILED")
            print("💡 Try running full training with more keywords")
        
        return report
        
    except Exception as e:
        print(f"❌ Quick training failed: {e}")
        return None

if __name__ == "__main__":
    asyncio.run(quick_train())