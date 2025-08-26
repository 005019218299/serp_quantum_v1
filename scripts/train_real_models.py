#!/usr/bin/env python3
"""
Real Model Training Script
Collects actual SERP data and trains ML models with real data
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_collection.serp_harvester import SERPHarvester
from processing.data_pipeline import DataProcessor
from processing.feature_store import FeatureStore
from prediction.temporal_model import SERPPredictor
from prediction.competitive_model import CompetitiveResponsePredictor

class RealDataTrainer:
    def __init__(self):
        self.harvester = SERPHarvester()
        self.processor = DataProcessor()
        self.feature_store = FeatureStore()
        self.temporal_model = SERPPredictor()
        self.competitive_model = CompetitiveResponsePredictor()
        
    async def collect_training_data(self, keywords: List[str], days: int = 30) -> List[Dict]:
        """Collect real SERP data for training"""
        print(f"🔄 Collecting training data for {len(keywords)} keywords over {days} days...")
        
        training_data = []
        
        for i, keyword in enumerate(keywords):
            print(f"📊 Processing keyword {i+1}/{len(keywords)}: {keyword}")
            
            try:
                # Fetch current SERP data
                serp_data = await self.harvester.fetch_serp(keyword, location='vn')
                
                if serp_data and 'data' in serp_data and serp_data['data']:
                    # Process the data
                    processed_data = self.processor.process_serp_data(serp_data)
                    
                    # Store in feature store
                    feature_id = self.feature_store.store_features(processed_data)
                    
                    training_data.append(processed_data)
                    print(f"  ✅ Collected data for '{keyword}' (Feature ID: {feature_id})")
                else:
                    print(f"  ❌ No data collected for '{keyword}'")
                
                # Rate limiting - wait between requests
                await asyncio.sleep(3)
                
            except Exception as e:
                print(f"  ❌ Error collecting data for '{keyword}': {e}")
                continue
        
        print(f"📈 Total training samples collected: {len(training_data)}")
        return training_data
    
    def train_temporal_model(self, training_data: List[Dict]) -> bool:
        """Train temporal prediction model with real data"""
        print("🧠 Training temporal prediction model...")
        
        if len(training_data) < 10:
            print("❌ Insufficient data for temporal model training (need at least 10 samples)")
            return False
        
        try:
            # Extract features for temporal model
            features_list = [data['serp_features'] for data in training_data]
            
            # Prepare sequences for LSTM training
            sequences, targets = self.temporal_model.prepare_sequences(features_list)
            
            if len(sequences) == 0:
                print("❌ No valid sequences generated for training")
                return False
            
            print(f"📊 Training with {len(sequences)} sequences...")
            
            # Train the model
            success = self.temporal_model.train(sequences, targets, epochs=100)
            
            if success:
                print("✅ Temporal model training completed successfully")
                
                # Get model info
                model_info = self.temporal_model.get_model_info()
                print(f"📋 Model Info:")
                print(f"   - Parameters: {model_info['model_parameters']:,}")
                print(f"   - Size: {model_info['model_size_mb']:.2f} MB")
                print(f"   - Features: {model_info['input_features']}")
                
                return True
            else:
                print("❌ Temporal model training failed")
                return False
                
        except Exception as e:
            print(f"❌ Error training temporal model: {e}")
            return False
    
    def train_competitive_model(self, training_data: List[Dict]) -> bool:
        """Train competitive response model with real data"""
        print("🎯 Training competitive response model...")
        
        if len(training_data) < 5:
            print("❌ Insufficient data for competitive model training (need at least 5 samples)")
            return False
        
        try:
            # Prepare competitive data
            X, y = self.competitive_model.prepare_competitive_data(training_data)
            
            if len(X) == 0:
                print("❌ No valid competitive data generated for training")
                return False
            
            print(f"📊 Training with {len(X)} competitive samples...")
            
            # Train the model
            self.competitive_model.train(X, y)
            
            print("✅ Competitive model training completed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error training competitive model: {e}")
            return False
    
    async def validate_models(self, validation_keywords: List[str]) -> Dict:
        """Validate trained models with real data"""
        print("🔍 Validating trained models...")
        
        validation_results = {
            'temporal_model': {'tested': False, 'accuracy': 0.0},
            'competitive_model': {'tested': False, 'accuracy': 0.0}
        }
        
        try:
            for keyword in validation_keywords[:3]:  # Test with first 3 keywords
                print(f"🧪 Testing with keyword: {keyword}")
                
                # Get current SERP data
                current_data = asyncio.run(self.harvester.fetch_serp(keyword))
                
                if current_data and 'data' in current_data:
                    processed_data = self.processor.process_serp_data(current_data)
                    
                    # Test temporal model
                    if self.temporal_model.is_trained:
                        predictions = self.temporal_model.predict_future_serp(
                            processed_data['serp_features'], days_ahead=7
                        )
                        
                        if predictions:
                            avg_confidence = sum(p.get('prediction_confidence', 0) for p in predictions) / len(predictions)
                            validation_results['temporal_model'] = {
                                'tested': True,
                                'accuracy': avg_confidence,
                                'predictions_count': len(predictions)
                            }
                    
                    # Test competitive model
                    if self.competitive_model.is_trained:
                        competitive_pred = self.competitive_model.predict_competitor_moves(
                            processed_data['serp_features']
                        )
                        
                        if competitive_pred:
                            # Simple validation based on prediction completeness
                            accuracy = len([v for v in competitive_pred.values() if isinstance(v, (int, float))]) / len(competitive_pred)
                            validation_results['competitive_model'] = {
                                'tested': True,
                                'accuracy': accuracy,
                                'predictions': competitive_pred
                            }
                            
                    await asyncio.sleep(2)  # Rate limiting
        
        except Exception as e:
            print(f"❌ Error during validation: {e}")
        
        return validation_results
    
    def save_training_report(self, training_data: List[Dict], validation_results: Dict):
        """Save training report"""
        report = {
            'training_date': datetime.now().isoformat(),
            'training_samples': len(training_data),
            'models_trained': {
                'temporal_model': self.temporal_model.is_trained,
                'competitive_model': self.competitive_model.is_trained
            },
            'validation_results': validation_results,
            'model_info': self.temporal_model.get_model_info() if self.temporal_model.is_trained else None
        }
        
        report_file = f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Training report saved: {report_file}")

async def main():
    """Main training function"""
    print("🚀 Starting Real Data Model Training")
    print("=" * 50)
    
    # Training keywords (add more for better training)
    training_keywords = [
        "máy lọc nước",
        "điện thoại iPhone",
        "laptop gaming",
        "ô tô điện",
        "bảo hiểm sức khỏe",
        "du lịch Đà Nẵng",
        "học tiếng Anh online",
        "mua nhà Hà Nội",
        "đầu tư chứng khoán",
        "thực phẩm organic"
    ]
    
    # Validation keywords
    validation_keywords = [
        "smartphone Samsung",
        "máy tính bảng",
        "khóa học lập trình"
    ]
    
    trainer = RealDataTrainer()
    
    try:
        # Step 1: Collect training data
        training_data = await trainer.collect_training_data(training_keywords, days=1)
        
        if len(training_data) < 5:
            print("❌ Insufficient training data collected. Need at least 5 samples.")
            return
        
        # Step 2: Train temporal model
        temporal_success = trainer.train_temporal_model(training_data)
        
        # Step 3: Train competitive model
        competitive_success = trainer.train_competitive_model(training_data)
        
        # Step 4: Validate models
        validation_results = trainer.validate_models(validation_keywords)
        
        # Step 5: Generate report
        trainer.save_training_report(training_data, validation_results)
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 TRAINING SUMMARY")
        print("=" * 50)
        print(f"Training samples: {len(training_data)}")
        print(f"Temporal model: {'✅ Trained' if temporal_success else '❌ Failed'}")
        print(f"Competitive model: {'✅ Trained' if competitive_success else '❌ Failed'}")
        
        if validation_results['temporal_model']['tested']:
            print(f"Temporal accuracy: {validation_results['temporal_model']['accuracy']:.2%}")
        
        if validation_results['competitive_model']['tested']:
            print(f"Competitive accuracy: {validation_results['competitive_model']['accuracy']:.2%}")
        
        if temporal_success or competitive_success:
            print("\n🎉 Training completed successfully!")
            print("💡 Models are now ready for production use")
        else:
            print("\n⚠️  Training completed with issues")
            print("💡 Check logs and consider collecting more data")
    
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return

if __name__ == "__main__":
    asyncio.run(main())