import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple
import joblib
import os
from datetime import datetime

class TemporalSERPPredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Remove dropout for inference
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, input_size)
        )
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use last time step
        last_out = attn_out[:, -1, :]
        return self.fc(last_out)

class SERPPredictor:
    def __init__(self, model_path: str = "./models"):
        # Set reproducible seeds
        torch.manual_seed(42)
        np.random.seed(42)
        
        self.model_path = model_path
        self.scaler = StandardScaler()
        self.feature_names = [
            'serp_complexity_score', 'has_featured_snippet', 'has_video_carousel',
            'has_people_also_ask', 'local_pack_presence', 'total_ads_count',
            'knowledge_graph_presence', 'paa_questions_count', 'organic_results_count',
            'top_domain_concentration', 'top_3_domains_ratio', 'unique_domains_count',
            'domain_diversity_score', 'hour_of_day', 'day_of_week'
        ]
        self.input_size = len(self.feature_names)
        self.model = TemporalSERPPredictor(self.input_size)
        self.model.eval()  # Set to evaluation mode
        self.is_trained = False
        self.device = torch.device('cpu')  # Use CPU for stability
        
        self._load_model()
        
    def prepare_sequences(self, features_list: List[Dict], sequence_length: int = 7) -> Tuple[np.ndarray, np.ndarray]:
        """Chuẩn bị chuỗi thời gian cho training"""
        if not features_list or len(features_list) < sequence_length + 1:
            return np.array([]), np.array([])
        
        try:
            # Vectorized feature extraction for better performance
            feature_arrays = np.array([[features.get(name, 0) for name in self.feature_names] 
                                     for features in features_list])
            
            # Normalize features
            features_normalized = self.scaler.fit_transform(feature_arrays)
            
            # Vectorized sequence creation
            num_sequences = len(features_normalized) - sequence_length
            sequences = np.array([features_normalized[i:i+sequence_length] for i in range(num_sequences)])
            targets = features_normalized[sequence_length:]
            
            return sequences, targets
        except Exception as e:
            print(f"Error preparing sequences: {e}")
            return np.array([]), np.array([])
    
    def train(self, sequences: np.ndarray, targets: np.ndarray, epochs: int = 100, validation_split: float = 0.2):
        """Train model with proper validation and early stopping"""
        if len(sequences) == 0:
            print("❌ Insufficient data for training")
            return False
        
        if len(sequences) < 20:
            print(f"⚠️  Warning: Only {len(sequences)} training samples. Recommend at least 50 for reliable training.")
        
        # Split data for validation
        val_size = int(len(sequences) * validation_split)
        if val_size > 0:
            train_sequences = sequences[:-val_size]
            train_targets = targets[:-val_size]
            val_sequences = sequences[-val_size:]
            val_targets = targets[-val_size:]
        else:
            train_sequences = sequences
            train_targets = targets
            val_sequences = None
            val_targets = None
        
        # Convert to tensors
        train_seq_tensor = torch.FloatTensor(train_sequences).to(self.device)
        train_tgt_tensor = torch.FloatTensor(train_targets).to(self.device)
        
        if val_sequences is not None:
            val_seq_tensor = torch.FloatTensor(val_sequences).to(self.device)
            val_tgt_tensor = torch.FloatTensor(val_targets).to(self.device)
        
        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        print(f"🚀 Starting training with {len(train_sequences)} samples...")
        
        self.model.train()
        for epoch in range(epochs):
            # Training step
            optimizer.zero_grad()
            outputs = self.model(train_seq_tensor)
            train_loss = criterion(outputs, train_tgt_tensor)
            train_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Validation step
            if val_sequences is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(val_seq_tensor)
                    val_loss = criterion(val_outputs, val_tgt_tensor)
                
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), f"{self.model_path}/best_temporal_model.pth")
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"🛑 Early stopping at epoch {epoch}")
                    break
                
                self.model.train()
            
            # Logging
            if epoch % 20 == 0:
                if val_sequences is not None:
                    print(f'Epoch {epoch:3d} | Train Loss: {train_loss.item():.4f} | Val Loss: {val_loss.item():.4f}')
                else:
                    print(f'Epoch {epoch:3d} | Train Loss: {train_loss.item():.4f}')
        
        # Load best model if validation was used
        if val_sequences is not None and os.path.exists(f"{self.model_path}/best_temporal_model.pth"):
            self.model.load_state_dict(torch.load(f"{self.model_path}/best_temporal_model.pth", map_location=self.device))
        
        self.is_trained = True
        self._save_model()
        
        print(f"✅ Training completed. Final loss: {train_loss.item():.4f}")
        return True
    
    def predict_future_serp(self, current_features: Dict, days_ahead: int = 30) -> List[Dict]:
        """Real SERP prediction with confidence scoring"""
        if not self.is_trained:
            # Return realistic baseline predictions when not trained
            return self._generate_baseline_predictions(current_features, days_ahead)
        
        self.model.eval()
        
        try:
            # Prepare current features with validation
            feature_array = [current_features.get(name, 0) for name in self.feature_names]
            
            # Validate input features
            if not self._validate_features(feature_array):
                return self._generate_baseline_predictions(current_features, days_ahead)
            
            feature_normalized = self.scaler.transform([feature_array])
            
            predictions = []
            current_seq = np.repeat(feature_normalized, 7, axis=0).reshape(1, 7, -1)
            
            with torch.no_grad():
                for day in range(days_ahead):
                    seq_tensor = torch.FloatTensor(current_seq).to(self.device)
                    pred = self.model(seq_tensor)
                    pred_numpy = pred.cpu().numpy()
                    
                    # Validate prediction and handle edge cases
                    if np.any(np.isnan(pred_numpy)) or np.any(np.isinf(pred_numpy)):
                        pred_numpy = feature_normalized[0].reshape(1, -1)  # Fallback to current state
                    
                    pred_denormalized = self.scaler.inverse_transform(pred_numpy)
                    
                    # Apply realistic constraints
                    pred_denormalized = self._apply_realistic_constraints(pred_denormalized[0])
                    
                    # Convert to feature dict with confidence
                    pred_features = dict(zip(self.feature_names, pred_denormalized))
                    
                    # Add prediction metadata
                    pred_features['prediction_confidence'] = self._calculate_prediction_confidence(day, days_ahead)
                    pred_features['prediction_day'] = day + 1
                    pred_features['model_trained'] = True
                    
                    predictions.append(pred_features)
                    
                    # Update sequence for next prediction
                    current_seq = np.roll(current_seq, -1, axis=1)
                    current_seq[0, -1] = pred_numpy[0]
            
            return predictions
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return self._generate_baseline_predictions(current_features, days_ahead)
    
    def _generate_baseline_predictions(self, current_features: Dict, days_ahead: int) -> List[Dict]:
        """Generate realistic baseline predictions when model is not trained"""
        predictions = []
        
        for day in range(days_ahead):
            # Create realistic variations of current features
            pred_features = current_features.copy()
            
            # Add small random variations (±5%)
            for feature in self.feature_names:
                if feature in pred_features:
                    current_val = pred_features[feature]
                    if isinstance(current_val, (int, float)):
                        variation = np.random.normal(0, 0.05)  # 5% std deviation
                        pred_features[feature] = max(0, current_val * (1 + variation))
            
            # Add prediction metadata
            pred_features['prediction_confidence'] = max(0.3 - (day * 0.01), 0.1)  # Decreasing confidence
            pred_features['prediction_day'] = day + 1
            pred_features['model_trained'] = False
            
            predictions.append(pred_features)
        
        return predictions
    
    def _validate_features(self, feature_array: List[float]) -> bool:
        """Validate input features"""
        if len(feature_array) != len(self.feature_names):
            return False
        
        # Check for reasonable values
        for val in feature_array:
            if not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
                return False
            if val < 0 or val > 1000:  # Reasonable bounds
                return False
        
        return True
    
    def _apply_realistic_constraints(self, predictions: np.ndarray) -> np.ndarray:
        """Apply realistic constraints to predictions"""
        # Ensure binary features stay binary
        binary_features = ['has_featured_snippet', 'has_video_carousel', 'has_people_also_ask', 
                          'local_pack_presence', 'knowledge_graph_presence']
        
        for i, feature_name in enumerate(self.feature_names):
            if feature_name in binary_features:
                predictions[i] = 1 if predictions[i] > 0.5 else 0
            elif feature_name.endswith('_count'):
                predictions[i] = max(0, int(predictions[i]))  # Non-negative integers
            elif feature_name.endswith('_score') or feature_name.endswith('_ratio'):
                predictions[i] = max(0, min(1, predictions[i]))  # 0-1 range
        
        return predictions
    
    def _calculate_prediction_confidence(self, day: int, total_days: int) -> float:
        """Calculate realistic prediction confidence"""
        # Confidence decreases with prediction horizon
        base_confidence = 0.8 if self.is_trained else 0.3
        decay_rate = 0.02  # 2% decrease per day
        
        confidence = base_confidence * (1 - decay_rate * day)
        return max(0.1, confidence)  # Minimum 10% confidence
    
    def _save_model(self):
        """Save trained model with metadata"""
        os.makedirs(self.model_path, exist_ok=True)
        
        # Save model state
        torch.save(self.model.state_dict(), f"{self.model_path}/temporal_model.pth")
        
        # Save scaler
        joblib.dump(self.scaler, f"{self.model_path}/scaler.pkl")
        
        # Save model metadata
        metadata = {
            "model_type": "TemporalSERPPredictor",
            "input_size": self.input_size,
            "feature_names": self.feature_names,
            "trained_at": datetime.now().isoformat(),
            "pytorch_version": torch.__version__,
            "model_parameters": sum(p.numel() for p in self.model.parameters())
        }
        
        import json
        with open(f"{self.model_path}/model_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Model saved to {self.model_path}")
    
    def _load_model(self):
        """Load trained model with validation"""
        model_file = f"{self.model_path}/temporal_model.pth"
        scaler_file = f"{self.model_path}/scaler.pkl"
        
        try:
            if os.path.exists(model_file) and os.path.exists(scaler_file):
                # Load model state
                state_dict = torch.load(model_file, map_location=self.device)
                self.model.load_state_dict(state_dict)
                
                # Load scaler
                self.scaler = joblib.load(scaler_file)
                
                # Validate loaded model
                if self._validate_loaded_model():
                    self.is_trained = True
                    print("✅ Temporal prediction model loaded successfully")
                else:
                    self.is_trained = False
                    print("❌ Loaded model failed validation")
            else:
                self.is_trained = False
                print("⚠️  No trained model found - using baseline predictions")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.is_trained = False
    
    def _validate_loaded_model(self) -> bool:
        """Validate that loaded model works correctly"""
        try:
            # Test with dummy data
            dummy_input = torch.randn(1, 7, self.input_size).to(self.device)
            
            with torch.no_grad():
                output = self.model(dummy_input)
                
            # Check output shape and values
            if output.shape != (1, self.input_size):
                return False
            
            if torch.any(torch.isnan(output)) or torch.any(torch.isinf(output)):
                return False
            
            return True
        except Exception:
            return False
    
    def get_model_info(self) -> Dict:
        """Get information about the current model state"""
        return {
            "is_trained": self.is_trained,
            "model_path": self.model_path,
            "input_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "device": str(self.device),
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
            "model_size_mb": sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024 / 1024
        }