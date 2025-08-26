"""
Model Manager - Fix AI/ML issues with proper model handling
"""
import torch
import numpy as np
from typing import Dict, Optional
import threading

class ModelManager:
    def __init__(self):
        self._models = {}
        self._lock = threading.Lock()
        
        # Set reproducible seeds
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
    
    def get_model(self, model_name: str, model_class, **kwargs):
        """Get model with lazy loading and proper inference mode"""
        with self._lock:
            if model_name not in self._models:
                model = model_class(**kwargs)
                if hasattr(model, 'eval'):
                    model.eval()  # Set to evaluation mode
                    if hasattr(model, 'requires_grad_'):
                        for param in model.parameters():
                            param.requires_grad_(False)  # Disable gradients for inference
                self._models[model_name] = model
            
            return self._models[model_name]
    
    def validate_prediction(self, prediction: Dict) -> Dict:
        """Validate model predictions"""
        validated = {}
        
        for key, value in prediction.items():
            if isinstance(value, (int, float)):
                # Check for NaN or infinite values
                if np.isnan(value) or np.isinf(value):
                    validated[key] = 0.0
                else:
                    validated[key] = float(value)
            else:
                validated[key] = value
        
        return validated

model_manager = ModelManager()