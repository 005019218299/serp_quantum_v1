import os
import threading
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from typing import Dict, Optional

class LazyModelLoader:
    """Lazy loading models chỉ khi cần thiết"""
    
    def __init__(self):
        self._models = {}
        self._tokenizers = {}
        self._lock = threading.Lock()
        self.cache_dir = os.getenv('HF_HOME', '/app/hf_cache')
    
    def get_model_and_tokenizer(self, model_name: str, model_type: str = 'bert'):
        """Load model và tokenizer chỉ khi được gọi"""
        with self._lock:
            if model_name not in self._models:
                print(f"Loading {model_name} for first time...")
                
                try:
                    if model_type == 'bert':
                        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.cache_dir)
                        model = AutoModel.from_pretrained(model_name, cache_dir=self.cache_dir)
                    elif model_type == 't5':
                        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.cache_dir)
                        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=self.cache_dir)
                    else:
                        raise ValueError(f"Unsupported model type: {model_type}")
                    
                    self._tokenizers[model_name] = tokenizer
                    self._models[model_name] = model
                    print(f"✅ {model_name} loaded successfully")
                    
                except Exception as e:
                    print(f"❌ Failed to load {model_name}: {e}")
                    return None, None
            
            return self._models.get(model_name), self._tokenizers.get(model_name)

# Global instance
lazy_loader = LazyModelLoader()