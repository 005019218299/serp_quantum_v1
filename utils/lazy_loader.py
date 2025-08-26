import asyncio
import hashlib
import time
from typing import Dict, Callable, Any
from functools import wraps
from .constants import CACHE_TTL_SECONDS, MAX_CACHE_SIZE

class LazyLoader:
    """Efficient lazy loading with caching"""
    
    def __init__(self):
        self._components = {}
        self._instances = {}
        self._cache = {}
        self._cache_times = {}
    
    def register_component(self, name: str, factory: Callable):
        """Register component factory"""
        self._components[name] = factory
    
    async def get_component(self, name: str):
        """Get component with lazy loading"""
        if name not in self._instances:
            if name not in self._components:
                raise ValueError(f"Component {name} not registered")
            self._instances[name] = self._components[name]()
        return self._instances[name]

class Calculator:
    """Utility calculator functions"""
    
    def safe_divide(self, a: float, b: float, default: float = 0.0) -> float:
        return a / b if b != 0 else default

class Hasher:
    """String hashing utilities"""
    
    def hash_string(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

def cached_result(ttl: int = CACHE_TTL_SECONDS):
    """Decorator for caching function results"""
    def decorator(func):
        cache = {}
        cache_times = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(sorted(kwargs.items()))
            current_time = time.time()
            
            if key in cache and (current_time - cache_times[key]) < ttl:
                return cache[key]
            
            result = func(*args, **kwargs)
            
            # Simple cache size management
            if len(cache) >= MAX_CACHE_SIZE:
                oldest_key = min(cache_times.keys(), key=lambda k: cache_times[k])
                del cache[oldest_key]
                del cache_times[oldest_key]
            
            cache[key] = result
            cache_times[key] = current_time
            return result
        
        return wrapper
    return decorator

lazy_loader = LazyLoader()
calculator = Calculator()
hasher = Hasher()