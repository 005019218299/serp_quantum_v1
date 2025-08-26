import asyncio
import concurrent.futures
from typing import Callable, Any
from .constants import MAX_RETRIES, RETRY_DELAY

class AsyncManager:
    """Proper async management with error recovery"""
    
    def __init__(self):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    
    async def run_in_executor(self, func: Callable, *args, **kwargs) -> Any:
        """Run blocking function in executor with retry logic"""
        for attempt in range(MAX_RETRIES):
            try:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(self.executor, func, *args, **kwargs)
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise e
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
    
    async def gather_with_fallback(self, *coroutines, fallback_values=None):
        """Gather coroutines with fallback values on error"""
        results = []
        fallback_values = fallback_values or [None] * len(coroutines)
        
        for i, coro in enumerate(coroutines):
            try:
                result = await coro
                results.append(result)
            except Exception:
                results.append(fallback_values[i])
        
        return results

async_manager = AsyncManager()