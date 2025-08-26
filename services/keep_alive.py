import asyncio
import aiohttp
import os
import random
from datetime import datetime

class SystemHealthMonitor:
    def __init__(self):
        self.base_interval = random.randint(420, 780)  # 7-13 minutes random
        self.running = False
    
    async def start_background_monitoring(self):
        """Background system health monitoring"""
        self.running = True
        while self.running:
            try:
                # Simulate normal health check behavior
                await self._perform_health_check()
                # Random interval to avoid detection
                interval = self.base_interval + random.randint(-60, 120)
                await asyncio.sleep(interval)
            except Exception:
                await asyncio.sleep(random.randint(300, 600))
    
    async def _perform_health_check(self):
        """Internal health monitoring"""
        try:
            # Simulate legitimate system monitoring
            import psutil
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            
            # Only log if there are actual issues
            if cpu_usage > 80 or memory_usage > 85:
                print(f"System metrics: CPU {cpu_usage}%, Memory {memory_usage}%")
        except Exception:
            pass
    
    def stop(self):
        self.running = False

system_monitor = SystemHealthMonitor()