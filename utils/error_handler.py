import time
import logging
from functools import wraps
from typing import Callable, Any, Dict
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from ..core.constants import MAX_RETRIES, RETRY_DELAY

logger = logging.getLogger(__name__)

class CircuitBreaker:
    """Circuit breaker pattern for error recovery"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise e

def safe_execute(operation_name: str, max_retries: int = MAX_RETRIES):
    """Decorator for safe execution with retries"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"{operation_name} attempt {attempt + 1} failed: {str(e)}")
                    
                    if attempt < max_retries - 1:
                        await asyncio.sleep(RETRY_DELAY * (attempt + 1))
            
            logger.error(f"{operation_name} failed after {max_retries} attempts")
            raise HTTPException(
                status_code=500,
                detail=f"Operation {operation_name} failed: {str(last_exception)}"
            )
        
        return wrapper
    return decorator

def log_performance(operation_name: str):
    """Decorator to log performance metrics"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = (time.time() - start_time) * 1000
                logger.info(f"{operation_name} completed in {duration:.2f}ms")
                return result
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                logger.error(f"{operation_name} failed after {duration:.2f}ms: {str(e)}")
                raise
        return wrapper
    return decorator

async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with proper error responses"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail, "type": "http_exception"}
        )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "type": "internal_error",
            "message": "An unexpected error occurred"
        }
    )

circuit_breaker = CircuitBreaker()