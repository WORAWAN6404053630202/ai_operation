"""
Production resilience utilities for Thai Regulatory AI.

Features:
- Circuit breaker pattern
- Retry logic with exponential backoff
- Request timeout handling
- Graceful shutdown
- Health checks (liveness, readiness)
- Error recovery mechanisms

Usage:
    from code.utils.resilience import CircuitBreaker, retry_with_backoff, HealthCheck
    
    # Circuit breaker
    breaker = CircuitBreaker(failure_threshold=5, timeout=60)
    
    @breaker
    def risky_operation():
        return external_api_call()
    
    # Retry with backoff
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def flaky_function():
        return may_fail()
    
    # Health checks
    health = HealthCheck()
    health.add_check("database", check_database)
    health.add_check("llm", check_llm)
    status = health.check_all()
"""

import time
import asyncio
import logging
import functools
from enum import Enum
from typing import Callable, Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import signal
import sys


logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes to close from half-open
    timeout: int = 60  # Seconds before trying half-open
    


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.
    
    Prevents cascading failures by failing fast when a service is down.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: int = 60
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Failures before opening circuit
            success_threshold: Successes to close from half-open
            timeout: Seconds before attempting recovery
        """
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call function through circuit breaker.
        
        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try recovery."""
        if self.last_failure_time is None:
            return True
        
        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.timeout
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                logger.info("Circuit breaker CLOSED")
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning("Circuit breaker reopened to OPEN")
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker OPEN after {self.failure_count} failures")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator usage."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Retry decorator with exponential backoff.
    
    Args:
        max_retries: Maximum retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff
        exceptions: Exceptions to catch and retry
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        logger.error(f"{func.__name__} failed after {max_retries} retries")
                        raise e
                    
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}). "
                        f"Retrying in {delay:.2f}s... Error: {e}"
                    )
                    time.sleep(delay)
            
        return wrapper
    return decorator


def async_retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Async retry decorator with exponential backoff.
    
    Args:
        max_retries: Maximum retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff
        exceptions: Exceptions to catch and retry
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        logger.error(f"{func.__name__} failed after {max_retries} retries")
                        raise e
                    
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}). "
                        f"Retrying in {delay:.2f}s... Error: {e}"
                    )
                    await asyncio.sleep(delay)
            
        return wrapper
    return decorator


@dataclass
class HealthCheckResult:
    """Health check result."""
    name: str
    healthy: bool
    message: str = ""
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class HealthCheck:
    """
    Health check system for liveness and readiness probes.
    """
    
    def __init__(self):
        """Initialize health check."""
        self.checks: Dict[str, Callable] = {}
    
    def add_check(self, name: str, check_func: Callable):
        """
        Add health check.
        
        Args:
            name: Check name
            check_func: Function that returns (healthy: bool, message: str)
        """
        self.checks[name] = check_func
        logger.info(f"Added health check: {name}")
    
    def check_all(self) -> Dict[str, Any]:
        """
        Run all health checks.
        
        Returns:
            Dict with overall status and individual results
        """
        results: List[HealthCheckResult] = []
        
        for name, check_func in self.checks.items():
            start = time.time()
            try:
                healthy, message = check_func()
                latency_ms = (time.time() - start) * 1000
                
                results.append(HealthCheckResult(
                    name=name,
                    healthy=healthy,
                    message=message,
                    latency_ms=latency_ms
                ))
            except Exception as e:
                latency_ms = (time.time() - start) * 1000
                results.append(HealthCheckResult(
                    name=name,
                    healthy=False,
                    message=str(e),
                    latency_ms=latency_ms
                ))
        
        # Overall status
        all_healthy = all(r.healthy for r in results)
        
        return {
            "healthy": all_healthy,
            "timestamp": datetime.now().isoformat(),
            "checks": [
                {
                    "name": r.name,
                    "healthy": r.healthy,
                    "message": r.message,
                    "latency_ms": round(r.latency_ms, 2)
                }
                for r in results
            ]
        }
    
    def liveness(self) -> bool:
        """
        Liveness probe - is the app running?
        
        Returns:
            True if alive
        """
        # Simple check - if we can respond, we're alive
        return True
    
    def readiness(self) -> Dict[str, Any]:
        """
        Readiness probe - is the app ready to serve traffic?
        
        Returns:
            Readiness status
        """
        return self.check_all()


class GracefulShutdown:
    """
    Handle graceful shutdown on SIGTERM/SIGINT.
    """
    
    def __init__(self):
        """Initialize shutdown handler."""
        self.shutdown_requested = False
        self.cleanup_callbacks: List[Callable] = []
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def register_cleanup(self, callback: Callable):
        """
        Register cleanup callback.
        
        Args:
            callback: Function to call on shutdown
        """
        self.cleanup_callbacks.append(callback)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
        self._cleanup()
        sys.exit(0)
    
    def _cleanup(self):
        """Run cleanup callbacks."""
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in cleanup callback: {e}")


# Example health check functions
def check_llm_health() -> tuple[bool, str]:
    """Check if LLM is available."""
    try:
        # Add your LLM health check here
        # Example: call a simple endpoint
        return True, "LLM service is healthy"
    except Exception as e:
        return False, f"LLM service error: {e}"


def check_database_health() -> tuple[bool, str]:
    """Check if database is available."""
    try:
        # Add your database health check here
        # Example: query a simple table
        return True, "Database is healthy"
    except Exception as e:
        return False, f"Database error: {e}"


def check_vector_store_health() -> tuple[bool, str]:
    """Check if vector store is available."""
    try:
        # Add your vector store health check here
        return True, "Vector store is healthy"
    except Exception as e:
        return False, f"Vector store error: {e}"


if __name__ == "__main__":
    # Example usage
    
    # Circuit breaker
    breaker = CircuitBreaker(failure_threshold=3, timeout=5)
    
    @breaker
    def flaky_operation():
        import random
        if random.random() < 0.5:
            raise Exception("Random failure")
        return "Success"
    
    # Retry with backoff
    @retry_with_backoff(max_retries=3, base_delay=0.5)
    def retry_example():
        print("Attempting operation...")
        return "Done"
    
    # Health checks
    health = HealthCheck()
    health.add_check("llm", check_llm_health)
    health.add_check("database", check_database_health)
    health.add_check("vector_store", check_vector_store_health)
    
    status = health.check_all()
    print(f"Health status: {status}")
