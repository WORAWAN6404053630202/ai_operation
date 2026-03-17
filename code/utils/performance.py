"""
Performance optimization utilities for Thai Regulatory AI.

Features:
- Connection pooling
- Batch processing
- Prefetching strategies
- Lazy loading
- Query optimization
- Benchmarking tools

Usage:
    from code.utils.performance import (
        batch_process, ConnectionPool, QueryOptimizer, Benchmark
    )
    
    # Batch processing
    results = batch_process(items, process_func, batch_size=10)
    
    # Connection pool
    pool = ConnectionPool(max_connections=50)
    with pool.get_connection() as conn:
        # Use connection
        pass
    
    # Benchmarking
    with Benchmark("operation_name"):
        # Code to benchmark
        expensive_operation()
"""

import time
import logging
from typing import List, Dict, Any, Callable, Optional, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from collections import defaultdict
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import statistics


logger = logging.getLogger(__name__)


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def batch_process(
    items: List[Any],
    process_func: Callable,
    batch_size: int = 10,
    parallel: bool = False,
    max_workers: int = 4
) -> List[Any]:
    """
    Process items in batches.
    
    Args:
        items: Items to process
        process_func: Function to process each item
        batch_size: Batch size
        parallel: Use parallel processing
        max_workers: Number of parallel workers
        
    Returns:
        List of results
    """
    results = []
    
    if parallel:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                batch_results = list(executor.map(process_func, batch))
                results.extend(batch_results)
    else:
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            for item in batch:
                results.append(process_func(item))
    
    return results


async def async_batch_process(
    items: List[Any],
    async_func: Callable,
    batch_size: int = 10,
    max_concurrent: int = 5
) -> List[Any]:
    """
    Process items in batches asynchronously.
    
    Args:
        items: Items to process
        async_func: Async function to process each item
        batch_size: Batch size
        max_concurrent: Maximum concurrent tasks
        
    Returns:
        List of results
    """
    results = []
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(item):
        async with semaphore:
            return await async_func(item)
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = await asyncio.gather(*[
            process_with_semaphore(item) for item in batch
        ])
        results.extend(batch_results)
    
    return results


# ============================================================================
# CONNECTION POOLING
# ============================================================================

class ConnectionPool:
    """
    Generic connection pool.
    
    Manages reusable connections to external services.
    """
    
    def __init__(
        self,
        create_connection: Callable,
        max_connections: int = 10,
        timeout: float = 30.0
    ):
        """
        Initialize connection pool.
        
        Args:
            create_connection: Function to create new connection
            max_connections: Maximum pool size
            timeout: Connection timeout
        """
        self.create_connection = create_connection
        self.max_connections = max_connections
        self.timeout = timeout
        
        self._pool: List[Any] = []
        self._in_use: set = set()
    
    @contextmanager
    def get_connection(self):
        """Get connection from pool."""
        conn = None
        
        try:
            # Try to get from pool
            if self._pool:
                conn = self._pool.pop()
            elif len(self._in_use) < self.max_connections:
                # Create new connection
                conn = self.create_connection()
            else:
                # Wait for available connection
                start = time.time()
                while not self._pool and (time.time() - start) < self.timeout:
                    time.sleep(0.1)
                
                if self._pool:
                    conn = self._pool.pop()
                else:
                    raise TimeoutError("Connection pool timeout")
            
            self._in_use.add(id(conn))
            yield conn
            
        finally:
            # Return to pool
            if conn:
                self._in_use.discard(id(conn))
                self._pool.append(conn)
    
    def close_all(self):
        """Close all connections."""
        for conn in self._pool:
            if hasattr(conn, "close"):
                conn.close()
        self._pool.clear()
        self._in_use.clear()


# ============================================================================
# QUERY OPTIMIZATION
# ============================================================================

class QueryOptimizer:
    """
    Database query optimizer.
    
    Analyzes and optimizes database queries.
    """
    
    def __init__(self):
        """Initialize query optimizer."""
        self.query_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "total_time": 0.0,
            "min_time": float("inf"),
            "max_time": 0.0,
            "times": []
        })
    
    def track_query(self, query_name: str, execution_time: float):
        """
        Track query execution.
        
        Args:
            query_name: Query identifier
            execution_time: Execution time in seconds
        """
        stats = self.query_stats[query_name]
        stats["count"] += 1
        stats["total_time"] += execution_time
        stats["min_time"] = min(stats["min_time"], execution_time)
        stats["max_time"] = max(stats["max_time"], execution_time)
        stats["times"].append(execution_time)
    
    def get_slow_queries(self, threshold: float = 1.0) -> List[Dict[str, Any]]:
        """
        Get slow queries.
        
        Args:
            threshold: Slow query threshold in seconds
            
        Returns:
            List of slow queries
        """
        slow_queries = []
        
        for query_name, stats in self.query_stats.items():
            avg_time = stats["total_time"] / stats["count"]
            
            if avg_time > threshold:
                slow_queries.append({
                    "query": query_name,
                    "count": stats["count"],
                    "avg_time": avg_time,
                    "max_time": stats["max_time"],
                    "total_time": stats["total_time"]
                })
        
        return sorted(slow_queries, key=lambda x: x["avg_time"], reverse=True)
    
    def get_report(self) -> Dict[str, Any]:
        """
        Get optimization report.
        
        Returns:
            Query statistics report
        """
        total_queries = sum(stats["count"] for stats in self.query_stats.values())
        total_time = sum(stats["total_time"] for stats in self.query_stats.values())
        
        return {
            "total_queries": total_queries,
            "total_time": total_time,
            "unique_queries": len(self.query_stats),
            "slow_queries": self.get_slow_queries(),
            "top_queries": sorted(
                [
                    {
                        "query": name,
                        "count": stats["count"],
                        "total_time": stats["total_time"]
                    }
                    for name, stats in self.query_stats.items()
                ],
                key=lambda x: x["count"],
                reverse=True
            )[:10]
        }


# ============================================================================
# PREFETCHING
# ============================================================================

class Prefetcher:
    """
    Data prefetching utility.
    
    Loads data in advance to reduce latency.
    """
    
    def __init__(self, load_func: Callable, cache_size: int = 100):
        """
        Initialize prefetcher.
        
        Args:
            load_func: Function to load data
            cache_size: Cache size
        """
        self.load_func = load_func
        self.cache: Dict[str, Any] = {}
        self.cache_size = cache_size
        self.access_order: List[str] = []
    
    def prefetch(self, keys: List[str]):
        """
        Prefetch data for keys.
        
        Args:
            keys: Keys to prefetch
        """
        for key in keys:
            if key not in self.cache:
                self._load_and_cache(key)
    
    def get(self, key: str) -> Any:
        """
        Get data with prefetching.
        
        Args:
            key: Data key
            
        Returns:
            Data value
        """
        if key not in self.cache:
            self._load_and_cache(key)
        
        # Update access order
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
        
        return self.cache[key]
    
    def _load_and_cache(self, key: str):
        """Load and cache data."""
        data = self.load_func(key)
        
        # Evict if cache full
        if len(self.cache) >= self.cache_size and self.access_order:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        self.cache[key] = data


# ============================================================================
# BENCHMARKING
# ============================================================================

@dataclass
class BenchmarkResult:
    """Benchmark result."""
    name: str
    execution_time: float
    iterations: int
    avg_time: float
    min_time: float
    max_time: float
    std_dev: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class Benchmark:
    """
    Benchmarking utility.
    
    Measures code execution time and performance.
    """
    
    def __init__(self, name: str):
        """
        Initialize benchmark.
        
        Args:
            name: Benchmark name
        """
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def __enter__(self):
        """Start benchmark."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        """End benchmark."""
        self.end_time = time.time()
        execution_time = self.end_time - self.start_time
        logger.info(f"Benchmark [{self.name}]: {execution_time:.4f}s")
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
    
    @staticmethod
    def run_benchmark(
        func: Callable,
        iterations: int = 100,
        warmup: int = 10,
        **kwargs
    ) -> BenchmarkResult:
        """
        Run comprehensive benchmark.
        
        Args:
            func: Function to benchmark
            iterations: Number of iterations
            warmup: Warmup iterations
            **kwargs: Arguments to pass to function
            
        Returns:
            Benchmark result
        """
        # Warmup
        for _ in range(warmup):
            func(**kwargs)
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.time()
            func(**kwargs)
            elapsed = time.time() - start
            times.append(elapsed)
        
        # Calculate statistics
        total_time = sum(times)
        avg_time = total_time / iterations
        min_time = min(times)
        max_time = max(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
        
        return BenchmarkResult(
            name=func.__name__,
            execution_time=total_time,
            iterations=iterations,
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            std_dev=std_dev
        )


# ============================================================================
# LAZY LOADING
# ============================================================================

class LazyLoader:
    """
    Lazy loading utility.
    
    Delays loading until data is accessed.
    """
    
    def __init__(self, load_func: Callable):
        """
        Initialize lazy loader.
        
        Args:
            load_func: Function to load data
        """
        self.load_func = load_func
        self._data: Optional[Any] = None
        self._loaded = False
    
    def __call__(self):
        """Load and return data."""
        if not self._loaded:
            self._data = self.load_func()
            self._loaded = True
        return self._data
    
    def invalidate(self):
        """Invalidate cached data."""
        self._data = None
        self._loaded = False


# ============================================================================
# PERFORMANCE PROFILER
# ============================================================================

class PerformanceProfiler:
    """
    Performance profiling utility.
    
    Tracks performance metrics across application.
    """
    
    def __init__(self):
        """Initialize profiler."""
        self.metrics: Dict[str, List[float]] = defaultdict(list)
    
    def track(self, metric_name: str, value: float):
        """
        Track metric value.
        
        Args:
            metric_name: Metric name
            value: Metric value
        """
        self.metrics[metric_name].append(value)
    
    @contextmanager
    def measure(self, metric_name: str):
        """
        Context manager to measure execution time.
        
        Args:
            metric_name: Metric name
        """
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            self.track(metric_name, elapsed)
    
    def get_stats(self, metric_name: str) -> Dict[str, float]:
        """
        Get statistics for metric.
        
        Args:
            metric_name: Metric name
            
        Returns:
            Statistics dictionary
        """
        values = self.metrics.get(metric_name, [])
        
        if not values:
            return {}
        
        return {
            "count": len(values),
            "total": sum(values),
            "avg": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "p95": statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
            "p99": statistics.quantiles(values, n=100)[98] if len(values) >= 100 else max(values)
        }
    
    def get_report(self) -> Dict[str, Dict[str, float]]:
        """
        Get full performance report.
        
        Returns:
            Performance statistics for all metrics
        """
        return {
            metric_name: self.get_stats(metric_name)
            for metric_name in self.metrics.keys()
        }


if __name__ == "__main__":
    # Example usage
    
    # Batch processing
    def square(x):
        return x * x
    
    results = batch_process(list(range(100)), square, batch_size=10)
    print(f"Batch results: {results[:5]}...")
    
    # Benchmarking
    def test_function():
        time.sleep(0.01)
        return sum(range(1000))
    
    result = Benchmark.run_benchmark(test_function, iterations=10)
    print(f"Benchmark: {result.name} - avg: {result.avg_time:.4f}s")
    
    # Performance profiler
    profiler = PerformanceProfiler()
    
    with profiler.measure("operation"):
        time.sleep(0.05)
    
    stats = profiler.get_stats("operation")
    print(f"Profiler stats: {stats}")
    
    # Lazy loading
    def expensive_load():
        time.sleep(1)
        return {"data": "loaded"}
    
    lazy_data = LazyLoader(expensive_load)
    print("Lazy loader created (not loaded yet)")
    data = lazy_data()  # Now it loads
    print(f"Lazy data: {data}")
