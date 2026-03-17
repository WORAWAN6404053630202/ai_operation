"""
Prometheus metrics for Thai Regulatory AI.

Features:
- Custom metrics collection
- LLM performance metrics
- Cache performance metrics
- Business metrics
- Request metrics
- Error tracking

Usage:
    from code.utils.prometheus_metrics import (
        track_llm_call, track_cache_operation, track_request
    )
    
    # Track LLM call
    with track_llm_call(model="gpt-4", persona="practical"):
        response = llm.invoke(prompt)
    
    # Track cache operation
    track_cache_operation("get", hit=True)
    
    # Expose metrics endpoint
    # GET /metrics
"""

import time
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager
from functools import wraps

from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info,
    generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry
)


logger = logging.getLogger(__name__)


# Create registry
registry = CollectorRegistry()


# ============================================================================
# REQUEST METRICS
# ============================================================================

# Request counter
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
    registry=registry
)

# Request duration
http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    registry=registry
)

# Active requests
http_requests_active = Gauge(
    "http_requests_active",
    "Number of active HTTP requests",
    registry=registry
)


# ============================================================================
# LLM METRICS
# ============================================================================

# LLM calls counter
llm_calls_total = Counter(
    "llm_calls_total",
    "Total LLM calls",
    ["model", "persona", "status"],
    registry=registry
)

# LLM call duration
llm_call_duration_seconds = Histogram(
    "llm_call_duration_seconds",
    "LLM call duration in seconds",
    ["model", "persona"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 60.0],
    registry=registry
)

# LLM tokens
llm_tokens_total = Counter(
    "llm_tokens_total",
    "Total LLM tokens used",
    ["model", "token_type"],  # prompt, completion
    registry=registry
)

# LLM cost
llm_cost_total = Counter(
    "llm_cost_total",
    "Total LLM cost in USD",
    ["model"],
    registry=registry
)

# LLM errors
llm_errors_total = Counter(
    "llm_errors_total",
    "Total LLM errors",
    ["model", "error_type"],
    registry=registry
)


# ============================================================================
# CACHE METRICS
# ============================================================================

# Cache operations
cache_operations_total = Counter(
    "cache_operations_total",
    "Total cache operations",
    ["operation", "result"],  # get/set/delete, hit/miss/success/error
    registry=registry
)

# Cache hit rate
cache_hit_rate = Gauge(
    "cache_hit_rate",
    "Cache hit rate (0-1)",
    registry=registry
)

# Cache size
cache_size_bytes = Gauge(
    "cache_size_bytes",
    "Cache size in bytes",
    registry=registry
)

# Cache items
cache_items_total = Gauge(
    "cache_items_total",
    "Total items in cache",
    registry=registry
)


# ============================================================================
# VECTOR STORE METRICS
# ============================================================================

# Vector store queries
vector_queries_total = Counter(
    "vector_queries_total",
    "Total vector store queries",
    ["status"],
    registry=registry
)

# Vector query duration
vector_query_duration_seconds = Histogram(
    "vector_query_duration_seconds",
    "Vector query duration in seconds",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
    registry=registry
)

# Documents retrieved
vector_documents_retrieved = Summary(
    "vector_documents_retrieved",
    "Number of documents retrieved per query",
    registry=registry
)


# ============================================================================
# BUSINESS METRICS
# ============================================================================

# Active sessions
sessions_active = Gauge(
    "sessions_active",
    "Number of active sessions",
    registry=registry
)

# Total conversations
conversations_total = Counter(
    "conversations_total",
    "Total conversations",
    ["persona"],
    registry=registry
)

# User satisfaction (if collected)
user_satisfaction = Histogram(
    "user_satisfaction",
    "User satisfaction scores",
    ["persona"],
    buckets=[1, 2, 3, 4, 5],
    registry=registry
)

# Messages per session
messages_per_session = Summary(
    "messages_per_session",
    "Number of messages per session",
    registry=registry
)


# ============================================================================
# SYSTEM METRICS
# ============================================================================

# Application info
app_info = Info(
    "app",
    "Application information",
    registry=registry
)

app_info.info({
    "name": "thai_regulatory_ai",
    "version": "2.0.0",
    "environment": "production"
})

# Database connections
database_connections_active = Gauge(
    "database_connections_active",
    "Active database connections",
    registry=registry
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def track_request(method: str, endpoint: str, status: int, duration: float):
    """
    Track HTTP request metrics.
    
    Args:
        method: HTTP method
        endpoint: Endpoint path
        status: Status code
        duration: Request duration in seconds
    """
    http_requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
    http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)


@contextmanager
def track_active_request():
    """Context manager to track active requests."""
    http_requests_active.inc()
    try:
        yield
    finally:
        http_requests_active.dec()


def track_llm_call_metric(
    model: str,
    persona: str,
    duration: float,
    tokens: Dict[str, int],
    cost: float,
    status: str = "success"
):
    """
    Track LLM call metrics.
    
    Args:
        model: Model name
        persona: Persona type
        duration: Call duration in seconds
        tokens: Dict with prompt_tokens and completion_tokens
        cost: Call cost in USD
        status: success or error
    """
    llm_calls_total.labels(model=model, persona=persona, status=status).inc()
    llm_call_duration_seconds.labels(model=model, persona=persona).observe(duration)
    
    if tokens:
        llm_tokens_total.labels(model=model, token_type="prompt").inc(tokens.get("prompt_tokens", 0))
        llm_tokens_total.labels(model=model, token_type="completion").inc(tokens.get("completion_tokens", 0))
    
    llm_cost_total.labels(model=model).inc(cost)


def track_llm_error(model: str, error_type: str):
    """
    Track LLM error.
    
    Args:
        model: Model name
        error_type: Error type
    """
    llm_errors_total.labels(model=model, error_type=error_type).inc()


def track_cache_operation(operation: str, result: str):
    """
    Track cache operation.
    
    Args:
        operation: Operation type (get, set, delete)
        result: Result (hit, miss, success, error)
    """
    cache_operations_total.labels(operation=operation, result=result).inc()


def update_cache_metrics(hit_rate: float, size_bytes: int, items: int):
    """
    Update cache metrics.
    
    Args:
        hit_rate: Cache hit rate (0-1)
        size_bytes: Cache size in bytes
        items: Number of items
    """
    cache_hit_rate.set(hit_rate)
    cache_size_bytes.set(size_bytes)
    cache_items_total.set(items)


def track_vector_query(duration: float, num_docs: int, status: str = "success"):
    """
    Track vector store query.
    
    Args:
        duration: Query duration in seconds
        num_docs: Number of documents retrieved
        status: success or error
    """
    vector_queries_total.labels(status=status).inc()
    vector_query_duration_seconds.observe(duration)
    vector_documents_retrieved.observe(num_docs)


def update_session_metrics(active_sessions: int):
    """
    Update session metrics.
    
    Args:
        active_sessions: Number of active sessions
    """
    sessions_active.set(active_sessions)


def track_conversation(persona: str, num_messages: int):
    """
    Track conversation metrics.
    
    Args:
        persona: Persona type
        num_messages: Number of messages in session
    """
    conversations_total.labels(persona=persona).inc()
    messages_per_session.observe(num_messages)


def track_user_satisfaction(persona: str, score: int):
    """
    Track user satisfaction.
    
    Args:
        persona: Persona type
        score: Satisfaction score (1-5)
    """
    user_satisfaction.labels(persona=persona).observe(score)


def update_database_connections(active: int):
    """
    Update database connection metrics.
    
    Args:
        active: Number of active connections
    """
    database_connections_active.set(active)


# ============================================================================
# DECORATORS
# ============================================================================

def monitor_llm_call(model: str, persona: str):
    """
    Decorator to monitor LLM calls.
    
    Args:
        model: Model name
        persona: Persona type
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start
                
                # Extract metrics from result if available
                tokens = getattr(result, "usage", {})
                cost = getattr(result, "cost", 0.0)
                
                track_llm_call_metric(
                    model=model,
                    persona=persona,
                    duration=duration,
                    tokens=tokens,
                    cost=cost,
                    status="success"
                )
                
                return result
            except Exception as e:
                duration = time.time() - start
                track_llm_call_metric(
                    model=model,
                    persona=persona,
                    duration=duration,
                    tokens={},
                    cost=0.0,
                    status="error"
                )
                track_llm_error(model=model, error_type=type(e).__name__)
                raise
        
        return wrapper
    return decorator


def monitor_vector_query(func):
    """Decorator to monitor vector queries."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start
            num_docs = len(result) if result else 0
            track_vector_query(duration, num_docs, status="success")
            return result
        except Exception as e:
            duration = time.time() - start
            track_vector_query(duration, 0, status="error")
            raise
    
    return wrapper


# ============================================================================
# METRICS ENDPOINT
# ============================================================================

def get_metrics() -> bytes:
    """
    Get Prometheus metrics.
    
    Returns:
        Metrics in Prometheus format
    """
    return generate_latest(registry)


def get_metrics_content_type() -> str:
    """
    Get metrics content type.
    
    Returns:
        Content type string
    """
    return CONTENT_TYPE_LATEST


if __name__ == "__main__":
    # Example usage
    
    # Track request
    track_request("POST", "/api/chat", 200, 1.5)
    
    # Track LLM call
    track_llm_call_metric(
        model="gpt-4",
        persona="practical",
        duration=2.3,
        tokens={"prompt_tokens": 100, "completion_tokens": 200},
        cost=0.01
    )
    
    # Track cache
    track_cache_operation("get", "hit")
    update_cache_metrics(hit_rate=0.75, size_bytes=1024000, items=100)
    
    # Get metrics
    metrics = get_metrics()
    print(metrics.decode())
