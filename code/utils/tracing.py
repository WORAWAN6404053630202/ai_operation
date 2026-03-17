"""
OpenTelemetry instrumentation for distributed tracing.

Traces:
- HTTP requests (FastAPI)
- LLM calls
- Vector store queries
- Cache operations
- Persona switching

Export to:
- Jaeger (default)
- Zipkin
- Console (development)

Usage:
    from code.utils.tracing import init_tracing, trace_function
    
    # Initialize tracing
    init_tracing(service_name="thai-regulatory-ai")
    
    # Trace function
    @trace_function(name="my_function")
    def my_function():
        pass
    
    # Manual span
    from code.utils.tracing import tracer
    with tracer.start_as_current_span("operation_name"):
        # Do work
        pass
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from typing import Optional, Callable, Any
from functools import wraps
import logging

logger = logging.getLogger(__name__)

# Global tracer
tracer: Optional[trace.Tracer] = None


def init_tracing(
    service_name: str = "thai-regulatory-ai",
    export_to: str = "console",  # "console", "jaeger", "zipkin"
    endpoint: Optional[str] = None
):
    """
    Initialize OpenTelemetry tracing.
    
    Args:
        service_name: Service name for tracing
        export_to: Export destination ("console", "jaeger", "zipkin")
        endpoint: Custom endpoint URL (optional)
    """
    global tracer
    
    # Create resource
    resource = Resource.create({
        "service.name": service_name,
        "service.version": "1.0.0",
        "deployment.environment": "production"
    })
    
    # Create provider
    provider = TracerProvider(resource=resource)
    
    # Add exporter
    if export_to == "console":
        exporter = ConsoleSpanExporter()
    elif export_to == "jaeger":
        try:
            from opentelemetry.exporter.jaeger.thrift import JaegerExporter
            exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=6831,
            )
        except ImportError:
            logger.warning("Jaeger exporter not available, falling back to console")
            exporter = ConsoleSpanExporter()
    elif export_to == "zipkin":
        try:
            from opentelemetry.exporter.zipkin.json import ZipkinExporter
            exporter = ZipkinExporter(
                endpoint=endpoint or "http://localhost:9411/api/v2/spans"
            )
        except ImportError:
            logger.warning("Zipkin exporter not available, falling back to console")
            exporter = ConsoleSpanExporter()
    else:
        exporter = ConsoleSpanExporter()
    
    # Add processor
    provider.add_span_processor(BatchSpanProcessor(exporter))
    
    # Set global provider
    trace.set_tracer_provider(provider)
    
    # Get tracer
    tracer = trace.get_tracer(__name__)
    
    # Instrument FastAPI
    try:
        FastAPIInstrumentor.instrument()
    except Exception as e:
        logger.warning(f"Failed to instrument FastAPI: {e}")
    
    # Instrument requests library
    try:
        RequestsInstrumentor().instrument()
    except Exception as e:
        logger.warning(f"Failed to instrument requests: {e}")
    
    logger.info(f"✅ Tracing initialized: {service_name} -> {export_to}")


def trace_function(name: Optional[str] = None, attributes: Optional[dict] = None):
    """
    Decorator to trace a function.
    
    Usage:
        @trace_function(name="my_function", attributes={"key": "value"})
        def my_function(arg1, arg2):
            return result
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if tracer is None:
                return func(*args, **kwargs)
            
            span_name = name or f"{func.__module__}.{func.__name__}"
            
            with tracer.start_as_current_span(span_name) as span:
                # Add attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)
                
                # Add function arguments as attributes
                if args:
                    span.set_attribute("args.count", len(args))
                if kwargs:
                    for key, value in kwargs.items():
                        if isinstance(value, (str, int, float, bool)):
                            span.set_attribute(f"arg.{key}", value)
                
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("success", True)
                    return result
                except Exception as e:
                    span.set_attribute("success", False)
                    span.set_attribute("error", str(e))
                    span.set_attribute("error.type", type(e).__name__)
                    raise
        
        return wrapper
    return decorator


async def trace_async_function(name: Optional[str] = None, attributes: Optional[dict] = None):
    """
    Decorator to trace an async function.
    
    Usage:
        @trace_async_function(name="my_async_function")
        async def my_async_function():
            return result
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            if tracer is None:
                return await func(*args, **kwargs)
            
            span_name = name or f"{func.__module__}.{func.__name__}"
            
            with tracer.start_as_current_span(span_name) as span:
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)
                
                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute("success", True)
                    return result
                except Exception as e:
                    span.set_attribute("success", False)
                    span.set_attribute("error", str(e))
                    raise
        
        return wrapper
    return decorator


def add_span_event(name: str, attributes: Optional[dict] = None):
    """Add an event to the current span."""
    if tracer is None:
        return
    
    span = trace.get_current_span()
    if span:
        span.add_event(name, attributes=attributes or {})


def set_span_attribute(key: str, value: Any):
    """Set an attribute on the current span."""
    if tracer is None:
        return
    
    span = trace.get_current_span()
    if span:
        span.set_attribute(key, value)


# Example instrumentation for LLM calls
def trace_llm_call(model: str, prompt_tokens: int, completion_tokens: int, cost: float):
    """Add LLM call metrics to current span."""
    if tracer is None:
        return
    
    span = trace.get_current_span()
    if span:
        span.set_attribute("llm.model", model)
        span.set_attribute("llm.prompt_tokens", prompt_tokens)
        span.set_attribute("llm.completion_tokens", completion_tokens)
        span.set_attribute("llm.total_tokens", prompt_tokens + completion_tokens)
        span.set_attribute("llm.cost_usd", cost)


# Example instrumentation for cache operations
def trace_cache_operation(operation: str, hit: bool, key: Optional[str] = None):
    """Add cache operation metrics to current span."""
    if tracer is None:
        return
    
    span = trace.get_current_span()
    if span:
        span.set_attribute("cache.operation", operation)
        span.set_attribute("cache.hit", hit)
        if key:
            span.set_attribute("cache.key", key)


# Example instrumentation for vector store queries
def trace_vector_search(query: str, num_results: int, similarity_threshold: float):
    """Add vector search metrics to current span."""
    if tracer is None:
        return
    
    span = trace.get_current_span()
    if span:
        span.set_attribute("vector.query_length", len(query))
        span.set_attribute("vector.num_results", num_results)
        span.set_attribute("vector.similarity_threshold", similarity_threshold)


if __name__ == "__main__":
    # Example usage
    init_tracing(service_name="thai-regulatory-ai-test", export_to="console")
    
    @trace_function(name="test_function")
    def test_function(x, y):
        add_span_event("processing_started")
        result = x + y
        set_span_attribute("result", result)
        add_span_event("processing_completed")
        return result
    
    result = test_function(5, 3)
    print(f"Result: {result}")
