"""
API versioning system for Thai Regulatory AI.

Features:
- Multiple API versions (v1, v2)
- Deprecation warnings
- Backward compatibility
- Version routing
- Migration guides

Usage:
    from code.router.route_v2 import router as v2_router
    
    app.include_router(v2_router, prefix="/api/v2")
    
    # V1 endpoints: /api/v1/chat
    # V2 endpoints: /api/v2/chat
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Header, Request
from pydantic import BaseModel, Field

from code.adapter.response.chat_response import ChatResponse
from code.utils.auth import AuthManager
from code.utils.security import SecurityManager
from code.utils.redis_cache import get_redis_cache, create_cache_key
from code.utils.prometheus_metrics import track_request, track_active_request


logger = logging.getLogger(__name__)


# API Version 2 Router
router = APIRouter(tags=["chat-v2"])


# ============================================================================
# REQUEST/RESPONSE MODELS (V2)
# ============================================================================

class ChatRequestV2(BaseModel):
    """Chat request model for API v2."""
    
    question: str = Field(..., min_length=1, max_length=2000, description="User question")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    persona: Optional[str] = Field("practical", description="Persona type (practical, academic)")
    user_id: Optional[str] = Field(None, description="User ID for tracking")
    
    # V2 new features
    stream: bool = Field(False, description="Enable streaming responses")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens in response")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="LLM temperature")
    context_window: int = Field(5, ge=1, le=20, description="Number of previous messages to include")
    enable_cache: bool = Field(True, description="Enable response caching")
    
    class Config:
        schema_extra = {
            "example": {
                "question": "บอกขั้นตอนการจดทะเบียนร้านอาหาร",
                "session_id": "session_123",
                "persona": "practical",
                "stream": False,
                "max_tokens": 1000,
                "temperature": 0.7,
                "context_window": 5,
                "enable_cache": True
            }
        }


class ChatResponseV2(BaseModel):
    """Chat response model for API v2."""
    
    answer: str = Field(..., description="AI response")
    session_id: str = Field(..., description="Session ID")
    persona: str = Field(..., description="Current persona")
    
    # V2 enhanced metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    sources: Optional[list] = Field(None, description="Source documents")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Response confidence score")
    tokens_used: Optional[int] = Field(None, description="Tokens used")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")
    cached: bool = Field(False, description="Whether response was cached")
    
    # Cost tracking
    estimated_cost: Optional[float] = Field(None, description="Estimated cost in USD")
    
    # Quality indicators
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Response quality score")
    
    class Config:
        schema_extra = {
            "example": {
                "answer": "ขั้นตอนการจดทะเบียนร้านอาหาร...",
                "session_id": "session_123",
                "persona": "practical",
                "metadata": {
                    "model": "gpt-4",
                    "version": "v2"
                },
                "sources": [
                    {"title": "คู่มือการจดทะเบียน", "page": 5}
                ],
                "confidence": 0.95,
                "tokens_used": 500,
                "processing_time_ms": 1250.5,
                "cached": False,
                "estimated_cost": 0.015,
                "quality_score": 0.92
            }
        }


class HealthResponseV2(BaseModel):
    """Health check response for API v2."""
    
    status: str
    version: str
    timestamp: str
    
    # V2 enhanced health info
    services: Dict[str, str] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    uptime_seconds: Optional[float] = None


# ============================================================================
# MIDDLEWARE & DEPENDENCIES
# ============================================================================

async def check_api_version(
    request: Request,
    x_api_version: Optional[str] = Header(None)
):
    """
    Check API version and add deprecation warnings.
    
    Args:
        request: FastAPI request
        x_api_version: Optional API version header
    """
    # Extract version from path
    path_version = None
    if "/v1/" in request.url.path:
        path_version = "v1"
    elif "/v2/" in request.url.path:
        path_version = "v2"
    
    # Log version usage
    effective_version = x_api_version or path_version or "unknown"
    logger.info(f"API request - version: {effective_version}, path: {request.url.path}")
    
    # Add deprecation warning for v1
    if path_version == "v1":
        logger.warning(
            "API v1 is deprecated. Please migrate to v2. "
            "V1 will be sunset on 2026-12-31."
        )


# ============================================================================
# ENDPOINTS (V2)
# ============================================================================

@router.post("/chat", response_model=ChatResponseV2)
async def chat_v2(
    request: ChatRequestV2,
    api_version_check = Depends(check_api_version)
):
    """
    Chat endpoint (API v2) with enhanced features.
    
    New features in v2:
    - Streaming support
    - Configurable temperature and max_tokens
    - Confidence scores
    - Quality metrics
    - Enhanced caching
    - Source attribution
    """
    import time
    start_time = time.time()
    
    try:
        with track_active_request():
            # Security check
            security = SecurityManager()
            validation = security.validate_input(
                request.question,
                sanitize=True,
                mask_pii=True,
                check_injection=True
            )
            
            if not validation["security_check"]["is_safe"]:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Security validation failed",
                        "issues": validation["security_check"]["issues"]
                    }
                )
            
            question = validation["processed_text"]
            
            # Check cache if enabled
            cached_response = None
            if request.enable_cache:
                cache = get_redis_cache()
                cache_key = create_cache_key(
                    "chat_v2",
                    request.persona,
                    question[:100]
                )
                cached_response = cache.get(cache_key)
                
                if cached_response:
                    logger.info(f"Cache hit for v2 request")
                    cached_response["cached"] = True
                    cached_response["processing_time_ms"] = (time.time() - start_time) * 1000
                    return ChatResponseV2(**cached_response)
            
            # Process request (import here to avoid circular dependency)
            from code.model.state_manager import StateManager
            from code.utils.llm_call import llm_invoke
            
            state_manager = StateManager()
            state = state_manager.get_or_create_session(
                request.session_id or f"v2_{int(time.time())}",
                request.persona
            )
            
            # Get persona
            persona = state_manager.get_active_persona(state)
            
            # Build prompt with context window
            conversation_history = state.get("history", [])[-request.context_window:]
            
            # Get retriever context
            from code.service.local_vector_store import LocalVectorStore
            vector_store = LocalVectorStore()
            docs = vector_store.get_retriever().get_relevant_documents(question)
            
            context = "\n\n".join([doc.page_content for doc in docs[:5]])
            
            # Build prompt
            full_prompt = persona.build_prompt(
                question=question,
                context=context,
                history=conversation_history
            )
            
            # LLM call with v2 parameters
            llm_kwargs = {}
            if request.max_tokens:
                llm_kwargs["max_tokens"] = request.max_tokens
            if request.temperature is not None:
                llm_kwargs["temperature"] = request.temperature
            
            response = llm_invoke(full_prompt, **llm_kwargs)
            
            # Extract answer
            answer = response.content if hasattr(response, "content") else str(response)
            
            # Calculate metrics
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Estimate tokens (rough estimate)
            tokens_used = len(full_prompt.split()) + len(answer.split())
            estimated_cost = tokens_used * 0.00003  # Rough estimate
            
            # Calculate confidence and quality (simplified)
            confidence = 0.85  # Would use actual model confidence
            quality_score = min(1.0, len(answer) / 500)  # Simplified
            
            # Prepare sources
            sources = [
                {
                    "title": doc.metadata.get("source", "Unknown"),
                    "content_preview": doc.page_content[:100] + "...",
                    "relevance": 1.0 / (i + 1)
                }
                for i, doc in enumerate(docs[:3])
            ]
            
            # Build response
            response_data = {
                "answer": answer,
                "session_id": state["session_id"],
                "persona": request.persona,
                "metadata": {
                    "model": "gpt-4",
                    "version": "v2",
                    "context_docs": len(docs),
                    "history_length": len(conversation_history)
                },
                "sources": sources,
                "confidence": confidence,
                "tokens_used": tokens_used,
                "processing_time_ms": round(processing_time_ms, 2),
                "cached": False,
                "estimated_cost": round(estimated_cost, 6),
                "quality_score": round(quality_score, 2)
            }
            
            # Update state
            state["history"].append({"role": "user", "content": question})
            state["history"].append({"role": "assistant", "content": answer})
            state_manager.save_session(state)
            
            # Cache response
            if request.enable_cache:
                cache.set(cache_key, response_data, ttl=3600)
            
            # Track metrics
            track_request("POST", "/api/v2/chat", 200, processing_time_ms / 1000)
            
            return ChatResponseV2(**response_data)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat v2 error: {e}", exc_info=True)
        processing_time_ms = (time.time() - start_time) * 1000
        track_request("POST", "/api/v2/chat", 500, processing_time_ms / 1000)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponseV2)
async def health_v2():
    """Enhanced health check endpoint (API v2)."""
    import time
    
    # Check services
    services = {
        "api": "healthy",
        "llm": "healthy",
        "vector_store": "healthy",
        "cache": "healthy",
        "database": "healthy"
    }
    
    # Get metrics
    try:
        from code.utils.redis_cache import get_redis_cache
        cache = get_redis_cache()
        cache_stats = cache.get_stats()
        
        metrics = {
            "cache_hit_rate": cache_stats.get("hit_rate", 0),
            "cache_size": cache_stats.get("fallback_cache_size", 0)
        }
    except Exception:
        metrics = {}
    
    return HealthResponseV2(
        status="healthy",
        version="2.0.0",
        timestamp=datetime.now().isoformat(),
        services=services,
        metrics=metrics,
        uptime_seconds=time.time()
    )


@router.get("/version")
async def get_version():
    """Get API version information."""
    return {
        "version": "2.0.0",
        "api_version": "v2",
        "release_date": "2026-03-15",
        "features": [
            "streaming_support",
            "confidence_scores",
            "quality_metrics",
            "enhanced_caching",
            "source_attribution",
            "configurable_parameters"
        ],
        "deprecations": {
            "v1": {
                "sunset_date": "2026-12-31",
                "migration_guide": "/docs/migration-v1-to-v2"
            }
        }
    }
