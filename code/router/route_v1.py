"""
FastAPI Router
API endpoints for Thai Regulatory AI
"""

import datetime
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from adapter.response.response_custom import HandleSuccess
from model.conversation_state import ConversationState
from model.state_manager import StateManager
from model.persona_supervisor import PersonaSupervisor
from utils.simple_cache import get_cache
from utils.rate_limiter import get_rate_limiter

import conf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api_v1 = APIRouter()

SESSION_RETENTION_DAYS = 7


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message to chatbot")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")


class SessionRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Session ID")


class NewSessionRequest(BaseModel):
    persona_id: str = Field(default="practical", description="practical or academic")


logger.info("Initializing services...")

try:
    if conf.USE_ZILLIZ:
        from service.vector_store import VectorStoreManager
        vs_manager = VectorStoreManager()
        retriever = vs_manager.connect_to_existing()
        logger.info("Using Milvus/Zilliz retriever")
    else:
        from service.local_vector_store import get_retriever
        retriever = get_retriever(fail_if_empty=False)
        logger.info("Using local Chroma retriever")

    supervisor = PersonaSupervisor(retriever=retriever)
    state_manager = StateManager()

    logger.info("Services initialized successfully")

except Exception:
    logger.error("Failed to initialize services", exc_info=True)
    supervisor = None
    state_manager = None
    raise


def _cleanup_old_sessions():
    try:
        state_manager.purge_older_than_days(SESSION_RETENTION_DAYS)
    except Exception:
        logger.warning("Session cleanup failed", exc_info=True)


def _build_topics_from_state(state: ConversationState):
    topics_raw: list = (state.context or {}).get("last_menu_topics") or []
    descs_raw: list = (state.context or {}).get("last_menu_topic_descs") or []

    selected = topics_raw[:2]
    topics = [
        {
            "title": t,
            "description": descs_raw[i] if i < len(descs_raw) else f"ผมจะแนะนำ{t} ตั้งแต่ต้นจนจบ พร้อมเอกสารที่ต้องใช้ ให้คุณทำตามได้ง่ายที่สุดครับ",
        }
        for i, t in enumerate(selected)
    ]
    return topics


@api_v1.post("/greeting")
async def start_session(payload: Optional[NewSessionRequest] = None):
    if supervisor is None or state_manager is None:
        raise HTTPException(status_code=503, detail="Services not initialized")

    _cleanup_old_sessions()

    persona_id = "practical"
    if payload and payload.persona_id in {"practical", "academic"}:
        persona_id = payload.persona_id

    session_id = f"s_{uuid.uuid4().hex[:8]}"
    state = ConversationState(session_id=session_id, persona_id=persona_id, context={})

    state, greeting_text = supervisor.handle(state, "")
    state_manager.save(session_id, state)

    topics = _build_topics_from_state(state)

    return HandleSuccess(
        message="Session created",
        session_id=session_id,
        response=greeting_text,
        topics=topics,
        persona_id=persona_id,
        retention_days=SESSION_RETENTION_DAYS,
    )


@api_v1.post("/reset")
async def reset_session(request: SessionRequest):
    if supervisor is None or state_manager is None:
        raise HTTPException(status_code=503, detail="Services not initialized")

    _cleanup_old_sessions()

    session_id = request.session_id or f"s_{uuid.uuid4().hex[:8]}"
    state = ConversationState(session_id=session_id, persona_id="practical", context={})

    state, greeting_text = supervisor.handle(state, "")
    state_manager.save(session_id, state)

    topics = _build_topics_from_state(state)

    return HandleSuccess(
        message="Session reset",
        session_id=session_id,
        response=greeting_text,
        topics=topics,
        retention_days=SESSION_RETENTION_DAYS,
    )


@api_v1.get("/sessions")
async def list_sessions():
    if state_manager is None:
        raise HTTPException(status_code=503, detail="State manager not initialized")

    _cleanup_old_sessions()

    sessions = state_manager.list_sessions(limit=20)
    return HandleSuccess(
        message="Sessions loaded",
        sessions=sessions,
        retention_days=SESSION_RETENTION_DAYS,
    )


@api_v1.post("/session/load")
async def load_session(request: SessionRequest):
    if state_manager is None:
        raise HTTPException(status_code=503, detail="State manager not initialized")

    if not request.session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    state = state_manager.load(request.session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return HandleSuccess(
        message="Session loaded",
        session_id=state.session_id,
        persona_id=state.persona_id,
        messages=state.messages or [],
    )


@api_v1.post("/session/delete")
async def delete_session(request: SessionRequest):
    if state_manager is None:
        raise HTTPException(status_code=503, detail="State manager not initialized")

    if not request.session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    state_manager.delete(request.session_id)

    return HandleSuccess(
        message="Session deleted",
        session_id=request.session_id,
    )


@api_v1.get("/healthcheck")
async def health_check():
    cache = get_cache()
    cache_stats = cache.get_stats()
    
    rate_limiter = get_rate_limiter()
    rate_stats = rate_limiter.get_stats()
    
    return {
        "status": "ok",
        "timestamp": datetime.datetime.now().isoformat(),
        "service": "Thai Regulatory AI - น้องโคโค่",
        "version": "1.0.0",
        "supervisor_initialized": supervisor is not None,
        "state_manager_initialized": state_manager is not None,
        "use_zilliz": conf.USE_ZILLIZ,
        "collection_name": conf.COLLECTION_NAME,
        "session_retention_days": SESSION_RETENTION_DAYS,
        "cache": cache_stats,
        "rate_limit": rate_stats
    }


@api_v1.post("/chat")
async def chat(request: ChatRequest):
    if supervisor is None or state_manager is None:
        logger.error("Services not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Services not initialized. Check server logs.",
        )

    if not request.message or not request.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty",
        )

    _cleanup_old_sessions()

    session_id = request.session_id or f"s_{uuid.uuid4().hex[:8]}"
    
    # Rate limiting check
    rate_limiter = get_rate_limiter()
    allowed, rate_info = rate_limiter.is_allowed(session_id)
    
    if not allowed:
        logger.warning(f"[{session_id}] 🚫 Rate limit exceeded - blocking request")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many requests. Please wait {rate_info['retry_after']} seconds.",
            headers={
                "X-RateLimit-Limit": str(rate_info["limit"]),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(rate_info["reset_in"]),
                "Retry-After": str(rate_info["retry_after"])
            }
        )
    
    logger.info(f"[{session_id}] ✅ Rate limit OK - {rate_info['remaining']}/{rate_info['limit']} remaining")

    try:
        # Load state
        saved = state_manager.load(session_id)
        state = saved if saved else ConversationState(session_id=session_id, persona_id="practical", context={})
        
        # Check cache first
        cache = get_cache()
        cached_result = cache.get(session_id, request.message, state.persona_id)
        
        if cached_result is not None:
            logger.info(f"[{session_id}] 🎯 Cache HIT! Skipping LLM call (saved ${cached_result.get('cost', 0):.3f})")
            
            # Update state with cached message (but don't call LLM)
            state.messages.append({"role": "user", "content": request.message})
            state.messages.append({"role": "assistant", "content": cached_result["response"]})
            state_manager.save(session_id, state)
            
            return HandleSuccess(
                message="Chat completed (cached)",
                response=cached_result["response"],
                session_id=session_id,
                persona_id=state.persona_id,
                cached=True,
                cache_stats=cache.get_stats()
            )
        
        # Cache miss - call LLM
        logger.info(f"[{session_id}] ❌ Cache MISS - Calling LLM")
        state, bot_reply = supervisor.handle(state, request.message)
        state_manager.save(session_id, state)
        
        # Store in cache for future use
        cache.set(
            session_id=session_id,
            question=request.message,
            value={
                "response": bot_reply,
                "cost": 0.033,  # Average cost (will be updated from actual metrics)
                "persona": state.persona_id
            },
            persona=state.persona_id
        )

        return HandleSuccess(
            message="Chat completed",
            response=bot_reply,
            session_id=session_id,
            persona_id=state.persona_id,
            cached=False,
            cache_stats=cache.get_stats()
        )

    except Exception as e:
        logger.error(f"[{session_id}] Chat failed", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat failed: {str(e)}",
        )
    