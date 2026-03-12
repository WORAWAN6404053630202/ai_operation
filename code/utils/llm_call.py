"""
llm_call.py — Thin wrapper around LangChain LLM .invoke() that logs
token usage and wall-clock time per call.

Usage:
    from code.utils.llm_call import llm_invoke
    response = llm_invoke(llm, messages, logger=_LOG, label="Practical/answer", state=state)
    text = (response.content or "").strip()
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, List, Optional

_MAX_RETRIES = 2
_RETRY_DELAYS = (1.0, 2.0)  # exponential backoff in seconds

if TYPE_CHECKING:
    from code.model.conversation_state import ConversationState

_ROOT_LOG = logging.getLogger("restbiz.llm")

_TOKEN_WARN_THRESHOLD = 50_000  # warn when session total exceeds this


def llm_invoke(
    llm: Any,
    messages: List[Any],
    *,
    logger: Optional[logging.Logger] = None,
    label: str = "LLM",
    state: Optional["ConversationState"] = None,
) -> Any:
    """
    Call llm.invoke(messages), log elapsed time + token usage, and
    accumulate tokens into state.add_token_usage() if state is provided.

    Returns the raw LangChain response object (same as llm.invoke would).
    """
    log = logger or _ROOT_LOG
    t0 = time.perf_counter()
    last_exc: Optional[Exception] = None
    for attempt in range(1 + _MAX_RETRIES):
        try:
            response = llm.invoke(messages)
            break
        except Exception as exc:
            last_exc = exc
            elapsed = time.perf_counter() - t0
            if attempt < _MAX_RETRIES:
                delay = _RETRY_DELAYS[attempt]
                log.warning(
                    "[%s] LLM call FAILED after %.2fs (attempt %d/%d) — retrying in %.0fs",
                    label, elapsed, attempt + 1, 1 + _MAX_RETRIES, delay,
                )
                time.sleep(delay)
            else:
                log.warning("[%s] LLM call FAILED after %.2fs (all %d attempts exhausted)", label, elapsed, 1 + _MAX_RETRIES)
                raise
    else:
        # loop exhausted without break (should not happen due to raise above)
        raise RuntimeError(f"[{label}] LLM call failed") from last_exc

    elapsed = time.perf_counter() - t0

    # Extract token counts — LangChain may expose them in different places
    prompt_tokens = 0
    completion_tokens = 0
    try:
        um = getattr(response, "usage_metadata", None) or {}
        if um:
            prompt_tokens = int(um.get("input_tokens") or um.get("prompt_tokens") or 0)
            completion_tokens = int(um.get("output_tokens") or um.get("completion_tokens") or 0)
        else:
            rm = getattr(response, "response_metadata", None) or {}
            tu = rm.get("token_usage") or rm.get("usage") or {}
            prompt_tokens = int(tu.get("prompt_tokens") or tu.get("input_tokens") or 0)
            completion_tokens = int(tu.get("completion_tokens") or tu.get("output_tokens") or 0)
    except Exception:
        pass

    total_call = prompt_tokens + completion_tokens
    log.info(
        "[%s] tokens=%d (in=%d out=%d) time=%.2fs",
        label, total_call, prompt_tokens, completion_tokens, elapsed,
    )

    if state is not None:
        try:
            state.add_token_usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
            session_total = getattr(state, "total_tokens", 0)
            if session_total > _TOKEN_WARN_THRESHOLD:
                log.warning(
                    "[%s] Session token budget exceeded %d (total=%d) — trimming history",
                    label, _TOKEN_WARN_THRESHOLD, session_total,
                )
                if hasattr(state, "trim_messages"):
                    state.trim_messages(keep_last=8)
        except Exception:
            pass

    return response
