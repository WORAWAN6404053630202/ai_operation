"""
State Manager Service
Handles persistence of conversation states

- stable persist directory
- best-effort file locking
- payload trimming
- list sessions
- purge old sessions
- filter by browser_id
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from model.conversation_state import ConversationState

try:
    import conf
except Exception:
    conf = None


class StateManager:
    def __init__(self, persist_dir: str | None = None):
        if persist_dir:
            base = Path(persist_dir)
        elif conf is not None and getattr(conf, "STATE_DIR", None):
            base = Path(getattr(conf, "STATE_DIR"))
        else:
            base = Path(__file__).resolve().parent.parent / "data" / "states"

        self.dir = base
        self.dir.mkdir(parents=True, exist_ok=True)

        self._lock_timeout_s = float(getattr(conf, "STATE_LOCK_TIMEOUT_S", 2.0) if conf is not None else 2.0)
        self._lock_poll_s = float(getattr(conf, "STATE_LOCK_POLL_S", 0.05) if conf is not None else 0.05)

        self._default_max_recent = int(getattr(conf, "MAX_RECENT_MESSAGES_SAVE", 18) if conf is not None else 18)
        self._default_max_internal = int(getattr(conf, "MAX_INTERNAL_MESSAGES_SAVE", 40) if conf is not None else 40)

    def _safe_session_id(self, session_id: str) -> str:
        return (session_id or "").replace("/", "_").replace("\\", "_").strip()

    def _state_path(self, session_id: str) -> Path:
        return self.dir / f"{self._safe_session_id(session_id)}.json"

    def _lock_path(self, session_id: str) -> Path:
        return self.dir / f"{self._safe_session_id(session_id)}.lock"

    def _acquire_lock(self, session_id: str) -> None:
        lock_path = self._lock_path(session_id)
        deadline = time.time() + self._lock_timeout_s

        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    payload = {"pid": os.getpid(), "ts": time.time()}
                    os.write(fd, json.dumps(payload).encode("utf-8"))
                finally:
                    os.close(fd)
                return
            except FileExistsError:
                try:
                    stat = lock_path.stat()
                    age = time.time() - float(stat.st_mtime)
                    stale_after = float(getattr(conf, "STATE_LOCK_STALE_S", 15.0) if conf is not None else 15.0)
                    if age > stale_after:
                        lock_path.unlink(missing_ok=True)
                        continue
                except Exception:
                    pass

                if time.time() >= deadline:
                    raise TimeoutError(f"Could not acquire state lock for session_id={session_id!r}")
                time.sleep(self._lock_poll_s)

    def _release_lock(self, session_id: str) -> None:
        try:
            self._lock_path(session_id).unlink(missing_ok=True)
        except Exception:
            pass

    def _trim_state_for_save(self, state: ConversationState) -> None:
        max_recent = None
        try:
            sp = getattr(state, "strict_profile", None) or {}
            if isinstance(sp, dict):
                v = sp.get("max_recent_messages")
                if v is not None:
                    max_recent = int(v)
        except Exception:
            max_recent = None

        if not max_recent or max_recent <= 0:
            max_recent = self._default_max_recent

        if isinstance(state.messages, list) and len(state.messages) > max_recent:
            state.messages = state.messages[-max_recent:]

        max_internal = self._default_max_internal
        if isinstance(state.internal_messages, list) and max_internal > 0 and len(state.internal_messages) > max_internal:
            state.internal_messages = state.internal_messages[-max_internal:]

    def save(self, session_id: str, state: ConversationState) -> None:
        if not session_id:
            raise ValueError("session_id is required")

        state.session_id = session_id
        self._trim_state_for_save(state)

        path = self._state_path(session_id)
        tmp_path = path.with_suffix(f".{os.getpid()}.tmp")

        self._acquire_lock(session_id)
        try:
            payload = state.model_dump()
            payload.setdefault("_meta", {})
            payload["_meta"]["schema_version"] = payload["_meta"].get("schema_version", "v1")
            payload["_meta"]["saved_at"] = time.time()

            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            tmp_path.replace(path)
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            self._release_lock(session_id)

    def load(self, session_id: str) -> Optional[ConversationState]:
        if not session_id:
            return None

        path = self._state_path(session_id)
        if not path.exists():
            return None

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data.pop("_meta", None)
        return ConversationState(**data)

    def delete(self, session_id: str) -> None:
        if not session_id:
            return

        path = self._state_path(session_id)
        lock_path = self._lock_path(session_id)

        try:
            self._acquire_lock(session_id)
        except Exception:
            pass

        try:
            if path.exists():
                path.unlink()
        finally:
            try:
                lock_path.unlink(missing_ok=True)
            except Exception:
                pass
            self._release_lock(session_id)

    def list_sessions(self, limit: int = 20, browser_id: Optional[str] = None) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []

        for path in sorted(self.dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                data.pop("_meta", None)
                ctx = data.get("context") or {}

                if browser_id:
                    owner = str(ctx.get("browser_id") or "").strip()
                    if owner != str(browser_id).strip():
                        continue

                session_id = str(data.get("session_id") or path.stem)
                persona_id = str(data.get("persona_id") or "practical")

                chat_title = str(ctx.get("chat_title") or "").strip()
                if not chat_title:
                    first_user = ""
                    messages = data.get("messages") or []
                    for m in messages:
                        if m.get("role") == "user" and (m.get("content") or "").strip():
                            first_user = (m.get("content") or "").strip()
                            break
                    chat_title = first_user[:60] if first_user else f"Chat {session_id[-4:]}"

                updated_at = path.stat().st_mtime

                out.append(
                    {
                        "session_id": session_id,
                        "persona_id": persona_id,
                        "title": chat_title,
                        "updated_at": updated_at,
                    }
                )
            except Exception:
                continue

            if len(out) >= limit:
                break

        return out

    def purge_older_than_days(self, days: int = 7) -> int:
        deleted = 0
        cutoff = time.time() - (max(1, int(days)) * 86400)

        for path in self.dir.glob("*.json"):
            try:
                if path.stat().st_mtime < cutoff:
                    self.delete(path.stem)
                    deleted += 1
            except Exception:
                continue

        return deleted