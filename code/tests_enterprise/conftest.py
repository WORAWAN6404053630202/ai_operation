#/Users/w.worawan/Downloads/ai-operation-microservice3_v2ori/code/tests_enterprise/conftest.py
from __future__ import annotations

import sys
from pathlib import Path
import pytest

# Add code directory to path for imports
code_dir = Path(__file__).parent.parent
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from model.conversation_state import ConversationState
from model.persona_supervisor import PersonaSupervisor

from .fakes import SpyRetriever, LLMCallStats, FakeLLMJSON


@pytest.fixture()
def retriever():
    return SpyRetriever()


@pytest.fixture()
def llm_stats():
    return LLMCallStats()


@pytest.fixture()
def llm_router():
    """
    Router returns JSON for:
    - intent classifier: {intent, meta}
    - confirm yes/no: {yes,no,confidence}
    - style wants_long/short: {wants_long,wants_short,confidence}
    - academic final answer: decision JSON
    """

    def _router(prompt: str):
        p = (prompt or "").lower()

        # ------------------------------------------------------------
        # 1) YES/NO confirmation classifier (persona switch confirm)
        # ------------------------------------------------------------
        if ("yes" in p and "no" in p) and ("confidence" in p) and ("ตีความ" in prompt or "classify" in p):
            if any(x in p for x in ["เยป", "yep", "ได้เลย", "ยืนยัน", "ok", "okay", "จัดไป", "ครับผม", "ได้ครับ"]):
                return {"yes": True, "no": False, "confidence": 0.9}
            if any(x in p for x in ["ไม่เอา", "ยกเลิก", "cancel", "nope", "ไม่ต้อง", "ไม่ครับ"]):
                return {"yes": False, "no": True, "confidence": 0.9}
            return {"yes": False, "no": False, "confidence": 0.2}

        # ------------------------------------------------------------
        # 2) Style classifier (wants_long / wants_short)
        # ------------------------------------------------------------
        if ("wants_long" in p and "wants_short" in p) and ("confidence" in p):
            if any(x in p for x in ["ละเอียด", "เชิงลึก", "อ้างอิง", "ตามกฎหมาย", "ลงรายละเอียด", "reasoning"]):
                return {"wants_long": True, "wants_short": False, "confidence": 0.9}
            if any(x in p for x in ["สั้น", "กระชับ", "สรุป", "เป็นข้อๆ", "tl;dr"]):
                return {"wants_long": False, "wants_short": True, "confidence": 0.9}
            return {"wants_long": False, "wants_short": False, "confidence": 0.2}

        # ------------------------------------------------------------
        # 3) INTENT classifier (CRITICAL for enterprise suite)
        # Expect: { "intent": "...", "meta": {...} }
        # We detect this prompt by presence of "intent" and "meta"
        # ------------------------------------------------------------
        if ("intent" in p and "meta" in p) and ("return json" in p or "ตอบเป็น json" in p or "json" in p):
            # normalize user_text extraction: take the last non-empty line as proxy
            user_text = ""
            for line in (prompt or "").splitlines()[::-1]:
                t = line.strip()
                if t:
                    user_text = t
                    break
            u = user_text.lower()

            # explicit switch
            if any(x in u for x in ["เปลี่ยนโหมด", "สลับโหมด", "switch mode", "switch persona"]):
                # no target
                if not any(x in u for x in ["practical", "academic", "สั้น", "ละเอียด"]):
                    return {"intent": "explicit_switch", "meta": {"kind": "no_target"}}
                # has target implied by style
                if any(x in u for x in ["ละเอียด", "เชิงลึก", "academic"]):
                    return {"intent": "explicit_switch", "meta": {"kind": "target", "wants_long": True}}
                if any(x in u for x in ["สั้น", "สรุป", "practical"]):
                    return {"intent": "explicit_switch", "meta": {"kind": "target", "wants_short": True}}
                return {"intent": "explicit_switch", "meta": {"kind": "no_target"}}

            # mode status
            if any(x in u for x in ["ตอนนี้โหมด", "อยู่โหมด", "mode status", "โหมดอะไร"]):
                return {"intent": "mode_status", "meta": {}}

            # greeting / thanks / noise
            if any(x in u for x in ["สวัสดี", "hi", "hello", "ดีจ้า", "ขอบคุณ", "thanks", "thank you"]):
                return {"intent": "greeting", "meta": {}}
            if any(x in u for x in ["555", "lol", "lmao", "haha", "ฮ่า", "😅", "😂"]):
                return {"intent": "noise", "meta": {}}

            # legal (must retrieve)
            if any(x in u for x in ["ขึ้นทะเบียน", "นายจ้าง", "ประกันสังคม", "กองทุน", "ภพ.20", "vat", "ภาษี", "ใบอนุญาต"]):
                return {"intent": "legal", "meta": {}}

            # default
            return {"intent": "unknown", "meta": {}}

        # ------------------------------------------------------------
        # 4) Academic final answer JSON
        # ------------------------------------------------------------
        if "return json" in p and "documents" in p and "user_question" in p:
            return {
                "input_type": "new_question",
                "analysis": "ok",
                "action": "answer",
                "execution": {
                    "answer": "สรุปคำตอบแบบ academic จากเอกสาร: ...\n- ค่าธรรมเนียม: ไม่มีค่าธรรมเนียม",
                    "context_update": {},
                },
            }

        # default: harmless
        return {"ok": True}

    return _router


@pytest.fixture()
def supervisor(monkeypatch, retriever, llm_stats, llm_router):
    """
    Supervisor + patch ChatOpenAI.invoke globally to deterministic FakeLLMJSON
    so test suite never calls real LLM.
    """
    from langchain_openai import ChatOpenAI

    fake = FakeLLMJSON(llm_stats, llm_router)
    monkeypatch.setattr(ChatOpenAI, "invoke", fake.invoke, raising=True)

    sup = PersonaSupervisor(retriever=retriever)
    return sup


@pytest.fixture()
def new_state():
    def _make(persona_id: str = "practical"):
        return ConversationState(
            session_id="test",
            persona_id=persona_id,
            context={},
            messages=[],
            internal_messages=[],
        )
    return _make