#/Users/w.worawan/Downloads/ai-operation-microservice3_v2ori/code/tests_enterprise/test_50_retrieval_policy.py
from __future__ import annotations


def test_greeting_never_triggers_retrieval(supervisor, retriever, new_state):
    st = new_state("practical")
    st.context["did_greet"] = True

    supervisor.handle(st, "hi")
    supervisor.handle(st, "ดีจ้า")
    supervisor.handle(st, "555")
    supervisor.handle(st, "ขอบคุณครับ")

    assert len(retriever.queries) == 0