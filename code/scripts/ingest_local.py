# code/scripts/ingest_local.py
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

# Add parent directory to path so we can import from code/
project_root = Path(__file__).parent.parent.parent
code_dir = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from code.service.data_loader import DataLoader
from code.service.local_vector_store import ingest_documents
from code.utils.text_chunker import should_chunk_document, chunk_document_smart
from code.utils.semantic_chunker import SemanticChunker
from langchain_core.documents import Document

SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/1YnLKV7gJXCu7jvcH1sUL9crMlBCJKOpQfp2wtulMszE/"
    "edit?gid=657201027#gid=657201027"
)

# -----------------------------
# Data normalization (ingest-time)
# -----------------------------
_WS_RE = re.compile(r"\s+", re.UNICODE)

# Match "token token" duplicated consecutively WITH whitespace.
# - Thai token: [\u0E00-\u0E7F]+
# - Latin/num token: [A-Za-z0-9]+
_DUP_TOKEN_WS_RE = re.compile(
    r"(?:(?<=\s)|^)"
    r"("
    r"(?:[\u0E00-\u0E7F]+|[A-Za-z0-9]+)"
    r")"
    r"(?:\s+)"
    r"\1"
    r"(?=(?:\s|$))",
    re.UNICODE,
)

# Match "token" duplicated consecutively WITHOUT whitespace (typo like "เอกสารเอกสาร", "VATVAT").
# Conservative:
# - token length >= 3 (avoid breaking short syllables)
# - Thai token OR latin/num token
_DUP_TOKEN_NOSPACE_RE = re.compile(
    r"((?:[\u0E00-\u0E7F]{3,}|[A-Za-z0-9]{3,}))\1",
    re.UNICODE,
)

_URLISH_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_EMAILISH_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.IGNORECASE)


def _normalize_text(s: str) -> str:
    """
    Conservative normalization (ingest-time):
    - normalize whitespace
    - remove obvious consecutive duplicate tokens
        - with whitespace: "เอกสาร เอกสาร" -> "เอกสาร"
        - no whitespace: "เอกสารเอกสาร" -> "เอกสาร"
    - fix a few known repeated phrases (safe targeted)
    """
    if s is None:
        return ""
    t = str(s)

    # Do not normalize URLs/emails aggressively (avoid breaking them)
    if _URLISH_RE.search(t) or _EMAILISH_RE.search(t):
        return t.strip()

    # Remove zero-width spaces and normalize whitespace
    t = t.replace("\u200b", "")
    t = _WS_RE.sub(" ", t).strip()
    if not t:
        return t

    # Targeted safe fixes (known typos)
    t = t.replace("การจดทะเบียนทะเบียนพาณิชย์", "การจดทะเบียนพาณิชย์")
    t = t.replace("จดทะเบียนทะเบียนพาณิชย์", "จดทะเบียนพาณิชย์")
    t = t.replace("ทะเบียนทะเบียนพาณิชย์", "ทะเบียนพาณิชย์")

    # Remove consecutive duplicate tokens (repeat until stable, but cap passes)
    # 1) whitespace duplicates: "ทะเบียน ทะเบียน พาณิชย์" -> "ทะเบียน พาณิชย์"
    # 2) no-space duplicates: "เอกสารเอกสารประเภ..." -> "เอกสารประเภ..."
    for _ in range(4):
        new_t = _DUP_TOKEN_WS_RE.sub(r"\1", t)
        new_t = _DUP_TOKEN_NOSPACE_RE.sub(r"\1", new_t)
        new_t = _WS_RE.sub(" ", new_t).strip()
        if new_t == t:
            break
        t = new_t

    return t


def _normalize_metadata(md: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(md, dict):
        return md

    out: Dict[str, Any] = {}
    for k, v in md.items():
        if isinstance(v, str):
            out[k] = _normalize_text(v)
        else:
            # keep non-str as-is (None, numbers, lists, dicts)
            out[k] = v
    return out


def normalize_documents(docs: Iterable[Any]) -> None:
    """
    Mutates doc objects in-place:
      - doc.page_content (if exists)
      - doc.metadata (if exists)
    Compatible with LangChain Document-like objects.
    """
    for d in docs or []:
        # page_content
        if hasattr(d, "page_content"):
            try:
                pc = getattr(d, "page_content", "") or ""
                setattr(d, "page_content", _normalize_text(pc))
            except Exception:
                pass

        # metadata
        if hasattr(d, "metadata"):
            try:
                md = getattr(d, "metadata", {}) or {}
                setattr(d, "metadata", _normalize_metadata(md))
            except Exception:
                pass


def chunk_documents(docs: List[Any], chunk_size: int = 400, overlap: int = 50) -> List[Any]:
    """
    🎯 Smart Chunking: แบ่งเอกสารยาวเป็น chunks เล็กๆ
    
    กฎการแบ่ง:
    - เอกสารสั้น (< 300 ตัวอักษร) → ไม่แบ่ง
    - เอกสารยาว (> 600 ตัวอักษร) → แบ่งเป็น chunks
    - แบ่งตามย่อหน้า/ประโยค (รักษา context)
    - มี overlap 50 ตัวอักษร (ไม่เสีย context)
    
    Args:
        docs: List of Document objects
        chunk_size: ขนาด chunk (ตัวอักษร)
        overlap: ส่วนทับซ้อน (ตัวอักษร)
    
    Returns:
        List of Documents (อาจมากกว่าเดิม ถ้ามีการแบ่ง)
    """
    chunked_docs = []
    chunked_count = 0
    kept_count = 0
    
    for doc in docs:
        content = getattr(doc, "page_content", "") or ""
        metadata = getattr(doc, "metadata", {}) or {}
        
        # ตัดสินใจว่าควร chunk หรือไม่
        if should_chunk_document(content, metadata):
            # แบ่ง document เป็น chunks
            chunks = chunk_document_smart(
                content=content,
                metadata=metadata,
                chunk_size=chunk_size,
                overlap=overlap
            )
            
            # สร้าง Document objects สำหรับแต่ละ chunk
            for chunk_data in chunks:
                chunked_docs.append(Document(
                    page_content=chunk_data['content'],
                    metadata=chunk_data['metadata']
                ))
            
            chunked_count += 1
        else:
            # เอกสารสั้น เก็บไว้ตามเดิม
            chunked_docs.append(doc)
            kept_count += 1
    
    print(f"[Chunking] Total docs: {len(docs)}")
    print(f"[Chunking] Chunked: {chunked_count} docs → {len(chunked_docs) - kept_count} chunks")
    print(f"[Chunking] Kept as-is: {kept_count} docs")
    print(f"[Chunking] Final total: {len(chunked_docs)} chunks")
    
    return chunked_docs


def main():
    dl = DataLoader(config={})
    dl.load_from_google_sheet(SHEET_URL, source_name="google_sheet")
    docs = dl.documents

    # ✅ Step 1: Normalize text before indexing
    normalize_documents(docs)
    
    # 🎯 Step 2: Semantic Chunking (Field-based)
    # เร็วกว่า sliding window 100 เท่า! (15 seconds vs 30+ minutes)
    print(f"\n🎯 Starting semantic field-based chunking...")
    print(f"   Original documents: {len(docs)}")
    
    chunker = SemanticChunker(min_chunk_chars=200, max_chunk_chars=600)
    chunked_docs = chunker.chunk_documents(docs)
    
    stats = chunker.get_stats(docs, chunked_docs)
    print(f"\n📊 Chunking Statistics:")
    print(f"   Original: {stats['original_count']} documents")
    print(f"   Chunked: {stats['chunked_count']} chunks")
    print(f"   Expansion: {stats['expansion_ratio']}x")
    print(f"   Avg chunk size: {stats['avg_chunk_size']} chars")
    print(f"   Chunk types: {stats['chunk_types']}")

    print(f"\n[Ingest] Final chunks={len(chunked_docs)}")
    ingest_documents(chunked_docs, reset=True)  # ✅ wipe old local chroma before rebuild
    print('[Ingest] ✅ Done! Start server: python code/app.py')


if __name__ == "__main__":
    main()
    