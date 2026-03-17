# code/utils/text_chunker.py
"""
Smart Text Chunker สำหรับ RAG
แบ่งเอกสารยาวเป็น chunks เล็กๆ พร้อม overlap

Strategies:
1. Semantic Chunking - แบ่งตามความหมาย (ประโยค, ย่อหน้า)
2. Sliding Window - แบ่งตามขนาด + overlap
3. Hybrid - รวมทั้ง 2 วิธี

เหมาะสำหรับ:
- เอกสารยาว > 500 ตัวอักษร
- ต้องการลด token usage
- ต้องการ precision สูง
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Chunk:
    """Chunk ของข้อความพร้อม metadata"""
    content: str
    chunk_index: int
    total_chunks: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]


class SmartTextChunker:
    """
    Smart chunker ที่เข้าใจภาษาไทย
    
    Features:
    - รักษา context ด้วย overlap
    - ไม่ตัดคำครึ่งประโยค
    - แบ่งตามย่อหน้า/หัวข้อถ้าเจอ
    """
    
    def __init__(
        self,
        chunk_size: int = 400,
        overlap: int = 50,
        min_chunk_size: int = 100,
        respect_paragraphs: bool = True
    ):
        """
        Args:
            chunk_size: ขนาด chunk (ตัวอักษร)
            overlap: ส่วนทับซ้อนระหว่าง chunks (ตัวอักษร)
            min_chunk_size: ขนาดขั้นต่ำของ chunk
            respect_paragraphs: พยายามไม่ตัดย่อหน้า
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.respect_paragraphs = respect_paragraphs
        
        # Patterns สำหรับภาษาไทย
        self.paragraph_pattern = re.compile(r'\n\s*\n')
        self.sentence_pattern = re.compile(r'[.!?។]\s+')
        self.list_item_pattern = re.compile(r'\n\s*[-•*\d]+[\.)]\s+')
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        แบ่งข้อความเป็น chunks
        
        Returns:
            List of Chunk objects
        """
        if not text or len(text) < self.min_chunk_size:
            # ข้อความสั้น ไม่ต้องแบ่ง
            return [Chunk(
                content=text,
                chunk_index=0,
                total_chunks=1,
                start_char=0,
                end_char=len(text),
                metadata=metadata or {}
            )]
        
        chunks = []
        
        # Strategy 1: ลองแบ่งตามย่อหน้าก่อน
        if self.respect_paragraphs:
            chunks = self._chunk_by_paragraphs(text, metadata)
        
        # Strategy 2: ถ้าย่อหน้ายาวเกิน ใช้ sliding window
        if not chunks or any(len(c.content) > self.chunk_size * 1.5 for c in chunks):
            chunks = self._chunk_by_sliding_window(text, metadata)
        
        return chunks
    
    def _chunk_by_paragraphs(self, text: str, metadata: Optional[Dict[str, Any]]) -> List[Chunk]:
        """แบ่งตามย่อหน้า รักษา semantic"""
        paragraphs = self.paragraph_pattern.split(text)
        chunks = []
        current_chunk = ""
        start_char = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # ถ้าเพิ่มย่อหน้านี้แล้วยาวเกิน → บันทึก chunk เดิม
            if current_chunk and len(current_chunk) + len(para) > self.chunk_size:
                chunks.append(Chunk(
                    content=current_chunk.strip(),
                    chunk_index=len(chunks),
                    total_chunks=0,  # จะอัพเดททีหลัง
                    start_char=start_char,
                    end_char=start_char + len(current_chunk),
                    metadata=metadata or {}
                ))
                
                # เริ่ม chunk ใหม่ พร้อม overlap
                overlap_text = self._get_overlap(current_chunk)
                current_chunk = overlap_text + para
                start_char = start_char + len(current_chunk) - len(para) - len(overlap_text)
            else:
                current_chunk += ("\n\n" if current_chunk else "") + para
        
        # บันทึก chunk สุดท้าย
        if current_chunk:
            chunks.append(Chunk(
                content=current_chunk.strip(),
                chunk_index=len(chunks),
                total_chunks=0,
                start_char=start_char,
                end_char=start_char + len(current_chunk),
                metadata=metadata or {}
            ))
        
        # อัพเดท total_chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _chunk_by_sliding_window(self, text: str, metadata: Optional[Dict[str, Any]]) -> List[Chunk]:
        """แบ่งแบบ sliding window + overlap"""
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            # คำนวณจุดสิ้นสุด
            end = min(start + self.chunk_size, text_len)
            
            # ถ้าไม่ใช่ chunk สุดท้าย พยายามตัดที่จุดที่เหมาะสม
            if end < text_len:
                # หาจุดตัดที่ดี: ท้ายประโยค > ท้ายบรรทัด > ช่องว่าง
                cutpoint = self._find_best_cutpoint(text, start, end)
                if cutpoint > start:
                    end = cutpoint
            
            chunk_text = text[start:end].strip()
            
            # ข้าม chunk ว่างๆ
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(Chunk(
                    content=chunk_text,
                    chunk_index=len(chunks),
                    total_chunks=0,
                    start_char=start,
                    end_char=end,
                    metadata=metadata or {}
                ))
            
            # เลื่อนไปจุดถัดไป (มี overlap)
            start = end - self.overlap
            
            # ป้องกัน infinite loop
            if start >= end:
                start = end
        
        # อัพเดท total_chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _find_best_cutpoint(self, text: str, start: int, end: int) -> int:
        """หาจุดตัดที่ดีที่สุด"""
        search_range = text[start:end]
        
        # 1. หาท้ายประโยค (. ! ? ตามด้วยช่องว่าง)
        for match in self.sentence_pattern.finditer(search_range):
            pos = start + match.end()
            if pos > start + self.min_chunk_size:
                return pos
        
        # 2. หาท้ายบรรทัด
        last_newline = search_range.rfind('\n')
        if last_newline > self.min_chunk_size:
            return start + last_newline + 1
        
        # 3. หาช่องว่าง
        for i in range(len(search_range) - 1, self.min_chunk_size, -1):
            if search_range[i].isspace():
                return start + i + 1
        
        # 4. ไม่เจอ ใช้ตำแหน่งเดิม
        return end
    
    def _get_overlap(self, text: str) -> str:
        """ดึงส่วน overlap จากท้ายข้อความ"""
        if len(text) <= self.overlap:
            return text
        
        # พยายามตัดที่จุดที่เหมาะสม
        overlap_text = text[-self.overlap:]
        
        # หาจุดเริ่มที่ดี (หลังช่องว่าง/ขึ้นบรรทัดใหม่)
        for i, char in enumerate(overlap_text):
            if char.isspace():
                return overlap_text[i+1:]
        
        return overlap_text


# 🎯 Utility functions สำหรับใช้ใน ingest_local.py

def should_chunk_document(content: str, metadata: Dict[str, Any]) -> bool:
    """
    ตัดสินใจว่าเอกสารควรถูก chunk หรือไม่
    
    กฎ:
    - เอกสารยาว > 600 ตัวอักษร → chunk
    - เอกสารที่มีขั้นตอน/รายการ → chunk
    - เอกสารสั้น < 300 → ไม่ chunk
    """
    content_len = len(content)
    
    # เอกสารสั้นมาก ไม่ต้อง chunk
    if content_len < 300:
        return False
    
    # เอกสารยาวมาก ต้อง chunk
    if content_len > 600:
        return True
    
    # เอกสารปานกลาง ดูว่ามีโครงสร้างไหม
    has_structure = (
        '\n\n' in content or  # มีย่อหน้าหลายย่อหน้า
        content.count('\n') > 5 or  # มีหลายบรรทัด
        bool(re.search(r'\d+[.)]\s', content))  # มีรายการลำดับ
    )
    
    return has_structure


def chunk_document_smart(
    content: str,
    metadata: Dict[str, Any],
    chunk_size: int = 400,
    overlap: int = 50
) -> List[Dict[str, Any]]:
    """
    แบ่งเอกสารเป็น chunks พร้อม metadata
    
    Returns:
        List of dicts with 'content' and 'metadata'
    """
    chunker = SmartTextChunker(
        chunk_size=chunk_size,
        overlap=overlap,
        min_chunk_size=100,
        respect_paragraphs=True
    )
    
    chunks = chunker.chunk_text(content, metadata)
    
    # แปลงเป็น format ที่ ingest ใช้
    result = []
    for chunk in chunks:
        chunk_metadata = chunk.metadata.copy()
        chunk_metadata.update({
            'chunk_index': chunk.chunk_index,
            'total_chunks': chunk.total_chunks,
            'chunk_start': chunk.start_char,
            'chunk_end': chunk.end_char,
            'is_chunked': True
        })
        
        result.append({
            'content': chunk.content,
            'metadata': chunk_metadata
        })
    
    return result


if __name__ == "__main__":
    # ทดสอบ
    sample_text = """
    การขอใบอนุญานประกอบกิจการร้านอาหาร
    
    ขั้นตอนที่ 1: เตรียมเอกสาร
    - สำเนาบัตรประชาชน
    - สำเนาทะเบียนบ้าน
    - หนังสือรับรองการจดทะเบียนนิติบุคคล
    
    ขั้นตอนที่ 2: ยื่นคำขอ
    ยื่นคำขอได้ที่สำนักงานเขต หรือผ่านระบบออนไลน์
    
    ขั้นตอนที่ 3: ตรวจสอบสถานที่
    เจ้าหน้าที่จะเข้าตรวจสอบสถานที่ประกอบการ
    """ * 3  # ทำให้ยาวขึ้น
    
    chunker = SmartTextChunker(chunk_size=200, overlap=30)
    chunks = chunker.chunk_text(sample_text, {"topic": "ใบอนุญาน"})
    
    print(f"แบ่งได้ {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1}/{len(chunks)} ---")
        print(f"Length: {len(chunk.content)} chars")
        print(f"Content: {chunk.content[:100]}...")
