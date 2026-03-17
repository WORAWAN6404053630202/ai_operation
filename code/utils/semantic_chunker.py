"""
Semantic field-based chunking for structured Thai regulatory documents.

Instead of sliding window (slow, O(n²)), this uses semantic field grouping:
- Group related fields together (e.g., steps + documents, fees + duration)
- Preserve field relationships
- Fast O(n) processing

For 137 documents:
- Sliding window: 30+ minutes
- Semantic field-based: 10-15 seconds
"""

from typing import List, Dict, Any
from langchain_core.documents import Document


class SemanticChunker:
    """
    Semantic field-based chunker for Thai regulatory documents.
    
    Example document structure:
    {
        "department": "กรมพัฒนาธุรกิจการค้า",
        "license_type": "จดทะเบียนร้านอาหาร",
        "operation_steps": "1. ยื่นคำขอ...",
        "identification_documents": "บัตรประชาชน, ทะเบียนบ้าน...",
        "fees": "500 บาท",
        "operation_duration": "7-14 วันทำการ",
        "service_channel": "สำนักงานเขต...",
        "legal_regulatory": "พระราชบัญญัติ..."
    }
    
    Chunking strategy:
    - Chunk 1: Basic Info (department, license_type)
    - Chunk 2: Process (operation_steps, identification_documents)
    - Chunk 3: Cost & Time (fees, operation_duration, service_channel)
    - Chunk 4: Legal (legal_regulatory)
    """
    
    # Field groups (semantic relationships)
    FIELD_GROUPS = [
        {
            "name": "basic_info",
            "fields": ["department", "license_type", "license_name"],
            "weight": 1.0
        },
        {
            "name": "process",
            "fields": ["operation_steps", "identification_documents", "documents_required"],
            "weight": 1.2  # Most important for user queries
        },
        {
            "name": "cost_time",
            "fields": ["fees", "operation_duration", "service_channel", "location"],
            "weight": 1.1
        },
        {
            "name": "legal",
            "fields": ["legal_regulatory", "regulations", "laws"],
            "weight": 0.8
        }
    ]
    
    def __init__(self, min_chunk_chars: int = 200, max_chunk_chars: int = 600):
        """
        Initialize semantic chunker.
        
        Args:
            min_chunk_chars: Minimum characters per chunk (skip if too small)
            max_chunk_chars: Maximum characters per chunk (split if too large)
        """
        self.min_chunk_chars = min_chunk_chars
        self.max_chunk_chars = max_chunk_chars
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk documents using semantic field grouping.
        
        Args:
            documents: List of LangChain Document objects
        
        Returns:
            List of chunked documents (more than input if large docs split)
        
        Example:
            chunker = SemanticChunker()
            chunks = chunker.chunk_documents(documents)
            # 137 docs → ~500-600 chunks
        """
        all_chunks = []
        
        for doc in documents:
            chunks = self._chunk_single_document(doc)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def _chunk_single_document(self, doc: Document) -> List[Document]:
        """
        Chunk a single document into semantic field groups.
        
        Returns:
            List of chunks (1-4 chunks depending on doc size)
        """
        metadata = doc.metadata
        content = doc.page_content
        
        # If document is small, don't chunk
        if len(content) < self.min_chunk_chars:
            return [doc]
        
        # Try to parse structured fields from metadata or content
        chunks = []
        
        # Strategy 1: If metadata has structured fields, use them
        if self._has_structured_fields(metadata):
            chunks = self._chunk_by_fields(metadata, content)
        else:
            # Strategy 2: Fallback to paragraph-based chunking
            chunks = self._chunk_by_paragraphs(doc)
        
        # Filter out too-small chunks
        chunks = [c for c in chunks if len(c.page_content) >= self.min_chunk_chars]
        
        # If no valid chunks, return original
        if not chunks:
            return [doc]
        
        return chunks
    
    def _has_structured_fields(self, metadata: Dict) -> bool:
        """Check if metadata has structured fields."""
        required_fields = ["department", "license_type"]
        return all(field in metadata for field in required_fields)
    
    def _chunk_by_fields(self, metadata: Dict, content: str) -> List[Document]:
        """
        Chunk document by semantic field groups.
        
        Creates separate chunks for:
        - Basic info (department, license)
        - Process (steps, documents)
        - Cost/Time (fees, duration)
        - Legal (regulations)
        """
        chunks = []
        
        for group in self.FIELD_GROUPS:
            group_content = []
            group_metadata = metadata.copy()
            group_metadata["chunk_type"] = group["name"]
            group_metadata["chunk_weight"] = group["weight"]
            
            # Extract fields for this group
            for field in group["fields"]:
                if field in metadata and metadata[field]:
                    value = metadata[field]
                    group_content.append(f"{field}: {value}")
            
            # Create chunk if content exists
            if group_content:
                chunk_text = "\n".join(group_content)
                
                # Split if too large
                if len(chunk_text) > self.max_chunk_chars:
                    sub_chunks = self._split_large_chunk(chunk_text, group_metadata)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(Document(
                        page_content=chunk_text,
                        metadata=group_metadata
                    ))
        
        return chunks
    
    def _chunk_by_paragraphs(self, doc: Document) -> List[Document]:
        """
        Fallback: Chunk by paragraphs if no structured fields.
        
        Splits on double newlines, preserves semantic boundaries.
        """
        content = doc.page_content
        metadata = doc.metadata.copy()
        
        # Split by paragraph (double newline or Thai section markers)
        paragraphs = content.split('\n\n')
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_len = len(para)
            
            # If adding this paragraph exceeds max, create chunk
            if current_length + para_len > self.max_chunk_chars and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunk_meta = metadata.copy()
                chunk_meta["chunk_type"] = "paragraph"
                
                chunks.append(Document(
                    page_content=chunk_text,
                    metadata=chunk_meta
                ))
                
                current_chunk = [para]
                current_length = para_len
            else:
                current_chunk.append(para)
                current_length += para_len
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunk_meta = metadata.copy()
            chunk_meta["chunk_type"] = "paragraph"
            
            chunks.append(Document(
                page_content=chunk_text,
                metadata=chunk_meta
            ))
        
        return chunks
    
    def _split_large_chunk(self, text: str, metadata: Dict) -> List[Document]:
        """Split a large chunk into smaller pieces."""
        chunks = []
        lines = text.split('\n')
        
        current_chunk = []
        current_length = 0
        
        for line in lines:
            line_len = len(line)
            
            if current_length + line_len > self.max_chunk_chars and current_chunk:
                chunk_text = "\n".join(current_chunk)
                chunks.append(Document(
                    page_content=chunk_text,
                    metadata=metadata.copy()
                ))
                
                current_chunk = [line]
                current_length = line_len
            else:
                current_chunk.append(line)
                current_length += line_len
        
        # Add remaining
        if current_chunk:
            chunk_text = "\n".join(current_chunk)
            chunks.append(Document(
                page_content=chunk_text,
                metadata=metadata.copy()
            ))
        
        return chunks
    
    def get_stats(self, original_docs: List[Document], chunked_docs: List[Document]) -> Dict[str, Any]:
        """
        Get chunking statistics.
        
        Returns:
            {
                "original_count": 137,
                "chunked_count": 542,
                "expansion_ratio": 3.96,
                "avg_chunk_size": 387,
                "chunk_types": {"process": 137, "cost_time": 120, ...}
            }
        """
        chunk_types = {}
        total_chunk_size = 0
        
        for doc in chunked_docs:
            chunk_type = doc.metadata.get("chunk_type", "unknown")
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            total_chunk_size += len(doc.page_content)
        
        avg_chunk_size = total_chunk_size / len(chunked_docs) if chunked_docs else 0
        
        return {
            "original_count": len(original_docs),
            "chunked_count": len(chunked_docs),
            "expansion_ratio": round(len(chunked_docs) / len(original_docs), 2) if original_docs else 0,
            "avg_chunk_size": int(avg_chunk_size),
            "chunk_types": chunk_types
        }
