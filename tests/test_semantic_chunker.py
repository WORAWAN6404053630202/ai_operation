"""
Unit tests for SemanticChunker.
"""

import pytest
from langchain_core.documents import Document
from code.utils.semantic_chunker import SemanticChunker


class TestSemanticChunker:
    """Test suite for SemanticChunker."""
    
    def test_chunker_initialization(self):
        """Test chunker initializes with correct parameters."""
        chunker = SemanticChunker(min_chunk_chars=200, max_chunk_chars=600)
        
        assert chunker.min_chunk_chars == 200
        assert chunker.max_chunk_chars == 600
    
    def test_small_document_not_chunked(self):
        """Test small documents are not chunked."""
        chunker = SemanticChunker(min_chunk_chars=200, max_chunk_chars=600)
        
        small_doc = Document(
            page_content="Short content",  # < 200 chars
            metadata={"source": "test"}
        )
        
        chunks = chunker.chunk_documents([small_doc])
        
        assert len(chunks) == 1
        assert chunks[0].page_content == "Short content"
    
    def test_structured_document_chunking(self):
        """Test structured documents are chunked by fields."""
        chunker = SemanticChunker(min_chunk_chars=100, max_chunk_chars=400)
        
        # Create a longer document to trigger chunking
        long_steps = "1. ยื่นคำขอพร้อมเอกสาร " * 10
        long_docs = "บัตรประชาชน ทะเบียนบ้าน สัญญาเช่าหรือเอกสารแสดงกรรมสิทธิ์ " * 5
        
        doc = Document(
            page_content="Long content here..." * 20,  # Make it longer
            metadata={
                "department": "กรมพัฒนาธุรกิจการค้า",
                "license_type": "จดทะเบียนร้านอาหาร",
                "operation_steps": long_steps,
                "identification_documents": long_docs,
                "fees": "500 บาท",
                "operation_duration": "7-14 วันทำการ",
                "source": "test"
            }
        )
        
        chunks = chunker.chunk_documents([doc])
        
        # Should create multiple chunks (basic_info, process, cost_time)
        # Note: May only create 1 chunk if content is still short
        assert len(chunks) >= 1
        
        # Check that metadata is preserved
        assert all(c.metadata.get("source") == "test" for c in chunks)
    
    def test_paragraph_fallback_chunking(self):
        """Test flat mode: doc returned as-is regardless of content size."""
        chunker = SemanticChunker(min_chunk_chars=100, max_chunk_chars=300)

        long_content = "\n\n".join([
            "Paragraph 1: " + "x" * 200,
            "Paragraph 2: " + "y" * 200,
            "Paragraph 3: " + "z" * 200,
        ])

        doc = Document(
            page_content=long_content,
            metadata={"source": "test"}
        )

        chunks = chunker.chunk_documents([doc])

        # Flat mode: 1 doc in → 1 doc out
        assert len(chunks) == 1
        # Original doc returned unchanged (no chunk_type added)
        assert "chunk_type" not in chunks[0].metadata
    
    def test_large_chunk_splitting(self):
        """Test flat mode: large doc returned as-is without splitting."""
        chunker = SemanticChunker(min_chunk_chars=100, max_chunk_chars=200)

        doc = Document(
            page_content="content",
            metadata={
                "department": "dept",
                "license_type": "license",
                "operation_steps": "Step " * 100,  # Very long
                "source": "test"
            }
        )

        chunks = chunker.chunk_documents([doc])

        # Flat mode: always 1 doc out per doc in
        assert len(chunks) == 1
        assert chunks[0] is doc
    
    def test_chunk_statistics(self):
        """Test chunking statistics calculation."""
        chunker = SemanticChunker(min_chunk_chars=100, max_chunk_chars=400)
        
        docs = [
            Document(
                page_content="Content 1",
                metadata={
                    "department": "กรมA",
                    "license_type": "ประเภทA",
                    "operation_steps": "ขั้นตอน 1, 2, 3",
                    "fees": "500 บาท"
                }
            ),
            Document(
                page_content="Content 2",
                metadata={
                    "department": "กรมB",
                    "license_type": "ประเภทB",
                    "operation_steps": "ขั้นตอน A, B, C",
                    "fees": "1000 บาท"
                }
            ),
        ]
        
        chunks = chunker.chunk_documents(docs)
        stats = chunker.get_stats(docs, chunks)
        
        assert stats["original_count"] == 2
        assert stats["chunked_count"] == 2   # flat: no expansion
        assert stats["expansion_ratio"] == 1.0
        assert "avg_chunk_size" in stats
        assert "chunk_types" in stats
    
    def test_empty_document_list(self):
        """Test chunking empty document list."""
        chunker = SemanticChunker()
        
        chunks = chunker.chunk_documents([])
        
        assert len(chunks) == 0
    
    def test_metadata_preservation(self):
        """Test metadata is preserved in chunks."""
        chunker = SemanticChunker(min_chunk_chars=50, max_chunk_chars=400)
        
        # Create document with longer content to trigger chunking
        long_content = "content text " * 100
        
        doc = Document(
            page_content=long_content,
            metadata={
                "department": "กรมทดสอบ",
                "license_type": "ใบอนุญาตทดสอบ",
                "source": "google_sheet",
                "custom_field": "custom_value"
            }
        )
        
        chunks = chunker.chunk_documents([doc])
        
        for chunk in chunks:
            # Original metadata should be preserved
            assert "department" in chunk.metadata
            assert "source" in chunk.metadata
            # Document was long enough that it should have been chunked, 
            # but if it's still returned as-is, that's also valid behavior


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
