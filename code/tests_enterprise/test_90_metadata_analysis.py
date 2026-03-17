"""
Test 90: Metadata Analysis & Quality Assessment
Tests actual metadata structure from ChromaDB to understand data quality
NO HARDCODED VALUES - all extracted from real data
"""
from __future__ import annotations

import pytest
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Set, Any

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


@pytest.fixture
def chroma_client():
    """Connect to local ChromaDB"""
    if not CHROMADB_AVAILABLE:
        pytest.skip("ChromaDB not available")
    
    # Try both possible locations
    import os
    possible_paths = [
        "./local_chroma_v2",
        "./code/local_chroma_v2",
        "../local_chroma_v2"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            client = chromadb.PersistentClient(path=path)
            return client
    
    pytest.skip(f"ChromaDB not found in any of: {possible_paths}")
    return None


@pytest.fixture
def collection_data(chroma_client):
    """Get all data from ChromaDB collection"""
    collections = chroma_client.list_collections()
    
    if not collections:
        pytest.skip("No collections found in ChromaDB")
    
    collection = collections[0]  # Use first collection
    results = collection.get()
    
    return {
        'collection': collection,
        'ids': results['ids'],
        'documents': results['documents'],
        'metadatas': results['metadatas']
    }


class TestMetadataStructure:
    """Test metadata field existence and quality"""
    
    def test_metadata_fields_exist(self, collection_data):
        """Verify critical metadata fields exist"""
        metadatas = collection_data['metadatas']
        
        if not metadatas:
            pytest.skip("No metadata found")
        
        # Collect all unique metadata keys
        all_keys = set()
        for metadata in metadatas:
            all_keys.update(metadata.keys())
        
        print(f"\n📊 Found {len(all_keys)} unique metadata fields:")
        for key in sorted(all_keys):
            count = sum(1 for m in metadatas if key in m)
            coverage = (count / len(metadatas)) * 100
            print(f"   - {key}: {coverage:.1f}% coverage ({count}/{len(metadatas)})")
        
        # At least some metadata should exist
        assert len(all_keys) > 0, "No metadata fields found"
        
        # Store for other tests
        return all_keys
    
    def test_topic_fields_coverage(self, collection_data):
        """Test coverage of topic-related fields"""
        metadatas = collection_data['metadatas']
        
        # Fields that could contain topic information
        topic_fields = [
            'operation_topic',
            'แนวคำตอบ',
            'หัวข้อการดำเนินการย่อย',
            'topic',
            'section',
            'doc_type'
        ]
        
        coverage = {}
        for field in topic_fields:
            count = sum(1 for m in metadatas if field in m and m.get(field))
            coverage[field] = (count / len(metadatas)) * 100
        
        print(f"\n📋 Topic field coverage:")
        for field, pct in sorted(coverage.items(), key=lambda x: -x[1]):
            print(f"   {field}: {pct:.1f}%")
        
        # At least ONE topic field should have >50% coverage
        max_coverage = max(coverage.values()) if coverage else 0
        assert max_coverage > 50, f"No topic field has >50% coverage (max: {max_coverage:.1f}%)"
    
    def test_entity_type_normalization(self, collection_data):
        """Test entity_type field quality"""
        metadatas = collection_data['metadatas']
        
        entity_types = []
        for m in metadatas:
            if 'entity_type_normalized' in m:
                entity_types.append(m['entity_type_normalized'])
            elif 'entity_type' in m:
                entity_types.append(m['entity_type'])
        
        if not entity_types:
            pytest.skip("No entity_type fields found")
        
        unique_types = set(entity_types)
        print(f"\n🏢 Entity types found: {unique_types}")
        
        # Should have normalized values
        assert len(unique_types) > 0, "No entity types found"
        assert len(unique_types) < 20, f"Too many entity types ({len(unique_types)}) - needs normalization"


class TestMenuQuality:
    """Test quality of data for menu generation"""
    
    def test_extract_menu_worthy_topics(self, collection_data):
        """Extract and validate menu-worthy topics from metadata"""
        metadatas = collection_data['metadatas']
        
        # Extract unique topics from various fields
        topics = set()
        topic_sources = defaultdict(set)
        
        # Priority fields for menu topics
        priority_fields = [
            'operation_topic',
            'แนวคำตอบ',
            'หัวข้อการดำเนินการย่อย',
        ]
        
        for metadata in metadatas:
            for field in priority_fields:
                if field in metadata and metadata[field]:
                    value = str(metadata[field]).strip()
                    if self._is_menu_worthy(value):
                        topics.add(value)
                        topic_sources[value].add(field)
        
        print(f"\n✨ Menu-worthy topics found: {len(topics)}")
        print(f"\nTop 10 topics:")
        for i, topic in enumerate(sorted(topics)[:10], 1):
            sources = ', '.join(topic_sources[topic])
            print(f"   {i}. {topic[:60]}... (from: {sources})")
        
        # Should have at least 5 menu-worthy topics
        assert len(topics) >= 5, f"Only {len(topics)} menu-worthy topics found, need at least 5"
        
        # Store for use in conversation tests
        return sorted(topics)
    
    def _is_menu_worthy(self, text: str) -> bool:
        """Check if text is suitable for menu display"""
        if not text or len(text) < 3:
            return False
        
        if len(text) > 150:  # Too long for menu
            return False
        
        # Reject obvious noise
        noise_patterns = [
            'nan', 'null', 'none', 'n/a', 
            'ไม่มี', 'ไม่ระบุ', '...', '???'
        ]
        
        text_lower = text.lower()
        if any(pattern in text_lower for pattern in noise_patterns):
            return False
        
        # Must have Thai or English content (not just numbers/symbols)
        has_content = any(c.isalpha() for c in text)
        if not has_content:
            return False
        
        return True
    
    def test_slot_data_availability(self, collection_data):
        """Test if data contains slot-relevant information"""
        metadatas = collection_data['metadatas']
        
        # Look for slot-related metadata
        slot_indicators = {
            'location': ['กรุงเทพ', 'ต่างจังหวัด', 'location', 'ที่ตั้ง'],
            'entity_type': ['นิติบุคคล', 'บุคคลธรรมดา', 'entity_type'],
            'business_type': ['ร้านอาหาร', 'business_type', 'ประเภท']
        }
        
        found_slots = defaultdict(int)
        
        for metadata in metadatas:
            # Check all metadata values
            all_values = ' '.join(str(v) for v in metadata.values())
            
            for slot_name, indicators in slot_indicators.items():
                if any(ind in all_values for ind in indicators):
                    found_slots[slot_name] += 1
        
        print(f"\n🎯 Slot-related data found:")
        for slot, count in found_slots.items():
            pct = (count / len(metadatas)) * 100
            print(f"   {slot}: {count} docs ({pct:.1f}%)")
        
        # Should find at least some slot-related data
        assert len(found_slots) > 0, "No slot-related data found in metadata"


class TestContentQuality:
    """Test quality of document content"""
    
    def test_document_content_length(self, collection_data):
        """Test that documents have reasonable content"""
        documents = collection_data['documents']
        
        lengths = [len(doc) for doc in documents]
        
        avg_length = sum(lengths) / len(lengths)
        min_length = min(lengths)
        max_length = max(lengths)
        
        print(f"\n📄 Document statistics:")
        print(f"   Total documents: {len(documents)}")
        print(f"   Average length: {avg_length:.0f} chars")
        print(f"   Min length: {min_length} chars")
        print(f"   Max length: {max_length} chars")
        
        # Documents should have meaningful content
        assert avg_length > 50, f"Average document too short ({avg_length:.0f} chars)"
        assert min_length > 0, "Some documents are empty"
    
    def test_thai_content_ratio(self, collection_data):
        """Test that documents contain Thai language content"""
        documents = collection_data['documents']
        
        thai_doc_count = 0
        for doc in documents:
            # Check if document has Thai characters
            if any('\u0e00' <= c <= '\u0e7f' for c in doc):
                thai_doc_count += 1
        
        thai_ratio = (thai_doc_count / len(documents)) * 100
        
        print(f"\n🇹🇭 Thai content ratio: {thai_ratio:.1f}%")
        
        # Most documents should have Thai content
        assert thai_ratio > 80, f"Only {thai_ratio:.1f}% documents have Thai content"
    
    def test_regulatory_keywords(self, collection_data):
        """Test presence of regulatory keywords in documents"""
        documents = collection_data['documents']
        
        # Thai regulatory keywords
        keywords = {
            'ใบอนุญาต': 0,
            'จดทะเบียน': 0,
            'ภาษี': 0,
            'เอกสาร': 0,
            'ขั้นตอน': 0,
            'ค่าธรรมเนียม': 0,
            'หน่วยงาน': 0,
        }
        
        all_content = ' '.join(documents)
        
        for keyword in keywords:
            keywords[keyword] = all_content.count(keyword)
        
        print(f"\n🔍 Regulatory keyword frequency:")
        for keyword, count in sorted(keywords.items(), key=lambda x: -x[1]):
            print(f"   {keyword}: {count} occurrences")
        
        # Should find regulatory keywords
        total_keywords = sum(keywords.values())
        assert total_keywords > 0, "No regulatory keywords found in documents"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
