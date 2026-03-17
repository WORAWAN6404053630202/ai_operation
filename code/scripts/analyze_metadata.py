"""
Script to analyze actual metadata from ChromaDB
Extracts real column names and sample values to understand data structure
"""
import sys
import json
from pathlib import Path

# Add code to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from collections import Counter, defaultdict

def analyze_metadata():
    """Analyze ChromaDB metadata structure"""
    
    # Connect to local ChromaDB
    client = chromadb.PersistentClient(path="./code/local_chroma_v2")
    
    # Get all collections
    collections = client.list_collections()
    print(f"\n{'='*80}")
    print(f"Found {len(collections)} collection(s)")
    print(f"{'='*80}\n")
    
    for collection in collections:
        print(f"\n{'='*80}")
        print(f"Collection: {collection.name}")
        print(f"{'='*80}")
        
        # Get all data
        results = collection.get()
        
        total_docs = len(results['ids'])
        print(f"\nTotal documents: {total_docs}")
        
        if total_docs == 0:
            print("No documents found in collection")
            continue
        
        # Analyze metadata structure
        metadata_keys = Counter()
        metadata_samples = defaultdict(set)
        
        for metadata in results['metadatas']:
            for key, value in metadata.items():
                metadata_keys[key] += 1
                # Store up to 10 unique sample values
                if len(metadata_samples[key]) < 10:
                    metadata_samples[key].add(str(value)[:100])
        
        # Print metadata field analysis
        print(f"\n{'─'*80}")
        print("METADATA FIELDS ANALYSIS:")
        print(f"{'─'*80}")
        
        for key, count in sorted(metadata_keys.items(), key=lambda x: -x[1]):
            coverage = (count / total_docs) * 100
            print(f"\n📊 Field: '{key}'")
            print(f"   Coverage: {count}/{total_docs} ({coverage:.1f}%)")
            print(f"   Sample values:")
            for i, sample in enumerate(sorted(metadata_samples[key])[:5], 1):
                print(f"      {i}. {sample}")
        
        # Analyze specific important fields
        print(f"\n{'─'*80}")
        print("CRITICAL FIELDS FOR MENU GENERATION:")
        print(f"{'─'*80}")
        
        critical_fields = [
            'operation_topic',
            'แนวคำตอบ',
            'เงื่อนไขและหลักเกณฑ์',
            'หัวข้อการดำเนินการย่อย',
            'entity_type',
            'entity_type_normalized',
            'หน่วยงาน',
            'section',
            'doc_type'
        ]
        
        for field in critical_fields:
            if field in metadata_keys:
                print(f"\n✅ '{field}' exists ({metadata_keys[field]} docs)")
                print(f"   Unique values: {len(metadata_samples[field])}")
                print(f"   Examples:")
                for i, val in enumerate(sorted(metadata_samples[field])[:3], 1):
                    print(f"      {i}. {val}")
            else:
                print(f"\n❌ '{field}' NOT FOUND")
        
        # Sample 5 complete documents
        print(f"\n{'─'*80}")
        print("SAMPLE DOCUMENTS (first 5):")
        print(f"{'─'*80}")
        
        for i in range(min(5, total_docs)):
            print(f"\n📄 Document {i+1}:")
            print(f"   ID: {results['ids'][i]}")
            print(f"   Content preview: {results['documents'][i][:150]}...")
            print(f"   Metadata:")
            for key, value in sorted(results['metadatas'][i].items()):
                print(f"      - {key}: {str(value)[:100]}")

if __name__ == "__main__":
    analyze_metadata()
