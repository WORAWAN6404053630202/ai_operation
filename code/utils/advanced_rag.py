"""
Advanced RAG techniques for Thai Regulatory AI.

Features:
- Hybrid search (semantic + keyword/BM25)
- Re-ranking with cross-encoder
- Query expansion
- Context compression
- Multi-query retrieval
- Reciprocal rank fusion

Usage:
    from code.utils.advanced_rag import HybridRetriever, QueryExpander, ContextCompressor
    
    # Hybrid search
    retriever = HybridRetriever(vector_store, alpha=0.5)
    docs = retriever.get_relevant_documents("query")
    
    # Query expansion
    expander = QueryExpander()
    expanded_queries = expander.expand("original query")
    
    # Context compression
    compressor = ContextCompressor()
    compressed = compressor.compress(docs, query)
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result with score."""
    document: Document
    score: float
    source: str  # semantic, keyword, or hybrid


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining semantic and keyword search.
    
    Uses reciprocal rank fusion to combine results.
    """
    
    def __init__(
        self,
        vector_store,
        alpha: float = 0.5,
        k: int = 10,
        enable_bm25: bool = True
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: Vector store for semantic search
            alpha: Weight for semantic search (0-1), 1-alpha for keyword
            k: Number of documents to retrieve
            enable_bm25: Enable BM25 keyword search
        """
        self.vector_store = vector_store
        self.alpha = alpha
        self.k = k
        self.enable_bm25 = enable_bm25
        
        # BM25 index (will be built from documents)
        self.bm25_index = None
        self.documents = []
    
    def _build_bm25_index(self, documents: List[Document]):
        """Build BM25 index from documents."""
        try:
            from rank_bm25 import BM25Okapi
            
            # Tokenize documents
            tokenized_docs = [
                doc.page_content.lower().split()
                for doc in documents
            ]
            
            self.bm25_index = BM25Okapi(tokenized_docs)
            self.documents = documents
            
            logger.info(f"Built BM25 index with {len(documents)} documents")
        except ImportError:
            logger.warning("rank_bm25 not installed, disabling keyword search")
            self.enable_bm25 = False
    
    def _semantic_search(self, query: str, k: int) -> List[SearchResult]:
        """Perform semantic search."""
        docs = self.vector_store.similarity_search_with_score(query, k=k)
        
        results = [
            SearchResult(
                document=doc,
                score=float(score),
                source="semantic"
            )
            for doc, score in docs
        ]
        
        return results
    
    def _keyword_search(self, query: str, k: int) -> List[SearchResult]:
        """Perform BM25 keyword search."""
        if not self.enable_bm25 or self.bm25_index is None:
            return []
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top k
        top_indices = np.argsort(scores)[-k:][::-1]
        
        results = [
            SearchResult(
                document=self.documents[idx],
                score=float(scores[idx]),
                source="keyword"
            )
            for idx in top_indices
        ]
        
        return results
    
    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[SearchResult],
        keyword_results: List[SearchResult],
        k: int = 60
    ) -> List[SearchResult]:
        """
        Combine results using reciprocal rank fusion.
        
        Args:
            semantic_results: Semantic search results
            keyword_results: Keyword search results
            k: RRF constant (default 60)
            
        Returns:
            Fused results
        """
        # Build score dict
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}
        
        # Add semantic results
        for rank, result in enumerate(semantic_results, start=1):
            doc_id = id(result.document)
            scores[doc_id] = scores.get(doc_id, 0) + self.alpha * (1.0 / (k + rank))
            doc_map[doc_id] = result.document
        
        # Add keyword results
        for rank, result in enumerate(keyword_results, start=1):
            doc_id = id(result.document)
            scores[doc_id] = scores.get(doc_id, 0) + (1 - self.alpha) * (1.0 / (k + rank))
            doc_map[doc_id] = result.document
        
        # Sort by score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        # Create results
        fused_results = [
            SearchResult(
                document=doc_map[doc_id],
                score=scores[doc_id],
                source="hybrid"
            )
            for doc_id in sorted_ids[:self.k]
        ]
        
        return fused_results
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Get relevant documents using hybrid search.
        
        Args:
            query: Search query
            
        Returns:
            List of relevant documents
        """
        # Semantic search
        semantic_results = self._semantic_search(query, k=self.k)
        
        # Keyword search
        keyword_results = []
        if self.enable_bm25:
            if self.bm25_index is None:
                # Build index from vector store
                all_docs = self.vector_store.similarity_search("", k=1000)
                self._build_bm25_index(all_docs)
            
            keyword_results = self._keyword_search(query, k=self.k)
        
        # Fuse results
        if keyword_results:
            fused_results = self._reciprocal_rank_fusion(
                semantic_results,
                keyword_results
            )
        else:
            fused_results = semantic_results
        
        return [result.document for result in fused_results]
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version of get_relevant_documents."""
        return self.get_relevant_documents(query)


class QueryExpander:
    """
    Expand queries for better retrieval.
    
    Techniques:
    - Synonym expansion
    - Related terms
    - Multi-query generation
    """
    
    def __init__(self):
        """Initialize query expander."""
        # Thai regulatory synonyms
        self.synonyms = {
            "ร้านอาหาร": ["ร้านค้า", "สถานประกอบการอาหาร", "ภัตตาคาร"],
            "จดทะเบียน": ["ลงทะเบียน", "ขึ้นทะเบียน", "ยื่นขอจดทะเบียน"],
            "ใบอนุญาต": ["ใบรับรอง", "ใบอนุมัติ", "ใบประกอบการ"],
            "ขั้นตอน": ["กระบวนการ", "วิธีการ", "ลำดับขั้น"],
            "เอกสาร": ["หลักฐาน", "เอกสารประกอบ", "ใบสำคัญ"]
        }
    
    def expand(self, query: str, max_expansions: int = 3) -> List[str]:
        """
        Expand query with synonyms.
        
        Args:
            query: Original query
            max_expansions: Maximum number of expanded queries
            
        Returns:
            List of expanded queries (including original)
        """
        expanded = [query]
        
        # Find synonyms in query
        for term, synonyms in self.synonyms.items():
            if term in query:
                for synonym in synonyms[:max_expansions - 1]:
                    expanded_query = query.replace(term, synonym)
                    if expanded_query not in expanded:
                        expanded.append(expanded_query)
        
        return expanded[:max_expansions]
    
    def generate_multi_queries(self, query: str, llm=None) -> List[str]:
        """
        Generate multiple query variations using LLM.
        
        Args:
            query: Original query
            llm: Language model for generation
            
        Returns:
            List of query variations
        """
        if llm is None:
            # Fallback to rule-based expansion
            return self.expand(query)
        
        # Use LLM to generate variations
        prompt = f"""Given this question in Thai, generate 3 different ways to ask the same question:

Original: {query}

Generate 3 variations:
1."""
        
        try:
            response = llm.invoke(prompt)
            variations = [query]  # Include original
            
            # Parse response (simplified)
            lines = response.content.split("\n")
            for line in lines:
                if line.strip() and not line.startswith("Original"):
                    # Remove numbering
                    variation = line.split(".", 1)[-1].strip()
                    if variation:
                        variations.append(variation)
            
            return variations[:4]  # Original + 3 variations
            
        except Exception as e:
            logger.error(f"Multi-query generation failed: {e}")
            return [query]


class ContextCompressor:
    """
    Compress retrieved context to most relevant parts.
    
    Reduces token usage while maintaining relevance.
    """
    
    def __init__(
        self,
        max_chars: int = 2000,
        relevance_threshold: float = 0.5
    ):
        """
        Initialize context compressor.
        
        Args:
            max_chars: Maximum characters in compressed context
            relevance_threshold: Minimum relevance score to include
        """
        self.max_chars = max_chars
        self.relevance_threshold = relevance_threshold
    
    def compress(
        self,
        documents: List[Document],
        query: str,
        preserve_order: bool = False
    ) -> str:
        """
        Compress documents to most relevant parts.
        
        Args:
            documents: Retrieved documents
            query: Original query
            preserve_order: Preserve document order
            
        Returns:
            Compressed context string
        """
        if not documents:
            return ""
        
        # Score each sentence by relevance
        scored_sentences = []
        
        for doc in documents:
            sentences = self._split_sentences(doc.page_content)
            
            for sentence in sentences:
                score = self._score_sentence(sentence, query)
                
                if score >= self.relevance_threshold:
                    scored_sentences.append({
                        "text": sentence,
                        "score": score,
                        "source": doc.metadata.get("source", "unknown")
                    })
        
        # Sort by score (unless preserving order)
        if not preserve_order:
            scored_sentences.sort(key=lambda x: x["score"], reverse=True)
        
        # Build compressed context
        compressed = []
        total_chars = 0
        
        for item in scored_sentences:
            sentence = item["text"]
            if total_chars + len(sentence) <= self.max_chars:
                compressed.append(sentence)
                total_chars += len(sentence)
            else:
                break
        
        return " ".join(compressed)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences (Thai-aware)."""
        import re
        
        # Split on Thai sentence endings
        sentences = re.split(r'[।\n]|(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _score_sentence(self, sentence: str, query: str) -> float:
        """
        Score sentence relevance to query.
        
        Args:
            sentence: Sentence text
            query: Query text
            
        Returns:
            Relevance score (0-1)
        """
        # Simple word overlap scoring
        query_words = set(query.lower().split())
        sentence_words = set(sentence.lower().split())
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words & sentence_words)
        score = overlap / len(query_words)
        
        return min(score, 1.0)


class ReRanker:
    """
    Re-rank documents using cross-encoder model.
    
    More accurate than bi-encoder for final ranking.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize re-ranker.
        
        Args:
            model_name: Cross-encoder model name
        """
        self.model_name = model_name
        self.model = None
    
    def _load_model(self):
        """Load cross-encoder model."""
        if self.model is None:
            try:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(self.model_name)
                logger.info(f"Loaded re-ranker model: {self.model_name}")
            except ImportError:
                logger.warning("sentence-transformers not installed, re-ranking disabled")
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """
        Re-rank documents by relevance.
        
        Args:
            query: Search query
            documents: Documents to re-rank
            top_k: Return top k documents
            
        Returns:
            List of (document, score) tuples
        """
        self._load_model()
        
        if self.model is None:
            # Fallback: return documents with dummy scores
            return [(doc, 1.0) for doc in documents]
        
        # Prepare pairs
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Score
        scores = self.model.predict(pairs)
        
        # Combine
        ranked = list(zip(documents, scores))
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        if top_k:
            ranked = ranked[:top_k]
        
        return ranked


if __name__ == "__main__":
    # Example usage
    from code.service.local_vector_store import LocalVectorStore
    
    # Initialize vector store
    vector_store = LocalVectorStore()
    
    # Hybrid retriever
    retriever = HybridRetriever(vector_store, alpha=0.7)
    docs = retriever.get_relevant_documents("ขั้นตอนการจดทะเบียนร้านอาหาร")
    print(f"Retrieved {len(docs)} documents")
    
    # Query expansion
    expander = QueryExpander()
    expanded = expander.expand("ใบอนุญาตร้านอาหาร")
    print(f"Expanded queries: {expanded}")
    
    # Context compression
    compressor = ContextCompressor(max_chars=500)
    compressed = compressor.compress(docs, "ขั้นตอนการจดทะเบียน")
    print(f"Compressed context: {compressed[:100]}...")
    
    # Re-ranking
    reranker = ReRanker()
    ranked = reranker.rerank("ขั้นตอนการจดทะเบียน", docs, top_k=3)
    print(f"Re-ranked top 3 documents")
