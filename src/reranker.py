from sentence_transformers import CrossEncoder
from typing import List, Tuple, Dict
import numpy as np
from logging_config import get_logger
logger = get_logger(__name__)


class Reranker:
    """
    Re-ranking module for improving retrieval quality in RAG systems.
    Uses cross-encoder models to re-score candidate documents.
    """
    
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Initialize the re-ranker with a cross-encoder model.
        
        Args:
            model_name: HuggingFace model name for cross-encoder
                       Popular options:
                       - 'cross-encoder/ms-marco-MiniLM-L-6-v2' (fast, good quality)
                       - 'cross-encoder/ms-marco-MiniLM-L-12-v2' (slower, better quality)
                       - 'cross-encoder/ms-marco-TinyBERT-L-2-v2' (fastest, lower quality)
        """
        logger.info(f"Loading re-ranker model: {model_name}")
        self.model = CrossEncoder(model_name)
        self.model_name = model_name
        logger.info(f"Re-ranker model loaded successfully: {model_name}")
                                                                                             
    
    def rerank(
        self, 
        query: str, 
        documents: List[str], 
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Re-rank documents based on their relevance to the query.
        
        Args:
            query: Search query string
            documents: List of document texts to re-rank
            top_k: Number of top documents to return
            
        Returns:
            List of (document_index, relevance_score) tuples sorted by relevance (descending)
        """
        if not documents:
            logger.warning("Rerank called with empty document list")
            return []
        
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Get relevance scores from cross-encoder
        scores = self.model.predict(pairs)
        
        # Sort by score (descending) and return top_k
        ranked_results = sorted(
            enumerate(scores), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        
        logger.debug(f"Re-ranked {len(documents)} documents, returning top {len(ranked_results)}")
        return ranked_results
    
    
    def rerank_with_metadata(
        self,
        query: str,
        candidates: List[Dict],
        text_field: str = 'content',
        top_k: int = 5
    ) -> List[Dict]:
        """
        Re-rank candidate documents with metadata.
        
        Args:
            query: Search query string
            candidates: List of candidate dictionaries containing document data {"doc_id": ..., "content": ...}
            text_field: Field name containing the text to re-rank on
            top_k: Number of top documents to return
            
        Returns:
            List of re-ranked candidate dictionaries with added 'rerank_score' field
        """
        if not candidates:
            return []
        
        # Extract texts for re-ranking
        documents = [str(candidate.get(text_field, '')) for candidate in candidates]
        
        # Get re-ranked results
        ranked_indices_scores = self.rerank(query, documents, top_k)
        
        # Build result list with rerank scores
        reranked_results = []
        for idx, score in ranked_indices_scores:
            result = candidates[idx].copy()
            result['rerank_score'] = float(score)
            reranked_results.append(result)
        
        return reranked_results
    
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'type': 'cross-encoder'
        }
