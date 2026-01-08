import numpy as np
import pandas as pd
import faiss
import json
from typing import List, Tuple, Dict
import os


class FAISS_Backend:
    """FAISS-based vector similarity search backend for RAG system."""
    
    def __init__(self, embedding_dim: int = 1536):
        """
        Initialize FAISS backend.
        
        Args:
            embedding_dim: Dimension of embeddings (default 1536 for text-embedding-3-small)
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.id_mapping = {}  # Maps FAISS index IDs to document IDs
        self.reverse_id_mapping = {}  # Maps document IDs to FAISS index IDs
        self.embeddings = {}  # Stores embeddings for reference
        
        
        
    def create_index(self):
        """Create a new FAISS index using L2 distance."""
        # Using IndexFlatL2 for exact search, can be upgraded to IndexIVFFlat for large datasets
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.id_mapping = {}
        self.reverse_id_mapping = {}
        self.embeddings = {}
        
        
        
    def add_embeddings(self, embeddings_dict: Dict[int, List[float]]) -> int:
        """
        Add embeddings to the FAISS index.
        
        Args:
            embeddings_dict: Dictionary mapping document IDs to embedding vectors
            
        Returns:
            Total number of embeddings in the index
        """
        if self.index is None:
            self.create_index()
        
        # Convert embeddings to numpy array
        embedding_list = []
        doc_ids = []
        
        for doc_id, embedding in embeddings_dict.items():
            embedding_array = np.array(embedding, dtype='float32').reshape(1, -1)
            embedding_list.append(embedding_array)
            doc_ids.append(doc_id)
            self.embeddings[doc_id] = embedding
        
        if embedding_list:
            embeddings_array = np.vstack(embedding_list).astype('float32')
            
            # Map FAISS index IDs to document IDs
            start_idx = self.index.ntotal
            for i, doc_id in enumerate(doc_ids):
                faiss_id = start_idx + i
                self.id_mapping[faiss_id] = doc_id
                self.reverse_id_mapping[doc_id] = faiss_id
            
            # Add to index
            self.index.add(embeddings_array)
        
        return self.index.ntotal
    
    
    
    def search(self, query_embedding: List[float], k: int = 5) -> List[Tuple[int, float]]:
        """
        Search for k nearest neighbors to the query embedding.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of nearest neighbors to return
            
        Returns:
            List of (document_id, similarity_score) tuples sorted by relevance
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        
        query_array = np.array(query_embedding, dtype='float32').reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_array, min(k, self.index.ntotal))
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx >= 0:  # Valid index
                doc_id = self.id_mapping.get(idx)
                if doc_id is not None:
                    # Convert L2 distance to similarity score (higher is better)
                    # similarity = 1 / (1 + distance)
                    similarity = float(1.0 / (1.0 + distance))
                    results.append((doc_id, similarity))
        
        return results
    
    
    
    def save_index(self, index_path: str, metadata_path: str):
        """
        Save FAISS index and metadata to disk.
        
        Args:
            index_path: Path to save FAISS index file
            metadata_path: Path to save ID mapping metadata
        """
        if self.index is None:
            raise ValueError("No index to save. Please add embeddings first.")
        
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        metadata = {
            'embedding_dim': self.embedding_dim,
            'id_mapping': {str(k): v for k, v in self.id_mapping.items()},
            'reverse_id_mapping': {str(k): v for k, v in self.reverse_id_mapping.items()}
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        print(f"Index saved to {index_path}")
        print(f"Metadata saved to {metadata_path}")
        
        
        
    
    def load_index(self, index_path: str, metadata_path: str):
        """
        Load FAISS index and metadata from disk.
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to ID mapping metadata file
        """
        self.index = faiss.read_index(index_path)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.embedding_dim = metadata['embedding_dim']
        self.id_mapping = {int(k): v for k, v in metadata['id_mapping'].items()}
        self.reverse_id_mapping = {int(k): v for k, v in metadata['reverse_id_mapping'].items()}
        
        print(f"Index loaded from {index_path}")
        print(f"Metadata loaded from {metadata_path}")
        
        
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the current index."""
        if self.index is None:
            return {'status': 'empty', 'total_vectors': 0}
        
        return {
            'status': 'ready',
            'total_vectors': self.index.ntotal,
            'embedding_dim': self.embedding_dim,
            'index_type': str(type(self.index).__name__)
        }
        
        
    
    def clear_index(self):
        """Clear the index and all mappings."""
        self.index = None
        self.id_mapping = {}
        self.reverse_id_mapping = {}
        self.embeddings = {}
