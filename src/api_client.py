import requests
import json
from typing import List, Dict, Tuple, Optional


class RAG_API_Client:
    """Client for interacting with RAG API server."""
    
    def __init__(self, host = "localhost", port = 5000):
        """
        Initialize the RAG API client.
        
        Args:
            host: Hostname of the API server
            port: Port number of the API server
        """
        self.base_url = f"http://{host}:{port}"
        self.api_prefix = f"{self.base_url}/api"
    
    def health_check(self) -> Dict:
        """Check API server health."""
        response = requests.get(f"{self.api_prefix}/health")
        response.raise_for_status()
        return response.json()
    
    def initialize(self) -> Dict:
        """Initialize the RAG system."""
        response = requests.post(f"{self.api_prefix}/initialize")
        response.raise_for_status()
        return response.json()
    
    def load_index(self) -> Dict:
        """Load pre-built FAISS index."""
        response = requests.post(f"{self.api_prefix}/load-index")
        response.raise_for_status()
        return response.json()
    
    def get_stats(self) -> Dict:
        """Get system statistics."""
        response = requests.get(f"{self.api_prefix}/stats")
        response.raise_for_status()
        return response.json()
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        response = requests.post(
            f"{self.api_prefix}/embed",
            json={"text": text}
        )
        response.raise_for_status()
        data = response.json()
        return data['embedding']
    
    def search(self, query: str, k: int = 5, use_rerank: bool = False, initial_k: int = 20, rewrite: bool = False) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results
            use_rerank: Whether to use re-ranking
            initial_k: Number of candidates to retrieve before re-ranking
            rewrite: Whether to rewrite query before retrieval
            
        Returns:
            List of search results
        """
        response = requests.post(
            f"{self.api_prefix}/search",
            json={"query": query, "k": k, "use_rerank": use_rerank, "initial_k": initial_k, "rewrite": rewrite}
        )
        response.raise_for_status()
        data = response.json()
        return data['results']
    
    def search_embedding(self, embedding: List[float], k: int = 5) -> List[Dict]:
        """
        Search using embedding vector.
        
        Args:
            embedding: Embedding vector
            k: Number of results
            
        Returns:
            List of search results
        """
        response = requests.post(
            f"{self.api_prefix}/search-embedding",
            json={"embedding": embedding, "k": k}
        )
        response.raise_for_status()
        data = response.json()
        return data['results']
    
    def query(self, question: str, k: int = 5, use_rerank: bool = True, initial_k: int = 20, rewrite: bool = True) -> Tuple[str, str]:
        """
        Answer a question using RAG.
        
        Args:
            question: Question to answer
            k: Number of context documents
            use_rerank: Whether to use re-ranking
            initial_k: Number of candidates before re-ranking
            rewrite: Whether to rewrite the question before retrieval
            
        Returns:
            Tuple of (prompt, answer)
        """
        response = requests.post(
            f"{self.api_prefix}/query",
            json={"question": question, "k": k, "use_rerank": use_rerank, "initial_k": initial_k, "rewrite": rewrite}
        )
        response.raise_for_status()
        data = response.json()
        return data['prompt'], data['answer']

    def rewrite(self, query: str, style: str = "concise") -> Dict:
        """Call the rewrite endpoint to get a rewritten query."""
        response = requests.post(
            f"{self.api_prefix}/rewrite",
            json={"query": query, "style": style}
        )
        response.raise_for_status()
        return response.json()
        
    def get_document(self, doc_id: int) -> Dict:
        """Get document by ID."""
        response = requests.get(f"{self.api_prefix}/documents/{doc_id}")
        response.raise_for_status()
        return response.json()
    
    def list_documents(self, page: int = 1, per_page: int = 10) -> Dict:
        """List all documents."""
        response = requests.get(
            f"{self.api_prefix}/documents",
            params={"page": page, "per_page": per_page}
        )
        response.raise_for_status()
        return response.json()






def main():
    # Initialize client
    client = RAG_API_Client(host="0.0.0.0", port=5001)
    
    # Health check
    print("Checking API health...")
    health = client.health_check()
    print(f"Health: {health}")
    
    # Initialize system if needed
    print("\nInitializing system...")
    try:
        init_result = client.initialize()
        print(f"Init result: {init_result}")
    except Exception as e:
        print(f"System already initialized or error: {e}")
        
    # Set question
    question = "What are the Seller's formatting options for Form 65?"
        
    # Search example
    print("\n" + "="*80)
    print("SEARCH EXAMPLE")
    print("="*80)
    
    print(f"\nSearching for: {question}")
    
    try:
        results = client.search(question, k=3)
        for result in results:
            print(f"\n - {result['document_id']}: {result['title']} (score: {result['similarity_score']:.4f})")
            print(f"Content: {result['content_preview']}...")
    except Exception as e:
        print(f"Error: {e}")
    
    # Query example
    print("\n" + "="*80)
    print("QUERY EXAMPLE")
    print("="*80)
    
    print(f"\nQuestion: \n{question}")
    
    try:
        prompt, answer = client.query(question, k=3)
        print("----------------------------------------------------------------------")
        print(f"\nPrompt: \n{prompt}")
        print("----------------------------------------------------------------------")
        print(f"\nAnswer: \n{answer}")
    except Exception as e:
        print(f"Error: {e}")
    
    


if __name__ == "__main__":
    main()
