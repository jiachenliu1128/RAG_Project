import requests
import json
from typing import List, Dict, Tuple, Optional


class RAG_API_Client:
    """Client for interacting with RAG API server."""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        """
        Initialize the RAG API client.
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url
        self.api_prefix = f"{base_url}/api"
    
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
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of search results
        """
        response = requests.post(
            f"{self.api_prefix}/search",
            json={"query": query, "k": k}
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
    
    def query(self, question: str, k: int = 5) -> Tuple[str, str]:
        """
        Answer a question using RAG.
        
        Args:
            question: Question to answer
            k: Number of context documents
            
        Returns:
            Tuple of (prompt, answer)
        """
        response = requests.post(
            f"{self.api_prefix}/query",
            json={"question": question, "k": k}
        )
        response.raise_for_status()
        data = response.json()
        return data['prompt'], data['answer']
        
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
    client = RAG_API_Client()
    
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
    
    # Query example
    print("\n" + "="*80)
    print("QUERY EXAMPLE")
    print("="*80)
    
    question = "What is the required use of Form 65?"
    print(f"\nQ: {question}")
    
    try:
        prompt, answer = client.query(question, k=3)
        print(f"Prompt: {prompt}")
        print(f"Answer: {answer}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Search example
    print("\n" + "="*80)
    print("SEARCH EXAMPLE")
    print("="*80)
    
    search_query = "Form 65 formatting"
    print(f"\nSearching for: {search_query}")
    
    try:
        results = client.search(search_query, k=3)
        for result in results:
            print(f"  - {result['document_id']}: {result['title']} (score: {result['similarity_score']:.4f})")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
