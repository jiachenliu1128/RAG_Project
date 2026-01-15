import os
import platform
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import json
from functools import wraps

from faiss_backend import FAISS_Backend
from reranker import Reranker
from rag_system import (
    load_data, preprocess_dataset, compute_doc_embeddings, 
    answer_query_with_context, get_embedding, save_embeddings, load_embeddings, rewrite_query
)
from logging_config import setup_logging, get_logger
logger = get_logger(__name__)


# Avoid macOS libomp duplicate runtime crash when faiss/numpy load OpenMP
if platform.system() == "Darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")





# =============================================================================
# Flask App Setup
# =============================================================================

# Initialize Flask app
app = Flask(__name__)
CORS(app)  
host = '0.0.0.0'
port = 5001

# Configuration
DATA_DIR = "./data"
DATA_FILE_STEM = "Series4000"
CSV_PATH = f"{DATA_DIR}/{DATA_FILE_STEM}.csv"
PROCESSED_CSV_PATH = f"{DATA_DIR}/{DATA_FILE_STEM}_processed.csv"
EMBEDDINGS_PATH = f"{DATA_DIR}/{DATA_FILE_STEM}_embeddings.json"
FAISS_INDEX_PATH = f"{DATA_DIR}/{DATA_FILE_STEM}_faiss_index.bin"
FAISS_METADATA_PATH = f"{DATA_DIR}/{DATA_FILE_STEM}_faiss_metadata.json"

# Global variables
faiss_backend = None
df = None
embeddings_dict = None
reranker = None










# =============================================================================
# Utility Decorator Functions
# =============================================================================

def require_init(f):
    """Decorator to check if system is initialized."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if faiss_backend is None or faiss_backend.index is None:
            return jsonify({'error': 'System not initialized. Please call /api/initialize first.'}), 400
        return f(*args, **kwargs)
    return decorated_function







# =============================================================================
# API Routes - System Management
# =============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    backend_status = 'uninitialized'
    if faiss_backend is not None and faiss_backend.index is not None:
        backend_status = 'ready'
    
    return jsonify({
        'status': 'healthy',
        'backend': backend_status,
        'faiss': faiss_backend.get_index_stats() if faiss_backend else None
    }), 200


@app.route('/api/initialize', methods=['POST'])
def initialize():
    """
    Initialize the RAG system with FAISS backend.
    Loads data, preprocesses it, and creates the FAISS index.
    """
    global faiss_backend, df, embeddings_dict, reranker
    
    try:
        # Initialize FAISS backend
        logger.info("Initializing FAISS backend...")
        faiss_backend = FAISS_Backend(embedding_dim=len(get_embedding("")))
        
        # Initialize reranker
        logger.info("Initializing reranker...")
        reranker = Reranker()
        
        # Load data
        logger.info("Loading data...")
        df = load_data(CSV_PATH)
        
        # Preprocess dataset
        logger.info("Preprocessing dataset...")
        df = preprocess_dataset(df, PROCESSED_CSV_PATH)
        
        # Load or compute embeddings
        if os.path.exists(EMBEDDINGS_PATH):
            logger.info("Loading embeddings...")
            embeddings_dict = load_embeddings(EMBEDDINGS_PATH)
        else:
            logger.info("Computing embeddings...")
            embeddings_dict = compute_doc_embeddings(df)
            save_embeddings(embeddings_dict, EMBEDDINGS_PATH)
            
        
        # Add embeddings to FAISS index
        logger.info("Building FAISS index...")
        total = faiss_backend.add_embeddings(embeddings_dict)
        
        # Save FAISS index
        logger.info("Saving FAISS index...")
        faiss_backend.save_index(FAISS_INDEX_PATH, FAISS_METADATA_PATH)
        
        stats = faiss_backend.get_index_stats()
        logger.info(f"System initialized successfully: {len(df)} rows, {total} embeddings")
        return jsonify({
            'status': 'success',
            'message': 'System initialized successfully',
            'data_rows': len(df),
            'embeddings': total,
            'index_stats': stats
        }), 200
        
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Initialization failed: {str(e)}'
        }), 500


@app.route('/api/load-index', methods=['POST'])
def load_index():
    """Load pre-built FAISS index from disk."""
    global faiss_backend, df, reranker
    
    try:
        if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(FAISS_METADATA_PATH):
            return jsonify({
                'status': 'error',
                'message': f'Index files not found. Please initialize first.'
            }), 404
        
        # Initialize and load FAISS backend
        faiss_backend = FAISS_Backend()
        faiss_backend.load_index(FAISS_INDEX_PATH, FAISS_METADATA_PATH)
        
        # Initialize reranker
        logger.info("Initializing reranker...")
        reranker = Reranker()
        
        # Load dataframe
        df = pd.read_csv(PROCESSED_CSV_PATH)
        
        stats = faiss_backend.get_index_stats()
        logger.info(f"Index loaded successfully: {stats['total_vectors']} vectors")
        return jsonify({
            'status': 'success',
            'message': 'Index loaded successfully',
            'index_stats': stats
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to load index: {str(e)}'
        }), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics."""
    stats = {
        'data_rows': len(df) if df is not None else 0,
        'faiss_backend': faiss_backend.get_index_stats() if faiss_backend else None
    }
    return jsonify(stats), 200








# =============================================================================
# API Routes - Embedding Operations
# =============================================================================

@app.route('/api/embed', methods=['POST'])
def embed_text():
    """
    Generate embedding for input text.
    
    Request body:
    {
        "text": "Your text here"
    }
    """
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" field in request'}), 400
        
        text = data['text']
        embedding = get_embedding(text)
        
        return jsonify({
            'status': 'success',
            'embedding': embedding,
            'dimension': len(embedding)
        }), 200
        
    except Exception as e:
        logger.error(f"Embedding failed: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Embedding failed: {str(e)}'
        }), 500
        
        
        
        
        
        
        
        
        


# =============================================================================
# API Routes - Search Operations
# =============================================================================

@app.route('/api/search', methods=['POST'])
@require_init
def search():
    """
    Search for similar documents using FAISS.
    
    Request body:
    {
        "query": "Your search query",
        "k": 5,  (optional, default 5)
        "use_rerank": true,  (optional, default true if reranker available)
        "initial_k": 20  (optional, default 20, used when reranking)
        "rewrite": true  (optional, default false)
    }
    """
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing "query" field in request'}), 400
        
        query = data['query']
        k = data.get('k', 5)
        use_rerank = data.get('use_rerank', reranker is not None)
        initial_k = data.get('initial_k', 20)
        rewrite = data.get('rewrite', False)
        
        logger.debug(f"Search request: query='{query[:50]}...', k={k}, use_rerank={use_rerank}, rewrite={rewrite}")

        # Optionally rewrite the query for retrieval
        effective_query = rewrite_query(query) if rewrite else query
        
        # Get query embedding
        query_embedding = get_embedding(effective_query)
        
        # Search using FAISS (retrieve more if re-ranking)
        search_k = initial_k if use_rerank and reranker is not None else k
        results = faiss_backend.search(query_embedding, search_k)
        
        # Apply re-ranking if requested
        if use_rerank and reranker is not None:
            candidates = []
            for doc_id, similarity in results:
                if doc_id < len(df):
                    candidates.append({
                        'doc_id': doc_id,
                        'initial_score': similarity,
                        'content': str(df.loc[doc_id].get('content', ''))
                    })
            
            # Re-rank
            reranked = reranker.rerank_with_metadata(
                query=query,
                candidates=candidates,
                text_field='content',
                top_k=k
            )
            
            # Convert back to results format
            results = [(r['doc_id'], r['rerank_score']) for r in reranked]
        
        # Build response with document context
        search_results = []
        for doc_id, similarity in results:
            if doc_id < len(df):
                doc = df.loc[doc_id]
                search_results.append({
                    'document_id': int(doc_id),
                    'similarity_score': float(similarity),
                    'title': str(doc.get('title', 'N/A')),
                    'content_preview': str(doc.get('content', ''))[:200]
                })
        
        logger.info(f"Search completed: {len(search_results)} results returned")
        return jsonify({
            'status': 'success',
            'query': query,
            'results_count': len(search_results),
            'results': search_results
        }), 200
        
    except Exception as e:
        logger.error(f"Search failed: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Search failed: {str(e)}'
        }), 500


@app.route('/api/search-embedding', methods=['POST'])
@require_init
def search_embedding():
    """
    Search using a pre-computed embedding vector.
    
    Request body:
    {
        "embedding": [0.1, 0.2, ...],
        "k": 5  (optional)
    }
    """
    try:
        data = request.get_json()
        if not data or 'embedding' not in data:
            return jsonify({'error': 'Missing "embedding" field in request'}), 400
        
        embedding = data['embedding']
        k = data.get('k', 5)
        
        # Validate embedding dimension
        if len(embedding) != faiss_backend.embedding_dim:
            return jsonify({
                'error': f'Invalid embedding dimension. Expected {faiss_backend.embedding_dim}, got {len(embedding)}'
            }), 400
        
        # Search using FAISS
        results = faiss_backend.search(embedding, k)
        
        # Build response
        search_results = []
        for doc_id, similarity in results:
            if doc_id < len(df):
                doc = df.loc[doc_id]
                search_results.append({
                    'document_id': int(doc_id),
                    'similarity_score': float(similarity),
                    'title': str(doc.get('title', 'N/A')),
                    'content_preview': str(doc.get('content', ''))[:200]
                })
        
        return jsonify({
            'status': 'success',
            'results_count': len(search_results),
            'results': search_results
        }), 200
        
    except Exception as e:
        logger.error(f"Search with embedding failed: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Search failed: {str(e)}'
        }), 500












# =============================================================================
# API Routes - Query RAG Operations
# =============================================================================

@app.route('/api/query', methods=['POST'])
@require_init
def query_rag():
    """
    Answer a question using RAG with FAISS backend.
    
    Request body:
    {
        "question": "Your question here",
        "k": 5,  (optional, number of context documents to use)
        "use_rerank": true,  (optional, default true if reranker available)
        "initial_k": 20  (optional, number of candidates before reranking)
        "rewrite": true  (optional, default false)
    }
    """
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({'error': 'Missing "question" field in request'}), 400
        
        question = data['question']
        k = data.get('k', 5)
        use_rerank = data.get('use_rerank', reranker is not None) 
        initial_k = data.get('initial_k', 20)
        rewrite = data.get('rewrite', False) 
        
        logger.debug(f"Query request: question='{question[:50]}...', k={k}, use_rerank={use_rerank}, rewrite={rewrite}")
        
        # Answer query using RAG
        answer, prompt = answer_query_with_context(
            question, 
            df, 
            faiss_backend=faiss_backend,
            k=k,
            reranker=reranker if use_rerank else None,
            initial_k=initial_k,
            rewrite=rewrite
        )
        
        logger.info(f"Query completed successfully for question: '{question[:50]}...'")
        return jsonify({
            'status': 'success',
            'question': question,
            'prompt': prompt,
            'answer': answer
        }), 200
        
    except Exception as e:
        logger.error(f"Query failed: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Query failed: {str(e)}'
        }), 500
        
        
        
        
        
        
        
 # =============================================================================
# API Routes - Query Rewrite
# =============================================================================

@app.route('/api/rewrite', methods=['POST'])
def api_rewrite():
    """Rewrite a query to a retrieval-optimized form.
    
    Request body:
    {
        "query": "Your query here",
        "style": "concise" or "expanded" (optional, default "concise")
    }
    """
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing "query" field in request'}), 400
        
        # Rewrite the query
        style = data.get('style', 'concise')
        rewritten = rewrite_query(data['query'], style=style)
        
        # Return the rewritten query
        return jsonify({
            'status': 'success',
            'original': data['query'],
            'rewritten': rewritten
        }), 200
        
    except Exception as e:
        logger.error(f"Rewrite endpoint failed: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500











# =============================================================================
# API Routes - Document Operations
# =============================================================================

@app.route('/api/documents/<int:doc_id>', methods=['GET'])
def get_document(doc_id):
    """Get full document content by ID.
    
    Args:
        doc_id: Document ID
    """
    if df is None or doc_id >= len(df) or doc_id < 0:
        return jsonify({'error': 'Document not found'}), 404
    
    try:
        doc = df.loc[doc_id]
        return jsonify({
            'status': 'success',
            'document_id': doc_id,
            'title': str(doc.get('title', 'N/A')),
            'content': str(doc.get('content', 'N/A')),
            'context': str(doc.get('context', 'N/A'))
        }), 200
    except Exception as e:
        return jsonify({'error': f'Failed to retrieve document: {str(e)}'}), 500


@app.route('/api/documents', methods=['GET'])
def list_documents():
    """List all documents with pagination.
    
    Request body:
    {
        "page": 1,  (optional, default 1)
        "per_page": 10  (optional, default 10)
    }
    """
    if df is None:
        return jsonify({'error': 'No documents loaded'}), 400
    
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        
        total = len(df)
        start = (page - 1) * per_page
        end = start + per_page
        
        documents = []
        for idx in range(start, min(end, total)):
            doc = df.loc[idx]
            documents.append({
                'id': idx,
                'title': str(doc.get('title', 'N/A')),
                'content_preview': str(doc.get('content', ''))[:200]
            })
        
        return jsonify({
            'status': 'success',
            'total': total,
            'page': page,
            'per_page': per_page,
            'documents': documents
        }), 200
    except Exception as e:
        return jsonify({'error': f'Failed to list documents: {str(e)}'}), 500











# =============================================================================
# Error Handlers
# =============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500












# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    # Setup logging
    setup_logging(log_file='logs/api_server.log')
    
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    logger.info("Starting RAG API Server...")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"API Server running on http://{host}:{port}")
    
    # Run Flask app
    app.run(debug=True, host=host, port=5001)
