import numpy as np
import pandas as pd
import openai
import tiktoken
import json
from transformers import GPT2TokenizerFast
import dotenv
from faiss_backend import FAISS_Backend

# =============================================================================
# Configuration
# =============================================================================

# Load environment variables from .env file and initialize OpenAI client
dotenv.load_dotenv()
client = openai.OpenAI()

# Model for text generation
COMPLETIONS_MODEL = "gpt-4o-mini-2024-07-18"

# Model for generating embeddings
EMBEDDING_MODEL = "text-embedding-3-small"

# Parameters for completions
COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 300,
    "model": COMPLETIONS_MODEL
}

# Context retrieval parameters
MAX_SECTION_LEN = 1000
SEPARATOR = "\n* "







# =============================================================================
# Data Preprocessing
# =============================================================================

def load_data(csv_path: str) -> pd.DataFrame:
    """Load and preprocess the dataset."""
    df = pd.read_csv(csv_path, header=0)
    print(f"{len(df)} rows in the data.")
    return df


def create_context_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a new column called 'context' that combines multiple fields into a single text string.
    
    Combines title, headings, and content fields with labels and newlines.
    """
    for index, row in df.iterrows():
        df.at[index, 'context'] = f"Title: {str(row['title']).strip()}\n" + \
            f"Heading1: {str(row['heading1']).strip()}\n" + \
            f"Heading2: {str(row['heading2']).strip()}\n" + \
            f"Heading3: {str(row['heading3']).strip()}\n" + \
            f"Heading4: {str(row['heading4']).strip()}\n" + \
            f"Content: {str(row['content']).strip()}"
    return df


def num_tokens(text: str, model: str = "gpt-4") -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


# def generate_questions_for_context(context: str) -> str:
#     """Generate questions based on the provided context text."""
#     try:
#         response = client.chat.completions.create(
#             model=COMPLETIONS_MODEL,
#             messages=[{"role": "user", "content": f"Write questions based on the text below\n\nText: {context}\n\nQuestions:\n1."}],
#             temperature=0,
#             max_tokens=257,
#             top_p=1,
#             frequency_penalty=0,
#             presence_penalty=0,
#             stop=["\n\n"]
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         print(f"Error generating questions: {e}")
#         return ""


# def generate_answers_for_context(row) -> str:
#     """Generate answers based on the context and questions."""
#     try:
#         response = client.chat.completions.create(
#             model=COMPLETIONS_MODEL,
#             messages=[{"role": "user", "content": f"Write answer based on the text below\n\nText: {row.context}\n\nQuestions:\n{row.questions}\n\nAnswers:\n1."}],
#             temperature=0,
#             max_tokens=257,
#             top_p=1,
#             frequency_penalty=0,
#             presence_penalty=0
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         print(f"Error generating answers: {e}")
#         return ""


def preprocess_dataset(df: pd.DataFrame, output_csv: str) -> pd.DataFrame:
    """
    Preprocess the dataset by creating context, questions, and answers.
    
    Args:
        df: Input DataFrame
        output_csv: Path to save the preprocessed data
        
    Returns:
        Preprocessed DataFrame
    """
    # Create context column
    df = create_context_column(df)
    
    # # Generate questions
    # print("Generating questions...")
    # df['questions'] = df.context.apply(generate_questions_for_context)
    # df['questions'] = "1. " + df.questions
    
    # # Generate answers
    # print("Generating answers...")
    # df['answers'] = df.apply(generate_answers_for_context, axis=1)
    # df['answers'] = "1. " + df.answers
    # df = df.dropna().reset_index().drop('index', axis=1)
    
    # Count tokens
    df['tokens'] = df.apply(lambda row: num_tokens(row['context']), axis=1)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Preprocessed data saved to {output_csv}")
    
    return df








# =============================================================================
# Functions for Document Embeddings
# =============================================================================

def get_embedding(text: str, model: str = EMBEDDING_MODEL):
    """Convert text into an embedding vector using the OpenAI Embeddings API."""
    resp = client.embeddings.create(model=model, input=text)
    return resp.data[0].embedding


def compute_doc_embeddings(df: pd.DataFrame):
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Args:
        df: DataFrame with a 'context' column containing text to embed
        
    Returns:
        Dictionary mapping row index -> embedding vector
    """
    embeddings = {}
    for idx, row in df.iterrows():
        embeddings[idx] = get_embedding(str(row.get('context', '')))
    return embeddings


def save_embeddings(embeddings: dict, filepath: str):
    """Save embeddings to JSON file."""
    with open(filepath, 'w') as fp:
        json.dump(embeddings, fp)
    print(f"Embeddings saved to {filepath}")
    
    
def jsonKeys2int(x):
    """Convert JSON object keys from strings to integers."""
    if isinstance(x, dict):
        return {int(k): v for k, v in x.items()}
    return x


def load_embeddings(filepath: str) -> dict:
    """Load embeddings from JSON file."""
    with open(filepath, 'r') as fp:
        embeddings = json.load(fp, object_hook=jsonKeys2int)
    print(f"Embeddings loaded from {filepath}")
    return embeddings







# =============================================================================
# Retrieval and Similarity
# =============================================================================

def vector_similarity(x, y):
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity
    is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))


def order_document_sections_by_query_similarity(
    query: str, 
    contexts: dict = None, 
    faiss_backend: FAISS_Backend = None
) -> list:
    """
    Find the query embedding for the supplied query, and compare it against all
    of the pre-calculated document embeddings to find the most relevant sections.
    
    Args:
        query: User query string
        contexts: Dictionary of document embeddings
        faiss_backend: FAISSBackend instance with indexed embeddings (optional)
        
    Returns:
        List of (similarity_score, doc_index) tuples sorted by relevance (descending)
    """
    query_embedding = get_embedding(query)
    
    if faiss_backend is not None:
        document_similarities = faiss_backend.search(query_embedding)
        document_similarities = [(score, doc_id) for doc_id, score in document_similarities]
    else:
        document_similarities = sorted([
            (vector_similarity(query_embedding, doc_embedding), doc_index)
            for doc_index, doc_embedding in contexts.items()
        ], reverse=True)
    
    return document_similarities








# =============================================================================
# Prompt Construction
# =============================================================================

def construct_prompt(
    question: str, 
    df: pd.DataFrame, 
    context_embeddings: dict = None, 
    faiss_backend: FAISS_Backend = None
) -> str:
    """
    Construct a prompt by retrieving relevant document sections and prepending them
    to the query.
    
    Args:
        question: User's question
        context_embeddings: Dictionary of document embeddings
        df: DataFrame with document content and token counts
        faiss_backend: FAISSBackend instance with indexed embeddings (optional)
        
    Returns:
        Constructed prompt with context and question
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(
        question, context_embeddings, faiss_backend
    )
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
    
    # Get tokenizer for separator
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    separator_len = len(tokenizer.tokenize(SEPARATOR))
    
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.
        document_section = df.loc[section_index]
        
        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
        
        chosen_sections.append(
            SEPARATOR + document_section.content.replace("\n", " ")
        )
        chosen_sections_indexes.append(str(section_index))
    
    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_indexes))
    
    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
    
    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"






# =============================================================================
# ChatGPT Interaction
# =============================================================================

def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embedding: dict = None,
    faiss_backend: FAISS_Backend = None,
    show_prompt: bool = False
) -> str:
    """
    Answer a user's query using retrieved context and GPT.
    
    Args:
        query: User's question
        df: DataFrame with document content
        document_embedding: Dictionary of document embeddings
        faiss_backend: FAISS_Backend instance with indexed embeddings
        show_prompt: If True, print the constructed prompt
        
    Returns:
        Answer string generated by GPT
    """
    if document_embedding is None and faiss_backend is None:
        raise ValueError("Either document_embedding or faiss_backend must be provided.")
    
    prompt = construct_prompt(query, df, document_embedding, faiss_backend)
    
    if show_prompt:
        print(prompt)
    
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        **COMPLETIONS_API_PARAMS
    )
    
    return (response.choices[0].message.content.strip(" \n"), prompt)








# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution function."""
    
    # Paths
    data_dir = "./data"
    csv_path = f"{data_dir}/Series4000.csv"
    processed_csv_path = f"{data_dir}/Series4000_processed.csv"
    embeddings_path = f"{data_dir}/embeddings.json"
    
    # Load data
    print("Loading data...")
    df = load_data(csv_path)
    
    # Preprocess dataset (create context, generate Q&A)
    print("\nPreprocessing dataset...")
    df = preprocess_dataset(df, processed_csv_path)
    
    # Compute and save embeddings
    print("\nComputing embeddings...")
    document_embeddings = compute_doc_embeddings(df)
    save_embeddings(document_embeddings, embeddings_path)
    
    # Display example embedding
    example_entry = list(document_embeddings.items())[0]
    print(f"\nExample embedding: {example_entry[0]} : {example_entry[1][:5]}... ({len(example_entry[1])} entries)")
    
    # Example queries
    print("\n" + "="*80)
    print("EXAMPLE QUERIES AND ANSWERS")
    print("="*80)
    
    test_queries = [
        "What is the required use of Form 65?",
        "What are the Seller's formatting options for Form 65?",
        "What are the translation aids for Form 65?",
        "The Seller has formatting options for Form 65, which must be in accordance with the UMDP Rendering Options."
    ]
    
    for query in test_queries:
        print(f"\nQ: {query}")
        answer, prompt = answer_query_with_context(query, df, document_embeddings)
        print(f"A: {answer}")
        print("-" * 80)

if __name__ == "__main__":
    main()
