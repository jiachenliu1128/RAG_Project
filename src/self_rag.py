"""
Self-RAG reflection utilities for the RAG system.

Provides LLM-based reflection methods (grounding check, utility check, etc.)
to enhance answer quality and reduce hallucinations.
"""

import openai
import os
from typing import Tuple, Optional
from logging_config import get_logger
from .retrieval_decider import RetrievalDecider
from .faiss_backend import FAISS_Backend

logger = get_logger(__name__)

# Initialize OpenAI client (assumes dotenv is loaded)
client = openai.OpenAI()

DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"


def should_retrieve(
    query: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    faiss_backend: Optional[FAISS_Backend] = None,
    decider_model_path: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Decide if retrieval is needed for the query.
    
    Args:
        query: User query
        model: OpenAI model to use
        temperature: Sampling temperature
        faiss_backend: If provided with a trained decider, uses ML classifier
        decider_model_path: Optional path to a trained decider (.pkl)
        
    Returns:
        Tuple of (should_retrieve: bool, reasoning: str)
    """
    # 1) Prefer ML decider if available
    try:
        decider_path = decider_model_path or os.getenv("RETRIEVAL_DECIDER_MODEL_PATH")
        if decider_path and faiss_backend is not None:
            decider = RetrievalDecider(decider_path)
            decider.load()
            pred = decider.predict_should_retrieve(query, faiss_backend)
            reason = (
                "ML-Decider: RETRIEVE based on top-k similarity features"
                if pred
                else "ML-Decider: NO_RETRIEVE based on top-k similarity features"
            )
            logger.info(f"Retrieval decision (ML): {'RETRIEVE' if pred else 'NO_RETRIEVE'}")
            return pred, reason
    except Exception as e:
        logger.warning(f"ML decider unavailable or failed, falling back to LLM: {e}")

    # 2) Fall back to LLM decision
    prompt = f"""Analyze this question and decide if it requires retrieving external documents to answer accurately.

Question: {query}

Consider:
- Does it ask about specific facts, dates, procedures, or documentation?
- Can it be answered with general knowledge alone?
- Does it reference specific forms, policies, or technical details?

Respond with ONLY "RETRIEVE" or "NO_RETRIEVE" followed by a brief reason.
Format: [DECISION] Reason

Examples:
Question: What is 2+2?
NO_RETRIEVE Simple arithmetic that doesn't require external documents

Question: What are the requirements for Form 65?
RETRIEVE Asks about specific form requirements that need documentation

Your response:"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=150
        )
        
        result = response.choices[0].message.content.strip()
        should_retrieve = result.upper().startswith("RETRIEVE")
        
        logger.info(f"Retrieval decision: {'RETRIEVE' if should_retrieve else 'NO_RETRIEVE'} - {result}")
        return should_retrieve, result
        
    except Exception as e:
        logger.error(f"Error in should_retrieve: {e}", exc_info=True)
        # Default to retrieving on error
        return True, f"Error occurred, defaulting to retrieve: {str(e)}"


def check_groundedness(query: str, context: str, answer: str, model: str = DEFAULT_MODEL, temperature: float = 0.0) -> Tuple[bool, float, str]:
    """
    Check if the answer is grounded in (supported by) the provided context.
    
    Args:
        query: Original query
        context: Retrieved context
        answer: Generated answer
        model: OpenAI model to use
        temperature: Sampling temperature
        
    Returns:
        Tuple of (is_grounded: bool, confidence: float, reasoning: str)
    """
    prompt = f"""Evaluate if the answer is fully supported by the provided context.

Context:
{context[:1500]}

Question: {query}

Answer: {answer}

Determine:
1. Is every claim in the answer supported by the context?
2. Does the answer contain any information not in the context?
3. Is the answer faithful to the source material?

Rate groundedness:
- FULLY_GROUNDED: All claims supported by context
- MOSTLY_GROUNDED: Most claims supported, minor unsupported details
- PARTIALLY_GROUNDED: Some claims supported, some speculation
- NOT_GROUNDED: Answer contradicts or ignores context

Respond with: [RATING] (Confidence: 0.0-1.0) Explanation

Your evaluation:"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=300
        )
        
        result = response.choices[0].message.content.strip()
        logger.debug(f"Groundedness check: {result}")
        
        # Parse result
        is_grounded = "FULLY_GROUNDED" in result.upper() or "MOSTLY_GROUNDED" in result.upper()
        
        # Extract confidence if present
        confidence = 0.5
        if "FULLY_GROUNDED" in result.upper():
            confidence = 0.95
        elif "MOSTLY_GROUNDED" in result.upper():
            confidence = 0.75
        elif "PARTIALLY_GROUNDED" in result.upper():
            confidence = 0.4
        else:  # NOT_GROUNDED
            confidence = 0.1
        
        logger.info(f"Groundedness: {is_grounded} (confidence={confidence:.2f})")
        return is_grounded, confidence, result
        
    except Exception as e:
        logger.error(f"Error in check_groundedness: {e}", exc_info=True)
        return True, 0.5, f"Error occurred: {str(e)}"


def check_utility(query: str, answer: str, model: str = DEFAULT_MODEL, temperature: float = 0.0) -> Tuple[bool, float, str]:
    """
    Check if the answer is useful and addresses the question.
    
    Args:
        query: Original query
        answer: Generated answer
        model: OpenAI model to use
        temperature: Sampling temperature
        
    Returns:
        Tuple of (is_useful: bool, score: float, reasoning: str)
    """
    prompt = f"""Evaluate if the answer effectively addresses the question.

Question: {query}

Answer: {answer}

Consider:
- Does it directly answer what was asked?
- Is it complete and informative?
- Is it clear and understandable?
- Does it acknowledge limitations appropriately?

Rate utility:
- HIGHLY_USEFUL: Comprehensive, direct answer
- USEFUL: Good answer, addresses the question
- SOMEWHAT_USEFUL: Partial answer or vague
- NOT_USEFUL: Doesn't answer the question

Respond with: [RATING] (Score: 0.0-1.0) Explanation

Your evaluation:"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=200
        )
        
        result = response.choices[0].message.content.strip()
        logger.debug(f"Utility check: {result}")
        
        # Parse result
        is_useful = "USEFUL" in result.upper() and "NOT_USEFUL" not in result.upper()
        
        # Extract score
        score = 0.5
        if "HIGHLY_USEFUL" in result.upper():
            score = 0.95
        elif "SOMEWHAT_USEFUL" in result.upper():
            score = 0.4
        elif "NOT_USEFUL" in result.upper():
            score = 0.1
        else:  # USEFUL
            score = 0.75
        
        logger.info(f"Utility: {is_useful} (score={score:.2f})")
        return is_useful, score, result
        
    except Exception as e:
        logger.error(f"Error in check_utility: {e}", exc_info=True)
        return True, 0.5, f"Error occurred: {str(e)}"

