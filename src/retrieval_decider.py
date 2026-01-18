import os
import pickle
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from logging_config import get_logger
from .faiss_backend import FAISS_Backend
from .rag_system import get_embedding

logger = get_logger(__name__)


def _features_from_similarities(sims: List[float], top_k: int = 5) -> np.ndarray:
    """Compute a compact feature vector from top similarity scores.

    Features:
    - max@k, mean@k, std@k
    - gap (s1 - s2) and sum@k
    - count_above_0_5, count_above_0_7 thresholds
    """
    if not sims:
        return np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

    s = np.array(sorted(sims, reverse=True)[: top_k], dtype=np.float32)
    max_s = float(np.max(s))
    mean_s = float(np.mean(s))
    std_s = float(np.std(s))
    gap = float(s[0] - s[1]) if len(s) > 1 else float(s[0])
    sum_s = float(np.sum(s))
    cnt_05 = float(np.sum(s >= 0.5))
    cnt_07 = float(np.sum(s >= 0.7))
    return np.array([max_s, mean_s, std_s, gap, sum_s, cnt_05, cnt_07], dtype=np.float32)


@dataclass
class RetrievalDeciderConfig:
    top_k: int = 10
    feature_top_k: int = 5


class RetrievalDecider:
    """Logistic-regression-based retrieval decision helper.

    Expects a FAISS index already built over your corpus using the same
    embedding function as your runtime system.
    """

    def __init__(self, model_path: str, config: Optional[RetrievalDeciderConfig] = None):
        self.model_path = model_path
        self.config = config or RetrievalDeciderConfig()
        self._clf = None
        self._scaler = None

    def load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Retrieval decider model not found: {self.model_path}")
        with open(self.model_path, "rb") as f:
            payload = pickle.load(f)
        # payload can be { 'clf': ..., 'scaler': ... }
        self._clf = payload.get("clf", None)
        self._scaler = payload.get("scaler", None)
        if self._clf is None:
            raise ValueError("Invalid decider model payload: missing 'clf'")
        logger.info(f"Loaded retrieval decider from {self.model_path}")

    def ready(self) -> bool:
        return self._clf is not None

    def features_for_query(self, query: str, faiss_backend: FAISS_Backend) -> np.ndarray:
        # Embed query with the system's embedding
        q_emb = get_embedding(query)
        # Retrieve top_k neighbors
        results = faiss_backend.search(q_emb, k=self.config.top_k)
        sims = [sim for _, sim in results]
        feats = _features_from_similarities(sims, top_k=self.config.feature_top_k)
        if self._scaler is not None:
            feats = self._scaler.transform([feats])[0].astype(np.float32)
        return feats

    def predict_should_retrieve(self, query: str, faiss_backend: FAISS_Backend) -> bool:
        feats = self.features_for_query(query, faiss_backend)
        prob = float(self._clf.predict_proba([feats])[0][1])  # class 1 = retrieve
        logger.debug(f"RetrievalDecider prob(retrieve)={prob:.3f}")
        return prob >= 0.5


__all__ = ["RetrievalDecider", "RetrievalDeciderConfig"]
