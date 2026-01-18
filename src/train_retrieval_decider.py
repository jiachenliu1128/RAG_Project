import os
import pickle
from typing import List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from logging_config import get_logger, setup_logging

# Optional: Use BEIR to build a bootstrap dataset
# pip install beir sentence-transformers faiss-cpu scikit-learn
try:
    from beir import util, LoggingHandler
    from beir.datasets.data_loader import GenericDataLoader
    from sentence_transformers import SentenceTransformer
    import faiss
except Exception:
    util = None


logger = get_logger(__name__)


def features_from_sims(sims: List[float], k_feat: int = 5) -> np.ndarray:
    s = np.array(sorted(sims, reverse=True)[: k_feat], dtype=np.float32)
    if s.size == 0:
        s = np.array([0.0], dtype=np.float32)
    max_s = float(np.max(s))
    mean_s = float(np.mean(s))
    std_s = float(np.std(s))
    gap = float(s[0] - s[1]) if s.size > 1 else float(s[0])
    sum_s = float(np.sum(s))
    cnt_05 = float(np.sum(s >= 0.5))
    cnt_07 = float(np.sum(s >= 0.7))
    return np.array([max_s, mean_s, std_s, gap, sum_s, cnt_05, cnt_07], dtype=np.float32)


def build_training_using_beir(dataset: str = "fiqa") -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a bootstrap training set using a BEIR dataset.

    Labeling scheme:
    - Positive (1 = retrieve): queries that have at least one relevant document (per qrels)
    - Negative (0 = no_retrieve): queries from a DIFFERENT dataset evaluated on this corpus (mismatched),
      which typically have no relevant docs here.

    NOTE: This is a pragmatic bootstrap and may not match your domain distribution.
    Prefer fine-tuning on your own queries when available.
    """
    if util is None:
        raise RuntimeError("Required packages not installed. Please install: beir sentence-transformers faiss-cpu scikit-learn")

    logger.info(f"Loading BEIR dataset: {dataset}")
    data_path = util.download_and_unzip("https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset), "./data")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    # Build corpus arrays
    doc_ids = list(corpus.keys())
    docs = [corpus[did]["text"] or "" for did in doc_ids]

    # Embed with a local model (keeps training offline)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb_docs = model.encode(docs, batch_size=128, show_progress_bar=True, convert_to_numpy=True).astype("float32")

    # Build FAISS index (L2) and store vectors
    d = emb_docs.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(emb_docs)

    # Helper to compute sims from query to top-k corpus docs
    def topk_sims(q: str, k: int = 10) -> List[float]:
        qe = model.encode([q], convert_to_numpy=True).astype("float32")
        D, I = index.search(qe, k)
        # Convert L2 distance to similarity ~ 1/(1+dist)
        sims = [float(1.0 / (1.0 + d)) for d in D[0]]
        return sims

    X, y = [], []

    # Positives: queries with relevant docs in this dataset
    for qid, qtext in queries.items():
        has_rel = qid in qrels and len(qrels[qid]) > 0
        if not has_rel:
            continue
        sims = topk_sims(qtext)
        X.append(features_from_sims(sims))
        y.append(1)

    # Negatives: pick queries from another dataset on this corpus (mismatch)
    neg_dataset = "scifact" if dataset != "scifact" else "fiqa"
    logger.info(f"Loading negative queries from another BEIR dataset: {neg_dataset}")
    neg_path = util.download_and_unzip("https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(neg_dataset), "./data")
    _, neg_queries, _ = GenericDataLoader(data_folder=neg_path).load(split="test")

    neg_count = len(y)  # balance
    for i, (_, qtext) in enumerate(neg_queries.items()):
        if i >= neg_count:
            break
        sims = topk_sims(qtext)
        X.append(features_from_sims(sims))
        y.append(0)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    logger.info(f"Training set built: X={X.shape}, positives={int(np.sum(y==1))}, negatives={int(np.sum(y==0))}")
    return X, y


def main():
    setup_logging()
    os.makedirs("./models", exist_ok=True)

    # Build data
    X, y = build_training_using_beir(dataset=os.getenv("BEIR_DATASET", "fiqa"))

    # Train/val split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale features
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Train logistic regression
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_tr_s, y_tr)

    # Eval
    y_pred = clf.predict(X_te_s)
    logger.info("\n" + classification_report(y_te, y_pred, digits=3))

    # Save
    out_path = os.getenv("OUT_PATH", "./models/retrieval_decider.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({"clf": clf, "scaler": scaler}, f)
    logger.info(f"Saved retrieval decider to {out_path}")


if __name__ == "__main__":
    main()
