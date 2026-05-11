"""Cross-encoder reranking for retrieved chunks."""

from __future__ import annotations

from pathlib import Path

from sentence_transformers import CrossEncoder

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
FINETUNED_MODEL_DIR = "models/finetuned-reranker"

_reranker: CrossEncoder | None = None


def get_reranker(prefer_finetuned: bool = True) -> CrossEncoder:
    """
    Load cross-encoder once (singleton).

    If prefer_finetuned=True and a fine-tuned model exists at FINETUNED_MODEL_DIR,
    loads that instead of the base model.
    """
    global _reranker
    if _reranker is None:
        model_path = RERANKER_MODEL
        if prefer_finetuned and Path(FINETUNED_MODEL_DIR).exists():
            config_file = Path(FINETUNED_MODEL_DIR) / "config.json"
            if config_file.exists():
                model_path = FINETUNED_MODEL_DIR
                print(f"[Reranker] Loading FINE-TUNED model from {model_path}")
            else:
                print(f"[Reranker] Fine-tuned dir exists but no config.json, using base model")
                print(f"[Reranker] Loading {RERANKER_MODEL}...")
        else:
            print(f"[Reranker] Loading {RERANKER_MODEL}...")
        _reranker = CrossEncoder(model_path)
    return _reranker


def rerank(
    query: str,
    chunks: list[str],
    top_n: int = 3,
    prefer_finetuned: bool = True,
) -> list[str]:
    """
    Score each (query, chunk) pair and return the top_n chunks by relevance.

    chunks should be the candidate set from bi-encoder retrieval (e.g. top_k from FAISS).
    If prefer_finetuned=True and a fine-tuned model exists, uses it automatically.
    """
    if not chunks:
        return []
    if top_n <= 0:
        return []
    if not query.strip():
        return chunks[: min(top_n, len(chunks))]

    reranker = get_reranker(prefer_finetuned=prefer_finetuned)
    pairs = [[query, c] for c in chunks]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    top_chunks = [c for _, c in ranked[:top_n]]
    print(f"[Reranker] Reranked {len(chunks)} candidates -> top {len(top_chunks)}")
    return top_chunks
