"""Cross-encoder reranking for retrieved chunks."""

from __future__ import annotations

from sentence_transformers import CrossEncoder

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    """Load cross-encoder once (singleton)."""
    global _reranker
    if _reranker is None:
        print(f"[Reranker] Loading {RERANKER_MODEL}...")
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank(query: str, chunks: list[str], top_n: int = 3) -> list[str]:
    """
    Score each (query, chunk) pair and return the top_n chunks by relevance.

    chunks should be the candidate set from bi-encoder retrieval (e.g. top_k from FAISS).
    """
    if not chunks:
        return []
    if top_n <= 0:
        return []
    if not query.strip():
        return chunks[: min(top_n, len(chunks))]

    reranker = get_reranker()
    pairs = [[query, c] for c in chunks]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    top_chunks = [c for _, c in ranked[:top_n]]
    print(f"[Reranker] Reranked {len(chunks)} candidates -> top {len(top_chunks)}")
    return top_chunks
