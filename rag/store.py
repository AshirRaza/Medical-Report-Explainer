"""FAISS index build, BM25 sparse index, and hybrid search."""

from __future__ import annotations

import numpy as np
import faiss
from rank_bm25 import BM25Okapi

from rag.embedder import embed_texts


def build_index(chunks: list[str]) -> tuple[faiss.Index, list[str]]:
    """
    Embed chunks and build an exact inner-product index (cosine on L2-normalized vectors).

    Returns the index and the same chunk list (for mapping row ids back to text).
    """
    if not chunks:
        raise ValueError("No chunks provided to build_index.")

    print(f"[Store] Building FAISS index for {len(chunks)} chunks...")
    embeddings = embed_texts(chunks)
    dim = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"[Store] Index built: vectors={index.ntotal}, dim={dim}")
    return index, chunks


def build_bm25_index(chunks: list[str]) -> BM25Okapi:
    """Build a BM25 sparse index over the chunks."""
    tokenized = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    print(f"[Store] BM25 index built over {len(chunks)} chunks")
    return bm25


def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """Normalize scores to [0, 1] range."""
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-9:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)


def search_index(
    query: str,
    index: faiss.Index,
    chunks: list[str],
    top_k: int = 5,
) -> list[str]:
    """Embed the query and return the top_k most similar chunks by inner product."""
    if not query.strip():
        return []

    if top_k <= 0:
        return []

    n = index.ntotal
    if n == 0:
        return []

    k = min(top_k, n)
    query_vec = embed_texts([query])
    scores, indices = index.search(query_vec, k)

    out: list[str] = []
    for idx in indices[0]:
        i = int(idx)
        if i < 0 or i >= len(chunks):
            continue
        out.append(chunks[i])
    print(f"[Store] Retrieved {len(out)} chunks for query (top_k={k})")
    return out


def hybrid_search(
    query: str,
    index: faiss.Index,
    bm25: BM25Okapi,
    chunks: list[str],
    top_k: int = 15,
    alpha: float = 0.5,
) -> list[str]:
    """
    Combine dense (FAISS) and sparse (BM25) retrieval with score fusion.

    alpha controls the blend: 1.0 = pure dense, 0.0 = pure BM25.
    """
    if not query.strip() or top_k <= 0:
        return []

    n = index.ntotal
    if n == 0:
        return []

    # Dense scores for all chunks
    query_vec = embed_texts([query])
    all_scores, all_ids = index.search(query_vec, n)
    dense_scores = np.zeros(len(chunks), dtype=np.float32)
    for score, idx in zip(all_scores[0], all_ids[0]):
        i = int(idx)
        if 0 <= i < len(chunks):
            dense_scores[i] = score

    # BM25 scores for all chunks
    bm25_scores = np.array(
        bm25.get_scores(query.lower().split()), dtype=np.float32
    )

    # Normalize both to [0, 1] and combine
    d_norm = _min_max_normalize(dense_scores)
    b_norm = _min_max_normalize(bm25_scores)
    combined = alpha * d_norm + (1 - alpha) * b_norm

    # Return top_k by combined score
    k = min(top_k, len(chunks))
    top_ids = combined.argsort()[::-1][:k]

    out = [chunks[i] for i in top_ids if combined[i] > 0]
    print(
        f"[Store] Hybrid search retrieved {len(out)} chunks "
        f"(alpha={alpha}, top_k={k})"
    )
    return out
