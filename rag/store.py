"""FAISS index build and dense vector search."""

from __future__ import annotations

import faiss

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
        # FAISS may use -1 as a placeholder in some index types; guard anyway.
        if i < 0 or i >= len(chunks):
            continue
        out.append(chunks[i])
    print(f"[Store] Retrieved {len(out)} chunks for query (top_k={k})")
    return out
