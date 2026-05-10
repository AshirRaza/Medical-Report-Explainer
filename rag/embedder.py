"""Sentence-Transformer embeddings for dense retrieval (FAISS inner product)."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

# PubMed-oriented bi-encoder; good for medical terms in context
EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"

_embedder: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    """Load embedding model once (singleton)."""
    global _embedder
    if _embedder is None:
        print(f"[Embedder] Loading {EMBEDDING_MODEL}...")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed strings to L2-normalized float32 vectors (cosine similarity = dot product with IndexFlatIP).
    """
    if not texts:
        raise ValueError("Cannot embed an empty list of texts.")

    model = get_embedder()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (embeddings / norms).astype(np.float32)
