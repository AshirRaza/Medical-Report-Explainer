"""RAG package: chunking, embeddings, FAISS retrieval, and reranking."""

from rag.chunker import chunk_text, clean_text
from rag.embedder import EMBEDDING_MODEL, embed_texts, get_embedder
from rag.reranker import RERANKER_MODEL, rerank, get_reranker
from rag.store import build_index, search_index

__all__ = [
    "clean_text",
    "chunk_text",
    "EMBEDDING_MODEL",
    "get_embedder",
    "embed_texts",
    "build_index",
    "search_index",
    "RERANKER_MODEL",
    "get_reranker",
    "rerank",
]
