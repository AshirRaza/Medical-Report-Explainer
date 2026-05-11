"""RAG package: chunking, embeddings, FAISS retrieval, and reranking."""

from rag.chunker import chunk_text, chunk_text_structured, clean_report_noise, clean_text
from rag.embedder import EMBEDDING_MODEL, embed_texts, get_embedder
from rag.reranker import FINETUNED_MODEL_DIR, RERANKER_MODEL, rerank, get_reranker
from rag.store import build_bm25_index, build_index, hybrid_search, search_index

__all__ = [
    "clean_text",
    "clean_report_noise",
    "chunk_text",
    "chunk_text_structured",
    "EMBEDDING_MODEL",
    "get_embedder",
    "embed_texts",
    "build_index",
    "build_bm25_index",
    "search_index",
    "hybrid_search",
    "RERANKER_MODEL",
    "FINETUNED_MODEL_DIR",
    "get_reranker",
    "rerank",
]
