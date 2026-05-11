"""End-to-end medical report OCR, retrieval, reranking, and Claude generation."""

from __future__ import annotations

import faiss
from rank_bm25 import BM25Okapi

from llm.generator import generate_answer, generate_summary
from ocr.extract import extract_text_from_pdf, load_qwen2vl
from rag.chunker import chunk_text_structured, clean_report_noise, clean_text
from rag.reranker import rerank
from rag.store import build_bm25_index, build_index, hybrid_search


class MedicalReportPipeline:
    """
    Load OCR once, then per PDF: OCR -> clean/chunk -> embed/index -> summary.
    Per question: hybrid retrieve -> rerank -> grounded answer.
    """

    def __init__(
        self,
        quantize_ocr: bool = True,
        top_k: int = 20,
        top_n: int = 5,
        ocr_dpi: int = 300,
        hybrid_alpha: float = 0.5,
    ) -> None:
        del quantize_ocr  # kept for API compatibility with local-Qwen plans
        print("[Pipeline] Loading OCR client...")
        self.qwen_model, self.qwen_processor = load_qwen2vl()
        self.top_k = top_k
        self.top_n = top_n
        self.ocr_dpi = ocr_dpi
        self.hybrid_alpha = hybrid_alpha
        self.index: faiss.Index | None = None
        self.bm25: BM25Okapi | None = None
        self.chunks: list[str] | None = None
        self.raw_text: str | None = None
        self.summary: str | None = None
        print("[Pipeline] Ready.")

    def process_pdf(self, pdf_path: str) -> str:
        """
        OCR the PDF, build the vector index, and return a patient-friendly summary.
        Call once per new PDF upload.
        """
        print(f"[Pipeline] Processing: {pdf_path}")
        self.raw_text = extract_text_from_pdf(
            pdf_path,
            self.qwen_model,
            self.qwen_processor,
            dpi=self.ocr_dpi,
        )

        clean = clean_text(self.raw_text)
        clean = clean_report_noise(clean)
        if not clean.strip():
            self.chunks = []
            self.index = None
            self.bm25 = None
            self.summary = generate_summary("")
            print("[Pipeline] OCR produced no text; index not built.")
            return self.summary

        self.chunks = chunk_text_structured(clean)
        if not self.chunks:
            self.chunks = [clean]

        self.index, self.chunks = build_index(self.chunks)
        self.bm25 = build_bm25_index(self.chunks)
        self.summary = generate_summary(clean)
        return self.summary

    def answer(self, query: str) -> tuple[str, list[str]]:
        """
        Hybrid retrieve, rerank, and answer using Claude. Returns (answer_text, chunks_used).
        """
        if self.index is None or self.chunks is None:
            raise RuntimeError("No PDF loaded. Call process_pdf() first.")

        candidates = hybrid_search(
            query, self.index, self.bm25, self.chunks,
            top_k=self.top_k, alpha=self.hybrid_alpha,
        )
        top_chunks = rerank(query, candidates, top_n=self.top_n) if candidates else []
        answer = generate_answer(query, top_chunks)
        return answer, top_chunks
