"""End-to-end medical report OCR, retrieval, reranking, and Claude generation."""

from __future__ import annotations

import faiss

from llm.generator import generate_answer, generate_summary
from ocr.extract import extract_text_from_pdf, load_qwen2vl
from rag.chunker import chunk_text, clean_text
from rag.reranker import rerank
from rag.store import build_index, search_index


class MedicalReportPipeline:
    """
    Load OCR once, then per PDF: OCR -> clean/chunk -> embed/index -> summary.
    Per question: retrieve -> rerank -> grounded answer.
    """

    def __init__(
        self,
        quantize_ocr: bool = True,
        top_k: int = 5,
        top_n: int = 3,
        ocr_dpi: int = 300,
    ) -> None:
        del quantize_ocr  # kept for API compatibility with local-Qwen plans
        print("[Pipeline] Loading OCR client...")
        self.qwen_model, self.qwen_processor = load_qwen2vl()
        self.top_k = top_k
        self.top_n = top_n
        self.ocr_dpi = ocr_dpi
        self.index: faiss.Index | None = None
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
        if not clean.strip():
            self.chunks = []
            self.index = None
            self.summary = generate_summary("")
            print("[Pipeline] OCR produced no text; index not built.")
            return self.summary

        self.chunks = chunk_text(clean)
        if not self.chunks:
            self.chunks = [clean]

        self.index, self.chunks = build_index(self.chunks)
        self.summary = generate_summary(clean)
        return self.summary

    def answer(self, query: str) -> tuple[str, list[str]]:
        """
        Retrieve, rerank, and answer using Claude. Returns (answer_text, chunks_used).
        """
        if self.index is None or self.chunks is None:
            raise RuntimeError("No PDF loaded. Call process_pdf() first.")

        candidates = search_index(query, self.index, self.chunks, top_k=self.top_k)
        top_chunks = rerank(query, candidates, top_n=self.top_n) if candidates else []
        answer = generate_answer(query, top_chunks)
        return answer, top_chunks
