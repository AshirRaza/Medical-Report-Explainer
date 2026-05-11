"""
RAG vs no-RAG side-by-side evaluation on real PDF medical reports.

For each question the script runs two modes:
  1. **rag** — OCR -> chunk -> embed -> FAISS retrieve -> rerank -> Claude answer
  2. **no_rag** — OCR -> full text (truncated to max_context_chars) -> Claude answer

Outputs a JSON report with both answers, the chunks RAG selected, and timing data
so you can compare quality and see where retrieval adds value on long documents.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from llm.generator import generate_answer
from ocr.extract import extract_text_from_pdf, load_qwen2vl
from rag.chunker import chunk_text, chunk_text_structured, clean_report_noise, clean_text
from rag.reranker import rerank
from rag.store import build_bm25_index, build_index, hybrid_search, search_index


def _load_questions(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.loads(f.read())


def _answer_no_rag(query: str, full_text: str, max_chars: int) -> str:
    """Pass a single truncated context block to the LLM (no retrieval)."""
    truncated = full_text[:max_chars]
    return generate_answer(query, [truncated])


def run_pdf_evaluation(
    pdf_path: str,
    questions_path: str,
    top_k: int = 20,
    top_n: int = 5,
    chunk_size: int = 200,
    chunk_overlap: int = 50,
    max_no_rag_chars: int = 12000,
    ocr_dpi: int = 300,
    skip_ocr: bool = False,
    ocr_cache_path: str | None = None,
    output_path: str = "Results/pdf_eval_report.json",
    use_hybrid: bool = True,
    hybrid_alpha: float = 0.5,
    use_structured_chunking: bool = True,
) -> dict[str, Any]:

    Path("Results").mkdir(parents=True, exist_ok=True)
    questions = _load_questions(questions_path)
    print(f"[PDF Eval] Loaded {len(questions)} questions from {questions_path}")

    # --- OCR ---
    if skip_ocr and ocr_cache_path and Path(ocr_cache_path).exists():
        print(f"[PDF Eval] Loading cached OCR text from {ocr_cache_path}")
        raw_text = Path(ocr_cache_path).read_text(encoding="utf-8")
    else:
        print(f"[PDF Eval] Running OCR on {pdf_path} (dpi={ocr_dpi})...")
        t0 = time.time()
        model, processor = load_qwen2vl()
        raw_text = extract_text_from_pdf(pdf_path, model, processor, dpi=ocr_dpi)
        ocr_time = time.time() - t0
        print(f"[PDF Eval] OCR completed in {ocr_time:.1f}s ({len(raw_text)} chars)")

        cache_out = ocr_cache_path or str(
            Path("Results") / (Path(pdf_path).stem + "_ocr_text.txt")
        )
        Path(cache_out).write_text(raw_text, encoding="utf-8")
        print(f"[PDF Eval] OCR text cached to {cache_out}")

    cleaned = clean_text(raw_text)
    # Remove repeated boilerplate (signatures, headers, QR code lines)
    cleaned = clean_report_noise(cleaned)
    word_count = len(cleaned.split())
    print(f"[PDF Eval] Cleaned text: {len(cleaned)} chars, {word_count} words")

    # --- Build RAG index once ---
    if use_structured_chunking:
        chunks = chunk_text_structured(
            cleaned, chunk_size=chunk_size, overlap=chunk_overlap
        )
    else:
        chunks = chunk_text(cleaned, chunk_size=chunk_size, overlap=chunk_overlap)
    if not chunks:
        chunks = [cleaned]
    index, kept_chunks = build_index(chunks)
    print(f"[PDF Eval] FAISS index built with {len(kept_chunks)} chunks")

    # Build BM25 index for hybrid search
    bm25 = build_bm25_index(kept_chunks) if use_hybrid else None

    # --- Evaluate each question in both modes ---
    results: list[dict[str, Any]] = []

    for i, qobj in enumerate(questions, start=1):
        qid = qobj.get("id", f"q{i:02d}")
        question = qobj["question"]
        category = qobj.get("category", "")
        expected_page = qobj.get("expected_page")
        print(f"\n[PDF Eval] [{i}/{len(questions)}] {qid}: {question}")

        # RAG mode
        t0 = time.time()
        if use_hybrid and bm25 is not None:
            candidates = hybrid_search(
                question, index, bm25, kept_chunks,
                top_k=top_k, alpha=hybrid_alpha,
            )
        else:
            candidates = search_index(question, index, kept_chunks, top_k=top_k)
        top_chunks = rerank(question, candidates, top_n=top_n) if candidates else []
        rag_answer = generate_answer(question, top_chunks)
        rag_time = time.time() - t0

        # No-RAG mode
        t1 = time.time()
        no_rag_answer = _answer_no_rag(question, cleaned, max_no_rag_chars)
        no_rag_time = time.time() - t1

        entry: dict[str, Any] = {
            "id": qid,
            "question": question,
            "category": category,
            "expected_page": expected_page,
            "rag": {
                "answer": rag_answer,
                "chunks_used": len(top_chunks),
                "chunks_text": top_chunks,
                "time_s": round(rag_time, 2),
            },
            "no_rag": {
                "answer": no_rag_answer,
                "context_chars": min(len(cleaned), max_no_rag_chars),
                "time_s": round(no_rag_time, 2),
            },
        }
        results.append(entry)
        print(f"  RAG ({rag_time:.1f}s, {len(top_chunks)} chunks) | no-RAG ({no_rag_time:.1f}s)")

    # --- Build summary ---
    report: dict[str, Any] = {
        "pdf": pdf_path,
        "questions_file": questions_path,
        "num_questions": len(questions),
        "ocr_text_chars": len(cleaned),
        "ocr_text_words": word_count,
        "num_chunks": len(kept_chunks),
        "config": {
            "top_k": top_k,
            "top_n": top_n,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "max_no_rag_chars": max_no_rag_chars,
            "ocr_dpi": ocr_dpi,
            "use_hybrid": use_hybrid,
            "hybrid_alpha": hybrid_alpha,
            "use_structured_chunking": use_structured_chunking,
        },
        "no_rag_coverage": round(
            min(len(cleaned), max_no_rag_chars) / max(len(cleaned), 1) * 100, 1
        ),
        "results": results,
    }

    Path(output_path).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[PDF Eval] Report saved to {output_path}")
    print(f"[PDF Eval] no-RAG saw {report['no_rag_coverage']}% of the document")
    print(f"[PDF Eval] RAG used {len(kept_chunks)} chunks, top_k={top_k}, top_n={top_n}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAG vs no-RAG evaluation on a real PDF medical report."
    )
    parser.add_argument(
        "--pdf",
        type=str,
        required=True,
        help="Path to the PDF medical report.",
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="Sample Reports/questions.json",
        help="Path to questions JSON file.",
    )
    parser.add_argument("--top-k", type=int, default=20, help="Top-k retrieval candidates.")
    parser.add_argument("--top-n", type=int, default=5, help="Chunks after reranking.")
    parser.add_argument("--chunk-size", type=int, default=200, help="Words per chunk.")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Word overlap.")
    parser.add_argument(
        "--max-no-rag-chars",
        type=int,
        default=12000,
        help="Max chars for no-RAG truncated context.",
    )
    parser.add_argument("--ocr-dpi", type=int, default=300, help="OCR DPI.")
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Skip OCR and load from --ocr-cache instead.",
    )
    parser.add_argument(
        "--ocr-cache",
        type=str,
        default=None,
        help="Path to cached OCR text file (used with --skip-ocr).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="Results/pdf_eval_report.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--no-hybrid",
        action="store_true",
        help="Disable hybrid BM25+dense search (use dense only).",
    )
    parser.add_argument(
        "--hybrid-alpha",
        type=float,
        default=0.5,
        help="Hybrid search blend: 1.0=pure dense, 0.0=pure BM25 (default 0.5).",
    )
    parser.add_argument(
        "--no-structured-chunking",
        action="store_true",
        help="Disable structure-aware chunking (use flat word-level chunking).",
    )
    args = parser.parse_args()

    run_pdf_evaluation(
        pdf_path=args.pdf,
        questions_path=args.questions,
        top_k=args.top_k,
        top_n=args.top_n,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_no_rag_chars=args.max_no_rag_chars,
        ocr_dpi=args.ocr_dpi,
        skip_ocr=args.skip_ocr,
        ocr_cache_path=args.ocr_cache,
        output_path=args.output,
        use_hybrid=not args.no_hybrid,
        hybrid_alpha=args.hybrid_alpha,
        use_structured_chunking=not args.no_structured_chunking,
    )
