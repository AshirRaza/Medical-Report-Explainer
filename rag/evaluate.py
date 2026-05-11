"""PubMedQA RAG benchmark: three retrieval settings, label accuracy, optional RAGAS."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
from anthropic import Anthropic
from datasets import load_dataset
from dotenv import load_dotenv

from llm.generator import generate_pubmedqa_answer
from rag.chunker import chunk_text, clean_text
from rag.embedder import EMBEDDING_MODEL
from rag.reranker import rerank
from rag.store import build_index, search_index

load_dotenv()

Mode = Literal["rag_rerank", "rag_no_rerank", "no_rag"]
MODES: tuple[Mode, ...] = ("rag_rerank", "rag_no_rerank", "no_rag")


def _document_from_row(row: dict[str, Any]) -> str:
    ctx = row.get("context") or {}
    parts = ctx.get("contexts") or []
    return clean_text("\n\n".join(str(p) for p in parts if p))


def _extract_label(text: str) -> str | None:
    lines = [ln.strip().lower() for ln in (text or "").strip().splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln in ("yes", "no", "maybe"):
            return ln
    m = re.search(r"\b(yes|no|maybe)\b", (text or "").lower())
    return m.group(1) if m else None


def _retrieve_chunks(
    mode: Mode,
    question: str,
    doc: str,
    top_k: int,
    top_n: int,
    max_no_rag_chars: int,
    chunk_size: int = 200,
    chunk_overlap: int = 50,
) -> list[str]:
    text = doc.strip()
    if not text:
        return []

    if mode == "no_rag":
        body = text[:max_no_rag_chars]
        return [body] if body.strip() else []

    chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
    if not chunks:
        chunks = [text]

    index, kept = build_index(chunks)
    candidates = search_index(question, index, kept, top_k=top_k)
    if not candidates:
        return []
    if mode == "rag_no_rerank":
        return candidates[: min(top_n, len(candidates))]
    return rerank(question, candidates, top_n=top_n)


def _make_ragas_llm() -> Any:
    from ragas.llms import llm_factory

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is required for RAGAS metrics. Set it in .env or use --skip-ragas."
        )
    model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6").strip()
    client = Anthropic(api_key=api_key)
    # Higher ceiling for RAGAS structured outputs (faithfulness NLI can emit long JSON).
    llm = llm_factory(
        model,
        provider="anthropic",
        client=client,
        max_tokens=4096,
    )
    # Ragas defaults include both temperature and top_p; current Anthropic models reject
    # that combination (400: use only one).
    model_args = getattr(llm, "model_args", None)
    if isinstance(model_args, dict):
        model_args.pop("top_p", None)
    return llm


def _run_ragas_on_rows(rows: list[dict[str, Any]], llm: Any) -> dict[str, Any]:
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.metrics._context_recall import context_recall
    from ragas.metrics._faithfulness import faithfulness
    from ragas.run_config import RunConfig

    samples: list[SingleTurnSample] = []
    for r in rows:
        ctxs = r.get("retrieved_contexts") or []
        if not ctxs:
            continue
        samples.append(
            SingleTurnSample(
                user_input=r["user_input"],
                retrieved_contexts=ctxs,
                response=r["response"],
                reference=r["reference"],
            )
        )
    if not samples:
        return {"skipped": True, "reason": "no_rows_with_context"}

    ed = EvaluationDataset(samples=samples)
    run_config = RunConfig(timeout=180)
    result = evaluate(
        ed,
        metrics=[faithfulness, context_recall],
        llm=llm,
        run_config=run_config,
        raise_exceptions=False,
        show_progress=True,
    )
    out: dict[str, Any] = {"skipped": False, "n_rows": len(samples)}
    for k, v in result._repr_dict.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            out[k] = None
        else:
            out[k] = round(float(v), 4) if isinstance(v, (float, np.floating)) else v
    return out


class _JsonEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        if isinstance(o, (np.floating, np.integer)):
            return float(o) if isinstance(o, np.floating) else int(o)
        return super().default(o)


def run_rag_evaluation(
    num_samples: int = 25,
    seed: int = 42,
    split: str = "train",
    top_k: int = 10,
    top_n: int = 3,
    max_no_rag_chars: int = 12000,
    chunk_size: int = 200,
    chunk_overlap: int = 50,
    skip_ragas: bool = False,
    ragas_on_mode: Mode = "rag_rerank",
    output_path: str = "Results/rag_pubmedqa_evaluation.json",
) -> dict[str, Any]:
    """
    Evaluate dense retrieval (+ optional rerank) on PubMedQA ``pqa_labeled``.

    Runs three modes: ``rag_rerank``, ``rag_no_rerank`` (top FAISS hits, no cross-encoder),
    ``no_rag`` (single truncated document chunk). Computes yes/no/maybe accuracy vs
    ``final_decision`` using ``generate_pubmedqa_answer``. Optionally runs legacy RAGAS
    ``faithfulness`` and ``context_recall`` on ``ragas_on_mode`` only.
    """
    if not os.getenv("ANTHROPIC_API_KEY", "").strip():
        raise RuntimeError(
            "ANTHROPIC_API_KEY is required for answer generation. Set it in .env."
        )

    print("[RAG Eval] Loading pubmed_qa / pqa_labeled...")
    ds = load_dataset("pubmed_qa", "pqa_labeled", split=split)
    n = min(int(num_samples), len(ds))
    ds = ds.shuffle(seed=seed).select(range(n))

    summary: dict[str, Any] = {
        "dataset": "pubmed_qa",
        "config": "pqa_labeled",
        "split": split,
        "seed": seed,
        "num_samples": n,
        "top_k": top_k,
        "top_n": top_n,
        "max_no_rag_chars": max_no_rag_chars,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": EMBEDDING_MODEL,
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "ragas_on_mode": ragas_on_mode,
        "modes": {},
    }

    ragas_rows: list[dict[str, Any]] = []

    for mode in MODES:
        correct = 0
        scored = 0
        print(f"[RAG Eval] Mode={mode}")
        for i, row in enumerate(ds, start=1):
            doc = _document_from_row(row)
            question = str(row.get("question", "")).strip()
            gold = str(row.get("final_decision", "")).strip().lower()
            if gold not in ("yes", "no", "maybe"):
                continue

            chunks = _retrieve_chunks(
                mode, question, doc, top_k, top_n, max_no_rag_chars,
                chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            )
            answer = generate_pubmedqa_answer(question, chunks)
            pred = _extract_label(answer)
            scored += 1
            if pred == gold:
                correct += 1

            if mode == ragas_on_mode and chunks:
                ragas_rows.append(
                    {
                        "user_input": question,
                        "retrieved_contexts": chunks,
                        "response": answer,
                        "reference": str(row.get("long_answer", "") or ""),
                    }
                )

            if i % 5 == 0 or i == n:
                print(f"  [{mode}] {i}/{n} processed")

        acc = (correct / scored) if scored else None
        summary["modes"][mode] = {
            "accuracy": round(float(acc), 4) if acc is not None else None,
            "n_scored": scored,
            "n_correct": correct,
        }

    summary["ragas"] = {"skipped": True}
    if skip_ragas:
        summary["ragas"]["reason"] = "--skip-ragas"
    else:
        try:
            llm = _make_ragas_llm()
            summary["ragas"] = _run_ragas_on_rows(ragas_rows, llm)
            summary["ragas"]["on_mode"] = ragas_on_mode
        except Exception as exc:  # noqa: BLE001
            summary["ragas"] = {"skipped": True, "error": str(exc)}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        json.dumps(summary, indent=2, cls=_JsonEncoder),
        encoding="utf-8",
    )
    print(f"[RAG Eval] Saved results to {output_path}")
    print(json.dumps(summary, indent=2, cls=_JsonEncoder))
    return summary


def _parse_ragas_mode(s: str) -> Mode:
    s = (s or "").strip().lower().replace("-", "_")
    if s in MODES:
        return s  # type: ignore[return-value]
    raise argparse.ArgumentTypeError(f"ragas mode must be one of {MODES}, got {s!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PubMedQA RAG evaluation: rerank vs no-rerank vs no-RAG + optional RAGAS."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=25,
        help="Number of PubMedQA examples (after shuffle).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split name (default train).",
    )
    parser.add_argument("--top-k", type=int, default=10, help="FAISS top-k.")
    parser.add_argument("--top-n", type=int, default=3, help="Chunks passed to the LLM.")
    parser.add_argument("--chunk-size", type=int, default=200, help="Words per chunk.")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Word overlap between chunks.")
    parser.add_argument(
        "--max-no-rag-chars",
        type=int,
        default=12000,
        help="Max characters for no_rag single-context mode.",
    )
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Skip RAGAS faithfulness/context_recall (saves time and API calls).",
    )
    parser.add_argument(
        "--ragas-on-mode",
        type=_parse_ragas_mode,
        default="rag_rerank",
        help="Which retrieval mode the RAGAS rows correspond to (default rag_rerank).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="Results/rag_pubmedqa_evaluation.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    run_rag_evaluation(
        num_samples=args.num_samples,
        seed=args.seed,
        split=args.split,
        top_k=args.top_k,
        top_n=args.top_n,
        max_no_rag_chars=args.max_no_rag_chars,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        skip_ragas=args.skip_ragas,
        ragas_on_mode=args.ragas_on_mode,
        output_path=args.output,
    )
