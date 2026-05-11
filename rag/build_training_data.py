"""Generate labeled (query, chunk, label) pairs for cross-encoder fine-tuning.

Sources:
1. PDF evaluation results — positive pairs from correct RAG answers,
   hard negatives from irrelevant retrieved chunks.
2. PubMedQA dataset — question + gold context = positive,
   question + random other context = negative.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset
from dotenv import load_dotenv

from rag.chunker import chunk_text, clean_text

load_dotenv()

random.seed(42)


def _load_pdf_eval_pairs(eval_path: str) -> list[dict[str, Any]]:
    """
    Extract training pairs from a PDF evaluation report.

    For each question:
    - If RAG answered correctly (answer doesn't contain "could not find"),
      the retrieved chunks are positives.
    - Chunks from OTHER questions serve as hard negatives.
    """
    data = json.loads(Path(eval_path).read_text(encoding="utf-8"))
    results = data.get("results", [])

    all_chunks: list[str] = []
    positives: list[tuple[str, str]] = []

    for entry in results:
        question = entry["question"]
        rag = entry.get("rag", {})
        answer = rag.get("answer", "")
        chunks = rag.get("chunks_text", [])

        all_chunks.extend(chunks)

        if "could not find" not in answer.lower() and chunks:
            for chunk in chunks:
                positives.append((question, chunk))

    pairs: list[dict[str, Any]] = []

    for query, pos_chunk in positives:
        pairs.append({"query": query, "chunk": pos_chunk, "label": 1})

        # Sample a hard negative from other chunks
        negatives = [c for c in all_chunks if c != pos_chunk]
        if negatives:
            neg_chunk = random.choice(negatives)
            pairs.append({"query": query, "chunk": neg_chunk, "label": 0})

    return pairs


def _load_pubmedqa_pairs(
    num_samples: int = 100, seed: int = 42
) -> list[dict[str, Any]]:
    """
    Extract training pairs from PubMedQA.

    Each sample has a question + context passages (positive).
    We pair each question with a random other sample's context as a negative.
    """
    ds = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    ds = ds.shuffle(seed=seed).select(range(min(num_samples, len(ds))))

    all_contexts: list[str] = []
    questions: list[str] = []

    for row in ds:
        ctx = row.get("context") or {}
        parts = ctx.get("contexts") or []
        doc = clean_text("\n\n".join(str(p) for p in parts if p))
        all_contexts.append(doc)
        questions.append(row["question"])

    pairs: list[dict[str, Any]] = []

    for i, (question, context) in enumerate(zip(questions, all_contexts)):
        # Chunk the context and use chunks as positives
        chunks = chunk_text(context, chunk_size=100, overlap=20)
        if not chunks:
            chunks = [context]

        # Positive: question paired with its own context chunks
        for chunk in chunks[:3]:
            pairs.append({"query": question, "chunk": chunk, "label": 1})

        # Negative: question paired with chunks from a different sample
        neg_idx = (i + random.randint(1, len(all_contexts) - 1)) % len(all_contexts)
        neg_chunks = chunk_text(all_contexts[neg_idx], chunk_size=100, overlap=20)
        if not neg_chunks:
            neg_chunks = [all_contexts[neg_idx]]
        for chunk in neg_chunks[:2]:
            pairs.append({"query": question, "chunk": chunk, "label": 0})

    return pairs


def build_training_data(
    pdf_eval_path: str | None = None,
    pubmedqa_samples: int = 100,
    output_path: str = "rag/training_data.json",
) -> list[dict[str, Any]]:
    """Combine all data sources into a single training set."""
    all_pairs: list[dict[str, Any]] = []

    # Source 1: PDF evaluation results
    if pdf_eval_path and Path(pdf_eval_path).exists():
        pdf_pairs = _load_pdf_eval_pairs(pdf_eval_path)
        print(f"[TrainData] PDF eval pairs: {len(pdf_pairs)}")
        all_pairs.extend(pdf_pairs)

    # Source 2: PubMedQA
    pubmed_pairs = _load_pubmedqa_pairs(num_samples=pubmedqa_samples)
    print(f"[TrainData] PubMedQA pairs: {len(pubmed_pairs)}")
    all_pairs.extend(pubmed_pairs)

    # Shuffle
    random.shuffle(all_pairs)

    # Summary
    n_pos = sum(1 for p in all_pairs if p["label"] == 1)
    n_neg = sum(1 for p in all_pairs if p["label"] == 0)
    print(f"[TrainData] Total: {len(all_pairs)} pairs ({n_pos} positive, {n_neg} negative)")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        json.dumps(all_pairs, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[TrainData] Saved to {output_path}")
    return all_pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build cross-encoder training data.")
    parser.add_argument(
        "--pdf-eval",
        type=str,
        default="Results/pdf_eval_report_v2.json",
        help="Path to PDF evaluation report JSON.",
    )
    parser.add_argument(
        "--pubmedqa-samples",
        type=int,
        default=100,
        help="Number of PubMedQA samples to use.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="rag/training_data.json",
        help="Output path for training data JSON.",
    )
    args = parser.parse_args()

    build_training_data(
        pdf_eval_path=args.pdf_eval,
        pubmedqa_samples=args.pubmedqa_samples,
        output_path=args.output,
    )
