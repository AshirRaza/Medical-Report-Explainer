"""Evaluate reranker quality before and after fine-tuning.

Measures Precision@k and Recall@k: given a query and candidate chunks,
does the reranker place the relevant chunks in the top-n?

Usage:
    python -m rag.evaluate_reranker --eval-data rag/training_data.json
    python -m rag.evaluate_reranker --eval-data rag/training_data.json --finetuned models/finetuned-reranker
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers.cross_encoder import CrossEncoder

BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_eval_data(path: str) -> dict[str, list[dict]]:
    """
    Load evaluation pairs and group by query.

    Returns: {query: [{"chunk": ..., "label": 0|1}, ...]}
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    grouped: dict[str, list[dict]] = defaultdict(list)
    for pair in data:
        grouped[pair["query"]].append(
            {"chunk": pair["chunk"], "label": pair["label"]}
        )

    # Only keep queries that have both positives and negatives
    filtered = {
        q: items
        for q, items in grouped.items()
        if any(i["label"] == 1 for i in items)
        and any(i["label"] == 0 for i in items)
    }
    return filtered


def evaluate_model(
    model: CrossEncoder, eval_data: dict[str, list[dict]], top_n: int = 3
) -> dict[str, float]:
    """
    Compute Precision@n and Recall@n for the reranker.

    For each query, score all candidate chunks, take top-n,
    and check how many are actually relevant.
    """
    precisions = []
    recalls = []
    aps = []  # Average Precision

    for query, items in eval_data.items():
        chunks = [it["chunk"] for it in items]
        labels = [it["label"] for it in items]

        # Score all (query, chunk) pairs
        pairs = [[query, chunk] for chunk in chunks]
        scores = model.predict(pairs)

        # Rank by score (descending)
        ranked_indices = np.argsort(scores)[::-1]
        ranked_labels = [labels[i] for i in ranked_indices]

        # Precision@n and Recall@n
        top_labels = ranked_labels[:top_n]
        n_relevant_in_top = sum(top_labels)
        n_total_relevant = sum(labels)

        precision = n_relevant_in_top / top_n if top_n > 0 else 0
        recall = (
            n_relevant_in_top / n_total_relevant if n_total_relevant > 0 else 0
        )
        precisions.append(precision)
        recalls.append(recall)

        # Average Precision
        hits = 0
        sum_precisions = 0.0
        for rank, label in enumerate(ranked_labels, 1):
            if label == 1:
                hits += 1
                sum_precisions += hits / rank
        ap = sum_precisions / n_total_relevant if n_total_relevant > 0 else 0
        aps.append(ap)

    return {
        f"precision@{top_n}": round(np.mean(precisions), 4),
        f"recall@{top_n}": round(np.mean(recalls), 4),
        "MAP": round(np.mean(aps), 4),
        "n_queries": len(eval_data),
    }


def compare_models(
    eval_data_path: str,
    base_model_name: str = BASE_MODEL,
    finetuned_path: str | None = None,
    top_n: int = 3,
) -> dict[str, dict]:
    """Run evaluation on base model and optionally the fine-tuned model."""
    eval_data = load_eval_data(eval_data_path)
    print(f"[EvalReranker] Loaded {len(eval_data)} queries for evaluation")

    results = {}

    # Evaluate base model
    print(f"[EvalReranker] Evaluating base model: {base_model_name}")
    base_model = CrossEncoder(base_model_name)
    base_metrics = evaluate_model(base_model, eval_data, top_n=top_n)
    results["base"] = {"model": base_model_name, **base_metrics}
    print(f"[EvalReranker] Base model: {base_metrics}")

    # Evaluate fine-tuned model if provided
    if finetuned_path and Path(finetuned_path).exists():
        print(f"[EvalReranker] Evaluating fine-tuned model: {finetuned_path}")
        ft_model = CrossEncoder(finetuned_path)
        ft_metrics = evaluate_model(ft_model, eval_data, top_n=top_n)
        results["finetuned"] = {"model": finetuned_path, **ft_metrics}
        print(f"[EvalReranker] Fine-tuned: {ft_metrics}")

        # Compute deltas
        delta = {
            f"precision@{top_n}_delta": round(
                ft_metrics[f"precision@{top_n}"] - base_metrics[f"precision@{top_n}"], 4
            ),
            f"recall@{top_n}_delta": round(
                ft_metrics[f"recall@{top_n}"] - base_metrics[f"recall@{top_n}"], 4
            ),
            "MAP_delta": round(ft_metrics["MAP"] - base_metrics["MAP"], 4),
        }
        results["improvement"] = delta
        print(f"[EvalReranker] Improvement: {delta}")

    # Save results
    output_path = "Results/reranker_eval.json"
    Path("Results").mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[EvalReranker] Results saved to {output_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate reranker quality (base vs fine-tuned)."
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        default="rag/training_data.json",
        help="Path to evaluation data JSON.",
    )
    parser.add_argument(
        "--finetuned",
        type=str,
        default=None,
        help="Path to fine-tuned model directory (optional).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Top-n for precision/recall computation.",
    )
    args = parser.parse_args()

    compare_models(
        eval_data_path=args.eval_data,
        finetuned_path=args.finetuned,
        top_n=args.top_n,
    )
