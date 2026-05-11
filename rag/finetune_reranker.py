"""Fine-tune the cross-encoder reranker on medical report retrieval pairs.

Uses sentence-transformers CrossEncoder training with binary cross-entropy loss.
Runs on CPU (~30-60 min for 500 pairs, 3 epochs) or GPU if available.

Usage:
    python -m rag.finetune_reranker --training-data rag/training_data.json
    python -m rag.finetune_reranker --training-data rag/training_data.json --epochs 5
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import (
    CEBinaryClassificationEvaluator,
)
from torch.utils.data import DataLoader

BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OUTPUT_DIR = "models/finetuned-reranker"


def load_training_data(path: str) -> list[dict]:
    """Load labeled pairs from JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    print(f"[FineTune] Loaded {len(data)} labeled pairs from {path}")
    return data


def split_train_eval(
    data: list[dict], eval_fraction: float = 0.15
) -> tuple[list[dict], list[dict]]:
    """Split data into train and eval sets."""
    n_eval = max(1, int(len(data) * eval_fraction))
    return data[n_eval:], data[:n_eval]


def to_input_examples(pairs: list[dict]) -> list[InputExample]:
    """Convert labeled pairs to sentence-transformers InputExamples."""
    examples = []
    for p in pairs:
        examples.append(
            InputExample(
                texts=[p["query"], p["chunk"]],
                label=float(p["label"]),
            )
        )
    return examples


def train(
    training_data_path: str,
    output_dir: str = OUTPUT_DIR,
    base_model: str = BASE_MODEL,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    warmup_fraction: float = 0.1,
    eval_fraction: float = 0.15,
) -> str:
    """
    Fine-tune the cross-encoder and save to output_dir.

    Returns the path to the saved model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[FineTune] Device: {device}")
    print(f"[FineTune] Base model: {base_model}")

    # Load and split data
    all_data = load_training_data(training_data_path)
    train_data, eval_data = split_train_eval(all_data, eval_fraction)
    print(f"[FineTune] Train: {len(train_data)}, Eval: {len(eval_data)}")

    # Convert to InputExamples
    train_examples = to_input_examples(train_data)
    eval_examples = to_input_examples(eval_data)

    # Build DataLoader
    train_dataloader = DataLoader(
        train_examples, shuffle=True, batch_size=batch_size
    )

    # Load model
    model = CrossEncoder(base_model, num_labels=1, device=device)

    # Build evaluator
    eval_pairs = [[e.texts[0], e.texts[1]] for e in eval_examples]
    eval_labels = [int(e.label) for e in eval_examples]

    evaluator = CEBinaryClassificationEvaluator(
        eval_pairs, eval_labels, name="medical-reranker"
    )

    # Training parameters
    total_steps = len(train_dataloader) * epochs
    warmup_steps = math.ceil(total_steps * warmup_fraction)

    print(f"[FineTune] Epochs: {epochs}")
    print(f"[FineTune] Batch size: {batch_size}")
    print(f"[FineTune] Total steps: {total_steps}")
    print(f"[FineTune] Warmup steps: {warmup_steps}")
    print(f"[FineTune] Learning rate: {learning_rate}")
    print(f"[FineTune] Training...")

    # Train
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": learning_rate},
        output_path=output_dir,
        save_best_model=True,
        show_progress_bar=True,
    )

    # Explicitly save the model (in case Trainer API didn't persist weights)
    model.save(output_dir)
    print(f"[FineTune] Model saved to {output_dir}")

    # Final evaluation
    print("[FineTune] Final evaluation on held-out set:")
    eval_results = evaluator(model, output_path=output_dir)
    if isinstance(eval_results, dict):
        for key, val in eval_results.items():
            print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")
    else:
        print(f"  Score: {eval_results:.4f}")

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune cross-encoder reranker on medical retrieval data."
    )
    parser.add_argument(
        "--training-data",
        type=str,
        default="rag/training_data.json",
        help="Path to training data JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help="Directory to save the fine-tuned model.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=BASE_MODEL,
        help="Base cross-encoder model to fine-tune.",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    args = parser.parse_args()

    train(
        training_data_path=args.training_data,
        output_dir=args.output_dir,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
