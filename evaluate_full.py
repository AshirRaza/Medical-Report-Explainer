"""Run MedOCR OCR evaluation and/or PubMedQA RAG evaluation; save a combined JSON report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def _compute_deltas(merged: dict[str, Any]) -> dict[str, Any]:
    """Derive improvement deltas across evaluation modes for the final report."""
    deltas: dict[str, Any] = {}

    rag = merged.get("rag")
    if rag and isinstance(rag.get("modes"), dict):
        modes = rag["modes"]
        rr = modes.get("rag_rerank", {}).get("accuracy")
        nr = modes.get("rag_no_rerank", {}).get("accuracy")
        base = modes.get("no_rag", {}).get("accuracy")

        rag_deltas: dict[str, Any] = {}
        if rr is not None and base is not None:
            rag_deltas["rag_rerank_vs_no_rag"] = round(rr - base, 4)
        if nr is not None and base is not None:
            rag_deltas["rag_no_rerank_vs_no_rag"] = round(nr - base, 4)
        if rr is not None and nr is not None:
            rag_deltas["rerank_vs_no_rerank"] = round(rr - nr, 4)

        ragas = rag.get("ragas") or {}
        if not ragas.get("skipped"):
            faith = ragas.get("faithfulness")
            recall = ragas.get("context_recall")
            if faith is not None:
                rag_deltas["faithfulness"] = round(faith, 4)
            if recall is not None:
                rag_deltas["context_recall"] = round(recall, 4)

        deltas["rag"] = rag_deltas

    ocr = merged.get("ocr")
    if ocr and isinstance(ocr, dict):
        ocr_deltas: dict[str, Any] = {}
        qwen_cer = ocr.get("qwen_mean_cer")
        paddle_cer = ocr.get("paddle_mean_cer")
        if qwen_cer is not None and paddle_cer is not None and paddle_cer > 0:
            ocr_deltas["cer_reduction_vs_paddle"] = round(
                (paddle_cer - qwen_cer) / paddle_cer * 100, 2
            )
        qwen_wer = ocr.get("qwen_mean_wer")
        paddle_wer = ocr.get("paddle_mean_wer")
        if qwen_wer is not None and paddle_wer is not None and paddle_wer > 0:
            ocr_deltas["wer_reduction_vs_paddle"] = round(
                (paddle_wer - qwen_wer) / paddle_wer * 100, 2
            )
        deltas["ocr"] = ocr_deltas

    return deltas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combined runner: optional MedOCR OCR + optional PubMedQA RAG benchmark."
    )
    parser.add_argument(
        "--ocr-num-samples",
        type=int,
        default=0,
        help="If >0, run MedOCR OCR evaluation with this many samples (default 0: skip).",
    )
    parser.add_argument(
        "--rag-num-samples",
        type=int,
        default=10,
        help="If >0, run PubMedQA RAG evaluation (default 10). Set 0 to skip.",
    )
    parser.add_argument(
        "--ocr-no-paddle",
        action="store_true",
        help="When OCR runs: disable PaddleOCR baseline.",
    )
    parser.add_argument(
        "--ocr-require-paddle",
        action="store_true",
        help="When OCR runs: fail if PaddleOCR is unavailable.",
    )
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="When RAG runs: skip RAGAS faithfulness/context_recall.",
    )
    parser.add_argument(
        "--ocr-output",
        type=str,
        default=None,
        help="OCR JSON path when OCR runs (default under Results/).",
    )
    parser.add_argument(
        "--rag-output",
        type=str,
        default=None,
        help="RAG JSON path when RAG runs (default under Results/).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Combined summary JSON (default Results/full_evaluation_<UTC>.json).",
    )
    args = parser.parse_args()

    if args.ocr_num_samples <= 0 and args.rag_num_samples <= 0:
        parser.error("At least one of --ocr-num-samples or --rag-num-samples must be > 0.")

    merged: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ocr": None,
        "rag": None,
    }

    Path("Results").mkdir(parents=True, exist_ok=True)

    if args.ocr_num_samples > 0:
        from ocr.evaluate import run_ocr_evaluation

        ocr_path = args.ocr_output or str(
            Path("Results") / "ocr_evaluation_full_run.json"
        )
        merged["ocr"] = run_ocr_evaluation(
            num_samples=args.ocr_num_samples,
            compare_paddle=not args.ocr_no_paddle,
            require_paddle=args.ocr_require_paddle,
            output_path=ocr_path,
        )

    if args.rag_num_samples > 0:
        from rag.evaluate import run_rag_evaluation

        rag_path = args.rag_output or str(
            Path("Results") / "rag_pubmedqa_full_run.json"
        )
        merged["rag"] = run_rag_evaluation(
            num_samples=args.rag_num_samples,
            skip_ragas=args.skip_ragas,
            output_path=rag_path,
        )

    merged["improvement_deltas"] = _compute_deltas(merged)

    out = args.output or str(
        Path("Results")
        / f"full_evaluation_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    Path(out).write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"[Full Eval] Wrote combined report to {out}")
    print(json.dumps(merged, indent=2))


if __name__ == "__main__":
    main()
