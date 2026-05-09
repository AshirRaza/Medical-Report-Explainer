from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from jiwer import cer, wer

from ocr.extract import load_qwen2vl, ocr_image


def _load_paddleocr_if_available():
    try:
        from paddleocr import PaddleOCR

        return PaddleOCR
    except Exception:
        return None


def run_ocr_evaluation(
    num_samples: int = 50,
    compare_paddle: bool = False,
    output_path: str = "ocr_evaluation_results.json",
) -> dict[str, Any]:
    """
    Evaluate OCR on MedOCR Vision with CER/WER.

    - Primary path: Qwen API OCR
    - Optional baseline: PaddleOCR (best-effort; skipped if unavailable)
    """
    print("[OCR Eval] Loading MedOCR Vision dataset...")
    dataset = load_dataset("naazimsnh02/medocr-vision-dataset", split="train")
    samples = dataset.select(range(min(num_samples, len(dataset))))

    qwen_client, qwen_processor = load_qwen2vl()
    qwen_cers: list[float] = []
    qwen_wers: list[float] = []

    paddle_cers: list[float] = []
    paddle_wers: list[float] = []
    paddle_ocr = None

    if compare_paddle:
        PaddleOCR = _load_paddleocr_if_available()
        if PaddleOCR is None:
            print("[OCR Eval] PaddleOCR is not installed. Continuing without baseline.")
            compare_paddle = False
        else:
            paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

    for i, sample in enumerate(samples, start=1):
        image = sample["image"]
        ground_truth = str(sample.get("text", "")).strip()

        if not ground_truth:
            continue

        print(f"[OCR Eval] Sample {i}/{len(samples)}")

        qwen_output = ocr_image(image, qwen_client, qwen_processor)
        qwen_cers.append(cer(ground_truth, qwen_output))
        qwen_wers.append(wer(ground_truth, qwen_output))

        if compare_paddle and paddle_ocr is not None:
            img_array = np.array(image)
            paddle_result = paddle_ocr.ocr(img_array, cls=True)
            paddle_text = (
                " ".join(line[1][0] for line in paddle_result[0])
                if paddle_result and paddle_result[0]
                else ""
            )
            paddle_cers.append(cer(ground_truth, paddle_text))
            paddle_wers.append(wer(ground_truth, paddle_text))

    results: dict[str, Any] = {
        "dataset": "naazimsnh02/medocr-vision-dataset",
        "samples_requested": num_samples,
        "samples_scored": len(qwen_cers),
        "qwen_api": {
            "mean_CER": round(float(np.mean(qwen_cers)), 4) if qwen_cers else None,
            "mean_WER": round(float(np.mean(qwen_wers)), 4) if qwen_wers else None,
        },
    }

    if compare_paddle and paddle_cers and paddle_wers:
        paddle_mean_cer = float(np.mean(paddle_cers))
        qwen_mean_cer = float(np.mean(qwen_cers))
        results["paddleocr"] = {
            "mean_CER": round(paddle_mean_cer, 4),
            "mean_WER": round(float(np.mean(paddle_wers)), 4),
        }
        results["improvement"] = {
            "CER_reduction_percent": round(
                ((paddle_mean_cer - qwen_mean_cer) / paddle_mean_cer) * 100, 2
            )
            if paddle_mean_cer > 0
            else None
        }

    Path(output_path).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[OCR Eval] Saved results to {output_path}")
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    run_ocr_evaluation()
