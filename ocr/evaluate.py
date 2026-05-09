from __future__ import annotations

import argparse
import json
import os
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


def _extract_text_from_paddle_result(paddle_output: Any) -> str:
    """
    Normalize PaddleOCR outputs across versions into plain text.
    """
    if paddle_output is None:
        return ""

    # Newer PaddleOCR often returns a list of result objects.
    if isinstance(paddle_output, list):
        texts: list[str] = []

        # Legacy shape: [ [ [box], (text, score) ], ... ]
        if paddle_output and isinstance(paddle_output[0], list):
            for row in paddle_output[0]:
                try:
                    texts.append(str(row[1][0]))
                except Exception:
                    continue
            return " ".join(texts).strip()

        # New shape can contain dict-like items or objects with rec_texts/texts.
        for item in paddle_output:
            if isinstance(item, dict):
                rec_texts = item.get("rec_texts") or item.get("texts") or []
                texts.extend([str(t) for t in rec_texts if t])
            else:
                rec_texts = getattr(item, "rec_texts", None) or getattr(item, "texts", None)
                if rec_texts:
                    texts.extend([str(t) for t in rec_texts if t])
        return " ".join(texts).strip()

    return ""


def run_ocr_evaluation(
    num_samples: int = 50,
    compare_paddle: bool = True,
    require_paddle: bool = False,
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
        # Workaround for common Windows CPU runtime issues in Paddle 3.x.
        os.environ["FLAGS_use_mkldnn"] = "0"
        os.environ["FLAGS_enable_pir_api"] = "0"
        os.environ["FLAGS_enable_pir_in_executor"] = "0"
        os.environ["FLAGS_set_to_1d"] = "0"

        PaddleOCR = _load_paddleocr_if_available()
        if PaddleOCR is None:
            message = "[OCR Eval] PaddleOCR is not installed."
            if require_paddle:
                raise RuntimeError(
                    f"{message} Install paddleocr and paddlepaddle, then retry."
                )
            print(f"{message} Continuing without baseline.")
            compare_paddle = False
        else:
            # PaddleOCR 3.x removed show_log and deprecated use_angle_cls.
            paddle_ocr = PaddleOCR(
                use_textline_orientation=True,
                lang="en",
                device="cpu",
                enable_mkldnn=False,
                cpu_threads=4,
            )

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
            try:
                paddle_result = paddle_ocr.predict(img_array)
            except NotImplementedError as exc:
                if "ConvertPirAttribute2RuntimeAttribute" in str(exc):
                    raise RuntimeError(
                        "PaddleOCR failed due to a Windows CPU Paddle runtime issue. "
                        "Try running with FLAGS_use_mkldnn=0 and FLAGS_enable_pir_api=0, "
                        "or install a compatible Paddle/PaddleOCR version pair."
                    ) from exc
                raise
            paddle_text = _extract_text_from_paddle_result(paddle_result)
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
    parser = argparse.ArgumentParser(description="Run OCR CER/WER evaluation on MedOCR.")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of MedOCR samples to evaluate.",
    )
    parser.add_argument(
        "--no-paddle",
        action="store_true",
        help="Disable PaddleOCR baseline comparison.",
    )
    parser.add_argument(
        "--require-paddle",
        action="store_true",
        help="Fail if PaddleOCR is unavailable.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ocr_evaluation_results.json",
        help="Path to save evaluation JSON.",
    )
    args = parser.parse_args()

    run_ocr_evaluation(
        num_samples=args.num_samples,
        compare_paddle=not args.no_paddle,
        require_paddle=args.require_paddle,
        output_path=args.output,
    )
