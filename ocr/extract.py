from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image

load_dotenv()


@dataclass
class QwenOCRClient:
    api_key: str
    base_url: str
    model: str
    http_referer: str
    app_title: str
    timeout_seconds: int = 90
    max_retries: int = 3

    def _endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def transcribe_image(self, image_base64: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Extract all text from this medical lab report exactly as it appears. "
                                "Preserve all numbers, units, and reference ranges with full precision. "
                                "Do not summarise or interpret - transcribe only."
                            ),
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 2048,
        }

        body = json.dumps(payload).encode("utf-8")
        endpoint = self._endpoint()

        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                endpoint,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": self.http_referer,
                    "X-Title": self.app_title,
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    raw = response.read().decode("utf-8")
                    parsed = json.loads(raw)
                    return parsed["choices"][0]["message"]["content"].strip()
            except urllib.error.HTTPError as exc:
                last_err = exc
                should_retry = exc.code in (408, 409, 429, 500, 502, 503, 504)
                if not should_retry or attempt == self.max_retries:
                    try:
                        detail = exc.read().decode("utf-8")
                    except Exception:
                        detail = str(exc)
                    raise RuntimeError(
                        f"Qwen OCR API request failed (status {exc.code}): {detail}"
                    ) from exc
                time.sleep(1.5 * (2**attempt))
            except Exception as exc:  # network/timeouts/json decode
                last_err = exc
                if attempt == self.max_retries:
                    break
                time.sleep(1.5 * (2**attempt))

        raise RuntimeError(f"Qwen OCR API failed after retries: {last_err}")


def pdf_to_images(pdf_path: str, dpi: int = 200) -> list[Image.Image]:
    """
    Convert each PDF page into PIL images.
    Set POPPLER_PATH in .env on Windows when poppler is not on PATH.
    """
    poppler_path = os.getenv("POPPLER_PATH", "").strip() or None
    pages = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
    print(f"[OCR] Converted {len(pages)} page(s) from {pdf_path}")
    return pages


def load_qwen2vl(quantize: bool = True) -> tuple[QwenOCRClient, dict[str, str]]:
    """
    Returns a Qwen API OCR client.

    Name kept for interface compatibility with the rest of the planned pipeline.
    """
    del quantize
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
    model = os.getenv("QWEN_VL_MODEL", "qwen/qwen3-vl-32b-instruct").strip()
    http_referer = os.getenv("OPENROUTER_HTTP_REFERER", "https://localhost").strip()
    app_title = os.getenv("OPENROUTER_APP_TITLE", "Medical-Report-Explainer").strip()

    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing. Set it in your .env file.")
    if not base_url:
        raise RuntimeError("OPENROUTER_BASE_URL is missing. Set it in your .env file.")

    client = QwenOCRClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        http_referer=http_referer,
        app_title=app_title,
    )
    processor = {"provider": "qwen_api", "model": model}
    print(f"[OCR] Qwen via OpenRouter loaded for model: {model}")
    return client, processor


def _pil_to_base64_png(pil_image: Image.Image) -> str:
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def ocr_image(pil_image: Image.Image, model: QwenOCRClient, processor: dict[str, str]) -> str:
    """
    OCR a single page image through Qwen API.
    """
    if processor.get("provider") != "qwen_api":
        raise ValueError("Unsupported OCR provider; expected qwen_api.")
    return model.transcribe_image(_pil_to_base64_png(pil_image))


def extract_text_from_pdf(pdf_path: str, model: QwenOCRClient, processor: dict[str, str]) -> str:
    """
    Full OCR flow: PDF -> page images -> OCR per page -> merged text.
    """
    images = pdf_to_images(pdf_path)
    full_text = []
    total = len(images)
    for i, image in enumerate(images, start=1):
        print(f"[OCR] Processing page {i}/{total}...")
        page_text = ocr_image(image, model, processor)
        full_text.append(f"--- Page {i} ---\n{page_text}")
    return "\n\n".join(full_text).strip()
