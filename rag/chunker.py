"""Split OCR text into overlapping chunks for retrieval."""

from __future__ import annotations

import re


def clean_text(raw_text: str) -> str:
    """Normalize whitespace and remove OCR page markers."""
    text = re.sub(r"\n{3,}", "\n\n", raw_text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"--- Page \d+ ---", "", text, flags=re.IGNORECASE)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping word-level chunks.

    chunk_size and overlap are in words. overlap must be smaller than chunk_size.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap.")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks: list[str] = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size]).strip()
        if len(chunk) > 20:
            chunks.append(chunk)

    print(f"[Chunker] Created {len(chunks)} chunks from {len(words)} words")
    return chunks
