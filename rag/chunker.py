"""Split OCR text into overlapping chunks for retrieval."""

from __future__ import annotations

import re


def clean_text(raw_text: str) -> str:
    """Normalize whitespace and remove OCR page markers."""
    text = re.sub(r"\n{3,}", "\n\n", raw_text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"--- Page \d+ ---", "", text, flags=re.IGNORECASE)
    return text.strip()


def clean_report_noise(text: str) -> str:
    """Remove repeated boilerplate from lab reports (headers, footers, signatures)."""
    # QR code / lab branding lines
    text = re.sub(
        r"Scan QR code to check report authenticity[\s\S]{0,30}?MC-2202",
        "",
        text,
    )
    text = re.sub(r"ACCURIS\s+Pathology lab that cares", "", text)
    text = re.sub(
        r"This is an Electronically Authenticated Report\.?", "", text
    )
    # Repeated doctor signature blocks
    text = re.sub(
        r"DR\.?\s*TEJASWINI DHOTE M\.?D\.?\s*Pathology\s*"
        r"Dr\.?\s*Sanjeev Shah M\.?D\.?\s*Path\s*"
        r"Dr\.?\s*Yash\s*Shah M\.?D\.?\s*Path",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"Dr\.?\s*Purvish Darji M\.?D\.?\s*\(?\s*Path\s*\)?\s*"
        r"Dr\.?\s*Sanjeev Shah M\.?D\.?\s*Path\s*"
        r"Dr\.?\s*\.?Yash\s*Shah M\.?D\.?\s*Path",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # "# Referred Test Page X of Y" markers
    text = re.sub(r"#\s*Referred Test\s*Page \d+ of \d+", "", text)
    # Standalone "Passport No :" without a value
    text = re.sub(r"Passport No\s*:\s*(?=\s)", "", text)

    # Remove duplicate patient info blocks (keep only the first occurrence)
    patient_block = (
        r"(LABORATORY TEST REPORT\s*Passport No\s*:?\s*Patient Information\s*"
        r"Name\s*:.*?Sample Type\s*:\s*\w[\w ]*)"
    )
    matches = list(re.finditer(patient_block, text, flags=re.DOTALL))
    if len(matches) > 1:
        for m in matches[1:]:
            text = text[: m.start()] + text[m.end() :]

    # Collapse resulting whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# Section header patterns commonly found in lab reports
_SECTION_SPLIT_PATTERN = re.compile(
    r"(?="
    r"(?:Biochemistry|Immunoassay|Thyroid Function Test|"
    r"HbA1c\s*\(Glycosylated|Lipid Profile|"
    r"Total WBC and Differential|Complete Blood Count|"
    r"HB Electrophoresis|Haematology|"
    r"Urine (?:Routine|Analysis)|Microscopic Examination|"
    r"Peripheral Smear|Iron Studies|"
    r"(?:Page \d+ of \d+))"
    r")",
    re.IGNORECASE,
)


def chunk_text_structured(
    text: str, chunk_size: int = 200, overlap: int = 50
) -> list[str]:
    """
    Structure-aware chunking: split on section/page boundaries first,
    then sub-chunk large sections with word-level overlap.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap.")

    # Split into sections using known lab report headers
    sections = _SECTION_SPLIT_PATTERN.split(text)
    sections = [s.strip() for s in sections if s.strip()]

    chunks: list[str] = []
    for section in sections:
        words = section.split()
        if not words:
            continue
        if len(words) <= chunk_size:
            if len(section) > 20:
                chunks.append(section)
        else:
            step = chunk_size - overlap
            for i in range(0, len(words), step):
                chunk = " ".join(words[i : i + chunk_size]).strip()
                if len(chunk) > 20:
                    chunks.append(chunk)

    print(
        f"[Chunker] Structured chunking: {len(chunks)} chunks "
        f"from {len(sections)} sections"
    )
    return chunks


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
