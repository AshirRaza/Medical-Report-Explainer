"""Claude API generation for grounded answers and report summaries."""

from __future__ import annotations

import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

_anthropic_client: anthropic.Anthropic | None = None

SYSTEM_PROMPT = """You are a helpful medical assistant helping patients understand their lab reports.

Rules you must follow:
1. Use ONLY the information in the context provided. Never add information from your training.
2. If the answer is not in the context, say exactly: "I could not find that in your report."
3. Explain in plain, patient-friendly language. Avoid jargon.
4. When mentioning a value, always include the unit and reference range if available.
5. Never provide medical diagnoses or treatment recommendations.
6. Keep answers concise and focused on what was asked."""


def _get_model_name() -> str:
    return os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6").strip()


def _get_client() -> anthropic.Anthropic:
    """Return a shared Anthropic client (one connection pool per process)."""
    global _anthropic_client
    if _anthropic_client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is missing. Set it in your .env file (see .env.example)."
            )
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


def generate_answer(query: str, context_chunks: list[str]) -> str:
    """
    Produce a patient-friendly answer grounded only in the retrieved context chunks.
    """
    query = (query or "").strip()
    if not query:
        return "Please ask a question about your report."

    chunks = [c.strip() for c in context_chunks if c and str(c).strip()]
    if not chunks:
        return "I could not find that in your report."

    client = _get_client()
    model = _get_model_name()
    context = "\n\n---\n\n".join(chunks)

    user_message = f"""Here is the relevant content from the patient's lab report:

<context>
{context}
</context>

Patient question: {query}

Answer based only on the context above:"""

    message = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=0.2,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return _first_text_block(message)


def generate_summary(full_text: str) -> str:
    """
    Produce a brief patient-friendly summary of the lab report from full OCR text.
    """
    text = (full_text or "").strip()
    if not text:
        return "No report text was available to summarize."

    client = _get_client()
    model = _get_model_name()
    excerpt = text[:12000]

    user_message = f"""Here is a patient's full lab report:

<report>
{excerpt}
</report>

Please provide a brief, patient-friendly summary of:
1. Which values are normal
2. Which values are outside the reference range (flag these clearly)
3. Any patterns worth noting

Use simple language. Do not diagnose."""

    message = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=0.2,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return _first_text_block(message)


def _first_text_block(message: anthropic.types.Message) -> str:
    for block in message.content:
        if block.type == "text":
            return block.text.strip()
    return ""


PUBMEDQA_SYSTEM = """You answer biomedical questions using ONLY the provided context passages.
Write 2–4 short sentences grounded in the context, then on the final line output exactly one word: yes, no, or maybe (lowercase), matching PubMedQA-style labels.
If the context is insufficient, say so briefly and end with maybe on the last line."""


def generate_pubmedqa_answer(query: str, context_chunks: list[str]) -> str:
    """
    PubMedQA-oriented answer for benchmarking: grounded prose plus a final-line yes/no/maybe.
    """
    query = (query or "").strip()
    if not query:
        return "No question was provided.\nmaybe"

    chunks = [c.strip() for c in context_chunks if c and str(c).strip()]
    if not chunks:
        return "The context is empty.\nmaybe"

    client = _get_client()
    model = _get_model_name()
    context = "\n\n---\n\n".join(chunks)

    user_message = f"""Context passages:

<context>
{context}
</context>

Question: {query}

Answer following the output rules in your instructions:"""

    message = client.messages.create(
        model=model,
        max_tokens=512,
        temperature=0.1,
        system=PUBMEDQA_SYSTEM,
        messages=[{"role": "user", "content": user_message}],
    )
    return _first_text_block(message)
