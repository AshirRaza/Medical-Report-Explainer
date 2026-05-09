# Medical-Report-Explainer

Deep learning–oriented project for **medical lab report Q&A** (full RAG pipeline, Claude generation, and Gradio UI are planned; **OCR is implemented and evaluated** as described below).

---

## Current implementation (status)

This section describes **what exists in the repository today** and how to use it.

### Implemented features

1. **PDF and image OCR via Qwen VL (OpenRouter)**  
   - `ocr/extract.py` converts PDF pages to images with [pdf2image](https://github.com/Belval/pdf2image) (requires **Poppler** on the system).  
   - Each page image is sent to a **vision-language model** through the [OpenRouter](https://openrouter.ai) API using an OpenAI-compatible `chat/completions` flow, with the image as a base64 data URL.  
   - The model is configured via environment variables (see below). Default model name in `.env.example` is `qwen/qwen3-vl-32b-instruct`; you can point to another OpenRouter vision model if you prefer.

2. **OCR evaluation on MedOCR Vision**  
   - `ocr/evaluate.py` loads the Hugging Face dataset [`naazimsnh02/medocr-vision-dataset`](https://huggingface.co/datasets/naazimsnh02/medocr-vision-dataset) and, for each sample, compares predictions to ground-truth text using **CER** and **WER** from [jiwer](https://github.com/jwerlinger/jiwer).  
   - **Two systems are evaluated:**  
     - **Qwen path:** same OpenRouter-backed OCR as in `extract.py`.  
     - **PaddleOCR baseline:** local PaddleOCR (PaddlePaddle + PaddleOCR stack on Windows).  
   - Results are written to JSON (e.g. `ocr_evaluation_results.json`) with mean metrics and a relative **CER reduction %** vs Paddle when both runs succeed.

3. **Environment and safety**  
   - `.env.example` lists required variables **without secrets** (safe to commit).  
   - `.gitignore` includes `.env` so real API keys are not pushed to GitHub.  
   - `test_keys.py` loads `.env` and prints whether keys and model names are present (useful smoke test after setup).

### Prerequisites

- **Python 3.12** (matches the current `venv` / wheel choices in this project).  
- **Poppler** for `pdf2image` (Windows: e.g. install via `winget` and set `POPPLER_PATH` to the Poppler `Library\bin` folder if `pdftoppm` is not on your `PATH`).  
- **Internet access** for Hugging Face datasets, OpenRouter, and first-time Paddle / model downloads.

### Environment variables

Copy `.env.example` to `.env` in the project root and fill in values:

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Reserved for upcoming Claude-based generation (not required for OCR-only flows today). |
| `OPENROUTER_API_KEY` | Bearer token for OpenRouter. |
| `OPENROUTER_BASE_URL` | API base, typically `https://openrouter.ai/api/v1`. |
| `QWEN_VL_MODEL` | OpenRouter model id for vision OCR (e.g. `qwen/qwen3-vl-32b-instruct`). |
| `CLAUDE_MODEL` | Claude model id for future LLM steps (e.g. `claude-sonnet-4-6`). |
| `POPPLER_PATH` | Optional. Directory containing Poppler binaries (e.g. `pdftoppm.exe`) if not on system `PATH`. |
| `OPENROUTER_HTTP_REFERER` | Optional. OpenRouter-recommended header (e.g. your site or `https://localhost`). |
| `OPENROUTER_APP_TITLE` | Optional. Short app name sent to OpenRouter. |

For faster or more reliable Hugging Face dataset downloads, you can set `HF_TOKEN` in `.env` (not listed in `.env.example` by default).

### Setup

```powershell
cd C:\Users\Ashir\Medical-Report-Explainer
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create `.env` from `.env.example`, add keys, set `POPPLER_PATH` if needed, then verify:

```powershell
python test_keys.py
```

### Usage

**OCR a PDF from Python (example: single PDF in `Sample Reports`):**

```powershell
.\venv\Scripts\python -c "from pathlib import Path; from ocr.extract import load_qwen2vl, extract_text_from_pdf; pdfs=list(Path('Sample Reports').glob('*.pdf')); pdf=str(pdfs[0]); m,p=load_qwen2vl(); t=extract_text_from_pdf(pdf,m,p); print(t[:3000]); Path('ocr_output.txt').write_text(t, encoding='utf-8')"
```

**Run MedOCR evaluation (Qwen + Paddle, 50 samples, require Paddle):**

```powershell
.\venv\Scripts\python -m ocr.evaluate --num-samples 50 --require-paddle --output ocr_evaluation_results.json
```

Use `--num-samples 10` for a quicker dry run. Use `--no-paddle` if you only want Qwen metrics.

### Project layout (relevant to current work)

```
Medical-Report-Explainer/
├── .env                 # local secrets (gitignored)
├── .env.example         # template for env vars
├── .gitignore
├── requirements.txt
├── test_keys.py         # verify .env loading
├── ocr/
│   ├── __init__.py      # re-exports extract helpers
│   ├── extract.py       # PDF → images → OpenRouter Qwen OCR
│   └── evaluate.py      # MedOCR CER/WER: Qwen vs PaddleOCR
├── rag/                 # package placeholder (RAG not wired yet)
├── llm/                 # package placeholder (Claude gen not wired yet)
└── Sample Reports/      # optional: place sample PDFs here
```

### Dependencies (high level)

Core entries in `requirements.txt` include: `anthropic`, `datasets`, `faiss-cpu`, `gradio`, `jiwer`, `numpy`, `pdf2image`, `pillow`, `python-dotenv`, `ragas`, `sentence-transformers`, `torch`, `transformers`, plus **`paddleocr`** and **`paddlepaddle`** for the OCR baseline.

### Evaluation notes (MedOCR)

- **Mean CER ~0.5** on a 50-sample run indicates both Qwen (API) and Paddle still differ substantially from reference strings in strict character alignment—common when layout, spacing, or punctuation do not match ground truth exactly.  
- In the same run, **Qwen showed lower mean CER and much lower mean WER than Paddle** on average; see `ocr_evaluation_results.json` for exact numbers.  
- For reporting, emphasize **relative comparison** (Qwen vs Paddle) unless you add **normalization** before CER/WER to better match how humans judge “same text.”

### Known limitations and next steps

- **RAG** (chunking, FAISS, reranker), **Claude answer generation**, **PubMedQA / RAGAS evaluation**, **full `pipeline.py`**, **`evaluate_full.py`**, and **Gradio `app.py`** are **not implemented yet** in this repo state; they are the natural follow-on after OCR is stable.  
- PaddleOCR on Windows may need compatible Paddle versions and can be slow on CPU; first run downloads several model bundles under the user profile (e.g. `.paddlex`).

---

## Original tagline

Deep Learning Project
