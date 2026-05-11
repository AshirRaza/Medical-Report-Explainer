# Medical-Report-Explainer

Deep learning project for **medical lab report Q&A** — combines OCR, hybrid RAG (dense + sparse retrieval), a **fine-tuned cross-encoder reranker**, and Claude LLM generation with comprehensive evaluation.

---

## Current implementation (status)

This section describes **what exists in the repository today** and how to use it.

### Implemented features

1. **PDF and image OCR via Qwen VL (OpenRouter)**  
   - `ocr/extract.py` converts PDF pages to images with [pdf2image](https://github.com/Belval/pdf2image) (requires **Poppler** on the system).  
   - **Image quality before OCR:** default **300 DPI** rendering, optional **margin crop** (header/footer noise), **deskew** for mild page tilt, and **contrast boost** (`preprocess_image_for_ocr` / `extract_text_from_pdf` parameters).  
   - Each page image is sent to a **vision-language model** through the [OpenRouter](https://openrouter.ai) API using an OpenAI-compatible `chat/completions` flow, with the image as a base64 data URL.  
   - The model is configured via environment variables (see below). Default model name in `.env.example` is `qwen/qwen3-vl-32b-instruct`; you can point to another OpenRouter vision model if you prefer.

2. **OCR evaluation on MedOCR Vision**  
   - `ocr/evaluate.py` loads the Hugging Face dataset [`naazimsnh02/medocr-vision-dataset`](https://huggingface.co/datasets/naazimsnh02/medocr-vision-dataset) and, for each sample, compares predictions to ground-truth text using **CER** and **WER** from [jiwer](https://github.com/jwerlinger/jiwer).  
   - **Two systems are evaluated:**  
     - **Qwen path:** same OpenRouter-backed OCR as in `extract.py` (including preprocessing when enabled in evaluation).  
     - **PaddleOCR baseline:** local PaddleOCR (PaddlePaddle + PaddleOCR stack on Windows).  
   - Results are written to JSON under `Results/` (e.g. `ocr_evaluation_results.json`, `ocr_evaluation_results_latest.json`) with mean metrics and a relative **CER reduction %** vs Paddle when both runs succeed.

3. **RAG core — hybrid retrieval + fine-tuned reranker**  
   - `rag/chunker.py` — `clean_text()` normalizes OCR text; `clean_report_noise()` strips boilerplate (QR codes, signatures, page markers); `chunk_text_structured()` splits by medical section headings before sub-chunking (200 words, 50 overlap).  
   - `rag/embedder.py` — bi-encoder **`pritamdeka/S-PubMedBert-MS-MARCO`** with **L2-normalized** embeddings for cosine-as-dot-product search.  
   - `rag/store.py` — **`faiss.IndexFlatIP`** dense search + **BM25Okapi** sparse keyword search + **hybrid fusion** (alpha-blended min-max normalized scores).  
   - `rag/reranker.py` — **fine-tuned cross-encoder** (base: `ms-marco-MiniLM-L-6-v2`, 22M params) to rerank candidates into **top-n** chunks. Automatically loads fine-tuned weights from `models/finetuned-reranker/` if available.

4. **LLM layer (Phase 5) — Anthropic Claude**  
   - `llm/generator.py` — **`generate_answer(query, context_chunks)`** answers patient questions using only the provided chunks (grounded, low temperature).  
   - **`generate_summary(full_text)`** produces a short patient-friendly overview from OCR text (first ~12k characters sent to control cost).  
   - Model id from **`CLAUDE_MODEL`** in `.env` (default `claude-sonnet-4-6`); requires **`ANTHROPIC_API_KEY`**.

5. **End-to-end pipeline**  
   - **`pipeline.py`** — class **`MedicalReportPipeline`**: loads the OpenRouter Qwen OCR client once; **`process_pdf(pdf_path)`** runs OCR → noise cleaning → structured chunking → build FAISS + BM25 indexes → **`generate_summary`**; **`answer(query)`** runs **hybrid search** → **rerank** (fine-tuned) → **`generate_answer`**, returning **`(answer, top_chunks)`**.  
   - Constructor kwargs: **`top_k=20`**, **`top_n=5`**, **`ocr_dpi=300`**, **`hybrid_alpha=0.5`**.  
   - If OCR yields no usable text, the index is not built and **`answer`** raises until a successful **`process_pdf`**.

6. **RAG evaluation (Phase 7) — PubMedQA + RAGAS**  
   - **`rag/evaluate.py`** — loads Hugging Face **`pubmed_qa`** / **`pqa_labeled`**, shuffles a fixed-size slice, and for each example builds chunks from the abstract(s), then compares **three** retrieval settings: **`rag_rerank`** (FAISS top‑k → cross‑encoder rerank → top‑n), **`rag_no_rerank`** (FAISS order, top‑n only), **`no_rag`** (single truncated full document context, no retrieval).  
   - **`generate_pubmedqa_answer`** in **`llm/generator.py`** asks Claude for a short grounded answer ending in **yes / no / maybe**; **accuracy** is the fraction of examples where that label matches **`final_decision`** (unparseable outputs count as wrong).  
   - **RAGAS** (optional, default on): legacy **`faithfulness`** and **`context_recall`** from **`ragas`** are run on the rows produced under **`--ragas-on-mode`** (default **`rag_rerank`**) using the same **`ANTHROPIC_API_KEY`** / **`CLAUDE_MODEL`** as the rest of the app. Use **`--skip-ragas`** for accuracy-only runs.  
   - **`evaluate_full.py`** — runs OCR evaluation and/or PubMedQA RAG in one invocation and writes a **combined** JSON report plus the usual per-task JSON files under **`Results/`**.

7. **Full evaluation runner (Phase 8)**  
   - **`evaluate_full.py`** — orchestrates optional OCR and/or PubMedQA RAG evaluation in a single run, saves per-task JSONs plus a combined report with **improvement deltas** (accuracy differences across modes, CER/WER reductions).

8. **Real-PDF RAG evaluation**  
   - **`evaluate_pdf.py`** — runs **25 targeted questions** against a real 19-page lab report (`Sample Reports/comprehensive_report.pdf`) in **RAG** vs **no-RAG** modes side-by-side. Uses hybrid retrieval + fine-tuned reranker. Caches OCR text for fast re-runs. Supports `--no-hybrid`, `--no-structured-chunking`, and configurable `--top-k` / `--hybrid-alpha` for ablation studies.

9. **Deep Learning: Cross-Encoder Fine-Tuning**  
   - `rag/build_training_data.py` — generates labeled `(query, chunk, relevance)` training pairs from PDF evaluation results and PubMedQA.  
   - `rag/finetune_reranker.py` — fine-tunes `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M params) using binary cross-entropy loss with AdamW optimizer, warmup scheduling, and train/eval splits. Runs on CPU (~14 min) or GPU.  
   - `rag/evaluate_reranker.py` — compares base vs fine-tuned model on Precision@k, Recall@k, and MAP.  
   - Training data: 720 labeled pairs (360 positive, 360 negative) from PDF eval + PubMedQA.  
   - Result: Precision@3 improved from **89.25% → 90.86%**, MAP from **97.00% → 97.31%**.

10. **Environment and safety**  
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
| `ANTHROPIC_API_KEY` | Required for **`generate_summary`** / **`generate_answer`** and the full **`MedicalReportPipeline`** (not required for OCR-only or RAG-only smoke tests). |
| `OPENROUTER_API_KEY` | Bearer token for OpenRouter. |
| `OPENROUTER_BASE_URL` | API base, typically `https://openrouter.ai/api/v1`. |
| `QWEN_VL_MODEL` | OpenRouter model id for vision OCR (e.g. `qwen/qwen3-vl-32b-instruct`). |
| `CLAUDE_MODEL` | Claude model id for `llm.generator` and the pipeline (e.g. `claude-sonnet-4-6`). |
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
.\venv\Scripts\python -c "from pathlib import Path; from ocr.extract import load_qwen2vl, extract_text_from_pdf; Path('Results').mkdir(exist_ok=True); pdfs=list(Path('Sample Reports').glob('*.pdf')); pdf=str(pdfs[0]); m,p=load_qwen2vl(); t=extract_text_from_pdf(pdf,m,p); print(t[:3000]); Path('Results/ocr_output.txt').write_text(t, encoding='utf-8')"
```

**Run MedOCR evaluation (Qwen + Paddle, 50 samples, require Paddle):**

```powershell
.\venv\Scripts\python -m ocr.evaluate --num-samples 50 --require-paddle --output Results/ocr_evaluation_results.json
```

Use `--num-samples 10` for a quicker dry run. Use `--no-paddle` if you only want Qwen metrics.

**RAG smoke test (loads embedding + reranker models on first run):**

```powershell
.\venv\Scripts\python -c "from rag.chunker import clean_text, chunk_text; from rag.store import build_index, search_index; from rag.reranker import rerank; text=clean_text('HbA1c 6.2 percent. Hemoglobin 12.5 g per dL. ' * 30); ch=chunk_text(text); idx,keep=build_index(ch); cands=search_index('What is hemoglobin?', idx, keep, top_k=5); print(rerank('What is hemoglobin?', cands, top_n=2))"
```

**Full pipeline (OCR + index + summary + Q&A; requires OpenRouter, Anthropic, Poppler, and a PDF path):**

```powershell
.\venv\Scripts\python -c "from pathlib import Path; from pipeline import MedicalReportPipeline; pdfs=list(Path('Sample Reports').glob('*.pdf')); assert pdfs, 'Add a PDF under Sample Reports/'; p=MedicalReportPipeline(); s=p.process_pdf(str(pdfs[0])); print('--- Summary ---'); print(s[:1500]); a, ch=p.answer('What are the main lab results?'); print('--- Answer ---'); print(a); print('--- Chunks used ---', len(ch))"
```

**PubMedQA RAG benchmark (requires `ANTHROPIC_API_KEY`; first run downloads PubMedQA + embedding/reranker weights):**

```powershell
.\venv\Scripts\python -m rag.evaluate --num-samples 15 --skip-ragas --output Results/rag_pubmedqa_eval.json
```

Omit `--skip-ragas` to also run RAGAS **faithfulness** and **context_recall** on the **`rag_rerank`** path (extra Claude calls). Use `--ragas-on-mode rag_no_rerank` or `no_rag` to score RAGAS on another mode’s retrieved rows.

**Combined OCR + RAG report (`evaluate_full.py`):**

```powershell
.\venv\Scripts\python evaluate_full.py --ocr-num-samples 0 --rag-num-samples 10 --skip-ragas
```

Use `--ocr-num-samples 20` (and optional `--ocr-no-paddle`) to include MedOCR in the same JSON. Default is **RAG only** with **10** PubMedQA examples; set `--rag-num-samples 0` and a positive OCR count to run OCR alone.

### Project layout

```
Medical-Report-Explainer/
├── .env                 # local secrets (gitignored)
├── .env.example         # template for env vars
├── .gitignore
├── requirements.txt
├── pipeline.py          # MedicalReportPipeline: OCR -> hybrid RAG -> summary / Q&A
├── evaluate_full.py     # optional OCR + optional PubMedQA RAG, combined JSON
├── evaluate_pdf.py      # RAG vs no-RAG side-by-side on real PDF reports
├── test_keys.py         # verify .env loading
├── Results/
│   ├── ocr_evaluation_results.json
│   ├── ocr_evaluation_results_latest.json
│   ├── pdf_eval_report.json               # v1 baseline RAG evaluation
│   ├── pdf_eval_report_v2.json            # v2 with hybrid search + structured chunking
│   ├── pdf_eval_report_v2_35k.json        # v2 with 35k no-RAG context (100% coverage)
│   ├── reranker_eval.json                 # base vs fine-tuned reranker comparison
│   └── results_summary.txt
├── ocr/
│   ├── __init__.py      # re-exports extract helpers
│   ├── extract.py       # PDF → images → OpenRouter Qwen OCR
│   └── evaluate.py      # MedOCR CER/WER: Qwen vs PaddleOCR
├── rag/
│   ├── __init__.py
│   ├── chunker.py       # clean + noise removal + structured chunking
│   ├── embedder.py      # S-PubMedBert bi-encoder embeddings
│   ├── store.py         # FAISS + BM25 + hybrid search
│   ├── reranker.py      # cross-encoder rerank (auto-loads fine-tuned weights)
│   ├── evaluate.py      # PubMedQA: 3 retrieval modes + accuracy + optional RAGAS
│   ├── build_training_data.py   # generate labeled pairs for fine-tuning
│   ├── finetune_reranker.py     # DL training script (cross-encoder fine-tuning)
│   ├── evaluate_reranker.py     # before/after reranker quality comparison
│   └── training_data.json       # 720 labeled (query, chunk, label) pairs
├── models/
│   └── finetuned-reranker/      # fine-tuned cross-encoder weights (22M params)
├── llm/
│   └── generator.py     # Claude: generate_answer, generate_summary
└── Sample Reports/
    ├── comprehensive_report.pdf  # 19-page lab report for real-PDF evaluation
    └── questions.json            # 25 targeted questions for evaluate_pdf.py
```

### Dependencies (high level)

Core entries in `requirements.txt` include: `accelerate`, `anthropic`, `datasets`, `faiss-cpu`, `gradio`, `jiwer`, `numpy`, `pdf2image`, `pillow`, `python-dotenv`, `ragas`, `rank-bm25`, `sentence-transformers`, `torch`, `transformers`, plus **`paddleocr`** and **`paddlepaddle`** for the OCR baseline.

### Evaluation notes (MedOCR)

- Metrics are **strict** CER/WER (`jiwer`), so layout, spacing, and punctuation mismatches still inflate error rates.  
- **Qwen vs Paddle:** Qwen remains better than the Paddle baseline on the same 50 samples.  
- For reporting, you can cite **Qwen vs Paddle** and **Qwen before vs after preprocessing**; see the summary table below.

### OCR evaluation summary (50 samples, MedOCR Vision)

Two committed runs are stored for comparison (same dataset split, **50 scored samples**, same Paddle baseline numbers):

| Run | File | Qwen mean CER | Qwen mean WER | Paddle mean CER | Paddle mean WER | CER reduction vs Paddle |
|-----|------|---------------|---------------|-----------------|-----------------|-------------------------|
| **Baseline** (Qwen on raw dataset images) | `Results/ocr_evaluation_results.json` | 0.5029 | 0.6425 | 0.5091 | 1.6043 | 1.22% |
| **Improved** (Qwen after crop + deskew + contrast, same as `extract.py` helpers) | `Results/ocr_evaluation_results_latest.json` | **0.4730** | **0.6170** | 0.5091 | 1.6043 | **7.09%** |

MedOCR images are already rasterized; **DPI changes apply to PDF OCR in `extract.py`**, not to dataset tiles. The gain here is from **spatial cleanup + contrast** before the vision model.

**Improvement from preprocessing (Qwen only, same 50 scored samples):**

- Mean CER: **0.5029 → 0.4730** (−0.0299, about **5.9%** lower relative to the baseline Qwen CER).  
- Mean WER: **0.6425 → 0.6170** (−0.0255, about **4.0%** lower relative to the baseline Qwen WER).  
- **Paddle metrics are unchanged** between the two JSON files, so the gain is from the **Qwen image pipeline**, not a different baseline run.

Interpretation:

- **Crop + deskew + contrast** measurably improved Qwen on this MedOCR slice; **PDF OCR** in `extract.py` also gains from **higher DPI** (300) on top of the same preprocessing.  
- Absolute CER is still in the **high-0.4** range, so continue to treat numbers as **benchmark-relative** unless you add normalization for fairer human-style comparison.

A short rolling log lives in `Results/results_summary.txt` (you can append future runs there).

### PubMedQA RAG evaluation (results on disk)

Runs use **`pubmed_qa` / `pqa_labeled`**, **`split=train`**, **`seed=42`**, default chunking (**200-word / 50-word overlap**), **`top_k=10`**, **`top_n=3`**, bi-encoder **`pritamdeka/S-PubMedBert-MS-MARCO`**, reranker **`cross-encoder/ms-marco-MiniLM-L-6-v2`**, and Claude via **`generate_pubmedqa_answer`** for **yes / no / maybe** accuracy vs **`final_decision`**. **`no_rag`** passes a **single** context string truncated to **`max_no_rag_chars` (12000)** (often nearly the full abstract for PubMedQA).

Committed JSON under **`Results/`**:

| File | `num_samples` | RAGAS | Label accuracy (`rag_rerank` / `rag_no_rerank` / `no_rag`) | RAGAS on `rag_rerank` rows |
|------|----------------|-------|-------------------------------------------------------------|----------------------------|
| `rag_pubmedqa_eval_100.json` | 100 | skipped (`--skip-ragas`) | **0.76 / 0.74 / 0.77** | — |
| `rag_pubmedqa_ragas_25.json` | **50** (output filename is legacy) | on | **0.74 / 0.74 / 0.76** | **faithfulness ≈ 0.84**, **context_recall ≈ 0.70** |
| `rag_pubmedqa_ragas_10.json` | 10 | on | **0.90 / 0.90 / 0.80** | **faithfulness ≈ 0.87**, **context_recall ≈ 0.56** |

**How to read these numbers**

- **At N=100 (accuracy-only):** **`no_rag` (0.77)** is slightly ahead of **`rag_rerank` (0.76)** and **`rag_no_rerank` (0.74)`**. The **two-point** gap between **`rag_rerank`** and **`rag_no_rerank`** suggests the **cross-encoder improves ordering** when only **top-n chunks** reach the LLM. **`no_rag`** is a **strong baseline** because PubMedQA abstracts are **short** and usually **fit (or nearly fit)** the 12k-character window, so the model sees **most of the same evidence** as in the chunked path.
- **At N=50 with RAGAS:** **`rag_rerank`** and **`rag_no_rerank`** **tie** on label accuracy (0.74); **`no_rag`** is slightly higher (0.76). **Faithfulness ~0.84** means answers are **largely grounded** in the **retrieved** chunks; **context_recall ~0.70** means retrieval **covers a substantial part** of the reference **`long_answer`** under RAGAS’s sentence-level test (still room to improve).
- **At N=10:** label accuracies are **high-variance**; treat RAGAS there as **directional** only.

**Why RAG can look “flat” vs full context on this benchmark**

- **`top_n=3`** drops any evidence in lower-ranked chunks; **`no_rag`** may still contain it in one block.
- **Yes / no / maybe** over a **single abstract** is often solvable from a **global** read; dense retrieval matters **more** on **long, noisy PDF lab reports** than on this Hugging Face slice.

**Follow-ups (experiments, not stored in JSON):** raise **`--top-k` / `--top-n`**, tune **chunk size**, or add **hybrid retrieval / domain rerankers** to push **`rag_rerank`** above **`no_rag`** on PubMedQA; **evaluate on real PDFs** for product-level conclusions.

### Real-PDF RAG evaluation (comprehensive_report.pdf)

The PubMedQA benchmark uses short abstracts where the full text fits in a single context window. To demonstrate RAG's real value, `evaluate_pdf.py` runs **25 targeted questions** against a **19-page lab report** in two modes side-by-side.

**Pipeline (v2 — current):**
- **RAG mode**: OCR → noise cleaning → structured chunking → hybrid search (FAISS + BM25, top_k=20) → fine-tuned cross-encoder rerank (top_n=5) → Claude answer
- **no-RAG mode**: OCR → truncate to `max_no_rag_chars` → Claude answer

**Document stats**: ~33k characters, ~5,300 words, 42 structured chunks.

**Run the evaluation:**

```powershell
# With cached OCR text (fast, ~5 min):
.\venv\Scripts\python evaluate_pdf.py --pdf "Sample Reports/comprehensive_report.pdf" --skip-ocr --ocr-cache "Results/comprehensive_report_ocr_text.txt" --output "Results/pdf_eval_report_v2.json"

# Compare with old pipeline (ablation):
.\venv\Scripts\python evaluate_pdf.py --pdf "Sample Reports/comprehensive_report.pdf" --skip-ocr --ocr-cache "Results/comprehensive_report_ocr_text.txt" --no-hybrid --no-structured-chunking --top-k 10 --output "Results/pdf_eval_report_baseline.json"
```

**Results comparison (v1 → v2):**

| Metric | v1 (baseline) | v2 (hybrid + structured + noise clean) |
|--------|--------------|----------------------------------------|
| RAG correct answers | 18/25 (72%) | **23/25 (92%)** |
| RAG "could not find" | 7/25 | **2/25** |
| no-RAG correct (12k chars) | 13/25 (52%) | 15/25 (60%) |

**v2 fixed 5 previously failing questions:** cholesterol/triglycerides, blood group, ESR, liver enzymes, and albumin/A-G ratio — all due to hybrid BM25 retrieval catching keyword matches that dense search alone missed.

**Full-context experiment** (`pdf_eval_report_v2_35k.json` — no-RAG with 35k chars = 100% document):

| Metric | RAG (v2) | no-RAG (12k) | no-RAG (35k) |
|--------|----------|--------------|--------------|
| Correct answers | 23/25 (92%) | 15/25 (60%) | **25/25 (100%)** |
| Avg time/question | ~7s | ~5s | ~6s |
| Input cost/question | ~5k chars | ~12k chars | ~33k chars |

**Key insight:** For short documents (~33k chars) that fit the LLM context window, full-context no-RAG achieves perfect accuracy. RAG's value for such documents is **cost efficiency (7x cheaper)** and **chunk traceability**. RAG becomes essential once documents exceed ~100k characters.

### Cross-Encoder Fine-Tuning (Deep Learning Component)

Fine-tuned the reranker to improve medical query-chunk relevance scoring.

**Training pipeline:**

```powershell
# Step 1: Generate training data (720 labeled pairs from eval results + PubMedQA)
.\venv\Scripts\python -m rag.build_training_data --pdf-eval "Results/pdf_eval_report_v2.json"

# Step 2: Fine-tune (22M param cross-encoder, ~14 min on CPU)
.\venv\Scripts\python -m rag.finetune_reranker --training-data rag/training_data.json --epochs 3

# Step 3: Evaluate improvement
.\venv\Scripts\python -m rag.evaluate_reranker --eval-data rag/training_data.json --finetuned models/finetuned-reranker
```

**Training details:**
- Base model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M parameters)
- Loss: Binary cross-entropy
- Optimizer: AdamW (lr=2e-5, warmup 10%)
- Data: 720 pairs (612 train / 108 eval), 3 epochs, batch size 16
- Hardware: CPU (~14 min) or GPU (~2 min)

**Results** (`Results/reranker_eval.json`):

| Metric | Base (MS-MARCO) | Fine-tuned | Delta |
|--------|----------------|------------|-------|
| Precision@3 | 89.25% | **90.86%** | +1.61% |
| Recall@3 | 88.39% | **89.89%** | +1.50% |
| MAP | 97.00% | **97.31%** | +0.31% |

The fine-tuned model is automatically loaded by the pipeline when weights exist at `models/finetuned-reranker/`.

### Known limitations and next steps

- **Gradio `app.py`** is **not implemented yet**; it would expose **`MedicalReportPipeline`** in a browser UI.  
- PaddleOCR on Windows may need compatible Paddle versions and can be slow on CPU; first run downloads several model bundles under the user profile (e.g. `.paddlex`).
- Fine-tuning gains are modest (+1.6% P@3) due to high baseline and small training set from a single report. More diverse reports would widen the gap.
- The 2 remaining RAG failures (q14: electrolytes, q25: list all abnormals) are not retrieval problems — q14 is an LLM generation issue, q25 is architecturally impossible with 5 chunks.
