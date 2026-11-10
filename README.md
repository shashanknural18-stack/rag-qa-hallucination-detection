# Question Answering System Using RAG with Multi-Method Hallucination Detection

A production-grade Retrieval-Augmented Generation (RAG) question-answering system that retrieves grounded evidence from a private document collection, generates answers using Mistral-7B-Instruct (quantised GGUF), and validates factual consistency using four independent detection methods (Cross-Encoder NLI, SelfCheckGPT stochastic consistency, Factuality judge, and Citation semantic matching).

---

## Architecture Diagram

```
                 ┌────────────────── User Question ──────────────────┐
                 │                                                   │
                 ▼                                                   ▼
┌─────────────────────────────────┐                 ┌─────────────────────────────────┐
│ Module 1: Ingestion Pipeline    │                 │ Module 2: Query Processing      │
│  - Text Extract (PDF, TXT, MD)  │                 │  - Contraction expansion        │
│  - Token Chunker (240 tokens)   │                 │  - Normalization & Cleaning     │
│  - all-MiniLM-L6-v2 Embeddings  │                 │  - Dense Embedding Vector       │
│  - FAISS Vector Store + Records │                 └────────────────┬────────────────┘
└────────────────┬────────────────┘                                  │
                 │                                                   │
                 ▼                                                   ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Module 3: Hybrid Retrieval & Reranking                                              │
│  - Dense inner-product search (top-10 candidates)                                   │
│  - Optional BM25 sparse search + Reciprocal Rank Fusion (RRF)                       │
│  - Cross-Encoder ms-marco-MiniLM-L-6-v2 Reranker (top-5 final chunks)               │
│  - Early refusal if no chunks exceed MIN_RERANK_SCORE                               │
└────────────────────────────────────────┬────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Module 4: Grounded Generation (Local CPU GGUF)                                      │
│  - Mistral-7B-Instruct Q4 GGUF via llama-cpp-python                                 │
│  - Strict [INST] prompt builder with numbered source chunks                         │
│  - In-line citation parser ([n] extraction & validation)                            │
│  - Stochastic sampling (N answers at temperature 0.7 for SelfCheckGPT)              │
└────────────────────────────────────────┬────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Module 5: Multi-Method Hallucination Detection & Aggregation                        │
│  1. NLI Entailment: DeBERTa-v3-small premise/hypothesis scoring (weight: 0.35)      │
│  2. SelfCheckGPT: Agreement across stochastic samples (weight: 0.25)                │
│  3. Factuality Judge: Atomic claim verification via LLM (weight: 0.25)              │
│  4. Citation Support: Cosine similarity between sentences & chunks (weight: 0.15)   │
│  -> Aggregator: Renormalized score + Hard Contradiction Cap                         │
│  -> Final Decision: SHOW (>= 0.75) | WARN (0.50 - 0.74) | REFUSE (< 0.50)           │
└────────────────────────────────────────┬────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Module 6: Interactive Web UI (Streamlit) & RAGPipeline Orchestrator                 │
│  - Decision banner, confidence meter, per-sentence verification breakdown           │
│  - Document uploader with live index rebuild                                        │
│  - Session history, source inspector, and latency profiling                         │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Setup & Installation

### Windows (PowerShell)

```powershell
# Clone or navigate to repository
cd rag-qa-hallucination-detection

# Create virtual environment
python -m venv venv

# If script execution is restricted in PowerShell, run:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

*Note:* You can also run commands directly using `.\venv\Scripts\python.exe` without activating.

### Linux / macOS

```bash
cd rag-qa-hallucination-detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Model Downloads

### 1. GGUF Generation Model (Mistral-7B-Instruct-v0.2 Q4_K_M)

```powershell
# Create models directory
New-Item -ItemType Directory -Force -Path models

# Download via huggingface-cli
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF mistral-7b-instruct-v0.2.Q4_K_M.gguf --local-dir models
```

If `llama-cpp-python` pre-built wheels are needed on Windows:
```powershell
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

### 2. Hugging Face Models (Auto-cached on first run)
The embedder, reranker, and NLI models download automatically on first run to the local Hugging Face cache:
- `sentence-transformers/all-MiniLM-L6-v2` (Dense Embedder)
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (Reranker)
- `cross-encoder/nli-deberta-v3-small` (NLI Detector)

---

## Ingesting Documents

Place PDF, TXT, MD, or DOCX documents into `data/raw/`:

```powershell
# Ingest new/changed documents into FAISS vector store
python -m src.ingestion.pipeline

# Rebuild index from scratch
python -m src.ingestion.pipeline --rebuild
```

---

## Running the Interactive Web App

Launch the Streamlit interface:

```powershell
streamlit run app.py
```
*(Or directly: `.\venv\Scripts\streamlit.exe run app.py`)*

Open your browser at `http://localhost:8501`.

---

## Command-Line Demos

```powershell
# 1. Full Detection Pipeline (Retrieval + Generation + Verification)
python demo_detection.py "How fast must a return be acknowledged?"

# 2. Generation Only (with citations)
python demo_generation.py "How fast must a return be acknowledged?"

# 3. Retrieval & Reranking Only
python demo_retrieval.py "How fast must a return be acknowledged?"

# 4. Multi-document query
python demo_retrieval.py "I ordered a phone online, it arrived with a manufacturing defect, and when I tried to return it the platform said it would only process my return if I shared my Aadhaar number. What are my rights?"
```

---

## Configuration Settings (`src/config.py`)

All thresholds, paths, weights, and model parameters are consolidated in `src/config.py`:

| Parameter | Default | Environment Override | Meaning |
|---|---|---|---|
| `LLM_MODEL_PATH` | `models/mistral-7b-instruct-v0.2.Q4_K_M.gguf` | `LLM_MODEL_PATH` | Path to quantised GGUF model |
| `LLM_N_THREADS` | CPU cores (e.g. 4-8) | `LLM_N_THREADS` | Inference CPU threads |
| `THRESHOLD_SHOW` | `0.75` | `THRESHOLD_SHOW` | Minimum confidence score to display answer |
| `THRESHOLD_WARN` | `0.50` | `THRESHOLD_WARN` | Minimum confidence score to display with warning |
| `WEIGHTS` | `{"nli": 0.35, "selfcheck": 0.25, "factuality": 0.25, "citation": 0.15}` | — | Detection method weights (must sum to 1.0) |
| `NLI_ENTAIL_THRESHOLD` | `0.50` | — | Entailment probability threshold for sentence support |
| `CITATION_SUPPORT_THRESHOLD` | `0.45` | — | Cosine similarity threshold for citation validity |
| `CONTRADICTION_CAP_THRESHOLD`| `0.80` | — | NLI contradiction probability that caps SHOW to WARN |
| `NUM_SAMPLES_UI` | `3` | — | SelfCheckGPT stochastic sample count in UI |
| `TOP_K_FINAL` | `5` | — | Top reranked chunks supplied to generation prompt |

---

## Running Test Suites

All tests use pytest with `--capture=sys` configured in `pytest.ini`:

```powershell
# Fast unit tests (skips slow model inference)
$env:HF_HUB_OFFLINE="1"; pytest tests/ -v

# Specific module tests:
pytest tests/test_ingestion.py -v
pytest tests/test_query.py -v
pytest tests/test_retrieval.py -v
pytest tests/test_generation.py -v
pytest tests/test_detection.py -v
pytest tests/test_pipeline.py -v
pytest tests/test_ui_helpers.py -v
pytest tests/test_eval.py -v

# End-to-end slow tests (requires local GGUF + NLI models)
pytest tests/test_end_to_end.py -v -m slow
```

---

## Evaluation Benchmark (`run_eval.py`)

Evaluate detection accuracy, latency, and hallucination resistance against `eval/questions.json`:

```powershell
# 1. Quick evaluation (5 questions, skips LLM judge)
python run_eval.py --limit 5 --no-judge

# 2. Full benchmark run
python run_eval.py

# 3. Test refusal behavior on out-of-corpus questions only
python run_eval.py --category unanswerable

# 4. Error injection sensitivity test (tamper factual details and measure downgrades)
python run_eval.py --inject-errors
```

Results and confusion matrices are printed to stdout and saved in `eval/results/eval_<timestamp>.json`.

---

## Decision Thresholds Explained

- **`SHOW` (Confidence ≥ 0.75):** All sentences are strongly supported by retrieved chunks, citations are semantically aligned, and SelfCheckGPT samples agree. Displayed in green with `[n]` citations.
- **`WARN` (0.50 ≤ Confidence < 0.75):** Some claims are partially unverified or contradicted. If NLI contradiction probability exceeds 0.80 on any sentence, the decision is capped at `WARN`. Displayed in yellow with warning text.
- **`REFUSE` (Confidence < 0.50 or Retrieval Empty):** The system cannot verify the answer against retrieved documents. The ungrounded answer is replaced with a polite refusal message.

---

## System Limitations & Design Trade-offs

1. **CPU Latency:** Running 7B-parameter quantised GGUF inference on CPU takes several seconds per completion. In the UI, default samples are set to 3 to balance speed and consistency verification.
2. **NLI Numeric Reasoning:** Cross-encoder NLI models excel at semantic entailment but can miss complex arithmetic, date math, or unit conversions.
3. **Small-Corpus Tuning:** Retrieval thresholds (`MIN_RERANK_SCORE = 0.0`) are tuned for high recall on targeted corpora.
4. **LLM Judge Edge Cases:** Mistral-7B-Instruct serves as factuality judge. While robust against JSON schemas, severe context truncation can result in a skipped detector status.

---

## Roadmap

- [x] **Module 1** — Document Ingestion (loader, chunker, embedder, FAISS vector store)
- [x] **Module 2** — Query Processing (cleaning, contractions, query vector)
- [x] **Module 3** — Retrieval (dense top-10, BM25 hybrid option, cross-encoder reranking to top-5)
- [x] **Module 4** — Generation (Mistral-7B-Instruct Q4 GGUF, citation parser, stochastic sampling)
- [x] **Module 5** — Hallucination Detection (NLI, SelfCheckGPT, Factuality judge, Citation matching, Aggregator)
- [x] **Module 6** — Streamlit UI & Orchestrator (`app.py`, `RAGPipeline`, evaluation benchmark)
