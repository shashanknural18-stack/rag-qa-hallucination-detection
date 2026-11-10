# Question Answering System Using RAG with Multi-Method Hallucination Detection

A Retrieval-Augmented Generation (RAG) question-answering system that retrieves evidence from a private document collection, generates a grounded answer, and verifies it with four independent hallucination-detection methods (NLI, SelfCheckGPT, Factuality judge, Citation matching). The system produces a confidence score that decides whether to show the answer, show it with a warning, or refuse.

## Module 1 Data Flow

```
data/raw/*  ->  Loader (1.1)  ->  Chunker (1.2)  ->  Embedder (1.3)  ->  FAISS index + records.json (+ manifest.json)
```

## Module 2: Query Processing

`QueryProcessor` takes one shared `EmbeddingGenerator`. It keeps a whitespace-normalized raw query for the reranker and later prompt, and a cleaned query (lowercased, contractions expanded, legal punctuation kept) that is embedded with `all-MiniLM-L6-v2`.

```
user text  ->  QueryProcessor.clean  ->  QueryProcessor.embed  ->  ProcessedQuery
```

## Module 3: Retrieval

`RetrievalModule` accepts only a `ProcessedQuery`. It searches the Module 1 FAISS store for 10 candidates, optionally fuses a BM25 list, then reranks with `cross-encoder/ms-marco-MiniLM-L-6-v2` and keeps the top 5 as a `RetrievalResult`.

```
ProcessedQuery  ->  dense top-10  ->  optional BM25 + RRF  ->  rerank top-5  ->  RetrievalResult
```

```bash
python demo_retrieval.py "How fast must a return be acknowledged?"
python demo_retrieval.py "How fast must a return be acknowledged?" --bm25
python demo_retrieval.py "How fast must a return be acknowledged?" --no-rerank
```

### Retrieval config

| Name | Default | Meaning |
|---|---|---|
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder used after dense search |
| `RERANK_BATCH_SIZE` | `16` | CPU batch size for reranking |
| `TOP_K_RETRIEVE` | `10` | Candidates from the first stage |
| `TOP_K_FINAL` | `5` | Chunks kept after reranking |
| `USE_BM25` | `False` | Hybrid BM25 fusion (off by default) |
| `RRF_K` | `60` | Reciprocal rank fusion constant |
| `MIN_RERANK_SCORE` | `0.0` | Drop reranked chunks below this 0-1 score |
| `MAX_QUERY_CHARS` | `1000` | Longer cleaned queries are truncated |

## Setup

### Windows (PowerShell)

```powershell
cd rag-qa-hallucination-detection
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Linux / macOS

```bash
cd rag-qa-hallucination-detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Adding Documents

Place PDF, TXT, MD, or DOCX files into `data/raw/`. The pipeline will pick them up automatically.

## Running the Pipeline

```bash
python -m src.ingestion.pipeline
```

### Rebuild from Scratch

If you edit or delete source documents, use `--rebuild` to clear stale chunks and re-index everything:

```bash
python -m src.ingestion.pipeline --rebuild
```

### Quick Check

```bash
python quick_check.py "How fast must a return be acknowledged?"
```

## Running Tests & Verification Guidelines

### 1. Pytest Test Suites

Run the full test suite (32 tests covering Ingestion, Query Processing, and Retrieval):

```bash
pytest tests/ -v
```

Run specific module test suites individually:

```bash
# Module 1 (Document Ingestion) - 11 tests
pytest tests/test_ingestion.py -v

# Module 2 (Query Processing) - 6 tests
pytest tests/test_query.py -v

# Module 3 (Retrieval & Reranking) - 15 tests
pytest tests/test_retrieval.py -v
```

> **Tip (Offline Execution):** If running offline or in restricted network environments, set `HF_HUB_OFFLINE=1` to skip Hugging Face hub connection attempts and use locally cached models directly:
> - **PowerShell:** `$env:HF_HUB_OFFLINE="1"; pytest tests/ -v`
> - **Bash:** `HF_HUB_OFFLINE=1 pytest tests/ -v`

### 2. End-to-End Acceptance Commands

Run the full pipeline acceptance sequence:

```powershell
# 1. Activate virtual environment
venv\Scripts\activate

# 2. Ingest documents into vector store
python -m src.ingestion.pipeline

# 3. Quick sanity check on the index
python quick_check.py "How fast must a return be acknowledged?"

# 4. Dense retrieval + Cross-encoder reranking (default)
python demo_retrieval.py "How fast must a return be acknowledged?"

# 5. Hybrid retrieval (Dense + BM25 sparse + Reciprocal Rank Fusion + Reranking)
python demo_retrieval.py "How fast must a return be acknowledged?" --bm25

# 6. First-stage retrieval only (skip cross-encoder)
python demo_retrieval.py "How fast must a return be acknowledged?" --no-rerank

# 7. Custom top-k chunks
python demo_retrieval.py "How fast must a return be acknowledged?" --k 3
```

### 3. Sanity Benchmark Queries (Section 9 Acceptance Criteria)

Verify retrieval quality against the sample legal corpus (`consumer_rights.txt` and `data_privacy.txt`):

| Query | Expected Top-1 / Selection | Test Command |
|---|---|---|
| How fast must a return be acknowledged? | `consumer_rights.txt` (rank 1) | `python demo_retrieval.py "How fast must a return be acknowledged?"` |
| What counts as a deficiency in goods? | `consumer_rights.txt` (rank 1) | `python demo_retrieval.py "What counts as a deficiency in goods?"` |
| Can a platform force me to share my ID number? | `data_privacy.txt` (rank 1) | `python demo_retrieval.py "Can a platform force me to share my ID number?"` |
| What kind of consent is valid? | `data_privacy.txt` (rank 1) | `python demo_retrieval.py "What kind of consent is valid?"` |
| Long Aadhaar demo query (manufacturing defect + mandatory ID sharing) | Top-5 contains chunks from **both** files | `python demo_retrieval.py "I ordered a phone online, it arrived with a manufacturing defect, and when I tried to return it the platform said it would only process my return if I shared my Aadhaar number with their third-party pickup partner. I wasn't given a clear option to refuse. What are my rights here?"` |

### 4. Syntax & Compilation Verification

Ensure all source files compile cleanly without syntax errors:

```bash
python -m compileall src tests demo_retrieval.py quick_check.py
```

## Roadmap

- [x] **Module 1** — Document Ingestion (loader, chunker, embedder, FAISS vector store)
- [x] **Module 2** — Query Processing
- [x] **Module 3** — Retrieval (dense top-10, cross-encoder rerank to top-5)
- [ ] **Module 4** — Generation (Mistral-7B-Instruct Q4 GGUF, temp 0.1)
- [ ] **Module 5** — Hallucination Detection (NLI, SelfCheckGPT, Factuality judge, Citation matching)
- [ ] **Module 6** — Streamlit UI
