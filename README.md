# Hybrid RAG Document Engine

A production-oriented Retrieval-Augmented Generation (RAG) engine built to query complex institutional PDFs (tested on the IIEST Shibpur UG Academic Ordinance) with layout-aware table parsing, SHA-256 incremental indexing, hybrid retrieval, versioned SQLite caching, and automated evaluation.

## Architecture

- **Ingestion (`src/ingestion.py`):** Pure-Python `pdfplumber` extraction converting PDF tables into intact Markdown tables, paired with SHA-256 file hashing (`manifest.json`) to incrementally sync ChromaDB and BM25 without index desync.
- **Retrieval (`src/retriever.py`):** Configurable 3-mode retriever (`vector`, `hybrid`, `hybrid_rerank`) combining ChromaDB semantic search, regex-tokenized BM25, Reciprocal Rank Fusion (RRF), deterministic domain synonym expansion, and prompt-injection filtering.
- **Generation & Caching (`src/generator.py`, `src/cache.py`):** Streaming Gemini client with fail-fast model fallback, strict citation grounding, telemetry logging, and a persistent SQLite cache versioned by corpus hash (sub-millisecond repeat hits, auto-invalidated on document edits).
- **UI (`app.py`):** Interactive Streamlit interface supporting multi-PDF uploads, mode toggling, live streaming, and expandable source citations.

## Benchmark Results (IIEST UG Ordinance)

Evaluated via `python -m eval.benchmark --llm` on an 8-query golden set (covering slang, tables, and multi-clause rules) plus out-of-scope hallucination traps:

| Pipeline Mode   | Hit@4 (%) |       MRR | Avg Retrieval Latency (ms) |
| --------------- | --------: | --------: | -------------------------: |
| `vector`        |    100.0% |     0.760 |                  166.06 ms |
| `hybrid`        |    100.0% | **0.875** |                  170.58 ms |
| `hybrid_rerank` |    100.0% |     0.750 |                  470.58 ms |

- **Hallucination Rejection Rate:** `100.0%`
- **Cache Hit Latency (SQLite):** `< 1 ms`
- **Engineering Takeaway:** `hybrid` (Vector + Regex BM25 + RRF) achieved the highest MRR (`0.875`) at `170ms`—outperforming the general-purpose 4MB `TinyBERT` reranker on domain-specific academic policy text while cutting latency by ~64%.

## Quickstart

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env  # Add your GEMINI_API_KEY
```
