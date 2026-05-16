# Architecture

LegalMind is a sequential pipeline that turns a messy legal document into
a grounded, cited draft an operator can edit. Operator edits feed back into
a three-layer learning system that improves future drafts on similar
documents. Everything runs on local GGUF models; nothing leaves the
machine.

## Design principles

1. **Grounded over fluent.** Every factual claim in a draft carries a
   `[E1]`-style citation. A deterministic verifier scores each claim
   against the cited evidence. Unsupported claims are flagged, not hidden.
2. **Local-only inference.** All models run through llama-cpp on GGUF
   weights. No API keys, no external services.
3. **Progressive learning.** Operator corrections affect the next similar
   document immediately (few-shot exemplars), generalize into rules at the
   batch level, and consolidate into the base prompt over time.
4. **Crash-isolated GPU work.** A single 10 GB GPU runs one large model at
   a time. Draft generation and rule extraction each run in a short-lived
   subprocess so a llama-cpp CUDA fault cannot take the server down.
5. **Modular and inspectable.** Each pipeline stage is its own module; the
   verifier is lexical and reproducible by hand.

## Data flow

```
Upload (PDF / PNG / JPG / TIFF / TXT)
  │
  ▼
ingestion.py ── text-extractable? ──► pdfplumber (rich text)
  │                       │
  │  sparse / scanned     └──── per-page text + page_spans
  ▼
ocr_engine.py  (LightOnOCR-1B, page images 150–300 DPI)
  │
  ▼
structurer.py  (Qwen3-VL-8B) ── schema-shaped fields  ◄── exemplars injected
  │            (cascade + single-pass fallback;
  │             _ensure_schema_shape() guards against malformed JSON on
  │             heavily degraded scans)
  ▼
chunker.py  (512-token target, 64 overlap) ── page-aware: each chunk
  │          records its real source page via page_spans
  ▼
indexer.py  ──► ChromaDB (bge-small embeddings, CPU) + chunk store
  │
  ▼
searcher.py  hybrid: ChromaDB semantic ⊕ BM25 ──► Reciprocal Rank Fusion
  │
  ▼
evidence.py  packages top-K chunks as [E1]…[EN] with page + char span
  │
  ▼
drafter.py  (Qwen3-VL-8B, in subprocess)
  prompt = system_base + learned_rules + top-3 exemplars + [E] evidence + task
  │
  ▼
firewall/runner.py  verifies (a) every [E]-cited markdown claim against its
  │                  cited evidence, (b) structured fields; skips _cascade_meta
  ▼
webapp (Jinja2 + HTMX)
  draft viewer: rendered markdown, clickable verification badge,
  citation "eye" → source page image or highlighted text span,
  per-claim operator-override → verified (with provenance recorded)
  │
  ▼
operator edits  (WYSIWYG in-place editing)
  │
  ▼
learning/
  Layer 1: exemplars        BM25 over correction source text → next draft
  Layer 2: pattern extractor every 20 corrections (or forced) → rules.yaml
  Layer 3: prompt consolidator every 50 → base prompt; old exemplars archived
  (Layer 2 and 3 each run in their own subprocess.)
```

## Components

| Module | Responsibility | Model |
|---|---|---|
| `code/pipeline/ingestion.py` | File-type routing, page extraction, builds `full_text` + `page_spans` | — |
| `code/pipeline/ocr_engine.py` | Scanned/handwritten OCR; graceful empty-result handling | LightOnOCR-1B Q8 |
| `code/pipeline/structurer.py` | Draft-type-specific field extraction; schema-shape guard | Qwen3-VL-8B Q4 |
| `code/pipeline/chunker.py` | Semantic chunking; page-accurate page numbers via `page_spans` | — |
| `code/retrieval/indexer.py` | Persistent ChromaDB index, bge-small on CPU | — |
| `code/retrieval/searcher.py` | Hybrid semantic + BM25 with RRF fusion | — |
| `code/retrieval/evidence.py` | `[E1]…` citation map (text, page, char range) | — |
| `code/generation/drafter.py` | Prompt assembly and draft generation | Qwen3-VL-8B Q4 |
| `code/generation/grounding.py` | 3-gram and token-containment claim verification | — |
| `code/firewall/runner.py` | Per-claim + per-field verification, summary | — |
| `code/learning/correction_store.py` | JSONL persistence of operator corrections | — |
| `code/learning/exemplar_retriever.py` | BM25 retrieval of similar past corrections | — |
| `code/learning/pattern_extractor.py` | Clusters corrections into reusable rules | Gemma-4-E4B Q4 |
| `code/learning/prompt_consolidator.py` | Folds rules into the base prompt | Gemma-4-E4B Q4 |
| `code/llm_interface/model_manager.py` | Singleton; one large model at a time; `reclear_gpu()` | — |
| `webapp/main.py` | FastAPI routes, SSE, job queue, subprocess orchestration | — |

## Reliability design

A few things took real failures during development to get right:

- **Subprocess isolation for heavy GPU jobs.** Loading multi-GB models
  inside the live web process contends with the server's own GPU
  allocation and, on Windows, llama-cpp's CUDA context can deadlock or
  OOM-kill the worker. Draft generation and pattern extraction both run
  in short-lived subprocesses. The OS reclaims the CUDA context on exit,
  and `reclear_gpu()` is called before each launch.
- **Schema-shape guard.** A heavily degraded scan can make the extraction
  LLM emit malformed JSON. `structurer._ensure_schema_shape()` guarantees
  the top-level keys for every draft type exist so no downstream consumer
  KeyErrors on a sparse extraction.
- **Per-document GPU reset on upload.** The OCR model and the structured
  extraction model are different; under GPU pressure the swap inside a
  single upload can fail. Every upload begins with `reclear_gpu()` so
  documents are independent.
- **Persistent job queue.** Queue state is snapshotted on every mutation;
  an in-flight job that survives a restart is marked failed with a clear
  reason instead of silently lost.
- **Page-accurate citations.** Chunks record their real source page from
  per-page text spans (not a `char_start // 3000` estimate), so the
  citation "eye" opens the correct page image on multi-page PDFs.

## Scalability

- Per-document pipeline; document-level parallelism is the natural
  scale-out axis (the subprocess model already isolates GPU work).
- Layer-3 prompt consolidation bounds correction-bank growth.
- Adding a new draft type means a new prompt template and a field schema;
  no code changes.
- Models swap via `config/models.yaml`; the inference interface abstracts
  the backend.
