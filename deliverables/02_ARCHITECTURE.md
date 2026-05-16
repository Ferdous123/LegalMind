# LegalMind — Architecture Overview

A document-intelligence pipeline that ingests messy legal documents, extracts
structured data, retrieves grounded evidence, generates cited drafts, and
improves from operator edits. Runs entirely on local GGUF models (no cloud).

## Design principles

1. **Grounding over fluency.** Every factual claim in a draft carries a
   `[E1]`-style citation; a verification firewall scores each claim against
   the cited evidence. Unsupported text is flagged, not hidden.
2. **Local-first.** All inference is local llama-cpp GGUF. No data leaves the
   machine; no API keys.
3. **Progressive improvement.** Operator corrections improve future drafts at
   three timescales: immediate (exemplars), medium (extracted rules),
   long-term (prompt consolidation).
4. **Resource-aware & crash-isolated.** One RTX 3080 (10 GB). A singleton
   `ModelManager` loads one large model at a time. Every heavy GPU job
   (draft generation, rule extraction) runs in a **short-lived subprocess**
   so a llama-cpp CUDA fault can never take the web server down.
5. **Modular & independently testable.** Each pipeline stage is its own
   module under `code/`.

## Data flow

```
Upload (PDF / PNG / JPG / TIFF / TXT)
  │
  ▼
ingestion.py ── text-extractable? ──► pdfplumber (rich text)
  │                       │
  │  sparse / scanned     └────────── per-page text + page_spans
  ▼
ocr_engine.py  (LightOnOCR-1B, page images 150–300 DPI)
  │
  ▼
structurer.py  (Qwen3-VL-8B) ── schema-shaped fields  ◄── exemplars injected
  │            (cascade + single-pass fallback; _ensure_schema_shape guards
  │             against malformed JSON on heavily degraded scans)
  ▼
chunker.py  (512-tok target, 64 overlap) ── page-aware: each chunk records
  │          its TRUE source page via page_spans (not a char//3000 estimate)
  ▼
indexer.py  ──► ChromaDB (bge-small embeddings, CPU) + chunk store
  │
  ▼
searcher.py  hybrid: ChromaDB semantic ⊕ BM25 keyword → Reciprocal Rank Fusion
  │
  ▼
evidence.py  packages top-K chunks as [E1]…[EN] with page + char span
  │
  ▼
drafter.py  (Qwen3-VL-8B, in subprocess)
  prompt = system_base + learned_rules + top-3 exemplars + [E] evidence + task
  │
  ▼
firewall/runner.py   verifies (a) every [E]-cited markdown claim against its
  │                  cited evidence, (b) structured fields; skips _cascade_meta
  ▼
webapp (Jinja2 + HTMX)  draft viewer: rendered markdown, clickable
  │                     verification badge, citation eye → source page/text
  ▼
operator edits (WYSIWYG, in-place contenteditable)
  │
  ▼
learning/   Layer1 exemplars (BM25 over correction source) ──► next draft
            Layer2 pattern_extractor (every 20 corr, or forced) ──► rules
            Layer3 prompt_consolidator (every 50) ──► base prompt
            (extraction runs in a subprocess — never crashes the server)
```

## Components

| Module | Responsibility | Model |
|---|---|---|
| `code/pipeline/ingestion.py` | File-type routing, page extraction, builds `full_text` + `page_spans` | — |
| `code/pipeline/ocr_engine.py` | Scanned/handwritten OCR; graceful empty-result fallback | LightOnOCR-1B Q8 |
| `code/pipeline/structurer.py` | Draft-type-specific structured fields; schema-shape guard | Qwen3-VL-8B Q4 |
| `code/pipeline/chunker.py` | Semantic chunking; page-accurate `page_number` from `page_spans` | — |
| `code/retrieval/indexer.py` | ChromaDB persistent index (bge-small, CPU) | — |
| `code/retrieval/searcher.py` | Hybrid semantic+BM25, RRF fusion | — |
| `code/retrieval/evidence.py` | `[E1]…` citation map (text, page, char span) | — |
| `code/generation/drafter.py` | Assembles grounded prompt, generates draft | Qwen3-VL-8B Q4 |
| `code/generation/grounding.py` | n-gram (3) + token-containment claim verification | — |
| `code/firewall/runner.py` | Per-claim + per-field verification, summary | — |
| `code/learning/correction_store.py` | JSONL correction persistence | — |
| `code/learning/exemplar_retriever.py` | BM25 retrieval of similar past corrections | — |
| `code/learning/pattern_extractor.py` | Clusters corrections → reusable rules | Gemma-4-E4B Q4 |
| `code/learning/prompt_consolidator.py` | Folds rules into base prompt | Gemma-4-E4B Q4 |
| `code/llm_interface/model_manager.py` | Singleton; one large model at a time; `reclear_gpu()` | — |
| `webapp/main.py` | FastAPI routes, SSE, job queue, subprocess orchestration | — |

## Reliability design (learned the hard way, documented in `docs/decisions.md`)

- **Subprocess isolation.** `drafter` and `pattern_extractor` load multi-GB
  models. Doing that in the web process contends with the server's own GPU
  allocation and, on Windows, llama-cpp's CUDA context deadlocks/OOM-kills the
  worker — taking the whole server down. Both now run in a tempfile runner
  subprocess with stdout/stderr to a log; the OS reclaims the CUDA context on
  exit. `reclear_gpu()` is called before each launch.
- **Schema-shape guard.** Heavily degraded scans can make the extraction LLM
  emit malformed JSON. `structurer._ensure_schema_shape()` guarantees every
  draft-type's top-level keys exist so no downstream consumer KeyErrors.
- **Queue persistence.** Job queue state is snapshotted to disk so an
  in-flight job survives a server restart (marked failed with a clear reason
  rather than silently lost).
- **Idempotent corpus processing.** `scripts/process_corpus.py` is
  single-process, strictly sequential, skip-if-done, filesystem-verified —
  resumable after any interruption.

## Scalability

- Per-document pipeline; document-level parallelism is the natural scale-out
  axis (the subprocess model already isolates GPU work).
- Layer-3 prompt consolidation bounds correction-bank growth.
- New draft type = a prompt template + a field schema; no code change.
- Models are swappable via `config/models.yaml`; the inference interface
  abstracts the backend.
