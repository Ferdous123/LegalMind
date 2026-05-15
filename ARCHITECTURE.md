# LegalMind Architecture

## System Overview

LegalMind is a document intelligence platform for legal workflows. It processes messy legal documents (scanned, handwritten, poorly formatted), extracts structured information, retrieves relevant evidence, generates grounded draft outputs, and continuously improves through operator feedback.

## Design Principles

1. **Grounding over fluency** — Every generated claim must cite its source. Unsupported text is flagged, not hidden.
2. **Local-first inference** — All LLM inference runs on local GGUF models. No cloud API calls, no data leaves the machine.
3. **Progressive improvement** — The system learns from operator corrections at three timescales: immediate (exemplars), medium (rules), long-term (prompt evolution).
4. **Modular pipeline** — Each stage (ingest → extract → retrieve → draft → verify) is independently testable.
5. **Resource-aware** — GPU memory is managed via a singleton ModelManager that swaps models to fit within 10GB VRAM.

## Component Architecture

### 1. Document Ingestion (`code/pipeline/ingestion.py`)

**Input**: PDF, PNG, JPG, TIFF files (potentially messy, scanned, handwritten)

**Processing logic**:
- Detect file type
- For PDFs: attempt text extraction via pdfplumber first
  - If extracted text is sparse (<100 chars/page), treat as scanned → route to OCR
  - If text is rich, use directly
- For images: always route to OCR
- Output: raw text + page metadata + confidence flag (ocr_used: bool)

**Output**: `ProcessedDocument` dataclass stored as JSON in `data/processed/`

### 2. OCR Engine (`code/pipeline/ocr_engine.py`)

**Model**: LightOnOCR-1B (Q8_0 GGUF + mmproj)

**Design**:
- Page images rendered at 300 DPI via pypdfium2
- Each page processed independently
- Results concatenated with page boundaries preserved
- Confidence heuristic: if model returns many `[UNK]` tokens or empty output, flag page as "low_confidence"

### 3. Structured Extraction (`code/pipeline/structurer.py`)

**Model**: Qwen3-VL-8B (Q4_K_M GGUF)

**Design**:
- Takes processed text + document type hint
- Extracts structured fields based on draft type:
  - Case facts: parties, dates, claims, procedural_history, evidence_items, outcomes
  - Title review: property_description, ownership_chain, encumbrances, gaps, recording_info
  - Notice summary: notice_type, deadlines, requirements, response_actions, compliance_steps
  - Document checklist: required_docs, present_docs, missing_docs, verification_status
- Each field includes: `value`, `source_span` (verbatim text from source), `confidence`
- Uses prompt injection of relevant exemplars (from learning system)

### 4. Text Chunking (`code/pipeline/chunker.py`)

**Strategy**: Semantic chunking with overlap
- Target chunk size: 512 tokens
- Overlap: 64 tokens
- Boundary detection: paragraph breaks > sentence breaks > token count
- Each chunk retains: `document_id`, `page_number`, `char_offset_start`, `char_offset_end`

### 5. Retrieval Layer (`code/retrieval/`)

**Vector Store**: ChromaDB (local, persistent)

**Indexing** (`indexer.py`):
- Embed chunks using BGE-M3 (sentence-transformers)
- Store in ChromaDB collection per document
- Metadata: document_id, page, offset, chunk_text

**Search** (`searcher.py`):
- Hybrid: semantic (cosine similarity) + keyword (BM25-style metadata filter)
- Returns top-K chunks with similarity scores
- Configurable: K=5 default, adjustable per draft type

**Evidence packaging** (`evidence.py`):
- Takes retrieval results + draft query
- Formats as numbered evidence list for prompt injection
- Each evidence item: `[E1] "verbatim text" (page X, chars Y-Z)`
- Enables downstream citation: draft can reference `[E1]` etc.

### 6. Draft Generation (`code/generation/`)

**Model**: Qwen3-VL-8B (same as extraction, already loaded)

**Prompt structure**:
```
[System prompt: base + learned rules]
[Few-shot exemplars from correction bank]
[Evidence block: [E1]...[EN]]
[Task instruction: generate {draft_type} from evidence]
```

**Grounding enforcement** (`grounding.py`):
- Post-generation: scan output for claims
- For each claim, verify it has a citation `[EX]` reference
- Claims without citations → flagged as "unsupported"
- Unsupported claims get a visual warning in the UI

**Output**: Markdown-formatted draft with inline citations + structured JSON

### 7. Verification Firewall (`code/firewall/`)

**Three layers**:
1. **Source Anchor** (`source_anchor.py`): Verify that cited evidence actually supports the claim (semantic similarity > threshold)
2. **Confidence Score** (`confidence.py`): Logprob-based scoring where available; fallback to embedding similarity between claim and evidence
3. **Runner** (`runner.py`): Orchestrates layers, produces per-field verification status

**Status values**: `verified`, `uncertain`, `unsupported`, `manual_review`

### 8. Learning System (`code/learning/`)

**Three-layer hybrid architecture**:

**Layer 1 — Exemplar Bank** (`correction_store.py`, `exemplar_retriever.py`):
- Stores every operator correction: {source_ocr, generated_text, edited_text, field_type, draft_type}
- On generation: retrieve top-3 most similar corrections (by BGE-M3 embedding of source_ocr)
- Inject as few-shot examples in the prompt
- Active exemplars capped at 50; excess archived

**Layer 2 — Pattern Extraction** (`pattern_extractor.py`):
- Triggered every 20 new corrections
- Uses Gemma-4-E4B to analyze correction clusters
- Outputs reusable rules: "Always include [pattern]", "Never omit [pattern]"
- Rules stored in `config/learned_rules.yaml`

**Layer 3 — Prompt Consolidation** (`prompt_consolidator.py`):
- Triggered every 50 new corrections
- Folds accumulated rules into the base system prompt
- Archives exemplars now covered by the updated prompt
- Keeps the system lean and generalizable over time

### 9. Web Application (`webapp/`)

**Backend**: FastAPI with Jinja2 SSR

**Pages**:
- `/` — Dashboard (overview stats, recent activity)
- `/documents` — Upload + document list with processing status
- `/documents/{id}/draft` — Draft viewer with inline edit
- `/audit` — Learning dashboard: correction history, extracted rules, improvement metrics
- `/settings` — Model status, system config
- `/api/v1/...` — RESTful API for programmatic access

**Real-time**: SSE for pipeline progress updates during document processing

**Edit flow**:
1. Draft rendered with editable field components
2. Click field → textarea appears with current value
3. Focus → "Update" button slides in
4. Submit → POST to `/api/v1/corrections` → exemplar stored → toast confirmation
5. Field shows "edited" badge; correction immediately available for future generations

## Data Flow

```
Upload PDF/Image
       │
       ▼
┌─────────────────┐
│ ingestion.py    │──── Is text extractable? ────┐
└─────────────────┘                              │
       │ no (scanned)                    yes     │
       ▼                                         ▼
┌─────────────────┐                    ┌─────────────────┐
│ ocr_engine.py   │                    │ text_extractor   │
└─────────────────┘                    └─────────────────┘
       │                                         │
       └──────────────────┬──────────────────────┘
                          ▼
                ┌─────────────────┐
                │ structurer.py   │ ← exemplars injected
                └─────────────────┘
                          │
                          ▼
                ┌─────────────────┐
                │ chunker.py      │
                └─────────────────┘
                          │
                          ▼
                ┌─────────────────┐
                │ indexer.py      │ → ChromaDB
                └─────────────────┘
                          │
                          ▼
                ┌─────────────────┐
                │ searcher.py     │ ← query from draft type
                └─────────────────┘
                          │
                          ▼
                ┌─────────────────┐
                │ drafter.py      │ ← evidence + exemplars + rules
                └─────────────────┘
                          │
                          ▼
                ┌─────────────────┐
                │ firewall/       │ → verified/uncertain/unsupported
                └─────────────────┘
                          │
                          ▼
                ┌─────────────────┐
                │ webapp (UI)     │ → operator reviews & edits
                └─────────────────┘
                          │
                          ▼
                ┌─────────────────┐
                │ learning/       │ → exemplars → rules → prompt
                └─────────────────┘
```

## Scalability Considerations

- **Multiple documents**: Process in queue; pipeline is per-document, parallelizable at document level
- **Growing correction bank**: Layer 3 consolidation prevents unbounded growth
- **New draft types**: Add a prompt template + field schema; no code changes needed
- **Alternative models**: Swap GGUF files in config/models.yaml; interface layer abstracts model specifics
