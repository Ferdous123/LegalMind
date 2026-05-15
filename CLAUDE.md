# LegalMind — AI-Powered Legal Document Intelligence Platform

## Project Identity

**Name**: LegalMind  
**Purpose**: Internal workflow for Pearson Specter Litt — ingests messy legal documents, extracts structured data, retrieves evidence, generates grounded drafts, and improves from operator edits.  
**Target**: 100/100 on assessment rubric + all optional deliverables.

---

## Development Workflow (MANDATORY)

**Use the superpowers plugin skills for ALL implementation work:**

1. **Before starting any phase** → invoke `superpowers:writing-plans` if the phase needs decomposition
2. **For implementation** → invoke `superpowers:subagent-driven-development` — dispatch implementer subagent per task, then spec reviewer, then code quality reviewer
3. **Before any refactoring** → use `gitnexus-impact-analysis` skill to check blast radius
4. **Before marking any phase done** → invoke `superpowers:verification-before-completion`
5. **Before committing** → invoke `superpowers:finishing-a-development-branch`

**MCP Tools Available:**
- `gitnexus_impact({target, direction})` — check what depends on / what is depended on
- `gitnexus_detect_changes()` — map staged git changes to affected code flows
- `mempalace` — semantic memory for project context and decisions

**Run `npx gitnexus analyze` once at the start of each session** to index the codebase for dependency tracking.

---

## Assessment Compliance Checklist (PDF is divine)

### Required Deliverables
- [x] Source code
- [x] README with setup and run instructions
- [x] Short architecture overview
- [x] Brief write-up of assumptions and tradeoffs
- [x] Sample inputs and outputs
- [x] Evaluation approach and results

### Optional Deliverables (ALL must be done)
- [x] API endpoints (FastAPI)
- [x] Simple UI (full web app)
- [x] Tests (pytest suite)
- [x] Docker setup (Dockerfile + compose)

### Rubric Targets (100 points)
1. **Document Processing (25pts)**: OCR + text extraction over scanned/noisy files, reasonable handling of partially unclear inputs, extracted text + structured data usable downstream
2. **Retrieval & Grounding (25pts)**: Surface relevant passages, feed evidence into generation, inspect which evidence supported which output, control unsupported generation
3. **Draft Quality (10pts)**: Useful, clear, structured, consistent with source documents
4. **Improvement from Edits (25pts)**: Capture edits, learn reusable patterns, future outputs improve meaningfully
5. **Code Quality & System Design (10pts)**: Organization, maintainability, modularity, error handling, scalability
6. **Documentation & Clarity (5pts)**: Ease of understanding, setup clarity, reviewer experience

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        LegalMind Platform                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐            │
│  │ Document  │───▶│  Extraction  │───▶│  Structured │            │
│  │ Ingestion │    │   Pipeline   │    │    Store    │            │
│  └──────────┘    └──────────────┘    └─────────────┘            │
│       │                                      │                    │
│       │ OCR/PDF                              │ indexed            │
│       ▼                                      ▼                    │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐            │
│  │   Text   │    │   Evidence   │◀───│   Vector    │            │
│  │  Chunks  │───▶│  Retrieval   │    │    Index    │            │
│  └──────────┘    └──────────────┘    └─────────────┘            │
│                         │                                         │
│                         │ grounded context                        │
│                         ▼                                         │
│  ┌──────────────────────────────────────────┐                    │
│  │         Draft Generation Engine           │                    │
│  │  (4 output types: case_fact, title_review,│                    │
│  │   notice_summary, document_checklist)      │                    │
│  └──────────────────────────────────────────┘                    │
│                         │                                         │
│                         ▼                                         │
│  ┌──────────────────────────────────────────┐                    │
│  │         Operator Review Panel             │                    │
│  │  (click-to-edit → save → learn loop)      │                    │
│  └──────────────────────────────────────────┘                    │
│                         │                                         │
│                         ▼                                         │
│  ┌──────────────────────────────────────────┐                    │
│  │       Improvement Engine (Hybrid D)       │                    │
│  │  - Exemplar bank (immediate few-shot)     │                    │
│  │  - Pattern extraction (every N edits)     │                    │
│  │  - Prompt consolidation (rules → prompt)  │                    │
│  └──────────────────────────────────────────┘                    │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Language | Python 3.11+ | Assessment standard |
| Web framework | FastAPI + Uvicorn | Async, fast, auto-docs |
| Frontend | Jinja2 SSR + Tailwind CSS + HTMX | No build step, professional |
| Local LLMs | llama-cpp-python (GGUF) | Proven, GPU-native |
| Models | Qwen3-VL-8B (Q4), LightOnOCR-1B (Q8), Gemma-4-E4B (Q4) | Local only, no API calls |
| Vector Store | ChromaDB | Simple, powerful, MemPalace-compatible |
| Embeddings | sentence-transformers BGE-M3 | Multilingual, high quality |
| PDF Processing | pdfplumber + pypdfium2 | Free, robust |
| Memory/Context | MemPalace (MCP) | Token-efficient retrieval, semantic search |
| Testing | pytest + httpx | Standard |
| Containerization | Docker + docker-compose | Assessment optional |

---

## Model Configuration

All models are stored at `F:/Research_Paper_Projects/LLMs/` (flat directory):

```yaml
equation_ocr:
  gguf_model: LightOnOCR-1B-1025-Q8_0.gguf
  gguf_mmproj: mmproj-LightOnOCR-1B-1025-Q8_0.gguf
  vram_gb: 1.5
  role: OCR for scanned documents, handwritten notes, low-res PDFs

structured_extraction:
  gguf_model: Qwen3VL-8B-Instruct-Q4_K_M.gguf
  gguf_mmproj: mmproj-Qwen3VL-8B-Instruct-F16.gguf
  vram_gb: 6.1
  role: Structured field extraction, draft generation

symbolic_reasoning:
  gguf_model: gemma-4-E4B-it-Q4_K_M.gguf
  vram_gb: 3.5
  role: Verification, pattern extraction from edits, rule consolidation

embeddings:
  model: BAAI/bge-m3
  backend: sentence_transformers
  device: cuda
```

**GPU constraint**: RTX 3080 10GB. Only ONE large model loaded at a time. Model swapping managed by ModelManager singleton.

---

## The Four Draft Output Types

1. **Case Fact Summary** — structured extraction of parties, dates, claims, evidence, outcomes from legal documents
2. **Title Review Summary** — property/title chain analysis showing ownership history, encumbrances, gaps
3. **Notice-Related Summary** — deadline-focused extraction of notice requirements, compliance steps, responses needed
4. **Document Checklist** — completeness audit listing present/missing documents, verification status per item

Each draft type has:
- A dedicated prompt template in `prompts/`
- Grounding citations back to source text
- Structured JSON output + human-readable markdown

---

## Improvement from Edits — Hybrid D Architecture

### The Three-Layer Learning System

**Layer 1: Exemplar Bank (Immediate)**
- Operator clicks a field → edits text → clicks "Update"
- System stores: `{source_ocr, generated_text, edited_text, field_type, timestamp}`
- On next generation, top-K similar exemplars are injected into prompt as few-shot examples
- Effect: Immediate improvement on similar documents

**Layer 2: Pattern Extraction (Every 20 corrections)**
- Background job clusters similar corrections
- Uses Gemma-4-E4B to analyze patterns: "These 8 edits all add party names that were in the source but omitted"
- Extracts reusable RULES: `"Always include all named parties from source text"`
- Rules stored in `config/learned_rules.yaml`

**Layer 3: Prompt Consolidation (Every 50 corrections)**
- Rules from Layer 2 are folded into the base system prompt
- Old exemplars now covered by prompt rules are ARCHIVED (not deleted)
- Keeps exemplar bank lean (<50 active entries)
- Net effect: System generalizes instead of memorizing

### Storage Schema (data/corrections/)
```json
{
  "id": "corr_001",
  "timestamp": "2026-05-15T10:30:00Z",
  "document_id": "doc_upload_xyz",
  "field_path": "parties.defendant",
  "draft_type": "case_fact_summary",
  "source_ocr_chunk": "...the defendant, Marcus Bell, filed...",
  "generated_text": "Defendant: [Not clearly identified]",
  "edited_text": "Defendant: Marcus Bell",
  "correction_type": "omission",
  "exemplar_active": true
}
```

---

## UI/UX Design Spec

### Layout: Three-Column Professional Legal-Tech
- **Left panel** (250px): Document list + upload area
- **Center panel** (flex): Draft output with inline-editable fields
- **Right panel** (350px): Evidence citations + source highlights

### Color Palette (Light mode primary, dark toggle)
- Background: `#f8fafc` (slate-50)
- Surface: `#ffffff`
- Primary: `#1e40af` (blue-800)
- Secondary: `#64748b` (slate-500)
- Accent/verified: `#059669` (emerald-600)
- Warning/uncertain: `#d97706` (amber-600)
- Error/unsupported: `#dc2626` (red-600)
- Text: `#1e293b` (slate-800)

### Navigation: Top horizontal bar
- Logo (left) + nav links (center) + user/settings (right)
- Pages: Dashboard | Documents | Drafts | Audit & Learning | Settings

### Operator Edit Flow
1. Operator views generated draft in center panel
2. Each field/section has a subtle edit icon on hover
3. Click → field becomes editable text input
4. "Update" button appears below the field
5. Click Update → correction saved → toast notification "Correction saved. Future drafts will improve."
6. Field shows a small "edited" badge
7. No re-generation of current document (corrections improve FUTURE documents only)

### Key UX Principles
- Zero AI aesthetic (no sparkles, no "magic" language)
- Professional legal-tech look (like Casetext, Relativity, NetDocuments)
- Clear source attribution for every generated claim
- Confidence indicators on extracted fields
- Responsive but not mobile-first (desktop workflow)

---

## Project Structure

```
LegalMind/
├── CLAUDE.md                 # THIS FILE — master instructions
├── ARCHITECTURE.md           # Detailed system design
├── HANDOFF.md               # Implementation sequence for Sonnet
├── README.md                # Setup + run instructions (assessment deliverable)
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container setup
├── docker-compose.yml       # Full stack compose
├── .mcp.json               # MemPalace MCP config
│
├── config/
│   ├── models.yaml          # Model paths and settings
│   ├── learned_rules.yaml   # Auto-generated rules from corrections
│   └── paths.py             # Filesystem path constants
│
├── code/
│   ├── __init__.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── ingestion.py     # PDF/image upload + preprocessing
│   │   ├── ocr_engine.py    # LightOnOCR integration
│   │   ├── text_extractor.py # pdfplumber text extraction
│   │   ├── structurer.py    # Qwen3-VL structured field extraction
│   │   └── chunker.py       # Text chunking for retrieval
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── indexer.py       # ChromaDB indexing
│   │   ├── searcher.py      # Semantic + keyword hybrid search
│   │   └── evidence.py      # Evidence packaging for generation
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── drafter.py       # Core draft generation engine
│   │   ├── grounding.py     # Citation injection + hallucination control
│   │   └── templates.py     # Output format templates per draft type
│   │
│   ├── learning/
│   │   ├── __init__.py
│   │   ├── correction_store.py  # Exemplar bank CRUD
│   │   ├── pattern_extractor.py # Layer 2: cluster + extract rules
│   │   ├── prompt_consolidator.py # Layer 3: rules → prompt rewrite
│   │   └── exemplar_retriever.py  # Semantic lookup of similar corrections
│   │
│   ├── llm_interface/
│   │   ├── __init__.py
│   │   ├── model_manager.py  # Singleton: load/unload/swap models
│   │   ├── inference.py      # Unified inference API
│   │   └── gpu_guard.py      # VRAM monitoring + OOM prevention
│   │
│   └── firewall/
│       ├── __init__.py
│       ├── source_anchor.py  # Verify claims against source text
│       ├── confidence.py     # Logprob-based confidence scoring
│       └── runner.py         # Orchestrate verification layers
│
├── prompts/
│   ├── case_fact_summary.txt
│   ├── title_review_summary.txt
│   ├── notice_summary.txt
│   ├── document_checklist.txt
│   └── system_base.txt       # Base system prompt (Layer 3 appends here)
│
├── webapp/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, routes, SSE
│   ├── static/
│   │   ├── css/
│   │   │   └── app.css      # Tailwind utilities + custom styles
│   │   ├── js/
│   │   │   ├── app.js       # Core UI logic
│   │   │   ├── editor.js    # Inline edit + correction submission
│   │   │   └── evidence.js  # Source highlighting + citation panel
│   │   └── icons/
│   │       └── logo.svg     # LegalMind logo
│   └── templates/
│       ├── base.html        # Layout shell
│       ├── dashboard.html   # Overview page
│       ├── documents.html   # Upload + document list
│       ├── drafts.html      # Draft viewer + editor
│       ├── audit.html       # Learning dashboard + correction history
│       └── settings.html    # System config
│
├── data/
│   ├── uploads/             # Raw uploaded documents
│   ├── processed/           # Extracted text + structured JSON
│   ├── corrections/         # Exemplar bank storage
│   └── sample/              # Sample inputs for assessment demo
│
├── exemplars/               # Active few-shot examples by field type
│   ├── parties.jsonl
│   ├── dates.jsonl
│   ├── claims.jsonl
│   ├── evidence_refs.jsonl
│   └── outcomes.jsonl
│
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_ocr_engine.py
│   │   ├── test_structurer.py
│   │   ├── test_retrieval.py
│   │   ├── test_drafter.py
│   │   └── test_learning.py
│   ├── integration/
│   │   ├── test_pipeline_e2e.py
│   │   ├── test_api_endpoints.py
│   │   └── test_correction_loop.py
│   └── fixtures/
│       ├── sample_legal_doc.pdf
│       ├── sample_scanned.png
│       └── expected_outputs/
│
├── scripts/
│   ├── bootstrap.py         # First-run setup (venv, models, deps)
│   ├── seed_sample_data.py  # Generate sample docs for demo
│   └── run_evaluation.py    # Automated evaluation script
│
└── docs/
    ├── architecture.md      # Detailed architecture (assessment deliverable)
    ├── assumptions.md       # Assumptions and tradeoffs write-up
    └── evaluation.md        # Evaluation approach and results
```

---

## Implementation Sequence (for Sonnet)

### Phase 1: Foundation (Do First)
1. `config/models.yaml` + `config/paths.py` — file paths and model config
2. `code/llm_interface/model_manager.py` — model loading singleton
3. `code/llm_interface/inference.py` — unified inference wrapper
4. `code/llm_interface/gpu_guard.py` — VRAM checks

### Phase 2: Document Processing Pipeline
5. `code/pipeline/text_extractor.py` — pdfplumber text extraction
6. `code/pipeline/ocr_engine.py` — LightOnOCR for scanned pages
7. `code/pipeline/ingestion.py` — upload handler + routing (text vs scan)
8. `code/pipeline/structurer.py` — Qwen3-VL structured extraction
9. `code/pipeline/chunker.py` — text chunking for retrieval index

### Phase 3: Retrieval & Grounding
10. `code/retrieval/indexer.py` — ChromaDB index builder
11. `code/retrieval/searcher.py` — hybrid semantic + keyword search
12. `code/retrieval/evidence.py` — package evidence for generation

### Phase 4: Draft Generation
13. `prompts/*.txt` — all 4 draft type prompts
14. `code/generation/drafter.py` — core generation engine
15. `code/generation/grounding.py` — citation + hallucination control
16. `code/generation/templates.py` — output formatting

### Phase 5: Learning from Edits
17. `code/learning/correction_store.py` — exemplar bank CRUD
18. `code/learning/exemplar_retriever.py` — semantic retrieval of similar corrections
19. `code/learning/pattern_extractor.py` — rule extraction from correction clusters
20. `code/learning/prompt_consolidator.py` — fold rules into base prompt

### Phase 6: Verification Firewall
21. `code/firewall/source_anchor.py` — verify claims against source
22. `code/firewall/confidence.py` — logprob confidence scoring
23. `code/firewall/runner.py` — orchestration

### Phase 7: Web Application
24. `webapp/main.py` — FastAPI routes + API endpoints
25. `webapp/templates/*.html` — all pages
26. `webapp/static/` — CSS, JS, logo

### Phase 8: Testing & Documentation
27. `tests/` — full test suite
28. `README.md` — setup/run instructions
29. `docs/` — architecture, assumptions, evaluation
30. `Dockerfile` + `docker-compose.yml`

### Phase 9: Sample Data & Evaluation
31. `scripts/seed_sample_data.py` — create demo documents
32. `scripts/run_evaluation.py` — automated eval
33. `data/sample/` — sample inputs + expected outputs

---

## Critical Rules for Implementation

### DO:
- Keep all generated claims grounded with source citations
- Store source_ocr + generated_text + edited_text for every correction
- Use model swapping (one large model at a time on GPU)
- Make exemplar retrieval semantic (BGE-M3 embeddings), not keyword
- Include confidence scores on every extracted field
- Show which evidence supports which draft section
- Use proper error handling with graceful degradation
- Write clean, modular, well-tested code
- Include type hints throughout

### DO NOT:
- Never reference "TRACE", "trace", or any related naming
- Never use online APIs (all inference is local)
- Never skip the grounding step — every claim needs a source
- Never bloat exemplar bank past 50 active entries (archive old ones)
- Never load two large models simultaneously (GPU OOM)
- Never generate text without feeding retrieved evidence
- Never hard-code file paths (use config/paths.py)

### NAMING CONVENTIONS:
- Project: LegalMind (always capitalized)
- Modules: snake_case
- Classes: PascalCase
- Functions: snake_case
- Constants: UPPER_SNAKE_CASE
- No emojis in code or UI
- Professional terminology: "operator" not "user", "correction" not "fix"

---

## Environment Setup

```bash
# Working directory
F:/Research_Paper_Projects/LegalMind/

# Model files location (flat directory, no subdirectories needed)
F:/Research_Paper_Projects/LLMs/

# CUDA setup (Windows, RTX 3080)
# CUDA 12.6 installed at: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin

# Python 3.11+ required
# llama-cpp-python must be compiled with CUDA support (CMAKE_ARGS=-DGGML_CUDA=on)
```

---

## MemPalace Integration

MemPalace provides token-efficient context management:
- Corrections and learned rules stored as semantic memories
- Retrieval uses semantic search (not full-scan) — only relevant context injected
- Replaces naive "dump all exemplars" with smart top-K retrieval
- MCP server provides tool-call access for the learning loop

Config in `.mcp.json`:
```json
{
  "mcpServers": {
    "mempalace": {
      "type": "stdio",
      "command": "mempalace",
      "args": ["serve", "--palace", "./data/palace"]
    }
  }
}
```

---

## Evaluation Strategy

### Automated Metrics (scripts/run_evaluation.py)
1. **Extraction accuracy**: Compare extracted fields against ground-truth annotations
2. **Retrieval precision@K**: Are top-K retrieved passages actually relevant?
3. **Grounding score**: % of generated claims with valid source citations
4. **Edit distance reduction**: Do corrections reduce future edit distances for similar docs?
5. **Rule coverage**: Do extracted rules actually reduce correction frequency?

### Manual Demo Flow
1. Upload messy PDF → show extraction working
2. View draft → show evidence citations
3. Make corrections → show they're stored
4. Upload similar document → show improved draft (fewer corrections needed)
5. View audit panel → show learned rules and patterns

---

## Submission Checklist

- [ ] All code compiles and runs
- [ ] `docker-compose up` works end-to-end
- [ ] README has clear setup instructions
- [ ] Sample documents included in data/sample/
- [ ] Evaluation script produces results
- [ ] Architecture doc is clear and concise
- [ ] No references to any external proprietary software
- [ ] GitHub repo created and collaborators invited
