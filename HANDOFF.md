# LegalMind — Implementation Handoff

## Agentic Development Protocol

You are implementing LegalMind as a coordinated development team. **Use superpowers skills.**

### Session Startup (EVERY session)
```
1. npx gitnexus analyze          ← index codebase for dependency tracking
2. Read CLAUDE.md                 ← master spec, architecture, constraints
3. Read this file (HANDOFF.md)    ← find next incomplete phase
4. Invoke superpowers:subagent-driven-development for remaining phases
```

### Per-Phase Workflow (superpowers-driven)
1. **PLAN**: Read the phase spec below. Invoke `superpowers:writing-plans` if decomposition needed.
2. **IMPLEMENT**: Dispatch implementer subagent per task. Implementer writes code, tests, commits.
3. **SPEC REVIEW**: Dispatch spec reviewer subagent — confirms code matches phase spec exactly.
4. **QUALITY REVIEW**: Dispatch code quality reviewer — checks modularity, error handling, naming.
5. **VERIFY**: Run `python -c "import code.{module}"` + `pytest tests/` for the phase.
6. **COMMIT**: Clean commit with descriptive message after reviews pass.

### Before Refactoring Anything
```
gitnexus_impact({target: "ClassName.method", direction: "upstream"})
```
Check what breaks. If >5 affected symbols, plan carefully.

**Critical constraints**:
- Read CLAUDE.md FIRST — it has the full project spec, naming rules, and architecture.
- Read config/models.yaml — it has model file paths and GPU constraints.
- NEVER reference "TRACE" or any proprietary system in code, comments, or docs.
- ALL inference is LOCAL (GGUF models via llama-cpp-python). No API calls.
- Model cache directory: `F:/Research_Paper_Projects/LLMs/` (models already downloaded).
- CUDA path: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin`

---

## Phase 1: LLM Interface Layer

**Goal**: ModelManager singleton that loads/unloads GGUF models, respects 10GB VRAM limit.

**Files to create**:
- `code/__init__.py` (empty)
- `code/llm_interface/__init__.py`
- `code/llm_interface/model_manager.py`
- `code/llm_interface/inference.py`
- `code/llm_interface/gpu_guard.py`

**Spec for `model_manager.py`**:
```python
class ModelManager:
    """Singleton. Loads one GGUF model at a time via llama-cpp-python.
    
    Usage:
        mgr = ModelManager.instance()
        model = mgr.load("extraction")  # loads Qwen3-VL-8B
        response = mgr.generate(prompt, images=None, max_tokens=2048)
        mgr.unload()  # frees VRAM
        model = mgr.load("ocr")  # loads LightOnOCR
    
    The embeddings model (BGE-M3) is handled separately via sentence-transformers
    and can coexist with any single GGUF model.
    """
```

Key implementation details:
- Read config from `config/models.yaml` via yaml.safe_load
- Use `llama_cpp.Llama` for loading with appropriate `n_ctx`, `n_gpu_layers=-1` (all on GPU)
- For VL models (ocr, extraction): use `chat_handler="qwen25vl"` and pass `gguf_mmproj`
- `gpu_guard.py`: Before loading, check if enough VRAM is free. Use `torch.cuda.mem_get_info()` or parse `nvidia-smi` output.
- On Windows, add CUDA DLL dirs via `os.add_dll_directory()`

**Spec for `inference.py`**:
```python
class InferenceEngine:
    """Unified interface for all inference calls.
    
    Methods:
        generate_text(prompt, max_tokens, temperature) -> str
        generate_structured(prompt, schema: dict) -> dict  # JSON mode
        generate_with_image(prompt, image_path, max_tokens) -> str
        embed_text(text) -> list[float]
        embed_batch(texts: list[str]) -> list[list[float]]
    """
```

**Verification**: `python -c "from code.llm_interface import ModelManager, InferenceEngine; print('OK')"`

---

## Phase 2: Document Processing Pipeline

**Goal**: Accept PDFs and images, extract text (OCR if needed), produce structured JSON.

**Files to create**:
- `code/pipeline/__init__.py`
- `code/pipeline/ingestion.py`
- `code/pipeline/text_extractor.py`
- `code/pipeline/ocr_engine.py`
- `code/pipeline/structurer.py`
- `code/pipeline/chunker.py`

**Spec for `ingestion.py`**:
```python
class DocumentIngester:
    """Orchestrates document processing.
    
    Pipeline:
    1. Detect file type (PDF, image)
    2. For PDF: try pdfplumber text extraction
       - If sparse (<100 chars/page avg): route to OCR
       - If rich: use extracted text directly
    3. For images: always OCR
    4. Run structurer on extracted text
    5. Run chunker for retrieval indexing
    6. Save ProcessedDocument to data/processed/{doc_id}.json
    
    Returns: ProcessedDocument dataclass
    """

@dataclass
class ProcessedDocument:
    id: str
    filename: str
    upload_timestamp: str
    pages: list[PageContent]
    structured_fields: dict
    chunks: list[TextChunk]
    ocr_used: bool
    confidence: float  # 0.0-1.0 overall extraction confidence
```

**Spec for `ocr_engine.py`**:
```python
class OCREngine:
    """Wraps LightOnOCR-1B for page-level OCR.
    
    Input: image path (PNG/JPG) or PDF page rendered as image
    Output: extracted text + confidence score
    
    For PDFs: use pypdfium2 to render pages at 300 DPI before OCR.
    """
```

**Spec for `structurer.py`**:
```python
class DocumentStructurer:
    """Uses Qwen3-VL-8B to extract structured fields from text.
    
    Input: raw text + draft_type (case_fact|title_review|notice|checklist)
    Output: dict of structured fields with source_span + confidence per field
    
    IMPORTANT: Inject exemplars from the learning system into the prompt.
    Load exemplars via code.learning.exemplar_retriever.get_relevant_exemplars()
    """
```

**Spec for `chunker.py`**:
```python
class TextChunker:
    """Splits text into retrieval-ready chunks.
    
    Strategy: semantic chunking with overlap
    - Target: 512 tokens per chunk
    - Overlap: 64 tokens
    - Boundaries: paragraph > sentence > token
    - Each chunk: {text, document_id, page_number, char_start, char_end}
    """
```

**Verification**: Create a simple test PDF, run the pipeline, verify JSON output.

---

## Phase 3: Retrieval & Evidence

**Goal**: Index chunks in ChromaDB, retrieve relevant evidence for draft generation.

**Files to create**:
- `code/retrieval/__init__.py`
- `code/retrieval/indexer.py`
- `code/retrieval/searcher.py`
- `code/retrieval/evidence.py`

**Spec for `indexer.py`**:
```python
class DocumentIndexer:
    """Indexes document chunks into ChromaDB.
    
    - One ChromaDB collection: "legalmind_documents"
    - Each entry: embedding (BGE-M3), metadata (doc_id, page, offsets), document text
    - Idempotent: re-indexing same doc_id overwrites previous entries
    """
```

**Spec for `searcher.py`**:
```python
class EvidenceSearcher:
    """Retrieves relevant chunks for a given query.
    
    Methods:
        search(query: str, doc_ids: list[str], top_k: int = 5) -> list[EvidenceChunk]
        search_by_field(field_name: str, doc_ids: list[str]) -> list[EvidenceChunk]
    
    Returns ranked chunks with similarity scores.
    """
```

**Spec for `evidence.py`**:
```python
class EvidencePackager:
    """Formats retrieved evidence for prompt injection.
    
    Input: list of EvidenceChunk from searcher
    Output: formatted string like:
        [E1] "verbatim text from source" (Document: X, Page: Y)
        [E2] "another passage" (Document: X, Page: Z)
        ...
    
    Also returns a mapping: {citation_id: chunk_metadata} for verification.
    """
```

**Verification**: Index a sample document, query it, verify results are relevant.

---

## Phase 4: Draft Generation

**Goal**: Generate grounded legal drafts using retrieved evidence.

**Files to create**:
- `code/generation/__init__.py`
- `code/generation/drafter.py`
- `code/generation/grounding.py`
- `code/generation/templates.py`
- `prompts/system_base.txt`
- `prompts/case_fact_summary.txt`
- `prompts/title_review_summary.txt`
- `prompts/notice_summary.txt`
- `prompts/document_checklist.txt`

**Spec for `drafter.py`**:
```python
class DraftGenerator:
    """Generates grounded drafts from evidence.
    
    Pipeline:
    1. Load base system prompt + learned rules
    2. Retrieve relevant exemplars from correction bank
    3. Package evidence from retrieval
    4. Build full prompt: system + exemplars + evidence + task instruction
    5. Generate via Qwen3-VL-8B
    6. Post-process: parse structured output + verify citations
    
    Methods:
        generate_draft(doc_id: str, draft_type: str) -> DraftOutput
    """

@dataclass  
class DraftOutput:
    draft_type: str
    document_id: str
    content_markdown: str  # Human-readable draft
    content_structured: dict  # JSON structured fields
    citations: dict  # {field_path: [evidence_ids]}
    verification_status: dict  # {field_path: "verified"|"uncertain"|"unsupported"}
    generation_timestamp: str
    exemplars_used: list[str]  # IDs of exemplars injected
    rules_applied: list[str]  # Rules from learned_rules.yaml that were active
```

**Prompt template structure** (for each .txt file in prompts/):
```
You are a legal document analyst. Your task is to generate a {draft_type} based ONLY on the provided evidence.

RULES:
- Every claim must cite its source using [E1], [E2], etc.
- If information is unclear or missing from the evidence, state "Not clearly established in source documents"
- Never infer facts not directly supported by evidence
- Structure your output as specified below

{learned_rules_block}

EVIDENCE:
{evidence_block}

SIMILAR CORRECTIONS FROM PAST REVIEWS:
{exemplars_block}

OUTPUT FORMAT:
{format_spec_per_draft_type}
```

**Verification**: Generate a draft from indexed sample doc, verify it contains citations.

---

## Phase 5: Learning System

**Goal**: Capture operator corrections, extract patterns, consolidate into prompts.

**Files to create**:
- `code/learning/__init__.py`
- `code/learning/correction_store.py`
- `code/learning/exemplar_retriever.py`
- `code/learning/pattern_extractor.py`
- `code/learning/prompt_consolidator.py`

**Spec for `correction_store.py`**:
```python
class CorrectionStore:
    """Persistent storage for operator corrections.
    
    Storage: JSONL files in data/corrections/ (one file per draft_type)
    
    Schema per entry:
    {
        "id": "corr_XXXXX",
        "timestamp": "ISO8601",
        "document_id": "doc_XXX",
        "draft_type": "case_fact_summary",
        "field_path": "parties.defendant",
        "source_ocr_chunk": "original text from document",
        "generated_text": "what the system produced",
        "edited_text": "what the operator corrected it to",
        "correction_type": "omission|error|style|restructure",
        "embedding": [float...],  # BGE-M3 embedding of source_ocr_chunk
        "active": true
    }
    
    Methods:
        save_correction(correction: Correction) -> str  # returns ID
        get_all_active(draft_type: str) -> list[Correction]
        get_count() -> int
        archive_corrections(ids: list[str]) -> None
        get_corrections_since(timestamp: str) -> list[Correction]
    """
```

**Spec for `exemplar_retriever.py`**:
```python
class ExemplarRetriever:
    """Retrieves relevant past corrections for prompt injection.
    
    Input: source text chunk being processed + field_type
    Output: top-K most similar past corrections (by embedding similarity)
    
    Methods:
        get_relevant_exemplars(source_text: str, field_type: str, k: int = 3) -> list[Correction]
    
    Uses ChromaDB collection "legalmind_corrections" for fast similarity search.
    """
```

**Spec for `pattern_extractor.py`**:
```python
class PatternExtractor:
    """Extracts reusable rules from correction clusters.
    
    Triggered when: correction_store.get_count() % 20 == 0
    
    Process:
    1. Get last 20 corrections
    2. Embed and cluster by similarity
    3. For each cluster (>= 3 similar corrections):
       - Feed to Gemma-4-E4B with prompt:
         "Analyze these corrections. What common pattern do they fix?
          Write ONE concise rule that would prevent this error in future."
    4. Append new rules to config/learned_rules.yaml
    
    Methods:
        extract_patterns() -> list[str]  # new rules
        should_trigger() -> bool
    """
```

**Spec for `prompt_consolidator.py`**:
```python
class PromptConsolidator:
    """Folds learned rules into base system prompt.
    
    Triggered when: len(learned_rules) >= 10 new rules since last consolidation
    
    Process:
    1. Read current prompts/system_base.txt
    2. Read all rules from config/learned_rules.yaml
    3. Use Gemma-4-E4B to rewrite system prompt incorporating rules naturally
    4. Archive exemplars whose corrections are now covered by the updated prompt
    5. Save updated system_base.txt (keep backup as system_base.txt.bak)
    
    Methods:
        consolidate() -> None
        should_trigger() -> bool
    """
```

**Verification**: Create mock corrections, trigger pattern extraction, verify rules are generated.

---

## Phase 6: Verification Firewall

**Goal**: Verify generated claims against source evidence.

**Files to create**:
- `code/firewall/__init__.py`
- `code/firewall/source_anchor.py`
- `code/firewall/confidence.py`
- `code/firewall/runner.py`

**Spec for `source_anchor.py`**:
```python
class SourceAnchorVerifier:
    """Verifies that claims are supported by cited evidence.
    
    For each claim + citation pair:
    1. Extract the cited evidence text
    2. Compute semantic similarity (BGE-M3 embedding cosine)
    3. If similarity > 0.75: "verified"
    4. If 0.5 < similarity <= 0.75: "uncertain"  
    5. If similarity <= 0.5 or no citation: "unsupported"
    
    Methods:
        verify_claim(claim: str, evidence_text: str) -> VerificationResult
        verify_draft(draft: DraftOutput) -> dict[str, VerificationResult]
    """
```

**Verification**: Run firewall on generated draft, check status assignments make sense.

---

## Phase 7: Web Application

**Goal**: Full FastAPI web app with professional UI.

**Files to create**:
- `webapp/__init__.py`
- `webapp/main.py`
- `webapp/templates/base.html`
- `webapp/templates/dashboard.html`
- `webapp/templates/documents.html`
- `webapp/templates/drafts.html`
- `webapp/templates/audit.html`
- `webapp/templates/settings.html`
- `webapp/static/css/app.css`
- `webapp/static/js/app.js`
- `webapp/static/js/editor.js`
- `webapp/static/js/evidence.js`
- `webapp/static/icons/logo.svg`

**FastAPI Routes**:
```python
# Pages (SSR)
GET  /                          → dashboard
GET  /documents                 → document list + upload
GET  /documents/{id}/draft      → draft viewer + editor
GET  /audit                     → learning dashboard
GET  /settings                  → system settings

# API (JSON)
POST /api/v1/documents/upload   → upload + process document
GET  /api/v1/documents          → list all documents
GET  /api/v1/documents/{id}     → document details + processed data
POST /api/v1/drafts/generate    → generate draft for document
GET  /api/v1/drafts/{id}        → get draft details
POST /api/v1/corrections        → submit operator correction
GET  /api/v1/corrections        → list corrections
GET  /api/v1/learning/rules     → get learned rules
GET  /api/v1/learning/metrics   → improvement metrics over time
GET  /api/v1/system/status      → model status, GPU info

# SSE
GET  /api/v1/events/pipeline    → real-time pipeline progress
```

**UI Design Rules** (see CLAUDE.md for full spec):
- Light mode primary (`#f8fafc` background)
- Top horizontal nav bar with logo left, links center, settings right
- Three-column layout on draft page: docs left | draft center | evidence right
- Tailwind CSS via CDN (no build step)
- HTMX for dynamic updates without full page reload
- Clean, corporate, professional — like legal SaaS products

**Logo** (`webapp/static/icons/logo.svg`):
- Simple geometric mark: hexagonal document icon with a subtle gavel/balance scale motif
- Colors: primary blue (#1e40af) + slate
- Wordmark: "LegalMind" in Inter or similar professional sans-serif

---

## Phase 8: Testing

**Goal**: Comprehensive test suite covering all modules.

**Files to create**:
- `tests/__init__.py`
- `tests/conftest.py` (shared fixtures)
- `tests/unit/test_ocr_engine.py`
- `tests/unit/test_structurer.py`
- `tests/unit/test_retrieval.py`
- `tests/unit/test_drafter.py`
- `tests/unit/test_learning.py`
- `tests/integration/test_pipeline_e2e.py`
- `tests/integration/test_api_endpoints.py`
- `tests/integration/test_correction_loop.py`

**Testing strategy**:
- Unit tests mock the LLM layer (fast, no GPU needed)
- Integration tests use real models (slower, require GPU)
- Fixtures in `tests/fixtures/`: sample PDFs, expected outputs
- Use `httpx.AsyncClient` for API endpoint tests

**conftest.py fixtures**:
```python
@pytest.fixture
def sample_pdf() -> Path: ...
@pytest.fixture  
def processed_document() -> ProcessedDocument: ...
@pytest.fixture
def mock_model_manager() -> ModelManager: ...
@pytest.fixture
def app_client() -> AsyncClient: ...
```

---

## Phase 9: Documentation & Docker

**Goal**: Assessment-ready documentation and containerization.

**Files to create**:
- `README.md`
- `docs/architecture.md` (can symlink to ARCHITECTURE.md)
- `docs/assumptions.md`
- `docs/evaluation.md`
- `Dockerfile`
- `docker-compose.yml`

**README.md structure**:
```markdown
# LegalMind
One-line description.

## Quick Start
docker-compose up / manual setup steps

## Architecture
Brief overview + link to detailed doc

## Features
- Document processing (OCR + text extraction)
- Evidence retrieval (semantic + keyword)
- Grounded draft generation (4 output types)
- Continuous improvement from operator edits

## Sample Usage
Screenshots or example API calls

## Evaluation
Link to evaluation doc + summary metrics
```

**Dockerfile**:
- Base: `nvidia/cuda:12.6.0-runtime-ubuntu22.04`
- Install Python 3.11, system deps
- Copy code, install requirements
- Mount model cache as volume (not copied into image)
- Expose port 8000

**docker-compose.yml**:
- Service: legalmind (the app)
- Volume: model cache mount
- Volume: data persistence
- GPU passthrough via `deploy.resources.reservations.devices`

---

## Phase 10: Sample Data & Evaluation

**Goal**: Create sample documents, run evaluation, produce metrics.

**Files to create**:
- `scripts/seed_sample_data.py`
- `scripts/run_evaluation.py`
- `data/sample/` (3-5 sample legal documents)
- `data/sample/expected_outputs/` (ground truth for evaluation)

**Sample documents to create** (synthetic, not real legal docs):
1. A scanned lease agreement (PDF with embedded images, poor quality)
2. A typed court filing (clean PDF with parties, dates, claims)
3. A handwritten note (image file, partially illegible)
4. A property deed with chain of title (structured but messy formatting)
5. A compliance notice with deadlines (mixed formatting)

**Evaluation metrics**:
- Extraction accuracy: compare structured fields against manually annotated ground truth
- Retrieval precision@5: are retrieved chunks actually relevant?
- Grounding score: % of claims with valid source citations
- Improvement delta: edit distance reduction after corrections are applied
- Rule quality: do extracted rules align with correction patterns?

---

## Critical Implementation Notes

### Model Loading Pattern
```python
# CORRECT: Sequential model usage
mgr = ModelManager.instance()

# Phase 1: OCR
mgr.load("ocr")
ocr_results = [mgr.generate_with_image(page) for page in pages]
mgr.unload()

# Phase 2: Extraction
mgr.load("extraction")
structured = mgr.generate_structured(prompt, schema)
draft = mgr.generate_text(draft_prompt)
mgr.unload()

# Phase 3: Verification  
mgr.load("reasoning")
rules = mgr.generate_text(pattern_prompt)
mgr.unload()

# Embeddings: always available (separate, lightweight)
engine = InferenceEngine()
embeddings = engine.embed_batch(texts)
```

### CUDA DLL Setup (Windows)
```python
import os, sys
if sys.platform == "win32":
    cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
    if os.path.exists(cuda_bin):
        os.add_dll_directory(cuda_bin)
        os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")
```

### Error Handling Pattern
```python
# Graceful degradation: if OCR fails, mark document as "needs_manual_review"
# Never crash the pipeline — log error, set status, continue with next document
try:
    result = ocr_engine.process(page)
except Exception as e:
    logger.warning("OCR failed for page %d: %s", page_num, e)
    result = PageContent(text="", confidence=0.0, ocr_error=str(e))
```

### Exemplar Injection Format
```
PAST CORRECTIONS (use these to avoid similar errors):

Example 1:
- Source text: "...the defendant, Marcus Bell, filed a motion..."
- System generated: "Defendant: [Not clearly identified]"  
- Operator corrected to: "Defendant: Marcus Bell"
- Lesson: Extract party names even when embedded in procedural text.

Example 2:
...
```

---

## Verification Checklist (Run After Each Phase)

- [ ] Phase 1: `python -c "from code.llm_interface import ModelManager; print('OK')"`
- [ ] Phase 2: `python -c "from code.pipeline import DocumentIngester; print('OK')"`
- [ ] Phase 3: `python -c "from code.retrieval import DocumentIndexer, EvidenceSearcher; print('OK')"`
- [ ] Phase 4: `python -c "from code.generation import DraftGenerator; print('OK')"`
- [ ] Phase 5: `python -c "from code.learning import CorrectionStore, PatternExtractor; print('OK')"`
- [ ] Phase 6: `python -c "from code.firewall import FirewallRunner; print('OK')"`
- [ ] Phase 7: `uvicorn webapp.main:app --host 0.0.0.0 --port 8000` (should serve pages)
- [ ] Phase 8: `pytest tests/ -v` (all pass)
- [ ] Phase 9: `docker-compose up --build` (app accessible at localhost:8000)
- [ ] Phase 10: `python scripts/run_evaluation.py` (produces metrics)
