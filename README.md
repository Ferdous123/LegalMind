# LegalMind

AI-powered legal document intelligence platform for Pearson Specter Litt. Ingests scanned and typed legal documents, extracts structured data, retrieves evidence, generates grounded drafts, and improves continuously from operator corrections.

---

## 📦 Assessment Submission — start here

All required deliverables are in **[`deliverables/`](deliverables/)**. Begin with
**[`deliverables/00_SUBMISSION_INDEX.md`](deliverables/00_SUBMISSION_INDEX.md)** —
it maps every "What to Submit" item and every rubric point to its file, and
gives a 15-minute reviewer path.

| Required item | File |
|---|---|
| Setup & run | [`deliverables/01_README.md`](deliverables/01_README.md) |
| Architecture overview | [`deliverables/02_ARCHITECTURE.md`](deliverables/02_ARCHITECTURE.md) |
| Assumptions & tradeoffs | [`deliverables/03_ASSUMPTIONS_AND_TRADEOFFS.md`](deliverables/03_ASSUMPTIONS_AND_TRADEOFFS.md) |
| Sample inputs & outputs | [`deliverables/04_SAMPLE_INPUTS_AND_OUTPUTS.md`](deliverables/04_SAMPLE_INPUTS_AND_OUTPUTS.md) |
| Evaluation approach & results | [`deliverables/05_EVALUATION.md`](deliverables/05_EVALUATION.md) |

`docs/decisions.md` is a candid running root-cause / decision log kept
deliberately to show the engineering process.

---

## Quick Start

### Option 1: Docker (recommended)

**Prerequisites**: Docker Desktop with NVIDIA Container Toolkit installed and configured.

```bash
# Clone the repository
git clone https://github.com/pearsonspecterlitt/legalmind.git
cd legalmind

# Configure the path to your local GGUF model cache
cp .env.example .env
# Edit .env and set MODEL_CACHE_PATH to your model directory, e.g.:
#   MODEL_CACHE_PATH=/home/user/models          (Linux/macOS)
#   MODEL_CACHE_PATH=F:/Research_Paper_Projects/LLMs   (Windows)

# Build and start
docker compose up --build

# The web UI is available at http://localhost:8000
# (manual setup uses port 7860; set --port in uvicorn invocation accordingly)
```

To run in the background:

```bash
docker compose up -d
docker compose logs -f legalmind
```

To stop:

```bash
docker compose down
```

**Notes**:
- Model GGUF files are mounted read-only from your local model cache — they are never copied into the image.
- Processed documents, corrections, and uploads persist in `./data/` on the host.
- GPU passthrough requires NVIDIA Container Toolkit (`nvidia-ctk runtime configure --runtime=docker`).

---

### Option 2: One-Click (Windows)

```cmd
:: First time setup (installs dependencies, checks models)
setup.bat

:: Start the server
run.bat
```

The web UI will be at `http://localhost:8000`.

---

### Option 3: Manual Setup

**Prerequisites**: Python 3.11+, CUDA 12.6 toolkit (for GPU inference), Git.

```bash
# Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate          # Linux/macOS
.venv\Scripts\activate             # Windows

# Install dependencies
pip install -r requirements.txt

# Install llama-cpp-python with CUDA support
CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall llama-cpp-python>=0.3.20

# On Windows (PowerShell):
# $env:CMAKE_ARGS="-DGGML_CUDA=on"; pip install --force-reinstall llama-cpp-python>=0.3.20

# Configure model paths
cp config/models.yaml.example config/models.yaml
# Edit config/models.yaml to point to your GGUF files

# Start the server
uvicorn webapp.main:app --host 0.0.0.0 --port 7860 --reload
```

The web UI is available at `http://localhost:7860`.

---

## Required Models

LegalMind runs entirely on local GGUF models. No API keys or internet connection required during inference.

| Model | Role | Recommended Quant |
|---|---|---|
| LightOnOCR-1B | OCR for scanned documents | Q8_0 + mmproj |
| Qwen3-VL-8B | Structured extraction + draft generation | Q4_K_M |
| Gemma-4-E4B | Pattern extraction from correction clusters | Q4_K_M |

All models must be placed in the directory configured as `LEGALMIND_MODEL_CACHE` (default: `/models` in Docker, or the path in `config/models.yaml` for manual setup).

---

## Architecture

LegalMind is structured as a sequential pipeline:

```
Upload -> Ingest (OCR or text extract) -> Structure -> Chunk -> Index
       -> Retrieve (semantic + keyword) -> Draft -> Verify -> UI
                                                          |
                                                   Operator edits
                                                          |
                                              Learning system (exemplars -> rules -> prompt)
```

Each stage is independently testable. The system is designed for a single RTX 3080 (10 GB VRAM): models are loaded and unloaded sequentially by a singleton `ModelManager`.

See [`deliverables/02_ARCHITECTURE.md`](deliverables/02_ARCHITECTURE.md) for the full component breakdown and data flow diagram.

See [`deliverables/03_ASSUMPTIONS_AND_TRADEOFFS.md`](deliverables/03_ASSUMPTIONS_AND_TRADEOFFS.md) for design decisions and tradeoffs.

---

## Features

### Document Processing
- Accepts PDF, PNG, JPG, and TIFF files
- Automatic routing: text-rich PDFs use `pdfplumber` directly; sparse or scanned PDFs and all images are processed through the OCR engine
- OCR threshold: fewer than 100 characters per page triggers the OCR path
- Page-level confidence flags preserve traceability

### Evidence Retrieval
- All document text is chunked (512-token target, 64-token overlap) and stored in ChromaDB
- Hybrid retrieval: semantic search (sentence-transformers embeddings on CPU) + BM25 keyword matching
- Results fused using Reciprocal Rank Fusion (RRF) for superior relevance
- Returns top-K evidence chunks with relevance scores and verbatim source spans
- Falls back to BM25-only if embedding model unavailable

### Grounded Draft Generation
- Four draft output types (see below)
- Every claim in the generated draft must reference a numbered evidence item (`[E1]`, `[E2]`, etc.)
- Claims without citations are flagged as "unsupported" and visually highlighted in the UI
- A verification firewall checks that cited evidence semantically supports each claim

### Continuous Improvement
- **Layer 1 (Exemplar bank)**: Every operator correction is stored and retrieved as a few-shot example for similar future documents
- **Layer 2 (Pattern extraction)**: Every 20 corrections, Gemma-4-E4B analyzes the correction cluster and extracts reusable rules (`config/learned_rules.yaml`)
- **Layer 3 (Prompt consolidation)**: When 10+ new rules accumulate, they are folded into the base system prompt; redundant exemplars are archived

### Verification Firewall
- Per-claim source anchoring: verifies each `[E]`-cited markdown claim against its cited evidence (deterministic 3-gram verbatim OR ≥0.30 token containment — auditable, no extra model)
- Also scores structured fields; internal `_cascade_meta` is excluded from verification
- Status per claim/field: `verified`, `uncertain`, `unsupported`, `manual_review`
- The draft viewer's verification badge is **clickable** — lists each claim by status with its evidence snippet

---

## Draft Output Types

| Type | Use Case | Key Extracted Fields |
|---|---|---|
| **Case Fact Summary** | Litigation review | Parties, dates, claims, procedural history, evidence items, outcomes |
| **Title Review Summary** | Real estate transactions | Property description, ownership chain, encumbrances, gaps, recording info |
| **Notice-Related Summary** | Compliance and regulatory | Notice type, deadlines, requirements, response actions, compliance steps |
| **Document Checklist** | Transaction due diligence | Required docs, present docs, missing docs, verification status |

---

## Sample Usage

### Via Web UI

1. Navigate to `http://localhost:7860` (manual) or `http://localhost:8000` (Docker)
2. Go to the **Pipeline** panel and drag in a PDF or image file
3. Select a draft type and process; watch the live SSE pipeline
4. Open the draft — it renders as formatted markdown. Hover any section, click
   **Edit**, and correct it **in place** (WYSIWYG); Save submits a correction
5. Corrections are stored immediately (Layer 1 exemplars) and, via the **Audit**
   panel, drive rule extraction that improves future drafts

### Via API

Upload a document and request a draft:

```bash
# Upload a document (returns document_id and full document record on completion)
curl -X POST http://localhost:7860/api/v1/documents/upload \
  -F "file=@/path/to/contract.pdf" \
  -F "draft_type=case_fact_summary"

# Retrieve the processed document record
curl http://localhost:7860/api/v1/documents/{document_id}

# Generate a draft for the document
curl -X POST http://localhost:7860/api/v1/drafts/generate \
  -H "Content-Type: application/json" \
  -d '{"document_id": "{document_id}", "draft_type": "case_fact_summary"}'

# Retrieve a previously generated draft
curl http://localhost:7860/api/v1/drafts/{document_id}_{draft_type}
```

Submit a correction:

```bash
curl -X POST http://localhost:7860/api/v1/corrections \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "{document_id}",
    "draft_type": "case_fact_summary",
    "field_path": "parties.plaintiff",
    "source_ocr_chunk": "Plaintiff John Doe filed suit on ...",
    "generated_text": "Plaintiff: John Doe",
    "edited_text": "Plaintiff: Jonathon K. Doe",
    "correction_type": "error"
  }'
```

Retrieve learned rules:

```bash
curl http://localhost:7860/api/v1/learning/rules
```

---

## Project Structure

```
legalmind/
├── code/
│   ├── pipeline/         # Ingestion, OCR, structuring, chunking
│   ├── retrieval/        # Indexer, searcher, evidence packaging
│   ├── generation/       # Drafter, grounding enforcement
│   ├── firewall/         # Source anchor, confidence, runner
│   └── learning/         # Correction store, exemplar retriever, pattern extractor, consolidator
├── webapp/               # FastAPI app, Jinja2 templates, HTMX frontend
├── config/               # models.yaml, paths.py, learned_rules.yaml
├── data/
│   ├── uploads/          # Raw uploaded files
│   ├── processed/        # ProcessedDocument JSON files
│   ├── corrections/      # JSONL correction log
│   └── sample/           # Synthetic demo documents + expected outputs
├── scripts/              # Seed data, evaluation, corpus/demo processors
├── tests/                # Pytest test suite
├── deliverables/         # Assessment submission set (00–05)
├── docs/                 # decisions.md (running RCA log) + deliverable pointers
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Running Tests

```bash
pytest tests/ -v --cov=code --cov=webapp --cov-report=term-missing
```

---

## Evaluation

To run the extraction accuracy evaluation against the sample documents:

```bash
# First, seed the sample data
python scripts/seed_sample_data.py

# Process the sample documents through the pipeline (requires models)
# Then run evaluation against pre-processed outputs
python scripts/run_evaluation.py
```

See [`deliverables/05_EVALUATION.md`](deliverables/05_EVALUATION.md) for the full methodology and the measured controlled-A/B results.

---

## Configuration

Key configuration files:

| File | Purpose |
|---|---|
| `config/models.yaml` | Model paths, context lengths, GPU layer counts |
| `config/paths.py` | Data directory paths |
| `config/learned_rules.yaml` | Auto-generated rules from correction analysis |

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `LEGALMIND_MODEL_DIR` | (from models.yaml) | Directory containing GGUF model files |
| `LEGALMIND_MODEL_CACHE` | `/models` | Docker-only: mount point for model cache |
| `HF_HUB_DISABLE_SYMLINKS_WARNING` | - | Set to `1` to suppress HuggingFace warnings |

---

## License

Internal use only — Pearson Specter Litt. Not licensed for redistribution.
