# LegalMind

LegalMind is an internal document-intelligence pipeline for a law firm. It
takes messy legal documents — scanned pages, low-resolution PDFs,
typed text — pulls structured data out of them, retrieves the relevant
passages back, and produces grounded first-pass drafts an operator can
review and edit. Operator edits feed back into the system so future drafts
get better on similar documents.

Everything runs locally on GGUF models. No external APIs, no data leaves
the machine.

## What it does

- **Ingest** PDFs, PNGs, JPGs, TIFFs, plain text. Text-rich PDFs go through
  `pdfplumber`; scanned pages and images go through LightOnOCR-1B.
- **Extract** structured fields per draft type (parties, dates, chain of
  title, deadlines, document inventory) with Qwen3-VL-8B.
- **Index** every chunk in ChromaDB with a small embedding model and a
  BM25 fallback; results fused with Reciprocal Rank Fusion.
- **Generate** one of four grounded draft types with inline `[E1]`
  citations to the retrieved evidence.
- **Verify** every cited claim against its evidence with a deterministic
  n-gram + token-containment check. The verification badge in the UI is
  clickable — you can see exactly which evidence supported which claim,
  and override an uncertain claim to verified with one click (recorded as
  an operator override, not silently laundered).
- **Learn** from operator corrections in three layers: immediate few-shot
  exemplars, periodic rule extraction (Gemma-4-E4B clusters similar
  corrections into reusable rules), and prompt consolidation that folds
  stable rules into the base prompt.

The four draft types are: **case fact summary**, **title review summary**,
**notice-related summary**, **document checklist**.

## Quick start

### Docker

```bash
cp .env.example .env
# set MODEL_CACHE_PATH in .env to the directory holding the GGUF files
docker compose up --build
```
The web UI is at `http://localhost:8000`.

### Windows

```cmd
setup.bat   :: one-time: venv + dependencies + model check
run.bat     :: starts the server at http://localhost:7860
```

### Manual

```bash
python3.11 -m venv .venv
.venv\Scripts\activate         # PowerShell:  .venv\Scripts\Activate.ps1
                                # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
# llama-cpp-python with CUDA, e.g. on Windows PowerShell:
#   $env:CMAKE_ARGS="-DGGML_CUDA=on"; pip install --force-reinstall llama-cpp-python

cp config/models.yaml.example config/models.yaml   # then edit model paths

python -m uvicorn webapp.main:app --host 0.0.0.0 --port 7860
```

Run uvicorn **without** `--reload` for any real session. Editing files
during a multi-GB model load can deadlock the GPU; the Docker and `run.bat`
paths already omit it.

## Models

Three GGUF files, all local, no downloads at inference time:

| Model | Role | Quant |
|---|---|---|
| LightOnOCR-1B (+ mmproj) | OCR for scans and degraded images | Q8_0 |
| Qwen3-VL-8B (+ mmproj) | Structured extraction + draft generation | Q4_K_M |
| Gemma-4-E4B | Pattern extraction from operator corrections | Q4_K_M |

Point `config/models.yaml` (or `$LEGALMIND_MODEL_CACHE`) at the directory
containing them. Designed for a single 10 GB GPU — one large model resident
at a time, swapped by a singleton `ModelManager`.

## First run

```bash
python scripts/seed_sample_data.py
```
Writes the synthetic sample inputs and expected outputs under `data/sample/`.
Then open the UI → **Pipeline** panel → drop in a file from
`data/sample/pdfs/` (clean text) or `data/sample/images/` (degraded scans)
and watch it move through ingest → OCR → extract → index → retrieve →
generate → verify in real time.

## API

```bash
# Upload a document and produce a draft
curl -X POST http://localhost:7860/api/v1/documents/upload \
  -F file=@/path/to/contract.pdf \
  -F draft_type=case_fact_summary

# Get a processed document or a draft
curl http://localhost:7860/api/v1/documents/{doc_id}
curl http://localhost:7860/api/v1/drafts/draft_{doc_id}_{draft_type}

# Submit an operator correction
curl -X POST http://localhost:7860/api/v1/corrections \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "{doc_id}",
    "draft_type": "case_fact_summary",
    "field_path": "parties.plaintiff",
    "generated_text": "Plaintiff: J. Doe",
    "edited_text": "Plaintiff: Jonathon K. Doe",
    "correction_type": "error"
  }'

# Trigger pattern extraction (also the Audit-tab button); runs in a subprocess
curl -X POST http://localhost:7860/api/v1/learning/rules

# Operator-verify a claim the firewall flagged uncertain/unsupported
curl -X POST http://localhost:7860/api/v1/drafts/{draft_id}/override \
  -H "Content-Type: application/json" \
  -d '{"field_path":"draft.claim[3]"}'
```

## Project structure

```
code/
  pipeline/    ingestion, OCR, structuring, page-aware chunking
  retrieval/   ChromaDB indexer, hybrid searcher, evidence packaging
  generation/  drafter, grounding verifier
  firewall/    per-claim and per-field verification
  learning/    correction store, exemplars, pattern extractor, consolidator
  llm_interface/  singleton model manager, inference wrapper, GPU guard
webapp/        FastAPI app, Jinja2 + HTMX UI (six panels)
prompts/       one prompt per draft type + base system prompt
config/        models.yaml, paths.py, learned_rules.yaml
data/
  uploads/     raw uploaded files
  processed/   processed document JSON and generated drafts
  pages/       rendered PDF page images for the citation viewer
  corrections/ operator-correction log (feeds the learning loop)
  sample/      synthetic inputs and expected outputs
scripts/       seed data, evaluation, controlled A/B, demo processors
tests/         pytest suite
deliverables/  written reports: architecture, assumptions, samples, evaluation
docs/          decisions.md (development log)
```

## Tests

```bash
pytest tests/ -v
```

## Evaluation

```bash
python scripts/run_evaluation.py    # extraction accuracy against expected outputs
python scripts/controlled_ab.py     # same-document A/B isolating the learning effect
```
Results and methodology are in [`deliverables/05_EVALUATION.md`](deliverables/05_EVALUATION.md).

## Documentation

- [`deliverables/02_ARCHITECTURE.md`](deliverables/02_ARCHITECTURE.md) — components and data flow
- [`deliverables/03_ASSUMPTIONS_AND_TRADEOFFS.md`](deliverables/03_ASSUMPTIONS_AND_TRADEOFFS.md) — design choices
- [`deliverables/04_SAMPLE_INPUTS_AND_OUTPUTS.md`](deliverables/04_SAMPLE_INPUTS_AND_OUTPUTS.md) — what's in `data/sample/` and `data/processed/`
- [`deliverables/05_EVALUATION.md`](deliverables/05_EVALUATION.md) — measured results
- [`docs/decisions.md`](docs/decisions.md) — development log

## Configuration

| File | Purpose |
|---|---|
| `config/models.yaml` | Model paths, context lengths, GPU layer counts |
| `config/paths.py` | Data directory paths |
| `config/learned_rules.yaml` | Rules extracted from operator corrections |

| Environment variable | Purpose |
|---|---|
| `LEGALMIND_MODEL_CACHE` | Directory containing the GGUF model files |
| `HF_HUB_DISABLE_SYMLINKS_WARNING` | Set to `1` to suppress HF Hub warnings |

## Troubleshooting

| Symptom | Fix |
|---|---|
| Upload stalls at "ocr active" | A model load is stuck. Restart the server without `--reload`; `POST /api/v1/system/gpu_reset` clears the VRAM budget. |
| `Pattern extraction failed` | Runs in a subprocess — its log path appears in the error. Usually transient GPU contention; retry once the system is idle. |
| Draft viewer shows raw markdown | Hard-refresh the page (Ctrl+Shift+R); a stale tab can hold the previous JS. |
| Eye icon won't load on a `.txt` doc | Falls back to a highlighted source-text popup; an outdated cached tab can hold the old image-only modal — hard-refresh. |
