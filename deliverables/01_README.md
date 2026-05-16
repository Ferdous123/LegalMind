# LegalMind — Setup & Run

AI-powered legal document intelligence: ingests messy legal documents,
extracts structured data, retrieves grounded evidence, generates cited
drafts, and improves from operator edits. 100% local inference (GGUF).

> This is the deliverable copy. The repo-root `README.md` has the same
> instructions plus the full feature/API reference.

## Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA 12.6 (developed on RTX 3080, 10 GB). CPU works but
  generation is very slow.
- ~12 GB free disk for the three GGUF models.
- Local GGUF model files (no downloads at inference time):

| Model | Role | Quant |
|---|---|---|
| LightOnOCR-1B (+ mmproj) | OCR for scanned/degraded docs | Q8_0 |
| Qwen3-VL-8B (+ mmproj) | Structured extraction + draft generation | Q4_K_M |
| Gemma-4-E4B | Pattern extraction from corrections | Q4_K_M |

Point `config/models.yaml` (or `$LEGALMIND_MODEL_CACHE`) at the directory
holding these files.

## Run — Option A: Docker (recommended for reviewers)

```bash
cp .env.example .env
# set MODEL_CACHE_PATH in .env to your local GGUF directory
docker compose up --build
# UI: http://localhost:8000
```
Models are mounted read-only; `./data/` persists on the host.

## Run — Option B: Windows one-click

```cmd
setup.bat      :: first time: venv + deps + model check
run.bat        :: start server  ->  http://localhost:7860
```

## Run — Option C: Manual

```bash
python3.11 -m venv .venv
.venv\Scripts\activate                 # Windows  (source .venv/bin/activate on *nix)
pip install -r requirements.txt
# llama-cpp-python with CUDA:
#   Windows PS:  $env:CMAKE_ARGS="-DGGML_CUDA=on"; pip install --force-reinstall llama-cpp-python
cp config/models.yaml.example config/models.yaml   # then edit model paths

python -m uvicorn webapp.main:app --host 0.0.0.0 --port 7860
# UI: http://localhost:7860
```

> **Reliability note:** run **without** `--reload` for any real session.
> `--reload` restarts the worker on file changes; if that happens while a
> multi-GB model is loading it can deadlock the GPU. The Docker and `run.bat`
> paths already omit it. (Full rationale in `docs/decisions.md`.)

## First run

```bash
python scripts/seed_sample_data.py     # writes synthetic inputs + expected outputs
```
Then open the UI → **Pipeline** panel → drag in a file from
`data/sample/pdfs/` or `data/sample/images/`.

## Quick API smoke test

```bash
# clean text doc
curl -X POST http://localhost:7860/api/v1/documents/upload \
  -F file=@data/sample/pdfs/doc_1_lease_agreement.pdf \
  -F draft_type=case_fact_summary

# degraded scan (OCR path)
curl -X POST http://localhost:7860/api/v1/documents/upload \
  -F file=@data/sample/images/img_1_level_4.png \
  -F draft_type=case_fact_summary

# submit an operator correction
curl -X POST http://localhost:7860/api/v1/corrections -H "Content-Type: application/json" \
  -d '{"document_id":"<id>","draft_type":"case_fact_summary",
       "field_path":"parties.plaintiff","generated_text":"Plaintiff: J. Doe",
       "edited_text":"Plaintiff: Jonathon K. Doe","correction_type":"error"}'

# trigger rule extraction (also the Audit-tab button) — runs in a subprocess
curl -X POST http://localhost:7860/api/v1/learning/rules

curl http://localhost:7860/api/v1/learning/rules    # view learned rules
```

## Tests

```bash
pytest tests/ -v
```

## Where things live

```
code/pipeline/     ingest, OCR, structure, chunk
code/retrieval/    index, hybrid search, evidence packaging
code/generation/   drafter, grounding
code/firewall/     per-claim verification
code/learning/     correction store, exemplars, pattern extractor, consolidator
webapp/            FastAPI + Jinja2/HTMX UI (6 panels)
config/            models.yaml, paths.py, learned_rules.yaml
data/sample/       synthetic inputs + expected outputs
scripts/           seed_sample_data.py, process_corpus.py, evaluation
deliverables/      this submission set
docs/decisions.md  candid running RCA / decision log
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| Upload stalls at "ocr active", server unresponsive | A model load deadlocked (usually after a `--reload`). Restart without `--reload`; `POST /api/v1/system/gpu_reset` to clear VRAM budget. |
| `Pattern extraction failed` | It runs in a subprocess; check the temp `*.log` path in the error. Usually GPU contention — retry once idle. |
| Draft viewer shows raw markdown | Hard-refresh (Ctrl+Shift+R) — static JS is cache-busted by mtime but a stale tab can hold the old file. |
| Citation "eye" 404 on a .txt doc | Expected for older docs; the text-span fallback popup handles new ones. |
