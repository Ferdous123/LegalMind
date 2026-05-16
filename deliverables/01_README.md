# LegalMind — Setup & Run

LegalMind is a local document-intelligence pipeline: it ingests messy legal
documents, extracts structured fields, retrieves grounded evidence,
generates cited drafts, and learns from operator corrections. All inference
is local (GGUF via llama-cpp); no external APIs.

The root `README.md` has a longer overview. This document focuses on getting
the system running.

## Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA 12.6 (developed on RTX 3080, 10 GB). CPU works but
  draft generation is impractically slow.
- ~12 GB free disk for the three GGUF models.
- The three model files on disk:

| Model | Role | Quantization |
|---|---|---|
| LightOnOCR-1B (+ mmproj) | OCR for scans / degraded images | Q8_0 |
| Qwen3-VL-8B (+ mmproj) | Structured extraction + draft generation | Q4_K_M |
| Gemma-4-E4B | Pattern extraction from operator corrections | Q4_K_M |

Set `config/models.yaml` (or the `LEGALMIND_MODEL_CACHE` env var) to the
directory containing the files.

## Run — Docker

```bash
cp .env.example .env
# set MODEL_CACHE_PATH in .env
docker compose up --build
# UI: http://localhost:8000
```
The model directory is mounted read-only; `./data/` persists on the host.

## Run — Windows one-click

```cmd
setup.bat       :: first time only: venv, dependencies, model check
run.bat         :: starts the server at http://localhost:7860
```

## Run — manual

```bash
python3.11 -m venv .venv
.venv\Scripts\activate                  # Windows
# source .venv/bin/activate              # Linux/macOS
pip install -r requirements.txt

# llama-cpp-python with CUDA. Windows PowerShell example:
# $env:CMAKE_ARGS="-DGGML_CUDA=on"; pip install --force-reinstall llama-cpp-python

cp config/models.yaml.example config/models.yaml   # edit model paths

python -m uvicorn webapp.main:app --host 0.0.0.0 --port 7860
```

Run uvicorn **without** `--reload`. Editing files while a multi-GB model is
loading deadlocks the GPU; the Docker and `run.bat` paths already omit it.

## First run

```bash
python scripts/seed_sample_data.py
```
This writes the synthetic inputs and expected outputs under `data/sample/`.
Open the UI → **Pipeline** panel → drop in a file from
`data/sample/pdfs/` or `data/sample/images/`.

## API smoke test

```bash
# clean text PDF
curl -X POST http://localhost:7860/api/v1/documents/upload \
  -F file=@data/sample/pdfs/doc_1_lease_agreement.pdf \
  -F draft_type=case_fact_summary

# degraded scan (OCR path)
curl -X POST http://localhost:7860/api/v1/documents/upload \
  -F file=@data/sample/images/img_1_level_4.png \
  -F draft_type=case_fact_summary

# submit an operator correction
curl -X POST http://localhost:7860/api/v1/corrections \
  -H "Content-Type: application/json" \
  -d '{"document_id":"<id>","draft_type":"case_fact_summary",
       "field_path":"parties.plaintiff","generated_text":"Plaintiff: J. Doe",
       "edited_text":"Plaintiff: Jonathon K. Doe","correction_type":"error"}'

# trigger pattern extraction (the Audit-tab button uses the same endpoint)
curl -X POST http://localhost:7860/api/v1/learning/rules

# view learned rules
curl http://localhost:7860/api/v1/learning/rules

# operator-verify a claim the firewall flagged
curl -X POST http://localhost:7860/api/v1/drafts/<draft_id>/override \
  -H "Content-Type: application/json" \
  -d '{"field_path":"draft.claim[3]"}'
```

## Tests

```bash
pytest tests/ -v
```

## Layout

```
code/pipeline/      ingest, OCR, structurer, page-aware chunker
code/retrieval/     index, hybrid searcher, evidence packaging
code/generation/    drafter, grounding verifier
code/firewall/      per-claim and per-field verification
code/learning/      correction store, exemplars, pattern extractor, consolidator
webapp/             FastAPI + Jinja2/HTMX UI (six panels)
prompts/            one per draft type + base system prompt
config/             models.yaml, paths.py, learned_rules.yaml
data/sample/        synthetic inputs and expected outputs
data/processed/     processed documents and generated drafts
scripts/            seed_sample_data, run_evaluation, controlled_ab, …
```

## Troubleshooting

| Symptom | What to do |
|---|---|
| Upload hangs at "ocr active" | A model load deadlocked. Restart the server without `--reload`. `POST /api/v1/system/gpu_reset` zeroes the VRAM budget. |
| `Pattern extraction failed` | Rule extraction runs in a subprocess; the log path appears in the error message. Usually GPU contention; retry once the system is idle. |
| Draft viewer shows raw `# headings` | Hard-refresh (Ctrl+Shift+R). The page renders markdown client-side and a stale tab can hold old JS. |
| Eye icon spins on a `.txt` document | A cached tab is using the old image-only modal. Hard-refresh; the current modal falls back to a highlighted text excerpt for documents without a page image. |
