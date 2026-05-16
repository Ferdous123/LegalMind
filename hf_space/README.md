---
title: LegalMind
emoji: ⚖️
colorFrom: indigo
colorTo: yellow
sdk: docker
app_port: 7860
app_file: hf_space/app.py
pinned: false
short_description: Grounded legal-document drafting with operator-edit learning
---

# LegalMind — read-only demo

This Space is a **read-only demo** of LegalMind. The repository ships
with a fully populated corpus: extracted documents, generated drafts,
operator corrections, and learned rules. You can browse all of it here
without a GPU.

Operations that need a local GPU and the GGUF model files — uploading a
new document, generating a new draft, triggering rule extraction — are
blocked with a 503 explaining the demo is read-only.

## What you can do here

- Browse processed documents in the **Library** and **Pipeline** panels
- Open any draft → see the grounded markdown with `[E1]`-style citations
- Click the **verification badge** (`N verified / N uncertain / N
  unsupported`) → see each claim with its evidence snippet
- Click the **eye** on a citation → the source page image (PDFs and
  scans) or a highlighted source text excerpt (`.txt`)
- Use **Verify (override)** to flip an uncertain claim to verified — the
  override is recorded as `operator override` in the draft JSON
- See accumulated operator corrections and extracted rules in the
  **Audit** panel

## What you cannot do here

| Endpoint | Why it's disabled |
|---|---|
| `POST /api/v1/documents/upload` | OCR + structured extraction need a GPU and the GGUF models |
| `POST /api/v1/drafts/generate` | 8B draft model + subprocess |
| `POST /api/v1/learning/rules` | 4B rule-extraction model + subprocess |
| `POST /api/v1/system/gpu_reset` | No GPU |

Each returns a clear JSON 503 with a hint.

## Run the full pipeline locally

Clone the repository and follow the root `README.md`. The full system
needs a single 10 GB NVIDIA GPU (developed on RTX 3080) and three local
GGUF files: LightOnOCR-1B, Qwen3-VL-8B, Gemma-4-E4B.
