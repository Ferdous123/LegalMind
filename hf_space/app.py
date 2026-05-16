"""
LegalMind — Hugging Face Spaces entry point (READ-ONLY DEMO).

This launcher reuses the full `webapp.main` application but adds a middleware
that refuses any operation requiring GPU / model loading. Spaces have no
local GGUF models and (on the free tier) no usable GPU, so attempting to
upload a document, generate a new draft, or run rule extraction would hang
or crash. The middleware returns a clear 503 explaining the demo is
read-only and pointing the visitor at the existing populated data.

Everything that DOES NOT touch the GPU stays live:
  - Browsing processed documents, drafts, corrections, learned rules
  - The clickable verification badge / per-claim breakdown
  - The citation eye (page images + text-span fallback)
  - Operator overriding an uncertain claim to verified
  - Reading the prompts / corpus

The full system (with model inference) runs locally per the repo README.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

# Make the repo root importable so `from webapp.main import app` resolves
# when this file runs from inside hf_space/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# Mark the process as a Spaces demo so any code that reads the env can
# behave accordingly. This is independent of the middleware below.
os.environ.setdefault("LEGALMIND_DEMO", "1")

from webapp.main import app  # noqa: E402  (after sys.path tweak)


# ---------------------------------------------------------------------------
# Read-only demo middleware
# ---------------------------------------------------------------------------
# Any endpoint whose handler would load a multi-GB model or spawn a GPU
# subprocess is blocked here. The list is conservative on purpose — a 503
# with a clear message is strictly better than a hung worker or a CUDA OOM
# on the Spaces runtime.

_BLOCKED = {
    ("POST", "/api/v1/documents/upload"):
        "Uploading a document triggers OCR and structured extraction. "
        "Those need a local GPU and the GGUF model files, which are not "
        "shipped with this demo Space.",
    ("POST", "/api/v1/drafts/generate"):
        "Draft generation runs an 8B model in a subprocess. The free "
        "Spaces runtime has no GPU and no model files — generation is "
        "disabled here.",
    ("POST", "/api/v1/learning/rules"):
        "Pattern extraction loads a 4B model in a subprocess. Same "
        "reason as draft generation — not available in the demo.",
    ("POST", "/api/v1/system/gpu_reset"):
        "There is no GPU to reset in the demo.",
}

_DEMO_HINT = (
    "This is a read-only demo. The repository already contains a fully "
    "populated set of processed documents, generated drafts, operator "
    "corrections, and learned rules — browse them through the UI. To run "
    "the live pipeline (upload your own documents, generate new drafts, "
    "trigger rule extraction), clone the repo and follow README.md."
)


class DemoReadOnlyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        key = (request.method, request.url.path)
        msg = _BLOCKED.get(key)
        if msg is not None:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "demo_read_only",
                    "detail": msg,
                    "hint": _DEMO_HINT,
                },
                headers={
                    "X-Demo-Mode": "read-only",
                    "X-Toast-Type": "warn",
                    "X-Toast-Message": "Demo mode: this action is disabled (no GPU).",
                },
            )
        return await call_next(request)


app.add_middleware(DemoReadOnlyMiddleware)


# ---------------------------------------------------------------------------
# Spaces entry point
# ---------------------------------------------------------------------------
# HF Spaces (Docker SDK) runs whatever is on port 7860 by default. Uvicorn
# is started from the Dockerfile CMD; nothing else needed here.
