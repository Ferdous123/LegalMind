import os
import sys

if sys.platform == "win32":
    cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
    if os.path.exists(cuda_bin):
        os.add_dll_directory(cuda_bin)
        os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")

import asyncio
import glob
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Optional

import yaml
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from config.paths import (
    CORRECTIONS_DIR,
    PROCESSED_DIR,
    STATIC_DIR,
    TEMPLATES_DIR,
    UPLOADS_DIR,
    ensure_dirs,
    LEARNED_RULES_YAML,
    MODELS_YAML,
)
from code.pipeline.ingestion import DocumentIngester
from code.generation.drafter import DraftGenerator
from code.retrieval.indexer import DocumentIndexer
from code.learning.correction_store import Correction, CorrectionStore
from code.learning.pattern_extractor import PatternExtractor
from code.learning.prompt_consolidator import PromptConsolidator
from code.firewall.runner import FirewallRunner
from code.generation.templates import get_draft_label, get_supported_types
from code.llm_interface.gpu_guard import get_gpu_info

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline SSE event bus
# ---------------------------------------------------------------------------
# Each SSE connection gets its own asyncio.Queue. _sse_broadcast() is awaited
# from the upload handler and puts the event into every subscriber's queue.
# The SSE endpoint sends a "connected" sentinel immediately on connect so the
# browser knows the stream is live before it starts the upload POST.

_sse_subscribers: list[asyncio.Queue] = []
_sse_log = Path(__file__).resolve().parent.parent / "logs" / "pipeline_events.jsonl"

# One active pipeline job at a time (upload or draft generation).
# Concurrent requests receive 409 — single pipeline job at a time.
_pipeline_lock: asyncio.Lock = asyncio.Lock()

# ---------------------------------------------------------------------------
# Job Queue
# ---------------------------------------------------------------------------
# In-memory queue: jobs go through queued → in_progress → done/failed.
# A background worker pulls from the queue and runs one job at a time.
# SSE events are broadcast for each stage so the UI can animate the pipeline.

import threading
import uuid as _uuid

_job_queue_state: dict = {
    "queued": [],        # [{id, document_id, draft_type, filename, requested_at}]
    "in_progress": None, # {id, document_id, draft_type, filename, started_at} or None
    "done": [],          # [{id, document_id, draft_type, filename, completed_at, duration_sec}]
    "failed": [],        # [{id, document_id, draft_type, filename, error, completed_at}]
}
_job_queue_lock = threading.Lock()
_job_queue_event: asyncio.Event = None  # set in lifespan
_job_cancel_requested: bool = False
_job_worker_task: asyncio.Task = None
_job_active_proc: dict = {}  # {"proc": Popen, "job_id": str} — for cancellation

# Persist the queue across uvicorn --reload restarts. Without this, every
# code edit wipes the queued/done/failed lists from memory and orphans any
# subprocess started before the reload.
_QUEUE_STATE_FILE = Path(__file__).resolve().parent.parent / "data" / "queue_state.json"


def _save_queue_state() -> None:
    """Snapshot _job_queue_state to disk. Caller MUST hold _job_queue_lock."""
    try:
        _QUEUE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _QUEUE_STATE_FILE.write_text(
            json.dumps(_job_queue_state, default=str), encoding="utf-8"
        )
    except Exception as exc:
        logger.warning("Could not persist queue state: %s", exc)


def _load_queue_state() -> None:
    """Restore queued/done/failed from disk on startup. Drops any in-flight
    job — its subprocess was already killed by the reload, and the new
    worker can't adopt orphaned children safely."""
    if not _QUEUE_STATE_FILE.exists():
        return
    try:
        data = json.loads(_QUEUE_STATE_FILE.read_text(encoding="utf-8"))
        with _job_queue_lock:
            _job_queue_state["queued"] = list(data.get("queued", []))
            _job_queue_state["done"] = list(data.get("done", []))[:50]
            _job_queue_state["failed"] = list(data.get("failed", []))[:50]
            # If an in_progress job was persisted, move it to failed with
            # a clear reason — its subprocess died with the previous worker.
            ip = data.get("in_progress")
            if ip:
                _job_queue_state["failed"].insert(0, {
                    **ip,
                    "error": "Worker reloaded mid-job; subprocess terminated.",
                    "completed_at": datetime.now().isoformat(),
                    "status": "failed",
                })
            _job_queue_state["in_progress"] = None
        logger.info("Restored queue state: %d queued, %d done, %d failed",
                    len(_job_queue_state["queued"]),
                    len(_job_queue_state["done"]),
                    len(_job_queue_state["failed"]))
    except Exception as exc:
        logger.warning("Could not load queue state: %s", exc)


async def _sse_broadcast(stage: str, status: str, message: str = "") -> None:
    """Put a pipeline event into every connected subscriber's queue."""
    payload = json.dumps({
        "stage": stage, "status": status,
        "message": message, "ts": datetime.now().isoformat(),
    })
    try:
        _sse_log.parent.mkdir(parents=True, exist_ok=True)
        with open(_sse_log, "a", encoding="utf-8") as fh:
            fh.write(payload + "\n")
    except Exception:
        pass
    dead: list[asyncio.Queue] = []
    for q in list(_sse_subscribers):
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        try:
            _sse_subscribers.remove(q)
        except ValueError:
            pass


PANELS = [
    {"id": "pipeline", "name": "Pipeline", "icon": "M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"},
    {"id": "library", "name": "Library", "icon": "M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"},
    {"id": "audit", "name": "Audit", "icon": "M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4"},
    {"id": "prompts", "name": "Prompts", "icon": "M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"},
    {"id": "corpus", "name": "Corpus", "icon": "M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.747 0-3.332-.523-4.5-1.247"},
    {"id": "settings", "name": "Settings", "icon": "M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z M15 12a3 3 0 11-6 0 3 3 0 016 0z"},
]

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
LEARNED_TERMS_FILE = Path(__file__).resolve().parent.parent / "data" / "learned_terms.json"
PAGES_DIR = Path(__file__).resolve().parent.parent / "data" / "pages"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    global _job_queue_event, _job_worker_task
    ensure_dirs()
    _load_queue_state()
    _job_queue_event = asyncio.Event()
    # If there were queued jobs from a previous worker, kick the worker.
    if _job_queue_state["queued"]:
        _job_queue_event.set()
    _job_worker_task = asyncio.create_task(_job_queue_worker())
    yield
    _job_worker_task.cancel()
    try:
        await _job_worker_task
    except asyncio.CancelledError:
        pass
    # Save final state on shutdown
    with _job_queue_lock:
        _save_queue_state()


app = FastAPI(title="LegalMind", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _static_version() -> str:
    """Compute a fingerprint over our custom JS so templates can cache-bust
    automatically when we edit them. Returns the max mtime of webapp/static/js
    rendered as a compact int. Falls back to process start time on error."""
    try:
        js_dir = STATIC_DIR / "js"
        if js_dir.exists():
            latest = max((p.stat().st_mtime for p in js_dir.rglob("*.js")), default=0.0)
            return str(int(latest))
    except Exception:
        pass
    return str(int(datetime.now().timestamp()))


# Expose static_v as a Jinja global so every template can `?v={{ static_v }}`
templates.env.globals["static_v"] = _static_version


@app.middleware("http")
async def no_cache_middleware(request: Request, call_next):
    response = await call_next(request)
    if "/static/" not in str(request.url):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


# ---------------------------------------------------------------------------
# SSR page routes
# ---------------------------------------------------------------------------


@app.get("/")
async def root():
    return RedirectResponse(url="/panel/pipeline", status_code=302)


@app.get("/panel/{panel_id}")
async def panel_page(request: Request, panel_id: str):
    valid = {p["id"] for p in PANELS}
    if panel_id not in valid:
        return RedirectResponse(url="/panel/pipeline", status_code=302)

    ctx = {"panels": PANELS, "active_panel": panel_id, "request": request}

    if panel_id == "pipeline":
        # Recent documents. Filter out draft_*.json FIRST, then take the
        # newest 10 — otherwise the slice is consumed by draft files (every
        # doc writes a draft that sorts newer), leaving only a handful of
        # actual documents visible.
        docs = []
        try:
            doc_paths = [
                p for p in sorted(PROCESSED_DIR.glob("*.json"), reverse=True)
                if not p.stem.startswith("draft_")
            ][:10]
            for path in doc_paths:
                doc = json.loads(path.read_text(encoding="utf-8"))
                doc["has_draft"] = _doc_has_draft(doc.get("id", ""))
                doc["draft_types_available"] = _doc_draft_types(doc.get("id", ""))
                docs.append(doc)
        except Exception:
            pass
        ctx["recent_docs"] = docs
        ctx["draft_types"] = get_supported_types()

    elif panel_id == "library":
        docs = []
        try:
            for path in sorted(PROCESSED_DIR.glob("*.json"), reverse=True):
                if not path.stem.startswith("draft_"):
                    doc = json.loads(path.read_text(encoding="utf-8"))
                    doc["has_draft"] = _doc_has_draft(doc.get("id", ""))
                    doc["draft_types_available"] = _doc_draft_types(doc.get("id", ""))
                    docs.append(doc)
        except Exception:
            pass
        ctx["documents"] = docs
        ctx["draft_types"] = get_supported_types()

    elif panel_id == "audit":
        corrections = []
        learned_rules = []
        metrics = {}
        total_corrections = 0
        try:
            store = CorrectionStore()
            corrections = [_correction_to_dict(c) for c in await asyncio.to_thread(store.get_recent, 50)]
            total_corrections = await asyncio.to_thread(store.get_count)
            if LEARNED_RULES_YAML.exists():
                data = yaml.safe_load(LEARNED_RULES_YAML.read_text(encoding="utf-8")) or {}
                learned_rules = data.get("rules", [])
            by_draft_type = {}
            by_correction_type = {}
            for t in get_supported_types():
                count = await asyncio.to_thread(store.get_count, t["id"])
                if count:
                    by_draft_type[t["id"]] = count
            for c in await asyncio.to_thread(store.get_recent, 500):
                ct = getattr(c, "correction_type", None) or "unknown"
                by_correction_type[ct] = by_correction_type.get(ct, 0) + 1
            last_extraction_date = None
            active_exemplars = 0
            if LEARNED_RULES_YAML.exists():
                rules_data = yaml.safe_load(LEARNED_RULES_YAML.read_text(encoding="utf-8")) or {}
                last_extraction_date = rules_data.get("last_extraction")
                active_exemplars = rules_data.get("total_corrections_processed", 0)
            metrics = {
                "by_draft_type": by_draft_type,
                "by_correction_type": by_correction_type,
                "last_extraction_date": str(last_extraction_date) if last_extraction_date else None,
                "active_exemplars": active_exemplars,
            }
        except Exception as exc:
            logger.warning("Audit context: %s", exc)
        ctx.update({"corrections": corrections, "learned_rules": learned_rules,
                    "metrics": metrics, "total_corrections": total_corrections})

    elif panel_id == "prompts":
        pass  # JS fetches via /api/v1/prompts

    elif panel_id == "corpus":
        pass  # JS fetches via /api/v1/corpus/terms

    elif panel_id == "settings":
        gpu = {}
        models = []
        dir_stats = []
        try:
            gpu = await asyncio.to_thread(get_gpu_info)
            if MODELS_YAML.exists():
                raw = yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8")) or {}
                cache_dir = os.environ.get("LEGALMIND_MODEL_CACHE") or raw.get("model_cache_dir", "")
                for key, val in raw.get("local_models", {}).items():
                    if isinstance(val, dict):
                        gguf_file = val.get("gguf_model", "")
                        file_path = str(Path(cache_dir) / gguf_file) if gguf_file and cache_dir else gguf_file
                        models.append({"name": val.get("name", key), "role": val.get("role", ""),
                                       "vram_gb": val.get("vram_gb", ""), "file_path": file_path,
                                       "file_exists": Path(file_path).exists() if file_path else False})
            from config.paths import UPLOADS_DIR, CHROMA_DIR
            for name, path in [("Uploads", UPLOADS_DIR), ("Processed", PROCESSED_DIR),
                               ("Corrections", CORRECTIONS_DIR), ("ChromaDB", CHROMA_DIR)]:
                exists = path.exists()
                dir_stats.append({"name": name, "path": str(path), "exists": exists,
                                  "file_count": len(list(path.iterdir())) if exists else 0})
        except Exception as exc:
            logger.warning("Settings context: %s", exc)
        ctx.update({"gpu_info": gpu, "models": models, "dir_stats": dir_stats})

    return templates.TemplateResponse(request, f"panels/{panel_id}.html", ctx)


@app.get("/documents")
async def page_documents(request: Request):
    documents = []
    try:
        for path in sorted(PROCESSED_DIR.glob("*.json"), reverse=True):
            if path.stem.startswith("draft_"):
                continue
            try:
                documents.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                pass
    except Exception as exc:
        logger.warning("Documents page load failed: %s", exc)
    return templates.TemplateResponse(request, "documents.html", {
        "active_page": "documents",
        "panels": PANELS, "active_panel": "library",
        "documents": documents, "draft_types": get_supported_types(),
    })


@app.get("/documents/{doc_id}/draft")
async def page_draft(request: Request, doc_id: str, draft_id: Optional[str] = None):
    from code.pipeline.ingestion import ProcessedDocument
    doc = None
    draft = None
    citations = {}
    verification_summary = {}
    draft_type_label = ""
    try:
        doc = await asyncio.to_thread(ProcessedDocument.load, doc_id)
        if draft_id:
            key = draft_id if draft_id.startswith("draft_") else f"draft_{draft_id}"
            path = PROCESSED_DIR / f"{key}.json"
            if path.exists():
                draft = json.loads(path.read_text(encoding="utf-8"))
                citations = draft.get("citations", {})
                draft_type_label = get_draft_label(draft.get("draft_type", ""))
                verification_summary = draft.get("firewall_summary", {})
        else:
            # Auto-find the most recently *generated* draft. Sort by mtime,
            # not filename — alphabetical order would arbitrarily prefer
            # e.g. "title_review_summary" over "case_fact_summary"
            # regardless of which was actually produced for this document.
            draft_paths = sorted(
                PROCESSED_DIR.glob(f"draft_{doc_id}_*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if draft_paths:
                draft = json.loads(draft_paths[0].read_text(encoding="utf-8"))
                citations = draft.get("citations", {})
                draft_type_label = get_draft_label(draft.get("draft_type", ""))
                verification_summary = draft.get("firewall_summary", {})
    except Exception as exc:
        logger.warning("Draft page load failed: %s", exc)
    return templates.TemplateResponse(request, "drafts.html", {
        "active_page": "documents",
        "panels": PANELS, "active_panel": "library",
        "doc": doc, "draft": draft, "doc_id": doc_id,
        "draft_type_label": draft_type_label, "citations": citations,
        "verification_summary": verification_summary,
        "draft_types": get_supported_types(),
    })


@app.get("/audit")
async def page_audit(request: Request):
    corrections = []
    learned_rules = []
    metrics = {}
    total_corrections = 0
    try:
        store = CorrectionStore()
        corrections = [_correction_to_dict(c) for c in await asyncio.to_thread(store.get_recent, 50)]
        total_corrections = await asyncio.to_thread(store.get_count)
        if LEARNED_RULES_YAML.exists():
            data = yaml.safe_load(LEARNED_RULES_YAML.read_text(encoding="utf-8")) or {}
            learned_rules = data.get("rules", [])
        by_draft_type: dict = {}
        by_correction_type: dict = {}
        for t in get_supported_types():
            count = await asyncio.to_thread(store.get_count, t["id"])
            if count:
                by_draft_type[t["id"]] = count
        # Aggregate correction_type breakdown from the full recent set
        for c in await asyncio.to_thread(store.get_recent, 500):
            ct = getattr(c, "correction_type", None) or "unknown"
            by_correction_type[ct] = by_correction_type.get(ct, 0) + 1
        last_extraction_date = None
        active_exemplars = 0
        if LEARNED_RULES_YAML.exists():
            rules_data = yaml.safe_load(LEARNED_RULES_YAML.read_text(encoding="utf-8")) or {}
            last_extraction_date = rules_data.get("last_extraction")
            active_exemplars = rules_data.get("total_corrections_processed", 0)
        metrics = {
            "by_draft_type": by_draft_type,
            "by_correction_type": by_correction_type,
            "last_extraction_date": str(last_extraction_date) if last_extraction_date else None,
            "active_exemplars": active_exemplars,
        }
    except Exception as exc:
        logger.warning("Audit page load failed: %s", exc)
    return templates.TemplateResponse(request, "audit.html", {
        "active_page": "audit",
        "panels": PANELS, "active_panel": "audit",
        "corrections": corrections, "learned_rules": learned_rules,
        "metrics": metrics, "total_corrections": total_corrections,
    })


@app.get("/settings")
async def page_settings(request: Request):
    gpu = {}
    models = []
    dir_stats = []
    try:
        gpu = await asyncio.to_thread(get_gpu_info)
        if MODELS_YAML.exists():
            raw = yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8")) or {}
            cache_dir = os.environ.get("LEGALMIND_MODEL_CACHE") or raw.get("model_cache_dir", "")
            for key, val in raw.get("local_models", {}).items():
                if isinstance(val, dict):
                    gguf_file = val.get("gguf_model", "")
                    file_path = str(Path(cache_dir) / gguf_file) if gguf_file and cache_dir else gguf_file
                    file_exists = Path(file_path).exists() if file_path else False
                    models.append({
                        "name": val.get("name", key),
                        "role": val.get("role", ""),
                        "vram_gb": val.get("vram_gb", ""),
                        "file_path": file_path,
                        "file_exists": file_exists,
                    })
        from config.paths import UPLOADS_DIR, CHROMA_DIR
        for name, path in [
            ("Uploads", UPLOADS_DIR), ("Processed", PROCESSED_DIR),
            ("Corrections", CORRECTIONS_DIR), ("ChromaDB", CHROMA_DIR),
        ]:
            exists = path.exists()
            dir_stats.append({
                "name": name, "path": str(path), "exists": exists,
                "file_count": len(list(path.iterdir())) if exists else 0,
            })
    except Exception as exc:
        logger.warning("Settings page load failed: %s", exc)
    return templates.TemplateResponse(request, "settings.html", {
        "active_page": "settings",
        "panels": PANELS, "active_panel": "settings",
        "gpu_info": gpu, "models": models, "dir_stats": dir_stats,
    })


# ---------------------------------------------------------------------------
# Document API
# ---------------------------------------------------------------------------


@app.post("/api/v1/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    draft_type: str = Form(...),
) -> dict:
    # Stash the file payload BEFORE we wait on the lock — otherwise the
    # UploadFile's spooled-temp buffer may be invalidated by the time we
    # actually read it (some ASGI servers reap it after the handler yields).
    filename = file.filename
    payload = await file.read()

    # Hold the lock for the full inline ingest+draft pipeline. If another
    # upload is in flight, this request blocks here instead of returning
    # 409. From the client's POV it's a slow upload, not a rejection —
    # the SSE stream keeps showing live progress for whichever upload is
    # currently active.
    async with _pipeline_lock:
        # Wrap the already-buffered bytes in a fake "file-like" so
        # _run_upload can keep its same signature.
        class _BufferedFile:
            def __init__(self, name: str, data: bytes):
                self.filename = name
                self._data = data
            async def read(self) -> bytes:
                return self._data
        return await _run_upload(_BufferedFile(filename, payload), draft_type)


async def _run_upload(file: UploadFile, draft_type: str) -> dict:
    # Stage 1: ingest — save file to disk
    await _sse_broadcast("ingest", "active", "Saving file…")
    dest = UPLOADS_DIR / file.filename
    try:
        contents = await file.read()
        dest.write_bytes(contents)
    except Exception as exc:
        await _sse_broadcast("ingest", "error", str(exc))
        raise HTTPException(status_code=500, detail=f"File save failed: {exc}") from exc
    await _sse_broadcast("ingest", "complete", "File saved")

    # Start every upload from a clean GPU. The previous document's OCR /
    # extraction model can leave the ModelManager's VRAM budget maxed; the
    # next upload's model load then fails (VRAMBudgetExceeded → empty OCR →
    # no draft → "Generation skipped"). Reclearing here makes each document
    # independent. (The end-of-upload reclear only helps the next *draft
    # job*, not the next upload's own OCR/extract stage.)
    try:
        from code.llm_interface.model_manager import ModelManager
        await asyncio.to_thread(ModelManager.instance().reclear_gpu)
    except Exception as exc:
        logger.warning("Pre-ingest GPU reclear failed: %s", exc)

    # Stage 2: ocr + extract — run full ingestion pipeline in thread
    await _sse_broadcast("ocr", "active", "Running OCR & text extraction…")
    try:
        ingester = DocumentIngester()
        doc = await asyncio.to_thread(ingester.process, str(dest), draft_type)
    except Exception as exc:
        await _sse_broadcast("ocr", "error", str(exc))
        logger.exception("Ingestion failed for %s", file.filename)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc
    await _sse_broadcast("ocr", "complete", "Text extracted")
    await _sse_broadcast("extract", "complete", f"{doc.page_count} page(s) structured")

    # Stage 3: index — build BM25 index
    await _sse_broadcast("index", "active", "Indexing chunks…")
    try:
        indexer = DocumentIndexer()
        await asyncio.wait_for(
            asyncio.to_thread(indexer.index_document, doc.id, doc.chunks),
            timeout=60.0,
        )
    except asyncio.TimeoutError:
        logger.warning("Indexing timed out for %s", doc.id)
    except Exception as exc:
        logger.warning("Indexing failed for %s: %s", doc.id, exc)
    await _sse_broadcast("index", "complete", f"{len(doc.chunks)} chunks indexed")

    # Stage 4: retrieve — search for evidence passages
    await _sse_broadcast("retrieve", "active", "Retrieving evidence…")
    from code.retrieval.searcher import EvidenceSearcher
    from code.retrieval.evidence import EvidencePackager
    evidence_package = None
    try:
        searcher = EvidenceSearcher()
        packager = EvidencePackager()
        query_map = {
            "case_fact_summary": "parties claims evidence dates procedural history relief",
            "title_review_summary": "property deed title transfer ownership chain lien encumbrance",
            "notice_summary": "notice deadline requirement compliance response demand",
            "document_checklist": "document filing record deed agreement contract certificate",
        }
        query = query_map.get(draft_type, "legal document content")
        chunks = await asyncio.to_thread(searcher.search, query, [doc.id], 8)
        evidence_package = packager.package(chunks)
        await _sse_broadcast("retrieve", "complete",
                             f"{evidence_package.chunk_count} evidence passages retrieved")
    except Exception as exc:
        logger.warning("Evidence retrieval failed for %s: %s", doc.id, exc)
        await _sse_broadcast("retrieve", "complete", "Retrieval skipped")

    # Stage 5: generate — produce grounded draft
    await _sse_broadcast("generate", "active", f"Generating {draft_type.replace('_', ' ')}…")
    draft_dict = None
    try:
        gen = DraftGenerator()
        output = await asyncio.to_thread(
            gen.generate_draft, doc.id, draft_type, doc.full_text, 8
        )
        draft_dict = _draft_to_dict(output)
        await _sse_broadcast("generate", "complete", "Draft generated")
    except Exception as exc:
        logger.warning("Draft generation failed for %s: %s", doc.id, exc)
        await _sse_broadcast("generate", "complete", "Generation skipped (CPU mode)")

    # Stage 6: verify — run firewall checks
    await _sse_broadcast("verify", "active", "Verifying grounding…")
    if draft_dict:
        try:
            runner = FirewallRunner()
            citation_map = {}
            if evidence_package:
                citation_map = evidence_package.citation_map
            results = await asyncio.to_thread(
                runner.verify_draft,
                draft_dict.get("content_markdown", ""),
                doc.structured_fields,
                citation_map,
                doc.confidence,
            )
            fw_summary = runner.get_summary(results)
            draft_dict["firewall_results"] = [_vr_to_dict(r) for r in results]
            draft_dict["firewall_summary"] = fw_summary
            draft_path = PROCESSED_DIR / f"draft_{doc.id}_{draft_type}.json"
            draft_path.write_text(json.dumps(draft_dict, default=str), encoding="utf-8")
            verified = fw_summary.get("verified", 0)
            total = fw_summary.get("total", 0)
            await _sse_broadcast("verify", "complete",
                                 f"Verified: {verified}/{total} claims grounded")
        except Exception as exc:
            logger.warning("Firewall check failed: %s", exc)
            await _sse_broadcast("verify", "complete", "Verification skipped")
    else:
        await _sse_broadcast("verify", "complete", "No draft to verify")

    # Free GPU after inline generation so the next subprocess-based
    # draft job (Generate button) has a clear runway on the RTX 3080.
    try:
        from code.llm_interface.model_manager import ModelManager
        await asyncio.to_thread(ModelManager.instance().reclear_gpu)
    except Exception as exc:
        logger.warning("Post-upload GPU reclear failed: %s", exc)

    # Signal complete
    await _sse_broadcast("all", "complete", "Processing complete")

    result = _doc_to_dict(doc)
    if draft_dict:
        result["draft"] = draft_dict
    return result


@app.get("/api/v1/documents")
async def list_documents() -> list[dict]:
    docs = []
    for path in sorted(PROCESSED_DIR.glob("*.json")):
        if path.stem.startswith("draft_"):
            continue
        try:
            docs.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.warning("Could not read %s: %s", path, exc)
    return docs


@app.get("/api/v1/documents/{doc_id}")
async def get_document(doc_id: str) -> dict:
    try:
        doc = await asyncio.to_thread(
            lambda: __import__(
                "code.pipeline.ingestion", fromlist=["ProcessedDocument"]
            ).ProcessedDocument.load(doc_id)
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return _doc_to_dict(doc)


@app.get("/api/v1/documents/{doc_id}/detail")
async def get_document_detail_partial(request: Request, doc_id: str):
    """HTML partial used by HTMX to load the document detail panel on the
    Documents page.  Returns an HTML fragment rendered from the document data."""
    from code.pipeline.ingestion import ProcessedDocument

    doc = None
    drafts = []
    try:
        doc = await asyncio.to_thread(ProcessedDocument.load, doc_id)
        for path in sorted(PROCESSED_DIR.glob(f"draft_{doc_id}_*.json"), reverse=True):
            try:
                drafts.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                pass
    except Exception as exc:
        logger.warning("Document detail partial failed for %s: %s", doc_id, exc)

    if doc is None:
        return templates.TemplateResponse(
            request, "partials/document_detail.html",
            {"doc": None, "drafts": [], "doc_id": doc_id,
             "draft_types": get_supported_types()},
        )
    return templates.TemplateResponse(
        request, "partials/document_detail.html",
        {"doc": _doc_to_dict(doc), "drafts": drafts, "doc_id": doc_id,
         "draft_types": get_supported_types()},
    )


# ---------------------------------------------------------------------------
# Job Queue Worker
# ---------------------------------------------------------------------------


async def _job_queue_worker():
    """Background worker: pulls jobs from queue, runs one at a time."""
    global _job_cancel_requested
    while True:
        await _job_queue_event.wait()
        _job_queue_event.clear()

        while True:
            job = None
            with _job_queue_lock:
                if not _job_queue_state["queued"]:
                    break
                job = _job_queue_state["queued"].pop(0)
                job["started_at"] = datetime.now().isoformat()
                _job_queue_state["in_progress"] = job
                _save_queue_state()

            _job_cancel_requested = False
            await _sse_broadcast("queue", "job_started", json.dumps({
                "job_id": job["id"], "document_id": job["document_id"],
                "draft_type": job["draft_type"], "filename": job["filename"],
            }))

            start_time = datetime.now()
            try:
                await _run_draft_job(job)
                duration = (datetime.now() - start_time).total_seconds()
                with _job_queue_lock:
                    _job_queue_state["in_progress"] = None
                    _job_queue_state["done"].insert(0, {
                        **job, "completed_at": datetime.now().isoformat(),
                        "duration_sec": round(duration, 1), "status": "done",
                    })
                    if len(_job_queue_state["done"]) > 50:
                        _job_queue_state["done"] = _job_queue_state["done"][:50]
                    _save_queue_state()
                await _sse_broadcast("queue", "job_done", json.dumps({
                    "job_id": job["id"], "document_id": job["document_id"],
                    "duration_sec": round(duration, 1),
                }))
            except Exception as exc:
                logger.exception("Job %s failed", job["id"])
                with _job_queue_lock:
                    _job_queue_state["in_progress"] = None
                    _job_queue_state["failed"].insert(0, {
                        **job, "error": str(exc),
                        "completed_at": datetime.now().isoformat(), "status": "failed",
                    })
                    if len(_job_queue_state["failed"]) > 50:
                        _job_queue_state["failed"] = _job_queue_state["failed"][:50]
                    _save_queue_state()
                await _sse_broadcast("queue", "job_failed", json.dumps({
                    "job_id": job["id"], "error": str(exc)[:200],
                }))
                await _sse_broadcast("all", "complete", "Job failed: " + str(exc)[:100])


async def _run_draft_job(job: dict):
    """Execute a single draft generation job in a subprocess.

    llama-cpp-python's CUDA context / mmap locks / chat_handler buffers leak
    on Windows. Per-job subprocess isolation forces OS-level cleanup on
    subprocess exit. The webapp stays alive, only the worker subprocess
    turns over.
    """
    global _job_cancel_requested
    document_id = job["document_id"]
    draft_type = job["draft_type"]

    import subprocess as _sp
    import tempfile

    repo_root = Path(__file__).resolve().parent.parent

    # Verify document exists before launching subprocess
    doc_path = PROCESSED_DIR / f"{document_id}.json"
    if not doc_path.exists():
        await _sse_broadcast("retrieve", "error", "Document not found")
        raise RuntimeError(f"Document not found: {document_id}")

    # Free webapp-side GPU before launching subprocess. The webapp's
    # ModelManager may still hold the Qwen3-VL model from a prior ingestion
    # (~6GB). The subprocess loads its own 6.1GB copy — with the webapp also
    # holding 6.1GB, the RTX 3080's 10GB total is exhausted and llama.cpp
    # aborts on CUDA OOM (exit code 1, empty stderr). Reclearing here drops
    # the webapp's GPU allocation so the subprocess has a clear runway.
    try:
        from code.llm_interface.model_manager import ModelManager
        await asyncio.to_thread(ModelManager.instance().reclear_gpu)
    except Exception as exc:
        logger.warning("Pre-subprocess GPU reclear failed: %s", exc)

    await _sse_broadcast("retrieve", "active", "Loading document & retrieving evidence…")

    # Create temp file for result + log file for subprocess stdout/stderr
    fd, result_path = tempfile.mkstemp(
        prefix=f"legalmind_draft_{document_id}_", suffix=".json"
    )
    os.close(fd)
    log_path = Path(result_path).with_suffix(".log")

    # Python code to run in subprocess — full pipeline: retrieve → generate → verify
    runner_code = f"""
import sys, json, os, time
sys.path.insert(0, r{str(repo_root)!r})
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

if sys.platform == "win32":
    cuda_bin = r"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6\\bin"
    if os.path.exists(cuda_bin):
        os.add_dll_directory(cuda_bin)
        os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")

result = {{"status": "failed", "error": None, "duration_sec": 0}}
t0 = time.time()
try:
    from code.pipeline.ingestion import ProcessedDocument
    from code.generation.drafter import DraftGenerator
    from code.firewall.runner import FirewallRunner

    doc = ProcessedDocument.load({document_id!r})
    if doc is None:
        raise RuntimeError("Document not found")

    # Generate
    gen = DraftGenerator()
    output = gen.generate_draft({document_id!r}, {draft_type!r}, doc.full_text, 8)

    # Verify
    runner = FirewallRunner()
    fw_results = runner.verify_draft(
        output.content_markdown,
        doc.structured_fields,
        output.citations or {{}},
        doc.confidence,
    )
    fw_summary = runner.get_summary(fw_results)

    # Build result dict
    draft_dict = {{
        "id": f"draft_{document_id}_{draft_type}",
        "draft_type": output.draft_type,
        "document_id": output.document_id,
        "content_markdown": output.content_markdown,
        "content_structured": output.content_structured,
        "citations": output.citations,
        "verification_status": output.verification_status,
        "generation_timestamp": str(output.generation_timestamp),
        "exemplars_used": output.exemplars_used,
        "rules_applied": output.rules_applied,
        "evidence_count": output.evidence_count,
        "confidence_overall": output.confidence_overall,
        "firewall_results": [{{
            "field_path": r.field_path, "status": r.status,
            "confidence_score": r.confidence_score, "confidence_level": r.confidence_level,
            "similarity": r.similarity, "evidence_snippet": r.evidence_snippet,
            "reasons": r.reasons,
        }} for r in fw_results],
        "firewall_summary": fw_summary,
    }}

    # Save draft to disk
    from config.paths import PROCESSED_DIR
    draft_path = PROCESSED_DIR / f"draft_{document_id}_{draft_type}.json"
    draft_path.write_text(json.dumps(draft_dict, default=str), encoding="utf-8")

    result["status"] = "done"
    result["duration_sec"] = round(time.time() - t0, 1)

except Exception as exc:
    result["status"] = "failed"
    result["error"] = f"{{type(exc).__name__}}: {{exc}}"
    result["duration_sec"] = round(time.time() - t0, 1)

with open(r{result_path!r}, "w", encoding="utf-8") as f:
    json.dump(result, f)
"""

    env = os.environ.copy()
    creationflags = 0
    if sys.platform == "win32":
        import subprocess as _sp2
        creationflags = _sp2.CREATE_NEW_PROCESS_GROUP

    # Write runner to a tempfile rather than passing via -c. llama.cpp emits
    # a large volume to stderr while loading; if stderr were a PIPE, the
    # ~64KB Windows pipe buffer fills, the subprocess blocks writing, and
    # the whole job deadlocks. Redirecting to a log file avoids the buffer
    # entirely and gives us a real artifact to surface on failure.
    fd_py, runner_path = tempfile.mkstemp(
        prefix=f"legalmind_runner_{document_id}_", suffix=".py"
    )
    os.close(fd_py)
    Path(runner_path).write_text(runner_code, encoding="utf-8")
    log_fh = open(log_path, "w", encoding="utf-8")

    proc = _sp.Popen(
        [sys.executable, "-u", runner_path],
        env=env,
        cwd=str(repo_root),
        stdout=log_fh, stderr=_sp.STDOUT,
        creationflags=creationflags,
    )

    # Register for cancellation
    _job_active_proc["proc"] = proc
    _job_active_proc["job_id"] = job["id"]

    await _sse_broadcast("retrieve", "complete", "Subprocess launched")
    await _sse_broadcast("generate", "active",
                         f"Generating {draft_type.replace('_', ' ')} (model loading)…")

    try:
        # Poll subprocess — check for cancel every 2 seconds
        while proc.poll() is None:
            if _job_cancel_requested:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except _sp.TimeoutExpired:
                    proc.kill()
                raise RuntimeError("Cancelled by operator")
            await asyncio.sleep(2)

        # Close the log handle so the subprocess's final writes flush to disk
        try:
            log_fh.close()
        except Exception:
            pass

        if proc.returncode != 0:
            log_excerpt = _tail_log(log_path, head_lines=80, tail_lines=80)
            raise RuntimeError(
                f"Subprocess exit code {proc.returncode}. "
                f"Log excerpt (full log: {log_path}):\n{log_excerpt}"
            )

        # Read result
        if not Path(result_path).exists() or Path(result_path).stat().st_size == 0:
            log_excerpt = _tail_log(log_path, head_lines=40, tail_lines=80)
            raise RuntimeError(
                f"Subprocess exited 0 but produced no result file. "
                f"Log excerpt (full log: {log_path}):\n{log_excerpt}"
            )
        result_data = json.loads(Path(result_path).read_text(encoding="utf-8"))

        if result_data["status"] == "done":
            await _sse_broadcast("generate", "complete", "Draft generated")
            await _sse_broadcast("verify", "complete",
                                 f"Complete ({result_data['duration_sec']}s)")
            await _sse_broadcast("all", "complete", "Processing complete")
        else:
            error_msg = result_data.get("error", "Unknown error")
            log_excerpt = _tail_log(log_path, head_lines=20, tail_lines=40)
            await _sse_broadcast("generate", "error", error_msg)
            raise RuntimeError(f"{error_msg}\nLog excerpt:\n{log_excerpt}")

    finally:
        _job_active_proc.clear()
        try:
            log_fh.close()
        except Exception:
            pass
        try:
            Path(result_path).unlink(missing_ok=True)
        except Exception:
            pass
        try:
            Path(runner_path).unlink(missing_ok=True)
        except Exception:
            pass
        # Keep the log file on failure so the operator can inspect it.
        # Successful runs delete it to avoid clutter.
        try:
            if proc.returncode == 0 and Path(log_path).exists():
                Path(log_path).unlink(missing_ok=True)
        except Exception:
            pass


def _tail_log(log_path: Path, head_lines: int = 40, tail_lines: int = 80) -> str:
    """Return the first head_lines and last tail_lines of a log file."""
    try:
        if not Path(log_path).exists():
            return "(log file missing)"
        lines = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()
        if len(lines) <= head_lines + tail_lines:
            return "\n".join(lines)
        head = lines[:head_lines]
        tail = lines[-tail_lines:]
        return "\n".join(head + [f"... ({len(lines) - head_lines - tail_lines} lines elided) ..."] + tail)
    except Exception as exc:
        return f"(could not read log: {exc})"


# ---------------------------------------------------------------------------
# Draft API (queue-based)
# ---------------------------------------------------------------------------


@app.post("/api/v1/drafts/generate")
async def generate_draft(body: dict) -> dict:
    """Enqueue a draft generation job. Returns immediately with job ID."""
    document_id: str = body.get("document_id", "")
    draft_type: str = body.get("draft_type", "")
    if not document_id or not draft_type:
        raise HTTPException(status_code=422, detail="document_id and draft_type required")

    # Validate document exists
    from code.pipeline.ingestion import ProcessedDocument
    doc = await asyncio.to_thread(ProcessedDocument.load, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    # Check for duplicates in queue
    with _job_queue_lock:
        for j in _job_queue_state["queued"]:
            if j["document_id"] == document_id and j["draft_type"] == draft_type:
                return {"id": j["id"], "status": "already_queued", "position": _job_queue_state["queued"].index(j) + 1}
        ip = _job_queue_state["in_progress"]
        if ip and ip["document_id"] == document_id and ip["draft_type"] == draft_type:
            return {"id": ip["id"], "status": "in_progress"}

    job_id = f"job_{_uuid.uuid4().hex[:12]}"
    job = {
        "id": job_id,
        "document_id": document_id,
        "draft_type": draft_type,
        "filename": doc.filename,
        "requested_at": datetime.now().isoformat(),
    }

    with _job_queue_lock:
        _job_queue_state["queued"].append(job)
        position = len(_job_queue_state["queued"])
        _save_queue_state()

    _job_queue_event.set()

    await _sse_broadcast("queue", "job_queued", json.dumps({
        "job_id": job_id, "document_id": document_id,
        "draft_type": draft_type, "filename": doc.filename,
        "position": position,
    }))

    return {"id": job_id, "status": "queued", "position": position}


@app.get("/api/v1/queue")
async def get_queue_state() -> dict:
    """Return current queue state: queued, in_progress, done, failed."""
    with _job_queue_lock:
        return {
            "queued": list(_job_queue_state["queued"]),
            "in_progress": _job_queue_state["in_progress"],
            "done": _job_queue_state["done"][:10],
            "failed": _job_queue_state["failed"][:10],
        }


@app.post("/api/v1/queue/cancel")
async def cancel_current_job() -> dict:
    """Cancel the currently running job (terminates subprocess)."""
    global _job_cancel_requested
    with _job_queue_lock:
        ip = _job_queue_state["in_progress"]
        if not ip:
            raise HTTPException(status_code=409, detail="No job currently running")
    _job_cancel_requested = True
    # Also terminate the subprocess immediately
    proc = _job_active_proc.get("proc")
    if proc and proc.poll() is None:
        proc.terminate()
    await _sse_broadcast("queue", "cancel_requested", json.dumps({"job_id": ip["id"]}))
    return {"ok": True, "job_id": ip["id"], "message": "Cancel requested"}


@app.delete("/api/v1/queue/{job_id}")
async def remove_queued_job(job_id: str) -> dict:
    """Remove a job from the queue (only if still queued, not in_progress)."""
    with _job_queue_lock:
        for i, j in enumerate(_job_queue_state["queued"]):
            if j["id"] == job_id:
                _job_queue_state["queued"].pop(i)
                return {"ok": True, "removed": job_id}
    raise HTTPException(status_code=404, detail="Job not found in queue")


@app.delete("/api/v1/queue")
async def clear_queue(scope: str = "queued") -> dict:
    """Clear queue entries. scope: queued, done, failed, all."""
    with _job_queue_lock:
        if scope in ("queued", "all"):
            _job_queue_state["queued"].clear()
        if scope in ("done", "all"):
            _job_queue_state["done"].clear()
        if scope in ("failed", "all"):
            _job_queue_state["failed"].clear()
    return {"ok": True, "cleared": scope}


@app.get("/api/v1/drafts/{draft_id}")
async def get_draft(draft_id: str) -> dict:
    # draft_id may be the composite key or already include the prefix
    if not draft_id.startswith("draft_"):
        draft_id = f"draft_{draft_id}"
    path = PROCESSED_DIR / f"{draft_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Draft not found")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Corrections API
# ---------------------------------------------------------------------------


@app.post("/api/v1/corrections")
async def save_correction(body: dict) -> dict:
    # editor.js only knows the field_path and the markdown text it just edited
    # — it does NOT have the relevant source-text chunk. Without a non-empty
    # source_ocr_chunk the exemplar retriever (BM25 over chunk text) can
    # never match this correction against future similar docs, which breaks
    # the learning loop. Fall back to the document's full_text so retrieval
    # has something meaningful to score against.
    if not body.get("source_ocr_chunk"):
        doc_id = body.get("document_id", "")
        if doc_id:
            try:
                from code.pipeline.ingestion import ProcessedDocument
                doc = await asyncio.to_thread(ProcessedDocument.load, doc_id)
                if doc and doc.full_text:
                    body["source_ocr_chunk"] = doc.full_text[:1500]
            except Exception as exc:
                logger.warning("Could not enrich correction with source text: %s", exc)

    try:
        correction = Correction(**body)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid correction payload: {exc}") from exc

    try:
        store = CorrectionStore()
        correction_id = await asyncio.to_thread(store.save_correction, correction)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Pattern extraction and consolidation triggered in background. We
    # instantiate the checker AFTER the save above so it sees the fresh
    # count — the old code created the extractor in module init and asked
    # `should_trigger()` before the new correction had been persisted, which
    # made the trigger race-prone.
    try:
        extractor = PatternExtractor()
        if await asyncio.to_thread(extractor.should_trigger):
            asyncio.create_task(_run_pattern_extraction(extractor))
    except Exception as exc:
        logger.warning("Pattern extractor check failed: %s", exc)

    try:
        consolidator = PromptConsolidator()
        if await asyncio.to_thread(consolidator.should_trigger):
            asyncio.create_task(_run_consolidation(consolidator))
    except Exception as exc:
        logger.warning("Prompt consolidator check failed: %s", exc)

    return {"id": correction_id, "status": "saved"}


@app.get("/api/v1/corrections")
async def list_corrections(draft_type: Optional[str] = None) -> list[dict]:
    try:
        store = CorrectionStore()
        if draft_type:
            corrections = await asyncio.to_thread(store.get_all_active, draft_type)
        else:
            corrections = await asyncio.to_thread(store.get_recent, 20)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return [_correction_to_dict(c) for c in corrections]


# ---------------------------------------------------------------------------
# Learning API
# ---------------------------------------------------------------------------


async def _run_extraction_subprocess() -> dict:
    """Run pattern extraction in an isolated subprocess.

    extract_patterns() loads the reasoning LLM. Doing that in-process (even
    via to_thread) contends with the webapp's own ModelManager on the single
    RTX 3080 and, on Windows, llama-cpp's CUDA context / mmap locks routinely
    deadlock or OOM-kill the worker — which takes the whole server down.
    Same failure mode and same fix as draft generation: spawn a short-lived
    subprocess, let the OS reclaim the CUDA context on its exit.
    """
    import subprocess as _sp
    import tempfile

    repo_root = Path(__file__).resolve().parent.parent

    # Free any model the webapp is holding so the subprocess has a clear GPU.
    try:
        from code.llm_interface.model_manager import ModelManager
        await asyncio.to_thread(ModelManager.instance().reclear_gpu)
    except Exception as exc:
        logger.warning("Pre-extraction GPU reclear failed: %s", exc)

    fd, result_path = tempfile.mkstemp(prefix="legalmind_extract_", suffix=".json")
    os.close(fd)
    log_path = Path(result_path).with_suffix(".log")

    runner_code = f"""
import sys, json, os, time
sys.path.insert(0, r{str(repo_root)!r})
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
if sys.platform == "win32":
    cuda_bin = r"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6\\bin"
    if os.path.exists(cuda_bin):
        os.add_dll_directory(cuda_bin)
        os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")
result = {{"status": "failed", "error": None, "rules": [], "duration_sec": 0}}
t0 = time.time()
try:
    from code.learning.pattern_extractor import PatternExtractor
    rules = PatternExtractor().extract_patterns(True)
    result["status"] = "done"
    result["rules"] = list(rules or [])
    result["duration_sec"] = round(time.time() - t0, 1)
except Exception as exc:
    result["status"] = "failed"
    result["error"] = f"{{type(exc).__name__}}: {{exc}}"
    result["duration_sec"] = round(time.time() - t0, 1)
with open(r{result_path!r}, "w", encoding="utf-8") as f:
    json.dump(result, f)
"""
    fd_py, runner_path = tempfile.mkstemp(prefix="legalmind_extract_runner_", suffix=".py")
    os.close(fd_py)
    Path(runner_path).write_text(runner_code, encoding="utf-8")
    log_fh = open(log_path, "w", encoding="utf-8")

    creationflags = 0
    if sys.platform == "win32":
        creationflags = _sp.CREATE_NEW_PROCESS_GROUP

    proc = _sp.Popen(
        [sys.executable, "-u", runner_path],
        env=os.environ.copy(), cwd=str(repo_root),
        stdout=log_fh, stderr=_sp.STDOUT, creationflags=creationflags,
    )
    try:
        while proc.poll() is None:
            await asyncio.sleep(2)
        try:
            log_fh.close()
        except Exception:
            pass
        if proc.returncode != 0 or not Path(result_path).exists():
            tail = _tail_log(log_path, 40, 60)
            raise RuntimeError(
                f"Extraction subprocess exit {proc.returncode}. Log:\n{tail}"
            )
        data = json.loads(Path(result_path).read_text(encoding="utf-8"))
        if data["status"] != "done":
            raise RuntimeError(data.get("error", "unknown extraction error"))
        return data
    finally:
        try:
            log_fh.close()
        except Exception:
            pass
        for p in (result_path, runner_path):
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass
        try:
            if proc.returncode == 0:
                Path(log_path).unlink(missing_ok=True)
        except Exception:
            pass


@app.post("/api/v1/learning/rules")
async def trigger_pattern_extraction() -> dict:
    """Manually trigger pattern extraction from accumulated corrections.

    Used by the Audit page's 'Trigger Pattern Extraction' button via HTMX.
    Runs in an isolated subprocess (loads the reasoning LLM) so it cannot
    take the server down. force=True so it runs regardless of count.
    """
    try:
        data = await _run_extraction_subprocess()
        rules = data.get("rules", [])
        return {"status": "ok", "rules_extracted": len(rules),
                "rules": rules,
                "message": f"Extraction complete — {len(rules)} rule(s) generated "
                           f"({data.get('duration_sec', 0)}s)."}
    except Exception as exc:
        logger.exception("Manual pattern extraction failed")
        raise HTTPException(status_code=500, detail=f"Pattern extraction failed: {exc}") from exc


@app.get("/api/v1/learning/rules")
async def get_learning_rules() -> dict:
    if not LEARNED_RULES_YAML.exists():
        return {"rules": []}
    try:
        data = yaml.safe_load(LEARNED_RULES_YAML.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {"rules": data}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/v1/learning/metrics")
async def get_learning_metrics() -> dict:
    try:
        store = CorrectionStore()
        supported = get_supported_types()
        per_type = {}
        for t in supported:
            tid = t["id"]
            per_type[tid] = {
                "label": t["label"],
                "count": await asyncio.to_thread(store.get_count, tid),
            }
        total = await asyncio.to_thread(store.get_count)
        return {"per_type": per_type, "total": total}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# System status API
# ---------------------------------------------------------------------------


@app.get("/api/v1/system/status")
async def system_status() -> dict:
    gpu: dict = {}
    try:
        gpu = await asyncio.to_thread(get_gpu_info)
    except Exception as exc:
        logger.warning("GPU info unavailable: %s", exc)

    models_config: dict = {}
    try:
        models_config = yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("Could not read models.yaml: %s", exc)

    correction_count = 0
    try:
        store = CorrectionStore()
        correction_count = await asyncio.to_thread(store.get_count)
    except Exception as exc:
        logger.warning("Correction count unavailable: %s", exc)

    return {
        "gpu": gpu,
        "models": models_config.get("local_models", {}),
        "mode": models_config.get("mode", "unknown"),
        "correction_count": correction_count,
        "supported_draft_types": get_supported_types(),
    }


@app.get("/api/v1/system/health")
async def system_health() -> dict:
    components = []
    healthy = True

    # Check GPU
    try:
        gpu = await asyncio.to_thread(get_gpu_info)
        gpu_ok = gpu.get("total_mb", 0) > 0
        components.append({
            "name": "GPU",
            "status": "ok" if gpu_ok else "warn",
            "detail": f"{gpu.get('name', 'Unknown')} — {gpu.get('vram_free_gb', 0):.1f}GB free" if gpu_ok else "No GPU detected",
        })
        # GPU missing is a warning, not a hard failure — CPU fallback is possible
    except Exception as e:
        components.append({"name": "GPU", "status": "fail", "detail": str(e)})
        healthy = False

    # Check models.yaml
    try:
        mode = "unknown"
        if MODELS_YAML.exists():
            raw = yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8")) or {}
            mode = raw.get("mode", "unknown")
        components.append({"name": "models.yaml mode", "status": "ok", "detail": f"mode = {mode}"})
    except Exception as e:
        components.append({"name": "models.yaml mode", "status": "fail", "detail": str(e)})
        healthy = False

    # Check corrections store
    try:
        store = CorrectionStore()
        count = await asyncio.to_thread(store.get_count)
        components.append({"name": "Correction Store", "status": "ok", "detail": f"{count} corrections"})
    except Exception as e:
        components.append({"name": "Correction Store", "status": "warn", "detail": str(e)})

    # Check ChromaDB path
    from config.paths import CHROMA_DIR
    chroma_ok = CHROMA_DIR.exists()
    components.append({
        "name": "ChromaDB",
        "status": "ok" if chroma_ok else "warn",
        "detail": "Ready" if chroma_ok else "Directory not yet created",
    })

    n_fail = sum(1 for c in components if c["status"] == "fail")
    n_warn = sum(1 for c in components if c["status"] == "warn")
    summary = f"{len(components)} components: {n_fail} failing, {n_warn} warnings" if (n_fail or n_warn) else f"All {len(components)} components healthy"

    return {
        "healthy": healthy,
        "summary": summary,
        "components": components,
        "pipeline": {"state": "idle", "label": "System ready"},
    }


@app.post("/api/v1/system/gpu_reset")
async def gpu_reset() -> dict:
    """Emergency GPU/VRAM reset — unloads all models, zeroes the VRAM budget.

    Use when the server is stuck in a VRAMBudgetExceeded state due to a timed-out
    or crashed pipeline run that left a model partially loaded.
    """
    try:
        from code.llm_interface.model_manager import ModelManager
        mgr = ModelManager.instance()
        await asyncio.to_thread(mgr.reclear_gpu)
        return {
            "ok": True,
            "vram_used_gb": mgr.vram_used_gb,
            "message": "GPU reset complete. VRAM budget zeroed.",
        }
    except Exception as exc:
        logger.exception("GPU reset failed")
        raise HTTPException(status_code=500, detail=f"GPU reset failed: {exc}") from exc


# ---------------------------------------------------------------------------
# SSE pipeline events
# ---------------------------------------------------------------------------


@app.get("/api/v1/events/pipeline")
async def pipeline_events(request: Request):
    """Stream pipeline events to the browser via Server-Sent Events.

    On connect: subscriber queue is created and a "connected/ready" sentinel is
    sent immediately so the browser knows the stream is live before it POSTs
    the upload. Events: {stage, status, message, ts}.
    Keepalive ping sent after 15s idle to prevent proxy timeout.
    """
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    _sse_subscribers.append(q)
    # Sentinel: tells the browser SSE is established — safe to start the upload
    await q.put(json.dumps({"stage": "connected", "status": "ready", "message": ""}))

    async def _event_generator() -> AsyncGenerator[dict, None]:
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield {"data": payload}
                    ev = json.loads(payload)
                    if ev.get("stage") == "all" and ev.get("status") == "complete":
                        return
                except asyncio.TimeoutError:
                    yield {"data": json.dumps({"stage": "ping", "status": "idle"})}
        finally:
            try:
                _sse_subscribers.remove(q)
            except ValueError:
                pass

    return EventSourceResponse(_event_generator())


# ---------------------------------------------------------------------------
# Prompts API
# ---------------------------------------------------------------------------


@app.get("/api/v1/prompts")
async def list_prompts() -> dict:
    prompts = []
    for path in sorted(PROMPTS_DIR.glob("*.txt")):
        content = path.read_text(encoding="utf-8")
        managed = "managed: true" in content[:200]
        prompts.append({
            "id": path.stem,
            "name": path.stem.replace("_", " ").title(),
            "path": str(path.relative_to(PROMPTS_DIR)),
            "managed": managed,
            "size": len(content),
        })
    return {"prompts": prompts}


@app.get("/api/v1/prompts/{prompt_id}")
async def get_prompt(prompt_id: str) -> dict:
    path = PROMPTS_DIR / f"{prompt_id}.txt"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Prompt not found")
    content = path.read_text(encoding="utf-8")
    managed = "managed: true" in content[:200]
    return {"id": prompt_id, "content": content, "managed": managed, "path": str(path)}


@app.put("/api/v1/prompts/{prompt_id}")
async def save_prompt(prompt_id: str, body: dict) -> dict:
    path = PROMPTS_DIR / f"{prompt_id}.txt"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Prompt not found")
    content = body.get("content", "")
    if not content:
        raise HTTPException(status_code=422, detail="content required")
    path.write_text(content, encoding="utf-8")
    return {"ok": True, "id": prompt_id}


# ---------------------------------------------------------------------------
# Corpus API
# ---------------------------------------------------------------------------


@app.get("/api/v1/corpus/terms")
async def get_corpus_terms() -> dict:
    from code.corpus.term_data import LEGAL_TERMS
    builtin = {cat: list(terms) for cat, terms in LEGAL_TERMS.items()}

    # Load learned terms
    learned = []
    if LEARNED_TERMS_FILE.exists():
        try:
            learned = json.loads(LEARNED_TERMS_FILE.read_text(encoding="utf-8"))
        except Exception:
            learned = []

    return {"builtin": builtin, "learned": learned, "total": sum(len(v) for v in builtin.values())}


@app.post("/api/v1/corpus/terms")
async def add_corpus_term(body: dict) -> dict:
    term = body.get("term", "").strip()
    category = body.get("category", "custom").strip()
    if not term:
        raise HTTPException(status_code=422, detail="term required")

    LEARNED_TERMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    learned = []
    if LEARNED_TERMS_FILE.exists():
        try:
            learned = json.loads(LEARNED_TERMS_FILE.read_text(encoding="utf-8"))
        except Exception:
            learned = []

    # Avoid duplicates
    if not any(t.get("term") == term for t in learned):
        learned.append({"term": term, "category": category, "added_at": datetime.now().isoformat()})
        LEARNED_TERMS_FILE.write_text(json.dumps(learned, indent=2), encoding="utf-8")

    return {"ok": True, "term": term}


@app.delete("/api/v1/corpus/terms/{term}")
async def delete_corpus_term(term: str) -> dict:
    if not LEARNED_TERMS_FILE.exists():
        raise HTTPException(status_code=404, detail="Term not found")

    learned = json.loads(LEARNED_TERMS_FILE.read_text(encoding="utf-8"))
    original_count = len(learned)
    learned = [t for t in learned if t.get("term") != term]

    if len(learned) == original_count:
        raise HTTPException(status_code=404, detail="Term not found in learned list")

    LEARNED_TERMS_FILE.write_text(json.dumps(learned, indent=2), encoding="utf-8")
    return {"ok": True}


# ---------------------------------------------------------------------------
# Page image API (for source verbatim human audit)
# ---------------------------------------------------------------------------


@app.get("/api/v1/documents/{doc_id}/pages/{page_num}")
async def get_document_page_image(doc_id: str, page_num: int):
    # Preferred path: PDFs get rendered to JPEGs at ingest time.
    page_path = PAGES_DIR / doc_id / f"page_{page_num:03d}.jpg"
    if page_path.exists():
        return FileResponse(str(page_path), media_type="image/jpeg")

    # Fallback: for image uploads (PNG/JPG/etc) the original upload IS the
    # single page — serve it directly from data/uploads/{filename}. We
    # consult the processed doc JSON for the original filename.
    if page_num == 1:
        doc_json = PROCESSED_DIR / f"{doc_id}.json"
        if doc_json.exists():
            try:
                meta = json.loads(doc_json.read_text(encoding="utf-8"))
                filename = meta.get("filename", "")
                if filename:
                    upload_path = UPLOADS_DIR / filename
                    if upload_path.exists():
                        suffix = upload_path.suffix.lower()
                        mime = {
                            ".png": "image/png",
                            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                            ".tif": "image/tiff", ".tiff": "image/tiff",
                            ".bmp": "image/bmp", ".webp": "image/webp",
                            ".gif": "image/gif",
                        }.get(suffix)
                        if mime:
                            return FileResponse(str(upload_path), media_type=mime)
            except Exception as exc:
                logger.warning("Page-image fallback failed for %s: %s", doc_id, exc)

    raise HTTPException(status_code=404, detail="Page image not found")


@app.get("/api/v1/documents/{doc_id}/text-span")
async def get_document_text_span(
    doc_id: str, start: int = 0, end: int = 0, pad: int = 600
) -> dict:
    """Return the cited text span plus surrounding context from full_text.

    Used by the citation eye-button for .txt documents (which have no page
    image): the modal renders {before}{span}{after} with the span
    highlighted, so the operator can see exactly where the claim came from.
    """
    doc_json = PROCESSED_DIR / f"{doc_id}.json"
    if not doc_json.exists():
        raise HTTPException(status_code=404, detail="Document not found")
    try:
        meta = json.loads(doc_json.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    full = meta.get("full_text", "") or ""
    n = len(full)
    if n == 0:
        raise HTTPException(status_code=404, detail="No document text available")

    # Clamp the requested range into the text
    s = max(0, min(start, n))
    e = max(s, min(end, n)) if end else s
    if e <= s:
        e = min(n, s + 400)  # no usable end → show a reasonable window

    ctx_start = max(0, s - pad)
    ctx_end = min(n, e + pad)

    return {
        "filename": meta.get("filename", ""),
        "before": full[ctx_start:s],
        "span": full[s:e],
        "after": full[e:ctx_end],
        "truncated_before": ctx_start > 0,
        "truncated_after": ctx_end < n,
        "char_start": s,
        "char_end": e,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _run_pattern_extraction(extractor: PatternExtractor) -> None:
    # Route through the same isolated subprocess as the manual endpoint —
    # extract_patterns() loads the reasoning LLM and must never run in the
    # server process (it crashes the worker on the shared GPU).
    try:
        data = await _run_extraction_subprocess()
        logger.info("Background pattern extraction produced %d rule(s)",
                    len(data.get("rules", [])))
    except Exception as exc:
        logger.warning("Background pattern extraction failed: %s", exc)


async def _run_consolidation(consolidator: PromptConsolidator) -> None:
    try:
        await asyncio.to_thread(consolidator.consolidate)
        logger.info("Prompt consolidation complete")
    except Exception as exc:
        logger.warning("Background consolidation failed: %s", exc)


def _doc_has_draft(doc_id: str) -> bool:
    """True if any generated draft file exists for this doc."""
    if not doc_id:
        return False
    try:
        return any(PROCESSED_DIR.glob(f"draft_{doc_id}_*.json"))
    except Exception:
        return False


def _doc_draft_types(doc_id: str) -> list[str]:
    """Return the list of draft_types that have been generated for this doc."""
    if not doc_id:
        return []
    types: list[str] = []
    try:
        for p in PROCESSED_DIR.glob(f"draft_{doc_id}_*.json"):
            stem = p.stem  # e.g. "draft_doc_xyz_case_fact_summary"
            prefix = f"draft_{doc_id}_"
            if stem.startswith(prefix):
                types.append(stem[len(prefix):])
    except Exception:
        pass
    return sorted(set(types))


def _doc_to_dict(doc) -> dict:
    return {
        "id": doc.id,
        "filename": doc.filename,
        "upload_timestamp": str(doc.upload_timestamp),
        "pages": doc.pages,
        "structured_fields": doc.structured_fields,
        "chunks": doc.chunks,
        "full_text": doc.full_text,
        "ocr_used": doc.ocr_used,
        "confidence": doc.confidence,
        "page_count": doc.page_count,
        "processing_errors": doc.processing_errors,
        "has_draft": _doc_has_draft(doc.id),
        "draft_types_available": _doc_draft_types(doc.id),
    }


def _draft_to_dict(output) -> dict:
    return {
        "id": f"draft_{output.document_id}_{output.draft_type}",
        "draft_type": output.draft_type,
        "document_id": output.document_id,
        "content_markdown": output.content_markdown,
        "content_structured": output.content_structured,
        "citations": output.citations,
        "verification_status": output.verification_status,
        "generation_timestamp": str(output.generation_timestamp),
        "exemplars_used": output.exemplars_used,
        "rules_applied": output.rules_applied,
        "evidence_count": output.evidence_count,
        "confidence_overall": output.confidence_overall,
    }


def _vr_to_dict(vr) -> dict:
    return {
        "field_path": vr.field_path,
        "status": vr.status,
        "confidence_score": vr.confidence_score,
        "confidence_level": vr.confidence_level,
        "similarity": vr.similarity,
        "evidence_snippet": vr.evidence_snippet,
        "reasons": vr.reasons,
    }


def _correction_to_dict(c) -> dict:
    return {
        "id": c.id,
        "timestamp": str(c.timestamp),
        "document_id": c.document_id,
        "draft_type": c.draft_type,
        "field_path": c.field_path,
        "source_ocr_chunk": c.source_ocr_chunk,
        "generated_text": c.generated_text,
        "edited_text": c.edited_text,
        "correction_type": c.correction_type,
        "active": c.active,
    }
