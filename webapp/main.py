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
# Pipeline SSE event bus  (TRACE subscriber-list pattern)
# ---------------------------------------------------------------------------
# Each SSE connection gets its own asyncio.Queue. _sse_broadcast() is awaited
# from the upload handler and puts the event into every subscriber's queue.
# The SSE endpoint sends a "connected" sentinel immediately on connect so the
# browser knows the stream is live before it starts the upload POST.

_sse_subscribers: list[asyncio.Queue] = []
_sse_log = Path(__file__).resolve().parent.parent / "logs" / "pipeline_events.jsonl"

# One active pipeline job at a time (upload or draft generation).
# Concurrent requests receive 409 — same pattern as TRACE.
_pipeline_lock: asyncio.Lock = asyncio.Lock()


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
    ensure_dirs()
    yield


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
        # Add recent documents
        docs = []
        try:
            for path in sorted(PROCESSED_DIR.glob("*.json"), reverse=True)[:10]:
                if not path.stem.startswith("draft_"):
                    docs.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            pass
        ctx["recent_docs"] = docs
        ctx["draft_types"] = get_supported_types()

    elif panel_id == "library":
        docs = []
        try:
            for path in sorted(PROCESSED_DIR.glob("*.json"), reverse=True):
                if not path.stem.startswith("draft_"):
                    docs.append(json.loads(path.read_text(encoding="utf-8")))
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
    if _pipeline_lock.locked():
        raise HTTPException(
            status_code=409,
            detail="Pipeline already running. Wait for the current job to finish."
        )

    async with _pipeline_lock:
        return await _run_upload(file, draft_type)


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
# Draft API
# ---------------------------------------------------------------------------


@app.post("/api/v1/drafts/generate")
async def generate_draft(body: dict) -> dict:
    document_id: str = body.get("document_id", "")
    draft_type: str = body.get("draft_type", "")
    if not document_id or not draft_type:
        raise HTTPException(status_code=422, detail="document_id and draft_type required")

    # Load the source document for full_text and structured_fields
    try:
        from code.pipeline.ingestion import ProcessedDocument

        doc = await asyncio.to_thread(ProcessedDocument.load, document_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        gen = DraftGenerator()
        output = await asyncio.to_thread(
            gen.generate_draft,
            document_id,
            draft_type,
            doc.full_text,
            8,
        )
    except Exception as exc:
        logger.exception("Draft generation failed for doc %s", document_id)
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    try:
        runner = FirewallRunner()
        results = await asyncio.to_thread(
            runner.verify_draft,
            output.content_markdown,
            doc.structured_fields,
            output.citations or {},
            doc.confidence,
        )
        fw_summary = runner.get_summary(results)
        firewall_results = [_vr_to_dict(r) for r in results]
    except Exception as exc:
        logger.warning("Firewall check failed: %s", exc)
        fw_summary = {}
        firewall_results = []

    draft_dict = _draft_to_dict(output)
    draft_dict["firewall_results"] = firewall_results
    draft_dict["firewall_summary"] = fw_summary

    draft_path = PROCESSED_DIR / f"draft_{document_id}_{draft_type}.json"
    try:
        draft_path.write_text(json.dumps(draft_dict, default=str), encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not save draft to disk: %s", exc)

    return draft_dict


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
    try:
        correction = Correction(**body)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid correction payload: {exc}") from exc

    try:
        store = CorrectionStore()
        correction_id = await asyncio.to_thread(store.save_correction, correction)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Pattern extraction and consolidation triggered in background so the
    # HTTP response is not delayed by potentially heavy LLM inference.
    try:
        extractor = PatternExtractor()
        if extractor.should_trigger():
            asyncio.create_task(_run_pattern_extraction(extractor))
    except Exception as exc:
        logger.warning("Pattern extractor check failed: %s", exc)

    try:
        consolidator = PromptConsolidator()
        if consolidator.should_trigger():
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


@app.post("/api/v1/learning/rules")
async def trigger_pattern_extraction() -> dict:
    """Manually trigger pattern extraction from accumulated corrections.

    Used by the Audit page's 'Trigger Pattern Extraction' button via HTMX.
    Passes force=True so extraction runs regardless of correction count —
    useful for demo and review where fewer than TRIGGER_INTERVAL edits exist.
    """
    try:
        extractor = PatternExtractor()
        patterns = await asyncio.to_thread(extractor.extract_patterns, True)
        rule_count = len(patterns)
        return {"status": "ok", "rules_extracted": rule_count,
                "message": f"Extraction complete — {rule_count} rule(s) generated."}
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
    """Stream pipeline events to the browser using the TRACE subscriber pattern.

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
    categories = [{"name": cat, "terms": list(terms)} for cat, terms in LEGAL_TERMS.items()]

    # Load learned terms
    learned = []
    if LEARNED_TERMS_FILE.exists():
        try:
            learned = json.loads(LEARNED_TERMS_FILE.read_text(encoding="utf-8"))
        except Exception:
            learned = []

    return {"categories": categories, "learned": learned, "total": sum(len(c["terms"]) for c in categories)}


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
    page_path = PAGES_DIR / doc_id / f"page_{page_num:03d}.jpg"
    if not page_path.exists():
        raise HTTPException(status_code=404, detail="Page image not found")
    return FileResponse(str(page_path), media_type="image/jpeg")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _run_pattern_extraction(extractor: PatternExtractor) -> None:
    try:
        patterns = await asyncio.to_thread(extractor.extract_patterns)
        logger.info("Pattern extraction produced %d patterns", len(patterns))
    except Exception as exc:
        logger.warning("Background pattern extraction failed: %s", exc)


async def _run_consolidation(consolidator: PromptConsolidator) -> None:
    try:
        await asyncio.to_thread(consolidator.consolidate)
        logger.info("Prompt consolidation complete")
    except Exception as exc:
        logger.warning("Background consolidation failed: %s", exc)


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
