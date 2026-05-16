"""Idempotent, strictly-sequential corpus processor.

Processes every sample text PDF + every PNG scan through the full upload
pipeline ONE AT A TIME. After each upload it verifies the processed JSON
and its draft actually landed on disk (the "watcher" gate) before moving
to the next file. Resumable: a file whose content is already processed is
skipped, so the script can be re-run after any interruption.

Single process, single blocking HTTP call at a time — no orphan curls,
no crossed responses (the failure mode of the old bash batch).

Run:  .venv/Scripts/python.exe scripts/process_corpus.py
Log:  logs/process_corpus.log   Progress JSON: logs/process_corpus_state.json
"""
from __future__ import annotations
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://localhost:7860"
PROC = ROOT / "data" / "processed"
PDF = ROOT / "data" / "sample" / "pdfs"
IMG = ROOT / "data" / "sample" / "images"
LOG = ROOT / "logs" / "process_corpus.log"
STATE = ROOT / "logs" / "process_corpus_state.json"

# img_N maps to the same draft type as doc_N
DTYPE_BY_N = {
    1: "case_fact_summary",
    2: "case_fact_summary",
    3: "title_review_summary",
    4: "notice_summary",
    5: "document_checklist",
}

# (path, draft_type)
WORK: list[tuple[Path, str]] = []
WORK += [
    (PDF / "doc_1_lease_agreement.pdf", "case_fact_summary"),
    (PDF / "doc_2_court_filing.pdf", "case_fact_summary"),
    (PDF / "doc_3_property_deed.pdf", "title_review_summary"),
    (PDF / "doc_4_compliance_notice.pdf", "notice_summary"),
    (PDF / "doc_5_document_checklist.pdf", "document_checklist"),
]
for n in (1, 2, 3, 4, 5):
    for lvl in (1, 2, 3, 4, 5):
        WORK.append((IMG / f"img_{n}_level_{lvl}.png", DTYPE_BY_N[n]))


def log(msg: str) -> None:
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def processed_filenames() -> set[str]:
    """Filenames already represented by a processed (non-draft) JSON."""
    names: set[str] = set()
    for p in PROC.glob("*.json"):
        if p.stem.startswith("draft_"):
            continue
        try:
            names.add(json.loads(p.read_text(encoding="utf-8")).get("filename", ""))
        except Exception:
            pass
    return names


def has_draft(doc_id: str, draft_type: str) -> bool:
    return (PROC / f"draft_{doc_id}_{draft_type}.json").exists()


def main() -> None:
    LOG.write_text("", encoding="utf-8")
    done, skipped, failed = [], [], []
    total = len(WORK)
    log(f"=== CORPUS PROCESSOR START — {total} files ===")

    for idx, (path, dtype) in enumerate(WORK, 1):
        if not path.exists():
            log(f"[{idx}/{total}] MISSING {path.name} — skip")
            failed.append(path.name)
            continue

        already = processed_filenames()
        if path.name in already:
            log(f"[{idx}/{total}] SKIP {path.name} (already processed)")
            skipped.append(path.name)
            _write_state(done, skipped, failed, total)
            continue

        log(f"[{idx}/{total}] UPLOAD {path.name} ({dtype})")
        t0 = time.time()
        try:
            with open(path, "rb") as fh:
                resp = requests.post(
                    f"{BASE}/api/v1/documents/upload",
                    files={"file": (path.name, fh)},
                    data={"draft_type": dtype},
                    timeout=900,
                )
        except Exception as exc:
            log(f"[{idx}/{total}] ERROR  {path.name}: {exc}")
            failed.append(path.name)
            _write_state(done, skipped, failed, total)
            time.sleep(3)
            continue

        dt = round(time.time() - t0, 1)
        if resp.status_code != 200:
            log(f"[{idx}/{total}] FAIL   {path.name} HTTP {resp.status_code}: {resp.text[:160]}")
            failed.append(path.name)
            _write_state(done, skipped, failed, total)
            continue

        try:
            doc_id = resp.json().get("id", "")
        except Exception:
            doc_id = ""

        # Watcher gate: confirm processed JSON + draft actually landed
        ok_doc = bool(doc_id) and (PROC / f"{doc_id}.json").exists()
        ok_draft = bool(doc_id) and has_draft(doc_id, dtype)
        if ok_doc and ok_draft:
            log(f"[{idx}/{total}] OK     {path.name} -> {doc_id} ({dt}s)")
            done.append(path.name)
        elif ok_doc:
            log(f"[{idx}/{total}] PARTIAL {path.name} -> {doc_id} processed but no draft ({dt}s)")
            done.append(path.name + " (no draft)")
        else:
            log(f"[{idx}/{total}] FAIL   {path.name} -> no processed JSON ({dt}s)")
            failed.append(path.name)

        _write_state(done, skipped, failed, total)

    log(f"=== DONE — {len(done)} ok, {len(skipped)} skipped, {len(failed)} failed ===")
    if failed:
        log("FAILED: " + ", ".join(failed))


def _write_state(done, skipped, failed, total) -> None:
    STATE.write_text(json.dumps({
        "total": total, "done": done, "skipped": skipped, "failed": failed,
        "completed": len(done) + len(skipped), "updated": datetime.now().isoformat(),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
