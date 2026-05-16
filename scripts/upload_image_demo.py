"""Upload a small, varied set of scanned-image documents to demonstrate the
OCR / messy-input path. Single process, strictly sequential, blocking —
each upload runs the full inline pipeline (OCR -> extract -> retrieve ->
draft -> verify) and returns only when its draft is on disk.
"""
from __future__ import annotations
import json, sys, time
from datetime import datetime
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://localhost:7860"
IMG = ROOT / "data" / "sample" / "images"
PROC = ROOT / "data" / "processed"
LOG = ROOT / "logs" / "image_demo.log"

# Spread: different source docs, different degradation levels, 3 draft types.
JOBS = [
    ("img_2_level_3.png", "case_fact_summary"),     # medium degradation
    ("img_3_level_2.png", "title_review_summary"),  # title path
    ("img_4_level_4.png", "notice_summary"),        # heavy degradation, notice
]


def log(m: str) -> None:
    line = f"[{datetime.now():%H:%M:%S}] {m}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    LOG.write_text("", encoding="utf-8")
    log(f"=== IMAGE DEMO START — {len(JOBS)} scans ===")
    ok = 0
    for i, (name, dtype) in enumerate(JOBS, 1):
        path = IMG / name
        if not path.exists():
            log(f"[{i}/{len(JOBS)}] MISSING {name}")
            continue
        log(f"[{i}/{len(JOBS)}] UPLOAD {name} ({dtype}) — OCR path")
        t0 = time.time()
        try:
            with open(path, "rb") as fh:
                r = requests.post(
                    f"{BASE}/api/v1/documents/upload",
                    files={"file": (name, fh)},
                    data={"draft_type": dtype},
                    timeout=900,
                )
        except Exception as exc:
            log(f"[{i}/{len(JOBS)}] ERROR {name}: {exc}")
            continue
        dt = round(time.time() - t0, 1)
        if r.status_code != 200:
            log(f"[{i}/{len(JOBS)}] FAIL {name} HTTP {r.status_code}: {r.text[:150]}")
            continue
        doc_id = ""
        try:
            doc_id = r.json().get("id", "")
        except Exception:
            pass
        draft = PROC / f"draft_{doc_id}_{dtype}.json"
        if doc_id and (PROC / f"{doc_id}.json").exists() and draft.exists():
            d = json.loads(draft.read_text(encoding="utf-8"))
            fw = d.get("firewall_summary", {})
            doc = json.loads((PROC / f"{doc_id}.json").read_text(encoding="utf-8"))
            log(f"[{i}/{len(JOBS)}] OK {name} -> {doc_id} ({dt}s) "
                f"ocr_conf={round(doc.get('confidence', 0), 2)} "
                f"verified={fw.get('verified', 0)}/{fw.get('total', 0)}")
            ok += 1
        elif doc_id and (PROC / f"{doc_id}.json").exists():
            log(f"[{i}/{len(JOBS)}] PARTIAL {name} -> {doc_id} processed, no draft ({dt}s)")
        else:
            log(f"[{i}/{len(JOBS)}] FAIL {name} no processed JSON ({dt}s)")
    log(f"=== IMAGE DEMO DONE — {ok}/{len(JOBS)} with drafts ===")


if __name__ == "__main__":
    sys.exit(main())
