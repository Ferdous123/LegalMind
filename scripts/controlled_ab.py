"""Controlled same-document A/B for the learning loop.

Same doc (doc_2 court filing, doc_6802495c90ea), case_fact_summary.
Only the learning state varies:

  BEFORE : 3 new rules deactivated  -> draft (pre-learning rule set)
  AFTER  : all 6 rules active       -> draft (post-learning rule set)

Isolates the effect of the 3 rules extracted from the agentic operator
edits. learned_rules.yaml is backed up and restored exactly.
"""
from __future__ import annotations
import json, shutil, sys, time
from datetime import datetime
from pathlib import Path
import requests, yaml

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://localhost:7860"
RULES = ROOT / "config" / "learned_rules.yaml"
BAK = ROOT / "config" / "learned_rules.yaml.expbak"
DOC = "doc_6802495c90ea"
DTYPE = "case_fact_summary"
DRAFT = ROOT / "data" / "processed" / f"draft_{DOC}_{DTYPE}.json"
LOG = ROOT / "logs" / "controlled_ab.log"
RESULT = ROOT / "logs" / "controlled_ab_result.json"


def log(m):
    line = f"[{datetime.now():%H:%M:%S}] {m}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def set_new_rules_active(active: bool):
    d = yaml.safe_load(RULES.read_text(encoding="utf-8"))
    n = 0
    for r in d.get("rules", []):
        if str(r.get("extracted_at", "")).startswith("2026-05-16"):
            r["active"] = active
            n += 1
    RULES.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    return n


def gen_and_wait(tag: str) -> dict:
    base_m = DRAFT.stat().st_mtime if DRAFT.exists() else 0
    r = requests.post(f"{BASE}/api/v1/drafts/generate",
                       json={"document_id": DOC, "draft_type": DTYPE}, timeout=30)
    log(f"{tag}: enqueue -> {r.status_code} {r.text[:120]}")
    for _ in range(150):  # up to ~12.5 min
        time.sleep(5)
        if DRAFT.exists() and DRAFT.stat().st_mtime > base_m:
            time.sleep(2)  # let the write settle
            d = json.loads(DRAFT.read_text(encoding="utf-8"))
            fw = d.get("firewall_summary", {})
            m = {
                "n_rules_applied": len(d.get("rules_applied", [])),
                "n_exemplars": len(d.get("exemplars_used", [])),
                "verified": fw.get("verified", 0), "total": fw.get("total", 0),
                "uncertain": fw.get("uncertain", 0),
                "unsupported": fw.get("unsupported", 0),
                "confidence": round(fw.get("overall_confidence", 0), 4),
                "content_len": len(d.get("content_markdown", "")),
            }
            log(f"{tag}: done {m}")
            return {"metrics": m, "content": d.get("content_markdown", ""),
                    "rules_applied": d.get("rules_applied", [])}
    raise RuntimeError(f"{tag}: draft did not regenerate in time")


def main():
    LOG.write_text("", encoding="utf-8")
    log("=== CONTROLLED SAME-DOC A/B START ===")
    out = {}
    try:
        # BEFORE — deactivate the 3 new rules
        off = set_new_rules_active(False)
        log(f"BEFORE: deactivated {off} new rules (original 3 remain active)")
        before = gen_and_wait("BEFORE")

        # AFTER — restore all rules active
        shutil.copy2(BAK, RULES)
        on = sum(1 for r in yaml.safe_load(RULES.read_text())["rules"] if r.get("active"))
        log(f"AFTER: restored learned_rules.yaml ({on} rules active)")
        after = gen_and_wait("AFTER")

        bm, am = before["metrics"], after["metrics"]
        out = {
            "document": DOC, "draft_type": DTYPE,
            "before_pre_learning": bm,
            "after_post_learning": am,
            "delta": {
                "rules_applied": am["n_rules_applied"] - bm["n_rules_applied"],
                "verified": am["verified"] - bm["verified"],
                "verified_pct_before": round(bm["verified"]/bm["total"], 3) if bm["total"] else 0,
                "verified_pct_after": round(am["verified"]/am["total"], 3) if am["total"] else 0,
                "confidence": round(am["confidence"] - bm["confidence"], 4),
            },
            "before_rules": before["rules_applied"],
            "after_rules": after["rules_applied"],
        }
        RESULT.write_text(json.dumps(out, indent=2), encoding="utf-8")
        log("=== RESULT ===")
        log(json.dumps(out["delta"]))
    finally:
        # Always restore the rules file to the full 6-active state
        if BAK.exists():
            shutil.copy2(BAK, RULES)
            log("rules file restored to full state (cleanup)")
    log("=== DONE ===")


if __name__ == "__main__":
    sys.exit(main())
