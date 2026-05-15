"""Seed the correction store with realistic operator corrections.

This script populates data/corrections/ with corrections that reflect real
operator edits on the sample documents.  Running it does NOT require the
server or any LLM — it writes directly to the JSONL store.

After seeding, it calls PatternExtractor.extract_patterns(force=True) to
demonstrate Layer 2 rule extraction.  Layer 2 DOES load the reasoning model,
so the server must not be holding the GPU when this is run.

Usage (from repo root):
    python scripts/seed_corrections.py [--no-extract]

Options:
    --no-extract    Seed corrections only; skip pattern extraction
                    (use this if models are not available / server is running)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is on the path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from code.learning.correction_store import Correction, CorrectionStore  # noqa: E402


# ---------------------------------------------------------------------------
# Correction payloads (realistic, based on actual processed documents)
# ---------------------------------------------------------------------------

_CORRECTIONS: list[dict] = [
    # --- Party name omissions (doc_8e01efbc6693 = doc_2_court_filing.pdf) ---
    {
        "document_id": "doc_8e01efbc6693",
        "draft_type": "case_fact_summary",
        "field_path": "parties.defendants",
        "source_ocr_chunk": (
            "NORWOOD FABRICATION GROUP LLC, a Delaware limited liability company, "
            "and GERALD P. NORWOOD, individually, Defendants."
        ),
        "generated_text": "Defendant: Norwood Fabrication Group LLC",
        "edited_text": (
            "Defendants: Norwood Fabrication Group LLC (Delaware LLC); "
            "Gerald P. Norwood (individually)"
        ),
        "correction_type": "omission",
    },
    {
        "document_id": "doc_c84f13b79d0f",
        "draft_type": "case_fact_summary",
        "field_path": "parties.defendants",
        "source_ocr_chunk": (
            "Defendants Norwood Fabrication Group LLC (\"Norwood\") and "
            "Gerald P. Norwood (\"Norwood Individual\") (collectively \"Defendants\")"
        ),
        "generated_text": "Defendant: [Not clearly identified]",
        "edited_text": (
            "Defendants: (1) Norwood Fabrication Group LLC; "
            "(2) Gerald P. Norwood, individually"
        ),
        "correction_type": "omission",
    },
    {
        "document_id": "doc_e35803ce3eb3",
        "draft_type": "case_fact_summary",
        "field_path": "parties.defendants",
        "source_ocr_chunk": (
            "NORWOOD FABRICATION GROUP LLC, a Delaware limited liability company, "
            "and GERALD P. NORWOOD, individually"
        ),
        "generated_text": "Defendant: Norwood Fabrication Group LLC",
        "edited_text": (
            "Defendants: Norwood Fabrication Group LLC; Gerald P. Norwood"
        ),
        "correction_type": "omission",
    },
    # --- Party name completeness for lease agreement ---
    {
        "document_id": "doc_33c39286752d",
        "draft_type": "case_fact_summary",
        "field_path": "parties.other_parties",
        "source_ocr_chunk": (
            "TENANT: Marcus Delacroix-Webb, an individual residing at "
            "814 Cedarbrook Lane, Apartment 3B, Thornfield, TX 77019"
        ),
        "generated_text": "Tenant: marcus Delacroix-Webb",
        "edited_text": "Tenant: Marcus Delacroix-Webb",
        "correction_type": "error",
    },
    {
        "document_id": "doc_eb71a78dd017",
        "draft_type": "case_fact_summary",
        "field_path": "parties.other_parties",
        "source_ocr_chunk": (
            "LANDLORD: Hargrove Property Holdings LLC, a limited liability company "
            "organised under the laws of the State of Thornfield"
        ),
        "generated_text": "Landlord: hargrove property holdings llc",
        "edited_text": "Landlord: Hargrove Property Holdings LLC",
        "correction_type": "error",
    },
    # --- Date format standardisation ---
    {
        "document_id": "doc_8e01efbc6693",
        "draft_type": "case_fact_summary",
        "field_path": "key_dates",
        "source_ocr_chunk": "Filing Date: March 14 2024",
        "generated_text": "Filing date: March 14 2024",
        "edited_text": "Filing date: 2024-03-14",
        "correction_type": "style",
    },
    {
        "document_id": "doc_33c39286752d",
        "draft_type": "case_fact_summary",
        "field_path": "key_dates",
        "source_ocr_chunk": (
            "This Residential Lease Agreement is entered into as of the 1st day "
            "of March, 2024"
        ),
        "generated_text": "Agreement date: March 1, 2024",
        "edited_text": "Agreement date: 2024-03-01",
        "correction_type": "style",
    },
    {
        "document_id": "doc_e35803ce3eb3",
        "draft_type": "case_fact_summary",
        "field_path": "key_dates",
        "source_ocr_chunk": (
            "Manufacturing Services Agreement dated January 15, 2023"
        ),
        "generated_text": "Contract date: January 15, 2023",
        "edited_text": "Contract date: 2023-01-15",
        "correction_type": "style",
    },
    # --- Claim description improvements ---
    {
        "document_id": "doc_8e01efbc6693",
        "draft_type": "case_fact_summary",
        "field_path": "claims",
        "source_ocr_chunk": (
            "COMPLAINT FOR BREACH OF CONTRACT, FRAUD, AND UNJUST ENRICHMENT"
        ),
        "generated_text": "Claims: breach of contract",
        "edited_text": (
            "Claims: (1) Breach of Contract; (2) Fraud; (3) Unjust Enrichment"
        ),
        "correction_type": "omission",
    },
    {
        "document_id": "doc_c84f13b79d0f",
        "draft_type": "case_fact_summary",
        "field_path": "claims",
        "source_ocr_chunk": (
            "Plaintiff alleges that Norwood failed to deliver conforming goods "
            "under the Manufacturing Services Agreement, constituting breach of "
            "contract, and that Norwood's representations were fraudulent."
        ),
        "generated_text": "Claims: breach of contract, fraud",
        "edited_text": (
            "Claims: (1) Breach of Contract — non-delivery of conforming goods; "
            "(2) Fraud — fraudulent misrepresentations; "
            "(3) Unjust Enrichment"
        ),
        "correction_type": "style",
    },
]


def main(run_extraction: bool = True) -> None:
    store = CorrectionStore()

    print(f"Seeding {len(_CORRECTIONS)} corrections into data/corrections/...")
    for i, payload in enumerate(_CORRECTIONS, 1):
        correction = Correction(**payload)
        correction_id = store.save_correction(correction)
        print(
            f"  [{i:02d}] {correction_id}  "
            f"{payload['draft_type']}  {payload['field_path']}  "
            f"({payload['correction_type']})"
        )

    total = store.get_count()
    print(f"\nCorrection store total: {total} corrections")

    if not run_extraction:
        print("\nSkipped pattern extraction (--no-extract)")
        return

    print("\nRunning pattern extraction (force=True)...")
    try:
        from code.learning.pattern_extractor import PatternExtractor
        extractor = PatternExtractor()
        rules = extractor.extract_patterns(force=True)
        if rules:
            print(f"\nExtracted {len(rules)} rule(s):")
            for j, rule in enumerate(rules, 1):
                print(f"  Rule {j}: {rule[:120]}{'...' if len(rule) > 120 else ''}")
        else:
            print(
                "\nNo rules extracted (clusters too small or no reasoning model available).\n"
                "Corrections are stored — rules will be extracted on next server trigger."
            )
    except Exception as exc:
        print(f"\nPattern extraction failed (server/model not available): {exc}")
        print("Corrections are stored. Run POST /api/v1/learning/rules when server is up.")

    from config.paths import LEARNED_RULES_YAML
    if LEARNED_RULES_YAML.exists():
        print(f"\nRules file: {LEARNED_RULES_YAML}")
    else:
        print(f"\nRules file not yet created — will be written after pattern extraction.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-extract", action="store_true",
        help="Seed corrections only; skip LLM pattern extraction"
    )
    args = parser.parse_args()
    main(run_extraction=not args.no_extract)
