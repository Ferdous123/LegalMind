"""Test script for the LegalMind learning loop.

Validates the end-to-end learning pipeline without requiring GPU inference:
1. Creates 5 synthetic Correction objects covering all draft types.
2. Saves them to CorrectionStore.
3. Verifies retrieval methods (get_all_active, get_recent, get_count).
4. Verifies ExemplarRetriever can load corrections (with mocked embeddings).
5. Checks PatternExtractor.should_trigger() logic.
6. Checks PromptConsolidator.should_trigger() logic.

Run from project root:
    python scripts/test_learning_loop.py
"""

import sys
import os
import json
import shutil
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path so imports work when run from any directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

class TestResult:
    def __init__(self):
        self.passed = []
        self.failed = []

    def ok(self, name: str):
        self.passed.append(name)
        print(f"  PASS  {name}")

    def fail(self, name: str, reason: str):
        self.failed.append(name)
        print(f"  FAIL  {name}: {reason}")

    def summary(self) -> bool:
        print()
        print("=" * 60)
        total = len(self.passed) + len(self.failed)
        if not self.failed:
            print(f"Learning loop: PASS ({len(self.passed)}/{total} checks passed)")
        else:
            print(f"Learning loop: FAIL ({len(self.failed)} failure(s) out of {total} checks)")
            for f in self.failed:
                print(f"  - {f}")
        print("=" * 60)
        return len(self.failed) == 0


def make_synthetic_correction(index: int, draft_type: str, field_path: str,
                               correction_type: str) -> "Correction":
    """Build a realistic synthetic Correction object."""
    from code.learning.correction_store import Correction

    source_chunks = {
        "case_fact_summary": (
            "The defendant, Marcus J. Bell, failed to appear at the scheduled "
            "hearing on February 14, 2024, despite proper service of summons."
        ),
        "title_review_summary": (
            "Deed of Trust recorded in Book 412, Page 87 of the Official Records "
            "of Riverside County, dated November 3, 2019."
        ),
        "notice_summary": (
            "You are hereby notified that payment of $4,250.00 is due within "
            "thirty (30) days of the date of this notice, or legal action will follow."
        ),
        "document_checklist": (
            "Certificate of Title, Policy No. 2024-RTL-00871, issued by First "
            "American Title Insurance Company, dated March 1, 2024."
        ),
    }

    generated_texts = {
        "case_fact_summary": "Defendant: [Not clearly identified in source]",
        "title_review_summary": "Deed of Trust: Book [illegible], Page 87",
        "notice_summary": "Amount due: [amount not specified]",
        "document_checklist": "Title policy: present",
    }

    edited_texts = {
        "case_fact_summary": "Defendant: Marcus J. Bell",
        "title_review_summary": "Deed of Trust: Book 412, Page 87, Riverside County, dated 2019-11-03",
        "notice_summary": "Amount due: $4,250.00 — due within 30 days of notice date",
        "document_checklist": (
            "Title policy: Present — Certificate of Title, Policy No. 2024-RTL-00871, "
            "First American Title Insurance Company, dated 2024-03-01"
        ),
    }

    return Correction(
        id=f"corr_test_{index:04d}",
        timestamp=datetime.now(timezone.utc).isoformat(),
        document_id=f"doc_test_{index:04d}",
        draft_type=draft_type,
        field_path=field_path,
        source_ocr_chunk=source_chunks[draft_type],
        generated_text=generated_texts[draft_type],
        edited_text=edited_texts[draft_type],
        correction_type=correction_type,
        embedding=[],  # no embedding for offline test
        active=True,
    )


# ---------------------------------------------------------------------------
# Main test suite
# ---------------------------------------------------------------------------

def run_tests():
    results = TestResult()

    # Use a temporary directory for all data so tests don't pollute production data
    tmp_dir = Path(tempfile.mkdtemp(prefix="legalmind_test_"))
    corrections_dir = tmp_dir / "corrections"
    corrections_dir.mkdir(parents=True)

    print(f"Test workspace: {tmp_dir}")
    print()
    print("--- Step 1: Import core learning modules ---")

    try:
        from code.learning.correction_store import Correction, CorrectionStore
        results.ok("Import: correction_store")
    except Exception as e:
        results.fail("Import: correction_store", str(e))
        print("Cannot continue — correction_store import failed.")
        results.summary()
        return False

    try:
        from code.learning.pattern_extractor import PatternExtractor, TRIGGER_INTERVAL
        results.ok("Import: pattern_extractor")
    except Exception as e:
        results.fail("Import: pattern_extractor", str(e))

    try:
        from code.learning.prompt_consolidator import PromptConsolidator, CONSOLIDATION_THRESHOLD
        results.ok("Import: prompt_consolidator")
    except Exception as e:
        results.fail("Import: prompt_consolidator", str(e))

    # ------------------------------------------------------------------
    print()
    print("--- Step 2: Create 5 synthetic corrections and save to store ---")

    synthetic_corrections = [
        make_synthetic_correction(1, "case_fact_summary",    "parties.defendant",    "omission"),
        make_synthetic_correction(2, "title_review_summary", "chain_of_title",       "error"),
        make_synthetic_correction(3, "notice_summary",       "deadlines",            "omission"),
        make_synthetic_correction(4, "document_checklist",   "documents_present",    "restructure"),
        make_synthetic_correction(5, "case_fact_summary",    "key_dates",            "style"),
    ]

    # Patch CORRECTIONS_DIR to use temp directory
    with patch("code.learning.correction_store.CORRECTIONS_DIR", corrections_dir):
        # Re-import within patch context won't work for already-imported classes
        # Instead, instantiate store with temp dir by monkey-patching on the instance
        store = CorrectionStore.__new__(CorrectionStore)
        store.__class__.__init__(store)

        # Override the _get_file method to use temp dir
        original_get_file = store._get_file
        store._get_file = lambda dt: corrections_dir / f"{dt}.jsonl"
        corrections_dir.mkdir(parents=True, exist_ok=True)

        saved_ids = []
        save_errors = []
        for c in synthetic_corrections:
            try:
                cid = store.save_correction(c)
                saved_ids.append(cid)
            except Exception as e:
                save_errors.append(str(e))

    if save_errors:
        results.fail("Save 5 corrections", f"Errors: {save_errors}")
    elif len(saved_ids) == 5:
        results.ok("Save 5 corrections (all 4 draft types covered)")
    else:
        results.fail("Save 5 corrections", f"Only saved {len(saved_ids)}")

    # ------------------------------------------------------------------
    print()
    print("--- Step 3: Verify retrieval methods ---")

    with patch("code.learning.correction_store.CORRECTIONS_DIR", corrections_dir):
        store2 = CorrectionStore.__new__(CorrectionStore)
        store2.__class__.__init__(store2)
        store2._get_file = lambda dt: corrections_dir / f"{dt}.jsonl"

        # get_all_active
        try:
            csf_active = store2.get_all_active("case_fact_summary")
            if len(csf_active) == 2:
                results.ok("get_all_active: case_fact_summary returns 2 corrections")
            else:
                results.fail("get_all_active: case_fact_summary",
                             f"Expected 2, got {len(csf_active)}")
        except Exception as e:
            results.fail("get_all_active", traceback.format_exc())

        # get_all_active for other types
        try:
            trs_active = store2.get_all_active("title_review_summary")
            ns_active = store2.get_all_active("notice_summary")
            dc_active = store2.get_all_active("document_checklist")
            if len(trs_active) == 1 and len(ns_active) == 1 and len(dc_active) == 1:
                results.ok("get_all_active: title_review, notice, checklist each return 1")
            else:
                results.fail("get_all_active: other types",
                             f"trs={len(trs_active)}, ns={len(ns_active)}, dc={len(dc_active)}")
        except Exception as e:
            results.fail("get_all_active: other types", traceback.format_exc())

        # get_recent
        try:
            recent = store2.get_recent(n=10)
            if len(recent) == 5:
                results.ok("get_recent(10) returns all 5 corrections")
            else:
                results.fail("get_recent", f"Expected 5, got {len(recent)}")
        except Exception as e:
            results.fail("get_recent", traceback.format_exc())

        try:
            recent3 = store2.get_recent(n=3)
            if len(recent3) == 3:
                results.ok("get_recent(3) correctly caps at 3")
            else:
                results.fail("get_recent(3)", f"Expected 3, got {len(recent3)}")
        except Exception as e:
            results.fail("get_recent(3)", traceback.format_exc())

        # get_count
        try:
            count = store2.get_count()
            if count == 5:
                results.ok("get_count() returns 5")
            else:
                results.fail("get_count", f"Expected 5, got {count}")
        except Exception as e:
            results.fail("get_count", traceback.format_exc())

        # get_count by draft_type
        try:
            csf_count = store2.get_count("case_fact_summary")
            if csf_count == 2:
                results.ok("get_count('case_fact_summary') returns 2")
            else:
                results.fail("get_count by draft_type",
                             f"Expected 2, got {csf_count}")
        except Exception as e:
            results.fail("get_count by draft_type", traceback.format_exc())

    # ------------------------------------------------------------------
    print()
    print("--- Step 4: ExemplarRetriever with mocked embeddings ---")

    try:
        from code.learning.exemplar_retriever import ExemplarRetriever
        results.ok("Import: exemplar_retriever")
    except Exception as e:
        results.fail("Import: exemplar_retriever", str(e))

    try:
        # Mock chromadb and ModelManager to avoid GPU/ChromaDB dependency
        mock_chroma_collection = MagicMock()
        mock_chroma_collection.query.return_value = {
            "ids": [["corr_test_0001", "corr_test_0002"]],
            "distances": [[0.1, 0.2]],
        }

        mock_chroma_client = MagicMock()
        mock_chroma_client.get_or_create_collection.return_value = mock_chroma_collection

        mock_model_mgr = MagicMock()
        mock_model_mgr.embed.return_value = [[0.1, 0.2, 0.3]]

        with (
            patch("code.learning.exemplar_retriever.chromadb.PersistentClient",
                  return_value=mock_chroma_client),
            patch("code.learning.exemplar_retriever.ModelManager") as mock_mgr_cls,
            patch("code.learning.exemplar_retriever.CHROMA_DIR", tmp_dir / "chromadb"),
            patch("code.learning.correction_store.CORRECTIONS_DIR", corrections_dir),
        ):
            mock_mgr_cls.instance.return_value = mock_model_mgr

            retriever = ExemplarRetriever()

            # Override store to use temp corrections
            retriever._store._get_file = lambda dt: corrections_dir / f"{dt}.jsonl"

            exemplars = retriever.get_relevant_exemplars(
                source_text="The defendant Marcus Bell failed to appear",
                draft_type="case_fact_summary",
                k=2,
            )

            # Should return corrections that match the mocked IDs
            if isinstance(exemplars, list):
                results.ok(
                    f"ExemplarRetriever.get_relevant_exemplars returns list "
                    f"(got {len(exemplars)} exemplars)"
                )
            else:
                results.fail("ExemplarRetriever.get_relevant_exemplars",
                             f"Expected list, got {type(exemplars)}")

            # Test format_exemplars_for_prompt
            formatted = retriever.format_exemplars_for_prompt(exemplars)
            if isinstance(formatted, str):
                results.ok("ExemplarRetriever.format_exemplars_for_prompt returns string")
            else:
                results.fail("format_exemplars_for_prompt",
                             f"Expected str, got {type(formatted)}")

    except Exception as e:
        results.fail("ExemplarRetriever integration (mocked)", traceback.format_exc())

    # ------------------------------------------------------------------
    print()
    print("--- Step 5: PatternExtractor.should_trigger() logic ---")

    try:
        from code.learning.pattern_extractor import PatternExtractor, TRIGGER_INTERVAL

        with (
            patch("code.learning.correction_store.CORRECTIONS_DIR", corrections_dir),
            patch("code.learning.pattern_extractor.LEARNED_RULES_YAML",
                  tmp_dir / "learned_rules.yaml"),
            patch("code.learning.pattern_extractor.InferenceEngine"),
        ):
            extractor = PatternExtractor()
            extractor._store._get_file = lambda dt: corrections_dir / f"{dt}.jsonl"

            # With 5 corrections and last_trigger=0, should NOT trigger
            # (TRIGGER_INTERVAL is 20, we only have 5)
            extractor._last_trigger_count = 0
            current_count = extractor._store.get_count()

            should = extractor.should_trigger()
            expected_should = current_count >= (0 + TRIGGER_INTERVAL)

            if should == expected_should:
                results.ok(
                    f"PatternExtractor.should_trigger() = {should} "
                    f"(count={current_count}, threshold=0+{TRIGGER_INTERVAL}={TRIGGER_INTERVAL}) — correct"
                )
            else:
                results.fail(
                    "PatternExtractor.should_trigger()",
                    f"Expected {expected_should}, got {should} "
                    f"(count={current_count}, last_trigger=0, interval={TRIGGER_INTERVAL})"
                )

            # Simulate being at the trigger boundary
            extractor._last_trigger_count = current_count - TRIGGER_INTERVAL
            should_boundary = extractor.should_trigger()
            if should_boundary:
                results.ok("PatternExtractor.should_trigger() = True at trigger boundary")
            else:
                results.fail(
                    "PatternExtractor.should_trigger() at boundary",
                    f"Expected True when count={current_count} >= "
                    f"last_trigger={current_count - TRIGGER_INTERVAL} + {TRIGGER_INTERVAL}"
                )

            # Verify TRIGGER_INTERVAL constant is 20
            if TRIGGER_INTERVAL == 20:
                results.ok("PatternExtractor.TRIGGER_INTERVAL == 20 (as specified)")
            else:
                results.fail("TRIGGER_INTERVAL value", f"Expected 20, got {TRIGGER_INTERVAL}")

    except Exception as e:
        results.fail("PatternExtractor.should_trigger()", traceback.format_exc())

    # ------------------------------------------------------------------
    print()
    print("--- Step 6: PromptConsolidator.should_trigger() logic ---")

    try:
        from code.learning.prompt_consolidator import PromptConsolidator, CONSOLIDATION_THRESHOLD

        rules_yaml_path = tmp_dir / "learned_rules.yaml"

        with (
            patch("code.learning.correction_store.CORRECTIONS_DIR", corrections_dir),
            patch("code.learning.prompt_consolidator.LEARNED_RULES_YAML", rules_yaml_path),
            patch("code.learning.prompt_consolidator.PROMPTS_DIR", tmp_dir / "prompts"),
            patch("code.learning.prompt_consolidator.InferenceEngine"),
            patch("code.learning.exemplar_retriever.chromadb.PersistentClient",
                  return_value=mock_chroma_client),
            patch("code.learning.exemplar_retriever.ModelManager") as mock_mgr_cls2,
            patch("code.learning.exemplar_retriever.CHROMA_DIR", tmp_dir / "chromadb"),
        ):
            mock_mgr_cls2.instance.return_value = mock_model_mgr

            consolidator = PromptConsolidator()
            consolidator._store._get_file = lambda dt: corrections_dir / f"{dt}.jsonl"

            # No rules file — should not trigger
            should_no_rules = consolidator.should_trigger()
            if not should_no_rules:
                results.ok("PromptConsolidator.should_trigger() = False with no rules file")
            else:
                results.fail(
                    "PromptConsolidator.should_trigger() with no rules",
                    f"Expected False, got {should_no_rules}"
                )

            # Write a rules file with fewer than CONSOLIDATION_THRESHOLD unconsolidated rules
            import yaml
            few_rules = {
                "rules": [
                    {"text": f"Rule {i}", "active": True, "consolidated": False}
                    for i in range(CONSOLIDATION_THRESHOLD - 1)
                ],
                "metadata": {}
            }
            rules_yaml_path.parent.mkdir(parents=True, exist_ok=True)
            with open(rules_yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(few_rules, f)

            should_few = consolidator.should_trigger()
            if not should_few:
                results.ok(
                    f"PromptConsolidator.should_trigger() = False with "
                    f"{CONSOLIDATION_THRESHOLD - 1} rules (below threshold={CONSOLIDATION_THRESHOLD})"
                )
            else:
                results.fail(
                    "PromptConsolidator.should_trigger() below threshold",
                    f"Expected False, got {should_few}"
                )

            # Write exactly CONSOLIDATION_THRESHOLD unconsolidated rules
            enough_rules = {
                "rules": [
                    {"text": f"Rule {i}", "active": True, "consolidated": False}
                    for i in range(CONSOLIDATION_THRESHOLD)
                ],
                "metadata": {}
            }
            with open(rules_yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(enough_rules, f)

            should_enough = consolidator.should_trigger()
            if should_enough:
                results.ok(
                    f"PromptConsolidator.should_trigger() = True with "
                    f"{CONSOLIDATION_THRESHOLD} unconsolidated rules"
                )
            else:
                results.fail(
                    "PromptConsolidator.should_trigger() at threshold",
                    f"Expected True with {CONSOLIDATION_THRESHOLD} rules, got {should_enough}"
                )

            # Verify CONSOLIDATION_THRESHOLD constant is 10
            if CONSOLIDATION_THRESHOLD == 10:
                results.ok("PromptConsolidator.CONSOLIDATION_THRESHOLD == 10 (as specified)")
            else:
                results.fail(
                    "CONSOLIDATION_THRESHOLD value",
                    f"Expected 10, got {CONSOLIDATION_THRESHOLD}"
                )

            # Test that consolidated rules are excluded from the trigger count
            mixed_rules = {
                "rules": [
                    {"text": f"Old rule {i}", "active": True, "consolidated": True}
                    for i in range(15)
                ] + [
                    {"text": f"New rule {i}", "active": True, "consolidated": False}
                    for i in range(CONSOLIDATION_THRESHOLD - 1)
                ],
                "metadata": {}
            }
            with open(rules_yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(mixed_rules, f)

            should_mixed = consolidator.should_trigger()
            if not should_mixed:
                results.ok(
                    "PromptConsolidator.should_trigger() correctly ignores already-consolidated rules"
                )
            else:
                results.fail(
                    "PromptConsolidator.should_trigger() with mixed consolidated/new",
                    f"Expected False (only {CONSOLIDATION_THRESHOLD - 1} unconsolidated), got True"
                )

    except Exception as e:
        results.fail("PromptConsolidator.should_trigger()", traceback.format_exc())

    # ------------------------------------------------------------------
    # Cleanup temp directory
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print()
    return results.summary()


if __name__ == "__main__":
    print("LegalMind — Learning Loop Validation")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print()

    success = run_tests()
    sys.exit(0 if success else 1)
