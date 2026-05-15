"""
Evaluation script for LegalMind.
Measures extraction accuracy against ground-truth annotations in
data/sample/expected_outputs/.

This script reads pre-existing ProcessedDocument JSON files from data/processed/.
It does NOT perform any LLM inference; models are not required.

Usage:
    python scripts/run_evaluation.py

Results are printed as a formatted report and saved to data/evaluation_results.json.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = REPO_ROOT / "data" / "sample"
EXPECTED_DIR = SAMPLE_DIR / "expected_outputs"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_PATH = REPO_ROOT / "data" / "evaluation_results.json"


# ---------------------------------------------------------------------------
# String normalisation helpers
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "of", "in", "at", "to", "for",
        "with", "by", "from", "as", "is", "are", "was", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did",
        "on", "into", "its", "it", "this", "that",
    }
)


def _normalise(value: Any) -> str:
    """Return a lowercase, whitespace-stripped string representation."""
    if value is None:
        return ""
    return str(value).lower().strip()


def _tokenise(text: str) -> set[str]:
    """Return a set of non-stopword word tokens from normalised text."""
    tokens = re.findall(r"[a-z0-9]+", _normalise(text))
    return {t for t in tokens if t not in _STOPWORDS}


def exact_match(extracted: Any, expected: Any) -> bool:
    """Return True if normalised string representations are identical."""
    return _normalise(extracted) == _normalise(expected)


def jaccard_similarity(extracted: Any, expected: Any) -> float:
    """Return the Jaccard similarity of token sets for two values."""
    tokens_a = _tokenise(extracted)
    tokens_b = _tokenise(expected)
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def soft_match(extracted: Any, expected: Any, threshold: float = 0.7) -> bool:
    """Return True if Jaccard similarity meets or exceeds threshold."""
    return jaccard_similarity(extracted, expected) >= threshold


# ---------------------------------------------------------------------------
# Flattening nested field dicts
# ---------------------------------------------------------------------------


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """
    Recursively flatten a nested dict/list structure into dot-separated keys.
    Lists are represented as comma-joined strings for comparison purposes.
    """
    result: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            full_key = f"{prefix}.{key}" if prefix else key
            result.update(_flatten(value, full_key))
    elif isinstance(obj, list):
        # Represent list as a single concatenated string for Jaccard comparison
        joined = " | ".join(_normalise(item) for item in obj)
        result[prefix] = joined
    else:
        result[prefix] = obj
    return result


# ---------------------------------------------------------------------------
# Loading data
# ---------------------------------------------------------------------------


def load_expected_outputs() -> dict[str, dict[str, Any]]:
    """
    Return a mapping from document filename stem to expected output dict.
    E.g. {"doc_1_lease_agreement": {...}, ...}
    """
    outputs: dict[str, dict[str, Any]] = {}
    if not EXPECTED_DIR.exists():
        return outputs
    for path in sorted(EXPECTED_DIR.glob("*_expected.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        doc_filename = data.get("document", "")
        stem = Path(doc_filename).stem if doc_filename else path.stem.replace(
            "_expected", ""
        )
        outputs[stem] = data
    return outputs


def load_processed_document(doc_stem: str) -> dict[str, Any] | None:
    """
    Attempt to load a ProcessedDocument JSON for the given document stem.
    Returns None if not found.
    """
    # Convention: processed files are stored as {doc_stem}.json
    candidate = PROCESSED_DIR / f"{doc_stem}.json"
    if candidate.exists():
        return json.loads(candidate.read_text(encoding="utf-8"))

    # Fallback: search for any JSON file whose 'document_id' or 'filename'
    # matches the stem.
    for path in PROCESSED_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if (
            _normalise(data.get("document_id", "")) == _normalise(doc_stem)
            or _normalise(Path(data.get("filename", "")).stem) == _normalise(doc_stem)
        ):
            return data
    return None


# ---------------------------------------------------------------------------
# Field comparison
# ---------------------------------------------------------------------------


def compare_fields(
    extracted_fields: dict[str, Any],
    expected_fields: dict[str, Any],
) -> dict[str, Any]:
    """
    Compare flattened extracted fields against flattened expected fields.
    Returns a per-field breakdown dict and aggregate accuracy scores.
    """
    flat_extracted = _flatten(extracted_fields)
    flat_expected = _flatten(expected_fields)

    all_keys = set(flat_expected.keys())

    exact_hits = 0
    soft_hits = 0
    field_results: list[dict[str, Any]] = []

    for key in sorted(all_keys):
        exp_val = flat_expected.get(key, "")
        ext_val = flat_extracted.get(key, None)

        is_exact = exact_match(ext_val, exp_val) if ext_val is not None else False
        is_soft = (
            soft_match(ext_val, exp_val) if ext_val is not None else False
        )
        sim = jaccard_similarity(ext_val, exp_val) if ext_val is not None else 0.0

        if is_exact:
            exact_hits += 1
        if is_soft:
            soft_hits += 1

        field_results.append(
            {
                "field": key,
                "expected": str(exp_val)[:120],
                "extracted": str(ext_val)[:120] if ext_val is not None else "(missing)",
                "exact_match": is_exact,
                "soft_match": is_soft,
                "jaccard": round(sim, 3),
            }
        )

    total = len(all_keys)
    return {
        "field_results": field_results,
        "total_fields": total,
        "exact_matches": exact_hits,
        "soft_matches": soft_hits,
        "exact_accuracy": round(exact_hits / total, 3) if total > 0 else 0.0,
        "soft_accuracy": round(soft_hits / total, 3) if total > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

_COL_FIELD = 45
_COL_EXACT = 7
_COL_SOFT = 7
_COL_SIM = 7


def _row(field: str, exact: str, soft: str, sim: str) -> str:
    return (
        f"  {field:<{_COL_FIELD}}"
        f"  {exact:<{_COL_EXACT}}"
        f"  {soft:<{_COL_SOFT}}"
        f"  {sim:<{_COL_SIM}}"
    )


def print_document_report(
    doc_stem: str,
    comparison: dict[str, Any],
    draft_type: str,
    processed_found: bool,
) -> None:
    sep = "-" * 80
    print()
    print(sep)
    print(f"  Document : {doc_stem}")
    print(f"  Draft type: {draft_type}")
    if not processed_found:
        print(
            "  WARNING: No ProcessedDocument found in data/processed/."
            " Showing ground-truth fields only."
        )
    print(sep)
    print(_row("Field", "Exact", "Soft", "Jaccard"))
    print(_row("-" * _COL_FIELD, "-" * _COL_EXACT, "-" * _COL_SOFT, "-" * _COL_SIM))
    for fr in comparison["field_results"]:
        exact_str = "YES" if fr["exact_match"] else "no"
        soft_str = "YES" if fr["soft_match"] else "no"
        print(_row(fr["field"][:_COL_FIELD], exact_str, soft_str, str(fr["jaccard"])))
    print()
    print(
        f"  Total fields : {comparison['total_fields']}"
    )
    print(
        f"  Exact matches: {comparison['exact_matches']} "
        f"({comparison['exact_accuracy'] * 100:.1f}%)"
    )
    print(
        f"  Soft matches : {comparison['soft_matches']} "
        f"({comparison['soft_accuracy'] * 100:.1f}%)"
    )


def print_summary_table(results: list[dict[str, Any]]) -> None:
    print()
    sep = "=" * 80
    print(sep)
    print("  EVALUATION SUMMARY")
    print(sep)
    header = (
        f"  {'Document':<40}  {'Draft Type':<25}"
        f"  {'Exact%':>7}  {'Soft%':>6}"
    )
    print(header)
    print("-" * 80)
    total_exact = 0.0
    total_soft = 0.0
    for r in results:
        print(
            f"  {r['document']:<40}  {r['draft_type']:<25}"
            f"  {r['exact_accuracy'] * 100:>6.1f}%"
            f"  {r['soft_accuracy'] * 100:>5.1f}%"
        )
        total_exact += r["exact_accuracy"]
        total_soft += r["soft_accuracy"]
    n = len(results)
    print("-" * 80)
    if n > 0:
        print(
            f"  {'MEAN':<40}  {'':25}"
            f"  {total_exact / n * 100:>6.1f}%"
            f"  {total_soft / n * 100:>5.1f}%"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    expected_outputs = load_expected_outputs()

    if not expected_outputs:
        print(
            "No expected output files found in:\n"
            f"  {EXPECTED_DIR}\n"
            "Run  python scripts/seed_sample_data.py  first.",
            file=sys.stderr,
        )
        sys.exit(1)

    all_results: list[dict[str, Any]] = []

    for doc_stem, ground_truth in sorted(expected_outputs.items()):
        draft_type: str = ground_truth.get("draft_type", "unknown")
        expected_fields: dict[str, Any] = ground_truth.get("expected_fields", {})

        processed = load_processed_document(doc_stem)
        processed_found = processed is not None

        if processed is not None:
            # Try common field locations in a ProcessedDocument
            extracted_fields: dict[str, Any] = (
                processed.get("structured_fields")
                or processed.get("extracted_fields")
                or processed.get("fields")
                or {}
            )
        else:
            extracted_fields = {}

        comparison = compare_fields(extracted_fields, expected_fields)
        print_document_report(doc_stem, comparison, draft_type, processed_found)

        all_results.append(
            {
                "document": doc_stem,
                "draft_type": draft_type,
                "processed_found": processed_found,
                "total_fields": comparison["total_fields"],
                "exact_matches": comparison["exact_matches"],
                "soft_matches": comparison["soft_matches"],
                "exact_accuracy": comparison["exact_accuracy"],
                "soft_accuracy": comparison["soft_accuracy"],
                "field_results": comparison["field_results"],
            }
        )

    print_summary_table(all_results)

    # Compute overall mean accuracies
    n = len(all_results)
    mean_exact = sum(r["exact_accuracy"] for r in all_results) / n if n > 0 else 0.0
    mean_soft = sum(r["soft_accuracy"] for r in all_results) / n if n > 0 else 0.0

    output: dict[str, Any] = {
        "evaluation_summary": {
            "documents_evaluated": n,
            "mean_exact_accuracy": round(mean_exact, 3),
            "mean_soft_accuracy": round(mean_soft, 3),
        },
        "document_results": all_results,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print()
    print(f"Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
