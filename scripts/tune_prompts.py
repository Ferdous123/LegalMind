"""
Prompt Tuning Script -- Analyzes extraction errors and refines prompts.

Reads pipeline_results.json, identifies systematic errors, and:
1. Categorizes errors by field type and error pattern
2. Generates suggested prompt additions for each pattern
3. Writes improved prompts to prompts/ directory
4. Appends tuning rules to config/learned_rules.yaml
5. Can re-run pipeline to measure improvement

Usage:
    python scripts/tune_prompts.py              # analyze + apply fixes
    python scripts/tune_prompts.py --dry-run    # show suggestions without applying
    python scripts/tune_prompts.py --rerun      # apply fixes then re-run pipeline
"""

import sys
import os
import json
import logging
import argparse
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from collections import defaultdict

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.paths import (
    DATA_DIR, PROMPTS_DIR, LEARNED_RULES_YAML, LOGS_DIR, CONFIG_DIR,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PIPELINE_RESULTS_PATH = DATA_DIR / "pipeline_results.json"

# Error pattern categories
PATTERN_OMISSION = "omission"
PATTERN_FORMATTING = "formatting"
PATTERN_PARTIAL = "partial"
PATTERN_WRONG_FIELD = "wrong_field"

# Draft type to prompt file mapping
DRAFT_TYPE_PROMPTS: dict[str, str] = {
    "case_fact_summary": "case_fact_summary.txt",
    "title_review_summary": "title_review_summary.txt",
    "notice_summary": "notice_summary.txt",
    "document_checklist": "document_checklist.txt",
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    """Configure logging for the tuning script."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"tune_prompts_{timestamp}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    ))
    root_logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)-8s | %(message)s"))
    root_logger.addHandler(ch)

    logger = logging.getLogger("legalmind.tune")
    logger.info("Tuning log: %s", log_file)
    return logger


# ---------------------------------------------------------------------------
# Error Pattern Detection
# ---------------------------------------------------------------------------

def classify_error(field_result: dict) -> str:
    """Classify a field-level error into a pattern category.

    Categories:
    - omission: expected field has value but extracted is empty
    - formatting: values are semantically equivalent but format differs
    - partial: extracted value is a substring of expected (incomplete extraction)
    - wrong_field: value appears but in a different field key
    """
    expected = str(field_result.get("expected", "")).strip()
    extracted = str(field_result.get("extracted", "")).strip()
    match_type = field_result.get("match_type", "miss")
    similarity = field_result.get("similarity", 0.0)

    if not extracted:
        return PATTERN_OMISSION

    # Formatting: high similarity but not exact
    if match_type == "soft" and similarity >= 0.7:
        # Check if it's just a formatting difference
        exp_stripped = expected.lower().replace("-", "").replace("/", "").replace(" ", "")
        ext_stripped = extracted.lower().replace("-", "").replace("/", "").replace(" ", "")
        if exp_stripped == ext_stripped:
            return PATTERN_FORMATTING
        return PATTERN_FORMATTING

    # Partial: extracted is contained in expected or vice versa
    if match_type == "partial" or (
        expected.lower() in extracted.lower() or extracted.lower() in expected.lower()
    ):
        return PATTERN_PARTIAL

    # If we have a miss but extracted is non-empty, it may be a wrong_field mapping
    if match_type == "miss" and extracted:
        return PATTERN_WRONG_FIELD

    return PATTERN_OMISSION


def analyze_errors(evaluations: list[dict]) -> dict:
    """Analyze all field-level errors across evaluations.

    Returns a structured analysis:
    {
        "by_pattern": {pattern: [field_info, ...]},
        "by_draft_type": {draft_type: {pattern: [field_info, ...]}},
        "by_field": {field_name: [error_info, ...]},
        "summary": {...}
    }
    """
    by_pattern: dict[str, list[dict]] = defaultdict(list)
    by_draft_type: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    by_field: dict[str, list[dict]] = defaultdict(list)

    total_errors = 0
    total_fields = 0

    for ev in evaluations:
        draft_type = ev.get("draft_type", "unknown")
        doc = ev.get("document", "unknown")

        for fr in ev.get("field_results", []):
            total_fields += 1
            if fr["match_type"] == "exact":
                continue  # not an error

            total_errors += 1
            pattern = classify_error(fr)

            error_info = {
                "document": doc,
                "draft_type": draft_type,
                "field": fr["field"],
                "expected": fr["expected"],
                "extracted": fr["extracted"],
                "similarity": fr["similarity"],
                "pattern": pattern,
            }

            by_pattern[pattern].append(error_info)
            by_draft_type[draft_type][pattern].append(error_info)
            by_field[fr["field"]].append(error_info)

    return {
        "by_pattern": dict(by_pattern),
        "by_draft_type": {k: dict(v) for k, v in by_draft_type.items()},
        "by_field": dict(by_field),
        "summary": {
            "total_fields": total_fields,
            "total_errors": total_errors,
            "error_rate": round(total_errors / max(total_fields, 1), 4),
            "pattern_counts": {p: len(errs) for p, errs in by_pattern.items()},
        },
    }


# ---------------------------------------------------------------------------
# Prompt Improvement Generation
# ---------------------------------------------------------------------------

def generate_improvements(analysis: dict) -> list[dict]:
    """Generate suggested prompt improvements based on error patterns.

    Each improvement is:
    {
        "pattern": str,
        "draft_type": str,
        "field": str,
        "rule_text": str,
        "prompt_addition": str,
        "priority": int (1=high, 3=low),
    }
    """
    improvements: list[dict] = []

    # --- Omission pattern: fields being missed entirely ---
    for error_info in analysis["by_pattern"].get(PATTERN_OMISSION, []):
        field_name = error_info["field"]
        draft_type = error_info["draft_type"]
        human_field = field_name.replace(".", " > ").replace("_", " ")

        rule_text = f"Always extract '{human_field}' even if embedded in procedural text or boilerplate sections."
        prompt_addition = (
            f"IMPORTANT: You must extract the '{human_field}' field. "
            f"Look for this information throughout the entire document, "
            f"including headers, footers, signature blocks, and procedural sections. "
            f"If present anywhere in the text, extract it."
        )

        improvements.append({
            "pattern": PATTERN_OMISSION,
            "draft_type": draft_type,
            "field": field_name,
            "rule_text": rule_text,
            "prompt_addition": prompt_addition,
            "priority": 1,
        })

    # --- Formatting pattern: correct value, wrong format ---
    for error_info in analysis["by_pattern"].get(PATTERN_FORMATTING, []):
        field_name = error_info["field"]
        draft_type = error_info["draft_type"]
        expected = error_info["expected"]
        extracted = error_info["extracted"]
        human_field = field_name.replace(".", " > ").replace("_", " ")

        # Detect format type from expected value
        format_hint = _detect_format(expected)

        rule_text = (
            f"Field '{human_field}' must use format: {format_hint}. "
            f"Example: '{expected}'"
        )
        prompt_addition = (
            f"FORMAT REQUIREMENT for '{human_field}': Output must match this format exactly: "
            f"{format_hint}. Example value: \"{expected}\". "
            f"Do NOT output as \"{extracted}\"."
        )

        improvements.append({
            "pattern": PATTERN_FORMATTING,
            "draft_type": draft_type,
            "field": field_name,
            "rule_text": rule_text,
            "prompt_addition": prompt_addition,
            "priority": 2,
        })

    # --- Partial pattern: incomplete extraction ---
    for error_info in analysis["by_pattern"].get(PATTERN_PARTIAL, []):
        field_name = error_info["field"]
        draft_type = error_info["draft_type"]
        human_field = field_name.replace(".", " > ").replace("_", " ")

        rule_text = (
            f"Include the COMPLETE value for '{human_field}'. "
            f"Do not truncate names, addresses, or multi-part identifiers."
        )
        prompt_addition = (
            f"COMPLETENESS for '{human_field}': Extract the FULL value including "
            f"all parts (full names with titles/suffixes, complete addresses with "
            f"apartment/suite numbers, full reference numbers). Never truncate."
        )

        improvements.append({
            "pattern": PATTERN_PARTIAL,
            "draft_type": draft_type,
            "field": field_name,
            "rule_text": rule_text,
            "prompt_addition": prompt_addition,
            "priority": 1,
        })

    # --- Wrong field pattern: value in wrong location ---
    for error_info in analysis["by_pattern"].get(PATTERN_WRONG_FIELD, []):
        field_name = error_info["field"]
        draft_type = error_info["draft_type"]
        human_field = field_name.replace(".", " > ").replace("_", " ")

        rule_text = (
            f"Ensure '{human_field}' contains ONLY the value described by its label. "
            f"Do not confuse with related but distinct fields."
        )
        prompt_addition = (
            f"DISAMBIGUATION for '{human_field}': This field must contain ONLY "
            f"the specific value matching its label. Do not place related but "
            f"different information here. Read the field name carefully."
        )

        improvements.append({
            "pattern": PATTERN_WRONG_FIELD,
            "draft_type": draft_type,
            "field": field_name,
            "rule_text": rule_text,
            "prompt_addition": prompt_addition,
            "priority": 2,
        })

    # Deduplicate: if same (draft_type, field, pattern) appears multiple times, keep one
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict] = []
    for imp in improvements:
        key = (imp["draft_type"], imp["field"], imp["pattern"])
        if key not in seen:
            seen.add(key)
            deduped.append(imp)

    # Sort by priority
    deduped.sort(key=lambda x: x["priority"])
    return deduped


def _detect_format(value: str) -> str:
    """Detect the expected format from a sample value."""
    import re

    # Date formats
    if re.match(r"\d{4}-\d{2}-\d{2}", value):
        return "YYYY-MM-DD (ISO date)"
    if re.match(r"\d{2}/\d{2}/\d{4}", value):
        return "MM/DD/YYYY"

    # Currency
    if value.startswith("$") and "," in value:
        return "$X,XXX.XX (USD with commas)"
    if value.startswith("$"):
        return "$X.XX (USD)"

    # Case numbers
    if re.match(r"[A-Z]{2,}-\d{4}-\d+", value):
        return "PREFIX-YEAR-NUMBER"

    # General
    return f"match format of: {value[:50]}"


# ---------------------------------------------------------------------------
# Apply improvements
# ---------------------------------------------------------------------------

def apply_to_learned_rules(improvements: list[dict], logger: logging.Logger, dry_run: bool = False) -> int:
    """Append new rules to config/learned_rules.yaml.

    Returns number of rules added.
    """
    import yaml

    # Load existing rules
    existing_rules: list[dict] = []
    if LEARNED_RULES_YAML.exists():
        with open(LEARNED_RULES_YAML, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        existing_rules = data.get("rules", [])

    existing_texts = {r.get("text", "") for r in existing_rules}

    # Add new rules (avoid duplicates)
    new_rules: list[dict] = []
    for imp in improvements:
        if imp["rule_text"] not in existing_texts:
            new_rules.append({
                "text": imp["rule_text"],
                "active": True,
                "draft_type": imp["draft_type"],
                "field": imp["field"],
                "pattern": imp["pattern"],
                "priority": imp["priority"],
                "added": datetime.now(timezone.utc).isoformat(),
                "source": "auto_tuning",
            })
            existing_texts.add(imp["rule_text"])

    if dry_run:
        logger.info("[DRY RUN] Would add %d new rules to learned_rules.yaml", len(new_rules))
        for rule in new_rules[:5]:
            logger.info("  - %s", rule["text"][:80])
        if len(new_rules) > 5:
            logger.info("  ... and %d more", len(new_rules) - 5)
        return len(new_rules)

    # Write updated rules
    all_rules = existing_rules + new_rules
    output_data = {
        "version": 2,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "tuning_runs": (data.get("tuning_runs", 0) if LEARNED_RULES_YAML.exists() else 0) + 1,
        "rules": all_rules,
    }

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LEARNED_RULES_YAML, "w", encoding="utf-8") as f:
        yaml.dump(output_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    logger.info("Added %d new rules to %s (total: %d)", len(new_rules), LEARNED_RULES_YAML, len(all_rules))
    return len(new_rules)


def apply_to_prompts(improvements: list[dict], logger: logging.Logger, dry_run: bool = False) -> int:
    """Update prompt templates with specific instructions based on errors.

    Appends a TUNING NOTES section to relevant prompt files.
    Returns number of prompts updated.
    """
    # Group improvements by draft type
    by_draft_type: dict[str, list[dict]] = defaultdict(list)
    for imp in improvements:
        by_draft_type[imp["draft_type"]].append(imp)

    updated_count = 0

    for draft_type, imps in by_draft_type.items():
        prompt_filename = DRAFT_TYPE_PROMPTS.get(draft_type)
        if not prompt_filename:
            logger.warning("No prompt file mapping for draft type: %s", draft_type)
            continue

        prompt_path = PROMPTS_DIR / prompt_filename
        if not prompt_path.exists():
            logger.warning("Prompt file not found: %s", prompt_path)
            continue

        # Read existing prompt
        existing_content = prompt_path.read_text(encoding="utf-8")

        # Check if tuning notes already exist
        tuning_marker = "\n\n---\nAUTO-TUNING NOTES"
        if tuning_marker in existing_content:
            # Replace existing tuning notes section
            base_content = existing_content.split(tuning_marker)[0]
        else:
            base_content = existing_content.rstrip()

        # Build tuning notes
        notes_lines = [
            f"\n\n---\nAUTO-TUNING NOTES (generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}):",
            "",
        ]

        # Group by pattern for clarity
        by_pattern: dict[str, list[dict]] = defaultdict(list)
        for imp in imps:
            by_pattern[imp["pattern"]].append(imp)

        if PATTERN_OMISSION in by_pattern:
            notes_lines.append("MISSING FIELD REMINDERS:")
            for imp in by_pattern[PATTERN_OMISSION][:10]:
                notes_lines.append(f"- {imp['prompt_addition']}")
            notes_lines.append("")

        if PATTERN_FORMATTING in by_pattern:
            notes_lines.append("FORMAT REQUIREMENTS:")
            for imp in by_pattern[PATTERN_FORMATTING][:10]:
                notes_lines.append(f"- {imp['prompt_addition']}")
            notes_lines.append("")

        if PATTERN_PARTIAL in by_pattern:
            notes_lines.append("COMPLETENESS REQUIREMENTS:")
            for imp in by_pattern[PATTERN_PARTIAL][:10]:
                notes_lines.append(f"- {imp['prompt_addition']}")
            notes_lines.append("")

        if PATTERN_WRONG_FIELD in by_pattern:
            notes_lines.append("FIELD DISAMBIGUATION:")
            for imp in by_pattern[PATTERN_WRONG_FIELD][:10]:
                notes_lines.append(f"- {imp['prompt_addition']}")
            notes_lines.append("")

        new_content = base_content + "\n".join(notes_lines)

        if dry_run:
            logger.info("[DRY RUN] Would update: %s (%d improvements)", prompt_path, len(imps))
            for line in notes_lines[:5]:
                if line.strip():
                    logger.info("    %s", line[:100])
        else:
            prompt_path.write_text(new_content, encoding="utf-8")
            logger.info("Updated prompt: %s (%d improvements applied)", prompt_path, len(imps))

        updated_count += 1

    return updated_count


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_analysis_report(analysis: dict, improvements: list[dict]) -> None:
    """Print a human-readable analysis report."""
    separator = "=" * 78
    thin_sep = "-" * 78

    print(f"\n{separator}")
    print("  PROMPT TUNING -- ERROR ANALYSIS REPORT")
    print(f"  Generated: {datetime.now(timezone.utc).isoformat()}")
    print(separator)

    summary = analysis["summary"]
    print(f"\n  Total fields evaluated:  {summary['total_fields']}")
    print(f"  Total errors found:      {summary['total_errors']}")
    print(f"  Error rate:              {summary['error_rate']:.1%}")

    print(f"\n  Error pattern breakdown:")
    for pattern, count in sorted(summary["pattern_counts"].items(), key=lambda x: -x[1]):
        label = {
            PATTERN_OMISSION: "Omission (field missed entirely)",
            PATTERN_FORMATTING: "Formatting (value correct, format wrong)",
            PATTERN_PARTIAL: "Partial (incomplete extraction)",
            PATTERN_WRONG_FIELD: "Wrong field (value in wrong slot)",
        }.get(pattern, pattern)
        print(f"    {label:<45} {count}")

    # Per draft type breakdown
    print(f"\n{thin_sep}")
    print("  ERRORS BY DRAFT TYPE")
    print(thin_sep)

    for draft_type, patterns in analysis["by_draft_type"].items():
        total = sum(len(errs) for errs in patterns.values())
        print(f"\n  {draft_type} ({total} errors):")
        for pattern, errs in sorted(patterns.items(), key=lambda x: -len(x[1])):
            print(f"    {pattern:<20} {len(errs)} errors")
            for err in errs[:3]:
                field_short = err["field"][:25]
                exp_short = str(err["expected"])[:30]
                print(f"      - {field_short}: expected='{exp_short}'")

    # Suggested improvements
    print(f"\n{separator}")
    print(f"  SUGGESTED IMPROVEMENTS ({len(improvements)} total)")
    print(separator)

    for i, imp in enumerate(improvements[:20], 1):
        priority_label = {1: "HIGH", 2: "MEDIUM", 3: "LOW"}.get(imp["priority"], "?")
        print(f"\n  [{i}] [{priority_label}] {imp['draft_type']} / {imp['field']}")
        print(f"      Pattern: {imp['pattern']}")
        print(f"      Rule: {imp['rule_text'][:80]}")

    if len(improvements) > 20:
        print(f"\n  ... and {len(improvements) - 20} more improvements")

    print(f"\n{separator}")


# ---------------------------------------------------------------------------
# Re-run pipeline
# ---------------------------------------------------------------------------

def rerun_pipeline(logger: logging.Logger) -> None:
    """Re-run the pipeline after applying improvements."""
    import subprocess

    logger.info("Re-running pipeline to measure improvement...")
    print("\nRe-running pipeline...\n")

    script_path = PROJECT_ROOT / "scripts" / "run_pipeline.py"
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=False,
        text=True,
    )

    if result.returncode != 0:
        logger.error("Pipeline re-run failed with exit code %d", result.returncode)
    else:
        logger.info("Pipeline re-run completed successfully")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_tuning(args: argparse.Namespace) -> None:
    """Execute the prompt tuning workflow."""
    logger = setup_logging()

    logger.info("LegalMind Prompt Tuning starting")
    logger.info("Mode: %s", "dry-run" if args.dry_run else "apply")

    # Step 1: Load pipeline results
    if not PIPELINE_RESULTS_PATH.exists():
        logger.error(
            "No pipeline_results.json found at %s. "
            "Run the pipeline first: python scripts/run_pipeline.py",
            PIPELINE_RESULTS_PATH,
        )
        sys.exit(1)

    with open(PIPELINE_RESULTS_PATH, encoding="utf-8") as f:
        pipeline_data = json.load(f)

    evaluations = pipeline_data.get("evaluations", [])
    if not evaluations:
        logger.error("No evaluations found in pipeline_results.json. Run with evaluation enabled.")
        sys.exit(1)

    logger.info("Loaded %d document evaluations", len(evaluations))

    # Step 2: Analyze errors
    analysis = analyze_errors(evaluations)
    logger.info(
        "Error analysis: %d errors across %d fields (%.1f%% error rate)",
        analysis["summary"]["total_errors"],
        analysis["summary"]["total_fields"],
        analysis["summary"]["error_rate"] * 100,
    )

    if analysis["summary"]["total_errors"] == 0:
        print("\nNo errors found -- all fields match expected outputs. Nothing to tune.")
        logger.info("No errors found. Exiting.")
        return

    # Step 3: Generate improvements
    improvements = generate_improvements(analysis)
    logger.info("Generated %d improvement suggestions", len(improvements))

    # Step 4: Print report
    print_analysis_report(analysis, improvements)

    # Step 5: Apply improvements
    if not improvements:
        print("\nNo actionable improvements generated.")
        return

    print(f"\n{'=' * 78}")
    if args.dry_run:
        print("  DRY RUN MODE -- showing what would be changed")
    else:
        print("  APPLYING IMPROVEMENTS")
    print(f"{'=' * 78}")

    rules_added = apply_to_learned_rules(improvements, logger, dry_run=args.dry_run)
    prompts_updated = apply_to_prompts(improvements, logger, dry_run=args.dry_run)

    print(f"\n  Rules {'would be ' if args.dry_run else ''}added:     {rules_added}")
    print(f"  Prompts {'would be ' if args.dry_run else ''}updated:  {prompts_updated}")

    # Step 6: Optionally re-run pipeline
    if args.rerun and not args.dry_run:
        rerun_pipeline(logger)
    elif args.rerun and args.dry_run:
        logger.info("[DRY RUN] Would re-run pipeline after applying changes")

    logger.info("Prompt tuning complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LegalMind Prompt Tuning -- Analyze errors and refine extraction prompts"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show suggestions without applying changes"
    )
    parser.add_argument(
        "--rerun", action="store_true",
        help="Re-run the pipeline after applying improvements to measure delta"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_tuning(args)
