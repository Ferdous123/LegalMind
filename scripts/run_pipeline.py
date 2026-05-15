"""
LegalMind Production Pipeline -- End-to-End Processing + Evaluation.

Processes all sample documents through the complete pipeline:
  Document -> Text Extraction -> Structuring -> Indexing -> Draft Generation -> Verification

Then evaluates against ground truth and reports metrics.

Usage:
    python scripts/run_pipeline.py              # process all samples
    python scripts/run_pipeline.py --doc 1      # process only doc 1
    python scripts/run_pipeline.py --eval-only  # skip processing, just evaluate
    python scripts/run_pipeline.py --no-gpu     # use mock models (for testing)
"""

import sys
import os
import json
import logging
import argparse
import time
from pathlib import Path
from datetime import datetime, timezone
from difflib import SequenceMatcher
from dataclasses import asdict
from typing import Optional

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# CUDA DLL setup on Windows
if sys.platform == "win32":
    cuda_paths = [
        Path(os.environ.get("CUDA_PATH", "")) / "bin",
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"),
    ]
    for cuda_bin in cuda_paths:
        if cuda_bin.exists():
            os.add_dll_directory(str(cuda_bin))
            os.environ["PATH"] = str(cuda_bin) + os.pathsep + os.environ.get("PATH", "")
            break

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from config.paths import (
    PROCESSED_DIR, SAMPLE_DIR, DATA_DIR, MODEL_CACHE_DIR,
    LOGS_DIR, ensure_dirs,
)

# Model GGUF files to verify
MODEL_FILES = [
    "LightOnOCR-1B-1025-Q8_0.gguf",
    "Qwen3VL-8B-Instruct-Q4_K_M.gguf",
    "gemma-4-E4B-it-Q4_K_M.gguf",
]

# Document -> draft type mapping
DOCUMENT_DRAFT_MAP: dict[str, str] = {
    "doc_1_lease_agreement.txt": "case_fact_summary",
    "doc_2_court_filing.txt": "case_fact_summary",
    "doc_3_property_deed.txt": "title_review_summary",
    "doc_4_compliance_notice.txt": "notice_summary",
    "doc_5_document_checklist.txt": "document_checklist",
}

# Expected output file mapping
EXPECTED_OUTPUTS_DIR = SAMPLE_DIR / "expected_outputs"

PIPELINE_RESULTS_PATH = DATA_DIR / "pipeline_results.json"

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    """Configure logging to both file and console."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"pipeline_run_{timestamp}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # File handler — verbose
    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    ))
    root_logger.addHandler(fh)

    # Console handler — concise
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)-8s | %(message)s"))
    root_logger.addHandler(ch)

    logger = logging.getLogger("legalmind.pipeline")
    logger.info("Log file: %s", log_file)
    return logger


# ---------------------------------------------------------------------------
# GPU / Model availability check
# ---------------------------------------------------------------------------

def check_models_available() -> tuple[bool, list[str]]:
    """Check if all required GGUF model files exist.

    Returns:
        (all_available, list_of_missing_files)
    """
    missing = []
    for model_file in MODEL_FILES:
        path = MODEL_CACHE_DIR / model_file
        if not path.exists():
            missing.append(model_file)
    return len(missing) == 0, missing


# ---------------------------------------------------------------------------
# Phase 1: Document Processing
# ---------------------------------------------------------------------------

def process_document_gpu(doc_path: Path, draft_type: str, logger: logging.Logger) -> dict:
    """Process a single document through the full GPU pipeline.

    Steps:
    1. DocumentIngester.process(doc_path, draft_type) -> ProcessedDocument
    2. DocumentIndexer.index_document(doc.id, doc.chunks)
    3. DraftGenerator.generate_draft(doc.id, draft_type, doc.full_text)
    4. FirewallRunner.verify_draft(...)
    5. Return full result dict with timing
    """
    from code.pipeline.ingestion import DocumentIngester
    from code.retrieval.indexer import DocumentIndexer
    from code.generation.drafter import DraftGenerator
    from code.firewall.runner import FirewallRunner

    result = {
        "document": doc_path.name,
        "draft_type": draft_type,
        "status": "success",
        "phases": {},
        "structured_fields": {},
        "draft_content": "",
        "verification_summary": {},
        "errors": [],
    }

    # Phase 1a: Ingestion
    t0 = time.perf_counter()
    try:
        ingester = DocumentIngester()
        doc = ingester.process(str(doc_path), draft_type=draft_type)
        result["phases"]["ingestion_sec"] = round(time.perf_counter() - t0, 2)
        result["structured_fields"] = doc.structured_fields
        result["document_id"] = doc.id
        result["full_text_length"] = len(doc.full_text)
        result["page_count"] = doc.page_count
        result["confidence"] = doc.confidence
        logger.info("  Ingestion complete: %d pages, confidence=%.2f",
                    doc.page_count, doc.confidence)
    except Exception as e:
        result["phases"]["ingestion_sec"] = round(time.perf_counter() - t0, 2)
        result["errors"].append(f"Ingestion failed: {e}")
        result["status"] = "partial_failure"
        logger.error("  Ingestion failed: %s", e)
        return result

    # Phase 1b: Indexing
    t0 = time.perf_counter()
    try:
        indexer = DocumentIndexer()
        chunk_count = indexer.index_document(doc.id, doc.chunks)
        result["phases"]["indexing_sec"] = round(time.perf_counter() - t0, 2)
        result["chunks_indexed"] = chunk_count
        logger.info("  Indexing complete: %d chunks", chunk_count)
    except Exception as e:
        result["phases"]["indexing_sec"] = round(time.perf_counter() - t0, 2)
        result["errors"].append(f"Indexing failed: {e}")
        logger.warning("  Indexing failed (non-fatal): %s", e)

    # Phase 1c: Draft generation
    t0 = time.perf_counter()
    try:
        drafter = DraftGenerator()
        draft_output = drafter.generate_draft(
            document_id=doc.id,
            draft_type=draft_type,
            full_text=doc.full_text,
        )
        result["phases"]["generation_sec"] = round(time.perf_counter() - t0, 2)
        result["draft_content"] = draft_output.content_markdown
        result["draft_confidence"] = draft_output.confidence_overall
        result["evidence_count"] = draft_output.evidence_count
        result["rules_applied"] = draft_output.rules_applied
        logger.info("  Draft generation complete: confidence=%.2f, evidence=%d",
                    draft_output.confidence_overall, draft_output.evidence_count)
    except Exception as e:
        result["phases"]["generation_sec"] = round(time.perf_counter() - t0, 2)
        result["errors"].append(f"Draft generation failed: {e}")
        result["status"] = "partial_failure"
        logger.error("  Draft generation failed: %s", e)

    # Phase 1d: Firewall verification
    t0 = time.perf_counter()
    try:
        firewall = FirewallRunner()
        verification_results = firewall.verify_draft(
            draft_content=result.get("draft_content", ""),
            structured_fields=doc.structured_fields,
            citation_map=draft_output.citations if "draft_output" in dir() else {},
            ocr_confidence=doc.confidence,
        )
        summary = firewall.get_summary(verification_results)
        result["phases"]["verification_sec"] = round(time.perf_counter() - t0, 2)
        result["verification_summary"] = summary
        logger.info("  Verification complete: %d/%d verified",
                    summary.get("verified", 0), summary.get("total", 0))
    except Exception as e:
        result["phases"]["verification_sec"] = round(time.perf_counter() - t0, 2)
        result["errors"].append(f"Verification failed: {e}")
        logger.warning("  Verification failed (non-fatal): %s", e)

    return result


def process_document_no_gpu(doc_path: Path, draft_type: str, logger: logging.Logger) -> dict:
    """Process a document without GPU models (pipeline flow validation).

    Reads the text file directly, creates a ProcessedDocument-like structure,
    performs chunking, and returns a result dict without LLM inference.
    """
    from code.pipeline.chunker import TextChunker

    result = {
        "document": doc_path.name,
        "draft_type": draft_type,
        "status": "success",
        "mode": "no-gpu",
        "phases": {},
        "structured_fields": {},
        "draft_content": "",
        "verification_summary": {},
        "errors": [],
    }

    # Phase 1a: Read and structure the text file directly
    t0 = time.perf_counter()
    try:
        text = doc_path.read_text(encoding="utf-8")
        result["full_text_length"] = len(text)
        result["page_count"] = 1
        result["confidence"] = 1.0  # direct text, no OCR
        result["document_id"] = f"doc_nogpu_{doc_path.stem}"
        result["phases"]["ingestion_sec"] = round(time.perf_counter() - t0, 2)
        logger.info("  Text read: %d characters", len(text))
    except Exception as e:
        result["phases"]["ingestion_sec"] = round(time.perf_counter() - t0, 2)
        result["errors"].append(f"Text read failed: {e}")
        result["status"] = "failed"
        logger.error("  Failed to read document: %s", e)
        return result

    # Phase 1b: Chunking (no GPU needed)
    t0 = time.perf_counter()
    try:
        chunker = TextChunker()
        chunks = chunker.chunk(text, result["document_id"])
        result["chunks_indexed"] = len(chunks)
        result["phases"]["indexing_sec"] = round(time.perf_counter() - t0, 2)
        logger.info("  Chunking complete: %d chunks", len(chunks))
    except Exception as e:
        result["phases"]["indexing_sec"] = round(time.perf_counter() - t0, 2)
        result["errors"].append(f"Chunking failed: {e}")
        logger.warning("  Chunking failed (non-fatal): %s", e)

    # Phase 1c: No draft generation without GPU
    result["phases"]["generation_sec"] = 0.0
    result["draft_content"] = "[no-gpu mode: draft generation skipped]"
    result["draft_confidence"] = 0.0
    logger.info("  Draft generation skipped (no-gpu mode)")

    # Phase 1d: No verification without GPU
    result["phases"]["verification_sec"] = 0.0
    result["verification_summary"] = {"total": 0, "verified": 0, "note": "skipped in no-gpu mode"}
    logger.info("  Verification skipped (no-gpu mode)")

    # In no-gpu mode, structured_fields is empty — LLM would normally extract these
    result["structured_fields"] = {}

    return result


# ---------------------------------------------------------------------------
# Phase 2: Evaluation
# ---------------------------------------------------------------------------

def normalize_value(val) -> str:
    """Normalize a value for comparison: lowercase, strip whitespace, remove extra spaces."""
    if val is None:
        return ""
    if isinstance(val, (list, dict)):
        return json.dumps(val, sort_keys=True, ensure_ascii=False).lower().strip()
    return str(val).lower().strip()


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict[str, str]:
    """Flatten a nested dict into dot-separated keys with string values."""
    items: list[tuple[str, str]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        elif isinstance(v, list):
            items.append((new_key, json.dumps(v, sort_keys=True, ensure_ascii=False)))
        else:
            items.append((new_key, str(v) if v is not None else ""))
    return dict(items)


def evaluate_document(processed: dict, expected: dict) -> dict:
    """Compare processed output against ground truth.

    For each expected field:
    - Exact match: normalized string comparison
    - Soft match: SequenceMatcher ratio >= 0.7
    - Partial match: key substring contained in extracted value
    - Miss: no match at all

    Returns per-field accuracy + overall scores.
    """
    expected_fields = expected.get("expected_fields", {})
    extracted_fields = processed.get("structured_fields", {})

    # Flatten both for comparison
    expected_flat = flatten_dict(expected_fields)
    extracted_flat = flatten_dict(extracted_fields)

    field_results = []
    exact_matches = 0
    soft_matches = 0
    partial_matches = 0
    misses = 0

    for field_key, expected_val in expected_flat.items():
        norm_expected = normalize_value(expected_val)
        if not norm_expected:
            continue  # skip empty expected values

        # Find best matching extracted field
        best_match_type = "miss"
        best_extracted = ""
        best_ratio = 0.0

        # Try exact key match first
        if field_key in extracted_flat:
            norm_extracted = normalize_value(extracted_flat[field_key])
            best_extracted = extracted_flat[field_key]
            ratio = SequenceMatcher(None, norm_expected, norm_extracted).ratio()
            best_ratio = ratio

            if norm_expected == norm_extracted:
                best_match_type = "exact"
            elif ratio >= 0.7:
                best_match_type = "soft"
            elif norm_expected in norm_extracted or norm_extracted in norm_expected:
                best_match_type = "partial"
        else:
            # Try to find the field in any key
            for ext_key, ext_val in extracted_flat.items():
                norm_extracted = normalize_value(ext_val)
                ratio = SequenceMatcher(None, norm_expected, norm_extracted).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_extracted = ext_val
                    if norm_expected == norm_extracted:
                        best_match_type = "exact"
                    elif ratio >= 0.7:
                        best_match_type = "soft"
                    elif norm_expected in norm_extracted or norm_extracted in norm_expected:
                        best_match_type = "partial"

        if best_match_type == "exact":
            exact_matches += 1
        elif best_match_type == "soft":
            soft_matches += 1
        elif best_match_type == "partial":
            partial_matches += 1
        else:
            misses += 1

        field_results.append({
            "field": field_key,
            "expected": expected_val,
            "extracted": best_extracted,
            "match_type": best_match_type,
            "similarity": round(best_ratio, 3),
        })

    total_fields = len(field_results)
    if total_fields == 0:
        return {
            "document": processed.get("document", ""),
            "total_fields": 0,
            "accuracy": 0.0,
            "field_results": [],
        }

    # Weighted accuracy: exact=1.0, soft=0.8, partial=0.5, miss=0.0
    weighted_score = (
        exact_matches * 1.0
        + soft_matches * 0.8
        + partial_matches * 0.5
        + misses * 0.0
    )
    accuracy = weighted_score / total_fields

    return {
        "document": processed.get("document", ""),
        "draft_type": processed.get("draft_type", ""),
        "total_fields": total_fields,
        "exact_matches": exact_matches,
        "soft_matches": soft_matches,
        "partial_matches": partial_matches,
        "misses": misses,
        "accuracy": round(accuracy, 4),
        "field_results": field_results,
    }


# ---------------------------------------------------------------------------
# Phase 3: Report Generation
# ---------------------------------------------------------------------------

def generate_report(results: list[dict], evaluations: list[dict], logger: logging.Logger) -> None:
    """Print formatted evaluation report.

    Shows:
    - Per-document: filename, draft_type, field accuracy, confidence
    - Per-field: expected vs. extracted, match type
    - Overall: extraction accuracy, retrieval quality, grounding score
    """
    separator = "=" * 78
    thin_sep = "-" * 78

    print(f"\n{separator}")
    print("  LEGALMIND PIPELINE -- EVALUATION REPORT")
    print(f"  Generated: {datetime.now(timezone.utc).isoformat()}")
    print(separator)

    # Per-document summary
    print(f"\n{'Document':<35} {'Draft Type':<22} {'Accuracy':<10} {'Fields':<8} {'Status'}")
    print(thin_sep)
    for res, ev in zip(results, evaluations):
        doc_name = res.get("document", "???")[:34]
        draft_type = res.get("draft_type", "???")[:21]
        accuracy = f"{ev.get('accuracy', 0.0):.1%}" if ev else "N/A"
        fields = str(ev.get("total_fields", 0)) if ev else "0"
        status = res.get("status", "unknown")
        print(f"{doc_name:<35} {draft_type:<22} {accuracy:<10} {fields:<8} {status}")

    # Detailed per-field breakdown
    print(f"\n{separator}")
    print("  FIELD-LEVEL DETAIL")
    print(separator)

    for ev in evaluations:
        if not ev or not ev.get("field_results"):
            continue
        print(f"\n  [{ev['document']}] ({ev['draft_type']})")
        print(f"  {'Field':<30} {'Match':<8} {'Sim':<6} {'Expected (truncated)'}")
        print(f"  {thin_sep}")
        for fr in ev["field_results"]:
            field_name = fr["field"][:29]
            match_type = fr["match_type"]
            sim = f"{fr['similarity']:.2f}"
            expected_trunc = str(fr["expected"])[:40]
            # Color-code match type in terminal
            marker = {
                "exact": "[OK]",
                "soft": "[~~]",
                "partial": "[..] ",
                "miss": "[XX]",
            }.get(match_type, "[??]")
            print(f"  {field_name:<30} {marker:<8} {sim:<6} {expected_trunc}")

    # Overall summary
    print(f"\n{separator}")
    print("  OVERALL METRICS")
    print(separator)

    if evaluations:
        total_fields_all = sum(e.get("total_fields", 0) for e in evaluations)
        total_exact = sum(e.get("exact_matches", 0) for e in evaluations)
        total_soft = sum(e.get("soft_matches", 0) for e in evaluations)
        total_partial = sum(e.get("partial_matches", 0) for e in evaluations)
        total_misses = sum(e.get("misses", 0) for e in evaluations)
        avg_accuracy = sum(e.get("accuracy", 0.0) for e in evaluations) / len(evaluations)

        print(f"\n  Total fields evaluated:   {total_fields_all}")
        print(f"  Exact matches:            {total_exact}")
        print(f"  Soft matches (>=0.7):     {total_soft}")
        print(f"  Partial matches:          {total_partial}")
        print(f"  Misses:                   {total_misses}")
        print(f"  Weighted accuracy:        {avg_accuracy:.1%}")
    else:
        print("\n  No evaluations available.")

    # Timing summary
    print(f"\n{thin_sep}")
    print("  TIMING (seconds)")
    print(thin_sep)
    print(f"  {'Document':<35} {'Ingest':<10} {'Index':<10} {'Generate':<10} {'Verify':<10} {'Total'}")
    for res in results:
        phases = res.get("phases", {})
        doc_name = res.get("document", "???")[:34]
        ingest = phases.get("ingestion_sec", 0)
        index = phases.get("indexing_sec", 0)
        gen = phases.get("generation_sec", 0)
        verify = phases.get("verification_sec", 0)
        total = ingest + index + gen + verify
        print(f"  {doc_name:<35} {ingest:<10.2f} {index:<10.2f} {gen:<10.2f} {verify:<10.2f} {total:.2f}")

    total_time = sum(
        sum(r.get("phases", {}).values())
        for r in results
    )
    print(f"\n  Total pipeline time: {total_time:.2f} seconds")
    print(separator)


# ---------------------------------------------------------------------------
# Load expected outputs
# ---------------------------------------------------------------------------

def load_expected(doc_filename: str) -> Optional[dict]:
    """Load the expected output JSON for a document."""
    stem = Path(doc_filename).stem
    expected_path = EXPECTED_OUTPUTS_DIR / f"{stem}_expected.json"
    if not expected_path.exists():
        return None
    with open(expected_path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main pipeline orchestration
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full pipeline: process -> evaluate -> report."""
    logger = setup_logging()
    ensure_dirs()

    logger.info("LegalMind Production Pipeline starting")
    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Mode: %s", "no-gpu" if args.no_gpu else "GPU")

    # Check model availability
    models_ok, missing_models = check_models_available()
    if not models_ok and not args.no_gpu:
        logger.warning(
            "Model files not found at %s: %s",
            MODEL_CACHE_DIR,
            ", ".join(missing_models),
        )
        logger.warning("Falling back to no-gpu mode. Use --no-gpu to suppress this warning.")
        args.no_gpu = True

    if not args.no_gpu:
        logger.info("GPU mode: all model files verified at %s", MODEL_CACHE_DIR)
    else:
        logger.info("No-GPU mode: pipeline flow validation only (no LLM inference)")

    # Determine which documents to process
    doc_files: list[tuple[str, str]] = []
    if args.doc:
        # Single document mode
        doc_num = args.doc
        matching = [
            (fname, dtype)
            for fname, dtype in DOCUMENT_DRAFT_MAP.items()
            if f"doc_{doc_num}_" in fname
        ]
        if not matching:
            logger.error("No document found for --doc %s", doc_num)
            sys.exit(1)
        doc_files = matching
    else:
        doc_files = list(DOCUMENT_DRAFT_MAP.items())

    # ---------------------------------------------------------------------------
    # Process documents (skip if --eval-only)
    # ---------------------------------------------------------------------------
    results: list[dict] = []

    if not args.eval_only:
        logger.info("Processing %d document(s)...", len(doc_files))
        print(f"\nProcessing {len(doc_files)} document(s)...\n")

        for i, (doc_filename, draft_type) in enumerate(doc_files, 1):
            doc_path = SAMPLE_DIR / doc_filename
            if not doc_path.exists():
                logger.error("Document not found: %s", doc_path)
                results.append({
                    "document": doc_filename,
                    "draft_type": draft_type,
                    "status": "not_found",
                    "phases": {},
                    "structured_fields": {},
                    "errors": [f"File not found: {doc_path}"],
                })
                continue

            logger.info("[%d/%d] Processing: %s (type: %s)",
                        i, len(doc_files), doc_filename, draft_type)
            print(f"  [{i}/{len(doc_files)}] {doc_filename} -> {draft_type}")

            try:
                if args.no_gpu:
                    result = process_document_no_gpu(doc_path, draft_type, logger)
                else:
                    result = process_document_gpu(doc_path, draft_type, logger)
                results.append(result)
            except Exception as e:
                logger.error("Unhandled error processing %s: %s", doc_filename, e, exc_info=True)
                results.append({
                    "document": doc_filename,
                    "draft_type": draft_type,
                    "status": "failed",
                    "phases": {},
                    "structured_fields": {},
                    "errors": [f"Unhandled error: {e}"],
                })

        # Save results
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        pipeline_output = {
            "run_timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": "no-gpu" if args.no_gpu else "gpu",
            "documents_processed": len(results),
            "results": results,
        }
        with open(PIPELINE_RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(pipeline_output, f, indent=2, ensure_ascii=False)
        logger.info("Pipeline results saved to: %s", PIPELINE_RESULTS_PATH)
    else:
        # Load previous results for eval-only mode
        if not PIPELINE_RESULTS_PATH.exists():
            logger.error("No pipeline_results.json found. Run the pipeline first.")
            sys.exit(1)
        with open(PIPELINE_RESULTS_PATH, encoding="utf-8") as f:
            pipeline_output = json.load(f)
        results = pipeline_output.get("results", [])
        logger.info("Loaded %d results from previous run (eval-only mode)", len(results))

    # ---------------------------------------------------------------------------
    # Evaluate against ground truth
    # ---------------------------------------------------------------------------
    logger.info("Evaluating against ground truth...")
    evaluations: list[dict] = []

    for result in results:
        doc_filename = result.get("document", "")
        expected = load_expected(doc_filename)
        if expected is None:
            logger.warning("No expected output for: %s", doc_filename)
            evaluations.append({
                "document": doc_filename,
                "draft_type": result.get("draft_type", ""),
                "total_fields": 0,
                "accuracy": 0.0,
                "field_results": [],
                "note": "no expected output file",
            })
            continue

        # Handle notice_summary vs notice_related_summary mismatch
        expected_draft_type = expected.get("draft_type", "")
        actual_draft_type = result.get("draft_type", "")
        if expected_draft_type in ("notice_related_summary", "notice_summary"):
            # These are equivalent — normalize for comparison
            pass

        evaluation = evaluate_document(result, expected)
        evaluations.append(evaluation)

    # Save evaluations alongside pipeline results
    pipeline_output["evaluations"] = evaluations
    with open(PIPELINE_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(pipeline_output, f, indent=2, ensure_ascii=False)

    # ---------------------------------------------------------------------------
    # Generate report
    # ---------------------------------------------------------------------------
    generate_report(results, evaluations, logger)

    logger.info("Pipeline run complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LegalMind Production Pipeline -- End-to-End Processing + Evaluation"
    )
    parser.add_argument(
        "--doc", type=int, default=None,
        help="Process only the specified document number (1-5)"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip processing, evaluate from existing pipeline_results.json"
    )
    parser.add_argument(
        "--no-gpu", action="store_true",
        help="Run without GPU models (validates pipeline flow only)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
