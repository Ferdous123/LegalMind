"""Correction store — persistent storage for operator corrections.

Storage format: JSONL files in data/corrections/ (one file per draft_type).
Each correction captures: source OCR text, generated output, operator's edit,
field metadata, and embedding for semantic retrieval.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config.paths import CORRECTIONS_DIR

logger = logging.getLogger(__name__)


@dataclass
class Correction:
    """A single operator correction record."""
    id: str = ""
    timestamp: str = ""
    document_id: str = ""
    draft_type: str = ""
    field_path: str = ""
    source_ocr_chunk: str = ""
    generated_text: str = ""
    edited_text: str = ""
    correction_type: str = ""  # omission, error, style, restructure
    embedding: list[float] = field(default_factory=list)
    active: bool = True

    def __post_init__(self):
        if not self.id:
            self.id = f"corr_{uuid.uuid4().hex[:8]}"
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class CorrectionStore:
    """Manages persistent storage of operator corrections.

    Corrections are stored as JSONL files, one per draft_type, in data/corrections/.
    Provides CRUD operations and count-based triggers for the learning pipeline.
    """

    def __init__(self):
        CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    def _get_file(self, draft_type: str) -> Path:
        return CORRECTIONS_DIR / f"{draft_type}.jsonl"

    def save_correction(self, correction: Correction) -> str:
        """Append a correction to the store. Returns correction ID."""
        filepath = self._get_file(correction.draft_type)
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(correction)) + "\n")
        logger.info("Saved correction %s for field %s", correction.id, correction.field_path)

        # Vocabulary learning hook: extract novel legal terms from the correction
        try:
            from code.corpus.vocabulary_learner import VocabularyLearner
            learner = VocabularyLearner()
            learner.learn_from_correction(asdict(correction))
        except Exception:
            pass  # vocabulary learning is non-critical

        return correction.id

    def get_all_active(self, draft_type: str) -> list[Correction]:
        """Get all active (non-archived) corrections for a draft type."""
        filepath = self._get_file(draft_type)
        if not filepath.exists():
            return []
        corrections = []
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if data.get("active", True):
                    corrections.append(Correction(**data))
        return corrections

    def get_count(self, draft_type: Optional[str] = None) -> int:
        """Count total corrections, optionally filtered by draft_type."""
        if draft_type:
            return len(self.get_all_active(draft_type))
        total = 0
        for f in CORRECTIONS_DIR.glob("*.jsonl"):
            with open(f, encoding="utf-8") as fh:
                total += sum(1 for line in fh if line.strip())
        return total

    def archive_corrections(self, ids: list[str]) -> None:
        """Mark corrections as inactive (archived). They remain in file but won't be retrieved."""
        for filepath in CORRECTIONS_DIR.glob("*.jsonl"):
            lines = []
            modified = False
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data["id"] in ids:
                        data["active"] = False
                        modified = True
                    lines.append(json.dumps(data))
            if modified:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")
        logger.info("Archived %d corrections", len(ids))

    def get_corrections_since(self, timestamp: str) -> list[Correction]:
        """Get all corrections created after the given ISO timestamp."""
        results = []
        for filepath in CORRECTIONS_DIR.glob("*.jsonl"):
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get("timestamp", "") > timestamp:
                        results.append(Correction(**data))
        return sorted(results, key=lambda c: c.timestamp)

    def get_recent(self, n: int = 20) -> list[Correction]:
        """Get the N most recent corrections across all draft types."""
        all_corrections = []
        for filepath in CORRECTIONS_DIR.glob("*.jsonl"):
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get("active", True):
                        all_corrections.append(Correction(**data))
        all_corrections.sort(key=lambda c: c.timestamp, reverse=True)
        return all_corrections[:n]
