"""Pattern extractor — analyzes correction clusters to extract reusable rules.

Triggered every 20 new corrections (or on forced manual run). Clusters similar
corrections by field category and correction type, then uses Gemma-4-E4B to
extract generalizable rules from each cluster.  Rules are stored in
config/learned_rules.yaml.
"""

import logging
from collections import defaultdict
from datetime import datetime, timezone

import yaml

from config.paths import LEARNED_RULES_YAML
from code.learning.correction_store import CorrectionStore, Correction
from code.llm_interface.inference import InferenceEngine

logger = logging.getLogger(__name__)

TRIGGER_INTERVAL = 20
CLUSTER_MIN_SIZE = 3
FORCE_CLUSTER_MIN_SIZE = 2  # lower threshold when manually forced

PATTERN_EXTRACTION_PROMPT = """Analyze these operator corrections to a legal document AI system.
Each correction shows: what the source text said, what the AI generated, and what the operator fixed it to.

Corrections:
{corrections_block}

Based on these corrections, identify the COMMON PATTERN they all fix.
Write exactly ONE concise rule (1-2 sentences) that would prevent this type of error in future generations.

The rule should be:
- Actionable (tells the system exactly what to do)
- General (applies beyond just these specific documents)
- Specific enough to be useful (not vague platitudes)

Output ONLY the rule text, nothing else."""


class PatternExtractor:
    """Extracts generalizable rules from clusters of similar corrections."""

    def __init__(self):
        self._store = CorrectionStore()
        self._engine = InferenceEngine()
        self._last_trigger_count = self._load_last_trigger_count()

    def should_trigger(self) -> bool:
        """Check if pattern extraction should run (every 20 corrections)."""
        current = self._store.get_count()
        return current >= self._last_trigger_count + TRIGGER_INTERVAL

    def extract_patterns(self, force: bool = False) -> list[str]:
        """Run pattern extraction on accumulated corrections.

        Args:
            force: If True, run even if fewer than TRIGGER_INTERVAL corrections
                   exist and use a lower minimum cluster size.

        Returns:
            List of newly extracted rule strings.
        """
        recent = self._store.get_recent(n=500)
        if not recent:
            return []
        if not force and len(recent) < TRIGGER_INTERVAL:
            return []

        work_set = recent if force else recent[:TRIGGER_INTERVAL]
        min_size = FORCE_CLUSTER_MIN_SIZE if (force and len(work_set) < TRIGGER_INTERVAL) else CLUSTER_MIN_SIZE

        clusters = self._cluster_corrections(work_set)
        new_rules = []

        for cluster in clusters:
            if len(cluster) < min_size:
                continue
            rule = self._extract_rule_from_cluster(cluster)
            if rule:
                new_rules.append(rule)

        if new_rules:
            self._save_rules(new_rules)
            self._update_last_trigger_count()

        logger.info("Extracted %d new rules from %d clusters", len(new_rules), len(clusters))
        return new_rules

    def _cluster_corrections(self, corrections: list[Correction]) -> list[list[Correction]]:
        """Cluster corrections by field category and correction type.

        Groups corrections that share the same field prefix (e.g. 'parties',
        'key_dates') and correction type. No embedding model required — this
        produces semantically meaningful clusters for legal field corrections.
        """
        groups: dict[str, list[Correction]] = defaultdict(list)
        for c in corrections:
            field_prefix = c.field_path.split(".")[0] if c.field_path else "general"
            key = f"{field_prefix}:{c.correction_type or 'unknown'}"
            groups[key].append(c)
        return list(groups.values())

    def _extract_rule_from_cluster(self, cluster: list[Correction]) -> str:
        """Use Gemma-4-E4B to extract a rule from a correction cluster."""
        corrections_block = ""
        for i, c in enumerate(cluster[:5], 1):
            corrections_block += (
                f"\nCorrection {i}:\n"
                f"  Source: \"{c.source_ocr_chunk[:150]}\"\n"
                f"  AI generated: \"{c.generated_text[:150]}\"\n"
                f"  Operator fixed to: \"{c.edited_text[:150]}\"\n"
                f"  Field: {c.field_path}\n"
            )

        prompt = PATTERN_EXTRACTION_PROMPT.format(corrections_block=corrections_block)

        try:
            rule = self._engine.generate_text(
                prompt, role="reasoning", max_tokens=200, temperature=0.1
            )
            return rule.strip()
        except Exception as e:
            logger.error("Rule extraction failed: %s", e)
            return ""

    def _save_rules(self, new_rules: list[str]) -> None:
        """Append new rules to config/learned_rules.yaml."""
        LEARNED_RULES_YAML.parent.mkdir(parents=True, exist_ok=True)

        existing = {"rules": [], "metadata": {}}
        if LEARNED_RULES_YAML.exists():
            with open(LEARNED_RULES_YAML, encoding="utf-8") as f:
                existing = yaml.safe_load(f) or {"rules": [], "metadata": {}}

        existing_texts = {r["text"].strip() for r in existing.get("rules", []) if isinstance(r, dict)}
        added = 0
        for rule in new_rules:
            if rule.strip() not in existing_texts:
                existing["rules"].append({
                    "text": rule,
                    "extracted_at": datetime.now(timezone.utc).isoformat(),
                    "active": True,
                })
                existing_texts.add(rule.strip())
                added += 1
        if not added:
            return

        existing.setdefault("metadata", {})["last_extraction"] = datetime.now(timezone.utc).isoformat()
        existing["metadata"]["total_rules"] = len(existing["rules"])

        with open(LEARNED_RULES_YAML, "w", encoding="utf-8") as f:
            yaml.dump(existing, f, default_flow_style=False, allow_unicode=True)

    def _load_last_trigger_count(self) -> int:
        """Load the correction count at last trigger from metadata."""
        if LEARNED_RULES_YAML.exists():
            with open(LEARNED_RULES_YAML, encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data and "metadata" in data:
                    return data["metadata"].get("trigger_count", 0)
        return 0

    def _update_last_trigger_count(self) -> None:
        """Update stored trigger count after extraction."""
        if LEARNED_RULES_YAML.exists():
            with open(LEARNED_RULES_YAML, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {"rules": [], "metadata": {}}
        data.setdefault("metadata", {})["trigger_count"] = self._store.get_count()
        with open(LEARNED_RULES_YAML, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
