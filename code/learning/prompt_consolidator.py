"""Prompt consolidator — folds learned rules into base system prompt.

Triggered when 10+ new rules have accumulated since last consolidation.
Uses Gemma-4-E4B to naturally integrate rules into the system prompt,
then archives exemplars that are now covered by the updated prompt.
"""

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

from config.paths import LEARNED_RULES_YAML, PROMPTS_DIR
from code.learning.correction_store import CorrectionStore
from code.learning.exemplar_retriever import ExemplarRetriever
from code.llm_interface.inference import InferenceEngine

logger = logging.getLogger(__name__)

CONSOLIDATION_THRESHOLD = 10

CONSOLIDATION_PROMPT = """You are rewriting a system prompt for a legal document AI.

Current system prompt:
---
{current_prompt}
---

The following rules were learned from operator corrections. Integrate them NATURALLY
into the system prompt. Do not simply append them as a list — weave them into the
relevant sections as if they were always part of the instructions.

New rules to integrate:
{rules_block}

Output the COMPLETE updated system prompt. Maintain the same overall structure and tone.
Do not remove any existing instructions — only add the new guidance where it fits best."""


class PromptConsolidator:
    """Consolidates learned rules into the base system prompt."""

    def __init__(self):
        self._engine = InferenceEngine()
        self._store = CorrectionStore()
        self._retriever = ExemplarRetriever()
        self._prompt_file = PROMPTS_DIR / "system_base.txt"

    def should_trigger(self) -> bool:
        """Check if consolidation should run (10+ new rules since last consolidation)."""
        rules_data = self._load_rules()
        if not rules_data or not rules_data.get("rules"):
            return False

        active_rules = [r for r in rules_data["rules"] if r.get("active", True)
                        and not r.get("consolidated", False)]
        return len(active_rules) >= CONSOLIDATION_THRESHOLD

    def consolidate(self) -> None:
        """Run prompt consolidation.

        1. Read current system prompt
        2. Read unconsolidated rules
        3. Use reasoning model to rewrite prompt with rules integrated
        4. Save updated prompt (with backup)
        5. Mark rules as consolidated
        6. Archive exemplars now covered by prompt rules
        """
        current_prompt = self._read_current_prompt()
        rules_data = self._load_rules()
        active_rules = [r for r in rules_data["rules"]
                        if r.get("active", True) and not r.get("consolidated", False)]

        if not active_rules:
            return

        rules_block = "\n".join(f"- {r['text']}" for r in active_rules)

        prompt = CONSOLIDATION_PROMPT.format(
            current_prompt=current_prompt,
            rules_block=rules_block,
        )

        try:
            updated_prompt = self._engine.generate_text(
                prompt, role="reasoning", max_tokens=3000, temperature=0.1
            )
        except Exception as e:
            logger.error("Prompt consolidation failed: %s", e)
            return

        if len(updated_prompt.strip()) < len(current_prompt) * 0.5:
            logger.warning("Consolidated prompt suspiciously short — aborting")
            return

        self._backup_and_save(updated_prompt.strip())
        self._mark_rules_consolidated(rules_data)
        self._archive_covered_exemplars(active_rules)

        logger.info("Prompt consolidation complete: integrated %d rules", len(active_rules))

    def _read_current_prompt(self) -> str:
        if self._prompt_file.exists():
            return self._prompt_file.read_text(encoding="utf-8")
        return ""

    def _load_rules(self) -> dict:
        if LEARNED_RULES_YAML.exists():
            with open(LEARNED_RULES_YAML, encoding="utf-8") as f:
                return yaml.safe_load(f) or {"rules": [], "metadata": {}}
        return {"rules": [], "metadata": {}}

    def _backup_and_save(self, new_prompt: str) -> None:
        """Backup current prompt and save the new one."""
        if self._prompt_file.exists():
            backup = self._prompt_file.with_suffix(
                f".bak.{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            )
            shutil.copy2(self._prompt_file, backup)

        self._prompt_file.write_text(new_prompt, encoding="utf-8")

    def _mark_rules_consolidated(self, rules_data: dict) -> None:
        """Mark all active rules as consolidated."""
        for rule in rules_data["rules"]:
            if rule.get("active", True) and not rule.get("consolidated", False):
                rule["consolidated"] = True
                rule["consolidated_at"] = datetime.now(timezone.utc).isoformat()

        rules_data.setdefault("metadata", {})["last_consolidation"] = (
            datetime.now(timezone.utc).isoformat()
        )

        with open(LEARNED_RULES_YAML, "w", encoding="utf-8") as f:
            yaml.dump(rules_data, f, default_flow_style=False, allow_unicode=True)

    def _archive_covered_exemplars(self, consolidated_rules: list[dict]) -> None:
        """Archive exemplars whose corrections are now covered by prompt rules.

        Heuristic: archive the oldest exemplars up to the count of consolidated rules,
        keeping the exemplar bank lean. More sophisticated matching could be added later.
        """
        all_active = []
        for draft_type in ["case_fact_summary", "title_review_summary",
                           "notice_summary", "document_checklist"]:
            all_active.extend(self._store.get_all_active(draft_type))

        all_active.sort(key=lambda c: c.timestamp)
        to_archive = all_active[:len(consolidated_rules)]

        if to_archive:
            self._store.archive_corrections([c.id for c in to_archive])
            logger.info("Archived %d old exemplars after consolidation", len(to_archive))
