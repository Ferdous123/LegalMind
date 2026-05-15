"""Draft generator — produces grounded legal drafts from evidence.

This is the core generation engine. It assembles the full prompt from:
1. Base system prompt (with learned rules baked in)
2. Few-shot exemplars from the correction bank
3. Retrieved evidence passages
4. Draft-type-specific task instructions
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from config.paths import PROMPTS_DIR, LEARNED_RULES_YAML
from code.llm_interface.inference import InferenceEngine
from code.retrieval.searcher import EvidenceSearcher
from code.retrieval.evidence import EvidencePackager
from code.learning.exemplar_retriever import ExemplarRetriever
from code.generation.grounding import GroundingVerifier

logger = logging.getLogger(__name__)


@dataclass
class DraftOutput:
    """Complete draft generation output."""
    draft_type: str = ""
    document_id: str = ""
    content_markdown: str = ""
    content_structured: dict = field(default_factory=dict)
    citations: dict = field(default_factory=dict)
    verification_status: dict = field(default_factory=dict)
    generation_timestamp: str = ""
    exemplars_used: list = field(default_factory=list)
    rules_applied: list = field(default_factory=list)
    evidence_count: int = 0
    confidence_overall: float = 0.0

    def __post_init__(self):
        if not self.generation_timestamp:
            self.generation_timestamp = datetime.now(timezone.utc).isoformat()


class DraftGenerator:
    """Generates grounded drafts using evidence + exemplars + learned rules."""

    def __init__(self):
        self._engine = InferenceEngine()
        self._searcher = EvidenceSearcher()
        self._packager = EvidencePackager()
        self._exemplar_retriever = ExemplarRetriever()
        self._grounding = GroundingVerifier()

    def generate_draft(self, document_id: str, draft_type: str,
                       full_text: str = "", top_k_evidence: int = 8) -> DraftOutput:
        """Generate a grounded draft for a processed document.

        Args:
            document_id: ID of the processed document.
            draft_type: Type of draft to generate.
            full_text: Full document text (for exemplar matching).
            top_k_evidence: Number of evidence chunks to retrieve.

        Returns:
            DraftOutput with generated content and metadata.
        """
        output = DraftOutput(draft_type=draft_type, document_id=document_id)

        # 1. Retrieve evidence
        query = self._build_retrieval_query(draft_type)
        chunks = self._searcher.search(query, doc_ids=[document_id], top_k=top_k_evidence)
        evidence_package = self._packager.package(chunks)
        output.evidence_count = evidence_package.chunk_count
        output.citations = evidence_package.citation_map

        # 2. Get relevant exemplars
        exemplars = self._exemplar_retriever.get_relevant_exemplars(
            source_text=full_text[:500] if full_text else "",
            draft_type=draft_type,
            k=3,
        )
        exemplars_block = self._exemplar_retriever.format_exemplars_for_prompt(exemplars)
        output.exemplars_used = [e.id for e in exemplars]

        # 3. Load system prompt + rules
        system_prompt = self._load_system_prompt()
        rules = self._load_learned_rules()
        output.rules_applied = [r["text"] for r in rules if r.get("active", True)]

        # 4. Load draft-type instruction
        task_instruction = self._load_task_prompt(draft_type)

        # 5. Assemble full prompt
        full_prompt = self._assemble_prompt(
            system_prompt=system_prompt,
            rules=output.rules_applied,
            exemplars_block=exemplars_block,
            evidence_block=evidence_package.formatted_text,
            task_instruction=task_instruction,
        )

        # 6. Generate
        raw_output = self._engine.generate_text(
            full_prompt, role="extraction", max_tokens=2048, temperature=0.3
        )
        output.content_markdown = raw_output

        # 7. Verify grounding
        output.verification_status = self._grounding.verify_draft_citations(
            draft_text=raw_output,
            citation_map=evidence_package.citation_map,
        )

        # 8. Calculate overall confidence
        statuses = list(output.verification_status.values())
        if statuses:
            verified_count = sum(1 for s in statuses if s == "verified")
            output.confidence_overall = verified_count / len(statuses)

        return output

    def _build_retrieval_query(self, draft_type: str) -> str:
        """Build a broad retrieval query based on draft type."""
        queries = {
            "case_fact_summary": "parties claims evidence dates procedural history relief",
            "title_review_summary": "property deed title transfer ownership chain lien encumbrance",
            "notice_summary": "notice deadline requirement compliance response demand",
            "document_checklist": "document filing record deed agreement contract certificate",
        }
        return queries.get(draft_type, "legal document content")

    def _load_system_prompt(self) -> str:
        """Load base system prompt."""
        prompt_file = PROMPTS_DIR / "system_base.txt"
        if prompt_file.exists():
            return prompt_file.read_text(encoding="utf-8")
        return ""

    def _load_task_prompt(self, draft_type: str) -> str:
        """Load draft-type-specific prompt template."""
        prompt_file = PROMPTS_DIR / f"{draft_type}.txt"
        if prompt_file.exists():
            return prompt_file.read_text(encoding="utf-8")
        return f"Generate a {draft_type.replace('_', ' ')} from the provided evidence."

    def _load_learned_rules(self) -> list[dict]:
        """Load learned rules from config."""
        if LEARNED_RULES_YAML.exists():
            with open(LEARNED_RULES_YAML, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return data.get("rules", [])
        return []

    def _assemble_prompt(self, system_prompt: str, rules: list[str],
                         exemplars_block: str, evidence_block: str,
                         task_instruction: str) -> str:
        """Assemble the complete generation prompt."""
        parts = [system_prompt]

        if rules:
            rules_text = "\n\nLEARNED RULES (from past operator corrections):\n"
            rules_text += "\n".join(f"- {r}" for r in rules)
            parts.append(rules_text)

        if exemplars_block:
            parts.append(f"\n\n{exemplars_block}")

        parts.append(f"\n\n{evidence_block}")
        parts.append(f"\n\nTASK:\n{task_instruction}")

        return "\n".join(parts)
