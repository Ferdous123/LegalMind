"""Document structurer — uses Qwen3-VL-8B to extract structured fields.

Takes raw extracted text and produces typed, labeled fields appropriate
to the selected draft type. Injects relevant exemplars for improved accuracy.

Supports two modes:
- Legacy single-pass extraction (fast, for tests/simple documents)
- Cascading multi-pass extraction (robust, for scanned/degraded documents)
"""

import json
import logging
from typing import Optional

from code.llm_interface.inference import InferenceEngine

logger = logging.getLogger(__name__)

FIELD_SCHEMAS = {
    "case_fact_summary": {
        "parties": {
            "plaintiffs": [{"name": "", "role": "", "source_span": ""}],
            "defendants": [{"name": "", "role": "", "source_span": ""}],
            "other_parties": [{"name": "", "role": "", "source_span": ""}],
        },
        "key_dates": [{"event": "", "date": "", "source_span": ""}],
        "claims": [{"type": "", "description": "", "legal_basis": "", "source_span": ""}],
        "procedural_history": [{"event": "", "date": "", "source_span": ""}],
        "evidence_items": [{"type": "", "description": "", "source_span": ""}],
        "relief_sought": [{"party": "", "relief": "", "amount": "", "source_span": ""}],
    },
    "title_review_summary": {
        "property_description": {"legal_description": "", "address": "", "parcel_number": "", "jurisdiction": ""},
        "chain_of_title": [{"grantor": "", "grantee": "", "date": "", "instrument_type": "", "recording_info": "", "source_span": ""}],
        "encumbrances": [{"type": "", "holder": "", "amount": "", "date": "", "status": "", "source_span": ""}],
        "defects": [{"description": "", "severity": "", "source_span": ""}],
        "current_owner": {"name": "", "source_span": ""},
    },
    "notice_summary": {
        "notice_type": "",
        "issuing_party": {"name": "", "source_span": ""},
        "receiving_party": {"name": "", "source_span": ""},
        "date_issued": "",
        "deadlines": [{"action": "", "due_date": "", "consequence": "", "source_span": ""}],
        "requirements": [{"description": "", "source_span": ""}],
        "legal_basis": [{"statute_or_provision": "", "source_span": ""}],
        "response_actions": [{"priority": "", "action": "", "deadline": "", "source_span": ""}],
    },
    "document_checklist": {
        "documents_present": [{"type": "", "date": "", "pages": 0, "quality": "", "source_span": ""}],
        "documents_missing": [{"type": "", "reason_expected": ""}],
        "documents_partial": [{"type": "", "issue": "", "source_span": ""}],
        "verification": [{"document": "", "signatures": "", "dates_consistent": "", "notarized": "", "source_span": ""}],
        "cross_references": {"date_consistency": "", "party_consistency": "", "amount_consistency": ""},
    },
}

EXTRACTION_PROMPT = """You are extracting structured fields from a legal document for use in attorney review.

Document text:
---
{document_text}
---

{exemplars_block}

EXTRACTION RULES:
1. Only extract information that is EXPLICITLY stated in the document text above. Do not infer, assume, or supply facts from background knowledge.
2. For every extracted field, set "source_span" to a short verbatim quote (10–40 words) from the document text that directly supports the extracted value. The source_span must be an exact substring of the document text — do not paraphrase.
3. If a field cannot be determined from the document text, set its value to null and set source_span to null. Do NOT fabricate a value.
4. OCR artifact awareness: the document text may contain OCR errors. Common patterns: "l" confused with "1" or "I", "0" confused with "O", garbled proper nouns, broken line endings. If an extracted value contains suspect characters, extract the most plausible reading and reflect that uncertainty in the source_span by quoting the garbled text verbatim.
5. For date fields: extract dates exactly as they appear in the source. Do not reformat unless the schema explicitly requires a specific format.
6. For name fields: if a name appears with variant spellings across the document (OCR-induced), use the most complete and most frequently appearing version.
7. For list fields (e.g., parties, claims, deadlines): include one entry per distinct item. Do not merge distinct items into a single entry.

Output ONLY valid JSON matching this schema, with no markdown fences, no explanatory text before or after:
{schema}"""


class DocumentStructurer:
    """Extracts structured fields from document text using Qwen3-VL-8B.

    Supports two extraction modes:
    - Single-pass (legacy): Fast extraction, suitable for clean documents.
    - Cascade (default): Multi-pass with corpus assistance, suitable for
      scanned pages, low-resolution PDFs, and partially illegible records.
    """

    def __init__(self):
        self._engine = InferenceEngine()
        self._cascade = None

    @property
    def cascade(self):
        """Lazy-load the extraction cascade to keep imports lightweight."""
        if self._cascade is None:
            from code.pipeline.cascade import ExtractionCascade
            self._cascade = ExtractionCascade()
        return self._cascade

    def extract(self, text: str, draft_type: str,
                exemplars_block: str = "",
                use_cascade: bool = True,
                source_text: str = "") -> dict:
        """Extract structured fields from document text.

        Args:
            text: Full extracted text from document.
            draft_type: One of the supported draft types.
            exemplars_block: Formatted exemplar string from learning system.
            use_cascade: If True, use multi-pass cascading extraction for
                robustness. If False, use legacy single-pass mode.
            source_text: Original source text for cascade anchor verification.
                If empty, the text parameter is used as source.

        Returns:
            Dict of structured fields matching the schema for draft_type.
            When use_cascade=True, includes additional metadata under
            '_cascade_meta' key with confidence scores and resolution methods.
        """
        if use_cascade:
            return self._extract_cascade(
                text, draft_type, exemplars_block, source_text
            )
        return self._extract_single_pass(text, draft_type, exemplars_block)

    def _extract_cascade(self, text: str, draft_type: str,
                         exemplars_block: str, source_text: str) -> dict:
        """Multi-pass cascading extraction with corpus assistance.

        Falls back to single-pass if the cascade encounters an unrecoverable
        error (e.g., models unavailable).
        """
        try:
            cascade_result = self.cascade.extract(
                text=text,
                draft_type=draft_type,
                source_text=source_text or text,
                exemplars_block=exemplars_block,
            )

            # Merge cascade results into the schema-shaped output
            # Use pass_results[0] as the base (full schema structure)
            if cascade_result.pass_results:
                base_result = cascade_result.pass_results[0]
            else:
                base_result = FIELD_SCHEMAS.get(
                    draft_type, FIELD_SCHEMAS["case_fact_summary"]
                )

            # Overlay resolved field values onto base
            for dotted_key, resolved_value in cascade_result.fields.items():
                if resolved_value and cascade_result.resolution_method.get(
                    dotted_key
                ) != "manual_review":
                    self._set_nested(base_result, dotted_key, resolved_value)

            # Attach cascade metadata
            base_result["_cascade_meta"] = {
                "confidence_map": cascade_result.confidence_map,
                "resolution_methods": cascade_result.resolution_method,
                "corpus_corrections": cascade_result.corpus_corrections,
                "unresolved_fields": cascade_result.unresolved_fields,
                "passes_executed": len(cascade_result.pass_results),
            }

            return base_result

        except Exception as e:
            logger.warning(
                "Cascade extraction failed, falling back to single-pass: %s", e
            )
            return self._extract_single_pass(text, draft_type, exemplars_block)

    def _extract_single_pass(self, text: str, draft_type: str,
                             exemplars_block: str) -> dict:
        """Legacy single-pass extraction (original behavior)."""
        schema = FIELD_SCHEMAS.get(draft_type, FIELD_SCHEMAS["case_fact_summary"])

        # Truncate text if too long for context window
        max_chars = 8000  # ~2000 tokens, leaving room for prompt + output
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[Document truncated for processing]"

        prompt = EXTRACTION_PROMPT.format(
            document_text=text,
            exemplars_block=exemplars_block if exemplars_block else "",
            schema=json.dumps(schema, indent=2),
        )

        result = self._engine.generate_structured(
            prompt, schema=schema, role="extraction", max_tokens=2048
        )

        if result.get("_parse_error"):
            logger.warning("Structured extraction returned unparseable JSON")
            return schema

        return result

    def get_schema(self, draft_type: str) -> dict:
        """Get the field schema for a given draft type."""
        return FIELD_SCHEMAS.get(draft_type, {})

    @staticmethod
    def supported_draft_types() -> list[str]:
        """Return list of supported draft types."""
        return list(FIELD_SCHEMAS.keys())

    @staticmethod
    def _set_nested(data: dict, dotted_key: str, value) -> None:
        """Set a value in a nested dict using a dot-separated key path.

        Args:
            data: The nested dictionary to modify in place.
            dotted_key: Dot-separated path (e.g., "property_description.address").
            value: The value to set.
        """
        parts = dotted_key.split(".")
        current = data
        for part in parts[:-1]:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return
        if isinstance(current, dict) and parts[-1] in current:
            current[parts[-1]] = value
