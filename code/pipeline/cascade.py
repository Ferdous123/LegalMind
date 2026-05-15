"""Cascading multi-pass extraction with progressive refinement.

Implements a 3-pass strategy for robust legal document extraction:
- Pass 1: Standard deterministic extraction
- Pass 2: Adjusted parameters with corpus vocabulary injection
- Pass 3: Reasoning model for difficult fields

Resolution uses majority vote, corpus anchoring, and source-span verification.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class FieldExtractionResult:
    """Result of extracting a single field across passes."""
    field_name: str
    resolved_value: str
    confidence: float
    resolution_method: str  # "majority_vote"|"corpus_anchor"|"source_anchor"|"manual_review"
    pass_values: list[str]
    corpus_correction: Optional[str] = None


@dataclass
class CascadeResult:
    """Result of cascading extraction."""
    fields: dict  # field_name -> resolved value
    confidence_map: dict  # field_name -> confidence (0.0-1.0)
    resolution_method: dict  # field_name -> resolution method string
    pass_results: list[dict]  # raw results from each pass
    corpus_corrections: list[dict]  # OCR corrections applied via corpus
    unresolved_fields: list[str]  # fields still needing review


class ExtractionCascade:
    """Multi-pass extraction with progressive refinement and legal corpus assistance.

    Implements a 3-pass cascading extraction strategy:

    Pass 1 (Primary): Standard extraction via the configured model.
        - Temperature: 0.1 (deterministic)
        - Full document context
        - Standard prompt

    Pass 2 (Refined): Re-extraction with adjusted parameters.
        - Temperature: 0.4 (more creative)
        - Chunk boundary shifted +/- overlap
        - Legal corpus terms injected into prompt as vocabulary hints

    Pass 3 (Specialist): Reasoning model for difficult fields.
        - Uses the reasoning model (Gemma-4-E4B)
        - Focused prompt on specific unclear fields only
        - Includes legal corpus suggestions as candidate values

    Resolution Strategy:
        1. Majority vote: if 2+ passes produce identical value -> accept (HIGH confidence)
        2. Corpus anchor: if extracted value fuzzy-matches a corpus term -> accept (MEDIUM confidence)
        3. Source anchor: if extracted value appears verbatim (5-gram) in source text -> accept (MEDIUM confidence)
        4. If unresolved after 3 passes: mark as "needs_manual_review" with all candidates listed
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        max_passes: int = 3,
        enable_reasoning_model: bool = True,
    ):
        """Initialize the extraction cascade.

        Args:
            confidence_threshold: Minimum confidence to accept a value.
            max_passes: Maximum passes to run (1-3).
            enable_reasoning_model: Whether Pass 3 (reasoning) is available.
        """
        self._confidence_threshold = confidence_threshold
        self._max_passes = min(3, max(1, max_passes))
        self._enable_reasoning = enable_reasoning_model
        self._engine = None
        self._lexicon = None
        self._anchor = None

    @property
    def _inference_engine(self):
        """Lazy-load the inference engine."""
        if self._engine is None:
            from code.llm_interface.inference import InferenceEngine
            self._engine = InferenceEngine()
        return self._engine

    @property
    def _legal_lexicon(self):
        """Lazy-load the legal lexicon."""
        if self._lexicon is None:
            from code.corpus.legal_lexicon import LegalLexicon
            self._lexicon = LegalLexicon()
        return self._lexicon

    @property
    def _source_anchor(self):
        """Lazy-load the source anchor verifier."""
        if self._anchor is None:
            from code.pipeline.source_anchor import SourceAnchor
            self._anchor = SourceAnchor()
        return self._anchor

    def extract(
        self,
        text: str,
        draft_type: str,
        source_text: str = "",
        exemplars_block: str = "",
    ) -> CascadeResult:
        """Run cascading multi-pass extraction on document text.

        Args:
            text: The extracted document text to process.
            draft_type: The draft type (determines field schema).
            source_text: Original source text for anchor verification.
                         If empty, uses `text` as source.
            exemplars_block: Optional past corrections for prompt injection.

        Returns:
            CascadeResult with resolved fields and confidence scores.
        """
        if not source_text:
            source_text = text

        from code.pipeline.structurer import FIELD_SCHEMAS
        schema = FIELD_SCHEMAS.get(draft_type, FIELD_SCHEMAS.get("case_fact_summary", {}))

        pass_results: list[dict] = []
        corpus_corrections: list[dict] = []

        # --- Pass 1: Standard deterministic extraction ---
        logger.info("Cascade Pass 1: Standard extraction (temp=0.1)")
        pass1_result = self._run_pass(
            text=text,
            schema=schema,
            draft_type=draft_type,
            temperature=0.1,
            role="extraction",
            exemplars_block=exemplars_block,
            vocabulary_hints="",
        )
        pass_results.append(pass1_result)

        # Determine which fields need further passes
        unresolved_fields = self._identify_weak_fields(pass1_result, source_text)

        if not unresolved_fields or self._max_passes < 2:
            # All fields resolved in pass 1
            return self._build_result(
                pass_results=pass_results,
                schema=schema,
                source_text=source_text,
                corpus_corrections=corpus_corrections,
            )

        # --- Pass 2: Refined extraction with corpus hints ---
        logger.info(
            "Cascade Pass 2: Refined extraction (temp=0.4, corpus hints) — "
            "%d unresolved fields",
            len(unresolved_fields),
        )
        vocabulary_hints = self._legal_lexicon.get_vocabulary_hints(max_hints=40)

        # Shift chunk boundaries by adding context padding
        shifted_text = self._shift_chunk_boundary(text)

        pass2_result = self._run_pass(
            text=shifted_text,
            schema=schema,
            draft_type=draft_type,
            temperature=0.4,
            role="extraction",
            exemplars_block=exemplars_block,
            vocabulary_hints=vocabulary_hints,
        )
        pass_results.append(pass2_result)

        # Apply corpus corrections to pass 2 results
        corrections = self._apply_corpus_corrections(pass2_result)
        corpus_corrections.extend(corrections)

        # Check if resolution is now possible
        unresolved_fields = self._identify_unresolved_after_votes(
            pass_results, source_text
        )

        if not unresolved_fields or self._max_passes < 3:
            return self._build_result(
                pass_results=pass_results,
                schema=schema,
                source_text=source_text,
                corpus_corrections=corpus_corrections,
            )

        # --- Pass 3: Reasoning model for remaining difficult fields ---
        if self._enable_reasoning:
            logger.info(
                "Cascade Pass 3: Reasoning model — %d unresolved fields",
                len(unresolved_fields),
            )
            pass3_result = self._run_reasoning_pass(
                text=text,
                schema=schema,
                draft_type=draft_type,
                unresolved_fields=unresolved_fields,
                prior_candidates=self._collect_candidates(pass_results, unresolved_fields),
                exemplars_block=exemplars_block,
            )
            pass_results.append(pass3_result)

            # Apply corpus corrections to pass 3
            corrections = self._apply_corpus_corrections(pass3_result)
            corpus_corrections.extend(corrections)
        else:
            logger.info("Cascade Pass 3 skipped: reasoning model not enabled")

        return self._build_result(
            pass_results=pass_results,
            schema=schema,
            source_text=source_text,
            corpus_corrections=corpus_corrections,
        )

    def extract_field(
        self,
        field_name: str,
        text: str,
        draft_type: str,
        source_text: str = "",
    ) -> FieldExtractionResult:
        """Extract a single specific field using the cascade.

        Useful when only one field needs re-extraction after manual review.

        Args:
            field_name: The specific field to extract.
            text: Document text.
            draft_type: Draft type for schema context.
            source_text: Original source text for verification.

        Returns:
            FieldExtractionResult for the requested field.
        """
        if not source_text:
            source_text = text

        pass_values: list[str] = []

        # Pass 1
        prompt = self._build_field_prompt(field_name, text, draft_type, "")
        value1 = self._inference_engine.generate_text(
            prompt, role="extraction", temperature=0.1, max_tokens=512
        ).strip()
        pass_values.append(value1)

        # Pass 2 with corpus hints
        hints = self._legal_lexicon.get_vocabulary_hints(max_hints=20)
        prompt2 = self._build_field_prompt(field_name, text, draft_type, hints)
        value2 = self._inference_engine.generate_text(
            prompt2, role="extraction", temperature=0.4, max_tokens=512
        ).strip()
        pass_values.append(value2)

        # Pass 3 if needed and enabled
        if value1 != value2 and self._enable_reasoning:
            prompt3 = self._build_arbitration_prompt(
                field_name, text, [value1, value2], hints
            )
            try:
                value3 = self._inference_engine.generate_text(
                    prompt3, role="reasoning", temperature=0.2, max_tokens=512
                ).strip()
                pass_values.append(value3)
            except RuntimeError as e:
                logger.warning("Reasoning model unavailable for field extraction: %s", e)

        # Resolve
        resolved, confidence, method = self._resolve_field_values(
            pass_values, source_text, field_name
        )

        # Check for corpus correction
        corpus_correction = None
        corrected, corr_conf = self._legal_lexicon.correct_ocr(resolved)
        if corr_conf > 0.0 and corrected != resolved:
            corpus_correction = corrected
            resolved = corrected
            confidence = max(confidence, corr_conf)

        return FieldExtractionResult(
            field_name=field_name,
            resolved_value=resolved,
            confidence=confidence,
            resolution_method=method,
            pass_values=pass_values,
            corpus_correction=corpus_correction,
        )

    def _run_pass(
        self,
        text: str,
        schema: dict,
        draft_type: str,
        temperature: float,
        role: str,
        exemplars_block: str,
        vocabulary_hints: str,
    ) -> dict:
        """Run a single extraction pass and return structured results.

        Args:
            text: Document text.
            schema: Expected output schema.
            draft_type: Draft type name.
            temperature: Sampling temperature.
            role: Model role to use.
            exemplars_block: Past corrections block.
            vocabulary_hints: Corpus vocabulary hints for the prompt.

        Returns:
            Dict of extracted fields matching the schema structure.
        """
        prompt = self._build_extraction_prompt(
            text, schema, draft_type, exemplars_block, vocabulary_hints
        )

        try:
            result = self._inference_engine.generate_structured(
                prompt, schema=schema, role=role, max_tokens=2048
            )
            if result.get("_parse_error"):
                logger.warning("Pass returned unparseable JSON, using empty schema")
                return schema
            return result
        except RuntimeError as e:
            logger.error("Model unavailable for pass (role=%s): %s", role, e)
            return schema

    def _run_reasoning_pass(
        self,
        text: str,
        schema: dict,
        draft_type: str,
        unresolved_fields: list[str],
        prior_candidates: dict[str, list[str]],
        exemplars_block: str,
    ) -> dict:
        """Run Pass 3 using the reasoning model on unresolved fields only.

        Args:
            text: Document text.
            schema: Full schema.
            draft_type: Draft type.
            unresolved_fields: Field names that need reasoning.
            prior_candidates: Prior pass values for each unresolved field.
            exemplars_block: Past corrections.

        Returns:
            Dict of extracted fields (may be partial — only unresolved fields).
        """
        # Build focused prompt for the reasoning model
        candidates_block = ""
        for field_name, values in prior_candidates.items():
            unique_values = list(set(v for v in values if v))
            if unique_values:
                candidates_block += (
                    f"\n  {field_name}: candidates = {unique_values}"
                )

        # Get corpus suggestions for unresolved fields
        corpus_suggestions = ""
        for field_name in unresolved_fields:
            categories = self._legal_lexicon.classify_segment(field_name)
            if categories:
                terms = self._legal_lexicon.get_terms(categories[0])[:10]
                corpus_suggestions += f"\n  {field_name}: possible terms = {terms}"

        prompt = (
            f"You are a legal document analysis expert. The following document text "
            f"has been processed, but some fields remain uncertain.\n\n"
            f"Document text:\n---\n{text[:6000]}\n---\n\n"
            f"Unresolved fields requiring your analysis: {unresolved_fields}\n\n"
            f"Prior extraction attempts produced these candidate values:"
            f"{candidates_block}\n\n"
            f"Legal vocabulary suggestions:{corpus_suggestions}\n\n"
            f"{exemplars_block}\n\n"
            f"For each unresolved field, determine the correct value based on:\n"
            f"1. What is explicitly stated in the document\n"
            f"2. Which candidate value best matches the source text\n"
            f"3. Legal domain knowledge about what values are plausible\n\n"
            f"Output valid JSON matching the full schema:\n"
            f"```json\n{json.dumps(schema, indent=2)}\n```"
        )

        try:
            result = self._inference_engine.generate_structured(
                prompt, schema=schema, role="reasoning", max_tokens=2048
            )
            if result.get("_parse_error"):
                logger.warning("Reasoning pass returned unparseable JSON")
                return schema
            return result
        except RuntimeError as e:
            logger.error("Reasoning model unavailable: %s", e)
            return schema

    def _build_result(
        self,
        pass_results: list[dict],
        schema: dict,
        source_text: str,
        corpus_corrections: list[dict],
    ) -> CascadeResult:
        """Build the final CascadeResult by resolving all fields across passes.

        Args:
            pass_results: Results from each pass.
            schema: The field schema.
            source_text: Source document text.
            corpus_corrections: Applied corpus corrections.

        Returns:
            CascadeResult with resolved fields.
        """
        fields: dict = {}
        confidence_map: dict = {}
        resolution_method: dict = {}
        unresolved_fields: list[str] = []

        # Flatten schema to get field names
        flat_fields = self._flatten_schema_keys(schema)

        for field_name in flat_fields:
            # Collect values from each pass for this field
            pass_values = []
            for result in pass_results:
                value = self._extract_nested_value(result, field_name)
                pass_values.append(value)

            resolved, confidence, method = self._resolve_field_values(
                pass_values, source_text, field_name
            )

            fields[field_name] = resolved
            confidence_map[field_name] = confidence
            resolution_method[field_name] = method

            if method == "manual_review":
                unresolved_fields.append(field_name)

        return CascadeResult(
            fields=fields,
            confidence_map=confidence_map,
            resolution_method=resolution_method,
            pass_results=pass_results,
            corpus_corrections=corpus_corrections,
            unresolved_fields=unresolved_fields,
        )

    def _resolve_field_values(
        self,
        pass_values: list[str],
        source_text: str,
        field_name: str,
    ) -> tuple[str, float, str]:
        """Resolve a field value from multiple pass outputs.

        Applies the resolution hierarchy:
        1. Majority vote
        2. Corpus anchor
        3. Source anchor
        4. Manual review

        Args:
            pass_values: Values from each pass.
            source_text: Source document text.
            field_name: Name of the field.

        Returns:
            Tuple of (resolved_value, confidence, resolution_method).
        """
        # Filter out empty/null values
        valid_values = [
            v for v in pass_values
            if v and str(v).strip() and str(v).lower() not in ("null", "none", "")
        ]

        if not valid_values:
            return ("", 0.0, "manual_review")

        # Strategy 1: Majority vote
        value_counts: dict[str, int] = {}
        for v in valid_values:
            normalized = str(v).strip().lower()
            value_counts[normalized] = value_counts.get(normalized, 0) + 1

        # Find majority (2+ agreeing)
        for value_norm, count in sorted(
            value_counts.items(), key=lambda x: x[1], reverse=True
        ):
            if count >= 2:
                # Find original-case version
                original = next(
                    str(v) for v in valid_values if str(v).strip().lower() == value_norm
                )
                return (original, 0.9, "majority_vote")

        # Strategy 2: Corpus anchor
        for v in valid_values:
            v_str = str(v).strip()
            matches = self._legal_lexicon.match_term(v_str, threshold=0.8)
            if matches:
                best_match = matches[0]
                confidence = min(0.85, 0.6 + best_match.similarity * 0.3)
                return (best_match.term, confidence, "corpus_anchor")

        # Strategy 3: Source anchor (5-gram)
        best_anchored = ""
        best_anchor_conf = 0.0
        for v in valid_values:
            v_str = str(v).strip()
            anchor_conf = self._source_anchor.anchor_confidence(v_str, source_text)
            if anchor_conf > best_anchor_conf:
                best_anchor_conf = anchor_conf
                best_anchored = v_str

        if best_anchor_conf >= 0.5:
            return (best_anchored, best_anchor_conf, "source_anchor")

        # Strategy 4: No strong resolution — return best candidate with low confidence
        # Prefer the first valid value (from the most deterministic pass)
        if valid_values:
            return (str(valid_values[0]).strip(), 0.3, "manual_review")

        return ("", 0.0, "manual_review")

    def _identify_weak_fields(self, result: dict, source_text: str) -> list[str]:
        """Identify fields from pass 1 that may need re-extraction.

        Fields are considered weak if:
        - They are empty/null
        - They fail source anchoring
        - They contain OCR-garbled text

        Args:
            result: Pass 1 extraction result.
            source_text: Original source text.

        Returns:
            List of field names that are weak.
        """
        weak: list[str] = []
        flat_fields = self._flatten_schema_keys(result)

        for field_name in flat_fields:
            value = self._extract_nested_value(result, field_name)
            if not value or not str(value).strip():
                weak.append(field_name)
                continue

            # Check source anchoring
            v_str = str(value).strip()
            anchor_conf = self._source_anchor.anchor_confidence(v_str, source_text)
            if anchor_conf < 0.4:
                weak.append(field_name)
                continue

            # Check if value looks OCR-garbled
            corrected, corr_conf = self._legal_lexicon.correct_ocr(v_str)
            if corr_conf > 0.0 and corrected != v_str:
                weak.append(field_name)

        return weak

    def _identify_unresolved_after_votes(
        self, pass_results: list[dict], source_text: str
    ) -> list[str]:
        """Identify fields still unresolved after majority voting across passes.

        Args:
            pass_results: Results from all completed passes.
            source_text: Source document text.

        Returns:
            List of field names that cannot be resolved by vote or anchoring.
        """
        if not pass_results:
            return []

        schema_keys = self._flatten_schema_keys(pass_results[0])
        unresolved: list[str] = []

        for field_name in schema_keys:
            pass_values = []
            for result in pass_results:
                value = self._extract_nested_value(result, field_name)
                pass_values.append(value)

            _, confidence, method = self._resolve_field_values(
                pass_values, source_text, field_name
            )

            if method == "manual_review" or confidence < self._confidence_threshold:
                unresolved.append(field_name)

        return unresolved

    def _collect_candidates(
        self, pass_results: list[dict], field_names: list[str]
    ) -> dict[str, list[str]]:
        """Collect candidate values for unresolved fields from all passes.

        Args:
            pass_results: Results from all passes.
            field_names: Fields to collect candidates for.

        Returns:
            Dict mapping field names to lists of candidate values.
        """
        candidates: dict[str, list[str]] = {}
        for field_name in field_names:
            values: list[str] = []
            for result in pass_results:
                value = self._extract_nested_value(result, field_name)
                if value and str(value).strip():
                    values.append(str(value).strip())
            candidates[field_name] = values
        return candidates

    def _apply_corpus_corrections(self, result: dict) -> list[dict]:
        """Apply corpus-based OCR corrections to extracted values.

        Args:
            result: Extraction result dict.

        Returns:
            List of corrections applied (for audit trail).
        """
        corrections: list[dict] = []
        flat_fields = self._flatten_schema_keys(result)

        for field_name in flat_fields:
            value = self._extract_nested_value(result, field_name)
            if not value or not str(value).strip():
                continue

            v_str = str(value).strip()
            # Check individual words for OCR corrections
            words = v_str.split()
            corrected_words: list[str] = []
            field_corrected = False

            for word in words:
                corrected, conf = self._legal_lexicon.correct_ocr(word)
                if conf > 0.0 and corrected != word:
                    corrected_words.append(corrected)
                    field_corrected = True
                    corrections.append({
                        "field": field_name,
                        "original": word,
                        "corrected": corrected,
                        "confidence": conf,
                    })
                else:
                    corrected_words.append(word)

            if field_corrected:
                new_value = " ".join(corrected_words)
                self._set_nested_value(result, field_name, new_value)

        return corrections

    def _build_extraction_prompt(
        self,
        text: str,
        schema: dict,
        draft_type: str,
        exemplars_block: str,
        vocabulary_hints: str,
    ) -> str:
        """Build the extraction prompt for a pass.

        Args:
            text: Document text.
            schema: Output schema.
            draft_type: Draft type.
            exemplars_block: Past correction exemplars.
            vocabulary_hints: Corpus vocabulary hints.

        Returns:
            Formatted prompt string.
        """
        # Truncate if needed
        max_chars = 8000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[Document truncated for processing]"

        prompt_parts = [
            "You are extracting structured information from a legal document.",
            "",
            f"Document text:\n---\n{text}\n---",
            "",
        ]

        if vocabulary_hints:
            prompt_parts.append(f"{vocabulary_hints}\n")

        if exemplars_block:
            prompt_parts.append(f"{exemplars_block}\n")

        prompt_parts.extend([
            "Extract the following structured fields from the document above.",
            "For each field, include the 'source_span' — a short verbatim quote from the document that supports the extracted value.",
            "If a field cannot be determined from the document, set it to null or empty string.",
            "",
            f"Output as valid JSON matching this schema:\n```json\n{json.dumps(schema, indent=2)}\n```",
            "",
            "IMPORTANT: Only extract information that is explicitly stated in the document text above.",
            "Do not infer or assume any facts not directly supported by the text.",
        ])

        return "\n".join(prompt_parts)

    def _build_field_prompt(
        self,
        field_name: str,
        text: str,
        draft_type: str,
        vocabulary_hints: str,
    ) -> str:
        """Build a prompt for single-field extraction.

        Args:
            field_name: The field to extract.
            text: Document text.
            draft_type: Draft type context.
            vocabulary_hints: Vocabulary hints.

        Returns:
            Formatted prompt string.
        """
        max_chars = 6000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[Document truncated]"

        prompt = (
            f"Extract the '{field_name}' field from this legal document.\n\n"
            f"Document text:\n---\n{text}\n---\n\n"
        )
        if vocabulary_hints:
            prompt += f"{vocabulary_hints}\n\n"
        prompt += (
            f"Return ONLY the extracted value for '{field_name}'. "
            f"Use only information explicitly present in the text. "
            f"If the field cannot be determined, respond with 'UNDETERMINED'."
        )
        return prompt

    def _build_arbitration_prompt(
        self,
        field_name: str,
        text: str,
        candidates: list[str],
        vocabulary_hints: str,
    ) -> str:
        """Build an arbitration prompt for the reasoning model.

        Args:
            field_name: The field being arbitrated.
            text: Document text.
            candidates: Candidate values from prior passes.
            vocabulary_hints: Corpus vocabulary hints.

        Returns:
            Formatted arbitration prompt.
        """
        max_chars = 5000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[Document truncated]"

        prompt = (
            f"You are arbitrating between conflicting extraction results for a legal document.\n\n"
            f"Field: '{field_name}'\n"
            f"Candidate values from prior extraction passes: {candidates}\n\n"
            f"Document text:\n---\n{text}\n---\n\n"
        )
        if vocabulary_hints:
            prompt += f"{vocabulary_hints}\n\n"
        prompt += (
            f"Determine which candidate value (or a corrected version) is the "
            f"correct extraction for '{field_name}' based on what the document "
            f"explicitly states. If none of the candidates are correct, provide "
            f"the correct value. If the field truly cannot be determined, respond "
            f"with 'UNDETERMINED'.\n\n"
            f"Return ONLY the correct value, nothing else."
        )
        return prompt

    @staticmethod
    def _shift_chunk_boundary(text: str) -> str:
        """Shift chunk boundaries by re-centering the text with overlap padding.

        Adds paragraph-boundary awareness: finds paragraph breaks and shifts
        the starting point to align with paragraph boundaries rather than
        arbitrary character positions.

        Args:
            text: Original document text.

        Returns:
            Text with shifted boundaries (may reorder paragraphs slightly).
        """
        paragraphs = text.split("\n\n")
        if len(paragraphs) <= 2:
            return text

        # Shift by rotating paragraphs: move first paragraph to the end
        # This changes which paragraph falls at chunk boundaries
        shifted = paragraphs[1:] + paragraphs[:1]
        return "\n\n".join(shifted)

    @staticmethod
    def _flatten_schema_keys(schema: dict, prefix: str = "") -> list[str]:
        """Flatten a nested schema dict into dot-separated field names.

        Only flattens to the first level of dict keys (not into lists).

        Args:
            schema: The schema dictionary.
            prefix: Key prefix for recursion.

        Returns:
            List of dot-separated field name strings.
        """
        keys: list[str] = []
        for key, value in schema.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if key.startswith("_"):
                continue
            if isinstance(value, dict):
                keys.extend(
                    ExtractionCascade._flatten_schema_keys(value, full_key)
                )
            else:
                keys.append(full_key)
        return keys

    @staticmethod
    def _extract_nested_value(data: dict, dotted_key: str):
        """Extract a value from a nested dict using a dot-separated key.

        Args:
            data: The nested dictionary.
            dotted_key: Dot-separated path (e.g., "property.address").

        Returns:
            The value at that path, or "" if not found.
        """
        parts = dotted_key.split(".")
        current = data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return ""
        if isinstance(current, (list, dict)):
            return json.dumps(current) if current else ""
        return str(current) if current is not None else ""

    @staticmethod
    def _set_nested_value(data: dict, dotted_key: str, value) -> None:
        """Set a value in a nested dict using a dot-separated key.

        Args:
            data: The nested dictionary to modify.
            dotted_key: Dot-separated path.
            value: Value to set.
        """
        parts = dotted_key.split(".")
        current = data
        for part in parts[:-1]:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return
        if isinstance(current, dict):
            current[parts[-1]] = value
