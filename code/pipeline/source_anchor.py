"""Source-span verification via n-gram matching.

Verifies that extracted values actually appear in (or closely match) the
original source text. Used by the cascade to confirm extractions are
grounded in the document rather than hallucinated.
"""

import logging
import re
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class SourceAnchor:
    """Verifies extracted values against source text using n-gram matching.

    Provides grounding verification: ensures that extracted fields trace back
    to specific spans in the original document text. This prevents model
    hallucinations from being accepted as valid extractions.
    """

    def __init__(self, default_n: int = 5):
        """Initialize with default n-gram size.

        Args:
            default_n: Default number of words for n-gram matching.
        """
        self._default_n = default_n

    def has_anchor(self, value: str, source_text: str, n: int = 5) -> bool:
        """Check if the extracted value has an n-gram anchor in the source text.

        Looks for any contiguous n-gram from the value that appears verbatim
        in the source text.

        Args:
            value: The extracted value to verify.
            source_text: The original document text.
            n: Number of words for n-gram matching.

        Returns:
            True if at least one n-gram from value appears in source_text.
        """
        if not value or not source_text:
            return False

        value_clean = self._normalize(value)
        source_clean = self._normalize(source_text)

        # If value is shorter than n words, check if entire value is in source
        value_words = value_clean.split()
        if len(value_words) < n:
            return value_clean in source_clean

        # Generate n-grams from value and check presence in source
        ngrams = self._extract_ngrams(value_words, n)
        for ngram in ngrams:
            if ngram in source_clean:
                return True

        return False

    def find_best_span(
        self, value: str, source_text: str, window: int = 200
    ) -> tuple[str, float]:
        """Find the best matching span in source text for a given value.

        Slides a window over the source text and finds the region with
        highest similarity to the extracted value.

        Args:
            value: The extracted value to locate in source.
            source_text: The full source document text.
            window: Character window size for span search.

        Returns:
            Tuple of (best_matching_span, similarity_score).
            Returns ("", 0.0) if no reasonable match found.
        """
        if not value or not source_text:
            return ("", 0.0)

        value_norm = self._normalize(value)
        source_norm = self._normalize(source_text)

        if not value_norm or not source_norm:
            return ("", 0.0)

        # If value is short, try direct substring match first
        if len(value_norm) <= window:
            if value_norm in source_norm:
                # Find the position in original text
                idx = source_norm.find(value_norm)
                start = max(0, idx - 20)
                end = min(len(source_norm), idx + len(value_norm) + 20)
                return (source_norm[start:end].strip(), 1.0)

        # Sliding window search
        best_span = ""
        best_sim = 0.0
        step = max(1, window // 4)

        for i in range(0, len(source_norm) - min(window, len(value_norm)) + 1, step):
            span = source_norm[i : i + window]
            sim = SequenceMatcher(None, value_norm, span).ratio()
            if sim > best_sim:
                best_sim = sim
                best_span = span

        # Refine around best position with smaller steps
        if best_sim > 0.3:
            best_idx = source_norm.find(best_span[:50])
            if best_idx >= 0:
                refine_start = max(0, best_idx - step)
                refine_end = min(
                    len(source_norm), best_idx + window + step
                )
                for i in range(refine_start, refine_end - len(value_norm) + 1, 1):
                    span = source_norm[i : i + window]
                    sim = SequenceMatcher(None, value_norm, span).ratio()
                    if sim > best_sim:
                        best_sim = sim
                        best_span = span

        return (best_span.strip(), round(best_sim, 4))

    def anchor_confidence(self, value: str, source_text: str) -> float:
        """Calculate a grounding confidence score for the extracted value.

        Combines multiple signals:
        - Exact n-gram presence (strong signal)
        - Best span similarity (weaker signal)
        - Word overlap ratio

        Args:
            value: The extracted value to score.
            source_text: The original source document text.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        if not value or not source_text:
            return 0.0

        value_norm = self._normalize(value)
        source_norm = self._normalize(source_text)

        if not value_norm:
            return 0.0

        # Signal 1: N-gram anchoring (strongest)
        has_5gram = self.has_anchor(value, source_text, n=5)
        has_3gram = self.has_anchor(value, source_text, n=3)

        # Signal 2: Word-level overlap
        value_words = set(value_norm.split())
        source_words = set(source_norm.split())
        if value_words:
            word_overlap = len(value_words & source_words) / len(value_words)
        else:
            word_overlap = 0.0

        # Signal 3: Best span similarity
        _, span_sim = self.find_best_span(value, source_text, window=len(value_norm) + 50)

        # Combine signals with weights
        if has_5gram:
            # Strong grounding: 5-gram found verbatim
            confidence = 0.85 + (word_overlap * 0.15)
        elif has_3gram:
            # Moderate grounding: 3-gram found
            confidence = 0.60 + (span_sim * 0.2) + (word_overlap * 0.15)
        else:
            # Weak grounding: rely on span similarity and word overlap
            confidence = (span_sim * 0.5) + (word_overlap * 0.4)

        return round(min(1.0, max(0.0, confidence)), 3)

    def verify_field_extraction(
        self,
        field_name: str,
        extracted_value: str,
        source_text: str,
        min_confidence: float = 0.5,
    ) -> dict:
        """Verify a single field extraction against source text.

        Args:
            field_name: Name of the extracted field.
            extracted_value: The value that was extracted.
            source_text: The source document text.
            min_confidence: Minimum confidence to consider anchored.

        Returns:
            Dict with keys: anchored (bool), confidence (float),
            best_span (str), method (str).
        """
        if not extracted_value or not extracted_value.strip():
            return {
                "anchored": False,
                "confidence": 0.0,
                "best_span": "",
                "method": "empty_value",
            }

        confidence = self.anchor_confidence(extracted_value, source_text)
        best_span, span_sim = self.find_best_span(extracted_value, source_text)
        has_exact = self.has_anchor(extracted_value, source_text, n=self._default_n)

        if has_exact:
            method = "exact_ngram"
        elif confidence >= min_confidence:
            method = "fuzzy_span"
        else:
            method = "unanchored"

        return {
            "anchored": confidence >= min_confidence,
            "confidence": confidence,
            "best_span": best_span,
            "method": method,
        }

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for comparison: lowercase, collapse whitespace."""
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _extract_ngrams(words: list[str], n: int) -> list[str]:
        """Extract word-level n-grams as joined strings.

        Args:
            words: List of words.
            n: N-gram size.

        Returns:
            List of n-gram strings (space-joined).
        """
        if len(words) < n:
            return [" ".join(words)]
        return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
