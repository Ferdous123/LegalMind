"""Legal terminology corpus for vocabulary-assisted extraction.

Provides fuzzy matching against a curated legal vocabulary to correct OCR
misreadings, identify legal entities, and boost extraction confidence.
"""

import logging
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional

from code.corpus.term_data import LEGAL_TERMS, OCR_CONFUSION_MAP

logger = logging.getLogger(__name__)


@dataclass
class TermMatch:
    """A match result from the legal lexicon."""
    term: str
    category: str
    similarity: float
    original_input: str


class LegalLexicon:
    """Legal terminology corpus for vocabulary-assisted extraction.

    Provides fuzzy matching against a curated legal vocabulary to:
    1. Correct common OCR misreadings (e.g., "plointiff" -> "plaintiff")
    2. Identify legal entities, terms, and phrases
    3. Boost extraction confidence when matches are found
    4. Disambiguate partial or unclear text against legal terminology

    The corpus is organized by category:
    - party_roles: plaintiff, defendant, respondent, petitioner, etc.
    - document_types: deed, motion, complaint, summons, affidavit, etc.
    - legal_actions: filed, granted, denied, sustained, overruled, etc.
    - property_terms: easement, encumbrance, lien, covenant, fee simple, etc.
    - temporal_markers: hereinafter, whereas, forthwith, notwithstanding, etc.
    - monetary_terms: damages, consideration, escrow, deposit, etc.
    - court_hierarchy: superior court, district court, circuit court, etc.
    - compliance_terms: violation, penalty, remediation, abatement, etc.
    """

    def __init__(self):
        self._terms = LEGAL_TERMS
        self._ocr_map = OCR_CONFUSION_MAP
        # Build a flat lookup for fast exact matching
        self._flat_terms: dict[str, str] = {}
        for category, terms in self._terms.items():
            for term in terms:
                self._flat_terms[term.lower()] = category

        # Augment with learned vocabulary from operator corrections
        self._learned_terms: list = []
        try:
            from code.corpus.vocabulary_learner import VocabularyLearner
            learner = VocabularyLearner()
            self._learned_terms = learner.get_learned_terms()
            count = learner.merge_into_lexicon(self)
            if count > 0:
                logger.info("Merged %d learned terms into lexicon", count)
        except Exception as e:
            logger.debug("Could not load learned vocabulary: %s", e)

    def match_term(
        self,
        text: str,
        category: Optional[str] = None,
        threshold: float = 0.7,
    ) -> list[TermMatch]:
        """Find legal terms that fuzzy-match the input text.

        Args:
            text: The text to match against the corpus.
            category: Optional category to restrict the search.
            threshold: Minimum similarity ratio (0.0-1.0) for a match.

        Returns:
            List of TermMatch objects sorted by similarity (descending).
        """
        text_lower = text.lower().strip()
        matches: list[TermMatch] = []

        # Check exact match first
        if text_lower in self._flat_terms:
            cat = self._flat_terms[text_lower]
            if category is None or cat == category:
                matches.append(TermMatch(
                    term=text_lower,
                    category=cat,
                    similarity=1.0,
                    original_input=text,
                ))
                return matches

        # Check OCR confusion map
        if text_lower in self._ocr_map:
            corrected = self._ocr_map[text_lower]
            if corrected.lower() in self._flat_terms:
                cat = self._flat_terms[corrected.lower()]
                if category is None or cat == category:
                    matches.append(TermMatch(
                        term=corrected,
                        category=cat,
                        similarity=0.95,
                        original_input=text,
                    ))

        # Fuzzy search across terms
        search_categories = (
            {category: self._terms[category]}
            if category and category in self._terms
            else self._terms
        )

        for cat, terms in search_categories.items():
            for term in terms:
                sim = self._similarity(text_lower, term.lower())
                if sim >= threshold:
                    matches.append(TermMatch(
                        term=term,
                        category=cat,
                        similarity=sim,
                        original_input=text,
                    ))

        # Sort by similarity descending, deduplicate
        matches.sort(key=lambda m: m.similarity, reverse=True)
        seen: set[str] = set()
        deduped: list[TermMatch] = []
        for m in matches:
            if m.term not in seen:
                seen.add(m.term)
                deduped.append(m)
        return deduped

    def correct_ocr(self, word: str) -> tuple[str, float]:
        """Attempt to correct an OCR-garbled word using the corpus.

        Args:
            word: The potentially garbled word from OCR output.

        Returns:
            Tuple of (corrected_word, confidence). If no correction found,
            returns (original_word, 0.0).
        """
        word_lower = word.lower().strip()

        # Direct lookup in OCR confusion map
        if word_lower in self._ocr_map:
            corrected = self._ocr_map[word_lower]
            return (corrected, 0.95)

        # Try fuzzy match against all known terms
        best_term = ""
        best_sim = 0.0

        for term in self._flat_terms:
            sim = self._similarity(word_lower, term)
            if sim > best_sim:
                best_sim = sim
                best_term = term

        # Only correct if similarity is high enough to be confident
        if best_sim >= 0.80:
            return (best_term, best_sim)

        return (word, 0.0)

    def classify_segment(self, text: str) -> list[str]:
        """Classify a text segment by relevant legal categories.

        Analyzes the text for presence of legal terms from each category
        and returns categories that are represented.

        Args:
            text: A text segment (sentence or paragraph) to classify.

        Returns:
            List of category names that have matching terms in the text,
            sorted by number of matches (most matches first).
        """
        text_lower = text.lower()
        category_scores: dict[str, int] = {}

        for category, terms in self._terms.items():
            count = 0
            for term in terms:
                # For multi-word terms, check if they appear as substring
                if len(term.split()) > 1:
                    if term.lower() in text_lower:
                        count += 2  # Multi-word matches are stronger signals
                else:
                    # Single-word terms: check word boundaries
                    words = text_lower.split()
                    for w in words:
                        # Strip punctuation for comparison
                        cleaned = w.strip(".,;:!?()[]{}\"'")
                        if cleaned == term.lower():
                            count += 1
                        elif self._similarity(cleaned, term.lower()) >= 0.85:
                            count += 1
            if count > 0:
                category_scores[category] = count

        # Sort by score descending
        sorted_categories = sorted(
            category_scores.keys(),
            key=lambda c: category_scores[c],
            reverse=True,
        )
        return sorted_categories

    def get_terms(self, category: str) -> list[str]:
        """Get all terms in a given category.

        Args:
            category: The category name to look up.

        Returns:
            List of terms in that category, or empty list if category unknown.
        """
        return list(self._terms.get(category, []))

    def boost_confidence(self, extracted_text: str, field_type: str) -> float:
        """Calculate a confidence boost based on corpus matches.

        Checks how well the extracted text aligns with known legal terminology
        for the given field type.

        Args:
            extracted_text: The text that was extracted.
            field_type: The type of field being extracted (maps to categories).

        Returns:
            Confidence boost value between 0.0 and 1.0.
            0.0 = no corpus support
            1.0 = strong corpus match
        """
        if not extracted_text or not extracted_text.strip():
            return 0.0

        # Map field types to relevant categories
        field_category_map: dict[str, list[str]] = {
            "party": ["party_roles"],
            "parties": ["party_roles"],
            "plaintiff": ["party_roles"],
            "defendant": ["party_roles"],
            "document_type": ["document_types"],
            "action": ["legal_actions"],
            "property": ["property_terms"],
            "temporal": ["temporal_markers"],
            "monetary": ["monetary_terms"],
            "amount": ["monetary_terms"],
            "court": ["court_hierarchy"],
            "compliance": ["compliance_terms"],
            "date": ["temporal_markers"],
            "encumbrance": ["property_terms"],
            "lien": ["property_terms"],
            "damages": ["monetary_terms"],
        }

        relevant_categories = field_category_map.get(
            field_type.lower(), list(self._terms.keys())
        )

        # Check each word in extracted text against relevant categories
        words = extracted_text.lower().split()
        total_words = len(words)
        if total_words == 0:
            return 0.0

        matched_words = 0
        for word in words:
            cleaned = word.strip(".,;:!?()[]{}\"'")
            if not cleaned:
                continue
            for cat in relevant_categories:
                terms = self._terms.get(cat, [])
                for term in terms:
                    if cleaned == term.lower():
                        matched_words += 1
                        break
                    elif self._similarity(cleaned, term.lower()) >= 0.85:
                        matched_words += 0.7
                        break
                else:
                    continue
                break

        # Also check multi-word terms
        text_lower = extracted_text.lower()
        multi_word_bonus = 0.0
        for cat in relevant_categories:
            for term in self._terms.get(cat, []):
                if len(term.split()) > 1 and term.lower() in text_lower:
                    multi_word_bonus += 0.2

        # Calculate boost: ratio of matched words + multi-word bonus
        ratio = matched_words / max(total_words, 1)
        boost = min(1.0, ratio + multi_word_bonus)
        return round(boost, 3)

    def get_vocabulary_hints(self, category: Optional[str] = None, max_hints: int = 30) -> str:
        """Generate a vocabulary hint block for injection into extraction prompts.

        Args:
            category: Optional category to focus hints on.
            max_hints: Maximum number of terms to include.

        Returns:
            A formatted string of vocabulary hints for prompt injection.
        """
        hints: list[str] = []

        if category and category in self._terms:
            terms = self._terms[category][:max_hints]
            hints.extend(terms)
        else:
            # Sample from each category
            per_category = max(3, max_hints // len(self._terms))
            for cat, terms in self._terms.items():
                hints.extend(terms[:per_category])
                if len(hints) >= max_hints:
                    break

        hints = hints[:max_hints]
        return "Legal vocabulary reference: " + ", ".join(hints)

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """Compute similarity ratio between two strings using SequenceMatcher.

        This provides a Levenshtein-like similarity measure without external
        dependencies.

        Args:
            a: First string.
            b: Second string.

        Returns:
            Similarity ratio between 0.0 and 1.0.
        """
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0
        return SequenceMatcher(None, a, b).ratio()

    @property
    def categories(self) -> list[str]:
        """Return all available category names."""
        return list(self._terms.keys())

    @property
    def total_terms(self) -> int:
        """Return total number of terms across all categories."""
        return sum(len(terms) for terms in self._terms.values())
