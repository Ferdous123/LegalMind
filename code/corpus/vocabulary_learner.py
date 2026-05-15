"""Vocabulary learner — expands the legal lexicon from operator corrections.

When an operator corrects a field, the system:
1. Extracts any legal-sounding terms from the correction
2. Checks if they're already in the corpus
3. If novel, appends them to a persistent learned vocabulary file
4. The learned vocabulary is loaded alongside the static corpus

This creates a continuously improving extraction vocabulary that adapts
to the specific legal domain the operators work in.
"""

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from config.paths import DATA_DIR
from code.corpus.term_data import LEGAL_TERMS

logger = logging.getLogger(__name__)

LEARNED_VOCAB_PATH = DATA_DIR / "learned_vocabulary.jsonl"

# Suffixes commonly found in legal terminology
_LEGAL_SUFFIXES = (
    "tion", "ment", "ance", "ence", "ity", "ure", "ory", "ive",
    "ible", "able", "ness", "ship",
)

# Known legal bigrams that strongly suggest a legal term
_LEGAL_BIGRAM_PREFIXES = (
    "court", "deed", "title", "lien", "trust", "estate", "filing",
    "motion", "order", "judgment", "decree", "injunction", "writ",
    "statute", "clause", "provision", "covenant", "easement",
    "encumbrance", "affidavit", "deposition", "subpoena",
)

# Category inference keyword map
_CATEGORY_HINTS: dict[str, list[str]] = {
    "court_hierarchy": ["court", "judge", "justice", "tribunal", "bench", "chamber"],
    "property_terms": ["deed", "title", "easement", "encumbrance", "lien", "parcel", "lot", "tract"],
    "party_roles": ["plaintiff", "defendant", "petitioner", "respondent", "appellant", "appellee"],
    "document_types": ["motion", "complaint", "affidavit", "summons", "brief", "petition", "writ"],
    "legal_actions": ["filed", "granted", "denied", "sustained", "overruled", "dismissed"],
    "monetary_terms": ["damages", "consideration", "escrow", "deposit", "restitution", "award"],
    "temporal_markers": ["hereinafter", "whereas", "forthwith", "notwithstanding", "hereunder"],
    "compliance_terms": ["violation", "penalty", "remediation", "abatement", "compliance"],
}


@dataclass
class LearnedTerm:
    """A term learned from operator corrections."""
    term: str
    category: str
    source_correction_id: str
    confidence: float
    frequency: int = 1


class VocabularyLearner:
    """Expands the legal lexicon from operator corrections.

    Methods:
        learn_from_correction(correction: dict) -> list[str]
        get_learned_terms() -> list[LearnedTerm]
        get_learned_terms_by_category(category: str) -> list[str]
        merge_into_lexicon(lexicon: LegalLexicon) -> int
    """

    def __init__(self):
        self._static_terms: set[str] = set()
        for category, terms in LEGAL_TERMS.items():
            for term in terms:
                self._static_terms.add(term.lower())

        # Ensure the data directory exists
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    def learn_from_correction(self, correction: dict) -> list[str]:
        """Analyze a correction and extract novel legal terms.

        Steps:
        1. Tokenize the edited_text and generated_text
        2. Find words/phrases in edited_text NOT in generated_text (the delta)
        3. Filter: keep only terms that look legal (capitalized entities,
           multi-word legal phrases, known suffixes like -tion, -ment, -ance)
        4. Check against existing corpus -- skip if already known
        5. If the term appears in 2+ corrections, boost confidence
        6. Save to LEARNED_VOCAB_PATH

        Returns list of newly learned term strings.
        """
        edited_text = correction.get("edited_text", "")
        generated_text = correction.get("generated_text", "")
        correction_id = correction.get("id", "unknown")

        if not edited_text:
            return []

        # Extract candidate terms from the delta between edited and generated
        delta_terms = self._extract_delta_terms(edited_text, generated_text)

        # Filter to legal-sounding candidates
        candidates = self._filter_legal_candidates(delta_terms)

        # Check against existing corpus and already-learned terms
        existing_learned = self._load_learned_map()
        newly_learned: list[str] = []

        for term in candidates:
            term_lower = term.lower()

            # Skip if already in static corpus
            if term_lower in self._static_terms:
                continue

            # If already learned, bump frequency and confidence
            if term_lower in existing_learned:
                existing_learned[term_lower].frequency += 1
                existing_learned[term_lower].confidence = min(
                    1.0, existing_learned[term_lower].confidence + 0.15
                )
                continue

            # Infer category
            category = self._infer_category(term)

            # Compute initial confidence
            confidence = self._compute_confidence(term)

            learned = LearnedTerm(
                term=term,
                category=category,
                source_correction_id=correction_id,
                confidence=confidence,
                frequency=1,
            )
            existing_learned[term_lower] = learned
            newly_learned.append(term)

        # Persist all learned terms
        self._save_learned_map(existing_learned)

        if newly_learned:
            logger.info(
                "Learned %d new terms from correction %s: %s",
                len(newly_learned), correction_id, newly_learned,
            )

        return newly_learned

    def get_learned_terms(self) -> list[LearnedTerm]:
        """Load all learned terms from the JSONL file."""
        if not LEARNED_VOCAB_PATH.exists():
            return []

        terms: list[LearnedTerm] = []
        with open(LEARNED_VOCAB_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    terms.append(LearnedTerm(**data))
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning("Skipping malformed learned term line: %s", e)
                    continue
        return terms

    def get_learned_terms_by_category(self, category: str) -> list[str]:
        """Get all learned terms belonging to a specific category.

        Args:
            category: The category to filter by.

        Returns:
            List of term strings in the given category.
        """
        terms = self.get_learned_terms()
        return [t.term for t in terms if t.category == category]

    def merge_into_lexicon(self, lexicon) -> int:
        """Inject learned terms into a LegalLexicon instance at runtime.

        Called during lexicon initialization to augment the static corpus.
        Only merges terms with confidence >= 0.6 OR frequency >= 2.

        Returns count of new terms added.
        """
        terms = self.get_learned_terms()
        count = 0

        for learned in terms:
            # Threshold: confidence >= 0.6 OR frequency >= 2
            if learned.confidence < 0.6 and learned.frequency < 2:
                continue

            term_lower = learned.term.lower()

            # Skip if already in the flat lookup
            if term_lower in lexicon._flat_terms:
                continue

            # Add to the lexicon's internal structures
            category = learned.category
            if category not in lexicon._terms:
                lexicon._terms[category] = []

            lexicon._terms[category].append(learned.term)
            lexicon._flat_terms[term_lower] = category
            count += 1

        return count

    def _extract_delta_terms(self, edited: str, generated: str) -> list[str]:
        """Extract terms present in edited text but not in generated text.

        Returns individual words and detected multi-word phrases from the delta.
        """
        edited_words = set(self._tokenize(edited))
        generated_words = set(self._tokenize(generated))

        # Single-word delta
        delta_words = edited_words - generated_words

        # Also extract multi-word phrases (bigrams and trigrams) from edited
        edited_phrases = self._extract_phrases(edited)
        generated_phrases = set(self._extract_phrases(generated))
        delta_phrases = [p for p in edited_phrases if p not in generated_phrases]

        # Combine single words and phrases
        results: list[str] = list(delta_words) + delta_phrases
        return results

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into cleaned words."""
        words = re.findall(r"[A-Za-z][\w'-]*[A-Za-z]|[A-Za-z]", text)
        return [w for w in words if len(w) >= 3]

    def _extract_phrases(self, text: str) -> list[str]:
        """Extract multi-word phrases (bigrams and trigrams) from text."""
        words = self._tokenize(text)
        phrases: list[str] = []

        # Bigrams
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i + 1]}"
            phrases.append(phrase)

        # Trigrams
        for i in range(len(words) - 2):
            phrase = f"{words[i]} {words[i + 1]} {words[i + 2]}"
            phrases.append(phrase)

        return phrases

    def _filter_legal_candidates(self, terms: list[str]) -> list[str]:
        """Filter terms to keep only those that appear to be legal terminology.

        A term qualifies if any of:
        - It is a multi-word phrase (2+ words, both capitalized or contains a legal keyword)
        - It ends with a known legal suffix
        - It contains a known legal bigram prefix
        - It is capitalized (proper noun, likely entity)
        """
        candidates: list[str] = []

        for term in terms:
            if not term or len(term) < 3:
                continue

            term_lower = term.lower()
            words = term.split()

            # Multi-word phrases with legal relevance
            if len(words) >= 2:
                # Check if any word is a legal keyword
                has_legal_word = any(
                    w.lower() in self._static_terms or
                    any(w.lower().startswith(prefix) for prefix in _LEGAL_BIGRAM_PREFIXES)
                    for w in words
                )
                # Or both words are capitalized (proper entity)
                all_capitalized = all(w[0].isupper() for w in words if w)

                if has_legal_word or all_capitalized:
                    candidates.append(term)
                    continue

            # Single words: check suffix patterns
            if any(term_lower.endswith(suffix) for suffix in _LEGAL_SUFFIXES):
                candidates.append(term)
                continue

            # Single words: check if contains a legal prefix/root
            if any(term_lower.startswith(prefix) for prefix in _LEGAL_BIGRAM_PREFIXES):
                candidates.append(term)
                continue

            # Capitalized single word (likely a proper legal entity/concept)
            if term[0].isupper() and len(term) >= 4:
                candidates.append(term)
                continue

        return candidates

    def _infer_category(self, term: str) -> str:
        """Infer the most likely category for a term based on keyword hints.

        Returns the best matching category or 'general_legal' as fallback.
        """
        term_lower = term.lower()

        best_category = "general_legal"
        best_score = 0

        for category, keywords in _CATEGORY_HINTS.items():
            score = 0
            for keyword in keywords:
                if keyword in term_lower:
                    score += 2
                # Also check if the term is semantically close to category keywords
                elif any(keyword.startswith(term_lower[:4]) for _ in [None] if len(term_lower) >= 4):
                    score += 1

            if score > best_score:
                best_score = score
                best_category = category

        return best_category

    def _compute_confidence(self, term: str) -> float:
        """Compute initial confidence that a term is a legitimate legal term.

        Factors:
        - Multi-word terms get higher confidence (0.5 base)
        - Legal suffix presence: +0.15
        - Legal keyword presence: +0.2
        - Length >= 6 chars: +0.05
        """
        confidence = 0.3  # base
        term_lower = term.lower()

        # Multi-word terms are more likely to be genuine
        if " " in term:
            confidence += 0.2

        # Legal suffix
        if any(term_lower.endswith(suffix) for suffix in _LEGAL_SUFFIXES):
            confidence += 0.15

        # Contains a known legal root
        if any(prefix in term_lower for prefix in _LEGAL_BIGRAM_PREFIXES):
            confidence += 0.2

        # Reasonable length
        if len(term) >= 6:
            confidence += 0.05

        return min(1.0, round(confidence, 2))

    def _load_learned_map(self) -> dict[str, LearnedTerm]:
        """Load learned terms into a dict keyed by lowercase term."""
        terms = self.get_learned_terms()
        return {t.term.lower(): t for t in terms}

    def _save_learned_map(self, term_map: dict[str, LearnedTerm]) -> None:
        """Persist the full learned term map to the JSONL file."""
        with open(LEARNED_VOCAB_PATH, "w", encoding="utf-8") as f:
            for learned in term_map.values():
                f.write(json.dumps(asdict(learned)) + "\n")
