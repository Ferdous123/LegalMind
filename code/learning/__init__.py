"""Learning system — captures corrections, extracts patterns, consolidates prompts."""

from code.learning.correction_store import CorrectionStore, Correction
from code.learning.exemplar_retriever import ExemplarRetriever
from code.learning.pattern_extractor import PatternExtractor
from code.learning.prompt_consolidator import PromptConsolidator

__all__ = [
    "CorrectionStore", "Correction",
    "ExemplarRetriever",
    "PatternExtractor",
    "PromptConsolidator",
]
