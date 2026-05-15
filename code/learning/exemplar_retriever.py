"""Exemplar retriever — finds relevant past corrections for prompt injection.

Uses ChromaDB to store correction embeddings and perform semantic search.
When generating a new draft, retrieves the top-K most similar corrections
to inject as few-shot examples in the prompt.
"""

import logging
from typing import Optional

import chromadb
from chromadb.config import Settings

from config.paths import CHROMA_DIR
from code.learning.correction_store import Correction, CorrectionStore
from code.llm_interface.model_manager import ModelManager

logger = logging.getLogger(__name__)

COLLECTION_NAME = "legalmind_corrections"


class ExemplarRetriever:
    """Retrieves semantically similar past corrections for few-shot injection."""

    def __init__(self):
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        self._store = CorrectionStore()

    def index_correction(self, correction: Correction) -> None:
        """Add a correction to the retrieval index."""
        if not correction.embedding:
            mgr = ModelManager.instance()
            correction.embedding = mgr.embed([correction.source_ocr_chunk])[0]

        self._collection.upsert(
            ids=[correction.id],
            embeddings=[correction.embedding],
            metadatas=[{
                "draft_type": correction.draft_type,
                "field_path": correction.field_path,
                "correction_type": correction.correction_type,
                "active": correction.active,
            }],
            documents=[correction.source_ocr_chunk],
        )

    def get_relevant_exemplars(self, source_text: str,
                                field_type: Optional[str] = None,
                                draft_type: Optional[str] = None,
                                k: int = 3) -> list[Correction]:
        """Retrieve top-K most similar corrections for few-shot injection.

        Args:
            source_text: The source text being processed (query).
            field_type: Optional filter by field path prefix.
            draft_type: Optional filter by draft type.
            k: Number of exemplars to retrieve.

        Returns:
            List of Correction objects, most similar first.
        """
        mgr = ModelManager.instance()
        query_embedding = mgr.embed([source_text])[0]

        where_filter = {"active": True}
        if draft_type:
            where_filter["draft_type"] = draft_type

        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_filter if len(where_filter) > 1 else None,
            )
        except Exception as e:
            logger.warning("Exemplar retrieval failed: %s", e)
            return []

        if not results["ids"] or not results["ids"][0]:
            return []

        corrections = []
        for corr_id in results["ids"][0]:
            active_corrections = []
            for draft_file_type in ["case_fact_summary", "title_review_summary",
                                     "notice_summary", "document_checklist"]:
                active_corrections.extend(self._store.get_all_active(draft_file_type))

            for c in active_corrections:
                if c.id == corr_id:
                    corrections.append(c)
                    break

        return corrections

    def format_exemplars_for_prompt(self, exemplars: list[Correction]) -> str:
        """Format exemplars as few-shot examples for prompt injection.

        Returns a formatted string block ready to insert into generation prompts.
        """
        if not exemplars:
            return ""

        lines = ["PAST CORRECTIONS (use these to avoid similar errors):", ""]
        for i, ex in enumerate(exemplars, 1):
            lines.append(f"Example {i}:")
            lines.append(f"- Source text: \"{ex.source_ocr_chunk[:200]}\"")
            lines.append(f"- System generated: \"{ex.generated_text[:200]}\"")
            lines.append(f"- Operator corrected to: \"{ex.edited_text[:200]}\"")
            lines.append(f"- Field: {ex.field_path}")
            lines.append("")

        return "\n".join(lines)

    def rebuild_index(self) -> int:
        """Rebuild the entire correction index from stored corrections."""
        count = 0
        for draft_type in ["case_fact_summary", "title_review_summary",
                           "notice_summary", "document_checklist"]:
            for correction in self._store.get_all_active(draft_type):
                self.index_correction(correction)
                count += 1
        logger.info("Rebuilt correction index: %d entries", count)
        return count
