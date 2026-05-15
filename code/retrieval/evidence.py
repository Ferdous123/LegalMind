"""Evidence packager — formats retrieved chunks for prompt injection."""

import logging
from dataclasses import dataclass

from code.retrieval.searcher import EvidenceChunk

logger = logging.getLogger(__name__)


@dataclass
class EvidencePackage:
    """Formatted evidence ready for prompt injection."""
    formatted_text: str
    citation_map: dict  # {citation_id: EvidenceChunk metadata}
    chunk_count: int


class EvidencePackager:
    """Formats retrieved evidence chunks for inclusion in generation prompts.

    Produces numbered citations like [E1], [E2], etc. that the draft generator
    can reference. Also produces a citation map for verification.
    """

    def package(self, chunks: list[EvidenceChunk],
                max_chunks: int = 10) -> EvidencePackage:
        """Package evidence chunks into formatted prompt text.

        Args:
            chunks: Retrieved evidence chunks, sorted by relevance.
            max_chunks: Maximum number to include (to respect context limits).

        Returns:
            EvidencePackage with formatted text and citation map.
        """
        if not chunks:
            return EvidencePackage(
                formatted_text="[No relevant evidence retrieved]",
                citation_map={},
                chunk_count=0,
            )

        selected = chunks[:max_chunks]
        lines = ["EVIDENCE FROM SOURCE DOCUMENTS:", ""]
        citation_map = {}

        for i, chunk in enumerate(selected, 1):
            citation_id = f"E{i}"
            text_preview = chunk.text.strip()

            # Truncate very long chunks for prompt efficiency
            if len(text_preview) > 500:
                text_preview = text_preview[:497] + "..."

            lines.append(
                f"[{citation_id}] \"{text_preview}\" "
                f"(Document: {chunk.document_id}, Page: {chunk.page_number}, "
                f"Relevance: {chunk.similarity_score:.2f})"
            )
            lines.append("")

            citation_map[citation_id] = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "page_number": chunk.page_number,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "text": chunk.text,
                "similarity_score": chunk.similarity_score,
            }

        return EvidencePackage(
            formatted_text="\n".join(lines),
            citation_map=citation_map,
            chunk_count=len(selected),
        )

    def get_evidence_for_field(self, chunks: list[EvidenceChunk],
                               field_name: str) -> str:
        """Get a compact evidence string for a specific field verification."""
        relevant = [c for c in chunks if c.similarity_score > 0.5]
        if not relevant:
            return ""
        return " | ".join(c.text[:200] for c in relevant[:3])
