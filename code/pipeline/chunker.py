"""Text chunker — splits extracted text into retrieval-ready chunks.

Uses semantic chunking with overlap: prefers paragraph boundaries, falls back
to sentence boundaries, last resort is token-count splitting.
"""

import logging
import re
import uuid
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 512  # target tokens per chunk
DEFAULT_OVERLAP = 64  # overlap tokens between chunks
APPROX_CHARS_PER_TOKEN = 4  # rough estimate for English text


@dataclass
class TextChunk:
    """A text chunk ready for retrieval indexing."""
    id: str
    text: str
    document_id: str
    page_number: int
    char_start: int
    char_end: int

    def __init__(self, text: str = "", document_id: str = "",
                 page_number: int = 0, char_start: int = 0, char_end: int = 0,
                 id: str = ""):
        self.id = id or f"chunk_{uuid.uuid4().hex[:8]}"
        self.text = text
        self.document_id = document_id
        self.page_number = page_number
        self.char_start = char_start
        self.char_end = char_end


class TextChunker:
    """Splits document text into overlapping chunks for retrieval indexing."""

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE,
                 overlap: int = DEFAULT_OVERLAP):
        self._chunk_size_chars = chunk_size * APPROX_CHARS_PER_TOKEN
        self._overlap_chars = overlap * APPROX_CHARS_PER_TOKEN

    def chunk(self, text: str, document_id: str,
              page_spans: list[tuple[int, int, int]] | None = None) -> list[TextChunk]:
        """Split text into chunks with semantic boundary detection.

        Strategy:
        1. Split by paragraphs (double newline)
        2. If paragraph > chunk_size, split by sentences
        3. Merge small paragraphs into chunks up to chunk_size
        4. Add overlap between chunks

        Args:
            text: The full document text.
            document_id: Owning document ID.
            page_spans: Optional list of (page_number, char_start, char_end)
                tuples describing where each source page sits inside `text`.
                When provided, each chunk's page_number is resolved from the
                page whose char range contains the chunk's start — the real
                page, not the char_start//3000 estimate. Built by the
                ingester from doc.pages so PDF citations open the correct
                page image.
        """
        if not text.strip():
            return []

        paragraphs = self._split_paragraphs(text)
        chunks = []
        current_text = ""
        current_start = 0
        position = 0

        for para in paragraphs:
            para_len = len(para)

            if para_len > self._chunk_size_chars:
                # Save current buffer as chunk
                if current_text.strip():
                    chunks.append(self._make_chunk(
                        current_text.strip(), document_id, current_start
                    ))

                # Split large paragraph by sentences
                sentences = self._split_sentences(para)
                sent_buffer = ""
                sent_start = position

                for sent in sentences:
                    if len(sent_buffer) + len(sent) > self._chunk_size_chars:
                        if sent_buffer.strip():
                            chunks.append(self._make_chunk(
                                sent_buffer.strip(), document_id, sent_start
                            ))
                        # Overlap: keep last portion
                        overlap_text = sent_buffer[-self._overlap_chars:]
                        sent_buffer = overlap_text + sent
                        sent_start = position + len(para) - len(sent_buffer)
                    else:
                        sent_buffer += sent

                if sent_buffer.strip():
                    chunks.append(self._make_chunk(
                        sent_buffer.strip(), document_id, sent_start
                    ))
                current_text = ""
                current_start = position + para_len

            elif len(current_text) + para_len > self._chunk_size_chars:
                # Current buffer full — save and start new
                if current_text.strip():
                    chunks.append(self._make_chunk(
                        current_text.strip(), document_id, current_start
                    ))
                # Overlap: keep end of current
                overlap = current_text[-self._overlap_chars:] if len(current_text) > self._overlap_chars else ""
                current_text = overlap + para + "\n\n"
                current_start = position - len(overlap)
            else:
                if not current_text:
                    current_start = position
                current_text += para + "\n\n"

            position += para_len + 2  # +2 for the \n\n separator

        # Flush remaining buffer
        if current_text.strip():
            chunks.append(self._make_chunk(
                current_text.strip(), document_id, current_start
            ))

        # Assign page numbers. If the ingester gave us real page spans, map
        # each chunk to the page whose char range contains its start. Else
        # fall back to the rough ~3000-chars-per-page heuristic (txt files,
        # or callers that don't supply spans).
        if page_spans:
            for chunk in chunks:
                chunk.page_number = self._resolve_page(chunk.char_start, page_spans)
        else:
            for chunk in chunks:
                chunk.page_number = max(1, chunk.char_start // 3000 + 1)

        return chunks

    @staticmethod
    def _resolve_page(char_start: int, page_spans: list[tuple[int, int, int]]) -> int:
        """Return the page_number whose [start, end) range contains char_start.

        Falls back to the nearest preceding page if the position lands in a
        page separator gap, or page 1 if before all spans.
        """
        resolved = page_spans[0][0] if page_spans else 1
        for page_number, start, end in page_spans:
            if start <= char_start < end:
                return page_number
            if char_start >= start:
                resolved = page_number
        return resolved

    def _make_chunk(self, text: str, document_id: str, char_start: int) -> TextChunk:
        return TextChunk(
            text=text,
            document_id=document_id,
            char_start=char_start,
            char_end=char_start + len(text),
        )

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        """Split text on double newlines (paragraph boundaries)."""
        paras = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paras if p.strip()]

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences using regex."""
        # Split on period/question/exclamation followed by space and uppercase
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        return [p + " " for p in parts]
