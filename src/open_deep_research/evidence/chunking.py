"""Markdown-aware chunking that preserves exact source offsets."""

import hashlib
import re
from dataclasses import dataclass

from open_deep_research.evidence.models import EvidenceChunk, EvidenceDocument

_HEADING_PATTERN = re.compile(r"(?m)^(#{1,6})[ \t]+(.+?)[ \t]*$")
_BOUNDARY_PATTERN = re.compile(
    r"\n\s*\n|^#{1,6}[ \t]+|[.!?]\s+",
    re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Character limits for deterministic source-preserving chunks."""

    max_chars: int = 1800
    overlap_chars: int = 180
    min_chars: int = 120

    def __post_init__(self) -> None:
        """Reject limits that could stall or produce invalid chunks."""
        if self.max_chars <= 0:
            raise ValueError("max_chars must be positive")
        if self.min_chars <= 0 or self.min_chars > self.max_chars:
            raise ValueError("min_chars must be between 1 and max_chars")
        if self.overlap_chars < 0 or self.overlap_chars >= self.max_chars:
            raise ValueError("overlap_chars must be smaller than max_chars")


class MarkdownChunker:
    """Split Markdown near semantic boundaries without changing source text."""

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        """Create a chunker with validated limits."""
        self.config = config or ChunkingConfig()

    def split(self, document: EvidenceDocument) -> list[EvidenceChunk]:
        """Split a document into deterministic, overlapping evidence chunks."""
        content = document.content
        headings = [
            (match.start(), match.group(2).strip())
            for match in _HEADING_PATTERN.finditer(content)
        ]
        chunks: list[EvidenceChunk] = []
        cursor = 0

        while cursor < len(content):
            hard_end = min(cursor + self.config.max_chars, len(content))
            end = self._choose_boundary(content, cursor, hard_end)
            start, stripped_end = self._trim_range(content, cursor, end)

            if stripped_end > start:
                chunk_content = content[start:stripped_end]
                chunks.append(
                    EvidenceChunk(
                        chunk_id=self._chunk_id(
                            document.document_id,
                            start,
                            stripped_end,
                            chunk_content,
                        ),
                        document_id=document.document_id,
                        content=chunk_content,
                        title=document.title,
                        uri=document.uri,
                        start_char=start,
                        end_char=stripped_end,
                        section=self._section_at(headings, start),
                        metadata=dict(document.metadata),
                    )
                )

            if end >= len(content):
                break
            next_cursor = max(end - self.config.overlap_chars, cursor + 1)
            cursor = self._skip_whitespace(content, next_cursor)

        return chunks

    def _choose_boundary(self, content: str, start: int, hard_end: int) -> int:
        """Choose the latest semantic boundary after the minimum chunk size."""
        if hard_end >= len(content):
            return len(content)

        minimum_end = min(start + self.config.min_chars, hard_end)
        candidates = [
            match.end()
            for match in _BOUNDARY_PATTERN.finditer(content, minimum_end, hard_end)
        ]
        return candidates[-1] if candidates else hard_end

    @staticmethod
    def _trim_range(content: str, start: int, end: int) -> tuple[int, int]:
        """Trim surrounding whitespace while retaining original offsets."""
        while start < end and content[start].isspace():
            start += 1
        while end > start and content[end - 1].isspace():
            end -= 1
        return start, end

    @staticmethod
    def _skip_whitespace(content: str, cursor: int) -> int:
        """Advance over whitespace before the next chunk."""
        while cursor < len(content) and content[cursor].isspace():
            cursor += 1
        return cursor

    @staticmethod
    def _section_at(headings: list[tuple[int, str]], offset: int) -> str | None:
        """Return the nearest heading at or before an offset."""
        section = None
        for heading_offset, heading in headings:
            if heading_offset > offset:
                break
            section = heading
        return section

    @staticmethod
    def _chunk_id(document_id: str, start: int, end: int, content: str) -> str:
        """Build a stable chunk identifier from source identity and content."""
        payload = f"{document_id}:{start}:{end}:{content}".encode()
        digest = hashlib.sha256(payload).hexdigest()[:20]
        return f"{document_id}:{digest}"
