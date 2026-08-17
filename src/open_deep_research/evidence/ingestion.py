"""Safe local text ingestion for evidence-backed research."""

import hashlib
import re
from pathlib import Path

from pydantic import BaseModel, Field

from open_deep_research.evidence.chunking import MarkdownChunker
from open_deep_research.evidence.models import EvidenceChunk, EvidenceDocument

_TITLE_PATTERN = re.compile(r"(?m)^#[ \t]+(.+?)[ \t]*$")


class IngestionIssue(BaseModel):
    """A source file skipped during ingestion and its reason."""

    path: str
    reason: str


class IngestionResult(BaseModel):
    """Documents, chunks, and non-fatal issues from one directory scan."""

    documents: list[EvidenceDocument] = Field(default_factory=list)
    chunks: list[EvidenceChunk] = Field(default_factory=list)
    issues: list[IngestionIssue] = Field(default_factory=list)


class DirectoryDocumentLoader:
    """Read bounded UTF-8 text files from one explicit root directory."""

    def __init__(
        self,
        root: str | Path,
        *,
        allowed_suffixes: tuple[str, ...] = (".md", ".markdown", ".txt"),
        max_file_bytes: int = 2_000_000,
    ) -> None:
        """Configure a read-only directory scan."""
        self.root = Path(root).expanduser().resolve()
        self.allowed_suffixes = tuple(suffix.lower() for suffix in allowed_suffixes)
        self.max_file_bytes = max_file_bytes
        if not self.allowed_suffixes:
            raise ValueError("allowed_suffixes must not be empty")
        if max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be positive")

    def load(self) -> tuple[list[EvidenceDocument], list[IngestionIssue]]:
        """Load supported files in deterministic relative-path order."""
        if not self.root.is_dir():
            raise ValueError(f"knowledge root is not a directory: {self.root}")

        documents: list[EvidenceDocument] = []
        issues: list[IngestionIssue] = []
        candidates = sorted(
            (
                path
                for path in self.root.rglob("*")
                if path.is_file() and path.suffix.lower() in self.allowed_suffixes
            ),
            key=lambda path: path.relative_to(self.root).as_posix(),
        )
        for path in candidates:
            relative_path = path.relative_to(self.root).as_posix()
            size = path.stat().st_size
            if size > self.max_file_bytes:
                issues.append(
                    IngestionIssue(
                        path=relative_path,
                        reason=f"file exceeds {self.max_file_bytes} bytes",
                    )
                )
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as error:
                issues.append(
                    IngestionIssue(
                        path=relative_path,
                        reason=f"unable to read UTF-8 text: {error.__class__.__name__}",
                    )
                )
                continue
            if not content.strip():
                issues.append(
                    IngestionIssue(path=relative_path, reason="file is empty")
                )
                continue

            title_match = _TITLE_PATTERN.search(content)
            title = title_match.group(1).strip() if title_match else path.stem
            document_id = hashlib.sha256(relative_path.encode()).hexdigest()[:20]
            documents.append(
                EvidenceDocument(
                    document_id=document_id,
                    content=content,
                    title=title,
                    uri=path.as_uri(),
                    metadata={
                        "relative_path": relative_path,
                        "suffix": path.suffix.lower(),
                        "bytes": size,
                    },
                )
            )
        return documents, issues


def ingest_directory(
    loader: DirectoryDocumentLoader,
    chunker: MarkdownChunker | None = None,
) -> IngestionResult:
    """Load and split a directory into one auditable ingestion result."""
    documents, issues = loader.load()
    active_chunker = chunker or MarkdownChunker()
    chunks = [
        chunk for document in documents for chunk in active_chunker.split(document)
    ]
    return IngestionResult(documents=documents, chunks=chunks, issues=issues)
