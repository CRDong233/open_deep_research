"""Evidence ingestion, retrieval, citation, and evaluation primitives."""

from open_deep_research.evidence.chunking import ChunkingConfig, MarkdownChunker
from open_deep_research.evidence.citations import (
    CitationAudit,
    audit_citations,
    format_evidence_context,
)
from open_deep_research.evidence.evaluation import (
    RetrievalExample,
    RetrievalMetrics,
    evaluate_retriever,
)
from open_deep_research.evidence.ingestion import (
    DirectoryDocumentLoader,
    IngestionIssue,
    IngestionResult,
    ingest_directory,
)
from open_deep_research.evidence.models import (
    EvidenceChunk,
    EvidenceDocument,
    RetrievalHit,
)
from open_deep_research.evidence.qdrant_store import (
    DEFAULT_EMBEDDING_MODEL,
    FastEmbedTextEmbedder,
    QdrantHybridRetriever,
)
from open_deep_research.evidence.retrieval import (
    Embedder,
    InMemoryHybridRetriever,
)

__all__ = [
    "ChunkingConfig",
    "CitationAudit",
    "DEFAULT_EMBEDDING_MODEL",
    "Embedder",
    "DirectoryDocumentLoader",
    "EvidenceChunk",
    "EvidenceDocument",
    "FastEmbedTextEmbedder",
    "InMemoryHybridRetriever",
    "IngestionIssue",
    "IngestionResult",
    "MarkdownChunker",
    "QdrantHybridRetriever",
    "RetrievalExample",
    "RetrievalHit",
    "RetrievalMetrics",
    "audit_citations",
    "evaluate_retriever",
    "format_evidence_context",
    "ingest_directory",
]
