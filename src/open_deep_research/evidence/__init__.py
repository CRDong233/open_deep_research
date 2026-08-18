"""Evidence ingestion, retrieval, citation, and evaluation primitives."""

from open_deep_research.evidence.chunking import ChunkingConfig, MarkdownChunker
from open_deep_research.evidence.citations import (
    CitationAudit,
    audit_citations,
    citation_id,
    format_evidence_context,
)
from open_deep_research.evidence.evaluation import (
    CitationMetrics,
    EvaluationRun,
    EvaluationSummary,
    RetrievalExample,
    RetrievalMetrics,
    evaluate_citation_references,
    evaluate_retriever,
    load_retrieval_examples,
    summarize_evaluation_runs,
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
    SnapshotSyncResult,
)
from open_deep_research.evidence.retrieval import (
    Embedder,
    InMemoryHybridRetriever,
)

__all__ = [
    "ChunkingConfig",
    "CitationMetrics",
    "CitationAudit",
    "DEFAULT_EMBEDDING_MODEL",
    "Embedder",
    "DirectoryDocumentLoader",
    "EvaluationRun",
    "EvaluationSummary",
    "EvidenceChunk",
    "EvidenceDocument",
    "FastEmbedTextEmbedder",
    "InMemoryHybridRetriever",
    "IngestionIssue",
    "IngestionResult",
    "MarkdownChunker",
    "QdrantHybridRetriever",
    "SnapshotSyncResult",
    "RetrievalExample",
    "RetrievalHit",
    "RetrievalMetrics",
    "audit_citations",
    "citation_id",
    "evaluate_retriever",
    "summarize_evaluation_runs",
    "evaluate_citation_references",
    "format_evidence_context",
    "ingest_directory",
    "load_retrieval_examples",
]
