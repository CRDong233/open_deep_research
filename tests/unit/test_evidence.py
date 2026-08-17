"""Unit tests for offline evidence ingestion, retrieval, and evaluation."""

from collections.abc import Sequence

import pytest

from open_deep_research.evidence import (
    ChunkingConfig,
    EvidenceChunk,
    EvidenceDocument,
    InMemoryHybridRetriever,
    MarkdownChunker,
    RetrievalExample,
    audit_citations,
    evaluate_retriever,
    format_evidence_context,
)
from open_deep_research.evidence.retrieval import tokenize


def make_chunk(chunk_id: str, content: str) -> EvidenceChunk:
    """Create a compact chunk fixture."""
    return EvidenceChunk(
        chunk_id=chunk_id,
        document_id="doc",
        content=content,
        title="Agent handbook",
        uri="file:///agent.md",
        start_char=0,
        end_char=len(content),
    )


def test_chunker_preserves_offsets_sections_and_stable_ids() -> None:
    """Chunks should remain traceable to exact source slices."""
    content = (
        "# Retrieval\n\nHybrid retrieval combines lexical and semantic ranking. "
        "Citations preserve evidence.\n\n"
        "## Recovery\n\nRetries must be bounded and observable. "
        "Fallbacks should preserve the original error."
    )
    document = EvidenceDocument(
        document_id="handbook",
        content=content,
        title="Agent handbook",
        uri="file:///agent.md",
        metadata={"team": "ai"},
    )
    chunker = MarkdownChunker(
        ChunkingConfig(max_chars=105, overlap_chars=20, min_chars=45)
    )

    first_run = chunker.split(document)
    second_run = chunker.split(document)

    assert len(first_run) >= 2
    assert [chunk.chunk_id for chunk in first_run] == [
        chunk.chunk_id for chunk in second_run
    ]
    for chunk in first_run:
        assert content[chunk.start_char : chunk.end_char] == chunk.content
        assert chunk.metadata == {"team": "ai"}
    assert first_run[0].section == "Retrieval"
    assert first_run[-1].section == "Recovery"


def test_lexical_retrieval_supports_english_and_chinese() -> None:
    """BM25 should rank relevant English and Chinese evidence first."""
    chunks = [
        make_chunk("retrieval", "Hybrid retrieval combines BM25 and embeddings."),
        make_chunk("recovery", "Tool failures use bounded retries and fallback."),
        make_chunk("memory", "长期记忆需要去重、衰减和冲突处理。"),
    ]
    retriever = InMemoryHybridRetriever()
    retriever.index(chunks)

    assert (
        retriever.retrieve("BM25 retrieval", top_k=1)[0].chunk.chunk_id == "retrieval"
    )
    assert retriever.retrieve("长期记忆冲突", top_k=1)[0].chunk.chunk_id == "memory"
    assert "长期" in tokenize("长期记忆")


class FakeEmbedder:
    """Deterministic semantic vectors for a hybrid-ranking unit test."""

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Map recovery text to the query direction."""
        return [[1.0, 0.0] if "recovery" in text else [0.0, 1.0] for text in texts]

    def embed_query(self, text: str) -> Sequence[float]:
        """Return a fixed recovery-oriented query vector."""
        return [1.0, 0.0]


def test_semantic_score_can_resolve_a_lexical_tie() -> None:
    """The optional semantic component should affect final ranking."""
    retriever = InMemoryHybridRetriever(embedder=FakeEmbedder(), semantic_weight=0.8)
    retriever.index(
        [
            make_chunk("recovery", "agent recovery"),
            make_chunk("planning", "agent planning"),
        ]
    )

    hits = retriever.retrieve("agent", top_k=2)

    assert hits[0].chunk.chunk_id == "recovery"
    assert hits[0].semantic_score == pytest.approx(1.0)
    assert hits[0].score > hits[1].score


def test_citation_context_and_audit_expose_unknown_references() -> None:
    """Citation audits should reject identifiers absent from the context."""
    retriever = InMemoryHybridRetriever()
    retriever.index([make_chunk("retrieval", "Evidence requires citations.")])
    hits = retriever.retrieve("citations", top_k=1)

    context = format_evidence_context(hits)
    audit = audit_citations("Supported [E1], invented [E9].", hits)

    assert "[E1] Agent handbook" in context
    assert "file:///agent.md#char=0,28" in context
    assert audit.cited_ids == ["E1", "E9"]
    assert audit.unknown_ids == ["E9"]
    assert not audit.valid


def test_evaluation_reports_recall_and_reciprocal_rank() -> None:
    """The local harness should calculate deterministic ranking metrics."""
    retriever = InMemoryHybridRetriever()
    retriever.index(
        [
            make_chunk("retrieval", "hybrid BM25 retrieval"),
            make_chunk("recovery", "bounded tool retry"),
            make_chunk("memory", "long term memory"),
        ]
    )
    examples = [
        RetrievalExample(query="BM25", relevant_chunk_ids={"retrieval"}),
        RetrievalExample(query="tool retry", relevant_chunk_ids={"recovery"}),
    ]

    metrics = evaluate_retriever(retriever, examples, top_k=2)

    assert metrics.recall_at_k == pytest.approx(1.0)
    assert metrics.mean_reciprocal_rank == pytest.approx(1.0)
    assert metrics.evaluated_queries == 2


def test_invalid_limits_and_empty_queries_fail_safely() -> None:
    """Public interfaces should reject invalid limits without partial work."""
    with pytest.raises(ValueError, match="overlap_chars"):
        ChunkingConfig(max_chars=100, overlap_chars=100, min_chars=20)

    retriever = InMemoryHybridRetriever()
    retriever.index([make_chunk("one", "content")])
    assert retriever.retrieve("   ") == []
    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("content", top_k=0)
