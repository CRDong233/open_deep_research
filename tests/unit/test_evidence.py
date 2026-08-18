"""Unit tests for offline evidence ingestion, retrieval, and evaluation."""

from collections.abc import Sequence

import pytest
from qdrant_client import QdrantClient

from open_deep_research.evidence import (
    ChunkingConfig,
    DirectoryDocumentLoader,
    EvidenceChunk,
    EvidenceDocument,
    FastEmbedTextEmbedder,
    InMemoryHybridRetriever,
    MarkdownChunker,
    QdrantHybridRetriever,
    RetrievalExample,
    audit_citations,
    citation_id,
    evaluate_retriever,
    format_evidence_context,
    ingest_directory,
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
    stable_id = citation_id(hits[0].chunk.chunk_id)
    audit = audit_citations(f"Supported [{stable_id}], legacy [E1], invented [E9].", hits)

    assert f"[{stable_id}] Agent handbook" in context
    assert "file:///agent.md#char=0,28" in context
    assert audit.cited_ids == [stable_id, "E1", "E9"]
    assert audit.unknown_ids == ["E1", "E9"]
    assert not audit.valid


def test_citation_id_is_stable_across_retrieval_rounds() -> None:
    """The same chunk must keep one identifier across separate searches."""
    chunk = make_chunk("document:stable-chunk", "Evidence requires citations.")
    first = citation_id(chunk.chunk_id)
    second = citation_id(chunk.model_copy().chunk_id)
    assert first == second
    assert first.startswith("E-")


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


def test_qdrant_adapter_persists_and_fuses_candidates() -> None:
    """Qdrant vectors and BM25 scores should produce one ranked result."""
    chunks = [
        make_chunk("recovery", "agent recovery"),
        make_chunk("planning", "agent planning"),
        make_chunk("memory", "long term memory"),
    ]
    retriever = QdrantHybridRetriever(
        QdrantClient(location=":memory:"),
        FakeEmbedder(),
        semantic_weight=0.8,
    )

    retriever.index(chunks)
    hits = retriever.retrieve("agent", top_k=2)

    assert retriever.indexed_count == 3
    assert hits[0].chunk.chunk_id == "recovery"
    assert hits[0].semantic_score == pytest.approx(1.0)
    assert hits[0].lexical_score > 0


def test_qdrant_adapter_requires_explicit_snapshot_replacement() -> None:
    """Existing collections should not be deleted without explicit ownership."""
    client = QdrantClient(location=":memory:")
    initial = QdrantHybridRetriever(client, FakeEmbedder())
    initial.index([make_chunk("one", "agent recovery")])

    guarded = QdrantHybridRetriever(client, FakeEmbedder())
    with pytest.raises(ValueError, match="replace_snapshot"):
        guarded.index([make_chunk("two", "agent planning")])

    replacement = QdrantHybridRetriever(
        client,
        FakeEmbedder(),
        replace_snapshot=True,
    )
    replacement.index([make_chunk("two", "agent planning")])
    assert replacement.indexed_count == 1


def test_fastembed_adapter_is_lazy() -> None:
    """Constructing the production embedder should not download model weights."""
    embedder = FastEmbedTextEmbedder(cache_dir="unused-in-construction")

    assert embedder.model_name
    assert embedder._model is None


def test_directory_ingestion_is_bounded_and_traceable(tmp_path) -> None:
    """Only bounded UTF-8 source files should become evidence documents."""
    (tmp_path / "guide.md").write_text(
        "# Recovery Guide\n\nRetries must be bounded.",
        encoding="utf-8",
    )
    (tmp_path / "empty.txt").write_text("  ", encoding="utf-8")
    (tmp_path / "ignored.json").write_text("{}", encoding="utf-8")
    (tmp_path / "large.md").write_text("x" * 200, encoding="utf-8")
    (tmp_path / "binary.txt").write_bytes(b"\xff\xfe")
    loader = DirectoryDocumentLoader(tmp_path, max_file_bytes=100)

    result = ingest_directory(
        loader,
        MarkdownChunker(ChunkingConfig(max_chars=80, overlap_chars=10, min_chars=20)),
    )

    assert [document.title for document in result.documents] == ["Recovery Guide"]
    assert result.documents[0].metadata["relative_path"] == "guide.md"
    assert len(result.chunks) == 1
    assert result.chunks[0].uri.startswith("file:")
    assert {issue.path for issue in result.issues} == {
        "binary.txt",
        "empty.txt",
        "large.md",
    }


def test_qdrant_snapshot_can_be_loaded_without_reindexing() -> None:
    """Stored payloads should restore local BM25 state after a restart."""
    client = QdrantClient(location=":memory:")
    writer = QdrantHybridRetriever(client, FakeEmbedder())
    writer.index(
        [
            make_chunk("recovery", "agent recovery"),
            make_chunk("planning", "agent planning"),
        ]
    )
    reader = QdrantHybridRetriever(client, FakeEmbedder())

    loaded = reader.load_snapshot()
    hits = reader.retrieve("recovery", top_k=1)

    assert loaded == 2
    assert hits[0].chunk.chunk_id == "recovery"
