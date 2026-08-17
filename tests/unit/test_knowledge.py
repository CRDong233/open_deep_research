"""Unit tests for the Qdrant-backed Agent knowledge tool."""

from collections.abc import Sequence

import pytest
from qdrant_client import QdrantClient

from open_deep_research.knowledge import (
    KnowledgeSettings,
    build_knowledge_service,
    create_knowledge_search_tool,
)


class KeywordEmbedder:
    """Deterministic vectors that avoid model downloads in unit tests."""

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Map retry evidence to one vector direction."""
        return [[1.0, 0.0] if "retry" in text.lower() else [0.0, 1.0] for text in texts]

    def embed_query(self, text: str) -> Sequence[float]:
        """Map retry queries to the matching direction."""
        return [1.0, 0.0] if "retry" in text.lower() else [0.0, 1.0]


def test_knowledge_service_ingests_and_returns_citations(tmp_path) -> None:
    """A configured source directory should become an Agent search tool."""
    (tmp_path / "recovery.md").write_text(
        "# Tool Recovery\n\nA retry must be bounded and observable.",
        encoding="utf-8",
    )
    settings = KnowledgeSettings(
        enabled=True,
        source_path=str(tmp_path),
        qdrant_location=":memory:",
        top_k=1,
    )
    service = build_knowledge_service(
        settings,
        client=QdrantClient(location=":memory:"),
        embedder=KeywordEmbedder(),
    )
    knowledge_tool = create_knowledge_search_tool(service)

    result = knowledge_tool.invoke({"query": "retry policy"})

    assert service.indexed_chunks == 1
    assert service.ingestion is not None
    assert "[E1] Tool Recovery" in result
    assert "#char=" in result
    assert "bounded and observable" in result


def test_knowledge_service_restores_existing_collection(tmp_path) -> None:
    """A second service should restore payloads without requiring source files."""
    (tmp_path / "memory.md").write_text(
        "# Memory\n\nLong-term memory needs conflict handling.",
        encoding="utf-8",
    )
    client = QdrantClient(location=":memory:")
    writer_settings = KnowledgeSettings(
        enabled=True,
        source_path=str(tmp_path),
        qdrant_location=":memory:",
    )
    writer = build_knowledge_service(
        writer_settings,
        client=client,
        embedder=KeywordEmbedder(),
    )
    reader_settings = KnowledgeSettings(
        enabled=True,
        qdrant_location=":memory:",
    )

    reader = build_knowledge_service(
        reader_settings,
        client=client,
        embedder=KeywordEmbedder(),
    )

    assert writer.indexed_chunks == reader.indexed_chunks == 1
    assert reader.ingestion is None
    assert "Memory" in reader.search("conflict handling")


def test_disabled_or_unconfigured_knowledge_fails_explicitly() -> None:
    """Knowledge indexing should never start without explicit configuration."""
    with pytest.raises(ValueError, match="disabled"):
        build_knowledge_service(KnowledgeSettings())
    with pytest.raises(ValueError, match="knowledge_base_path"):
        build_knowledge_service(
            KnowledgeSettings(enabled=True, qdrant_location=":memory:"),
            client=QdrantClient(location=":memory:"),
            embedder=KeywordEmbedder(),
        )
