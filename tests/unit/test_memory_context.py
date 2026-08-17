"""Unit tests for long-term memory and context budgeting."""

from collections.abc import Sequence

import pytest
from qdrant_client import QdrantClient

from open_deep_research.context_budget import (
    ContextBudget,
    ContextMessage,
    compress_context,
)
from open_deep_research.memory import MemoryKind, QdrantMemoryStore


class MemoryEmbedder:
    """Deterministic memory vectors for offline tests."""

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Embed documents through the same keyword function."""
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> Sequence[float]:
        """Separate preference and recovery memories."""
        lowered = text.lower()
        if "python" in lowered:
            return [1.0, 0.0]
        return [0.0, 1.0]


def test_memory_is_deduplicated_scoped_and_forgettable() -> None:
    """Memory IDs should deduplicate content without crossing user boundaries."""
    store = QdrantMemoryStore(QdrantClient(location=":memory:"), MemoryEmbedder())
    first = store.remember(
        user_id="alice",
        content="Prefers Python examples",
        kind=MemoryKind.PREFERENCE,
        importance=0.9,
    )
    duplicate = store.remember(
        user_id="alice",
        content="  PREFERS   python examples ",
        kind=MemoryKind.PREFERENCE,
        importance=0.8,
    )
    store.remember(
        user_id="bob",
        content="Prefers Python examples",
        kind=MemoryKind.PREFERENCE,
    )

    alice_hits = store.recall(user_id="alice", query="Python", top_k=5)
    bob_hits = store.recall(user_id="bob", query="Python", top_k=5)

    assert not first.deduplicated
    assert duplicate.deduplicated
    assert duplicate.record.memory_id == first.record.memory_id
    assert len(alice_hits) == len(bob_hits) == 1
    assert store.forget(user_id="bob", memory_id=first.record.memory_id) is False
    assert store.forget(user_id="alice", memory_id=first.record.memory_id) is True
    assert store.recall(user_id="alice", query="Python") == []


def test_memory_recall_combines_similarity_and_importance() -> None:
    """Importance should break semantic ties without replacing relevance."""
    store = QdrantMemoryStore(
        QdrantClient(location=":memory:"),
        MemoryEmbedder(),
        importance_weight=0.2,
    )
    store.remember(user_id="u", content="Python preference", importance=0.9)
    store.remember(user_id="u", content="Python fact", importance=0.2)

    hits = store.recall(user_id="u", query="Python", top_k=2)

    assert hits[0].record.content == "Python preference"
    assert hits[0].similarity == pytest.approx(hits[1].similarity)
    assert hits[0].score > hits[1].score


class RecordingSummarizer:
    """Bounded summarizer fixture that records dropped messages."""

    def __init__(self) -> None:
        """Initialize captured messages."""
        self.seen: list[ContextMessage] = []

    def summarize(self, messages: Sequence[ContextMessage], max_chars: int) -> str:
        """Join dropped content and honor the requested budget."""
        self.seen = list(messages)
        return " | ".join(message.content for message in messages)[:max_chars]


def test_context_compression_preserves_pinned_and_recent_messages() -> None:
    """System constraints and recent turns should survive compression."""
    messages = [
        ContextMessage(role="system", content="Never invent citations.", pinned=True),
        ContextMessage(role="user", content="old question " * 5),
        ContextMessage(role="assistant", content="old answer " * 5),
        ContextMessage(role="tool", content="[E1] retained evidence", pinned=True),
        ContextMessage(role="user", content="latest question"),
        ContextMessage(role="assistant", content="latest answer"),
    ]
    summarizer = RecordingSummarizer()

    result = compress_context(
        messages,
        ContextBudget(max_chars=100, summary_max_chars=20, preserve_recent_messages=2),
        summarizer,
    )

    assert result.dropped_messages == 2
    assert [message.role for message in result.retained_messages] == [
        "system",
        "tool",
        "user",
        "assistant",
    ]
    assert len(result.summary) <= 20
    assert result.final_chars <= 100
    assert result.estimated_tokens == (result.final_chars + 3) // 4
    assert len(summarizer.seen) == 2


def test_context_budget_rejects_uncompressible_pinned_content() -> None:
    """Compression should fail visibly instead of truncating hard constraints."""
    messages = [ContextMessage(role="system", content="x" * 101, pinned=True)]

    with pytest.raises(ValueError, match="exceed"):
        compress_context(
            messages,
            ContextBudget(max_chars=100, summary_max_chars=20),
            RecordingSummarizer(),
        )
