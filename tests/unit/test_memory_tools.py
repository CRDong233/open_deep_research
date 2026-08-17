"""Unit tests for explicit user-scoped memory tools."""

import json
from collections.abc import Sequence

from qdrant_client import QdrantClient

from open_deep_research.memory import QdrantMemoryStore
from open_deep_research.memory_tools import create_memory_tools


class ToolEmbedder:
    """Deterministic vectors for memory tool tests."""

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Embed documents using the query implementation."""
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> Sequence[float]:
        """Return one stable vector for compact tool tests."""
        return [1.0, 0.0]


def test_memory_tools_write_recall_deduplicate_and_forget() -> None:
    """Agent tools should expose the complete explicit memory lifecycle."""
    store = QdrantMemoryStore(QdrantClient(location=":memory:"), ToolEmbedder())
    tools = {
        memory_tool.name: memory_tool
        for memory_tool in create_memory_tools(store, user_id="alice", top_k=3)
    }

    first = json.loads(
        tools["memory_write"].invoke(
            {"content": "Prefers concise Python examples", "kind": "preference"}
        )
    )
    duplicate = json.loads(
        tools["memory_write"].invoke(
            {"content": "  PREFERS concise python examples  ", "kind": "preference"}
        )
    )
    recalled = json.loads(tools["memory_recall"].invoke({"query": "Python preference"}))
    forgotten = json.loads(
        tools["memory_forget"].invoke({"memory_id": first["memory_id"]})
    )

    assert first["deduplicated"] is False
    assert duplicate["deduplicated"] is True
    assert first["memory_id"] == duplicate["memory_id"]
    assert recalled["memories"][0]["kind"] == "preference"
    assert forgotten["deleted"] is True
    assert json.loads(tools["memory_recall"].invoke({"query": "Python"})) == {
        "memories": []
    }


def test_memory_tools_cannot_delete_another_users_memory() -> None:
    """The tool closure must enforce the configured user namespace."""
    store = QdrantMemoryStore(QdrantClient(location=":memory:"), ToolEmbedder())
    alice_tools = {
        item.name: item for item in create_memory_tools(store, user_id="alice")
    }
    bob_tools = {item.name: item for item in create_memory_tools(store, user_id="bob")}
    written = json.loads(
        alice_tools["memory_write"].invoke({"content": "Alice preference"})
    )

    result = json.loads(
        bob_tools["memory_forget"].invoke({"memory_id": written["memory_id"]})
    )

    assert result["deleted"] is False
    assert json.loads(alice_tools["memory_recall"].invoke({"query": "Alice"}))[
        "memories"
    ]
