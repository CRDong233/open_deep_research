"""Explicitly enabled Agent tools for user-scoped long-term memory."""

import asyncio
import json
import threading
from dataclasses import dataclass

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool

from open_deep_research.configuration import Configuration
from open_deep_research.evidence import FastEmbedTextEmbedder
from open_deep_research.knowledge import create_qdrant_client
from open_deep_research.memory import MemoryKind, QdrantMemoryStore


@dataclass(frozen=True, slots=True)
class MemorySettings:
    """Hashable settings for one user-scoped memory tool set."""

    enabled: bool = False
    user_id: str | None = None
    qdrant_location: str = ".qdrant"
    collection_name: str = "agent_memories"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    top_k: int = 5

    def __post_init__(self) -> None:
        """Require a user namespace whenever memory is enabled."""
        if self.enabled and (self.user_id is None or not self.user_id.strip()):
            raise ValueError("memory_user_id is required when memory is enabled")
        if self.top_k <= 0:
            raise ValueError("memory_top_k must be positive")

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig) -> "MemorySettings":
        """Read memory settings from the shared Agent configuration."""
        configured = Configuration.from_runnable_config(config)
        return cls(
            enabled=configured.enable_long_term_memory,
            user_id=configured.memory_user_id,
            qdrant_location=configured.memory_qdrant_location,
            collection_name=configured.memory_collection,
            embedding_model=configured.embedding_model,
            top_k=configured.memory_top_k,
        )


def build_memory_store(settings: MemorySettings) -> QdrantMemoryStore:
    """Build a Qdrant memory store without writing any memory."""
    if not settings.enabled:
        raise ValueError("long-term memory is disabled")
    return QdrantMemoryStore(
        create_qdrant_client(settings.qdrant_location),
        FastEmbedTextEmbedder(settings.embedding_model),
        collection_name=settings.collection_name,
    )


def create_memory_tools(
    store: QdrantMemoryStore,
    *,
    user_id: str,
    top_k: int = 5,
) -> list[BaseTool]:
    """Create recall, explicit-write, and explicit-forget tools for one user."""
    if not user_id.strip():
        raise ValueError("user_id must not be empty")

    @tool("memory_recall")
    def memory_recall(query: str) -> str:
        """Recall relevant user preferences or facts from long-term memory."""
        hits = store.recall(user_id=user_id, query=query, top_k=top_k)
        return json.dumps(
            {
                "memories": [
                    {
                        "memory_id": hit.record.memory_id,
                        "kind": hit.record.kind.value,
                        "content": hit.record.content,
                        "score": round(hit.score, 4),
                    }
                    for hit in hits
                ]
            },
            ensure_ascii=False,
        )

    @tool("memory_write")
    def memory_write(
        content: str,
        kind: MemoryKind = MemoryKind.FACT,
        importance: float = 0.5,
    ) -> str:
        """Store non-sensitive information only when the user explicitly asks."""
        result = store.remember(
            user_id=user_id,
            content=content,
            kind=kind,
            importance=importance,
        )
        return json.dumps(
            {
                "memory_id": result.record.memory_id,
                "deduplicated": result.deduplicated,
            },
            ensure_ascii=False,
        )

    @tool("memory_forget")
    def memory_forget(memory_id: str) -> str:
        """Delete one memory only when the user explicitly requests it."""
        deleted = store.forget(user_id=user_id, memory_id=memory_id)
        return json.dumps(
            {"memory_id": memory_id, "deleted": deleted},
            ensure_ascii=False,
        )

    return [memory_recall, memory_write, memory_forget]


_STORE_CACHE: dict[MemorySettings, QdrantMemoryStore] = {}
_STORE_LOCK = threading.Lock()


def _get_or_build_store(settings: MemorySettings) -> QdrantMemoryStore:
    """Build one memory store per process and configuration snapshot."""
    with _STORE_LOCK:
        store = _STORE_CACHE.get(settings)
        if store is None:
            store = build_memory_store(settings)
            _STORE_CACHE[settings] = store
        return store


async def get_memory_tools(config: RunnableConfig) -> list[BaseTool]:
    """Return cached memory tools only when explicitly enabled."""
    settings = MemorySettings.from_runnable_config(config)
    if not settings.enabled:
        return []
    store = await asyncio.to_thread(_get_or_build_store, settings)
    assert settings.user_id is not None
    return create_memory_tools(store, user_id=settings.user_id, top_k=settings.top_k)
