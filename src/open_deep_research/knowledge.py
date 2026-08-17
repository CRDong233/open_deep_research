"""Configurable local knowledge service exposed as an Agent tool."""

import asyncio
import threading
from dataclasses import dataclass
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from qdrant_client import QdrantClient

from open_deep_research.configuration import Configuration
from open_deep_research.evidence import (
    DEFAULT_EMBEDDING_MODEL,
    DirectoryDocumentLoader,
    Embedder,
    FastEmbedTextEmbedder,
    IngestionResult,
    QdrantHybridRetriever,
    format_evidence_context,
    ingest_directory,
)


@dataclass(frozen=True, slots=True)
class KnowledgeSettings:
    """Hashable settings for one cached knowledge service."""

    enabled: bool = False
    source_path: str | None = None
    qdrant_location: str = ".qdrant"
    collection_name: str = "evidence_chunks"
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    top_k: int = 5
    max_file_bytes: int = 2_000_000
    rebuild: bool = False

    def __post_init__(self) -> None:
        """Validate user-controlled knowledge settings."""
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be positive")
        if not self.collection_name.strip():
            raise ValueError("collection_name must not be empty")

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig) -> "KnowledgeSettings":
        """Read knowledge settings from the existing Agent configuration."""
        configured = Configuration.from_runnable_config(config)
        return cls(
            enabled=configured.enable_knowledge_search,
            source_path=configured.knowledge_base_path,
            qdrant_location=configured.qdrant_location,
            collection_name=configured.qdrant_collection,
            embedding_model=configured.embedding_model,
            top_k=configured.knowledge_top_k,
            max_file_bytes=configured.knowledge_max_file_bytes,
            rebuild=configured.rebuild_knowledge_index,
        )


@dataclass(slots=True)
class KnowledgeSearchService:
    """Search facade with ingestion metadata for observability."""

    retriever: QdrantHybridRetriever
    top_k: int
    ingestion: IngestionResult | None = None

    @property
    def indexed_chunks(self) -> int:
        """Return the active evidence chunk count."""
        return self.retriever.indexed_count

    def search(self, query: str) -> str:
        """Return prompt-ready evidence with stable citation identifiers."""
        hits = self.retriever.retrieve(query, top_k=self.top_k)
        if not hits:
            return '{"ok": false, "reason": "no_evidence"}'
        return format_evidence_context(hits)


def build_knowledge_service(
    settings: KnowledgeSettings,
    *,
    client: QdrantClient | None = None,
    embedder: Embedder | None = None,
) -> KnowledgeSearchService:
    """Build or restore a knowledge service from explicit settings."""
    if not settings.enabled:
        raise ValueError("knowledge search is disabled")
    active_client = client or create_qdrant_client(settings.qdrant_location)
    active_embedder = embedder or FastEmbedTextEmbedder(settings.embedding_model)
    retriever = QdrantHybridRetriever(
        active_client,
        active_embedder,
        collection_name=settings.collection_name,
        replace_snapshot=settings.rebuild,
    )

    collection_exists = active_client.collection_exists(settings.collection_name)
    if collection_exists and not settings.rebuild:
        retriever.load_snapshot()
        return KnowledgeSearchService(retriever=retriever, top_k=settings.top_k)

    if not settings.source_path:
        raise ValueError(
            "knowledge_base_path is required when no reusable collection exists"
        )
    loader = DirectoryDocumentLoader(
        settings.source_path,
        max_file_bytes=settings.max_file_bytes,
    )
    ingestion = ingest_directory(loader)
    if not ingestion.chunks:
        raise ValueError("knowledge source produced no evidence chunks")
    retriever.index(ingestion.chunks)
    return KnowledgeSearchService(
        retriever=retriever,
        top_k=settings.top_k,
        ingestion=ingestion,
    )


def create_qdrant_client(location: str) -> QdrantClient:
    """Create a local-path, in-memory, or URL-backed Qdrant client."""
    normalized = location.strip()
    if not normalized:
        raise ValueError("qdrant_location must not be empty")
    if normalized == ":memory:":
        return QdrantClient(location=normalized)
    if normalized.startswith(("http://", "https://")):
        return QdrantClient(url=normalized)
    path = Path(normalized).expanduser().resolve()
    return QdrantClient(path=str(path))


def create_knowledge_search_tool(service: KnowledgeSearchService) -> BaseTool:
    """Expose a configured service as a LangChain tool."""

    @tool("knowledge_search")
    def knowledge_search(query: str) -> str:
        """Search the local knowledge base and return source-linked evidence."""
        return service.search(query)

    return knowledge_search


_SERVICE_CACHE: dict[KnowledgeSettings, KnowledgeSearchService] = {}
_SERVICE_LOCK = threading.Lock()


def _get_or_build_service(settings: KnowledgeSettings) -> KnowledgeSearchService:
    """Build a service once per process and configuration snapshot."""
    with _SERVICE_LOCK:
        service = _SERVICE_CACHE.get(settings)
        if service is None:
            service = build_knowledge_service(settings)
            _SERVICE_CACHE[settings] = service
        return service


async def get_knowledge_search_tool(config: RunnableConfig) -> BaseTool | None:
    """Return a cached local knowledge tool when explicitly enabled."""
    settings = KnowledgeSettings.from_runnable_config(config)
    if not settings.enabled:
        return None
    service = await asyncio.to_thread(_get_or_build_service, settings)
    return create_knowledge_search_tool(service)
