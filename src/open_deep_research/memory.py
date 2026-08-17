"""User-scoped long-term memory backed by Qdrant vectors."""

import hashlib
import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models

from open_deep_research.evidence import Embedder

_WHITESPACE = re.compile(r"\s+")


class MemoryKind(str, Enum):
    """Supported long-term memory categories."""

    PREFERENCE = "preference"
    FACT = "fact"
    EPISODE = "episode"


class MemoryRecord(BaseModel):
    """One durable user memory with lifecycle metadata."""

    memory_id: str
    user_id: str = Field(min_length=1)
    kind: MemoryKind
    content: str = Field(min_length=1)
    importance: float = Field(default=0.5, ge=0, le=1)
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryWriteResult(BaseModel):
    """Result of a deduplicating memory write."""

    record: MemoryRecord
    deduplicated: bool


class MemoryHit(BaseModel):
    """A recalled memory with semantic and fused scores."""

    record: MemoryRecord
    similarity: float
    score: float


class QdrantMemoryStore:
    """Persist and retrieve user-scoped memories in a dedicated collection."""

    def __init__(
        self,
        client: QdrantClient,
        embedder: Embedder,
        *,
        collection_name: str = "agent_memories",
        importance_weight: float = 0.15,
    ) -> None:
        """Create a memory store with a bounded importance contribution."""
        if not collection_name.strip():
            raise ValueError("collection_name must not be empty")
        if not 0 <= importance_weight <= 1:
            raise ValueError("importance_weight must be between 0 and 1")
        self.client = client
        self.embedder = embedder
        self.collection_name = collection_name
        self.importance_weight = importance_weight

    def remember(
        self,
        *,
        user_id: str,
        content: str,
        kind: MemoryKind = MemoryKind.FACT,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryWriteResult:
        """Insert or update a normalized memory without creating duplicates."""
        normalized = normalize_memory_content(content)
        if not user_id.strip():
            raise ValueError("user_id must not be empty")
        if not normalized:
            raise ValueError("content must not be empty")
        if not 0 <= importance <= 1:
            raise ValueError("importance must be between 0 and 1")

        memory_id = build_memory_id(user_id, kind, normalized)
        point_id = self._point_id(memory_id)
        existing = self._retrieve_record(point_id)
        now = datetime.now(timezone.utc)
        record = MemoryRecord(
            memory_id=memory_id,
            user_id=user_id,
            kind=kind,
            content=content.strip(),
            importance=importance,
            created_at=existing.created_at if existing else now,
            updated_at=now,
            metadata=metadata or {},
        )
        vector = list(self.embedder.embed_query(record.content))
        if not vector:
            raise ValueError("embedder returned an empty memory vector")
        self._ensure_collection(len(vector))
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "user_id": user_id,
                        "kind": kind.value,
                        "record": record.model_dump(mode="json"),
                    },
                )
            ],
            wait=True,
        )
        return MemoryWriteResult(record=record, deduplicated=existing is not None)

    def recall(self, *, user_id: str, query: str, top_k: int = 5) -> list[MemoryHit]:
        """Recall memories from one user namespace only."""
        if not user_id.strip() or not query.strip():
            return []
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not self.client.collection_exists(self.collection_name):
            return []

        query_vector = list(self.embedder.embed_query(query))
        points = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id),
                    )
                ]
            ),
            limit=top_k,
            with_payload=True,
        ).points
        hits: list[MemoryHit] = []
        for point in points:
            payload = point.payload or {}
            record_payload = payload.get("record")
            if not isinstance(record_payload, dict):
                continue
            record = MemoryRecord.model_validate(record_payload)
            score = (
                1 - self.importance_weight
            ) * point.score + self.importance_weight * record.importance
            hits.append(MemoryHit(record=record, similarity=point.score, score=score))
        hits.sort(key=lambda hit: (-hit.score, hit.record.memory_id))
        return hits

    def forget(self, *, user_id: str, memory_id: str) -> bool:
        """Delete a memory only when it belongs to the requesting user."""
        point_id = self._point_id(memory_id)
        existing = self._retrieve_record(point_id)
        if existing is None or existing.user_id != user_id:
            return False
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=[point_id],
            wait=True,
        )
        return True

    def _ensure_collection(self, vector_size: int) -> None:
        """Create the dedicated memory collection on first write."""
        if self.client.collection_exists(self.collection_name):
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def _retrieve_record(self, point_id: str) -> MemoryRecord | None:
        """Read one memory payload when its collection exists."""
        if not self.client.collection_exists(self.collection_name):
            return None
        records = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id],
            with_payload=True,
        )
        if not records:
            return None
        payload = records[0].payload or {}
        record_payload = payload.get("record")
        if not isinstance(record_payload, dict):
            return None
        return MemoryRecord.model_validate(record_payload)

    @staticmethod
    def _point_id(memory_id: str) -> str:
        """Map a readable memory identifier to a Qdrant UUID."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, memory_id))


def normalize_memory_content(content: str) -> str:
    """Normalize content used for deterministic duplicate detection."""
    return _WHITESPACE.sub(" ", content).strip().casefold()


def build_memory_id(user_id: str, kind: MemoryKind, normalized_content: str) -> str:
    """Build a stable identifier scoped by user and memory kind."""
    payload = f"{user_id}:{kind.value}:{normalized_content}".encode()
    return hashlib.sha256(payload).hexdigest()[:24]
