"""Qdrant persistence and FastEmbed adapters for traceable evidence."""

import math
import uuid
from collections.abc import Sequence

from fastembed import TextEmbedding
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models

from open_deep_research.evidence.models import EvidenceChunk, RetrievalHit
from open_deep_research.evidence.retrieval import (
    Embedder,
    InMemoryHybridRetriever,
    Reranker,
    apply_reranker,
)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class SnapshotSyncResult(BaseModel):
    """Counts returned by an incremental evidence snapshot update."""

    added_chunks: int = Field(ge=0)
    updated_chunks: int = Field(ge=0)
    removed_chunks: int = Field(ge=0)
    unchanged_chunks: int = Field(ge=0)


class FastEmbedTextEmbedder:
    """Lazy multilingual FastEmbed implementation of the Embedder contract."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        cache_dir: str | None = None,
        threads: int | None = None,
    ) -> None:
        """Store model settings without downloading weights eagerly."""
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.threads = threads
        self._model: TextEmbedding | None = None

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed documents with the configured multilingual model."""
        return [vector.tolist() for vector in self._get_model().embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        """Embed a query using the model's query-specific path."""
        vectors = self._get_model().query_embed(text)
        return next(iter(vectors)).tolist()

    def _get_model(self) -> TextEmbedding:
        """Initialize the model only when an embedding is requested."""
        if self._model is None:
            self._model = TextEmbedding(
                model_name=self.model_name,
                cache_dir=self.cache_dir,
                threads=self.threads,
                lazy_load=True,
            )
        return self._model


class QdrantHybridRetriever:
    """Combine Qdrant dense retrieval with local BM25 candidate ranking."""

    def __init__(
        self,
        client: QdrantClient,
        embedder: Embedder,
        *,
        collection_name: str = "evidence_chunks",
        semantic_weight: float = 0.6,
        candidate_multiplier: int = 4,
        replace_snapshot: bool = False,
        reranker: Reranker | None = None,
    ) -> None:
        """Create a retriever that owns one explicitly named collection."""
        if not collection_name.strip():
            raise ValueError("collection_name must not be empty")
        if not 0 <= semantic_weight <= 1:
            raise ValueError("semantic_weight must be between 0 and 1")
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be at least 1")
        self.client = client
        self.embedder = embedder
        self.collection_name = collection_name
        self.semantic_weight = semantic_weight
        self.candidate_multiplier = candidate_multiplier
        self.replace_snapshot = replace_snapshot
        self.reranker = reranker
        self._lexical = InMemoryHybridRetriever()
        self._chunks: dict[str, EvidenceChunk] = {}

    @property
    def indexed_count(self) -> int:
        """Return the number of chunks in the active local snapshot."""
        return len(self._chunks)

    def index(self, chunks: Sequence[EvidenceChunk]) -> None:
        """Persist a complete evidence snapshot and build its BM25 index."""
        snapshot = list(chunks)
        chunk_ids = [chunk.chunk_id for chunk in snapshot]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("chunk_id values must be unique")

        collection_exists = self.client.collection_exists(self.collection_name)
        if collection_exists and not self.replace_snapshot:
            raise ValueError(
                "collection already exists; set replace_snapshot=True to rebuild it"
            )
        if collection_exists:
            self.client.delete_collection(self.collection_name)

        self._chunks = {chunk.chunk_id: chunk for chunk in snapshot}
        self._lexical.index(snapshot)
        if not snapshot:
            return

        vectors = [
            list(vector)
            for vector in self.embedder.embed_documents(
                [chunk.content for chunk in snapshot]
            )
        ]
        self._validate_vectors(vectors, len(snapshot))
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=len(vectors[0]),
                distance=models.Distance.COSINE,
            ),
        )
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                vector=vector,
                payload={"chunk": chunk.model_dump(mode="json")},
            )
            for chunk, vector in zip(snapshot, vectors)
        ]
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    def sync_snapshot(self, chunks: Sequence[EvidenceChunk]) -> SnapshotSyncResult:
        """Incrementally upsert changed chunks and remove deleted chunks."""
        snapshot = list(chunks)
        chunk_ids = [chunk.chunk_id for chunk in snapshot]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("chunk_id values must be unique")
        if not self.client.collection_exists(self.collection_name):
            self.index(snapshot)
            return SnapshotSyncResult(
                added_chunks=len(snapshot),
                updated_chunks=0,
                removed_chunks=0,
                unchanged_chunks=0,
            )
        if not self._chunks:
            self.load_snapshot()

        incoming = {chunk.chunk_id: chunk for chunk in snapshot}
        existing_ids = set(self._chunks)
        incoming_ids = set(incoming)
        removed_ids = existing_ids - incoming_ids
        changed = [
            chunk
            for chunk_id, chunk in incoming.items()
            if self._chunks.get(chunk_id) != chunk
        ]
        added = sum(chunk_id not in existing_ids for chunk_id in incoming)
        updated = len(changed) - added
        unchanged = len(incoming_ids & existing_ids) - updated
        self.update(changed, removed_chunk_ids=removed_ids)
        return SnapshotSyncResult(
            added_chunks=added,
            updated_chunks=updated,
            removed_chunks=len(removed_ids),
            unchanged_chunks=unchanged,
        )

    def update(
        self,
        chunks: Sequence[EvidenceChunk],
        *,
        removed_chunk_ids: set[str] | None = None,
    ) -> None:
        """Apply a bounded chunk delta without deleting the collection."""
        upserts = list(chunks)
        upsert_ids = [chunk.chunk_id for chunk in upserts]
        if len(set(upsert_ids)) != len(upsert_ids):
            raise ValueError("chunk_id values must be unique")
        if not self.client.collection_exists(self.collection_name):
            raise ValueError("collection must exist before applying a delta")
        if not self._chunks:
            self.load_snapshot()

        removed = set(removed_chunk_ids or ()) & set(self._chunks)
        if removed:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[self._point_id(chunk_id) for chunk_id in removed]
                ),
                wait=True,
            )
            for chunk_id in removed:
                self._chunks.pop(chunk_id, None)

        if upserts:
            vectors = [
                list(vector)
                for vector in self.embedder.embed_documents(
                    [chunk.content for chunk in upserts]
                )
            ]
            self._validate_vectors(vectors, len(upserts))
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=self._point_id(chunk.chunk_id),
                        vector=vector,
                        payload={"chunk": chunk.model_dump(mode="json")},
                    )
                    for chunk, vector in zip(upserts, vectors)
                ],
                wait=True,
            )
            self._chunks.update({chunk.chunk_id: chunk for chunk in upserts})
        self._lexical.index(list(self._chunks.values()))

    def load_snapshot(self) -> int:
        """Restore chunk payloads and BM25 state from an existing collection."""
        if not self.client.collection_exists(self.collection_name):
            raise ValueError(f"collection does not exist: {self.collection_name}")

        chunks: list[EvidenceChunk] = []
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection_name,
                offset=offset,
                limit=256,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                payload = record.payload or {}
                chunk_payload = payload.get("chunk")
                if isinstance(chunk_payload, dict):
                    chunks.append(EvidenceChunk.model_validate(chunk_payload))
            if offset is None:
                break

        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("stored collection contains duplicate chunk_id values")
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self._lexical.index(chunks)
        return len(chunks)

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievalHit]:
        """Fuse dense and lexical candidates into one inspectable ranking."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not query.strip() or not self._chunks:
            return []

        candidate_limit = min(
            len(self._chunks),
            max(top_k, top_k * self.candidate_multiplier),
        )
        lexical_hits = self._lexical.retrieve(query, top_k=candidate_limit)
        lexical_scores = {hit.chunk.chunk_id: hit.score for hit in lexical_hits}
        lexical_raw_scores = {
            hit.chunk.chunk_id: hit.lexical_score for hit in lexical_hits
        }

        query_vector = list(self.embedder.embed_query(query))
        semantic_points = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=candidate_limit,
            with_payload=True,
        ).points
        semantic_raw_scores: dict[str, float] = {}
        for point in semantic_points:
            payload = point.payload or {}
            chunk_payload = payload.get("chunk")
            if isinstance(chunk_payload, dict) and isinstance(
                chunk_payload.get("chunk_id"), str
            ):
                semantic_raw_scores[chunk_payload["chunk_id"]] = point.score

        semantic_scores = self._normalize(semantic_raw_scores)
        candidate_ids = set(lexical_scores) | set(semantic_scores)
        ranked = []
        for chunk_id in candidate_ids:
            lexical_score = lexical_scores.get(chunk_id, 0.0)
            semantic_score = semantic_scores.get(chunk_id, 0.0)
            combined = (
                1 - self.semantic_weight
            ) * lexical_score + self.semantic_weight * semantic_score
            ranked.append((chunk_id, combined))
        ranked.sort(key=lambda item: (-item[1], item[0]))

        hits = [
            RetrievalHit(
                chunk=self._chunks[chunk_id],
                rank=rank,
                score=score,
                lexical_score=lexical_raw_scores.get(chunk_id, 0.0),
                semantic_score=semantic_raw_scores.get(chunk_id),
            )
            for rank, (chunk_id, score) in enumerate(ranked[:top_k], start=1)
        ]
        return apply_reranker(self.reranker, query, hits, top_k=top_k)

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        """Map a stable chunk ID to Qdrant's deterministic point ID."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    @staticmethod
    def _normalize(scores: dict[str, float]) -> dict[str, float]:
        """Normalize a named score map for weighted fusion."""
        if not scores:
            return {}
        minimum = min(scores.values())
        maximum = max(scores.values())
        if math.isclose(minimum, maximum):
            value = 1.0 if maximum > 0 else 0.0
            return dict.fromkeys(scores, value)
        return {
            key: (score - minimum) / (maximum - minimum)
            for key, score in scores.items()
        }

    @staticmethod
    def _validate_vectors(vectors: Sequence[Sequence[float]], expected: int) -> None:
        """Ensure vector count and dimensions match the evidence snapshot."""
        if len(vectors) != expected:
            raise ValueError("embedder returned the wrong number of vectors")
        dimensions = {len(vector) for vector in vectors}
        if not dimensions or 0 in dimensions or len(dimensions) != 1:
            raise ValueError("embedding vectors must share one non-zero dimension")
