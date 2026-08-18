"""Offline-capable hybrid retrieval with inspectable lexical scores."""

import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

from open_deep_research.evidence.models import EvidenceChunk, RetrievalHit

_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]", re.IGNORECASE)


class Embedder(Protocol):
    """Minimal embedding contract used by the hybrid retriever."""

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Embed document texts in input order."""

    def embed_query(self, text: str) -> Sequence[float]:
        """Embed one query using the same vector space."""


class Reranker(Protocol):
    """Optional post-retrieval ranking contract."""

    def rerank(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
        *,
        top_k: int,
    ) -> Sequence[RetrievalHit]:
        """Return a reordered subset without changing evidence payloads."""


def apply_reranker(
    reranker: Reranker | None,
    query: str,
    hits: Sequence[RetrievalHit],
    *,
    top_k: int,
) -> list[RetrievalHit]:
    """Apply an optional reranker and normalize ranks deterministically."""
    selected = list(hits)
    if reranker is not None:
        selected = list(reranker.rerank(query, selected, top_k=top_k))
    selected = selected[:top_k]
    return [hit.model_copy(update={"rank": rank}) for rank, hit in enumerate(selected, 1)]


def tokenize(text: str) -> list[str]:
    """Tokenize Latin words and Chinese unigrams/bigrams for lexical search."""
    raw_tokens = [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text)]
    tokens: list[str] = []
    chinese_run: list[str] = []

    def flush_chinese() -> None:
        if not chinese_run:
            return
        tokens.extend(chinese_run)
        tokens.extend(
            chinese_run[index] + chinese_run[index + 1]
            for index in range(len(chinese_run) - 1)
        )
        chinese_run.clear()

    for token in raw_tokens:
        if "\u4e00" <= token <= "\u9fff":
            chinese_run.append(token)
        else:
            flush_chinese()
            tokens.append(token)
    flush_chinese()
    return tokens


class InMemoryHybridRetriever:
    """Rank chunks with BM25 and an optional semantic embedder."""

    def __init__(
        self,
        *,
        embedder: Embedder | None = None,
        semantic_weight: float = 0.6,
        k1: float = 1.5,
        b: float = 0.75,
        reranker: Reranker | None = None,
    ) -> None:
        """Create an empty retriever with validated ranking parameters."""
        if not 0 <= semantic_weight <= 1:
            raise ValueError("semantic_weight must be between 0 and 1")
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("invalid BM25 parameters")
        self.embedder = embedder
        self.semantic_weight = semantic_weight
        self.k1 = k1
        self.b = b
        self.reranker = reranker
        self._chunks: list[EvidenceChunk] = []
        self._term_frequencies: list[Counter[str]] = []
        self._document_frequencies: Counter[str] = Counter()
        self._document_lengths: list[int] = []
        self._vectors: list[list[float]] | None = None

    def index(self, chunks: Sequence[EvidenceChunk]) -> None:
        """Replace the current index with a deterministic chunk snapshot."""
        self._chunks = list(chunks)
        tokenized = [tokenize(chunk.content) for chunk in self._chunks]
        self._term_frequencies = [Counter(tokens) for tokens in tokenized]
        self._document_lengths = [len(tokens) for tokens in tokenized]
        self._document_frequencies = Counter(
            token for tokens in tokenized for token in set(tokens)
        )
        self._vectors = None

        if self.embedder and self._chunks:
            vectors = self.embedder.embed_documents(
                [chunk.content for chunk in self._chunks]
            )
            self._vectors = [list(vector) for vector in vectors]
            if len(self._vectors) != len(self._chunks):
                raise ValueError("embedder returned the wrong number of vectors")
            self._validate_vector_dimensions(self._vectors)

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievalHit]:
        """Return the highest-ranked chunks with score components."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not self._chunks or not query.strip():
            return []

        lexical_scores = self._bm25_scores(tokenize(query))
        semantic_scores = self._semantic_scores(query)
        normalized_lexical = self._normalize(lexical_scores)
        normalized_semantic = (
            self._normalize(semantic_scores) if semantic_scores is not None else None
        )
        lexical_weight = 1 - self.semantic_weight if semantic_scores else 1.0

        scored: list[tuple[int, float]] = []
        for index, lexical_score in enumerate(normalized_lexical):
            semantic_score = (
                normalized_semantic[index] if normalized_semantic is not None else 0.0
            )
            combined = lexical_weight * lexical_score
            if normalized_semantic is not None:
                combined += self.semantic_weight * semantic_score
            scored.append((index, combined))

        scored.sort(key=lambda item: (-item[1], self._chunks[item[0]].chunk_id))
        hits: list[RetrievalHit] = []
        for rank, (index, score) in enumerate(scored[:top_k], start=1):
            hits.append(
                RetrievalHit(
                    chunk=self._chunks[index],
                    rank=rank,
                    score=score,
                    lexical_score=lexical_scores[index],
                    semantic_score=(
                        semantic_scores[index] if semantic_scores is not None else None
                    ),
                )
            )
        return apply_reranker(self.reranker, query, hits, top_k=top_k)

    def _bm25_scores(self, query_tokens: Sequence[str]) -> list[float]:
        """Calculate BM25 scores for the indexed snapshot."""
        document_count = len(self._chunks)
        average_length = sum(self._document_lengths) / document_count
        scores: list[float] = []

        for frequencies, document_length in zip(
            self._term_frequencies,
            self._document_lengths,
        ):
            score = 0.0
            for token in query_tokens:
                frequency = frequencies[token]
                if not frequency:
                    continue
                document_frequency = self._document_frequencies[token]
                inverse_document_frequency = math.log(
                    1
                    + (document_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * document_length / max(average_length, 1)
                )
                score += inverse_document_frequency * (
                    frequency * (self.k1 + 1) / denominator
                )
            scores.append(score)
        return scores

    def _semantic_scores(self, query: str) -> list[float] | None:
        """Calculate cosine similarity when an embedder is configured."""
        if not self.embedder or self._vectors is None:
            return None
        query_vector = list(self.embedder.embed_query(query))
        self._validate_vector_dimensions([*self._vectors, query_vector])
        return [self._cosine(query_vector, vector) for vector in self._vectors]

    @staticmethod
    def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
        """Return cosine similarity, treating zero vectors as unrelated."""
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if not left_norm or not right_norm:
            return 0.0
        return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)

    @staticmethod
    def _normalize(scores: Sequence[float]) -> list[float]:
        """Min-max normalize scores while retaining positive ties."""
        if not scores:
            return []
        minimum = min(scores)
        maximum = max(scores)
        if math.isclose(minimum, maximum):
            return [1.0 if maximum > 0 else 0.0 for _ in scores]
        return [(score - minimum) / (maximum - minimum) for score in scores]

    @staticmethod
    def _validate_vector_dimensions(vectors: Sequence[Sequence[float]]) -> None:
        """Reject empty or inconsistent embedding vectors."""
        dimensions = {len(vector) for vector in vectors}
        if not dimensions or 0 in dimensions or len(dimensions) != 1:
            raise ValueError("embedding vectors must share one non-zero dimension")
