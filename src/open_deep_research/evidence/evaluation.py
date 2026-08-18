"""Deterministic retrieval metrics for local regression evaluation."""

from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, Field

from open_deep_research.evidence.citations import audit_citations, citation_id
from open_deep_research.evidence.models import RetrievalHit


class Retriever(Protocol):
    """Retrieval interface required by the local evaluation harness."""

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievalHit]:
        """Return ranked hits for a query."""


class RetrievalExample(BaseModel):
    """One query and the chunk identifiers considered relevant."""

    query: str = Field(min_length=1)
    relevant_chunk_ids: set[str] = Field(min_length=1)


class RetrievalMetrics(BaseModel):
    """Aggregate retrieval quality metrics."""

    recall_at_k: float = Field(ge=0, le=1)
    mean_reciprocal_rank: float = Field(ge=0, le=1)
    evaluated_queries: int = Field(ge=0)
    top_k: int = Field(gt=0)


class CitationMetrics(BaseModel):
    """Deterministic source-reference and required-evidence metrics."""

    reference_validity: float = Field(ge=0, le=1)
    evidence_coverage: float = Field(ge=0, le=1)
    cited_references: int = Field(ge=0)
    valid_references: int = Field(ge=0)
    required_evidence: int = Field(ge=0)
    covered_evidence: int = Field(ge=0)
    unknown_ids: list[str]


def evaluate_retriever(
    retriever: Retriever,
    examples: Sequence[RetrievalExample],
    *,
    top_k: int = 5,
) -> RetrievalMetrics:
    """Calculate macro Recall@K and mean reciprocal rank."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not examples:
        return RetrievalMetrics(
            recall_at_k=0,
            mean_reciprocal_rank=0,
            evaluated_queries=0,
            top_k=top_k,
        )

    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    for example in examples:
        hits = retriever.retrieve(example.query, top_k=top_k)
        retrieved_ids = [hit.chunk.chunk_id for hit in hits]
        relevant_retrieved = example.relevant_chunk_ids.intersection(retrieved_ids)
        recalls.append(len(relevant_retrieved) / len(example.relevant_chunk_ids))
        reciprocal_ranks.append(
            next(
                (
                    1 / rank
                    for rank, chunk_id in enumerate(retrieved_ids, start=1)
                    if chunk_id in example.relevant_chunk_ids
                ),
                0.0,
            )
        )

    return RetrievalMetrics(
        recall_at_k=sum(recalls) / len(recalls),
        mean_reciprocal_rank=sum(reciprocal_ranks) / len(reciprocal_ranks),
        evaluated_queries=len(examples),
        top_k=top_k,
    )


def evaluate_citation_references(
    answer: str,
    hits: Sequence[RetrievalHit],
    *,
    required_chunk_ids: set[str],
) -> CitationMetrics:
    """Measure traceable references and coverage of expected evidence chunks.

    This validates that citation identifiers point to supplied evidence. It does
    not determine whether a cited source semantically entails the surrounding
    claim; that requires a labelled review or a separate judge.
    """
    audit = audit_citations(answer, list(hits))
    cited_ids = set(audit.cited_ids)
    unknown_ids = set(audit.unknown_ids)
    valid_ids = cited_ids - unknown_ids
    required_ids = {citation_id(chunk_id) for chunk_id in required_chunk_ids}
    covered_ids = valid_ids.intersection(required_ids)

    return CitationMetrics(
        reference_validity=len(valid_ids) / len(cited_ids) if cited_ids else 0,
        evidence_coverage=(
            len(covered_ids) / len(required_ids) if required_ids else 1
        ),
        cited_references=len(cited_ids),
        valid_references=len(valid_ids),
        required_evidence=len(required_ids),
        covered_evidence=len(covered_ids),
        unknown_ids=sorted(unknown_ids),
    )
