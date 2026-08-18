"""Deterministic retrieval metrics for local regression evaluation."""

from collections.abc import Sequence
from math import ceil
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


class EvaluationRun(BaseModel):
    """Versioned result record for one reproducible Agent evaluation run."""

    schema_version: int = Field(default=1, ge=1)
    case_id: str = Field(min_length=1)
    succeeded: bool
    duration_ms: float = Field(ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    citation_metrics: CitationMetrics | None = None
    failure_kind: str | None = None


class EvaluationSummary(BaseModel):
    """Aggregate only values actually recorded by evaluation runs."""

    evaluated_runs: int = Field(ge=0)
    successful_runs: int = Field(ge=0)
    task_success_rate: float = Field(ge=0, le=1)
    average_duration_ms: float | None = Field(default=None, ge=0)
    p95_duration_ms: float | None = Field(default=None, ge=0)
    average_total_tokens: float | None = Field(default=None, ge=0)
    average_reference_validity: float | None = Field(default=None, ge=0, le=1)
    average_evidence_coverage: float | None = Field(default=None, ge=0, le=1)
    failure_kinds: dict[str, int]


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


def summarize_evaluation_runs(
    runs: Sequence[EvaluationRun],
) -> EvaluationSummary:
    """Aggregate versioned run records without inventing absent measurements."""
    if not runs:
        return EvaluationSummary(
            evaluated_runs=0,
            successful_runs=0,
            task_success_rate=0,
            failure_kinds={},
        )

    durations = sorted(run.duration_ms for run in runs)
    successful_runs = sum(run.succeeded for run in runs)
    token_values = [run.total_tokens for run in runs if run.total_tokens is not None]
    citation_values = [
        run.citation_metrics for run in runs if run.citation_metrics is not None
    ]
    failure_kinds: dict[str, int] = {}
    for run in runs:
        if run.succeeded or not run.failure_kind:
            continue
        failure_kinds[run.failure_kind] = failure_kinds.get(run.failure_kind, 0) + 1

    return EvaluationSummary(
        evaluated_runs=len(runs),
        successful_runs=successful_runs,
        task_success_rate=successful_runs / len(runs),
        average_duration_ms=sum(durations) / len(durations),
        p95_duration_ms=durations[ceil(0.95 * len(durations)) - 1],
        average_total_tokens=(
            sum(token_values) / len(token_values) if token_values else None
        ),
        average_reference_validity=(
            sum(metric.reference_validity for metric in citation_values)
            / len(citation_values)
            if citation_values
            else None
        ),
        average_evidence_coverage=(
            sum(metric.evidence_coverage for metric in citation_values)
            / len(citation_values)
            if citation_values
            else None
        ),
        failure_kinds=failure_kinds,
    )
