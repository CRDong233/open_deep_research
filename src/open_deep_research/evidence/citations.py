"""Formatting and validation helpers for evidence-grounded answers."""

import re

from pydantic import BaseModel

from open_deep_research.evidence.models import RetrievalHit

_CITATION_PATTERN = re.compile(r"\[E(\d+)]")


class CitationAudit(BaseModel):
    """Summary of citation references found in generated text."""

    cited_ids: list[str]
    unknown_ids: list[str]
    available_ids: list[str]

    @property
    def valid(self) -> bool:
        """Return whether every cited identifier was supplied as evidence."""
        return not self.unknown_ids


def format_evidence_context(hits: list[RetrievalHit]) -> str:
    """Format ranked hits as a prompt-ready context with stable citations."""
    blocks = []
    for hit in hits:
        citation_id = f"E{hit.rank}"
        section = f" | section: {hit.chunk.section}" if hit.chunk.section else ""
        blocks.append(
            f"[{citation_id}] {hit.chunk.title}{section}\n"
            f"source: {hit.chunk.citation_uri}\n"
            f"score: {hit.score:.4f}\n"
            f"{hit.chunk.content}"
        )
    return "\n\n".join(blocks)


def audit_citations(answer: str, hits: list[RetrievalHit]) -> CitationAudit:
    """Detect citations that do not correspond to supplied evidence."""
    cited_ids = list(
        dict.fromkeys(f"E{value}" for value in _CITATION_PATTERN.findall(answer))
    )
    available_ids = [f"E{hit.rank}" for hit in hits]
    available = set(available_ids)
    return CitationAudit(
        cited_ids=cited_ids,
        unknown_ids=[
            citation_id for citation_id in cited_ids if citation_id not in available
        ],
        available_ids=available_ids,
    )
