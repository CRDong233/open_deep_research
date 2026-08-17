"""Validated data models for traceable retrieval evidence."""

from typing import Any

from pydantic import BaseModel, Field, model_validator


class EvidenceDocument(BaseModel):
    """A source document that can be split into retrievable evidence chunks."""

    document_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    title: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceChunk(BaseModel):
    """A source-preserving document segment with stable character offsets."""

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    title: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    section: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_offsets(self) -> "EvidenceChunk":
        """Ensure each chunk has a non-empty source range."""
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")
        return self

    @property
    def citation_uri(self) -> str:
        """Return a stable URI that identifies the source character range."""
        return f"{self.uri}#char={self.start_char},{self.end_char}"


class RetrievalHit(BaseModel):
    """A ranked evidence chunk with inspectable component scores."""

    chunk: EvidenceChunk
    rank: int = Field(ge=1)
    score: float
    lexical_score: float
    semantic_score: float | None = None
