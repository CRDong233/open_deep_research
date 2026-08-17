"""Deterministic context selection with pluggable bounded summarization."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, Field


class ContextMessage(BaseModel):
    """A role/content pair with optional retention protection."""

    role: str = Field(min_length=1)
    content: str
    pinned: bool = False


@dataclass(frozen=True, slots=True)
class ContextBudget:
    """Character budget and retention policy for one model invocation."""

    max_chars: int = 24_000
    summary_max_chars: int = 4_000
    preserve_recent_messages: int = 8

    def __post_init__(self) -> None:
        """Reject limits that cannot produce a bounded context."""
        if self.max_chars <= 0 or self.summary_max_chars < 0:
            raise ValueError("context character limits are invalid")
        if self.summary_max_chars > self.max_chars:
            raise ValueError("summary_max_chars must not exceed max_chars")
        if self.preserve_recent_messages < 0:
            raise ValueError("preserve_recent_messages must not be negative")


class ContextSummarizer(Protocol):
    """Contract for summarizing messages removed from the active window."""

    def summarize(self, messages: Sequence[ContextMessage], max_chars: int) -> str:
        """Return a summary no longer than the requested character limit."""


class CompressedContext(BaseModel):
    """Auditable output of a context-budget decision."""

    summary: str = ""
    retained_messages: list[ContextMessage]
    dropped_messages: int = Field(ge=0)
    original_chars: int = Field(ge=0)
    final_chars: int = Field(ge=0)

    @property
    def estimated_tokens(self) -> int:
        """Return a conservative model-agnostic four-characters estimate."""
        return math.ceil(self.final_chars / 4)


def compress_context(
    messages: Sequence[ContextMessage],
    budget: ContextBudget,
    summarizer: ContextSummarizer,
) -> CompressedContext:
    """Preserve pinned/recent messages and summarize the dropped middle."""
    snapshot = list(messages)
    original_chars = sum(len(message.content) for message in snapshot)
    if original_chars <= budget.max_chars:
        return CompressedContext(
            retained_messages=snapshot,
            dropped_messages=0,
            original_chars=original_chars,
            final_chars=original_chars,
        )

    pinned_indexes = {index for index, message in enumerate(snapshot) if message.pinned}
    unpinned_indexes = [
        index for index, message in enumerate(snapshot) if not message.pinned
    ]
    recent_indexes = set(unpinned_indexes[-budget.preserve_recent_messages :])
    retained_indexes = pinned_indexes | recent_indexes
    retained = [
        message for index, message in enumerate(snapshot) if index in retained_indexes
    ]
    dropped = [
        message
        for index, message in enumerate(snapshot)
        if index not in retained_indexes
    ]
    retained_chars = sum(len(message.content) for message in retained)
    if retained_chars > budget.max_chars:
        raise ValueError("pinned and recent messages exceed the context budget")

    summary_budget = min(
        budget.summary_max_chars,
        budget.max_chars - retained_chars,
    )
    summary = summarizer.summarize(dropped, summary_budget) if summary_budget else ""
    if len(summary) > summary_budget:
        summary = summary[:summary_budget]
    final_chars = retained_chars + len(summary)
    return CompressedContext(
        summary=summary,
        retained_messages=retained,
        dropped_messages=len(dropped),
        original_chars=original_chars,
        final_chars=final_chars,
    )
