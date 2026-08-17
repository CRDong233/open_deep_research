"""Bounded retries and structured failures for Agent tool execution."""

import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(api[_-]?key|authorization|access[_-]?token|token)\s*[:=]\s*\S+"
)
_OPENAI_STYLE_KEY = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")


class ToolErrorKind(str, Enum):
    """Stable error categories used by retry and recovery policies."""

    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    INVALID_INPUT = "invalid_input"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ToolExecutionPolicy:
    """Limits for one logical Agent tool call."""

    timeout_seconds: float = 30.0
    max_attempts: int = 2
    initial_backoff_seconds: float = 0.25
    max_backoff_seconds: float = 2.0

    def __post_init__(self) -> None:
        """Reject unsafe or non-terminating retry settings."""
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.initial_backoff_seconds < 0:
            raise ValueError("initial_backoff_seconds must not be negative")
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError("max_backoff_seconds must not be smaller than initial")


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    """Inspectable outcome of a bounded tool execution."""

    success: bool
    attempts: int
    duration_ms: float
    value: Any = None
    error_kind: ToolErrorKind | None = None
    error_type: str | None = None
    error_message: str | None = None
    retryable: bool = False

    def as_tool_message(self) -> str:
        """Serialize an Agent-safe result without exposing exception internals."""
        if self.success:
            if isinstance(self.value, str):
                return self.value
            return json.dumps(self.value, ensure_ascii=False, default=str)
        return json.dumps(
            {
                "ok": False,
                "error": {
                    "kind": self.error_kind.value if self.error_kind else "unknown",
                    "type": self.error_type,
                    "message": self.error_message,
                    "retryable": self.retryable,
                    "attempts": self.attempts,
                },
            },
            ensure_ascii=False,
        )


async def execute_with_policy(
    operation: Callable[[], Awaitable[Any]],
    policy: ToolExecutionPolicy,
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> ToolExecutionResult:
    """Execute an async operation with bounded, category-aware retries."""
    started_at = time.perf_counter()
    last_error: Exception | None = None
    last_kind = ToolErrorKind.UNKNOWN

    for attempt in range(1, policy.max_attempts + 1):
        try:
            value = await asyncio.wait_for(
                operation(),
                timeout=policy.timeout_seconds,
            )
            return ToolExecutionResult(
                success=True,
                value=value,
                attempts=attempt,
                duration_ms=(time.perf_counter() - started_at) * 1000,
            )
        except Exception as error:
            last_error = error
            last_kind = classify_tool_error(error)
            retryable = is_retryable(last_kind)
            if not retryable or attempt >= policy.max_attempts:
                break
            backoff = min(
                policy.initial_backoff_seconds * (2 ** (attempt - 1)),
                policy.max_backoff_seconds,
            )
            await sleep(backoff)

    assert last_error is not None
    return ToolExecutionResult(
        success=False,
        attempts=attempt,
        duration_ms=(time.perf_counter() - started_at) * 1000,
        error_kind=last_kind,
        error_type=last_error.__class__.__name__,
        error_message=sanitize_error_message(str(last_error)),
        retryable=is_retryable(last_kind),
    )


def classify_tool_error(error: Exception) -> ToolErrorKind:
    """Map provider-specific exceptions into stable recovery categories."""
    if isinstance(error, TimeoutError):
        return ToolErrorKind.TIMEOUT

    status_code = getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)
    if status_code in {401, 403}:
        return ToolErrorKind.AUTHENTICATION
    if status_code in {400, 404, 409, 422}:
        return ToolErrorKind.INVALID_INPUT
    if status_code == 429:
        return ToolErrorKind.RATE_LIMIT
    if isinstance(status_code, int) and status_code >= 500:
        return ToolErrorKind.UNAVAILABLE

    message = str(error).lower()
    if "rate limit" in message or "too many requests" in message:
        return ToolErrorKind.RATE_LIMIT
    if "unauthorized" in message or "invalid api key" in message:
        return ToolErrorKind.AUTHENTICATION
    if isinstance(error, ConnectionError | OSError):
        return ToolErrorKind.UNAVAILABLE
    return ToolErrorKind.UNKNOWN


def is_retryable(kind: ToolErrorKind) -> bool:
    """Return whether retrying could succeed without changing the request."""
    return kind in {
        ToolErrorKind.TIMEOUT,
        ToolErrorKind.RATE_LIMIT,
        ToolErrorKind.UNAVAILABLE,
    }


def sanitize_error_message(message: str, *, max_length: int = 500) -> str:
    """Redact common credentials and bound error text sent back to the Agent."""
    redacted = _SECRET_ASSIGNMENT.sub(r"\1=[REDACTED]", message)
    redacted = _OPENAI_STYLE_KEY.sub("[REDACTED_KEY]", redacted)
    if len(redacted) <= max_length:
        return redacted
    return redacted[: max_length - 3] + "..."
