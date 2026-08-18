"""Unit tests for bounded Agent tool execution."""

import asyncio
import json

import pytest

from open_deep_research.deep_researcher import execute_tool_safely
from open_deep_research.reliability import (
    ToolErrorKind,
    ToolExecutionPolicy,
    classify_tool_error,
    execute_with_policy,
    sanitize_error_message,
)


@pytest.mark.asyncio
async def test_retryable_failure_recovers_with_backoff() -> None:
    """Transient connection failures should retry within the policy limit."""
    attempts = 0
    sleeps: list[float] = []

    async def operation() -> dict[str, bool]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("temporary outage")
        return {"recovered": True}

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    result = await execute_with_policy(
        operation,
        ToolExecutionPolicy(
            timeout_seconds=1,
            max_attempts=3,
            initial_backoff_seconds=0.1,
            max_backoff_seconds=1,
        ),
        sleep=fake_sleep,
    )

    assert result.success
    assert result.attempts == 2
    assert result.value == {"recovered": True}
    assert sleeps == [0.1]


@pytest.mark.asyncio
async def test_success_telemetry_envelope_is_versioned() -> None:
    """Telemetry should expose duration and attempts without changing the legacy API."""

    async def operation() -> dict[str, bool]:
        return {"ok": True}

    result = await execute_with_policy(operation, ToolExecutionPolicy(max_attempts=1))
    payload = result.as_tool_message(include_telemetry=True)

    parsed = json.loads(payload)
    assert parsed["schema_version"] == 1
    assert parsed["ok"] is True
    assert parsed["value"] == {"ok": True}
    assert parsed["telemetry"]["attempts"] == 1
    assert parsed["telemetry"]["duration_ms"] >= 0


@pytest.mark.asyncio
async def test_timeout_is_bounded_and_reported() -> None:
    """A hanging operation should stop after the configured attempts."""

    async def operation() -> None:
        await asyncio.sleep(1)

    async def no_wait(_: float) -> None:
        return None

    result = await execute_with_policy(
        operation,
        ToolExecutionPolicy(
            timeout_seconds=0.001,
            max_attempts=2,
            initial_backoff_seconds=0,
            max_backoff_seconds=0,
        ),
        sleep=no_wait,
    )

    assert not result.success
    assert result.attempts == 2
    assert result.error_kind == ToolErrorKind.TIMEOUT
    assert result.retryable


class HttpError(Exception):
    """Small status-bearing exception fixture."""

    def __init__(self, status_code: int, message: str) -> None:
        """Store an HTTP status like provider SDK exceptions do."""
        super().__init__(message)
        self.status_code = status_code


@pytest.mark.asyncio
async def test_authentication_failure_does_not_retry_or_leak_key() -> None:
    """Permanent authentication errors should fail once with redaction."""
    attempts = 0

    async def operation() -> None:
        nonlocal attempts
        attempts += 1
        raise HttpError(401, "api_key=sk-secretvalue123 was rejected")

    result = await execute_with_policy(
        operation,
        ToolExecutionPolicy(max_attempts=3),
    )

    assert attempts == 1
    assert result.error_kind == ToolErrorKind.AUTHENTICATION
    assert not result.retryable
    assert "secretvalue" not in result.as_tool_message()
    assert "[REDACTED]" in result.as_tool_message()


def test_error_classification_and_policy_validation() -> None:
    """Stable categories should distinguish retryable provider failures."""
    assert classify_tool_error(HttpError(429, "limited")) == ToolErrorKind.RATE_LIMIT
    assert (
        classify_tool_error(HttpError(422, "bad args")) == ToolErrorKind.INVALID_INPUT
    )
    assert classify_tool_error(HttpError(503, "down")) == ToolErrorKind.UNAVAILABLE
    assert "[REDACTED_KEY]" in sanitize_error_message("failed for sk-abcdefgh1234")
    with pytest.raises(ValueError, match="max_attempts"):
        ToolExecutionPolicy(max_attempts=0)


class FlakyTool:
    """LangChain-like tool fixture that succeeds on its second invocation."""

    def __init__(self) -> None:
        """Initialize the invocation counter."""
        self.attempts = 0

    async def ainvoke(self, args, config) -> str:
        """Fail once, then return a stable tool observation."""
        self.attempts += 1
        if self.attempts == 1:
            raise ConnectionError("temporary provider outage")
        return f"recovered:{args['query']}"


@pytest.mark.asyncio
async def test_deep_researcher_uses_reliability_configuration() -> None:
    """The graph helper should apply retry settings from RunnableConfig."""
    tool = FlakyTool()
    config = {
        "configurable": {
            "tool_timeout_seconds": 1,
            "tool_max_attempts": 2,
            "tool_retry_backoff_seconds": 0,
        }
    }

    result = await execute_tool_safely(tool, {"query": "RAG"}, config)
    unknown = await execute_tool_safely(None, {}, config)

    payload = json.loads(result)
    assert payload["schema_version"] == 1
    assert payload["value"] == "recovered:RAG"
    assert payload["telemetry"]["attempts"] == 2
    assert tool.attempts == 2
    assert '"kind": "invalid_input"' in unknown
