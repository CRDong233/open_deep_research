"""Run one bounded LangGraph API smoke test without printing model content."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import requests  # type: ignore[import-untyped]


def build_payload(prompt: str, timeout_seconds: int) -> dict[str, Any]:
    """Build a deliberately small, bounded graph run request."""
    limits = {
        "allow_clarification": False,
        "max_researcher_iterations": 1,
        "max_react_tool_calls": 1,
        "max_concurrent_research_units": 1,
        "tool_max_attempts": 1,
        "tool_timeout_seconds": timeout_seconds,
        "summarization_model_max_tokens": 300,
        "research_model_max_tokens": 400,
        "compression_model_max_tokens": 300,
        "final_report_model_max_tokens": 400,
        "max_content_length": 4000,
    }
    return {
        "assistant_id": "Deep Researcher",
        "input": {"messages": [{"type": "human", "content": prompt}]},
        "config": {"configurable": limits},
    }


def summarize_response(response: dict[str, Any], duration_ms: float) -> dict[str, Any]:
    """Return non-sensitive run facts, excluding answer text and raw messages."""
    serialized = json.dumps(response, ensure_ascii=False)
    return {
        "status": "response_received",
        "duration_ms": round(duration_ms, 2),
        "final_report_present": bool(response.get("final_report")),
        "usage_metadata_present": "usage_metadata" in serialized,
        "url_mentions": serialized.count("http://") + serialized.count("https://"),
        "top_level_keys": sorted(response),
    }


def main() -> int:
    """Execute one bounded request and print a redacted JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:2024/runs/wait")
    parser.add_argument(
        "--prompt",
        default="Briefly define retrieval-augmented generation. Use one web source and include its URL.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=20)
    args = parser.parse_args()

    started = time.perf_counter()
    try:
        response = requests.post(
            args.url,
            json=build_payload(args.prompt, args.timeout_seconds),
            timeout=max(30, args.timeout_seconds * 6),
        )
        response.raise_for_status()
        result = summarize_response(response.json(), (time.perf_counter() - started) * 1000)
    except requests.RequestException as exc:
        result = {
            "status": "request_failed",
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            "error_type": type(exc).__name__,
            "message": str(exc)[:240],
        }
        sys.stdout.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
        return 1
    except (TypeError, ValueError) as exc:
        result = {
            "status": "invalid_response",
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            "error_type": type(exc).__name__,
            "message": str(exc)[:240],
        }
        sys.stdout.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
        return 1

    sys.stdout.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
