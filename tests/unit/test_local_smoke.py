"""Tests for the bounded local LangGraph smoke runner."""

import importlib.util
from pathlib import Path


def load_smoke_module():
    """Load the standalone script without requiring scripts to be a package."""
    path = Path(__file__).parents[2] / "scripts" / "run_local_smoke.py"
    spec = importlib.util.spec_from_file_location("run_local_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_payload_keeps_online_smoke_bounded():
    """Use one research and tool iteration even when the timeout changes."""
    smoke = load_smoke_module()

    payload = smoke.build_payload("What is RAG?", timeout_seconds=11)

    configurable = payload["config"]["configurable"]
    assert payload["assistant_id"] == "Deep Researcher"
    assert payload["input"]["messages"][0]["content"] == "What is RAG?"
    assert configurable["max_researcher_iterations"] == 1
    assert configurable["max_react_tool_calls"] == 1
    assert configurable["tool_max_attempts"] == 1
    assert configurable["tool_timeout_seconds"] == 11


def test_summarize_response_excludes_answer_content():
    """Report run facts without preserving response messages in the summary."""
    smoke = load_smoke_module()

    summary = smoke.summarize_response(
        {
            "final_report": "Sensitive answer body",
            "messages": [{"content": "https://example.com"}],
            "notes": [],
        },
        duration_ms=123.456,
    )

    assert summary == {
        "status": "response_received",
        "duration_ms": 123.46,
        "final_report_present": True,
        "usage_metadata_present": False,
        "url_mentions": 1,
        "top_level_keys": ["final_report", "messages", "notes"],
    }
