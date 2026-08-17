"""Unit tests for Skill loading, permissions, and trace evaluation."""

from pathlib import Path

import pytest

from open_deep_research.skill_registry import (
    SkillRegistry,
    SkillTestCase,
    SkillTrace,
    evaluate_skill,
    load_skill,
)
from open_deep_research.skill_runtime import (
    filter_tools_for_skill,
    format_skill_prompt,
    get_active_skill,
)


def test_repository_skills_load_with_versioned_permissions() -> None:
    """Committed Skill assets should parse into explicit contracts."""
    registry = SkillRegistry(Path("skills"))

    skills = registry.load()
    evidence = registry.get("evidence-research")

    assert set(skills) == {"evidence-research", "repository-analysis"}
    assert evidence.version == "1.0.0"
    assert evidence.read_only
    assert "knowledge_search" in evidence.allowed_tools
    assert "Never invent" in evidence.prompt


def test_skill_harness_reports_permissions_and_citations() -> None:
    """Trace evaluation should expose exact contract violations."""
    spec = load_skill(Path("skills/evidence-research.md"))
    cases = [
        SkillTestCase(
            name="grounded",
            required_tools={"knowledge_search"},
            requires_citations=True,
        ),
        SkillTestCase(
            name="read-only",
            forbidden_tools={"shell"},
        ),
    ]
    traces = [
        SkillTrace(
            case_name="grounded",
            tool_calls=["knowledge_search", "think_tool"],
            output="The policy is bounded [E1].",
        ),
        SkillTrace(
            case_name="read-only",
            tool_calls=["shell"],
            output="changed files",
        ),
    ]

    evaluation = evaluate_skill(spec, cases, traces)

    assert evaluation.passed_cases == 1
    assert evaluation.pass_rate == pytest.approx(0.5)
    assert evaluation.results[0].passed
    assert "unauthorized tools: shell" in evaluation.results[1].violations
    assert "forbidden tools: shell" in evaluation.results[1].violations


def test_skill_loader_rejects_missing_frontmatter_and_duplicate_names(tmp_path) -> None:
    """Malformed or ambiguous Skill assets should fail before Agent execution."""
    (tmp_path / "invalid.md").write_text("# no metadata", encoding="utf-8")
    with pytest.raises(ValueError, match="frontmatter"):
        load_skill(tmp_path / "invalid.md")

    skill_text = """---
name: duplicate
version: 1.0.0
description: duplicate fixture
allowed_tools: [think_tool]
---
Do the bounded task.
"""
    (tmp_path / "a.md").write_text(skill_text, encoding="utf-8")
    (tmp_path / "b.md").write_text(skill_text, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate skill"):
        SkillRegistry(tmp_path).load()


class NamedTool:
    """Minimal named tool fixture for runtime allow-list tests."""

    def __init__(self, name: str) -> None:
        """Store the public tool name."""
        self.name = name


def test_active_skill_injects_prompt_and_enforces_tool_allow_list() -> None:
    """Runtime selection should remove tools absent from the Skill contract."""
    config = {
        "configurable": {
            "active_skill": "repository-analysis",
            "skills_path": "skills",
        }
    }
    skill = get_active_skill(config)
    tools = [
        NamedTool("knowledge_search"),
        NamedTool("think_tool"),
        NamedTool("shell"),
    ]

    filtered = filter_tools_for_skill(tools, skill)
    prompt = format_skill_prompt(skill)

    assert [item.name for item in filtered] == ["knowledge_search", "think_tool"]
    assert 'name="repository-analysis"' in prompt
    assert "Allowed tools are enforced" in prompt
