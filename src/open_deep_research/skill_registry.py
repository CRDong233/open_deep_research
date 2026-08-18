"""Versioned Markdown Skill loading and deterministic trace evaluation."""

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

_FRONTMATTER_DELIMITER = "---"
_CITATION_PATTERN = re.compile(r"\[E(?:-[0-9a-f]{10}|\d+)]", re.IGNORECASE)


class SkillSpec(BaseModel):
    """Validated behavior, permissions, and prompt body for one Skill."""

    name: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    description: str = Field(min_length=1)
    allowed_tools: list[str] = Field(min_length=1)
    read_only: bool = True
    input_schema: dict[str, object] = Field(default_factory=dict)
    output_requirements: list[str] = Field(default_factory=list)
    prompt: str = Field(min_length=1)
    source_path: str

    @field_validator("allowed_tools")
    @classmethod
    def validate_unique_tools(cls, tools: list[str]) -> list[str]:
        """Reject ambiguous duplicate tool declarations."""
        if len(set(tools)) != len(tools):
            raise ValueError("allowed_tools must not contain duplicates")
        return tools


class SkillRegistry:
    """Load versioned Skill Markdown files from one explicit directory."""

    def __init__(self, root: str | Path) -> None:
        """Resolve the Skill root without loading any file yet."""
        self.root = Path(root).expanduser().resolve()
        self._skills: dict[str, SkillSpec] = {}

    def load(self) -> dict[str, SkillSpec]:
        """Load all Skill files and reject duplicate names."""
        if not self.root.is_dir():
            raise ValueError(f"skill root is not a directory: {self.root}")
        loaded: dict[str, SkillSpec] = {}
        for path in sorted(self.root.glob("*.md"), key=lambda item: item.name):
            spec = load_skill(path)
            if spec.name in loaded:
                raise ValueError(f"duplicate skill name: {spec.name}")
            loaded[spec.name] = spec
        self._skills = loaded
        return dict(loaded)

    def get(self, name: str) -> SkillSpec:
        """Return a loaded Skill or fail with an actionable name."""
        if name not in self._skills:
            raise KeyError(f"skill is not loaded: {name}")
        return self._skills[name]


class SkillTestCase(BaseModel):
    """Expected tool and output behavior for one Skill scenario."""

    name: str
    required_tools: set[str] = Field(default_factory=set)
    forbidden_tools: set[str] = Field(default_factory=set)
    requires_citations: bool = False


class SkillTrace(BaseModel):
    """Observed tools and final output for one Skill scenario."""

    case_name: str
    tool_calls: list[str] = Field(default_factory=list)
    output: str = ""


class SkillCaseResult(BaseModel):
    """Contract violations found in one evaluated Skill trace."""

    case_name: str
    passed: bool
    violations: list[str] = Field(default_factory=list)


class SkillEvaluation(BaseModel):
    """Aggregate deterministic results for one Skill version."""

    skill_name: str
    skill_version: str
    passed_cases: int
    total_cases: int
    pass_rate: float = Field(ge=0, le=1)
    results: list[SkillCaseResult]


def load_skill(path: str | Path) -> SkillSpec:
    """Parse a Markdown Skill with YAML frontmatter."""
    skill_path = Path(path).expanduser().resolve()
    text = skill_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != _FRONTMATTER_DELIMITER:
        raise ValueError(f"skill is missing YAML frontmatter: {skill_path.name}")
    try:
        closing_index = next(
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.strip() == _FRONTMATTER_DELIMITER
        )
    except StopIteration as error:
        raise ValueError(
            f"skill frontmatter is not closed: {skill_path.name}"
        ) from error

    metadata = yaml.safe_load("\n".join(lines[1:closing_index]))
    if not isinstance(metadata, dict):
        raise ValueError(f"skill frontmatter must be a mapping: {skill_path.name}")
    prompt = "\n".join(lines[closing_index + 1 :]).strip()
    return SkillSpec(**metadata, prompt=prompt, source_path=str(skill_path))


def evaluate_skill(
    spec: SkillSpec,
    cases: list[SkillTestCase],
    traces: list[SkillTrace],
) -> SkillEvaluation:
    """Evaluate permissions, required tools, and citation contracts."""
    traces_by_case = {trace.case_name: trace for trace in traces}
    results: list[SkillCaseResult] = []
    allowed = set(spec.allowed_tools)

    for case in cases:
        trace = traces_by_case.get(case.name)
        violations: list[str] = []
        if trace is None:
            violations.append("missing trace")
        else:
            called = set(trace.tool_calls)
            unauthorized = called - allowed
            missing = case.required_tools - called
            forbidden = case.forbidden_tools & called
            if unauthorized:
                violations.append(
                    f"unauthorized tools: {', '.join(sorted(unauthorized))}"
                )
            if missing:
                violations.append(f"missing tools: {', '.join(sorted(missing))}")
            if forbidden:
                violations.append(f"forbidden tools: {', '.join(sorted(forbidden))}")
            if case.requires_citations and not _CITATION_PATTERN.search(trace.output):
                violations.append("missing evidence citation")
        results.append(
            SkillCaseResult(
                case_name=case.name,
                passed=not violations,
                violations=violations,
            )
        )

    passed_cases = sum(result.passed for result in results)
    total_cases = len(cases)
    return SkillEvaluation(
        skill_name=spec.name,
        skill_version=spec.version,
        passed_cases=passed_cases,
        total_cases=total_cases,
        pass_rate=passed_cases / total_cases if total_cases else 0,
        results=results,
    )
