"""Runtime Skill selection, prompt injection, and tool allow-listing."""

import threading
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig

from open_deep_research.configuration import Configuration
from open_deep_research.skill_registry import SkillRegistry, SkillSpec

_REGISTRY_CACHE: dict[str, SkillRegistry] = {}
_REGISTRY_LOCK = threading.Lock()


def get_active_skill(config: RunnableConfig) -> SkillSpec | None:
    """Resolve the configured Skill from a cached validated registry."""
    configured = Configuration.from_runnable_config(config)
    if not configured.active_skill:
        return None
    root = str(Path(configured.skills_path).expanduser().resolve())
    with _REGISTRY_LOCK:
        registry = _REGISTRY_CACHE.get(root)
        if registry is None:
            registry = SkillRegistry(root)
            registry.load()
            _REGISTRY_CACHE[root] = registry
    return registry.get(configured.active_skill)


def filter_tools_for_skill(tools: list[Any], skill: SkillSpec | None) -> list[Any]:
    """Enforce a Skill's tool allow-list before model binding."""
    if skill is None:
        return tools
    allowed = set(skill.allowed_tools)
    return [item for item in tools if _tool_name(item) in allowed]


def format_skill_prompt(skill: SkillSpec | None) -> str:
    """Wrap a Skill prompt with its version and enforced permissions."""
    if skill is None:
        return ""
    tools = ", ".join(skill.allowed_tools)
    return (
        f'\n\n<ActiveSkill name="{skill.name}" version="{skill.version}">\n'
        f"Allowed tools are enforced by the runtime: {tools}.\n"
        f"{skill.prompt}\n"
        "</ActiveSkill>"
    )


def _tool_name(item: Any) -> str:
    """Read names from LangChain tools or provider-native tool mappings."""
    if hasattr(item, "name"):
        return str(item.name)
    if isinstance(item, dict):
        return str(item.get("name", item.get("type", "")))
    return ""
