---
name: repository-analysis
version: 1.0.0
description: Analyze an indexed repository without modifying files or Git state.
allowed_tools:
  - think_tool
  - knowledge_search
  - ResearchComplete
read_only: true
input_schema:
  required:
    - question
    - repository
output_requirements:
  - Cite indexed files and character ranges.
  - Separate observed behavior from proposed changes.
  - Do not claim that a patch or test was executed.
---
# Repository Analysis

Use `knowledge_search` to locate relevant code and documentation. Build conclusions from retrieved source ranges, and clearly distinguish code that exists from changes that are only recommendations.

This Skill is read-only. Do not request shell, write, Git push, or pull-request tools. If the indexed evidence is insufficient, identify the missing file or test instead of fabricating its behavior.

