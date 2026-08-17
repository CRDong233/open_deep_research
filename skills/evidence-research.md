---
name: evidence-research
version: 1.0.0
description: Produce a source-linked answer from local and web evidence.
allowed_tools:
  - think_tool
  - knowledge_search
  - tavily_search
  - ResearchComplete
read_only: true
input_schema:
  required:
    - question
output_requirements:
  - Every material factual claim must retain its source citation.
  - Distinguish missing evidence from negative evidence.
  - End when the available evidence is sufficient for the question.
---
# Evidence Research

Plan the minimum research needed to answer the question. Search the local knowledge base first when it is configured, then use web search only for missing or time-sensitive facts.

Preserve every `[E#]` identifier returned by `knowledge_search` next to the claim it supports. Never invent a citation identifier. If evidence conflicts, present the conflict and source boundaries instead of silently choosing one source.

Use `think_tool` between search rounds. Stop when the evidence is sufficient or the tool budget is exhausted, then call `ResearchComplete`.

