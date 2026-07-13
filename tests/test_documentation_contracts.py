from __future__ import annotations

from pathlib import Path

from cayu.runtime.context import (
    _KNOWLEDGE_INJECTION_CLOSE_TAG,
    _KNOWLEDGE_INJECTION_OPEN_TAG,
    _KNOWLEDGE_INJECTION_TOOL_CALL_ID_PREFIX,
    _KNOWLEDGE_INJECTION_TOOL_NAME,
    _KNOWLEDGE_INJECTION_TRUNCATION_MARKER,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _section(path: Path, *, start: str, end: str) -> str:
    text = path.read_text(encoding="utf-8")
    assert start in text, f"documentation section start not found in {path.name}: {start}"
    section = text.split(start, 1)[1]
    assert end in section, f"documentation section end not found in {path.name}: {end}"
    return " ".join(section.split(end, 1)[0].split())


def test_knowledge_injection_runtime_contract_pins_synthetic_tool_round() -> None:
    section = _section(
        _REPO_ROOT / "docs" / "runtime-contracts.md",
        start="Apps can also use `KnowledgeInjectionPolicy`",
        end="This slice does not add graph retrieval",
    )

    assert "synthetic user context" not in section
    for required in (
        "latest user message",
        "skips the search instead of falling back to an earlier turn",
        "`query_max_chars`",
        "`max_hits`",
        "`max_bytes`",
        "synthetic tool",
        "after the latest real user message",
        f"`{_KNOWLEDGE_INJECTION_TOOL_NAME}`",
        f"`{_KNOWLEDGE_INJECTION_TOOL_CALL_ID_PREFIX}{{step}}`",
        "tool-result message",
        "untrusted reference data",
        "configurable `prefix` is followed by an explicit warning",
        "retrieved snippets are enclosed",
        f"`{_KNOWLEDGE_INJECTION_OPEN_TAG}`",
        f"`{_KNOWLEDGE_INJECTION_CLOSE_TAG}`",
        f"`{_KNOWLEDGE_INJECTION_TRUNCATION_MARKER.strip()}`",
        "projection-only",
        "durable transcript",
        "`knowledge.search.started`",
        "`knowledge.search.completed`",
        "`knowledge.search.failed`",
        "`knowledge.injected`",
        "fail closed by default",
        "`fail_open=True`",
    ):
        assert required in section


def test_readme_knowledge_injection_description_preserves_trust_boundary() -> None:
    section = _section(
        _REPO_ROOT / "README.md",
        start="`KnowledgeInjectionPolicy` searches the active environment's `knowledge_store`",
        end="Scope tool authority per agent:",
    )

    assert "synthetic user context" not in section
    for required in (
        "synthetic assistant call",
        f"`{_KNOWLEDGE_INJECTION_TOOL_NAME}`",
        "matching tool result",
        f"`{_KNOWLEDGE_INJECTION_OPEN_TAG}`",
        "not user or system instruction",
        "durable transcript",
        "fail closed by default",
        "`fail_open=True`",
    ):
        assert required in section
