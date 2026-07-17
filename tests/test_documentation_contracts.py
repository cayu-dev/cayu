from __future__ import annotations

import tomllib
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


def _heading_section(path: Path, *, heading: str) -> str:
    text = path.read_text(encoding="utf-8")
    marker = f"## {heading}\n"
    assert marker in text, f"documentation heading not found in {path.name}: {heading}"
    section = text.split(marker, 1)[1].split("\n## ", 1)[0]
    return " ".join(section.split())


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


def test_readme_surfaces_reviewed_knowledge_and_links_runtime_contracts() -> None:
    readme_path = _REPO_ROOT / "README.md"
    capabilities = _heading_section(readme_path, heading="What Cayu provides")
    for required in (
        "Reviewed knowledge",
        "approval state",
        "keyword/vector retrieval",
    ):
        assert required in capabilities

    readme = readme_path.read_text(encoding="utf-8")
    assert "[Runtime contracts](" in readme
    assert "/docs/runtime-contracts.md" in readme


def test_release_facing_metadata_uses_public_urls() -> None:
    readme = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "https://github.com/vertexkg/cayu" not in readme

    with (_REPO_ROOT / "pyproject.toml").open("rb") as source:
        project = tomllib.load(source)["project"]

    assert project["urls"] == {
        "Homepage": "https://cayu.dev",
        "Repository": "https://github.com/cayu-dev/cayu",
        "Documentation": "https://github.com/cayu-dev/cayu#readme",
        "Issues": "https://github.com/cayu-dev/cayu/issues",
        "Changelog": "https://github.com/cayu-dev/cayu/blob/main/docs/release-notes.md",
    }


def test_application_anatomy_guide_is_linked_from_release_facing_docs() -> None:
    guide = _REPO_ROOT / "src" / "cayu" / "guides" / "application-anatomy.md"
    assert guide.is_file()

    readme = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "application-anatomy.md" in readme

    for relative_path in (
        "docs/console.md",
        "docs/project-layout.md",
        "docs/runtime-contracts.md",
    ):
        text = (_REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert "application-anatomy.md" in text, (
            f"canonical anatomy guide not linked from {relative_path}"
        )


def test_server_auth_tenant_isolation_boundary_is_documented() -> None:
    readme = " ".join((_REPO_ROOT / "README.md").read_text(encoding="utf-8").split())
    runtime_contract = " ".join(
        (_REPO_ROOT / "docs" / "runtime-contracts.md").read_text(encoding="utf-8").split()
    )
    recipe = " ".join(
        (_REPO_ROOT / "docs" / "recipes" / "server-auth-tenancy.md")
        .read_text(encoding="utf-8")
        .split()
    )

    assert "`AuthContext.tenant` records authenticated operator provenance" in readme
    assert "server-auth-tenancy.md" in readme

    for required in (
        "Authentication and tenant isolation are separate contracts",
        "session, event, transcript, task, knowledge, artifact, usage",
        "not filtered by it",
        "same raw Cayu record",
        "Labels, metadata, namespaced identifiers, and query filters",
        "operator boundary",
        "session compaction, and message enqueue",
    ):
        assert required in runtime_contract

    for required in (
        "WHERE public_id = :public_id AND tenant_id = :authenticated_tenant",
        "require_owned_run",
        "before constructing a Cayu query",
        "allocates non-null `public_id`, `cayu_session_id`, and `task_id`",
        "`(tenant_id, idempotency_key)` constraint",
        "request_fingerprint",
        "IdempotencyConflict",
        "status_code=409",
        "owned.cayu_session_id is None or owned.task_id is None",
        "raise HTTPException(status_code=404",
        "passing `None` to an optional Cayu query filter",
        "status_code=202",
        "create_task",
        "load_task",
        "task_matches_product_run",
        "except ValueError",
        "lost HTTP response and concurrent task-creation race",
        "existing task is accepted only after its immutable creation fields",
        "durable `TaskStore`",
        "run_task_worker",
        "outside the HTTP request",
        "rejects ordinary product members",
        "does not rewrite their store queries",
        "none is sufficient authorization by itself",
        "never raw Cayu events or payloads",
        "application-owned allowlist",
        "row-level security",
        "Native tenant-aware storage is not currently part",
    ):
        assert required in recipe

    assert "run_to_completion" not in recipe
    assert "return await cayu_app.session_store.query_events" not in recipe
    assert "product_runs.require_owned" not in recipe
