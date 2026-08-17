from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path

from cayu.runtime.context import (
    _KNOWLEDGE_CANDIDATES_CLOSE_TAG,
    _KNOWLEDGE_CANDIDATES_FORMAT_VERSION,
    _KNOWLEDGE_CANDIDATES_OPEN_TAG,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_INTERNAL_MARKDOWN_REFERENCE_PATTERNS = (
    re.compile(r"https://github\.com/cayu-dev/cayu/(?:issues|pull)/[0-9]+", re.I),
    re.compile(r"(?:\.\./)+(?:issues|pull)/[0-9]+", re.I),
    re.compile(r"\b(?:cayu|cayu-dev/cayu)#[0-9]+\b", re.I),
    re.compile(r"(?<![A-Za-z0-9_/-])#[0-9]+\b"),
    re.compile(r"\bpre-#[0-9]+", re.I),
    re.compile(r"\bmetadata-isolation-[0-9a-f]{8,}\b", re.I),
)


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


def test_knowledge_injection_runtime_contract_pins_candidate_manifest() -> None:
    section = _section(
        _REPO_ROOT / "docs" / "runtime-contracts.md",
        start="Apps can also use `KnowledgeInjectionPolicy`",
        end="This slice does not add graph retrieval",
    )

    assert "synthetic user context" not in section
    for required in (
        "latest eligible user message",
        "never falls back to an earlier turn",
        "structured-output repair or before-stop continuation",
        "same atomic write that appends them to the transcript",
        "a later real user message is not suppressed",
        "`query_max_chars`",
        "`max_hits`",
        "`max_bytes`",
        "capped at 128 KiB",
        "first text part",
        "before invoking the wrapped context policy",
        "provider token counting, compaction",
        "discarded rather than being attached to a different message",
        "original user text and file parts remain separate and unchanged",
        f"`cayu.knowledge_candidates.v{_KNOWLEDGE_CANDIDATES_FORMAT_VERSION}`",
        f"`{_KNOWLEDGE_CANDIDATES_OPEN_TAG}`",
        f"`{_KNOWLEDGE_CANDIDATES_CLOSE_TAG}`",
        "valid JSON",
        "entry ids, kinds, source metadata, and bounded excerpts",
        "ordinary `read_knowledge` tool",
        "does not fabricate assistant tool calls or tool-result messages",
        "strict provider tool-history validation",
        "any knowledge kind",
        "no kind-specific loader",
        "runtime-authored context",
        "durable transcript",
        "frozen once per user turn",
        "runtime checkpoint",
        "`knowledge_injection`",
        "tool-result follow-ups, retries, cache-prefix construction",
        "provider-managed reasoning state",
        "does not mutate the frozen candidates",
        "Zero-hit searches and `fail_open=True` failures are also frozen",
        "absence of an active knowledge store",
        "retroactively injecting into a user message",
        "prunes its checkpoint frame",
        "prompt-cache prefix stable",
        "retains every indistinguishable anchor",
        "rehydrated even when a later context build has no active knowledge store",
        "`max_checkpoint_bytes` default is 1 MiB",
        "configured up to 16 MiB",
        "fails closed before provider dispatch",
        "`knowledge.search.started`",
        "`knowledge.search.completed`",
        "`knowledge.search.failed`",
        "`knowledge.injected`",
        "candidate count, byte count, format version, truncation state",
        "does not emit a second search or injection event",
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
    private_repository_url = "https://github.com/" + "vertex" + "kg/cayu"
    assert private_repository_url not in readme

    with (_REPO_ROOT / "pyproject.toml").open("rb") as source:
        project = tomllib.load(source)["project"]

    assert project["urls"] == {
        "Homepage": "https://cayu.dev",
        "Repository": "https://github.com/cayu-dev/cayu",
        "Documentation": "https://github.com/cayu-dev/cayu#readme",
        "Issues": "https://github.com/cayu-dev/cayu/issues",
        "Changelog": "https://github.com/cayu-dev/cayu/blob/main/docs/release-notes.md",
    }


def test_tracked_tree_excludes_private_and_internal_identifiers() -> None:
    forbidden_terms = ("vertex" + "kg", "lane" + "-" + "agent")
    tracked_paths = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout.lower()
    assert all(term.encode("utf-8") not in tracked_paths for term in forbidden_terms)

    completed = subprocess.run(
        [
            "git",
            "grep",
            "--ignore-case",
            "--line-number",
            *(argument for term in forbidden_terms for argument in ("-e", term)),
        ],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1, completed.stdout or completed.stderr


def test_public_markdown_excludes_internal_tracker_and_run_identifiers() -> None:
    tracked_paths = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.md"],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")

    for encoded_path in tracked_paths:
        if not encoded_path:
            continue
        relative_path = encoded_path.decode("utf-8")
        contents = (_REPO_ROOT / relative_path).read_text(encoding="utf-8")
        matches = [
            match.group(0)
            for pattern in _INTERNAL_MARKDOWN_REFERENCE_PATTERNS
            if (match := pattern.search(contents)) is not None
        ]
        assert not matches, (
            f"{relative_path}: internal tracker or run identifiers remain: {matches}"
        )


def test_internal_markdown_reference_patterns_preserve_generic_pull_syntax() -> None:
    rejected = (
        "tracked in #" + "174",
        "cayu#" + "16",
        "Issue #" + "243",
        "(#" + "212)",
        "pre-#" + "336",
        "https://github.com/cayu-dev/cayu/issues/" + "243",
        "../../../pull/" + "352",
        "metadata-isolation-" + "2e0cb47627d6",
    )
    for reference in rejected:
        assert any(
            pattern.search(reference) for pattern in _INTERNAL_MARKDOWN_REFERENCE_PATTERNS
        ), reference

    generic_repository_pull = "owner/repo#" + "123"
    assert all(
        pattern.search(generic_repository_pull) is None
        for pattern in _INTERNAL_MARKDOWN_REFERENCE_PATTERNS
    )


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


def test_application_anatomy_guide_tracks_shipped_process_roles() -> None:
    process_roles = _heading_section(
        _REPO_ROOT / "src" / "cayu" / "guides" / "application-anatomy.md",
        heading="Process roles",
    )

    assert "`cayu serve` and `cayu worker` are shipped process-role adapters" in (process_roles)
    assert "Cayu does not ship a `cayu script` command" in process_roles
    assert "`cayu server`" not in process_roles
    assert "`cayu worker` are not shipped commands" not in process_roles


def test_release_notes_preserve_storage_revision_chronology() -> None:
    release_notes = _REPO_ROOT / "docs" / "release-notes.md"
    unreleased = _heading_section(release_notes, heading="Unreleased")
    v0_2_1 = _heading_section(release_notes, heading="v0.2.1")
    v0_2_0 = _heading_section(release_notes, heading="v0.2.0")

    assert "advances from revision 29 to revision 34" in v0_2_0
    assert "confirm revision 34 with no pending migrations" in v0_2_0
    assert "revision 39" not in v0_2_0

    assert "advances from revision 34 to revision 36" in v0_2_1
    assert "confirm revision 36 with no pending migrations" in v0_2_1
    assert "revision 39" not in v0_2_1

    assert "Breaking schema revision 39" in unreleased
    assert "confirm revision 39 before starting current workers" in unreleased


def test_project_server_command_is_release_visible_and_fail_closed() -> None:
    guide = " ".join(
        (_REPO_ROOT / "docs" / "project-server.md").read_text(encoding="utf-8").split()
    )
    readme = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")
    runtime_contract = (_REPO_ROOT / "docs" / "runtime-contracts.md").read_text(encoding="utf-8")

    assert "docs/project-server.md" in readme
    assert "`cayu serve` is a process-role adapter" in runtime_contract
    assert "Without `--dev`, the command fails closed" in guide
    assert "Autoreload and multi-process workers are v1 non-goals" in guide
    assert "Incomplete-session startup recovery is off by default" in guide


def test_project_worker_command_is_release_visible() -> None:
    guide = " ".join(
        (_REPO_ROOT / "docs" / "project-workers.md").read_text(encoding="utf-8").split()
    )
    readme = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")
    runtime_contract = (_REPO_ROOT / "docs" / "runtime-contracts.md").read_text(encoding="utf-8")

    assert "docs/project-workers.md" in readme
    assert "`cayu worker <name>` is a process-role adapter" in runtime_contract
    assert "performs no implicit incomplete-session recovery" in guide
    assert "TaskStoreDispatcher.run_worker" in guide
    assert "run_task_worker(" in guide
    for exit_code in ("`130`", "`143`", "`124`"):
        assert exit_code in guide


def test_mcp_transport_limits_are_documented_as_one_shared_contract() -> None:
    runtime_contract = " ".join(
        (_REPO_ROOT / "docs" / "runtime-contracts.md").read_text(encoding="utf-8").split()
    )

    for required in (
        "`McpTransportLimits`",
        "`max_message_bytes`",
        "`max_response_bytes`",
        "`idle_timeout_s`",
        "`total_call_timeout_s`",
        "The exact ceiling is accepted and the next byte is rejected",
        "including runtime tool-adapter calls through either built-in transport",
        "Invalid JSON values inside object-shaped outbound requests fail through a detached",
        "Inbound JSON and SSE messages use the same nesting envelope",
        "Fatal stdio framing",
        "HTTP response failure is isolated",
        "including during initialization and built-in-session tool discovery",
        "An injected `McpSession` that cannot make that positive fencing guarantee",
        "a private settlement task retains that exact work",
        "established session's response has completed transport cleanup",
        "session identifiers are installed only after",
        "accept only 2xx responses (including session-termination DELETEs)",
    ):
        assert required in runtime_contract


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
