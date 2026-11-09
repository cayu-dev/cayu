"""Generated projects: the coding reviewer's environment and who closes the stores.

The coding scaffold's background reviewer runs in its own ``coding-reviewer``
environment, so it never re-binds the source workspace that the primary's
Docker ``coding`` environment holds. Generated ``build_app()`` functions (coding
and agent presets) hand the stores they open to the app as ``owned_resources``,
so ``app.aclose()`` closes them; injected stores stay with their caller. The Git
baseline probe ignores repository ``core.filemode`` and ``core.ignorecase``.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import sqlite3
import subprocess
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from tests.coding_scaffold_fixtures import admit_fixture_image

from cayu import EventType, Message, RunRequest
from cayu.cli.project import project_context
from cayu.cli.scaffold import project_files
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import ExecutionProfileBehaviorIdentity
from cayu.sessions import InMemorySessionStore, SessionQuery
from cayu.storage import InMemoryKnowledgeStore
from cayu.tasks import InMemoryTaskStore


@pytest.fixture(autouse=True)
def _local_database(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("CAYU_DATABASE_URL", "CAYU_REQUIRE_POSTGRES", "CAYU_DATABASE_DIRECT_URL"):
        monkeypatch.delenv(name, raising=False)


class _ReviewerProvider(ModelProvider):
    """Primary delegates one background review and waits; the tool-free reviewer answers."""

    name = "scaffold-reviewer-test"

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests.scaffold_reviewer_provider",
            behavior_version="1",
            implementation_version="1",
        )

    def __init__(self) -> None:
        self.primary_step = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        if not request.tools:
            yield ModelStreamEvent.text_delta("Reviewer found no blocking issue.")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
            return
        calls = (
            ("read_file", {"path": "example.py"}),
            ("subagent", {"agent": "reviewer", "task": "Review example.py."}),
            ("subagent_result", {"all": True, "wait": True}),
        )
        if self.primary_step < len(calls):
            name, arguments = calls[self.primary_step]
            self.primary_step += 1
            yield ModelStreamEvent.tool_call(
                id=f"call_{self.primary_step}", name=name, arguments=arguments
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        yield ModelStreamEvent.text_delta("Review complete.")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


def _write_project(root: Path, files: dict[str, str]) -> Path:
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


def _git_repository(path: Path) -> Path:
    path.mkdir()
    (path / "example.py").write_text("VALUE = 1\n", encoding="utf-8")
    for command in (
        ["init", "-q", "-b", "main"],
        ["add", "example.py"],
        [
            *("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid"),
            *("commit", "-q", "-m", "baseline"),
        ],
    ):
        subprocess.run(["git", "-C", str(path), *command], check=True, capture_output=True)
    return path


def _generated_fixture(project: Path):
    """The generated project's own test doubles (its fake Docker factory)."""

    spec = importlib.util.spec_from_file_location(
        "scaffold_reviewer_fixture", project / "tests/test_coding_composition.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_closed(app, outcome) -> None:
    assert outcome.settled, outcome.summary()
    assert outcome.owned_resources == "released"
    # The generated SQLite session store's connection is closed, not just idle.
    with pytest.raises(sqlite3.ProgrammingError):
        app.session_store._connection.execute("SELECT 1")


@pytest.mark.parametrize(
    "execution,with_capabilities",
    [
        (None, ()),
        # Git publishing is optional; the reviewer must not depend on it.
        ("docker", ()),
        ("docker", ("github-delivery",)),
    ],
    ids=["local", "docker", "docker-github"],
)
def test_generated_reviewer_runs_in_its_own_environment_and_aclose_closes_stores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    execution: str | None,
    with_capabilities: tuple[str, ...],
) -> None:
    project = _write_project(
        tmp_path / "consumer",
        project_files(
            "review-coder",
            preset="coding",
            execution=execution,
            coding_toolchain="python" if execution == "docker" else None,
            with_capabilities=with_capabilities,
        ),
    )
    source = _git_repository(tmp_path / "source")
    created_runners: list[object] = []
    with project_context(project):
        operations = importlib.import_module("operations.coding")
        monkeypatch.setattr(operations, "_verify_coding_dependencies", lambda root: None)
        if execution == "docker":
            admit_fixture_image(monkeypatch, operations)
            _generated_fixture(project)._install_fake_docker_factory(
                monkeypatch, target=tmp_path / "target", created_runners=created_runners
            )
            (tmp_path / "target").mkdir()
        build_app = importlib.import_module("app").build_app

        async def scenario():
            # No injected stores: build_app opens SQLite under .cayu/runtime.
            app = build_app(provider=_ReviewerProvider(), workspace_root=source)
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="review-coder",
                        session_id="review",
                        messages=[Message.text("user", "Review the example.")],
                    )
                )
            ]
            children = (
                await app.session_store.list_sessions(SessionQuery(parent_session_id="review"))
            ).sessions
            return app, events, children, await app.aclose()

        app, events, children, outcome = asyncio.run(scenario())

    assert events[-1].type is EventType.SESSION_COMPLETED, events[-1].payload
    results = {
        event.tool_name: event.payload["result"]
        for event in events
        if event.type is EventType.TOOL_CALL_COMPLETED
    }
    reviewed = results["subagent_result"]["structured"]
    assert [child["status"] for child in reviewed["children"]] == ["completed"]
    assert reviewed["children"][0]["result_text"] == "Reviewer found no blocking issue."
    assert [child.environment_name for child in children] == ["coding-reviewer"]
    if execution == "docker":
        # Only the primary allocated a sandbox and bound the source workspace.
        assert len(created_runners) == 1
    _assert_closed(app, outcome)


def test_injected_coding_stores_stay_with_their_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _write_project(tmp_path / "consumer", project_files("inject-coder", preset="coding"))
    with project_context(project):
        operations = importlib.import_module("operations.coding")
        monkeypatch.setattr(operations, "_verify_coding_dependencies", lambda root: None)
        app = importlib.import_module("app").build_app(
            provider=ScriptedModelProvider([]),
            session_store=InMemorySessionStore(),
            task_store=InMemoryTaskStore(),
            knowledge_store=InMemoryKnowledgeStore(access_scope=operations._knowledge_scope()),
            workspace_root=_git_repository(tmp_path / "source"),
        )
        outcome = asyncio.run(app.aclose())
    assert outcome.settled, outcome.summary()
    assert outcome.owned_resources == "none"


def test_generated_agent_app_closes_the_stores_it_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CAYU_MEMORY_EVIDENCE_KEY", "k" * 64)
    project = _write_project(tmp_path / "consumer", project_files("store-agent"))
    with project_context(project):
        app = importlib.import_module("app").build_app(
            provider=ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.text_delta("Hello."),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ]
                ]
            )
        )

        async def scenario():
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="store-agent",
                        messages=[Message.text("user", "Hi")],
                    )
                )
            ]
            return events, await app.aclose()

        events, outcome = asyncio.run(scenario())
    assert events[-1].type is EventType.SESSION_COMPLETED, events[-1].payload
    _assert_closed(app, outcome)


@pytest.mark.parametrize("execution", [None, "docker"])
def test_git_baseline_ignores_repository_filemode_and_ignorecase(
    tmp_path: Path, execution: str | None
) -> None:
    project = _write_project(
        tmp_path / "consumer",
        project_files(
            "probe-coder",
            preset="coding",
            execution=execution,
            coding_toolchain="python" if execution == "docker" else None,
        ),
    )
    mode_source = _git_repository(tmp_path / "mode-source")
    subprocess.run(["git", "-C", str(mode_source), "config", "core.filemode", "false"], check=True)
    (mode_source / "example.py").chmod(0o755)
    case_source = _git_repository(tmp_path / "case-source")
    subprocess.run(["git", "-C", str(case_source), "config", "core.ignorecase", "true"], check=True)
    (case_source / "EXAMPLE.py").write_text("VALUE = 3\n", encoding="utf-8")
    case_sensitive = (case_source / "example.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    with project_context(project):
        operations = importlib.import_module("operations.coding")
        argv = operations._safe_git_probe_argv("git", "status")
        with pytest.raises(RuntimeError, match="clean committed Git source baseline"):
            operations.observe_clean_coding_product_git_baseline(mode_source)
        if case_sensitive:
            with pytest.raises(RuntimeError, match="clean committed Git source baseline"):
                operations.observe_clean_coding_product_git_baseline(case_source)
    overrides = [argv[index + 1] for index, value in enumerate(argv[:-1]) if value == "-c"]
    assert {"core.filemode=true", "core.ignorecase=false"} <= set(overrides)
