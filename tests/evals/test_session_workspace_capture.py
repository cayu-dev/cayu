"""A coding session records the commit its checkout was at; add-session starts the case there.

Sessions run in a Cloud-brokered coding environment served by the repository's fake broker,
over a source workspace that is a Git checkout, and are added to suites through the real
server routes.
"""

from __future__ import annotations

import asyncio
import subprocess
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from tests.core.test_human_review import identity
from tests.evals.test_structured_output_field_assertions import _workflow_target
from tests.sandbox_broker.fake_broker import FakeBroker
from tests.server.test_server_eval_session_additions import _destination
from tests.server.test_server_eval_session_additions import (
    addition_environment as addition_environment,
)
from tests.server.test_server_evaluation_promotion import _AUTH_HEADERS, _SESSION_ID

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu.environments import EnvironmentSpec
from cayu.evals import WorkspaceFixtureSpec
from cayu.evals.execution import CorpusTarget
from cayu.evals.read_views import EvalSuiteViewIdentity
from cayu.evals.session_additions import session_addition_workspace
from cayu.evals.trajectory import trajectory_from_session
from cayu.events import EventType
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.sandbox_broker import BrokeredSandboxCodingEnvironmentFactory
from cayu.workspaces import GitWorkspaceSource, LocalWorkspace


class _AnswerProvider(ModelProvider):
    name = "answer-provider"

    @property
    def execution_profile_identity(self):
        return identity("answer-provider")

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        yield ModelStreamEvent.text_delta("the claim holds")
        yield ModelStreamEvent.completed(
            {
                "finish_reason": "stop",
                "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            }
        )


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=Eval", "-c", "user.email=eval@example.test", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _checkout(root: Path) -> tuple[Path, str]:
    """A Git checkout with one commit, and the commit."""

    root.mkdir(parents=True)
    _git(root, "init", "--quiet")
    (root / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    _git(root, "add", "-A")
    _git(root, "commit", "--quiet", "-m", "first")
    return root, _git(root, "rev-parse", "HEAD")


def _app(
    tmp_path: Path,
    source: Path,
    *,
    repository: str | None = "payments",
    register: bool = True,
) -> CayuApp:
    app = CayuApp(enable_logging=False)
    app.register_provider(_AnswerProvider(), default=True)
    app.register_environment_factory(
        EnvironmentSpec(name="sandbox", execution_profile_identity=identity("brokered")),
        BrokeredSandboxCodingEnvironmentFactory(
            broker=FakeBroker(tmp_path / "cloud").client(),
            source_workspace=LocalWorkspace(
                source, workspace_id="checkout", excluded_directory_names=("scratch",)
            ),
            egress="public_web",
            workspace_repository=repository,
        ),
        default=True,
    )
    if register:
        app.register_workspace_source(GitWorkspaceSource("payments", source))
    app.register_agent(AgentSpec(name="assistant", model="answer-model"))
    return app


def _run(app: CayuApp, session_id: str) -> None:
    async def run() -> None:
        async for _ in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "Does add() add?")],
            )
        ):
            pass

    asyncio.run(run())


def _origin(app: CayuApp, session_id: str) -> dict | None:
    trajectory = asyncio.run(trajectory_from_session(app, session_id))
    bound = next(
        event
        for event in trajectory.events
        if event.type == EventType.ENVIRONMENT_BINDING_COMPLETED
    )
    return bound.payload["bound_metadata"].get("workspace_origin")


# What a session records ----------------------------------------------------------------


def test_a_coding_session_records_the_commit_its_checkout_was_at(tmp_path: Path) -> None:
    source, head = _checkout(tmp_path / "checkout")
    app = _app(tmp_path, source)

    _run(app, "clean")
    assert _origin(app, "clean") == {"repository": "payments", "revision": head, "clean": True}

    # The environment copies nothing in from a directory it excludes.
    (source / "scratch").mkdir()
    (source / "scratch" / "notes.txt").write_text("scratch\n")
    _run(app, "excluded")
    assert _origin(app, "excluded") == {
        "repository": "payments",
        "revision": head,
        "clean": True,
    }

    (source / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    _run(app, "modified")
    assert _origin(app, "modified") == {
        "repository": "payments",
        "revision": head,
        "clean": False,
    }
    # Reading the revision leaves the checkout's index and files as they were.
    assert _git(source, "status", "--porcelain") == "M calc.py\n?? scratch/".strip()


def test_a_session_records_no_commit_its_checkout_cannot_name(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    app = _app(tmp_path / "plain-app", plain)
    _run(app, "plain")
    assert _origin(app, "plain") is None

    # A directory inside a checkout is no checkout of the source's own tree.
    outer, _ = _checkout(tmp_path / "outer")
    (outer / "service").mkdir()
    (outer / "service" / "main.py").write_text("print('ok')\n")
    _git(outer, "add", "-A")
    _git(outer, "commit", "--quiet", "-m", "service")
    nested = _app(tmp_path / "nested-app", outer / "service")
    _run(nested, "nested")
    assert _origin(nested, "nested") is None

    # Without workspace_repository the environment doesn't look.
    source, _ = _checkout(tmp_path / "unnamed")
    unnamed = _app(tmp_path / "unnamed-app", source, repository=None)
    _run(unnamed, "unnamed")
    assert _origin(unnamed, "unnamed") is None


# What add-session makes of it -----------------------------------------------------------


def _target(app: CayuApp) -> CorpusTarget:
    return CorpusTarget(
        key="assistant.default",
        app=app,
        request_base=RunRequest(agent_name="assistant", messages=[]),
        application_release_id="release-one",
    )


def test_the_eval_targets_application_must_register_the_source(tmp_path: Path) -> None:
    source, head = _checkout(tmp_path / "checkout")
    served = _app(tmp_path / "served", source)
    _run(served, "served")
    trajectory = asyncio.run(trajectory_from_session(served, "served"))

    # The target's application runs the case, so its registration is the one that counts.
    unregistered = session_addition_workspace(
        trajectory, target=_target(_app(tmp_path / "target", source, register=False))
    )
    assert (unregistered.state, unregistered.revision) == ("unregistered", head)
    assert unregistered.fixture() is None
    registered = session_addition_workspace(
        trajectory, target=_target(_app(tmp_path / "other", source))
    )
    assert registered.fixture() == WorkspaceFixtureSpec(repository="payments", revision=head)

    # A workflow target builds its own runtime for each trial, which takes no fixture.
    workflow = session_addition_workspace(trajectory, target=_workflow_target())
    assert (workflow.state, workflow.repository) == ("unsupported", "payments")
    assert "workflow" in workflow.message and workflow.fixture() is None


def _addition(destination: dict, *, mode: str = "case") -> dict:
    return {
        "mode": mode,
        "destination": destination,
        "case_id": "captured-claim",
        "case_name": "Captured claim",
    }


def _preview(client: TestClient, request: dict) -> dict:
    response = client.post(
        f"/api/evals/sessions/{_SESSION_ID}/additions/preview",
        headers=_AUTH_HEADERS,
        json=request,
    )
    assert response.status_code == 200, response.text
    return response.json()


def _publish(client: TestClient, request: dict, plan: dict, key: str) -> dict:
    response = client.post(
        f"/api/evals/sessions/{_SESSION_ID}/additions",
        headers={**_AUTH_HEADERS, "Idempotency-Key": key},
        json={
            **request,
            "expected_plan_revision": plan["revision"],
            "expected_source_revision": plan["source_revision"],
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_add_session_starts_the_case_at_the_sessions_commit(
    tmp_path: Path, addition_environment
) -> None:
    source, head = _checkout(tmp_path / "checkout")
    app = _app(tmp_path, source)
    _run(app, _SESSION_ID)
    server, _, store = addition_environment(app_override=app)
    fixture = WorkspaceFixtureSpec(repository="payments", revision=head)

    with TestClient(server) as client:
        request = _addition({"kind": "new", "suite_id": "claims", "name": "Claims"})
        plan = _preview(client, request)
        assert plan["workspace"] == {
            "state": "captured",
            "repository": "payments",
            "revision": head,
            "clean": True,
            "message": (
                f"The case starts from workspace source 'payments' at commit {head}, where "
                "the session's checkout was."
            ),
        }
        assert plan["case_definition"]["workspace"] == fixture.model_dump(mode="json")
        receipt = _publish(client, request, plan, "claims-from-session")
        assert receipt["plan"] == plan

        # A corpus suite gets the same fixture on its case.
        selector, _ = _destination(client, "corpus")
        corpus_plan = _preview(client, _addition({"kind": "existing", "selector": selector}))
        assert corpus_plan["workspace"] == plan["workspace"]
        assert corpus_plan["case_definition"]["workspace"] == fixture.model_dump(mode="json")

        # A scored capture doesn't run, so it has no workspace.
        scored = _preview(
            client, _addition({"kind": "existing", "selector": selector}, mode="score")
        )
        assert "workspace" not in scored and "workspace" not in scored["case_definition"]

        # The published suite keeps the fixture on its case.
        destination = EvalSuiteViewIdentity.model_validate(plan["destination"])
        saved = asyncio.run(store.load_authored_suite(destination.revision))
        assert saved is not None and saved.cases[0].workspace == fixture


def test_add_session_says_why_a_case_has_no_fixture_and_still_adds_it(
    tmp_path: Path, addition_environment
) -> None:
    source, head = _checkout(tmp_path / "checkout")
    (source / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    dirty = _app(tmp_path / "dirty", source)
    _run(dirty, _SESSION_ID)
    server, _, _ = addition_environment(app_override=dirty)
    with TestClient(server) as client:
        plan = _preview(client, _addition({"kind": "new", "suite_id": "claims", "name": "Claims"}))
    assert plan["workspace"]["state"] == "captured" and plan["workspace"]["clean"] is False
    assert "uncommitted changes" in plan["workspace"]["message"]
    assert plan["case_definition"]["workspace"]["revision"] == head

    # The target application never registered the source the session ran on.
    unregistered = _app(tmp_path / "unregistered", source, register=False)
    _run(unregistered, _SESSION_ID)
    server, _, _ = addition_environment(app_override=unregistered)
    with TestClient(server) as client:
        request = _addition({"kind": "new", "suite_id": "claims", "name": "Claims"})
        plan = _preview(client, request)
        assert plan["workspace"]["state"] == "unregistered"
        assert plan["workspace"]["revision"] == head
        assert "hasn't registered" in plan["workspace"]["message"]
        assert "register_workspace_source" in plan["workspace"]["message"]
        assert "workspace" not in plan["case_definition"]
        _publish(client, request, plan, "claims-without-source")

    # The session's environment recorded no revision.
    plain = tmp_path / "plain"
    plain.mkdir()
    unrecorded = _app(tmp_path / "unrecorded", plain)
    _run(unrecorded, _SESSION_ID)
    server, _, _ = addition_environment(app_override=unrecorded)
    with TestClient(server) as client:
        request = _addition({"kind": "new", "suite_id": "claims", "name": "Claims"})
        plan = _preview(client, request)
        assert plan["workspace"] == {
            "state": "unrecorded",
            "repository": None,
            "revision": None,
            "clean": None,
            "message": (
                "The session's environment recorded no workspace revision, so the case has no "
                "workspace fixture."
            ),
        }
        assert "workspace" not in plan["case_definition"]
        _publish(client, request, plan, "claims-without-revision")
