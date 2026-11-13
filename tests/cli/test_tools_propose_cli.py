"""`cayu tools propose`: one exact tool call from the terminal, through exposure and policy."""

from __future__ import annotations

import asyncio
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

import cayu.cli.tools as tools_cli
from cayu import AgentSpec, CayuApp, Message
from cayu.cli import main
from cayu.environments import (
    EnvironmentFactory,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    EnvironmentSpec,
)
from cayu.evals.testing import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.sessions.requests import ForkSessionRequest, ResumeRequest, RunRequest
from cayu.storage import SQLiteSessionStore
from cayu.tools import StaticToolExposurePolicy
from cayu.tools.base import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.tools.policy import (
    TOOL_POLICY_DRY_RUN_METADATA_KEY,
    ToolPolicy,
    ToolPolicyRequest,
    ToolPolicyResult,
)
from cayu.vaults.redaction import SecretRedactor

RAN: list[tuple[str, dict]] = []
SEEN: list[dict] = []


def _tool(name: str, effect: ToolEffect, *, fails: bool = False) -> Tool:
    class _Tool(Tool):
        spec = ToolSpec(
            name=name,
            description=f"The {name} tool.",
            input_schema={
                "type": "object",
                "properties": {"repository": {"type": "string"}, "number": {"type": "integer"}},
            },
            effect=effect,
            max_terminal_payload_bytes=64 * 1024,
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            RAN.append((name, dict(args)))
            if fails:
                raise RuntimeError("the remote said no")
            return ToolResult(content=f"{name} done.")

    return _Tool()


class _Policy(ToolPolicy):
    """Merge needs approval, publish is denied, everything else is allowed."""

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        SEEN.append(dict(request.metadata))
        if request.tool_name == "merge_pull_request":
            return ToolPolicyResult(decision="require_approval", reason="A person merges.")
        if request.tool_name == "publish":
            return ToolPolicyResult(decision="deny", reason="Publishing is off.")
        return ToolPolicyResult(decision="allow")


def _build(path) -> CayuApp:
    app = CayuApp(session_store=SQLiteSessionStore(path), enable_logging=False)
    app.register_agent(
        AgentSpec(name="shepherd", model="unconfigured-model"),
        tools=[
            _tool("look_up", ToolEffect.NONE),
            _tool("label", ToolEffect.EXTERNAL),
            _tool("publish", ToolEffect.EXTERNAL),
            _tool("flaky", ToolEffect.NONE, fails=True),
            _tool("hidden", ToolEffect.NONE),
        ],
        proposal_tools=[_tool("merge_pull_request", ToolEffect.EXTERNAL)],
        tool_exposure_policy=StaticToolExposurePolicy(
            profile_id="visible", tools=["look_up", "label", "publish", "flaky"]
        ),
        tool_policy=_Policy(),
    )
    return app


@pytest.fixture
def project(tmp_path, monkeypatch):
    RAN.clear()
    SEEN.clear()
    path = tmp_path / "sessions.sqlite"
    monkeypatch.setattr(
        tools_cli,
        "resolve_project",
        lambda target, **_kwargs: SimpleNamespace(
            root=tmp_path, target="app:build", service_target=None
        ),
    )
    monkeypatch.setattr(tools_cli, "project_context", lambda _root: nullcontext())
    monkeypatch.setattr(tools_cli, "build_project_app", lambda target, **_kwargs: _build(path))
    return path


def _propose(capsys, *argv: str) -> tuple[int, dict]:
    code = main(["tools", "propose", *argv, "--json"])
    out = capsys.readouterr().out.strip().splitlines()
    return code, json.loads(out[-1]) if out else {}


def _session(path, session_id: str):
    async def load():
        store = SQLiteSessionStore(path)
        try:
            session = await store.load(session_id)
            return session, [] if session is None else await store.load_events(session_id)
        finally:
            await store.close()

    return asyncio.run(load())


def test_a_proposal_only_tool_waits_for_approval_as_the_named_person(project, capsys) -> None:
    code, payload = _propose(
        capsys,
        "shepherd",
        "merge_pull_request",
        "--args",
        '{"repository": "o/a", "number": 12}',
        "--session-id",
        "merge-12",
        "--as",
        "alice",
        "--yes",
    )
    assert code == 0
    assert payload["outcome"]["status"] == "pending_approval"
    assert RAN == []
    session, _events = _session(project, "merge-12")
    # The session's provenance names who proposed it (session.started carries no origin).
    assert session.invocation.origin.subject == "alice"
    assert session.invocation.origin.trust == "host_asserted"


def test_each_outcome_sets_the_exit_status(project, capsys, tmp_path) -> None:
    arguments = tmp_path / "args.json"
    arguments.write_text('{"repository": "o/a"}')
    code, payload = _propose(capsys, "shepherd", "look_up", "--args", f"@{arguments}")
    assert (code, payload["outcome"]["status"]) == (0, "completed")
    assert RAN == [("look_up", {"repository": "o/a"})]
    code, payload = _propose(capsys, "shepherd", "publish", "--yes")
    assert (code, payload["outcome"]["status"]) == (1, "blocked")
    assert payload["outcome"]["reason"] == "Publishing is off."
    code, payload = _propose(capsys, "shepherd", "flaky")
    assert (code, payload["outcome"]["status"]) == (1, "failed")


def test_an_ordinary_tool_that_is_not_exposed_is_blocked(project, capsys) -> None:
    code, payload = _propose(capsys, "shepherd", "hidden", "--yes")
    assert code == 1
    assert payload["outcome"]["status"] == "blocked"
    assert ("hidden", {}) not in RAN


def test_a_dry_run_records_nothing_and_tells_the_policy(project, capsys) -> None:
    code, payload = _propose(
        capsys, "shepherd", "merge_pull_request", "--dry-run", "--session-id", "dry"
    )
    assert code == 0
    assert payload["dry_run"] is True
    assert payload["preview"]["decision"] == "require_approval"
    assert payload["preview"]["proposal_only"] is True
    assert SEEN and SEEN[-1]["cayu:dry_run"] is True
    assert _session(project, "dry")[0] is None
    code, payload = _propose(capsys, "shepherd", "hidden", "--dry-run")
    assert (code, payload["preview"]["blocked_by"]) == (1, "tool_exposure")
    code, payload = _propose(capsys, "shepherd", "publish", "--dry-run")
    assert (code, payload["preview"]["blocked_by"]) == (1, "tool_policy")


def test_the_session_id_makes_the_command_idempotent(project, capsys) -> None:
    argv = (
        "shepherd",
        "merge_pull_request",
        "--args",
        '{"number": 1}',
        "--session-id",
        "m-1",
        "--yes",
    )
    first_code, first = _propose(capsys, *argv)
    again_code, again = _propose(capsys, *argv)
    assert (first_code, again_code) == (0, 0)
    assert first["existing"] is False and again["existing"] is True
    assert again["outcome"]["approval_id"] == first["outcome"]["approval_id"]
    _session_record, events = _session(project, "m-1")
    assert sum(event.type.value == "tool.call.approval_requested" for event in events) == 1
    code, payload = _propose(
        capsys, "shepherd", "merge_pull_request", "--args", '{"number": 2}', "--session-id", "m-1"
    )
    assert code == 1
    assert "different call" in payload["error"]["message"]


def test_an_allowed_external_call_needs_yes_without_a_terminal(project, capsys) -> None:
    code, payload = _propose(capsys, "shepherd", "label", "--session-id", "label-1")
    assert code == 1
    assert "--yes" in payload["error"]["message"]
    assert RAN == []
    assert _session(project, "label-1")[0] is None
    code, payload = _propose(capsys, "shepherd", "label", "--yes")
    assert (code, payload["outcome"]["status"]) == (0, "completed")
    assert RAN == [("label", {})]


class _PreviewOnlyApproval(ToolPolicy):
    """Asks for approval in a preview and allows the real call, as a live check can."""

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        if request.metadata.get(TOOL_POLICY_DRY_RUN_METADATA_KEY) is True:
            return ToolPolicyResult(decision="require_approval", reason="Preview.")
        return ToolPolicyResult(decision="allow")


def test_an_external_call_needs_yes_whatever_the_preview_said(project, capsys, monkeypatch) -> None:
    def build(_target, **_kwargs) -> CayuApp:
        app = CayuApp(session_store=SQLiteSessionStore(project), enable_logging=False)
        app.register_agent(
            AgentSpec(name="shepherd", model="unconfigured-model"),
            tools=[_tool("label", ToolEffect.EXTERNAL)],
            tool_policy=_PreviewOnlyApproval(),
        )
        return app

    monkeypatch.setattr(tools_cli, "build_project_app", build)
    code, payload = _propose(capsys, "shepherd", "label", "--session-id", "label-2")
    assert code == 1
    assert "--yes" in payload["error"]["message"]
    assert "would wait for approval" in payload["error"]["message"]
    assert RAN == []
    assert _session(project, "label-2")[0] is None


def test_only_a_preview_can_tell_the_policy_it_is_a_dry_run(project) -> None:
    for build in (
        lambda: RunRequest(
            agent_name="shepherd",
            messages=[Message.text("user", "Hi")],
            metadata={TOOL_POLICY_DRY_RUN_METADATA_KEY: True},
        ),
        lambda: ForkSessionRequest(
            source_session_id="chat", metadata={TOOL_POLICY_DRY_RUN_METADATA_KEY: True}
        ),
    ):
        with pytest.raises(ValueError, match="dry-run authority"):
            build()

    async def scenario():
        app = _build(project)
        app.register_provider(
            ScriptedModelProvider(
                [
                    [ModelStreamEvent.text_delta("Hi."), ModelStreamEvent.completed({})],
                    [
                        ModelStreamEvent.tool_call(id="call-1", name="look_up", arguments={}),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ],
                    [ModelStreamEvent.text_delta("Done."), ModelStreamEvent.completed({})],
                ]
            ),
            default=True,
        )
        with pytest.raises(ValueError, match="dry-run authority"):
            await app.propose_tool_call(
                "shepherd", "look_up", {}, metadata={TOOL_POLICY_DRY_RUN_METADATA_KEY: True}
            )
        async for _event in app.run(
            RunRequest(
                agent_name="shepherd", session_id="chat", messages=[Message.text("user", "Hi")]
            )
        ):
            pass
        # A resume's metadata reaches the policy, but admission drops a forged marker.
        async for _event in app.resume(
            ResumeRequest(
                session_id="chat",
                messages=[Message.text("user", "Look it up.")],
                metadata={TOOL_POLICY_DRY_RUN_METADATA_KEY: True, "ticket": "T-1"},
            )
        ):
            pass
        seen_by_real_call = dict(SEEN[-1])
        preview = await app.preview_tool_call(
            "shepherd", "look_up", {}, metadata={TOOL_POLICY_DRY_RUN_METADATA_KEY: False}
        )
        await app.aclose()
        return seen_by_real_call, preview

    seen_by_real_call, preview = asyncio.run(scenario())
    assert RAN == [("look_up", {})]
    assert seen_by_real_call.get("ticket") == "T-1"
    assert TOOL_POLICY_DRY_RUN_METADATA_KEY not in seen_by_real_call
    assert preview.decision == "allow"
    assert SEEN[-1][TOOL_POLICY_DRY_RUN_METADATA_KEY] is True


class _ProductionNeedsApproval(ToolPolicy):
    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        SEEN.append({"environment_name": request.environment_name})
        if request.environment_name == "production":
            return ToolPolicyResult(decision="require_approval", reason="Production.")
        return ToolPolicyResult(decision="allow")


class _UnallocatedFactory(EnvironmentFactory):
    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        raise AssertionError("a preview must not allocate an environment")


def test_a_preview_sees_a_factory_backed_environment_by_name(project) -> None:
    async def scenario():
        app = CayuApp(session_store=SQLiteSessionStore(project), enable_logging=False)
        app.register_environment_factory(
            EnvironmentSpec(name="production"), _UnallocatedFactory(), default=True
        )
        app.register_agent(
            AgentSpec(name="shepherd", model="unconfigured-model"),
            tools=[_tool("label", ToolEffect.EXTERNAL)],
            tool_policy=_ProductionNeedsApproval(),
        )
        try:
            return await app.preview_tool_call("shepherd", "label", {})
        finally:
            await app.aclose()

    preview = asyncio.run(scenario())
    assert SEEN[-1] == {"environment_name": "production"}
    assert preview.decision == "require_approval"


class _LeakyDenial(ToolPolicy):
    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        return ToolPolicyResult(
            decision="deny",
            reason="Refused by https://svc:hunter2-secret@internal.example",
            metadata={"url": "https://svc:hunter2-secret@internal.example"},
        )


def test_a_preview_redacts_secrets_in_the_policy_reason(project) -> None:
    async def scenario():
        app = CayuApp(secret_redactor=SecretRedactor("hunter2-secret"), enable_logging=False)
        app.register_agent(
            AgentSpec(name="shepherd", model="unconfigured-model"),
            tools=[_tool("label", ToolEffect.EXTERNAL)],
            tool_policy=_LeakyDenial(),
        )
        try:
            return await app.preview_tool_call("shepherd", "label", {})
        finally:
            await app.aclose()

    preview = asyncio.run(scenario())
    assert preview.decision == "deny"
    assert "hunter2-secret" not in json.dumps(preview.model_dump(mode="json"))
    assert preview.reason is not None and preview.reason.startswith("Refused by https://svc:")


def test_read_tool_call_proposal_reads_a_proposal_and_refuses_other_sessions(project) -> None:
    async def scenario():
        app = _build(project)
        app.register_provider(
            ScriptedModelProvider(
                [[ModelStreamEvent.text_delta("Hi."), ModelStreamEvent.completed({})]]
            ),
            default=True,
        )
        pending = await app.propose_tool_call(
            "shepherd", "merge_pull_request", {"number": 3}, session_id="p-3"
        )
        read = await app.read_tool_call_proposal("p-3")
        missing = await app.read_tool_call_proposal("nobody")
        async for _event in app.run(
            RunRequest(
                agent_name="shepherd", session_id="chat", messages=[Message.text("user", "Hi")]
            )
        ):
            pass
        with pytest.raises(ValueError, match="not a tool-call proposal"):
            await app.read_tool_call_proposal("chat")
        await app.aclose()
        return pending, read, missing

    pending, read, missing = asyncio.run(scenario())
    assert read == pending
    assert missing is None
