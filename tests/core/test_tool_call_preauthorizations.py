from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from tests._session_provenance import fixture_session_invocation

from cayu import AgentSpec, CayuApp, Tool, ToolEffect, ToolResult, ToolSpec
from cayu.approvals import (
    TOOL_CALL_PREAUTHORIZATION_METADATA_KEY,
    TOOL_MANDATORY_APPROVAL_METADATA_KEY,
    InMemoryToolCallPreauthorizationStore,
    PreauthorizedToolCallPolicy,
    ToolAuthorizationGrantActor,
    ToolAuthorizationGrantConflict,
    ToolCallPreauthorization,
)
from cayu.runtime import check_manifest
from cayu.sessions.records import Session, SessionStatus
from cayu.tools import (
    AlwaysRequireApprovalToolPolicy,
    StaticToolPolicy,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
)

NOW = datetime(2026, 10, 1, tzinfo=UTC)
OPERATOR = ToolAuthorizationGrantActor(subject="operator")
EMAIL = {"email_id": "e-1", "digest": "abc", "to": ["ap@example.com"]}


def _session(session_id: str = "s", *, task_id: str | None = "task-1") -> Session:
    return Session(
        id=session_id,
        agent_name="mailer",
        provider_name="scripted",
        model="fake-model",
        invocation=fixture_session_invocation(session_id),
        status=SessionStatus.RUNNING,
        metadata={} if task_id is None else {"cayu:task_id": task_id},
    )


def _request(
    *,
    session: Session | None = None,
    arguments: dict | None = None,
    call: str = "call_1",
    tool: str = "send_email",
    agent: str = "mailer",
    metadata: dict | None = None,
) -> ToolPolicyRequest:
    return ToolPolicyRequest(
        session=session or _session(),
        agent=AgentSpec(name=agent, model="fake-model"),
        tool_name=tool,
        tool_call_id=call,
        tool_round_id="round_1",
        arguments=dict(EMAIL) if arguments is None else arguments,
        metadata=metadata or {},
    )


def _record(**overrides) -> ToolCallPreauthorization:
    values = {
        "preauthorization_id": "pre-1",
        "task_id": "task-1",
        "tool_name": "send_email",
        "arguments": EMAIL,
        "authorized_by": OPERATOR,
        "created_at": NOW,
        **overrides,
    }
    return ToolCallPreauthorization.for_call(**values)


def _policy(store, then: ToolPolicy | None = None) -> PreauthorizedToolCallPolicy:
    return PreauthorizedToolCallPolicy(
        store=store, tools=["send_email"], then=then, clock=lambda: NOW
    )


def test_exact_call_in_its_task_is_allowed_once_and_names_the_person() -> None:
    async def scenario() -> None:
        store = InMemoryToolCallPreauthorizationStore()
        await store.issue(_record())
        policy = _policy(store)
        session = _session()

        first = await policy.authorize(_request(session=session))
        assert first.decision is ToolPolicyDecision.ALLOW
        assert first.metadata[TOOL_CALL_PREAUTHORIZATION_METADATA_KEY] == {
            "preauthorization_id": "pre-1",
            "task_id": "task-1",
            "authorized_by": {"subject": "operator"},
        }
        # A retry of the same execution is decided the same way.
        retry = await policy.authorize(_request(session=session))
        assert retry.decision is ToolPolicyDecision.ALLOW
        # Any other call, even with the same arguments, asks a person.
        again = await policy.authorize(_request(session=session, call="call_2"))
        assert again.decision is ToolPolicyDecision.REQUIRE_APPROVAL
        used = await store.get("pre-1")
        assert used.use is not None and used.use.tool_call_id == "call_1"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "request_kwargs",
    (
        {"arguments": {**EMAIL, "to": ["someone@example.com"]}},
        {"arguments": {**EMAIL, "extra": True}},
        {"session": _session(task_id="task-2")},
        {"session": _session(task_id=None)},
        {"agent": "other"},
    ),
)
def test_anything_but_the_exact_call_in_its_task_asks_a_person(request_kwargs) -> None:
    async def scenario() -> None:
        store = InMemoryToolCallPreauthorizationStore()
        await store.issue(_record(agent_name="mailer"))
        result = await _policy(store).authorize(_request(**request_kwargs))
        assert result.decision is ToolPolicyDecision.REQUIRE_APPROVAL
        assert (await store.get("pre-1")).use is None

    asyncio.run(scenario())


def test_preauthorization_never_overrides_deny_mandatory_or_expiry() -> None:
    class Mandatory(ToolPolicy):
        async def authorize(self, request):
            return ToolPolicyResult(
                decision=ToolPolicyDecision.REQUIRE_APPROVAL,
                metadata={TOOL_MANDATORY_APPROVAL_METADATA_KEY: True},
            )

    class Forging(ToolPolicy):
        async def authorize(self, request):
            return ToolPolicyResult(
                decision=ToolPolicyDecision.DENY,
                metadata={TOOL_CALL_PREAUTHORIZATION_METADATA_KEY: {"forged": True}},
            )

    async def scenario() -> None:
        store = InMemoryToolCallPreauthorizationStore()
        await store.issue(_record())
        denied = await _policy(store, StaticToolPolicy(deny=["send_email"])).authorize(_request())
        assert denied.decision is ToolPolicyDecision.DENY
        forged = await _policy(store, Forging()).authorize(_request())
        assert TOOL_CALL_PREAUTHORIZATION_METADATA_KEY not in forged.metadata
        mandatory = await _policy(store, Mandatory()).authorize(_request())
        assert mandatory.decision is ToolPolicyDecision.REQUIRE_APPROVAL
        assert (await store.get("pre-1")).use is None

        expiring = InMemoryToolCallPreauthorizationStore()
        await expiring.issue(_record(lifetime_seconds=60))
        later = PreauthorizedToolCallPolicy(
            store=expiring, tools=["send_email"], clock=lambda: NOW + timedelta(seconds=60)
        )
        assert (await later.authorize(_request())).decision is ToolPolicyDecision.REQUIRE_APPROVAL

        revoked = InMemoryToolCallPreauthorizationStore()
        await revoked.issue(_record())
        await revoked.revoke("pre-1", revoked_by=OPERATOR, now=NOW)
        result = await _policy(revoked).authorize(_request())
        assert result.decision is ToolPolicyDecision.REQUIRE_APPROVAL

    asyncio.run(scenario())


def test_dry_run_and_failing_store_do_not_use_the_record() -> None:
    class Broken(InMemoryToolCallPreauthorizationStore):
        async def use(self, **kwargs):
            raise RuntimeError("store unavailable")

    async def scenario() -> None:
        store = InMemoryToolCallPreauthorizationStore()
        await store.issue(_record())
        preview = await _policy(store).authorize(_request(metadata={"cayu:dry_run": True}))
        assert preview.decision is ToolPolicyDecision.ALLOW
        assert (await store.get("pre-1")).use is None

        broken = Broken()
        await broken.issue(_record())
        result = await _policy(broken).authorize(_request())
        assert result.decision is ToolPolicyDecision.REQUIRE_APPROVAL

    asyncio.run(scenario())


def test_issue_is_idempotent_and_records_are_exact() -> None:
    async def scenario() -> None:
        store = InMemoryToolCallPreauthorizationStore()
        first = await store.issue(_record())
        assert await store.issue(_record()) == first
        with pytest.raises(ToolAuthorizationGrantConflict):
            await store.issue(_record(arguments={**EMAIL, "to": ["x@example.com"]}))
        await _policy(store).authorize(_request())
        with pytest.raises(ToolAuthorizationGrantConflict):
            await store.revoke("pre-1", revoked_by=OPERATOR, now=NOW)

    asyncio.run(scenario())
    with pytest.raises(ValueError, match="arguments_digest"):
        ToolCallPreauthorization(
            preauthorization_id="p",
            task_id="t",
            tool_name="send_email",
            arguments_digest="sha256:short",
            authorized_by=OPERATOR,
            created_at=NOW,
        )


class _SendTool(Tool):
    spec = ToolSpec(
        name="send_email",
        effect=ToolEffect.EXTERNAL,
        input_schema={"type": "object"},
        max_terminal_payload_bytes=65_536,
    )

    async def run(self, ctx, args):
        return ToolResult(content="sent")


def test_cayu_check_reads_the_policy_as_conditional_coverage() -> None:
    app = CayuApp(enable_logging=False)
    app.register_agent(
        AgentSpec(name="mailer", model="model"),
        tools=[_SendTool()],
        tool_policy=_policy(
            InMemoryToolCallPreauthorizationStore(),
            AlwaysRequireApprovalToolPolicy(tools=["send_email"]),
        ),
    )
    manifest = app.describe()
    assert manifest.agents[0].tools[0].policy_coverage == "conditional"
    codes = {item.code for item in check_manifest(manifest).diagnostics}
    assert not codes & {"EXTERNAL_TOOL_COVERAGE_UNKNOWN", "EXTERNAL_TOOL_UNGUARDED"}


def test_mandatory_control_of_a_wrapped_grant_policy_still_asks_a_person() -> None:
    from cayu.approvals import (
        InMemoryToolAuthorizationGrantStore,
        ToolAuthorizationGrantControl,
        ToolAuthorizationGrantPolicy,
    )

    grants = ToolAuthorizationGrantPolicy(
        policy_id="p",
        store=InMemoryToolAuthorizationGrantStore(),
        grantable={"send_email": "tool_call"},
        mandatory_controls=[ToolAuthorizationGrantControl(tool_name="send_email")],
    )

    async def scenario() -> None:
        store = InMemoryToolCallPreauthorizationStore()
        await store.issue(_record())
        result = await _policy(store, grants).authorize(_request())
        assert result.decision is ToolPolicyDecision.REQUIRE_APPROVAL
        assert (await store.get("pre-1")).use is None

    asyncio.run(scenario())

    app = CayuApp(enable_logging=False)
    app.register_agent(
        AgentSpec(name="mailer", model="model"),
        tools=[_SendTool()],
        tool_policy=_policy(InMemoryToolCallPreauthorizationStore(), grants),
    )
    assert app.describe().agents[0].tools[0].policy_coverage == "approval_required"


def test_a_batched_call_matches_by_its_whole_arguments() -> None:
    # A batched built-in passes one argument set per target, with the whole call beside it.
    whole = {"files": [{"path": "a.txt"}, {"path": "b.txt"}]}

    def target(path: str) -> ToolPolicyRequest:
        request = _request(tool="read_file", arguments={**whole, "path": path}, session=session)
        request.call_arguments = whole
        return request

    session = _session()

    async def scenario() -> None:
        store = InMemoryToolCallPreauthorizationStore()
        await store.issue(_record(tool_name="read_file", arguments=whole))
        policy = PreauthorizedToolCallPolicy(store=store, tools=["read_file"], clock=lambda: NOW)
        first = await policy.authorize(target("a.txt"))
        second = await policy.authorize(target("b.txt"))
        assert first.decision is second.decision is ToolPolicyDecision.ALLOW
        assert first.metadata == second.metadata
        other = await policy.authorize(_request(tool="read_file", arguments=whole, call="call_2"))
        assert other.decision is ToolPolicyDecision.REQUIRE_APPROVAL

    asyncio.run(scenario())


def test_the_runtime_gives_policies_the_whole_call_of_a_projected_call() -> None:
    from cayu.tools.files import ReadFileTool

    seen: list[ToolPolicyRequest] = []

    class Recording(ToolPolicy):
        async def authorize(self, request):
            seen.append(request)
            return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)

    app = CayuApp(enable_logging=False)
    app.register_agent(
        AgentSpec(name="reader", model="model"), tools=[ReadFileTool()], tool_policy=Recording()
    )
    whole = {"files": [{"path": "a.txt"}, {"path": "b.txt"}]}
    asyncio.run(app.preview_tool_call("reader", "read_file", whole))
    assert [request.arguments.get("path") for request in seen] == ["a.txt", "b.txt"]
    assert all(request.call_arguments == whole for request in seen)
    seen.clear()
    asyncio.run(app.preview_tool_call("reader", "read_file", {"path": "a.txt"}))
    assert [request.call_arguments for request in seen] == [None]


def test_an_outer_grant_policys_gate_specific_mandatory_control_still_asks() -> None:
    from cayu.approvals import (
        InMemoryToolAuthorizationGrantStore,
        ToolAuthorizationGrantControl,
        ToolAuthorizationGrantPolicy,
    )

    async def scenario() -> None:
        store = InMemoryToolCallPreauthorizationStore()
        await store.issue(_record())
        inner = _policy(store, AlwaysRequireApprovalToolPolicy(approval_gate="email"))
        alone = await inner.authorize(_request())
        assert alone.decision is ToolPolicyDecision.ALLOW
        assert alone.metadata["cayu:approval_gate"] == "email"
        fresh = InMemoryToolCallPreauthorizationStore()
        await fresh.issue(_record())
        outer = ToolAuthorizationGrantPolicy(
            policy_id="p",
            store=InMemoryToolAuthorizationGrantStore(),
            grantable={"send_email": "tool_call"},
            mandatory_controls=[
                ToolAuthorizationGrantControl(tool_name="send_email", approval_gate="email")
            ],
            then=_policy(fresh, AlwaysRequireApprovalToolPolicy(approval_gate="email")),
        )
        result = await outer.authorize(_request())
        assert result.decision is ToolPolicyDecision.REQUIRE_APPROVAL

    asyncio.run(scenario())


def test_issuing_again_later_is_the_same_issue() -> None:
    async def scenario() -> None:
        store = InMemoryToolCallPreauthorizationStore()
        first = await store.issue(_record(lifetime_seconds=60))
        later = _record(lifetime_seconds=60, created_at=NOW + timedelta(seconds=5))
        assert await store.issue(later) == first
        with pytest.raises(ToolAuthorizationGrantConflict):
            await store.issue(_record(lifetime_seconds=120))

    asyncio.run(scenario())
