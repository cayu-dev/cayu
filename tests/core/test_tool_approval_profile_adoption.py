"""A pending tool approval can still be settled after the application is upgraded."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import cayu.runtime._execution_profile_continuation as execution_profile_continuation
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.tools import (
    ResolutionActor,
    ResolutionActorSource,
    ToolApprovalDecision,
    ToolApprovalProfileAdoptionRefused,
    ToolApprovalRequest,
)
from cayu.evals.testing import ScriptedModelProvider
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.providers.base import ModelRequest, ModelStreamEvent
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.execution_profiles import (
    EXECUTION_PROFILE_METADATA_KEY,
    ExecutionProfileAdoptionIntent,
    ExecutionProfileAdoptionRejected,
    ExecutionProfileAuthorityDecision,
    ExecutionProfileMismatchError,
    ExecutionProfilePolicy,
    ExecutionProfilePolicyAction,
    ExecutionProfilePolicyRequest,
    ExecutionProfilePolicyResult,
)
from cayu.sessions import _pending_approval_reader as pending_approval_reader
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.requests import RunRequest
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.base import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.tools.policy import ToolPolicy, ToolPolicyDecision, ToolPolicyRequest, ToolPolicyResult


class _FileBug(Tool):
    def __init__(self, behavior_version: str = "1") -> None:
        self.calls: list[dict[str, Any]] = []
        self.spec = ToolSpec(
            name="file_bug",
            description="File a bug in the tracker.",
            input_schema={"type": "object", "properties": {"title": {"type": "string"}}},
            effect=ToolEffect.EXTERNAL,
            max_terminal_payload_bytes=1024,
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name="tests:file-bug",
                behavior_version=behavior_version,
                implementation_version="1",
            ),
        )
        super().__init__()

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.calls.append(args)
        return ToolResult(content="filed")


class _AskFirst(ToolPolicy):
    def __init__(self, decision: ToolPolicyDecision = ToolPolicyDecision.REQUIRE_APPROVAL) -> None:
        self._decision = decision

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:ask-first",
            behavior_version=self._decision.value,
            implementation_version="1",
        )

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        return ToolPolicyResult(decision=self._decision)


class _AdoptExplicitRequests(ExecutionProfilePolicy):
    def __init__(self, action: ExecutionProfilePolicyAction = ExecutionProfilePolicyAction.ADOPT):
        self.action = action
        self.requests: list[ExecutionProfilePolicyRequest] = []

    @property
    def identity(self) -> str:
        return "tests:adopt-explicit-requests:v1"

    async def decide(self, request: ExecutionProfilePolicyRequest) -> ExecutionProfilePolicyResult:
        self.requests.append(request)
        if request.intent is None:
            return ExecutionProfilePolicyResult(
                action=ExecutionProfilePolicyAction.REJECT, reason="No adoption was requested."
            )
        return ExecutionProfilePolicyResult(
            action=self.action,
            reason=request.intent.reason,
            authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
        )


class _Reporter(ScriptedModelProvider):
    """Proposes the bug, then reports it filed; the same identity in every process."""

    def __init__(self) -> None:
        super().__init__(response_factory=self._respond, name="fake")

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:reporter", behavior_version="1", implementation_version="1"
        )

    @staticmethod
    def _respond(request: ModelRequest) -> list[ModelStreamEvent]:
        if any(message.role == "tool" for message in request.messages):
            return [ModelStreamEvent.text_delta("Filed."), ModelStreamEvent.completed()]
        return [
            ModelStreamEvent.tool_call(
                id="call-1", name="file_bug", arguments={"title": "Crash on save"}
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]


def _app(
    store,
    *,
    tool: _FileBug | None = None,
    tool_policy: ToolPolicy | None = None,
    profile_policy: ExecutionProfilePolicy | None = None,
) -> CayuApp:
    app = CayuApp(
        session_store=store, execution_profile_policy=profile_policy, enable_logging=False
    )
    app.register_provider(
        _Reporter(),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="reporter", model="fake-model", system_prompt="File bugs."),
        tools=[tool or _FileBug()],
        tool_policy=tool_policy or _AskFirst(),
    )
    return app


def _use_cayu(monkeypatch, version: str) -> None:
    monkeypatch.setattr(execution_profile_continuation, "_runtime_version", lambda: version)


async def _collect(stream) -> list[Event]:
    return [event async for event in stream]


async def _pause(app: CayuApp, session_id: str = "bug-report") -> dict[str, Any]:
    events = await _collect(
        app.run(
            RunRequest(
                agent_name="reporter",
                session_id=session_id,
                messages=[Message.text("user", "Save crashes the editor.")],
            )
        )
    )
    return next(e for e in events if e.type == EventType.TOOL_CALL_APPROVAL_REQUESTED).payload


def _approve(requested: dict[str, Any], **changes) -> ToolApprovalRequest:
    return ToolApprovalRequest(
        session_id="bug-report",
        approval_id=requested["approval_id"],
        tool_round_id=requested["tool_round_id"],
        tool_call_id=requested["tool_call_id"],
        decision=ToolApprovalDecision.APPROVE,
        **changes,
    )


def _intent(key: str = "adopt-0.16") -> ExecutionProfileAdoptionIntent:
    return ExecutionProfileAdoptionIntent(
        idempotency_key=key,
        reason="The bug filer moved to Cayu 0.16 and a new tool version.",
        requested_by=ResolutionActor(subject="priya", source=ResolutionActorSource.REQUEST),
    )


def _still_pending(store, session_id: str = "bug-report") -> bool:
    checkpoint = asyncio.run(store.load_checkpoint(session_id))
    return pending_approval_reader.pending_approval_from_checkpoint(checkpoint) is not None


def _decided(events: list[Event]) -> list[Event]:
    return [e for e in events if e.type == EventType.SESSION_EXECUTION_PROFILE_DECIDED]


def _durable_store(store_kind: str, tmp_path, request):
    if store_kind == "memory":
        return InMemorySessionStore()
    if store_kind == "sqlite":
        return SQLiteSessionStore(tmp_path / "sessions.sqlite")
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresSessionStore

    return PostgresSessionStore(
        request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
    )


@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "postgres"])
def test_a_cayu_upgrade_no_longer_strands_a_pending_approval(
    monkeypatch, tmp_path, request, store_kind: str
) -> None:
    async def exercise() -> None:
        store = _durable_store(store_kind, tmp_path, request)
        try:
            _use_cayu(monkeypatch, "0.15.0")
            requested = await _pause(_app(store))

            # The application now runs on a new Cayu version and nothing else changed.
            _use_cayu(monkeypatch, "0.16.0")
            tool = _FileBug()
            events = await _collect(
                _app(store, tool=tool).resolve_tool_approval(_approve(requested))
            )

            assert tool.calls == [{"title": "Crash on save"}]
            [decided] = _decided(events)
            assert decided.payload["decision"] == "adopted"
            assert decided.payload["policy_identity"] == "cayu:runtime-upgrade-adoption:v1"
            assert decided.payload["changed_component_classes"] == ["runtime"]
            session = await store.load("bug-report")
            assert session is not None
            assert session.runtime_version == "0.16.0"
            assert (
                session.metadata[EXECUTION_PROFILE_METADATA_KEY]["expected"]
                == decided.payload["candidate_profile"]
            )
            assert events[-1].type == EventType.SESSION_COMPLETED

            # The next approval on 0.16 needs no further decision.
            restarted = _FileBug()
            requested = await _pause(_app(store, tool=restarted), "bug-report-2")
            again = await _collect(
                _app(store, tool=restarted).resolve_tool_approval(
                    _approve(requested).model_copy(update={"session_id": "bug-report-2"})
                )
            )
            assert _decided(again) == []
            assert restarted.calls == [{"title": "Crash on save"}]
        finally:
            if not isinstance(store, InMemorySessionStore):
                await store.close()

    asyncio.run(exercise())


def test_an_upgrade_that_changes_a_tool_needs_an_explicit_adoption(monkeypatch) -> None:
    store = InMemorySessionStore()
    _use_cayu(monkeypatch, "0.15.0")
    requested = asyncio.run(_pause(_app(store)))
    _use_cayu(monkeypatch, "0.16.0")

    # Without an adoption request the approval keeps its profile and stays pending.
    tool = _FileBug(behavior_version="2")
    with pytest.raises(ExecutionProfileMismatchError) as mismatch:
        asyncio.run(_collect(_app(store, tool=tool).resolve_tool_approval(_approve(requested))))
    assert mismatch.value.boundary == "continuation"
    assert "`profile_adoption` field of the ToolApprovalRequest" in str(mismatch.value)
    assert tool.calls == []
    assert _still_pending(store)

    # An adoption request needs an application policy that adopts it.
    with pytest.raises(ExecutionProfileAdoptionRejected) as rejected:
        asyncio.run(
            _collect(
                _app(store, tool=tool).resolve_tool_approval(
                    _approve(requested, profile_adoption=_intent("adopt-without-policy"))
                )
            )
        )
    assert rejected.value.application_policy_configured is False
    assert tool.calls == []
    assert _still_pending(store)

    policy = _AdoptExplicitRequests()
    events = asyncio.run(
        _collect(
            _app(store, tool=tool, profile_policy=policy).resolve_tool_approval(
                _approve(requested, profile_adoption=_intent())
            )
        )
    )
    assert tool.calls == [{"title": "Crash on save"}]
    [policy_request] = policy.requests
    assert policy_request.intent == _intent()
    assert policy_request.authority_review_required is True
    assert set(policy_request.changed_component_classes) == {"runtime", "tool_implementations"}
    [decided] = _decided(events)
    assert decided.payload["decision"] == "adopted"
    assert decided.payload["actor"]["subject"] == "priya"


def test_a_policy_can_refuse_an_explicit_adoption(monkeypatch) -> None:
    store = InMemorySessionStore()
    _use_cayu(monkeypatch, "0.15.0")
    requested = asyncio.run(_pause(_app(store)))
    _use_cayu(monkeypatch, "0.16.0")
    tool = _FileBug(behavior_version="2")
    policy = _AdoptExplicitRequests(ExecutionProfilePolicyAction.REJECT)

    with pytest.raises(ExecutionProfileAdoptionRejected) as rejected:
        asyncio.run(
            _collect(
                _app(store, tool=tool, profile_policy=policy).resolve_tool_approval(
                    _approve(requested, profile_adoption=_intent())
                )
            )
        )

    assert rejected.value.boundary == "continuation"
    assert rejected.value.application_policy_configured is True
    assert tool.calls == []
    assert _still_pending(store)
    events = asyncio.run(store.load_events("bug-report"))
    [refusal] = [e for e in events if e.type == EventType.SESSION_EXECUTION_PROFILE_REJECTED]
    assert refusal.payload["policy_identity"] == "tests:adopt-explicit-requests:v1"


@pytest.mark.parametrize("change", ["policy_denies", "policy_allows", "tool_removed"])
def test_adoption_checks_the_paused_call_against_the_current_tool_and_policy(
    monkeypatch, change: str
) -> None:
    store = InMemorySessionStore()
    _use_cayu(monkeypatch, "0.15.0")
    requested = asyncio.run(_pause(_app(store)))
    _use_cayu(monkeypatch, "0.16.0")
    tool = _FileBug()
    policy = _AdoptExplicitRequests()
    app = _app(
        store,
        tool=tool,
        tool_policy=_AskFirst(
            ToolPolicyDecision.DENY if change == "policy_denies" else ToolPolicyDecision.ALLOW
        ),
        profile_policy=policy,
    )
    if change == "tool_removed":
        app = CayuApp(session_store=store, execution_profile_policy=policy, enable_logging=False)
        app.register_provider(_Reporter(), default=True)
        app.register_agent(
            AgentSpec(name="reporter", model="fake-model", system_prompt="File bugs."),
            tool_policy=_AskFirst(),
        )

    approve = app.resolve_tool_approval(_approve(requested, profile_adoption=_intent()))
    if change == "policy_allows":
        # A policy that now allows the call needs no more than the approval it got.
        asyncio.run(_collect(approve))
        assert tool.calls == [{"title": "Crash on save"}]
        return
    # A removed tool already fails the current profile; a refusing policy fails the
    # re-check. Either way nothing ran and the approval is still pending.
    with pytest.raises(
        ValueError if change == "tool_removed" else ToolApprovalProfileAdoptionRefused
    ):
        asyncio.run(_collect(approve))
    assert tool.calls == []
    assert _still_pending(store)
    events = asyncio.run(store.load_events("bug-report"))
    assert _decided(events) == []

    # The approval can still be settled where it was requested.
    _use_cayu(monkeypatch, "0.15.0")
    original = _FileBug()
    asyncio.run(_collect(_app(store, tool=original).resolve_tool_approval(_approve(requested))))
    assert original.calls == [{"title": "Crash on save"}]


def test_changed_arguments_are_approved_under_the_adopted_profile(monkeypatch) -> None:
    store = InMemorySessionStore()
    _use_cayu(monkeypatch, "0.15.0")
    requested = asyncio.run(_pause(_app(store)))
    _use_cayu(monkeypatch, "0.16.0")
    tool = _FileBug()

    asyncio.run(
        _collect(
            _app(store, tool=tool).resolve_tool_approval(
                _approve(requested, arguments={"title": "Editor crashes on save"})
            )
        )
    )

    assert tool.calls == [{"title": "Editor crashes on save"}]


def test_a_continued_admission_moves_profiles_only_with_its_exact_adoption(monkeypatch) -> None:
    import cayu.runtime._pending_profile_adoption as pending_profile_adoption
    from cayu.sessions._invocation_lifecycle import AdmitInvocationCommand

    prepared: list[AdmitInvocationCommand] = []
    prepare = pending_profile_adoption.prepare_adopting_continuation_command

    def capture(*args, **kwargs):
        prepared.append(prepare(*args, **kwargs))
        return prepared[-1]

    monkeypatch.setattr(pending_profile_adoption, "prepare_adopting_continuation_command", capture)
    store = InMemorySessionStore()
    _use_cayu(monkeypatch, "0.15.0")
    requested = asyncio.run(_pause(_app(store)))
    _use_cayu(monkeypatch, "0.16.0")
    asyncio.run(_collect(_app(store).resolve_tool_approval(_approve(requested))))

    [command] = prepared
    fields = {name: getattr(command, name) for name in type(command).model_fields}
    decision = command.execution_profile_decision
    assert decision is not None
    assert command.continued_interaction_id == command.target_active_profile.interaction_id
    assert command.expected_active_profile is not None
    assert decision.expected_profile == command.expected_active_profile.profile
    assert decision.candidate_profile == command.target_active_profile.profile

    # Without the adoption decision the paused interaction keeps its profile.
    with pytest.raises(ValueError, match="cannot replace its execution profile"):
        AdmitInvocationCommand(
            **(fields | {"execution_profile_decision": None, "adopted_runtime_identity": None})
        )
    # A decision that names other profiles is refused too.
    with pytest.raises(ValueError):
        AdmitInvocationCommand(
            **(
                fields
                | {
                    "execution_profile_decision": decision.model_copy(
                        update={"candidate_profile": decision.expected_profile}
                    )
                }
            )
        )


def test_a_compatible_reuse_policy_keeps_the_strict_continuation_check(monkeypatch) -> None:
    store = InMemorySessionStore()
    _use_cayu(monkeypatch, "0.15.0")
    requested = asyncio.run(_pause(_app(store)))
    _use_cayu(monkeypatch, "0.16.0")
    tool = _FileBug()
    policy = _AdoptExplicitRequests(ExecutionProfilePolicyAction.COMPATIBLE_REUSE)

    with pytest.raises(ExecutionProfileMismatchError) as mismatch:
        asyncio.run(
            _collect(
                _app(store, tool=tool, profile_policy=policy).resolve_tool_approval(
                    _approve(requested, profile_adoption=_intent())
                )
            )
        )

    assert type(mismatch.value) is ExecutionProfileMismatchError
    assert tool.calls == []
    assert _still_pending(store)


class _RedescribedFileBug(_FileBug):
    def __init__(self) -> None:
        super().__init__()
        self.spec = self.spec.model_copy(update={"description": "File a bug, with labels."})


@pytest.mark.parametrize(
    ("decision", "adopt"),
    [
        (ToolApprovalDecision.APPROVE, True),
        (ToolApprovalDecision.DENY, True),
        (ToolApprovalDecision.DENY, False),
    ],
)
def test_a_changed_tool_catalogue_is_refused_before_anything_is_committed(
    monkeypatch, decision: ToolApprovalDecision, adopt: bool
) -> None:
    store = InMemorySessionStore()
    _use_cayu(monkeypatch, "0.15.0")
    requested = asyncio.run(_pause(_app(store)))
    _use_cayu(monkeypatch, "0.16.0")
    tool = _RedescribedFileBug()
    policy = _AdoptExplicitRequests()
    settle = _approve(requested, profile_adoption=_intent() if adopt else None)

    # Settling either way continues the round under the catalogue the model saw.
    # Without an adoption request it stays the ordinary continuation mismatch.
    with pytest.raises(
        ToolApprovalProfileAdoptionRefused if adopt else ExecutionProfileMismatchError,
        match="whether approved or denied" if adopt else "direct_tools",
    ):
        asyncio.run(
            _collect(
                _app(store, tool=tool, profile_policy=policy).resolve_tool_approval(
                    settle.model_copy(update={"decision": decision})
                )
            )
        )

    if adopt:
        assert policy.requests == []
    assert tool.calls == []
    assert _still_pending(store)
    assert _decided(asyncio.run(store.load_events("bug-report"))) == []


class _StopBeforeDispatch(InMemorySessionStore):
    """Stop once right before the approved call starts, after its claim is durable."""

    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.stopped = False

    async def append_events(self, session_id, events):
        if not self.stopped and any(e.type == EventType.TOOL_CALL_STARTED for e in events):
            self.stopped = True
            raise ConnectionError("the process stopped before the approved call was dispatched")
        await super().append_events(session_id, events)


class _RefuseEditedTitle(_AskFirst):
    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:refuse-edited-title", behavior_version="1", implementation_version="1"
        )

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        if request.arguments.get("title") == "Edited":
            return ToolPolicyResult(decision=ToolPolicyDecision.DENY, reason="Not this title.")
        return ToolPolicyResult(decision=ToolPolicyDecision.REQUIRE_APPROVAL)


def test_a_retried_edit_is_checked_with_the_arguments_its_claim_pinned(monkeypatch) -> None:
    store = _StopBeforeDispatch()
    _use_cayu(monkeypatch, "0.15.0")
    original = _app(store)
    requested = asyncio.run(_pause(original))
    edited = _approve(requested, arguments={"title": "Edited"})
    asyncio.run(_collect(original.resolve_tool_approval(edited)))
    assert store.stopped
    asyncio.run(original.drain_background_interruptions())

    # The same edited decision is retried after an upgrade whose policy refuses the edit.
    _use_cayu(monkeypatch, "0.16.0")
    tool = _FileBug()
    upgraded = _app(
        store, tool=tool, tool_policy=_RefuseEditedTitle(), profile_policy=_AdoptExplicitRequests()
    )
    with pytest.raises(ToolApprovalProfileAdoptionRefused, match="refuses the paused call"):
        asyncio.run(
            _collect(
                upgraded.resolve_tool_approval(
                    edited.model_copy(update={"profile_adoption": _intent()})
                )
            )
        )
    assert tool.calls == []
