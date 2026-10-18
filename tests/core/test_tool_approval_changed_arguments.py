"""Approving a tool call with changed arguments, through the public resolve entrance."""

from __future__ import annotations

import asyncio
import base64
import hashlib
from contextlib import aclosing
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import SecretStr, ValidationError
from tests.core.test_approval_lifecycle_execution_identities import _RecordingTool
from tests.core.test_tool_round_publication_failure_matrix import store_factory as store_factory

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu._validation import canonical_durable_json_bytes
from cayu.approvals.review import HumanReviewContext, HumanReviewReference
from cayu.approvals.tools import (
    MAX_APPROVED_TOOL_ARGUMENTS_BYTES,
    ToolApprovalArgumentsRefused,
    ToolApprovalDecision,
    ToolApprovalRequest,
    copy_tool_approval_request,
)
from cayu.environments import Environment, EnvironmentSpec
from cayu.evals import ScriptedModelProvider
from cayu.events import EventType
from cayu.messages import ToolCallPart, ToolResultPart
from cayu.observability.hooks import BeforeToolCallDecision, RuntimeHook
from cayu.providers import ModelStreamEvent
from cayu.runtime import _approval_support as approval_support
from cayu.runtime._tool_effect_state import ToolEffectStateOwner
from cayu.runtime._tool_execution import APPROVED_ARGUMENTS_NOTICE
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.sessions import SupervisionEventKind, SupervisionEventQuery
from cayu.sessions import _pending_approval_reader as pending_approval_reader
from cayu.sessions._checkpoint_mutations import runtime_publication_checkpoint_value_digest
from cayu.tools.policy import ToolPolicy, ToolPolicyDecision, ToolPolicyResult
from cayu.vaults import StaticVault
from cayu.vaults.redaction import SecretRedactor

_TERMINALS = {
    EventType.TOOL_CALL_COMPLETED,
    EventType.TOOL_CALL_FAILED,
    EventType.TOOL_CALL_BLOCKED,
    EventType.TOOL_CALL_APPROVAL_DENIED,
}
_SECRET = "edit-secret-7f3a"


class _EditFaultStore:
    """Fail once on a chosen durable write, above the real backend boundary."""

    def __init__(self, *args, fault="none", **kwargs):
        super().__init__(*args, **kwargs)
        self.fault = fault
        self.faulted = False

    async def append_events(self, session_id, events):
        start = any(
            event.type is EventType.TOOL_CALL_STARTED
            and event.payload.get("tool_call_id") == "pause"
            for event in events
        )
        if self.fault == "pause-start" and start and not self.faulted:
            self.faulted = True
            raise ConnectionError("the process stopped before the edited call was dispatched")
        await super().append_events(session_id, events)
        terminal = any(event.type in _TERMINALS for event in events)
        if self.fault == "terminal-ack" and terminal and not self.faulted:
            self.faulted = True
            raise ConnectionError("edited terminal acknowledgement lost")


@pytest.fixture
def edit_store_factory(store_factory):
    key = base64.urlsafe_b64encode(bytes([47]) * 32).decode().rstrip("=")
    codec = PublicAuthorityAliasCodec(
        PublicAuthorityAliasKeyring(active_key_id="test", keys={"test": SecretStr(key)})
    )

    def open_store(**options):
        return store_factory(_EditFaultStore, public_authority_alias_codec=codec, **options)

    return open_store


_IDENTITY = {"behavior_version": "1", "implementation_version": "1"}


class _EditTool(_RecordingTool):
    # Declared identities let a restarted app continue the same session.
    spec = _RecordingTool.spec.model_copy(
        update={
            "execution_profile_identity": ExecutionProfileBehaviorIdentity(
                name="tests:edit-tool", **_IDENTITY
            )
        }
    )


class _EditPolicy(ToolPolicy):
    """Gate ``pause`` behind a payments approval; its value selects other outcomes."""

    def __init__(self) -> None:
        self.requests = []

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(name="tests:edit-policy", **_IDENTITY)

    async def authorize(self, request):
        self.requests.append(request)
        if request.tool_call_id == "denied":
            return ToolPolicyResult(decision=ToolPolicyDecision.DENY, reason="Denied sibling.")
        if request.tool_call_id != "pause":
            return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)
        # Authorization depends on the run's context, never the approver's.
        if request.metadata.get("tenant") != "acme":
            return ToolPolicyResult(decision=ToolPolicyDecision.DENY, reason="Wrong tenant.")
        if request.metadata.get("route") == "treasury":
            return ToolPolicyResult(
                decision=ToolPolicyDecision.REQUIRE_APPROVAL,
                metadata={"cayu:approval_gate": "treasury"},
            )
        value = request.arguments.get("value")
        if value == "forbidden":
            return ToolPolicyResult(decision=ToolPolicyDecision.DENY, reason="Never allowed.")
        if value == "allowed-now":
            return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)
        gate = "treasury" if value == "large" else "payments"
        return ToolPolicyResult(
            decision=ToolPolicyDecision.REQUIRE_APPROVAL,
            metadata={"cayu:approval_gate": gate},
        )


_SIBLING_VALUES = {"allowed": "second", "denied": "never"}


def _runtime(store, *, dynamic=False, siblings=("allowed", "denied"), restarted=False, hooks=()):
    tool = _EditTool()
    policy = _EditPolicy()
    round_turn = [
        [
            ModelStreamEvent.tool_call(
                id="pause", name=tool.spec.name, arguments={"value": "first"}
            ),
            *(
                ModelStreamEvent.tool_call(
                    id=call_id, name=tool.spec.name, arguments={"value": _SIBLING_VALUES[call_id]}
                )
                for call_id in siblings
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ],
    ]
    done_turn = [
        ModelStreamEvent.text_delta("done"),
        ModelStreamEvent.completed({"finish_reason": "stop"}),
    ]
    # A restarted process only serves the continuation after the round.
    provider = ScriptedModelProvider([*([] if restarted else round_turn), done_turn])
    app = CayuApp(
        session_store=store,
        enable_logging=False,
        secret_redactor=SecretRedactor(_SECRET),
        runtime_hooks=list(hooks),
    )
    app.register_provider(provider, default=True)
    if dynamic:
        app.register_environment(
            Environment(EnvironmentSpec(name="dynamic"), vault=StaticVault({"key": _SECRET})),
            default=True,
        )
    app.register_agent(
        AgentSpec(name="assistant", model="scripted-model"),
        tools=[tool],
        tool_policy=policy,
    )
    return app, provider, tool, policy


async def _drain(stream):
    async with aclosing(stream):
        return [event async for event in stream]


async def _pause(app, session_id) -> ToolApprovalRequest:
    # The PostgreSQL backend shares one database across parameters.
    session_id = f"{session_id}-{uuid4().hex[:12]}"
    events = await _drain(
        app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "Run the round.")],
                metadata={"tenant": "acme"},
            )
        )
    )
    event = next(event for event in events if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED)
    return ToolApprovalRequest(
        session_id=session_id,
        approval_id=event.payload["approval_id"],
        tool_round_id=event.payload["tool_round_id"],
        tool_call_id=event.payload["tool_call_id"],
        decision=ToolApprovalDecision.APPROVE,
    )


def _edit(request: ToolApprovalRequest, value: object) -> ToolApprovalRequest:
    return request.model_copy(update={"arguments": {"value": value}})


async def _intent(store, session_id):
    return pending_approval_reader.approval_resolution_intent_from_checkpoint(
        await store.load_checkpoint(session_id)
    )


def _sha256(arguments) -> str:
    return hashlib.sha256(canonical_durable_json_bytes(arguments, "effect_arguments")).hexdigest()


# S1: the request field, its copy and the retry digest.


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"decision": ToolApprovalDecision.DENY}, "Approved arguments require decision=approve"),
        ({"arguments": ["value"]}, "arguments"),
        (
            {"arguments": {"value": "x" * (MAX_APPROVED_TOOL_ARGUMENTS_BYTES + 1)}},
            "maximum encoded size",
        ),
        # A human review is bound to the model's proposal, not to an edit of it.
        (
            {
                "review_reference": HumanReviewReference(
                    context=HumanReviewContext(recipient="reviewer", purpose="approval"),
                    policy_version="1",
                    content_tag="a" * 64,
                )
            },
            "cannot change the arguments of a human-reviewed proposal",
        ),
    ],
)
def test_approved_arguments_are_a_bounded_object_on_an_approval(changes, message):
    fields = {
        "session_id": "sess",
        "approval_id": "approval",
        "tool_round_id": "round",
        "tool_call_id": "call",
        "decision": ToolApprovalDecision.APPROVE,
        "arguments": {"value": "edited"},
        **changes,
    }
    with pytest.raises(ValidationError, match=message):
        ToolApprovalRequest(**fields)


def test_approved_arguments_are_copied_and_bound_into_the_retry_digest():
    plain = ToolApprovalRequest(
        session_id="sess",
        approval_id="approval",
        tool_round_id="round",
        tool_call_id="call",
        decision=ToolApprovalDecision.APPROVE,
        reason="checked",
    )
    edited = _edit(plain, "edited")
    assert copy_tool_approval_request(edited).arguments == {"value": "edited"}
    assert "edited" not in repr(edited)
    digest = approval_support.approval_resolution_request_digest
    # Without approved arguments the digest is the one already stored in
    # receipts and grants, so no earlier resolution changes identity.
    assert digest(plain) == runtime_publication_checkpoint_value_digest(
        plain.model_dump(
            mode="json",
            include={"review_reference", "decision", "reason", "metadata", "resolved_by"},
        )
    )
    assert len({digest(plain), digest(edited), digest(_edit(plain, "other"))}) == 3
    assert digest(edited) == digest(copy_tool_approval_request(edited))


# S2: request-time admission refuses an edit and leaves the approval pending.


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"value": 5}, "Invalid arguments: value fails type"),
        ({}, "Invalid arguments: value fails required"),
        ({"value": "forbidden"}, "Tool policy denied the approved arguments"),
        ({"value": "large"}, "need authority this approval did not cover"),
        ({"value": _SECRET}, "must not contain a workload secret"),
    ],
)
def test_refused_edit_keeps_the_approval_pending_and_runs_nothing(
    edit_store_factory, arguments, message
):
    async def scenario():
        async with edit_store_factory() as store:
            app, provider, tool, _policy = _runtime(store, dynamic=True)
            request = await _pause(app, "edit-refused")
            before = await store.load_checkpoint(request.session_id)
            status = (await store.load(request.session_id)).status
            with pytest.raises(ToolApprovalArgumentsRefused, match=message):
                await _drain(
                    app.resolve_tool_approval(request.model_copy(update={"arguments": arguments}))
                )
            assert tool.calls == [] and len(provider.requests) == 1
            assert await store.load_checkpoint(request.session_id) == before
            assert await _intent(store, request.session_id) is None
            assert (await store.load(request.session_id)).status is status
            assert not any(
                event.type in {EventType.TOOL_CALL_APPROVED, EventType.TOOL_CALL_STARTED}
                for event in await store.load_events(request.session_id)
            )
            # The approval is still open: an unchanged approval proceeds.
            events = await _drain(app.resolve_tool_approval(request))
            assert events[-1].type is EventType.SESSION_COMPLETED
            assert tool.calls == [{"value": "first"}, {"value": "second"}]

    asyncio.run(scenario())


def test_edit_that_policy_now_allows_is_accepted(edit_store_factory):
    async def scenario():
        async with edit_store_factory() as store:
            app, _provider, tool, _policy = _runtime(store, siblings=("allowed",))
            request = await _pause(app, "edit-allowed-now")
            events = await _drain(app.resolve_tool_approval(_edit(request, "allowed-now")))
            assert events[-1].type is EventType.SESSION_COMPLETED
            assert tool.calls == [{"value": "allowed-now"}, {"value": "second"}]
            intent_policy = [
                event
                for event in await store.load_events(request.session_id)
                if event.type is EventType.TOOL_CALL_APPROVED
            ]
            assert len(intent_policy) == 1

    asyncio.run(scenario())


# S3 and S4: exact execution, evidence and the model's view.


def test_edit_runs_approved_arguments_once_and_siblings_keep_the_models(edit_store_factory):
    async def scenario():
        async with edit_store_factory() as store:
            app, provider, tool, policy = _runtime(store)
            request = await _pause(app, "edit-static")
            edited = _edit(request, "edited")
            events = await _drain(app.resolve_tool_approval(edited))
            assert events[-1].type is EventType.SESSION_COMPLETED
            # Only the gating call runs the approved arguments; the allowed
            # sibling keeps the model's, and the denied sibling never runs.
            assert tool.calls == [{"value": "edited"}, {"value": "second"}]
            reauthorized = policy.requests[-1]
            assert reauthorized.arguments == {"value": "edited"}
            assert reauthorized.metadata["cayu:tool_policy_reauthorization"] is True

            retained = await store.load_events(request.session_id)
            approved = [e for e in retained if e.type is EventType.TOOL_CALL_APPROVED]
            assert [e.payload["tool_call_id"] for e in approved] == ["pause"]
            revision = approved[0].payload["arguments_revision"]
            assert revision["changed_fields"] == ["value"]
            assert revision["proposed_arguments_digest"] != revision["approved_arguments_digest"]
            # Salted digests never equal a plain digest of the arguments.
            assert _sha256({"value": "edited"}) not in revision.values()
            terminals = {
                e.payload["tool_call_id"]: e.payload for e in retained if e.type in _TERMINALS
            }
            # A multi-call continuation publishes no arguments, so neither version.
            assert terminals["pause"]["arguments_state"] == "unavailable"
            assert not {"arguments", "approved_arguments", "effective_arguments"} & set(
                terminals["pause"]
            )
            assert terminals["pause"]["result"]["content"].startswith(APPROVED_ARGUMENTS_NOTICE)
            assert not terminals["allowed"]["result"]["content"].startswith(
                APPROVED_ARGUMENTS_NOTICE
            )
            record = await ToolEffectStateOwner(store).resolve_call(
                await store.load(request.session_id),
                tool_round_id=approved[0].payload["tool_round_id"],
                tool_call_id="pause",
            )
            assert record.intent.arguments_digest == _sha256({"value": "edited"})

            transcript = await store.load_transcript(request.session_id)
            calls = {
                part.tool_call_id: part
                for message in transcript
                for part in message.content
                if type(part) is ToolCallPart
            }
            # The model's call is kept (or unavailable), never replaced by the edit.
            assert calls["pause"].arguments in ({"value": "first"}, {})
            results = {
                part.tool_call_id: part
                for message in transcript
                for part in message.content
                if type(part) is ToolResultPart
            }
            assert results["pause"].content.startswith(APPROVED_ARGUMENTS_NOTICE)
            # The continuation model request carries the same notice.
            continuation = provider.requests[-1]
            assert any(
                type(part) is ToolResultPart
                and part.tool_call_id == "pause"
                and part.content.startswith(APPROVED_ARGUMENTS_NOTICE)
                for message in continuation.messages
                for part in message.content
            )

            # An exact retry replays; changed arguments conflict.
            await _drain(app.resolve_tool_approval(edited))
            assert tool.calls == [{"value": "edited"}, {"value": "second"}]
            assert await store.load_transcript(request.session_id) == transcript
            with pytest.raises(RuntimeError, match="conflicting identity or decision"):
                await _drain(app.resolve_tool_approval(_edit(request, "other")))
            with pytest.raises(RuntimeError, match="conflicting identity or decision"):
                await _drain(app.resolve_tool_approval(request))

    asyncio.run(scenario())


def test_dynamic_scope_keeps_approved_arguments_out_of_events(edit_store_factory):
    async def scenario():
        async with edit_store_factory() as store:
            app, _provider, tool, _policy = _runtime(store, dynamic=True)
            request = await _pause(app, "edit-dynamic")
            public = await _drain(app.resolve_tool_approval(_edit(request, "edited")))
            assert tool.calls == [{"value": "edited"}, {"value": "second"}]
            durable = await store.load_events(request.session_id)
            for event in [*public, *durable]:
                if event.type in _TERMINALS:
                    assert "approved_arguments" not in event.payload
                    assert "edited" not in str(event.payload)
                if event.type is EventType.TOOL_CALL_APPROVED:
                    assert event.payload["arguments_revision"]["changed_fields"] == ["value"]
                    assert "edited" not in str(event.payload)
            transcript = await store.load_transcript(request.session_id)
            assert any(
                type(part) is ToolResultPart
                and part.tool_call_id == "pause"
                and part.content.startswith(APPROVED_ARGUMENTS_NOTICE)
                for message in transcript
                for part in message.content
            )

    asyncio.run(scenario())


def test_restart_after_the_claim_runs_the_approved_arguments(edit_store_factory):
    async def scenario():
        async with edit_store_factory(fault="pause-start") as store:
            app, _provider, tool, _policy = _runtime(store)
            request = await _pause(app, "edit-restart")
            edited = _edit(request, "edited")
            first = await _drain(app.resolve_tool_approval(edited))
            assert store.faulted and first[-1].type is EventType.SESSION_INTERRUPTED
            assert tool.calls == []
            intent = await _intent(store, request.session_id)
            assert intent is not None and intent.approved_arguments is not None
            assert intent.approved_arguments.arguments == {"value": "edited"}
            assert await app.drain_background_interruptions()

            # A new process over the same store resumes from the durable intent.
            restarted, _provider, restarted_tool, _ = _runtime(store, restarted=True)
            events = await _drain(restarted.resolve_tool_approval(edited))
            assert events[-1].type is EventType.SESSION_COMPLETED, events[-1].payload
            assert restarted_tool.calls == [{"value": "edited"}, {"value": "second"}]
            assert tool.calls == []
            assert await restarted.drain_background_interruptions()

    asyncio.run(scenario())


def test_lost_terminal_acknowledgement_never_reruns_the_edit(edit_store_factory):
    async def scenario():
        async with edit_store_factory(fault="terminal-ack") as store:
            app, provider, tool, _policy = _runtime(store, dynamic=True, siblings=("allowed",))
            request = await _pause(app, "edit-terminal-ack")
            edited = _edit(request, "edited")
            first = await _drain(app.resolve_tool_approval(edited))
            assert store.faulted and first[-1].type is EventType.SESSION_INTERRUPTED
            assert tool.calls == [{"value": "edited"}, {"value": "second"}]
            events = await _drain(app.resolve_tool_approval(edited))
            assert events[-1].type is EventType.SESSION_COMPLETED, events[-1].payload
            assert tool.calls == [{"value": "edited"}, {"value": "second"}]
            assert len(provider.requests) == 2
            assert await app.drain_background_interruptions()

    asyncio.run(scenario())


def test_single_call_round_publishes_both_versions_in_its_terminal(edit_store_factory):
    async def scenario():
        async with edit_store_factory() as store:
            app, _provider, tool, _policy = _runtime(store, siblings=())
            request = await _pause(app, "edit-single")
            public = await _drain(app.resolve_tool_approval(_edit(request, "edited")))
            assert tool.calls == [{"value": "edited"}]
            durable = await store.load_events(request.session_id)
            for events in (public, durable):
                [terminal] = [event for event in events if event.type in _TERMINALS]
                assert terminal.payload["arguments_state"] == "finalized"
                assert terminal.payload["arguments"] == {"value": "first"}
                assert terminal.payload["approved_arguments"] == {"value": "edited"}
                assert "effective_arguments" not in terminal.payload
                [started] = [e for e in events if e.type is EventType.TOOL_CALL_STARTED]
                assert "approved_arguments" not in started.payload
                assert "edited" not in str(started.payload)
                [approved] = [e for e in events if e.type is EventType.TOOL_CALL_APPROVED]
                assert "edited" not in str(approved.payload)
            transcript = await store.load_transcript(request.session_id)
            [call] = [
                part
                for message in transcript
                for part in message.content
                if type(part) is ToolCallPart
            ]
            assert call.arguments == {"value": "first"}

    asyncio.run(scenario())


# S5: supervision reports the edit from runtime evidence.


def test_supervision_reports_an_edit_and_its_content_on_request(edit_store_factory):
    async def scenario():
        async with edit_store_factory() as store:
            app, _provider, _tool, _policy = _runtime(store, siblings=())
            request = await _pause(app, "edit-supervision")
            await _drain(app.resolve_tool_approval(_edit(request, "edited")))
            window = SupervisionEventQuery(
                start_at=datetime.now(UTC) - timedelta(days=1),
                end_at=datetime.now(UTC) + timedelta(days=1),
            )
            plain = await store.query_supervision_events(window, include_content=False)
            full = await store.query_supervision_events(window, include_content=True)
            return request.session_id, plain, full

    session_id, plain, full = asyncio.run(scenario())

    def edits(page):
        # The PostgreSQL backend shares its database with other parameters.
        return [
            event
            for event in page.events
            if event.kind is SupervisionEventKind.EDIT and event.session_id == session_id
        ]

    (touch,) = edits(plain)
    assert touch.content is None
    assert touch.approval_revision is None
    assert touch.arguments_revision is not None
    assert touch.arguments_revision.changed_fields == ("value",)
    assert "edited" not in touch.model_dump_json()
    (detailed,) = edits(full)
    assert detailed.content is not None
    assert list(detailed.content.tool_arguments.values()) == [{"value": "first"}]
    assert list(detailed.content.approved_arguments.values()) == [{"value": "edited"}]


class _ChangingHook(RuntimeHook):
    """A before-call hook that sees the approved arguments and changes them again."""

    def __init__(self) -> None:
        self.seen = []

    async def before_tool_call(self, context):
        self.seen.append(dict(context.arguments))
        return BeforeToolCallDecision(
            action="proceed_modified", modified_arguments={"value": "allowed-now"}
        )


def test_hooks_see_approved_arguments_and_effective_arguments_report_only_hook_changes(
    edit_store_factory,
):
    async def scenario():
        async with edit_store_factory() as store:
            hook = _ChangingHook()
            app, _provider, tool, _policy = _runtime(store, siblings=(), hooks=[hook])
            request = await _pause(app, "edit-hook")
            # Execution-time hook re-authorization sees the resolution metadata
            # (pre-existing resume behavior), so the approver restates the tenant.
            edited = _edit(request, "edited").model_copy(update={"metadata": {"tenant": "acme"}})
            await _drain(app.resolve_tool_approval(edited))
            assert hook.seen == [{"value": "edited"}]
            assert tool.calls == [{"value": "allowed-now"}]
            [terminal] = [
                event
                for event in await store.load_events(request.session_id)
                if event.type in _TERMINALS
            ]
            assert terminal.payload["arguments"] == {"value": "first"}
            assert terminal.payload["approved_arguments"] == {"value": "edited"}
            assert terminal.payload["effective_arguments"] == {"value": "allowed-now"}

    asyncio.run(scenario())


def test_reauthorization_uses_the_run_context_not_the_approvers_metadata(edit_store_factory):
    async def scenario():
        async with edit_store_factory() as store:
            app, _provider, tool, policy = _runtime(store, siblings=())
            request = await _pause(app, "edit-run-context")
            # An approver key that would route the call to another gate, and a
            # tenant override, never reach re-authorization.
            steered = _edit(request, "edited").model_copy(
                update={"metadata": {"route": "treasury", "tenant": "other"}}
            )
            events = await _drain(app.resolve_tool_approval(steered))
            assert events[-1].type is EventType.SESSION_COMPLETED
            assert tool.calls == [{"value": "edited"}]
            reauthorized = policy.requests[-1]
            assert reauthorized.arguments == {"value": "edited"}
            assert reauthorized.metadata["tenant"] == "acme"
            assert "route" not in reauthorized.metadata

    asyncio.run(scenario())


def _unkeyed_guess(session_id: str, tool_call_id: str, value: str) -> str:
    """An offline guess from the public session and call identities alone."""

    return hashlib.sha256(
        canonical_durable_json_bytes(
            ["cayu:approval-arguments-revision:v1", session_id, tool_call_id, {"value": value}],
            "arguments_revision",
        )
    ).hexdigest()


def test_published_argument_digests_cannot_be_guessed_from_public_identities(
    edit_store_factory,
):
    async def scenario():
        async with edit_store_factory() as store:
            app, _provider, _tool, _policy = _runtime(store, dynamic=True)
            request = await _pause(app, "edit-guess")
            public = await _drain(app.resolve_tool_approval(_edit(request, "edited")))
            durable = await store.load_events(request.session_id)
            return public, durable

    public, durable = asyncio.run(scenario())

    [approved] = [e for e in durable if e.type is EventType.TOOL_CALL_APPROVED]
    revision = approved.payload["arguments_revision"]
    session_id, call_id = approved.session_id, approved.payload["tool_call_id"]
    published = {revision["proposed_arguments_digest"], revision["approved_arguments_digest"]}
    candidates = {"first", "edited", "second", "never", ""}
    guesses = {_unkeyed_guess(session_id, call_id, value) for value in candidates}
    guesses |= {_sha256({"value": value}) for value in candidates}
    assert not published & guesses
    assert revision["digest_key_id"] == "test"
    [public_approved] = [e for e in public if e.type is EventType.TOOL_CALL_APPROVED]
    assert public_approved.payload["arguments_revision"] == revision


def test_resume_time_argument_error_still_tells_the_model_about_the_edit(
    edit_store_factory, monkeypatch
):
    from cayu.runtime._tool_invocation import invocation

    async def scenario():
        async with edit_store_factory() as store:
            app, _provider, tool, _policy = _runtime(store, siblings=())
            request = await _pause(app, "edit-resume-invalid")
            # The tool's input contract tightens after the claim was admitted.
            monkeypatch.setattr(
                invocation,
                "_registered_tool_argument_error",
                lambda tool, arguments: (
                    "Invalid arguments: value fails enum." if arguments.get("value") else None
                ),
            )
            await _drain(app.resolve_tool_approval(_edit(request, "edited")))
            assert tool.calls == []
            [terminal] = [
                event
                for event in await store.load_events(request.session_id)
                if event.type in _TERMINALS
            ]
            assert terminal.type is EventType.TOOL_CALL_BLOCKED
            assert terminal.payload["result"]["content"].startswith(APPROVED_ARGUMENTS_NOTICE)

    asyncio.run(scenario())
