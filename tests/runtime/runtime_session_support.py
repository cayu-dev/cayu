"""Session support."""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

import pytest
from pydantic import SecretStr
from tests._session_provenance import fixture_session_invocation
from tests.core._execution_profile_fixtures import rebind_test_invocation
from tests.runtime.runtime_provider_support import FakeProvider

import cayu.runtime.execution_profiles as execution_profiles_module
import cayu.sessions._model_completion_publication as model_completion_publication_module
from cayu._exception_groups import (
    exception_cause,
    exception_context,
    exception_group_children,
    exception_suppresses_context,
)
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.tools import (
    PendingToolApproval,
    PendingToolApprovalEventView,
    ResolutionActor,
    ResolutionActorSource,
    ToolApprovalRecoveryRequest,
    ToolApprovalRequest,
)
from cayu.approvals.user_input import UserInputResponse
from cayu.events import Event, EventType
from cayu.execution_units import ToolRoundIdentity
from cayu.providers.base import ModelStreamEvent
from cayu.runtime._event_projection import PRIVATE_EVENT_AUTHORITY, public_event_sequence
from cayu.runtime.execution_profiles import (
    ExecutionProfileAdoptionIntent,
    ExecutionProfileAuthorityDecision,
    ExecutionProfilePolicy,
    ExecutionProfilePolicyAction,
    ExecutionProfilePolicyRequest,
    ExecutionProfilePolicyResult,
)
from cayu.sessions import _pending_approval_reader as pending_approval_reader
from cayu.sessions import _pending_tool_round_reader as pending_round_reader
from cayu.sessions._invocation_terminal_decision import (
    settled_invocation_terminal_decision_from_checkpoint,
)
from cayu.sessions.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
    INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
    SETTLED_INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
)
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.forks import ForkExecutionProfileSelection
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import EventRecord, Session, SessionStatus
from cayu.sessions.requests import (
    ForkSessionRequest,
    InterruptSessionRequest,
    ResumeRequest,
    RunRequest,
)
from cayu.sessions.store import SessionStore
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tasks.dispatch import DispatchRequest
from cayu.tools.base import Tool
from cayu.tools.policy import TaintAwareToolPolicy, ToolPolicy, ToolPolicyDecision
from cayu.tools.rounds import ToolRoundRecoveryRequest

# A drain returns as soon as its work settles, so a check that work did settle
# needs no tight deadline: this budget only bounds a failing run. Measured under
# CPU load, these drains took up to ~330 ms; 2 s leaves margin for slower CI
# runners. Checks that a blocked cleanup does not settle keep short budgets.
_SETTLED_DRAIN_TIMEOUT_S = 2.0


_INTERACTION_LIFECYCLE_EVENT_TYPES = {
    EventType.INTERACTION_STARTED,
    EventType.INTERACTION_RESUMED,
    EventType.INTERACTION_PAUSED,
    EventType.INTERACTION_COMPLETED,
    EventType.INTERACTION_FAILED,
    EventType.INTERACTION_INTERRUPTED,
}


def _without_interaction_lifecycle(events: list[Event]) -> list[Event]:
    """Keep legacy event-sequence assertions focused on their original contract."""
    return [event for event in events if event.type not in _INTERACTION_LIFECYCLE_EVENT_TYPES]


def _without_request_evidence(events: list[Event]) -> list[Event]:
    """Keep pre-evidence event-sequence assertions focused on their contracts."""
    request_evidence_types = {
        EventType.REQUEST_FOOTPRINT_RECORDED,
        EventType.TOOL_EXPOSURE_RECORDED,
        EventType.ENVIRONMENT_LIFECYCLE_TRANSITION,
    }
    return [event for event in events if event.type not in request_evidence_types]


def _runtime_exception_graph(error: BaseException) -> list[BaseException]:
    """Collect one runtime-owned exception graph, including explicit causes."""

    pending = [error]
    collected: list[BaseException] = []
    visited: set[int] = set()
    while pending:
        candidate = pending.pop()
        if id(candidate) in visited:
            continue
        visited.add(id(candidate))
        collected.append(candidate)
        if isinstance(candidate, BaseExceptionGroup):
            children = exception_group_children(candidate)
            if children is not None:
                pending.extend(children)
        cause = exception_cause(candidate)
        if cause is not None:
            pending.append(cause)
        elif not exception_suppresses_context(candidate):
            context = exception_context(candidate)
            if context is not None:
                pending.append(context)
    return collected


async def collect_events(app: CayuApp, request: RunRequest) -> list[Event]:
    events = [event async for event in app.run(request)]
    return _without_request_evidence(_without_interaction_lifecycle(events))


def _private_record_for_public_event(
    records: list[EventRecord],
    event: Event,
) -> EventRecord:
    """Resolve one public app event to its private record without comparing raw IDs."""

    sequence = public_event_sequence(event.id)
    assert sequence is not None
    matches = [record for record in records if record.sequence == sequence]
    assert len(matches) == 1
    return matches[0]


async def _private_events_for_public_events(
    store: SessionStore,
    session_id: str,
    events: list[Event],
) -> list[Event]:
    records = await store.query_events(EventQuery(session_id=session_id))
    return [_private_record_for_public_event(records, event).event for event in events]


async def _pending_tool_approval_from_public_event(
    store: InMemorySessionStore,
    event: Event,
) -> PendingToolApproval:
    private_event = (
        await _private_events_for_public_events(
            store,
            event.session_id,
            [event],
        )
    )[0]
    projected_approval = PendingToolApprovalEventView.from_event(private_event)
    checkpoint_approval = pending_approval_reader.pending_approval_from_checkpoint(
        await store.load_checkpoint(event.session_id)
    )
    assert checkpoint_approval is not None
    assert (
        projected_approval.approval_id,
        projected_approval.tool_call_id,
        projected_approval.model_step_id,
        projected_approval.model_attempt_id,
        projected_approval.tool_round_id,
    ) == (
        checkpoint_approval.approval_id,
        checkpoint_approval.tool_call_id,
        checkpoint_approval.model_step_id,
        checkpoint_approval.model_attempt_id,
        checkpoint_approval.tool_round_id,
    )
    return checkpoint_approval


def tool_round_identity_payload(event: Event) -> dict[str, str]:
    return {
        key: event.payload[key] for key in ("model_step_id", "model_attempt_id", "tool_round_id")
    }


def _tool_round_identity() -> ToolRoundIdentity:
    return ToolRoundIdentity(
        model_step_id=f"mstep_{'1' * 32}",
        model_attempt_id=f"matt_{'2' * 32}",
        tool_round_id=f"tround_{'3' * 32}",
    )


async def collect_resume_events(app: CayuApp, request: ResumeRequest) -> list[Event]:
    events = [event async for event in app.resume(request)]
    return _without_request_evidence(_without_interaction_lifecycle(events))


def assert_only_model_step_publication_checkpoint(
    checkpoint: dict[str, Any] | None,
) -> None:
    assert checkpoint is not None
    settled = settled_invocation_terminal_decision_from_checkpoint(checkpoint)
    optional_keys = (
        {SETTLED_INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY} if settled is not None else set()
    )
    assert set(checkpoint) - optional_keys == {
        CHECKPOINT_SCHEMA_VERSION_KEY,
        execution_profiles_module.ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
        INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
        model_completion_publication_module.LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY,
    }
    assert checkpoint[CHECKPOINT_SCHEMA_VERSION_KEY] == CURRENT_CHECKPOINT_SCHEMA_VERSION
    assert (
        model_completion_publication_module.model_step_publication_from_checkpoint(checkpoint)
        is not None
    )
    assert (
        execution_profiles_module.active_invocation_execution_profile_from_checkpoint(checkpoint)
        is not None
    )


def checkpoint_without_model_step_publication(
    checkpoint: dict[str, Any] | None,
) -> dict[str, Any]:
    assert checkpoint is not None
    assert (
        model_completion_publication_module.model_step_publication_from_checkpoint(checkpoint)
        is not None
    )
    copied = dict(checkpoint)
    copied.pop(model_completion_publication_module.LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY)
    copied.pop(execution_profiles_module.ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY)
    copied.pop(INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY, None)
    if SETTLED_INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY in copied:
        assert settled_invocation_terminal_decision_from_checkpoint(copied) is not None
        copied.pop(SETTLED_INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY)
    return copied


def _expected_failure_evidence(
    session_id: str, *exception_types: str, run_epoch: int = 1
) -> dict[str, Any]:
    return {
        "classification": "failure",
        "deadline": None,
        "deadline_phase": None,
        "exception_types": list(exception_types),
        "run_epoch": run_epoch,
        "secondary_failures": False,
        "session_id": session_id,
        "settlement": "unknown",
        "terminal_event_id": None,
        "truncated": False,
    }


def checkpoint_without_active_invocation_profile(
    checkpoint: dict[str, Any] | None,
) -> dict[str, Any]:
    assert checkpoint is not None
    assert (
        execution_profiles_module.active_invocation_execution_profile_from_checkpoint(checkpoint)
        is not None
    )
    copied = dict(checkpoint)
    copied.pop(execution_profiles_module.ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY)
    copied.pop(INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY, None)
    if SETTLED_INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY in copied:
        assert settled_invocation_terminal_decision_from_checkpoint(copied) is not None
        copied.pop(SETTLED_INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY)
    return copied


async def transition_runtime_session_to_running(
    store: SessionStore,
    session_id: str,
) -> Session:
    return await rebind_test_invocation(store, session_id)


def interruption_payload_without_request_id(payload: dict[str, Any]) -> dict[str, Any]:
    copied = dict(payload)
    request_id = copied.pop("interruption_request_id")
    if request_id != PRIVATE_EVENT_AUTHORITY:
        assert UUID(request_id)
    return copied


@pytest.fixture(params=("memory", "sqlite"), ids=("memory", "sqlite"))
def request_footprint_session_store(request, tmp_path):
    if request.param == "memory":
        store = InMemorySessionStore()
    else:
        import base64

        from cayu.runtime.public_authority import (
            PublicAuthorityAliasCodec,
            PublicAuthorityAliasKeyring,
        )

        alias_codec = PublicAuthorityAliasCodec(
            PublicAuthorityAliasKeyring(
                active_key_id="test",
                keys={
                    "test": SecretStr(
                        base64.urlsafe_b64encode(bytes(range(32))).decode("ascii").rstrip("=")
                    )
                },
            )
        )
        store = SQLiteSessionStore(
            tmp_path / "request-footprint-matrix.sqlite",
            public_authority_alias_codec=alias_codec,
        )
    yield store
    close = getattr(store, "close", None)
    if close is not None:
        asyncio.run(close())


async def collect_interrupt_events(app: CayuApp, request: InterruptSessionRequest) -> list[Event]:
    return [event async for event in app.interrupt_session(request)]


async def collect_fork_events(app: CayuApp, request: ForkSessionRequest) -> list[Event]:
    return [event async for event in app.fork_session(request)]


def _current_fork_profile_selection(
    idempotency_key: str,
) -> dict[str, object]:
    return {
        "execution_profile_selection": ForkExecutionProfileSelection.CURRENT_CHILD,
        "profile_adoption": ExecutionProfileAdoptionIntent(
            idempotency_key=idempotency_key,
            reason="Install the explicitly selected current child execution profile.",
            requested_by=ResolutionActor(
                subject="test-caller",
                source=ResolutionActorSource.REQUEST,
            ),
        ),
    }


class _AllowForkProfileAdoption(ExecutionProfilePolicy):
    @property
    def identity(self) -> str:
        return "test:allow-fork-profile-adoption:v1"

    async def decide(
        self,
        request: ExecutionProfilePolicyRequest,
    ) -> ExecutionProfilePolicyResult:
        return ExecutionProfilePolicyResult(
            action=ExecutionProfilePolicyAction.ADOPT,
            reason="The test explicitly authorizes the selected child profile.",
            authority_decision=(
                ExecutionProfileAuthorityDecision.AUTHORIZED
                if request.authority_review_required
                else ExecutionProfileAuthorityDecision.NOT_REQUIRED
            ),
        )


async def collect_dispatch_events(app: CayuApp, request: DispatchRequest) -> list[Event]:
    events = [event async for event in app.dispatch_inline(request)]
    return _without_request_evidence(_without_interaction_lifecycle(events))


async def collect_tool_approval_events(
    app: CayuApp,
    request: ToolApprovalRequest,
) -> list[Event]:
    events = [event async for event in app.resolve_tool_approval(request)]
    return _without_request_evidence(_without_interaction_lifecycle(events))


async def collect_user_input_events(
    app: CayuApp,
    response: UserInputResponse,
) -> list[Event]:
    events = [event async for event in app.resolve_user_input(response)]
    return _without_request_evidence(_without_interaction_lifecycle(events))


async def collect_tool_approval_recovery_events(
    app: CayuApp,
    request: ToolApprovalRecoveryRequest,
) -> list[Event]:
    events = [event async for event in app.recover_tool_approval(request)]
    return _without_request_evidence(_without_interaction_lifecycle(events))


async def collect_tool_round_recovery_events(
    app: CayuApp,
    request: ToolRoundRecoveryRequest,
) -> list[Event]:
    events = [event async for event in app.recover_tool_round(request)]
    return _without_request_evidence(_without_interaction_lifecycle(events))


def _test_session() -> Session:
    return Session(
        id="sess_context",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        invocation=fixture_session_invocation("sess_context"),
    )


def _taint_aware_policy() -> TaintAwareToolPolicy:
    return TaintAwareToolPolicy(
        taint_sources={"read_web": ["web"]},
        protected_tools={"send_email": ["web"]},
        decision=ToolPolicyDecision.DENY,
    )


def _reviewer_actor() -> ResolutionActor:
    return ResolutionActor(
        subject="reviewer@example.com",
        tenant="acme",
        source=ResolutionActorSource.REQUEST,
        claims={"role": "admin"},
    )


def _approval_pause_app(
    *,
    store: InMemorySessionStore,
    tool: Tool,
    policy: ToolPolicy,
    clock=None,
) -> tuple[CayuApp, FakeProvider]:
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={"value": "secret"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("resolved"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store, clock=clock)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=policy,
    )
    return app, provider


async def _assert_proxy_tool_effect_remains_unknown(store, session_id: str) -> None:
    from cayu.runtime._tool_effect_state import ToolEffectStateOwner

    session = await store.load(session_id)
    assert session is not None
    assert session.status is SessionStatus.INTERRUPTED
    checkpoint = await store.load_checkpoint(session_id)
    pending = pending_round_reader.pending_tool_round_from_checkpoint(checkpoint)
    assert pending is not None
    assert len(pending.tool_calls) == 1
    record = await ToolEffectStateOwner(store).resolve_call(
        session,
        tool_round_id=pending.tool_round_id,
        tool_call_id=pending.tool_calls[0].tool_call_id,
    )
    assert record is not None
    assert record.state == "outcome_unknown"
    assert record.terminal is None
    transcript = await store.load_transcript(session_id)
    assert not any(message.role == "tool" for message in transcript)
