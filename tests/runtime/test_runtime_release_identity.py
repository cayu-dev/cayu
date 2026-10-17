"""Every invocation records the Release its process runs, and touches report it."""

from __future__ import annotations

import asyncio

from tests.runtime.runtime_environment_support import RecordingWorkspaceBinding
from tests.runtime.runtime_provider_support import FakeProvider, VersionedFakeProvider
from tests.runtime.runtime_recovery_support import (
    FailingTerminalToolEventStore,
    _crashed_tool_round_app,
)
from tests.runtime.runtime_session_support import (
    _pending_tool_approval_from_public_event,
    _reviewer_actor,
    collect_events,
    collect_interrupt_events,
    collect_resume_events,
    collect_tool_approval_events,
    collect_tool_approval_recovery_events,
    collect_tool_round_recovery_events,
    collect_user_input_events,
)
from tests.runtime.runtime_supervision_support import (
    approve_pending,
    pause_for_approval,
    release_app,
    resolve_approval,
    rollout_provider,
    run_across_releases,
    supervision_page,
)
from tests.runtime.runtime_tool_support import PureRecordingTool, RequireApprovalPolicy

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.tools import (
    ToolApprovalDecision,
    ToolApprovalRecoveryOutcome,
    ToolApprovalRecoveryRequest,
)
from cayu.approvals.user_input import UserInputRecoveryRequest, UserInputResponse
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.releases import (
    RELEASE_EVENT_PAYLOAD_KEY,
    RELEASE_ID_ENV,
    ReleaseIdentity,
    recorded_release_identity,
)
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions import SupervisionEventKind
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import InterruptSessionRequest, ResumeRequest, RunRequest
from cayu.tools.rounds import ToolRoundRecoveryRequest
from cayu.tools.user_input import UserInputTool

FIRST = ReleaseIdentity(id="release-1")
SECOND = ReleaseIdentity(id="release-2", snapshot_root="ab" * 32)
_RECORDING_EVENTS = (EventType.SESSION_STARTED, EventType.SESSION_RESUMED, EventType.MODEL_STARTED)


async def _recorded_releases(store, session_id: str) -> list[tuple[str, str | None]]:
    records = await store.query_events(
        EventQuery(session_id=session_id, event_types=_RECORDING_EVENTS)
    )
    return [(str(record.event.type), _release_id(record.event.payload)) for record in records]


def _release_id(payload: dict) -> str | None:
    release = recorded_release_identity(payload)
    return None if release is None else release.id


def test_a_session_resumed_after_a_rollout_reports_each_events_release():
    store = InMemorySessionStore()

    async def run():
        await run_across_releases(store, "sess_release_rollout", FIRST, SECOND)
        return (
            await _recorded_releases(store, "sess_release_rollout"),
            await supervision_page(store),
        )

    recorded, page = asyncio.run(run())

    assert recorded == [
        ("session.started", "release-1"),
        ("model.started", "release-1"),
        ("session.resumed", "release-1"),
        ("model.started", "release-1"),
        ("session.resumed", "release-2"),
        ("model.started", "release-2"),
        ("session.resumed", "release-2"),
        ("model.started", "release-2"),
    ]
    # Each approval belongs to the invocation that proposed its call, even when a
    # process on the next Release resolved it.
    assert [
        (event.kind, event.tool_call_ids, event.release_id, event.release_snapshot_root)
        for event in page.events
    ] == [
        (SupervisionEventKind.APPROVE, ("call_1",), "release-1", None),
        (SupervisionEventKind.APPROVE, ("call_2",), "release-1", None),
        (SupervisionEventKind.APPROVE, ("call_3",), "release-2", SECOND.snapshot_root),
    ]


def test_a_touch_without_a_new_invocation_keeps_the_invocation_release():
    store = InMemorySessionStore()
    provider = rollout_provider()

    async def run():
        await pause_for_approval(release_app(store, provider, FIRST), store, "sess_release_paused")
        await collect_interrupt_events(
            release_app(store, provider, SECOND),
            InterruptSessionRequest(
                session_id="sess_release_paused",
                reason="stop",
                requested_by=_reviewer_actor(),
            ),
        )
        return await supervision_page(store)

    page = asyncio.run(run())

    assert [(event.kind, event.release_id) for event in page.events] == [
        (SupervisionEventKind.INTERRUPT, "release-1")
    ]


def test_a_process_without_a_stated_release_records_none():
    store = InMemorySessionStore()
    provider = rollout_provider()

    async def run():
        local = release_app(store, provider, None)
        await pause_for_approval(local, store, "sess_release_local")
        await approve_pending(local, store, "sess_release_local")
        return (
            await _recorded_releases(store, "sess_release_local"),
            await supervision_page(store),
            await supervision_page(store, release_id="release-1"),
        )

    recorded, page, filtered = asyncio.run(run())

    assert recorded == [
        ("session.started", None),
        ("model.started", None),
        ("session.resumed", None),
        ("model.started", None),
    ]
    assert [(event.release_id, event.release_snapshot_root) for event in page.events] == [
        (None, None)
    ]
    assert filtered.events == ()


def test_public_event_streams_expose_each_invocations_release():
    store = InMemorySessionStore()

    async def run():
        app = release_app(store, rollout_provider(), SECOND)
        started = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_release_public",
                messages=[Message.text("user", "use the tool")],
            ),
        )
        pending = await _pending_tool_approval_from_public_event(
            store,
            next(
                event for event in started if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
            ),
        )
        resumed = await collect_tool_approval_events(
            app,
            resolve_approval("sess_release_public", pending, ToolApprovalDecision.APPROVE),
        )
        return [*started, *resumed]

    events = asyncio.run(run())

    assert [
        (str(event.type), event.payload[RELEASE_EVENT_PAYLOAD_KEY])
        for event in events
        if event.type in _RECORDING_EVENTS
    ] == [
        ("session.started", SECOND.model_dump(mode="json")),
        ("model.started", SECOND.model_dump(mode="json")),
        ("session.resumed", SECOND.model_dump(mode="json")),
        ("model.started", SECOND.model_dump(mode="json")),
    ]


def test_an_invocation_that_fails_before_its_start_event_records_no_release():
    """Setup that fails before session.started or session.resumed records no Release.

    No event of that invocation records one, so its failure is attributed to the
    Release in force before it, and a new session that fails there records none.
    """

    store = InMemorySessionStore()
    binding = RecordingWorkspaceBinding()
    provider = VersionedFakeProvider(
        [
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ]
    )
    environment = EnvironmentSpec(
        name="bound",
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests:release:bound-environment", behavior_version="1", implementation_version="1"
        ),
    )

    def bound_app(release: ReleaseIdentity) -> CayuApp:
        app = CayuApp(session_store=store, release_identity=release)
        app.register_provider(provider, default=True)
        app.register_environment(Environment(environment, binding=binding), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        return app

    async def run():
        await collect_events(
            bound_app(FIRST),
            RunRequest(
                agent_name="assistant",
                session_id="sess_release_setup_resumed",
                messages=[Message.text("user", "first")],
            ),
        )
        binding.fail_bind = True
        after = bound_app(SECOND)
        await collect_resume_events(
            after,
            ResumeRequest(
                session_id="sess_release_setup_resumed", messages=[Message.text("user", "again")]
            ),
        )
        await collect_events(
            after,
            RunRequest(
                agent_name="assistant",
                session_id="sess_release_setup_new",
                messages=[Message.text("user", "first")],
            ),
        )
        outcomes = {}
        for session_id in ("sess_release_setup_resumed", "sess_release_setup_new"):
            session = await store.load(session_id)
            assert session is not None
            records = await store.query_events(EventQuery(session_id=session_id))
            outcomes[session_id] = (
                session.status,
                [
                    (str(record.event.type), _release_id(record.event.payload))
                    for record in records
                    if RELEASE_EVENT_PAYLOAD_KEY in record.event.payload
                ],
            )
        return outcomes

    outcomes = asyncio.run(run())

    assert outcomes == {
        "sess_release_setup_resumed": (
            SessionStatus.FAILED,
            [("session.started", "release-1"), ("model.started", "release-1")],
        ),
        "sess_release_setup_new": (SessionStatus.FAILED, []),
    }


STATED = ReleaseIdentity(id="release-stated")


def _stated_app(monkeypatch, store, *, turns, tools, policy=None) -> CayuApp:
    """An application whose process the environment states as running ``STATED``."""

    monkeypatch.setenv(RELEASE_ID_ENV, STATED.id)
    app = CayuApp(session_store=store)
    app.register_provider(FakeProvider(turns), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"), tools=tools, tool_policy=policy
    )
    return app


async def _resumed_releases(store, session_id: str) -> list[str | None]:
    records = await store.query_events(
        EventQuery(session_id=session_id, event_types=(EventType.SESSION_RESUMED,))
    )
    return [_release_id(record.event.payload) for record in records]


def _ask_then(*turns) -> list:
    return [
        [
            ModelStreamEvent.tool_call(
                id="call_ask", name="ask_user", arguments={"question": "Q?"}
            ),
            *turns,
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ],
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ],
    ]


async def _pause_for_input(app: CayuApp, session_id: str) -> UserInputResponse:
    paused = await collect_events(
        app,
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[Message.text("user", "ask first")],
        ),
    )
    awaiting = next(e for e in paused if e.type == EventType.SESSION_AWAITING_USER_INPUT)
    return UserInputResponse(
        session_id=session_id, input_id=awaiting.payload["input_id"], answer="September"
    )


def test_a_user_input_answer_records_the_release(monkeypatch):
    store = InMemorySessionStore()
    app = _stated_app(monkeypatch, store, turns=_ask_then(), tools=[UserInputTool()])

    async def run():
        await collect_user_input_events(app, await _pause_for_input(app, "sess_release_answer"))
        return await _resumed_releases(store, "sess_release_answer"), await supervision_page(store)

    resumed, page = asyncio.run(run())

    assert resumed == [STATED.id]
    assert [(event.kind, event.release_id) for event in page.events] == [
        (SupervisionEventKind.ANSWER, STATED.id)
    ]


def test_a_user_input_manual_recovery_records_the_release(monkeypatch):
    store = FailingTerminalToolEventStore()
    app = _stated_app(
        monkeypatch,
        store,
        turns=_ask_then(ModelStreamEvent.tool_call(id="call_1", name="side_effect", arguments={})),
        tools=[UserInputTool(), PureRecordingTool()],
    )

    async def run():
        response = await _pause_for_input(app, "sess_release_input_recovery")
        crashed = await collect_user_input_events(app, response)
        assert crashed[-1].type == EventType.SESSION_INTERRUPTED
        _ = [
            event
            async for event in app.recover_user_input(
                UserInputRecoveryRequest(
                    session_id=response.session_id,
                    input_id=response.input_id,
                    answer=response.answer,
                    tool_call_id="call_1",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="recorded outside",
                )
            )
        ]
        return await _resumed_releases(store, response.session_id)

    assert asyncio.run(run()) == [STATED.id, STATED.id]


def test_an_approval_manual_recovery_records_the_release(monkeypatch):
    store = FailingTerminalToolEventStore()
    app = _stated_app(
        monkeypatch,
        store,
        turns=[
            [
                ModelStreamEvent.tool_call(id="call_1", name="side_effect", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ],
        tools=[PureRecordingTool()],
        policy=RequireApprovalPolicy(),
    )

    async def run():
        pending = await pause_for_approval(app, store, "sess_release_approval_recovery")
        crashed = await collect_tool_approval_events(
            app,
            resolve_approval(
                "sess_release_approval_recovery", pending, ToolApprovalDecision.APPROVE
            ),
        )
        assert crashed[-1].type == EventType.SESSION_INTERRUPTED
        await collect_tool_approval_recovery_events(
            app,
            ToolApprovalRecoveryRequest(
                session_id="sess_release_approval_recovery",
                approval_id=pending.approval_id,
                tool_round_id=pending.tool_round_id,
                tool_call_id=pending.tool_call_id,
                outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                message="done outside",
            ),
        )
        return await _resumed_releases(store, "sess_release_approval_recovery")

    assert asyncio.run(run()) == [STATED.id, STATED.id]


def test_a_tool_round_manual_recovery_records_the_release(monkeypatch):
    monkeypatch.setenv(RELEASE_ID_ENV, STATED.id)
    session_id = "sess_release_round_recovery"
    app, store, _tool, checkpoint = _crashed_tool_round_app(session_id, tool=PureRecordingTool())

    async def run():
        await collect_tool_round_recovery_events(
            app,
            ToolRoundRecoveryRequest(
                session_id=session_id,
                round_id=checkpoint["pending_tool_round"]["tool_round_id"],
                tool_call_id="call_1",
                outcome=ToolApprovalRecoveryOutcome.FAILED,
                message="the export did not happen",
            ),
        )
        return await _resumed_releases(store, session_id), await _recorded_releases(
            store, session_id
        )

    resumed, recorded = asyncio.run(run())

    assert resumed == [STATED.id]
    assert recorded[-1] == ("model.started", STATED.id)
