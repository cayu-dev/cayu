from __future__ import annotations

import asyncio

import pytest
from tests.core._execution_profile_fixtures import profiled_session_identity
from tests.core.postgres_contention_support import drop_cayu_tables

from cayu import (
    AgentSpec,
    CayuApp,
    Event,
    EventQuery,
    EventType,
    ForkSessionRequest,
    InMemorySessionStore,
    Message,
    ModelStreamEvent,
    PostgresSessionStore,
    RunRequest,
    ScriptedModelProvider,
    SessionIdentity,
    SessionLineageResult,
    SessionStatus,
    SessionStore,
    SessionTrajectoryBounds,
    SessionTrajectoryError,
    SessionTrajectoryErrorCode,
    SQLiteSessionStore,
    TerminalSessionEvidenceErrorCode,
    ToolCapabilityCeiling,
    ToolsCalledInOrder,
    Trajectory,
    evaluate_assertions,
    load_trajectory,
    trajectory_from_session,
    write_trajectory_json,
)
from cayu._validation import compact_json_utf8_size
from cayu.core.events import (
    event_payload_authority_is_runtime_generated,
    event_with_runtime_payload_authority,
)
from cayu.evals.trajectory import _build_child_trajectories, _CaptureState, _IncompleteFlag
from cayu.storage.migrations import SchemaMode

pytestmark = pytest.mark.postgres


@pytest.fixture(params=("memory", "sqlite", "postgres"))
def trajectory_store_case(request, tmp_path):
    if request.param == "postgres":
        return request.param, tmp_path, request.getfixturevalue("postgres_dsn")
    return request.param, tmp_path, None


async def _open_store(case) -> SessionStore:
    kind, tmp_path, postgres_dsn = case
    if kind == "memory":
        return InMemorySessionStore()
    if kind == "sqlite":
        return SQLiteSessionStore(tmp_path / "session-trajectory.sqlite")
    await drop_cayu_tables(postgres_dsn)
    return PostgresSessionStore(
        postgres_dsn,
        min_size=1,
        max_size=4,
        schema_mode=SchemaMode.CREATE,
    )


async def _close_store(store: SessionStore) -> None:
    close = getattr(store, "close", None)
    if close is not None:
        await close()


async def _create_running_session(
    store: SessionStore,
    session_id: str,
    *,
    parent_session_id: str | None = None,
    started_payload: dict[str, object] | None = None,
    origin_event_type: EventType = EventType.SESSION_STARTED,
    untrusted_origin_fields: tuple[str, ...] = (),
    agent_name: str = "assistant",
    model: str = "fake-model",
) -> str:
    interaction_id = f"{session_id}-interaction"
    user_message = Message.text("user", f"request for {session_id}")
    await store.create(
        RunRequest(
            agent_name=agent_name,
            session_id=session_id,
            parent_session_id=parent_session_id,
            messages=[user_message],
            tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
        ),
        identity=profiled_session_identity(
            provider_name="fake",
            model=model,
        ),
        interaction_started_event=Event(
            id=f"{session_id}-interaction-started",
            type=EventType.INTERACTION_STARTED,
            session_id=session_id,
            interaction_id=interaction_id,
        ),
        interaction_source_messages=[user_message],
    )
    if started_payload is None:
        origin_payload: dict[str, object] = {"agent_name": "assistant"}
        if parent_session_id is not None:
            origin_payload["parent_session_id"] = parent_session_id
            if origin_event_type == EventType.SESSION_FORKED:
                origin_payload["source_session_id"] = parent_session_id
    else:
        origin_payload = started_payload
    origin_event = Event(
        id=f"{session_id}-started",
        type=origin_event_type,
        session_id=session_id,
        payload=origin_payload,
    )
    authority_fields = tuple(
        field_name
        for field_name in ("parent_session_id", "source_session_id")
        if type(origin_event.payload.get(field_name)) is str
        and field_name not in untrusted_origin_fields
    )
    if authority_fields:
        origin_event = event_with_runtime_payload_authority(origin_event, *authority_fields)
    await store.append_event(session_id, origin_event)
    await store.replace_initial_transcript_messages(
        session_id,
        [user_message],
        [Message.text("system", "Be precise."), user_message],
        interaction_id=interaction_id,
    )
    return interaction_id


async def _finish_session(
    store: SessionStore,
    session_id: str,
    interaction_id: str,
    *,
    status: SessionStatus = SessionStatus.COMPLETED,
    append_output: bool = True,
) -> None:
    if append_output:
        await store.append_transcript_messages(
            session_id,
            [Message.text("assistant", f"answer from {session_id}")],
            interaction_id=interaction_id,
        )
    interaction_event_type = (
        EventType.INTERACTION_COMPLETED
        if status is SessionStatus.COMPLETED
        else EventType.INTERACTION_FAILED
    )
    terminal_event_type = (
        EventType.SESSION_COMPLETED
        if status is SessionStatus.COMPLETED
        else EventType.SESSION_FAILED
    )
    await store.publish_interaction_transition(
        session_id,
        event=Event(
            id=f"{session_id}-{status.value}-interaction",
            type=interaction_event_type,
            session_id=session_id,
            interaction_id=interaction_id,
        ),
        from_statuses={SessionStatus.RUNNING},
        to_status=status,
    )
    await store.append_event(
        session_id,
        Event(
            id=f"{session_id}-{status.value}-session",
            type=terminal_event_type,
            session_id=session_id,
        ),
    )


async def _create_zero_transcript_running_session(
    store: SessionStore,
    session_id: str,
    *,
    parent_session_id: str | None = None,
) -> str:
    interaction_id = f"{session_id}-interaction"
    await store.create(
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            parent_session_id=parent_session_id,
            messages=[],
            tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
        ),
        identity=profiled_session_identity(
            provider_name="fake",
            model="fake-model",
        ),
        interaction_started_event=Event(
            id=f"{session_id}-interaction-started",
            type=EventType.INTERACTION_STARTED,
            session_id=session_id,
            interaction_id=interaction_id,
        ),
        interaction_source_messages=[],
    )
    await store.replace_initial_transcript_messages(
        session_id,
        [],
        [],
        interaction_id=interaction_id,
    )
    origin_payload = {"agent_name": "assistant"}
    if parent_session_id is not None:
        origin_payload["parent_session_id"] = parent_session_id
    origin_event = Event(
        id=f"{session_id}-started",
        type=EventType.SESSION_STARTED,
        session_id=session_id,
        payload=origin_payload,
    )
    if parent_session_id is not None:
        origin_event = event_with_runtime_payload_authority(
            origin_event,
            "parent_session_id",
        )
    await store.append_event(session_id, origin_event)
    return interaction_id


def test_trajectory_from_session_admits_preterminal_tree_and_excludes_later_fork(
    trajectory_store_case,
):
    async def scenario():
        store = await _open_store(trajectory_store_case)
        root_interaction = await _create_running_session(store, "root")
        child_interaction = await _create_running_session(
            store,
            "child-before-cutoff",
            parent_session_id="root",
        )
        grandchild_interaction = await _create_running_session(
            store,
            "grandchild-before-cutoff",
            parent_session_id="child-before-cutoff",
        )
        await _finish_session(store, "root", root_interaction)
        # A background child may finish after its parent. Its origin is before the
        # parent's terminal cutoff, so it remains part of the admitted run tree.
        await _finish_session(store, "child-before-cutoff", child_interaction)
        await _finish_session(
            store,
            "grandchild-before-cutoff",
            grandchild_interaction,
        )

        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(ScriptedModelProvider([], name="fake"), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        fork_events = [
            event
            async for event in app.fork_session(
                ForkSessionRequest(
                    source_session_id="root",
                    session_id="fork-after-cutoff",
                )
            )
        ]
        assert [event.type for event in fork_events] == [EventType.SESSION_FORKED]

        # The fork is a real completed child, but it began after the root's
        # terminal boundary and therefore is not part of the evaluated run.
        fork = await store.load("fork-after-cutoff")
        assert fork is not None
        assert fork.status is SessionStatus.COMPLETED

        first = await trajectory_from_session(app, "root")
        second = await trajectory_from_session(app, "root")
        root_evidence = await store.load_terminal_session_evidence("root")
        fresh_state = _CaptureState(bounds=SessionTrajectoryBounds(), strict=False)
        fresh_state.retain(root_evidence)
        fresh_incomplete = _IncompleteFlag()
        fresh_children = await _build_child_trajectories(
            app,
            "root",
            visited={"root"},
            incomplete=fresh_incomplete,
            parent_terminal_sequence=root_evidence.boundary.terminal_event_sequence,
            state=fresh_state,
        )
        await _close_store(store)
        return first, second, fresh_children, fresh_incomplete.value

    first, second, fresh_children, fresh_incomplete = asyncio.run(scenario())
    assert first == second
    assert first.session is not None
    assert first.session.id == "root"
    assert first.final_output == "answer from root"
    assert [child.session.id for child in first.children if child.session is not None] == [
        "child-before-cutoff"
    ]
    assert first.children[0].final_output == "answer from child-before-cutoff"
    assert [
        child.session.id for child in first.children[0].children if child.session is not None
    ] == ["grandchild-before-cutoff"]
    assert [child.session.id for child in fresh_children if child.session is not None] == [
        "child-before-cutoff"
    ]
    assert fresh_incomplete is False
    assert first.children_incomplete is False
    assert first.probes.workspace_available is False
    assert first.probes.artifacts_available is False


def test_trajectory_from_session_reopens_sqlite_without_runtime_registrations(tmp_path):
    async def scenario():
        path = tmp_path / "restart-trajectory.sqlite"
        producer_store = SQLiteSessionStore(path)
        producer = CayuApp(session_store=producer_store, enable_logging=False)
        producer.register_provider(
            ScriptedModelProvider(
                [
                    ModelStreamEvent.text_delta("durable answer"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            ),
            default=True,
        )
        producer.register_agent(AgentSpec(name="assistant", model="fake-model"))
        async for _ in producer.run(
            RunRequest(
                agent_name="assistant",
                session_id="restart-root",
                messages=[Message.text("user", "answer once")],
            )
        ):
            pass
        await producer_store.close()

        reopened_store = SQLiteSessionStore(path)
        consumer = CayuApp(session_store=reopened_store, enable_logging=False)
        trajectory = await trajectory_from_session(consumer, "restart-root")
        await reopened_store.close()
        return trajectory

    trajectory = asyncio.run(scenario())
    assert trajectory.session is not None
    assert trajectory.session.status is SessionStatus.COMPLETED
    assert trajectory.final_output == "durable answer"
    assert trajectory.children == ()


def test_trajectory_from_session_keeps_canonical_acceptance_across_stores(
    trajectory_store_case,
):
    async def scenario():
        store = await _open_store(trajectory_store_case)
        interaction_id = await _create_running_session(store, "numeric-root")
        await store.append_event(
            "numeric-root",
            Event(
                id="numeric-payload",
                type=EventType.MODEL_COMPLETED,
                session_id="numeric-root",
                payload={"samples": [1e-7] * 10_000},
            ),
        )
        await _finish_session(store, "numeric-root", interaction_id)
        trajectory = await trajectory_from_session(
            CayuApp(session_store=store, enable_logging=False),
            "numeric-root",
        )
        await _close_store(store)
        return trajectory

    trajectory = asyncio.run(scenario())
    assert trajectory.session is not None
    assert trajectory.session.id == "numeric-root"
    assert trajectory.events[-1].type is EventType.SESSION_COMPLETED


def test_trajectory_from_session_accepts_a_failed_terminal_root():
    async def scenario():
        store = InMemorySessionStore()
        interaction_id = await _create_running_session(store, "failed-root")
        await _finish_session(
            store,
            "failed-root",
            interaction_id,
            status=SessionStatus.FAILED,
        )
        return await trajectory_from_session(
            CayuApp(session_store=store, enable_logging=False),
            "failed-root",
        )

    trajectory = asyncio.run(scenario())
    assert trajectory.session is not None
    assert trajectory.session.status is SessionStatus.FAILED
    assert trajectory.events[-1].type is EventType.SESSION_FAILED


def test_trajectory_from_session_requires_the_bounded_lineage_capability():
    class NoLineageStore(InMemorySessionStore):
        supports_session_lineage = False

    with pytest.raises(SessionTrajectoryError) as captured:
        asyncio.run(
            trajectory_from_session(
                CayuApp(session_store=NoLineageStore(), enable_logging=False),
                "root",
            )
        )
    assert captured.value.code is SessionTrajectoryErrorCode.STORE_UNSUPPORTED
    assert captured.value.session_id == "root"


def test_tools_called_in_order_scores_a_promoted_and_reloaded_trajectory(tmp_path):
    async def scenario():
        store = InMemorySessionStore()
        interaction_id = await _create_running_session(store, "tool-root")
        await store.append_transcript_messages(
            "tool-root",
            [
                Message.tool_call(
                    tool_call_id="search-call",
                    tool_name="search",
                    arguments={"query": "cayu"},
                ),
                Message.tool_result(
                    tool_call_id="search-call",
                    tool_name="search",
                    content="found",
                ),
            ],
            interaction_id=interaction_id,
        )
        await _finish_session(store, "tool-root", interaction_id)
        trajectory = await trajectory_from_session(
            CayuApp(session_store=store, enable_logging=False),
            "tool-root",
        )
        path = tmp_path / "tool-root.json"
        write_trajectory_json(trajectory, path)
        restored = load_trajectory(path)
        results = await evaluate_assertions(restored, [ToolsCalledInOrder(["search"])])
        return trajectory, restored, results

    trajectory, restored, results = asyncio.run(scenario())
    assert restored == trajectory
    assert len(results) == 1
    assert results[0].passed is True


@pytest.mark.parametrize(
    ("root_status", "expected_terminal_code"),
    [
        (SessionStatus.RUNNING, TerminalSessionEvidenceErrorCode.SESSION_NOT_TERMINAL),
        (SessionStatus.INTERRUPTED, TerminalSessionEvidenceErrorCode.SESSION_INTERRUPTED),
    ],
)
def test_trajectory_from_session_rejects_noneligible_root(
    root_status,
    expected_terminal_code,
):
    async def scenario():
        store = InMemorySessionStore()
        interaction_id = await _create_running_session(store, "root")
        if root_status is SessionStatus.INTERRUPTED:
            await store.publish_interaction_transition(
                "root",
                event=Event(
                    id="root-interaction-interrupted",
                    type=EventType.INTERACTION_INTERRUPTED,
                    session_id="root",
                    interaction_id=interaction_id,
                ),
                from_statuses={SessionStatus.RUNNING},
                to_status=SessionStatus.INTERRUPTED,
            )
            await store.append_event(
                "root",
                Event(
                    id="root-session-interrupted",
                    type=EventType.SESSION_INTERRUPTED,
                    session_id="root",
                ),
            )
        return await trajectory_from_session(
            CayuApp(session_store=store, enable_logging=False),
            "root",
        )

    with pytest.raises(SessionTrajectoryError) as captured:
        asyncio.run(scenario())
    assert captured.value.code is SessionTrajectoryErrorCode.TERMINAL_EVIDENCE_REJECTED
    assert captured.value.terminal_code is expected_terminal_code
    assert captured.value.session_id == "root"


def test_trajectory_from_session_rejects_active_admitted_child():
    async def scenario():
        store = InMemorySessionStore()
        root_interaction = await _create_running_session(store, "root")
        await _create_running_session(store, "active-child", parent_session_id="root")
        await _finish_session(store, "root", root_interaction)
        return await trajectory_from_session(
            CayuApp(session_store=store, enable_logging=False),
            "root",
        )

    with pytest.raises(SessionTrajectoryError) as captured:
        asyncio.run(scenario())
    assert captured.value.code is SessionTrajectoryErrorCode.TERMINAL_EVIDENCE_REJECTED
    assert captured.value.terminal_code is TerminalSessionEvidenceErrorCode.SESSION_NOT_TERMINAL
    assert captured.value.session_id == "active-child"
    assert captured.value.parent_session_id == "root"


def test_trajectory_from_session_enforces_one_global_session_bound():
    async def scenario():
        store = InMemorySessionStore()
        root_interaction = await _create_running_session(store, "root")
        child_interaction = await _create_running_session(
            store,
            "child",
            parent_session_id="root",
        )
        await _finish_session(store, "root", root_interaction)
        await _finish_session(store, "child", child_interaction)
        return await trajectory_from_session(
            CayuApp(session_store=store, enable_logging=False),
            "root",
            bounds=SessionTrajectoryBounds(max_sessions=1),
        )

    with pytest.raises(SessionTrajectoryError) as captured:
        asyncio.run(scenario())
    assert captured.value.code is SessionTrajectoryErrorCode.SESSION_LIMIT_EXCEEDED
    assert captured.value.limit == 1
    assert captured.value.observed == 2


def test_trajectory_from_session_enforces_the_tree_depth_bound():
    async def scenario():
        store = InMemorySessionStore()
        root_interaction = await _create_running_session(store, "root")
        child_interaction = await _create_running_session(
            store,
            "child",
            parent_session_id="root",
        )
        grandchild_interaction = await _create_running_session(
            store,
            "grandchild",
            parent_session_id="child",
        )
        await _finish_session(store, "root", root_interaction)
        await _finish_session(store, "child", child_interaction)
        await _finish_session(store, "grandchild", grandchild_interaction)
        return await trajectory_from_session(
            CayuApp(session_store=store, enable_logging=False),
            "root",
            bounds=SessionTrajectoryBounds(max_depth=2),
        )

    with pytest.raises(SessionTrajectoryError) as captured:
        asyncio.run(scenario())
    assert captured.value.code is SessionTrajectoryErrorCode.DEPTH_LIMIT_EXCEEDED
    assert captured.value.session_id == "grandchild"
    assert captured.value.parent_session_id == "child"
    assert captured.value.limit == 2
    assert captured.value.observed == 3


def test_session_trajectory_hard_depth_is_exportable_and_reloadable(tmp_path):
    trajectory = Trajectory()
    for _ in range(SessionTrajectoryBounds().max_depth - 1):
        trajectory = Trajectory().model_copy(update={"children": (trajectory,)})

    path = tmp_path / "max-depth-trajectory.json"
    write_trajectory_json(trajectory, path)

    assert load_trajectory(path) == trajectory


def test_trajectory_from_session_validates_one_completed_tree_once(monkeypatch):
    import cayu.evals.trajectory as trajectory_module

    validations = 0
    validate = trajectory_module._validate_trajectory_record_contract

    def count_validation(trajectory):
        nonlocal validations
        validations += 1
        validate(trajectory)

    monkeypatch.setattr(
        trajectory_module,
        "_validate_trajectory_record_contract",
        count_validation,
    )

    async def scenario():
        store = InMemorySessionStore()
        root_interaction = await _create_running_session(store, "root")
        child_interaction = await _create_running_session(
            store,
            "child",
            parent_session_id="root",
        )
        grandchild_interaction = await _create_running_session(
            store,
            "grandchild",
            parent_session_id="child",
        )
        await _finish_session(store, "root", root_interaction)
        await _finish_session(store, "child", child_interaction)
        await _finish_session(store, "grandchild", grandchild_interaction)
        return await trajectory_from_session(
            CayuApp(session_store=store, enable_logging=False),
            "root",
        )

    trajectory = asyncio.run(scenario())
    assert trajectory.children[0].children[0].session is not None
    assert trajectory.children[0].children[0].session.id == "grandchild"
    assert validations == 1


def test_excluded_post_terminal_child_does_not_consume_retained_session_bound():
    async def scenario():
        store = InMemorySessionStore()
        root_interaction = await _create_running_session(store, "root")
        child_interaction = await _create_running_session(
            store,
            "admitted-child",
            parent_session_id="root",
        )
        await _finish_session(store, "root", root_interaction)
        await _finish_session(store, "admitted-child", child_interaction)
        excluded_interaction = await _create_running_session(
            store,
            "post-terminal-child",
            parent_session_id="root",
        )
        await _finish_session(store, "post-terminal-child", excluded_interaction)

        return await trajectory_from_session(
            CayuApp(session_store=store, enable_logging=False),
            "root",
            bounds=SessionTrajectoryBounds(max_sessions=2),
        )

    trajectory = asyncio.run(scenario())

    assert trajectory.session is not None
    assert trajectory.session.id == "root"
    assert [child.session.id for child in trajectory.children if child.session is not None] == [
        "admitted-child"
    ]


@pytest.mark.parametrize(
    ("bound_name", "boundary_name", "terminal_code"),
    [
        (
            "max_events",
            "event_count",
            TerminalSessionEvidenceErrorCode.EVENT_LIMIT_EXCEEDED,
        ),
        (
            "max_transcript_records",
            "transcript_count",
            TerminalSessionEvidenceErrorCode.TRANSCRIPT_LIMIT_EXCEEDED,
        ),
        (
            "max_total_bytes",
            "total_bytes",
            TerminalSessionEvidenceErrorCode.TOTAL_BYTES_EXCEEDED,
        ),
    ],
)
def test_trajectory_from_session_enforces_global_evidence_bounds(
    bound_name,
    boundary_name,
    terminal_code,
):
    async def scenario():
        store = InMemorySessionStore()
        root_interaction = await _create_running_session(store, "root")
        child_interaction = await _create_running_session(
            store,
            "child",
            parent_session_id="root",
        )
        await _finish_session(store, "root", root_interaction)
        await _finish_session(store, "child", child_interaction)
        root = await store.load_terminal_session_evidence("root")
        child = await store.load_terminal_session_evidence("child")
        combined_limit = (
            getattr(root.boundary, boundary_name) + getattr(child.boundary, boundary_name) - 1
        )
        bound_values = SessionTrajectoryBounds().model_dump(mode="python")
        bound_values[bound_name] = combined_limit
        return await trajectory_from_session(
            CayuApp(session_store=store, enable_logging=False),
            "root",
            bounds=SessionTrajectoryBounds.model_validate(bound_values),
        )

    with pytest.raises(SessionTrajectoryError) as captured:
        asyncio.run(scenario())
    assert captured.value.code is SessionTrajectoryErrorCode.TERMINAL_EVIDENCE_REJECTED
    assert captured.value.terminal_code is terminal_code
    assert captured.value.session_id == "child"


def test_trajectory_from_session_accepts_an_exact_transcript_bound_with_zero_record_child(
    trajectory_store_case,
):
    async def scenario():
        store = await _open_store(trajectory_store_case)
        try:
            root_interaction = await _create_running_session(store, "root")
            child_interaction = await _create_zero_transcript_running_session(
                store,
                "child",
                parent_session_id="root",
            )
            await _finish_session(
                store,
                "child",
                child_interaction,
                status=SessionStatus.FAILED,
                append_output=False,
            )
            await _finish_session(store, "root", root_interaction)
            root = await store.load_terminal_session_evidence("root")
            child = await store.load_terminal_session_evidence("child")
            app = CayuApp(session_store=store, enable_logging=False)
            trajectory = await trajectory_from_session(
                app,
                "root",
                bounds=SessionTrajectoryBounds(
                    max_transcript_records=root.boundary.transcript_count,
                ),
            )
            zero_transcript_root = await trajectory_from_session(
                app,
                "child",
                bounds=SessionTrajectoryBounds(max_transcript_records=0),
            )
            return trajectory, child.boundary.transcript_count, zero_transcript_root
        finally:
            await _close_store(store)

    trajectory, child_transcript_count, zero_transcript_root = asyncio.run(scenario())
    assert child_transcript_count == 0
    assert [child.session.id for child in trajectory.children if child.session is not None] == [
        "child"
    ]
    assert zero_transcript_root.transcript == ()


def test_trajectory_from_session_ignores_excluded_origin_payloads_outside_retained_budget():
    class IdentityOnlyOriginStore(InMemorySessionStore):
        async def query_events_bounded(self, query, *, max_bytes):
            raise AssertionError("Trajectory admission must not hydrate origin event payloads.")

        async def query_session_topology(self, query):
            raise AssertionError("Trajectory admission must use the minimal lineage projection.")

    async def scenario():
        store = IdentityOnlyOriginStore()
        root_interaction = await _create_running_session(store, "root")
        await _finish_session(store, "root", root_interaction)
        root = await store.load_terminal_session_evidence("root")
        for index in range(12):
            await _create_running_session(
                store,
                f"late-child-{index}",
                parent_session_id="root",
                started_payload={"padding": " " * 100_000},
            )
        return await trajectory_from_session(
            CayuApp(session_store=store, enable_logging=False),
            "root",
            bounds=SessionTrajectoryBounds(max_total_bytes=root.boundary.total_bytes),
        )

    trajectory = asyncio.run(scenario())
    assert trajectory.children == ()


def test_trajectory_from_session_excludes_unbounded_display_fields_without_hydrating_them(
    trajectory_store_case,
):
    async def scenario():
        store = await _open_store(trajectory_store_case)
        try:
            root_interaction = await _create_running_session(store, "root")
            await _finish_session(store, "root", root_interaction)
            root = await store.load_terminal_session_evidence("root")
            await _create_running_session(
                store,
                "late-child",
                parent_session_id="root",
                model="m" * 2_000_000,
            )
            return await trajectory_from_session(
                CayuApp(session_store=store, enable_logging=False),
                "root",
                bounds=SessionTrajectoryBounds(max_total_bytes=root.boundary.total_bytes),
            )
        finally:
            await _close_store(store)

    trajectory = asyncio.run(scenario())
    assert trajectory.children == ()


def test_trajectory_from_session_origin_identity_is_backend_neutral_for_jsonb_expansion(
    trajectory_store_case,
):
    async def scenario():
        store = await _open_store(trajectory_store_case)
        try:
            root_interaction = await _create_running_session(store, "root")
            await _finish_session(store, "root", root_interaction)
            await _create_running_session(
                store,
                "late-child",
                parent_session_id="root",
                started_payload={"values": [1e-100] * 10_000},
            )
            origin_records = await store.query_events(
                EventQuery(
                    session_id="late-child",
                    event_type=EventType.SESSION_STARTED,
                    limit=2,
                )
            )
            assert len(origin_records) == 1
            canonical_bytes = compact_json_utf8_size(origin_records[0].model_dump(mode="json"))
            trajectory = await trajectory_from_session(
                CayuApp(session_store=store, enable_logging=False),
                "root",
                bounds=SessionTrajectoryBounds(max_record_bytes=canonical_bytes),
            )
            return trajectory
        finally:
            await _close_store(store)

    trajectory = asyncio.run(scenario())
    assert trajectory.children == ()


def test_trajectory_from_session_enforces_the_record_byte_bound():
    async def scenario():
        store = InMemorySessionStore()
        interaction_id = await _create_running_session(store, "root")
        await _finish_session(store, "root", interaction_id)
        evidence = await store.load_terminal_session_evidence("root")
        return await trajectory_from_session(
            CayuApp(session_store=store, enable_logging=False),
            "root",
            bounds=SessionTrajectoryBounds(
                max_record_bytes=evidence.boundary.largest_record_bytes - 1
            ),
        )

    with pytest.raises(SessionTrajectoryError) as captured:
        asyncio.run(scenario())
    assert captured.value.code is SessionTrajectoryErrorCode.TERMINAL_EVIDENCE_REJECTED
    assert captured.value.terminal_code is TerminalSessionEvidenceErrorCode.RECORD_BYTES_EXCEEDED
    assert captured.value.session_id == "root"


def test_trajectory_from_session_classifies_descendant_enumeration_failures():
    class FailingLineageStore(InMemorySessionStore):
        async def query_session_lineage(self, query):
            raise RuntimeError("lineage unavailable")

    async def scenario():
        store = FailingLineageStore()
        interaction_id = await _create_running_session(store, "root")
        await _finish_session(store, "root", interaction_id)
        return await trajectory_from_session(
            CayuApp(session_store=store, enable_logging=False),
            "root",
        )

    with pytest.raises(SessionTrajectoryError) as captured:
        asyncio.run(scenario())
    assert captured.value.code is SessionTrajectoryErrorCode.DESCENDANT_ENUMERATION_FAILED
    assert captured.value.session_id == "root"


def test_trajectory_from_session_rejects_a_contradictory_parent_projection():
    class ContradictoryParentStore(InMemorySessionStore):
        async def query_session_lineage(self, query):
            result = await super().query_session_lineage(query)
            if query.parent_session_id != "root" or not result.children:
                return result
            child = result.children[0].model_copy(update={"parent_session_id": "different-parent"})
            return result.model_copy(update={"children": (child,)})

    async def scenario():
        store = ContradictoryParentStore()
        root_interaction = await _create_running_session(store, "root")
        child_interaction = await _create_running_session(
            store,
            "child",
            parent_session_id="root",
        )
        await _finish_session(store, "root", root_interaction)
        await _finish_session(store, "child", child_interaction)
        return await trajectory_from_session(
            CayuApp(session_store=store, enable_logging=False),
            "root",
        )

    with pytest.raises(SessionTrajectoryError) as captured:
        asyncio.run(scenario())
    assert captured.value.code is SessionTrajectoryErrorCode.PARENT_CONTRADICTION
    assert captured.value.session_id == "child"
    assert captured.value.parent_session_id == "root"


def test_trajectory_from_session_rejects_missing_child_origin_evidence():
    class MissingOriginStore(InMemorySessionStore):
        async def query_session_lineage(self, query):
            result = await super().query_session_lineage(query)
            if query.parent_session_id != "root" or not result.children:
                return result
            child = result.children[0].model_copy(update={"origin_events": ()})
            return result.model_copy(update={"children": (child,)})

    async def scenario():
        store = MissingOriginStore()
        root_interaction = await _create_running_session(store, "root")
        child_interaction = await _create_running_session(
            store,
            "child",
            parent_session_id="root",
        )
        await _finish_session(store, "root", root_interaction)
        await _finish_session(store, "child", child_interaction)
        return await trajectory_from_session(
            CayuApp(session_store=store, enable_logging=False),
            "root",
        )

    with pytest.raises(SessionTrajectoryError) as captured:
        asyncio.run(scenario())
    assert captured.value.code is SessionTrajectoryErrorCode.ORIGIN_EVIDENCE_REJECTED
    assert captured.value.session_id == "child"


@pytest.mark.parametrize(
    ("origin_event_type", "origin_payload"),
    [
        (EventType.SESSION_STARTED, {"agent_name": "assistant"}),
        (EventType.SESSION_STARTED, {"parent_session_id": "wrong-parent"}),
        (EventType.SESSION_FORKED, {"parent_session_id": "root"}),
        (
            EventType.SESSION_FORKED,
            {"parent_session_id": "root", "source_session_id": "wrong-parent"},
        ),
    ],
    ids=("missing-parent", "wrong-parent", "missing-fork-source", "wrong-fork-source"),
)
def test_trajectory_from_session_rejects_contradictory_authoritative_origin_lineage(
    trajectory_store_case,
    origin_event_type,
    origin_payload,
):
    async def scenario():
        store = await _open_store(trajectory_store_case)
        try:
            root_interaction = await _create_running_session(store, "root")
            child_interaction = await _create_running_session(
                store,
                "child",
                parent_session_id="root",
                started_payload=origin_payload,
                origin_event_type=origin_event_type,
            )
            await _finish_session(store, "root", root_interaction)
            await _finish_session(store, "child", child_interaction)
            app = CayuApp(session_store=store, enable_logging=False)
            for target_session_id in ("root", "child"):
                with pytest.raises(SessionTrajectoryError) as captured:
                    await trajectory_from_session(app, target_session_id)
                assert captured.value.code is SessionTrajectoryErrorCode.ORIGIN_EVIDENCE_REJECTED
                assert captured.value.session_id == "child"
                assert captured.value.parent_session_id == "root"
        finally:
            await _close_store(store)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("origin_event_type", "origin_payload", "untrusted_origin_fields"),
    [
        (
            EventType.SESSION_STARTED,
            {"parent_session_id": "root"},
            ("parent_session_id",),
        ),
        (
            EventType.SESSION_FORKED,
            {"parent_session_id": "root", "source_session_id": "root"},
            ("source_session_id",),
        ),
    ],
    ids=("matching-untrusted-parent", "matching-untrusted-fork-source"),
)
def test_trajectory_from_session_rejects_matching_untrusted_origin_lineage(
    trajectory_store_case,
    origin_event_type,
    origin_payload,
    untrusted_origin_fields,
):
    async def scenario():
        store = await _open_store(trajectory_store_case)
        try:
            root_interaction = await _create_running_session(store, "root")
            child_interaction = await _create_running_session(
                store,
                "child",
                parent_session_id="root",
                started_payload=origin_payload,
                origin_event_type=origin_event_type,
                untrusted_origin_fields=untrusted_origin_fields,
            )
            await _finish_session(store, "root", root_interaction)
            await _finish_session(store, "child", child_interaction)
            app = CayuApp(session_store=store, enable_logging=False)
            for target_session_id in ("root", "child"):
                with pytest.raises(SessionTrajectoryError) as captured:
                    await trajectory_from_session(app, target_session_id)
                assert captured.value.code is SessionTrajectoryErrorCode.ORIGIN_EVIDENCE_REJECTED
                assert captured.value.session_id == "child"
                assert captured.value.parent_session_id == "root"
        finally:
            await _close_store(store)

    asyncio.run(scenario())


def test_trajectory_from_session_rejects_matching_lineage_when_reader_loses_authority():
    class AuthorityDroppingStore(InMemorySessionStore):
        async def load_terminal_session_evidence(self, session_id, *, limits=None):
            evidence = await super().load_terminal_session_evidence(session_id, limits=limits)
            return type(evidence).model_validate(evidence.model_dump(mode="python"))

    async def scenario():
        store = AuthorityDroppingStore()
        root_interaction = await _create_running_session(store, "root")
        child_interaction = await _create_running_session(
            store,
            "child",
            parent_session_id="root",
        )
        await _finish_session(store, "root", root_interaction)
        await _finish_session(store, "child", child_interaction)

        child_evidence = await store.load_terminal_session_evidence("child")
        child_origin = next(
            record.event
            for record in child_evidence.events
            if record.event.type is EventType.SESSION_STARTED
        )
        assert child_origin.payload["parent_session_id"] == "root"
        assert not event_payload_authority_is_runtime_generated(
            child_origin,
            field_name="parent_session_id",
            value="root",
        )

        return await trajectory_from_session(
            CayuApp(session_store=store, enable_logging=False),
            "root",
        )

    with pytest.raises(SessionTrajectoryError) as captured:
        asyncio.run(scenario())
    assert captured.value.code is SessionTrajectoryErrorCode.ORIGIN_EVIDENCE_REJECTED
    assert captured.value.session_id == "child"
    assert captured.value.parent_session_id == "root"


def test_trajectory_from_session_accepts_an_authoritative_fork_origin(
    trajectory_store_case,
):
    async def scenario():
        store = await _open_store(trajectory_store_case)
        try:
            root_interaction = await _create_running_session(store, "root")
            child_interaction = await _create_running_session(
                store,
                "child",
                parent_session_id="root",
                origin_event_type=EventType.SESSION_FORKED,
            )
            await _finish_session(store, "root", root_interaction)
            await _finish_session(store, "child", child_interaction)
            return await trajectory_from_session(
                CayuApp(session_store=store, enable_logging=False),
                "root",
            )
        finally:
            await _close_store(store)

    trajectory = asyncio.run(scenario())
    assert [child.session.id for child in trajectory.children if child.session is not None] == [
        "child"
    ]


def test_runtime_child_start_persists_authoritative_parent_lineage():
    async def scenario():
        store = InMemorySessionStore()
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="root",
                messages=[Message.text("user", "parent")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.text_delta("done"),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ]
                ],
                name="fake",
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        async for _event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="child",
                parent_session_id="root",
                messages=[Message.text("user", "child")],
            )
        ):
            pass
        records = await store.query_events(
            EventQuery(
                session_id="child",
                event_type=EventType.SESSION_STARTED,
                limit=2,
            )
        )
        assert len(records) == 1
        return records[0].event

    origin = asyncio.run(scenario())
    assert origin.payload["parent_session_id"] == "root"
    assert event_payload_authority_is_runtime_generated(
        origin,
        field_name="parent_session_id",
        value="root",
    )


def test_trajectory_from_session_rejects_an_admitted_child_read_failure():
    class FailingChildReadStore(InMemorySessionStore):
        async def load_terminal_session_evidence(self, session_id, *, limits=None):
            if session_id == "child":
                raise RuntimeError("store read failed")
            return await super().load_terminal_session_evidence(session_id, limits=limits)

    async def scenario():
        store = FailingChildReadStore()
        root_interaction = await _create_running_session(store, "root")
        child_interaction = await _create_running_session(
            store,
            "child",
            parent_session_id="root",
        )
        await _finish_session(store, "root", root_interaction)
        await _finish_session(store, "child", child_interaction)
        return await trajectory_from_session(
            CayuApp(session_store=store, enable_logging=False),
            "root",
        )

    with pytest.raises(SessionTrajectoryError) as captured:
        asyncio.run(scenario())
    assert captured.value.code is SessionTrajectoryErrorCode.EVIDENCE_READ_FAILED
    assert captured.value.session_id == "child"
    assert captured.value.parent_session_id == "root"


@pytest.mark.parametrize("max_sessions", [100, 1])
def test_trajectory_from_session_rejects_a_closure_that_changes_during_capture(
    max_sessions,
):
    class ChangingClosureStore(InMemorySessionStore):
        root_queries = 0

        async def query_session_lineage(self, query):
            result = await super().query_session_lineage(query)
            if query.parent_session_id != "root":
                return result
            self.root_queries += 1
            if self.root_queries != 1:
                return result
            document = result.model_dump(mode="python")
            document["children"] = ()
            document["next_cursor"] = None
            document["has_more"] = False
            return SessionLineageResult.model_validate(document)

    async def scenario():
        store = ChangingClosureStore()
        root_interaction = await _create_running_session(store, "root")
        child_interaction = await _create_running_session(
            store,
            "child",
            parent_session_id="root",
        )
        await _finish_session(store, "root", root_interaction)
        await _finish_session(store, "child", child_interaction)
        return await trajectory_from_session(
            CayuApp(session_store=store, enable_logging=False),
            "root",
            bounds=SessionTrajectoryBounds(max_sessions=max_sessions),
        )

    with pytest.raises(SessionTrajectoryError) as captured:
        asyncio.run(scenario())
    assert captured.value.code is SessionTrajectoryErrorCode.CLOSURE_CHANGED
    assert captured.value.session_id == "root"


def test_trajectory_from_session_bounds_lineage_candidates_independently(monkeypatch):
    import cayu.evals.trajectory as trajectory_module

    monkeypatch.setattr(
        trajectory_module,
        "_SESSION_TRAJECTORY_HARD_MAX_LINEAGE_CANDIDATES",
        1,
    )

    async def scenario():
        store = InMemorySessionStore()
        root_interaction = await _create_running_session(store, "root")
        await _finish_session(store, "root", root_interaction)
        for child_id in ("post-terminal-child-1", "post-terminal-child-2"):
            interaction_id = await _create_running_session(
                store,
                child_id,
                parent_session_id="root",
            )
            await _finish_session(store, child_id, interaction_id)

        return await trajectory_from_session(
            CayuApp(session_store=store, enable_logging=False),
            "root",
            bounds=SessionTrajectoryBounds(max_sessions=1),
        )

    with pytest.raises(SessionTrajectoryError) as captured:
        asyncio.run(scenario())
    assert captured.value.code is SessionTrajectoryErrorCode.SESSION_LIMIT_EXCEEDED
    assert captured.value.limit == 1
    assert captured.value.observed == 2


def test_fresh_descendant_capture_marks_a_global_session_limit_incomplete():
    async def scenario():
        store = InMemorySessionStore()
        root_interaction = await _create_running_session(store, "root")
        await _finish_session(store, "root", root_interaction)
        for child_id in ("child-1", "child-2"):
            interaction_id = await _create_running_session(
                store,
                child_id,
                parent_session_id="root",
            )
            await _finish_session(store, child_id, interaction_id)

        state = _CaptureState(
            bounds=SessionTrajectoryBounds(max_sessions=2),
            strict=False,
        )
        root_evidence = await store.load_terminal_session_evidence("root")
        state.retain(root_evidence)
        incomplete = _IncompleteFlag()
        children = await _build_child_trajectories(
            CayuApp(session_store=store, enable_logging=False),
            "root",
            visited={"root"},
            incomplete=incomplete,
            state=state,
        )
        return children, incomplete.value

    children, incomplete = asyncio.run(scenario())
    assert [child.session.id for child in children if child.session is not None] == ["child-1"]
    assert incomplete is True


def test_fresh_descendant_capture_without_lineage_excludes_post_terminal_child():
    class NoLineageStore(InMemorySessionStore):
        supports_session_lineage = False

    async def scenario():
        store = NoLineageStore()
        root_interaction = await _create_running_session(store, "root")
        child_interaction = await _create_running_session(
            store,
            "child-before-cutoff",
            parent_session_id="root",
        )
        await _finish_session(store, "root", root_interaction)
        await _finish_session(store, "child-before-cutoff", child_interaction)
        late_interaction = await _create_running_session(
            store,
            "child-after-cutoff",
            parent_session_id="root",
        )
        await _finish_session(store, "child-after-cutoff", late_interaction)

        state = _CaptureState(
            bounds=SessionTrajectoryBounds(max_sessions=2),
            strict=False,
        )
        root_evidence = await store.load_terminal_session_evidence("root")
        state.retain(root_evidence)
        incomplete = _IncompleteFlag()
        children = await _build_child_trajectories(
            CayuApp(session_store=store, enable_logging=False),
            "root",
            visited={"root"},
            incomplete=incomplete,
            parent_terminal_sequence=root_evidence.boundary.terminal_event_sequence,
            state=state,
        )
        return children, incomplete.value, state.retained_session_ids

    children, incomplete, retained_session_ids = asyncio.run(scenario())
    assert [child.session.id for child in children if child.session is not None] == [
        "child-before-cutoff"
    ]
    assert incomplete is False
    assert retained_session_ids == {"root", "child-before-cutoff"}


def test_fresh_descendant_capture_marks_a_depth_limit_incomplete():
    async def scenario():
        store = InMemorySessionStore()
        root_interaction = await _create_running_session(store, "root")
        child_interaction = await _create_running_session(
            store,
            "child",
            parent_session_id="root",
        )
        grandchild_interaction = await _create_running_session(
            store,
            "grandchild",
            parent_session_id="child",
        )
        await _finish_session(store, "root", root_interaction)
        await _finish_session(store, "child", child_interaction)
        await _finish_session(store, "grandchild", grandchild_interaction)

        state = _CaptureState(
            bounds=SessionTrajectoryBounds(max_depth=2),
            strict=False,
        )
        root_evidence = await store.load_terminal_session_evidence("root")
        state.retain(root_evidence)
        incomplete = _IncompleteFlag()
        children = await _build_child_trajectories(
            CayuApp(session_store=store, enable_logging=False),
            "root",
            visited={"root"},
            incomplete=incomplete,
            state=state,
        )
        return children, incomplete.value

    children, incomplete = asyncio.run(scenario())
    assert len(children) == 1
    assert children[0].session is not None
    assert children[0].session.id == "child"
    assert children[0].children == ()
    assert children[0].children_incomplete is True
    assert incomplete is True
