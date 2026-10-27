"""Which execution deadline a paused session continues under, and where it came from."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
from tests.runtime.runtime_tool_support import SideEffectTool

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.approvals import ToolApprovalDecision, ToolApprovalRequest, UserInputResponse
from cayu.deadlines import (
    EXECUTION_DEADLINE_METADATA_KEY,
    EXECUTION_DEADLINE_ORIGIN_METADATA_KEY,
    ExecutionDeadline,
    ExecutionDeadlineExceeded,
    ExecutionDeadlineReport,
    HandlerTimeLimit,
    deadline_from_metadata,
    default_run_deadline,
    execution_deadline_scope,
    waiting_for_a_person_counts,
)
from cayu.evals.testing import ScriptedModelProvider
from cayu.messages import MessageRole
from cayu.providers import ModelRequest, ModelStreamEvent
from cayu.sessions import (
    InMemorySessionStore,
    PendingActionQuery,
    ResumeRequest,
    SessionRunFenced,
    SessionStatus,
    SessionStatusConflict,
)
from cayu.storage import SQLiteSessionStore
from cayu.tasks import InMemoryTaskStore, TaskCreate, TaskHandlerOutcome, TaskQuery
from cayu.tasks.worker import run_task_worker
from cayu.tools import AlwaysRequireApprovalToolPolicy, UserInputTool

_HUMAN_WAITS = {"tool_approval_required", "user_input_required"}


@pytest.fixture
def skip_ahead(monkeypatch):
    """Move the deadline clock (wall and monotonic) forward; events keep real time."""

    import cayu.deadlines as module

    offset = SimpleNamespace(seconds=0.0)
    real_datetime = datetime

    class Clock(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return real_datetime.now(tz) + timedelta(seconds=offset.seconds)

    monkeypatch.setattr(module, "datetime", Clock)
    monkeypatch.setattr(
        module, "time", SimpleNamespace(monotonic=lambda: time.monotonic() + offset.seconds)
    )

    def advance(seconds: float) -> None:
        offset.seconds += seconds

    advance.now = lambda: datetime.now(UTC) + timedelta(seconds=offset.seconds)
    return advance


def _respond(request: ModelRequest) -> list[ModelStreamEvent]:
    if request.model == "plain" or any(
        message.role is MessageRole.TOOL for message in request.messages
    ):
        return [
            ModelStreamEvent.text_delta("Done."),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    call = (
        ModelStreamEvent.tool_call(id="call", name="ask_user", arguments={"question": "Where?"})
        if request.model == "ask"
        else ModelStreamEvent.tool_call(id="call", name="side_effect", arguments={})
    )
    return [call, ModelStreamEvent.completed({"finish_reason": "tool_calls"})]


def _app(store, **options) -> CayuApp:
    app = CayuApp(session_store=store, enable_logging=False, **options)
    app.register_provider(ScriptedModelProvider(response_factory=_respond), default=True)
    app.register_agent(AgentSpec(name="plain", model="plain"))
    app.register_agent(AgentSpec(name="user_input", model="ask"), tools=[UserInputTool()])
    app.register_agent(
        AgentSpec(name="approval", model="approve"),
        tools=[SideEffectTool()],
        tool_policy=AlwaysRequireApprovalToolPolicy(),
    )
    return app


def _store(backend: str, tmp_path, request):
    if backend == "memory":
        return InMemorySessionStore()
    if backend == "sqlite":
        return SQLiteSessionStore(tmp_path / "sessions.db")
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresSessionStore

    return PostgresSessionStore(
        request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
    )


async def _run(app: CayuApp, session_id: str, agent: str, **options) -> list:
    return [
        event
        async for event in app.run(
            RunRequest(
                agent_name=agent,
                session_id=session_id,
                messages=[Message.text("user", "go")],
                **options,
            )
        )
    ]


async def _pause_in_handler(app: CayuApp, session_id: str, gate: str) -> None:
    # What run_task_worker(execution_timeout_s=120, execution_grace_s=10) binds.
    async with HandlerTimeLimit(120, grace_s=10, source="task_worker") as limit:
        with default_run_deadline(limit.run_deadline()):
            events = await _run(app, session_id, gate)
    assert events[-1].payload["interruption_type"] in _HUMAN_WAITS


async def _settle(app: CayuApp, session_id: str, gate: str, deadline=None) -> list:
    checkpoint = await app.session_store.load_checkpoint(session_id)
    if gate == "approval":
        pending = checkpoint["pending_tool_approval"]
        stream = app.resolve_tool_approval(
            ToolApprovalRequest(
                session_id=session_id,
                approval_id=pending["approval_id"],
                tool_round_id=pending["tool_round_id"],
                tool_call_id=pending["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
                execution_deadline=deadline,
            )
        )
    else:
        stream = app.resolve_user_input(
            UserInputResponse(
                session_id=session_id,
                input_id=checkpoint["pending_user_input"]["input_id"],
                answer="staging",
                execution_deadline=deadline,
            )
        )
    return [event async for event in stream]


async def _pending_report(app: CayuApp, session_id: str) -> ExecutionDeadlineReport | None:
    page = await app.session_store.query_pending_actions(PendingActionQuery(session_id=session_id))
    [action] = page.actions
    return action.session.execution_deadline


def test_deadline_exceeded_names_expiry_origin_and_source() -> None:
    expires_at = datetime(2026, 10, 26, 22, 37, 5, 123456, tzinfo=UTC)
    handler_default = ExecutionDeadline(expires_at=expires_at, source="task_worker")
    error = ExecutionDeadlineExceeded(handler_default.with_origin("run_default"), "execution")
    assert str(error) == (
        "Execution deadline expired before execution; the deadline was "
        "2026-10-26T22:37:05Z, the default for runs started inside a task handler "
        "(source task_worker)."
    )
    assert str(ExecutionDeadlineExceeded(handler_default.with_origin("explicit"), "resume")) == (
        "Execution deadline expired before resume; the deadline was 2026-10-26T22:37:05Z, "
        "set explicitly with execution_deadline (source task_worker)."
    )
    # A deadline stored before origins existed still names its expiry and source.
    legacy = deadline_from_metadata(
        {EXECUTION_DEADLINE_METADATA_KEY: handler_default.model_dump(mode="json")}
    )
    assert legacy.origin is None
    assert str(ExecutionDeadlineExceeded(legacy, "execution")).endswith(
        "the deadline was 2026-10-26T22:37:05Z (source task_worker)."
    )
    with pytest.raises(ValueError, match="Unknown execution deadline origin"):
        handler_default.with_origin("guess")
    # Only a top-level session's inherited handler default leaves a person's wait out.
    assert not waiting_for_a_person_counts(
        handler_default.with_origin("run_default"), top_level=True
    )
    assert waiting_for_a_person_counts(handler_default.with_origin("run_default"), top_level=False)
    assert waiting_for_a_person_counts(handler_default.with_origin("explicit"), top_level=True)
    assert waiting_for_a_person_counts(legacy, top_level=True)


def test_new_runs_record_where_their_deadline_came_from() -> None:
    async def scenario() -> None:
        app = _app(InMemorySessionStore())
        try:
            await _run(app, "explicit", "plain", execution_deadline=ExecutionDeadline.after(60))
            with default_run_deadline(ExecutionDeadline.after(60, source="task_worker")):
                await _run(app, "handler", "plain")
                # An explicit deadline replaces the default, even a longer one.
                await _run(
                    app, "override", "plain", execution_deadline=ExecutionDeadline.after(3600)
                )
            async with execution_deadline_scope(ExecutionDeadline.after(60, source="workflow")):
                await _run(app, "enclosing", "plain")
            await _run(app, "unbounded", "plain")

            stored = {
                session_id: (await app.session_store.load(session_id))
                for session_id in ("explicit", "handler", "override", "enclosing", "unbounded")
            }
            assert {key: session.execution_deadline.origin for key, session in stored.items()} == {
                "explicit": "explicit",
                "handler": "run_default",
                "override": "explicit",
                "enclosing": "enclosing",
                "unbounded": None,
            }
            assert stored["handler"].execution_deadline.source == "task_worker"
            assert stored["handler"].metadata[EXECUTION_DEADLINE_ORIGIN_METADATA_KEY] == (
                "run_default"
            )
            assert EXECUTION_DEADLINE_ORIGIN_METADATA_KEY not in stored["unbounded"].metadata
            # The origin is runtime-owned, like the deadline itself.
            with pytest.raises(ValueError, match="execution-deadline authority"):
                RunRequest(
                    agent_name="plain",
                    messages=[Message.text("user", "go")],
                    metadata={EXECUTION_DEADLINE_ORIGIN_METADATA_KEY: "explicit"},
                )
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_task_worker_runs_record_the_handler_default() -> None:
    async def scenario() -> None:
        tasks = InMemoryTaskStore()
        app = _app(InMemorySessionStore(), task_store=tasks)

        async def handle(runtime, task, worker_id):
            await _run(runtime, "scan-filing", "approval")
            return TaskHandlerOutcome.COMPLETED

        try:
            await tasks.create_task(TaskCreate(task_id="scan", type="scan"))
            processed = await run_task_worker(
                app,
                tasks,
                handle,
                worker_id="worker",
                query=TaskQuery(type="scan"),
                max_tasks=1,
                execution_timeout_s=120,
            )
            assert processed == 1
            session = await app.session_store.load("scan-filing")
            assert session.status is SessionStatus.INTERRUPTED
            deadline = session.execution_deadline
            assert (deadline.origin, deadline.source) == ("run_default", "task_worker")
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("gate", ["approval", "user_input"])
def test_a_persons_wait_does_not_use_up_an_inherited_handler_deadline(
    skip_ahead, tmp_path, request, backend, gate
) -> None:
    session_id = f"filing-{uuid4().hex[:12]}"

    async def scenario() -> None:
        app = _app(_store(backend, tmp_path, request))
        try:
            await _pause_in_handler(app, session_id, gate)
            paused = await app.session_store.load(session_id)
            handler_default = paused.execution_deadline
            assert handler_default.origin == "run_default"
            report = await _pending_report(app, session_id)
            assert report == ExecutionDeadlineReport(
                expires_at=handler_default.expires_at,
                source="task_worker",
                origin="run_default",
                waiting_counts=False,
            )
            [interrupted] = [
                event
                for event in await app.session_store.load_events(session_id)
                if event.type is EventType.SESSION_INTERRUPTED
            ]
            left_at_pause = (handler_default.expires_at - interrupted.timestamp).total_seconds()

            # The person decides long after the handler's limit passed.
            skip_ahead(3600)
            settled_at = skip_ahead.now()
            events = await _settle(app, session_id, gate)
            assert events[-1].type is EventType.SESSION_COMPLETED

            session = await app.session_store.load(session_id)
            renewed = session.execution_deadline
            assert renewed.origin == "run_default"
            assert renewed.source == "task_worker"
            expected = settled_at + timedelta(seconds=left_at_pause)
            assert abs((renewed.expires_at - expected).total_seconds()) < 5
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("gate", ["approval", "user_input"])
def test_an_explicit_deadline_still_expires_until_a_settle_replaces_it(
    skip_ahead, tmp_path, request, backend, gate
) -> None:
    async def scenario() -> None:
        app = _app(_store(backend, tmp_path, request))
        try:
            events = await _run(app, "filing", gate, execution_deadline=ExecutionDeadline.after(60))
            assert events[-1].payload["interruption_type"] in _HUMAN_WAITS
            stored = (await app.session_store.load("filing")).execution_deadline
            report = await _pending_report(app, "filing")
            assert (report.origin, report.waiting_counts) == ("explicit", True)

            skip_ahead(120)
            expiry = stored.expires_at.isoformat(timespec="seconds").replace("+00:00", "Z")
            with pytest.raises(ExecutionDeadlineExceeded) as expired:
                await _settle(app, "filing", gate)
            assert f"the deadline was {expiry}, set explicitly with execution_deadline" in str(
                expired.value
            )
            # A replacement that has already passed fails the same admission as a new run.
            with pytest.raises(ExecutionDeadlineExceeded, match="before resume"):
                await _settle(app, "filing", gate, ExecutionDeadline.after(0))
            session = await app.session_store.load("filing")
            assert session.execution_deadline.expires_at == stored.expires_at
            assert session.status is SessionStatus.INTERRUPTED
            assert await _pending_report(app, "filing") is not None

            events = await _settle(app, "filing", gate, ExecutionDeadline.after(600))
            assert events[-1].type is EventType.SESSION_COMPLETED
            replaced = (await app.session_store.load("filing")).execution_deadline
            assert replaced.origin == "explicit"
            assert replaced.expires_at > stored.expires_at
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_a_settle_for_another_pause_leaves_the_deadline_alone(skip_ahead) -> None:
    async def scenario() -> None:
        app = _app(InMemorySessionStore())
        try:
            await _run(app, "filing", "approval", execution_deadline=ExecutionDeadline.after(60))
            stored = (await app.session_store.load("filing")).execution_deadline
            pending = (await app.session_store.load_checkpoint("filing"))["pending_tool_approval"]
            skip_ahead(120)
            with pytest.raises(ExecutionDeadlineExceeded, match="set explicitly"):
                [
                    event
                    async for event in app.resolve_tool_approval(
                        ToolApprovalRequest(
                            session_id="filing",
                            approval_id="some-other-approval",
                            tool_round_id=pending["tool_round_id"],
                            tool_call_id=pending["tool_call_id"],
                            decision=ToolApprovalDecision.APPROVE,
                            execution_deadline=ExecutionDeadline.after(600),
                        )
                    )
                ]
            session = await app.session_store.load("filing")
            assert session.execution_deadline.expires_at == stored.expires_at
            assert session.execution_deadline.origin == "explicit"
            # A resume can't continue a session that waits on a person either.
            with pytest.raises(ExecutionDeadlineExceeded, match="set explicitly"):
                [
                    event
                    async for event in app.resume(
                        ResumeRequest(
                            session_id="filing",
                            messages=[Message.text("user", "again")],
                            execution_deadline=ExecutionDeadline.after(600),
                        )
                    )
                ]
            session = await app.session_store.load("filing")
            assert session.execution_deadline.expires_at == stored.expires_at
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_resume_can_carry_a_new_deadline(skip_ahead, tmp_path, request, backend) -> None:
    async def scenario() -> None:
        app = _app(_store(backend, tmp_path, request))
        try:
            await _run(app, "chat", "plain", execution_deadline=ExecutionDeadline.after(60))
            skip_ahead(120)

            def resume(deadline=None) -> ResumeRequest:
                return ResumeRequest(
                    session_id="chat",
                    messages=[Message.text("user", "again")],
                    execution_deadline=deadline,
                )

            with pytest.raises(ExecutionDeadlineExceeded, match="set explicitly"):
                [event async for event in app.resume(resume())]
            events = [event async for event in app.resume(resume(ExecutionDeadline.after(600)))]
            assert events[-1].type is EventType.SESSION_COMPLETED
            assert (await app.session_store.load("chat")).execution_deadline.origin == "explicit"
            # An unbounded deadline is explicit too: it removes the stored one.
            events = [event async for event in app.resume(resume(ExecutionDeadline()))]
            assert events[-1].type is EventType.SESSION_COMPLETED
            session = await app.session_store.load("chat")
            assert session.execution_deadline.expires_at is None
            assert EXECUTION_DEADLINE_ORIGIN_METADATA_KEY not in session.metadata
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_store_replaces_a_deadline_only_between_runs_of_the_session_read(
    tmp_path, request, backend
) -> None:
    session_id = f"chat-{uuid4().hex[:12]}"

    async def scenario() -> None:
        store = _store(backend, tmp_path, request)
        app = _app(store)
        try:
            await _run(app, session_id, "plain", execution_deadline=ExecutionDeadline.after(60))
            session = await store.load(session_id)
            replacement = ExecutionDeadline.after(600, source="operator").with_origin("explicit")
            for instance_id, epoch in (
                ("another-incarnation", session.run_epoch),
                (session.instance_id, session.run_epoch + 1),
            ):
                with pytest.raises(SessionRunFenced):
                    await store.replace_execution_deadline(
                        session_id,
                        replacement,
                        expected_session_instance_id=instance_id,
                        expected_run_epoch=epoch,
                    )
            replaced = await store.replace_execution_deadline(
                session_id,
                replacement,
                expected_session_instance_id=session.instance_id,
                expected_run_epoch=session.run_epoch,
            )
            assert replaced.execution_deadline.expires_at == replacement.expires_at
            assert replaced.execution_deadline.origin == "explicit"
            assert (await store.load(session_id)).execution_deadline.source == "operator"

            running = await store.transition_status(
                session_id, from_statuses={SessionStatus.COMPLETED}, to_status=SessionStatus.RUNNING
            )
            with pytest.raises(SessionStatusConflict, match="only between runs"):
                await store.replace_execution_deadline(
                    session_id,
                    ExecutionDeadline.after(60),
                    expected_session_instance_id=running.instance_id,
                    expected_run_epoch=running.run_epoch,
                )
            with pytest.raises(KeyError):
                await store.replace_execution_deadline(
                    "missing",
                    replacement,
                    expected_session_instance_id=None,
                    expected_run_epoch=0,
                )
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_work_attempts_refuse_a_resume_deadline_they_would_ignore() -> None:
    async def scenario() -> None:
        app = _app(InMemorySessionStore())
        try:
            with pytest.raises(ValueError, match="applied by CayuApp.resume"):
                await app.admit_work_attempt(
                    ResumeRequest(
                        session_id="chat",
                        messages=[Message.text("user", "again")],
                        execution_deadline=ExecutionDeadline.after(60),
                    ),
                    execution=None,
                )
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_renewing_the_same_pause_twice_gives_back_the_same_time(skip_ahead) -> None:
    from cayu.runtime._continuation_deadline import prepare_continuation_deadline

    async def scenario() -> None:
        app = _app(InMemorySessionStore())
        try:
            await _pause_in_handler(app, "filing", "approval")
            paused = await app.session_store.load("filing")
            [interrupted] = [
                event
                for event in await app.session_store.load_events("filing")
                if event.type is EventType.SESSION_INTERRUPTED
            ]
            left = (paused.execution_deadline.expires_at - interrupted.timestamp).total_seconds()
            pending = (await app.session_store.load_checkpoint("filing"))["pending_tool_approval"]
            settle = ToolApprovalRequest(
                session_id="filing",
                approval_id=pending["approval_id"],
                tool_round_id=pending["tool_round_id"],
                tool_call_id=pending["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
            )
            # The renewal is stored, then the process stops before the decision runs.
            skip_ahead(3600)
            await prepare_continuation_deadline(app, paused, settle, participant_context=None)
            # A retry an hour later must not count the first renewal as time left.
            skip_ahead(3600)
            retried_at = skip_ahead.now()
            renewed = await prepare_continuation_deadline(
                app, paused, settle, participant_context=None
            )
            expected = retried_at + timedelta(seconds=left)
            assert abs((renewed.execution_deadline.expires_at - expected).total_seconds()) < 5
            events = await _settle(app, "filing", "approval")
            assert events[-1].type is EventType.SESSION_COMPLETED
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_a_refused_resume_or_settle_leaves_the_deadline_alone(tmp_path, request, backend) -> None:
    async def scenario() -> None:
        app = _app(_store(backend, tmp_path, request))
        try:
            await _run(app, "chat", "plain", execution_deadline=ExecutionDeadline.after(600))
            stale = await app.session_store.load("chat")
            await _run_resume(app, "chat")
            stored = await app.session_store.load("chat")
            for fence in (
                {"expected_run_epoch": stale.run_epoch},
                {"expected_session_instance_id": "another-incarnation"},
            ):
                with pytest.raises((SessionRunFenced, SessionStatusConflict)):
                    await _run_resume(
                        app, "chat", execution_deadline=ExecutionDeadline.after(86400), **fence
                    )
                session = await app.session_store.load("chat")
                assert session.metadata == stored.metadata
                assert session.run_epoch == stored.run_epoch
                assert session.instance_id == stored.instance_id

            await _run(app, "filing", "approval", execution_deadline=ExecutionDeadline.after(600))
            before = (await app.session_store.load("filing")).execution_deadline
            pending = (await app.session_store.load_checkpoint("filing"))["pending_tool_approval"]
            with pytest.raises(ValueError):
                [
                    event
                    async for event in app.resolve_tool_approval(
                        ToolApprovalRequest(
                            session_id="filing",
                            approval_id=pending["approval_id"],
                            tool_round_id=pending["tool_round_id"],
                            tool_call_id="another-call",
                            decision=ToolApprovalDecision.APPROVE,
                            execution_deadline=ExecutionDeadline.after(86400),
                        )
                    )
                ]
            after = (await app.session_store.load("filing")).execution_deadline
            assert after.expires_at == before.expires_at
        finally:
            await app.aclose()

    asyncio.run(scenario())


async def _run_resume(app: CayuApp, session_id: str, **options) -> list:
    return [
        event
        async for event in app.resume(
            ResumeRequest(
                session_id=session_id, messages=[Message.text("user", "again")], **options
            )
        )
    ]


def test_send_turn_keeps_an_existing_sessions_deadline() -> None:
    from cayu.sessions.turns import TurnRequest, send_turn

    async def scenario() -> None:
        app = _app(InMemorySessionStore())
        try:

            def turn(**options) -> TurnRequest:
                return TurnRequest(
                    agent_name="plain",
                    session_id="chat",
                    messages=[Message.text("user", "hi")],
                    **options,
                )

            await send_turn(app, turn(execution_deadline=ExecutionDeadline.after(60)))
            stored = (await app.session_store.load("chat")).execution_deadline
            outcome = await send_turn(app, turn())
            assert outcome.ok
            assert (await app.session_store.load("chat")).execution_deadline.expires_at == (
                stored.expires_at
            )
            with pytest.raises(ValueError, match="execution_deadline"):
                await send_turn(app, turn(execution_deadline=ExecutionDeadline.after(3600)))
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_another_subjects_binding_cannot_change_a_sessions_deadline(tmp_path) -> None:
    from cayu.resource_access import ResourceAccessPolicy, execution_access
    from cayu.sessions._access_rules import (
        SessionAccessRule,
        SessionAccessScope,
        SessionAccessSelector,
    )

    def scope(organization: str) -> SessionAccessScope:
        rule = SessionAccessRule(
            selectors=(SessionAccessSelector(key="organization", values=(organization,)),)
        )
        return SessionAccessScope(
            read=(rule,), create=(rule,), execute=(rule,), inspect_state=(rule,)
        )

    class Policy(ResourceAccessPolicy):
        authority = "deadline-test-policy-v1"

        async def resolve(self, subject):
            return scope({"alice": "acme", "bob": "globex"}[subject])

    async def scenario() -> None:
        policy = Policy()
        app = _app(SQLiteSessionStore(tmp_path / "access.db"), resource_access_policy=policy)
        try:
            for subject, session_id, organization in (
                ("bob", "b", "globex"),
                ("alice", "a", "acme"),
            ):
                access = await app.access(subject)
                [
                    event
                    async for event in access.run(
                        RunRequest(
                            agent_name="plain",
                            session_id=session_id,
                            messages=[Message.text("user", "hi")],
                            labels={"organization": organization},
                            execution_deadline=ExecutionDeadline.after(3600),
                        )
                    )
                ]
            bob_deadline = (await app.session_store.load("b")).execution_deadline
            alice = (await app.session_store.load("a")).invocation.resource_access
            with pytest.raises(Exception, match="not authorized"):
                async with execution_access(alice, policy, {"organization": "acme"}):
                    await _run_resume(app, "b", execution_deadline=ExecutionDeadline.after(1))
            after = (await app.session_store.load("b")).execution_deadline
            assert after.expires_at == bob_deadline.expires_at
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_revoked_execution_access_cannot_replace_a_sessions_deadline(tmp_path) -> None:
    from cayu._resource_access_errors import ResourceAccessDenied
    from cayu.resource_access import ResourceAccessPolicy
    from cayu.sessions._access_rules import (
        SessionAccessRule,
        SessionAccessScope,
        SessionAccessSelector,
    )

    class Policy(ResourceAccessPolicy):
        authority = "revoked-deadline-test-policy-v1"
        allow_execution = True

        async def resolve(self, subject):
            assert subject == "alice"
            rule = SessionAccessRule(
                selectors=(SessionAccessSelector(key="organization", values=("acme",)),)
            )
            return SessionAccessScope(
                read=(rule,),
                create=(rule,),
                execute=(rule,) if self.allow_execution else (),
                inspect_state=(rule,),
            )

    async def scenario() -> None:
        policy = Policy()
        app = _app(SQLiteSessionStore(tmp_path / "revoked.db"), resource_access_policy=policy)
        try:
            access = await app.access("alice")
            [
                event
                async for event in access.run(
                    RunRequest(
                        agent_name="plain",
                        session_id="chat",
                        messages=[Message.text("user", "hi")],
                        labels={"organization": "acme"},
                        execution_deadline=ExecutionDeadline.after(600),
                    )
                )
            ]
            before = await app.session_store.load("chat")
            policy.allow_execution = False
            with pytest.raises(ResourceAccessDenied):
                await _run_resume(app, "chat", execution_deadline=ExecutionDeadline.after(86400))
            after = await app.session_store.load("chat")
            assert after.metadata == before.metadata
            assert after.run_epoch == before.run_epoch
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("gate", ["approval", "user_input", "plain"])
@pytest.mark.parametrize("expired", [False, True])
def test_store_without_deadline_replacement_keeps_ordinary_continuations(
    skip_ahead, monkeypatch, gate, expired
) -> None:
    from cayu.sessions.store import SessionStore

    async def scenario() -> None:
        store = InMemorySessionStore()
        monkeypatch.setattr(
            store,
            "replace_execution_deadline",
            SessionStore.replace_execution_deadline.__get__(store),
        )
        app = _app(store)
        try:
            if gate == "plain":
                with default_run_deadline(ExecutionDeadline.after(120)):
                    await _run(app, "chat", gate)
            else:
                await _pause_in_handler(app, "chat", gate)
            before = await app.session_store.load("chat")
            assert before.execution_deadline.origin == "run_default"
            if expired:
                skip_ahead(3600)

            async def continue_session(deadline=None):
                if gate == "plain":
                    return await _run_resume(app, "chat", execution_deadline=deadline)
                return await _settle(app, "chat", gate, deadline)

            # Explicit replacements still require adapter support, without a
            # partial metadata write or accepting the pending human decision.
            with pytest.raises(NotImplementedError, match="replace_execution_deadline"):
                await continue_session(ExecutionDeadline.after(86400))
            assert (await app.session_store.load("chat")).metadata == before.metadata

            if expired:
                with pytest.raises(ExecutionDeadlineExceeded):
                    await continue_session()
            else:
                events = await continue_session()
                assert events[-1].type is EventType.SESSION_COMPLETED
            after = await app.session_store.load("chat")
            assert after.execution_deadline.expires_at == before.execution_deadline.expires_at
            assert after.execution_deadline.origin == before.execution_deadline.origin
        finally:
            await app.aclose()

    asyncio.run(scenario())
