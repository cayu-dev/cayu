"""A process that dies inside a woken round surfaces as manual recovery on its own.

The woken round started a tool that never recorded a result. After recovery the
session waits on its wake with no live attempt: reconciliation must find it and
give it an attempt, which settles it as ``manual_recovery_required`` without
running the tool again, and hands a task-owned session's task off unparked.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from tests.core.test_sleep_until import _Clock, _StableScriptedProvider, _tool_results

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu.providers.base import ModelRequest, ModelStreamEvent
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.pending_action_contracts import PendingActionQuery
from cayu.sessions.records import PendingActionKind, SessionStatus
from cayu.sessions.recovery import IncompleteSessionRecoveryRequest
from cayu.storage import SQLiteSessionStore, SQLiteTaskStore
from cayu.tasks import (
    InMemoryTaskStore,
    SessionWakeOutcome,
    TaskCreate,
    TaskHandlerOutcome,
    TaskStatus,
    deliver_session_wake,
    reconcile_session_wakes,
    run_task_worker,
    run_task_worker_until_idle,
    session_wake_attempts,
)
from cayu.tasks.queries import TaskQuery
from cayu.tasks.session_wakes import _cancel_session_wake_attempts
from cayu.tools import SleepUntilTool
from cayu.tools.base import Tool, ToolEffect, ToolSpec

_REPO = Path(__file__).resolve().parents[2]
_DIES = "CAYU_TEST_CHARGE_EXITS"


class _Charge(Tool):
    """Dies while it charges: in a subprocess it ends the process, here it raises."""

    spec = ToolSpec(
        name="charge",
        description="Charge the card once.",
        effect=ToolEffect.IDEMPOTENT,
        input_schema={"type": "object"},
        max_terminal_payload_bytes=4096,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests:sleep-until-crash:charge", behavior_version="1", implementation_version="1"
        ),
    )

    async def run(self, ctx, args):
        if os.environ.get(_DIES) == "1":
            os._exit(9)
        raise SystemExit("worker died mid-charge")


def _respond(request: ModelRequest):
    if not _tool_results(request):
        return [
            ModelStreamEvent.tool_call(
                id="call_sleep", name="sleep_until", arguments={"duration_seconds": 1}
            ),
            ModelStreamEvent.tool_call(id="call_charge", name="charge", arguments={}),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    return [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()]


def _app(sessions, tasks, clock=None) -> CayuApp:
    app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False, clock=clock)
    app.register_provider(_StableScriptedProvider(response_factory=_respond), default=True)
    app.register_agent(
        AgentSpec(name="sleeper", model="scripted-model"), tools=[SleepUntilTool(), _Charge()]
    )
    return app


def _stores(backend: str, location: str, clock=None):
    if backend == "memory":
        return InMemorySessionStore(), InMemoryTaskStore(clock=clock)
    if backend == "sqlite":
        return (
            SQLiteSessionStore(Path(location) / "sessions.db"),
            SQLiteTaskStore(Path(location) / "tasks.db"),
        )
    from cayu.storage import PostgresSessionStore
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.tasks_postgres import PostgresTaskStore

    return (
        PostgresSessionStore(location, schema_mode=SchemaMode.CREATE),
        PostgresTaskStore(location, schema_mode=SchemaMode.CREATE),
    )


def _location(backend: str, tmp_path, request) -> str:
    return request.getfixturevalue("postgres_dsn") if backend == "postgres" else str(tmp_path)


async def _wake_id(app: CayuApp, session_id: str) -> str:
    checkpoint = await app.session_store.load_checkpoint(session_id)
    pending = (checkpoint.get("pending_wake") or {}).get("continuation") or checkpoint[
        "pending_user_input"
    ]
    return pending["input_id"]


async def _actions(app: CayuApp, session_id: str):
    page = await app.session_store.query_pending_actions(PendingActionQuery(session_id=session_id))
    return [(action.kind, action.tool_call_id) for action in page.actions], page.issues


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_a_top_level_round_that_died_mid_tool_is_found_and_settled_as_manual_recovery(
    backend, tmp_path, request
) -> None:
    from tests.core._execution_profile_fixtures import rebind_test_invocation

    async def scenario() -> None:
        clock = _Clock(datetime.now(UTC))
        sessions, tasks = _stores(backend, _location(backend, tmp_path, request), clock)
        app = _app(sessions, tasks, clock)
        session_id = f"s_died_{backend}"
        try:
            async for _ in app.run(
                RunRequest(
                    agent_name="sleeper",
                    session_id=session_id,
                    messages=[Message.text("user", "wait, then charge")],
                )
            ):
                pass
            wake_id = await _wake_id(app, session_id)
            clock.advance(1.1)
            with pytest.raises(SystemExit):
                await deliver_session_wake(app, session_id=session_id, wake_id=wake_id)
            await rebind_test_invocation(sessions, session_id)
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id)
            )

            # Still waiting on its wake, never an inconsistent source.
            actions, issues = await _actions(app, session_id)
            assert issues == [] and actions == [(PendingActionKind.WAKE, "call_sleep")]
            # Its attempt is still live, so reconciliation adds nothing.
            assert await reconcile_session_wakes(app) == 0

            # The attempt ended with its worker; reconciliation replaces it.
            await _cancel_session_wake_attempts(app, wake_id)
            assert await reconcile_session_wakes(app) == 1
            attempts = await session_wake_attempts(app, wake_id)
            assert [attempt.status for attempt in attempts] == [
                TaskStatus.CANCELLED,
                TaskStatus.PENDING,
            ]
            if backend == "memory":
                clock.now = attempts[-1].available_at
                assert (
                    await run_task_worker(
                        app, tasks, _no_tasks, worker_id="w", max_tasks=1, poll_interval_s=0.01
                    )
                    == 1
                )
                [_, successor] = await session_wake_attempts(app, wake_id)
                assert successor.result == {"outcome": "manual_recovery_required"}
            else:
                assert (
                    await deliver_session_wake(app, session_id=session_id, wake_id=wake_id)
                    is SessionWakeOutcome.MANUAL_RECOVERY_REQUIRED
                )
            actions, issues = await _actions(app, session_id)
            assert issues == [] and actions == [(PendingActionKind.MANUAL_RECOVERY, "call_charge")]
            # Settled: nothing replaces the attempt again.
            assert await reconcile_session_wakes(app) == 0
        finally:
            await app.aclose()

    asyncio.run(scenario())


async def _no_tasks(app, task, worker_id):
    raise AssertionError(f"Unexpected task {task.type}.")


async def _run_job(app, task, worker_id):
    async for _ in app.run(
        RunRequest(
            agent_name="sleeper",
            session_id=f"s-{task.id}",
            task_id=task.id,
            task_worker_id=worker_id,
            task_lease_expires_at=task.lease_expires_at,
            messages=[Message.text("user", "wait, then charge")],
        )
    ):
        pass
    session = await app.session_store.load(f"s-{task.id}")
    return (
        TaskHandlerOutcome.SESSION_INTERRUPTED
        if session.status is SessionStatus.INTERRUPTED
        else None
    )


async def _never_continued(app, task, worker_id):
    raise AssertionError(f"A generic continuation claimed task {task.id}.")


async def _die_in_woken_round(backend: str, location: str, short_lease: int) -> None:
    """Subprocess entry: run the due wake and die inside the woken round's charge."""

    from cayu.tasks import session_wakes

    session_wakes._CONTINUATION_LEASE_SECONDS = short_lease
    sessions, tasks = _stores(backend, location)
    app = _app(sessions, tasks)
    await run_task_worker(app, tasks, _no_tasks, worker_id="doomed", lease_seconds=short_lease)


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_a_task_owned_round_that_died_mid_tool_hands_its_task_off_unparked(
    backend, tmp_path, request
) -> None:
    short_lease = 2

    async def scenario() -> None:
        location = _location(backend, tmp_path, request)
        sessions, tasks = _stores(backend, location)
        app = _app(sessions, tasks)
        task_id = f"job-died-{backend}"
        session_id = f"s-{task_id}"
        try:
            await tasks.create_task(TaskCreate(task_id=task_id, type="job"))
            await run_task_worker(
                app,
                tasks,
                _run_job,
                worker_id="first",
                query=TaskQuery(type="job"),
                max_tasks=1,
                poll_interval_s=0.05,
            )
            wake_id = await _wake_id(app, session_id)
            await asyncio.sleep(1.2)
            died = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import asyncio, sys; from tests.core.test_sleep_until_crash_recovery "
                    "import _die_in_woken_round; "
                    f"asyncio.run(_die_in_woken_round({backend!r}, sys.argv[1], {short_lease}))",
                    location,
                ],
                cwd=_REPO,
                env={**os.environ, _DIES: "1"},
                timeout=120,
            )
            assert died.returncode == 9
            # The owning task's continuation lease and the attempt's lease run out.
            await asyncio.sleep(short_lease + 1.5)
            await run_task_worker_until_idle(
                app,
                tasks,
                _run_job,
                worker_id="after-crash",
                recovered_interrupted_task_handler=_never_continued,
            )
            # Recovery found the round waiting on its wake with no live attempt.
            attempts = await session_wake_attempts(app, wake_id)
            assert len(attempts) == 2 and attempts[-1].status is TaskStatus.PENDING
            assert (
                await deliver_session_wake(app, session_id=session_id, wake_id=wake_id)
                is SessionWakeOutcome.MANUAL_RECOVERY_REQUIRED
            )
            actions, issues = await _actions(app, session_id)
            assert issues == [] and actions == [(PendingActionKind.MANUAL_RECOVERY, "call_charge")]
            owner = await tasks.load_task(task_id)
            assert owner.status is TaskStatus.RUNNING and owner.worker_id is None
            receipt = await tasks.load_interrupted_task_handoff_receipt(
                owner.id, owner.interrupted_handoff_id
            )
            assert receipt.request.parked is None
        finally:
            await app.aclose()

    asyncio.run(asyncio.wait_for(scenario(), 120))
