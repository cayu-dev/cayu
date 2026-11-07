"""`cayu task show` says how Cayu settled a task whose worker was lost.

No settlement here is written by hand. A worker dies holding the task, its lease expires,
and Cayu's own recovery settles it: a worker declared the only one, the next worker that
recovers a session-attached task or a lost session wake, the dying worker's own drain, or
the application for a retry attempt. The evidence reads the same from a local store,
through a Cayu server, and from a Cloud Agent.
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from tests.cli.test_session_cli_cloud import _FakeCloud
from tests.core.test_cancel_orphaned_session_task import (
    _app,
    _cancel_fenced,
    _charge_then_answer,
    _died_mid_tool,
    _next_worker,
    _sleep_then_charge,
    _sleep_then_die_in_the_woken_round,
    _wake_id_of,
)
from tests.core.test_external_waits_in_tasks_edges import _die, _dies, _expire, _stores
from tests.core.test_sleep_until import _Clock

from cayu import CayuApp
from cayu.approvals.actors import ResolutionActor, ResolutionActorSource
from cayu.cli import main
from cayu.cli._task_readers import task_projection
from cayu.cli.task import cancellation_summary
from cayu.server import ServerConfig, create_server
from cayu.server.auth import BasicAuth
from cayu.storage import SQLiteTaskStore
from cayu.tasks import TaskCreate, TaskStatus, reconcile_abandoned_task_cancellation, session_wakes
from cayu.tasks import worker as task_worker
from cayu.tasks.abandoned import SINGLE_WORKER_VALIDATOR_ID
from cayu.tasks.cancellation import (
    TaskRetryCancellationReconciliationEvidence,
    TaskRetryCancellationReconciliationOutcome,
    TaskRetryCancellationReconciliationRequest,
)
from cayu.tasks.records import Task, TaskRetryPolicy
from cayu.tasks.session_wakes import WAKE_ATTEMPT_OF_ENDED_TASK_VALIDATOR_ID

_DEAD_WORKER = "worker-1"  # `_die`'s worker
_NEXT_WORKER = "after-crash"  # `_next_worker`'s worker
_OPERATOR = ("operator", "s3cret-password")  # `_FakeCloud`'s operator login
_APPLICATION = "bug-hunter-agent"


@dataclass(frozen=True)
class _Settled:
    """A task Cayu settled after its worker was lost, and what its evidence must say."""

    task: Task
    lease_expires_at: datetime
    settled_by: str
    evidence_kind: str
    evidence_id: str
    validator_id: str
    explanation: str | None = None
    session: tuple[str, int] | None = None
    restart_limit: int | None = None


async def _lost_with_its_worker(app: CayuApp, clock: Any, backend: str) -> _Settled:
    """The next worker, declared the only one, cancels a task fenced under a dead one."""

    await app.task_store.create_task(TaskCreate(task_id="scan", type="scan"))
    await _die(app, _dies)
    held = await app.task_store.load_task("scan")
    await _expire(backend, clock)
    await _next_worker(app, reconcile_abandoned_after_s=0)
    return _Settled(
        task=await app.task_store.load_task("scan"),
        lease_expires_at=held.lease_expires_at,
        settled_by=_NEXT_WORKER,
        evidence_kind="lease_expired",
        evidence_id=f"lease-expired-at:{held.lease_expires_at.isoformat()}",
        validator_id=SINGLE_WORKER_VALIDATOR_ID,
    )


async def _out_of_restarts(app: CayuApp, clock: Any, backend: str) -> _Settled:
    """A restartable task that may not restart again is cancelled instead."""

    await app.task_store.create_task(
        TaskCreate(task_id="scan", type="scan", restartable=True, max_restarts=0)
    )
    await _die(app, _dies)
    held = await app.task_store.load_task("scan")
    await _expire(backend, clock)
    await _next_worker(app, reconcile_abandoned_after_s=0, restart_abandoned=True)
    return _Settled(
        task=await app.task_store.load_task("scan"),
        lease_expires_at=held.lease_expires_at,
        settled_by=_NEXT_WORKER,
        evidence_kind="restart_limit_reached",
        evidence_id=f"restart-limit-reached:0:lease-expired-at:{held.lease_expires_at.isoformat()}",
        validator_id=SINGLE_WORKER_VALIDATOR_ID,
        explanation="It wasn't restarted: it had reached its limit of 0 restarts after a lost worker.",
        restart_limit=0,
    )


async def _attached_to_a_session(app: CayuApp, clock: Any, backend: str) -> _Settled:
    """A session-attached task cancelled after its worker died mid-tool."""

    task_id = await _died_mid_tool(app)
    held = await _cancel_fenced(app, task_id)
    await _expire(backend, clock)
    await _next_worker(app)
    session = await app.session_store.load(held.session_id)
    return _Settled(
        task=await app.task_store.load_task(task_id),
        lease_expires_at=held.lease_expires_at,
        settled_by=_NEXT_WORKER,
        evidence_kind="session_interrupted",
        evidence_id=f"session-interrupted:{session.instance_id}:{session.run_epoch}",
        validator_id="cayu.task-worker.expired-attached-owner",
        explanation=(
            f"Its session {session.id} was interrupted (run epoch {session.run_epoch}), "
            "so the lost run couldn't go on."
        ),
        session=(session.instance_id, session.run_epoch),
    )


async def _a_lost_session_wake(app: CayuApp, clock: Any, backend: str) -> _Settled:
    """The wake of a sleeping task dies in its woken round; the task is then cancelled."""

    task_id = await _sleep_then_die_in_the_woken_round(
        app, app.task_store, clock, attempt_lease=600
    )
    held = await app.task_store.load_task(task_id)
    [attempt] = await session_wakes.session_wake_attempts(app, _wake_id_of(held))
    await _cancel_fenced(app, task_id)
    # The wake's claim of the task expires first and the task is settled; then the wake.
    clock.advance(2)
    await _next_worker(app)
    clock.advance(601)
    await _next_worker(app)
    session = await app.session_store.load(held.session_id)
    return _Settled(
        task=await app.task_store.load_task(attempt.id),
        lease_expires_at=attempt.lease_expires_at,
        settled_by=_NEXT_WORKER,
        evidence_kind="task_ended",
        evidence_id=f"task-ended:{session.instance_id}:{session.run_epoch}",
        validator_id=WAKE_ATTEMPT_OF_ENDED_TASK_VALIDATOR_ID,
        explanation="The task this session wake would have continued had already ended.",
        session=(session.instance_id, session.run_epoch),
    )


async def _a_retry_attempt(app: CayuApp, clock: Any, backend: str) -> _Settled:
    """The application settles a retry attempt fenced under a dead worker, as it must."""

    tasks = app.task_store
    await tasks.create_task(
        TaskCreate(task_id="review", type="review", retry_policy=TaskRetryPolicy(max_attempts=3))
    )
    await _die(app, _dies)
    await _expire(backend, clock)
    await _next_worker(app)
    fenced = await tasks.load_task("review")
    assert fenced.status_reason == "retry_cancellation_requested"
    payload = fenced.status_payload
    requested_at = datetime.fromisoformat(payload["event"]["occurred_at"])
    validated_at = max(requested_at, fenced.lease_expires_at)
    await tasks.reconcile_task_retry_cancellation(
        TaskRetryCancellationReconciliationRequest(
            task_id=fenced.id,
            series_id=fenced.retry_series.series_id,
            attempt=fenced.retry_series.attempt,
            causal_budget_id=fenced.retry_series.causal_budget_id,
            original_worker_id=fenced.worker_id,
            original_lease_expires_at=fenced.lease_expires_at,
            cancellation_requested_at=requested_at,
            cancellation_idempotency_key=payload["settlement_idempotency_key"],
            reconciliation_idempotency_key="worker-registry-1",
            reconciliation_requested_at=validated_at,
            reconciled_by=ResolutionActor(
                subject="ops:worker-registry", source=ResolutionActorSource.SYSTEM
            ),
            evidence=TaskRetryCancellationReconciliationEvidence(
                outcome=TaskRetryCancellationReconciliationOutcome.QUIESCENT,
                validator_id="ops.worker-registry",
                validator_version="1",
                evidence_id=f"container-exited:{fenced.worker_id}",
                evidence_sha256="a" * 64,
                validated_at=validated_at,
            ),
        )
    )
    return _Settled(
        task=await tasks.load_task("review"),
        lease_expires_at=fenced.lease_expires_at,
        settled_by="ops:worker-registry",
        evidence_kind="other",
        evidence_id=f"container-exited:{_DEAD_WORKER}",
        validator_id="ops.worker-registry",
    )


async def _drained_after_its_lease(app: CayuApp, clock: Any, backend: str) -> _Settled:
    """A worker whose lease expired while it drained a cancellation settles it itself."""

    tasks = app.task_store
    await tasks.create_task(TaskCreate(task_id="drain", type="scan"))
    claimed = await tasks.claim_task(_DEAD_WORKER, lease_seconds=5)
    assert claimed is not None
    requested = await tasks.cancel_task("drain", {"code": "operator", "message": "not needed"})
    await _expire(backend, clock)
    await task_worker._settle_ordinary_task_cancellation_after_quiescence(
        tasks, "drain", _DEAD_WORKER, claim_generation=claimed.claim_generation
    )
    settled = await tasks.load_task("drain")
    evidence_id = settled.status_payload["cancellation_reconciliation"]["evidence"]["evidence_id"]
    assert evidence_id.startswith("task-worker-quiescence:")
    return _Settled(
        task=settled,
        lease_expires_at=requested.lease_expires_at,
        settled_by="cayu:task-worker-quiescence",
        evidence_kind="handler_stopped",
        evidence_id=evidence_id,
        validator_id="cayu.task-worker",
    )


_SCENARIOS: dict[str, tuple[Any, Callable[[CayuApp, Any, str], Awaitable[_Settled]]]] = {
    "lease_expired": (_charge_then_answer, _lost_with_its_worker),
    "restart_limit_reached": (_charge_then_answer, _out_of_restarts),
    "session_interrupted": (_charge_then_answer, _attached_to_a_session),
    "task_ended": (_sleep_then_charge, _a_lost_session_wake),
    "retry_attempt": (_charge_then_answer, _a_retry_attempt),
    "handler_stopped": (_charge_then_answer, _drained_after_its_lease),
}


def _settle(
    scenario: str, backend: str, tmp_path: Path, request: Any, monkeypatch: Any
) -> _Settled:
    # A wake's claim of its task expires a second after its worker dies.
    monkeypatch.setattr(session_wakes, "_CONTINUATION_LEASE_SECONDS", 1)
    respond, build = _SCENARIOS[scenario]

    async def run() -> _Settled:
        clock = _Clock(datetime.now(UTC))
        sessions, tasks = _stores(backend, clock, tmp_path, request)
        app, _, _ = _app(sessions, tasks, clock, respond)
        try:
            settled = await build(app, clock, backend)
        finally:
            await app.aclose()
        assert settled.task.status is TaskStatus.CANCELLED
        return settled

    return asyncio.run(run())


def _assert_worker_loss(loss: dict[str, Any], settled: _Settled) -> None:
    lease_expired_at, settled_at = loss["lease_expired_at"], loss["settled_at"]
    assert datetime.fromisoformat(lease_expired_at) == settled.lease_expires_at
    assert datetime.fromisoformat(settled_at) == settled.task.completed_at
    sentences = [
        f"Worker {_DEAD_WORKER}'s lease expired at {lease_expired_at} before it settled "
        "the cancellation, and it saw its handler stop."
        if settled.evidence_kind == "handler_stopped"
        else f"Worker {_DEAD_WORKER} stopped renewing its lease, which expired at "
        f"{lease_expired_at}.",
        settled.explanation,
        f"Cayu cancelled the task at {settled_at} (settled by {settled.settled_by}).",
    ]
    assert loss == {
        "summary": " ".join(sentence for sentence in sentences if sentence),
        "worker_id": _DEAD_WORKER,
        "lease_expired_at": lease_expired_at,
        "settled_at": settled_at,
        "settled_by": settled.settled_by,
        "settled_by_source": "system",
        "evidence_kind": settled.evidence_kind,
        "evidence_id": settled.evidence_id,
        "validator_id": settled.validator_id,
        "validator_version": "1",
        "session_instance_id": None if settled.session is None else settled.session[0],
        "session_run_epoch": None if settled.session is None else settled.session[1],
        "restart_limit": settled.restart_limit,
    }


@pytest.mark.parametrize("scenario", sorted(_SCENARIOS))
def test_an_in_memory_settlement_names_the_lost_worker_and_its_evidence(
    scenario: str, tmp_path: Path, request: Any, monkeypatch: Any
) -> None:
    settled = _settle(scenario, "memory", tmp_path, request, monkeypatch)

    cancellation = cancellation_summary(task_projection(settled.task, detail=True))

    assert cancellation is not None and cancellation["state"] == "cancelled"
    _assert_worker_loss(cancellation["worker_loss"], settled)


@contextmanager
def _served(database: Path, monkeypatch: Any) -> Iterator[None]:
    """``database`` served by a Cayu server, reached as a server URL and as a Cloud Agent."""

    monkeypatch.setattr("cayu.cli.cloud.cloud_client", lambda **_: _FakeCloud())
    monkeypatch.setenv(
        "CAYU_API_AUTHORIZATION",
        "Basic " + base64.b64encode(":".join(_OPERATOR).encode()).decode(),
    )
    server = create_server(
        CayuApp(task_store=SQLiteTaskStore(database), enable_logging=False),
        config=ServerConfig.protected(BasicAuth(username=_OPERATOR[0], password=_OPERATOR[1])),
    )
    with TestClient(server, base_url="http://127.0.0.1") as client:
        # Each command closes the client it opened; keep this shared one for the next.
        monkeypatch.setattr(client, "close", lambda: None)
        monkeypatch.setattr("cayu.cli._cloud_agent_api.httpx.Client", lambda **_: client)
        yield


def _task(capsys: Any, *argv: str) -> str:
    assert main(["task", *argv]) == 0, capsys.readouterr()
    return capsys.readouterr().out


@pytest.mark.parametrize("scenario", sorted(_SCENARIOS))
def test_show_tells_the_same_settlement_locally_through_a_server_and_from_a_cloud_agent(
    scenario: str, tmp_path: Path, request: Any, monkeypatch: Any, capsys: Any
) -> None:
    settled = _settle(scenario, "sqlite", tmp_path, request, monkeypatch)
    database = str(tmp_path / "tasks.sqlite")
    task_id = settled.task.id
    targets = {
        "local": ("--sqlite", database),
        "server": ("--server-url", "http://127.0.0.1"),
        "cloud": ("--application", _APPLICATION),
    }

    shown: dict[str, dict[str, Any]] = {}
    tables: dict[str, str] = {}
    listed: dict[str, list[dict[str, Any]]] = {}
    with _served(Path(database), monkeypatch):
        for name, target in targets.items():
            shown[name] = json.loads(_task(capsys, "show", task_id, *target, "--json"))
            tables[name] = _task(capsys, "show", task_id, *target)
            listed[name] = json.loads(_task(capsys, "list", *target, "--json"))["tasks"]

    cancellation = shown["local"]["cancellation"]
    loss = cancellation["worker_loss"]
    _assert_worker_loss(loss, settled)
    for name in targets:
        assert shown[name]["cancellation"] == cancellation, name
        # List rows leave the error out, but carry the whole settlement.
        [row] = [row for row in listed[name] if row["id"] == task_id]
        assert row["cancellation"]["worker_loss"] == loss, name
        assert tables[name] == tables["local"], name
    # The table says it in words, then lists each detail as a row of its own.
    section = tables["local"].split("\nWorker loss\n", 1)[1].split("\n\n", 1)[0]
    summary, header, *rows = section.splitlines()
    assert summary == loss["summary"]
    assert header.split() == ["field", "value"]
    assert {field: value.strip() for field, value in (row.split(maxsplit=1) for row in rows)} == {
        key: str(value) for key, value in loss.items() if value is not None and key != "summary"
    }
    assert "worker_loss" not in tables["local"]


@pytest.mark.parametrize(
    ("evidence_id", "kind"),
    [
        ("restart-limit-reached:²:lease-expired-at:then", "restart_limit_reached"),
        ("session-interrupted:instance:²", "session_interrupted"),
    ],
)
def test_application_evidence_that_only_looks_like_cayus_is_read_without_its_numbers(
    evidence_id: str, kind: str, tmp_path: Path, request: Any, capsys: Any
) -> None:
    """Evidence ids are the application's to choose, so a number in one may not parse."""

    async def run() -> None:
        clock = _Clock(datetime.now(UTC))
        sessions, tasks = _stores("sqlite", clock, tmp_path, request)
        app, _, _ = _app(sessions, tasks, clock, _charge_then_answer)
        try:
            await tasks.create_task(TaskCreate(task_id="scan", type="scan"))
            await _die(app, _dies)
            await _expire("sqlite", clock)
            await _next_worker(app)
            await reconcile_abandoned_task_cancellation(
                tasks,
                "scan",
                validator_id="ops.worker-registry",
                validator_version="1",
                evidence_id=evidence_id,
            )
        finally:
            await app.aclose()

    asyncio.run(run())
    database = str(tmp_path / "tasks.sqlite")

    listed = json.loads(_task(capsys, "list", "--sqlite", database, "--json"))["tasks"]
    shown = json.loads(_task(capsys, "show", "scan", "--sqlite", database, "--json"))

    loss = shown["cancellation"]["worker_loss"]
    assert [row["cancellation"]["worker_loss"] for row in listed] == [loss]
    assert (loss["evidence_kind"], loss["evidence_id"]) == (kind, evidence_id)
    assert loss["restart_limit"] is None and loss["session_run_epoch"] is None


def _fence_four_tasks(database: Path) -> None:
    """Two tasks fenced by worker loss and two by a request; one of each settled."""

    from tests.core.test_task_restart_after_worker_loss import SETTLE
    from tests.core.test_task_restart_after_worker_loss import _Clock as _StoreClock

    from cayu.tasks import reconcile_abandoned_tasks
    from cayu.tasks.queries import TaskQuery

    async def fence(store: SQLiteTaskStore, clock: _StoreClock, task_id: str, asked: bool) -> None:
        await store.create_task(TaskCreate(task_id=task_id, type=task_id))
        claimed = await store.claim_task("worker-a", TaskQuery(type=task_id), lease_seconds=30)
        assert claimed is not None and claimed.lease_expires_at is not None
        await store.mark_claimed_task_execution_started(
            task_id, "worker-a", claimed.lease_expires_at
        )
        if asked:
            await store.cancel_task(task_id)
        else:
            clock.value = claimed.lease_expires_at + timedelta(seconds=1)
            await store.reclaim_expired(query=TaskQuery(type=task_id))

    async def build() -> None:
        clock = _StoreClock()
        store = SQLiteTaskStore(database, ownership_clock=clock)
        try:
            await fence(store, clock, "lost-settled", asked=False)
            await fence(store, clock, "asked-settled", asked=True)
            clock.value += timedelta(seconds=31)
            await store.reclaim_expired(query=TaskQuery(type="asked-settled"))
            settled = await reconcile_abandoned_tasks(store, **{**SETTLE, "settlement": "cancel"})
            assert {task.id for task in settled} == {"lost-settled", "asked-settled"}
            await fence(store, clock, "lost-waiting", asked=False)
            await fence(store, clock, "asked-waiting", asked=True)
        finally:
            await store.close()

    asyncio.run(build())


_FENCED_BY = {
    "lost-waiting": "worker_loss",
    "asked-waiting": "request",
    "lost-settled": "worker_loss",
    "asked-settled": "request",
}


def _table_rows(capsys: Any, *target: str) -> dict[str, str]:
    return {
        line.split()[0]: line
        for line in _task(capsys, "list", *target).splitlines()[1:]
        if line.strip()
    }


def test_list_and_show_say_whether_worker_loss_or_a_request_fenced_a_cancelled_task(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    """A list row says what fenced a task both while it waits to stop and once it's settled.

    While it waits, the fence's error is in its status payload. Once it's settled the error
    is only the task's own, which list rows leave out except for its code.
    """

    database = tmp_path / "tasks.sqlite"
    _fence_four_tasks(database)
    targets = {
        "local": ("--sqlite", str(database)),
        "server": ("--server-url", "http://127.0.0.1"),
        "cloud": ("--application", _APPLICATION),
    }
    with _served(database, monkeypatch):
        for name, target in targets.items():
            rows = {
                row["id"]: row
                for row in json.loads(_task(capsys, "list", *target, "--json"))["tasks"]
            }
            shown = {
                task_id: json.loads(_task(capsys, "show", task_id, *target, "--json"))
                for task_id in _FENCED_BY
            }
            listed = {task_id: row["cancellation"]["fenced_by"] for task_id, row in rows.items()}
            assert listed == _FENCED_BY, name
            assert {
                task_id: shown[task_id]["cancellation"]["fenced_by"] for task_id in shown
            } == _FENCED_BY, name
            for task_id in ("lost-settled", "asked-settled"):
                row, code = rows[task_id], shown[task_id]["task"]["error"]["code"]
                assert "error" not in row and row["error_code"] == code, name
            for task_id in ("lost-waiting", "asked-waiting"):
                assert rows[task_id]["error_code"] is None, name
            table = _table_rows(capsys, *target)
            assert "(worker lost)" in table["lost-waiting"], name
            assert "(requested)" in table["asked-waiting"], name
            assert "(quiescent, worker lost)" in table["lost-settled"], name
            assert "(quiescent, requested)" in table["asked-settled"], name


def test_a_server_without_error_codes_in_its_rows_still_lists_settled_cancellations(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    """A server older than ``error_code`` lists cleanly; settled rows don't guess the cause."""

    from cayu.cli._task_readers import ApiTaskReader

    database = tmp_path / "tasks.sqlite"
    _fence_four_tasks(database)
    list_tasks = ApiTaskReader.list_tasks

    async def older_server_rows(self: ApiTaskReader, request: Any) -> list[dict[str, Any]]:
        rows = await list_tasks(self, request)
        for row in rows:
            del row["error_code"]
        return rows

    monkeypatch.setattr(ApiTaskReader, "list_tasks", older_server_rows)
    target = ("--server-url", "http://127.0.0.1")
    with _served(database, monkeypatch):
        rows = json.loads(_task(capsys, "list", *target, "--json"))["tasks"]
        table = _table_rows(capsys, *target)

    assert {row["id"]: row["cancellation"]["fenced_by"] for row in rows} == {
        **_FENCED_BY,
        "lost-settled": None,
        "asked-settled": None,
    }
    assert all("error_code" not in row for row in rows)
    for task_id in ("lost-settled", "asked-settled"):
        assert "(quiescent)" in table[task_id]
