"""Settling fenced tasks from Cayu Cloud's worker exit record."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import cloud_worker_exits as exits_module
from cayu.tasks.cloud_worker_exits import (
    CLOUD_WORKER_EXIT_VALIDATOR_ID,
    WorkerExit,
    WorkerExitRecord,
    WorkerExitsUnavailable,
    WorkerExitsUnsupported,
    cloud_worker_id,
    exited_run_references,
    read_cloud_worker_exits,
    reconcile_tasks_of_exited_workers,
    worker_id_for_run,
)
from cayu.tasks.creation import TaskCreate
from cayu.tasks.queries import TaskQuery
from cayu.tasks.records import Task, TaskStatus
from cayu.tasks.store import TaskStore

LEASE_SECONDS = 30
ARN_OLD = "arn:aws:ecs:region:account:task/agents/old"
ARN_LIVE = "arn:aws:ecs:region:account:task/agents/live"
MARKER = "0123456789abcdef0123456789abcdef"
EXITS_URL = "https://cloud.example.test/v1/agent-worker/exits"
STOPPED_AT = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _fresh_process_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(exits_module, "_process_reference", None)
    monkeypatch.setattr(exits_module, "_process_reference_read", False)


class _Clock:
    def __init__(self) -> None:
        self.value = datetime.now(UTC).replace(microsecond=0) - timedelta(hours=1)

    def __call__(self) -> datetime:
        return self.value


async def _started_claim(store: TaskStore, task_id: str, worker_id: str, **create) -> Task:
    await store.create_task(TaskCreate(task_id=task_id, type="scan", **create))
    claimed = await store.claim_task(worker_id, TaskQuery(type="scan"), lease_seconds=LEASE_SECONDS)
    assert claimed is not None and claimed.id == task_id
    assert claimed.lease_expires_at is not None
    return await store.mark_claimed_task_execution_started(
        task_id, worker_id, claimed.lease_expires_at
    )


def _record(*entries: WorkerExit) -> WorkerExitRecord:
    return WorkerExitRecord(exits=entries)


def test_settles_only_tasks_of_runs_cloud_recorded_stopped(tmp_path: Path) -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = SQLiteTaskStore(tmp_path / "tasks.db", ownership_clock=clock)
        old = worker_id_for_run("jobs", ARN_OLD)
        k8s = worker_id_for_run("jobs", f"cayu-worker-run:{MARKER}")
        started = [
            await _started_claim(store, "ecs-old", old),
            await _started_claim(store, "k8s-old", k8s),
            await _started_claim(store, "still-running", worker_id_for_run("jobs", ARN_LIVE)),
            await _started_claim(store, "plain-id", "jobs"),
            await _started_claim(store, "gone-run", worker_id_for_run("jobs", "arn:aws:x")),
            await _started_claim(store, "restartable", old, restartable=True),
        ]
        clock.value = max(t.lease_expires_at for t in started) + timedelta(seconds=1)  # type: ignore[type-var]
        assert await store.reclaim_expired(query=TaskQuery(type="scan")) == []

        record = _record(
            WorkerExit(ARN_OLD, "stopped", stopped_at=STOPPED_AT, reason="OOMKilled"),
            WorkerExit("containerd://abc", "stopped", stopped_at=STOPPED_AT, run_marker=MARKER),
            WorkerExit(ARN_LIVE, "stopping"),
            WorkerExit("arn:aws:x", "gone"),
        )
        settled = await reconcile_tasks_of_exited_workers(store, record, settlement="restart")
        by_id = {task.id: task for task in settled}
        assert sorted(by_id) == ["ecs-old", "k8s-old", "restartable"]
        assert by_id["ecs-old"].status is TaskStatus.CANCELLED
        assert by_id["restartable"].status is TaskStatus.PENDING
        payload = by_id["ecs-old"].status_payload
        assert payload is not None
        evidence = payload["cancellation_reconciliation"]["evidence"]
        assert evidence["validator_id"] == CLOUD_WORKER_EXIT_VALIDATOR_ID
        assert evidence["evidence_id"].startswith(
            f"cloud-worker-stopped-at:{STOPPED_AT.isoformat()}:lease-expired-at:"
        )
        fenced = TaskQuery(fenced_after_worker_loss=True)
        assert sorted(t.id for t in await store.list_tasks(fenced)) == [
            "gone-run",
            "plain-id",
            "still-running",
        ]
        # "gone" proves nothing unless the caller accepts it.
        [gone] = await reconcile_tasks_of_exited_workers(store, record, accept_gone=True)
        assert gone.id == "gone-run"
        await store.close()

    asyncio.run(scenario())


def test_exited_references_and_worker_ids() -> None:
    record = _record(
        WorkerExit(ARN_OLD, "stopped"),
        WorkerExit("containerd://abc", "stopped", run_marker=MARKER),
        WorkerExit(ARN_LIVE, "stopping"),
        WorkerExit("arn:aws:x", "gone"),
    )
    assert exited_run_references(record) == {
        ARN_OLD,
        "containerd://abc",
        f"cayu-worker-run:{MARKER}",
    }
    assert "arn:aws:x" in exited_run_references(record, accept_gone=True)
    assert worker_id_for_run("jobs", None) == "jobs"
    assert worker_id_for_run("jobs", ARN_OLD) != worker_id_for_run("jobs", ARN_LIVE)
    assert ARN_OLD not in worker_id_for_run("jobs", ARN_OLD)
    with pytest.raises(ValueError):
        worker_id_for_run("a@cloud-run:b", ARN_OLD)


def test_cloud_worker_id_reads_the_ecs_task_arn_once() -> None:
    calls: list[str] = []

    def metadata(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, json={"TaskARN": ARN_OLD})

    environ = {
        "ECS_CONTAINER_METADATA_URI_V4": "http://169.254.170.2/v4/abc",
        "CAYU_WORKER_EXITS_URL": EXITS_URL,
    }
    transport = httpx.MockTransport(metadata)
    first = asyncio.run(cloud_worker_id("jobs", environ, transport=transport))
    second = asyncio.run(cloud_worker_id("timers", environ, transport=transport))
    assert first == worker_id_for_run("jobs", ARN_OLD)
    assert second == worker_id_for_run("timers", ARN_OLD)
    assert calls == ["http://169.254.170.2/v4/abc/task"]


def test_cloud_worker_id_writes_a_kubernetes_run_marker(tmp_path: Path) -> None:
    log = tmp_path / "termination-log"
    log.write_text("", encoding="utf-8")
    environ = {"KUBERNETES_SERVICE_HOST": "10.0.0.1", "CAYU_WORKER_EXITS_URL": EXITS_URL}
    worker_id = asyncio.run(cloud_worker_id("jobs", environ, termination_log=log))
    written = log.read_text(encoding="utf-8")
    assert written.startswith("cayu-worker-run:") and len(written) == len("cayu-worker-run:") + 32
    assert worker_id == worker_id_for_run("jobs", written)

    # Another process of the same container run reuses the marker instead of replacing it.
    exits_module._process_reference_read = False
    again = asyncio.run(cloud_worker_id("jobs", environ, termination_log=log))
    assert again == worker_id
    assert log.read_text(encoding="utf-8") == written


def test_kubernetes_outside_cayu_cloud_leaves_the_termination_log_alone(tmp_path: Path) -> None:
    log = tmp_path / "termination-log"
    log.write_text("", encoding="utf-8")
    environ = {"KUBERNETES_SERVICE_HOST": "10.0.0.1"}
    assert asyncio.run(cloud_worker_id("jobs", environ, termination_log=log)) == "jobs"
    assert log.read_text(encoding="utf-8") == ""


def test_cloud_worker_id_outside_cloud_is_the_name(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    assert asyncio.run(cloud_worker_id("jobs", {}, termination_log=missing)) == "jobs"


def test_read_cloud_worker_exits() -> None:
    seen: list[httpx.Request] = []
    body = {
        "application_id": "invoices",
        "looked_at": "2026-01-02T03:05:00Z",
        "exits": [
            {
                "instance": ARN_OLD,
                "state": "stopped",
                "stopped_at": "2026-01-02T03:04:05+00:00",
                "reason": "OOMKilled",
                "exit_code": 137,
            },
            {"instance": "containerd://abc", "state": "stopped", "run_marker": MARKER},
            {"instance": "containerd://def", "state": "stopped", "run_marker": "not-hex"},
        ],
    }

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=body)

    environ = {"CAYU_WORKER_EXITS_URL": EXITS_URL, "CAYU_ATTENTION_HANDOFF_TOKEN": "agent-token"}
    record = asyncio.run(read_cloud_worker_exits(environ, transport=httpx.MockTransport(handle)))
    assert seen[0].headers["authorization"] == "Bearer agent-token"
    assert record.looked_at == datetime(2026, 1, 2, 3, 5, tzinfo=UTC)
    assert record.exits[0] == WorkerExit(
        ARN_OLD, "stopped", stopped_at=STOPPED_AT, reason="OOMKilled", exit_code=137
    )
    assert record.exits[1].run_marker == MARKER
    assert record.exits[2].run_marker is None

    with pytest.raises(WorkerExitsUnsupported):
        asyncio.run(read_cloud_worker_exits({}))
    not_served = httpx.MockTransport(lambda request: httpx.Response(404))
    with pytest.raises(WorkerExitsUnsupported):
        asyncio.run(read_cloud_worker_exits(environ, transport=not_served))
    refused = httpx.MockTransport(lambda request: httpx.Response(401, json={}))
    with pytest.raises(WorkerExitsUnavailable) as raised:
        asyncio.run(read_cloud_worker_exits(environ, transport=refused))
    assert "agent-token" not in str(raised.value)
