"""Policy and host-generation characterization for the installed example.

Actual application execution/restart coverage is in test_coding_host_process_recovery.
These tests do not dispatch Docker or establish live Docker qualification.
"""

import asyncio
import threading
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from tests.cli.test_coding_host_settlement_example import _events

from cayu import Event, EventType

_WORKSPACE_TOOLS = (
    "read_file",
    "write_file",
    "edit_file",
    "apply_patch",
    "delete_file",
    "list_files",
    "search_text",
    "git_changes",
)


def _fixture_ordinals(events):
    # Controlled serial fixture linkage; production derives this from private
    # Runtime dispatch tuples, never from names or these test events.
    current = -1
    values = []
    for event in events:
        if event.type is EventType.TOOL_CALL_STARTED:
            current += 1
        values.append(current if event.tool_name else None)
    return tuple(values)


@pytest.mark.parametrize(
    "fault", [None, "missing", "boolean", "duplicate-start", "duplicate-terminal", "wrong-terminal"]
)
def test_batched_history_requires_exact_inspected_correlation(fault):
    from cayu.guides import coding_host_evidence as module

    base = _events()
    read_start, read_end = _file_events("read_file")
    edit_start, edit_end = _file_events("edit_file")
    events = (*base[:2], read_start, edit_start, read_end, edit_end, *base[2:])
    keys = [None, None, 0, 1, 0, 1, 2, 2]
    if fault == "missing":
        keys[3] = None
    elif fault == "boolean":
        keys[3] = True
    elif fault == "duplicate-start":
        keys[3] = 0
    elif fault == "duplicate-terminal":
        events = (*events[:5], read_end, *events[6:])
        keys[5] = 0
    elif fault == "wrong-terminal":
        keys[4] = 1
    if fault is None:
        module._require_serial_check_quiescence(events, tool_call_ordinals=tuple(keys))
    else:
        with pytest.raises(module.MaintenanceReconciliationUnavailable):
            module._require_serial_check_quiescence(events, tool_call_ordinals=tuple(keys))


def _file_events(name):
    common = {"path": "range_ops.py"}
    values = {
        "read_file": {
            **common,
            "source": "workspace",
            "encoding": "utf-8",
            "bytes": 10,
            "total_bytes": 10,
            "offset": 0,
            "truncated": False,
        },
        "write_file": {
            **common,
            "revision": "revision",
            "sha256": "a" * 64,
            "bytes": 10,
            "encoding": "utf-8",
            "mode": "overwrite",
        },
        "edit_file": {
            **common,
            "before_revision": "before",
            "after_revision": "after",
            "before_sha256": "a" * 64,
            "after_sha256": "b" * 64,
            "before_bytes": 10,
            "after_bytes": 11,
            "edit_count": 1,
            "replacement_count": 1,
        },
        "apply_patch": {
            "version": 2,
            "patch_id": "patch",
            "behavior_profile_id": "profile",
            "outcome": "applied",
            "failure_category": None,
            "requires_fresh_read": False,
            "operation_count": 1,
        },
        "delete_file": {
            **common,
            "deleted_bytes": 10,
            "deleted_revision": "revision",
            "deleted_sha256": "a" * 64,
        },
        "list_files": {
            "pattern": "**/*",
            "files": [],
            "offset": 0,
            "total_files": 0,
            "truncated": False,
        },
        "search_text": {
            **common,
            "pattern": "range",
            "mode": "files",
            "matches": [],
            "returned": 0,
            "offset": 0,
            "limit": 10,
            "stdout_bytes": 0,
            "truncated": False,
        },
        "git_changes": {
            "mode": "diff",
            "scope": "all",
            "changes": [],
            "returned": 0,
            "offset": 0,
            "limit": 10,
            "truncated": False,
        },
    }
    return (
        Event(type=EventType.TOOL_CALL_STARTED, session_id="fixture", tool_name=name),
        Event(
            type=EventType.TOOL_CALL_COMPLETED,
            session_id="fixture",
            tool_name=name,
            payload={"result": {"is_error": False, "structured": values[name]}},
        ),
    )


@pytest.mark.parametrize("name", _WORKSPACE_TOOLS)
@pytest.mark.parametrize("fault", [None, "failed", "missing", "boolean", "unterminated"])
def test_completed_file_history_requires_positive_evidence(name, fault):
    from cayu.guides import coding_host_evidence as module

    started, terminal = _file_events(name)
    structured = terminal.payload["result"]["structured"]
    if fault == "failed":
        terminal.payload["result"]["is_error"] = True
    elif fault == "missing":
        structured.clear()
    elif fault == "boolean":
        key = {
            "read_file": "bytes",
            "write_file": "bytes",
            "edit_file": "edit_count",
            "apply_patch": "operation_count",
            "delete_file": "deleted_bytes",
            "list_files": "offset",
            "search_text": "returned",
            "git_changes": "returned",
        }[name]
        structured[key] = True
    events = _events()
    middle = (started,) if fault == "unterminated" else (started, terminal)
    if fault is None:
        module._require_serial_check_quiescence((*events[:2], *middle, *events[2:]))
    else:
        with pytest.raises(module.MaintenanceReconciliationUnavailable):
            module._require_serial_check_quiescence((*events[:2], *middle, *events[2:]))


@pytest.mark.parametrize("fault", [None, "missing-total", "boolean-total", "not-truncated"])
def test_bounded_file_listing_requires_explicit_settled_shape(fault):
    from cayu.guides import coding_host_evidence as module

    started, terminal = _file_events("list_files")
    value = terminal.payload["result"]["structured"]
    value.update(total_files=None, truncated=True)
    if fault == "missing-total":
        del value["total_files"]
    elif fault == "boolean-total":
        value["total_files"] = True
    elif fault == "not-truncated":
        value["truncated"] = False
    events = _events()
    history = (*events[:2], started, terminal, *events[2:])
    if fault is None:
        module._require_serial_check_quiescence(history)
    else:
        with pytest.raises(module.MaintenanceReconciliationUnavailable):
            module._require_serial_check_quiescence(history)


@pytest.mark.parametrize("outcome", ["partial", "ambiguous", "cancelled", "failed", "unknown"])
def test_patch_success_flag_does_not_override_unsettled_outcome(outcome):
    from cayu.guides import coding_host_evidence as module

    started, terminal = _file_events("apply_patch")
    terminal.payload["result"]["structured"]["outcome"] = outcome
    events = _events()
    with pytest.raises(module.MaintenanceReconciliationUnavailable):
        module._require_serial_check_quiescence((*events[:2], started, terminal, *events[2:]))


@pytest.mark.parametrize(
    "fault",
    ["missing-model", "missing-check", "parallel", "other-tool", "uncertain", "bool-attempt"],
)
def test_effect_census_refuses_ambiguous_evidence(fault):
    from cayu.guides import coding_host_evidence as module

    events = list(_events())
    if fault == "missing-model":
        del events[1]
    elif fault == "missing-check":
        events.pop()
    elif fault == "parallel":
        events.insert(3, events[2])
    elif fault == "other-tool":
        events[2].tool_name = "run_command"
    elif fault == "uncertain":
        events[3].payload["result"]["structured"]["cleanup_uncertain"] = True
    else:
        events[1].payload["attempt"] = True
    with pytest.raises(module.MaintenanceReconciliationUnavailable):
        module._require_serial_check_quiescence(events)


@pytest.mark.parametrize(
    "task_type",
    [
        "maintenance.coding",
        "maintenance.git_preparation",
        "maintenance.git_delivery",
        "maintenance.github_delivery",
    ],
)
def test_worker_generation_requires_positive_host_observation(monkeypatch, task_type):
    from cayu.guides import coding_host_owner as module

    observed = {
        "id": "a" * 64,
        "started": "2026-09-13T00:00:00Z",
        "status": "running",
        "running": True,
        "finished": "0001-01-01T00:00:00Z",
    }

    async def inspect(target, observed_task_type):
        observed_id = observed["id"]
        assert isinstance(observed_id, str) and observed_id.startswith(target)
        assert observed_task_type == task_type
        return dict(observed)

    monkeypatch.setenv("CAYU_MAINTENANCE_WORKER_OWNER", "docker")
    monkeypatch.setattr(module.socket, "gethostname", lambda: "a" * 12)
    monkeypatch.setattr(module, "_observe", inspect)

    async def scenario():
        worker = await module.maintenance_worker_id(task_type)
        with pytest.raises(module.WorkerOwnerUnavailable):
            await module.inspect_stopped_worker(worker, task_type=task_type)
        observed.update(status="exited", running=False, finished="2026-09-13T00:01:00Z")
        proof = await module.inspect_stopped_worker(worker, task_type=task_type)
        observed.update(status="running", running=True, started="2026-09-13T00:02:00Z")
        assert await module.inspect_stopped_worker(worker, task_type=task_type) == proof
        assert await module.maintenance_worker_id(task_type) != worker
        with pytest.raises(module.WorkerOwnerUnavailable):
            await module.inspect_stopped_worker("legacy-worker", task_type=task_type)
        other = (
            "maintenance.git_delivery"
            if task_type == "maintenance.coding"
            else "maintenance.coding"
        )
        with pytest.raises(module.WorkerOwnerUnavailable):
            await module.inspect_stopped_worker(worker, task_type=other)

    asyncio.run(scenario())


def test_cancelled_host_observer_retains_bounded_read_owner(monkeypatch):
    from cayu.guides import coding_host_owner as module

    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def blocked(target, task_type):
        assert task_type == "maintenance.coding"
        entered.set()
        assert release.wait(5)
        finished.set()
        return {"target": target}

    monkeypatch.setattr(module, "_inspect", blocked)

    async def scenario():
        task = asyncio.create_task(module._observe("a" * 64))
        try:
            async with asyncio.timeout(5):
                while not entered.is_set():
                    await asyncio.sleep(0)
            task.cancel("first")
            await asyncio.sleep(0)
            task.cancel("second")
            await asyncio.sleep(0)
            assert task.cancelling() == 2 and not task.done()
            assert not finished.is_set()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError, match="first"):
            await task
        assert task.cancelled() and task.cancelling() == 2 and finished.is_set()

    asyncio.run(scenario())


@pytest.mark.parametrize("task_type", [None, True, 1, "", "coding", "maintenance.unknown"])
def test_worker_owner_rejects_unknown_roles_before_observation(monkeypatch, task_type):
    from cayu.guides import coding_host_owner as module

    async def forbidden(*args):
        raise AssertionError("Invalid role reached host inspection")

    monkeypatch.setattr(module, "_observe", forbidden)

    async def scenario():
        with pytest.raises(module.WorkerOwnerUnavailable):
            await module.maintenance_worker_id(task_type)
        with pytest.raises(module.WorkerOwnerUnavailable):
            await module.inspect_stopped_worker("untrusted", task_type=task_type)

    asyncio.run(scenario())


@pytest.mark.parametrize("fault", ["absent", "malformed", "wrong-command", "invalid-time"])
def test_docker_observer_rejects_missing_or_conflicting_evidence(monkeypatch, fault):
    import json
    import subprocess

    from cayu.guides import coding_host_owner as module

    value = {
        "id": "a" * 64,
        "started": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": "running",
        "running": True,
        "finished": "0001-01-01T00:00:00Z",
        "command": ["cayu", "worker", "coding", "--shutdown-grace-seconds", "30"],
    }
    if fault == "wrong-command":
        value["command"] = ["other"]
    if fault == "invalid-time":
        value["started"] = "2026-99-99T00:00:00Z"

    def inspect(command, **kwargs):
        assert command[:3] == ["/usr/local/bin/docker", "--host", "unix:///var/run/docker.sock"]
        assert kwargs["timeout"] == 5
        if fault == "absent":
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(
            stdout=b"null" if fault == "malformed" else json.dumps(value).encode()
        )

    monkeypatch.setattr(module.subprocess, "run", inspect)
    with pytest.raises(module.WorkerOwnerUnavailable):
        module._inspect("a" * 64)
