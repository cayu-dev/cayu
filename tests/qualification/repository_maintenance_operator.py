"""Read-only reserved-task observations, never effect or recovery authority."""

from datetime import UTC, datetime
from uuid import UUID

from cayu import TaskStatus
from tests.qualification.repository_maintenance_git_intake import (
    _stored_delivery,
    _stored_preparation,
)
from tests.qualification.repository_maintenance_github_intake import _stored_github_delivery
from tests.qualification.repository_maintenance_identity import (
    MaintenanceTaskPhase,
    copy_identity,
    task_id_for,
)
from tests.qualification.repository_maintenance_intake import (
    MaintenanceTaskConflict,
    _matches,
    _task_request,
)


def _validate_task(task, identity, phase):
    if phase is MaintenanceTaskPhase.CODING:
        if not _matches(task, _task_request(identity)):
            raise MaintenanceTaskConflict()
    elif phase is MaintenanceTaskPhase.GIT_PREPARATION:
        _stored_preparation(task, identity)
    elif phase is MaintenanceTaskPhase.GIT_DELIVERY:
        _stored_delivery(task, identity)
    elif phase is MaintenanceTaskPhase.GITHUB_DELIVERY:
        _stored_github_delivery(task, identity)
    else:
        raise MaintenanceTaskConflict()


def _timestamp(value):
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() is None:
        raise MaintenanceTaskConflict()
    return value.astimezone(UTC).isoformat()


def _owner(worker_id, phase):
    if worker_id is None:
        return {"kind": "not_recorded"}
    if type(worker_id) is not str:
        raise MaintenanceTaskConflict()
    # Only the generated roles' opaque IDs are projected. Other owners remain
    # explicitly present without echoing arbitrary application-controlled text.
    prefix = f"maintenance.{phase.value}-"
    if len(worker_id) == len(prefix) + 32 and worker_id.startswith(prefix):
        suffix = worker_id[len(prefix) :]
        try:
            parsed = UUID(hex=suffix)
            if len(suffix) == 32 and parsed.version == 4 and parsed.hex == suffix:
                return {"kind": "registered_role", "id": worker_id}
        except ValueError:
            pass
    return {"kind": "present_unprojected"}


def _observation(task, phase):
    if type(task.status) is not TaskStatus:
        raise MaintenanceTaskConflict()
    reason = task.status_reason
    if reason is not None and type(reason) is not str:
        raise MaintenanceTaskConflict()
    observed_at = datetime.now(UTC)
    lease = task.lease_expires_at
    lease_at = None if lease is None else _timestamp(lease)
    return {
        "task_status": task.status.value,
        "updated_at": _timestamp(task.updated_at),
        "observed_at": observed_at.isoformat(),
        "recorded_owner": _owner(task.worker_id, phase),
        "recorded_lease_expires_at": lease_at,
        "lease_observation": (
            "not_recorded"
            if lease is None
            else "expired"
            if lease <= observed_at
            else "not_expired"
        ),
        "cancellation_marker": (
            "requested"
            if reason == "cancellation_requested"
            else "not_recorded"
            if reason is None
            else "unrecognized"
        ),
    }


async def inspect_reserved_tasks(app, expected):
    """Caller authenticates and resolves the tenant-owned reservation first.

    Four independent historical observations, not an atomic snapshot or lease
    grant. No native effects, cleanup, or result artifacts are inspected here.
    """
    identity = copy_identity(expected)
    phases = {}
    for phase in MaintenanceTaskPhase:
        task_id = task_id_for(identity, phase)
        row = {"task_id": task_id, "evidence": "unavailable"}
        try:
            if app.task_store is None:
                raise RuntimeError
            task = await app.task_store.load_task(task_id)
        except Exception:
            # Do not format opaque storage errors, nor swallow cancellation.
            phases[phase.value] = row
            continue
        if task is None:
            row["evidence"] = "absent"
        else:
            try:
                _validate_task(task, identity, phase)
                observation = _observation(task, phase)
            except Exception:
                # These validators only inspect the stored record. An opaque
                # malformed value must not erase another phase's evidence.
                row["evidence"] = "conflicting"
            else:
                row = {**row, "evidence": "recorded", **observation}
        phases[phase.value] = row
    return {
        "id": identity.public_id,
        "phases": phases,
        "effect_evidence": "not_inspected",
        "cleanup_evidence": "not_inspected",
    }
