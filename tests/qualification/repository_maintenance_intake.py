"""Trusted application bridge from immutable reservation to an outer coding task."""

from uuid import UUID

from cayu import (
    CayuApp,
    InvocationOrigin,
    InvocationOriginClaim,
    InvocationOriginTrust,
    Task,
    TaskCreate,
    TaskExecutionSource,
    TaskInvocation,
    TaskStatus,
)
from tests.qualification.repository_maintenance_identity import (
    MaintenanceRunIdentity,
    copy_identity,
)
from tests.qualification.repository_maintenance_runs import (
    PostgresMaintenanceRunStore,
    SQLiteMaintenanceRunStore,
)


class MaintenanceTaskConflict(ValueError):
    def __init__(self):
        super().__init__("Reserved maintenance task conflicts with durable application authority.")


def _matches(task, request):
    if type(task) is not Task:
        return False
    expected = {
        "id": request.task_id,
        "type": request.type,
        "title": request.title,
        "description": None,
        "session_id": None,
        "session_instance_id": None,
        "parent_task_id": None,
        "assigned_agent_name": None,
        "retry_series": None,
        "work_contract": None,
    }
    for name, value in expected.items():
        actual = getattr(task, name)
        if type(actual) is not type(value) or actual != value:
            return False
    for name in ("input", "metadata"):
        actual = getattr(task, name)
        wanted = getattr(request, name)
        if (
            type(actual) is not dict
            or any(type(key) is not str for key in actual)
            or set(actual) != set(wanted)
        ):
            return False
        if any(
            type(actual[key]) is not str or actual[key] != value for key, value in wanted.items()
        ):
            return False
    invocation = task.invocation
    if type(invocation) is not TaskInvocation or type(invocation.origin) is not InvocationOrigin:
        return False
    origin = invocation.origin
    if (
        origin.trust is not InvocationOriginTrust.HOST_ASSERTED
        or type(origin.subject) is not str
        or origin.subject != request.invocation_origin.subject
        or type(origin.tenant) is not str
        or origin.tenant != request.invocation_origin.tenant
        or invocation.source is not TaskExecutionSource.SDK_TASK
        or invocation.root_session_id is not None
        or type(invocation.root_invocation_id) is not str
    ):
        return False
    try:
        root = UUID(invocation.root_invocation_id)
    except ValueError:
        return False
    return root.version == 4 and str(root) == invocation.root_invocation_id


def _task_request(identity):
    return TaskCreate(
        task_id=identity.task_id,
        type="maintenance.coding",
        title="Repository maintenance coding",
        input={"maintenance_run_id": identity.public_id},
        metadata={"maintenance_intent_fingerprint": identity.intent.fingerprint},
        invocation_origin=InvocationOriginClaim(
            subject=identity.intent.subject, tenant=identity.intent.tenant
        ),
    )


async def load_claimed_coding_identity(
    app: CayuApp,
    reservations: PostgresMaintenanceRunStore | SQLiteMaintenanceRunStore,
    claimed: Task,
    worker_id: str,
) -> MaintenanceRunIdentity:
    """Read-only trusted handler lookup; Runtime still owns lease/dispatch authority."""
    if (
        type(claimed) is not Task
        or type(worker_id) is not str
        or not worker_id
        or claimed.status is not TaskStatus.CLAIMED
        or type(claimed.worker_id) is not str
        or claimed.worker_id != worker_id
    ):
        raise MaintenanceTaskConflict()
    identity = await reservations.load_for_task(claimed.id)
    if type(identity) is not MaintenanceRunIdentity or app.task_store is None:
        raise MaintenanceTaskConflict()
    identity = copy_identity(identity)
    request = _task_request(identity)
    if not _matches(claimed, request):
        raise MaintenanceTaskConflict()
    current = await app.task_store.load_task(identity.task_id)
    if (
        type(current) is not Task
        or not _matches(current, request)
        or current.status is not TaskStatus.CLAIMED
        or type(current.worker_id) is not str
        or current.worker_id != worker_id
        or current.invocation != claimed.invocation
    ):
        raise MaintenanceTaskConflict()
    return identity


async def load_owned_coding_task(app: CayuApp, identity: MaintenanceRunIdentity) -> Task | None:
    """Read-only projection helper after a trusted tenant-qualified reservation lookup."""
    identity = copy_identity(identity)
    if app.task_store is None:
        raise MaintenanceTaskConflict()
    task = await app.task_store.load_task(identity.task_id)
    if task is not None and not _matches(task, _task_request(identity)):
        raise MaintenanceTaskConflict()
    return task


async def ensure_coding_task(
    app: CayuApp,
    reservations: PostgresMaintenanceRunStore | SQLiteMaintenanceRunStore,
    expected: MaintenanceRunIdentity,
) -> Task:
    """Host-only entrance; expected data is not an authentication token.

    The caller must authenticate and validate current configuration before this
    operation. No returned Runtime record may be exposed directly as a product response.
    """
    expected = copy_identity(expected)
    identity = await reservations.load_owned(
        tenant=expected.intent.tenant, public_id=expected.public_id
    )
    if type(identity) is not MaintenanceRunIdentity:
        raise MaintenanceTaskConflict()
    identity = copy_identity(identity)
    if identity != expected:
        raise MaintenanceTaskConflict()
    request = _task_request(identity)
    return await _ensure_task(app, request)


async def _ensure_task(app, request):
    """Shared application-local insertion; caller owns reservation and origin checks."""
    if app.task_store is None:
        raise RuntimeError("Maintenance intake requires a task store.")
    existing = await app.task_store.load_task(request.task_id)
    if existing is None:
        try:
            existing = await app.create_task(request)
        except ValueError:
            existing = await app.task_store.load_task(request.task_id)
            if existing is None:
                raise
    if existing is None or not _matches(existing, request):
        raise MaintenanceTaskConflict()
    return existing
