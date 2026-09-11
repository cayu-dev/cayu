"""Host-only Git task intake and store-owned claim reconstruction."""

import json

from configuration.maintenance import (  # ty: ignore[unresolved-import]
    configured_maintenance_git_authority,
)

from cayu import (
    InvocationOrigin,
    InvocationOriginClaim,
    InvocationOriginTrust,
    RemoteGitDeliveryApproval,
    RemoteGitDeliveryRepository,
    RemoteGitDeliveryRequest,
    RemoteGitDeliveryState,
    Task,
    TaskCreate,
    TaskInvocation,
    TaskStatus,
    approve_remote_git_delivery,
    remote_git_delivery_request,
)
from tests.qualification.repository_maintenance_identity import copy_identity
from tests.qualification.repository_maintenance_intake import (
    MaintenanceTaskConflict,
    _ensure_task,
    _matches,
)
from tests.qualification.repository_maintenance_results import load_verified_coding_result


async def ensure_git_preparation_task(application, reservations, expected, *, actor_subject):
    """Caller must authenticate the operator; matching identity is not authentication.

    Persist full first-accepted request in Runtime task input. Changed actor or
    configuration conflicts on replay; worker restart must use the stored input.
    """
    identity = copy_identity(expected)
    try:
        if type(actor_subject) is not str:
            raise ValueError
        origin = InvocationOriginClaim(subject=actor_subject, tenant=identity.intent.tenant)
    except ValueError:
        raise ValueError("Invalid maintenance preparation actor.") from None
    _task, publication = await load_verified_coding_result(application, reservations, identity)
    native = _configured_git_request(identity, publication)
    encoded = _encode(native)
    request = _preparation_request(identity, encoded, origin)
    return await _ensure_task(application.app, request)


def _configured_git_request(identity, publication):
    repository, commit, security, limits = configured_maintenance_git_authority()
    return remote_git_delivery_request(
        publication,
        delivery_id=identity.git_delivery_task_id,
        session_id=identity.session_id,
        idempotency_key=identity.git_delivery_task_id,
        repository=repository,
        commit=commit,
        security=security,
        limits=limits,
    )


def _encode(native):
    encoded = json.dumps(
        native.model_dump(mode="json", warnings=False),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    if len(encoded.encode("utf-8")) > 65536:
        raise ValueError("Maintenance Git request exceeds its storage bound.")
    return encoded


def _preparation_request(identity, encoded, origin):
    return TaskCreate(
        task_id=identity.git_preparation_task_id,
        type="maintenance.git_preparation",
        title="Repository maintenance Git preparation",
        input={"maintenance_run_id": identity.public_id, "request_json": encoded},
        metadata={"maintenance_intent_fingerprint": identity.intent.fingerprint},
        invocation_origin=origin,
    )


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _nonfinite(_value):
    raise ValueError


async def load_claimed_git_preparation(app, reservations, claimed, worker_id):
    """Reconstruct from a store-owned claim; no execution permission from raw JSON."""
    if type(claimed) is not Task or type(worker_id) is not str or not worker_id:
        raise MaintenanceTaskConflict()
    identity = await reservations.load_for_task(claimed.id)
    if identity is None or app.task_store is None:
        raise MaintenanceTaskConflict()
    identity = copy_identity(identity)
    current = await app.task_store.load_task(identity.git_preparation_task_id)
    native, request = _stored_preparation(current, identity)
    _validate_claim(current, claimed, request, worker_id)
    return identity, native


def _validate_claim(current, claimed, request, worker_id):
    if (
        not _matches(claimed, request)
        or current.status is not TaskStatus.CLAIMED
        or claimed.status is not TaskStatus.CLAIMED
        or type(current.worker_id) is not str
        or type(claimed.worker_id) is not str
        or current.worker_id != worker_id
        or claimed.worker_id != worker_id
        or current.invocation != claimed.invocation
    ):
        raise MaintenanceTaskConflict()


def _stored_preparation(current, identity):
    origin = _stored_origin(current, identity)
    native = _decode(current.input.get("request_json"), RemoteGitDeliveryRequest)
    request = _preparation_request(identity, _encode(native), origin)
    _validate_native_identity(native, identity)
    if not _matches(current, request):
        raise MaintenanceTaskConflict()
    return native, request


def _stored_origin(current, identity):
    if (
        type(current) is not Task
        or type(current.invocation) is not TaskInvocation
        or type(current.invocation.origin) is not InvocationOrigin
        or current.invocation.origin.trust is not InvocationOriginTrust.HOST_ASSERTED
        or type(current.invocation.origin.subject) is not str
        or type(current.invocation.origin.tenant) is not str
        or current.invocation.origin.tenant != identity.intent.tenant
        or type(current.input) is not dict
        or any(type(key) is not str for key in current.input)
    ):
        raise MaintenanceTaskConflict()
    try:
        return InvocationOriginClaim(
            subject=current.invocation.origin.subject, tenant=identity.intent.tenant
        )
    except ValueError:
        raise MaintenanceTaskConflict() from None


def _decode(encoded, model):
    try:
        if type(encoded) is not str or len(encoded.encode("utf-8")) > 65536:
            raise ValueError
        raw = json.loads(encoded, object_pairs_hook=_unique, parse_constant=_nonfinite)
        return model.model_validate(raw, strict=True)
    except (ValueError, TypeError, RecursionError):
        raise MaintenanceTaskConflict() from None


def _validate_native_identity(native, identity):
    if (
        native.delivery_id != identity.git_delivery_task_id
        or native.idempotency_key != identity.git_delivery_task_id
        or native.session_id != identity.session_id
        or native.source.product_run_id != identity.product_run_id
    ):
        raise MaintenanceTaskConflict()


async def load_claimed_git_delivery(app, reservations, claimed, worker_id):
    """Restore the saved explicit approval only from the matching Runtime claim."""
    if type(claimed) is not Task or type(worker_id) is not str or not worker_id:
        raise MaintenanceTaskConflict()
    identity = await reservations.load_for_task(claimed.id)
    if identity is None or app.task_store is None:
        raise MaintenanceTaskConflict()
    identity = copy_identity(identity)
    current = await app.task_store.load_task(identity.git_delivery_task_id)
    native, approval, request = _stored_delivery(current, identity)
    _validate_claim(current, claimed, request, worker_id)
    return identity, native, approval


def _stored_delivery(current, identity):
    origin = _stored_origin(current, identity)
    native = _decode(current.input.get("request_json"), RemoteGitDeliveryRequest)
    approval = _decode(current.input.get("approval_json"), RemoteGitDeliveryApproval)
    _validate_native_identity(native, identity)
    if (
        approval.request_fingerprint != native.fingerprint
        or approval.policy_fingerprint != native.security.policy_fingerprint
        or approval.commit_approved is not True
        or approval.push_approved is not True
    ):
        raise MaintenanceTaskConflict()
    request = _delivery_request(identity, native, approval, origin)
    if not _matches(current, request):
        raise MaintenanceTaskConflict()
    return native, approval, request


async def load_git_approval_request(application, reservations, expected):
    """Read exact historical pending evidence after caller authentication/authorization."""
    identity = copy_identity(expected)
    _task, product = await load_verified_coding_result(application, reservations, identity)
    completed = await application.app.task_store.load_task(identity.git_preparation_task_id)
    native, _request = _stored_preparation(completed, identity)
    digest = _completed_digest(completed, native)
    if native != _configured_git_request(identity, product):
        raise MaintenanceTaskConflict()
    repository = RemoteGitDeliveryRepository(application.artifact_store)
    pending = await repository.load_result(native, digest)
    prepared = await repository.load_prepared(native)
    if (
        pending.result.state is not RemoteGitDeliveryState.APPROVAL_REQUIRED
        or prepared is None
        or pending.result.tree != prepared.tree
    ):
        raise MaintenanceTaskConflict()
    return native, prepared, pending


def _completed_digest(completed, native):
    if completed.status is not TaskStatus.COMPLETED:
        raise MaintenanceTaskConflict()
    result = completed.result
    if (
        type(result) is not dict
        or any(type(key) is not str for key in result)
        or set(result) != {"request_fingerprint", "result_digest"}
        or any(type(value) is not str for value in result.values())
        or result["request_fingerprint"] != native.fingerprint
    ):
        raise MaintenanceTaskConflict()
    digest = result["result_digest"]
    if (
        len(digest) != 71
        or not digest.startswith("sha256:")
        or any(char not in "0123456789abcdef" for char in digest[7:])
    ):
        raise MaintenanceTaskConflict()
    return digest


async def load_verified_git_result(application, reservations, expected):
    """Read completed approved push evidence, not permission for a new remote effect."""
    identity = copy_identity(expected)
    task, product = await load_verified_coding_result(application, reservations, identity)
    completed = await application.app.task_store.load_task(identity.git_delivery_task_id)
    native, approval, _request = _stored_delivery(completed, identity)
    digest = _completed_digest(completed, native)
    if native != _configured_git_request(identity, product):
        raise MaintenanceTaskConflict()
    repository = RemoteGitDeliveryRepository(application.artifact_store)
    publication = await repository.load_result(native, digest)
    result = publication.result
    receipts = await repository.load_lifecycle(native)
    if (
        result.state is not RemoteGitDeliveryState.PUSHED
        or result.cleanup_settled is not True
        or result.approval_id != approval.approval_id
        or result.approval_fingerprint != approval.fingerprint
        or result.tree != approval.prepared_tree
        or not receipts
        or receipts[-1].state is not RemoteGitDeliveryState.PUSHED
        or receipts[-1].evidence_sha256 != publication.artifact.sha256
    ):
        raise MaintenanceTaskConflict()
    return task, product, publication


async def ensure_git_delivery_task(
    application,
    reservations,
    expected,
    *,
    actor_subject,
    expected_request_fingerprint,
    expected_tree,
    approval_id,
):
    """Host-only explicit approval; enqueue does not publish native approval or Git effects."""
    identity = copy_identity(expected)
    if any(
        type(value) is not str or not value
        for value in (
            actor_subject,
            expected_request_fingerprint,
            expected_tree,
            approval_id,
        )
    ):
        raise MaintenanceTaskConflict()
    try:
        origin = InvocationOriginClaim(subject=actor_subject, tenant=identity.intent.tenant)
    except ValueError:
        raise MaintenanceTaskConflict() from None
    native, prepared, pending = await load_git_approval_request(application, reservations, identity)
    if native.fingerprint != expected_request_fingerprint or prepared.tree != expected_tree:
        raise MaintenanceTaskConflict()
    try:
        approval = approve_remote_git_delivery(native, prepared, approval_id=approval_id)
    except ValueError:
        raise MaintenanceTaskConflict() from None
    request = _delivery_request(identity, native, approval, origin)
    existing = await application.app.task_store.load_task(identity.git_delivery_task_id)
    if existing is None:
        repository = RemoteGitDeliveryRepository(application.artifact_store)
        receipts = await repository.load_lifecycle(native)
        if (
            not receipts
            or receipts[-1].state is not RemoteGitDeliveryState.APPROVAL_REQUIRED
            or receipts[-1].evidence_sha256 != pending.artifact.sha256
        ):
            raise MaintenanceTaskConflict()
    return await _ensure_task(application.app, request)


def _delivery_request(identity, native, approval, origin):
    inputs = {
        "maintenance_run_id": identity.public_id,
        "request_json": _encode(native),
        "approval_json": _encode(approval),
    }
    if len(json.dumps(inputs, ensure_ascii=False).encode("utf-8")) > 65536:
        raise MaintenanceTaskConflict()
    return TaskCreate(
        task_id=identity.git_delivery_task_id,
        type="maintenance.git_delivery",
        title="Repository maintenance approved Git delivery",
        input=inputs,
        metadata={"maintenance_intent_fingerprint": identity.intent.fingerprint},
        invocation_origin=origin,
    )
