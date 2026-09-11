"""Host-only PR review/approval queue; no GitHub requests or native artifact writes."""

import json

from configuration.maintenance import (  # ty: ignore[unresolved-import]
    configured_maintenance_github_authority,
)

from cayu import (
    GitHubCheckState,
    GitHubDeliveryApproval,
    GitHubDeliveryRepository,
    GitHubDeliveryState,
    GitHubPullRequestDeliveryRequest,
    GitHubReviewState,
    InvocationOriginClaim,
    Task,
    TaskCreate,
    approve_github_delivery,
    github_pull_request_delivery_request,
)
from tests.qualification.repository_maintenance_git_intake import (
    _completed_digest,
    _encode,
    _nonfinite,
    _stored_origin,
    _unique,
    _validate_claim,
    load_verified_git_result,
)
from tests.qualification.repository_maintenance_identity import copy_identity
from tests.qualification.repository_maintenance_intake import (
    MaintenanceTaskConflict,
    _ensure_task,
    _matches,
)


def _configured_github_request(identity, product, remote):
    try:
        return github_pull_request_delivery_request(
            product,
            remote,
            connector_run_id=identity.github_delivery_task_id,
            session_id=identity.session_id,
            idempotency_key=identity.github_delivery_task_id,
            **configured_maintenance_github_authority(),
        )
    except (ValueError, TypeError):
        raise MaintenanceTaskConflict() from None


async def load_github_approval_request(application, reservations, expected):
    """Caller must authorize the run; review is not permission or a provider observation."""
    identity = copy_identity(expected)
    _task, product, remote = await load_verified_git_result(application, reservations, identity)
    return _configured_github_request(identity, product, remote)


async def load_verified_github_result(application, reservations, expected):
    """Read recorded verified delivery; never observe GitHub or authorize a retry."""
    identity = copy_identity(expected)
    task, product, remote = await load_verified_git_result(application, reservations, identity)
    completed = await application.app.task_store.load_task(identity.github_delivery_task_id)
    request, approval, _ = _stored_github_delivery(completed, identity)
    digest = _completed_digest(completed, request)
    if (
        completed.worker_id is not None
        or completed.lease_expires_at is not None
        or request != _configured_github_request(identity, product, remote)
    ):
        raise MaintenanceTaskConflict()
    publication = await GitHubDeliveryRepository(application.artifact_store).latest(request)
    if publication is None or publication.artifact.sha256 != digest:
        raise MaintenanceTaskConflict()
    result = publication.result
    pr = result.pull_request
    if (
        result.state not in {GitHubDeliveryState.CHECKS_PASSED, GitHubDeliveryState.APPROVED}
        or result.checks_state is not GitHubCheckState.PASSED
        or result.review_state is GitHubReviewState.CHANGES_REQUESTED
        or (
            request.reviews.approval_required
            and result.review_state is not GitHubReviewState.APPROVED
        )
        or result.checks_truncated
        or result.feedback_truncated
        or result.next_poll_at is not None
        or result.next_poll_after_seconds is not None
        or (result.approval_id, result.approval_fingerprint)
        != (approval.approval_id, approval.fingerprint)
        or pr is None
        or pr.state != "open"
        or pr.merged
        or (pr.base_ref, pr.base_commit, pr.head_ref, pr.head_commit)
        != (
            request.repository.base_ref,
            request.repository.expected_base_commit,
            request.repository.head_ref,
            request.repository.head_commit,
        )
    ):
        raise MaintenanceTaskConflict()
    return task, product, remote, publication


def _task_request(identity, native, approval, origin):
    inputs = {
        "maintenance_run_id": identity.public_id,
        "request_json": _encode(native),
        "approval_json": _encode(approval),
    }
    if len(json.dumps(inputs, ensure_ascii=False).encode("utf-8")) > 65536:
        raise MaintenanceTaskConflict()
    return TaskCreate(
        task_id=identity.github_delivery_task_id,
        type="maintenance.github_delivery",
        title="Repository maintenance approved GitHub delivery",
        input=inputs,
        metadata={"maintenance_intent_fingerprint": identity.intent.fingerprint},
        invocation_origin=origin,
    )


async def ensure_github_delivery_task(
    application, reservations, expected, *, actor_subject, expected_request_fingerprint, approval_id
):
    """Persist separate exact GitHub consent; never infer it from approved Git push."""
    identity = copy_identity(expected)
    if any(
        type(value) is not str or not value
        for value in (actor_subject, expected_request_fingerprint, approval_id)
    ):
        raise MaintenanceTaskConflict()
    try:
        origin = InvocationOriginClaim(subject=actor_subject, tenant=identity.intent.tenant)
    except ValueError:
        raise MaintenanceTaskConflict() from None
    native = await load_github_approval_request(application, reservations, identity)
    if native.fingerprint != expected_request_fingerprint:
        raise MaintenanceTaskConflict()
    try:
        approval = approve_github_delivery(native, approval_id=approval_id)
    except ValueError:
        raise MaintenanceTaskConflict() from None
    return await _ensure_task(application.app, _task_request(identity, native, approval, origin))


def _decode_json(encoded, model):
    try:
        if type(encoded) is not str or len(encoded.encode("utf-8")) > 65536:
            raise ValueError
        raw = json.loads(encoded, object_pairs_hook=_unique, parse_constant=_nonfinite)
        return model.model_validate_json(json.dumps(raw, allow_nan=False), strict=True)
    except (ValueError, TypeError, RecursionError):
        raise MaintenanceTaskConflict() from None


async def load_claimed_github_delivery(app, reservations, claimed, worker_id):
    if type(claimed) is not Task or type(worker_id) is not str or not worker_id:
        raise MaintenanceTaskConflict()
    identity = await reservations.load_for_task(claimed.id)
    if identity is None or app.task_store is None:
        raise MaintenanceTaskConflict()
    identity = copy_identity(identity)
    current = await app.task_store.load_task(identity.github_delivery_task_id)
    native, approval, request = _stored_github_delivery(current, identity)
    _validate_claim(current, claimed, request, worker_id)
    return identity, native, approval


def _stored_github_delivery(current, identity):
    origin = _stored_origin(current, identity)
    native = _decode_json(current.input.get("request_json"), GitHubPullRequestDeliveryRequest)
    approval = _decode_json(current.input.get("approval_json"), GitHubDeliveryApproval)
    if (
        native.connector_run_id != identity.github_delivery_task_id
        or native.idempotency_key != identity.github_delivery_task_id
        or native.session_id != identity.session_id
        or native.source.product_run_id != identity.product_run_id
        or native.source.remote_delivery_id != identity.git_delivery_task_id
        or approval != approve_github_delivery(native, approval_id=approval.approval_id)
    ):
        raise MaintenanceTaskConflict()
    request = _task_request(identity, native, approval, origin)
    if not _matches(current, request):
        raise MaintenanceTaskConflict()
    return native, approval, request
