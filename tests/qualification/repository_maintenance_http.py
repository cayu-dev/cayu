"""Bounded coding-stage HTTP intake; execution belongs to Runtime workers."""

import json
from uuid import UUID

from domain.coding_product import CodingProductTask  # ty: ignore[unresolved-import]
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from operations.maintenance_requests import (  # ty: ignore[unresolved-import]
    capture_accepted_request,
)

from cayu import Task, TaskStatus
from cayu.server import AuthenticatedAccess, mount_cayu
from tests.qualification.repository_maintenance_auth import MaintenanceAccess
from tests.qualification.repository_maintenance_budget import require_maintenance_budget
from tests.qualification.repository_maintenance_cost import inspect_cost_evidence
from tests.qualification.repository_maintenance_delivery_view import inspect_delivery_evidence
from tests.qualification.repository_maintenance_final import (
    MaintenanceAcceptanceRejected,
    inspect_acceptance_evidence,
    inspect_final_result,
)
from tests.qualification.repository_maintenance_git_intake import (
    ensure_git_delivery_task,
    ensure_git_preparation_task,
    load_git_approval_request,
)
from tests.qualification.repository_maintenance_github_intake import (
    ensure_github_delivery_task,
    load_github_approval_request,
)
from tests.qualification.repository_maintenance_identity import (
    MaintenanceRunIntent,
    MaintenanceTaskPhase,
    copy_identity,
    task_id_for,
)
from tests.qualification.repository_maintenance_intake import (
    MaintenanceTaskConflict,
    ensure_coding_task,
    load_owned_coding_task,
)
from tests.qualification.repository_maintenance_operator import inspect_reserved_tasks
from tests.qualification.repository_maintenance_request import bounded_text, encode_request
from tests.qualification.repository_maintenance_results import MaintenanceResultUnavailable
from tests.qualification.repository_maintenance_runs import ReservationConflict


def _unique(pairs):
    data = {}
    for key, value in pairs:
        if key in data:
            raise ValueError
        data[key] = value
    return data


def _nonfinite(_value):
    raise ValueError


async def _json_body(request):
    raw = bytearray()
    async for chunk in request.stream():
        if len(raw) + len(chunk) > 8192:
            raise HTTPException(status_code=413, detail="Request body too large.")
        raw.extend(chunk)
    try:
        data = json.loads(bytes(raw), object_pairs_hook=_unique, parse_constant=_nonfinite)
        if type(data) is not dict:
            raise ValueError
        return data
    except (ValueError, UnicodeError, RecursionError):
        raise HTTPException(status_code=422, detail="Invalid maintenance request.") from None


async def _body(request):
    data = await _json_body(request)
    try:
        if set(data) != {"instruction", "idempotency_key"}:
            raise ValueError
        instruction = bounded_text(data["instruction"], bound=4096)
        key = bounded_text(data["idempotency_key"])
        if len(key) > 128 or key.strip() != key or any(ord(c) < 32 or ord(c) == 127 for c in key):
            raise ValueError
        return instruction, key
    except (ValueError, UnicodeError, RecursionError):
        raise HTTPException(status_code=422, detail="Invalid maintenance request.") from None


def _response(identity, task):
    if task is not None and type(task.status) is not TaskStatus:
        raise MaintenanceTaskConflict()
    return {
        "id": identity.public_id,
        "coding_task_status": "intake_pending" if task is None else task.status.value,
    }


async def _approval_body(request, *, require_tree):
    body = await _json_body(request)
    try:
        fields = {"request_fingerprint", "approval_id"}
        if require_tree:
            fields.add("prepared_tree")
        if set(body) != fields or any(type(value) is not str for value in body.values()):
            raise ValueError
        fingerprint, key = body["request_fingerprint"], body["approval_id"]
        bounded_text(key, bound=512)
        if (
            len(fingerprint) != 71
            or not fingerprint.startswith("sha256:")
            or any(c not in "0123456789abcdef" for c in fingerprint[7:])
            or not 1 <= len(key) <= 128
            or key.strip() != key
            or any(ord(c) < 32 or ord(c) == 127 for c in key)
        ):
            raise ValueError
        if require_tree:
            tree = body["prepared_tree"]
            if len(tree) not in {40, 64} or any(c not in "0123456789abcdef" for c in tree):
                raise ValueError
        return body
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid maintenance approval.") from None


def build_maintenance_server(
    application, reservations, access: MaintenanceAccess, *, lifespan=None
):
    """Construct the host with caller-owned dependencies and optional lifespan.

    The host lifespan surrounds the mounted Runtime startup/shutdown. Mount exit
    alone is not quiescence proof: dependency owners must verify drain outcomes
    before closing resources still usable by retained work.
    """
    if type(access) is not MaintenanceAccess:
        raise ValueError("Maintenance host requires explicit authentication.")
    require_maintenance_budget(application.app.budget_policy)
    server = FastAPI(lifespan=lifespan)

    @server.exception_handler(RequestValidationError)
    async def invalid_http(_request, _error):
        return JSONResponse(status_code=422, content={"detail": "Invalid maintenance request."})

    @server.post("/runs", status_code=202)
    async def create_run(request: Request):
        principal = await access.product(request)
        instruction, key = await _body(request)
        try:
            require_maintenance_budget(application.app.budget_policy)
            provisional = CodingProductTask(
                product_run_id="inspection-product",
                session_id="inspection-session",
                task_id="inspection-task",
                instruction=instruction,
            )
            accepted = await capture_accepted_request(application, provisional)
            identity = await reservations.reserve(
                MaintenanceRunIntent(
                    tenant=principal.tenant_id,
                    subject=principal.subject_id,
                    idempotency_key=key,
                    request_json=encode_request(accepted),
                )
            )
            task = await ensure_coding_task(application.app, reservations, identity)
            return _response(identity, task)
        except (ReservationConflict, MaintenanceTaskConflict):
            raise HTTPException(
                status_code=409, detail="Conflicting maintenance request."
            ) from None
        except Exception:
            # A commit may have succeeded. Retry the same key; never claim rollback.
            raise HTTPException(status_code=503, detail="Maintenance intake unavailable.") from None

    async def lookup_identity(public_id, tenant):
        try:
            if len(public_id) != 36 or str(UUID(public_id)) != public_id:
                raise ValueError
        except ValueError:
            raise HTTPException(status_code=404, detail="Run not found.") from None
        try:
            identity = await reservations.load_owned(tenant=tenant, public_id=public_id)
            if identity is None:
                raise HTTPException(status_code=404, detail="Run not found.")
            return copy_identity(identity)
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=503, detail="Maintenance state unavailable.") from None

    async def coding_response(identity):
        try:
            task = await load_owned_coding_task(application.app, identity)
            return _response(identity, task)
        except Exception:
            raise HTTPException(status_code=503, detail="Maintenance state unavailable.") from None

    @server.get("/runs/{public_id}")
    async def read_run(public_id: str, request: Request):
        principal = await access.product(request)
        identity = await lookup_identity(public_id, principal.tenant_id)
        return await coding_response(identity)

    @server.get("/operator/runs/{public_id}")
    async def inspect_run(public_id: str, request: Request):
        _actor, identity, response = await operator_lookup(public_id, request)
        try:
            response["allocated_references"] = {
                "product_run_id": identity.product_run_id,
                "coding_session_id": application.app.project_session_id_for_exposure(
                    identity.session_id
                ),
                "workflow_session_id": application.app.project_session_id_for_exposure(
                    identity.workflow_session_id
                ),
                "tasks": {
                    phase.value: task_id_for(identity, phase) for phase in MaintenanceTaskPhase
                },
            }
            return response
        except Exception:
            raise HTTPException(status_code=503, detail="Maintenance state unavailable.") from None

    @server.get("/operator/runs/{public_id}/tasks")
    async def inspect_tasks(public_id: str, request: Request):
        _actor, identity = await operator_identity(public_id, request)
        return await inspect_reserved_tasks(application.app, identity)

    @server.get("/operator/runs/{public_id}/delivery")
    async def inspect_delivery(public_id: str, request: Request):
        _actor, identity = await operator_identity(public_id, request)
        return await inspect_delivery_evidence(application, identity)

    @server.get("/operator/runs/{public_id}/cost")
    async def inspect_cost(public_id: str, request: Request):
        _actor, identity = await operator_identity(public_id, request)
        try:
            return await inspect_cost_evidence(application.app, identity)
        except Exception:
            raise HTTPException(status_code=503, detail="Maintenance cost unavailable.") from None

    async def operator_identity(public_id, request):
        actor = await access.operator(request)
        tenants = request.query_params.getlist("tenant")
        try:
            if len(tenants) != 1:
                raise ValueError
            tenant = MaintenanceRunIntent(
                tenant=tenants[0], subject="lookup", idempotency_key="lookup", request_json="{}"
            ).tenant
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid maintenance lookup.") from None
        return actor, await lookup_identity(public_id, tenant)

    async def result_evidence(public_id, request, reader):
        _actor, identity = await operator_identity(public_id, request)
        try:
            return await reader(application, reservations, identity)
        except (
            MaintenanceTaskConflict,
            MaintenanceResultUnavailable,
            MaintenanceAcceptanceRejected,
        ):
            raise HTTPException(
                status_code=409, detail="Maintenance result is not verified."
            ) from None
        except Exception:
            raise HTTPException(status_code=503, detail="Maintenance result unavailable.") from None

    @server.get("/operator/runs/{public_id}/result")
    async def final_result(public_id: str, request: Request):
        return await result_evidence(public_id, request, inspect_final_result)

    @server.get("/operator/runs/{public_id}/acceptance")
    async def acceptance_result(public_id: str, request: Request):
        return await result_evidence(public_id, request, inspect_acceptance_evidence)

    async def operator_lookup(public_id, request):
        actor, identity = await operator_identity(public_id, request)
        return actor, identity, await coding_response(identity)

    def phase_response(identity, task, phase):
        if type(task) is not Task or type(task.status) is not TaskStatus:
            raise MaintenanceTaskConflict()
        return {"id": identity.public_id, "phase": phase, "task_status": task.status.value}

    @server.post("/operator/runs/{public_id}/git/preparation", status_code=202)
    async def prepare_git(public_id: str, request: Request):
        actor, identity, _response = await operator_lookup(public_id, request)
        if await _json_body(request) != {}:
            raise HTTPException(status_code=422, detail="Invalid maintenance request.")
        try:
            task = await ensure_git_preparation_task(
                application, reservations, identity, actor_subject=actor.subject
            )
            return phase_response(identity, task, "git_preparation")
        except (MaintenanceTaskConflict, MaintenanceResultUnavailable):
            raise HTTPException(
                status_code=409, detail="Conflicting maintenance preparation."
            ) from None
        except Exception:
            raise HTTPException(
                status_code=503, detail="Maintenance preparation unavailable."
            ) from None

    @server.get("/operator/runs/{public_id}/git/approval")
    async def inspect_git_approval(public_id: str, request: Request):
        _actor, identity, _response = await operator_lookup(public_id, request)
        try:
            native, prepared, pending = await load_git_approval_request(
                application, reservations, identity
            )
            return {
                "id": identity.public_id,
                "recorded_state": "approval_required",
                "request_fingerprint": native.fingerprint,
                "prepared_tree": prepared.tree,
                "pending_result_digest": pending.artifact.sha256,
                "request": native.model_dump(mode="json", warnings=False),
            }
        except (MaintenanceTaskConflict, MaintenanceResultUnavailable):
            raise HTTPException(
                status_code=409, detail="Maintenance approval evidence unavailable."
            ) from None
        except Exception:
            raise HTTPException(status_code=503, detail="Maintenance state unavailable.") from None

    @server.post("/operator/runs/{public_id}/git/approval", status_code=202)
    async def approve_git(public_id: str, request: Request):
        actor, identity, _response = await operator_lookup(public_id, request)
        body = await _approval_body(request, require_tree=True)
        try:
            task = await ensure_git_delivery_task(
                application,
                reservations,
                identity,
                actor_subject=actor.subject,
                expected_request_fingerprint=body["request_fingerprint"],
                expected_tree=body["prepared_tree"],
                approval_id=body["approval_id"],
            )
            return phase_response(identity, task, "git_delivery")
        except (MaintenanceTaskConflict, MaintenanceResultUnavailable):
            raise HTTPException(
                status_code=409, detail="Conflicting maintenance approval."
            ) from None
        except Exception:
            raise HTTPException(
                status_code=503, detail="Maintenance approval unavailable."
            ) from None

    @server.get("/operator/runs/{public_id}/github/approval")
    async def inspect_github_approval(public_id: str, request: Request):
        _actor, identity, _response = await operator_lookup(public_id, request)
        try:
            native = await load_github_approval_request(application, reservations, identity)
            return {
                "id": identity.public_id,
                "phase": "github_delivery",
                "request_fingerprint": native.fingerprint,
                "request": native.model_dump(mode="json", warnings=False),
            }
        except (MaintenanceTaskConflict, MaintenanceResultUnavailable):
            raise HTTPException(
                status_code=409, detail="Maintenance approval evidence unavailable."
            ) from None
        except Exception:
            raise HTTPException(status_code=503, detail="Maintenance state unavailable.") from None

    @server.post("/operator/runs/{public_id}/github/approval", status_code=202)
    async def approve_github(public_id: str, request: Request):
        actor, identity, _response = await operator_lookup(public_id, request)
        body = await _approval_body(request, require_tree=False)
        try:
            task = await ensure_github_delivery_task(
                application,
                reservations,
                identity,
                actor_subject=actor.subject,
                expected_request_fingerprint=body["request_fingerprint"],
                approval_id=body["approval_id"],
            )
            return phase_response(identity, task, "github_delivery")
        except (MaintenanceTaskConflict, MaintenanceResultUnavailable):
            raise HTTPException(
                status_code=409, detail="Conflicting maintenance approval."
            ) from None
        except Exception:
            raise HTTPException(
                status_code=503, detail="Maintenance approval unavailable."
            ) from None

    mount_cayu(
        server,
        application.app,
        path="/internal/cayu",
        access=AuthenticatedAccess(dependency=access.operator),
    )
    return server
