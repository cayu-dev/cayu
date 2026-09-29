"""HTTP interface for the Cayu Lambda MicroVM command sidecar."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from fastapi import FastAPI, HTTPException

from .supervisor import (
    CommandConflictError,
    CommandExecutionBoundary,
    CommandRequestError,
    CommandSupervisor,
    OwnerFence,
    OwnerLifecycleLeasedError,
    OwnerSupersededError,
)

ROOT = os.environ.get("CAYU_MICROVM_WORKSPACE_ROOT", "/workspace")
PROTOCOL_VERSION = "3"
AGENT_UID = int(os.environ.get("CAYU_MICROVM_AGENT_UID", "1000"))
AGENT_GID = int(os.environ.get("CAYU_MICROVM_AGENT_GID", "1000"))
AGENT_NETNS = os.environ.get("CAYU_MICROVM_AGENT_NETNS", "cayu-agent")
EXECUTION_BOUNDARY = CommandExecutionBoundary(
    agent_uid=AGENT_UID,
    agent_gid=AGENT_GID,
    agent_netns=AGENT_NETNS,
)
SUPERVISOR = CommandSupervisor(root=ROOT, execution_boundary=EXECUTION_BOUNDARY)
OWNER = OwnerFence()
# 412 is reserved for a superseded owner and 423 for a lifecycle lease held by
# the current owner, so the host never parses error bodies.
OWNER_SUPERSEDED_STATUS = 412
OWNER_LIFECYCLE_LEASED_STATUS = 423
OWNER_SUPERSEDED_REASON = "owner_superseded"
app = FastAPI(title="Cayu Lambda MicroVM sidecar", docs_url=None, redoc_url=None)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "protocol_version": PROTOCOL_VERSION}


@app.post("/aws/lambda-microvms/runtime/v1/ready")
async def ready_hook() -> dict[str, str]:
    """Tell the image builder the command sidecar is ready to snapshot."""
    return {"status": "ok"}


@app.post("/v1/owner")
async def claim_owner(payload: dict[str, Any]) -> dict[str, Any]:
    """Make the caller the only host allowed to start commands.

    A claim that supersedes another resets the agent proxy relay before the new
    owner can start anything, then cancels every command of earlier owners. The
    response is sent only after that cancellation, so a completed takeover
    leaves no earlier owner's command able to run.
    """
    try:
        generation, superseded = await asyncio.to_thread(
            OWNER.claim,
            payload.get("claim_id"),
            on_supersede=EXECUTION_BOUNDARY.close,
        )
    except CommandRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OwnerLifecycleLeasedError as exc:
        raise HTTPException(status_code=OWNER_LIFECYCLE_LEASED_STATUS, detail=str(exc)) from exc
    if superseded:
        await asyncio.to_thread(
            SUPERVISOR.cancel_all,
            reason=OWNER_SUPERSEDED_REASON,
            before_generation=generation,
        )
    return {"generation": generation, "superseded_previous": superseded}


@app.post("/v1/owner/check")
async def check_owner(payload: dict[str, Any]) -> dict[str, bool]:
    return {"current": OWNER.is_current(payload.get("claim_id"))}


@app.post("/v1/owner/lifecycle")
async def acquire_lifecycle(payload: dict[str, Any]) -> dict[str, Any]:
    """Confirm the caller still owns the MicroVM and block takeover until it acts.

    The host calls this immediately before the provider's suspend or terminate.
    The lease ends only when the guest's hook for the leased action runs or the
    host releases it after its attempt was refused or definitively rejected;
    it never expires by time.
    """
    try:
        return await asyncio.to_thread(
            OWNER.acquire_lifecycle, payload.get("claim_id"), payload.get("action")
        )
    except CommandRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OwnerSupersededError as exc:
        raise HTTPException(status_code=OWNER_SUPERSEDED_STATUS, detail=str(exc)) from exc


@app.post("/v1/owner/lifecycle/release")
async def release_lifecycle(payload: dict[str, Any]) -> dict[str, str]:
    try:
        await asyncio.to_thread(OWNER.release_lifecycle, payload.get("claim_id"))
    except CommandRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OwnerSupersededError as exc:
        raise HTTPException(status_code=OWNER_SUPERSEDED_STATUS, detail=str(exc)) from exc
    return {"status": "released"}


def _start_admitted(
    owner_claim: object, command_id: str, payload: dict[str, Any]
) -> dict[str, Any]:
    # Validation, which may install the agent proxy relay, and registration run
    # while the fence is held, so a takeover either precedes them (and this
    # start is refused) or follows them (and cancels this command).
    with OWNER.admitted(owner_claim) as generation:
        return SUPERVISOR.start(command_id, payload, owner_generation=generation)


@app.post("/v1/commands", status_code=202)
async def start_command(payload: dict[str, Any]) -> dict[str, Any]:
    command_id = payload.get("command_id")
    if not isinstance(command_id, str):
        raise HTTPException(status_code=400, detail="command_id must be a string")
    command_payload = dict(payload)
    command_payload.pop("command_id", None)
    owner_claim = command_payload.pop("owner_claim", None)
    try:
        return await asyncio.to_thread(_start_admitted, owner_claim, command_id, command_payload)
    except OwnerSupersededError as exc:
        raise HTTPException(status_code=OWNER_SUPERSEDED_STATUS, detail=str(exc)) from exc
    except OwnerLifecycleLeasedError as exc:
        raise HTTPException(status_code=OWNER_LIFECYCLE_LEASED_STATUS, detail=str(exc)) from exc
    except CommandConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except CommandRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/v1/commands/{command_id}")
async def get_command(command_id: str) -> dict[str, Any]:
    try:
        result = await asyncio.to_thread(SUPERVISOR.get, command_id)
    except CommandRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if result["state"] == "not_found":
        raise HTTPException(status_code=404, detail="command not found")
    return result


@app.delete("/v1/commands/{command_id}")
async def cancel_command(command_id: str) -> dict[str, Any]:
    try:
        return await asyncio.to_thread(SUPERVISOR.cancel, command_id)
    except CommandRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/aws/lambda-microvms/runtime/v1/run")
async def run_hook(payload: Any = None) -> dict[str, str]:
    del payload
    return {"status": "ok"}


@app.post("/aws/lambda-microvms/runtime/v1/resume")
async def resume_hook(payload: dict[str, Any] | None = None) -> dict[str, str]:
    del payload
    OWNER.settle_lifecycle("resume")
    return {"status": "ok"}


@app.post("/aws/lambda-microvms/runtime/v1/suspend")
async def suspend_hook(payload: dict[str, Any] | None = None) -> dict[str, str]:
    del payload
    await asyncio.to_thread(SUPERVISOR.cancel_all)
    # The provider applied the leased suspend, and the owner never retries it.
    OWNER.settle_lifecycle("suspend")
    return {"status": "ok"}


@app.post("/aws/lambda-microvms/runtime/v1/terminate")
async def terminate_hook(payload: dict[str, Any] | None = None) -> dict[str, str]:
    del payload
    await asyncio.to_thread(SUPERVISOR.cancel_all)
    OWNER.settle_lifecycle("terminate")
    return {"status": "ok"}
