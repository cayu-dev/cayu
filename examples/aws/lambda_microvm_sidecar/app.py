"""HTTP interface for the Cayu Lambda MicroVM command sidecar."""

from __future__ import annotations

import asyncio
import functools
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import FastAPI, HTTPException

from .forwards import FORWARD_PORTS, MAX_FORWARD_CONNECTIONS, PortForwardRegistry
from .supervisor import (
    CommandConflictError,
    CommandExecutionBoundary,
    CommandNotFoundError,
    CommandRequestError,
    CommandSupervisor,
    OwnerFence,
    OwnerLifecycleLeasedError,
    OwnerSupersededError,
)

ROOT = os.environ.get("CAYU_MICROVM_WORKSPACE_ROOT", "/workspace")
PROTOCOL_VERSION = "7"
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
FORWARDS = PortForwardRegistry()
# 412 is reserved for a superseded owner and 423 for a lifecycle lease held by
# the current owner, so the host never parses error bodies.
OWNER_SUPERSEDED_STATUS = 412
OWNER_LIFECYCLE_LEASED_STATUS = 423
OWNER_SUPERSEDED_REASON = "owner_superseded"
# Interactive stdio blocks on pipes and long polls. Its own pool keeps that
# waiting from starving the default pool that serves every other endpoint.
INTERACTIVE_IO = ThreadPoolExecutor(max_workers=32, thread_name_prefix="cayu-interactive")
# Each relayed connection keeps one read long poll open, and each forward one
# accept long poll, so forwards get their own pools too. The poll pool fits every
# poll the forward limits allow. Writes and connection closes have a separate pool
# sized to the connection limit, since a write can block while a guest stops
# reading, so host bytes never queue behind a long poll. Threads start on demand.
FORWARD_POLL_IO = ThreadPoolExecutor(
    max_workers=len(FORWARD_PORTS) * (MAX_FORWARD_CONNECTIONS + 1),
    thread_name_prefix="cayu-forward-poll",
)
FORWARD_IO = ThreadPoolExecutor(
    max_workers=len(FORWARD_PORTS) * MAX_FORWARD_CONNECTIONS,
    thread_name_prefix="cayu-forward",
)
app = FastAPI(title="Cayu Lambda MicroVM sidecar", docs_url=None, redoc_url=None)


def _close_guest_relays() -> None:
    EXECUTION_BOUNDARY.close()
    FORWARDS.close_all()


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

    A claim that supersedes another resets the agent proxy relay and closes
    every port forward before the new owner can start anything, then cancels
    every command of earlier owners. The
    response is sent only after that cancellation, so a completed takeover
    leaves no earlier owner's command able to run.
    """
    try:
        generation, superseded = await asyncio.to_thread(
            OWNER.claim,
            payload.get("claim_id"),
            on_supersede=_close_guest_relays,
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


def _configure_control_relay_admitted(
    owner_claim: object, host: object, port: object
) -> dict[str, Any]:
    # Like command admission, the relay is configured while the fence is held,
    # so a takeover (which closes the execution boundary) either precedes this
    # and refuses it, or follows it and clears the relay.
    with OWNER.admitted(owner_claim):
        return EXECUTION_BOUNDARY.configure_control_relay(host, port)


@app.post("/v1/control-relay")
async def configure_control_relay(payload: dict[str, Any]) -> dict[str, Any]:
    """Relay the agent namespace's ``cayu-control`` name to one private control server.

    Only the current owner may configure it. The relay forwards TCP only; the
    guest verifies the control server's TLS certificate end to end.
    """
    try:
        return await asyncio.to_thread(
            _configure_control_relay_admitted,
            payload.get("owner_claim"),
            payload.get("host"),
            payload.get("port"),
        )
    except OwnerSupersededError as exc:
        raise HTTPException(status_code=OWNER_SUPERSEDED_STATUS, detail=str(exc)) from exc
    except OwnerLifecycleLeasedError as exc:
        raise HTTPException(status_code=OWNER_LIFECYCLE_LEASED_STATUS, detail=str(exc)) from exc
    except CommandConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except CommandRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _open_forward_admitted(payload: dict[str, Any]) -> dict[str, Any]:
    # Opened while the fence is held, like a command start, so a takeover
    # either refuses this open or closes the forward it created.
    with OWNER.admitted(payload.get("owner_claim")) as generation:
        return FORWARDS.open(
            payload.get("forward_id"),
            owner_generation=generation,
            timeout_s=payload.get("timeout_s"),
        )


@app.post("/v1/forwards")
async def open_forward(payload: dict[str, Any]) -> dict[str, Any]:
    """Listen on the agent veth gateway and hold its connections for the owner to relay.

    The owner long-polls for accepted connections and moves their bytes; the
    guest never gets a route to the host.
    """
    return await _forward_call(lambda: _open_forward_admitted(payload), executor=None)


@app.post("/v1/forwards/{forward_id}/accept")
async def accept_forward(forward_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Long-poll for connections accepted after the ``after`` connection number."""
    return await _owned_forward_call(
        FORWARD_POLL_IO,
        payload,
        FORWARDS.accept,
        forward_id,
        after=payload.get("after"),
        wait_s=payload.get("wait_s", 0),
    )


@app.post("/v1/forwards/{forward_id}/connections/{connection}/read")
async def read_forward(forward_id: str, connection: int, payload: dict[str, Any]) -> dict[str, Any]:
    """Long-poll a connection's guest bytes from ``offset``, acknowledging earlier ones."""
    return await _owned_forward_call(
        FORWARD_POLL_IO,
        payload,
        FORWARDS.read,
        forward_id,
        connection,
        offset=payload.get("offset"),
        max_bytes=payload.get("max_bytes"),
        wait_s=payload.get("wait_s", 0),
    )


@app.post("/v1/forwards/{forward_id}/connections/{connection}/write")
async def write_forward(
    forward_id: str, connection: int, payload: dict[str, Any]
) -> dict[str, Any]:
    """Send host bytes to a connection; ``offset`` makes a retry idempotent."""
    return await _owned_forward_call(
        FORWARD_IO,
        payload,
        FORWARDS.write,
        forward_id,
        connection,
        offset=payload.get("offset"),
        data_base64=payload.get("data_base64", ""),
        eof=payload.get("eof", False),
    )


@app.post("/v1/forwards/{forward_id}/connections/{connection}/close")
async def close_forward_connection(
    forward_id: str, connection: int, payload: dict[str, Any]
) -> dict[str, Any]:
    return await _owned_forward_call(
        FORWARD_IO, payload, FORWARDS.close_connection, forward_id, connection
    )


@app.post("/v1/forwards/{forward_id}/close")
async def close_forward(forward_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Close a forward and its connections; an unknown ID leaves a tombstone."""

    def close() -> dict[str, Any]:
        OWNER.require(payload.get("owner_claim"))
        return FORWARDS.close(forward_id)

    return await _forward_call(close, executor=None)


async def _owned_forward_call(
    executor: ThreadPoolExecutor,
    payload: dict[str, Any],
    operation: Any,
    *arguments: Any,
    **keywords: Any,
) -> dict[str, Any]:
    # The fence is checked, not held: a takeover closes every forward itself.
    def call() -> dict[str, Any]:
        generation = OWNER.require(payload.get("owner_claim"))
        return operation(*arguments, owner_generation=generation, **keywords)

    return await _forward_call(call, executor=executor)


async def _forward_call(call: Any, *, executor: ThreadPoolExecutor | None) -> dict[str, Any]:
    try:
        return await asyncio.get_running_loop().run_in_executor(executor, call)
    except OwnerSupersededError as exc:
        raise HTTPException(status_code=OWNER_SUPERSEDED_STATUS, detail=str(exc)) from exc
    except OwnerLifecycleLeasedError as exc:
        raise HTTPException(status_code=OWNER_LIFECYCLE_LEASED_STATUS, detail=str(exc)) from exc
    except CommandNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
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


@app.post("/v1/commands/{command_id}/stdin")
async def write_command_stdin(command_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Write stdin of an interactive command; ``offset`` makes a retry idempotent."""
    return await _interactive_call(
        payload,
        SUPERVISOR.write_stdin,
        command_id,
        offset=payload.get("offset"),
        data_base64=payload.get("data_base64", ""),
        eof=payload.get("eof", False),
    )


@app.post("/v1/commands/{command_id}/stdout")
async def read_command_stdout(command_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Long-poll stdout of an interactive command from ``offset``.

    Reading at an offset acknowledges every earlier byte, so the sidecar holds
    only output the host has not yet confirmed.
    """
    return await _interactive_call(
        payload,
        SUPERVISOR.read_stdout,
        command_id,
        offset=payload.get("offset"),
        max_bytes=payload.get("max_bytes"),
        wait_s=payload.get("wait_s", 0),
    )


async def _interactive_call(
    payload: dict[str, Any], operation: Any, command_id: str, **arguments: Any
) -> dict[str, Any]:
    # Only the current owner moves a command's stdio. The fence is checked, not
    # held: a takeover cancels the earlier owner's commands itself.
    try:
        await asyncio.to_thread(OWNER.require, payload.get("owner_claim"))
        return await asyncio.get_running_loop().run_in_executor(
            INTERACTIVE_IO, functools.partial(operation, command_id, **arguments)
        )
    except OwnerSupersededError as exc:
        raise HTTPException(status_code=OWNER_SUPERSEDED_STATUS, detail=str(exc)) from exc
    except OwnerLifecycleLeasedError as exc:
        raise HTTPException(status_code=OWNER_LIFECYCLE_LEASED_STATUS, detail=str(exc)) from exc
    except CommandNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except CommandRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/commands/{command_id}/release")
async def release_command(command_id: str) -> dict[str, Any]:
    try:
        result = await asyncio.to_thread(SUPERVISOR.release, command_id)
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
    try:
        await asyncio.to_thread(SUPERVISOR.cancel_all)
        await asyncio.to_thread(FORWARDS.close_all)
    finally:
        # The provider applied the leased suspend even if cleanup failed.
        # Retained forward custody is settled separately by its next close.
        OWNER.settle_lifecycle("suspend")
    return {"status": "ok"}


@app.post("/aws/lambda-microvms/runtime/v1/terminate")
async def terminate_hook(payload: dict[str, Any] | None = None) -> dict[str, str]:
    del payload
    try:
        await asyncio.to_thread(SUPERVISOR.cancel_all)
        await asyncio.to_thread(FORWARDS.close_all)
    finally:
        OWNER.settle_lifecycle("terminate")
    return {"status": "ok"}
