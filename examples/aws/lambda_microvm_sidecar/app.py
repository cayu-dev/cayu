"""HTTP interface for the Cayu Lambda MicroVM command sidecar."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from fastapi import FastAPI, HTTPException

from .supervisor import CommandConflictError, CommandRequestError, CommandSupervisor

ROOT = os.environ.get("CAYU_MICROVM_WORKSPACE_ROOT", "/workspace")
PROTOCOL_VERSION = "1"
SUPERVISOR = CommandSupervisor(root=ROOT)
app = FastAPI(title="Cayu Lambda MicroVM sidecar", docs_url=None, redoc_url=None)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "protocol_version": PROTOCOL_VERSION}


@app.post("/aws/lambda-microvms/runtime/v1/ready")
async def ready_hook() -> dict[str, str]:
    """Tell the image builder the command sidecar is ready to snapshot."""
    return {"status": "ok"}


@app.post("/v1/commands", status_code=202)
async def start_command(payload: dict[str, Any]) -> dict[str, Any]:
    command_id = payload.get("command_id")
    if not isinstance(command_id, str):
        raise HTTPException(status_code=400, detail="command_id must be a string")
    command_payload = dict(payload)
    command_payload.pop("command_id", None)
    try:
        return await asyncio.to_thread(SUPERVISOR.start, command_id, command_payload)
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
async def run_hook(payload: dict[str, Any] | None = None) -> dict[str, str]:
    del payload
    return {"status": "ok"}


@app.post("/aws/lambda-microvms/runtime/v1/resume")
async def resume_hook(payload: dict[str, Any] | None = None) -> dict[str, str]:
    del payload
    return {"status": "ok"}


@app.post("/aws/lambda-microvms/runtime/v1/suspend")
async def suspend_hook(payload: dict[str, Any] | None = None) -> dict[str, str]:
    del payload
    await asyncio.to_thread(SUPERVISOR.cancel_all)
    return {"status": "ok"}


@app.post("/aws/lambda-microvms/runtime/v1/terminate")
async def terminate_hook(payload: dict[str, Any] | None = None) -> dict[str, str]:
    del payload
    await asyncio.to_thread(SUPERVISOR.cancel_all)
    return {"status": "ok"}
