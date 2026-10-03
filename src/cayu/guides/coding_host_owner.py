"""Host-only Docker lifetime observations; never infer death from absence."""

import asyncio
import hashlib
import json
import os
import re
import socket
import subprocess
from datetime import datetime
from uuid import uuid4

from cayu.guides.coding_host_lifetime import wait_owned_task

_HEX = re.compile(r"[0-9a-f]{64}")
_START = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.]+Z")
_TASK_TYPES = frozenset(
    {
        "maintenance.coding",
        "maintenance.git_preparation",
        "maintenance.git_delivery",
        "maintenance.github_delivery",
    }
)


class WorkerOwnerUnavailable(ValueError):
    def __init__(self):
        super().__init__("Original maintenance worker termination is not established.")


def _require_task_type(task_type):
    if type(task_type) is not str or task_type not in _TASK_TYPES:
        raise WorkerOwnerUnavailable()
    return task_type


def _inspect(target, task_type="maintenance.coding"):
    task_type = _require_task_type(task_type)
    # Select the deployment's local daemon explicitly. Do not inherit a remote
    # Docker context, credential helper, or caller-supplied inspect expression.
    try:
        result = subprocess.run(
            [
                "/usr/local/bin/docker",
                "--host",
                "unix:///var/run/docker.sock",
                "container",
                "inspect",
                "--format",
                '{"id":{{json .Id}},"started":{{json .State.StartedAt}},'
                '"status":{{json .State.Status}},"running":{{json .State.Running}},'
                '"finished":{{json .State.FinishedAt}},"command":{{json .Config.Cmd}}}',
                target,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=True,
            env={"PATH": "/usr/local/bin:/usr/bin:/bin"},
        )
        if len(result.stdout) > 4096:
            raise ValueError
        value = json.loads(result.stdout)
        if (
            type(value) is not dict
            or set(value) != {"id", "started", "status", "running", "finished", "command"}
            or type(value["id"]) is not str
            or not _HEX.fullmatch(value["id"])
            or not value["id"].startswith(target)
            or type(value["started"]) is not str
            or not _START.fullmatch(value["started"])
            or value["started"].startswith("0001-")
            or type(value["finished"]) is not str
            or not _START.fullmatch(value["finished"])
            or type(value["running"]) is not bool
            or value["status"] not in {"running", "paused", "exited"}
            or value["command"]
            != [
                "cayu",
                "worker",
                task_type.removeprefix("maintenance."),
                "--shutdown-grace-seconds",
                "30",
            ]
        ):
            raise ValueError
        datetime.fromisoformat(value["started"])
        datetime.fromisoformat(value["finished"])
        return value
    except (OSError, ValueError, TypeError, subprocess.SubprocessError):
        raise WorkerOwnerUnavailable() from None


async def _observe(target, task_type="maintenance.coding"):
    # The bounded read-only client is killed/reaped by subprocess.run on timeout.
    # Retain ownership even when the HTTP/worker caller is cancelled.
    return await wait_owned_task(
        asyncio.create_task(asyncio.to_thread(_inspect, target, task_type))
    )


def _generation(value):
    return hashlib.sha256(value["started"].encode("ascii")).hexdigest()


async def maintenance_worker_id(task_type):
    task_type = _require_task_type(task_type)
    mode = os.environ.get("CAYU_MAINTENANCE_WORKER_OWNER")
    if mode is None:
        return task_type + "-" + uuid4().hex
    if mode != "docker":
        raise WorkerOwnerUnavailable()
    hostname = socket.gethostname()
    if re.fullmatch(r"[0-9a-f]{12}", hostname) is None:
        raise WorkerOwnerUnavailable()
    observed = await _observe(hostname, task_type)
    if observed["status"] != "running" or observed["running"] is not True:
        raise WorkerOwnerUnavailable()
    return task_type + ":docker:" + observed["id"] + ":" + _generation(observed)


async def inspect_stopped_worker(worker_id, *, task_type="maintenance.coding"):
    task_type = _require_task_type(task_type)
    prefix = task_type + ":docker:"
    if type(worker_id) is not str or not worker_id.startswith(prefix):
        raise WorkerOwnerUnavailable()
    parts = worker_id.removeprefix(prefix).split(":")
    if len(parts) != 2 or any(_HEX.fullmatch(part) is None for part in parts):
        raise WorkerOwnerUnavailable()
    container, generation = parts
    observed = await _observe(container, task_type)
    if _generation(observed) == generation and (
        observed["status"] != "exited"
        or observed["running"] is not False
        or observed["finished"].startswith("0001-")
    ):
        raise WorkerOwnerUnavailable()
    # Docker reuses a container ID across restart, but not a StartedAt generation.
    # A changed generation establishes termination of the recorded lifetime, not
    # quiescence of its external effects (which require separate Runtime proof).
    return {"worker_id": worker_id, "container_id": container, "generation": generation}
