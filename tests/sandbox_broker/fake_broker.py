"""An in-memory Cayu Cloud sandbox broker (contract ``cayu.cloud.sandbox`` v1) for tests.

Each sandbox's ``/workspace`` is a host directory. Commands run there as real
processes with a scrubbed environment, so Runtime code runs unchanged against
the fake. Arguments naming ``/workspace`` are mapped to that directory, and the
directory in a command's output back to ``/workspace``, as the guest's mount would.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import signal
import tarfile
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from cayu.sandbox_broker import SandboxBrokerClient, SandboxBrokerSettings

TOKEN = "cayu_sbx_test_token"
APPLICATION = "compound"
API_URL = "https://cloud.example.test"
GUARANTEES_V1 = (
    "untrusted_code_isolation",
    "no_provider_credentials",
    "unprivileged_commands",
    "egress_only_via_broker",
)
ARCHIVE_LIMIT = 8 * 1024 * 1024


@dataclass
class FakeCommand:
    body: dict[str, Any]
    root: Path
    status: str = "running"
    exit_code: int | None = None
    stdout: bytes = b""
    stderr: bytes = b""
    process: asyncio.subprocess.Process | None = None
    task: asyncio.Task | None = None

    def view(self) -> dict[str, Any]:
        limit = self.body["output_limit_bytes"]
        stdout, stdout_truncated = _json_capped(self.stdout, limit)
        stderr, stderr_truncated = _json_capped(self.stderr, limit)
        return {
            "command_id": self.body["command_id"],
            "status": self.status,
            "exit_code": self.exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
            "started_at": "2026-10-18T12:00:00Z",
            "finished_at": None if self.status == "running" else "2026-10-18T12:00:01Z",
            "duration_ms": 1,
        }


@dataclass
class FakeSandbox:
    id: str
    body: dict[str, Any]
    root: Path
    guarantees: tuple[str, ...]
    status: str = "provisioning"
    polls_until_ready: int = 1
    lease_expires_at: datetime | None = None
    max_expires_at: datetime | None = None
    destroy_reason: str | None = None
    failure: dict[str, str] | None = None
    commands: dict[str, FakeCommand] = field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "application_id": APPLICATION,
            "image": self.body["image"],
            "egress": self.body["egress"],
            "status": self.status,
            "failure": self.failure,
            "labels": self.body.get("labels", {}),
            "provider": "aws-lambda-microvm",
            "created_at": "2026-10-18T12:00:00Z",
            "ready_at": None,
            "lease_expires_at": None
            if self.lease_expires_at is None
            else self.lease_expires_at.isoformat(),
            "max_expires_at": None
            if self.max_expires_at is None
            else self.max_expires_at.isoformat(),
            "destroyed_at": None,
            "destroy_reason": self.destroy_reason,
            "guarantees": {"version": 1, "items": list(self.guarantees)},
        }


class FakeBroker:
    def __init__(self, base: Path) -> None:
        self.base = base
        self.sandboxes: dict[str, FakeSandbox] = {}
        self.by_key: dict[str, str] = {}
        self.requests: list[httpx.Request] = []
        self.calls: list[tuple[str, str]] = []
        self.started: list[list[str]] = []
        self.renewals = 0
        self.refuse_create: tuple[int, str] | None = None
        self.fail_allocation = False
        self.fail_destroy = False
        self.busy_responses = 0
        self.drop_start_responses = 0
        self.unknown_errors = 0
        # Responses without a contract code, as authentication, a proxy or the ingress send.
        self.bare_statuses: list[int] = []
        self.cancel_confirms = True
        # Seconds a start takes to register, so a cancel can overtake it.
        self.start_delay = 0.0
        self.archive_limit = ARCHIVE_LIMIT
        self.archive_gets = 0
        self.guarantees = GUARANTEES_V1
        self.catalog = {"coding-python"}

    def client(self, **settings: Any) -> SandboxBrokerClient:
        return SandboxBrokerClient(
            SandboxBrokerSettings(
                api_url=API_URL, application_id=APPLICATION, token=TOKEN, **settings
            ),
            transport=httpx.MockTransport(self.transport),
        )

    def live(self) -> list[FakeSandbox]:
        return [sandbox for sandbox in self.sandboxes.values() if sandbox.status != "destroyed"]

    def running(self) -> list[FakeCommand]:
        return [
            command
            for sandbox in self.sandboxes.values()
            for command in sandbox.commands.values()
            if command.status == "running"
        ]

    def expire(self, sandbox_id: str) -> None:
        sandbox = self.sandboxes[sandbox_id]
        sandbox.status = "destroyed"
        sandbox.destroy_reason = "lease_expired"

    async def transport(self, request: httpx.Request) -> httpx.Response:
        if self.unknown_errors:
            self.unknown_errors -= 1
            return httpx.Response(502, text="<html>bad gateway</html>")
        if self.bare_statuses:
            self.requests.append(request)
            return httpx.Response(self.bare_statuses.pop(0), text="")
        response = await self.handle(request)
        if (
            self.drop_start_responses
            and request.method == "POST"
            and request.url.path.endswith("/commands")
        ):
            # The broker applied the start, but the answer was lost.
            self.drop_start_responses -= 1
            raise httpx.ReadError("connection reset", request=request)
        return response

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.headers.get("authorization") != f"Bearer {TOKEN}":
            return httpx.Response(401, json={"error": {"code": "x", "message": "", "detail": {}}})
        path = request.url.path
        prefix = f"/v1/applications/{APPLICATION}/sandboxes"
        self.calls.append((request.method, path.removeprefix(prefix) or "/"))
        if path == "/v1/sandbox-images":
            return httpx.Response(
                200,
                json={
                    "images": [
                        {
                            "name": "coding-python",
                            "guarantees": {"version": 1, "items": list(GUARANTEES_V1)},
                        }
                    ]
                },
            )
        if not path.startswith(prefix):
            return httpx.Response(404, json={"error": {"code": "x", "message": "", "detail": {}}})
        parts = path.removeprefix(prefix).strip("/").split("/") if path != prefix else []
        if not parts or parts == [""]:
            if request.method == "POST":
                return self._create(json.loads(request.content))
            items = [sandbox.record() for sandbox in self.sandboxes.values()]
            return httpx.Response(200, json={"items": items, "next_cursor": None})
        sandbox = self.sandboxes.get(parts[0])
        if sandbox is None:
            return httpx.Response(404, json={"error": {"code": "x", "message": "", "detail": {}}})
        action = parts[1:]
        if request.method == "GET" and not action:
            if sandbox.status == "provisioning":
                sandbox.polls_until_ready -= 1
                if sandbox.polls_until_ready <= 0:
                    if self.fail_allocation:
                        sandbox.status = "failed"
                        sandbox.failure = {"code": "allocation_failed", "message": "no capacity"}
                    else:
                        now = datetime.now(UTC)
                        sandbox.status = "ready"
                        sandbox.lease_expires_at = now + timedelta(
                            seconds=sandbox.body["lease_seconds"]
                        )
                        sandbox.max_expires_at = now + timedelta(hours=8)
            return httpx.Response(200, json=sandbox.record())
        if request.method == "DELETE" and not action:
            if self.fail_destroy:
                return _error(503, "internal")
            if sandbox.status != "destroyed":
                for command in sandbox.commands.values():
                    await self._kill(command, "failed")
                sandbox.status = "destroyed"
                sandbox.destroy_reason = "requested"
            return httpx.Response(202, json=sandbox.record())
        if action == ["events"]:
            return httpx.Response(200, json={"items": []})
        if action[:1] == ["commands"] and len(action) >= 2 and request.method == "GET":
            command = sandbox.commands.get(action[1])
            if command is None:
                return httpx.Response(
                    404, json={"error": {"code": "x", "message": "", "detail": {}}}
                )
            wait = int(request.url.params.get("wait_seconds", "0"))
            if command.task is not None and wait:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(asyncio.shield(command.task), wait)
            return httpx.Response(200, json=command.view())
        if sandbox.status != "ready":
            return _error(409, "sandbox_not_ready")
        if action == ["renew"]:
            seconds = json.loads(request.content)["lease_seconds"]
            sandbox.lease_expires_at = datetime.now(UTC) + timedelta(seconds=seconds)
            self.renewals += 1
            return httpx.Response(200, json=sandbox.record())
        if action == ["commands"] and request.method == "POST":
            return await self._start(sandbox, json.loads(request.content))
        if action[:1] == ["commands"] and action[2:] == ["cancel"]:
            command = sandbox.commands.get(action[1])
            if command is None:
                return httpx.Response(
                    404, json={"error": {"code": "x", "message": "", "detail": {}}}
                )
            if self.cancel_confirms:
                await self._kill(command, "cancelled")
            return httpx.Response(202, json=command.view())
        if action == ["archive"] and request.method == "PUT":
            if len(request.content) > self.archive_limit:
                return _error(413, "archive_too_large", {"bytes": len(request.content)})
            target = sandbox.root / request.url.params.get("path", ".")
            with tarfile.open(fileobj=io.BytesIO(request.content), mode="r:") as archive:
                members = archive.getmembers()
                archive.extractall(target, filter="data")
            return httpx.Response(200, json={"files": len(members), "bytes": len(request.content)})
        if action == ["archive"] and request.method == "GET":
            self.archive_gets += 1
            return self._read_archive(sandbox, request)
        return httpx.Response(404, json={"error": {"code": "x", "message": "", "detail": {}}})

    def _create(self, body: dict[str, Any]) -> httpx.Response:
        if self.refuse_create is not None:
            status, code = self.refuse_create
            detail = {"scope": "agent", "limit": 8} if code == "sandbox_quota_exceeded" else {}
            return _error(status, code, detail)
        if body["image"] not in self.catalog:
            return _error(422, "image_unknown")
        key = body["idempotency_key"]
        if key in self.by_key:
            existing = self.sandboxes[self.by_key[key]]
            if existing.body != body:
                return _error(409, "sandbox_idempotency_conflict")
            return httpx.Response(200, json=existing.record())
        sandbox_id = f"sbx_{len(self.sandboxes) + 1:04d}"
        root = self.base / sandbox_id
        root.mkdir(parents=True)
        sandbox = FakeSandbox(id=sandbox_id, body=body, root=root, guarantees=self.guarantees)
        self.sandboxes[sandbox_id] = sandbox
        self.by_key[key] = sandbox_id
        return httpx.Response(202, json=sandbox.record())

    async def _start(self, sandbox: FakeSandbox, body: dict[str, Any]) -> httpx.Response:
        if self.start_delay:
            await asyncio.sleep(self.start_delay)
        existing = sandbox.commands.get(body["command_id"])
        if existing is not None:
            if existing.body != body:
                return _error(409, "command_id_conflict")
            return httpx.Response(200, json=existing.view())
        sizes = [len(item.encode()) for item in body["argv"]]
        if max(sizes) > 131_071 or sum(sizes) > 256 * 1024:
            return _error(422, "sandbox_invalid")
        if self.busy_responses:
            self.busy_responses -= 1
            return _error(409, "sandbox_busy")
        self.started.append(body["argv"])
        argv = [
            str(sandbox.root) + item.removeprefix("/workspace")
            if item == "/workspace" or item.startswith("/workspace/")
            else item
            for item in body["argv"]
        ]
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(sandbox.root)}
        env.update(body.get("env") or {})
        command = FakeCommand(body=body, root=sandbox.root)
        command.process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=sandbox.root / body.get("cwd", "."),
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        command.task = asyncio.create_task(self._wait(command))
        sandbox.commands[body["command_id"]] = command
        return httpx.Response(202, json=command.view())

    async def _wait(self, command: FakeCommand) -> None:
        process = command.process
        assert process is not None
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate((command.body.get("stdin") or "").encode()),
                command.body["timeout_seconds"],
            )
        except TimeoutError:
            await self._kill(command, "timed_out")
            return
        if command.status == "running":
            command.stdout = _as_guest(stdout, command.root)
            command.stderr = _as_guest(stderr, command.root)
            command.status = "exited"
            command.exit_code = process.returncode

    async def _kill(self, command: FakeCommand, status: str) -> None:
        if command.status != "running":
            return
        command.status = status
        process = command.process
        if process is not None and process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()

    def _read_archive(self, sandbox: FakeSandbox, request: httpx.Request) -> httpx.Response:
        paths = request.url.params.get_list("path") or ["."]
        excludes = set(request.url.params.get_list("exclude"))
        buffer = io.BytesIO()
        missing: list[str] = []
        with tarfile.open(fileobj=buffer, mode="w:") as archive:
            for path in paths:
                source = sandbox.root / path
                if not source.exists():
                    missing.append(path)
                    continue
                archive.add(
                    source,
                    arcname=path,
                    filter=lambda info: None if set(Path(info.name).parts) & excludes else info,
                )
        data = buffer.getvalue()
        if len(data) > self.archive_limit:
            return _error(413, "archive_too_large", {"bytes": len(data)})
        headers = {"Content-Type": "application/x-tar"}
        if missing:
            headers["X-Cayu-Missing-Paths"] = ",".join(quote(item, safe="") for item in missing)
        return httpx.Response(200, content=data, headers=headers)


def _as_guest(output: bytes, root: Path) -> bytes:
    """Name the sandbox's directory as the guest does, ``/workspace``."""

    # The longer name first: one may end with the other (/private/var and /var).
    for host in sorted({str(root.resolve()), str(root)}, key=len, reverse=True):
        output = output.replace(host.encode(), b"/workspace")
    return output


def _json_capped(raw: bytes, limit: int) -> tuple[str, bool]:
    """Cap a stream at ``limit`` bytes as it is encoded in the response JSON."""

    text = raw.decode("utf-8", "replace")

    def encoded(value: str) -> int:
        return len(json.dumps(value, ensure_ascii=False).encode()) - 2

    if encoded(text) <= limit:
        return text, False
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if encoded(text[:middle]) <= limit:
            low = middle
        else:
            high = middle - 1
    return text[:low], True


def _error(status: int, code: str, detail: dict[str, Any] | None = None) -> httpx.Response:
    return httpx.Response(
        status, json={"error": {"code": code, "message": code, "detail": detail or {}}}
    )
