"""An in-memory Cayu Cloud sandbox broker (contract ``cayu.cloud.sandbox`` v1) for tests.

Each sandbox's ``/workspace`` is a host directory. Commands run there as real
processes with a scrubbed environment, so Runtime code runs unchanged against
the fake. Arguments naming ``/workspace`` are mapped to that directory, and the
directory in a command's output back to ``/workspace``, as the guest's mount would.
``guest_paths`` maps further guest path prefixes (an image's ``/opt/...`` toolchain)
to host directories the same way, and ``image_identities`` gives a catalog image
the identity a broker reports for an image it built.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
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
STORAGE_URL = "https://storage.example.test"


# Names the broker owns in a command's environment; it refuses them.
CLOUD_ENVIRONMENT_NAMES = frozenset({"PATH", "HOME", "HTTPS_PROXY", "https_proxy"})


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
    # An interactive command's streams: stdin written so far, and stdout from stdout_start.
    stdin_offset: int = 0
    stdin_closed: bool = False
    stdout_start: int = 0
    stdout_eof: bool = False
    stdout_changed: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def interactive(self) -> bool:
        return self.body.get("interactive") is True

    def view(self) -> dict[str, Any]:
        limit = self.body.get("output_limit_bytes", 1024 * 1024)
        stdout, stdout_truncated = _json_capped(b"" if self.interactive else self.stdout, limit)
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
            "interactive": self.interactive,
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
    copies: dict[str, dict[str, Any]] = field(default_factory=dict)
    staged_copies: dict[str, int] | None = None
    interactive_commands: bool = True
    image_identity: dict[str, Any] | None = None

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
            "staged_copies": self.staged_copies,
            "interactive_commands": self.interactive_commands,
            "guarantees": {"version": 1, "items": list(self.guarantees)},
            **({} if self.image_identity is None else {"image_identity": self.image_identity}),
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
        # What new sandboxes offer for staged copies; None offers none.
        self.staged_copies: dict[str, int] | None = None
        # Whether new sandboxes run interactive commands; Cloud before them refuses the field.
        self.interactive_commands = True
        # Staged objects by key, and every request the storage received.
        self.objects: dict[str, bytes] = {}
        self.storage_requests: list[httpx.Request] = []
        # Stdio requests whose answer is lost after the broker applied them.
        self.drop_stdio_responses = 0
        # Guest path prefixes mapped to host directories, beyond /workspace.
        self.guest_paths: dict[str, Path] = {}
        # The image identity reported for sandboxes of each image name.
        self.image_identities: dict[str, dict[str, Any]] = {}

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
        if self.drop_stdio_responses and request.url.path.endswith(("/stdin", "/stdout")):
            self.drop_stdio_responses -= 1
            raise httpx.ReadError("connection reset", request=request)
        return response

    async def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == "storage.example.test":
            return self._storage(request)
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
            if action[2:] == ["stdout"]:
                return await self._read_stdout(command, request)
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
        if action[:1] == ["commands"] and action[2:] == ["stdin"]:
            command = sandbox.commands.get(action[1])
            if command is None:
                return httpx.Response(
                    404, json={"error": {"code": "x", "message": "", "detail": {}}}
                )
            return await self._write_stdin(command, json.loads(request.content))
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
        if action[:1] == ["copies"]:
            return self._copy(sandbox, request, action[1:])
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
        sandbox = FakeSandbox(
            id=sandbox_id,
            body=body,
            root=root,
            guarantees=self.guarantees,
            staged_copies=self.staged_copies,
            interactive_commands=self.interactive_commands,
            image_identity=self.image_identities.get(body["image"]),
        )
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
        if "interactive" in body and not sandbox.interactive_commands:
            # A broker without interactive commands rejects the unknown field.
            return _error(422, "sandbox_invalid")
        sizes = [len(item.encode()) for item in body["argv"]]
        if max(sizes) > 131_071 or sum(sizes) > 256 * 1024:
            return _error(422, "sandbox_invalid")
        if CLOUD_ENVIRONMENT_NAMES & set(body.get("env") or {}):
            return _error(422, "sandbox_invalid")
        if body.get("interactive") and ("stdin" in body or "output_limit_bytes" in body):
            return _error(422, "sandbox_invalid")
        if self.busy_responses:
            self.busy_responses -= 1
            return _error(409, "sandbox_busy")
        self.started.append(body["argv"])
        argv = [self._host_path(sandbox, item) for item in body["argv"]]
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
        command.task = asyncio.create_task(
            self._relay(command) if command.interactive else self._wait(command)
        )
        sandbox.commands[body["command_id"]] = command
        return httpx.Response(202, json=command.view())

    def _host_path(self, sandbox: FakeSandbox, value: str) -> str:
        prefixes = {"/workspace": sandbox.root, **self.guest_paths}
        for prefix in sorted(prefixes, key=len, reverse=True):
            if value == prefix or value.startswith(prefix + "/"):
                return str(prefixes[prefix]) + value.removeprefix(prefix)
        return value

    async def _relay(self, command: FakeCommand) -> None:
        """Hold an interactive command's stdout until it is read; it exits after its output."""

        process = command.process
        assert process is not None and process.stdout is not None
        try:
            async with asyncio.timeout(command.body["timeout_seconds"]):
                while chunk := await process.stdout.read(65536):
                    command.stdout += chunk
                    command.stdout_changed.set()
                exit_code = await process.wait()
        except TimeoutError:
            await self._kill(command, "timed_out")
            return
        command.stdout_eof = True
        command.stdout_changed.set()
        if command.status == "running":
            command.status = "exited"
            command.exit_code = exit_code

    async def _write_stdin(self, command: FakeCommand, body: dict[str, Any]) -> httpx.Response:
        if not command.interactive:
            return _error(409, "command_not_interactive")
        offset = body["offset"]
        if offset > command.stdin_offset:
            return _error(409, "stdin_offset_mismatch", {"offset": command.stdin_offset})
        if command.stdin_closed or command.status != "running":
            return httpx.Response(200, json={"offset": command.stdin_offset, "closed": True})
        process = command.process
        assert process is not None and process.stdin is not None
        pending = base64.b64decode(body["data"])[command.stdin_offset - offset :]
        try:
            if pending:
                process.stdin.write(pending)
                await process.stdin.drain()
            if body.get("eof"):
                process.stdin.close()
        except (BrokenPipeError, ConnectionResetError):
            command.stdin_closed = True
            return httpx.Response(200, json={"offset": command.stdin_offset, "closed": True})
        command.stdin_offset += len(pending)
        command.stdin_closed = bool(body.get("eof"))
        return httpx.Response(200, json={"offset": command.stdin_offset, "closed": False})

    async def _read_stdout(self, command: FakeCommand, request: httpx.Request) -> httpx.Response:
        if not command.interactive:
            return _error(409, "command_not_interactive")
        offset = int(request.url.params["offset"])
        wait = int(request.url.params.get("wait_seconds", "0"))
        max_bytes = int(request.url.params.get("max_bytes", str(1024 * 1024)))
        if command.status not in {"running", "exited"}:
            # Cancelled, timed out or lost: nothing more will be delivered.
            return httpx.Response(200, json={"offset": offset, "data": "", "eof": True})
        end = command.stdout_start + len(command.stdout)
        if not command.stdout_start <= offset <= end:
            return _error(
                409, "stdout_offset_mismatch", {"start": command.stdout_start, "end": end}
            )
        command.stdout = command.stdout[offset - command.stdout_start :]
        command.stdout_start = offset
        if not command.stdout and not command.stdout_eof and wait:
            command.stdout_changed.clear()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(command.stdout_changed.wait(), wait)
        data = command.stdout[:max_bytes]
        eof = command.stdout_eof and len(data) == len(command.stdout)
        return httpx.Response(
            200,
            json={"offset": offset, "data": base64.b64encode(data).decode(), "eof": eof},
        )

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

    def _storage(self, request: httpx.Request) -> httpx.Response:
        self.storage_requests.append(request)
        key = request.url.path.lstrip("/")
        if request.method == "PUT":
            if request.headers.get("content-length") != str(len(request.content)):
                return httpx.Response(400)
            self.objects[key] = request.content
            return httpx.Response(200)
        if key not in self.objects:
            return httpx.Response(404)
        return httpx.Response(200, content=self.objects[key])

    def _copy(
        self, sandbox: FakeSandbox, request: httpx.Request, action: list[str]
    ) -> httpx.Response:
        if sandbox.staged_copies is None:
            return _error(503, "copies_unavailable")
        if not action and request.method == "POST":
            body = json.loads(request.content)
            existing = sandbox.copies.get(body["copy_id"])
            if existing is not None:
                return httpx.Response(200, json=_copy_view(existing))
            key = f"{sandbox.id}/{body['copy_id']}.tar"
            copy = {**body, "key": key, "status": "awaiting_upload", "missing": []}
            if body["direction"] == "in" and body["bytes"] > sandbox.staged_copies["max_bytes"]:
                return _error(413, "copy_too_large", {"bytes": body["bytes"]})
            sandbox.copies[body["copy_id"]] = copy
            if body["direction"] == "out":
                self._copy_out(sandbox, copy)
            return httpx.Response(202, json=_copy_view(copy))
        copy = sandbox.copies.get(action[0]) if action else None
        if copy is None:
            return httpx.Response(404, json={"error": {"code": "x", "message": "", "detail": {}}})
        if action[1:] == ["commit"] and copy["status"] == "awaiting_upload":
            self._copy_in(sandbox, copy)
        return httpx.Response(200, json=_copy_view(copy))

    def _copy_in(self, sandbox: FakeSandbox, copy: dict[str, Any]) -> None:
        data = self.objects.pop(copy["key"], None)
        if data is None:
            copy.update(status="failed", error={"code": "copy_upload_missing", "message": ""})
            return
        if len(data) != copy["bytes"] or hashlib.sha256(data).hexdigest() != copy["sha256"]:
            copy.update(status="failed", error={"code": "copy_digest_mismatch", "message": ""})
            return
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
            members = archive.getmembers()
            archive.extractall(sandbox.root / copy.get("path", "."), filter="data")
        copy.update(status="done", files=len(members))

    def _copy_out(self, sandbox: FakeSandbox, copy: dict[str, Any]) -> None:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:") as archive:
            for path in copy["paths"] or ["."]:
                source = sandbox.root / path
                if not source.exists():
                    copy["missing"].append(path)
                    continue
                archive.add(source, arcname=path)
        data = buffer.getvalue()
        copy["bytes"] = len(data)
        assert sandbox.staged_copies is not None
        if len(data) > sandbox.staged_copies["max_bytes"]:
            copy.update(status="failed", error={"code": "copy_too_large", "message": ""})
            return
        self.objects[copy["key"]] = data
        copy.update(status="done", sha256=hashlib.sha256(data).hexdigest())


def _copy_view(copy: dict[str, Any]) -> dict[str, Any]:
    signed = {"url": f"{STORAGE_URL}/{copy['key']}", "expires_at": "2026-10-18T12:15:00Z"}
    view = {
        "copy_id": copy["copy_id"],
        "direction": copy["direction"],
        "status": copy["status"],
        "bytes": copy.get("bytes"),
        "sha256": copy.get("sha256"),
        "missing": copy["missing"],
        "files": copy.get("files"),
        "error": copy.get("error"),
        "upload": None,
        "download": None,
    }
    if copy["status"] == "awaiting_upload":
        view["upload"] = {
            **signed,
            "method": "PUT",
            "headers": {"Content-Type": "application/x-tar", "Content-Length": str(copy["bytes"])},
        }
    if copy["status"] == "done" and copy["direction"] == "out":
        view["download"] = {**signed, "method": "GET", "headers": {}}
    return view


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
