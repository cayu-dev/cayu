"""Process supervisor used by the Cayu Lambda MicroVM command sidecar."""

from __future__ import annotations

import base64
import contextlib
import ipaddress
import json
import os
import signal
import socket
import subprocess
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

DEFAULT_OUTPUT_LIMIT_BYTES = 1024 * 1024
DEFAULT_CANCEL_TIMEOUT_SECONDS = 5.0
MAX_STDIN_BYTES = 1024 * 1024
READ_CHUNK_BYTES = 64 * 1024

_TERMINAL_STATES = frozenset({"completed", "cancelled", "failed"})
_PROXY_ENV_KEYS = ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy")
_AGENT_PROXY_RELAY_PORT = 18080

ExecutionProfile = Literal["agent", "trusted"]


class CommandRequestError(ValueError):
    """The command request is invalid."""


class CommandConflictError(RuntimeError):
    """A command ID was reused with a different payload."""


class _TcpRelay:
    """Root-namespace TCP relay exposed only on the agent veth gateway."""

    def __init__(self, host: str, port: int, *, listen_host: str = "192.0.2.1") -> None:
        self.target = (host, port)
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind((listen_host, _AGENT_PROXY_RELAY_PORT))
        self._listener.listen(32)
        self._listener.settimeout(0.2)
        self.proxy_url = f"http://{listen_host}:{self._listener.getsockname()[1]}"
        self._stop = threading.Event()
        self._connections: set[socket.socket] = set()
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._accept,
            name="cayu-agent-proxy-relay",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        with contextlib.suppress(OSError):
            self._listener.close()
        with self._lock:
            connections = tuple(self._connections)
        for connection in connections:
            with contextlib.suppress(OSError):
                connection.close()
        self._thread.join(timeout=1)

    def _accept(self) -> None:
        while not self._stop.is_set():
            try:
                client, _address = self._listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            threading.Thread(target=self._bridge, args=(client,), daemon=True).start()

    def _bridge(self, client: socket.socket) -> None:
        upstream: socket.socket | None = None
        try:
            upstream = socket.create_connection(self.target, timeout=5)
            upstream.settimeout(None)
            client.settimeout(None)
            with self._lock:
                self._connections.update((client, upstream))
            left = threading.Thread(target=_copy_socket, args=(client, upstream), daemon=True)
            right = threading.Thread(target=_copy_socket, args=(upstream, client), daemon=True)
            left.start()
            right.start()
            left.join()
            right.join()
        except OSError:
            return
        finally:
            with self._lock:
                self._connections.discard(client)
                if upstream is not None:
                    self._connections.discard(upstream)
            with contextlib.suppress(OSError):
                client.close()
            if upstream is not None:
                with contextlib.suppress(OSError):
                    upstream.close()


class CommandExecutionBoundary:
    """Separate ordinary agent processes from authenticated system commands."""

    def __init__(
        self,
        *,
        agent_uid: int | None = None,
        agent_gid: int | None = None,
        agent_netns: str | None = None,
        relay_factory: Any | None = None,
    ) -> None:
        if (agent_uid is None) != (agent_gid is None):
            raise ValueError("agent_uid and agent_gid must be configured together")
        for value, name in ((agent_uid, "agent_uid"), (agent_gid, "agent_gid")):
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer")
        self.agent_uid = agent_uid
        self.agent_gid = agent_gid
        if agent_netns is not None and (
            not agent_netns or not agent_netns.replace("-", "").isalnum()
        ):
            raise ValueError("agent_netns must contain only letters, digits, and hyphens")
        self.agent_netns = agent_netns
        self._relay_factory = relay_factory or _TcpRelay
        self._relay: Any | None = None
        self._relay_target: tuple[str, int] | None = None
        self._relay_lock = threading.Lock()

    def argv_for(
        self,
        argv: list[str],
        *,
        execution_profile: ExecutionProfile,
    ) -> list[str]:
        if execution_profile == "trusted":
            return list(argv)
        if self.agent_uid is None or self.agent_gid is None:
            return list(argv)
        prefix = (
            ["/usr/sbin/ip", "netns", "exec", self.agent_netns]
            if self.agent_netns is not None
            else []
        )
        return [
            *prefix,
            "/usr/bin/setpriv",
            f"--reuid={self.agent_uid}",
            f"--regid={self.agent_gid}",
            "--clear-groups",
            "--no-new-privs",
            "--inh-caps=-all",
            "--ambient-caps=-all",
            "--bounding-set=-all",
            "--",
            *argv,
        ]

    def environment_for(
        self,
        environment: dict[str, str],
        *,
        execution_profile: ExecutionProfile,
    ) -> dict[str, str]:
        copied = dict(environment)
        if execution_profile == "trusted" or self.agent_netns is None:
            return copied
        configured = [copied[key] for key in _PROXY_ENV_KEYS if key in copied]
        if not configured:
            return copied
        targets = {_private_http_proxy(value) for value in configured}
        if len(targets) != 1:
            raise CommandRequestError("agent proxy environment must name one private endpoint")
        target = next(iter(targets))
        with self._relay_lock:
            if self._relay is None:
                self._relay = self._relay_factory(*target)
                self._relay_target = target
            elif self._relay_target != target:
                raise CommandRequestError("agent proxy endpoint changed within one MicroVM")
            proxy_url = self._relay.proxy_url
        for key in _PROXY_ENV_KEYS:
            if key in copied:
                copied[key] = proxy_url
        return copied

    def close(self) -> None:
        with self._relay_lock:
            relay = self._relay
            self._relay = None
            self._relay_target = None
        if relay is not None:
            relay.close()


@dataclass
class _CommandRecord:
    command_id: str
    payload_fingerprint: str | None
    state: str = "accepted"
    process: subprocess.Popen[bytes] | None = None
    cancel_requested: bool = False
    result: dict[str, Any] | None = None
    finished_at: float | None = None
    finished: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)


class _LimitedBuffer:
    def __init__(self, limit: int | None) -> None:
        self.limit = limit
        self.content = bytearray()
        self.total_bytes = 0
        self.truncated = False

    def add(self, chunk: bytes) -> None:
        self.total_bytes += len(chunk)
        if self.limit is None:
            self.content.extend(chunk)
            return
        remaining = self.limit - len(self.content)
        if remaining > 0:
            self.content.extend(chunk[:remaining])
        if len(chunk) > remaining:
            self.truncated = True


class CommandSupervisor:
    """Own guest processes by command ID and expose idempotent lifecycle operations."""

    def __init__(
        self,
        *,
        root: str | Path = "/workspace",
        cancel_timeout_s: float = DEFAULT_CANCEL_TIMEOUT_SECONDS,
        result_ttl_s: float = 300.0,
        execution_boundary: CommandExecutionBoundary | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if type(cancel_timeout_s) not in {int, float} or cancel_timeout_s <= 0:
            raise ValueError("cancel_timeout_s must be greater than zero")
        self.cancel_timeout_s = float(cancel_timeout_s)
        if type(result_ttl_s) not in {int, float} or result_ttl_s <= 0:
            raise ValueError("result_ttl_s must be greater than zero")
        self.result_ttl_s = float(result_ttl_s)
        self.execution_boundary = execution_boundary or CommandExecutionBoundary()
        self._records: dict[str, _CommandRecord] = {}
        self._lock = threading.Lock()

    def start(self, command_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        identifier = _command_id(command_id)
        validated = _validated_payload(
            payload,
            root=self.root,
            execution_boundary=self.execution_boundary,
        )
        fingerprint = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        self._prune()
        with self._lock:
            existing = self._records.get(identifier)
            if existing is not None:
                if existing.payload_fingerprint is None:
                    existing.payload_fingerprint = fingerprint
                    return self._snapshot(existing)
                if existing.payload_fingerprint != fingerprint:
                    raise CommandConflictError(
                        f"Command id {identifier!r} was already used with another payload"
                    )
                return self._snapshot(existing)
            record = _CommandRecord(command_id=identifier, payload_fingerprint=fingerprint)
            self._records[identifier] = record
        threading.Thread(
            target=self._run,
            args=(record, validated),
            name=f"cayu-command-{identifier}",
            daemon=True,
        ).start()
        return {"command_id": identifier, "state": "accepted"}

    def get(self, command_id: str) -> dict[str, Any]:
        identifier = _command_id(command_id)
        self._prune()
        with self._lock:
            record = self._records.get(identifier)
        if record is None:
            return {"command_id": identifier, "state": "not_found"}
        return self._snapshot(record)

    def cancel(self, command_id: str) -> dict[str, Any]:
        identifier = _command_id(command_id)
        self._prune()
        with self._lock:
            record = self._records.get(identifier)
            if record is None:
                record = _CommandRecord(
                    command_id=identifier,
                    payload_fingerprint=None,
                    state="cancelled",
                    cancel_requested=True,
                    result=_cancelled_result(identifier),
                    finished_at=time.monotonic(),
                )
                record.finished.set()
                self._records[identifier] = record
                return self._snapshot(record)
        with record.lock:
            if record.state in _TERMINAL_STATES:
                return self._snapshot_locked(record)
            record.cancel_requested = True
            process = record.process
        if process is not None:
            _stop_process_group(process)
        record.finished.wait(timeout=self.cancel_timeout_s)
        return self._snapshot(record)

    def cancel_all(self) -> None:
        with self._lock:
            command_ids = list(self._records)
        for command_id in command_ids:
            self.cancel(command_id)
        self.execution_boundary.close()

    def _run(self, record: _CommandRecord, payload: dict[str, Any]) -> None:
        stdout = _LimitedBuffer(payload["output_limit_bytes"])
        stderr = _LimitedBuffer(payload["output_limit_bytes"])
        process: subprocess.Popen[bytes] | None = None
        timed_out = False
        try:
            argv = payload["argv"]
            process = subprocess.Popen(
                argv,
                cwd=payload["cwd"],
                env=payload["env"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            with record.lock:
                record.process = process
                record.state = "running"
                cancel_requested = record.cancel_requested
            if cancel_requested:
                _stop_process_group(process)

            readers = [
                threading.Thread(target=_drain, args=(process.stdout, stdout), daemon=True),
                threading.Thread(target=_drain, args=(process.stderr, stderr), daemon=True),
            ]
            for reader in readers:
                reader.start()
            writer = threading.Thread(
                target=_feed_stdin,
                args=(process.stdin, payload["stdin"]),
                daemon=True,
            )
            writer.start()
            try:
                process.wait(timeout=payload["timeout_s"])
            except subprocess.TimeoutExpired:
                timed_out = True
                _stop_process_group(process)
                process.wait(timeout=self.cancel_timeout_s)
            for reader in readers:
                reader.join(timeout=self.cancel_timeout_s)
            writer.join(timeout=self.cancel_timeout_s)
            with record.lock:
                cancelled = record.cancel_requested and not timed_out
                state = "cancelled" if cancelled else "completed"
                record.state = state
                record.result = _result(
                    record.command_id,
                    state=state,
                    exit_code=process.returncode if process.returncode is not None else -9,
                    stdout=stdout,
                    stderr=stderr,
                    timed_out=timed_out,
                    cancelled=cancelled,
                )
        except BaseException as exc:
            if process is not None and process.poll() is None:
                _stop_process_group(process)
            stderr.add(f"{type(exc).__name__}: {exc}\n".encode("utf-8", errors="replace"))
            with record.lock:
                cancelled = record.cancel_requested
                state = "cancelled" if cancelled else "failed"
                record.state = state
                record.result = _result(
                    record.command_id,
                    state=state,
                    exit_code=-1,
                    stdout=stdout,
                    stderr=stderr,
                    timed_out=timed_out,
                    cancelled=cancelled,
                    error=exc,
                )
        finally:
            with record.lock:
                record.finished_at = time.monotonic()
            record.finished.set()

    def _prune(self) -> None:
        cutoff = time.monotonic() - self.result_ttl_s
        with self._lock:
            expired: list[str] = []
            for command_id, record in self._records.items():
                with record.lock:
                    if record.finished_at is not None and record.finished_at <= cutoff:
                        expired.append(command_id)
            for command_id in expired:
                self._records.pop(command_id, None)

    def _snapshot(self, record: _CommandRecord) -> dict[str, Any]:
        with record.lock:
            return self._snapshot_locked(record)

    @staticmethod
    def _snapshot_locked(record: _CommandRecord) -> dict[str, Any]:
        if record.result is not None:
            return dict(record.result)
        return {"command_id": record.command_id, "state": record.state}


def _validated_payload(
    payload: object,
    *,
    root: Path,
    execution_boundary: CommandExecutionBoundary,
) -> dict[str, Any]:
    if type(payload) is not dict:
        raise CommandRequestError("command payload must be an object")
    request = cast("dict[str, Any]", payload)
    kind = request.get("kind")
    execution_profile = _validated_execution_profile(request.get("execution_profile", "agent"))
    if kind not in {"process", "shell"}:
        raise CommandRequestError("kind must be process or shell")
    if kind == "process":
        raw_argv = request.get("argv")
        if type(raw_argv) is not list or not raw_argv:
            raise CommandRequestError("process commands require non-empty argv")
        argv = [_nonblank_string(value, "argv") for value in raw_argv]
    else:
        shell = _nonblank_string(request.get("shell"), "shell")
        argv = ["/bin/bash", "-c", shell]
    argv = execution_boundary.argv_for(
        argv,
        execution_profile=execution_profile,
    )

    cwd = Path(_nonblank_string(request.get("cwd"), "cwd")).resolve()
    if not cwd.is_relative_to(root):
        raise CommandRequestError("cwd escapes the workspace root")
    if not cwd.is_dir():
        raise CommandRequestError("cwd does not exist or is not a directory")

    raw_env = request.get("env", {})
    if type(raw_env) is not dict:
        raise CommandRequestError("env must be an object")
    env: dict[str, str] = {}
    for key, value in raw_env.items():
        env[_nonblank_string(key, "env key")] = _string(value, "env value")
    env = execution_boundary.environment_for(env, execution_profile=execution_profile)

    raw_stdin = request.get("stdin_base64")
    if raw_stdin is None:
        stdin = b""
    elif type(raw_stdin) is str:
        try:
            stdin = base64.b64decode(raw_stdin, validate=True)
        except ValueError as exc:
            raise CommandRequestError("stdin_base64 must be valid base64") from exc
    else:
        raise CommandRequestError("stdin_base64 must be a string or null")
    if len(stdin) > MAX_STDIN_BYTES:
        raise CommandRequestError(f"stdin exceeds {MAX_STDIN_BYTES} bytes")

    timeout_s = request.get("timeout_s")
    if timeout_s is not None and (type(timeout_s) not in {int, float} or timeout_s <= 0):
        raise CommandRequestError("timeout_s must be null or greater than zero")
    output_limit = request.get("output_limit_bytes", DEFAULT_OUTPUT_LIMIT_BYTES)
    if output_limit is not None and (type(output_limit) is not int or output_limit <= 0):
        raise CommandRequestError("output_limit_bytes must be null or a positive integer")
    return {
        "kind": kind,
        "argv": argv,
        "cwd": str(cwd),
        "env": env,
        "stdin": stdin,
        "timeout_s": timeout_s,
        "output_limit_bytes": output_limit,
    }


def _private_http_proxy(value: str) -> tuple[str, int]:
    try:
        parsed = urllib.parse.urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise CommandRequestError("agent proxy URL is malformed") from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname is None
        or port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise CommandRequestError("agent proxy must be an unauthenticated HTTP origin")
    try:
        address = ipaddress.IPv4Address(parsed.hostname)
    except ipaddress.AddressValueError as exc:
        raise CommandRequestError("agent proxy host must be a private IPv4 literal") from exc
    address_value = int(address)
    if not (
        int(ipaddress.IPv4Address("10.0.0.0"))
        <= address_value
        <= int(ipaddress.IPv4Address("10.255.255.255"))
        or int(ipaddress.IPv4Address("172.16.0.0"))
        <= address_value
        <= int(ipaddress.IPv4Address("172.31.255.255"))
        or int(ipaddress.IPv4Address("192.168.0.0"))
        <= address_value
        <= int(ipaddress.IPv4Address("192.168.255.255"))
    ):
        raise CommandRequestError("agent proxy host must be a private IPv4 literal")
    return str(address), port


def _validated_execution_profile(value: object) -> ExecutionProfile:
    if value not in ("agent", "trusted"):
        raise CommandRequestError("execution_profile must be agent or trusted")
    return cast("ExecutionProfile", value)


def _copy_socket(source: socket.socket, destination: socket.socket) -> None:
    try:
        while True:
            chunk = source.recv(64 * 1024)
            if not chunk:
                return
            destination.sendall(chunk)
    except OSError:
        return
    finally:
        with contextlib.suppress(OSError):
            destination.shutdown(socket.SHUT_WR)


def _drain(pipe: Any, output: _LimitedBuffer) -> None:
    if pipe is None:
        return
    while True:
        chunk = pipe.read(READ_CHUNK_BYTES)
        if not chunk:
            return
        output.add(chunk)


def _feed_stdin(pipe: Any, content: bytes) -> None:
    if pipe is None:
        return
    try:
        pipe.write(content)
        pipe.close()
    except BrokenPipeError:
        pass


def _stop_process_group(process: subprocess.Popen[bytes]) -> None:
    process_group_id = process.pid
    if not _process_group_exists(process_group_id):
        return
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        if not _process_group_exists(process_group_id):
            return
        time.sleep(0.01)
    # macOS can report EPERM for a just-disappeared process group; a live group
    # created by this supervisor has our uid, so a real descendant remains
    # signalable here.
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process_group_id, signal.SIGKILL)


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _result(
    command_id: str,
    *,
    state: str,
    exit_code: int,
    stdout: _LimitedBuffer,
    stderr: _LimitedBuffer,
    timed_out: bool,
    cancelled: bool,
    error: BaseException | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "command_id": command_id,
        "state": state,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "cancelled": cancelled,
        "stdout_base64": base64.b64encode(stdout.content).decode("ascii"),
        "stderr_base64": base64.b64encode(stderr.content).decode("ascii"),
        "stdout_bytes": stdout.total_bytes,
        "stderr_bytes": stderr.total_bytes,
        "stdout_truncated": stdout.truncated,
        "stderr_truncated": stderr.truncated,
    }
    if error is not None:
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
    return result


def _cancelled_result(command_id: str) -> dict[str, Any]:
    return {
        "command_id": command_id,
        "state": "cancelled",
        "exit_code": -1,
        "timed_out": False,
        "cancelled": True,
        "stdout_base64": "",
        "stderr_base64": "",
        "stdout_bytes": 0,
        "stderr_bytes": 0,
        "stdout_truncated": False,
        "stderr_truncated": False,
    }


def _command_id(value: object) -> str:
    identifier = _nonblank_string(value, "command_id")
    if len(identifier.encode("utf-8")) > 256 or any(char in identifier for char in "/\\"):
        raise CommandRequestError("command_id is invalid")
    return identifier


def _nonblank_string(value: object, field_name: str) -> str:
    if type(value) is not str or not value.strip():
        raise CommandRequestError(f"{field_name} must be a non-empty string")
    return value


def _string(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise CommandRequestError(f"{field_name} must be a string")
    return value
