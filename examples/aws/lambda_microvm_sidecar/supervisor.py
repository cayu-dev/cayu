"""Process supervisor used by the Cayu Lambda MicroVM command sidecar."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import hmac
import ipaddress
import json
import os
import signal
import socket
import subprocess
import threading
import time
import urllib.parse
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

DEFAULT_OUTPUT_LIMIT_BYTES = 1024 * 1024
# Sized for the built-in browser worker: a default session response envelope
# plus a browser-profile checkpoint, and a default upload batch on stdin.
MAX_OUTPUT_LIMIT_BYTES = 32 * 1024 * 1024
DEFAULT_CANCEL_TIMEOUT_SECONDS = 5.0
MAX_STDIN_BYTES = 24 * 1024 * 1024
READ_CHUNK_BYTES = 64 * 1024

_TERMINAL_STATES = frozenset({"completed", "cancelled", "failed", "released"})
_PROXY_ENV_KEYS = ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy")
_AGENT_PROXY_RELAY_PORT = 18080
LIFECYCLE_ACTIONS = frozenset({"suspend", "terminate"})
# A per-session token the relay presents to the Cayu proxy before any agent
# bytes, so a listener reachable from other MicroVMs on the shared connector is
# usable only through this MicroVM's root relay. Agent commands never see it.
TRANSPORT_TOKEN_ENV = "CAYU_EGRESS_PROXY_TRANSPORT_TOKEN"
_TRANSPORT_TUNNEL_TARGET = "cayu-transport.invalid:443"
_MAX_TRANSPORT_RESPONSE_BYTES = 4096
# The only route from the agent namespace to the Cayu control server. The
# trusted host configures one private target; the agent namespace resolves
# AGENT_CONTROL_HOSTNAME to the gateway, and TLS stays end to end.
AGENT_CONTROL_RELAY_PORT = 18443
AGENT_CONTROL_HOSTNAME = "cayu-control"
_AGENT_GATEWAY = "192.0.2.1"
DEFAULT_COMMAND_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

ExecutionProfile = Literal["agent", "trusted"]


class CommandRequestError(ValueError):
    """The command request is invalid."""


class OwnerSupersededError(RuntimeError):
    """A command was submitted by an owner that a later claim has fenced."""


class OwnerLifecycleLeasedError(RuntimeError):
    """The current owner holds a lifecycle lease; no claim or command is admitted."""


@dataclass(frozen=True)
class _LifecycleLease:
    action: str
    generation: int


class OwnerFence:
    """Single current host owner of this MicroVM's command execution.

    Each claim is a random host-held secret; only its digest is kept here. A
    new claim supersedes every earlier one, and the caller must cancel the
    earlier owner's commands. Re-presenting the current claim is idempotent.

    Command admission and lifecycle leases hold the fence's lock while they
    check the owner, so neither can interleave with a takeover. A lifecycle
    lease lets the current owner call the provider's suspend or terminate
    without a successor claiming the MicroVM between the check and the call.

    A lease never expires by time: an accepted provider request is not proven
    finished by any elapsed interval. It ends only on authoritative evidence:
    the owner releasing it after its single attempt was refused or definitively
    rejected, or the guest hook for the leased action (``settle_lifecycle``).
    Until then no claim or command is admitted, so a host that dies while
    holding a lease blocks takeover of this MicroVM until it ends (at the
    latest at its maximum duration) or the allocation reap terminates it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._generation = 0
        self._digest: bytes | None = None
        self._lease: _LifecycleLease | None = None

    def claim(
        self,
        claim_id: object,
        *,
        on_supersede: Callable[[], None] | None = None,
    ) -> tuple[int, bool]:
        """Return the current generation and whether this claim superseded another.

        ``on_supersede`` runs under the fence's lock after the swap, before any
        command of the new owner can be admitted.
        """

        digest = _claim_digest(claim_id)
        with self._lock:
            if self._digest is not None and hmac.compare_digest(self._digest, digest):
                return self._generation, False
            if self._lease is not None:
                raise OwnerLifecycleLeasedError(
                    "the current MicroVM owner is suspending or terminating it"
                )
            superseded = self._digest is not None
            self._generation += 1
            self._digest = digest
            if superseded and on_supersede is not None:
                on_supersede()
            return self._generation, superseded

    def is_current(self, claim_id: object) -> bool:
        try:
            digest = _claim_digest(claim_id)
        except CommandRequestError:
            return False
        with self._lock:
            return self._is_current_locked(digest)

    @contextlib.contextmanager
    def admitted(self, claim_id: object) -> Iterator[int]:
        """Hold the fence while the caller registers work for the current owner.

        Yields the owner's generation. A takeover waits until the block exits,
        so work registered here is visible to the takeover's cancellation.
        """

        try:
            digest: bytes | None = _claim_digest(claim_id)
        except CommandRequestError:
            digest = None
        with self._lock:
            if digest is None or not self._is_current_locked(digest):
                raise OwnerSupersededError("command owner is not the current MicroVM owner")
            if self._lease is not None:
                raise OwnerLifecycleLeasedError("the MicroVM owner is suspending or terminating it")
            yield self._generation

    def require(self, claim_id: object) -> int:
        with self.admitted(claim_id) as generation:
            return generation

    def acquire_lifecycle(self, claim_id: object, action: object) -> dict[str, Any]:
        """Atomically confirm the owner and block takeover until the lease ends."""

        if action not in LIFECYCLE_ACTIONS:
            raise CommandRequestError("lifecycle action must be suspend or terminate")
        digest = _claim_digest(claim_id)
        with self._lock:
            if not self._is_current_locked(digest):
                raise OwnerSupersededError("lifecycle caller is not the current MicroVM owner")
            self._lease = _LifecycleLease(action=str(action), generation=self._generation)
            return {"generation": self._generation, "action": action}

    def release_lifecycle(self, claim_id: object) -> None:
        digest = _claim_digest(claim_id)
        with self._lock:
            if not self._is_current_locked(digest):
                raise OwnerSupersededError("lifecycle caller is not the current MicroVM owner")
            self._lease = None

    def settle_lifecycle(self, hook: str) -> None:
        """Called by the guest's own lifecycle hooks: the provider has acted.

        A terminate hook settles any lease, since nothing survives it. A
        suspend or resume hook settles only a suspend lease: either proves the
        MicroVM went through suspension, and the leased attempt, which the host
        never retries, is the only suspend a fenced host can send while the
        lease is held. (A MicroVM idle policy suspends on its own; the runtime
        contract describes that limit.) A terminate lease stays held until its
        own hook.
        """

        with self._lock:
            lease = self._lease
            if lease is None:
                return
            if hook == "terminate" or (hook in {"suspend", "resume"} and lease.action == "suspend"):
                self._lease = None

    def _is_current_locked(self, digest: bytes) -> bool:
        return self._digest is not None and hmac.compare_digest(self._digest, digest)


def _claim_digest(claim_id: object) -> bytes:
    if type(claim_id) is not str or not 32 <= len(claim_id) <= 128 or not claim_id.isascii():
        raise CommandRequestError("owner claim must be 32 to 128 ASCII characters")
    return hashlib.sha256(claim_id.encode("ascii")).digest()


class CommandConflictError(RuntimeError):
    """A command ID was reused with a different payload."""


class _TcpRelay:
    """Root-namespace TCP relay exposed only on the agent veth gateway."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        transport_token: bytes | None = None,
        listen_host: str = _AGENT_GATEWAY,
        listen_port: int | None = None,
    ) -> None:
        self.target = (host, port)
        self._transport_request = (
            None
            if transport_token is None
            else (
                f"CONNECT {_TRANSPORT_TUNNEL_TARGET} HTTP/1.1\r\n"
                f"Host: {_TRANSPORT_TUNNEL_TARGET}\r\n"
                "Proxy-Authorization: Basic "
                f"{base64.b64encode(b'cayu:' + transport_token).decode('ascii')}\r\n\r\n"
            ).encode("ascii")
        )
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(
            (listen_host, _AGENT_PROXY_RELAY_PORT if listen_port is None else listen_port)
        )
        self._listener.listen(32)
        self._listener.settimeout(0.2)
        self.proxy_url = f"http://{listen_host}:{self._listener.getsockname()[1]}"
        self._stop = threading.Event()
        self._connections: set[socket.socket] = set()
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._accept,
            name=f"cayu-agent-relay-{self._listener.getsockname()[1]}",
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
            if self._transport_request is not None and not _open_transport_tunnel(
                upstream, self._transport_request
            ):
                return
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


def _open_transport_tunnel(upstream: socket.socket, request: bytes) -> bool:
    """Authenticate this connection to the Cayu proxy before relaying agent bytes.

    The proxy answers with a bare status head and then waits, so reading up to
    the blank line never consumes tunneled data.
    """
    upstream.sendall(request)
    head = bytearray()
    while not head.endswith(b"\r\n\r\n"):
        if len(head) >= _MAX_TRANSPORT_RESPONSE_BYTES:
            return False
        chunk = upstream.recv(1)
        if not chunk:
            return False
        head.extend(chunk)
    status = head.split(b"\r\n", 1)[0].split(b" ", 2)
    return len(status) >= 2 and status[1] == b"200"


class CommandExecutionBoundary:
    """Separate ordinary agent processes from authenticated system commands."""

    def __init__(
        self,
        *,
        agent_uid: int | None = None,
        agent_gid: int | None = None,
        agent_netns: str | None = None,
        relay_factory: Any | None = None,
        control_relay_factory: Any | None = None,
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
        self._relay_target: tuple[tuple[str, int], bytes | None] | None = None
        self._relay_lock = threading.Lock()
        self._control_relay_factory = control_relay_factory or _control_relay
        self._control_relay: Any | None = None
        self._control_target: tuple[str, int] | None = None

    def configure_control_relay(self, host: object, port: object) -> dict[str, Any]:
        """Relay the agent namespace's control hostname to one trusted private target.

        Only the owner-fenced host calls this. The target is fixed for the
        MicroVM's current owner: repeating it is idempotent, changing it is a
        conflict, and a superseding owner claim or suspension clears it.
        """
        if self.agent_netns is None:
            raise CommandRequestError("control relay requires the agent network namespace")
        target = (_private_ipv4(host, what="control relay target"), _tcp_port(port))
        with self._relay_lock:
            if self._control_relay is None:
                self._control_relay = self._control_relay_factory(*target)
                self._control_target = target
            elif self._control_target != target:
                raise CommandConflictError("control relay target changed within one MicroVM")
        return {
            "hostname": AGENT_CONTROL_HOSTNAME,
            "port": AGENT_CONTROL_RELAY_PORT,
            "target": f"{target[0]}:{target[1]}",
        }

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
        if execution_profile == "trusted":
            return copied
        raw_token = copied.pop(TRANSPORT_TOKEN_ENV, None)
        if self.agent_netns is None:
            return copied
        configured = [copied[key] for key in _PROXY_ENV_KEYS if key in copied]
        if not configured:
            return copied
        targets = {_private_http_proxy(value) for value in configured}
        if len(targets) != 1:
            raise CommandRequestError("agent proxy environment must name one private endpoint")
        target = next(iter(targets))
        token = None if raw_token is None else _transport_token(raw_token)
        relay_key = (target, token)
        with self._relay_lock:
            if self._relay is None:
                self._relay = (
                    self._relay_factory(*target)
                    if token is None
                    else self._relay_factory(*target, transport_token=token)
                )
                self._relay_target = relay_key
            elif self._relay_target != relay_key:
                raise CommandRequestError("agent proxy endpoint changed within one MicroVM")
            proxy_url = self._relay.proxy_url
        for key in _PROXY_ENV_KEYS:
            if key in copied:
                copied[key] = proxy_url
        return copied

    def close(self) -> None:
        with self._relay_lock:
            relays = (self._relay, self._control_relay)
            self._relay = None
            self._relay_target = None
            self._control_relay = None
            self._control_target = None
        for relay in relays:
            if relay is not None:
                relay.close()


def _control_relay(host: str, port: int) -> _TcpRelay:
    return _TcpRelay(host, port, listen_port=AGENT_CONTROL_RELAY_PORT)


@dataclass
class _CommandRecord:
    command_id: str
    payload_fingerprint: str | None
    owner_generation: int | None = None
    state: str = "accepted"
    process: subprocess.Popen[bytes] | None = None
    cancel_requested: bool = False
    cancel_reason: str | None = None
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

    def start(
        self,
        command_id: str,
        payload: dict[str, Any],
        *,
        owner_generation: int | None = None,
    ) -> dict[str, Any]:
        """Register and start one command.

        ``owner_generation`` records which owner admitted it, so a takeover
        cancels exactly the earlier owners' commands. The sidecar calls this
        inside ``OwnerFence.admitted`` so registration cannot straddle a claim.
        """

        identifier = _command_id(command_id)
        validated = _validated_payload(
            payload,
            root=self.root,
            execution_boundary=self.execution_boundary,
        )
        # Only a digest is retained for idempotent replays: the payload can carry
        # stdin and environment values that must not outlive the command.
        fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
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
            record = _CommandRecord(
                command_id=identifier,
                payload_fingerprint=fingerprint,
                owner_generation=owner_generation,
            )
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

    def release(self, command_id: str) -> dict[str, Any]:
        """Drop a delivered terminal result so its output is not retained.

        The host releases a command after it has read the terminal result. The
        record keeps only its payload digest, so a replayed start stays
        idempotent while stdout and stderr are gone. Releasing a running or
        unknown command changes nothing.
        """
        identifier = _command_id(command_id)
        self._prune()
        with self._lock:
            record = self._records.get(identifier)
        if record is None:
            return {"command_id": identifier, "state": "not_found"}
        with record.lock:
            if record.state in _TERMINAL_STATES:
                record.state = "released"
                record.result = None
            return self._snapshot_locked(record)

    def cancel(self, command_id: str, *, reason: str | None = None) -> dict[str, Any]:
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
            record.cancel_reason = reason
            process = record.process
        if process is not None:
            _stop_process_group(process)
        record.finished.wait(timeout=self.cancel_timeout_s)
        return self._snapshot(record)

    def cancel_all(
        self,
        *,
        reason: str | None = None,
        before_generation: int | None = None,
    ) -> None:
        """Cancel commands; ``reason`` is reported on commands it stops.

        With ``before_generation`` only commands admitted by earlier owners are
        cancelled, so a takeover never stops a newer owner's work, and the
        execution boundary is left to the takeover. Without it every command is
        cancelled and the agent proxy relay is closed.
        """
        with self._lock:
            command_ids = [
                command_id
                for command_id, record in self._records.items()
                if before_generation is None
                or (
                    record.owner_generation is not None
                    and record.owner_generation < before_generation
                )
            ]
        for command_id in command_ids:
            self.cancel(command_id, reason=reason)
        if before_generation is None:
            self.execution_boundary.close()

    def _run(self, record: _CommandRecord, payload: dict[str, Any]) -> None:
        stdout = _LimitedBuffer(payload["output_limit_bytes"])
        stderr = _LimitedBuffer(payload["output_limit_bytes"])
        process: subprocess.Popen[bytes] | None = None
        readers: list[tuple[threading.Thread, _LimitedBuffer]] = []
        timed_out = False
        try:
            argv = payload["argv"]
            # Spawning under the record lock orders it with cancel(): a command
            # cancelled before this point never starts, and one cancelled after
            # it has a process for cancel() to stop.
            with record.lock:
                if record.cancel_requested:
                    record.state = "cancelled"
                    record.result = {
                        **_cancelled_result(record.command_id),
                        **(
                            {"cancel_reason": record.cancel_reason}
                            if record.cancel_reason is not None
                            else {}
                        ),
                    }
                    return
                process = subprocess.Popen(
                    argv,
                    cwd=payload["cwd"],
                    env=payload["env"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                )
                record.process = process
                record.state = "running"

            pending_readers = [
                (
                    threading.Thread(target=_drain, args=(process.stdout, stdout), daemon=True),
                    stdout,
                ),
                (
                    threading.Thread(target=_drain, args=(process.stderr, stderr), daemon=True),
                    stderr,
                ),
            ]
            for reader, output in pending_readers:
                reader.start()
                # Only a successfully started thread may be joined during
                # failure cleanup; Thread.join() rejects unstarted threads.
                readers.append((reader, output))
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
            _join_output_readers(readers, timeout_s=self.cancel_timeout_s)
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
                    omit_truncated_output=payload["omit_truncated_output"],
                    cancel_reason=record.cancel_reason if cancelled else None,
                )
        except BaseException as exc:
            if process is not None and process.poll() is None:
                _stop_process_group(process)
            _join_output_readers(readers, timeout_s=self.cancel_timeout_s)
            if not payload["omit_truncated_output"]:
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
                    omit_truncated_output=payload["omit_truncated_output"],
                    error=exc,
                    cancel_reason=record.cancel_reason if cancelled else None,
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
    if "PATH" not in env:
        # Resolve programs exactly as the guest shell does. Without this the
        # launcher falls back to /bin:/usr/bin while /bin/sh (used by admission
        # probes and shell commands) also searches /usr/local, so a probe could
        # admit a program that a process command then cannot start.
        env["PATH"] = DEFAULT_COMMAND_PATH

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
    output_limit = min(
        MAX_OUTPUT_LIMIT_BYTES,
        MAX_OUTPUT_LIMIT_BYTES if output_limit is None else output_limit,
    )
    return {
        "kind": kind,
        "argv": argv,
        "cwd": str(cwd),
        "env": env,
        "stdin": stdin,
        "timeout_s": timeout_s,
        "output_limit_bytes": output_limit,
        "omit_truncated_output": _boolean(
            request.get("omit_truncated_output", False),
            "omit_truncated_output",
        ),
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
    return _private_ipv4(parsed.hostname, what="agent proxy host"), port


_PRIVATE_IPV4_NETWORKS = tuple(
    ipaddress.IPv4Network(network) for network in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
)


def _private_ipv4(value: object, *, what: str) -> str:
    if type(value) is not str:
        raise CommandRequestError(f"{what} must be a private IPv4 literal")
    try:
        address = ipaddress.IPv4Address(value)
    except ipaddress.AddressValueError as exc:
        raise CommandRequestError(f"{what} must be a private IPv4 literal") from exc
    if not any(address in network for network in _PRIVATE_IPV4_NETWORKS):
        raise CommandRequestError(f"{what} must be a private IPv4 literal")
    return str(address)


def _tcp_port(value: object) -> int:
    if type(value) is not int or not 1 <= value <= 65535:
        raise CommandRequestError("control relay port must be an integer from 1 to 65535")
    return value


def _transport_token(value: str) -> bytes:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise CommandRequestError("agent proxy transport token is malformed")
    return value.encode("ascii")


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
    try:
        while True:
            chunk = pipe.read(READ_CHUNK_BYTES)
            if not chunk:
                return
            output.add(chunk)
    except BaseException:
        # Thread failures cannot propagate back through ``Thread.join``. Mark
        # the observed bytes as an incomplete prefix so negotiated redacted
        # executions omit them instead of treating thread termination as EOF.
        output.truncated = True


def _join_output_readers(
    readers: list[tuple[threading.Thread, _LimitedBuffer]],
    *,
    timeout_s: float,
) -> None:
    for reader, output in readers:
        reader.join(timeout=timeout_s)
        if reader.is_alive():
            # A descendant can retain the pipe after the command's direct
            # process exits. The bytes observed so far are only a prefix, even
            # when they fit within the configured buffer.
            output.truncated = True


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
    omit_truncated_output: bool,
    error: BaseException | None = None,
    cancel_reason: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "command_id": command_id,
        "state": state,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "cancelled": cancelled,
        "stdout_base64": (
            ""
            if omit_truncated_output and stdout.truncated
            else base64.b64encode(stdout.content).decode("ascii")
        ),
        "stderr_base64": (
            ""
            if omit_truncated_output and stderr.truncated
            else base64.b64encode(stderr.content).decode("ascii")
        ),
        "stdout_bytes": stdout.total_bytes,
        "stderr_bytes": stderr.total_bytes,
        "stdout_truncated": stdout.truncated,
        "stderr_truncated": stderr.truncated,
    }
    if error is not None:
        result["error_type"] = type(error).__name__
    if cancel_reason is not None:
        result["cancel_reason"] = cancel_reason
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


def _boolean(value: object, field_name: str) -> bool:
    if type(value) is not bool:
        raise CommandRequestError(f"{field_name} must be a boolean")
    return value
