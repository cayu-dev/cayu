"""Owned process boundaries for stdio MCP transports."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import math
import os
import platform
import secrets
import signal
import socket
import sys
from contextlib import suppress
from enum import StrEnum
from pathlib import Path
from weakref import WeakKeyDictionary

from cayu._validation import require_clean_nonblank
from cayu.capabilities import CapabilityClaim, CapabilityEvidence, CapabilityProofSource
from cayu.mcp._jsonrpc import McpProtocolError, validate_positive_number
from cayu.mcp._stdio_containment import (
    _LINUX_F_ADD_SEALS,
    _LINUX_MFD_ALLOW_SEALING,
    _LINUX_MFD_CLOEXEC,
    _LINUX_REQUIRED_MEMFD_SEALS,
    _linux_pidfd_signaling_supported,
)

DEFAULT_MCP_CONTAINMENT_STARTUP_TIMEOUT_S = 5.0
DEFAULT_MCP_CONTAINMENT_TERM_TIMEOUT_S = 2.0
DEFAULT_MCP_CONTAINMENT_KILL_TIMEOUT_S = 2.0
_MAX_CONTAINMENT_CONTROL_BYTES = 65_536
_MAX_SERVER_ENV_BYTES = 16 * 1024 * 1024
_LINUX_CONTAINMENT_ARCHITECTURES = frozenset({"aarch64", "arm64", "x86_64", "amd64"})
_PARENT_DEATH_CONTAINMENT_PREFLIGHT_SUCCEEDED: bool | None = None
_PARENT_DEATH_CONTAINMENT_PREFLIGHT_PROCESS_ID: int | None = None
_PARENT_DEATH_CONTAINMENT_PREFLIGHT_AUTHORITY = object()
_CONTAINMENT_RENDEZVOUS_AUTHORITY = object()


class _ContainmentPreflightProof:
    """Private authority that one exact owner process completed preflight."""

    __slots__ = ("_authority", "owner_pid")

    def __init__(self, *, owner_pid: int, authority: object) -> None:
        self.owner_pid = owner_pid
        self._authority = authority


class _PreparedContainmentRendezvous:
    """Private single-use authority for one pre-secret rendezvous identity."""

    __slots__ = ("_active", "_authority", "identity", "owner_pid")

    def __init__(
        self,
        *,
        identity: str,
        owner_pid: int,
        authority: object,
    ) -> None:
        self._active = True
        self.identity = identity
        self.owner_pid = owner_pid
        self._authority = authority

    def consume(self) -> str:
        if (
            self._authority is not _CONTAINMENT_RENDEZVOUS_AUTHORITY
            or self.owner_pid != os.getpid()
            or not self._active
        ):
            raise RuntimeError("Stdio MCP containment rendezvous authority was invalid or stale.")
        self._active = False
        return self.identity

    def close(self) -> None:
        self._active = False


def _current_parent_death_containment_preflight_result() -> bool | None:
    if os.getpid() != _PARENT_DEATH_CONTAINMENT_PREFLIGHT_PROCESS_ID:
        return None
    return _PARENT_DEATH_CONTAINMENT_PREFLIGHT_SUCCEEDED


def _validate_containment_preflight_proof(proof: _ContainmentPreflightProof) -> None:
    if (
        type(proof) is not _ContainmentPreflightProof
        or proof._authority is not _PARENT_DEATH_CONTAINMENT_PREFLIGHT_AUTHORITY
        or proof.owner_pid != os.getpid()
    ):
        raise RuntimeError("Stdio MCP containment preflight authority was invalid or stale.")


def stdio_mcp_parent_death_containment_platform_candidate() -> bool:
    """Return whether this host can attempt the exact containment preflight."""

    return (
        os.name == "posix"
        and sys.platform == "linux"
        and platform.machine().lower() in _LINUX_CONTAINMENT_ARCHITECTURES
        and all(hasattr(os, name) for name in ("waitid", "WNOWAIT"))
        and hasattr(os, "memfd_create")
        and hasattr(socket, "AF_UNIX")
        and _linux_pidfd_signaling_supported()
    )


def stdio_mcp_parent_death_containment_supported() -> bool:
    """Return whether this process has proved the complete-tree contract."""

    return (
        stdio_mcp_parent_death_containment_platform_candidate()
        and _current_parent_death_containment_preflight_result() is True
    )


class StdioMcpProcessLifetime(StrEnum):
    """Ownership policy for a local stdio MCP subprocess."""

    PARENT_DEATH_CONTAINMENT = "parent_death_containment"
    GRACEFUL_CLEANUP = "graceful_cleanup"
    PERSISTENT_DETACHED = "persistent_detached"


_DIRECT_PROCESS_LIFETIMES: WeakKeyDictionary[
    asyncio.subprocess.Process, StdioMcpProcessLifetime
] = WeakKeyDictionary()
_RETAINED_CONTAINMENT_STARTUP_CLEANUPS: set[asyncio.Task[int]] = set()


def _move_fd_above_stdio(fd: int) -> int:
    import fcntl

    if fd > 2:
        return fd
    duplicate = fcntl.fcntl(fd, fcntl.F_DUPFD_CLOEXEC, 3)
    os.close(fd)
    return duplicate


def _move_socket_above_stdio(value: socket.socket) -> socket.socket:
    import fcntl

    if value.fileno() > 2:
        return value
    duplicate = fcntl.fcntl(value.fileno(), fcntl.F_DUPFD_CLOEXEC, 3)
    value.close()
    try:
        return socket.socket(fileno=duplicate)
    except BaseException:
        os.close(duplicate)
        raise


def _command_containment_rendezvous_identity(command: tuple[str, ...]) -> str:
    encoded = json.dumps(
        {
            "command": command,
            "schema": "cayu.mcp.stdio_containment_rendezvous.v1",
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_containment_rendezvous_identity(value: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("MCP stdio containment rendezvous identity was invalid.")
    return value


def _prepare_stdio_mcp_containment_rendezvous(
    identity: str,
) -> _PreparedContainmentRendezvous:
    """Own one exact, pre-secret identity for the trusted supervisor bind."""

    return _PreparedContainmentRendezvous(
        identity=_validate_containment_rendezvous_identity(identity),
        owner_pid=os.getpid(),
        authority=_CONTAINMENT_RENDEZVOUS_AUTHORITY,
    )


def _retain_containment_startup_cleanup(task: asyncio.Task[int]) -> None:
    _RETAINED_CONTAINMENT_STARTUP_CLEANUPS.add(task)

    def completed(completed_task: asyncio.Task[int]) -> None:
        _RETAINED_CONTAINMENT_STARTUP_CLEANUPS.discard(completed_task)
        with suppress(BaseException):
            completed_task.result()

    task.add_done_callback(completed)


def validate_stdio_mcp_process_lifetime(
    value: StdioMcpProcessLifetime | str,
) -> StdioMcpProcessLifetime:
    if isinstance(value, StdioMcpProcessLifetime):
        return value
    if type(value) is not str:
        raise TypeError("process_lifetime must be a StdioMcpProcessLifetime or string.")
    try:
        return StdioMcpProcessLifetime(require_clean_nonblank(value, "process_lifetime"))
    except ValueError:
        raise ValueError(
            "process_lifetime must be parent_death_containment, graceful_cleanup, "
            "or persistent_detached."
        ) from None


def validate_stdio_mcp_containment_timeout(value: float, field_name: str) -> float:
    timeout_s = validate_positive_number(value, field_name)
    if not math.isfinite(timeout_s):
        raise ValueError(f"{field_name} must be finite.")
    return timeout_s


def _available(
    capability: str,
    *,
    proof_source: CapabilityProofSource = "integration_validation",
) -> CapabilityClaim:
    return CapabilityClaim(
        capability=capability,
        state="available",
        proof_source=proof_source,
        observation="available",
    )


def _declared(capability: str) -> CapabilityClaim:
    return CapabilityClaim(
        capability=capability,
        state="declared",
        proof_source="integration_declaration",
        observation="supported",
    )


def _unsupported(capability: str, reason: str, remediation: str) -> CapabilityClaim:
    return CapabilityClaim(
        capability=capability,
        state="unsupported",
        proof_source="integration_declaration",
        observation="unavailable",
        reason_code=reason,
        remediation_code=remediation,
    )


def stdio_mcp_process_capability_evidence(
    lifetime: StdioMcpProcessLifetime,
    *,
    _parent_death_containment_proved: bool = False,
) -> CapabilityEvidence:
    """Return bounded configured evidence for one selected stdio lifecycle."""

    if lifetime is StdioMcpProcessLifetime.PARENT_DEATH_CONTAINMENT:
        if _parent_death_containment_proved or stdio_mcp_parent_death_containment_supported():
            graceful_claim = _available(
                "graceful_cleanup",
                proof_source="process_preflight",
            )
            parent_claim = _available(
                "parent_death_containment",
                proof_source="process_preflight",
            )
        elif stdio_mcp_parent_death_containment_platform_candidate() and (
            _current_parent_death_containment_preflight_result() is None
        ):
            graceful_claim = _declared("graceful_cleanup")
            parent_claim = _declared("parent_death_containment")
        elif stdio_mcp_parent_death_containment_platform_candidate():
            graceful_claim = _unsupported(
                "graceful_cleanup",
                "graceful_cleanup_prerequisite_unavailable",
                "select_graceful_cleanup_or_enable_linux_containment_primitives",
            )
            parent_claim = _unsupported(
                "parent_death_containment",
                "parent_death_containment_prerequisite_unavailable",
                "select_graceful_cleanup_or_enable_linux_containment_primitives",
            )
        else:
            graceful_claim = _unsupported(
                "graceful_cleanup",
                "graceful_cleanup_platform_unsupported",
                "select_parent_death_containment_on_supported_linux",
            )
            parent_claim = _unsupported(
                "parent_death_containment",
                "parent_death_containment_platform_unsupported",
                "select_parent_death_containment_on_supported_linux",
            )
    else:
        graceful_claim = _unsupported(
            "graceful_cleanup",
            "graceful_cleanup_complete_tree_unavailable",
            "select_parent_death_containment_on_supported_linux",
        )
        parent_reason = "parent_death_containment_not_selected"
        parent_claim = _unsupported(
            "parent_death_containment",
            parent_reason,
            "select_parent_death_containment_on_supported_linux",
        )
    claims: list[CapabilityClaim] = [graceful_claim, parent_claim]
    if lifetime is StdioMcpProcessLifetime.PERSISTENT_DETACHED and os.name == "posix":
        claims.append(_available("persistent_detached"))
    else:
        detached_reason = (
            "persistent_detached_platform_unsupported"
            if lifetime is StdioMcpProcessLifetime.PERSISTENT_DETACHED
            else "persistent_detached_not_selected"
        )
        claims.append(
            _unsupported(
                "persistent_detached",
                detached_reason,
                "select_persistent_detached_on_posix",
            )
        )
    return CapabilityEvidence(subject="stdio_mcp", claims=tuple(claims))


async def preflight_stdio_mcp_parent_death_containment(
    timeout_s: float = DEFAULT_MCP_CONTAINMENT_STARTUP_TIMEOUT_S,
) -> _ContainmentPreflightProof:
    """Prove the exact Linux wrapper primitives before secrets are resolved."""

    global _PARENT_DEATH_CONTAINMENT_PREFLIGHT_PROCESS_ID
    global _PARENT_DEATH_CONTAINMENT_PREFLIGHT_SUCCEEDED
    timeout_s = validate_stdio_mcp_containment_timeout(
        timeout_s,
        "containment_startup_timeout_s",
    )
    if not stdio_mcp_parent_death_containment_platform_candidate():
        raise RuntimeError("Stdio MCP parent-death containment is unavailable on this host.")
    if signal.getsignal(signal.SIGCHLD) == signal.SIG_IGN:
        _PARENT_DEATH_CONTAINMENT_PREFLIGHT_SUCCEEDED = False
        _PARENT_DEATH_CONTAINMENT_PREFLIGHT_PROCESS_ID = os.getpid()
        raise RuntimeError(
            "Stdio MCP parent-death containment requires waitable child-process exits."
        )
    helper = str(Path(__file__).with_name("_stdio_containment.py"))
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        "-S",
        helper,
        "--role",
        "preflight",
        "--nonce",
        "preflight",
        "--expected-parent-pid",
        str(os.getpid()),
        "--owner-fd",
        "-1",
        "--term-timeout-s",
        "1",
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        env={},
        start_new_session=True,
    )
    wait_task = asyncio.create_task(process.wait())
    cancellation: asyncio.CancelledError | None = None
    timed_out = False
    try:
        try:
            returncode = await asyncio.wait_for(
                asyncio.shield(wait_task),
                timeout=timeout_s,
            )
        except asyncio.CancelledError as error:
            cancellation = error
            returncode = None
        except TimeoutError:
            returncode = None
            timed_out = True
        if returncode is None and process.returncode is None:
            process.kill()
        while not wait_task.done():
            try:
                await asyncio.shield(wait_task)
            except asyncio.CancelledError as error:
                if cancellation is None:
                    cancellation = error
        if cancellation is not None:
            raise cancellation
        returncode = wait_task.result()
    except BaseException:
        if process.returncode is None:
            process.kill()
            with suppress(BaseException):
                await asyncio.shield(wait_task)
        raise
    if timed_out:
        raise TimeoutError("Stdio MCP parent-death containment preflight timed out.")
    if returncode != 0:
        _PARENT_DEATH_CONTAINMENT_PREFLIGHT_SUCCEEDED = False
        _PARENT_DEATH_CONTAINMENT_PREFLIGHT_PROCESS_ID = os.getpid()
        raise RuntimeError(
            "Stdio MCP parent-death containment prerequisites are unavailable on this host."
        )
    _PARENT_DEATH_CONTAINMENT_PREFLIGHT_SUCCEEDED = True
    _PARENT_DEATH_CONTAINMENT_PREFLIGHT_PROCESS_ID = os.getpid()
    return _ContainmentPreflightProof(
        owner_pid=os.getpid(),
        authority=_PARENT_DEATH_CONTAINMENT_PREFLIGHT_AUTHORITY,
    )


def _create_sealed_server_environment_fd(environment: dict[str, str]) -> int:
    """Own one non-durable server environment without exposing it to helpers."""

    import fcntl

    payload_text = json.dumps(environment, sort_keys=True, separators=(",", ":"))
    payload = bytearray(payload_text.encode("utf-8"))
    payload_text = ""
    if len(payload) > _MAX_SERVER_ENV_BYTES:
        payload.clear()
        raise ValueError("MCP stdio server environment exceeded its transfer limit.")
    fd = -1
    try:
        memfd_create = getattr(os, "memfd_create", None)
        if not callable(memfd_create):
            raise RuntimeError("MCP stdio anonymous environment transfer is unavailable.")
        fd = memfd_create(
            "cayu-mcp-server-environment",
            flags=_LINUX_MFD_CLOEXEC | _LINUX_MFD_ALLOW_SEALING,
        )
        fd = _move_fd_above_stdio(fd)
        written = 0
        while written < len(payload):
            written += os.write(fd, payload[written:])
        os.lseek(fd, 0, os.SEEK_SET)
        fcntl.fcntl(fd, _LINUX_F_ADD_SEALS, _LINUX_REQUIRED_MEMFD_SEALS)
        return fd
    except BaseException:
        if fd >= 0:
            with suppress(OSError):
                os.close(fd)
        raise
    finally:
        payload.clear()


def stdio_mcp_process_capability_evidence_for_process(
    process: asyncio.subprocess.Process | ContainedStdioMcpProcess,
) -> CapabilityEvidence:
    """Derive lifecycle evidence only from a runtime-owned process identity."""

    if isinstance(process, ContainedStdioMcpProcess):
        return process.capability_evidence
    lifetime = StdioMcpProcessLifetime.GRACEFUL_CLEANUP
    if isinstance(process, asyncio.subprocess.Process):
        lifetime = _DIRECT_PROCESS_LIFETIMES.get(process, lifetime)
    return stdio_mcp_process_capability_evidence(lifetime)


class ContainedStdioMcpProcess:
    """Asyncio-process-compatible owner for a contained stdio server tree."""

    def __init__(
        self,
        *,
        process: asyncio.subprocess.Process,
        control: socket.socket,
        owner_write_fd: int,
        nonce: str,
        settlement_timeout_s: float,
    ) -> None:
        self._process = process
        self._control = control
        self._owner_write_fd = owner_write_fd
        self._nonce = nonce
        self._capability_evidence = stdio_mcp_process_capability_evidence(
            StdioMcpProcessLifetime.PARENT_DEATH_CONTAINMENT,
            _parent_death_containment_proved=True,
        )
        self.settlement_timeout_s = settlement_timeout_s
        self._anchor_pid: int | None = None
        self._anchor_pgid: int | None = None
        self._server_pid: int | None = None
        self._settled = False
        self._settlement_reason: str | None = None
        loop = asyncio.get_running_loop()
        self._rendezvous_ready = loop.create_future()
        self._ready = loop.create_future()
        self._launch_authorized = False
        self._control_task = asyncio.create_task(self._read_control())
        self._wait_task = asyncio.create_task(self._wait_and_close_owner())
        self.stdin = process.stdin
        self.stdout = process.stdout
        self.stderr = process.stderr

    @property
    def pid(self) -> int:
        return self._process.pid

    @property
    def returncode(self) -> int | None:
        return self._process.returncode

    @property
    def anchor_pid(self) -> int | None:
        return self._anchor_pid

    @property
    def anchor_pgid(self) -> int | None:
        return self._anchor_pgid

    @property
    def server_pid(self) -> int | None:
        return self._server_pid

    @property
    def capability_evidence(self) -> CapabilityEvidence:
        return self._capability_evidence.model_copy(deep=True)

    async def await_ready(self, timeout_s: float) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        try:
            await self._await_startup_evidence(
                self._rendezvous_ready,
                timeout_s=max(0.0, deadline - loop.time()),
            )
            if deadline - loop.time() <= 0.0:
                raise TimeoutError
            self._authorize_launch()
            await self._await_startup_evidence(
                self._ready,
                timeout_s=max(0.0, deadline - loop.time()),
            )
        except TimeoutError:
            self.kill()
            raise McpProtocolError(
                "MCP stdio containment supervisor did not become ready."
            ) from None

    @staticmethod
    async def _await_startup_evidence(
        future: asyncio.Future[None],
        *,
        timeout_s: float,
    ) -> None:
        """Wait without transferring cancellation ownership to startup evidence."""

        done, _pending = await asyncio.wait((future,), timeout=timeout_s)
        if future not in done:
            raise TimeoutError
        future.result()

    def _authorize_launch(self) -> None:
        if self._launch_authorized:
            raise McpProtocolError("MCP stdio containment launch was already authorized.")
        try:
            self._control.sendall(
                json.dumps(
                    {"nonce": self._nonce, "type": "launch"},
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )
        except (BlockingIOError, BrokenPipeError, OSError):
            if self._owner_write_fd >= 0:
                with suppress(OSError):
                    os.close(self._owner_write_fd)
                self._owner_write_fd = -1
            raise McpProtocolError(
                "MCP stdio containment supervisor could not accept launch authority."
            ) from None
        self._launch_authorized = True

    async def wait(self) -> int:
        returncode = await asyncio.shield(self._wait_task)
        self._raise_for_failed_settlement(returncode)
        return returncode

    async def wait_for_settlement(self) -> int:
        """Await the runtime-owned supervisor independently of public wait wrappers."""

        returncode = await asyncio.shield(self._wait_task)
        self._raise_for_failed_settlement(returncode)
        return returncode

    def terminate(self) -> None:
        if self.returncode is not None:
            return
        try:
            self._control.sendall(
                json.dumps(
                    {"nonce": self._nonce, "type": "shutdown"},
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )
        except (BlockingIOError, BrokenPipeError, OSError):
            # Killing only the supervisor is safe: the anchor verifies that its
            # direct parent remains the authenticated supervisor and fences its
            # own group if that relationship changes.
            self.kill()

    def kill(self) -> None:
        if self.returncode is not None:
            return
        try:
            self._control.sendall(
                json.dumps(
                    {"nonce": self._nonce, "type": "force"},
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )
        except (BlockingIOError, BrokenPipeError, OSError):
            # Closing the sole owner writer is the safe fallback. Both wrapper
            # processes independently treat EOF as loss of ownership.
            if self._owner_write_fd >= 0:
                with suppress(OSError):
                    os.close(self._owner_write_fd)
                self._owner_write_fd = -1

    async def _read_control(self) -> None:
        buffer = bytearray()
        failure: BaseException | None = None
        try:
            loop = asyncio.get_running_loop()
            while True:
                chunk = await loop.sock_recv(self._control, 65536)
                if not chunk:
                    break
                buffer.extend(chunk)
                if len(buffer) > _MAX_CONTAINMENT_CONTROL_BYTES:
                    failure = McpProtocolError(
                        "MCP stdio containment control evidence exceeded its byte limit."
                    )
                    break
                while True:
                    newline = buffer.find(b"\n")
                    if newline < 0:
                        break
                    raw = bytes(buffer[:newline])
                    del buffer[: newline + 1]
                    try:
                        message = json.loads(raw)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    candidate_nonce = message.get("nonce") if type(message) is dict else None
                    if type(candidate_nonce) is not str or not hmac.compare_digest(
                        candidate_nonce,
                        self._nonce,
                    ):
                        continue
                    message_type = message.get("type")
                    if message_type == "rendezvous_ready":
                        if self._rendezvous_ready.done():
                            failure = McpProtocolError(
                                "MCP stdio containment supervisor returned duplicate rendezvous evidence."
                            )
                            break
                        self._rendezvous_ready.set_result(None)
                    elif message_type == "ready" and not self._ready.done():
                        if not self._rendezvous_ready.done() or not self._launch_authorized:
                            failure = McpProtocolError(
                                "MCP stdio containment supervisor started without launch authority."
                            )
                            break
                        anchor_pid = message.get("anchor_pid")
                        pgid = message.get("pgid")
                        server_pid = message.get("server_pid")
                        if not all(
                            type(value) is int and value > 0
                            for value in (anchor_pid, pgid, server_pid)
                        ):
                            failure = McpProtocolError(
                                "MCP stdio containment supervisor returned invalid startup evidence."
                            )
                            break
                        self._anchor_pid = anchor_pid
                        self._anchor_pgid = pgid
                        self._server_pid = server_pid
                        self._ready.set_result(None)
                    elif message_type == "start_failed" and not self._ready.done():
                        failure = McpProtocolError("MCP stdio contained process failed to start.")
                        break
                    elif message_type == "settled":
                        self._settled = True
                        reason = message.get("reason")
                        self._settlement_reason = reason if type(reason) is str else None
                if failure is not None:
                    break
        except asyncio.CancelledError:
            raise
        except BaseException:
            failure = McpProtocolError("MCP stdio containment control channel failed.")
        finally:
            buffer.clear()
        if failure is None:
            failure = McpProtocolError("MCP stdio containment supervisor exited before startup.")
        if not self._rendezvous_ready.done():
            self._rendezvous_ready.set_exception(failure)
            if not self._ready.done():
                self._ready.cancel()
        elif not self._ready.done():
            self._ready.set_exception(failure)

    async def _wait_and_close_owner(self) -> int:
        returncode = await self._process.wait()
        if not self._control_task.done():
            with suppress(BaseException):
                # The supervisor alone owns this socket endpoint. Its process
                # exit therefore guarantees EOF, so receipt draining is
                # naturally bounded without a scheduler-sensitive timeout.
                await asyncio.shield(self._control_task)
        self._consume_startup_future_outcomes()
        self._close_owner_handles()
        return returncode

    def _consume_startup_future_outcomes(self) -> None:
        """Retire startup evidence after the supervisor has settled.

        A timeout or cancellation can stop the sole startup waiter before the
        control task publishes its final failure.  Settlement still owns that
        failure, so retrieve it here rather than letting asyncio report an
        unobserved-future diagnostic after otherwise successful cleanup.
        """

        for future in (self._rendezvous_ready, self._ready):
            if future.done() and not future.cancelled():
                with suppress(BaseException):
                    future.exception()

    def _raise_for_failed_settlement(self, returncode: int) -> None:
        if self._settlement_reason == "cleanup_failed":
            raise McpProtocolError("MCP stdio containment cleanup could not prove settlement.")
        if not self._settled:
            raise McpProtocolError("MCP stdio containment supervisor exited without settlement.")

    def _close_owner_handles(self) -> None:
        if self._owner_write_fd >= 0:
            with suppress(OSError):
                os.close(self._owner_write_fd)
            self._owner_write_fd = -1
        with suppress(OSError):
            self._control.close()


async def create_contained_stdio_mcp_process(
    *command: str,
    env: dict[str, str],
    limit: int,
    startup_timeout_s: float,
    term_timeout_s: float,
    kill_timeout_s: float,
    _preflight_proof: _ContainmentPreflightProof | None = None,
    _rendezvous_identity: str | None = None,
    _prepared_rendezvous: _PreparedContainmentRendezvous | None = None,
) -> ContainedStdioMcpProcess:
    """Launch a supported Linux stdio server behind the authenticated supervisor."""

    startup_timeout_s = validate_stdio_mcp_containment_timeout(
        startup_timeout_s,
        "containment_startup_timeout_s",
    )
    term_timeout_s = validate_stdio_mcp_containment_timeout(
        term_timeout_s,
        "containment_term_timeout_s",
    )
    kill_timeout_s = validate_stdio_mcp_containment_timeout(
        kill_timeout_s,
        "containment_kill_timeout_s",
    )
    preflight_proof = _preflight_proof
    if preflight_proof is None:
        preflight_proof = await preflight_stdio_mcp_parent_death_containment(startup_timeout_s)
    _validate_containment_preflight_proof(preflight_proof)
    nonce = secrets.token_hex(32)
    server_env_fd = -1
    parent_control: socket.socket | None = None
    child_control: socket.socket | None = None
    owner_read_fd = -1
    owner_write_fd = -1
    try:
        expected_rendezvous_identity = _validate_containment_rendezvous_identity(
            _rendezvous_identity
            if _rendezvous_identity is not None
            else _command_containment_rendezvous_identity(command)
        )
        if _prepared_rendezvous is None:
            _prepared_rendezvous = _prepare_stdio_mcp_containment_rendezvous(
                expected_rendezvous_identity
            )
        if type(_prepared_rendezvous) is not _PreparedContainmentRendezvous:
            raise RuntimeError("Stdio MCP containment rendezvous authority was invalid.")
        rendezvous_identity = _validate_containment_rendezvous_identity(
            _prepared_rendezvous.consume()
        )
        if not hmac.compare_digest(
            rendezvous_identity,
            expected_rendezvous_identity,
        ):
            raise RuntimeError(
                "Stdio MCP containment rendezvous authority did not match the command."
            )
        server_env_fd = _create_sealed_server_environment_fd(env)
        parent_control, child_control = socket.socketpair()
        parent_control = _move_socket_above_stdio(parent_control)
        child_control = _move_socket_above_stdio(child_control)
        owner_read_fd, owner_write_fd = os.pipe()
        owner_read_fd = _move_fd_above_stdio(owner_read_fd)
        owner_write_fd = _move_fd_above_stdio(owner_write_fd)
    except BaseException:
        if _prepared_rendezvous is not None:
            _prepared_rendezvous.close()
        if server_env_fd >= 0:
            with suppress(OSError):
                os.close(server_env_fd)
        if parent_control is not None:
            parent_control.close()
        if child_control is not None:
            child_control.close()
        for fd in (owner_read_fd, owner_write_fd):
            if fd >= 0:
                with suppress(OSError):
                    os.close(fd)
        raise
    assert parent_control is not None
    assert child_control is not None
    helper = str(Path(__file__).with_name("_stdio_containment.py"))
    process: asyncio.subprocess.Process | None = None
    try:
        parent_control.setblocking(False)
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-S",
            helper,
            "--role",
            "supervisor",
            "--nonce",
            nonce,
            "--expected-parent-pid",
            str(os.getpid()),
            "--owner-fd",
            str(owner_read_fd),
            "--control-fd",
            str(child_control.fileno()),
            "--server-env-fd",
            str(server_env_fd),
            "--rendezvous-identity",
            rendezvous_identity,
            "--term-timeout-s",
            str(term_timeout_s),
            "--kill-timeout-s",
            str(kill_timeout_s),
            "--",
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={},
            limit=limit,
            pass_fds=(
                owner_read_fd,
                child_control.fileno(),
                server_env_fd,
            ),
            start_new_session=True,
        )
    finally:
        # These are child-side parent copies, not lifecycle authorities. Close
        # each independently so one local descriptor failure cannot interrupt
        # ownership handoff after the supervisor has already been created.
        with suppress(OSError):
            os.close(server_env_fd)
        with suppress(OSError):
            os.close(owner_read_fd)
        with suppress(OSError):
            child_control.close()
        if process is None:
            with suppress(OSError):
                os.close(owner_write_fd)
            with suppress(OSError):
                parent_control.close()
    assert process is not None
    try:
        owner = ContainedStdioMcpProcess(
            process=process,
            control=parent_control,
            owner_write_fd=owner_write_fd,
            nonce=nonce,
            # The anchor normally consumes one TERM/KILL sequence. If it loses
            # ownership before authenticating settlement, the supervisor may need
            # one additional bounded sequence as the adopted-tree cleanup owner.
            settlement_timeout_s=2 * (term_timeout_s + kill_timeout_s) + 1.0,
        )
    except BaseException:
        # The process exists, so dropping its sole owner writer is the exact
        # fail-closed handoff: both trusted wrappers classify EOF as owner loss
        # and retain process-tree settlement independently of this task.
        with suppress(OSError):
            os.close(owner_write_fd)
        with suppress(OSError):
            parent_control.close()
        cleanup_task = asyncio.create_task(process.wait())
        _retain_containment_startup_cleanup(cleanup_task)
        raise
    startup_failure: BaseException | None = None
    cleanup_cancellation: asyncio.CancelledError | None = None
    try:
        await owner.await_ready(startup_timeout_s)
    except BaseException as error:
        startup_failure = error
        owner.kill()
        cleanup_task = asyncio.create_task(owner.wait_for_settlement())
        _retain_containment_startup_cleanup(cleanup_task)
        try:
            await asyncio.wait((cleanup_task,), timeout=owner.settlement_timeout_s)
        except asyncio.CancelledError as cancellation:
            cleanup_cancellation = cancellation
        error = None
    if cleanup_cancellation is not None and not isinstance(
        startup_failure,
        (KeyboardInterrupt, SystemExit, GeneratorExit),
    ):
        raise cleanup_cancellation
    if startup_failure is not None:
        raise startup_failure
    return owner


async def create_direct_stdio_mcp_process(
    *command: str,
    env: dict[str, str],
    limit: int,
    lifetime: StdioMcpProcessLifetime,
) -> asyncio.subprocess.Process:
    """Launch an explicitly weaker direct stdio process."""

    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        limit=limit,
        start_new_session=(
            os.name == "posix" and lifetime is StdioMcpProcessLifetime.PERSISTENT_DETACHED
        ),
    )
    _DIRECT_PROCESS_LIFETIMES[process] = lifetime
    return process


def validate_containment_platform(lifetime: StdioMcpProcessLifetime) -> None:
    """Reject unsupported guarantees before secrets or process side effects."""

    if (
        lifetime is StdioMcpProcessLifetime.PARENT_DEATH_CONTAINMENT
        and not stdio_mcp_parent_death_containment_platform_candidate()
    ):
        raise RuntimeError(
            "Stdio MCP parent-death containment is unavailable on this platform; "
            "select graceful_cleanup explicitly to accept the weaker lifecycle. "
            "Complete-tree containment currently requires supported Linux process-tree enforcement "
            "with usable pidfd signaling."
        )
    if lifetime is StdioMcpProcessLifetime.PERSISTENT_DETACHED and os.name != "posix":
        raise RuntimeError("Stdio MCP persistent_detached is supported only on POSIX systems.")


__all__ = [
    "DEFAULT_MCP_CONTAINMENT_KILL_TIMEOUT_S",
    "DEFAULT_MCP_CONTAINMENT_STARTUP_TIMEOUT_S",
    "DEFAULT_MCP_CONTAINMENT_TERM_TIMEOUT_S",
    "ContainedStdioMcpProcess",
    "StdioMcpProcessLifetime",
    "create_contained_stdio_mcp_process",
    "create_direct_stdio_mcp_process",
    "preflight_stdio_mcp_parent_death_containment",
    "stdio_mcp_parent_death_containment_platform_candidate",
    "stdio_mcp_parent_death_containment_supported",
    "stdio_mcp_process_capability_evidence",
    "stdio_mcp_process_capability_evidence_for_process",
    "validate_containment_platform",
    "validate_stdio_mcp_containment_timeout",
    "validate_stdio_mcp_process_lifetime",
]
