from __future__ import annotations

import asyncio
import fcntl
import json
import os
import secrets
import socket
import stat
import sys
import traceback as traceback_module
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cayu._exception_groups import exception_tree_contains
from cayu._task_wait import (
    await_shielded_task_outcome,
    restore_task_cancellation_requests,
)
from cayu._validation import canonical_durable_json_bytes
from cayu.runtime.local_execution_attempts import (
    MAX_LOCAL_EXECUTION_ENVIRONMENT_BYTES,
    MAX_LOCAL_EXECUTION_RECEIPT_BYTES,
    MAX_LOCAL_EXECUTION_RECOVERY_BATCH,
    LocalExecutionAttemptAuthority,
    LocalExecutionAttemptConflict,
    LocalExecutionAttemptListCursor,
    LocalExecutionAttemptPhase,
    LocalExecutionAttemptQuiescence,
    LocalExecutionAttemptReceipt,
    LocalExecutionAttemptRecord,
    LocalExecutionAttemptRecoveryClaim,
    LocalExecutionAttemptRequest,
    LocalExecutionAttemptResult,
    LocalExecutionAttemptSettlement,
    LocalExecutionAttemptStart,
    LocalExecutionAttemptUnavailable,
    LocalExecutionAttemptUnsettled,
    LocalExecutionProcessIdentity,
    _authenticate_local_execution_attempt_settlement,
    _clean_local_execution_async_boundary,
    build_local_execution_attempt_authority,
    local_execution_attempt_list_cursor,
    local_execution_attempt_receipt_sha256,
    local_execution_boot_id,
    local_execution_host_identity,
)

if TYPE_CHECKING:
    from cayu.runtime.app import CayuApp
    from cayu.runtime.tasks import TaskStore


_MEMFD_CLOEXEC = 0x0001
_MEMFD_ALLOW_SEALING = 0x0002
_REQUIRED_SEALS = 0x0001 | 0x0002 | 0x0004 | 0x0008
_MAX_CONTROL_BYTES = 65_536
_MAX_PREPARATION_CLAIM_REFRESHES = 8
_MAX_LOCAL_EXECUTION_RECOVERY_SCAN_RECORDS = MAX_LOCAL_EXECUTION_RECOVERY_BATCH
_RETAINED_LOCAL_EXECUTION_TASKS: set[asyncio.Task[Any]] = set()


class _LocalExecutionRecoveryDeadlineElapsed(Exception):
    """Stop one recovery scan without weakening unsettled evidence."""


@dataclass(frozen=True, slots=True)
class _LocalExecutionRecoveryScan:
    records: tuple[LocalExecutionAttemptRecord, ...]
    after: LocalExecutionAttemptListCursor | None
    reached_end: bool
    deadline_elapsed: bool


def _retain_local_execution_task(
    task: asyncio.Task[Any],
    *,
    remove_path_on_success: Path | None = None,
) -> None:
    _RETAINED_LOCAL_EXECUTION_TASKS.add(task)

    def settled(done: asyncio.Task[Any]) -> None:
        _RETAINED_LOCAL_EXECUTION_TASKS.discard(done)
        try:
            done.result()
        except BaseException:
            return
        if remove_path_on_success is not None:
            with suppress(OSError):
                remove_path_on_success.unlink()

    task.add_done_callback(settled)


def _create_local_execution_task(
    operation: Any,
    *,
    name: str | None = None,
) -> asyncio.Task[Any]:
    """Create one owned task without leaking its coroutine on setup failure."""

    try:
        return asyncio.create_task(operation, name=name)
    except BaseException:
        close = getattr(operation, "close", None)
        if callable(close):
            close()
        raise


def _fallback_process_wait_task(process: asyncio.subprocess.Process) -> asyncio.Task[int]:
    operation = process.wait()
    try:
        task = asyncio.ensure_future(operation)
    except BaseException:
        operation.close()
        raise
    if not isinstance(task, asyncio.Task):
        task.cancel()
        raise RuntimeError("Local execution supervisor wait ownership was unavailable.")
    return task


def _move_fd(fd: int) -> int:
    if fd > 2:
        return fd
    duplicate = fcntl.fcntl(fd, fcntl.F_DUPFD_CLOEXEC, 3)
    os.close(fd)
    return duplicate


def _move_socket(value: socket.socket) -> socket.socket:
    if value.fileno() > 2:
        return value
    duplicate = fcntl.fcntl(value.fileno(), fcntl.F_DUPFD_CLOEXEC, 3)
    value.close()
    return socket.socket(fileno=duplicate)


def _sealed_launch_fd(request: LocalExecutionAttemptRequest) -> int:
    environment = request.effective_environment()
    payload = bytearray(
        json.dumps(
            {
                "argv": list(request.argv),
                "cwd": request.cwd,
                "env": environment,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    environment.clear()
    if len(payload) > MAX_LOCAL_EXECUTION_ENVIRONMENT_BYTES:
        payload.clear()
        raise ValueError("Local execution launch payload exceeded its transfer limit.")
    fd = -1
    view: memoryview | None = None
    try:
        memfd_create = getattr(os, "memfd_create", None)
        if not callable(memfd_create):
            raise RuntimeError("Anonymous local execution launch transfer is unavailable.")
        fd = _move_fd(
            memfd_create(
                "cayu-local-execution-launch",
                flags=_MEMFD_CLOEXEC | _MEMFD_ALLOW_SEALING,
            )
        )
        view = memoryview(payload)
        written = 0
        while written < len(view):
            written += os.write(fd, view[written:])
        os.lseek(fd, 0, os.SEEK_SET)
        fcntl.fcntl(fd, 1033, _REQUIRED_SEALS)
        return fd
    except BaseException:
        if fd >= 0:
            with suppress(OSError):
                os.close(fd)
        raise
    finally:
        if view is not None:
            view.release()
        payload.clear()


def _receipt_path(state_dir: Path, authority: LocalExecutionAttemptAuthority) -> Path:
    digest = authority.request_sha256
    return state_dir / f"{digest}.receipt.json"


def _unlink_receipt_from_existing_owner_directory(path: Path) -> None:
    """Remove replay evidence only through an authenticated existing directory."""

    directory_fd = -1
    try:
        directory_fd = os.open(
            path.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        )
        observed = os.fstat(directory_fd)
        if (
            not stat.S_ISDIR(observed.st_mode)
            or observed.st_uid != os.geteuid()
            or stat.S_IMODE(observed.st_mode) & 0o077
        ):
            return
        try:
            os.unlink(path.name, dir_fd=directory_fd)
        except OSError:
            return
    except OSError:
        return
    finally:
        if directory_fd >= 0:
            os.close(directory_fd)


def _prepare_state_dir(state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        observed = state_dir.lstat()
    except OSError:
        raise LocalExecutionAttemptUnsettled(
            "Local execution settlement directory could not be authenticated."
        ) from None
    if (
        not stat.S_ISDIR(observed.st_mode)
        or observed.st_uid != os.geteuid()
        or stat.S_IMODE(observed.st_mode) & 0o077
    ):
        raise LocalExecutionAttemptUnavailable(
            "Local execution settlement directory must be an owner-private directory."
        )


def _rendezvous_identity(
    authority: LocalExecutionAttemptAuthority,
    *,
    state_dir: Path,
) -> str:
    import hashlib

    coordinator_namespace = hashlib.sha256(
        b"cayu.local_execution.coordinator.v1\0" + os.fsencode(str(state_dir.resolve()))
    ).hexdigest()
    payload = {
        "coordinator_namespace": coordinator_namespace,
        "effect_lineage_id": authority.effect_lineage_id,
        "retry_scope": (
            f"retry:{authority.retry_series_id}"
            if authority.retry_series_id is not None
            else f"task:{authority.task_id}"
        ),
        "schema": "cayu.local_execution.rendezvous.v1",
        "task_id": authority.task_id,
        "task_created_at": authority.task_created_at.isoformat().replace("+00:00", "Z"),
    }

    return hashlib.sha256(
        canonical_durable_json_bytes(payload, "local execution rendezvous identity")
    ).hexdigest()


def _rendezvous_address(identity: str) -> bytes:
    return (
        b"\0cayu-local-execution-v1-u"
        + str(os.geteuid()).encode("ascii")
        + b"-"
        + identity.encode("ascii")
    )


async def _probe_rendezvous(
    record: LocalExecutionAttemptRecord,
    *,
    state_dir: Path,
    timeout: float = 0.1,
    deadline: float | None = None,
) -> dict[str, Any] | None:
    if record.start is None:
        identity = _rendezvous_identity(record.authority, state_dir=state_dir)
        expected_nonce = None
    else:
        identity = record.start.rendezvous_identity
        expected_nonce = record.start.supervisor_nonce
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.setblocking(False)
    try:
        loop = asyncio.get_running_loop()

        def probe_timeout() -> tuple[float, bool]:
            operation_timeout = timeout
            deadline_limited = False
            if deadline is not None:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise _LocalExecutionRecoveryDeadlineElapsed
                if remaining <= operation_timeout:
                    operation_timeout = remaining
                    deadline_limited = True
            return operation_timeout, deadline_limited

        async def await_probe(
            operation: Any,
            operation_timeout: float,
            deadline_limited: bool,
        ) -> Any:
            try:
                return await asyncio.wait_for(operation, timeout=operation_timeout)
            except TimeoutError:
                if deadline_limited:
                    raise _LocalExecutionRecoveryDeadlineElapsed from None
                raise

        operation_timeout, deadline_limited = probe_timeout()
        await await_probe(
            loop.sock_connect(client, _rendezvous_address(identity)),
            operation_timeout,
            deadline_limited,
        )
        operation_timeout, deadline_limited = probe_timeout()
        payload = await await_probe(
            loop.sock_recv(client, _MAX_CONTROL_BYTES + 1),
            operation_timeout,
            deadline_limited,
        )
    except (OSError, TimeoutError):
        return None
    finally:
        client.close()
    if len(payload) > _MAX_CONTROL_BYTES or not payload.endswith(b"\n"):
        return None
    try:
        document = json.loads(payload[:-1].decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        return None
    if (
        type(document) is not dict
        or document.get("attempt_id") != record.authority.attempt_id
        or (expected_nonce is not None and document.get("nonce") != expected_nonce)
        or document.get("state") not in {"ready", "running", "settled"}
    ):
        return None
    return document


def _exact_process_is_live(identity: LocalExecutionProcessIdentity) -> bool:
    try:
        stat_text = Path(f"/proc/{identity.pid}/stat").read_text(encoding="ascii")
        proc_inode = Path(f"/proc/{identity.pid}").stat().st_ino
    except (OSError, UnicodeError):
        return False
    close = stat_text.rfind(")")
    if close < 0:
        return False
    fields = stat_text[close + 2 :].split()
    if len(fields) <= 19:
        return False
    try:
        process_group = int(fields[2])
        start_tick = int(fields[19])
    except ValueError:
        return False
    return (
        process_group == identity.process_group
        and start_tick == identity.start_tick
        and proc_inode == identity.proc_inode
        and fields[0] not in {"Z", "X"}
    )


def _recovery_receipt(
    record: LocalExecutionAttemptRecord,
    *,
    reason: str,
    quiescence: str,
    effect_outcome: str,
) -> LocalExecutionAttemptReceipt:
    now = datetime.now(UTC)
    start = record.start
    payload: dict[str, Any] = {
        "attempt_id": record.authority.attempt_id,
        "boot_id": local_execution_boot_id() if start is None else start.boot_id,
        "descendants_observed": 0,
        "effect_outcome": effect_outcome,
        "exit_code": None,
        "host_identity": local_execution_host_identity() if start is None else start.host_identity,
        "kill_sent": False,
        "quiescence": quiescence,
        "request_sha256": record.authority.request_sha256,
        "root": None if start is None or start.root is None else start.root.model_dump(mode="json"),
        "settled_at": now.isoformat().replace("+00:00", "Z"),
        "supervisor_nonce": "recovery-not-dispatched" if start is None else start.supervisor_nonce,
        "term_sent": False,
        "terminal_reason": reason,
    }
    payload["receipt_sha256"] = local_execution_attempt_receipt_sha256(payload)
    return LocalExecutionAttemptReceipt.model_validate(payload)


async def _read_control_message(
    control: socket.socket,
    buffer: bytearray,
    *,
    timeout: float,
) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        if b"\n" in buffer:
            line, _, remainder = buffer.partition(b"\n")
            buffer[:] = remainder
            value = json.loads(line.decode("utf-8"))
            if type(value) is not dict:
                raise RuntimeError("Local execution supervisor control message was malformed.")
            return value
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise TimeoutError("Local execution supervisor control deadline elapsed.")
        chunk = await asyncio.wait_for(loop.sock_recv(control, 65_536), timeout=remaining)
        if not chunk:
            raise RuntimeError("Local execution supervisor closed its control channel.")
        buffer.extend(chunk)
        if len(buffer) > _MAX_CONTROL_BYTES:
            raise RuntimeError("Local execution supervisor control message exceeded its limit.")


async def _send_control(control: socket.socket, payload: dict[str, object]) -> None:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    await asyncio.get_running_loop().sock_sendall(control, encoded + b"\n")


async def _read_bounded(
    stream: asyncio.StreamReader | None,
    limit: int,
    *,
    app: CayuApp,
) -> tuple[str, bool]:
    if stream is None:
        return "", False
    collected = bytearray()
    total = 0
    try:
        while True:
            chunk = await stream.read(65_536)
            if not chunk:
                break
            total += len(chunk)
            if len(collected) < limit:
                collected.extend(chunk[: max(0, limit - len(collected))])
        source_truncated = total > limit
        if limit == 0:
            return "", source_truncated
        projected, projection_truncated = app.redact_utf8_head(
            bytes(collected),
            max_bytes=limit,
            source_complete=not source_truncated,
        )
        return projected, source_truncated or projection_truncated
    finally:
        collected.clear()


def _start_from_message(
    *,
    authority: LocalExecutionAttemptAuthority,
    nonce: str,
    rendezvous_identity: str,
    message: dict[str, Any],
    root: LocalExecutionProcessIdentity | None,
) -> LocalExecutionAttemptStart:
    return LocalExecutionAttemptStart(
        attempt_id=authority.attempt_id,
        request_sha256=authority.request_sha256,
        host_identity=local_execution_host_identity(),
        boot_id=local_execution_boot_id(),
        supervisor_nonce=nonce,
        rendezvous_identity=rendezvous_identity,
        supervisor=LocalExecutionProcessIdentity.model_validate(message["supervisor"]),
        root=root,
        started_at=datetime.now(UTC),
    )


def _load_receipt(
    path: Path, authority: LocalExecutionAttemptAuthority
) -> LocalExecutionAttemptReceipt:
    fd = -1
    try:
        fd = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        )
        observed = os.fstat(fd)
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_uid != os.geteuid()
            or not 0 <= observed.st_size <= MAX_LOCAL_EXECUTION_RECEIPT_BYTES
        ):
            raise ValueError("receipt file authority was invalid")
        encoded = bytearray()
        while len(encoded) < observed.st_size:
            chunk = os.read(fd, min(65_536, observed.st_size - len(encoded)))
            if not chunk:
                break
            encoded.extend(chunk)
        if len(encoded) != observed.st_size:
            raise ValueError("receipt file was incomplete")
        try:
            payload = json.loads(encoded.decode("utf-8"))
        finally:
            encoded.clear()
    except (OSError, RecursionError, UnicodeError, ValueError, json.JSONDecodeError):
        raise LocalExecutionAttemptUnsettled(
            "Local execution supervisor did not publish a valid settlement receipt."
        ) from None
    finally:
        if fd >= 0:
            os.close(fd)
    if type(payload) is not dict:
        raise LocalExecutionAttemptUnsettled("Local execution settlement receipt was malformed.")
    receipt: LocalExecutionAttemptReceipt | None = None
    receipt_error: str | None = None
    try:
        expected = local_execution_attempt_receipt_sha256(payload)
        if payload.get("receipt_sha256") != expected:
            receipt_error = "Local execution settlement receipt digest conflicted."
        else:
            try:
                receipt = LocalExecutionAttemptReceipt.model_validate(payload)
            except (RecursionError, TypeError, ValueError):
                receipt_error = "Local execution settlement receipt was malformed."
    except (RecursionError, TypeError, ValueError):
        receipt_error = "Local execution settlement receipt was malformed."
    finally:
        payload.clear()
    if receipt is None:
        raise LocalExecutionAttemptUnsettled(
            receipt_error or "Local execution settlement receipt was malformed."
        ) from None
    if (
        receipt.attempt_id != authority.attempt_id
        or receipt.request_sha256 != authority.request_sha256
    ):
        raise LocalExecutionAttemptUnsettled(
            "Local execution settlement receipt authority conflicted."
        )
    return receipt


async def _load_receipt_or_exact_durable_settlement(
    *,
    task_store: TaskStore,
    path: Path,
    authority: LocalExecutionAttemptAuthority,
) -> LocalExecutionAttemptReceipt:
    """Read the supervisor receipt or reconcile a recovery-owned settlement."""

    try:
        return _load_receipt(path, authority)
    except LocalExecutionAttemptUnsettled as receipt_failure:
        record = await task_store.load_local_execution_attempt(authority.attempt_id)
        if (
            record is None
            or record.authority != authority
            or record.phase is not LocalExecutionAttemptPhase.TERMINAL
            or record.receipt is None
        ):
            raise receipt_failure
        return record.receipt


def _promote_authenticated_staged_receipt(
    path: Path,
    authority: LocalExecutionAttemptAuthority,
) -> LocalExecutionAttemptReceipt | None:
    """Finish a supervisor receipt rename from authenticated settlement evidence."""

    staging = path.with_name(f"{path.name}.staging")
    if not staging.exists():
        return None
    receipt = _load_receipt(staging, authority)
    try:
        os.replace(staging, path)
    except OSError:
        # Another recovery owner may have promoted the same exact stage, or a
        # successful rename may have lost its local acknowledgement.  Only an
        # exact authenticated final receipt reconciles that boundary.
        try:
            promoted = _load_receipt(path, authority)
        except LocalExecutionAttemptUnsettled:
            raise LocalExecutionAttemptUnsettled(
                "Local execution settlement receipt promotion failed."
            ) from None
        if promoted != receipt:
            raise LocalExecutionAttemptUnsettled(
                "Local execution settlement receipt promotion conflicted."
            ) from None
    try:
        directory_fd = os.open(
            path.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        raise LocalExecutionAttemptUnsettled(
            "Local execution settlement receipt promotion failed."
        ) from None
    return receipt


def _authenticated_settlement(
    authority: LocalExecutionAttemptAuthority,
    receipt: LocalExecutionAttemptReceipt,
    *,
    recovery_owner_id: str | None = None,
    expected_recovery_generation: int | None = None,
) -> LocalExecutionAttemptSettlement:
    return _authenticate_local_execution_attempt_settlement(
        LocalExecutionAttemptSettlement(
            attempt_id=authority.attempt_id,
            request_sha256=authority.request_sha256,
            receipt=receipt,
            recovery_owner_id=recovery_owner_id,
            expected_recovery_generation=expected_recovery_generation,
        )
    )


def _same_local_execution_lineage(
    original: LocalExecutionAttemptAuthority,
    candidate: LocalExecutionAttemptAuthority,
) -> bool:
    return (
        original.task_id == candidate.task_id
        and original.task_created_at == candidate.task_created_at
        and original.task_invocation_sha256 == candidate.task_invocation_sha256
        and original.worker_id == candidate.worker_id
        and original.retry_series_id == candidate.retry_series_id
        and original.retry_attempt == candidate.retry_attempt
        and original.session_id == candidate.session_id
        and original.session_instance_id == candidate.session_instance_id
        and original.effect_lineage_id == candidate.effect_lineage_id
        and original.command_sha256 == candidate.command_sha256
        and original.execution_profile_fingerprint == candidate.execution_profile_fingerprint
        and original.workspace_identity == candidate.workspace_identity
        and original.lifetime is candidate.lifetime
        and original.effect_policy is candidate.effect_policy
        and original.idempotency_key_sha256 == candidate.idempotency_key_sha256
        and original.containment_backend == candidate.containment_backend
    )


async def _prepare_current_local_execution_attempt(
    *,
    app: CayuApp,
    task_store: TaskStore,
    authority: LocalExecutionAttemptAuthority,
    request: LocalExecutionAttemptRequest,
) -> tuple[LocalExecutionAttemptAuthority, LocalExecutionAttemptRecord]:
    """Bind preparation to a current claim without crossing task lineage."""

    from cayu.runtime.tasks import TaskClaimLost

    original = authority
    current = authority
    last_conflict: LocalExecutionAttemptConflict | None = None
    for _ in range(_MAX_PREPARATION_CLAIM_REFRESHES):
        try:
            prepared = await task_store.prepare_local_execution_attempt(current)
        except TaskClaimLost:
            raise LocalExecutionAttemptConflict(
                "Local execution attempt no longer owns the exact task claim generation."
            ) from None
        except LocalExecutionAttemptConflict as error:
            last_conflict = error
            if await task_store.load_local_execution_attempt(current.attempt_id) is not None:
                # A claim refresh may repair only a stale task snapshot.  It
                # must never route around an occupied or corrupt exact
                # attempt identity.
                raise error from None
            latest_task = await task_store.load_task(original.task_id)
            if latest_task is None:
                raise error from None
            try:
                latest = build_local_execution_attempt_authority(
                    app=app,
                    task=latest_task,
                    worker_id=original.worker_id,
                    request=request,
                )
            except (AttributeError, LocalExecutionAttemptConflict):
                raise error from None
            if not _same_local_execution_lineage(original, latest) or (
                latest.task_claim_updated_at == current.task_claim_updated_at
                and latest.task_claim_lease_expires_at == current.task_claim_lease_expires_at
            ):
                raise error from None
            current = latest
            continue
        return current, prepared
    if last_conflict is not None:
        raise last_conflict from None
    raise LocalExecutionAttemptConflict(
        "Local execution attempt claim changed too frequently during preparation."
    )


def _exact_terminal_replay_result(
    record: LocalExecutionAttemptRecord,
    *,
    receipt_path: Path,
) -> LocalExecutionAttemptResult | None:
    """Return one exact durable terminal result without preparing new work."""

    if record.phase is not LocalExecutionAttemptPhase.TERMINAL:
        return None
    if record.receipt is None:
        raise LocalExecutionAttemptUnsettled(
            "The exact local execution attempt is terminal without settlement evidence."
        )
    _unlink_receipt_from_existing_owner_directory(receipt_path)
    output_unavailable = record.quiescence is not LocalExecutionAttemptQuiescence.NOT_DISPATCHED
    return LocalExecutionAttemptResult(
        attempt=record,
        stdout_truncated=output_unavailable,
        stderr_truncated=output_unavailable,
    )


@_clean_local_execution_async_boundary
async def run_owned_local_execution_attempt(
    *,
    app: CayuApp,
    task_store: TaskStore,
    state_dir: Path,
    authority: LocalExecutionAttemptAuthority,
    request: LocalExecutionAttemptRequest,
) -> LocalExecutionAttemptResult:
    receipt_path = _receipt_path(state_dir, authority)
    existing = await task_store.load_local_execution_attempt(authority.attempt_id)
    if existing is not None:
        if existing.authority != authority:
            raise LocalExecutionAttemptConflict(
                "Local execution attempt identity is bound to different authority."
            )
        replay = _exact_terminal_replay_result(existing, receipt_path=receipt_path)
        if replay is not None:
            return replay

    _prepare_state_dir(state_dir)
    from cayu.runtime.local_execution_attempts import (
        local_execution_parent_death_containment_platform_candidate,
    )

    if not local_execution_parent_death_containment_platform_candidate():
        raise LocalExecutionAttemptUnavailable(
            "General local execution attempts require supported Linux process primitives."
        )
    # Reuse the production parent-death preflight owned by the stdio MCP
    # containment boundary. The general-attempt supervisor uses the same
    # subreaper, pidfd, and proc-identity primitives; launch is not authorized
    # and no retry fence is published until that shared boundary has proved
    # them in this process.
    from cayu.mcp._stdio_process import preflight_stdio_mcp_parent_death_containment

    preflight_failure: LocalExecutionAttemptUnavailable | None = None
    try:
        await preflight_stdio_mcp_parent_death_containment(request.limits.startup_timeout_seconds)
    except Exception as error:
        if error.__traceback__ is not None:
            traceback_module.clear_frames(error.__traceback__)
        preflight_failure = LocalExecutionAttemptUnavailable(
            "General local execution containment preflight failed."
        )
    if preflight_failure is not None:
        raise preflight_failure from None
    authority, prior = await _prepare_current_local_execution_attempt(
        app=app,
        task_store=task_store,
        authority=authority,
        request=request,
    )
    receipt_path = _receipt_path(state_dir, authority)
    replay = _exact_terminal_replay_result(prior, receipt_path=receipt_path)
    if replay is not None:
        return replay
    if prior.phase is not LocalExecutionAttemptPhase.PREPARED:
        raise LocalExecutionAttemptUnsettled(
            "The exact local execution attempt is already active without settlement."
        )
    launch_fd = _sealed_launch_fd(request)
    parent_control, child_control = socket.socketpair()
    parent_control = _move_socket(parent_control)
    child_control = _move_socket(child_control)
    parent_control.setblocking(False)
    owner_read_fd, owner_write_fd = os.pipe()
    owner_read_fd = _move_fd(owner_read_fd)
    owner_write_fd = _move_fd(owner_write_fd)
    nonce = secrets.token_hex(32)
    rendezvous_identity = _rendezvous_identity(authority, state_dir=state_dir)
    helper = str(Path(__file__).with_name("_local_execution_supervisor.py"))
    process: asyncio.subprocess.Process | None = None
    control_buffer = bytearray()
    start: LocalExecutionAttemptStart | None = None
    stdout_task: asyncio.Task[tuple[str, bool]] | None = None
    stderr_task: asyncio.Task[tuple[str, bool]] | None = None
    process_wait_task: asyncio.Task[int] | None = None
    primary_failure: BaseException | None = None
    cleanup_failures: list[BaseException] = []
    deferred_cancellation: asyncio.CancelledError | None = None
    cancellation_requests_consumed = 0
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-S",
            helper,
            "--attempt-id",
            authority.attempt_id,
            "--request-sha256",
            authority.request_sha256,
            "--nonce",
            nonce,
            "--expected-parent-pid",
            str(os.getpid()),
            "--owner-fd",
            str(owner_read_fd),
            "--control-fd",
            str(child_control.fileno()),
            "--launch-fd",
            str(launch_fd),
            "--receipt-path",
            str(receipt_path),
            "--rendezvous-identity",
            rendezvous_identity,
            "--host-identity",
            local_execution_host_identity(),
            "--boot-id",
            local_execution_boot_id(),
            "--effect-policy",
            request.effect_policy.value,
            "--lifetime",
            request.lifetime.value,
            "--term-grace-seconds",
            str(request.limits.term_grace_seconds),
            "--kill-grace-seconds",
            str(request.limits.kill_grace_seconds),
            *(
                ()
                if request.limits.deadline_seconds is None
                else ("--deadline-seconds", str(request.limits.deadline_seconds))
            ),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={},
            pass_fds=(owner_read_fd, child_control.fileno(), launch_fd),
            start_new_session=True,
        )
    finally:
        with suppress(OSError):
            os.close(owner_read_fd)
        with suppress(OSError):
            os.close(launch_fd)
        child_control.close()
        if process is None:
            with suppress(OSError):
                os.close(owner_write_fd)
            parent_control.close()
    try:
        assert process is not None
        process_wait_task = _create_local_execution_task(
            process.wait(),
            name=f"cayu-local-execution-supervisor-{authority.attempt_id}",
        )
        stdout_task = _create_local_execution_task(
            _read_bounded(
                process.stdout,
                request.limits.max_output_bytes,
                app=app,
            )
        )
        stderr_task = _create_local_execution_task(
            _read_bounded(
                process.stderr,
                request.limits.max_output_bytes,
                app=app,
            )
        )
        ready = await _read_control_message(
            parent_control,
            control_buffer,
            timeout=request.limits.startup_timeout_seconds,
        )
        if ready.get("nonce") != nonce or ready.get("type") != "ready":
            raise RuntimeError("Local execution supervisor did not publish ready authority.")
        start = _start_from_message(
            authority=authority,
            nonce=nonce,
            rendezvous_identity=rendezvous_identity,
            message=ready,
            root=None,
        )
        await task_store.start_local_execution_attempt(start)
        await _send_control(parent_control, {"nonce": nonce, "type": "launch"})
        started = await _read_control_message(
            parent_control,
            control_buffer,
            timeout=request.limits.startup_timeout_seconds,
        )
        if started.get("nonce") != nonce or started.get("type") != "started":
            raise RuntimeError("Local execution supervisor did not publish root authority.")
        start = start.model_copy(
            update={"root": LocalExecutionProcessIdentity.model_validate(started["root"])}
        )
        await task_store.start_local_execution_attempt(start)
        settlement_timeout = (
            (request.limits.deadline_seconds or 31_536_000)
            + 2 * request.limits.term_grace_seconds
            + request.limits.kill_grace_seconds
            + 2.0
        )
        await asyncio.wait_for(asyncio.shield(process_wait_task), settlement_timeout)
    except asyncio.CancelledError as cancellation:
        with suppress(BaseException):
            await _send_control(parent_control, {"nonce": nonce, "type": "shutdown"})
        with suppress(OSError):
            os.close(owner_write_fd)
        owner_write_fd = -1
        assert process is not None
        if process_wait_task is None:
            try:
                process_wait_task = _fallback_process_wait_task(process)
            except BaseException as wait_ownership_failure:
                if stdout_task is not None:
                    _retain_local_execution_task(stdout_task)
                if stderr_task is not None:
                    _retain_local_execution_task(stderr_task)
                raise cancellation from wait_ownership_failure
        settlement_outcome = await await_shielded_task_outcome(
            process_wait_task,
            cancellation=cancellation,
            timeout_s=(request.limits.term_grace_seconds + request.limits.kill_grace_seconds + 2.0),
        )
        if settlement_outcome.timed_out:
            _retain_local_execution_task(process_wait_task)
            if stdout_task is not None:
                _retain_local_execution_task(stdout_task)
            if stderr_task is not None:
                _retain_local_execution_task(stderr_task)
            restore_task_cancellation_requests(
                settlement_outcome.cancellation_requests_consumed,
                cancellation=cancellation,
            )
            raise cancellation from LocalExecutionAttemptUnsettled(
                "Local execution tree cleanup remains in flight and retry stays fenced."
            )
        cleanup_failure = settlement_outcome.error
        cancellation_requests_consumed = settlement_outcome.cancellation_requests_consumed
        if cleanup_failure is None:
            try:
                receipt = await _load_receipt_or_exact_durable_settlement(
                    task_store=task_store,
                    path=receipt_path,
                    authority=authority,
                )
            except BaseException as receipt_failure:
                cleanup_failure = receipt_failure
            else:
                settlement_task = asyncio.create_task(
                    task_store.settle_local_execution_attempt(
                        _authenticated_settlement(authority, receipt)
                    ),
                    name=f"cayu-local-execution-settlement-{authority.attempt_id}",
                )
                durable_outcome = await await_shielded_task_outcome(
                    settlement_task,
                    cancellation=cancellation,
                    timeout_s=(
                        request.limits.term_grace_seconds + request.limits.kill_grace_seconds + 2.0
                    ),
                )
                cancellation_requests_consumed += durable_outcome.cancellation_requests_consumed
                if durable_outcome.timed_out:
                    _retain_local_execution_task(
                        settlement_task,
                        remove_path_on_success=receipt_path,
                    )
                    cleanup_failure = LocalExecutionAttemptUnsettled(
                        "Local execution settlement remains in flight and retry stays fenced."
                    )
                else:
                    cleanup_failure = durable_outcome.error
                    if cleanup_failure is None:
                        with suppress(OSError):
                            receipt_path.unlink()
        if stdout_task is not None:
            _retain_local_execution_task(stdout_task)
        if stderr_task is not None:
            _retain_local_execution_task(stderr_task)
        restore_task_cancellation_requests(
            cancellation_requests_consumed,
            cancellation=cancellation,
        )
        if cleanup_failure is not None:
            if exception_tree_contains(
                cleanup_failure,
                (KeyboardInterrupt, SystemExit, GeneratorExit),
            ):
                raise cleanup_failure from cancellation
            raise cancellation from cleanup_failure
        raise cancellation
    except BaseException as exc:
        primary_failure = exc
        with suppress(BaseException):
            await _send_control(parent_control, {"nonce": nonce, "type": "kill"})
        with suppress(OSError):
            os.close(owner_write_fd)
        owner_write_fd = -1
        assert process is not None
        if process_wait_task is None:
            try:
                process_wait_task = _fallback_process_wait_task(process)
            except BaseException as wait_ownership_failure:
                cleanup_failures.append(wait_ownership_failure)
        cleanup_outcome = (
            None
            if process_wait_task is None
            else await await_shielded_task_outcome(
                process_wait_task,
                timeout_s=(
                    request.limits.term_grace_seconds + request.limits.kill_grace_seconds + 2.0
                ),
            )
        )
        if cleanup_outcome is not None:
            deferred_cancellation = (
                cleanup_outcome.cancellation or cleanup_outcome.subsequent_cancellation
            )
            cancellation_requests_consumed += cleanup_outcome.cancellation_requests_consumed
        if cleanup_outcome is None:
            cleanup_failures.append(
                LocalExecutionAttemptUnsettled(
                    "Local execution supervisor cleanup ownership was unavailable."
                )
            )
        elif cleanup_outcome.timed_out:
            cleanup_failures.append(
                LocalExecutionAttemptUnsettled(
                    "Local execution tree cleanup remains in flight and retry stays fenced."
                )
            )
            assert process_wait_task is not None
            _retain_local_execution_task(process_wait_task)
        elif cleanup_outcome.error is not None:
            cleanup_failures.append(cleanup_outcome.error)
        if receipt_path.exists():
            try:
                receipt = await _load_receipt_or_exact_durable_settlement(
                    task_store=task_store,
                    path=receipt_path,
                    authority=authority,
                )
                settlement_task = asyncio.create_task(
                    task_store.settle_local_execution_attempt(
                        _authenticated_settlement(authority, receipt)
                    ),
                    name=f"cayu-local-execution-failure-settlement-{authority.attempt_id}",
                )
                settlement_outcome = await await_shielded_task_outcome(
                    settlement_task,
                    cancellation=deferred_cancellation,
                    timeout_s=(
                        request.limits.term_grace_seconds + request.limits.kill_grace_seconds + 2.0
                    ),
                )
                deferred_cancellation = (
                    deferred_cancellation
                    or settlement_outcome.cancellation
                    or settlement_outcome.subsequent_cancellation
                )
                cancellation_requests_consumed += settlement_outcome.cancellation_requests_consumed
                if settlement_outcome.timed_out:
                    _retain_local_execution_task(
                        settlement_task,
                        remove_path_on_success=receipt_path,
                    )
                    cleanup_failures.append(
                        LocalExecutionAttemptUnsettled(
                            "Local execution settlement remains in flight and retry stays fenced."
                        )
                    )
                elif settlement_outcome.error is not None:
                    cleanup_failures.append(settlement_outcome.error)
                else:
                    with suppress(OSError):
                        receipt_path.unlink()
            except BaseException as settlement_failure:
                cleanup_failures.append(settlement_failure)
        if stdout_task is not None:
            _retain_local_execution_task(stdout_task)
        if stderr_task is not None:
            _retain_local_execution_task(stderr_task)
    finally:
        if owner_write_fd >= 0:
            with suppress(OSError):
                os.close(owner_write_fd)
        parent_control.close()

    if primary_failure is not None:
        ordered_failures = [primary_failure, *cleanup_failures]
        fatal_failure = next(
            (
                failure
                for failure in ordered_failures
                if exception_tree_contains(
                    failure,
                    (KeyboardInterrupt, SystemExit, GeneratorExit),
                )
            ),
            None,
        )
        if fatal_failure is not None:
            secondary_failures = [
                failure for failure in ordered_failures if failure is not fatal_failure
            ]
            if deferred_cancellation is not None:
                secondary_failures.append(deferred_cancellation)
            if secondary_failures:
                raise fatal_failure from BaseExceptionGroup(
                    "Local execution failed while settlement retained additional signals.",
                    secondary_failures,
                )
            raise fatal_failure
        if deferred_cancellation is not None:
            restore_task_cancellation_requests(
                cancellation_requests_consumed,
                cancellation=deferred_cancellation,
            )
            raise deferred_cancellation from BaseExceptionGroup(
                "Local execution failed before cancellation reached settlement.",
                ordered_failures,
            )
        if cleanup_failures:
            if all(isinstance(failure, Exception) for failure in ordered_failures):
                exception_failures = [
                    failure for failure in ordered_failures if isinstance(failure, Exception)
                ]
                raise ExceptionGroup(
                    "Local execution failed and cleanup did not settle cleanly.",
                    exception_failures,
                )
            raise BaseExceptionGroup(
                "Local execution failed and cleanup did not settle cleanly.",
                ordered_failures,
            )
        raise primary_failure

    receipt = await _load_receipt_or_exact_durable_settlement(
        task_store=task_store,
        path=receipt_path,
        authority=authority,
    )
    settlement_task = asyncio.create_task(
        task_store.settle_local_execution_attempt(_authenticated_settlement(authority, receipt)),
        name=f"cayu-local-execution-final-settlement-{authority.attempt_id}",
    )
    settlement_outcome = await await_shielded_task_outcome(
        settlement_task,
        timeout_after_cancellation_s=(
            request.limits.term_grace_seconds + request.limits.kill_grace_seconds + 2.0
        ),
    )
    final_cancellation = (
        settlement_outcome.cancellation or settlement_outcome.subsequent_cancellation
    )
    if settlement_outcome.timed_out:
        _retain_local_execution_task(
            settlement_task,
            remove_path_on_success=receipt_path,
        )
        if stdout_task is not None:
            _retain_local_execution_task(stdout_task)
        if stderr_task is not None:
            _retain_local_execution_task(stderr_task)
        if final_cancellation is not None:
            restore_task_cancellation_requests(
                settlement_outcome.cancellation_requests_consumed,
                cancellation=final_cancellation,
            )
            raise final_cancellation from LocalExecutionAttemptUnsettled(
                "Local execution settlement remains in flight and retry stays fenced."
            )
        raise LocalExecutionAttemptUnsettled(
            "Local execution settlement remains in flight and retry stays fenced."
        )
    if settlement_outcome.error is not None:
        if stdout_task is not None:
            _retain_local_execution_task(stdout_task)
        if stderr_task is not None:
            _retain_local_execution_task(stderr_task)
        if final_cancellation is not None:
            restore_task_cancellation_requests(
                settlement_outcome.cancellation_requests_consumed,
                cancellation=final_cancellation,
            )
            if exception_tree_contains(
                settlement_outcome.error,
                (KeyboardInterrupt, SystemExit, GeneratorExit),
            ):
                raise settlement_outcome.error from final_cancellation
            raise final_cancellation from settlement_outcome.error
        raise settlement_outcome.error
    settled = settlement_outcome.result
    if settled is None:
        raise RuntimeError("Local execution settlement returned no durable record.")
    with suppress(OSError):
        receipt_path.unlink()
    if final_cancellation is not None:
        if stdout_task is not None:
            _retain_local_execution_task(stdout_task)
        if stderr_task is not None:
            _retain_local_execution_task(stderr_task)
        restore_task_cancellation_requests(
            settlement_outcome.cancellation_requests_consumed,
            cancellation=final_cancellation,
        )
        raise final_cancellation
    if settled.quiescence is LocalExecutionAttemptQuiescence.TERMINAL_NOT_QUIESCENT:
        if stdout_task is not None:
            _retain_local_execution_task(stdout_task)
        if stderr_task is not None:
            _retain_local_execution_task(stderr_task)
        return LocalExecutionAttemptResult(
            attempt=settled,
            stdout="",
            stderr="",
            stdout_truncated=True,
            stderr_truncated=True,
        )
    assert stdout_task is not None
    assert stderr_task is not None
    stdout, stdout_truncated = await stdout_task
    stderr, stderr_truncated = await stderr_task
    return LocalExecutionAttemptResult(
        attempt=settled,
        stdout=stdout,
        stderr=stderr,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
    )


async def _recover_owned_local_execution_attempt(
    *,
    task_store: TaskStore,
    state_dir: Path,
    worker_id: str,
    record: LocalExecutionAttemptRecord,
    deadline: float | None = None,
) -> LocalExecutionAttemptRecord | None:
    authority = record.authority
    receipt_path = _receipt_path(state_dir, authority)
    if record.receipt is not None:
        # Terminal evidence that did not prove containment remains visible to
        # drain/inspection, but it cannot be replaced by inference.
        return None
    receipt: LocalExecutionAttemptReceipt | None = None
    if receipt_path.exists():
        try:
            receipt = _load_receipt(receipt_path, authority)
        except LocalExecutionAttemptUnsettled:
            # A corrupt final path is not positive settlement evidence.  A
            # still-live supervisor remains authoritative and may replace it
            # atomically; otherwise exact durable state decides this attempt.
            receipt = None
        if receipt is not None:
            settled = await task_store.settle_local_execution_attempt(
                _authenticated_settlement(authority, receipt)
            )
            with suppress(OSError):
                receipt_path.unlink()
            return settled

    supervisor_state = (
        await _probe_rendezvous(record, state_dir=state_dir)
        if deadline is None
        else await _probe_rendezvous(record, state_dir=state_dir, deadline=deadline)
    )
    if supervisor_state is not None and supervisor_state.get("state") != "settled":
        return None
    if receipt_path.exists():
        try:
            receipt = _load_receipt(receipt_path, authority)
        except LocalExecutionAttemptUnsettled:
            receipt = None
    current_host_identity = local_execution_host_identity()
    current_boot_id = local_execution_boot_id()
    exact_machine = (
        record.start is not None
        and current_host_identity != "unavailable"
        and record.start.host_identity == current_host_identity
    )
    if (
        receipt is None
        and record.start is not None
        and _exact_process_is_live(record.start.supervisor)
    ):
        # Exact local PID authority may only retain the retry fence when
        # machine identity is unavailable; it never authorizes signaling or
        # quiescence inference. The supervisor closes its rendezvous listener
        # immediately before publishing the final receipt, so let that exact
        # publisher complete or reconcile the rename itself.
        return None
    if receipt is None:
        # A complete authenticated staging file is written only after the
        # supervisor has positively classified the process tree and fsynced
        # the receipt. It therefore remains authoritative if the supervisor
        # dies before the final rename.
        try:
            receipt = _promote_authenticated_staged_receipt(
                receipt_path,
                authority,
            )
        except LocalExecutionAttemptUnsettled:
            receipt = None
    if receipt is None:
        # The supervisor may have renamed the authenticated staging receipt
        # and exited after recovery's preceding final-path check but before
        # the exact-process liveness check. Once that exact supervisor is no
        # longer live and staging promotion misses, authenticate the final
        # path once more before synthesizing weaker recovery evidence.
        try:
            receipt = _load_receipt(receipt_path, authority)
        except LocalExecutionAttemptUnsettled:
            receipt = None
    if receipt is not None:
        settled = await task_store.settle_local_execution_attempt(
            _authenticated_settlement(authority, receipt)
        )
        with suppress(OSError):
            receipt_path.unlink()
        with suppress(OSError):
            receipt_path.with_name(f"{receipt_path.name}.staging").unlink()
        return settled
    try:
        claim = await task_store.claim_local_execution_attempt_recovery(
            LocalExecutionAttemptRecoveryClaim(
                attempt_id=authority.attempt_id,
                request_sha256=authority.request_sha256,
                recovery_owner_id=worker_id,
                expected_recovery_generation=record.recovery_generation,
            )
        )
    except LocalExecutionAttemptConflict:
        return None
    if claim.receipt is not None:
        # Another owner terminalized the record while this claim was being
        # acquired. It owns any corresponding filesystem evidence cleanup;
        # preserve late stronger evidence instead of deleting it here.
        return claim

    # Recovery ownership makes inferred settlement exclusive, but a stronger
    # runtime-authenticated supervisor receipt may still arrive across the
    # durable claim boundary. Reconcile that evidence before committing any
    # synthetic terminal classification.
    try:
        receipt = _load_receipt(receipt_path, authority)
    except LocalExecutionAttemptUnsettled:
        receipt = None
    if (
        receipt is None
        and claim.start is not None
        and _exact_process_is_live(claim.start.supervisor)
    ):
        return None
    if receipt is None:
        try:
            receipt = _promote_authenticated_staged_receipt(
                receipt_path,
                authority,
            )
        except LocalExecutionAttemptUnsettled:
            receipt = None
    if receipt is None:
        try:
            receipt = _load_receipt(receipt_path, authority)
        except LocalExecutionAttemptUnsettled:
            receipt = None
    if receipt is not None:
        settled = await task_store.settle_local_execution_attempt(
            _authenticated_settlement(authority, receipt)
        )
        with suppress(OSError):
            receipt_path.unlink()
        with suppress(OSError):
            receipt_path.with_name(f"{receipt_path.name}.staging").unlink()
        return settled

    if claim.start is None:
        receipt = _recovery_receipt(
            claim,
            reason="recovered_before_dispatch",
            quiescence="not_dispatched",
            effect_outcome="not_started",
        )
    elif (
        exact_machine
        and current_boot_id != "unavailable"
        and claim.start.boot_id != current_boot_id
    ):
        receipt = _recovery_receipt(
            claim,
            reason="host_reboot",
            quiescence="quiescent",
            effect_outcome="outcome_unknown",
        )
    else:
        receipt = _recovery_receipt(
            claim,
            reason="supervisor_authority_unavailable",
            quiescence="unavailable",
            effect_outcome="outcome_unknown",
        )
    settled = await task_store.settle_local_execution_attempt(
        _authenticated_settlement(
            authority,
            receipt,
            recovery_owner_id=worker_id,
            expected_recovery_generation=claim.recovery_generation,
        )
    )
    with suppress(OSError):
        receipt_path.unlink()
    with suppress(OSError):
        receipt_path.with_name(f"{receipt_path.name}.staging").unlink()
    return settled


async def recover_owned_local_execution_attempts(
    *,
    task_store: TaskStore,
    state_dir: Path,
    worker_id: str,
    limit: int,
    after: LocalExecutionAttemptListCursor | None = None,
    max_scanned: int = _MAX_LOCAL_EXECUTION_RECOVERY_SCAN_RECORDS,
    deadline: float | None = None,
) -> _LocalExecutionRecoveryScan:
    _prepare_state_dir(state_dir)
    recovered: list[LocalExecutionAttemptRecord] = []
    scanned = 0
    page_cursor = after
    while len(recovered) < limit and scanned < max_scanned:
        if deadline is not None and asyncio.get_running_loop().time() >= deadline:
            return _LocalExecutionRecoveryScan(
                records=tuple(recovered),
                after=page_cursor,
                reached_end=False,
                deadline_elapsed=True,
            )
        page_limit = min(max_scanned - scanned, max(limit, 32))
        records = await task_store.list_unsettled_local_execution_attempts(
            limit=page_limit,
            after=page_cursor,
        )
        if type(records) is not tuple or len(records) > page_limit:
            raise LocalExecutionAttemptConflict(
                "The task store returned an invalid local execution recovery page."
            )
        if not records:
            return _LocalExecutionRecoveryScan(
                records=tuple(recovered),
                after=page_cursor,
                reached_end=True,
                deadline_elapsed=False,
            )
        for record in records:
            if not isinstance(record, LocalExecutionAttemptRecord):
                raise LocalExecutionAttemptConflict(
                    "The task store returned an invalid local execution recovery record."
                )
            record_cursor = local_execution_attempt_list_cursor(record)
            if page_cursor is not None and (
                record_cursor.created_at,
                record_cursor.attempt_id,
            ) <= (page_cursor.created_at, page_cursor.attempt_id):
                raise LocalExecutionAttemptConflict(
                    "The task store returned a non-monotonic local execution recovery page."
                )
            if deadline is not None and asyncio.get_running_loop().time() >= deadline:
                return _LocalExecutionRecoveryScan(
                    records=tuple(recovered),
                    after=page_cursor,
                    reached_end=False,
                    deadline_elapsed=True,
                )
            try:
                settled = await _recover_owned_local_execution_attempt(
                    task_store=task_store,
                    state_dir=state_dir,
                    worker_id=worker_id,
                    record=record,
                    deadline=deadline,
                )
            except _LocalExecutionRecoveryDeadlineElapsed:
                return _LocalExecutionRecoveryScan(
                    records=tuple(recovered),
                    after=page_cursor,
                    reached_end=False,
                    deadline_elapsed=True,
                )
            page_cursor = record_cursor
            scanned += 1
            if settled is not None:
                recovered.append(settled)
                if len(recovered) == limit:
                    return _LocalExecutionRecoveryScan(
                        records=tuple(recovered),
                        after=page_cursor,
                        reached_end=False,
                        deadline_elapsed=False,
                    )
    return _LocalExecutionRecoveryScan(
        records=tuple(recovered),
        after=page_cursor,
        reached_end=False,
        deadline_elapsed=False,
    )


__all__ = ["recover_owned_local_execution_attempts", "run_owned_local_execution_attempt"]
