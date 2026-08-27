"""Linux process-tree owner for one process-isolated tool invocation."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Coroutine
from contextlib import suppress
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from math import isfinite
from typing import Any, BinaryIO, Final, cast
from uuid import UUID, uuid4

from cayu._exception_groups import (
    exception_cause,
    exception_tree_contains,
    iter_exception_tree,
    set_exception_cause,
)
from cayu._task_wait import (
    await_shielded_task_outcome,
    restore_task_cancellation_requests,
)
from cayu._validation import (
    canonical_durable_json_bytes,
    copy_bounded_durable_json_value,
    copy_durable_json_object,
    inspect_bounded_durable_json,
)
from cayu.core.isolated_tools import (
    MAX_ISOLATED_TOOL_JSON_NODES,
    ProcessIsolatedTool,
    ProcessIsolatedToolContext,
)
from cayu.core.tools import ToolContext, ToolResult, _runtime_tool_invocation_authority
from cayu.runtime._isolated_tool_protocol import (
    ISOLATED_TOOL_TERMINAL_FRAME_HEADER_BYTES,
    IsolatedToolChildErrorCode,
    IsolatedToolProtocolError,
    build_isolated_tool_request,
    decode_isolated_tool_response,
    isolated_tool_terminal_frame_payload_length,
)
from cayu.runtime.tool_gateway import validate_effective_tool_arguments
from cayu.vaults.redaction import SecretRedactor

_WORKER_MODULE: Final = "cayu.runtime._isolated_tool_worker"
_SUPERVISOR_SCRIPT: Final = os.path.join(
    os.path.dirname(__file__),
    "_isolated_tool_supervisor.py",
)
_CHILD_BASE_ENVIRONMENT: Final = {
    "LC_ALL": "C",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUTF8": "1",
}
_READ_CHUNK_BYTES: Final = 64 << 10
_CLEANUP_SETTLEMENT_HEADROOM_SECONDS: Final = 2.0
_TEMPORARY_DIRECTORY_SETTLEMENT_SECONDS: Final = 1.0
_PIPE_CLOSE_SETTLEMENT_SECONDS: Final = 0.5
_ISOLATED_TOOL_DISPATCH_RECORD_TYPE: Final = "cayu.isolated-tool-dispatch"
_ISOLATED_TOOL_DISPATCH_RECORD_VERSION: Final = 2
_ISOLATED_TOOL_DISPATCH_AUTHORITY_RECORD_TYPE: Final = "cayu.isolated-tool-dispatch-authority"
_ISOLATED_TOOL_DISPATCH_AUTHORITY_RECORD_VERSION: Final = 1
_ISOLATED_TOOL_DISPATCH_SETTLEMENT_RECORD_TYPE: Final = "cayu.isolated-tool-dispatch-settlement"
_ISOLATED_TOOL_DISPATCH_SETTLEMENT_RECORD_VERSION: Final = 1
_ISOLATED_TOOL_ZERO_DISPATCH_REASONS: Final = frozenset(
    {
        "caller_cancelled_before_admission",
        "hard_process_deadline_exceeded",
        "prior_process_cleanup_pending",
        "process_boundary_setup_failed",
        "spawn_failed",
    }
)
_PROBE_CHILD_SUBREAPER_ARGUMENT: Final = "--probe-child-subreaper"
_SUPERVISOR_SETTLEMENT_ACK_COMPLETED: Final = b"\x01\x00"
_SUPERVISOR_SETTLEMENT_ACK_FAILED: Final = b"\x01\x01"
_SUPERVISOR_WORKER_ADMISSION: Final = b"\x01"

_RETAINED_ISOLATED_TOOL_OWNERS: dict[
    asyncio.Task[Any],
    Callable[[], Coroutine[Any, Any, Any]] | None,
] = {}


class _ChildSubreaperProbeUnavailable(RuntimeError):
    """One capability probe failed without establishing a durable result."""


@dataclass(frozen=True, slots=True)
class _SupervisorSettlement:
    return_code: int
    supervisor_failed: bool


class IsolatedToolFailure(Exception):
    """Runtime-owned fixed diagnostic for one isolated execution failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"Isolated tool execution failed: {code}.")


class IsolatedToolPreDispatchFailure(IsolatedToolFailure):
    """Failure positively known to precede child invocation dispatch."""


class IsolatedToolInvalidOutput(IsolatedToolFailure):
    """Child output failed the bounded protocol or ToolResult contract."""


class IsolatedToolDeadlineExceeded(TimeoutError):
    """The isolated invocation exceeded its hard process deadline."""

    code = "hard_process_deadline_exceeded"

    def __init__(self) -> None:
        super().__init__("Isolated tool exceeded its hard process deadline.")


class IsolatedToolCleanupUnproven(IsolatedToolFailure):
    """The parent could not prove process-tree settlement within its bound."""


class IsolatedToolSettlementFailure(IsolatedToolFailure):
    """A primary terminal condition was followed by unproven cleanup."""

    def __init__(
        self,
        *,
        primary: BaseException,
        cleanup: IsolatedToolCleanupUnproven,
    ) -> None:
        self.primary = primary
        self.cleanup_code = cleanup.code
        if isinstance(primary, IsolatedToolDeadlineExceeded):
            self.primary_kind = "timeout"
        elif isinstance(primary, IsolatedToolInvalidOutput):
            self.primary_kind = "invalid_output"
        else:
            self.primary_kind = "execution_error"
        super().__init__("process_cleanup_unproven")


def _ordered_failure_evidence(
    message: str,
    *failures: BaseException | None,
) -> BaseException | None:
    """Retain distinct failures once while preserving their phase order."""

    ordered: list[BaseException] = []
    seen: set[int] = set()
    for failure in failures:
        if failure is None or id(failure) in seen:
            continue
        ordered.append(failure)
        seen.update(id(candidate) for candidate in iter_exception_tree(failure))
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    return BaseExceptionGroup(message, ordered)


def _post_terminal_failure(
    *,
    supervisor_failure: IsolatedToolFailure | None,
    terminal_failure: IsolatedToolFailure | None,
    diagnostic_failure: IsolatedToolFailure | None,
) -> BaseException | None:
    """Keep supervisor classification while retaining every later stream failure."""

    if supervisor_failure is None:
        return _ordered_failure_evidence(
            "Isolated tool terminal and diagnostic settlement failed.",
            terminal_failure,
            diagnostic_failure,
        )
    secondary = _ordered_failure_evidence(
        "Isolated tool terminal and diagnostic settlement also failed.",
        terminal_failure,
        diagnostic_failure,
    )
    if secondary is not None:
        set_exception_cause(supervisor_failure, secondary)
    return supervisor_failure


def _independent_cleanup_evidence(
    primary: BaseException,
    cleanup: BaseException | None,
) -> BaseException | None:
    """Remove duplicate or primary-induced cleanup representations."""

    if cleanup is None:
        return None
    primary_nodes = tuple(iter_exception_tree(primary))
    primary_node_ids = {id(candidate) for candidate in primary_nodes}
    primary_codes = {
        (type(candidate), candidate.code)
        for candidate in primary_nodes
        if isinstance(candidate, IsolatedToolFailure)
    }

    def independent(candidate: BaseException) -> BaseException | None:
        if id(candidate) in primary_node_ids:
            return None
        if (
            isinstance(candidate, IsolatedToolFailure)
            and (type(candidate), candidate.code) in primary_codes
        ):
            return None
        if isinstance(primary, IsolatedToolDeadlineExceeded) and (
            type(candidate) is IsolatedToolInvalidOutput
            and candidate.code
            in {
                "missing_terminal_output",
                "response_read_failed",
                "terminal_stream_unsettled",
            }
        ):
            return None
        if isinstance(candidate, BaseExceptionGroup):
            children = [
                filtered
                for child in candidate.exceptions
                if (filtered := independent(child)) is not None
            ]
            if not children:
                return None
            if len(children) == 1:
                return children[0]
            return BaseExceptionGroup(
                "Independent isolated tool cleanup failures.",
                children,
            )
        return candidate

    return independent(cleanup)


class _FileDescriptorOwner:
    """Idempotent owner of one parent-only inherited pipe descriptor."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream

    @classmethod
    def adopt(cls, descriptor: int, *, mode: str) -> _FileDescriptorOwner:
        try:
            return cls(cast("BinaryIO", os.fdopen(descriptor, mode, buffering=0)))
        except BaseException:
            with suppress(OSError):
                os.close(descriptor)
            raise

    @classmethod
    def create_pipe_with_owned_writer(cls) -> tuple[int, _FileDescriptorOwner]:
        read_descriptor, write_descriptor = os.pipe()
        try:
            return read_descriptor, cls.adopt(write_descriptor, mode="wb")
        except BaseException:
            with suppress(OSError):
                os.close(read_descriptor)
            raise

    @property
    def descriptor(self) -> int:
        descriptor = self._stream.fileno()
        if type(descriptor) is not int or descriptor < 0:
            raise IsolatedToolCleanupUnproven("parent_pipe_descriptor_invalid")
        return descriptor

    def close_best_effort(self) -> None:
        with suppress(OSError):
            self._stream.close()


class _SupervisorControlOwner:
    """Owner of one supervisor generation's admission and shutdown channel."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream

    @classmethod
    def create(cls) -> tuple[int, _SupervisorControlOwner]:
        read_descriptor, write_descriptor = os.pipe()
        try:
            stream = os.fdopen(write_descriptor, "wb", buffering=0)
        except BaseException:
            with suppress(OSError):
                os.close(read_descriptor)
            with suppress(OSError):
                os.close(write_descriptor)
            raise
        return read_descriptor, cls(stream)

    @classmethod
    def create_with_owned_reader(
        cls,
    ) -> tuple[_FileDescriptorOwner, _SupervisorControlOwner]:
        read_descriptor: int | None = None
        control_owner: _SupervisorControlOwner | None = None
        try:
            read_descriptor, control_owner = cls.create()
            read_owner = _FileDescriptorOwner.adopt(read_descriptor, mode="rb")
            read_descriptor = None
            return read_owner, control_owner
        except BaseException:
            if read_descriptor is not None:
                with suppress(OSError):
                    os.close(read_descriptor)
            if control_owner is not None:
                control_owner.close_best_effort()
            raise

    def request_shutdown(self) -> None:
        try:
            self._stream.close()
        except OSError as exc:
            raise IsolatedToolCleanupUnproven("supervisor_control_close_failed") from exc

    def admit_worker(self) -> None:
        """Authorize worker creation only after the supervisor handle is owned."""

        # The invocation owner records uncertainty before calling this method.
        # A signal or short write cannot be converted into positive
        # zero-dispatch evidence merely because the caller did not observe it.
        try:
            written = self._stream.write(_SUPERVISOR_WORKER_ADMISSION)
        except (OSError, ValueError):
            raise IsolatedToolPreDispatchFailure("worker_admission_failed") from None
        if written != len(_SUPERVISOR_WORKER_ADMISSION):
            raise IsolatedToolPreDispatchFailure("worker_admission_failed")

    def close_best_effort(self) -> None:
        with suppress(OSError):
            self._stream.close()


class _SupervisorSettlementProofOwner:
    """Exact read authority for one supervisor's post-reaping acknowledgement."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._supervisor_failed: bool | None = None
        self._invalid = False

    @classmethod
    def create(cls) -> _SupervisorSettlementProofOwner:
        return cls(tempfile.TemporaryFile(mode="w+b", prefix="cayu-isolated-settlement-"))

    @property
    def descriptor(self) -> int:
        descriptor = self._stream.fileno()
        if type(descriptor) is not int or descriptor < 0:
            raise IsolatedToolCleanupUnproven("supervisor_settlement_descriptor_invalid")
        return descriptor

    def require_after_exit(self) -> bool:
        if self._supervisor_failed is not None:
            self.close_best_effort()
            return self._supervisor_failed
        if self._invalid or self._stream.closed:
            raise IsolatedToolCleanupUnproven("supervisor_settlement_ack_missing")
        try:
            acknowledgement = os.pread(
                self.descriptor,
                max(
                    len(_SUPERVISOR_SETTLEMENT_ACK_COMPLETED),
                    len(_SUPERVISOR_SETTLEMENT_ACK_FAILED),
                )
                + 1,
                0,
            )
        except InterruptedError as exc:
            raise IsolatedToolCleanupUnproven("supervisor_settlement_ack_interrupted") from exc
        except OSError as exc:
            self._invalid = True
            self.close_best_effort()
            raise IsolatedToolCleanupUnproven("supervisor_settlement_ack_failed") from exc
        if acknowledgement == _SUPERVISOR_SETTLEMENT_ACK_COMPLETED:
            supervisor_failed = False
        elif acknowledgement == _SUPERVISOR_SETTLEMENT_ACK_FAILED:
            supervisor_failed = True
        else:
            self._invalid = True
            self.close_best_effort()
            raise IsolatedToolCleanupUnproven("supervisor_settlement_ack_missing")
        self._supervisor_failed = supervisor_failed
        self.close_best_effort()
        return supervisor_failed

    def close_best_effort(self) -> None:
        with suppress(OSError):
            self._stream.close()


class _LateSpawnSettlementOwner:
    """Single owner for a spawn that outlived the public execution deadline."""

    def __init__(
        self,
        *,
        spawn_task: asyncio.Task[asyncio.subprocess.Process],
        limits: Any,
        parent_result_write_owner: _FileDescriptorOwner | None,
        parent_control_read_owner: _FileDescriptorOwner | None,
        parent_control_owner: _SupervisorControlOwner | None,
        settlement_proof_owner: _SupervisorSettlementProofOwner,
    ) -> None:
        self._spawn_task = spawn_task
        self._limits = limits
        self._parent_result_write_owner = parent_result_write_owner
        self._parent_control_read_owner = parent_control_read_owner
        self._parent_control_owner = parent_control_owner
        self._settlement_proof_owner = settlement_proof_owner
        self._lock = asyncio.Lock()
        self._settled = False
        self._failure_code: str | None = None

    @property
    def settled(self) -> bool:
        """Return positive process-local evidence that the exact spawn settled."""

        return self._settled

    async def settle(self) -> None:
        """Settle the exact spawn once; concurrent foreground cleanup joins it."""

        async with self._lock:
            if self._settled:
                _retire_late_spawn_retained_tasks(self)
                if self._failure_code is not None:
                    raise IsolatedToolFailure(self._failure_code)
                return
            self._failure_code = await _settle_late_spawn(
                self._spawn_task,
                self._limits,
                parent_result_write_owner=self._parent_result_write_owner,
                parent_control_read_owner=self._parent_control_read_owner,
                parent_control_owner=self._parent_control_owner,
                settlement_proof_owner=self._settlement_proof_owner,
            )
            self._settled = True
            _retire_late_spawn_retained_tasks(self)
            if self._failure_code is not None:
                raise IsolatedToolFailure(self._failure_code)


def _child_subreaper_probe_succeeds() -> bool:
    """Prove the supervisor can enable and observe subreaper ownership."""

    try:
        return _child_subreaper_probe_succeeds_for_process(os.getpid())
    except _ChildSubreaperProbeUnavailable:
        return False


@lru_cache(maxsize=4)
def _child_subreaper_probe_succeeds_for_process(process_id: int) -> bool:
    """Cache host capability by process generation, including after fork."""

    del process_id
    try:
        completed = subprocess.run(
            [sys.executable, "-I", _SUPERVISOR_SCRIPT, _PROBE_CHILD_SUBREAPER_ARGUMENT],
            check=False,
            close_fds=True,
            env=_CHILD_BASE_ENVIRONMENT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        # functools caches return values but not raised exceptions.  Keep a
        # transient launch/deadline failure retryable in this process.
        raise _ChildSubreaperProbeUnavailable from exc
    if completed.returncode != 0:
        raise _ChildSubreaperProbeUnavailable
    return True


def _complete_process_tree_supervision_available() -> bool:
    """Return whether this host can retain children that escape their group."""

    return (
        sys.platform == "linux"
        and hasattr(os, "killpg")
        and os.path.exists(f"/proc/self/task/{os.getpid()}/children")
        and _child_subreaper_probe_succeeds()
    )


def isolated_tool_dispatch_storage_key(
    *,
    session_id: str,
    model_step_id: str,
    model_attempt_id: str,
    tool_round_id: str,
    tool_call_id: str,
    tool_name: str,
    idempotency_key: str,
) -> str:
    """Return the stable private operation key for one isolated dispatch boundary."""

    digest = sha256(
        canonical_durable_json_bytes(
            {
                "session_id": session_id,
                "model_step_id": model_step_id,
                "model_attempt_id": model_attempt_id,
                "tool_round_id": tool_round_id,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "idempotency_key": idempotency_key,
            },
            "isolated_tool_dispatch_identity",
        )
    ).hexdigest()
    return f"cayu:isolated-tool-dispatch:sha256:{digest}"


def isolated_tool_dispatch_authority_storage_key(
    *,
    session_id: str,
    model_step_id: str,
    model_attempt_id: str,
    tool_round_id: str,
    tool_call_id: str,
    tool_name: str,
    idempotency_key: str,
) -> str:
    """Return the stable key for independently reconstructed dispatch authority."""

    return (
        isolated_tool_dispatch_storage_key(
            session_id=session_id,
            model_step_id=model_step_id,
            model_attempt_id=model_attempt_id,
            tool_round_id=tool_round_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            idempotency_key=idempotency_key,
        )
        + ":authority"
    )


def isolated_tool_dispatch_settlement_storage_key(
    *,
    session_id: str,
    model_step_id: str,
    model_attempt_id: str,
    tool_round_id: str,
    tool_call_id: str,
    tool_name: str,
    idempotency_key: str,
) -> str:
    """Return the stable key for an exact isolated-dispatch settlement."""

    return (
        isolated_tool_dispatch_storage_key(
            session_id=session_id,
            model_step_id=model_step_id,
            model_attempt_id=model_attempt_id,
            tool_round_id=tool_round_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            idempotency_key=idempotency_key,
        )
        + ":settlement"
    )


def isolated_tool_dispatch_settlement_matches(
    record: object,
    *,
    dispatch_record: dict[str, Any],
) -> bool:
    """Authenticate positive zero-dispatch evidence for one preparation record."""

    if type(record) is not dict or set(record) != {
        "record_type",
        "version",
        "outcome",
        "reason",
        "dispatch_record_sha256",
    }:
        return False
    digest = (
        "sha256:"
        + sha256(
            canonical_durable_json_bytes(
                dispatch_record,
                "isolated_tool_dispatch_record",
            )
        ).hexdigest()
    )
    copied = cast("dict[str, Any]", record)
    return all(
        (
            copied.get("record_type") == _ISOLATED_TOOL_DISPATCH_SETTLEMENT_RECORD_TYPE,
            type(copied.get("version")) is int,
            copied.get("version") == _ISOLATED_TOOL_DISPATCH_SETTLEMENT_RECORD_VERSION,
            copied.get("outcome") == "worker_not_admitted",
            copied.get("reason") in _ISOLATED_TOOL_ZERO_DISPATCH_REASONS,
            copied.get("dispatch_record_sha256") == digest,
        )
    )


def isolated_tool_dispatch_authority_digests(
    record: object,
    *,
    session_id: str,
    parent_task_id: str | None,
    parent_run_epoch: int,
    model_step_id: str,
    model_attempt_id: str,
    tool_round_id: str,
    tool_call_id: str,
    tool_name: str,
    idempotency_key: str,
    execution_profile_fingerprint: str,
    environment_allocation_fingerprint: str | None,
) -> tuple[str, str] | None:
    """Return request/argument digests only for exact reconstructed authority."""

    expected = {
        "record_type",
        "version",
        "request_sha256",
        "session_id",
        "parent_task_id",
        "parent_run_epoch",
        "model_step_id",
        "model_attempt_id",
        "tool_round_id",
        "tool_call_id",
        "tool_name",
        "idempotency_key",
        "effective_arguments_sha256",
        "execution_profile_fingerprint",
        "environment_allocation_fingerprint",
    }
    if type(record) is not dict or set(record) != expected:
        return None
    copied = cast("dict[str, Any]", record)
    request_sha256 = copied.get("request_sha256")
    effective_arguments_sha256 = copied.get("effective_arguments_sha256")
    if (
        type(request_sha256) is not str
        or len(request_sha256) != 71
        or not request_sha256.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in request_sha256[7:])
        or type(effective_arguments_sha256) is not str
        or len(effective_arguments_sha256) != 64
        or any(character not in "0123456789abcdef" for character in effective_arguments_sha256)
    ):
        return None
    matches = all(
        (
            copied.get("record_type") == _ISOLATED_TOOL_DISPATCH_AUTHORITY_RECORD_TYPE,
            type(copied.get("version")) is int,
            copied.get("version") == _ISOLATED_TOOL_DISPATCH_AUTHORITY_RECORD_VERSION,
            copied.get("session_id") == session_id,
            copied.get("parent_task_id") == parent_task_id,
            type(copied.get("parent_run_epoch")) is int,
            copied.get("parent_run_epoch") == parent_run_epoch,
            copied.get("model_step_id") == model_step_id,
            copied.get("model_attempt_id") == model_attempt_id,
            copied.get("tool_round_id") == tool_round_id,
            copied.get("tool_call_id") == tool_call_id,
            copied.get("tool_name") == tool_name,
            copied.get("idempotency_key") == idempotency_key,
            copied.get("execution_profile_fingerprint") == execution_profile_fingerprint,
            copied.get("environment_allocation_fingerprint") == environment_allocation_fingerprint,
        )
    )
    return (request_sha256, effective_arguments_sha256) if matches else None


def isolated_tool_dispatch_record_matches(
    record: object,
    *,
    session_id: str,
    parent_task_id: str | None,
    parent_run_epoch: int,
    model_step_id: str,
    model_attempt_id: str,
    tool_round_id: str,
    tool_call_id: str,
    tool_name: str,
    idempotency_key: str,
    request_sha256: str,
    effective_arguments_sha256: str,
    execution_profile_fingerprint: str,
    environment_allocation_fingerprint: str | None,
) -> bool:
    """Authenticate one durable dispatch marker against its recovery authority."""

    if type(record) is not dict or set(record) != {
        "record_type",
        "version",
        "dispatch_owner_id",
        "request_sha256",
        "session_id",
        "parent_task_id",
        "parent_run_epoch",
        "model_step_id",
        "model_attempt_id",
        "tool_round_id",
        "tool_call_id",
        "tool_name",
        "idempotency_key",
        "effective_arguments_sha256",
        "execution_profile_fingerprint",
        "environment_allocation_fingerprint",
    }:
        return False
    copied = cast("dict[str, Any]", record)
    owner_id = copied.get("dispatch_owner_id")
    stored_request_sha256 = copied.get("request_sha256")
    arguments_sha256 = copied.get("effective_arguments_sha256")
    environment_fingerprint = copied.get("environment_allocation_fingerprint")
    if (
        type(owner_id) is not str
        or type(stored_request_sha256) is not str
        or len(stored_request_sha256) != 71
        or not stored_request_sha256.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in stored_request_sha256[7:])
        or type(arguments_sha256) is not str
        or len(arguments_sha256) != 64
        or any(character not in "0123456789abcdef" for character in arguments_sha256)
        or (environment_fingerprint is not None and type(environment_fingerprint) is not str)
    ):
        return False
    try:
        parsed_owner_id = UUID(owner_id)
    except (TypeError, ValueError):
        return False
    if str(parsed_owner_id) != owner_id:
        return False
    return all(
        (
            copied.get("record_type") == _ISOLATED_TOOL_DISPATCH_RECORD_TYPE,
            type(copied.get("version")) is int,
            copied.get("version") == _ISOLATED_TOOL_DISPATCH_RECORD_VERSION,
            copied.get("session_id") == session_id,
            copied.get("parent_task_id") == parent_task_id,
            type(copied.get("parent_run_epoch")) is int,
            copied.get("parent_run_epoch") == parent_run_epoch,
            copied.get("model_step_id") == model_step_id,
            copied.get("model_attempt_id") == model_attempt_id,
            copied.get("tool_round_id") == tool_round_id,
            copied.get("tool_call_id") == tool_call_id,
            copied.get("tool_name") == tool_name,
            copied.get("idempotency_key") == idempotency_key,
            copied.get("request_sha256") == request_sha256,
            copied.get("effective_arguments_sha256") == effective_arguments_sha256,
            copied.get("execution_profile_fingerprint") == execution_profile_fingerprint,
            copied.get("environment_allocation_fingerprint") == environment_allocation_fingerprint,
        )
    )


def validate_process_isolated_tool_registration(
    tool: ProcessIsolatedTool,
    *,
    redactor: SecretRedactor,
) -> None:
    """Fail closed before registration when the declared boundary is unavailable."""

    if type(tool) is not ProcessIsolatedTool:
        raise TypeError("Isolated tool registration requires an exact ProcessIsolatedTool.")
    if not _complete_process_tree_supervision_available():
        raise RuntimeError(
            "Hard process-isolated tools require Linux subreaper process-tree support."
        )
    if not isinstance(redactor, SecretRedactor):
        raise TypeError("redactor must be a SecretRedactor.")
    public_configuration = {
        "factory": tool.factory.model_dump(mode="json"),
        "factory_config": tool.factory_config_copy(),
        "environment": tool.environment_copy(),
        "context_projection": tool.context_projection.model_dump(mode="json"),
        "limits": tool.limits.model_dump(mode="json"),
    }
    try:
        redacted = redactor.redact_json(public_configuration)
    except Exception:
        raise ValueError("Isolated tool configuration could not be checked for secrets.") from None
    if redacted != public_configuration:
        raise ValueError("Isolated tool configuration cannot contain registered secrets.")


def isolated_tool_execution_contract(
    tool: ProcessIsolatedTool,
    *,
    runtime_timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Return bounded callable-free material for descriptors and manifests."""

    if type(tool) is not ProcessIsolatedTool:
        raise TypeError("tool must be an exact ProcessIsolatedTool.")
    if runtime_timeout_seconds is not None and (
        type(runtime_timeout_seconds) not in {int, float}
        or not isfinite(float(runtime_timeout_seconds))
        or runtime_timeout_seconds <= 0
    ):
        raise ValueError("runtime_timeout_seconds must be positive or None.")
    private_material = {
        "protocol": "cayu.isolated-tool",
        "protocol_version": 1,
        "factory_identity": tool.factory.identity.model_dump(mode="json"),
        "factory_module": tool.factory.module,
        "factory_qualname": tool.factory.qualname,
        "factory_config": tool.factory_config_copy(),
        "context_projection": tool.context_projection.model_dump(mode="json"),
        "environment": tool.environment_copy(),
        "limits": tool.limits.model_dump(mode="json"),
    }
    configuration_sha256 = (
        "sha256:"
        + sha256(
            canonical_durable_json_bytes(
                private_material,
                "isolated_tool_execution_configuration",
            )
        ).hexdigest()
    )
    material = {
        "boundary": "posix_process",
        "timeout_strength": "hard_process_deadline",
        "sandboxed": False,
        "adapter_identity": tool.factory.identity.model_dump(mode="json"),
        "adapter_configuration_sha256": configuration_sha256,
        "hard_deadline_seconds": min(
            tool.limits.deadline_seconds,
            (
                tool.limits.deadline_seconds
                if runtime_timeout_seconds is None
                else float(runtime_timeout_seconds)
            ),
        ),
        "protocol": "cayu.isolated-tool",
        "protocol_version": 1,
    }
    return copy_durable_json_object(material, "isolated_tool_execution_contract")


async def execute_process_isolated_tool(
    *,
    tool: ProcessIsolatedTool,
    context: ToolContext,
    arguments: dict[str, Any],
    registered_schema: dict[str, Any],
    redactor: SecretRedactor,
) -> ToolResult:
    """Execute one exact isolated tool after trusted-parent validation."""

    if type(tool) is not ProcessIsolatedTool:
        raise IsolatedToolPreDispatchFailure("adapter_type_invalid")
    if not _complete_process_tree_supervision_available():
        raise IsolatedToolPreDispatchFailure("platform_unsupported")
    authority = _runtime_tool_invocation_authority(context)
    if authority is None:
        raise IsolatedToolPreDispatchFailure("runtime_authority_missing")
    try:
        copied_arguments = copy_bounded_durable_json_value(
            arguments,
            "isolated_tool_arguments",
            max_bytes=tool.limits.max_request_bytes,
            max_nodes=MAX_ISOLATED_TOOL_JSON_NODES,
        )
    except Exception:
        raise IsolatedToolPreDispatchFailure("arguments_invalid_or_too_large") from None
    if type(copied_arguments) is not dict:
        raise IsolatedToolPreDispatchFailure("arguments_invalid")
    arguments_sha256 = sha256(
        canonical_durable_json_bytes(
            copied_arguments,
            "isolated_tool_arguments",
        )
    ).hexdigest()
    if (
        authority.tool_name != tool.spec.name
        or authority.effective_arguments_sha256 != arguments_sha256
        or context.idempotency_key != authority.idempotency_key
    ):
        raise IsolatedToolPreDispatchFailure("runtime_authority_mismatch")
    if _retained_isolated_tool_cleanup_pending():
        raise IsolatedToolPreDispatchFailure("prior_process_cleanup_pending")
    try:
        valid_arguments = validate_effective_tool_arguments(
            copied_arguments,
            registered_schema,
        )
    except Exception:
        raise IsolatedToolPreDispatchFailure("arguments_invalid") from None
    if not valid_arguments:
        raise IsolatedToolPreDispatchFailure("arguments_invalid")

    projected_context = _project_context(tool=tool, context=context)
    factory_config = tool.factory_config_copy()
    environment = tool.environment_copy()
    child_payload = {
        "arguments": copied_arguments,
        "context": projected_context.model_dump(mode="json"),
        "factory_config": factory_config,
        "environment": environment,
    }
    try:
        inspect_bounded_durable_json(
            child_payload,
            "isolated_tool_child_payload",
            max_bytes=tool.limits.max_request_bytes,
            max_nodes=MAX_ISOLATED_TOOL_JSON_NODES,
        )
    except Exception:
        raise IsolatedToolPreDispatchFailure("request_invalid_or_too_large") from None
    try:
        redacted = redactor.redact_json(child_payload)
    except Exception:
        raise IsolatedToolPreDispatchFailure("child_payload_secret_check_failed") from None
    if redacted != child_payload:
        raise IsolatedToolPreDispatchFailure("child_payload_contains_secret")

    authority_document = {
        "parent_task_id": authority.parent_task_id,
        "parent_run_epoch": authority.parent_run_epoch,
        "model_step_id": authority.model_step_id,
        "model_attempt_id": authority.model_attempt_id,
        "tool_round_id": authority.tool_round_id,
        "tool_call_id": authority.tool_call_id,
        "tool_name": authority.tool_name,
        "idempotency_key": authority.idempotency_key,
        "effective_arguments_sha256": authority.effective_arguments_sha256,
        "execution_profile_fingerprint": authority.execution_profile_fingerprint,
        "environment_allocation_fingerprint": authority.environment_allocation_fingerprint,
    }
    try:
        envelope, request_bytes = build_isolated_tool_request(
            session_id=context.session_id,
            authority=authority_document,
            factory=tool.factory,
            limits=tool.limits,
            factory_config=factory_config,
            arguments=copied_arguments,
            context=projected_context,
            environment=environment,
        )
    except Exception:
        raise IsolatedToolPreDispatchFailure("request_invalid_or_too_large") from None
    del child_payload, factory_config, environment, copied_arguments

    dispatch_key = isolated_tool_dispatch_storage_key(
        session_id=context.session_id,
        model_step_id=authority.model_step_id,
        model_attempt_id=authority.model_attempt_id,
        tool_round_id=authority.tool_round_id,
        tool_call_id=authority.tool_call_id,
        tool_name=authority.tool_name,
        idempotency_key=authority.idempotency_key,
    )
    dispatch_authority_key = isolated_tool_dispatch_authority_storage_key(
        session_id=context.session_id,
        model_step_id=authority.model_step_id,
        model_attempt_id=authority.model_attempt_id,
        tool_round_id=authority.tool_round_id,
        tool_call_id=authority.tool_call_id,
        tool_name=authority.tool_name,
        idempotency_key=authority.idempotency_key,
    )
    dispatch_settlement_key = isolated_tool_dispatch_settlement_storage_key(
        session_id=context.session_id,
        model_step_id=authority.model_step_id,
        model_attempt_id=authority.model_attempt_id,
        tool_round_id=authority.tool_round_id,
        tool_call_id=authority.tool_call_id,
        tool_name=authority.tool_name,
        idempotency_key=authority.idempotency_key,
    )
    dispatch_authority_record = copy_durable_json_object(
        {
            "record_type": _ISOLATED_TOOL_DISPATCH_AUTHORITY_RECORD_TYPE,
            "version": _ISOLATED_TOOL_DISPATCH_AUTHORITY_RECORD_VERSION,
            "request_sha256": envelope.request_sha256,
            "session_id": context.session_id,
            **authority_document,
        },
        "isolated_tool_dispatch_authority_record",
    )
    dispatch_record = copy_durable_json_object(
        {
            "record_type": _ISOLATED_TOOL_DISPATCH_RECORD_TYPE,
            "version": _ISOLATED_TOOL_DISPATCH_RECORD_VERSION,
            "dispatch_owner_id": str(uuid4()),
            "request_sha256": envelope.request_sha256,
            "session_id": context.session_id,
            **authority_document,
        },
        "isolated_tool_dispatch_record",
    )
    try:
        prior_dispatch = await authority.load_durable_operation(dispatch_key)
        prior_authority = await authority.load_durable_operation(dispatch_authority_key)
        prior_settlement = await authority.load_durable_operation(dispatch_settlement_key)
    except (asyncio.CancelledError, KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except BaseException:
        raise IsolatedToolPreDispatchFailure("dispatch_evidence_lookup_failed") from None
    if prior_dispatch is not None or prior_authority is not None or prior_settlement is not None:
        raise IsolatedToolFailure("prior_dispatch_recorded")

    async def record_dispatch() -> None:
        try:
            published = await authority.compare_and_set_durable_operation(
                dispatch_key,
                None,
                dispatch_record,
                {dispatch_authority_key: dispatch_authority_record},
            )
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except BaseException:
            try:
                published = await authority.load_durable_operation(dispatch_key)
            except (asyncio.CancelledError, KeyboardInterrupt, SystemExit, GeneratorExit):
                raise
            except BaseException:
                raise IsolatedToolFailure("dispatch_evidence_reconciliation_failed") from None
        if published != dispatch_record:
            raise IsolatedToolFailure("dispatch_evidence_conflict")
        try:
            published_authority = await authority.load_durable_operation(dispatch_authority_key)
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except BaseException:
            raise IsolatedToolFailure("dispatch_evidence_reconciliation_failed") from None
        if published_authority != dispatch_authority_record:
            raise IsolatedToolFailure("dispatch_evidence_conflict")

    async def record_zero_dispatch(reason: str) -> None:
        """Publish exact positive evidence that the admission byte never crossed."""

        if reason not in _ISOLATED_TOOL_ZERO_DISPATCH_REASONS:
            raise IsolatedToolFailure("dispatch_settlement_reason_invalid")
        dispatch_settlement_record = copy_durable_json_object(
            {
                "record_type": _ISOLATED_TOOL_DISPATCH_SETTLEMENT_RECORD_TYPE,
                "version": _ISOLATED_TOOL_DISPATCH_SETTLEMENT_RECORD_VERSION,
                "outcome": "worker_not_admitted",
                "reason": reason,
                "dispatch_record_sha256": "sha256:"
                + sha256(
                    canonical_durable_json_bytes(
                        dispatch_record,
                        "isolated_tool_dispatch_record",
                    )
                ).hexdigest(),
            },
            "isolated_tool_dispatch_settlement_record",
        )
        try:
            published = await authority.compare_and_set_durable_operation(
                dispatch_settlement_key,
                None,
                dispatch_settlement_record,
                {},
            )
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except BaseException:
            try:
                published = await authority.load_durable_operation(dispatch_settlement_key)
            except (asyncio.CancelledError, KeyboardInterrupt, SystemExit, GeneratorExit):
                raise
            except BaseException:
                raise IsolatedToolFailure("dispatch_settlement_reconciliation_failed") from None
        if published != dispatch_settlement_record:
            raise IsolatedToolFailure("dispatch_settlement_conflict")

    async def restore_cancellation_after_zero_dispatch(
        cancellation: asyncio.CancelledError,
        *,
        reconcile_preparation: bool,
    ) -> None:
        """Retain exact zero-dispatch evidence before restoring caller cancellation."""

        async def publish_settlement() -> None:
            if reconcile_preparation:
                try:
                    current_dispatch = await authority.load_durable_operation(dispatch_key)
                    current_authority = await authority.load_durable_operation(
                        dispatch_authority_key
                    )
                except (KeyboardInterrupt, SystemExit, GeneratorExit):
                    raise
                except BaseException:
                    raise IsolatedToolFailure("dispatch_settlement_reconciliation_failed") from None
                if current_dispatch is None and current_authority is None:
                    return
                if (
                    current_dispatch != dispatch_record
                    or current_authority != dispatch_authority_record
                ):
                    raise IsolatedToolFailure("dispatch_evidence_conflict")
            await record_zero_dispatch("caller_cancelled_before_admission")

        settlement_task = asyncio.create_task(
            publish_settlement(),
            name="cayu-isolated-tool-zero-dispatch-cancellation-settlement",
        )
        outcome = await await_shielded_task_outcome(
            settlement_task,
            cancellation=cancellation,
        )
        evidence = exception_cause(cancellation)
        if outcome.error is not None and (
            evidence is None
            or not any(candidate is outcome.error for candidate in iter_exception_tree(evidence))
        ):
            evidence = (
                outcome.error
                if evidence is None
                else BaseExceptionGroup(
                    "Isolated tool cancellation and zero-dispatch settlement evidence.",
                    [evidence, outcome.error],
                )
            )
        _restore_and_raise_cancellation(outcome, cause=evidence)
        raise AssertionError("Cancellation restoration must re-raise.") from None

    try:
        await record_dispatch()
    except asyncio.CancelledError as cancellation:
        await restore_cancellation_after_zero_dispatch(
            cancellation,
            reconcile_preparation=True,
        )
        raise AssertionError("Cancellation settlement must re-raise.") from None
    # Durable admission can block behind another store operation.  Recheck the
    # process-local cleanup fence after that await so a sibling invocation
    # cannot enter retained cleanup while this call is waiting and still let a
    # new child cross the hard-process boundary.  Bind positive zero-dispatch
    # evidence to the committed preparation record before reporting rejection.
    if _retained_isolated_tool_cleanup_pending():
        await record_zero_dispatch("prior_process_cleanup_pending")
        raise IsolatedToolPreDispatchFailure("prior_process_cleanup_pending")

    owner = _IsolatedToolProcessOwner(
        tool=tool,
        request_bytes=request_bytes,
        request_sha256=envelope.request_sha256,
    )
    try:
        return await owner.run()
    except asyncio.CancelledError as cancellation:
        if not owner.worker_admission_may_have_crossed and not exception_tree_contains(
            cancellation, (IsolatedToolCleanupUnproven,)
        ):
            await restore_cancellation_after_zero_dispatch(
                cancellation,
                reconcile_preparation=False,
            )
            raise AssertionError("Cancellation settlement must re-raise.") from None
        raise
    except (IsolatedToolPreDispatchFailure, IsolatedToolDeadlineExceeded) as error:
        if (
            not owner.worker_admission_may_have_crossed
            and error.code in _ISOLATED_TOOL_ZERO_DISPATCH_REASONS
        ):
            await record_zero_dispatch(error.code)
        raise
    except BaseException:
        # Cleanup can make supervisor health the public classification after a
        # pre-admission deadline or fence.  Preserve the earlier positive
        # zero-dispatch fact independently of that later classification.
        zero_dispatch_reason = owner.known_zero_dispatch_reason
        if not owner.worker_admission_may_have_crossed and zero_dispatch_reason is not None:
            await record_zero_dispatch(zero_dispatch_reason)
        raise


def _project_context(
    *,
    tool: ProcessIsolatedTool,
    context: ToolContext,
) -> ProcessIsolatedToolContext:
    projected: dict[str, Any] = {
        field_name: getattr(context, field_name) for field_name in tool.context_projection.fields
    }
    metadata = {
        key: context.metadata[key]
        for key in tool.context_projection.metadata_keys
        if key in context.metadata
    }
    projected["metadata"] = metadata
    try:
        bounded = copy_bounded_durable_json_value(
            projected,
            "isolated_tool_context_projection",
            max_bytes=tool.limits.max_request_bytes,
            max_nodes=MAX_ISOLATED_TOOL_JSON_NODES,
        )
        if type(bounded) is not dict:
            raise TypeError
        return ProcessIsolatedToolContext.model_validate(bounded)
    except Exception:
        raise IsolatedToolPreDispatchFailure("context_projection_invalid") from None


class _IsolatedToolProcessOwner:
    def __init__(
        self,
        *,
        tool: ProcessIsolatedTool,
        request_bytes: bytes,
        request_sha256: str,
    ) -> None:
        self._tool = tool
        self._request_bytes = request_bytes
        self._request_sha256 = request_sha256
        self._spawn_task: asyncio.Task[asyncio.subprocess.Process] | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._wait_task: asyncio.Task[int] | None = None
        self._io_tasks: set[asyncio.Task[Any]] = set()
        self._diagnostic_tasks: set[asyncio.Task[None]] = set()
        self._cleanup_task: asyncio.Task[BaseException | None] | None = None
        self._retained_cleanup_tasks: set[asyncio.Task[Any]] = set()
        self._terminal_read_task: asyncio.Task[None] | None = None
        self._temporary_directory: str | None = None
        self._result_read_fd: int | None = None
        self._result_write_owner: _FileDescriptorOwner | None = None
        self._control_read_owner: _FileDescriptorOwner | None = None
        self._control_owner: _SupervisorControlOwner | None = None
        self._settlement_proof_owner: _SupervisorSettlementProofOwner | None = None
        self._late_spawn_settlement_owner: _LateSpawnSettlementOwner | None = None
        self._worker_admission_may_have_crossed = False
        self._known_zero_dispatch_reason: str | None = None

    @property
    def worker_admission_may_have_crossed(self) -> bool:
        """Whether the irreversible supervisor admission byte may have crossed."""

        return self._worker_admission_may_have_crossed

    @property
    def known_zero_dispatch_reason(self) -> str | None:
        """Return the pre-admission outcome retained across later cleanup failure."""

        return self._known_zero_dispatch_reason

    async def run(self) -> ToolResult:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._tool.limits.deadline_seconds
        settlement_attempted = False
        try:
            await self._spawn(deadline=deadline)
            terminal = await self._exchange(deadline=deadline)
            if isinstance(terminal, IsolatedToolChildErrorCode):
                raise IsolatedToolFailure(f"child_{terminal.value}")
            settlement_attempted = True
            post_terminal_failure = await self._settle()
            if post_terminal_failure is not None:
                raise post_terminal_failure
        except asyncio.CancelledError as cancellation:
            if settlement_attempted:
                self._retain_pending_cleanup()
                raise
            await self._settle(cancellation=cancellation)
            raise AssertionError("Cancellation settlement must re-raise.") from None
        except BaseException as primary:
            if settlement_attempted:
                self._retain_pending_cleanup()
                raise
            if not self._worker_admission_may_have_crossed and (
                isinstance(primary, IsolatedToolDeadlineExceeded)
                or type(primary) is IsolatedToolPreDispatchFailure
            ):
                reason = primary.code
                if reason in _ISOLATED_TOOL_ZERO_DISPATCH_REASONS:
                    self._known_zero_dispatch_reason = reason
            try:
                post_primary_failure = await self._settle()
            except asyncio.CancelledError as cancellation:
                cause = cancellation.__cause__
                evidence: BaseException = (
                    primary
                    if cause is None
                    else BaseExceptionGroup(
                        "Isolated tool failure preceded caller cancellation and cleanup failure.",
                        [primary, cause],
                    )
                )
                raise cancellation from evidence
            except IsolatedToolCleanupUnproven as cleanup:
                if isinstance(primary, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                    prior_cause = exception_cause(primary)
                    cause: BaseException = cleanup
                    if prior_cause is not None:
                        cause = BaseExceptionGroup(
                            "Isolated tool process-control and cleanup evidence.",
                            [prior_cause, cleanup],
                        )
                    raise primary from cause
                settlement = IsolatedToolSettlementFailure(
                    primary=primary,
                    cleanup=cleanup,
                )
                raise settlement from BaseExceptionGroup(
                    "Isolated tool execution and cleanup failed.",
                    [primary, cleanup],
                )
            except BaseException as cleanup:
                if isinstance(cleanup, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                    if isinstance(primary, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                        prior_primary = exception_cause(primary)
                        process_control_evidence: BaseException = cleanup
                        if prior_primary is not None:
                            process_control_evidence = (
                                prior_primary
                                if any(
                                    candidate is cleanup
                                    for candidate in iter_exception_tree(prior_primary)
                                )
                                else BaseExceptionGroup(
                                    "Isolated tool process-control and cleanup evidence.",
                                    [prior_primary, cleanup],
                                )
                            )
                        raise primary from process_control_evidence
                    prior_cleanup = exception_cause(cleanup)
                    process_control_evidence = primary
                    if prior_cleanup is not None:
                        process_control_evidence = (
                            prior_cleanup
                            if any(
                                candidate is primary
                                for candidate in iter_exception_tree(prior_cleanup)
                            )
                            else BaseExceptionGroup(
                                "Isolated tool failure and process-control cleanup evidence.",
                                [primary, prior_cleanup],
                            )
                        )
                    raise cleanup from process_control_evidence
                raise BaseExceptionGroup(
                    "Isolated tool execution and cleanup failed.",
                    [primary, cleanup],
                ) from None
            if (
                type(post_primary_failure) is IsolatedToolFailure
                and post_primary_failure.code == "supervisor_failed"
            ):
                if exception_tree_contains(
                    primary,
                    (KeyboardInterrupt, SystemExit, GeneratorExit),
                ):
                    prior_cause = exception_cause(primary)
                    supervisor_evidence: BaseException = post_primary_failure
                    if prior_cause is not None and not any(
                        candidate is post_primary_failure
                        for candidate in iter_exception_tree(prior_cause)
                    ):
                        supervisor_evidence = BaseExceptionGroup(
                            "Isolated tool process-control and supervisor evidence.",
                            [prior_cause, post_primary_failure],
                        )
                    raise primary from supervisor_evidence
                prior_supervisor_evidence = exception_cause(post_primary_failure)
                independent_supervisor_evidence = _independent_cleanup_evidence(
                    primary,
                    prior_supervisor_evidence,
                )
                combined_evidence = _ordered_failure_evidence(
                    "Isolated tool execution and supervisor settlement failed.",
                    primary,
                    independent_supervisor_evidence,
                )
                raise post_primary_failure from combined_evidence
            if post_primary_failure is not None:
                independent_cleanup = _independent_cleanup_evidence(
                    primary,
                    post_primary_failure,
                )
                if independent_cleanup is None:
                    raise
                prior_primary = exception_cause(primary)
                evidence = _ordered_failure_evidence(
                    "Isolated tool execution and cleanup evidence.",
                    prior_primary,
                    independent_cleanup,
                )
                if exception_tree_contains(
                    primary,
                    (KeyboardInterrupt, SystemExit, GeneratorExit),
                ):
                    raise primary from evidence
                raise primary from evidence
            raise
        finally:
            self._request_bytes = b""

        if type(terminal) is not ToolResult:  # pragma: no cover - decoder invariant
            raise IsolatedToolInvalidOutput("response_type_invalid")
        return terminal

    async def _spawn(self, *, deadline: float) -> None:
        try:
            self._temporary_directory = tempfile.mkdtemp(prefix="cayu-isolated-tool-")
            (
                self._result_read_fd,
                self._result_write_owner,
            ) = _FileDescriptorOwner.create_pipe_with_owned_writer()
            (
                self._control_read_owner,
                self._control_owner,
            ) = _SupervisorControlOwner.create_with_owned_reader()
            self._settlement_proof_owner = _SupervisorSettlementProofOwner.create()
        except OSError:
            self._close_parent_pipe_fds()
            raise IsolatedToolPreDispatchFailure("process_boundary_setup_failed") from None
        environment = {**_CHILD_BASE_ENVIRONMENT, **self._tool.environment_copy()}
        spawn_task = asyncio.create_task(
            asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                _SUPERVISOR_SCRIPT,
                "--result-fd",
                str(self._result_write_owner.descriptor),
                "--control-fd",
                str(self._control_read_owner.descriptor),
                "--settlement-fd",
                str(self._settlement_proof_owner.descriptor),
                "--worker-module",
                _WORKER_MODULE,
                "--term-grace-seconds",
                str(self._tool.limits.term_grace_seconds),
                "--kill-grace-seconds",
                str(self._tool.limits.kill_grace_seconds),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._temporary_directory,
                env=environment,
                close_fds=True,
                pass_fds=(
                    self._result_write_owner.descriptor,
                    self._control_read_owner.descriptor,
                    self._settlement_proof_owner.descriptor,
                ),
                start_new_session=True,
            ),
            name="cayu-isolated-tool-spawn",
        )
        self._spawn_task = spawn_task
        outcome = await await_shielded_task_outcome(
            spawn_task,
            timeout_s=max(0.0, deadline - asyncio.get_running_loop().time()),
        )
        if outcome.timed_out:
            late_spawn_owner = _LateSpawnSettlementOwner(
                spawn_task=spawn_task,
                limits=self._tool.limits,
                parent_result_write_owner=self._result_write_owner,
                parent_control_read_owner=self._control_read_owner,
                parent_control_owner=self._control_owner,
                settlement_proof_owner=self._settlement_proof_owner,
            )
            # Publish the replacement owner before clearing any source field.
            # Foreground cleanup can therefore join it after process control at
            # every subsequent handoff point, including task construction and
            # retained-owner registration.
            self._late_spawn_settlement_owner = late_spawn_owner
            self._spawn_task = None
            self._result_write_owner = None
            self._control_read_owner = None
            self._control_owner = None
            _retain_task(
                asyncio.create_task(
                    late_spawn_owner.settle(),
                    name="cayu-isolated-tool-late-spawn-cleanup",
                ),
                retry_factory=late_spawn_owner.settle,
            )
            deadline_error = IsolatedToolDeadlineExceeded()
            if outcome.cancellation is not None:
                _restore_and_raise_cancellation(outcome, cause=deadline_error)
            raise deadline_error
        if outcome.error is not None:
            self._spawn_task = None
            self._close_parent_pipe_fds()
            if outcome.cancellation is not None:
                _restore_and_raise_cancellation(outcome, cause=IsolatedToolFailure("spawn_failed"))
            raise IsolatedToolPreDispatchFailure("spawn_failed") from None
        process = outcome.result
        if process is None:  # pragma: no cover - subprocess result invariant
            self._spawn_task = None
            raise AssertionError("Subprocess spawn produced no process.")
        self._process = process
        self._wait_task = asyncio.create_task(
            process.wait(),
            name="cayu-isolated-tool-process-wait",
        )
        self._spawn_task = None
        if self._result_write_owner is not None:
            self._result_write_owner.close_best_effort()
            self._result_write_owner = None
        if self._control_read_owner is not None:
            self._control_read_owner.close_best_effort()
            self._control_read_owner = None
        if outcome.cancellation is not None:
            _restore_and_raise_cancellation(outcome, cause=None)
        if asyncio.get_running_loop().time() >= deadline:
            raise IsolatedToolDeadlineExceeded()
        control_owner = self._control_owner
        if control_owner is None:  # pragma: no cover - construction invariant
            raise IsolatedToolCleanupUnproven("supervisor_control_owner_missing")
        if _retained_isolated_tool_cleanup_pending():
            raise IsolatedToolPreDispatchFailure("prior_process_cleanup_pending")
        # Copy uncertainty into the invocation owner before the irreversible
        # write so cleanup can retire the control stream without losing it.
        self._worker_admission_may_have_crossed = True
        control_owner.admit_worker()

    async def _exchange(
        self,
        *,
        deadline: float,
    ) -> ToolResult | IsolatedToolChildErrorCode:
        process = self._process
        if process is None or self._wait_task is None or self._result_read_fd is None:
            raise AssertionError("Isolated process exchange started before spawn completed.")
        if process.stdin is None or process.stdout is None or process.stderr is None:
            raise AssertionError("Isolated process pipes are unavailable.")

        write_task = asyncio.create_task(
            _write_request(process.stdin, self._request_bytes),
            name="cayu-isolated-tool-request-write",
        )
        terminal_ready: asyncio.Future[ToolResult | IsolatedToolChildErrorCode] = (
            asyncio.get_running_loop().create_future()
        )
        terminal_ready.add_done_callback(_consume_future_exception)
        result_task = asyncio.create_task(
            _read_terminal(
                self._result_read_fd,
                max_bytes=self._tool.limits.max_response_bytes,
                expected_request_sha256=self._request_sha256,
                terminal_ready=terminal_ready,
            ),
            name="cayu-isolated-tool-result-read",
        )
        self._terminal_read_task = result_task
        self._result_read_fd = None
        stdout_task = asyncio.create_task(
            _drain_bounded(process.stdout, self._tool.limits.max_stdout_bytes, "stdout"),
            name="cayu-isolated-tool-stdout-drain",
        )
        stderr_task = asyncio.create_task(
            _drain_bounded(process.stderr, self._tool.limits.max_stderr_bytes, "stderr"),
            name="cayu-isolated-tool-stderr-drain",
        )
        self._diagnostic_tasks.update({stdout_task, stderr_task})
        deadline_task = asyncio.create_task(
            asyncio.sleep(max(0.0, deadline - asyncio.get_running_loop().time())),
            name="cayu-isolated-tool-hard-deadline",
        )
        self._io_tasks.update({write_task, result_task, stdout_task, stderr_task, deadline_task})
        watched: set[asyncio.Future[Any]] = {
            write_task,
            terminal_ready,
            result_task,
            stdout_task,
            stderr_task,
            deadline_task,
            self._wait_task,
        }
        done: set[asyncio.Future[Any]] = set()
        try:
            while watched:
                done, _ = await asyncio.wait(watched, return_when=asyncio.FIRST_COMPLETED)
                if deadline_task in done:
                    raise IsolatedToolDeadlineExceeded()
                if self._wait_task in done and terminal_ready not in done:
                    remaining = max(0.0, deadline - asyncio.get_running_loop().time())
                    await asyncio.wait(
                        {terminal_ready},
                        timeout=min(0.05, remaining),
                    )
                    if not terminal_ready.done():
                        raise IsolatedToolFailure("child_exited_without_terminal_output")
                    done.add(terminal_ready)
                if write_task in done:
                    watched.remove(write_task)
                    try:
                        write_task.result()
                    except (KeyboardInterrupt, SystemExit, GeneratorExit):
                        raise
                    except BaseException:
                        raise IsolatedToolFailure("request_write_failed") from None
                for stream_task in (stdout_task, stderr_task):
                    if stream_task not in done:
                        continue
                    watched.discard(stream_task)
                    failure_code: str | None = None
                    try:
                        stream_task.result()
                    except (KeyboardInterrupt, SystemExit, GeneratorExit):
                        raise
                    except BaseException as error:
                        failure_code = (
                            error.code
                            if type(error) is IsolatedToolFailure
                            else "diagnostic_stream_failed"
                        )
                    if failure_code is not None:
                        raise IsolatedToolFailure(failure_code) from None
                if result_task in done:
                    watched.discard(result_task)
                    self._terminal_read_task = None
                    self._io_tasks.discard(result_task)
                    try:
                        result_task.result()
                    except IsolatedToolProtocolError as error:
                        if (
                            terminal_ready.done()
                            and not terminal_ready.cancelled()
                            and terminal_ready.exception() is error
                        ):
                            done.add(terminal_ready)
                        else:
                            raise IsolatedToolInvalidOutput(error.code) from None
                    except (KeyboardInterrupt, SystemExit, GeneratorExit):
                        raise
                    except BaseException as error:
                        if (
                            terminal_ready.done()
                            and not terminal_ready.cancelled()
                            and terminal_ready.exception() is error
                        ):
                            done.add(terminal_ready)
                        else:
                            raise IsolatedToolInvalidOutput("response_read_failed") from None
                    if not terminal_ready.done():  # pragma: no cover - reader invariant
                        raise IsolatedToolInvalidOutput("missing_terminal_output")
                    done.add(terminal_ready)
                if terminal_ready in done:
                    protocol_failure_code: str | None = None
                    try:
                        return terminal_ready.result()
                    except IsolatedToolProtocolError as error:
                        protocol_failure_code = error.code
                    except (KeyboardInterrupt, SystemExit, GeneratorExit):
                        raise
                    except BaseException:
                        protocol_failure_code = "response_read_failed"
                    if protocol_failure_code == "missing_terminal_output":
                        remaining = max(0.0, deadline - asyncio.get_running_loop().time())
                        await asyncio.wait(
                            {self._wait_task},
                            timeout=min(0.05, remaining),
                        )
                        if self._wait_task.done():
                            return_code = self._wait_task.result()
                            raise IsolatedToolFailure(
                                "child_signaled"
                                if return_code < 0
                                else "child_exited_without_terminal_output"
                            )
                    if protocol_failure_code is not None:
                        raise IsolatedToolInvalidOutput(protocol_failure_code) from None
        finally:
            deadline_task.cancel()
            done.clear()
            watched.clear()
            write_task = None
            result_task = None
            terminal_ready = asyncio.get_running_loop().create_future()
            terminal_ready.cancel()
            stdout_task = None
            stderr_task = None
            deadline_task = None
            stream_task = None
            process = None
        raise AssertionError("Isolated process exchange ended without a terminal result.")

    async def _settle(
        self,
        *,
        cancellation: asyncio.CancelledError | None = None,
    ) -> BaseException | None:
        cleanup_task = self._cleanup_task
        if cleanup_task is None:
            cleanup_task = asyncio.create_task(
                self._cleanup_impl(),
                name="cayu-isolated-tool-process-cleanup",
            )
            self._cleanup_task = cleanup_task
        timeout = (
            self._tool.limits.term_grace_seconds
            + self._tool.limits.kill_grace_seconds
            + _CLEANUP_SETTLEMENT_HEADROOM_SECONDS
        )
        try:
            outcome = await await_shielded_task_outcome(
                cleanup_task,
                cancellation=cancellation,
                timeout_s=timeout,
                timeout_after_cancellation_s=timeout,
            )
        except BaseException:
            # A supervisory signal can interrupt the settlement waiter without
            # cancelling the shielded cleanup task. Transfer that exact owner
            # before allowing control to leave this boundary.
            self._retain_pending_cleanup()
            raise
        if outcome.timed_out:
            self._retain_cleanup_task(cleanup_task)
            cleanup_error: BaseException = IsolatedToolCleanupUnproven(
                "process_cleanup_deadline_exceeded"
            )
        else:
            cleanup_error = outcome.error
            if type(cleanup_error) is IsolatedToolCleanupUnproven and (
                self._process is not None
                or self._spawn_task is not None
                or self._late_spawn_settlement_owner is not None
            ):
                # Exact process-control failures leave this owner holding the
                # supervisor generation. Retain a retry before publishing the
                # bounded failure so later dispatch remains fenced.
                self._start_retained_cleanup_retry()
            elif (
                cleanup_error is not None and type(cleanup_error) is not IsolatedToolCleanupUnproven
            ):
                # A terminal cleanup task no longer owns the process state it
                # abandoned. Install the exact retry owner before propagating
                # process control or publishing a bounded cleanup failure.
                self._start_retained_cleanup_retry()
                if exception_tree_contains(
                    cleanup_error,
                    (KeyboardInterrupt, SystemExit, GeneratorExit),
                ):
                    raise cleanup_error
                if type(cleanup_error) is not IsolatedToolCleanupUnproven:
                    cleanup_failure = IsolatedToolCleanupUnproven("process_cleanup_failed")
                    set_exception_cause(cleanup_failure, cleanup_error)
                    cleanup_error = cleanup_failure
        if outcome.cancellation is not None:
            _restore_and_raise_cancellation(
                outcome,
                cause=cleanup_error or outcome.result,
            )
        if cleanup_error is not None:
            raise cleanup_error
        return outcome.result

    def _retain_pending_cleanup(self) -> None:
        cleanup_task = self._cleanup_task
        if cleanup_task is not None and not cleanup_task.done():
            self._retain_cleanup_task(cleanup_task)

    def _retain_cleanup_task(self, cleanup_task: asyncio.Task[Any]) -> None:
        if cleanup_task in self._retained_cleanup_tasks:
            return
        _retain_task(cleanup_task, retry_factory=self._cleanup_impl)
        # Publish the process-global fence before the local dedupe marker.  A
        # process-control signal between these steps may cause a harmless
        # duplicate callback registration, but cannot make a later handoff
        # mistake an unpublished owner for an already retained one.
        self._retained_cleanup_tasks.add(cleanup_task)

    def _start_retained_cleanup_retry(self) -> None:
        cleanup_task = asyncio.create_task(
            self._cleanup_impl(),
            name="cayu-isolated-tool-process-cleanup-retry",
        )
        self._cleanup_task = cleanup_task
        self._retain_cleanup_task(cleanup_task)

    async def _cleanup_impl(self) -> BaseException | None:
        late_spawn_failure = await self._adopt_pending_spawn_for_cleanup()
        process = self._process
        wait_task = self._wait_task
        control_owner = self._control_owner
        settlement_proof_owner = self._settlement_proof_owner
        supervisor_failure = late_spawn_failure
        if process is not None and wait_task is not None:
            if settlement_proof_owner is None:
                raise IsolatedToolCleanupUnproven("supervisor_settlement_owner_missing")
            supervisor_settlement = await _settle_owned_supervisor(
                process,
                wait_task,
                control_owner=control_owner,
                settlement_proof_owner=settlement_proof_owner,
                term_grace_seconds=self._tool.limits.term_grace_seconds,
                kill_grace_seconds=self._tool.limits.kill_grace_seconds,
            )
            if supervisor_settlement.supervisor_failed:
                supervisor_failure = IsolatedToolFailure("supervisor_failed")
        elif control_owner is not None:
            control_owner.request_shutdown()
        if self._control_owner is control_owner:
            self._control_owner = None
        # Retire the process identity before dropping its replayable settlement
        # proof. Process-control delivered after proof validation can therefore
        # either replay that proof or observe that no process remains owned.
        self._process = None
        self._wait_task = None
        if (
            process is not None
            and wait_task is not None
            and self._settlement_proof_owner is settlement_proof_owner
        ):
            self._settlement_proof_owner = None

        self._close_parent_pipe_fds()
        terminal_failure = await self._settle_terminal_reader()
        pending_diagnostics = {task for task in self._diagnostic_tasks if not task.done()}
        if pending_diagnostics:
            await asyncio.wait(
                pending_diagnostics,
                timeout=_PIPE_CLOSE_SETTLEMENT_SECONDS,
            )
        pending_io = {task for task in self._io_tasks if not task.done()}
        current_loop = asyncio.get_running_loop()
        if any(task.get_loop() is not current_loop for task in pending_io):
            raise IsolatedToolCleanupUnproven("process_io_cleanup_loop_unavailable")
        for task in pending_io:
            task.cancel()
        if pending_io:
            await asyncio.gather(*pending_io, return_exceptions=True)
        for task in self._io_tasks - pending_io:
            if not task.cancelled():
                with suppress(BaseException):
                    task.exception()
        diagnostic_failure: IsolatedToolFailure | None = None
        for task in self._diagnostic_tasks:
            if task.cancelled():
                diagnostic_failure = IsolatedToolFailure("diagnostic_stream_unsettled")
                break
            error = task.exception()
            if error is None:
                continue
            diagnostic_failure = (
                error
                if type(error) is IsolatedToolFailure
                else IsolatedToolFailure("diagnostic_stream_failed")
            )
            break
        self._diagnostic_tasks.clear()
        self._io_tasks.clear()
        post_terminal_failure = _post_terminal_failure(
            supervisor_failure=supervisor_failure,
            terminal_failure=terminal_failure,
            diagnostic_failure=diagnostic_failure,
        )
        if self._temporary_directory is not None:
            directory = self._temporary_directory
            removal = asyncio.create_task(
                _remove_temporary_directory(directory),
                name="cayu-isolated-tool-directory-cleanup",
            )
            removal_outcome = await await_shielded_task_outcome(
                removal,
                timeout_s=_TEMPORARY_DIRECTORY_SETTLEMENT_SECONDS,
            )
            if removal_outcome.timed_out:
                self._temporary_directory = None
                _retain_task(
                    removal,
                    retry_factory=lambda: _remove_temporary_directory(directory),
                )
                directory_failure = IsolatedToolCleanupUnproven(
                    "temporary_directory_cleanup_deadline_exceeded"
                )
                if post_terminal_failure is not None:
                    raise directory_failure from post_terminal_failure
                raise directory_failure
            if removal_outcome.cancellation is not None:
                _restore_and_raise_cancellation(
                    removal_outcome,
                    cause=_ordered_failure_evidence(
                        "Isolated tool stream and directory settlement failed.",
                        post_terminal_failure,
                        removal_outcome.error,
                    ),
                )
            if removal_outcome.error is not None:  # pragma: no cover - retry owner invariant
                if post_terminal_failure is not None:
                    if exception_tree_contains(
                        removal_outcome.error,
                        (KeyboardInterrupt, SystemExit, GeneratorExit),
                    ):
                        raise removal_outcome.error from post_terminal_failure
                    raise BaseExceptionGroup(
                        "Isolated tool stream and directory settlement failed.",
                        [post_terminal_failure, removal_outcome.error],
                    ) from None
                raise removal_outcome.error
            self._temporary_directory = None
        # A normal-looking terminal frame cannot authenticate a supervisor
        # generation that itself reported an internal failure.
        return post_terminal_failure

    async def _adopt_pending_spawn_for_cleanup(self) -> IsolatedToolFailure | None:
        """Transfer a process-control-interrupted spawn into this lifecycle owner."""

        late_spawn_owner = self._late_spawn_settlement_owner
        if late_spawn_owner is not None:
            late_spawn_failure: IsolatedToolFailure | None = None
            try:
                await late_spawn_owner.settle()
            except IsolatedToolFailure as error:
                if (
                    type(late_spawn_owner) is not _LateSpawnSettlementOwner
                    or not late_spawn_owner.settled
                    or type(error) is not IsolatedToolFailure
                    or error.code != "supervisor_failed"
                ):
                    raise
                late_spawn_failure = error
            finally:
                if type(late_spawn_owner) is _LateSpawnSettlementOwner and late_spawn_owner.settled:
                    self._late_spawn_settlement_owner = None
                    # The late-spawn owner consumed these exact fields even when a
                    # signal interrupted the producer before it cleared its mirrors.
                    self._spawn_task = None
                    self._result_write_owner = None
                    self._control_read_owner = None
                    self._control_owner = None
                    self._settlement_proof_owner = None
            return late_spawn_failure
        spawn_task = self._spawn_task
        if spawn_task is None:
            return
        outcome = await await_shielded_task_outcome(spawn_task)
        if outcome.cancellation is not None:
            _restore_and_raise_cancellation(outcome, cause=outcome.error)
        if outcome.error is not None:
            self._spawn_task = None
            return
        process = outcome.result
        if process is None:  # pragma: no cover - subprocess result invariant
            self._spawn_task = None
            return
        if self._process is not None and self._process is not process:
            raise IsolatedToolCleanupUnproven("spawn_process_owner_conflict")
        self._process = process
        if self._wait_task is None:
            self._wait_task = asyncio.create_task(
                process.wait(),
                name="cayu-isolated-tool-process-wait",
            )
        if process.stdin is not None:
            stdin_close_task = asyncio.create_task(
                _close_process_stdin(process.stdin),
                name="cayu-isolated-tool-stdin-close",
            )
            self._io_tasks.add(stdin_close_task)
        for stream, limit, stream_name in (
            (process.stdout, self._tool.limits.max_stdout_bytes, "stdout"),
            (process.stderr, self._tool.limits.max_stderr_bytes, "stderr"),
        ):
            if stream is None:
                continue
            drain_task = asyncio.create_task(
                _drain_bounded(stream, limit, stream_name),
                name=f"cayu-isolated-tool-{stream_name}-drain",
            )
            self._diagnostic_tasks.add(drain_task)
            self._io_tasks.add(drain_task)
        self._spawn_task = None

    async def _settle_terminal_reader(self) -> IsolatedToolFailure | None:
        task = self._terminal_read_task
        if task is None:
            return None
        current_loop = asyncio.get_running_loop()
        if not task.done() and task.get_loop() is current_loop:
            await asyncio.wait({task}, timeout=_PIPE_CLOSE_SETTLEMENT_SECONDS)
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                self._terminal_read_task = None
                self._io_tasks.discard(task)
                return IsolatedToolInvalidOutput("terminal_stream_unsettled")
        if not task.done():
            return None
        self._terminal_read_task = None
        self._io_tasks.discard(task)
        if task.cancelled():
            return IsolatedToolInvalidOutput("terminal_stream_unsettled")
        try:
            task.result()
        except IsolatedToolProtocolError as error:
            return IsolatedToolInvalidOutput(error.code)
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except BaseException:
            return IsolatedToolInvalidOutput("response_read_failed")
        return None

    def _close_parent_pipe_fds(self) -> None:
        for attribute in ("_result_read_fd",):
            descriptor = getattr(self, attribute)
            if descriptor is None:
                continue
            with suppress(OSError):
                os.close(descriptor)
            setattr(self, attribute, None)
        for attribute in ("_result_write_owner", "_control_read_owner"):
            descriptor_owner = getattr(self, attribute)
            if descriptor_owner is None:
                continue
            descriptor_owner.close_best_effort()
            setattr(self, attribute, None)
        control_owner = self._control_owner
        if control_owner is not None:
            control_owner.close_best_effort()
            if self._control_owner is control_owner:
                self._control_owner = None
        settlement_proof_owner = self._settlement_proof_owner
        if settlement_proof_owner is not None:
            settlement_proof_owner.close_best_effort()
            if self._settlement_proof_owner is settlement_proof_owner:
                self._settlement_proof_owner = None


async def _write_request(writer: asyncio.StreamWriter, data: bytes) -> None:
    try:
        writer.write(data)
        await writer.drain()
    finally:
        writer.close()
        with suppress(BrokenPipeError, ConnectionResetError):
            await asyncio.wait_for(
                writer.wait_closed(),
                timeout=_PIPE_CLOSE_SETTLEMENT_SECONDS,
            )


async def _close_process_stdin(writer: asyncio.StreamWriter) -> None:
    writer.close()
    with suppress(BrokenPipeError, ConnectionResetError):
        await asyncio.wait_for(
            writer.wait_closed(),
            timeout=_PIPE_CLOSE_SETTLEMENT_SECONDS,
        )


def _consume_future_exception(future: asyncio.Future[Any]) -> None:
    if future.cancelled():
        return
    with suppress(BaseException):
        future.exception()


async def _read_terminal(
    descriptor: int,
    *,
    max_bytes: int,
    expected_request_sha256: str,
    terminal_ready: asyncio.Future[ToolResult | IsolatedToolChildErrorCode],
) -> None:
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(limit=max_bytes + ISOLATED_TOOL_TERMINAL_FRAME_HEADER_BYTES + 1)
    protocol = asyncio.StreamReaderProtocol(reader)
    pipe = os.fdopen(descriptor, "rb", buffering=0, closefd=True)
    transport: asyncio.ReadTransport | None = None
    try:
        transport, _ = await loop.connect_read_pipe(lambda: protocol, pipe)
        header = await _read_exact_terminal_bytes(
            reader,
            ISOLATED_TOOL_TERMINAL_FRAME_HEADER_BYTES,
            empty_code="missing_terminal_output",
        )
        payload_bytes = isolated_tool_terminal_frame_payload_length(
            header,
            max_bytes=max_bytes,
        )
        header = b""
        data = await _read_exact_terminal_bytes(
            reader,
            payload_bytes,
            empty_code="response_invalid",
        )
        try:
            terminal = decode_isolated_tool_response(
                data,
                expected_request_sha256=expected_request_sha256,
                max_bytes=max_bytes,
            )
        finally:
            data = b""
        terminal_ready.set_result(terminal)
        terminal = None
        trailing = await reader.read(1)
        if trailing:
            trailing = b""
            raise IsolatedToolProtocolError("response_invalid")
    except BaseException as error:
        if not terminal_ready.done():
            terminal_ready.set_exception(error)
        raise
    finally:
        if transport is None:
            pipe.close()
        else:
            transport.close()


async def _read_exact_terminal_bytes(
    reader: asyncio.StreamReader,
    count: int,
    *,
    empty_code: str,
) -> bytes:
    retained = bytearray()
    try:
        while len(retained) < count:
            chunk = await reader.read(min(_READ_CHUNK_BYTES, count - len(retained)))
            if not chunk:
                code = empty_code if not retained else "response_invalid"
                raise IsolatedToolProtocolError(code) from None
            retained.extend(chunk)
        return bytes(retained)
    finally:
        retained.clear()


async def _drain_bounded(
    reader: asyncio.StreamReader,
    max_bytes: int,
    stream_name: str,
) -> None:
    observed = 0
    while True:
        chunk = await reader.read(_READ_CHUNK_BYTES)
        if not chunk:
            return
        observed += len(chunk)
        if observed > max_bytes:
            raise IsolatedToolFailure(f"{stream_name}_too_large")


async def _remove_temporary_directory(directory: str) -> None:
    retry_delay = 0.05
    while True:
        try:
            await asyncio.to_thread(shutil.rmtree, directory)
        except FileNotFoundError:
            return
        except asyncio.CancelledError:
            raise
        except Exception:
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 1.0)
        else:
            return


async def _settle_owned_supervisor(
    process: asyncio.subprocess.Process,
    wait_task: asyncio.Task[int],
    *,
    control_owner: _SupervisorControlOwner | None,
    settlement_proof_owner: _SupervisorSettlementProofOwner,
    term_grace_seconds: float,
    kill_grace_seconds: float,
) -> _SupervisorSettlement:
    if control_owner is not None:
        control_owner.request_shutdown()
    retry_delay = 0.05
    while True:
        completed_settlement = _completed_process_wait_settlement(
            process,
            wait_task,
            settlement_proof_owner=settlement_proof_owner,
        )
        if completed_settlement is not None:
            return completed_settlement
        try:
            settlement = await _wait_for_supervisor(
                process,
                wait_task,
                settlement_proof_owner=settlement_proof_owner,
                timeout=term_grace_seconds,
            )
            if settlement is None:
                settlement = await _wait_for_supervisor(
                    process,
                    wait_task,
                    settlement_proof_owner=settlement_proof_owner,
                    timeout=kill_grace_seconds,
                )
            if settlement is not None:
                return settlement
        except asyncio.CancelledError:
            raise
        except IsolatedToolCleanupUnproven:
            # A transient wait failure remains retryable while the exact
            # supervisor is pending. A completed wait is authoritative;
            # cancellation or failure remains unproven and escapes this owner.
            completed_settlement = _completed_process_wait_settlement(
                process,
                wait_task,
                settlement_proof_owner=settlement_proof_owner,
            )
            if completed_settlement is not None:
                return completed_settlement
        await asyncio.sleep(retry_delay)
        retry_delay = min(retry_delay * 2, 1.0)


def _completed_process_wait_settlement(
    process: asyncio.subprocess.Process,
    wait_task: asyncio.Task[int],
    *,
    settlement_proof_owner: _SupervisorSettlementProofOwner,
) -> _SupervisorSettlement | None:
    """Return the supervisor outcome after exact post-reaping proof."""

    if not wait_task.done():
        return None
    if wait_task.cancelled():
        raise IsolatedToolCleanupUnproven("process_wait_cancelled")
    try:
        waited_return_code = wait_task.result()
    except BaseException as exc:
        raise IsolatedToolCleanupUnproven("process_wait_failed") from exc
    if type(waited_return_code) is not int:
        raise IsolatedToolCleanupUnproven("process_wait_result_invalid")
    observed_return_code = process.returncode
    if observed_return_code is not None:
        if type(observed_return_code) is not int:
            raise IsolatedToolCleanupUnproven("process_returncode_invalid")
        if observed_return_code != waited_return_code:
            raise IsolatedToolCleanupUnproven("process_wait_result_conflict")
    supervisor_failed = settlement_proof_owner.require_after_exit()
    return _SupervisorSettlement(
        return_code=waited_return_code,
        supervisor_failed=supervisor_failed,
    )


async def _wait_for_supervisor(
    process: asyncio.subprocess.Process,
    wait_task: asyncio.Task[int],
    *,
    settlement_proof_owner: _SupervisorSettlementProofOwner,
    timeout: float,
) -> _SupervisorSettlement | None:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        completed_settlement = _completed_process_wait_settlement(
            process,
            wait_task,
            settlement_proof_owner=settlement_proof_owner,
        )
        if completed_settlement is not None:
            return completed_settlement
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return None
        await asyncio.sleep(min(0.01, remaining))


async def _settle_late_spawn(
    spawn_task: asyncio.Task[asyncio.subprocess.Process],
    limits: Any,
    *,
    parent_result_write_owner: _FileDescriptorOwner | None,
    parent_control_read_owner: _FileDescriptorOwner | None,
    parent_control_owner: _SupervisorControlOwner | None,
    settlement_proof_owner: _SupervisorSettlementProofOwner,
) -> str | None:
    # The hard deadline owns the shutdown decision, not delivery of the
    # delayed ``Process`` handle. Closing the exact inherited channel first
    # prevents worker admission while the parent is still waiting and asks an
    # already-admitted supervisor to settle. The owner remains usable for
    # idempotent settlement retries.
    if parent_control_owner is not None:
        parent_control_owner.request_shutdown()
    try:
        try:
            outcome = await await_shielded_task_outcome(spawn_task)
        except BaseException:
            if parent_control_owner is not None:
                parent_control_owner.close_best_effort()
            raise
    finally:
        if parent_result_write_owner is not None:
            parent_result_write_owner.close_best_effort()
        if parent_control_read_owner is not None:
            parent_control_read_owner.close_best_effort()
    if outcome.error is not None:
        if parent_control_owner is not None:
            parent_control_owner.close_best_effort()
        settlement_proof_owner.close_best_effort()
        if isinstance(outcome.error, Exception):
            return
        raise IsolatedToolCleanupUnproven("late_spawn_outcome_unproven") from None
    process = outcome.result
    if process is None:
        if parent_control_owner is not None:
            parent_control_owner.close_best_effort()
        settlement_proof_owner.close_best_effort()
        return
    diagnostic_drains: list[asyncio.Task[None]] = []
    try:
        if process.stdin is not None:
            process.stdin.close()
        diagnostic_drains = [
            asyncio.create_task(_drain_bounded(stream, limit, stream_name))
            for stream, limit, stream_name in (
                (process.stdout, limits.max_stdout_bytes, "stdout"),
                (process.stderr, limits.max_stderr_bytes, "stderr"),
            )
            if stream is not None
        ]
        wait_task = asyncio.create_task(process.wait())
        settlement = await _settle_owned_supervisor(
            process,
            wait_task,
            control_owner=parent_control_owner,
            settlement_proof_owner=settlement_proof_owner,
            term_grace_seconds=limits.term_grace_seconds,
            kill_grace_seconds=limits.kill_grace_seconds,
        )
        return "supervisor_failed" if settlement.supervisor_failed else None
    finally:
        for task in diagnostic_drains:
            if not task.done():
                task.cancel()
        if diagnostic_drains:
            await asyncio.gather(*diagnostic_drains, return_exceptions=True)


def _retain_task(
    task: asyncio.Task[Any],
    *,
    retry_factory: Callable[[], Coroutine[Any, Any, Any]] | None = None,
) -> None:
    # Task and retry ownership form one admission-fencing record.  Publishing
    # them through one mapping assignment prevents interruption from exposing
    # a retained task without the retry owner needed to settle it.
    if retry_factory is not None or task not in _RETAINED_ISOLATED_TOOL_OWNERS:
        _RETAINED_ISOLATED_TOOL_OWNERS[task] = retry_factory

    def discard(completed: asyncio.Task[Any]) -> None:
        if _retained_task_completed_successfully(completed):
            _discard_retained_task(completed)

    task.add_done_callback(discard)


def _late_spawn_retry_owner(
    retry_factory: Callable[[], Coroutine[Any, Any, Any]] | None,
) -> _LateSpawnSettlementOwner | None:
    if retry_factory is None:
        return None
    owner = getattr(retry_factory, "__self__", None)
    function = getattr(retry_factory, "__func__", None)
    if type(owner) is _LateSpawnSettlementOwner and function is _LateSpawnSettlementOwner.settle:
        return owner
    return None


def _retire_late_spawn_retained_tasks(owner: _LateSpawnSettlementOwner) -> None:
    """Remove every retained mirror after any joiner proves exact settlement."""

    for task, retry_factory in tuple(_RETAINED_ISOLATED_TOOL_OWNERS.items()):
        if _late_spawn_retry_owner(retry_factory) is owner:
            _discard_retained_task(task)


def _retained_task_completed_successfully(task: asyncio.Task[Any]) -> bool:
    if not task.done() or task.cancelled():
        return False
    try:
        return task.exception() is None
    except BaseException:
        return False


def _discard_retained_task(task: asyncio.Task[Any]) -> None:
    _RETAINED_ISOLATED_TOOL_OWNERS.pop(task, None)


def _retained_isolated_tool_cleanup_pending() -> bool:
    """Return positive process-local evidence of an unresolved cleanup owner."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    for task, retry_factory in tuple(_RETAINED_ISOLATED_TOOL_OWNERS.items()):
        late_spawn_owner = _late_spawn_retry_owner(retry_factory)
        if late_spawn_owner is not None and late_spawn_owner.settled:
            # A signal can interrupt the successful joiner between recording
            # proof and retiring its task mirrors.  The admission fence must
            # still derive from the shared owner's positive state.
            _discard_retained_task(task)
            continue
        if _retained_task_completed_successfully(task):
            _discard_retained_task(task)
            continue
        if not task.done() or retry_factory is None or loop is None:
            continue
        try:
            retry_task = loop.create_task(
                retry_factory(),
                name="cayu-isolated-tool-retained-cleanup-retry",
            )
        except BaseException:
            continue
        _retain_task(retry_task, retry_factory=retry_factory)
        # Publish the replacement before retiring the completed owner so an
        # interrupt cannot temporarily remove the process-global fence.
        _discard_retained_task(task)
    return bool(_RETAINED_ISOLATED_TOOL_OWNERS)


def _restore_and_raise_cancellation(outcome: Any, *, cause: BaseException | None) -> None:
    cancellation = outcome.cancellation
    if cancellation is None:  # pragma: no cover - caller invariant
        raise AssertionError("Cancellation restoration requires a retained cancellation.")
    restore_task_cancellation_requests(
        outcome.cancellation_requests_consumed,
        cancellation=cancellation,
    )
    raise cancellation from cause


__all__ = [
    "IsolatedToolCleanupUnproven",
    "IsolatedToolDeadlineExceeded",
    "IsolatedToolFailure",
    "IsolatedToolInvalidOutput",
    "IsolatedToolPreDispatchFailure",
    "IsolatedToolSettlementFailure",
    "execute_process_isolated_tool",
    "isolated_tool_execution_contract",
    "validate_process_isolated_tool_registration",
]
