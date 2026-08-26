"""POSIX lifecycle owner for one process-isolated tool invocation."""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import sys
import tempfile
from collections.abc import Callable, Coroutine
from contextlib import suppress
from hashlib import sha256
from math import isfinite
from typing import Any, Final, cast
from uuid import UUID, uuid4

from cayu._exception_groups import (
    exception_cause,
    exception_tree_contains,
    iter_exception_tree,
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
_ISOLATED_TOOL_DISPATCH_RECORD_VERSION: Final = 1

_RETAINED_ISOLATED_TOOL_TASKS: set[asyncio.Task[Any]] = set()
_RETAINED_ISOLATED_TOOL_RETRIES: dict[
    asyncio.Task[Any],
    Callable[[], Coroutine[Any, Any, Any]],
] = {}


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
    """The parent could not prove process-group settlement within its bound."""


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
    execution_profile_fingerprint: str,
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
    request_sha256 = copied.get("request_sha256")
    arguments_sha256 = copied.get("effective_arguments_sha256")
    environment_fingerprint = copied.get("environment_allocation_fingerprint")
    if (
        type(owner_id) is not str
        or type(request_sha256) is not str
        or len(request_sha256) != 71
        or not request_sha256.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in request_sha256[7:])
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
            copied.get("execution_profile_fingerprint") == execution_profile_fingerprint,
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
    if os.name != "posix" or not hasattr(os, "killpg"):
        raise RuntimeError("Hard process-isolated tools require POSIX process-group support.")
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
    if os.name != "posix" or not hasattr(os, "killpg"):
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
    except (asyncio.CancelledError, KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except BaseException:
        raise IsolatedToolPreDispatchFailure("dispatch_evidence_lookup_failed") from None
    if prior_dispatch is not None:
        raise IsolatedToolFailure("prior_dispatch_recorded")

    async def record_dispatch() -> None:
        try:
            published = await authority.compare_and_set_durable_operation(
                dispatch_key,
                None,
                dispatch_record,
                {},
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

    await record_dispatch()
    # Durable admission can block behind another store operation.  Recheck the
    # process-local cleanup fence after that await so a sibling invocation
    # cannot enter retained cleanup while this call is waiting and still let a
    # new child cross the hard-process boundary.  A committed dispatch marker
    # remains conservative recovery evidence if this final gate rejects.
    if _retained_isolated_tool_cleanup_pending():
        raise IsolatedToolPreDispatchFailure("prior_process_cleanup_pending")

    owner = _IsolatedToolProcessOwner(
        tool=tool,
        request_bytes=request_bytes,
        request_sha256=envelope.request_sha256,
    )
    return await owner.run()


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
        self._cleanup_task: asyncio.Task[IsolatedToolFailure | None] | None = None
        self._retained_cleanup_tasks: set[asyncio.Task[Any]] = set()
        self._terminal_read_task: asyncio.Task[None] | None = None
        self._temporary_directory: str | None = None
        self._result_read_fd: int | None = None
        self._result_write_fd: int | None = None

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
            try:
                await self._settle()
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
            raise
        finally:
            self._request_bytes = b""

        if type(terminal) is not ToolResult:  # pragma: no cover - decoder invariant
            raise IsolatedToolInvalidOutput("response_type_invalid")
        return terminal

    async def _spawn(self, *, deadline: float) -> None:
        try:
            self._temporary_directory = tempfile.mkdtemp(prefix="cayu-isolated-tool-")
            self._result_read_fd, self._result_write_fd = os.pipe()
        except OSError:
            raise IsolatedToolPreDispatchFailure("process_boundary_setup_failed") from None
        environment = {**_CHILD_BASE_ENVIRONMENT, **self._tool.environment_copy()}
        spawn_task = asyncio.create_task(
            asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                "-m",
                _WORKER_MODULE,
                "--result-fd",
                str(self._result_write_fd),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._temporary_directory,
                env=environment,
                close_fds=True,
                pass_fds=(self._result_write_fd,),
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
            self._spawn_task = None
            result_write_fd = self._result_write_fd
            self._result_write_fd = None
            if self._result_read_fd is not None:
                with suppress(OSError):
                    os.close(self._result_read_fd)
                self._result_read_fd = None
            _retain_task(
                asyncio.create_task(
                    _settle_late_spawn(
                        spawn_task,
                        self._tool.limits,
                        parent_result_write_fd=result_write_fd,
                    ),
                    name="cayu-isolated-tool-late-spawn-cleanup",
                ),
                retry_factory=lambda: _settle_late_spawn(
                    spawn_task,
                    self._tool.limits,
                    parent_result_write_fd=None,
                ),
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
        if self._result_write_fd is not None:
            os.close(self._result_write_fd)
            self._result_write_fd = None
        if outcome.cancellation is not None:
            _restore_and_raise_cancellation(outcome, cause=None)

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
    ) -> IsolatedToolFailure | None:
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(
                self._cleanup_impl(),
                name="cayu-isolated-tool-process-cleanup",
            )
        timeout = (
            self._tool.limits.term_grace_seconds
            + self._tool.limits.kill_grace_seconds
            + _CLEANUP_SETTLEMENT_HEADROOM_SECONDS
        )
        try:
            outcome = await await_shielded_task_outcome(
                self._cleanup_task,
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
            self._retain_cleanup_task(self._cleanup_task)
            cleanup_error: BaseException = IsolatedToolCleanupUnproven(
                "process_cleanup_deadline_exceeded"
            )
        else:
            cleanup_error = outcome.error
            if cleanup_error is not None and type(cleanup_error) is not IsolatedToolCleanupUnproven:
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
                    cleanup_error = IsolatedToolCleanupUnproven("process_cleanup_failed")
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
        self._retained_cleanup_tasks.add(cleanup_task)
        _retain_task(cleanup_task, retry_factory=self._cleanup_impl)

    def _start_retained_cleanup_retry(self) -> None:
        self._cleanup_task = asyncio.create_task(
            self._cleanup_impl(),
            name="cayu-isolated-tool-process-cleanup-retry",
        )
        self._retain_cleanup_task(self._cleanup_task)

    async def _cleanup_impl(self) -> IsolatedToolFailure | None:
        await self._adopt_pending_spawn_for_cleanup()
        process = self._process
        wait_task = self._wait_task
        if process is not None and wait_task is not None:
            await _settle_owned_process_group(
                process.pid,
                wait_task,
                term_grace_seconds=self._tool.limits.term_grace_seconds,
                kill_grace_seconds=self._tool.limits.kill_grace_seconds,
            )
        self._process = None
        self._wait_task = None

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
            diagnostic_failure = IsolatedToolFailure(
                error.code if type(error) is IsolatedToolFailure else "diagnostic_stream_failed"
            )
            break
        self._diagnostic_tasks.clear()
        self._io_tasks.clear()
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
                raise IsolatedToolCleanupUnproven("temporary_directory_cleanup_deadline_exceeded")
            if removal_outcome.cancellation is not None:
                _restore_and_raise_cancellation(
                    removal_outcome,
                    cause=removal_outcome.error,
                )
            if removal_outcome.error is not None:  # pragma: no cover - retry owner invariant
                raise removal_outcome.error
            self._temporary_directory = None
        return terminal_failure or diagnostic_failure

    async def _adopt_pending_spawn_for_cleanup(self) -> None:
        """Transfer a process-control-interrupted spawn into this lifecycle owner."""

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
        for attribute in ("_result_read_fd", "_result_write_fd"):
            descriptor = getattr(self, attribute)
            if descriptor is None:
                continue
            with suppress(OSError):
                os.close(descriptor)
            setattr(self, attribute, None)


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


def _signal_process_group(process_group_id: int, selected_signal: signal.Signals) -> None:
    try:
        os.killpg(process_group_id, selected_signal)
    except ProcessLookupError:
        return
    except OSError as exc:
        raise IsolatedToolCleanupUnproven("process_group_signal_failed") from exc


async def _settle_owned_process_group(
    process_group_id: int,
    wait_task: asyncio.Task[int],
    *,
    term_grace_seconds: float,
    kill_grace_seconds: float,
) -> None:
    retry_delay = 0.05
    while True:
        try:
            _signal_process_group(process_group_id, signal.SIGTERM)
            settled = await _wait_for_group_and_child(
                process_group_id,
                wait_task,
                timeout=term_grace_seconds,
            )
            if not settled:
                _signal_process_group(process_group_id, signal.SIGKILL)
                settled = await _wait_for_group_and_child(
                    process_group_id,
                    wait_task,
                    timeout=kill_grace_seconds,
                )
            if settled:
                return
        except asyncio.CancelledError:
            raise
        except IsolatedToolCleanupUnproven:
            pass
        await asyncio.sleep(retry_delay)
        retry_delay = min(retry_delay * 2, 1.0)


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        raise IsolatedToolCleanupUnproven("process_group_probe_failed") from exc
    return True


async def _wait_for_group_and_child(
    process_group_id: int,
    wait_task: asyncio.Task[int],
    *,
    timeout: float,
) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        if wait_task.done() and not _process_group_exists(process_group_id):
            with suppress(BaseException):
                wait_task.result()
            return True
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return False
        await asyncio.sleep(min(0.01, remaining))


async def _settle_late_spawn(
    spawn_task: asyncio.Task[asyncio.subprocess.Process],
    limits: Any,
    *,
    parent_result_write_fd: int | None,
) -> None:
    try:
        outcome = await await_shielded_task_outcome(spawn_task)
    finally:
        if parent_result_write_fd is not None:
            with suppress(OSError):
                os.close(parent_result_write_fd)
    if outcome.error is not None:
        if isinstance(outcome.error, Exception):
            return
        raise IsolatedToolCleanupUnproven("late_spawn_outcome_unproven") from None
    process = outcome.result
    if process is None:
        return
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
    try:
        await _settle_owned_process_group(
            process.pid,
            wait_task,
            term_grace_seconds=limits.term_grace_seconds,
            kill_grace_seconds=limits.kill_grace_seconds,
        )
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
    _RETAINED_ISOLATED_TOOL_TASKS.add(task)
    if retry_factory is not None:
        _RETAINED_ISOLATED_TOOL_RETRIES[task] = retry_factory

    def discard(completed: asyncio.Task[Any]) -> None:
        if _retained_task_completed_successfully(completed):
            _discard_retained_task(completed)

    task.add_done_callback(discard)


def _retained_task_completed_successfully(task: asyncio.Task[Any]) -> bool:
    if not task.done() or task.cancelled():
        return False
    try:
        return task.exception() is None
    except BaseException:
        return False


def _discard_retained_task(task: asyncio.Task[Any]) -> None:
    _RETAINED_ISOLATED_TOOL_TASKS.discard(task)
    _RETAINED_ISOLATED_TOOL_RETRIES.pop(task, None)


def _retained_isolated_tool_cleanup_pending() -> bool:
    """Return positive process-local evidence of an unresolved cleanup owner."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    for task in tuple(_RETAINED_ISOLATED_TOOL_TASKS):
        if _retained_task_completed_successfully(task):
            _discard_retained_task(task)
            continue
        retry_factory = _RETAINED_ISOLATED_TOOL_RETRIES.get(task)
        if not task.done() or retry_factory is None or loop is None:
            continue
        try:
            retry_task = loop.create_task(
                retry_factory(),
                name="cayu-isolated-tool-retained-cleanup-retry",
            )
        except BaseException:
            continue
        _discard_retained_task(task)
        _retain_task(retry_task, retry_factory=retry_factory)
    return bool(_RETAINED_ISOLATED_TOOL_TASKS)


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
