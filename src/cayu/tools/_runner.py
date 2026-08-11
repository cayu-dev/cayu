"""Invocation-aware runner capability handed to model-invoked tools."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from math import isfinite
from pathlib import Path
from typing import Any, NoReturn, cast

from pydantic import ValidationError

from cayu._exception_groups import (
    exception_group_children,
    rebuild_exception_group,
    set_exception_cause,
)
from cayu._validation import require_durable_nonblank
from cayu.runners import (
    DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ExecCommand,
    ExecResult,
    LocalRunner,
    Runner,
    RunnerCancelledError,
    RunnerExecutionError,
    RunnerUnavailableError,
)
from cayu.runners._cleanup import (
    attach_runner_cancellation_failure,
    runner_cancellation_failure,
    sanitize_runner_artifacts,
)
from cayu.runners._diagnostics import (
    trusted_runner_error_type_name,
    trusted_runner_exception_type_name,
)
from cayu.runners._redacted_output import redact_completed_exec_result
from cayu.runners._subprocess import (
    copy_runner_env,
    validate_output_limit,
    validate_runner_env_remove,
    validate_stdin,
    validate_timeout,
)
from cayu.runners.base import (
    _clear_preflight_traceback_frames,
    copy_exec_command,
    runner_execution_error,
)
from cayu.tools._operation_boundary import (
    await_invocation_cancellation_checkpoint,
    await_invocation_operation,
)
from cayu.tools._redaction import resolve_invocation_redactor_snapshot
from cayu.tools._resources import invocation_workspace_authenticated_cwd
from cayu.vaults import REDACTED_SECRET, SecretRedactor
from cayu.workspaces import LocalWorkspace, RunnerBoundWorkspace

_KNOWN_RUNNER_ADAPTERS = frozenset({"docker", "e2b", "lambda-microvm", "local", "microsandbox"})
_KNOWN_UNAVAILABLE_REASONS = frozenset(
    {
        "guest_agent_unavailable_after_incomplete_exec",
        "guest_agent_unavailable_after_signal_9",
    }
)
_KNOWN_PROBE_STATUSES = frozenset({"failed", "timed_out"})
_CURRENT_CANCELLATION_GROUP_ATTRIBUTE = "_cayu_current_runner_cancellation_group"
_CURRENT_CANCELLATION_GROUP_TOKEN = object()
_SAFE_RUNNER_CANCELLATION_MESSAGE = "Runner command was cancelled."
_MAX_RUNNER_CANCELLATION_REASON_BYTES = 2048


class InvocationRunnerHandle:
    """Expose command dispatch without leaking the raw lifecycle-owning runner."""

    __slots__ = (
        "__ambiguous_capture_observer",
        "__redactor_snapshot_provider",
        "__runner",
    )

    def __init__(
        self,
        runner: Runner,
        *,
        redactor_snapshot_provider: Callable[[], Any],
        ambiguous_capture_observer: Callable[[int], None] | None = None,
    ) -> None:
        if not isinstance(runner, Runner):
            raise TypeError("Invocation runner handle requires a Runner.")
        if not callable(redactor_snapshot_provider):
            raise TypeError("redactor_snapshot_provider must be callable.")
        if ambiguous_capture_observer is not None and not callable(ambiguous_capture_observer):
            raise TypeError("ambiguous_capture_observer must be callable or None.")
        self.__runner = runner
        self.__redactor_snapshot_provider = redactor_snapshot_provider
        self.__ambiguous_capture_observer = ambiguous_capture_observer

    async def preflight_exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        env_remove: tuple[str, ...] = (),
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ) -> None:
        """Validate the selected runner without consulting invocation secrets."""

        command_validation_failure = False
        preflight_failure: BaseException | None = None
        try:
            try:
                owned_command = copy_exec_command(command)
            except ValidationError:
                command_validation_failure = True
                raise
            owned_cwd = None if cwd is None else require_durable_nonblank(cwd, "cwd")
            owned_env = None if env is None else copy_runner_env(env, inherit_env=False)
            owned_env_remove = validate_runner_env_remove(env_remove)
            owned_timeout = validate_timeout(timeout_s)
            owned_stdin = validate_stdin(stdin)
            owned_output_limit = validate_output_limit(output_limit_bytes)
            self.__runner.preflight_exec(
                owned_command,
                cwd=owned_cwd,
                env=owned_env,
                env_remove=owned_env_remove,
                timeout_s=owned_timeout,
                stdin=owned_stdin,
                output_limit_bytes=owned_output_limit,
            )
        except BaseException as error:
            preflight_failure = _classify_invocation_preflight_failure(
                self.__runner,
                error,
                command_validation_failure=command_validation_failure,
            )
            _clear_preflight_traceback_frames(error)
            owned_command = None
            owned_cwd = None
            owned_env = None
            owned_env_remove = ()
            owned_timeout = None
            owned_stdin = None
            owned_output_limit = None
            cwd = None
            env = None
            env_remove = ()
            timeout_s = None
            stdin = None
            output_limit_bytes = None
        finally:
            del command
        if preflight_failure is not None:
            preflight_failure.__traceback__ = None
            caller_cancellation = await await_invocation_cancellation_checkpoint()
            if caller_cancellation is not None:
                cancellation_type, cancellation_args, cancellation_artifacts = (
                    _detached_runner_cancellation_state(
                        caller_cancellation,
                        redactor=_current_runner_redactor(
                            self.__redactor_snapshot_provider,
                        ),
                        caller_cancelled=True,
                    )
                )
                cancellation_cause = preflight_failure
                del caller_cancellation, preflight_failure, self
                _raise_clean_runner_cancellation(
                    cancellation_type,
                    cancellation_args,
                    cancellation_artifacts,
                    cause=cancellation_cause,
                )
            published_failure = preflight_failure
            del preflight_failure, self
            raise published_failure from None

    async def exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        env_remove: tuple[str, ...] = (),
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ) -> ExecResult:
        """Dispatch with the invocation registry current at the call boundary."""

        # Validate and own the complete portable request before consulting the
        # invocation secret registry or reaching a cancellation checkpoint.
        command_validation_failure = False
        preflight_failure: BaseException | None = None
        try:
            try:
                owned_command = copy_exec_command(command)
            except ValidationError:
                command_validation_failure = True
                raise
            owned_cwd = None if cwd is None else require_durable_nonblank(cwd, "cwd")
            owned_env = None if env is None else copy_runner_env(env, inherit_env=False)
            owned_env_remove = validate_runner_env_remove(env_remove)
            owned_timeout = validate_timeout(timeout_s)
            owned_stdin = validate_stdin(stdin)
            owned_output_limit = validate_output_limit(output_limit_bytes)
            self.__runner.preflight_exec(
                owned_command,
                cwd=owned_cwd,
                env=owned_env,
                env_remove=owned_env_remove,
                timeout_s=owned_timeout,
                stdin=owned_stdin,
                output_limit_bytes=owned_output_limit,
            )
        except BaseException as error:
            preflight_failure = _classify_invocation_preflight_failure(
                self.__runner,
                error,
                command_validation_failure=command_validation_failure,
            )
            _clear_preflight_traceback_frames(error)
            owned_command = None
            owned_cwd = None
            owned_env = None
            owned_env_remove = ()
            owned_timeout = None
            owned_stdin = None
            owned_output_limit = None
            cwd = None
            env = None
            env_remove = ()
            timeout_s = None
            stdin = None
            output_limit_bytes = None
        finally:
            del command
        if preflight_failure is not None:
            preflight_failure.__traceback__ = None
            caller_cancellation = await await_invocation_cancellation_checkpoint()
            if caller_cancellation is not None:
                cancellation_type, cancellation_args, cancellation_artifacts = (
                    _detached_runner_cancellation_state(
                        caller_cancellation,
                        redactor=_current_runner_redactor(
                            self.__redactor_snapshot_provider,
                        ),
                        caller_cancelled=True,
                    )
                )
                cancellation_cause = preflight_failure
                del caller_cancellation, preflight_failure, self
                _raise_clean_runner_cancellation(
                    cancellation_type,
                    cancellation_args,
                    cancellation_artifacts,
                    cause=cancellation_cause,
                )
            published_failure = preflight_failure
            del preflight_failure, self
            raise published_failure from None
        if owned_command is None:  # pragma: no cover - preflight construction invariant
            raise AssertionError("Runner preflight completed without an owned command.")
        initial = resolve_invocation_redactor_snapshot(self.__redactor_snapshot_provider)
        initial_revision = initial.revision
        result: ExecResult | None = None
        failure: Exception | None = None
        grouped_failure: BaseExceptionGroup | None = None
        cancellation_cause: BaseException | None = None
        cancellation: (
            tuple[
                type[asyncio.CancelledError],
                tuple[object, ...],
                list[dict[str, Any]],
            ]
            | None
        ) = None
        kwargs: dict[str, Any] = {
            "redactor": initial.redactor,
            "cwd": owned_cwd,
            "env": owned_env,
            "timeout_s": owned_timeout,
            "stdin": owned_stdin,
            "output_limit_bytes": owned_output_limit,
        }
        if owned_env_remove:
            kwargs["env_remove"] = owned_env_remove

        def operation_factory(
            runner: Runner = self.__runner,
            command: ExecCommand = owned_command,
            kwargs: dict[str, Any] = kwargs,
        ) -> Awaitable[ExecResult]:
            return runner.exec_redacted(command, **kwargs)

        operation = await_invocation_operation(operation_factory)
        del operation_factory
        # The boundary coroutine now exclusively owns the request factory.
        # Remove raw command input before this public frame starts awaiting.
        del (
            owned_command,
            owned_cwd,
            owned_env,
            owned_env_remove,
            owned_timeout,
            owned_stdin,
            cwd,
            env,
            env_remove,
            timeout_s,
            stdin,
            output_limit_bytes,
            kwargs,
            initial,
        )
        try:
            outcome = await operation
        finally:
            del operation
        result = outcome.result
        raw_error = outcome.error
        caller_cancellation = outcome.cancellation
        if caller_cancellation is not None:
            current_redactor = _current_runner_redactor(
                self.__redactor_snapshot_provider,
            )
            cancellation = _detached_runner_cancellation_state(
                caller_cancellation,
                redactor=current_redactor,
                caller_cancelled=True,
            )
            del current_redactor
        if isinstance(raw_error, BaseExceptionGroup):
            if isinstance(raw_error, Exception):
                safe_failure = _safe_runner_execution_error(self.__runner, raw_error)
                if caller_cancellation is not None:
                    cancellation_cause = safe_failure
                else:
                    failure = safe_failure
            else:
                safe_group = sanitize_runner_failure_group(
                    raw_error,
                    runner=self.__runner,
                    caller_cancelled=caller_cancellation is not None,
                )
                if caller_cancellation is not None:
                    cleanup_failures, child_cancellation_artifacts = (
                        _partition_runner_cancellation_group(safe_group)
                    )
                    if child_cancellation_artifacts:
                        if cancellation is None:  # pragma: no cover - construction invariant
                            raise AssertionError("Caller cancellation state was not captured.")
                        cancellation = (
                            cancellation[0],
                            cancellation[1],
                            sanitize_runner_artifacts(
                                [
                                    *cancellation[2],
                                    *child_cancellation_artifacts,
                                ]
                            ),
                        )
                    cancellation_cause = cleanup_failures
                else:
                    grouped_failure = safe_group
        elif isinstance(raw_error, asyncio.CancelledError):
            child_cancellation = _detached_runner_cancellation_state(
                raw_error,
                redactor=_current_runner_redactor(
                    self.__redactor_snapshot_provider,
                ),
                caller_cancelled=False,
            )
            if caller_cancellation is None:
                cancellation = child_cancellation
            else:
                if cancellation is None:  # pragma: no cover - construction invariant
                    raise AssertionError("Caller cancellation state was not captured.")
                cancellation = (
                    cancellation[0],
                    cancellation[1],
                    sanitize_runner_artifacts([*cancellation[2], *child_cancellation[2]]),
                )
        elif isinstance(raw_error, RunnerUnavailableError):
            safe_failure = _safe_runner_unavailable_error(self.__runner, raw_error)
            if caller_cancellation is not None:
                cancellation_cause = safe_failure
            else:
                failure = safe_failure
        elif isinstance(raw_error, Exception):
            safe_failure = _safe_runner_execution_error(self.__runner, raw_error)
            if caller_cancellation is not None:
                cancellation_cause = safe_failure
            else:
                failure = safe_failure
        elif raw_error is not None:
            if caller_cancellation is not None:
                cancellation_cause = _safe_runner_execution_error(
                    self.__runner,
                    raw_error,
                )
            else:
                fatal_error = raw_error
                del outcome, raw_error, caller_cancellation, self, result
                raise fatal_error from None
        del outcome, raw_error, caller_cancellation
        if cancellation is not None:
            # Do not leave an object path from the public traceback back into
            # adapter/SDK state that may still retain the transferred request.
            cancellation_type, cancellation_args, cancellation_artifacts = cancellation
            safe_cause = cancellation_cause
            del (
                cancellation,
                cancellation_cause,
                self,
                result,
                failure,
                grouped_failure,
            )
            _raise_clean_runner_cancellation(
                cancellation_type,
                cancellation_args,
                cancellation_artifacts,
                cause=safe_cause,
            )
        if grouped_failure is not None:
            safe_grouped_failure = grouped_failure
            del grouped_failure, self, result, failure
            raise safe_grouped_failure from None
        if failure is not None:
            safe_failure = failure
            del failure, self, result
            raise safe_failure from None
        if result is None:
            raise RuntimeError("Runner command returned without a result.")

        current = resolve_invocation_redactor_snapshot(self.__redactor_snapshot_provider)
        projected = redact_completed_exec_result(
            result,
            redactor=current.redactor,
            output_limit_bytes=owned_output_limit,
            omit_pretruncated=current.revision != initial_revision,
        )
        if self.__ambiguous_capture_observer is not None and _has_ambiguous_capture(
            projected,
            output_limit_bytes=owned_output_limit,
        ):
            self.__ambiguous_capture_observer(current.revision)
        return projected.model_copy(
            update={"artifacts": sanitize_runner_artifacts(projected.artifacts)}
        )

    def resolve_cwd(self, cwd: str | None = None) -> str:
        """Delegate canonicalization without exposing runner lifecycle methods."""

        return self.__runner.resolve_cwd(cwd)

    def authenticated_workspace_cwd(self, workspace: Any) -> str | None:
        """Resolve a cwd only when the hidden runner positively owns the workspace."""

        if workspace is None:
            return self.__runner.resolve_cwd(None)
        invocation_cwd = invocation_workspace_authenticated_cwd(workspace, self.__runner)
        if invocation_cwd is not None:
            return invocation_cwd
        if isinstance(workspace, RunnerBoundWorkspace):
            if not workspace.is_bound_to_runner(self.__runner):
                return None
            return workspace.runner_cwd
        if isinstance(workspace, LocalWorkspace) and isinstance(self.__runner, LocalRunner):
            runner_root = Path(self.__runner.resolve_cwd(None)).resolve()
            workspace_root = workspace.root.resolve()
            try:
                workspace_root.relative_to(runner_root)
            except ValueError:
                return None
            return str(workspace_root)
        return None


def _raise_clean_runner_cancellation(
    cancellation_type: type[asyncio.CancelledError],
    args: tuple[object, ...],
    artifacts: list[dict[str, Any]],
    *,
    cause: BaseException | None = None,
) -> NoReturn:
    """Raise cancellation without retaining a runner or raw-request traceback.

    Python may subsequently attach an exception active in the tool's caller.
    The runtime removes that caller-added context at its external cancellation
    boundary, after the complete tool failure tree is available.
    """

    cancellation = _clean_runner_cancellation(cancellation_type, args, artifacts)
    if cause is not None:
        attach_runner_cancellation_failure(cancellation, cause)
    raise cancellation from cause


def _clean_runner_cancellation(
    cancellation_type: type[asyncio.CancelledError],
    args: tuple[object, ...],
    artifacts: list[dict[str, Any]],
) -> asyncio.CancelledError:
    if cancellation_type is RunnerCancelledError:
        cancellation = RunnerCancelledError(
            artifacts=artifacts,
        )
        BaseException.__dict__["args"].__set__(cancellation, args)
    else:
        cancellation = asyncio.CancelledError(*args)
    if artifacts:
        namespace = BaseException.__dict__["__dict__"].__get__(
            cancellation,
            BaseException,
        )
        if type(namespace) is dict:
            dict.__setitem__(namespace, "artifacts", artifacts)
    return cancellation


def _detached_runner_cancellation_state(
    cancellation: asyncio.CancelledError,
    *,
    redactor: SecretRedactor | None,
    caller_cancelled: bool,
) -> tuple[
    type[asyncio.CancelledError],
    tuple[object, ...],
    list[dict[str, Any]],
]:
    """Capture only cancellation state safe to carry across the runner boundary."""

    cancellation_type = _safe_runner_cancellation_type(
        cancellation,
        redactor=redactor,
        caller_cancelled=caller_cancelled,
    )
    args = _safe_runner_cancellation_args(
        (_SAFE_RUNNER_CANCELLATION_MESSAGE,),
        redactor=redactor,
        cancellation_type=cancellation_type,
    )
    if caller_cancelled and redactor is not None:
        try:
            source_args = BaseException.__dict__["args"].__get__(
                cancellation,
                BaseException,
            )
        except BaseException:
            source_args = ()
        args = _safe_runner_cancellation_args(
            source_args,
            redactor=redactor,
            cancellation_type=cancellation_type,
        )
    return (
        cancellation_type,
        args,
        sanitize_runner_artifacts(_base_exception_namespace_value(cancellation, "artifacts")),
    )


def _safe_runner_cancellation_type(
    cancellation: asyncio.CancelledError,
    *,
    redactor: SecretRedactor | None,
    caller_cancelled: bool,
) -> type[asyncio.CancelledError]:
    """Preserve a legacy subtype only when it is runner-owned and safe to render."""

    if (
        caller_cancelled
        or not isinstance(cancellation, RunnerCancelledError)
        or redactor is None
        or not _runner_cancellation_rendering_is_safe(
            (),
            redactor=redactor,
            cancellation_type=RunnerCancelledError,
        )
    ):
        return asyncio.CancelledError
    return RunnerCancelledError


def _safe_runner_cancellation_args(
    source_args: object,
    *,
    redactor: SecretRedactor | None,
    cancellation_type: type[asyncio.CancelledError],
) -> tuple[object, ...]:
    """Project one ordinary reason and validate the complete exception rendering."""

    candidates: list[tuple[object, ...]] = []
    if redactor is not None and type(source_args) is tuple and len(source_args) <= 1:
        if not source_args:
            candidates.append(())
        else:
            value = source_args[0]
            try:
                if type(value) is str:
                    rendered = value
                elif value is None or type(value) in {bool, int, float}:
                    rendered = str(value)
                else:
                    rendered = None
                if rendered is not None:
                    projected = redactor.redact_text_bounded(
                        rendered,
                        max_bytes=_MAX_RUNNER_CANCELLATION_REASON_BYTES,
                    )
                    candidates.append(
                        (value,)
                        if type(value) is not str and projected == rendered
                        else (projected,)
                    )
            except BaseException:
                pass
    candidates.extend(
        (
            (_SAFE_RUNNER_CANCELLATION_MESSAGE,),
            (REDACTED_SECRET,),
            (),
        )
    )
    for candidate_args in candidates:
        if redactor is None or _runner_cancellation_rendering_is_safe(
            candidate_args,
            redactor=redactor,
            cancellation_type=cancellation_type,
        ):
            return candidate_args
    return ()


def _runner_cancellation_rendering_is_safe(
    args: tuple[object, ...],
    *,
    redactor: SecretRedactor,
    cancellation_type: type[asyncio.CancelledError],
) -> bool:
    try:
        if cancellation_type is RunnerCancelledError:
            candidate = RunnerCancelledError()
            BaseException.__dict__["args"].__set__(candidate, args)
        else:
            candidate = asyncio.CancelledError(*args)
        rendered = (str(candidate), repr(candidate))
    except BaseException:
        return False
    return all(redactor.redact_text(value) == value for value in rendered)


def _current_runner_redactor(
    provider: Callable[[], Any],
) -> SecretRedactor | None:
    """Resolve the latest registry, failing closed when it is unavailable."""

    try:
        return resolve_invocation_redactor_snapshot(provider).redactor
    except BaseException:
        return None


def invocation_runner_handle(
    runner: Any,
    *,
    redactor_snapshot_provider: Callable[[], Any],
    ambiguous_capture_observer: Callable[[int], None] | None = None,
) -> InvocationRunnerHandle | None:
    """Build the narrow runtime runner capability for one tool invocation."""

    if runner is None:
        return None
    if not isinstance(runner, Runner):
        raise TypeError("Registered environment runner must implement Runner.")
    return InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=redactor_snapshot_provider,
        ambiguous_capture_observer=ambiguous_capture_observer,
    )


def _safe_runner_execution_error(
    runner: Runner,
    error: BaseException,
) -> RunnerExecutionError:
    return runner_execution_error(
        error,
        adapter=_safe_runner_adapter(runner),
    )


def _classify_invocation_preflight_failure(
    runner: Runner,
    error: BaseException,
    *,
    command_validation_failure: bool,
) -> BaseException:
    """Preserve only caller-owned validation and authenticated control signals."""

    if command_validation_failure:
        return error
    if isinstance(error, asyncio.CancelledError):
        return _safe_runner_execution_error(runner, error)
    if not isinstance(error, Exception):
        return error
    if isinstance(error, RunnerUnavailableError):
        return _safe_runner_unavailable_error(runner, error)
    return _safe_runner_execution_error(runner, error)


def _has_ambiguous_capture(
    result: ExecResult,
    *,
    output_limit_bytes: int | None,
) -> bool:
    for channel in ("stdout", "stderr"):
        if getattr(result, f"{channel}_truncated"):
            return True
        byte_count = getattr(result, f"{channel}_bytes")
        if (
            output_limit_bytes is not None
            and type(byte_count) is int
            and byte_count > output_limit_bytes
        ):
            return True
    return False


def sanitize_runner_failure_group(
    error: BaseExceptionGroup,
    *,
    runner: Runner | None = None,
    caller_cancelled: bool = False,
) -> BaseExceptionGroup:
    """Detach a mixed runner failure tree without losing ordered cancellation."""

    if not isinstance(error, BaseExceptionGroup):
        raise TypeError("error must be a BaseExceptionGroup.")

    def map_leaf(leaf: BaseException) -> BaseException:
        if isinstance(leaf, asyncio.CancelledError):
            cancellation = asyncio.CancelledError("Runner command was cancelled.")
            artifacts = _base_exception_namespace_value(leaf, "artifacts")
            safe_artifacts = sanitize_runner_artifacts(artifacts)
            if safe_artifacts:
                namespace = BaseException.__dict__["__dict__"].__get__(
                    cancellation,
                    BaseException,
                )
                if type(namespace) is dict:
                    dict.__setitem__(namespace, "artifacts", safe_artifacts)
            cleanup_failure = runner_cancellation_failure(leaf)
            if cleanup_failure is not None:
                safe_cleanup_failure = sanitize_runner_failure(
                    cleanup_failure,
                    runner=runner,
                )
                attach_runner_cancellation_failure(
                    cancellation,
                    safe_cleanup_failure,
                )
                set_exception_cause(cancellation, safe_cleanup_failure)
            return cancellation
        adapter = _safe_runner_adapter(runner) if runner is not None else _safe_leaf_adapter(leaf)
        if isinstance(leaf, RunnerUnavailableError):
            return _safe_runner_unavailable_error_for_adapter(leaf, adapter=adapter)
        return runner_execution_error(leaf, adapter=adapter)

    rebuilt = rebuild_exception_group(
        error,
        group_message="Runner command reported multiple failures.",
        leaf_mapper=map_leaf,
        invalid_leaf_factory=lambda: runner_execution_error(
            RuntimeError("Invalid runner failure group."),
            adapter=_safe_runner_adapter(runner) if runner is not None else "unknown",
        ),
    )
    if caller_cancelled or is_current_runner_cancellation_group(error):
        _mark_current_runner_cancellation_group(rebuilt)
    return rebuilt


def sanitize_runner_failure(
    error: BaseException,
    *,
    runner: Runner | None = None,
) -> BaseException:
    """Rebuild one opaque runner failure without retaining mutable extension state."""

    if not isinstance(error, BaseException):
        raise TypeError("error must be a BaseException.")
    if isinstance(error, BaseExceptionGroup):
        return sanitize_runner_failure_group(error, runner=runner)
    adapter = _safe_runner_adapter(runner) if runner is not None else _safe_leaf_adapter(error)
    if isinstance(error, RunnerUnavailableError):
        return _safe_runner_unavailable_error_for_adapter(error, adapter=adapter)
    return runner_execution_error(error, adapter=adapter)


def _mark_current_runner_cancellation_group(error: BaseExceptionGroup) -> None:
    try:
        namespace = BaseException.__dict__["__dict__"].__get__(
            error,
            BaseException,
        )
        if type(namespace) is dict:
            dict.__setitem__(
                namespace,
                _CURRENT_CANCELLATION_GROUP_ATTRIBUTE,
                _CURRENT_CANCELLATION_GROUP_TOKEN,
            )
    except BaseException:
        pass


def is_current_runner_cancellation_group(error: BaseException) -> bool:
    """Return whether a sanitized group carries authenticated current cancellation."""

    if not isinstance(error, BaseExceptionGroup):
        return False
    return (
        _base_exception_namespace_value(
            error,
            _CURRENT_CANCELLATION_GROUP_ATTRIBUTE,
        )
        is _CURRENT_CANCELLATION_GROUP_TOKEN
    )


def _safe_runner_unavailable_error(
    runner: Runner,
    error: RunnerUnavailableError,
) -> RunnerUnavailableError:
    return _safe_runner_unavailable_error_for_adapter(
        error,
        adapter=_safe_runner_adapter(runner),
    )


def _safe_runner_unavailable_error_for_adapter(
    error: RunnerUnavailableError,
    *,
    adapter: str,
) -> RunnerUnavailableError:
    diagnostic: dict[str, Any] = {
        "type": "cayu.runner_unavailable.v1",
        "adapter": (
            adapter if type(adapter) is str and adapter in _KNOWN_RUNNER_ADAPTERS else "unknown"
        ),
        "status": "unavailable",
        "error_type": trusted_runner_exception_type_name(error),
    }
    source = _base_exception_namespace_value(error, "diagnostic")
    if type(source) is dict:
        source = cast("dict[str, Any]", source)
        reason = source.get("reason")
        if type(reason) is str and reason in _KNOWN_UNAVAILABLE_REASONS:
            diagnostic["reason"] = reason
        last_command = _safe_last_command_diagnostic(source.get("last_command"))
        if last_command:
            diagnostic["last_command"] = last_command
        probe = _safe_probe_diagnostic(source.get("probe"))
        if probe:
            diagnostic["probe"] = probe
    return RunnerUnavailableError(
        "Runner is unavailable.",
        diagnostic=diagnostic,
    )


def _safe_leaf_adapter(error: BaseException) -> str:
    diagnostic = _base_exception_namespace_value(error, "diagnostic")
    if type(diagnostic) is dict:
        adapter = cast("dict[str, Any]", diagnostic).get("adapter")
        if type(adapter) is str and adapter in _KNOWN_RUNNER_ADAPTERS:
            return cast("str", adapter)
    return "unknown"


def _safe_last_command_diagnostic(value: object) -> dict[str, Any]:
    if type(value) is not dict:
        return {}
    value = cast("dict[str, Any]", value)
    safe: dict[str, Any] = {}
    exit_code = value.get("exit_code")
    if exit_code is None or type(exit_code) is int:
        safe["exit_code"] = exit_code
    for field in ("timed_out", "cancelled"):
        field_value = value.get(field)
        if type(field_value) is bool:
            safe[field] = field_value
    for field in ("stdout_bytes", "stderr_bytes"):
        field_value = value.get(field)
        if type(field_value) is int and field_value >= 0:
            safe[field] = field_value
    error_type = value.get("error_type")
    trusted_error_type = trusted_runner_error_type_name(error_type)
    if trusted_error_type is not None:
        safe["error_type"] = trusted_error_type
    return safe


def _safe_probe_diagnostic(value: object) -> dict[str, Any]:
    if type(value) is not dict:
        return {}
    value = cast("dict[str, Any]", value)
    safe: dict[str, Any] = {}
    if value.get("method") == "Sandbox.ping":
        safe["method"] = "Sandbox.ping"
    status = value.get("status")
    if type(status) is str and status in _KNOWN_PROBE_STATUSES:
        safe["status"] = status
    timeout_s = _positive_float(value.get("timeout_s"))
    if timeout_s is not None:
        safe["timeout_s"] = timeout_s
    for field in ("error_type", "status_error_type"):
        trusted_field_value = trusted_runner_error_type_name(value.get(field))
        if trusted_field_value is not None:
            safe[field] = trusted_field_value
    return safe


def _safe_runner_adapter(runner: Runner) -> str:
    try:
        adapter = type.__getattribute__(type(runner), "isolation")
    except BaseException:
        adapter = None
    if type(adapter) is str and adapter in _KNOWN_RUNNER_ADAPTERS:
        return cast("str", adapter)
    try:
        namespace = object.__getattribute__(runner, "__dict__")
    except BaseException:
        namespace = None
    if type(namespace) is dict:
        adapter = dict.get(namespace, "isolation")
        if type(adapter) is str and adapter in _KNOWN_RUNNER_ADAPTERS:
            return cast("str", adapter)
    return "unknown"


def _partition_runner_cancellation_group(
    error: BaseExceptionGroup,
) -> tuple[BaseExceptionGroup | None, list[dict[str, Any]]]:
    """Remove cancellation leaves without recursively traversing an opaque group."""

    group_message = "Runner command reported cleanup failures."
    artifacts: list[dict[str, Any]] = []
    pending: list[tuple[BaseException, bool]] = [(error, False)]
    children_by_group: dict[int, tuple[BaseException, ...]] = {}
    rebuilt: dict[int, BaseException | None] = {}
    while pending:
        candidate, expanded = pending.pop()
        candidate_id = id(candidate)
        if candidate_id in rebuilt:
            continue
        if not isinstance(candidate, BaseExceptionGroup):
            if isinstance(candidate, asyncio.CancelledError):
                artifacts.extend(
                    sanitize_runner_artifacts(
                        _base_exception_namespace_value(candidate, "artifacts")
                    )
                )
                rebuilt[candidate_id] = None
            else:
                rebuilt[candidate_id] = candidate
            continue
        if expanded:
            children = children_by_group.pop(candidate_id, ())
            cleanup_children = [
                rebuilt_child
                for child in children
                if (rebuilt_child := rebuilt.get(id(child))) is not None
            ]
            rebuilt[candidate_id] = (
                BaseExceptionGroup(group_message, cleanup_children) if cleanup_children else None
            )
            continue
        children = exception_group_children(candidate)
        if children is None:
            rebuilt[candidate_id] = BaseExceptionGroup(
                group_message,
                [
                    runner_execution_error(
                        RuntimeError("Invalid runner cleanup failure group."),
                        adapter="unknown",
                    )
                ],
            )
            continue
        children_by_group[candidate_id] = children
        pending.append((candidate, True))
        pending.extend((child, False) for child in reversed(children))

    cleanup_failure = rebuilt.get(id(error))
    return (
        cleanup_failure if isinstance(cleanup_failure, BaseExceptionGroup) else None,
        sanitize_runner_artifacts(artifacts),
    )


def _positive_float(value: object) -> float | None:
    if type(value) is int:
        try:
            number = float(value)
        except OverflowError:
            return None
    elif type(value) is float:
        number = value
    else:
        return None
    return number if isfinite(number) and number > 0 else None


def _base_exception_namespace_value(error: BaseException, name: str) -> object:
    try:
        namespace = BaseException.__dict__["__dict__"].__get__(error, BaseException)
    except BaseException:
        return None
    return dict.get(namespace, name) if type(namespace) is dict else None
