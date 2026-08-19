"""Invocation-aware runner capability handed to model-invoked tools."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from time import monotonic
from typing import Any, Literal, NoReturn, cast

from pydantic import ValidationError

from cayu._exception_groups import (
    exception_group_children,
    rebuild_exception_group,
    set_exception_cause,
)
from cayu._task_wait import await_shielded_task_outcome
from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    compact_json_utf8_size,
    require_durable_nonblank,
)
from cayu._workspace_mutation import detached_workspace_mutation_process_signal
from cayu.environments.admission import ExecutionAdmissionCandidate
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
    RunnerWorkspaceMutationSettlement,
    _clear_preflight_traceback_frames,
    copy_exec_command,
    runner_execution_error,
    runner_pending_command_settlement_cancellation_safe,
    runner_workspace_mutation_settlement,
)
from cayu.tools._operation_boundary import (
    _RetainedInvocationOperationProbe,
    await_invocation_cancellation_checkpoint,
    await_invocation_operation,
)
from cayu.tools._redaction import resolve_invocation_redactor_snapshot
from cayu.tools._resources import (
    InvocationWorkspaceMutationOwner,
    invocation_workspace_authenticated_cwd,
)
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
_RUNNER_MUTATION_SETTLEMENT_FOREGROUND_TIMEOUT_SECONDS = 30.0
_RUNNER_COMMAND_EVIDENCE_MAX_BYTES = 4096
_RUNNER_COMMAND_EVIDENCE_PREVIEW_MAX_BYTES = 1536
RunnerExecutionObserver = Callable[
    [Literal["started", "completed"], dict[str, Any], int],
    Awaitable[None],
]


class InvocationRunnerHandle:
    """Expose command dispatch without leaking the raw lifecycle-owning runner."""

    __slots__ = (
        "__ambiguous_capture_observer",
        "__execution_observer",
        "__mutation_owner",
        "__publish_execution_arguments",
        "__redactor_snapshot_provider",
        "__runner",
    )

    def __init__(
        self,
        runner: Runner,
        *,
        redactor_snapshot_provider: Callable[[], Any],
        ambiguous_capture_observer: Callable[[int], None] | None = None,
        mutation_owner: InvocationWorkspaceMutationOwner | None = None,
        execution_observer: RunnerExecutionObserver | None = None,
        publish_execution_arguments: bool = True,
    ) -> None:
        if not isinstance(runner, Runner):
            raise TypeError("Invocation runner handle requires a Runner.")
        if not callable(redactor_snapshot_provider):
            raise TypeError("redactor_snapshot_provider must be callable.")
        if ambiguous_capture_observer is not None and not callable(ambiguous_capture_observer):
            raise TypeError("ambiguous_capture_observer must be callable or None.")
        if (
            mutation_owner is not None
            and type(mutation_owner) is not InvocationWorkspaceMutationOwner
        ):
            raise TypeError("Invocation runner mutation owner is invalid.")
        if execution_observer is not None and not callable(execution_observer):
            raise TypeError("execution_observer must be callable or None.")
        if type(publish_execution_arguments) is not bool:
            raise TypeError("publish_execution_arguments must be a bool.")
        self.__runner = runner
        self.__redactor_snapshot_provider = redactor_snapshot_provider
        self.__ambiguous_capture_observer = ambiguous_capture_observer
        self.__mutation_owner = mutation_owner
        self.__execution_observer = execution_observer
        self.__publish_execution_arguments = publish_execution_arguments

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

    def execution_admission_candidate(self) -> ExecutionAdmissionCandidate | None:
        """Return a detached capability snapshot without exposing runner ownership."""

        candidate = self.__runner.execution_admission_candidate()
        if candidate is None:
            return None
        if type(candidate) is not ExecutionAdmissionCandidate:
            raise TypeError(
                "Runner execution_admission_candidate() must return "
                "ExecutionAdmissionCandidate or None."
            )
        return ExecutionAdmissionCandidate.model_validate(
            candidate.model_dump(mode="python", warnings=False)
        )

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

        mutation_tracker = None
        mutation_admission_failure: RunnerExecutionError | None = None
        if self.__mutation_owner is not None:
            try:
                mutation_tracker = self.__mutation_owner.track_current_task()
                mutation_tracker.__enter__()
            except RuntimeError as error:
                mutation_admission_failure = _safe_runner_execution_error(
                    self.__runner,
                    error,
                )
                error.__traceback__ = None
                del error
        if mutation_admission_failure is not None:
            published_failure = mutation_admission_failure
            del (
                mutation_admission_failure,
                mutation_tracker,
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
                self,
                result,
                failure,
                grouped_failure,
                cancellation_cause,
                cancellation,
            )
            raise published_failure from None

        def operation_factory(
            runner: Runner = self.__runner,
            command: ExecCommand = owned_command,
            kwargs: dict[str, Any] = kwargs,
            capture_settlement: bool = self.__mutation_owner is not None,
            execution_observer: RunnerExecutionObserver | None = self.__execution_observer,
            publish_execution_arguments: bool = self.__publish_execution_arguments,
            command_evidence_revision: int = initial_revision,
        ) -> Awaitable[_RunnerDispatchOutcome]:
            return _capture_runner_dispatch_outcome(
                runner=runner,
                command=command,
                kwargs=kwargs,
                capture_settlement=capture_settlement,
                execution_observer=execution_observer,
                publish_execution_arguments=publish_execution_arguments,
                command_evidence_revision=command_evidence_revision,
            )

        operation = None
        caller_cancellation: asyncio.CancelledError | None = None
        settlement_failure: BaseException | None = None
        restore_settlement_cancellation_requests = 0

        def retain_unsettled_runner_operation(
            retained_operation: _RetainedInvocationOperationProbe,
            *,
            mutation_owner: InvocationWorkspaceMutationOwner | None = self.__mutation_owner,
            runner: Runner = self.__runner,
        ) -> None:
            if mutation_owner is None:  # pragma: no cover - callback installation invariant
                raise AssertionError("Runner mutation owner is unavailable.")
            mutation_owner.fail_closed(
                _RetainedRunnerDispatchSettlementProbe(
                    operation=retained_operation,
                    runner=runner,
                )
            )

        try:
            operation = await_invocation_operation(
                operation_factory,
                on_unsettled_supervisory_exit=(
                    retain_unsettled_runner_operation if self.__mutation_owner is not None else None
                ),
            )
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
            dispatch_outcome = (
                outcome.result if type(outcome.result) is _RunnerDispatchOutcome else None
            )
            if dispatch_outcome is None:
                runner_result = None
                runner_error = outcome.error
                settlement = "runner_quiescent" if outcome.operation_started is False else None
                settlement_error = None
            else:
                runner_result = dispatch_outcome.result
                runner_error = dispatch_outcome.error
                settlement = dispatch_outcome.settlement
                settlement_error = dispatch_outcome.settlement_error
            caller_cancellation = outcome.cancellation
            if self.__mutation_owner is not None:
                settlement_outcome = await _settle_invocation_runner_mutation(
                    runner=self.__runner,
                    owner=self.__mutation_owner,
                    settlement=settlement,
                    settlement_error=settlement_error,
                    cancellation=caller_cancellation,
                )
                if caller_cancellation is None:
                    caller_cancellation = settlement_outcome.cancellation
                settlement_failure = settlement_outcome.failure
                restore_settlement_cancellation_requests = (
                    settlement_outcome.restore_cancellation_requests
                )
                del settlement_outcome
        finally:
            if mutation_tracker is not None:
                mutation_tracker.__exit__(None, None, None)
        del mutation_tracker
        if settlement_failure is not None and caller_cancellation is None:
            published_settlement_signal = settlement_failure
            del (
                cancellation,
                cancellation_cause,
                caller_cancellation,
                failure,
                grouped_failure,
                outcome,
                dispatch_outcome,
                runner_error,
                runner_result,
                settlement,
                settlement_error,
                result,
                restore_settlement_cancellation_requests,
                self,
                settlement_failure,
            )
            _raise_clean_runner_settlement_signal(published_settlement_signal)
        result = cast("ExecResult | None", runner_result)
        raw_error = runner_error
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
                del (
                    outcome,
                    dispatch_outcome,
                    raw_error,
                    runner_error,
                    runner_result,
                    settlement,
                    settlement_error,
                    caller_cancellation,
                    self,
                    result,
                )
                raise fatal_error from None
        if settlement_failure is not None:
            cancellation_cause = _combine_runner_cancellation_causes(
                cancellation_cause,
                settlement_failure,
            )
        del (
            outcome,
            dispatch_outcome,
            raw_error,
            runner_error,
            runner_result,
            settlement,
            settlement_error,
            caller_cancellation,
            settlement_failure,
        )
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
                restore_cancellation_requests=restore_settlement_cancellation_requests,
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


@dataclass(frozen=True, slots=True)
class _RunnerSettlementCallOutcome:
    result: object | None = None
    error: BaseException | None = None


@dataclass(frozen=True, slots=True)
class _RunnerDispatchOutcome:
    """Extension result plus immutable runtime-owned settlement authority."""

    result: object | None = None
    error: BaseException | None = None
    settlement: RunnerWorkspaceMutationSettlement | None = None
    settlement_error: BaseException | None = None


@dataclass(frozen=True, slots=True)
class _RunnerMutationSettlementOutcome:
    cancellation: asyncio.CancelledError | None = None
    failure: BaseException | None = None
    restore_cancellation_requests: int = 0


async def _capture_runner_dispatch_outcome(
    *,
    runner: Runner,
    command: ExecCommand,
    kwargs: dict[str, Any],
    capture_settlement: bool,
    execution_observer: RunnerExecutionObserver | None,
    publish_execution_arguments: bool,
    command_evidence_revision: int,
) -> _RunnerDispatchOutcome:
    """Freeze settlement evidence before an extension can mutate its outcome."""

    started_at = monotonic()
    if type(command_evidence_revision) is not int or command_evidence_revision < 0:
        raise TypeError("command_evidence_revision must be a non-negative integer.")
    command_evidence = None
    if execution_observer is not None:
        command_evidence = _runner_command_evidence(
            command,
            redactor=cast("SecretRedactor", kwargs["redactor"]),
            publish_arguments=publish_execution_arguments,
        )
        try:
            await execution_observer(
                "started",
                {
                    "adapter": _safe_runner_adapter(runner),
                    "command": command_evidence,
                },
                command_evidence_revision,
            )
        except BaseException as error:
            return _RunnerDispatchOutcome(
                error=error,
                settlement="runner_quiescent" if capture_settlement else None,
            )

    result: object | None = None
    error: BaseException | None = None
    try:
        result = await runner.exec_redacted(command, **kwargs)
    except BaseException as exc:
        error = exc

    settlement = None
    settlement_error = None
    if capture_settlement:
        try:
            settlement = _invocation_runner_mutation_settlement(
                operation_started=True,
                result=result,
                error=error,
            )
        except BaseException as exc:
            settlement_error = exc

    outcome = _RunnerDispatchOutcome(
        result=result,
        error=error,
        settlement=settlement,
        settlement_error=settlement_error,
    )
    if execution_observer is None:
        return outcome
    if command_evidence is None:  # pragma: no cover - observer construction invariant
        raise AssertionError("Runner execution observer has no command evidence.")
    completed_payload: dict[str, Any] = {
        "adapter": _safe_runner_adapter(runner),
        "command": command_evidence,
        "duration_ms": min(
            MAX_DURABLE_JSON_INTEGER,
            max(0, int((monotonic() - started_at) * 1000)),
        ),
    }
    if type(result) is ExecResult:
        completed_payload.update(
            {
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
            }
        )
    if error is not None:
        completed_payload["error_type"] = (
            trusted_runner_exception_type_name(error) or "runner_execution_error"
        )
    try:
        await execution_observer(
            "completed",
            completed_payload,
            command_evidence_revision,
        )
    except BaseException as observer_error:
        return _RunnerDispatchOutcome(
            error=observer_error,
            settlement=settlement,
            settlement_error=settlement_error,
        )
    return outcome


def _runner_command_evidence(
    command: ExecCommand,
    *,
    redactor: SecretRedactor,
    publish_arguments: bool,
) -> dict[str, Any]:
    """Return a secret-safe, byte-bounded command projection for audit events."""

    if type(command) is not ExecCommand:
        raise TypeError("Runner command evidence requires an ExecCommand.")
    if not isinstance(redactor, SecretRedactor):
        raise TypeError("Runner command evidence requires a SecretRedactor.")
    if type(publish_arguments) is not bool:
        raise TypeError("publish_arguments must be a bool.")
    if not publish_arguments:
        return {
            "kind": command.kind,
            "arguments_state": "unavailable",
        }

    source_prefix, source_complete = _runner_command_json_prefix(
        command,
        max_bytes=_RUNNER_COMMAND_EVIDENCE_PREVIEW_MAX_BYTES,
    )
    preview, redaction_truncated = redactor.redact_utf8_head(
        source_prefix,
        max_bytes=_RUNNER_COMMAND_EVIDENCE_PREVIEW_MAX_BYTES,
        source_complete=source_complete,
    )
    truncated = not source_complete or redaction_truncated
    redacted = REDACTED_SECRET in preview
    evidence = {
        "kind": command.kind,
        "arguments_state": (
            "redacted_and_truncated"
            if redacted and truncated
            else ("redacted" if redacted else ("truncated" if truncated else "available"))
        ),
        "preview_format": "exec_command_json_prefix",
        "preview": preview,
        "truncated": truncated,
    }
    if compact_json_utf8_size(evidence) > _RUNNER_COMMAND_EVIDENCE_MAX_BYTES:
        raise AssertionError("Runner command evidence exceeded its hard byte bound.")
    return evidence


def _runner_command_json_prefix(
    command: ExecCommand,
    *,
    max_bytes: int,
) -> tuple[bytes, bool]:
    """Serialize at most one bounded exact prefix of compact command JSON."""

    if type(command) is not ExecCommand:
        raise TypeError("Runner command JSON prefix requires an ExecCommand.")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer.")

    retained = bytearray()

    def append_literal(value: bytes) -> bool:
        remaining = max_bytes - len(retained)
        if len(value) <= remaining:
            retained.extend(value)
            return True
        retained.extend(value[:remaining])
        return False

    def append_text(value: str) -> bool:
        remaining = max_bytes - len(retained)
        if remaining <= 0:
            return False
        # Every source character occupies at least one serialized byte. Slicing
        # first therefore keeps the temporary JSON string bounded even when a
        # command argument itself is arbitrarily large.
        source_complete = len(value) <= remaining
        rendered = json.dumps(
            value if source_complete else value[:remaining],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        serialized_complete = append_literal(rendered)
        return source_complete and serialized_complete

    if command.kind == "process":
        if not append_literal(b'{"argv":['):
            return bytes(retained), False
        argv = command.argv
        if argv is None:  # pragma: no cover - ExecCommand shape invariant
            raise AssertionError("Process command has no argv.")
        for index, argument in enumerate(argv):
            if index and not append_literal(b","):
                return bytes(retained), False
            if not append_text(argument):
                return bytes(retained), False
        complete = append_literal(b'],"kind":"process","shell":null}')
        return bytes(retained), complete

    if not append_literal(b'{"argv":null,"kind":"shell","shell":'):
        return bytes(retained), False
    shell = command.shell
    if shell is None:  # pragma: no cover - ExecCommand shape invariant
        raise AssertionError("Shell command has no script.")
    if not append_text(shell):
        return bytes(retained), False
    complete = append_literal(b"}")
    return bytes(retained), complete


async def _run_runner_settlement_call(
    factory: Callable[[], Awaitable[bool]],
) -> _RunnerSettlementCallOutcome:
    """Contain extension-owned control signals inside the owned task."""

    try:
        return _RunnerSettlementCallOutcome(result=await factory())
    except BaseException as error:
        return _RunnerSettlementCallOutcome(error=error)


class _RetainedRunnerSettlementProbe:
    """Join one exact settlement task before allowing a later fresh probe."""

    __slots__ = ("__factory", "__task")

    def __init__(
        self,
        task: asyncio.Task[_RunnerSettlementCallOutcome],
        factory: Callable[[], Awaitable[bool]],
    ) -> None:
        self.__task: asyncio.Task[_RunnerSettlementCallOutcome] | None = task
        self.__factory = factory
        task.add_done_callback(_observe_retained_runner_settlement_task)

    async def __call__(self) -> bool:
        task = self.__task
        if task is None:
            return await self.__factory()
        if not task.done() and task.get_loop() is not asyncio.get_running_loop():
            # An active task is affine to its original event loop. Starting a
            # replacement probe here could clear the fence while that task is
            # still operating, so cross-loop uncertainty remains quarantined.
            return False
        try:
            outcome = await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                self.__task = None
            raise
        except BaseException:
            self.__task = None
            raise
        self.__task = None
        if outcome.error is not None:
            raise outcome.error
        return outcome.result is True


class _RetainedRunnerDispatchSettlementProbe:
    """Join one abandoned runner dispatch through positive mutation settlement."""

    __slots__ = ("__operation", "__runner")

    def __init__(
        self,
        *,
        operation: _RetainedInvocationOperationProbe,
        runner: Runner,
    ) -> None:
        self.__operation = operation
        self.__runner = runner

    async def __call__(self) -> bool:
        outcome = await self.__operation.outcome()
        if outcome is None:
            return False
        if outcome.operation_started is False:
            return True
        dispatch_outcome = (
            outcome.result if type(outcome.result) is _RunnerDispatchOutcome else None
        )
        if dispatch_outcome is None:
            return False
        if dispatch_outcome.settlement_error is not None:
            raise dispatch_outcome.settlement_error
        settlement = dispatch_outcome.settlement
        if settlement is None:
            return False
        if settlement in {"complete", "runner_quiescent"}:
            return True
        if settlement != "deferred":
            return False
        if not runner_pending_command_settlement_cancellation_safe(self.__runner):
            return False
        settled = await self.__runner.await_pending_command_settlement()
        return type(settled) is bool and settled


def _observe_retained_runner_settlement_task(
    task: asyncio.Task[_RunnerSettlementCallOutcome],
) -> None:
    """Consume diagnostics while retaining the task's replayable outcome."""

    try:
        task.exception()
    except BaseException:
        return


async def _unavailable_runner_settlement_probe() -> bool:
    """Keep an unsafe extension waiter permanently fail-closed."""

    return False


def _runner_settlement_probe(
    runner: Runner,
    factory: Callable[[], Awaitable[bool]],
) -> tuple[Callable[[], Awaitable[bool]], bool]:
    """Return a probe only when its observer task may be cancelled safely."""

    cancellation_safe = runner_pending_command_settlement_cancellation_safe(runner)
    if cancellation_safe:
        return factory, True
    return _unavailable_runner_settlement_probe, False


def _runner_settlement_failure() -> RuntimeError:
    return RuntimeError("Runner mutation settlement failed.")


def _invocation_runner_mutation_settlement(
    *,
    operation_started: object,
    result: object,
    error: BaseException | None,
) -> RunnerWorkspaceMutationSettlement | None:
    """Validate one owned dispatch outcome before classifying quiescence."""

    if type(operation_started) is not bool:
        return None
    if not operation_started:
        return "runner_quiescent"
    if result is not None and type(result) is not ExecResult:
        return None
    return runner_workspace_mutation_settlement(
        result=result,
        error=error,
    )


async def _settle_invocation_runner_mutation(
    *,
    runner: Runner,
    owner: InvocationWorkspaceMutationOwner,
    settlement: RunnerWorkspaceMutationSettlement | None,
    settlement_error: BaseException | None,
    cancellation: asyncio.CancelledError | None,
) -> _RunnerMutationSettlementOutcome:
    """Keep the receipt owner until runner cleanup proves mutation quiescence."""

    def settlement_factory() -> Awaitable[bool]:
        return runner.await_pending_command_settlement()

    retry_probe, cancellation_safe = _runner_settlement_probe(runner, settlement_factory)

    if settlement_error is not None:
        owner.fail_closed(retry_probe)
        process_signal = detached_workspace_mutation_process_signal(settlement_error)
        if process_signal is not None:
            return _RunnerMutationSettlementOutcome(
                cancellation=cancellation,
                failure=process_signal,
            )
        return _RunnerMutationSettlementOutcome(
            cancellation=cancellation,
            failure=_runner_settlement_failure() if cancellation is not None else None,
        )
    if settlement is None:
        owner.fail_closed(retry_probe)
        return _RunnerMutationSettlementOutcome(cancellation=cancellation)
    if settlement in {"complete", "runner_quiescent"}:
        return _RunnerMutationSettlementOutcome(cancellation=cancellation)
    if settlement == "uncertain":
        owner.fail_closed(retry_probe)
        return _RunnerMutationSettlementOutcome(cancellation=cancellation)
    if not cancellation_safe:
        owner.fail_closed(retry_probe)
        return _RunnerMutationSettlementOutcome(cancellation=cancellation)
    if cancellation is not None:
        # Do not start an extension-owned waiter after cancellation is already
        # authoritative. The process fence can invoke the declared-safe probe
        # if this environment is reused without leaving a task in a closing loop.
        owner.fail_closed(retry_probe)
        return _RunnerMutationSettlementOutcome(cancellation=cancellation)

    settlement_task = asyncio.create_task(
        _run_runner_settlement_call(settlement_factory),
        name="cayu-runner-mutation-settlement",
    )
    retained_probe = _RetainedRunnerSettlementProbe(
        settlement_task,
        settlement_factory,
    )
    settlement_owner_resolved = False
    try:
        settlement_outcome = await await_shielded_task_outcome(
            settlement_task,
            timeout_s=_RUNNER_MUTATION_SETTLEMENT_FOREGROUND_TIMEOUT_SECONDS,
            timeout_after_cancellation_s=0.0,
        )
        if settlement_outcome.timed_out:
            owner.fail_closed(retained_probe)
            settlement_owner_resolved = True
            return _RunnerMutationSettlementOutcome(
                cancellation=settlement_outcome.cancellation,
                restore_cancellation_requests=(settlement_outcome.cancellation_requests_consumed),
            )
        settlement_call_outcome = settlement_outcome.result
        if type(settlement_call_outcome) is not _RunnerSettlementCallOutcome:
            settlement_call_outcome = _RunnerSettlementCallOutcome(
                error=RuntimeError("Runner mutation settlement returned an invalid outcome."),
            )
        settlement_error = settlement_outcome.error or settlement_call_outcome.error
        if settlement_error is not None:
            process_signal = detached_workspace_mutation_process_signal(settlement_error)
        else:
            process_signal = None
        if process_signal is not None:
            owner.fail_closed(retry_probe)
            settlement_owner_resolved = True
            settlement_cancellation = settlement_outcome.cancellation
            settlement_cancellation_requests = settlement_outcome.cancellation_requests_consumed
            del (
                retained_probe,
                settlement_call_outcome,
                settlement_error,
                settlement_outcome,
                settlement_task,
            )
            return _RunnerMutationSettlementOutcome(
                cancellation=settlement_cancellation,
                failure=process_signal,
                restore_cancellation_requests=settlement_cancellation_requests,
            )
        if settlement_error is not None or settlement_call_outcome.result is not True:
            owner.fail_closed(retry_probe)
            settlement_owner_resolved = True
            return _RunnerMutationSettlementOutcome(
                cancellation=settlement_outcome.cancellation,
                failure=(
                    _runner_settlement_failure()
                    if settlement_outcome.cancellation is not None
                    else None
                ),
                restore_cancellation_requests=(settlement_outcome.cancellation_requests_consumed),
            )
        settlement_owner_resolved = True
        return _RunnerMutationSettlementOutcome(
            cancellation=settlement_outcome.cancellation,
            restore_cancellation_requests=(settlement_outcome.cancellation_requests_consumed),
        )
    finally:
        if not settlement_owner_resolved:
            # Once the exact task exists, every abnormal supervisor exit must
            # transfer it before the original control signal continues.
            owner.fail_closed(retained_probe)


def _raise_clean_runner_cancellation(
    cancellation_type: type[asyncio.CancelledError],
    args: tuple[object, ...],
    artifacts: list[dict[str, Any]],
    *,
    cause: BaseException | None = None,
    restore_cancellation_requests: int = 0,
) -> NoReturn:
    """Raise cancellation without retaining a runner or raw-request traceback.

    Python may subsequently attach an exception active in the tool's caller.
    The runtime removes that caller-added context at its external cancellation
    boundary, after the complete tool failure tree is available.
    """

    cancellation = _clean_runner_cancellation(cancellation_type, args, artifacts)
    if cause is not None:
        attach_runner_cancellation_failure(cancellation, cause)
    current_task = asyncio.current_task()
    if current_task is not None:
        for _request in range(restore_cancellation_requests):
            current_task.cancel()
    raise cancellation from cause


def _combine_runner_cancellation_causes(
    command_failure: BaseException | None,
    settlement_failure: BaseException,
) -> BaseException:
    if command_failure is None or command_failure is settlement_failure:
        return settlement_failure
    return BaseExceptionGroup(
        "Runner command and mutation settlement failed after caller cancellation.",
        [command_failure, settlement_failure],
    )


def _raise_clean_runner_settlement_signal(signal: BaseException) -> NoReturn:
    """Raise a detached process signal without any runner-owned frame locals."""

    signal.__traceback__ = None
    signal.__cause__ = None
    signal.__context__ = None
    raise signal from None


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
    mutation_owner: InvocationWorkspaceMutationOwner | None = None,
    execution_observer: RunnerExecutionObserver | None = None,
    publish_execution_arguments: bool = True,
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
        mutation_owner=mutation_owner,
        execution_observer=execution_observer,
        publish_execution_arguments=publish_execution_arguments,
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
