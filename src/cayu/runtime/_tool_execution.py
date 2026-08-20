from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from cayu._exception_groups import exception_tree_contains
from cayu._task_wait import restore_task_cancellation_requests
from cayu._validation import (
    FrozenJsonDict,
    copy_durable_json_object,
    copy_json_value,
    freeze_json_value,
    thaw_json_value,
)
from cayu.core.tools import Tool, ToolContext, ToolEffect, ToolResult
from cayu.runners import RunnerExecutionError, RunnerUnavailableError
from cayu.runtime import _tool_results as tool_results
from cayu.runtime._durable_subagents import (
    durable_subagent_committed_cancellation_outcome,
    durable_subagent_unsettled_cancellation_outcome,
    is_durable_subagent_submission_unsettled,
)
from cayu.runtime._tool_identity import tool_idempotency_key as tool_idempotency_key
from cayu.runtime.tool_policy import ToolPolicyResult
from cayu.tools._runner import (
    is_current_runner_cancellation_group,
    sanitize_runner_failure_group,
)
from cayu.vaults import SecretRedactor

if TYPE_CHECKING:
    from cayu.runtime._invocation_secrets import InvocationPublicationSnapshot

_OUTCOME_CONSTRUCTION_TOKEN = object()
_BASE_EXCEPTION_ARGS_DESCRIPTOR = BaseException.__dict__["args"]


class _ToolTimeoutCancellationMarker:
    """Per-call authority for the cancellation requested by the tool deadline."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<tool-timeout-cancellation>"

    __str__ = __repr__


class _ToolTimeoutOwner:
    """Own one marker-authenticated timeout request without hiding caller cancellation."""

    __slots__ = (
        "_cancellation_baseline",
        "_expired",
        "_external_request_count",
        "_handle",
        "_latest_external_message",
        "_marker",
        "_request_dispatched",
        "_task",
    )

    def __init__(self, timeout_seconds: float) -> None:
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("Tool timeout requires an active asyncio task.")
        self._task = task
        self._cancellation_baseline = task.cancelling()
        self._marker = _ToolTimeoutCancellationMarker()
        self._expired = False
        self._external_request_count = 0
        self._latest_external_message: object | None = None
        self._request_dispatched = False
        self._handle: asyncio.TimerHandle | None = asyncio.get_running_loop().call_later(
            timeout_seconds,
            self._expire,
        )

    def _expire(self) -> None:
        self._handle = None
        self._expired = True
        # Do not overwrite a caller request that is already pending delivery.
        # The expired owner still classifies the later durable handoff, but it
        # contributes no cancellation request for that handoff to subtract.
        if self._task.cancelling() > self._cancellation_baseline:
            return
        self._request_dispatched = True
        self._task.cancel(self._marker)

    def close(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is not None:
            handle.cancel()

    def consume_request(self) -> None:
        """Remove this owner's request after its timeout callback has fired."""

        pending_requests = max(
            self._task.cancelling() - self._cancellation_baseline,
            0,
        )
        owned_request_is_pending = self._request_dispatched and pending_requests > 0
        self._external_request_count = max(
            pending_requests - int(owned_request_is_pending),
            0,
        )
        if self._external_request_count > 0:
            self._latest_external_message = getattr(self._task, "_cancel_message", None)
        if owned_request_is_pending:
            self._task.uncancel()

    def cancellation_request_count(self) -> int:
        return int(self._request_dispatched)

    def external_request_count(self) -> int:
        return self._external_request_count

    def expired(self) -> bool:
        return self._expired

    def owns(self, cancellation: asyncio.CancelledError) -> bool:
        try:
            args = _BASE_EXCEPTION_ARGS_DESCRIPTOR.__get__(cancellation, BaseException)
        except BaseException:
            return False
        return type(args) is tuple and len(args) == 1 and args[0] is self._marker

    def caller_cancellation(
        self,
        observed: asyncio.CancelledError,
    ) -> asyncio.CancelledError:
        """Return caller evidence without exposing the private deadline marker."""

        if self._external_request_count < 1 or not self.owns(observed):
            return observed
        if self._latest_external_message is None:
            return asyncio.CancelledError()
        return asyncio.CancelledError(self._latest_external_message)


def _caller_cancellation_after_tool_timeout(
    cancellation: asyncio.CancelledError,
    subsequent_cancellation: asyncio.CancelledError | None,
    *,
    timer: _ToolTimeoutOwner,
) -> asyncio.CancelledError:
    """Select the first retained cancellation not owned by the tool deadline."""

    for candidate in (cancellation, subsequent_cancellation):
        if candidate is not None and not timer.owns(candidate):
            return candidate
    return asyncio.CancelledError("Tool execution was cancelled during timeout settlement.")


@dataclass(frozen=True, slots=True, init=False)
class ToolExecutionOutcome:
    """Runtime-owned result plus terminal controls a tool cannot supply.

    Tools are accepted only when they return an exact ``ToolResult``. This
    envelope is created after execution by this module and carries control
    evidence separately, so a forged result or hook cannot clear replay and
    reconciliation semantics.
    """

    result: ToolResult
    _terminal_payload: FrozenJsonDict

    def __init__(
        self,
        result: ToolResult,
        terminal_payload: dict[str, Any],
        *,
        _token: object,
    ) -> None:
        if _token is not _OUTCOME_CONSTRUCTION_TOKEN:
            raise TypeError("ToolExecutionOutcome is runtime-owned.")
        validated_result = tool_results.normalize_tool_result(
            tool_results.validate_tool_result(result)
        )
        copied_payload = copy_durable_json_object(
            terminal_payload,
            "tool_execution_terminal_payload",
        )
        frozen_payload = freeze_json_value(copied_payload)
        if type(frozen_payload) is not FrozenJsonDict:
            raise AssertionError("Frozen terminal tool payload must be an object.")
        object.__setattr__(self, "result", validated_result)
        object.__setattr__(self, "_terminal_payload", frozen_payload)

    @property
    def allows_hook_modification(self) -> bool:
        return not self.publish_before_hooks

    @property
    def publish_before_hooks(self) -> bool:
        """Whether authoritative terminal evidence must precede observational hooks."""

        return bool(self._terminal_payload) and (
            self._terminal_payload.get("tool_effect") != ToolEffect.NONE.value
        )

    def terminal_payload_fields(self) -> dict[str, Any]:
        fields = thaw_json_value(self._terminal_payload)
        if type(fields) is not dict:
            raise AssertionError("Terminal tool payload must thaw to an object.")
        return copy_durable_json_object(fields, "tool_execution_terminal_payload")


def _execution_outcome(
    result: ToolResult,
    terminal_payload: dict[str, Any] | None = None,
) -> ToolExecutionOutcome:
    return ToolExecutionOutcome(
        result,
        terminal_payload or {},
        _token=_OUTCOME_CONSTRUCTION_TOKEN,
    )


async def run_tool(
    *,
    tool: Tool,
    effect: ToolEffect,
    ctx: ToolContext,
    arguments: dict[str, Any],
    redactor: Callable[[], SecretRedactor],
    finalize_publication: Callable[[], InvocationPublicationSnapshot] | None = None,
    timeout_seconds: float | None = None,
) -> ToolExecutionOutcome:
    """Execute one tool and seal its evolving secret scope before publication."""

    if finalize_publication is not None and not callable(finalize_publication):
        raise TypeError("finalize_publication must be callable or None.")
    try:
        outcome = await _run_tool(
            tool=tool,
            effect=effect,
            ctx=ctx,
            arguments=arguments,
            redactor=redactor,
            timeout_seconds=timeout_seconds,
        )
    except BaseException:
        if finalize_publication is not None:
            finalize_publication()
        raise
    if finalize_publication is None:
        return outcome
    publication = finalize_publication()
    if not publication.unsafe_output:
        return outcome
    ctx._discard_policy_denials_for(tool)
    result, controls = tool_results.terminal_failure_result(
        terminal_outcome="invalid_tool_output",
        effect=effect,
        message=(
            "Tool output was omitted because its secret-redaction scope could "
            "not be finalized safely before publication."
        ),
        redactor=publication.redactor,
    )
    return _execution_outcome(result, controls)


async def _run_tool(
    *,
    tool: Tool,
    effect: ToolEffect,
    ctx: ToolContext,
    arguments: dict[str, Any],
    redactor: Callable[[], SecretRedactor],
    timeout_seconds: float | None,
) -> ToolExecutionOutcome:
    timer: _ToolTimeoutOwner | None = None
    grouped_failure: BaseExceptionGroup | None = None
    current_task = asyncio.current_task()
    cancellation_baseline = 0 if current_task is None else current_task.cancelling()
    if type(effect) is not ToolEffect:
        raise TypeError("effect must be a ToolEffect.")
    if not callable(redactor):
        raise TypeError("redactor must be a callable returning SecretRedactor.")
    try:
        if timeout_seconds is None:
            raw_result = await tool.run(ctx, arguments)
        else:

            async def invoke_tool() -> ToolResult:
                return await tool.run(ctx, arguments)

            operation_task = asyncio.create_task(
                invoke_tool(),
                name="cayu-tool-execution",
            )
            timer = _ToolTimeoutOwner(timeout_seconds)
            try:
                # Keep the opaque tool/runner task separate from the deadline
                # owner. Extension code may normalize its own cancellation,
                # but it cannot forge or erase the caller task's cancellation
                # count used to distinguish the deadline from caller control.
                raw_result = await operation_task
            finally:
                timer.close()
                timer.consume_request()
    except asyncio.CancelledError as exc:
        if timer is None:
            raise
        if timer.external_request_count() > 0:
            raise timer.caller_cancellation(exc) from None
        if not timer.cancellation_request_count():
            raise
        # A runner boundary may replace the delivered cancellation while
        # retaining the task's cancellation request. The timeout owner, not
        # the replacement exception's arguments, remains authoritative when
        # no separate caller request exists.
        ctx._discard_policy_denials_for(tool)
        result, controls = tool_results.terminal_failure_result(
            terminal_outcome="tool_execution_timeout",
            effect=effect,
            message=f"Tool call timed out after {timeout_seconds} seconds.",
            redactor=_active_redactor(redactor),
        )
        return _execution_outcome(result, controls)
    except TimeoutError as exc:
        ctx._discard_policy_denials_for(tool)
        if timer is not None and timer.expired():
            active_redactor = _active_redactor(redactor)
            result, controls = tool_results.terminal_failure_result(
                terminal_outcome="tool_execution_timeout",
                effect=effect,
                message=f"Tool call timed out after {timeout_seconds} seconds.",
                redactor=active_redactor,
            )
            return _execution_outcome(result, controls)
        active_redactor = _active_redactor(redactor)
        diagnostic = tool_results.exception_diagnostic(
            exc,
            empty_message="tool execution failed",
            nonportable_message="Tool execution failed with a non-portable diagnostic.",
            redactor=active_redactor,
        )
        result, controls = tool_results.terminal_failure_result(
            terminal_outcome="tool_execution_error",
            effect=effect,
            message=diagnostic.message,
            diagnostic=diagnostic,
            redactor=active_redactor,
        )
        return _execution_outcome(result, controls)
    except BaseExceptionGroup as exc:
        ctx._discard_policy_denials_for(tool)
        if exception_tree_contains(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
            raise
        if timer is not None and timer.expired():
            if timer.external_request_count() > 0:
                grouped_failure = sanitize_runner_failure_group(
                    exc,
                    caller_cancelled=True,
                )
            else:
                result, controls = tool_results.terminal_failure_result(
                    terminal_outcome="tool_execution_timeout",
                    effect=effect,
                    message=f"Tool call timed out after {timeout_seconds} seconds.",
                    redactor=_active_redactor(redactor),
                )
                return _execution_outcome(result, controls)
        elif isinstance(exc, Exception):
            active_redactor = _active_redactor(redactor)
            result, controls = tool_results.terminal_failure_result(
                terminal_outcome="tool_execution_error",
                effect=effect,
                message="Tool execution reported multiple failures.",
                redactor=active_redactor,
            )
            return _execution_outcome(result, controls)
        else:
            current_cancellation = is_current_runner_cancellation_group(exc) or (
                current_task is not None and current_task.cancelling() > cancellation_baseline
            )
            if not current_cancellation:
                active_redactor = _active_redactor(redactor)
                result, controls = tool_results.terminal_failure_result(
                    terminal_outcome="tool_execution_error",
                    effect=effect,
                    message="Tool execution reported multiple failures.",
                    redactor=active_redactor,
                )
                return _execution_outcome(result, controls)
            grouped_failure = sanitize_runner_failure_group(
                exc,
                caller_cancelled=True,
            )
    except (RunnerExecutionError, RunnerUnavailableError) as exc:
        ctx._discard_policy_denials_for(tool)
        active_redactor = _active_redactor(redactor)
        result, controls = _runner_failure_result(
            exc,
            effect=effect,
            redactor=active_redactor,
        )
        return _execution_outcome(result, controls)
    except Exception as exc:
        committed_cancellation = durable_subagent_committed_cancellation_outcome(exc)
        unsettled_cancellation = durable_subagent_unsettled_cancellation_outcome(exc)
        if committed_cancellation is not None:
            (
                committed_result,
                cancellation,
                subsequent_cancellation,
                cancellation_requests_consumed,
            ) = committed_cancellation
            if timer is not None:
                # The durable settlement ran in the isolated tool task, so its
                # consumed count belongs to that child. Caller authority remains
                # pending on this wrapper task and must not be restored twice,
                # whether settlement finishes before or after the deadline.
                external_cancellation_requests = timer.external_request_count()
                if external_cancellation_requests > 0:
                    caller_cancellation = _caller_cancellation_after_tool_timeout(
                        cancellation,
                        subsequent_cancellation,
                        timer=timer,
                    )
                    raise caller_cancellation from None
                if timer.expired():
                    raw_result = committed_result
                else:
                    restore_task_cancellation_requests(
                        cancellation_requests_consumed,
                        cancellation=cancellation,
                    )
                    raise cancellation from None
            else:
                restore_task_cancellation_requests(
                    cancellation_requests_consumed,
                    cancellation=cancellation,
                )
                raise cancellation from None
        elif unsettled_cancellation is not None:
            (
                unsettled,
                cancellation,
                subsequent_cancellation,
                cancellation_requests_consumed,
            ) = unsettled_cancellation
            if timer is not None:
                # Preserve caller requests already pending on the wrapper;
                # child settlement counts are evidence for signal selection,
                # not requests that should be replayed into this task, whether
                # settlement finishes before or after the deadline.
                external_cancellation_requests = timer.external_request_count()
                if external_cancellation_requests > 0:
                    caller_cancellation = _caller_cancellation_after_tool_timeout(
                        cancellation,
                        subsequent_cancellation,
                        timer=timer,
                    )
                    raise caller_cancellation from unsettled
                if timer.expired():
                    raise unsettled from None
                restore_task_cancellation_requests(
                    cancellation_requests_consumed,
                    cancellation=cancellation,
                )
                raise cancellation from unsettled
            else:
                restore_task_cancellation_requests(
                    cancellation_requests_consumed,
                    cancellation=cancellation,
                )
                raise cancellation from unsettled
        elif is_durable_subagent_submission_unsettled(
            exc,
            parent_session_id=ctx.session_id,
            idempotency_key=ctx.idempotency_key,
        ):
            raise
        else:
            ctx._discard_policy_denials_for(tool)
            active_redactor = _active_redactor(redactor)
            diagnostic = tool_results.exception_diagnostic(
                exc,
                empty_message="tool execution failed",
                nonportable_message="Tool execution failed with a non-portable diagnostic.",
                redactor=active_redactor,
            )
            result, controls = tool_results.terminal_failure_result(
                terminal_outcome="tool_execution_error",
                effect=effect,
                message=diagnostic.message,
                diagnostic=diagnostic,
                redactor=active_redactor,
            )
            return _execution_outcome(result, controls)

    if grouped_failure is not None:
        raise grouped_failure
    if type(raw_result) is not ToolResult:
        ctx._discard_policy_denials_for(tool)
        active_redactor = _active_redactor(redactor)
        result, controls = tool_results.terminal_failure_result(
            terminal_outcome="invalid_tool_output",
            effect=effect,
            message=(
                "Tool returned invalid result type: "
                f"{_safe_type_name(raw_result)}. Expected ToolResult."
            ),
            raw_evidence=tool_results.raw_tool_result_evidence(raw_result),
            redactor=active_redactor,
        )
        return _execution_outcome(result, controls)
    try:
        validated_result = tool_results.normalize_tool_result(
            tool_results.validate_tool_result(raw_result)
        )
    except Exception as exc:
        ctx._discard_policy_denials_for(tool)
        active_redactor = _active_redactor(redactor)
        diagnostic = tool_results.exception_diagnostic(
            exc,
            empty_message="tool result validation failed",
            nonportable_message="Tool returned a non-portable result after execution.",
            redactor=active_redactor,
        )
        result, controls = tool_results.terminal_failure_result(
            terminal_outcome="invalid_tool_output",
            effect=effect,
            message=diagnostic.message,
            raw_evidence=tool_results.raw_tool_result_evidence(raw_result),
            diagnostic=diagnostic,
            redactor=active_redactor,
        )
        return _execution_outcome(result, controls)
    return _execution_outcome(validated_result)


def _runner_failure_result(
    error: RunnerExecutionError | RunnerUnavailableError,
    *,
    effect: ToolEffect,
    redactor: SecretRedactor,
) -> tuple[ToolResult, dict[str, Any]]:
    """Preserve only the runner boundary's fixed typed failure evidence."""

    if isinstance(error, RunnerUnavailableError):
        source = _base_exception_namespace_value(error, "diagnostic")
        if type(source) is not dict:
            source = {}
        source = cast("dict[str, Any]", source)
        adapter = source.get("adapter")
        diagnostic = {
            "type": "cayu.runner_unavailable.v1",
            "adapter": (
                adapter
                if adapter in {"docker", "e2b", "lambda-microvm", "local", "microsandbox"}
                else "unknown"
            ),
            "status": "unavailable",
            "error_type": "RunnerUnavailableError",
        }
        error_code = "runner_unavailable"
    else:
        diagnostic = copy_durable_json_object(
            error.diagnostic,
            "runner_failure_diagnostic",
        )
        error_code = "runner_execution_failed"
    result, controls = tool_results.terminal_failure_result(
        terminal_outcome="tool_execution_error",
        effect=effect,
        message=(
            "Runner is unavailable."
            if isinstance(error, RunnerUnavailableError)
            else "Runner command execution failed."
        ),
        redactor=redactor,
    )
    structured = dict(result.structured or {})
    structured.update(
        {
            "error": error_code,
            "diagnostic": diagnostic,
        }
    )
    return (
        ToolResult(
            content=result.content,
            structured=structured,
            artifacts=[diagnostic],
            is_error=result.is_error,
        ),
        controls,
    )


def _base_exception_namespace_value(error: BaseException, name: str) -> object:
    try:
        namespace = BaseException.__dict__["__dict__"].__get__(error, BaseException)
    except BaseException:
        return None
    return dict.get(namespace, name) if type(namespace) is dict else None


def _active_redactor(redactor: Callable[[], SecretRedactor]) -> SecretRedactor:
    active = redactor()
    if not isinstance(active, SecretRedactor):
        raise TypeError("redactor must return a SecretRedactor.")
    return active


def _safe_type_name(value: Any) -> str:
    try:
        name = type.__getattribute__(type(value), "__name__")
    except Exception:
        return "unknown"
    return name if type(name) is str and name.isidentifier() else "unknown"


def policy_denial_reason(policy_result: ToolPolicyResult) -> str:
    return policy_result.reason or "Tool call denied by policy."


def blocked_tool_result(policy_result: ToolPolicyResult, *, reason: str) -> ToolResult:
    return ToolResult(
        content=reason,
        structured={
            "decision": policy_result.decision.value,
            "reason": reason,
            "metadata": policy_result.metadata,
        },
        is_error=True,
    )


def context_metadata(
    *,
    request_metadata: dict[str, Any] | None = None,
    tool_call_id: str,
    approval_id: str | None,
    idempotency_key: str | None = None,
    tool_effect: ToolEffect | None = None,
    input_id: str | None = None,
) -> dict[str, Any]:
    metadata = copy_json_value(request_metadata or {}, "request_metadata")
    metadata["tool_call_id"] = tool_call_id
    if idempotency_key is not None:
        metadata["idempotency_key"] = idempotency_key
    if tool_effect is not None:
        metadata["tool_effect"] = tool_effect.value
    if approval_id is not None:
        metadata["approval_id"] = approval_id
    if input_id is not None:
        metadata["input_id"] = input_id
    return metadata


def validate_tool_policy_result(result: ToolPolicyResult) -> ToolPolicyResult:
    if type(result) is not ToolPolicyResult:
        raise TypeError(
            "Tool policies must return ToolPolicyResult instances. "
            f"Received {type(result).__name__}."
        )
    return ToolPolicyResult(
        decision=result.decision,
        reason=result.reason,
        metadata=copy_json_value(result.metadata, "metadata"),
        approval_expires_in_seconds=result.approval_expires_in_seconds,
    )
