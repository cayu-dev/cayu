"""Journaled command steps for deterministic workflow checks.

``command`` runs one process through a registered environment's runner (a test
suite, a linter, a checker script) and journals its outcome under the workflow
run id, the same way ``step`` journals an agent call. On resume the journaled
result is returned instead of running the command again, unless the caller
asks for a fresh run.

On a factory-backed environment the command runs in the allocation the
workflow's steps share (``allocation_scope="parent"`` or ``"causal_budget"``),
so a test command sees the files the steps wrote.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from time import monotonic
from typing import TYPE_CHECKING, Any

from cayu._validation import MAX_DURABLE_JSON_INTEGER, DurableValueError, require_clean_nonblank
from cayu.environments.bindings import (
    DeterministicWorkspaceBinding,
    NativeBinding,
    NoWorkspaceBinding,
)
from cayu.environments.factory import EnvironmentFactoryRequest
from cayu.environments.shared_allocations import AllocationSharingScope, allocation_sharing_scope
from cayu.events import Event, EventType
from cayu.failure_evidence import exception_evidence
from cayu.runners.base import ExecCommand, ExecResult, Runner
from cayu.tools._terminal_bounds import (
    MAX_TOOL_TERMINAL_PAYLOAD_BYTES,
    json_payload_bytes,
    prefix_within_json_bytes,
)
from cayu.tools.commands import DEFAULT_OUTPUT_LIMIT_BYTES, MAX_OUTPUT_LIMIT_BYTES
from cayu.workflows._step_identity import GATED_LOOP_STEP_ID_PREFIX
from cayu.workflows.journal import WorkflowJournalStepCompletion
from cayu.workflows.models import CommandResult, StepError
from cayu.workflows.workflow import (
    WorkflowContext,
    WorkflowSupersededError,
    _workflow_lineage_from_anchor,
)

if TYPE_CHECKING:
    from cayu.runtime._runtime_records import RegisteredEnvironment
    from cayu.sessions.records import Session

COMMAND_STEP_KIND = "command"
# Both streams share one durable workflow.step.completed event.
MAX_COMMAND_OUTPUT_LIMIT_BYTES = MAX_OUTPUT_LIMIT_BYTES // 2
# The completion payload's share of one durable event; the rest is envelope.
_MAX_COMPLETED_PAYLOAD_BYTES = MAX_TOOL_TERMINAL_PAYLOAD_BYTES
# argv, step id and the other completion fields may use at most this much, so
# each stream always keeps at least a quarter of the payload.
_MAX_COMPLETED_METADATA_BYTES = _MAX_COMPLETED_PAYLOAD_BYTES // 2
_REPLACEMENT_CHARACTER = "\ufffd"
# Bindings whose bind() and finalize() only pass the configured workspace and
# runner through, so the runner already sees the workspace without a session.
# Exact types: a subclass may add setup to bind().
_SETUP_FREE_BINDINGS = frozenset({NativeBinding, DeterministicWorkspaceBinding, NoWorkspaceBinding})
_UNSAFE_TEXT = re.compile("[\x00\ud800-\udfff]")


@dataclass(frozen=True, slots=True)
class _CommandEnvironmentRequest:
    """Where one command step runs, in the terms an environment factory uses.

    The lineage mirrors what ``step`` puts on a child ``RunRequest``, so on a
    factory-backed environment the command resolves the same shared allocation
    key as the workflow's agent steps.
    """

    workflow_session_id: str
    workflow_name: str
    step_id: str
    environment_name: str | None
    parent_session_id: str | None
    causal_budget_id: str
    allocation_scope: AllocationSharingScope | None = None


async def command(
    ctx: WorkflowContext,
    *,
    step_id: str,
    argv: Sequence[str],
    stdin: str | None = None,
    timeout_s: int | None = None,
    environment_name: str | None = None,
    cwd: str | None = None,
    env: Mapping[str, str] | None = None,
    output_limit_bytes: int = DEFAULT_OUTPUT_LIMIT_BYTES,
    rerun: bool = False,
    allocation_scope: AllocationSharingScope | str | None = None,
    serialize_writers: bool = False,
) -> CommandResult:
    """Run one command in a registered environment and journal its outcome.

    The command runs through the runner of ``environment_name`` (the app's
    default environment when omitted), so the same workflow runs against a
    local, Docker, or Lambda MicroVM runner. ``workflow.step.started`` and
    ``workflow.step.completed`` record the argv, exit code, duration,
    ``timed_out`` and stdout/stderr redacted with the app's secret redactor and
    bounded to ``output_limit_bytes`` per stream.

    A non-zero exit or a timeout is a normal result, not an error: check
    :attr:`CommandResult.passed` or return the result from a ``gated_loop``
    gate. A runner failure raises :class:`StepError` without journaling a
    completion, so a resume runs the command again.

    On resume, a completed ``step_id`` returns its journaled result with
    ``replayed=True``. Pass ``rerun=True`` when the check must see current state.
    ``env`` values are passed to the process but are not treated as secrets;
    supply secrets through the environment's runner or vault instead.

    A factory-backed environment runs the command in the allocation this run's
    steps share, so it needs a shared ``allocation_scope``: the environment's
    registered one, or ``allocation_scope=`` here (matching the steps'
    ``StepRunOptions.allocation_scope``). The command attaches to that
    allocation, or creates it when no step has yet; either way the workflow
    run releases it, never the command.

    ``serialize_writers=True`` makes the command take turns with the run's
    steps and commands that set it too (``StepRunOptions.serialize_writers``):
    it waits for the run's writer turn, then holds it while it attaches and
    runs. Without it, a command runs alongside parallel steps in the same
    allocation and can race with them on files.
    """
    if not isinstance(ctx, WorkflowContext):
        raise TypeError("command() requires a WorkflowContext.")
    step_id = require_clean_nonblank(step_id, "step_id")
    if step_id.startswith(GATED_LOOP_STEP_ID_PREFIX):
        raise ValueError(
            f"step_id {step_id!r} uses the reserved {GATED_LOOP_STEP_ID_PREFIX!r} namespace."
        )
    if isinstance(argv, str) or not isinstance(argv, Sequence):
        raise TypeError("command() argv must be a sequence of strings.")
    exec_command = ExecCommand.process(*argv)
    owned_argv = tuple(exec_command.argv or ())
    if environment_name is not None:
        environment_name = require_clean_nonblank(environment_name, "environment_name")
    if type(rerun) is not bool:
        raise TypeError("command() rerun must be a bool.")
    if type(serialize_writers) is not bool:
        raise TypeError("command() serialize_writers must be a bool.")
    if (
        type(output_limit_bytes) is not int
        or not 1 <= output_limit_bytes <= MAX_COMMAND_OUTPUT_LIMIT_BYTES
    ):
        raise ValueError(
            "command() output_limit_bytes must be an integer from 1 to "
            f"{MAX_COMMAND_OUTPUT_LIMIT_BYTES}."
        )
    owned_env = None if env is None else dict(env)
    scope = None if allocation_scope is None else allocation_sharing_scope(allocation_scope)

    await ctx._claim_step_id(step_id)
    await ctx._check_fence()

    if not rerun:
        replayed = await _replayed_command_result(ctx, step_id=step_id)
        if replayed is not None:
            return replayed

    # Admitted application work, like an agent step's run: shutdown waits for
    # it, and a sealed application refuses a new one.
    return await ctx.app._run_admitted_workflow_command(
        lambda: _execute_command(
            ctx,
            step_id=step_id,
            exec_command=exec_command,
            owned_argv=owned_argv,
            environment_name=environment_name,
            cwd=cwd,
            owned_env=owned_env,
            timeout_s=timeout_s,
            stdin=stdin,
            output_limit_bytes=output_limit_bytes,
            allocation_scope=scope,
            serialize_writers=serialize_writers,
        )
    )


async def _execute_command(
    ctx: WorkflowContext,
    *,
    step_id: str,
    exec_command: ExecCommand,
    owned_argv: tuple[str, ...],
    environment_name: str | None,
    cwd: str | None,
    owned_env: dict[str, str] | None,
    timeout_s: int | None,
    stdin: str | None,
    output_limit_bytes: int,
    allocation_scope: AllocationSharingScope | None,
    serialize_writers: bool,
) -> CommandResult:
    (await ctx.execution_deadline()).require_admission("workflow_command")
    anchor = await ctx._workflow_anchor()
    parent_session_id, causal_budget_id = _workflow_lineage_from_anchor(ctx, anchor)
    request = _CommandEnvironmentRequest(
        workflow_session_id=ctx.session_id,
        workflow_name=ctx.workflow_name,
        step_id=step_id,
        environment_name=environment_name,
        parent_session_id=parent_session_id,
        causal_budget_id=causal_budget_id,
        allocation_scope=allocation_scope,
    )
    registered = _command_environment(ctx, request)
    resolved_environment_name = registered.spec.name
    redactor = ctx.app._secret_redactor
    journaled_argv = [redactor.redact_text(item) for item in owned_argv]
    # Refused before any runner is chosen, so an oversized argv never creates
    # a shared allocation.
    stream_limit = _journal_stream_limit(
        ctx,
        step_id=step_id,
        argv=journaled_argv,
        environment_name=resolved_environment_name,
        output_limit_bytes=output_limit_bytes,
    )
    async with contextlib.AsyncExitStack() as stack:
        if serialize_writers:
            turns = ctx.app._environment_lifecycle.shared_allocations.writer_turns
            await stack.enter_async_context(turns.turn(ctx.session_id, holder=step_id))
        runner = await stack.enter_async_context(
            _command_runner(ctx, request, registered, anchor=anchor)
        )
        return await _run_and_journal(
            ctx,
            step_id=step_id,
            exec_command=exec_command,
            journaled_argv=journaled_argv,
            resolved_environment_name=resolved_environment_name,
            runner=runner,
            cwd=cwd,
            owned_env=owned_env,
            timeout_s=timeout_s,
            stdin=stdin,
            stream_limit=stream_limit,
        )


async def _run_and_journal(
    ctx: WorkflowContext,
    *,
    step_id: str,
    exec_command: ExecCommand,
    journaled_argv: list[str],
    resolved_environment_name: str,
    runner: Runner,
    cwd: str | None,
    owned_env: dict[str, str] | None,
    timeout_s: int | None,
    stdin: str | None,
    stream_limit: int,
) -> CommandResult:
    redactor = ctx.app._secret_redactor
    started_event = ctx.event(
        EventType.WORKFLOW_STEP_STARTED,
        payload={
            "step_id": step_id,
            "kind": COMMAND_STEP_KIND,
            "argv": journaled_argv,
            "environment_name": resolved_environment_name,
        },
    )
    await ctx._check_fence(establish=False)
    if not await ctx.journal.append_step_started(started_event, attempt_id=ctx.attempt_id):
        await _raise_lost_reservation(ctx, step_id)

    started_at = monotonic()
    try:
        exec_result = await runner.exec_redacted(
            exec_command,
            redactor=redactor,
            cwd=cwd,
            env=owned_env,
            timeout_s=timeout_s,
            stdin=stdin,
            output_limit_bytes=stream_limit,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        raise StepError(
            f"command step {step_id!r} could not run: {type(exc).__name__}",
            step_id=step_id,
            evidence=exception_evidence(exc),
            workflow_attempt_id=ctx.attempt_id,
        ) from exc
    duration_ms = min(MAX_DURABLE_JSON_INTEGER, max(0, int((monotonic() - started_at) * 1000)))
    if type(exec_result) is not ExecResult:
        raise StepError(
            f"command step {step_id!r}: runner did not return an ExecResult",
            step_id=step_id,
            workflow_attempt_id=ctx.attempt_id,
        )

    stdout, stdout_cut = _journal_text(exec_result.stdout, stream_limit)
    stderr, stderr_cut = _journal_text(exec_result.stderr, stream_limit)
    result = CommandResult(
        step_id=step_id,
        argv=tuple(journaled_argv),
        exit_code=exec_result.exit_code,
        timed_out=exec_result.timed_out,
        cancelled=exec_result.cancelled,
        stdout=stdout,
        stderr=stderr,
        stdout_truncated=exec_result.stdout_truncated or stdout_cut,
        stderr_truncated=exec_result.stderr_truncated or stderr_cut,
        duration_ms=duration_ms,
        environment_name=resolved_environment_name,
    )
    completed_event = ctx.event(
        EventType.WORKFLOW_STEP_COMPLETED,
        payload=_completed_payload(result),
    )
    try:
        appended = await ctx.journal.append_current_attempt(
            completed_event, attempt_id=ctx.attempt_id
        )
    except DurableValueError as exc:
        raise StepError(
            f"command step {step_id!r} ran but its completion could not be journaled ({exc.code})",
            step_id=step_id,
            evidence=exception_evidence(exc),
            workflow_attempt_id=ctx.attempt_id,
        ) from exc
    if not appended:
        latest = await ctx.journal.latest_attempt_id()
        if latest is not None:
            ctx._raise_superseded(latest)
        raise WorkflowSupersededError(
            f"Workflow run {ctx.session_id!r} no longer owns attempt "
            f"{ctx.attempt_id!r}; command step completion was not journaled."
        )
    return result


def _command_environment(
    ctx: WorkflowContext, request: _CommandEnvironmentRequest
) -> RegisteredEnvironment:
    """Return the registered environment a command step runs in."""
    try:
        registered = ctx.app._get_registered_environment(request.environment_name)
    except KeyError as exc:
        raise ValueError(f"command step {request.step_id!r}: {exc.args[0]}") from None
    if registered is None:
        raise ValueError(
            f"command step {request.step_id!r} needs an environment: register a "
            "default environment or pass environment_name=."
        )
    return registered


@contextlib.asynccontextmanager
async def _command_runner(
    ctx: WorkflowContext,
    request: _CommandEnvironmentRequest,
    registered: RegisteredEnvironment,
    *,
    anchor: Session | None,
) -> AsyncIterator[Runner]:
    """Yield the runner a command step executes on, for as long as it runs."""
    if registered.factory is None:
        yield _registered_command_runner(request, registered)
        return
    async with _shared_allocation_runner(ctx, request, registered, anchor=anchor) as runner:
        yield runner


def _registered_command_runner(
    request: _CommandEnvironmentRequest, registered: RegisteredEnvironment
) -> Runner:
    """Return the runner of an environment registered with ``register_environment``."""
    name = registered.spec.name
    binding = registered.environment.binding
    if binding is not None and type(binding) not in _SETUP_FREE_BINDINGS:
        # The runtime binds and finalizes a workspace binding (checkout, copy-in,
        # copy-back) only around an agent session. Running here would execute
        # against a workspace the binding never prepared.
        raise ValueError(
            f"command step {request.step_id!r}: environment {name!r} uses "
            f"{type(binding).__name__}, which prepares the workspace per session; "
            "command() runs only on environments whose runner already sees the "
            "workspace (no binding, NativeBinding or NoWorkspaceBinding)."
        )
    runner = registered.environment.runner
    if runner is None:
        raise ValueError(f"command step {request.step_id!r}: environment {name!r} has no runner.")
    return runner


@contextlib.asynccontextmanager
async def _shared_allocation_runner(
    ctx: WorkflowContext,
    request: _CommandEnvironmentRequest,
    registered: RegisteredEnvironment,
    *,
    anchor: Session | None,
) -> AsyncIterator[Runner]:
    """Lend the command the runner of the allocation this run's steps share.

    The factory's own binding prepared that allocation's workspace when the
    allocation was created, so the command runs against a prepared workspace
    whatever the binding is. The command attaches only while it runs; the
    workflow run, as the scope's owner, releases the allocation.
    """
    name = registered.spec.name
    factory = registered.factory
    if factory is None:  # pragma: no cover - checked by the caller
        raise AssertionError("A shared allocation needs a factory-backed environment.")
    shared_allocations = ctx.app._environment_lifecycle.shared_allocations
    scope = request.allocation_scope or shared_allocations.default_scope(name)
    if scope is AllocationSharingScope.SESSION:
        raise ValueError(
            f"command step {request.step_id!r}: environment {name!r} is factory-backed "
            "and is provisioned per session, and a command has no session to own an "
            "allocation. Give the environment, or this command and the steps it should "
            "share a sandbox with, allocation_scope='parent' or 'causal_budget'."
        )
    if anchor is None or request.parent_session_id is None:
        # Without the journal's anchor session, steps aren't children of the run
        # and share nothing, so there is no allocation for the command to join.
        raise ValueError(
            f"command step {request.step_id!r}: environment {name!r} shares one "
            f"allocation per {scope.value} scope, but workflow run {ctx.session_id!r} "
            "has no anchor session for its steps to share it under."
        )
    from cayu.resource_access import require_dispatch

    # The same dispatch check a step's session makes before its factory is called.
    await require_dispatch()
    factory_request = EnvironmentFactoryRequest(
        # Names only the command's attachment. A created allocation is requested
        # under the scope owner's identity, as it is for a step.
        session_id=f"{ctx.session_id}.command.{request.step_id}",
        agent_name=ctx.workflow_name,
        environment_name=name,
        parent_session_id=request.parent_session_id,
        causal_budget_id=request.causal_budget_id,
    )
    async with contextlib.AsyncExitStack() as stack:
        # Claimed with no await after the fence check, as a step claims it, so a
        # superseded attempt never takes ownership back from a newer one.
        await ctx._check_fence(establish=False)
        shared_allocations.open_owner(
            ctx.session_id, attempt_id=ctx.attempt_id, fence=ctx._shared_allocation_fence
        )
        try:
            runner = await stack.enter_async_context(
                shared_allocations.attach_command(
                    factory, scope=scope, parent=anchor, request=factory_request
                )
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise StepError(
                f"command step {request.step_id!r} could not use the shared allocation "
                f"of environment {name!r}: {type(exc).__name__}",
                step_id=request.step_id,
                evidence=exception_evidence(exc),
                workflow_attempt_id=ctx.attempt_id,
            ) from exc
        yield runner


def _journal_stream_limit(
    ctx: WorkflowContext,
    *,
    step_id: str,
    argv: list[str],
    environment_name: str,
    output_limit_bytes: int,
) -> int:
    """Return the per-stream bound that keeps the completion inside one durable event.

    Every completion field except the two streams is measured at its largest
    (integer fields at their maximum) and the streams split what is left.
    Metadata too large to leave room for output is refused before the command
    runs, so a command never runs without being able to journal its outcome.
    """
    skeleton = CommandResult(
        step_id=step_id,
        argv=tuple(argv),
        exit_code=MAX_DURABLE_JSON_INTEGER,
        duration_ms=MAX_DURABLE_JSON_INTEGER,
        environment_name=environment_name,
    )
    metadata_bytes = json_payload_bytes(
        {**_completed_payload(skeleton), "attempt_id": ctx.attempt_id}
    )
    if metadata_bytes > _MAX_COMPLETED_METADATA_BYTES:
        raise ValueError(
            f"command step {step_id!r}: argv and step metadata need {metadata_bytes} bytes "
            f"of the journaled completion; at most {_MAX_COMPLETED_METADATA_BYTES} are allowed."
        )
    return min(output_limit_bytes, (_MAX_COMPLETED_PAYLOAD_BYTES - metadata_bytes) // 2)


def _journal_text(text: str, limit: int) -> tuple[str, bool]:
    """Make redacted output safe for a durable event and bound its JSON size."""
    text = _UNSAFE_TEXT.sub(_REPLACEMENT_CHARACTER, text)
    bounded = prefix_within_json_bytes(text, limit)
    return bounded, len(bounded) != len(text)


def _completed_payload(result: CommandResult) -> dict[str, Any]:
    return {
        "step_id": result.step_id,
        "kind": COMMAND_STEP_KIND,
        "argv": list(result.argv),
        "environment_name": result.environment_name,
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "cancelled": result.cancelled,
        "passed": result.passed,
        "duration_ms": result.duration_ms,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "stdout_truncated": result.stdout_truncated,
        "stderr_truncated": result.stderr_truncated,
    }


async def _replayed_command_result(ctx: WorkflowContext, *, step_id: str) -> CommandResult | None:
    journal = ctx.journal
    if not isinstance(journal, WorkflowJournalStepCompletion):
        raise TypeError(
            "command() replay needs a journal with the WorkflowJournalStepCompletion "
            "capability; pass rerun=True to run the command on every attempt."
        )
    event = await journal.completed_step_event(step_id=step_id, attempt_id=ctx.attempt_id)
    if event is None:
        return None
    return _command_result_from_event(event, step_id=step_id)


def _command_result_from_event(event: Event, *, step_id: str) -> CommandResult:
    payload = event.payload
    if payload.get("step_id") != step_id or payload.get("kind") != COMMAND_STEP_KIND:
        raise StepError(
            f"step_id {step_id!r} already completed as a different kind of step; "
            "give the command step its own step_id.",
            step_id=step_id,
        )
    argv = payload.get("argv")
    environment_name = payload.get("environment_name")
    fields: dict[str, Any] = {}
    for name, expected in (
        ("exit_code", int),
        ("timed_out", bool),
        ("cancelled", bool),
        ("duration_ms", int),
        ("stdout", str),
        ("stderr", str),
        ("stdout_truncated", bool),
        ("stderr_truncated", bool),
    ):
        value = payload.get(name)
        if type(value) is not expected:
            raise StepError(
                f"journaled command step {step_id!r} has a malformed {name!r}; "
                "pass rerun=True to run it again.",
                step_id=step_id,
            )
        fields[name] = value
    if type(argv) is not list or not all(type(item) is str for item in argv):
        raise StepError(
            f"journaled command step {step_id!r} has a malformed 'argv'; "
            "pass rerun=True to run it again.",
            step_id=step_id,
        )
    if environment_name is not None and type(environment_name) is not str:
        environment_name = None
    return CommandResult(
        step_id=step_id,
        argv=tuple(argv),
        environment_name=environment_name,
        replayed=True,
        **fields,
    )


async def _raise_lost_reservation(ctx: WorkflowContext, step_id: str) -> None:
    latest = await ctx.journal.latest_attempt_id()
    if latest is not None and latest != ctx.attempt_id:
        ctx._raise_superseded(latest)
    raise WorkflowSupersededError(
        f"Workflow command step {step_id!r} in run {ctx.session_id!r} was already "
        "reserved in this attempt; this attempt stops to avoid running it twice."
    )
