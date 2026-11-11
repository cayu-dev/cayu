"""Do not swallow the caller's cancellation while waiting for something.

``with suppress(CancelledError): await child`` (or ``except CancelledError:
pass`` around an await) is meant to absorb the child's own cancellation after
``child.cancel()``. It also absorbs a cancellation of the task doing the wait
that arrives meanwhile, and the caller then goes on as if nothing happened: a
worker that was told to stop keeps claiming work. Stop a child with
``cayu._task_wait.cancel_and_join``, or wait with ``await_shielded_task_outcome``
and re-raise its ``cancellation``; both consume only the child's cancellation.

The ratchet covers every module of the package, discovered by glob. It flags a
``suppress`` of ``CancelledError`` or ``BaseException`` whose body awaits (also
entered on an ``ExitStack`` in a function that awaits), and a
``CancelledError``, ``BaseException`` or bare handler of a ``try`` that awaits
whose body only passes, continues, breaks, returns a constant, sets a flag to a
constant or logs. Import aliases (also inside functions), module- and
class-level constants of exception types and ``suppress(*types)`` are resolved.
A handler that returns what it captured, or flags the stop and acts on it
later, is not told apart from one that drops it: those are listed with why. The allow-list holds the sites left today, with why each stays, and
may only shrink.
"""

from __future__ import annotations

import ast
import collections
from collections.abc import Iterable
from pathlib import Path

import pytest
from tests.core.test_settlement_ratchet import _BARE, _handler_type_names, _type_aliases

_ROOT = Path(__file__).resolve().parents[2] / "src" / "cayu"
_SWALLOWED = frozenset({"CancelledError", "BaseException"})
_LOG_METHODS = frozenset({"debug", "info", "warning", "error", "exception", "critical", "log"})

# path -> function qualname -> count of remaining sites, with why each stays.
ALLOWED: dict[str, dict[str, int]] = {
    "runtime/_environment_exposure.py": {
        # ``await asyncio.sleep(0)`` taking the delivery of a cancellation the code
        # itself just restored, immediately before raising that same cancellation.
        "_await_exposure_admission_settlement": 1,
        "await_environment_exposure_settlement": 1,
    },
    "runtime/_environment_lifecycle.py": {
        # ``await asyncio.sleep(0)`` taking the delivery of a cancellation the code
        # itself just restored, immediately before raising that same cancellation.
        "EnvironmentLifecycle._await_pre_exposure_admission_settlement": 1,
    },
    "runtime/_tool_invocation/cancellation.py": {
        # ``await asyncio.sleep(0)`` taking the delivery of a cancellation the code
        # itself just restored, immediately before raising that same cancellation.
        "_receive_restored_post_tool_cancellation": 1,
    },
    "runtime/_local_execution_attempt_owner.py": {
        # Best-effort cleanup inside a handler that re-raises the cancellation or
        # failure it caught, or raises its own: a second cancellation during the
        # cleanup has nothing left to stop.
        "run_owned_local_execution_attempt": 2,
    },
    "runtime/_session_control.py": {
        # Best-effort cleanup inside a handler that re-raises the cancellation or
        # failure it caught, or raises its own: a second cancellation during the
        # cleanup has nothing left to stop.
        "SessionControl.stream_with_out_of_band_events": 1,
    },
    "runtime/_tool_round_staging.py": {
        # Best-effort cleanup inside a handler that re-raises the cancellation or
        # failure it caught, or raises its own: a second cancellation during the
        # cleanup has nothing left to stop.
        "_ToolRoundPublicationCoordinator._stage_terminal_owned": 1,
    },
    "egress/aws_lambda_microvm_adapter.py": {
        # Best-effort cleanup inside a handler that re-raises the cancellation or
        # failure it caught, or raises its own: a second cancellation during the
        # cleanup has nothing left to stop.
        "LambdaMicroVMEgressAdapter._admit_runner": 2,
    },
    "egress/transitions.py": {
        # Best-effort cleanup inside a handler that re-raises the cancellation or
        # failure it caught, or raises its own: a second cancellation during the
        # cleanup has nothing left to stop.
        "EgressAuthorityTransitionCoordinator.install": 1,
    },
    "environments/deferred.py": {
        # Best-effort cleanup inside a handler that re-raises the cancellation or
        # failure it caught, or raises its own: a second cancellation during the
        # cleanup has nothing left to stop.
        "DeferredMaterialization._materialize_once": 2,
    },
    "mcp/_stdio_process.py": {
        # Best-effort cleanup inside a handler that re-raises the cancellation or
        # failure it caught, or raises its own: a second cancellation during the
        # cleanup has nothing left to stop.
        "preflight_stdio_mcp_parent_death_containment": 1,
        # Stopping a task it owns, where a caller cancellation arriving during the
        # wait is absorbed too: a close, stop or lifespan exit while the component
        # shuts down, a start that was cancelled, or a finished process's control
        # task. Candidates for cancel_and_join.
        "ContainedStdioMcpProcess._wait_and_close_owner": 1,
    },
    "mcp/tools.py": {
        # Best-effort cleanup inside a handler that re-raises the cancellation or
        # failure it caught, or raises its own: a second cancellation during the
        # cleanup has nothing left to stop.
        "_McpToolSource.call_tool": 1,
        # A background task ending on its own cancellation: it is the task being
        # stopped, not a caller waiting for one.
        "_McpToolSource._run_notification_refresh": 1,
        # Stopping a task it owns, where a caller cancellation arriving during the
        # wait is absorbed too: a close, stop or lifespan exit while the component
        # shuts down, a start that was cancelled, or a finished process's control
        # task. Candidates for cancel_and_join.
        "_McpToolSource.close": 1,
    },
    "memory/recall.py": {
        # Best-effort cleanup inside a handler that re-raises the cancellation or
        # failure it caught, or raises its own: a second cancellation during the
        # cleanup has nothing left to stop.
        "KnowledgeRecallSource.retrieve": 1,
    },
    "connectors/snowflake/toolkit.py": {
        # Best-effort cleanup inside a handler that re-raises the cancellation or
        # failure it caught, or raises its own: a second cancellation during the
        # cleanup has nothing left to stop.
        "_Statement._poll": 1,
    },
    "server/routes.py": {
        # Best-effort cleanup inside a handler that re-raises the cancellation or
        # failure it caught, or raises its own: a second cancellation during the
        # cleanup has nothing left to stop.
        "_start_accepted_mutation": 2,
        # Best-effort close of an observer's event stream; it must not replace the
        # stream's own outcome.
        "_close_event_stream": 1,
    },
    "sandbox_broker/runner.py": {
        # Best-effort cleanup inside a handler that re-raises the cancellation or
        # failure it caught, or raises its own: a second cancellation during the
        # cleanup has nothing left to stop.
        "BrokeredSandboxRunner._settle_cancelled.settle": 1,
        # Answers what it observed (closed, settled, readable) for an operation the
        # caller is tearing down or probing; a caller cancellation that would matter
        # is re-raised by an earlier handler or checked against its baseline.
        "BrokeredSandboxRunner._settle_cancelled": 1,
    },
    "workflows/workflow.py": {
        # Best-effort cleanup inside a handler that re-raises the cancellation or
        # failure it caught, or raises its own: a second cancellation during the
        # cleanup has nothing left to stop.
        "_run_step.settle_cancelled_child": 1,
    },
    "runtime/_compaction/explicit.py": {
        # Waits out repeated cancellation inside the handler of the cancellation it
        # already holds, until owned work settles; that cancellation is raised (or
        # kept) afterwards.
        "SessionCompaction.run._run_provider_dispatch": 1,
    },
    "runtime/_model_policy.py": {
        # Waits out repeated cancellation inside the handler of the cancellation it
        # already holds, until owned work settles; that cancellation is raised (or
        # kept) afterwards.
        "ModelPolicy.close": 1,
    },
    "runtime/evidence_spool.py": {
        # Waits out repeated cancellation inside the handler of the cancellation it
        # already holds, until owned work settles; that cancellation is raised (or
        # kept) afterwards.
        "EvidenceSpool.run_owned": 1,
    },
    "storage/_postgres_verified_work.py": {
        # Waits out repeated cancellation inside the handler of the cancellation it
        # already holds, until owned work settles; that cancellation is raised (or
        # kept) afterwards.
        "PostgresMutationMixin._await_owned_store_mutation": 1,
        # Captures nothing: waits for owned work on behalf of a supervising caller
        # that already holds the cancellation, absorbing a repeated delivery so the
        # wait (or its deadline) completes.
        "_wait_for_postgres_mutation_task": 1,
    },
    "runtime/_recovery_claims.py": {
        # Captures nothing: waits for owned work on behalf of a supervising caller
        # that already holds the cancellation, absorbing a repeated delivery so the
        # wait (or its deadline) completes.
        "_RecoveryWorkerSettlement.await_worker_settlement": 1,
    },
    "collaboration/_ownership.py": {
        # Records the stop in a flag that the code after it acts on: it re-raises a
        # cancellation, or reports the stop as the outcome it returns.
        "_MutationOwners.run": 1,
    },
    "mcp/http.py": {
        # Records the stop in a flag that the code after it acts on: it re-raises a
        # cancellation, or reports the stop as the outcome it returns.
        "HttpMcpSession._delete_session": 1,
        # Answers what it observed (closed, settled, readable) for an operation the
        # caller is tearing down or probing; a caller cancellation that would matter
        # is re-raised by an earlier handler or checked against its baseline.
        "_wait_for_http_close_task": 1,
        # A background task ending on its own cancellation: it is the task being
        # stopped, not a caller waiting for one.
        "HttpMcpSession._run_server_listener": 1,
        # Stopping a task it owns, where a caller cancellation arriving during the
        # wait is absorbed too: a close, stop or lifespan exit while the component
        # shuts down, a start that was cancelled, or a finished process's control
        # task. Candidates for cancel_and_join.
        "HttpMcpSession._stop_server_listener": 1,
    },
    "runtime/_process_workers.py": {
        # Two record the stop in a flag and re-raise the cancellation once the
        # spawn or cleanup settled. One is in the finally, which must go on to
        # signal and reap the process groups even when the supervisor is
        # cancelled; raising there would leak them.
        "supervise_process_workers": 3,
        # Process supervision: the finally must go on to signal and reap the process
        # groups even when the supervisor is cancelled; raising there would leak them.
        "supervise_process_workers.drain_group": 1,
    },
    "tools/patches.py": {
        # Records the stop in a flag that the code after it acts on: it re-raises a
        # cancellation, or reports the stop as the outcome it returns.
        "ApplyPatchTool._apply": 2,
        # Answers what it observed (closed, settled, readable) for an operation the
        # caller is tearing down or probing; a caller cancellation that would matter
        # is re-raised by an earlier handler or checked against its baseline.
        "_settle_patch_journal": 1,
        "_observe_path": 1,
    },
    "providers/openai.py": {
        # Answers what it observed (closed, settled, readable) for an operation the
        # caller is tearing down or probing; a caller cancellation that would matter
        # is re-raised by an earlier handler or checked against its baseline.
        "_close_openai_operation_stream": 1,
    },
    "mcp/_stdio_subscription.py": {
        # A background task ending on its own cancellation: it is the task being
        # stopped, not a caller waiting for one.
        "StdioToolSubscription._run": 1,
    },
    "sessions/cleanup.py": {
        # A background task ending on its own cancellation: it is the task being
        # stopped, not a caller waiting for one.
        "RecoveryCleanupSupervisor._run_background_continuation": 1,
    },
    "connections/models.py": {
        # Stopping a task it owns, where a caller cancellation arriving during the
        # wait is absorbed too: a close, stop or lifespan exit while the component
        # shuts down, a start that was cancelled, or a finished process's control
        # task. Candidates for cancel_and_join.
        "ModelSlots.close": 1,
    },
    "managed_settings.py": {
        # Stopping a task it owns, where a caller cancellation arriving during the
        # wait is absorbed too: a close, stop or lifespan exit while the component
        # shuts down, a start that was cancelled, or a finished process's control
        # task. Candidates for cancel_and_join.
        "ManagedSettings.close": 1,
    },
    "mcp/stdio.py": {
        # Stopping a task it owns, where a caller cancellation arriving during the
        # wait is absorbed too: a close, stop or lifespan exit while the component
        # shuts down, a start that was cancelled, or a finished process's control
        # task. Candidates for cancel_and_join.
        "StdioMcpSession._cancel_background_task": 1,
    },
    "sandbox_broker/lease.py": {
        # Stopping a task it owns, where a caller cancellation arriving during the
        # wait is absorbed too: a close, stop or lifespan exit while the component
        # shuts down, a start that was cancelled, or a finished process's control
        # task. Candidates for cancel_and_join.
        "SandboxLeaseKeeper.stop": 1,
    },
    "server/__init__.py": {
        # Stopping a task it owns, where a caller cancellation arriving during the
        # wait is absorbed too: a close, stop or lifespan exit while the component
        # shuts down, a start that was cancelled, or a finished process's control
        # task. Candidates for cancel_and_join.
        "_cancel_background_task": 1,
    },
    "server/evals_worker.py": {
        # Stopping a task it owns, where a caller cancellation arriving during the
        # wait is absorbed too: a close, stop or lifespan exit while the component
        # shuts down, a start that was cancelled, or a finished process's control
        # task. Candidates for cancel_and_join.
        "EvalRunCoordinator.drain": 1,
        "EvalRunCoordinator.stop": 1,
        "EvalRunCoordinator._cancel_task": 1,
    },
    "storage/tasks_postgres.py": {
        # Stopping a task it owns, where a caller cancellation arriving during the
        # wait is absorbed too: a close, stop or lifespan exit while the component
        # shuts down, a start that was cancelled, or a finished process's control
        # task. Candidates for cancel_and_join.
        "PostgresTaskStore.close": 1,
    },
    "cloud_inventory.py": {
        # Stopping a task it owns, where a caller cancellation arriving during the
        # wait is absorbed too: a close, stop or lifespan exit while the component
        # shuts down, a start that was cancelled, or a finished process's control
        # task. Candidates for cancel_and_join.
        "cloud_inventory_lifespan": 1,
    },
    "external_agents/claude.py": {
        # Stopping a task it owns, where a caller cancellation arriving during the
        # wait is absorbed too: a close, stop or lifespan exit while the component
        # shuts down, a start that was cancelled, or a finished process's control
        # task. Candidates for cancel_and_join.
        "_owned_messages.close": 1,
    },
    "runners/docker.py": {
        # Stopping a task it owns, where a caller cancellation arriving during the
        # wait is absorbed too: a close, stop or lifespan exit while the component
        # shuts down, a start that was cancelled, or a finished process's control
        # task. Candidates for cancel_and_join.
        "_defer_docker_probe_settlement.settle_guest": 1,
    },
    "runners/e2b.py": {
        # Stopping a task it owns, where a caller cancellation arriving during the
        # wait is absorbed too: a close, stop or lifespan exit while the component
        # shuts down, a start that was cancelled, or a finished process's control
        # task. Candidates for cancel_and_join.
        "E2BRunner._resolve_started_handle_after_interruption": 1,
        "E2BRunner._cancel_or_track_start_task": 1,
    },
    "tools/_browser_guest.py": {
        # Stopping a task it owns, where a caller cancellation arriving during the
        # wait is absorbed too: a close, stop or lifespan exit while the component
        # shuts down, a start that was cancelled, or a finished process's control
        # task. Candidates for cancel_and_join.
        "_InteractiveDaemon._close_context_after_limit_abort": 1,
        # Process and guest entry points: a finally on the way out of a CLI worker,
        # an eval worker process or a browser guest command, where the outcome in
        # flight ends the process.
        "_interactive_download_path": 2,
    },
    "cli/_eval_processes.py": {
        # Process and guest entry points: a finally on the way out of a CLI worker,
        # an eval worker process or a browser guest command, where the outcome in
        # flight ends the process.
        "_worker": 1,
    },
    "cli/worker.py": {
        # Process and guest entry points: a finally on the way out of a CLI worker,
        # an eval worker process or a browser guest command, where the outcome in
        # flight ends the process.
        "_invoke_worker": 3,
        "_until_idle_with_evals": 1,
    },
    "evals/internal/_browser_page_fault_guest.py": {
        # Process and guest entry points: a finally on the way out of a CLI worker,
        # an eval worker process or a browser guest command, where the outcome in
        # flight ends the process.
        "main.start.handle": 1,
    },
    "egress/runtime.py": {
        # Egress teardown: after a timed-out drain, stopping the drain before its
        # timeout is raised, and a best-effort audit drain.
        "_EgressManagedRunner.prepare_workspace_sync": 1,
        "_EgressTeardownBinding._drain_audit": 1,
    },
    "knowledge/enrichment.py": {
        # The enrichment worker owns its claims outside the settlement structure; its
        # finally joins the coordination task before the heartbeat join that does
        # keep cancellation. Moving the worker onto the claim structure is separate.
        "KnowledgeEnrichmentWorker._process_claimed": 1,
    },
    "runtime/_run_limits.py": {
        # Finally blocks on the failure and cancellation paths: the exception in
        # flight already ends the caller. The success paths stop their heartbeat with
        # cancel_and_join.
        "RunLimitController.model_step_events_with_heartbeat": 2,
        "RunLimitController.run_operation_with_reservation_heartbeat": 2,
    },
    "runtime/_session_finalization.py": {
        # Best-effort abandonment finalization. Its callers are abandoning a run,
        # usually because of the very cancellation, and the abandoned-session sweep
        # repeats whatever this attempt did not finish.
        "SessionFinalization.finalize_abandoned_session_run": 1,
        "SessionFinalization.finalize_abandoned_session_by_id": 2,
        "SessionFinalization.finalize_abandoned_without_registered_runtime": 1,
    },
    "runtime/_shared_allocation_records.py": {
        # Stops the disposal-lease renewal when the lease holder's block exits; a
        # renewal failure must not replace the block's own outcome.
        "keeping_disposal_lease": 1,
    },
    "tasks/worker.py": {
        # Joins the handler it cancelled on the worker's shutdown path, before the
        # claim is restarted or its cancellation settled; the shutdown is already
        # the outcome.
        "_restart_task_handler_after_shutdown.cleanup": 1,
    },
}


class _Module:
    """Names that stand for ``suppress`` or for exception types in one module."""

    def __init__(self, tree: ast.AST) -> None:
        self.suppress = {"suppress"}
        self.types = _type_aliases(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "suppress" and alias.asname is not None:
                        self.suppress.add(alias.asname)

    def type_names(self, node: ast.expr | None) -> list[str]:
        return _handler_type_names(node, self.types)

    def suppressed(self, context: ast.expr) -> set[str]:
        """The exception types a ``suppress(...)`` call names; empty for anything else."""

        if not isinstance(context, ast.Call):
            return set()
        func = context.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name not in self.suppress:
            return set()
        return {type_name for arg in context.args for type_name in self.type_names(arg)}


def _functions(tree: ast.AST) -> list[tuple[str, ast.AST]]:
    found: list[tuple[str, ast.AST]] = [("<module>", tree)]

    def visit(node: ast.AST, scope: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                if not isinstance(child, ast.ClassDef):
                    found.append((".".join([*scope, child.name]), child))
                visit(child, [*scope, child.name])
            else:
                visit(child, scope)

    visit(tree, [])
    return found


def _own_nodes(nodes: list[ast.AST]) -> list[ast.AST]:
    """``nodes`` and their descendants, without nested functions or classes."""

    found: list[ast.AST] = []
    pending = list(nodes)
    while pending:
        node = pending.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda):
            continue
        found.append(node)
        pending.extend(ast.iter_child_nodes(node))
    return found


def _awaits(statements: Iterable[ast.AST]) -> bool:
    return any(
        isinstance(node, ast.Await | ast.AsyncFor | ast.AsyncWith)
        for node in _own_nodes(list(statements))
    )


def _is_log_call(statement: ast.stmt) -> bool:
    """``logger.warning(...)``, or a bare logging function such as ``log(...)``."""

    if not (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)):
        return False
    func = statement.value.func
    if isinstance(func, ast.Attribute):
        return func.attr in _LOG_METHODS
    return isinstance(func, ast.Name) and "log" in func.id.lower()


def _is_constant(node: ast.expr | None) -> bool:
    if isinstance(node, ast.Tuple):
        return all(_is_constant(element) for element in node.elts)
    return node is None or isinstance(node, ast.Constant)


def _absorbs(statement: ast.stmt) -> bool:
    """A handler statement that neither re-raises nor hands the signal on."""

    if isinstance(statement, ast.Pass | ast.Continue | ast.Break):
        return True
    if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
        return True
    if isinstance(statement, ast.Return):
        return _is_constant(statement.value)
    if isinstance(statement, ast.Assign | ast.AnnAssign | ast.AugAssign):
        # Only records that it stopped (``stopped = True``), handing nothing on.
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        return all(isinstance(target, ast.Name | ast.Attribute) for target in targets) and (
            statement.value is not None and _is_constant(statement.value)
        )
    return _is_log_call(statement)


def _swallows(node: ast.AST, module: _Module, function_awaits: bool) -> bool:
    if isinstance(node, ast.With | ast.AsyncWith):
        return _awaits(node.body) and any(
            _SWALLOWED & module.suppressed(item.context_expr) for item in node.items
        )
    if isinstance(node, ast.Try | ast.TryStar):
        return _awaits(node.body) and any(
            (_SWALLOWED | {_BARE}) & set(module.type_names(handler.type))
            and all(_absorbs(statement) for statement in handler.body)
            for handler in node.handlers
        )
    if isinstance(node, ast.Call):
        # ``stack.enter_context(suppress(CancelledError))`` in a function that awaits.
        func = node.func
        return (
            function_awaits
            and isinstance(func, ast.Attribute)
            and func.attr == "enter_context"
            and bool(node.args)
            and bool(_SWALLOWED & module.suppressed(node.args[0]))
        )
    return False


def _swallowing_sites(tree: ast.AST) -> collections.Counter[str]:
    module = _Module(tree)
    found: collections.Counter[str] = collections.Counter()
    for name, function in _functions(tree):
        nodes = _own_nodes(list(ast.iter_child_nodes(function)))
        function_awaits = name != "<module>" and _awaits(list(ast.iter_child_nodes(function)))
        found[name] += sum(_swallows(node, module, function_awaits) for node in nodes)
    return +found


def _discovered() -> dict[str, collections.Counter[str]]:
    found: dict[str, collections.Counter[str]] = {}
    for path in sorted(_ROOT.rglob("*.py")):
        sites = _swallowing_sites(ast.parse(path.read_text()))
        if sites:
            found[path.relative_to(_ROOT).as_posix()] = sites
    return found


def test_no_new_wait_swallows_the_callers_cancellation() -> None:
    found = _discovered()
    grown = {
        (relative, name): count
        for relative, sites in found.items()
        for name, count in sites.items()
        if count > ALLOWED.get(relative, {}).get(name, 0)
    }
    assert not grown, (
        f"Waits that swallow the caller's cancellation: {grown}. Stop a child with "
        "cayu._task_wait.cancel_and_join, or re-raise await_shielded_task_outcome's "
        "cancellation."
    )
    shrunk = {
        (relative, name): count
        for relative, allowed in ALLOWED.items()
        for name, count in allowed.items()
        if found.get(relative, collections.Counter())[name] < count
    }
    assert not shrunk, f"Sites were fixed; shrink the allow-list: {shrunk}"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "async def stop(task):\n    task.cancel()\n"
            "    with contextlib.suppress(asyncio.CancelledError):\n        await task\n",
            {"stop": 1},
        ),
        (
            "async def stop(task):\n    with suppress(Exception, BaseException):\n"
            "        await task\n",
            {"stop": 1},
        ),
        (
            "async def stop(task):\n    try:\n        await task\n"
            "    except asyncio.CancelledError:\n        pass\n",
            {"stop": 1},
        ),
        (
            "async def stop(task):\n    try:\n        await task\n    except:\n        pass\n",
            {"stop": 1},
        ),
        (
            "async def stop(task):\n    failure = await cancel_and_join(task)\n",
            {},
        ),
        (
            "def stop(task):\n    with contextlib.suppress(asyncio.CancelledError):\n"
            "        task.result()\n",
            {},
        ),
        (
            "async def stop(task):\n    with contextlib.suppress(TimeoutError):\n"
            "        await task\n",
            {},
        ),
        (
            "async def stop(task):\n    try:\n        await task\n"
            "    except asyncio.CancelledError:\n        log()\n        raise\n",
            {},
        ),
        (
            "async def outer():\n    with contextlib.suppress(asyncio.CancelledError):\n"
            "        async def inner():\n            await task\n",
            {},
        ),
        (
            "async def stop(task):\n    try:\n        await task\n"
            "    except asyncio.CancelledError:\n        logger.debug('stopped')\n"
            "        return\n",
            {"stop": 1},
        ),
        (
            "async def loop(task):\n    while True:\n        try:\n            await task\n"
            "        except BaseException:\n            continue\n",
            {"loop": 1},
        ),
        (
            "from asyncio import CancelledError as Stopped\n"
            "from contextlib import suppress as quietly\n"
            "async def stop(task):\n    with quietly(Stopped):\n        await task\n",
            {"stop": 1},
        ),
        (
            "import contextlib as c\n_STOP = (asyncio.CancelledError, TimeoutError)\n"
            "async def stop(task):\n    with c.suppress(*_STOP):\n        await task\n",
            {"stop": 1},
        ),
        (
            "async def stop(task):\n    with contextlib.ExitStack() as stack:\n"
            "        stack.enter_context(contextlib.suppress(asyncio.CancelledError))\n"
            "        await task\n",
            {"stop": 1},
        ),
        (
            "async def stop(task):\n    try:\n        await task\n"
            "    except asyncio.CancelledError:\n        return (None, False)\n",
            {"stop": 1},
        ),
        (
            "async def stop(task):\n    try:\n        await task\n"
            "    except asyncio.CancelledError:\n        self.stopped = True\n",
            {"stop": 1},
        ),
        (
            "async def stop(task):\n    try:\n        await task\n"
            "    except asyncio.CancelledError:\n        _log_stop(task)\n",
            {"stop": 1},
        ),
        (
            "async def stop(task):\n    try:\n        await task\n"
            "    except asyncio.CancelledError as error:\n"
            "        return Outcome(error=error)\n",
            {},
        ),
        (
            "class Owner:\n    _STOP: tuple = (asyncio.CancelledError,)\n\n"
            "    async def stop(self, task):\n"
            "        from asyncio import CancelledError as Gone\n"
            "        try:\n            await task\n        except self._STOP:\n"
            "            pass\n        try:\n            await task\n"
            "        except Gone:\n            pass\n",
            {"Owner.stop": 2},
        ),
    ],
    ids=[
        "suppress-cancelled",
        "suppress-base-exception",
        "except-pass",
        "bare-except-pass",
        "cancel-and-join",
        "no-await",
        "other-exception",
        "handler-reraises",
        "await-in-nested-function",
        "log-and-return",
        "continue",
        "aliased-names",
        "starred-module-tuple",
        "exit-stack",
        "constant-return",
        "flag-only",
        "bare-log-function",
        "capture-return-is-not-flagged",
        "class-constant-and-local-alias",
    ],
)
def test_the_ratchet_sees_each_swallowing_shape(source: str, expected: dict[str, int]) -> None:
    assert _swallowing_sites(ast.parse(source)) == collections.Counter(expected)
