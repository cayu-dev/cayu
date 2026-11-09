"""Keep caught environment cleanup failures from dropping their teardown successor.

A runner, binding or factory release can fail while its teardown continues in an
authenticated cleanup task carried by the failure. A handler around such a call
that does not re-raise the failure must pass it to
``cayu.environments._cleanup_handoff.keep_cleanup_owners`` so the task reaches
the owner that cleanup drains wait for. ``contextlib.suppress`` around such a
call cannot do that.

This is a heuristic over call names, not a proof: it sees ``try`` blocks and
``suppress`` around calls named like environment cleanup, not failures read
from ``task.exception()`` or gathered, and it trusts any call of the helper.
The allow-list records the handlers that remain today and may only shrink.
"""

from __future__ import annotations

import ast
import collections
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2] / "src" / "cayu"

_HELPER = "keep_cleanup_owners"
_CLEANUP_METHODS = frozenset(
    {"aclose", "close", "finalize", "release", "release_unbound", "teardown"}
)
_CLEANUP_RECEIVER = re.compile(r"runner|binding|environment|inner|factory", re.IGNORECASE)
# In the modules that own environment cleanup, any call or reference named like
# a cleanup step counts too (``self._close_pending()``, ``close_binding``, a
# bare ``close`` alias, ``_discard_resources``, ``backend.discard_spare``).
_CLEANUP_STEP = re.compile(
    r"^_?(a?close|finalize|release|teardown|discard|settle)(_\w+)?$|^create_spare$"
)
_CLEANUP_MODULES = re.compile(
    r"^(environments|egress|runners|sandbox_broker)/"
    r"|^runtime/_(environment|retained_environment|shared_environment|shared_allocation)"
)
# A cancellation can carry an owner too, so only these are exempt.
_CONTROL_TYPES = frozenset(
    {
        "GeneratorExit",
        "KeyboardInterrupt",
        "StopAsyncIteration",
        "StopIteration",
        "SystemExit",
    }
)

# (module, function qualname) -> count of handlers that may absorb a cleanup failure.
_ALLOWED: dict[tuple[str, str], int] = {
    # Concrete runners, processes and transports that never hand off their
    # teardown; brokered sandboxes are destroyed by the broker once their lease lapses.
    ("egress/aws_lambda_microvm_adapter.py", "LambdaMicroVMEgressAdapter._admit_runner"): 1,
    ("environments/aws_lambda_microvm.py", "_close_quietly"): 1,
    ("runners/_exit_cleanup.py", "close_runners_at_exit"): 1,
    ("runners/_subprocess.py", "_detach_process"): 1,
    ("runners/_subprocess.py", "_write_stdin"): 1,
    ("runners/aws_lambda_microvm.py", "LambdaMicroVMRunner._close_owned"): 1,
    ("runners/aws_lambda_microvm.py", "LambdaMicroVMRunner._close_transports"): 2,
    ("runners/aws_lambda_microvm.py", "LambdaMicroVMRunner.create.allocate"): 1,
    ("runners/aws_lambda_microvm.py", "LambdaMicroVMRunner.from_existing.attach"): 1,
    ("runners/e2b.py", "E2BRunner._exec"): 1,
    ("sandbox_broker/coding.py", "BrokeredSandboxCodingBinding.finalize"): 1,
    ("sandbox_broker/coding.py", "BrokeredSandboxCodingEnvironmentFactory._discard"): 1,
    # Not an environment teardown: egress sidecars, listeners, sockets, files
    # and authorities, snapshot pins, an archive check, the workspace dispatch
    # gate, a fenced invocation release, and a drain's own deadline.
    ("egress/_docker_reconnect.py", "DockerReconnect.claim"): 1,
    ("egress/_docker_reconnect.py", "_Claim.write"): 1,
    ("egress/_remote_adapter.py", "prepare_exposed_proxy_binding.cleanup"): 2,
    ("egress/broker.py", "TransparentEgressBroker._run_upstream_operation"): 1,
    ("egress/docker_adapter.py", "DockerEgressAdapter._prepare"): 1,
    ("egress/docker_adapter.py", "DockerEgressAdapter._teardown"): 2,
    ("egress/docker_adapter.py", "DockerEgressAdapter.cutover_authority"): 2,
    ("egress/e2b_adapter.py", "E2BEgressAdapter.cutover_authority"): 2,
    ("egress/proxy_server.py", "TransparentEgressProxyServer._accept_loop"): 2,
    ("egress/proxy_server.py", "TransparentEgressProxyServer._close_owned_resources"): 3,
    ("egress/proxy_server.py", "TransparentEgressProxyServer._handle_connect"): 1,
    ("egress/proxy_server.py", "TransparentEgressProxyServer._safe_handle"): 1,
    ("egress/proxy_server.py", "_shutdown"): 1,
    ("egress/runtime.py", "_EgressManagedRunner._finalize"): 1,
    ("environments/bindings.py", "_validate_sync_tar_stream"): 1,
    ("environments/snapshot_lifecycle.py", "ExecutionSnapshots.delete"): 1,
    ("runtime/_environment_lifecycle.py", "EnvironmentLifecycle.drain_retained_cleanups"): 1,
    ("runtime/_environment_lifecycle.py", "EnvironmentLifecycle.release_settled_invocation"): 1,
    # The failure is kept and raised, or its owners transferred, after the
    # step (egress phase failures and cancellations carry theirs onto what is
    # kept); or the owner that failed stays retained and is retried.
    ("egress/runtime.py", "_EgressManagedRunner._finalize_serialized"): 3,
    ("egress/runtime.py", "_EgressTeardownBinding._finalize_locked"): 2,
    ("egress/runtime.py", "_EgressTeardownBinding.release_unbound"): 3,
    (
        "egress/runtime.py",
        "_EgressManagedRunner.defer_finalization_settlement.settle.run_finalization_attempt",
    ): 1,
    ("environments/_sync_staging.py", "SyncBindingStagingCapacity._release_archive_reference"): 1,
    ("environments/deferred.py", "DeferredMaterialization._materialize_once"): 2,
    ("environments/docker_coding.py", "DockerCodingEnvironmentFactory._create"): 2,
    ("environments/docker_coding.py", "_release_immutable_input_attachments"): 1,
    ("runtime/_environment_lifecycle.py", "EnvironmentLifecycle.bind"): 1,
    ("runtime/_environment_lifecycle.py", "EnvironmentLifecycle._finalize_terminal_event_once"): 1,
    ("runtime/_environment_lifecycle.py", "EnvironmentLifecycle._abort_environment_setup_once"): 2,
    (
        "runtime/_environment_lifecycle.py",
        "EnvironmentLifecycle._abort_environment_setup_once.retry_binding_finalize",
    ): 1,
    ("runtime/_environment_lifecycle.py", "_release_unclaimed_factory_result"): 1,
    ("runtime/_environment_lifecycle.py", "_await_bounded_environment_factory_release"): 1,
    ("runtime/_retained_environment_end.py", "RetainedEnvironmentEndWatches._run"): 1,
    ("runtime/_shared_environment_allocations.py", "SharedEnvironmentAllocations.end_owner"): 1,
    (
        "runtime/_shared_environment_allocations.py",
        "SharedEnvironmentAllocations._release_recorded",
    ): 1,
    (
        "runtime/_shared_environment_allocations.py",
        "SharedEnvironmentAllocations._settle_in_background",
    ): 1,
}


def _type_names(node: ast.expr | None) -> list[str]:
    if node is None:
        return ["BaseException"]
    if isinstance(node, ast.Tuple):
        return [name for element in node.elts for name in _type_names(element)]
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return [node.attr]
    return ["?"]


def _own_nodes(nodes: list[ast.stmt], *, lambdas: bool = False) -> list[ast.AST]:
    found: list[ast.AST] = []
    pending: list[ast.AST] = list(nodes)
    while pending:
        node = pending.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) or (
            isinstance(node, ast.Lambda) and not lambdas
        ):
            continue
        found.append(node)
        pending.extend(ast.iter_child_nodes(node))
    return found


def _terminal_name(node: ast.expr) -> str:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Call):
        return _terminal_name(node.func)
    return ""


def _is_cleanup_reference(node: ast.expr, *, local_steps: bool) -> bool:
    if isinstance(node, ast.Name):
        return local_steps and _CLEANUP_STEP.match(node.id) is not None
    if not isinstance(node, ast.Attribute):
        return False
    receiver = _terminal_name(node.value)
    if local_steps and _CLEANUP_STEP.match(node.attr) is not None:
        return True
    return (node.attr in _CLEANUP_METHODS and _CLEANUP_RECEIVER.search(receiver) is not None) or (
        node.attr == "release" and receiver.endswith("result")
    )


def _calls_environment_cleanup(body: list[ast.stmt], module: str) -> bool:
    local_steps = _CLEANUP_MODULES.match(module) is not None
    return any(
        isinstance(node, ast.Call)
        and any(
            _is_cleanup_reference(reference, local_steps=local_steps)
            for reference in (node.func, *node.args)
        )
        for node in _own_nodes(body, lambdas=True)
    )


def _keeps_or_raises(handler: ast.ExceptHandler) -> bool:
    # Only a re-raise every path reaches counts; a conditional one leaves the
    # other paths absorbing the failure.
    statements = list(handler.body)
    while statements:
        statement = statements.pop()
        if isinstance(statement, ast.With | ast.AsyncWith):
            statements.extend(statement.body)
        elif isinstance(statement, ast.Raise) and (
            statement.exc is None
            or (isinstance(statement.exc, ast.Name) and statement.exc.id == handler.name)
        ):
            return True
    for node in _own_nodes(handler.body):
        if (
            isinstance(node, ast.Call)
            and _terminal_name(node.func) == _HELPER
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == handler.name
        ):
            return True
    return False


def _absorbing_handlers() -> collections.Counter[tuple[str, str]]:
    found: collections.Counter[tuple[str, str]] = collections.Counter()

    def visit(node: ast.AST, scope: list[str], module: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                visit(child, [*scope, child.name], module)
                continue
            if isinstance(child, ast.Try | ast.TryStar) and _calls_environment_cleanup(
                child.body, module
            ):
                for handler in child.handlers:
                    if set(_type_names(handler.type)) <= _CONTROL_TYPES:
                        continue
                    if not _keeps_or_raises(handler):
                        found[module, ".".join(scope)] += 1
            if isinstance(child, ast.With | ast.AsyncWith) and _calls_environment_cleanup(
                child.body, module
            ):
                for item in child.items:
                    context = item.context_expr
                    if (
                        isinstance(context, ast.Call)
                        and _terminal_name(context.func) == "suppress"
                        and not {name for arg in context.args for name in _type_names(arg)}
                        <= _CONTROL_TYPES
                    ):
                        found[module, ".".join(scope)] += 1
            visit(child, scope, module)

    for path in sorted(_ROOT.rglob("*.py")):
        visit(ast.parse(path.read_text()), [], path.relative_to(_ROOT).as_posix())
    return found


def test_no_new_handler_drops_a_cleanup_successor() -> None:
    found = _absorbing_handlers()
    grown = {key: count for key, count in found.items() if count > _ALLOWED.get(key, 0)}
    assert not grown, (
        f"Handlers absorb environment cleanup failures without keeping their owner: {grown}. "
        f"Pass the caught failure to {_HELPER} before recording it, or raise it."
    )
    shrunk = {key: count for key, count in _ALLOWED.items() if found.get(key, 0) < count}
    assert not shrunk, f"Handlers were removed; shrink the allow-list in this test: {shrunk}"
