"""Claimed task mutations inside ``src/cayu`` present the claim generation.

Fencing a claimed mutation by its exact lease expiry alone is deprecated: every
renewal moves the expiry, so a write that carries it races the claim's own
heartbeats, and a value read back from the row can name a newer
claim. This ratchet fails on any new call inside ``src/cayu`` that
presents ``lease_expires_at`` without ``claim_generation`` (a generation spelled
``None`` counts as none), on a cancellation request built from the row without
the drain's generation, and on a ``model_copy`` in the task-claim packages that
sets a lease without a generation. A generation passed through a variable is
taken on trust. The allow-list holds the remaining sites and may only shrink: a
listed site that no longer offends fails too, until it is removed from the list.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SOURCE = Path(__file__).resolve().parents[2] / "src" / "cayu"

# Store mutations that take the lease as a keyword.
_KEYWORD_LEASE_METHODS = frozenset(
    {
        "attach_task",
        "complete_task",
        "enforce_task_retry_deadline",
        "fail_task",
        "heartbeat",
        "hold_claimed_work_contract_task",
        "release_attached_task_worker",
        "release_task",
        "restart_task",
        "task_retry_deadline_elapsed",
    }
)
# Store mutations whose third positional argument is the lease.
_POSITIONAL_LEASE_METHODS = frozenset(
    {"mark_claimed_task_execution_started", "request_claimed_task_cancellation"}
)
_REQUEST_MODELS = frozenset(
    {
        "TaskInterruptedHandoffRequest",
        "TaskRetrySettlementRequest",
        "TaskTerminalizationRequest",
    }
)
# Request builders that fall back to the row's lease unless given the generation.
_GENERATION_BUILDERS = frozenset({"_task_cancellation_terminalization_request"})
# Where a ``model_copy`` that sets a lease builds a task claim's write. The stores
# (which own the row), work-attempt admission (fenced by its own claim generation)
# and other claim kinds are outside it.
_LEASE_COPY_PACKAGES = ("cayu/knowledge/", "cayu/runtime/", "cayu/tasks/", "cayu/verification/")
_LEASE_COPY_EXCLUDED = frozenset({"cayu/tasks/admission.py", "cayu/tasks/memory.py"})

# "path::function::callee" of each remaining lease-only site, with why it stays.
_ALLOWED: frozenset[str] = frozenset(
    {
        # The request digest is stored in the durable runtime-failure decision, which
        # the worker recomputes; changing its material changes the decision format.
        # Called without a lease (attached, lease-less path) or for replay matching.
        "cayu/runtime/_continuation_task_failure.py::runtime_task_failure_receipt_matches"
        "::TaskTerminalizationRequest",
        "cayu/runtime/_continuation_task_failure.py::runtime_task_failure_terminalization_request"
        "::TaskTerminalizationRequest",
        # Expired-owner recovery of a committed runtime failure: same decision digest.
        "cayu/runtime/_session_task_lifecycle.py::SessionTaskLifecycle"
        ".recover_committed_runtime_task_failure::TaskTerminalizationRequest",
        # The public attachment path for a caller outside a managed handler; inside
        # one, the branch above it presents the handler's generation.
        "cayu/runtime/_session_task_lifecycle.py::SessionTaskLifecycle.start_task::attach_task",
        # Validates an owner-lost cancellation against the exact expired lease, which
        # reclaim never requeues for a cancellation-fenced claim.
        "cayu/tasks/cancellation.py::TaskCancellationReconciliationResult.validate_result"
        "::TaskTerminalizationRequest",
        # A store-side receipt, not a claim write: the runtime retry settlement and
        # its reconciliation rebuild one request identity from the row, with a
        # synthetic worker and no generation. Its idempotency key names the
        # attempt; changing its material changes the digest reconciliation compares.
        "cayu/tasks/retry.py::_task_retry_runtime_terminal_request::TaskRetrySettlementRequest",
        # The retry-deadline receipt is derived from the lease the worker still holds
        # under its lock; the store fences the write by the generation it presents.
        "cayu/tasks/worker.py::_enforce_task_retry_deadline_with_retry::model_copy",
        # Owner-lost cancellation settlement and validation read only the cancellation's
        # idempotency key from the row; the claim is identified by its exact expired
        # lease, which reclaim never requeues for a cancellation-fenced claim.
        "cayu/tasks/_cancellation.py::_validated_task_cancellation"
        "::_task_cancellation_terminalization_request",
        "cayu/tasks/dispatch.py::TaskStoreDispatcher._settle_dispatch_cancellation_after_quiescence"
        "::_task_cancellation_terminalization_request",
        "cayu/tasks/restart.py::_abandoned_restarted_task"
        "::_task_cancellation_terminalization_request",
        "cayu/tasks/worker.py::_reconcile_observed_abandoned_task_cancellation"
        "::_task_cancellation_terminalization_request",
        "cayu/tasks/worker.py::_settle_expired_owner_cancellation"
        "::_task_cancellation_terminalization_request",
        "cayu/tasks/worker.py::_settle_ordinary_task_cancellation_after_quiescence"
        "::_task_cancellation_terminalization_request",
        # Still fenced by the lease alone, and open: the verified worker serializes
        # renewal and every write under its owner lock, but a lost renewal
        # acknowledgement still fences its own holder, which the generation removes
        # for the main worker. Its claims already carry claim_generation.
        "cayu/verification/verified_task_worker.py::VerifiedTaskWorker._prepare_and_run"
        "::mark_claimed_task_execution_started",
        "cayu/verification/verified_task_worker.py::_LeaseOwner.heartbeat::heartbeat",
    }
)


def _callee(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return None


def _is_none(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _lease_copy(call: ast.Call) -> bool:
    """``x.model_copy(update={"lease_expires_at": lease, ...})`` without a generation."""

    for keyword in call.keywords:
        if keyword.arg == "update" and isinstance(keyword.value, ast.Dict):
            fields = {
                key.value: value
                for key, value in zip(keyword.value.keys, keyword.value.values, strict=True)
                if isinstance(key, ast.Constant)
            }
            lease = fields.get("lease_expires_at")
            return lease is not None and not _is_none(lease) and "claim_generation" not in fields
    return False


def _offends(call: ast.Call, relative: str) -> bool:
    name = _callee(call)
    if name == "model_copy":
        return (
            relative.startswith(_LEASE_COPY_PACKAGES)
            and relative not in _LEASE_COPY_EXCLUDED
            and _lease_copy(call)
        )
    if None in {keyword.arg for keyword in call.keywords}:
        return False
    # A generation spelled ``None`` is no generation.
    keywords = {
        keyword.arg
        for keyword in call.keywords
        if not (keyword.arg == "claim_generation" and _is_none(keyword.value))
    }
    if name in _GENERATION_BUILDERS:
        return "claim_generation" not in keywords
    if "claim_generation" in keywords:
        return False
    if name in _KEYWORD_LEASE_METHODS or name in _REQUEST_MODELS:
        return "lease_expires_at" in keywords
    if name in _POSITIONAL_LEASE_METHODS:
        return "lease_expires_at" in keywords or len(call.args) >= 3
    return False


def _collect_sites(node: ast.AST, scope: str, relative: str, sites: set[str]) -> None:
    for child in ast.iter_child_nodes(node):
        child_scope = scope
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            child_scope = f"{scope}.{child.name}" if scope else child.name
        if isinstance(child, ast.Call) and _offends(child, relative):
            sites.add(f"{relative}::{scope or '<module>'}::{_callee(child)}")
        _collect_sites(child, child_scope, relative, sites)


def _lease_only_sites() -> set[str]:
    sites: set[str] = set()
    for path in sorted(_SOURCE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        _collect_sites(tree, "", path.relative_to(_SOURCE.parent).as_posix(), sites)
    return sites


def test_claimed_mutations_present_the_claim_generation() -> None:
    sites = _lease_only_sites()
    new = sorted(sites - _ALLOWED)
    assert not new, (
        "Present claim_generation= (the claim's durable generation) instead of only "
        f"lease_expires_at= at: {new}"
    )
    fixed = sorted(_ALLOWED - sites)
    assert not fixed, f"Remove these sites from the ratchet's allow-list: {fixed}"


def test_ratchet_sees_lease_only_spellings() -> None:
    def offends(source: str, relative: str = "cayu/tasks/worker.py") -> bool:
        call = ast.parse(source, mode="eval").body
        assert isinstance(call, ast.Call)
        return _offends(call, relative)

    assert offends("store.heartbeat(task_id, worker, lease_expires_at=lease)")
    assert offends(
        "store.heartbeat(task_id, worker, lease_expires_at=lease, claim_generation=None)"
    )
    assert not offends("store.heartbeat(task_id, worker, claim_generation=generation)")
    assert offends("_task_cancellation_terminalization_request(task, worker_id=worker)")
    assert offends(
        "_task_cancellation_terminalization_request(task, worker_id=worker, claim_generation=None)"
    )
    assert not offends(
        "_task_cancellation_terminalization_request(task, worker_id=w, claim_generation=g)"
    )
    assert offends('task.model_copy(update={"lease_expires_at": lease})')
    assert not offends('task.model_copy(update={"lease_expires_at": None})')
    assert not offends('task.model_copy(update={"lease_expires_at": l, "claim_generation": g})')
    assert not offends('task.model_copy(update={"lease_expires_at": lease})', "cayu/storage/x.py")
