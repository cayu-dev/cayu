"""Fail-closed workspace-mutation attribution.

The registry in this module is process-local evidence only.  Durable events
carry its conclusions, while a fresh process starts with no assumed knowledge
of earlier writers.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Literal, cast

from cayu._validation import canonical_durable_json_bytes, require_clean_nonblank
from cayu.workspaces import (
    Workspace,
    WorkspaceDirectMutationReconciliation,
    WorkspaceMutationAttribution,
    WorkspaceMutationAttributionConfidence,
    WorkspaceMutationResult,
    WorkspaceRevisionDelta,
    WorkspaceRevisionObservation,
    WorkspaceRevisionObservationStatus,
    WorkspaceWriterIsolationEvidence,
    WorkspaceWriterIsolationStatus,
)

_MAX_ACTIVE_WINDOWS = 4096
_MAX_RETAINED_RESOURCE_OBSERVATIONS = 64
_MAX_DIRECT_MUTATION_OPERATIONS = 64
_MAX_INLINE_DIRECT_MUTATION_OPERATIONS = 16
_MAX_DIRECT_MUTATION_PATH_BYTES = 4096

DirectWorkspaceMutationMethod = Literal[
    "write_bytes",
    "delete",
    "create_bytes",
    "replace_bytes",
    "delete_if_revision",
]


@dataclass
class _ActiveWindow:
    overlap_detected: bool = False


@dataclass(frozen=True)
class DirectWorkspaceMutationOperation:
    """Private, bounded evidence captured at the invocation workspace facade."""

    method: DirectWorkspaceMutationMethod
    path: str
    result: WorkspaceMutationResult | None
    result_valid: bool = True


class DirectWorkspaceMutationCollector:
    """Retain a bounded sequence of successful direct workspace operations."""

    def __init__(self, *, max_operations: int = _MAX_DIRECT_MUTATION_OPERATIONS) -> None:
        if type(max_operations) is not int or max_operations <= 0:
            raise ValueError("max_operations must be a positive integer.")
        self._max_operations = max_operations
        self._operations: list[DirectWorkspaceMutationOperation] = []
        self._total_operations = 0

    def record(self, method: str, path: str, result: object) -> None:
        self._total_operations += 1
        if type(method) is not str or method not in {
            "write_bytes",
            "delete",
            "create_bytes",
            "replace_bytes",
            "delete_if_revision",
        }:
            return
        if (
            type(path) is not str
            or not path.strip()
            or len(path.encode("utf-8", "surrogatepass")) > _MAX_DIRECT_MUTATION_PATH_BYTES
        ):
            return
        if len(self._operations) >= self._max_operations:
            return
        expects_result = method in {
            "create_bytes",
            "replace_bytes",
            "delete_if_revision",
        }
        normalized_method = cast("DirectWorkspaceMutationMethod", method)
        self._operations.append(
            DirectWorkspaceMutationOperation(
                method=normalized_method,
                path=path,
                result=result if type(result) is WorkspaceMutationResult else None,
                result_valid=not expects_result or type(result) is WorkspaceMutationResult,
            )
        )

    @property
    def operations(self) -> tuple[DirectWorkspaceMutationOperation, ...]:
        return tuple(self._operations)

    @property
    def total_operations(self) -> int:
        return self._total_operations

    @property
    def truncated(self) -> bool:
        return self._total_operations > len(self._operations)


class WorkspaceMutationWindow:
    """One process-observed writer window for a private workspace resource."""

    def __init__(
        self,
        *,
        window_id: str,
        resource_key: tuple[object, ...] | None,
        registry_token: object | None,
        state: _ActiveWindow | None,
        prior_observation: WorkspaceRevisionObservation | None,
    ) -> None:
        self.window_id = require_clean_nonblank(window_id, "window_id")
        self._resource_key = resource_key
        self._registry_token = registry_token
        self._state = state
        self._prior_observation = prior_observation
        self._closed = False

    @property
    def resource_identity_available(self) -> bool:
        return self._resource_key is not None and self._state is not None

    @property
    def overlap_detected(self) -> bool:
        return self._state is not None and self._state.overlap_detected

    @property
    def prior_observation(self) -> WorkspaceRevisionObservation | None:
        return self._prior_observation

    def close(
        self,
        observation: WorkspaceRevisionObservation | None = None,
        *,
        discard_history: bool = False,
    ) -> None:
        global _ACTIVE_WINDOW_COUNT

        if self._closed:
            return
        self._closed = True
        if self._resource_key is None or self._registry_token is None or self._state is None:
            return
        with _REGISTRY_LOCK:
            active = _ACTIVE_WINDOWS.get(self._resource_key)
            if active is not None:
                removed = active.pop(self._registry_token, None)
                if removed is not None:
                    _ACTIVE_WINDOW_COUNT -= 1
                if not active:
                    _ACTIVE_WINDOWS.pop(self._resource_key, None)
            if discard_history:
                _LAST_OBSERVATIONS.pop(self._resource_key, None)
            elif observation is not None:
                _LAST_OBSERVATIONS.pop(self._resource_key, None)
                if len(_LAST_OBSERVATIONS) >= _MAX_RETAINED_RESOURCE_OBSERVATIONS:
                    oldest_resource_key = next(iter(_LAST_OBSERVATIONS))
                    _LAST_OBSERVATIONS.pop(oldest_resource_key, None)
                _LAST_OBSERVATIONS[self._resource_key] = observation


_REGISTRY_LOCK = threading.Lock()
_ACTIVE_WINDOWS: dict[tuple[object, ...], dict[object, _ActiveWindow]] = {}
_ACTIVE_WINDOW_COUNT = 0
_LAST_OBSERVATIONS: dict[tuple[object, ...], WorkspaceRevisionObservation] = {}


def begin_workspace_mutation_window(
    workspace: Workspace,
    *,
    window_id: str,
) -> WorkspaceMutationWindow:
    """Open one best-effort process-local overlap window without publishing its key."""

    global _ACTIVE_WINDOW_COUNT

    resource_key = _safe_workspace_resource_key(workspace)
    if resource_key is None:
        return WorkspaceMutationWindow(
            window_id=window_id,
            resource_key=None,
            registry_token=None,
            state=None,
            prior_observation=None,
        )
    with _REGISTRY_LOCK:
        active = _ACTIVE_WINDOWS.get(resource_key)
        if active is None:
            active = {}
            _ACTIVE_WINDOWS[resource_key] = active
        if _ACTIVE_WINDOW_COUNT >= _MAX_ACTIVE_WINDOWS:
            for current in active.values():
                current.overlap_detected = True
            if not active:
                _ACTIVE_WINDOWS.pop(resource_key, None)
            return WorkspaceMutationWindow(
                window_id=window_id,
                resource_key=None,
                registry_token=None,
                state=None,
                prior_observation=None,
            )
        registry_token = object()
        state = _ActiveWindow(overlap_detected=bool(active))
        if active:
            for current in active.values():
                current.overlap_detected = True
        active[registry_token] = state
        _ACTIVE_WINDOW_COUNT += 1
        prior = _LAST_OBSERVATIONS.get(resource_key)
    return WorkspaceMutationWindow(
        window_id=window_id,
        resource_key=resource_key,
        registry_token=registry_token,
        state=state,
        prior_observation=prior,
    )


def _safe_workspace_resource_key(workspace: Workspace) -> tuple[object, ...] | None:
    try:
        resource_key = workspace.resource_key
        if type(resource_key) is not tuple or not resource_key:
            return None
        hash(resource_key)
    except Exception:
        return None
    return resource_key


def reconcile_direct_workspace_mutations(
    *,
    before: WorkspaceRevisionObservation,
    after: WorkspaceRevisionObservation,
    collector: DirectWorkspaceMutationCollector,
) -> WorkspaceDirectMutationReconciliation:
    """Reconcile direct operation receipts with independently observed endpoints."""

    operations = collector.operations
    if not operations:
        return WorkspaceDirectMutationReconciliation.NOT_OBSERVED
    if collector.truncated:
        return WorkspaceDirectMutationReconciliation.TRUNCATED
    if (
        before.status is not WorkspaceRevisionObservationStatus.SUPPORTED
        or after.status is not WorkspaceRevisionObservationStatus.SUPPORTED
        or before.path_scope != "complete"
        or after.path_scope != "complete"
    ):
        return WorkspaceDirectMutationReconciliation.INCOMPLETE
    if any(not operation.result_valid for operation in operations):
        return WorkspaceDirectMutationReconciliation.CONTRADICTORY

    before_paths = {entry.path: entry for entry in before.paths}
    after_paths = {entry.path: entry for entry in after.paths}
    by_path: dict[str, list[DirectWorkspaceMutationOperation]] = {}
    for operation in operations:
        by_path.setdefault(operation.path, []).append(operation)

    incomplete = False
    for path, path_operations in by_path.items():
        exact = [operation for operation in path_operations if operation.result is not None]
        if not exact:
            incomplete = True
            continue
        for operation in exact:
            result = operation.result
            if result is None:  # pragma: no cover - narrowed above
                raise AssertionError("Direct mutation reconciliation lost exact evidence.")
            if result.operation != _expected_result_operation(operation.method):
                return WorkspaceDirectMutationReconciliation.CONTRADICTORY
        for previous, current in pairwise(path_operations):
            previous_result = previous.result
            current_result = current.result
            if previous_result is None or current_result is None:
                incomplete = True
                continue
            if (
                previous_result.after_revision != current_result.before_revision
                or previous_result.after_sha256 != current_result.before_sha256
                or previous_result.after_bytes != current_result.before_bytes
            ):
                return WorkspaceDirectMutationReconciliation.CONTRADICTORY
        first = exact[0].result
        last = exact[-1].result
        if first is None or last is None:  # pragma: no cover - narrowed above
            raise AssertionError("Direct mutation reconciliation lost exact evidence.")
        observed_before = before_paths.get(path)
        if first.operation == "create":
            if observed_before is not None:
                return WorkspaceDirectMutationReconciliation.CONTRADICTORY
        elif observed_before is None or observed_before.content_sha256 != first.before_sha256:
            return WorkspaceDirectMutationReconciliation.CONTRADICTORY

        observed_after = after_paths.get(path)
        if last.operation == "delete":
            if observed_after is not None:
                return WorkspaceDirectMutationReconciliation.CONTRADICTORY
        elif observed_after is None or observed_after.content_sha256 != last.after_sha256:
            return WorkspaceDirectMutationReconciliation.CONTRADICTORY

        if len(exact) != len(path_operations):
            incomplete = True
    return (
        WorkspaceDirectMutationReconciliation.INCOMPLETE
        if incomplete
        else WorkspaceDirectMutationReconciliation.CONSISTENT
    )


def _expected_result_operation(method: str) -> str:
    if method == "create_bytes":
        return "create"
    if method == "replace_bytes":
        return "replace"
    if method == "delete_if_revision":
        return "delete"
    raise ValueError("A non-conditional mutation cannot carry exact result evidence.")


def classify_workspace_mutation_attribution(
    *,
    window: WorkspaceMutationWindow,
    isolation_before: WorkspaceWriterIsolationEvidence,
    isolation_after: WorkspaceWriterIsolationEvidence,
    direct_reconciliation: WorkspaceDirectMutationReconciliation,
) -> WorkspaceMutationAttribution:
    """Classify a window without upgrading absent evidence into causality."""

    exclusive = (
        window.resource_identity_available
        and isolation_before.status is WorkspaceWriterIsolationStatus.EXCLUSIVE
        and isolation_after.status is WorkspaceWriterIsolationStatus.EXCLUSIVE
        and isolation_before.mechanism == isolation_after.mechanism
        and isolation_before.generation == isolation_after.generation
    )
    isolation = (
        WorkspaceWriterIsolationStatus.EXCLUSIVE
        if exclusive
        else (
            WorkspaceWriterIsolationStatus.SHARED
            if WorkspaceWriterIsolationStatus.SHARED
            in {isolation_before.status, isolation_after.status}
            else WorkspaceWriterIsolationStatus.UNKNOWN
        )
    )
    if window.overlap_detected:
        confidence = WorkspaceMutationAttributionConfidence.CONCURRENT_AMBIGUITY
        detail_code = "overlapping_workspace_mutation_windows"
    elif direct_reconciliation is WorkspaceDirectMutationReconciliation.CONTRADICTORY:
        confidence = WorkspaceMutationAttributionConfidence.CONCURRENT_AMBIGUITY
        detail_code = "direct_and_observed_workspace_evidence_conflict"
    elif exclusive:
        confidence = WorkspaceMutationAttributionConfidence.EXCLUSIVE_TOOL
        detail_code = "exclusive_writer_isolation_verified"
    else:
        confidence = WorkspaceMutationAttributionConfidence.EXTERNAL_OR_UNKNOWN
        detail_code = (
            "workspace_resource_identity_unavailable"
            if not window.resource_identity_available
            else "exclusive_writer_isolation_unproven"
        )
    return WorkspaceMutationAttribution(
        confidence=confidence,
        writer_isolation=isolation,
        overlap_detected=window.overlap_detected,
        direct_reconciliation=direct_reconciliation,
        detail_code=detail_code,
    )


def direct_workspace_mutation_payload(
    collector: DirectWorkspaceMutationCollector,
    *,
    window_id: str,
    evidence_available: bool,
) -> dict[str, Any]:
    """Return a content-free, bounded durable projection of direct operations."""

    if not evidence_available:
        return {
            "operations": [],
            "retained_operations": 0,
            "total_operations": 0,
            "truncated": True,
        }
    operations: list[dict[str, Any]] = []
    retained = collector.operations[:_MAX_INLINE_DIRECT_MUTATION_OPERATIONS]
    for sequence, operation in enumerate(retained):
        item: dict[str, Any] = {
            "sequence": sequence,
            "method": operation.method,
            "path_sha256": hashlib.sha256(
                f"{window_id}\0{operation.path}".encode("utf-8", "surrogatepass")
            ).hexdigest(),
            "result_valid": operation.result_valid,
            "result_operation": "unavailable",
            "result_evidence_sha256": "unavailable",
        }
        if operation.result is not None:
            result_payload = {
                "operation": operation.result.operation,
                "before_revision": operation.result.before_revision,
                "after_revision": operation.result.after_revision,
                "before_sha256": operation.result.before_sha256,
                "after_sha256": operation.result.after_sha256,
                "before_bytes": operation.result.before_bytes,
                "after_bytes": operation.result.after_bytes,
            }
            item["result_operation"] = operation.result.operation
            item["result_evidence_sha256"] = hashlib.sha256(
                canonical_durable_json_bytes(result_payload, "workspace_mutation_result")
            ).hexdigest()
        operations.append(item)
    return {
        "operations": operations,
        "retained_operations": len(operations),
        "total_operations": collector.total_operations,
        "truncated": collector.truncated or len(retained) < len(collector.operations),
    }


def observed_pre_window_change(
    window: WorkspaceMutationWindow,
    before: WorkspaceRevisionObservation,
) -> WorkspaceRevisionDelta | None:
    """Compare the last process-observed endpoint with this window's start."""

    prior = window.prior_observation
    if prior is None or prior.identity != before.identity:
        return None
    from cayu.workspaces import compare_workspace_revisions

    return compare_workspace_revisions(prior, before)
