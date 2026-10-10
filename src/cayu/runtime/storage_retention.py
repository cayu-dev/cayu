"""Application-level storage retention: one policy over every configured store.

:func:`apply_storage_retention` composes the per-store parts. It collects the
references each configured store holds, so a session that an eval in another
database names, a session that a task in another database still runs, or an id
that an agent snapshot store pins is kept even though the store being pruned
cannot see that reference. Targets run in dependency order: workspaces (which
need the session's durable allocation records), evals (which free the sessions
they name), sessions, then artifacts (which belong to sessions).

:func:`apply_workspace_retention` cleans leftover runner and coding workspaces
through the runtime's incomplete-session recovery, so every disposal goes
through the environment factory's own reaping fence. The optional
:func:`run_storage_retention_worker` applies a policy on an interval. Nothing
here starts on import.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Collection, Iterable, Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, TypeVar

from cayu.storage.retention import (
    MAX_RETENTION_ITEMS_PER_RUN,
    RetentionAuditEntry,
    RetentionAuditSink,
    RetentionDisposition,
    RetentionItem,
    RetentionPhase,
    RetentionPolicy,
    RetentionProgress,
    RetentionProgressCallback,
    RetentionProtection,
    RetentionReport,
    StorageRetentionPolicy,
    StorageRetentionReport,
    StorageRetentionTarget,
    WorkspaceRetentionPolicy,
    bounded_retention_detail,
    retention_audit_summary,
)

if TYPE_CHECKING:
    from cayu.applications import CayuApp

logger = logging.getLogger(__name__)
_PolicyT = TypeVar("_PolicyT", bound=RetentionPolicy)

WORKSPACE_STORE_KIND = "workspaces"
_LIVE_TASK_STATUSES = (
    "pending",
    "waiting_dependencies",
    "waiting_group",
    "claimed",
    "running",
    "paused",
    "blocked",
    "needs_attention",
)
_SESSION_PAGE = 500
_WORKSPACE_ALLOCATION_KEYS = (
    "environment_factory_allocation_intents",
    "environment_factory_pending_disposals",
)


# -- cross-store references ---------------------------------------------------------


async def live_task_session_ids(task_store: Any) -> frozenset[str]:
    """Session ids that non-terminal tasks in ``task_store`` reference."""

    from cayu.tasks.queries import TaskQuery
    from cayu.tasks.records import TaskStatus

    found: set[str] = set()
    for status in _LIVE_TASK_STATUSES:
        offset = 0
        while True:
            page = await task_store.list_tasks(
                TaskQuery(status=TaskStatus(status), limit=1000, offset=offset)
            )
            found.update(task.session_id for task in page if task.session_id is not None)
            if len(page) < 1000:
                break
            offset += len(page)
    return frozenset(found)


async def _held_identifiers(snapshot_stores: Iterable[Any]) -> frozenset[str]:
    held: set[str] = set()
    for store in snapshot_stores:
        held |= await store.retention_held_identifiers()
    return frozenset(held)


def _merge(
    *pairs: tuple[RetentionProtection, Collection[str]],
) -> dict[RetentionProtection, frozenset[str]]:
    merged: dict[RetentionProtection, set[str]] = {}
    for protection, identifiers in pairs:
        if identifiers:
            merged.setdefault(protection, set()).update(identifiers)
    return {protection: frozenset(identifiers) for protection, identifiers in merged.items()}


# -- workspaces ---------------------------------------------------------------------


def _workspace_item(
    session: Any,
    disposition: RetentionDisposition,
    *,
    counts: Mapping[str, int] | None = None,
    protections: Collection[RetentionProtection] = (),
    detail: str | None = None,
) -> RetentionItem:
    return RetentionItem(
        item_id=session.id,
        status=str(session.status),
        last_updated_at=session.updated_at,
        disposition=disposition,
        protections=tuple(sorted(set(protections), key=lambda value: value.value)),
        detail=bounded_retention_detail(detail),
        counts=dict(counts or {}),
    )


async def _workspace_leftovers(app: CayuApp, session: Any) -> tuple[dict[str, int], str | None]:
    """Count unsettled environment work a terminal session still owns."""

    from cayu.sessions._completion_finalization import (
        pending_completion_finalization_from_checkpoint,
    )

    checkpoint = await app.session_store.load_checkpoint(session.id) or {}
    if not any(key in checkpoint for key in _WORKSPACE_ALLOCATION_KEYS) and (
        pending_completion_finalization_from_checkpoint(checkpoint) is None
    ):
        return {}, None
    try:
        pending = await app._environment_lifecycle.pending_allocation_names(session)
    except ValueError as invalid:
        return {}, str(invalid)
    disposals = checkpoint.get("environment_factory_pending_disposals") or {}
    counts = {
        "pending_allocations": len(pending),
        "pending_disposals": len(disposals) if isinstance(disposals, dict) else 1,
        "pending_completion_finalizations": int(
            pending_completion_finalization_from_checkpoint(checkpoint) is not None
        ),
    }
    return ({} if not any(counts.values()) else counts), None


def _workspace_checkpoint_busy(checkpoint: Mapping[str, Any]) -> bool:
    receipts = checkpoint.get("workspace_checkpoints")
    if not isinstance(receipts, dict):
        return False
    return any(
        isinstance(receipt, dict) and receipt.get("phase") in {"mutating", "checkpointing"}
        for receipt in receipts.values()
    )


def _durable_branch_state(app: CayuApp, environment_name: str | None) -> bool | None:
    """Whether the environment's workspace keeps durable branch state (None: unknown)."""

    from cayu.workspaces.branches import WorkspaceBranchRetentionStrength

    if environment_name is None:
        return False
    registered = next(
        (
            registration
            for registration in app.list_environment_registrations()
            if registration.spec.name == environment_name
        ),
        None,
    )
    if registered is None:
        return None
    workspaces = []
    environment = getattr(registered, "environment", None)
    if environment is not None and getattr(environment, "workspace", None) is not None:
        workspaces.append(environment.workspace)
    source = getattr(getattr(registered, "factory", None), "source_workspace", None)
    if source is not None:
        workspaces.append(source)
    return any(
        workspace.branch_capabilities().retention is WorkspaceBranchRetentionStrength.DURABLE
        for workspace in workspaces
    )


async def apply_workspace_retention(
    app: CayuApp,
    policy: WorkspaceRetentionPolicy,
    *,
    audit: RetentionAuditSink | None = None,
    protected_session_ids: Collection[str] = (),
    references: Mapping[RetentionProtection, Collection[str]] | None = None,
    progress: RetentionProgressCallback | None = None,
) -> RetentionReport:
    """Dispose leftover workspaces of old terminal sessions; an operator-only action.

    A session is selected when it is terminal, last updated before the cutoff,
    and still owns unsettled environment allocations, disposals or completion
    finalization. It is kept while the session store reports a protection for
    it (live task, execution lease, pending action, snapshot pin, closure),
    while a workspace checkpoint is mutating or checkpointing, while its
    environment's workspace keeps durable branch state, or while the caller or
    another store names it. Disposal runs the runtime's incomplete-session
    recovery for that session, which reaps through the environment factory.
    """

    from cayu.sessions.queries import SessionOrder, SessionQuery
    from cayu.sessions.recovery import (
        IncompleteSessionRecoveryAction,
        IncompleteSessionRecoveryRequest,
    )
    from cayu.storage._session_retention import external_protections

    if type(policy) is not WorkspaceRetentionPolicy:
        raise TypeError("policy must be a WorkspaceRetentionPolicy.")
    if not policy.dry_run and audit is None:
        raise ValueError("A workspace retention apply requires a durable audit sink.")
    external = external_protections(protected_session_ids, references)
    store = app.session_store
    # Session timestamps follow the application's clock.
    started_at = app._clock()
    cutoff = started_at - policy.older_than

    candidates: list[tuple[Any, dict[str, int]]] = []
    protected: list[RetentionItem] = []
    deferred = 0
    for status in sorted(policy.statuses, key=lambda value: value.value):
        offset = 0
        while True:
            page = await store.list_sessions(
                SessionQuery(
                    status=status,
                    order_by=SessionOrder.UPDATED_AT_ASC,
                    limit=_SESSION_PAGE,
                    offset=offset,
                )
            )
            for session in page.sessions:
                if session.updated_at > cutoff:
                    continue
                counts, invalid = await _workspace_leftovers(app, session)
                if invalid is not None:
                    protected.append(
                        _workspace_item(
                            session,
                            RetentionDisposition.PROTECTED,
                            protections=(RetentionProtection.ERASURE_GUARD,),
                            detail=invalid,
                        )
                    )
                elif counts:
                    candidates.append((session, counts))
            if len(page.sessions) < _SESSION_PAGE:
                break
            offset += len(page.sessions)
    candidates.sort(key=lambda pair: (pair[0].updated_at, pair[0].id))

    async def protections_for(session: Any) -> tuple[set[RetentionProtection], str | None]:
        reasons = set(external.get(session.id, ()))
        inspected = await store.inspect_session_retention([session.id], include_store_guards=False)
        reasons.update(inspected.get(session.id, ()))
        reasons.discard(RetentionProtection.SESSION_EXPORT)
        detail = None
        checkpoint = await store.load_checkpoint(session.id) or {}
        if _workspace_checkpoint_busy(checkpoint):
            reasons.add(RetentionProtection.WORKSPACE_CHECKPOINT)
        durable = _durable_branch_state(app, session.environment_name)
        if durable is None:
            reasons.add(RetentionProtection.ERASURE_GUARD)
            detail = "the session's environment is not registered in this application"
        elif durable:
            reasons.add(RetentionProtection.BRANCH_RETENTION)
        return reasons, detail

    selected: list[tuple[Any, dict[str, int]]] = []
    for index, (session, counts) in enumerate(candidates):
        if len(selected) >= policy.max_items:
            deferred = len(candidates) - index
            break
        reasons, detail = await protections_for(session)
        if reasons:
            protected.append(
                _workspace_item(
                    session,
                    RetentionDisposition.PROTECTED,
                    protections=reasons,
                    detail=detail,
                )
            )
            continue
        selected.append((session, counts))
    policy_document = policy.policy_document()
    if progress is not None:
        await progress(
            RetentionProgress(
                store_kind=WORKSPACE_STORE_KIND,
                phase=RetentionPhase.PLANNED,
                dry_run=policy.dry_run,
                planned_items=len(selected),
                protected=tuple(protected[:MAX_RETENTION_ITEMS_PER_RUN]),
            )
        )
    if policy.dry_run:
        return _workspace_report(
            policy,
            started_at,
            cutoff,
            None,
            [
                _workspace_item(session, RetentionDisposition.SELECTED, counts=counts)
                for session, counts in selected
            ],
            protected,
            deferred,
        )
    assert audit is not None
    audit_id = await audit.begin_retention_audit(
        store_kind=WORKSPACE_STORE_KIND,
        mode=policy.mode,
        policy=policy_document,
        started_at=started_at,
    )
    applied: list[RetentionItem] = []
    for index, (planned, _counts) in enumerate(selected):
        batch_applied: tuple[RetentionItem, ...] = ()
        batch_protected: tuple[RetentionItem, ...] = ()
        session = await store.load(planned.id)
        reasons: set[RetentionProtection] = set()
        detail: str | None = None
        if (
            session is None
            or session.status not in policy.statuses
            or session.updated_at != planned.updated_at
        ):
            reasons, detail = {RetentionProtection.CHANGED}, "session changed after planning"
        else:
            reasons, detail = await protections_for(session)
        counts: dict[str, int] = {}
        if not reasons:
            assert session is not None
            counts, invalid = await _workspace_leftovers(app, session)
            if invalid is not None:
                reasons, detail = {RetentionProtection.ERASURE_GUARD}, invalid
        if not reasons and counts:
            try:
                result = await app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(
                        session_id=planned.id,
                        reason="storage_retention_workspace_cleanup",
                    )
                )
            except (RuntimeError, ValueError, KeyError) as refused:
                reasons, detail = {RetentionProtection.ERASURE_GUARD}, str(refused)
            else:
                counts = {
                    **counts,
                    "allocations_reaped": int(
                        IncompleteSessionRecoveryAction.REAPED_ALLOCATION in result.actions
                    ),
                    "cleanup_pending": int(
                        IncompleteSessionRecoveryAction.PENDING_ALLOCATION_CLEANUP in result.actions
                    ),
                }
        if reasons:
            batch_protected = (
                _workspace_item(
                    planned,
                    RetentionDisposition.PROTECTED,
                    protections=reasons,
                    detail=detail,
                ),
            )
        elif counts:
            item = _workspace_item(planned, RetentionDisposition.APPLIED, counts=counts)
            await audit.record_retention_entry(
                audit_id,
                RetentionAuditEntry(
                    item_id=planned.id,
                    action=policy.mode,
                    counts=item.counts,
                    bytes=0,
                    recorded_at=datetime.now(UTC),
                ),
            )
            batch_applied = (item,)
        applied.extend(batch_applied)
        protected.extend(batch_protected)
        if progress is not None:
            await progress(
                RetentionProgress(
                    store_kind=WORKSPACE_STORE_KIND,
                    phase=RetentionPhase.BATCH,
                    dry_run=False,
                    audit_id=audit_id,
                    batch_index=index,
                    planned_items=len(selected),
                    applied=batch_applied,
                    protected=batch_protected,
                )
            )
        await asyncio.sleep(0)
    report = _workspace_report(policy, started_at, cutoff, audit_id, applied, protected, deferred)
    await audit.complete_retention_audit(
        audit_id, completed_at=report.completed_at, summary=retention_audit_summary(report)
    )
    return report


def _workspace_report(
    policy: WorkspaceRetentionPolicy,
    started_at: datetime,
    cutoff: datetime,
    audit_id: str | None,
    items: list[RetentionItem],
    protected: list[RetentionItem],
    deferred: int,
) -> RetentionReport:
    return RetentionReport(
        store_kind=WORKSPACE_STORE_KIND,
        mode=policy.mode,
        dry_run=policy.dry_run,
        policy=policy.policy_document(),
        started_at=started_at,
        completed_at=datetime.now(UTC),
        cutoff=cutoff,
        audit_id=audit_id,
        items=tuple(items),
        protected=tuple(protected[:MAX_RETENTION_ITEMS_PER_RUN]),
        protected_truncated=len(protected) > MAX_RETENTION_ITEMS_PER_RUN,
        deferred_count=deferred,
    )


# -- application-level policy ---------------------------------------------------------


def _bounded_error(error: BaseException) -> str:
    return bounded_retention_detail(f"{type(error).__name__}: {error}") or type(error).__name__


def _with_dry_run(policy: _PolicyT | None, dry_run: bool) -> _PolicyT | None:
    return None if policy is None else policy.model_copy(update={"dry_run": dry_run})


def _supports(store: Any) -> bool:
    return bool(getattr(store, "supports_storage_retention", False))


async def apply_storage_retention(
    policy: StorageRetentionPolicy,
    *,
    session_store: Any,
    task_store: Any = None,
    eval_store: Any = None,
    artifact_stores: Iterable[Any] = (),
    snapshot_stores: Iterable[Any] = (),
    app: CayuApp | None = None,
    progress: RetentionProgressCallback | None = None,
) -> StorageRetentionReport:
    """Apply an application-level policy across the given stores.

    The session store is also the audit sink for targets without a database
    of their own (artifacts, workspaces). A target the policy names but that no
    configured store can serve is reported in ``skipped``; a target that fails
    is reported in ``errors`` and the remaining targets still run.
    """

    if type(policy) is not StorageRetentionPolicy:
        raise TypeError("policy must be a StorageRetentionPolicy.")
    started_at = datetime.now(UTC)
    snapshot_stores = tuple(snapshot_stores)
    artifact_stores = tuple(
        {id(store): store for store in artifact_stores if store is not None}.values()
    )
    reports: list[RetentionReport] = []
    skipped: dict[str, str] = {}
    errors: dict[str, str] = {}
    audit = session_store if _supports(session_store) else None
    held = await _held_identifiers(snapshot_stores)
    live_tasks = await live_task_session_ids(task_store) if task_store is not None else frozenset()

    async def eval_sessions() -> frozenset[str]:
        if eval_store is None or not _supports(eval_store):
            return frozenset()
        return await eval_store.retention_session_references()

    async def run(target: StorageRetentionTarget, operation: Callable[[], Any]) -> None:
        try:
            reports.append(await operation())
        except Exception as error:
            logger.warning("Storage retention for %s failed: %s", target.value, error)
            errors[target.value] = _bounded_error(error)

    workspace_policy = _with_dry_run(policy.workspaces, policy.dry_run)
    if workspace_policy is not None:
        if app is None:
            skipped["workspaces"] = "workspace retention requires the CayuApp"
        elif audit is None and not policy.dry_run:
            skipped["workspaces"] = "the session store cannot record retention audits"
        else:
            await run(
                StorageRetentionTarget.WORKSPACES,
                lambda: apply_workspace_retention(
                    app,
                    workspace_policy,
                    audit=audit,
                    protected_session_ids=policy.protected_session_ids,
                    references=_merge(
                        (RetentionProtection.LIVE_TASK, live_tasks),
                        (RetentionProtection.SNAPSHOT_PIN, held),
                    ),
                    progress=progress,
                ),
            )

    eval_policy = _with_dry_run(policy.evals, policy.dry_run)
    if eval_policy is not None:
        if eval_store is None or not _supports(eval_store):
            skipped["evals"] = "no eval store that supports retention is configured"
        else:
            await run(
                StorageRetentionTarget.EVALS,
                lambda: eval_store.apply_retention_policy(
                    eval_policy,
                    protected_ids=policy.protected_eval_ids,
                    references=_merge((RetentionProtection.SNAPSHOT_PIN, held)),
                    progress=progress,
                ),
            )

    session_policy = _with_dry_run(policy.sessions, policy.dry_run)
    if session_policy is not None:
        if not _supports(session_store):
            skipped["sessions"] = "the session store does not support retention"
        else:
            references = _merge(
                (RetentionProtection.LIVE_TASK, live_tasks),
                (RetentionProtection.EVAL_REFERENCE, await eval_sessions()),
                (RetentionProtection.SNAPSHOT_PIN, held),
            )
            await run(
                StorageRetentionTarget.SESSIONS,
                lambda: session_store.apply_retention_policy(
                    session_policy,
                    protected_session_ids=policy.protected_session_ids,
                    references=references,
                    progress=progress,
                ),
            )

    artifact_policy = _with_dry_run(policy.artifacts, policy.dry_run)
    if artifact_policy is not None:
        if not artifact_stores:
            skipped["artifacts"] = "no artifact store is registered"
        elif audit is None and not policy.dry_run:
            skipped["artifacts"] = "the session store cannot record retention audits"
        else:
            from cayu.artifacts.retention import apply_artifact_retention_policy

            try:
                artifact_references = _merge(
                    (
                        RetentionProtection.SESSION_REFERENCE,
                        await session_store.retention_artifact_references()
                        if _supports(session_store)
                        else frozenset(),
                    ),
                    (
                        RetentionProtection.EVAL_REFERENCE,
                        await eval_store.retention_artifact_references()
                        if eval_store is not None and _supports(eval_store)
                        else frozenset(),
                    ),
                    (RetentionProtection.SNAPSHOT_PIN, held),
                )
            except Exception as error:
                errors["artifacts"] = _bounded_error(error)
            else:
                for artifact_store in artifact_stores:
                    await run(
                        StorageRetentionTarget.ARTIFACTS,
                        lambda artifact_store=artifact_store: apply_artifact_retention_policy(
                            artifact_store,
                            artifact_policy,
                            session_store=session_store,
                            eval_store=eval_store,
                            snapshot_stores=snapshot_stores,
                            references=artifact_references,
                            protected_artifact_ids=policy.protected_artifact_ids,
                            audit=audit,
                            progress=progress,
                        ),
                    )

    return StorageRetentionReport(
        dry_run=policy.dry_run,
        started_at=started_at,
        completed_at=datetime.now(UTC),
        reports=tuple(reports),
        skipped=skipped,
        errors=errors,
    )


async def run_storage_retention_worker(
    app: CayuApp,
    stop: asyncio.Event,
    *,
    policy: StorageRetentionPolicy,
    interval_seconds: float,
    eval_store: Any = None,
    snapshot_stores: Iterable[Any] = (),
    on_report: Callable[[StorageRetentionReport], Any] | None = None,
) -> None:
    """Apply ``policy`` every ``interval_seconds`` until ``stop`` is set.

    A worker entrypoint with the ``(app, stop)`` contract wraps this loop; it
    is never started implicitly. A failed pass is logged and retried on the
    next interval; ``on_report`` (sync or async) receives every report.
    """

    if type(policy) is not StorageRetentionPolicy:
        raise TypeError("policy must be a StorageRetentionPolicy.")
    if isinstance(interval_seconds, bool) or not isinstance(interval_seconds, int | float):
        raise TypeError("interval_seconds must be a number.")
    if not 1 <= float(interval_seconds) <= 7 * 86_400:
        raise ValueError("interval_seconds must be between 1 second and 7 days.")
    snapshot_stores = tuple(snapshot_stores)
    while not stop.is_set():
        try:
            report = await app.apply_storage_retention(
                policy, eval_store=eval_store, snapshot_stores=snapshot_stores
            )
        except Exception:
            logger.exception("Storage retention pass failed.")
        else:
            if on_report is not None:
                outcome = on_report(report)
                if asyncio.iscoroutine(outcome):
                    await outcome
        try:
            await asyncio.wait_for(stop.wait(), timeout=float(interval_seconds))
        except TimeoutError:
            continue


__all__ = [
    "apply_storage_retention",
    "apply_workspace_retention",
    "live_task_session_ids",
    "run_storage_retention_worker",
]
