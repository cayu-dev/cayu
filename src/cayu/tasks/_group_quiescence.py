"""Pure group barrier planning inside the existing graph transaction.

External work is never cancelled here. The plan publishes intent and retains
positive execution obligations independently of mutable task lifecycle fields.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256

from cayu._validation import canonical_durable_json_bytes
from cayu.tasks._graphs import GRAPH_TERMINAL_STATUSES, GraphTransition, plan_graph_transition
from cayu.tasks._groups import GroupPublication, plan_group_transition
from cayu.tasks.base import (
    Task,
    TaskRetrySettlementResult,
    TaskStatus,
    _cancelled_task_retry_settlement,
    _task_cancellation_requested_task,
    _task_retry_cancellation_requested_task,
    copy_task,
)
from cayu.tasks.graphs import graph_identifier
from cayu.tasks.groups import (
    TaskGroupConflict,
    TaskGroupEvent,
    TaskGroupEventType,
    TaskGroupExecutionObligation,
    TaskGroupInvocationObligation,
    TaskGroupQuiescenceResolution,
    TaskGroupSnapshot,
    TaskGroupStatus,
)
from cayu.tasks.groups import (
    TaskGroupFinalizerStatus as F,
)
from cayu.tasks.groups import (
    TaskGroupQuiescenceStatus as Q,
)


@dataclass(frozen=True)
class GroupGraphPublication:
    transition: GraphTransition
    group: GroupPublication


def validate_reconciliation_page(after_group_id: str | None, limit: int) -> None:
    if after_group_id is not None:
        graph_identifier(after_group_id)
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("Group reconciliation pages must contain at most 100 identities.")


def prepare_resolution(
    request: TaskGroupQuiescenceResolution,
) -> tuple[TaskGroupQuiescenceResolution, str]:
    if type(request) is not TaskGroupQuiescenceResolution:
        raise TaskGroupConflict("Resolution requires typed group authority.")
    request = TaskGroupQuiescenceResolution.model_validate(
        request.model_dump(mode="json", warnings=False)
    )
    return request, sha256(
        canonical_durable_json_bytes(
            request.model_dump(mode="json", warnings=False), "group resolution"
        )
    ).hexdigest()


def require_resolution(snapshot: TaskGroupSnapshot, request: TaskGroupQuiescenceResolution) -> None:
    if (
        snapshot.receipt.group_id != request.group_id
        or snapshot.receipt.request_sha256 != request.request_sha256
        or snapshot.last_sequence != request.expected_sequence
        or snapshot.quiescence.status is not Q.ATTENTION_REQUIRED
    ):
        raise TaskGroupConflict("Resolution conflicts with exact attention-required authority.")


def waiting_finalizer(snapshot: TaskGroupSnapshot) -> str | None:
    if snapshot.quiescence.finalizer_status is F.WAITING:
        return snapshot.receipt.finalizer_task_id
    return None


def reconciliation_is_complete(snapshot: TaskGroupSnapshot) -> bool:
    """Retained terminal evidence suffices for ordinary read-only reconciliation.

    This is not authority for a new settlement or attention resolution. Callers
    must keep those requests on the exact mutation/validation path.
    """
    return (
        snapshot.decision is not None
        and snapshot.quiescence.status is Q.QUIESCENT
        and snapshot.quiescence.finalizer_status in {F.ABSENT, F.INELIGIBLE, F.SETTLED}
        and all(member.status in GRAPH_TERMINAL_STATUSES for member in snapshot.members)
    )


def retained_execution_settlement_matches(
    snapshot: TaskGroupSnapshot, settled_execution: tuple[str, str, datetime]
) -> bool:
    """Authenticate acknowledgement replay without requiring deleted task rows.

    Only an already-settled exact execution is proof. A terminal group outcome
    by itself never authorizes a new settlement or a different execution.
    """
    identity, worker_id, started_at = settled_execution
    execution = next(
        (item for item in snapshot.quiescence.executions if item.task_id == identity), None
    )
    if execution is None or execution.worker_id != worker_id or execution.started_at != started_at:
        raise TaskGroupConflict("Execution settlement does not match its retained group owner.")
    return execution.settled_at is not None and (
        execution.result_resolution is None or execution.result_resolution.settled_at is not None
    )


def retained_invocation_settlement_matches(
    snapshot: TaskGroupSnapshot, invocation: TaskGroupInvocationObligation
) -> bool:
    """Authenticate an owner-return replay against the retained release receipt."""
    execution = next(
        (item for item in snapshot.quiescence.executions if item.task_id == invocation.task_id),
        None,
    )
    if execution is not None and execution.worker_id is not None:
        # A nested session does not own this execution. Keep its observation on
        # the live-task validation path; only the outer worker can settle it.
        return False
    prior = None if execution is None else execution.invocation
    if (
        prior is None
        or not invocation.owner_settled
        or prior.model_copy(update={"release_record_sha256": None})
        != invocation.model_copy(update={"release_record_sha256": None})
        or invocation.release_record_sha256 not in {None, prior.release_record_sha256}
    ):
        raise TaskGroupConflict("Invocation settlement does not match its retained group owner.")
    assert execution is not None
    return (
        execution.worker_id is None
        and execution.settled_at is not None
        and prior.release_record_sha256 is not None
        and (
            execution.result_resolution is None
            or execution.result_resolution.settled_at is not None
        )
    )


def require_group_deletion_ready(snapshot: TaskGroupSnapshot) -> None:
    """Keep planner inputs until every execution acknowledgement has committed.

    Finalizer eligibility only waits for losers; evidence retention must also
    protect winning executions and their acknowledgement-only retry owners.
    """
    from cayu.tasks.graphs import TaskGraphConflict

    if snapshot.receipt.quiescence is not None and (
        snapshot.quiescence.status is not Q.QUIESCENT
        or any(
            execution.settled_at is None
            or (
                execution.result_resolution is not None
                and execution.result_resolution.settled_at is None
            )
            for execution in snapshot.quiescence.executions
        )
    ):
        raise TaskGraphConflict("Unsettled task group retains its execution evidence.")


def require_group_mutation(
    snapshot: TaskGroupSnapshot,
    prior: Task,
    proposed: Task,
    *,
    root_task_id: str | None = None,
) -> None:
    """A durable loser cannot regain execution or publish a successful effect."""
    if (root_task_id or prior.id) not in snapshot.quiescence.loser_task_ids:
        return
    if prior.status in GRAPH_TERMINAL_STATUSES:
        return  # The graph owner separately authenticates immutable terminal replay.
    if proposed.status is TaskStatus.COMPLETED:
        raise TaskGroupConflict("A fenced group loser cannot publish successful completion.")
    if proposed.status in {TaskStatus.CLAIMED, TaskStatus.RUNNING} and (
        prior.status not in {TaskStatus.CLAIMED, TaskStatus.RUNNING}
        or prior.worker_id != proposed.worker_id
        or prior.started_at != proposed.started_at
        or prior.session_instance_id != proposed.session_instance_id
    ):
        raise TaskGroupConflict("A fenced group loser cannot start or resume execution.")


def observe_result_resolution(
    snapshot: TaskGroupSnapshot,
    root_id: str,
    task_id: str,
    decision_id: str,
    owner_id: str,
    *,
    settled: bool,
    now: datetime,
) -> TaskGroupSnapshot:
    """Serialize callback dispatch/settlement with the group election.

    This marker has no lease: an expired observer cannot prove that an opaque
    callback stopped. Only its exact retained owner can acknowledge settlement.
    """
    from cayu.tasks.groups import (
        TaskGroupResultResolutionObligation,
        TaskGroupResultResolutionPending,
    )

    proposed = TaskGroupResultResolutionObligation(decision_id=decision_id, owner_id=owner_id)
    if type(settled) is not bool:
        raise TaskGroupConflict("Invalid resolver settlement.")
    if snapshot.receipt.quiescence is None or root_id not in snapshot.receipt.member_task_ids:
        return snapshot
    executions = list(snapshot.quiescence.executions)
    index = next((i for i, item in enumerate(executions) if item.task_id == task_id), None)
    if index is None:
        raise TaskGroupConflict("Result resolution has no execution owner.")
    execution = executions[index]
    prior = execution.result_resolution
    if settled and prior is None:
        # Entry failed before publication; the runtime retained proof that it
        # never dispatched the callback. There is no owner to retire.
        return snapshot
    exact = prior is not None and (prior.decision_id, prior.owner_id) == (decision_id, owner_id)
    if settled:
        if not exact:
            raise TaskGroupConflict("Resolver settlement conflicts with its owner.")
        assert prior is not None
        proposed = prior.model_copy(update={"settled_at": prior.settled_at or now})
    else:
        if root_id in snapshot.quiescence.loser_task_ids or execution.settled_at is not None:
            raise TaskGroupConflict("A cancelled group member cannot dispatch result resolution.")
        if prior is not None and prior.settled_at is None:
            if not exact:
                raise TaskGroupResultResolutionPending(
                    "The prior result resolver is still running."
                )
            return snapshot
        if exact:
            raise TaskGroupConflict("A settled result resolver cannot dispatch again.")
    executions[index] = execution.model_copy(update={"result_resolution": proposed})
    return snapshot.model_copy(
        update={
            "quiescence": snapshot.quiescence.model_copy(update={"executions": tuple(executions)})
        }
    )


def plan_group_graph_transition(
    snapshot: TaskGroupSnapshot,
    transition: GraphTransition,
    *,
    current: Mapping[str, Task],
    prerequisites: Mapping[str, tuple[str, ...]],
    first_sequence: int,
    now: datetime,
    settled_execution: tuple[str, str, datetime] | None = None,
    resolve_attention: bool = False,
    unsettled_effects: frozenset[str] = frozenset(),
    lineage_roots: Mapping[str, str] | None = None,
    invocation: TaskGroupInvocationObligation | None = None,
) -> GroupGraphPublication:
    """Plan the complete graph/group write set before publishing any part."""
    if snapshot.receipt.quiescence is not None:
        transition = _fresh_resumed_execution(snapshot, transition, current)
    publication = plan_group_transition(snapshot, transition, now=now)
    if snapshot.receipt.quiescence is None:
        if resolve_attention:
            raise TaskGroupConflict("Outcome-only group has no quiescence authority.")
        return GroupGraphPublication(transition, publication)
    original = dict(current)
    tasks = dict(current)
    tasks.update({task.id: task for task in transition.tasks})
    lineage_roots = {} if lineage_roots is None else lineage_roots
    from cayu.tasks._group_lineage import validate_lineage

    validate_lineage(snapshot, tasks, lineage_roots)
    graph_events = list(transition.events)
    retry_settlements = list(transition.retry_settlements)
    group_events = list(publication.events)
    result = publication.snapshot
    barrier = result.quiescence
    executions = {item.task_id: item for item in barrier.executions}

    # A marker is positive dispatch ownership, not an inference from task outcome.
    for identity in (*snapshot.receipt.member_task_ids, *lineage_roots):
        task = tasks[identity]
        if task.status in {TaskStatus.CLAIMED, TaskStatus.RUNNING} and task.started_at is not None:
            prior = executions.get(identity)
            if (
                prior is None
                or (task.worker_id is not None and prior.worker_id != task.worker_id)
                or prior.started_at != task.started_at
            ):
                if prior is not None and prior.settled_at is None:
                    raise TaskGroupConflict("Prior group execution is still unsettled.")
                executions[identity] = TaskGroupExecutionObligation(
                    task_id=identity,
                    worker_id=task.worker_id,
                    started_at=task.started_at,
                )
    if invocation is not None:
        invocation = TaskGroupInvocationObligation.model_validate(
            invocation.model_dump(mode="python", warnings=False)
        )
        task = tasks.get(invocation.task_id)
        execution = executions.get(invocation.task_id)
        if (
            task is None
            or execution is None
            or task.session_id != invocation.session_id
            or task.session_instance_id != invocation.session_instance_id
            or task.started_at != execution.started_at
        ):
            raise TaskGroupConflict(
                "Invocation observation conflicts with the exact attached task."
            )
        if execution.worker_id is None and task.work_contract is None:
            prior_invocation = execution.invocation
            expected = invocation.model_copy(
                update={"release_record_sha256": None, "owner_settled": False}
            )
            if (
                prior_invocation is not None
                and prior_invocation.model_copy(
                    update={"release_record_sha256": None, "owner_settled": False}
                )
                != expected
            ):
                if (
                    execution.settled_at is None
                    or invocation.release_record_sha256 is not None
                    or invocation.run_epoch <= prior_invocation.run_epoch
                ):
                    raise TaskGroupConflict(
                        "A different invocation still owns the group execution."
                    )
            elif (
                prior_invocation is not None
                and prior_invocation.owner_settled
                and not invocation.owner_settled
            ):
                raise TaskGroupConflict("Settled invocation cannot regain execution ownership.")
            elif (
                prior_invocation is not None and prior_invocation.release_record_sha256 is not None
            ):
                if invocation.release_record_sha256 not in {
                    None,
                    prior_invocation.release_record_sha256,
                }:
                    raise TaskGroupConflict("Invocation release conflicts with its exact receipt.")
                # A repeated owner acknowledgement cannot erase a later exact
                # release. This is reconciliation, not a new execution grant.
                invocation = prior_invocation
            if invocation.release_record_sha256 is None and execution.settled_at is not None:
                if prior_invocation == invocation:
                    raise TaskGroupConflict(
                        "Released invocation cannot regain execution authority."
                    )
                if invocation.task_id in barrier.loser_task_ids:
                    raise TaskGroupConflict("A decided loser cannot start another invocation.")
            if (
                prior_invocation is not None
                and prior_invocation.release_record_sha256 is not None
                and prior_invocation != invocation
                and invocation.release_record_sha256 is not None
            ):
                raise TaskGroupConflict("Invocation release conflicts with its exact receipt.")
            executions[task.id] = execution.model_copy(
                update={
                    "invocation": invocation,
                    "settled_at": (
                        execution.settled_at or now
                        if invocation.release_record_sha256 is not None
                        else None
                    ),
                }
            )
        # A session nested inside a worker does not settle that worker's later
        # callback work. Its natural return/lifecycle receipt owns that proof.
    if settled_execution is not None and settled_execution[0] in {
        *snapshot.receipt.member_task_ids,
        *lineage_roots,
    }:
        identity, worker_id, started_at = settled_execution
        execution = executions.get(identity)
        if (
            execution is None
            or execution.worker_id != worker_id
            or execution.started_at != started_at
        ):
            raise TaskGroupConflict("Execution settlement does not match its group owner.")
        if (
            execution.result_resolution is not None
            and execution.result_resolution.settled_at is None
        ):
            from cayu.tasks.groups import TaskGroupResultResolutionPending

            raise TaskGroupResultResolutionPending("The result resolver has not settled.")
        if execution.settled_at is None:
            executions[identity] = execution.model_copy(update={"settled_at": now})
    barrier = barrier.model_copy(
        update={"executions": tuple(executions[k] for k in sorted(executions))}
    )

    def event(kind: TaskGroupEventType, task_id: str | None = None) -> None:
        group_events.append(
            TaskGroupEvent(
                group_id=snapshot.receipt.group_id,
                sequence=snapshot.last_sequence + len(group_events) + 1,
                type=kind,
                occurred_at=now,
                task_id=task_id,
                # Events carry bounded barrier summaries; full per-attempt
                # obligations remain inspectable in the authoritative snapshot.
                quiescence=barrier.model_copy(update={"executions": ()}),
            )
        )

    def change(task: Task, settlement: TaskRetrySettlementResult | None = None) -> None:
        nonlocal result, tasks
        if task.id in lineage_roots:
            tasks[task.id] = copy_task(task)
            if settlement is not None:
                retry_settlements.append(settlement)
            return
        extra = plan_graph_transition(
            graph_id=snapshot.receipt.graph.graph_id,
            prerequisites=prerequisites,
            current={identity: tasks[identity] for identity in prerequisites},
            proposed=task,
            proposed_readiness_recorded=False,
            first_sequence=first_sequence + len(graph_events),
            now=now,
            group_waiting_task_id=(
                snapshot.receipt.finalizer_task_id
                if barrier.finalizer_status is F.WAITING
                else None
            ),
        )
        next_group = plan_group_transition(result, extra, now=now)
        # Barrier events share the same cursor with member/decision events.
        for emitted in next_group.events:
            group_events.append(
                emitted.model_copy(
                    update={
                        "sequence": snapshot.last_sequence + len(group_events) + 1,
                    }
                )
            )
        result = next_group.snapshot
        tasks.update({item.id: item for item in extra.tasks})
        graph_events.extend(extra.events)
        retry_settlements.extend(extra.retry_settlements)
        if settlement is not None:
            retry_settlements.append(settlement)

    decision = result.decision
    if decision is not None:
        new_decision = barrier.status is Q.WAITING_DECISION
        if new_decision:
            losers = tuple(
                identity
                for identity in snapshot.receipt.member_task_ids
                if decision.status is TaskGroupStatus.FAILED
                or identity not in decision.successful_task_ids
            )
            barrier = barrier.model_copy(
                update={
                    "status": Q.DRAINING,
                    "deadline": decision.decided_at
                    + timedelta(seconds=snapshot.receipt.quiescence.timeout_seconds),
                    "loser_task_ids": losers,
                }
            )
        losing_work = tuple(barrier.loser_task_ids) + tuple(
            identity for identity, root in lineage_roots.items() if root in barrier.loser_task_ids
        )
        for identity in losing_work:
            task = tasks[identity]
            if task.status in GRAPH_TERMINAL_STATUSES:
                continue
            observed_execution = executions.get(identity)
            if (
                task.worker_id is None
                and task.work_contract is None
                and observed_execution is not None
                and observed_execution.worker_id is None
                and observed_execution.settled_at is not None
                and observed_execution.invocation is not None
                and observed_execution.invocation.release_record_sha256 is not None
                and identity not in unsettled_effects
            ):
                # SessionStore's authenticated release proves this exact
                # ownerless invocation stopped. No worker lease is erased here.
                change(
                    copy_task(
                        task.model_copy(
                            update={
                                "status": TaskStatus.CANCELLED,
                                "status_reason": "task_group_decided",
                                "status_payload": None,
                                "error": {"code": "task_group_decided"},
                                "lease_expires_at": None,
                                "completed_at": now,
                                "updated_at": now,
                            }
                        )
                    )
                )
                continue
            if (
                task.status is TaskStatus.CLAIMED
                and (
                    task.started_at is None
                    or (
                        observed_execution is not None and observed_execution.settled_at is not None
                    )
                )
                and task.session_id is None
                and task.lease_expires_at is not None
                and task.lease_expires_at <= now
                and identity not in unsettled_effects
            ):
                # Require nonexecution or a positive execution-owner handoff,
                # plus the independent effect ledger. Expiry alone is not proof.
                if task.retry_series is not None:
                    cancellation = _cancelled_task_retry_settlement(
                        task,
                        error={"code": "task_group_decided"},
                        committed_at=now,
                    )
                    change(cancellation.task, cancellation)
                else:
                    change(
                        copy_task(
                            task.model_copy(
                                update={
                                    "status": TaskStatus.CANCELLED,
                                    "status_reason": "task_group_decided",
                                    "status_payload": None,
                                    "error": {"code": "task_group_decided"},
                                    "worker_id": None,
                                    "lease_expires_at": None,
                                    "completed_at": now,
                                    "updated_at": now,
                                }
                            )
                        )
                    )
                continue
            if task.status_reason in {"cancellation_requested", "retry_cancellation_requested"}:
                continue
            if task.status in {TaskStatus.CLAIMED, TaskStatus.RUNNING}:
                if task.worker_id is None or task.lease_expires_at is None:
                    # An attached ownerless invocation is not proof of nonexecution.
                    if new_decision:
                        event(TaskGroupEventType.CANCELLATION_REQUESTED, identity)
                    continue
                # Admitted verified work has its own stop/settlement owner.
                if task.work_contract is not None and task.session_id is not None:
                    if new_decision:
                        event(TaskGroupEventType.CANCELLATION_REQUESTED, identity)
                    continue
                prepare = (
                    _task_retry_cancellation_requested_task
                    if task.retry_series is not None
                    else _task_cancellation_requested_task
                )
                change(prepare(task, error={"code": "task_group_decided"}, updated_at=now))
            elif task.retry_series is not None:
                cancellation = _cancelled_task_retry_settlement(
                    task,
                    error={"code": "task_group_decided"},
                    committed_at=now,
                )
                change(cancellation.task, cancellation)
            else:
                change(
                    copy_task(
                        task.model_copy(
                            update={
                                "status": TaskStatus.CANCELLED,
                                "status_reason": "task_group_decided",
                                "status_payload": None,
                                "error": {"code": "task_group_decided"},
                                "worker_id": None,
                                "lease_expires_at": None,
                                "completed_at": now,
                                "updated_at": now,
                            }
                        )
                    )
                )
            event(TaskGroupEventType.CANCELLATION_REQUESTED, identity)
        unsettled = tuple(
            identity
            for identity in barrier.loser_task_ids
            if tasks[identity].status not in GRAPH_TERMINAL_STATUSES
            or (identity in executions and executions[identity].settled_at is None)
            or identity in unsettled_effects
            # A root terminal snapshot cannot prove its retry successor stopped.
            # Keep the root fenced until lineage settlement is reconciled.
            or any(
                tasks[child].status not in GRAPH_TERMINAL_STATUSES
                or (child in executions and executions[child].settled_at is None)
                or child in unsettled_effects
                for child, root in lineage_roots.items()
                if root == identity
            )
        )
        previous_unsettled = barrier.unsettled_task_ids
        barrier = barrier.model_copy(update={"unsettled_task_ids": unsettled})
        if new_decision or previous_unsettled != unsettled:
            event(TaskGroupEventType.DRAINING)
        if barrier.status is Q.DRAINING:
            assert barrier.deadline is not None
            if now >= barrier.deadline:
                barrier = barrier.model_copy(update={"status": Q.ATTENTION_REQUIRED})
                event(TaskGroupEventType.TIMEOUT)
            elif not unsettled:
                barrier = barrier.model_copy(update={"status": Q.QUIESCENT})
                event(TaskGroupEventType.QUIESCENT)
        if resolve_attention:
            if barrier.status is not Q.ATTENTION_REQUIRED or unsettled:
                raise TaskGroupConflict(
                    "Attention resolution requires positive complete settlement."
                )
            barrier = barrier.model_copy(update={"status": Q.QUIESCENT})
            event(TaskGroupEventType.RESOLVED)
            event(TaskGroupEventType.QUIESCENT)
        finalizer_id = snapshot.receipt.finalizer_task_id
        if finalizer_id is not None:
            finalizer = tasks[finalizer_id]
            if barrier.finalizer_status is F.WAITING:
                if decision.status is TaskGroupStatus.FAILED:
                    barrier = barrier.model_copy(update={"finalizer_status": F.INELIGIBLE})
                    if finalizer.status not in GRAPH_TERMINAL_STATUSES:
                        change(
                            copy_task(
                                finalizer.model_copy(
                                    update={
                                        "status": TaskStatus.CANCELLED,
                                        "status_reason": "task_group_failed",
                                        "error": {"code": "task_group_failed"},
                                        "completed_at": now,
                                        "updated_at": now,
                                    }
                                )
                            )
                        )
                    event(TaskGroupEventType.FINALIZER_INELIGIBLE, finalizer_id)
                elif barrier.status is Q.QUIESCENT:
                    barrier = barrier.model_copy(update={"finalizer_status": F.RELEASED})
                    if finalizer.status is TaskStatus.WAITING_GROUP:
                        change(
                            finalizer.model_copy(
                                update={
                                    "status": TaskStatus.PENDING,
                                    "updated_at": now,
                                }
                            )
                        )
                    event(TaskGroupEventType.FINALIZER_RELEASED, finalizer_id)
            if (
                barrier.finalizer_status is F.RELEASED
                and tasks[finalizer_id].status in GRAPH_TERMINAL_STATUSES
            ):
                barrier = barrier.model_copy(update={"finalizer_status": F.SETTLED})
                event(TaskGroupEventType.FINALIZER_SETTLED, finalizer_id)
    result = TaskGroupSnapshot(
        receipt=result.receipt,
        members=result.members,
        decision=result.decision,
        last_sequence=snapshot.last_sequence + len(group_events),
        quiescence=barrier,
    )
    return GroupGraphPublication(
        GraphTransition(
            tasks=tuple(tasks[k] for k in sorted(tasks) if tasks[k] != original.get(k)),
            events=tuple(graph_events),
            retry_settlements=tuple(retry_settlements),
        ),
        GroupPublication(result, tuple(group_events)),
    )


def _fresh_resumed_execution(
    snapshot: TaskGroupSnapshot, transition: GraphTransition, current: Mapping[str, Task]
) -> GraphTransition:
    """A resumed, sessionless callback must not reuse an old dispatch marker.

    The marker remains a durable execution identity, not lease-time authority.
    Advance it strictly past its predecessor even if store time repeats or moves
    backwards. No task-wide schema or unrelated task history is changed.
    """
    executions = {item.task_id: item for item in snapshot.quiescence.executions}
    changed = []
    for task in transition.tasks:
        prior = current.get(task.id)
        execution = executions.get(task.id)
        if prior is not None and execution is not None and task.session_id is None:
            if prior.status in {
                TaskStatus.PAUSED,
                TaskStatus.BLOCKED,
                TaskStatus.NEEDS_ATTENTION,
            } and task.status in {TaskStatus.PENDING, TaskStatus.WAITING_DEPENDENCIES}:
                if execution.settled_at is None or (
                    prior.started_at is not None and execution.started_at != prior.started_at
                ):
                    raise TaskGroupConflict("Task resume requires exact prior settlement.")
                # A prior resume may already have cleared this marker without
                # another dispatch. Repeated holds still require the retained
                # execution's positive settlement, not a new acknowledgement.
                task = task.model_copy(update={"started_at": None})
            elif (
                task.status is TaskStatus.CLAIMED
                and prior.started_at is None
                and task.started_at is not None
            ):
                if execution.settled_at is None:
                    raise TaskGroupConflict("The prior task execution has not settled.")
                task = task.model_copy(
                    update={
                        "started_at": max(
                            task.started_at, execution.started_at + timedelta(microseconds=1)
                        )
                    }
                )
        changed.append(task)
    return GraphTransition(
        tasks=tuple(changed),
        events=transition.events,
        retry_settlements=transition.retry_settlements,
    )
