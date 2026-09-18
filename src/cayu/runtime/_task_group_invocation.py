"""Bridge exact invocation release into the TaskStore-owned group barrier.

This observes existing lifecycle receipts; it neither releases environments nor
equates a terminal session status with quiescence. Worker callbacks have their
own execution obligations and are never acknowledged by this bridge.
"""

from __future__ import annotations

import asyncio

from cayu._exception_groups import exception_cause, set_exception_cause
from cayu.runtime._invocation_lifecycle import (
    AdmittedInvocationBinding,
    InvocationContext,
    released_invocation_evidence,
    require_invocation_rebind_lineage,
)
from cayu.runtime._task_store_operation_boundary import (
    TaskStoreOperationOutcome,
    capture_task_store_operation,
    raise_task_store_operation_failure,
)
from cayu.runtime.execution_profiles import (
    ActiveInvocationExecutionProfile,
    active_invocation_execution_profile_from_checkpoint,
)
from cayu.sessions.base import SessionRunFenced, SessionStore
from cayu.tasks.base import TaskStore
from cayu.tasks.groups import TaskGroupInvocationObligation, TaskGroupUnavailable
from cayu.vaults.redaction import SecretRedactor


class TaskGroupInvocationSettlementPending(RuntimeError):
    """Retain ``settlement`` and await its ``retry()`` to publish owner return.

    Retry never executes the invocation again. Process loss before publication
    deliberately leaves the durable barrier fenced.
    """

    def __init__(self, settlement: _InvocationSettlement) -> None:
        super().__init__("Task group invocation acknowledgement remains pending.")
        self.settlement = settlement


class _InvocationSettlement:
    """Own exact acknowledgement independently of the completed invocation."""

    def __init__(
        self,
        store: TaskStore,
        session_store: SessionStore,
        observation: TaskGroupInvocationObligation,
        redactor: SecretRedactor,
    ) -> None:
        self._store = store
        self._session_store = session_store
        self._observation = observation.model_copy(deep=True, update={"owner_settled": True})
        self._redactor = redactor
        self._operation: asyncio.Task[TaskStoreOperationOutcome[None]] | None = None
        self._lock = asyncio.Lock()
        self._settled = False

    async def _publish(self) -> None:
        await self._store._observe_task_group_invocation(self._observation.model_copy(deep=True))
        await observe_release(
            self._store, self._session_store, self._observation, redactor=self._redactor
        )

    async def retry(self) -> None:
        if self._settled:
            return
        if self._lock.locked():
            raise TaskGroupInvocationSettlementPending(self)
        async with self._lock:
            if self._operation is None:
                self._operation = asyncio.create_task(
                    capture_task_store_operation(
                        self._publish,
                        operation_name="Task group invocation owner return",
                        redactor=self._redactor,
                    ),
                    name="cayu-group-invocation-settlement",
                )
            try:
                # Observation is bounded; cancellation never aborts the write.
                # Retries join this same task until it naturally settles.
                done, _ = await asyncio.wait((self._operation,), timeout=1.0)
            except BaseException as error:
                pending = TaskGroupInvocationSettlementPending(self)
                set_exception_cause(pending, exception_cause(error))
                set_exception_cause(error, pending)
                raise
            if not done:
                raise TaskGroupInvocationSettlementPending(self)
            outcome = self._operation.result()
            self._operation = None
            if outcome.failure is not None:
                pending = TaskGroupInvocationSettlementPending(self)
                if not isinstance(outcome.failure, Exception):
                    set_exception_cause(pending, exception_cause(outcome.failure))
                    set_exception_cause(outcome.failure, pending)
                    raise_task_store_operation_failure(outcome.failure)
                raise pending from outcome.failure
            self._settled = True


def invocation_obligation(
    task_id: str, context: InvocationContext
) -> TaskGroupInvocationObligation:
    binding = context.binding
    if type(binding) is not AdmittedInvocationBinding:
        raise TypeError("Group invocation observation requires an admitted runtime binding.")
    return TaskGroupInvocationObligation(
        task_id=task_id,
        session_id=binding.session_id,
        session_instance_id=binding.session_instance_id,
        interaction_id=binding.interaction_id,
        run_epoch=binding.run_epoch,
        profile_fingerprint=context.profile.fingerprint,
    )


async def prepare_invocation_obligation(
    store: TaskStore | None,
    *,
    task_id: str,
    context: InvocationContext,
    redactor: SecretRedactor,
) -> TaskGroupInvocationObligation | None:
    """Resolve opt-in participation before applying the group's evidence schema."""
    if (
        store is None
        or not store.supports_task_group_quiescence
        or context.work_attempt is not None
    ):
        return None

    async def prepare():
        retained = await store._task_group_retains_execution(task_id)
        if type(retained) is not bool:
            raise TaskGroupUnavailable("Group invocation membership requires boolean authority.")
        return invocation_obligation(task_id, context) if retained else None

    outcome = await capture_task_store_operation(
        prepare,
        operation_name="Task group invocation membership",
        redactor=redactor,
    )
    if outcome.failure is not None:
        raise_task_store_operation_failure(outcome.failure)
    return outcome.result


async def bind_invocation(
    store: TaskStore | None,
    *,
    task_id: str,
    context: InvocationContext,
    redactor: SecretRedactor,
) -> None:
    observation = await prepare_invocation_obligation(
        store, task_id=task_id, context=context, redactor=redactor
    )
    if observation is None:
        return
    assert store is not None
    outcome = await capture_task_store_operation(
        lambda: store._observe_task_group_invocation(observation),
        operation_name="Task group invocation binding",
        redactor=redactor,
    )
    if outcome.failure is not None:
        raise_task_store_operation_failure(outcome.failure)


async def observe_release(
    store: TaskStore | None,
    session_store: SessionStore,
    observation: TaskGroupInvocationObligation,
    *,
    redactor: SecretRedactor,
) -> None:
    if store is None or not store.supports_task_group_quiescence or not observation.owner_settled:
        return

    async def read_release():
        session = await session_store.load(observation.session_id)
        if session is None or session.instance_id != observation.session_instance_id:
            return None
        checkpoint = await session_store.load_checkpoint(observation.session_id)
        active = active_invocation_execution_profile_from_checkpoint(checkpoint)
        if (
            active is None
            or active.profile.fingerprint != observation.profile_fingerprint
            or active.interaction_id != observation.interaction_id
        ):
            return None
        original = ActiveInvocationExecutionProfile(
            session_id=observation.session_id,
            interaction_id=observation.interaction_id,
            run_epoch=observation.run_epoch,
            profile=active.profile,
        )
        try:
            require_invocation_rebind_lineage(
                checkpoint,
                session_instance_id=observation.session_instance_id,
                original=original,
                current=active,
            )
            return released_invocation_evidence(
                session,
                checkpoint,
                session_id=observation.session_id,
                session_instance_id=observation.session_instance_id,
                active_profile=active,
            )
        except SessionRunFenced:
            # Active/uncertain cleanup remains owned by the existing runtime.
            return None

    read = await capture_task_store_operation(
        read_release,
        operation_name="Task group exact invocation release",
        redactor=redactor,
    )
    if read.failure is not None:
        raise_task_store_operation_failure(read.failure)
    if read.result is None:
        return
    released = observation.model_copy(update={"release_record_sha256": read.result.record_sha256})
    published = await capture_task_store_operation(
        lambda: store._observe_task_group_invocation(released),
        operation_name="Task group invocation settlement",
        redactor=redactor,
    )
    if published.failure is not None:
        raise_task_store_operation_failure(published.failure)


async def observe_owner_return(
    store: TaskStore,
    session_store: SessionStore,
    observation: TaskGroupInvocationObligation,
    *,
    redactor: SecretRedactor,
) -> None:
    """Record the execution owner's return separately from environment release.

    An interruption coordinator can publish release while the live invocation
    is still draining. Recovery must retain that invocation's obligation until
    this exact owner acknowledges return; a release receipt alone cannot do so.
    """
    await _InvocationSettlement(store, session_store, observation, redactor).retry()
