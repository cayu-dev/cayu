"""Optional one-shot TaskStore hints over the standalone external-wait owner."""

from cayu._validation import canonical_durable_json_bytes
from cayu.external_waits import (
    ExternalEventWaits,
    ExternalWaitContext,
    ExternalWaitSnapshot,
    _snapshot,
)
from cayu.runtime._external_wait_timer import timer_scope
from cayu.sessions.external_waits import (
    ExternalWaitConflict,
    ExternalWaitRegistration,
    ExternalWaitScope,
    ExternalWaitTimer,
    external_wait_digest,
    external_wait_timer,
)
from cayu.tasks.base import TaskCreate, TaskStore
from cayu.tasks.scheduling import TaskSchedulePolicy

EXTERNAL_WAIT_TASK_TYPE = "cayu.external-wait-deadline.v1"
EXTERNAL_WAIT_TIMER_BYTES = 16 * 1024


class TaskStoreWaitScheduler:
    """An explicitly driven adapter, not a worker or a second outcome clock.

    A hint can be delivered early or late. The wait store alone elects timeout;
    the host's bounded discovery remains the fallback if clocks differ or a
    notification is lost. Neither task input nor its lease grants execution.
    """

    def __init__(self, *, waits: ExternalEventWaits, task_store: TaskStore, scheduler_id: str):
        if task_store.supports_task_scheduling is not True:
            raise NotImplementedError("External timers require managed task scheduling.")
        if type(scheduler_id) is not str or not scheduler_id.strip() or len(scheduler_id) > 256:
            raise ValueError("External scheduler identity must be bounded and nonblank.")
        self.waits = waits
        self.task_store = task_store
        self.scheduler_id = scheduler_id

    @staticmethod
    def _request(timer: ExternalWaitTimer) -> TaskCreate:
        return TaskCreate(
            task_id=timer.task_id,
            type=EXTERNAL_WAIT_TASK_TYPE,
            available_at=timer.correlation.request.deadline,
            schedule_policy=TaskSchedulePolicy(),
            input=timer.model_dump(mode="json"),
        )

    async def schedule(
        self, registration: ExternalWaitRegistration, *, context: ExternalWaitContext
    ) -> ExternalWaitTimer | None:
        registration = _snapshot(registration, ExternalWaitRegistration)
        context = _snapshot(context, ExternalWaitContext)
        correlation = registration.correlation
        self.waits._authorize(correlation.request, context, "service")
        if correlation.request.deadline is None:
            return None
        timer = external_wait_timer(registration, self.scheduler_id)
        prepare = self.waits._command(
            "prepare_timer", correlation, registration=registration, timer=timer
        )
        with timer_scope(prepare):
            retained = await self.waits._mutate(prepare)
        if retained.timer_published:
            return timer
        self.waits._authorize(correlation.request, context, "service")
        await self.waits._observe_operation(
            lambda: self.task_store.create_task(self._request(timer)),
            key=("external-timer", external_wait_digest(timer)),
            expectation=external_wait_digest(timer).encode(),
        )
        self.waits._authorize(correlation.request, context, "service")
        publish = self.waits._command(
            "publish_timer", correlation, registration=registration, timer=timer
        )
        with timer_scope(publish):
            await self.waits._mutate(publish)
        return timer

    async def reconcile(
        self,
        *,
        scope: ExternalWaitScope,
        source: str,
        context: ExternalWaitContext,
        after: str = "",
        limit: int = 32,
    ) -> tuple[ExternalWaitTimer, ...]:
        """Reconstruct pending scheduling intent with bounded, caller-driven discovery."""
        rows = await self.waits.list(
            scope=scope, source=source, context=context, after=after, limit=limit
        )
        result = []
        for row in rows:
            if not row.pending_timer:
                continue
            record = await self.waits.store._read_external_wait(
                scope, row.correlation.request.correlation_key
            )
            if (
                record is not None
                and record.timer is not None
                and record.timer.scheduler_id == self.scheduler_id
                and not record.timer_published
            ):
                assert record.registration is not None
                timer = await self.schedule(record.registration, context=context)
                assert timer is not None
                result.append(timer)
        return tuple(result)

    async def notify(
        self, timer: ExternalWaitTimer, *, context: ExternalWaitContext
    ) -> ExternalWaitSnapshot | None:
        """Observe the owner clock; return None only for an authenticated obsolete hint."""
        timer = _snapshot(timer, ExternalWaitTimer)
        self.waits._authorize(timer.correlation.request, context, "service")
        if timer.scheduler_id != self.scheduler_id:
            raise ExternalWaitConflict("External timer scheduler identity conflicts.")
        record = await self.waits.store._read_external_wait(
            timer.correlation.request.scope, timer.correlation.request.correlation_key
        )
        if record is not None and record.timer != timer:
            raise ExternalWaitConflict("External timer no longer matches its retained owner.")
        if record is not None:
            try:
                return await self.waits.observe(timer.correlation, context=context)
            except ExternalWaitConflict:
                # Retirement can win between read and observation. Only its
                # durable tombstone, not absence or an arbitrary error, is proof.
                if not await self._obsolete(timer, context):
                    raise
                return None
        if await self._obsolete(timer, context):
            return None
        raise ExternalWaitConflict("External timer no longer matches its retained owner.")

    async def _obsolete(self, timer: ExternalWaitTimer, context: ExternalWaitContext) -> bool:
        retirement = await self.waits.store._read_external_wait_retirement(
            timer.correlation.request.scope
        )
        if retirement is None:
            return False
        task = await self.task_store.load_task(timer.task_id)
        if (
            retirement.request.limits != timer.correlation.limits
            or self.waits.limits != timer.correlation.limits
            or task is None
            or task.type != EXTERNAL_WAIT_TASK_TYPE
            or canonical_durable_json_bytes(task.input, "external timer")
            != canonical_durable_json_bytes(timer.model_dump(mode="json"), "external timer")
        ):
            raise ExternalWaitConflict("Obsolete external timer identity conflicts.")
        self.waits._authorize(timer.correlation.request, context, "service")
        return True

    def worker_handler(self, *, context: ExternalWaitContext):
        """Compose with run_task_worker; construction never starts a worker."""
        from cayu.tasks.worker import complete_managed_task

        context = _snapshot(context, ExternalWaitContext)

        async def handle(app, task, worker_id):
            if app.task_store is not self.task_store or task.type != EXTERNAL_WAIT_TASK_TYPE:
                raise ExternalWaitConflict("External timer worker has a different task owner.")
            try:
                encoded = canonical_durable_json_bytes(task.input, "external timer")
                if len(encoded) > EXTERNAL_WAIT_TIMER_BYTES:
                    raise ValueError("External timer exceeds its envelope.")
                timer = ExternalWaitTimer.model_validate_json(encoded)
            except (ValueError, TypeError):
                raise ExternalWaitConflict("External timer task is malformed.") from None
            if task.id != timer.task_id:
                raise ExternalWaitConflict("External timer task identity conflicts.")
            snapshot = await self.notify(timer, context=context)
            await complete_managed_task(
                self.task_store,
                task,
                worker_id,
                {
                    "outcome_kind": None
                    if snapshot is None or snapshot.outcome is None
                    else snapshot.outcome.kind,
                    "pending_handoff": False if snapshot is None else snapshot.pending_handoff,
                    **({"obsolete": True} if snapshot is None else {}),
                },
            )

        return handle
