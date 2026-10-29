"""External-event waits that ``cayu worker`` services and ``cayu serve`` delivers.

``app.host_external_waits(...)`` builds the standalone parts once from the
application's own stores: :class:`~cayu.external_waits.ExternalEventWaits`, its
:class:`~cayu.session_external_waits.SessionExternalWaitAdapter`, an
:class:`~cayu.external_wait_host.ExternalWaitHost` and a
:class:`~cayu.external_wait_scheduler.TaskStoreWaitScheduler`. The application
then registers, parks and delivers through the returned hosting:

- every deadline becomes a delayed task and every accepted event a task due now,
  which ``run_task_worker`` runs itself, continuing the parked session;
- the worker also sweeps every configured scope and source on its reclaim
  cadence, starting with its first pass, and recreates the task of any wait that
  still needs one, so a task a crash left out is repaired;
- ``cayu serve`` mounts the delivery route of the configured event sources.

Every continuation runs inside a task, under its lease, never in the web process
or a sweep. Each part stays usable on its own.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import TYPE_CHECKING, Any, ClassVar

from cayu._validation import canonical_durable_json_bytes
from cayu.external_wait_host import ExternalWaitHost
from cayu.external_wait_scheduler import (
    EXTERNAL_WAIT_TASK_TYPE,
    TaskStoreWaitScheduler,
    timer_from_task,
    timer_result,
)
from cayu.external_waits import (
    ExternalEventWaits,
    ExternalWaitAccessPolicy,
    ExternalWaitContext,
    ExternalWaitProjector,
    ExternalWaitSnapshot,
    _snapshot,
    _view,
)
from cayu.session_external_waits import SessionExternalWaitAdapter, SessionExternalWaitReceipt
from cayu.sessions.external_waits import (
    EXTERNAL_WAIT_PAGE_SIZE,
    ExternalCorrelation,
    ExternalCorrelationRequest,
    ExternalDeliveryReceipt,
    ExternalEventDelivery,
    ExternalWaitConflict,
    ExternalWaitLimits,
    ExternalWaitRegistration,
    ExternalWaitScope,
    ExternalWaitTimer,
    Identifier,
    _Value,
)
from cayu.tasks.creation import TaskCreate
from cayu.tasks.queries import TaskQuery
from cayu.tasks.scheduling import TaskSchedulePolicy
from cayu.tasks.session_wakes import _attempt_is_live
from cayu.tasks.worker import complete_managed_task

if TYPE_CHECKING:
    from fastapi import APIRouter, Request

    from cayu.applications import CayuApp
    from cayu.runtime.loop_policies import LoopPolicy
    from cayu.sessions.requests import ResumeRequest, RunRequest
    from cayu.tasks.records import Task
    from cayu.tasks.worker import TaskHandler
    from cayu.vaults.redaction import SecretRedactor

#: The task that continues one parked wait after its event or timeout was elected.
EXTERNAL_WAIT_SERVICE_TASK_TYPE = "cayu.external-wait-service.v1"
#: How long a service task that found its wait not ready yet keeps retrying.
EXTERNAL_WAIT_SERVICE_RETRY_WINDOW = timedelta(hours=24)
_SERVICE_RETRY_BACKOFF = timedelta(seconds=30)
# Backoff and window bound the attempts; this only bounds a scan of malformed data.
_MAX_SERVICE_ATTEMPTS = 128
_MAX_SERVICE_RETRY_BACKOFF = timedelta(hours=1)
_DEFAULT_SCHEDULER_ID = "cayu-external-waits"
# A webhook body carries the event payload plus the sender's own envelope.
_DELIVERY_ENVELOPE_BYTES = 16 * 1024
_MAX_DELIVERY_REQUEST_BYTES = 1024 * 1024
_PATH_SEGMENT_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~"
)
_logger = logging.getLogger(__name__)


class ReceivedExternalEvent(_Value):
    """One authenticated event, as an :class:`ExternalEventSource` read it from a request.

    ``scope`` is the exact scope generation the correlation was issued in, read from
    the signed request (for example a generation the application put in the callback
    URL or the correlation key it gave the sender). A key is never reopened within a
    generation, so scope and key name exactly one wait; a request replayed after the
    application moved to a new generation that reuses the key is refused.
    """

    context: ExternalWaitContext
    scope: ExternalWaitScope
    correlation_key: Identifier
    delivery_id: Identifier
    payload_json: str


class ExternalEventSource(ABC):
    """Authenticates and reads the webhooks of one external-wait ``source``.

    The delivery route is outside ``cayu serve --auth``: each source authenticates
    its own callers.
    """

    source: ClassVar[str]

    @abstractmethod
    async def receive(self, request: Request) -> ReceivedExternalEvent | None:
        """Return the authenticated event, or ``None`` to refuse the request with 401.

        Verify the sender (a signature or a shared secret) and build the context
        from that verification, never from the event JSON. The body is already
        bounded; a ``ValueError``, ``KeyError`` or ``TypeError`` raised while reading
        it answers 422.
        """


def host_external_waits(
    app: CayuApp,
    *,
    access_policy: ExternalWaitAccessPolicy,
    context: ExternalWaitContext,
    scopes: Mapping[ExternalWaitScope, Iterable[str]],
    sources: Iterable[ExternalEventSource] = (),
    delivery_path: str | None = None,
    max_request_bytes: int | None = None,
    limits: ExternalWaitLimits | None = None,
    projectors: Mapping[tuple[str, int], ExternalWaitProjector] | None = None,
    redactor: SecretRedactor | None = None,
    request_loop_policies: Iterable[LoopPolicy] = (),
    scheduler_id: str = _DEFAULT_SCHEDULER_ID,
) -> ExternalWaitHosting:
    """Have ``cayu worker`` service external-event waits and ``cayu serve`` deliver them.

    - ``access_policy``: the :class:`ExternalWaitAccessPolicy` of every operation.
    - ``context``: the principal the worker services waits as, and schedules their
      deadlines as. The policy must grant it ``read`` and ``service`` for each
      configured scope and source.
    - ``scopes``: each :class:`ExternalWaitScope` the worker services, with its sources.
    - ``sources``: the :class:`ExternalEventSource` webhooks ``cayu serve`` delivers,
      at ``POST {delivery_path}/{source}``; none by default.
    - ``delivery_path``: a fixed path such as ``/webhooks/waits``, required with
      ``sources``. ``max_request_bytes`` bounds a webhook body; it defaults to the
      payload limit plus 16 KiB and is at most 1 MiB.
    - ``limits``, ``projectors``, ``redactor`` and ``request_loop_policies`` pass
      through to the wait owner and session adapter; ``scheduler_id`` names the
      deadline scheduler.
    """
    if app._external_wait_hosting is not None:
        raise ValueError("This application already hosts its external-event waits.")
    app._external_wait_hosting = ExternalWaitHosting(
        app,
        access_policy=access_policy,
        context=context,
        scopes=scopes,
        sources=sources,
        delivery_path=delivery_path,
        max_request_bytes=max_request_bytes,
        limits=limits,
        projectors=projectors,
        redactor=redactor,
        request_loop_policies=request_loop_policies,
        scheduler_id=scheduler_id,
    )
    return app._external_wait_hosting


class ExternalWaitHosting:
    """The external waits of one application, run by its worker and its web process.

    Create it with ``app.host_external_waits(...)``. The principals it meets:

    - the hosting ``context`` reads, services and schedules every configured wait;
    - an application caller needs ``reserve`` and ``register`` to start a wait, and
      ``service`` to park a session on it (``run_to_wait``/``resume_to_wait``);
    - a webhook's principal, from its :class:`ExternalEventSource`, needs ``deliver``.
    """

    def __init__(
        self,
        app: CayuApp,
        *,
        access_policy: ExternalWaitAccessPolicy,
        context: ExternalWaitContext,
        scopes: Mapping[ExternalWaitScope, Iterable[str]],
        sources: Iterable[ExternalEventSource],
        delivery_path: str | None,
        max_request_bytes: int | None,
        limits: ExternalWaitLimits | None,
        projectors: Mapping[tuple[str, int], ExternalWaitProjector] | None,
        redactor: SecretRedactor | None,
        request_loop_policies: Iterable[LoopPolicy],
        scheduler_id: str,
    ) -> None:
        task_store = app.task_store
        if task_store is None or task_store.supports_task_scheduling is not True:
            raise ValueError(
                "Hosted external waits need a task store that schedules tasks, because "
                "the worker continues their sessions from deadline and service tasks."
            )
        self.app = app
        self.context = _snapshot(context, ExternalWaitContext)
        self.waits = ExternalEventWaits(
            store=app.session_store,
            access_policy=access_policy,
            limits=limits,
            redactor=redactor,
            projectors=projectors,
        )
        self.adapter = SessionExternalWaitAdapter(
            app, self.waits, request_loop_policies=request_loop_policies
        )
        self.host = ExternalWaitHost(self.adapter, context=self.context)
        self.scheduler = TaskStoreWaitScheduler(
            waits=self.waits, task_store=task_store, scheduler_id=scheduler_id
        )
        self.scopes = _scopes(scopes)
        for scope, scope_sources in self.scopes.items():
            for source in scope_sources:
                for action in ("read", "service"):
                    try:
                        self._authorize(scope, source, "discovery", self.context, action)
                    except PermissionError:
                        raise ValueError(
                            f"The external-wait access policy must grant {action!r} to "
                            f"the hosting context for scope "
                            f"{scope.application_scope!r} and source {source!r}."
                        ) from None
        self.sources = _sources(sources, self.scopes)
        if self.sources and delivery_path is None:
            raise ValueError("External event sources need a delivery_path to be served at.")
        if delivery_path is not None and not self.sources:
            raise ValueError("A delivery_path needs at least one external event source.")
        self.delivery_path = None if delivery_path is None else _delivery_path(delivery_path)
        # The endpoints delivery_router() built, so a host can tell whether it serves one.
        self._delivery_endpoints: set[Any] = set()
        self.max_request_bytes = _request_bound(
            max_request_bytes, self.waits.limits.payload_bytes + _DELIVERY_ENVELOPE_BYTES
        )

    def _authorize(
        self,
        scope: ExternalWaitScope,
        source: str,
        correlation_key: str,
        context: ExternalWaitContext,
        action: Any,
    ) -> None:
        self.waits._authorize(
            ExternalCorrelationRequest(scope=scope, source=source, correlation_key=correlation_key),
            context,
            action,
        )

    async def register(
        self, registration: ExternalWaitRegistration, *, context: ExternalWaitContext
    ) -> ExternalWaitSnapshot:
        """Register a wait as ``context`` and schedule its deadline task as the hosting."""
        await self.waits.register(registration, context=context)
        await self.scheduler.schedule(registration, context=self.context)
        return await self.waits.inspect(registration.correlation, context=self.context)

    async def run_to_wait(
        self,
        request: RunRequest,
        registration: ExternalWaitRegistration,
        *,
        context: ExternalWaitContext,
    ) -> SessionExternalWaitReceipt:
        """Run a new session to its wait; an event that already won is serviced next."""
        receipt = await self.adapter.run_to_wait(request, registration, context=context)
        await self._schedule_service(receipt.wait)
        return receipt

    async def resume_to_wait(
        self,
        request: ResumeRequest,
        registration: ExternalWaitRegistration,
        *,
        context: ExternalWaitContext,
    ) -> SessionExternalWaitReceipt:
        """Resume a session to its wait; an event that already won is serviced next."""
        receipt = await self.adapter.resume_to_wait(request, registration, context=context)
        await self._schedule_service(receipt.wait)
        return receipt

    async def deliver(
        self, delivery: ExternalEventDelivery, *, context: ExternalWaitContext
    ) -> ExternalDeliveryReceipt:
        """Deliver an event; a wait it wins for gets a service task the worker runs now.

        A session that hasn't prepared its wait yet gets its task when it parks. If the task
        can't be created the receipt still stands, and so does a parked session's
        receipt: the worker's sweep creates the task instead.
        """
        receipt = await self.waits.deliver(delivery, context=context)
        if receipt.disposition == "accepted":
            request = delivery.correlation.request
            record = await self.waits.store._read_external_wait(
                request.scope, request.correlation_key
            )
            if record is not None and record.correlation == delivery.correlation:
                await self._schedule_service(_view(record))
        return receipt

    async def aclose(self, *, timeout_s: float = 10.0) -> bool:
        """Drain the wait owner's observations; durable waits stay as they are."""
        return await self.waits.aclose(timeout_s=timeout_s)

    async def _schedule_service(self, wait: ExternalWaitSnapshot) -> None:
        # The event is durable either way, so a failure here is logged, not raised.
        try:
            await self._ensure_service(wait)
        except Exception as exc:
            _logger.warning(
                "Scheduling the service of external wait %s failed (%s); the worker's "
                "sweep schedules it.",
                wait.correlation.request.correlation_key,
                type(exc).__name__,
            )

    async def _ensure_service(self, wait: ExternalWaitSnapshot) -> bool:
        """Give a wait that needs service a live attempt; returns whether one was created.

        Every publisher (delivery, parking, the deadline task and the sweep) shares
        one attempt chain per elected outcome, so they converge on the same tasks.
        An attempt that ended while the wait still needs service gets a successor.
        """
        if not _needs_service(wait):
            return False
        assert wait.outcome is not None
        chain = f"outcome:{wait.outcome.selected_at_ms}:{wait.retirement_requested}"
        return await self._ensure_attempt(
            wait.correlation, chain, first_at=_instant(wait.outcome.selected_at_ms)
        )

    async def _ensure_attempt(
        self, correlation: ExternalCorrelation, chain: str, *, first_at: datetime
    ) -> bool:
        task_store = self.scheduler.task_store
        attempts = []
        for attempt in range(1, _MAX_SERVICE_ATTEMPTS + 1):
            task = await task_store.load_task(_attempt_id(correlation, chain, attempt))
            if task is None:
                break
            attempts.append(task)
        if not attempts:
            request: TaskCreate | None = _attempt(correlation, chain, 1, first_at)
        elif _attempt_is_live(attempts[-1], now=self.app._clock()):
            return False
        else:
            request = _successor(attempts[-1])
        if request is None:
            _logger.warning(
                "External wait %s used its whole retry window without being serviced; "
                "service it with ExternalWaitHost.service_once.",
                correlation.request.correlation_key,
            )
            return False
        await task_store.create_task(request)
        return True

    async def sweep(self) -> int:
        """Schedule the tasks hosted waits still need; returns how many it scheduled.

        It only reads a wait that needs nothing, and it never continues a session
        itself: a continuation always runs inside a task, under its lease.
        """
        scheduled = 0
        for scope, scope_sources in self.scopes.items():
            for source in scope_sources:
                cursor = ""
                while True:
                    rows = await self.waits.list(
                        scope=scope,
                        source=source,
                        context=self.context,
                        after=cursor,
                        limit=EXTERNAL_WAIT_PAGE_SIZE,
                    )
                    for row in rows:
                        try:
                            scheduled += await self._sweep_row(row)
                        except Exception as exc:
                            _logger.warning(
                                "Sweeping external wait %s/%s of scope %s generation %d "
                                "failed (%s); the next sweep retries it.",
                                source,
                                row.correlation.request.correlation_key,
                                scope.application_scope,
                                scope.generation,
                                type(exc).__name__,
                            )
                    if len(rows) < EXTERNAL_WAIT_PAGE_SIZE:
                        break
                    cursor = rows[-1].correlation.request.correlation_key
        return scheduled

    async def _sweep_row(self, row: ExternalWaitSnapshot) -> int:
        scheduled = 0
        if row.pending_timer:
            scheduled += await self.scheduler.reconcile_row(row, context=self.context) is not None
        scheduled += await self._ensure_service(row)
        return scheduled

    def _worker_handler(self, handler: TaskHandler) -> TaskHandler:
        async def handle(app: CayuApp, task: Task, worker_id: str):
            if task.type == EXTERNAL_WAIT_TASK_TYPE:
                timer = timer_from_task(task)
                # A scheduler the application wired by hand keeps its own handler.
                if timer.scheduler_id != self.scheduler.scheduler_id:
                    return await handler(app, task, worker_id)
                return await self._handle_deadline(task, timer, worker_id)
            if task.type == EXTERNAL_WAIT_SERVICE_TASK_TYPE:
                return await self._handle_service(task, worker_id)
            return await handler(app, task, worker_id)

        return handle

    async def _handle_deadline(self, task: Task, timer: ExternalWaitTimer, worker_id: str) -> None:
        snapshot = await self.scheduler.notify(timer, context=self.context)
        result: dict[str, object] = timer_result(snapshot)
        deadline = timer.correlation.request.deadline
        if snapshot is not None and snapshot.outcome is None and deadline is not None:
            # The task store's clock ran ahead of the wait store's, which hasn't
            # elected the timeout yet: service the wait a little after the deadline,
            # retrying with backoff.
            retry_at = deadline + _SERVICE_RETRY_BACKOFF
            await self._ensure_attempt(
                snapshot.correlation, "deadline:" + task.id, first_at=retry_at
            )
            result["retry_at"] = retry_at.isoformat()
        elif snapshot is not None and snapshot.pending_handoff:
            result["service"] = await self._service(snapshot)
            # A timeout it couldn't finish servicing, such as behind a live writer,
            # continues in a service task, so the worker knows to come back.
            await self._ensure_service(await self._current(snapshot.correlation))
        await complete_managed_task(self.scheduler.task_store, task, worker_id, result)

    async def _current(self, correlation: ExternalCorrelation) -> ExternalWaitSnapshot:
        request = correlation.request
        record = await self.waits.store._read_external_wait(request.scope, request.correlation_key)
        if record is None or record.correlation != correlation:
            raise ExternalWaitConflict("External wait incarnation is unavailable.")
        return _view(record)

    async def _handle_service(self, task: Task, worker_id: str) -> None:
        try:
            scope = ExternalWaitScope.model_validate(task.input["scope"])
            source = task.input["source"]
            key = task.input["correlation_key"]
            incarnation = task.input["incarnation"]
        except (KeyError, TypeError, ValueError):
            raise ExternalWaitConflict("External wait service task is malformed.") from None
        self._authorize(scope, source, key, self.context, "service")
        record = await self.waits.store._read_external_wait(scope, key)
        if (
            record is None
            or record.correlation.request.source != source
            or record.correlation.incarnation != incarnation
        ):
            await complete_managed_task(
                self.scheduler.task_store, task, worker_id, {"service": "obsolete"}
            )
            return
        try:
            result: dict[str, object] = {"service": await self._service(_view(record))}
            current = await self._current(record.correlation)
        except Exception:
            # The failed attempt keeps its error; its successor keeps the wait from
            # staying parked.
            await self._publish_successor(task)
            raise
        if current.handoff == "pending":
            # A parked session that couldn't continue yet, such as behind a live
            # writer or before the wait store elected the timeout.
            result["retry_at"] = await self._publish_successor(task)
        await complete_managed_task(self.scheduler.task_store, task, worker_id, result)

    async def _publish_successor(self, task: Task) -> str | None:
        successor = _successor(task)
        if successor is None:
            _logger.warning(
                "External wait %s used its whole retry window without being serviced; "
                "service it with ExternalWaitHost.service_once.",
                task.input.get("correlation_key"),
            )
            return None
        await self.scheduler.task_store.create_task(successor)
        assert successor.available_at is not None
        return successor.available_at.isoformat()

    async def _service(self, row: ExternalWaitSnapshot) -> str:
        disposition, failure = await self.host.service_row(row)
        if failure is not None:
            _logger.warning(
                "External wait %s requires reconciliation (%s); its service task retries it.",
                row.correlation.request.correlation_key,
                failure,
            )
            return failure
        return disposition or "nothing_due"

    async def _deliver_received(self, source: str, event: ReceivedExternalEvent):
        scope = event.scope
        # Authorize before reading, so a refused caller learns nothing of the key or
        # of which scopes are served.
        self._authorize(scope, source, event.correlation_key, event.context, "deliver")
        if source not in self.scopes.get(scope, ()):
            return 404, {"code": "unknown_correlation"}
        record = await self.waits.store._read_external_wait(scope, event.correlation_key)
        if record is None or record.correlation.request.source != source:
            return 404, {"code": "unknown_correlation"}
        receipt = await self.deliver(
            ExternalEventDelivery(
                correlation=record.correlation,
                delivery_id=event.delivery_id,
                payload_json=event.payload_json,
            ),
            context=event.context,
        )
        return 202, receipt.model_dump(mode="json")

    def serves_delivery(self, server: Any) -> bool:
        """Whether ``server``'s routes include a route ``delivery_router()`` built."""
        return any(
            getattr(route, "endpoint", None) in self._delivery_endpoints
            for route in getattr(server, "routes", ())
        )

    def delivery_router(self) -> APIRouter:
        """The webhook route of the configured sources, at ``POST {delivery_path}/{source}``.

        ``cayu serve`` mounts it; any other ASGI host can include it. It is outside
        ``cayu serve --auth``: each source authenticates its own callers. It answers
        202 with the delivery receipt; 401 for an unknown source or a request the
        source refuses; 403 when the access policy refuses every scope of the source;
        404 for an unknown correlation; 409 for a conflicting retry or a key in two
        scopes; 413 for a body over ``max_request_bytes``, before the source reads
        it; 422 for a body the source can't read or a malformed or oversized
        payload; 429 with ``Retry-After`` when the wait owner is at capacity (there
        is no per-sender rate limit); and 503 when the acknowledgement is unavailable.
        """
        from cayu.server._external_wait_delivery import external_wait_delivery_router

        if self.delivery_path is None:
            raise ValueError("This hosting serves no external event sources.")
        return external_wait_delivery_router(self, self.delivery_path)


def claims_hosted_tasks(query: TaskQuery | None) -> bool:
    """Whether a worker with this ``query`` claims the hosted deadline and service tasks."""
    if query is None:
        return True
    narrowed = query.model_dump(
        include={
            "q",
            "status",
            "session_id",
            "parent_task_id",
            "assigned_agent_name",
            "has_work_contract",
        }
    )
    return query.type in {None, EXTERNAL_WAIT_TASK_TYPE, EXTERNAL_WAIT_SERVICE_TASK_TYPE} and all(
        value is None for value in narrowed.values()
    )


def _needs_service(wait: ExternalWaitSnapshot) -> bool:
    """A wait whose outcome is elected and whose handoff the runtime still owes.

    A wait with no execution preparation owes nothing yet (``pending_handoff`` is
    false): its session hasn't parked, and parking schedules its service. A retained
    preparation whose binding wasn't published does owe one, and servicing it
    reconciles the binding first.
    """
    return wait.pending_handoff and wait.outcome is not None


def _attempt_id(correlation: ExternalCorrelation, chain: str, attempt: int) -> str:
    request = correlation.request
    identity = canonical_durable_json_bytes(
        {
            "scope": request.scope.model_dump(mode="json"),
            "source": request.source,
            "correlation_key": request.correlation_key,
            "incarnation": correlation.incarnation,
            "chain": chain,
        },
        "external wait service",
    )
    base = "cayu-external-wait-service-" + sha256(identity).hexdigest()
    return base if attempt == 1 else f"{base}.{attempt}"


def _attempt(
    correlation: ExternalCorrelation,
    chain: str,
    attempt: int,
    available_at: datetime,
    retry_until: datetime | None = None,
) -> TaskCreate:
    # A managed one-shot task, fixed by its chain and predecessor so every publisher
    # builds the same one and a replay converges on it.
    request = correlation.request
    return TaskCreate(
        task_id=_attempt_id(correlation, chain, attempt),
        type=EXTERNAL_WAIT_SERVICE_TASK_TYPE,
        available_at=available_at,
        schedule_policy=TaskSchedulePolicy(),
        input={
            "scope": request.scope.model_dump(mode="json"),
            "source": request.source,
            "correlation_key": request.correlation_key,
            "incarnation": correlation.incarnation,
            "chain": chain,
            "attempt": attempt,
            # Fixed by the first attempt's run; ``None`` on the first attempt itself.
            "retry_until": None if retry_until is None else retry_until.isoformat(),
        },
    )


def _successor(previous: Task) -> TaskCreate | None:
    """The attempt after ``previous``, timed from when it ran, or ``None`` once spent.

    The retry window starts when the first attempt ran, so attempts that only run
    long after their wait was elected still get their retries.
    """
    try:
        chain = previous.input["chain"]
        attempt = previous.input["attempt"]
        raw_until = previous.input["retry_until"]
        retry_until = None if raw_until is None else datetime.fromisoformat(raw_until)
        if type(attempt) is not int or not 1 <= attempt <= _MAX_SERVICE_ATTEMPTS:
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise ExternalWaitConflict("External wait service task is malformed.") from None
    moments = [
        moment
        for moment in (previous.available_at, previous.started_at, previous.created_at)
        if moment is not None
    ]
    if retry_until is None:
        retry_until = max(moments) + EXTERNAL_WAIT_SERVICE_RETRY_WINDOW
    available_at = max(moments) + min(
        _SERVICE_RETRY_BACKOFF * 2 ** (attempt - 1), _MAX_SERVICE_RETRY_BACKOFF
    )
    if available_at > retry_until or attempt >= _MAX_SERVICE_ATTEMPTS:
        return None
    return TaskCreate(
        task_id=previous.id.split(".", 1)[0] + f".{attempt + 1}",
        type=EXTERNAL_WAIT_SERVICE_TASK_TYPE,
        available_at=available_at,
        schedule_policy=TaskSchedulePolicy(),
        input={
            **previous.input,
            "chain": chain,
            "attempt": attempt + 1,
            "retry_until": retry_until.isoformat(),
        },
    )


def _instant(epoch_ms: int) -> datetime:
    return datetime.fromtimestamp(epoch_ms / 1000, UTC)


def _scopes(
    scopes: Mapping[ExternalWaitScope, Iterable[str]],
) -> dict[ExternalWaitScope, tuple[str, ...]]:
    if not isinstance(scopes, Mapping) or not scopes:
        raise ValueError("Hosted external waits need at least one scope and its sources.")
    result: dict[ExternalWaitScope, tuple[str, ...]] = {}
    for scope, scope_sources in scopes.items():
        scope = _snapshot(scope, ExternalWaitScope)
        names = tuple(scope_sources) if not isinstance(scope_sources, str) else ()
        if (
            not names
            or any(type(name) is not str or not name.strip() for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError(
                f"Scope {scope.application_scope!r} needs distinct, nonblank source names."
            )
        result[scope] = names
    return result


def _sources(
    sources: Iterable[ExternalEventSource], scopes: Mapping[ExternalWaitScope, tuple[str, ...]]
) -> dict[str, ExternalEventSource]:
    configured = {name for names in scopes.values() for name in names}
    result: dict[str, ExternalEventSource] = {}
    for endpoint in sources:
        if not isinstance(endpoint, ExternalEventSource):
            raise TypeError("External event sources must implement ExternalEventSource.")
        name = getattr(endpoint, "source", None)
        if type(name) is not str or name not in configured:
            raise ValueError(
                f"External event source {name!r} must name a source of a configured scope."
            )
        if name in result:
            raise ValueError(f"External event source {name!r} is configured twice.")
        result[name] = endpoint
    return result


def _delivery_path(path: str) -> str:
    segments = path.split("/") if type(path) is str else []
    if (
        len(segments) < 2
        or segments[0]
        or any(
            not segment or segment in {".", ".."} or not set(segment) <= _PATH_SEGMENT_CHARACTERS
            for segment in segments[1:]
        )
    ):
        raise ValueError(
            "delivery_path must be a fixed path such as /webhooks/waits: segments of "
            "letters, digits, '-', '.', '_' or '~', without a trailing slash."
        )
    return path


def _request_bound(value: int | None, default: int) -> int:
    if value is None:
        return default
    if type(value) is not int or not 1 <= value <= _MAX_DELIVERY_REQUEST_BYTES:
        raise ValueError("max_request_bytes must be an int from 1 to 1048576.")
    return value
