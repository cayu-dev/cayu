"""When an agent next has scheduled work, and which runs are parked on an outside event."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta, timezone
from uuid import uuid4

import pytest
from tests.external_wait_support import CONTEXT, Policy, registration, reservation, stores

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu.cloud_reporting import read_cloud_activity, read_cloud_contract, read_cloud_schedule
from cayu.evals import ScriptedModelProvider
from cayu.external_wait_host import ExternalWaitHost
from cayu.external_waits import ExternalEventWaits
from cayu.providers import ModelStreamEvent
from cayu.session_external_waits import SessionExternalWaitAdapter
from cayu.sessions.authority import SessionRunFenced
from cayu.sessions.external_waits import (
    ExternalCorrelationRequest,
    ExternalEventDelivery,
    ExternalWaitScope,
    ExternalWaitUnavailable,
)
from cayu.sessions.memory import InMemorySessionStore
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.scheduling import TaskSchedulePolicy

T0 = datetime(2026, 10, 28, 9, 0, tzinfo=UTC)
PAYLOAD = '{"secret_payload":"render-payload-canary"}'


def _turn() -> list[ModelStreamEvent]:
    return [ModelStreamEvent.text_delta("Submitted."), ModelStreamEvent.completed({})]


def test_the_contract_advertises_the_schedule_and_external_event_waits() -> None:
    async def scenario():
        app = CayuApp(enable_logging=False)
        try:
            return await read_cloud_contract(app)
        finally:
            await app.aclose()

    features = asyncio.run(scenario()).features
    assert "schedule" in features
    assert "external_event_waits" in features


def test_an_agent_without_a_task_store_reports_no_task_schedule() -> None:
    async def scenario():
        app = CayuApp(enable_logging=False)
        try:
            return await read_cloud_schedule(app)
        finally:
            await app.aclose()

    schedule = asyncio.run(scenario())
    assert schedule.external_waits is not None
    assert schedule.model_dump(mode="json") == {
        "contract": "cayu.cloud-agent/v1",
        "tasks": None,
        "external_waits": {
            "as_of": schedule.model_dump(mode="json")["external_waits"]["as_of"],
            "next_deadline_at": None,
            "awaiting_service": 0,
        },
    }
    assert schedule.external_waits.as_of.utcoffset() == timedelta(0)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_the_schedule_reports_every_delayed_task_in_the_store(tmp_path, backend) -> None:
    async def scenario():
        clock = {"now": T0}
        if backend == "memory":
            tasks = InMemoryTaskStore(clock=lambda: clock["now"])
        else:
            tasks = SQLiteTaskStore(tmp_path / "tasks.db", clock=lambda: clock["now"])
        app = CayuApp(task_store=tasks, enable_logging=False)
        try:
            empty = await read_cloud_schedule(app)
            # Different task types: the summary counts work whichever worker serves it.
            for task_id, task_type, available_at, expires_at in (
                ("report", "report", T0 + timedelta(hours=3), None),
                ("follow-up", "email", T0 + timedelta(hours=1), T0 + timedelta(hours=2)),
                ("now", "email", None, None),
            ):
                await app.create_task(
                    TaskCreate(
                        task_id=task_id,
                        type=task_type,
                        available_at=available_at,
                        schedule_policy=(
                            None
                            if available_at is None
                            else TaskSchedulePolicy(expires_at=expires_at)
                        ),
                    )
                )
            scheduled = await read_cloud_schedule(app)
            clock["now"] = T0 + timedelta(hours=2, minutes=30)
            later = await read_cloud_schedule(app)
            return empty, scheduled, later
        finally:
            await app.aclose()
            if backend == "sqlite":
                await tasks.close()

    empty, scheduled, later = asyncio.run(scenario())
    assert empty.tasks is not None
    assert empty.tasks.model_dump() == {
        "as_of": T0,
        "next_available_at": None,
        "next_expiry_at": None,
        "maintenance_required": False,
        "scheduled": 0,
        "due": 0,
    }
    assert scheduled.model_dump(mode="json")["tasks"] == {
        "as_of": "2026-10-28T09:00:00Z",
        "next_available_at": "2026-10-28T10:00:00Z",
        "next_expiry_at": "2026-10-28T11:00:00Z",
        "maintenance_required": False,
        "scheduled": 2,
        "due": 1,
    }
    # The follow-up expired unclaimed: it stays pending until a worker's sweep, due now.
    assert later.tasks is not None
    assert later.tasks.next_available_at == T0 + timedelta(hours=3)
    assert later.tasks.next_expiry_at is None
    assert later.tasks.maintenance_required is True
    assert (later.tasks.scheduled, later.tasks.due) == (1, 2)


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_the_schedule_reports_the_earliest_open_external_wait_deadline(
    tmp_path, request, backend
) -> None:
    async def scenario():
        now = datetime.now(UTC)
        async with stores(backend, tmp_path, request, [now]) as (store, _reopen):
            waits = ExternalEventWaits(store=store, access_policy=Policy())
            app = CayuApp(session_store=store, enable_logging=False)
            scope = ExternalWaitScope(application_scope=uuid4().hex, generation=1)

            async def wait(key: str, deadline: datetime | None, *, register: bool = True):
                correlation = await waits.reserve_correlation(
                    ExternalCorrelationRequest(
                        scope=scope, source="renderer", correlation_key=key, deadline=deadline
                    ),
                    context=CONTEXT,
                )
                if register:
                    await waits.register(registration(correlation), context=CONTEXT)
                return correlation

            try:
                # Later as text than the UTC deadline below, but earlier as an instant.
                ahead = timezone(timedelta(hours=5))
                earliest = (now + timedelta(hours=2)).astimezone(ahead)
                await wait("later", now + timedelta(hours=3))
                await wait("earliest", earliest)
                await wait("event-only", None)
                # Neither an unregistered reservation nor a decided wait has a deadline to keep.
                await wait("unregistered", now + timedelta(minutes=5), register=False)
                answered = await wait("answered", now + timedelta(minutes=10))
                await waits.deliver(
                    ExternalEventDelivery(
                        correlation=answered, delivery_id="done", payload_json=PAYLOAD
                    ),
                    context=CONTEXT,
                )
                return earliest, await read_cloud_schedule(app)
            finally:
                await app.aclose()
                await waits.aclose()

    earliest, schedule = asyncio.run(scenario())
    assert schedule.external_waits is not None
    deadline = schedule.external_waits.next_deadline_at
    assert deadline == earliest and deadline is not None
    assert deadline.utcoffset() == timedelta(0)


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_a_run_parked_on_an_external_event_reports_it_without_content(
    tmp_path, request, backend
) -> None:
    async def scenario():
        async with stores(backend, tmp_path, request, [datetime.now(UTC)]) as (store, _reopen):
            provider = ScriptedModelProvider([_turn(), _turn()])
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="renderer-agent", model="model"))
            waits = ExternalEventWaits(store=store, access_policy=Policy())
            adapter = SessionExternalWaitAdapter(app, waits)
            # Far ahead, so this wait never becomes another test's earliest deadline.
            deadline = (datetime.now(UTC) + timedelta(days=300)).astimezone(
                timezone(timedelta(hours=2))
            )
            scope = ExternalWaitScope(application_scope=uuid4().hex, generation=1)
            correlation = await waits.reserve_correlation(
                ExternalCorrelationRequest(
                    scope=scope,
                    source="renderer",
                    correlation_key="video-key-canary",
                    deadline=deadline,
                ),
                context=CONTEXT,
            )
            registered = registration(correlation)
            await waits.register(registered, context=CONTEXT)
            session_id = "parked-" + uuid4().hex
            try:
                await adapter.run_to_wait(
                    RunRequest(
                        agent_name="renderer-agent",
                        session_id=session_id,
                        messages=[Message.text("user", "Render the video.")],
                    ),
                    registered,
                    context=CONTEXT,
                )
                parked = await read_cloud_activity(app)
                parked_schedule = await read_cloud_schedule(app)
                await waits.deliver(
                    ExternalEventDelivery(
                        correlation=correlation, delivery_id="rendered", payload_json=PAYLOAD
                    ),
                    context=CONTEXT,
                )
                delivered = await read_cloud_activity(app)
                delivered_schedule = await read_cloud_schedule(app)
                await adapter.service_wait(registered, context=CONTEXT)
                serviced = await read_cloud_activity(app)
                serviced_schedule = await read_cloud_schedule(app)
                schedules = (parked_schedule, delivered_schedule, serviced_schedule)
                return session_id, scope, deadline, parked, delivered, serviced, schedules
            finally:
                await app.aclose()
                await waits.aclose()

    session_id, scope, deadline, parked, delivered, serviced, schedules = asyncio.run(scenario())
    # A delivered event the application has not serviced yet needs it now.
    observed = [schedule.external_waits for schedule in schedules]
    assert all(waits is not None for waits in observed)
    assert [waits.awaiting_service for waits in observed if waits is not None] == [0, 1, 0]
    (run,) = [run for run in parked.runs if run.session_id == session_id]
    assert run.status == "interrupted" and run.ended_at is not None
    (item,) = run.waiting_for
    assert item.model_dump(mode="json") == {
        "kind": "external_event",
        "attention_id": None,
        "tool_name": None,
        "requested_at": item.model_dump(mode="json")["requested_at"],
        "child_session_id": None,
        "child_run_id": None,
        "external_wait": {
            "scope": scope.application_scope,
            "generation": 1,
            "source": "renderer",
            "deadline": deadline.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        },
        "wake_at": None,
    }
    assert item.requested_at == run.ended_at
    for page in (parked, delivered):
        dumped = page.model_dump_json()
        assert "video-key-canary" not in dumped
        assert "render-payload-canary" not in dumped
    # An event that arrived but has not been serviced leaves the run parked.
    (still,) = [run for run in delivered.runs if run.session_id == session_id]
    assert [item.kind for item in still.waiting_for] == ["external_event"]
    (done,) = [run for run in serviced.runs if run.session_id == session_id]
    assert done.status == "completed" and done.waiting_for == ()


def test_a_store_without_external_waits_reports_none() -> None:
    class PlainStore(InMemorySessionStore):
        def supports_external_waits(self) -> bool:
            return False

    async def scenario():
        app = CayuApp(session_store=PlainStore(), enable_logging=False)
        try:
            return await read_cloud_schedule(app)
        finally:
            await app.aclose()

    assert asyncio.run(scenario()).external_waits is None


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_a_run_that_stops_before_it_parks_is_not_reported_as_parked(
    tmp_path, request, backend, monkeypatch
) -> None:
    async def scenario():
        async with stores(backend, tmp_path, request, [datetime.now(UTC)]) as (store, _reopen):
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(ScriptedModelProvider([_turn()]), default=True)
            app.register_agent(AgentSpec(name="renderer-agent", model="model"))
            waits = ExternalEventWaits(store=store, access_policy=Policy())
            correlation = await waits.reserve_correlation(reservation(), context=CONTEXT)
            registered = registration(correlation)
            await waits.register(registered, context=CONTEXT)
            session_id = "unparked-" + uuid4().hex

            async def lost(*args, **kwargs):
                raise RuntimeError("The process stopped before the session parked.")

            # Bound to its continuation, but the ticket never reaches WAITING.
            monkeypatch.setattr(type(store), "mark_continuation_waiting", lost)
            try:
                with pytest.raises(ExternalWaitUnavailable):
                    await SessionExternalWaitAdapter(app, waits).run_to_wait(
                        RunRequest(
                            agent_name="renderer-agent",
                            session_id=session_id,
                            messages=[Message.text("user", "Render the video.")],
                        ),
                        registered,
                        context=CONTEXT,
                    )
                record = await store._read_external_wait(
                    correlation.request.scope, correlation.request.correlation_key
                )
                return session_id, record, await read_cloud_activity(app)
            finally:
                await app.aclose()
                await waits.aclose()

    session_id, record, page = asyncio.run(scenario())
    assert record is not None and record.handoff == "pending"
    (run,) = [run for run in page.runs if run.session_id == session_id]
    assert run.ended_at is not None
    assert run.waiting_for == ()


def test_task_wake_times_are_reported_when_the_store_cannot_count_tasks(monkeypatch) -> None:
    async def uncounted(filters=None):
        raise NotImplementedError

    async def scenario():
        tasks = InMemoryTaskStore(clock=lambda: T0)
        monkeypatch.setattr(tasks, "aggregate_operational_snapshot", uncounted)
        app = CayuApp(task_store=tasks, enable_logging=False)
        try:
            await app.create_task(
                TaskCreate(
                    task_id="later",
                    type="report",
                    available_at=T0 + timedelta(hours=1),
                    schedule_policy=TaskSchedulePolicy(),
                )
            )
            return await read_cloud_schedule(app)
        finally:
            await app.aclose()

    schedule = asyncio.run(scenario())
    assert schedule.tasks is not None
    assert schedule.tasks.next_available_at == T0 + timedelta(hours=1)
    assert (schedule.tasks.scheduled, schedule.tasks.due) == (None, None)


async def _parked_app(store, provider_turns: int):
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(ScriptedModelProvider([_turn()] * provider_turns), default=True)
    app.register_agent(AgentSpec(name="renderer-agent", model="model"))
    waits = ExternalEventWaits(store=store, access_policy=Policy())
    adapter = SessionExternalWaitAdapter(app, waits)
    correlation = await waits.reserve_correlation(reservation(), context=CONTEXT)
    registered = registration(correlation)
    await waits.register(registered, context=CONTEXT)
    session_id = "parked-" + uuid4().hex
    await adapter.run_to_wait(
        RunRequest(
            agent_name="renderer-agent",
            session_id=session_id,
            messages=[Message.text("user", "Render the video.")],
        ),
        registered,
        context=CONTEXT,
    )
    return app, waits, adapter, correlation, registered, session_id


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_an_interrupted_retirement_still_needs_its_application(
    tmp_path, request, backend, monkeypatch
) -> None:
    async def scenario():
        async with stores(backend, tmp_path, request, [datetime.now(UTC)]) as (store, _reopen):
            app, waits, adapter, correlation, registered, _ = await _parked_app(store, 1)
            try:
                await waits.deliver(
                    ExternalEventDelivery(
                        correlation=correlation, delivery_id="rendered", payload_json=PAYLOAD
                    ),
                    context=CONTEXT,
                )
                baseline = await read_cloud_schedule(app)
                mutate = waits._mutate

                async def lose_acknowledgement(command):
                    result = await mutate(command)
                    if command.kind == "settle":
                        raise RuntimeError("retirement acknowledgement lost")
                    return result

                with monkeypatch.context() as patch:
                    patch.setattr(waits, "_mutate", lose_acknowledgement)
                    with pytest.raises(RuntimeError, match="acknowledgement lost"):
                        await adapter.retire_execution(
                            registered, operation_key="retire", context=CONTEXT
                        )
                interrupted = await waits.inspect(correlation, context=CONTEXT)
                during = await read_cloud_schedule(app)
                # Servicing finishes the retirement, which releases the application.
                host = ExternalWaitHost(adapter, context=CONTEXT)
                await host.service_once(scope=correlation.request.scope, source="renderer")
                finished = await adapter.retire_execution(
                    registered, operation_key="retire", context=CONTEXT
                )
                after = await read_cloud_schedule(app)
                return baseline, interrupted, during, finished, after
            finally:
                await app.aclose()
                await waits.aclose()

    baseline, interrupted, during, finished, after = asyncio.run(scenario())
    assert interrupted.handoff == "excluded" and interrupted.pending_handoff
    assert not finished.pending_handoff
    counts = [
        schedule.external_waits.awaiting_service
        for schedule in (baseline, during, after)
        if schedule.external_waits is not None
    ]
    assert counts == [counts[0], counts[0], counts[0] - 1]


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("change", ["deleted", "replaced", "still_there"])
def test_a_session_gone_during_the_parked_read_leaves_the_page(
    tmp_path, request, backend, change, monkeypatch
) -> None:
    async def scenario():
        async with stores(backend, tmp_path, request, [datetime.now(UTC)]) as (store, _reopen):
            app, waits, _adapter, _correlation, _registered, session_id = await _parked_app(
                store, 1
            )
            store_type = type(store)
            load = store_type.load
            load_ticket = store_type.load_continuation_ticket

            async def gone_while_reading(self, wanted, *args, **kwargs):
                # Other tests' sessions share a PostgreSQL database; read them as usual.
                if wanted != session_id:
                    return await load_ticket(self, wanted, *args, **kwargs)
                if change == "deleted":
                    patch.setattr(store_type, "load", vanished)
                    raise KeyError(session_id)
                if change == "replaced":
                    patch.setattr(store_type, "load", reused)
                    raise SessionRunFenced("The session ID was reused.")
                raise KeyError("an unrelated store failure")

            async def vanished(self, wanted, *args, **kwargs):
                return None if wanted == session_id else await load(self, wanted, *args, **kwargs)

            async def reused(self, wanted, *args, **kwargs):
                current = await load(self, wanted, *args, **kwargs)
                if wanted != session_id or current is None:
                    return current
                return current.model_copy(update={"instance_id": "replacement-" + uuid4().hex})

            try:
                with monkeypatch.context() as patch:
                    patch.setattr(store_type, "load_continuation_ticket", gone_while_reading)
                    if change == "still_there":
                        with pytest.raises(KeyError, match="unrelated"):
                            await read_cloud_activity(app)
                        return session_id, None
                    return session_id, await read_cloud_activity(app)
            finally:
                await app.aclose()
                await waits.aclose()

    session_id, page = asyncio.run(scenario())
    if change != "still_there":
        assert page is not None
        assert session_id not in [run.session_id for run in page.runs]
