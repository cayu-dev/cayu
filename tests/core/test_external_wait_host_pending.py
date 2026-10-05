"""Pending native ownership must not stop an explicitly driven host loop."""

import asyncio
import logging
import time
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from tests.external_wait_support import CONTEXT, Policy, registration, reservation, stores

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu.evals.testing import ScriptedModelProvider
from cayu.external_wait_host import ExternalWaitHost
from cayu.external_waits import ExternalEventWaits
from cayu.providers.base import ModelStreamEvent
from cayu.runtime._external_execution_to_wait import _ExternalExecutionToWait
from cayu.runtime._session_continuation_owner import SessionContinuationOwner
from cayu.session_external_waits import SessionExternalWaitAdapter
from cayu.sessions._session_continuation import ContinuationConflict, ContinuationUnavailable
from cayu.sessions.base import InMemorySessionStore
from cayu.sessions.external_waits import ExternalEventDelivery, ExternalWaitConflict


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("before_ticket", [False, True])
def test_host_continues_past_owned_work(backend, before_ticket, tmp_path, request, monkeypatch):
    async def scenario():
        async with stores(backend, tmp_path, request, [datetime.now(UTC)]) as (store, reopen):
            entered, release, ready, stop = (asyncio.Event() for _ in range(4))

            class Provider(ScriptedModelProvider):
                dispatches = 0

                async def stream(self, request):
                    self.dispatches += 1
                    if not before_ticket and self.dispatches == 3:
                        entered.set()
                        await release.wait()
                    async for event in super().stream(request):
                        yield event

            provider = Provider(
                [[ModelStreamEvent.completed({"finish_reason": "stop"})] for _ in range(4)]
            )
            waits = ExternalEventWaits(store=store, access_policy=Policy())
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="root", model="model"))
            adapter = SessionExternalWaitAdapter(app, waits)
            first_request = reservation().model_copy(update={"correlation_key": "a-pending"})
            second_request = first_request.model_copy(update={"correlation_key": "b-ready"})
            registrations = []
            session_ids = ["host-" + uuid4().hex for _ in range(2)]
            for item in (first_request, second_request):
                correlation = await waits.reserve_correlation(item, context=CONTEXT)
                registered = registration(correlation)
                await waits.register(registered, context=CONTEXT)
                registrations.append(registered)

            def run_request(index):
                return RunRequest(
                    agent_name="root",
                    session_id=session_ids[index],
                    messages=[Message.text("user", "submit")],
                )

            # Exercise the real ten-second observation boundary while the
            # provider remains owned and blocked; do not fabricate its exception.
            original_prepare = _ExternalExecutionToWait.prepare

            async def held_prepare(boundary, invocation):
                if before_ticket and invocation.binding.session_id == session_ids[0]:
                    entered.set()
                    await release.wait()
                return await original_prepare(boundary, invocation)

            monkeypatch.setattr(_ExternalExecutionToWait, "prepare", held_prepare)
            running = None
            polling = None
            pages = []

            class Host(ExternalWaitHost):
                async def service_once(self, **kwargs):
                    page = await super().service_once(**kwargs)
                    pages.append(page)
                    if "b-ready" in page.settled:
                        ready.set()
                        await release.wait()
                    if "a-pending" in page.settled:
                        stop.set()
                    return page

            try:
                if before_ticket:
                    await adapter.run_to_wait(run_request(1), registrations[1], context=CONTEXT)
                    running = asyncio.create_task(
                        adapter.run_to_wait(run_request(0), registrations[0], context=CONTEXT)
                    )
                    await asyncio.wait_for(entered.wait(), 10)
                    await waits.cancel(
                        registrations[0].correlation, operation_key="cancel", context=CONTEXT
                    )
                else:
                    for index, registered in enumerate(registrations):
                        await adapter.run_to_wait(run_request(index), registered, context=CONTEXT)
                for registered in registrations[int(before_ticket) :]:
                    await waits.deliver(
                        ExternalEventDelivery(
                            correlation=registered.correlation,
                            delivery_id="done",
                            payload_json="{}",
                        ),
                        context=CONTEXT,
                    )
                host = Host(adapter, context=CONTEXT)
                polling = asyncio.create_task(
                    host.run(
                        scope=first_request.scope, source="renderer", stop=stop, interval_s=0.01
                    )
                )
                await asyncio.wait_for(ready.wait(), 30)
                assert entered.is_set()
                assert not polling.done()
                assert any("a-pending" in page.pending for page in pages)
                assert (
                    await waits.inspect(registrations[0].correlation, context=CONTEXT)
                ).pending_handoff
                assert provider.dispatches == (2 if before_ticket else 4), [
                    event.model_dump(mode="json")
                    for session_id in session_ids
                    for event in await store.load_events(session_id)
                    if "error" in str(event.type) or "failed" in str(event.type)
                ]
                if before_ticket:
                    running.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await running
                    assert running.cancelled() and running.cancelling() == 1
                release.set()
                await asyncio.wait_for(polling, 20)
                for registered in registrations:
                    assert not (
                        await waits.inspect(registered.correlation, context=CONTEXT)
                    ).pending_handoff
                restored_waits = ExternalEventWaits(store=reopen(), access_policy=Policy())
                restored_app = CayuApp(session_store=restored_waits.store, enable_logging=False)
                restored_app.register_provider(provider, default=True)
                restored_app.register_agent(AgentSpec(name="root", model="model"))
                restored_host = ExternalWaitHost(
                    SessionExternalWaitAdapter(restored_app, restored_waits), context=CONTEXT
                )
                assert not (
                    await restored_host.service_once(scope=first_request.scope, source="renderer")
                ).pending
                assert provider.dispatches == (2 if before_ticket else 4)
                await restored_app.aclose()
                await restored_waits.aclose()
            finally:
                release.set()
                stop.set()
                for task in (polling, running):
                    if task is not None and not task.done():
                        task.cancel()
                await asyncio.gather(
                    *(task for task in (polling, running) if task is not None),
                    return_exceptions=True,
                )
                await app.aclose()
                await waits.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "failure_type", [PermissionError, ContinuationConflict, ExternalWaitConflict]
)
def test_host_preserves_receiving_authority_failures(failure_type, monkeypatch):
    async def scenario():
        waits = ExternalEventWaits(store=InMemorySessionStore(), access_policy=Policy())
        app = CayuApp(session_store=waits.store, enable_logging=False)
        provider = ScriptedModelProvider([ModelStreamEvent.completed({"finish_reason": "stop"})])
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="root", model="model"))
        adapter = SessionExternalWaitAdapter(app, waits)
        correlation = await waits.reserve_correlation(reservation(), context=CONTEXT)
        registered = registration(correlation)
        await waits.register(registered, context=CONTEXT)
        try:
            await adapter.run_to_wait(
                RunRequest(agent_name="root", messages=[Message.text("user", "submit")]),
                registered,
                context=CONTEXT,
            )
            await waits.deliver(
                ExternalEventDelivery(
                    correlation=correlation, delivery_id="done", payload_json="{}"
                ),
                context=CONTEXT,
            )
            failure = failure_type("Receiving authority rejected")

            async def reject(*args, **kwargs):
                raise failure

            monkeypatch.setattr(SessionContinuationOwner, "service", reject)
            host = ExternalWaitHost(adapter, context=CONTEXT)
            if failure_type is PermissionError:
                with pytest.raises(failure_type) as caught:
                    await host.run(
                        scope=correlation.request.scope, source="renderer", stop=asyncio.Event()
                    )
                assert caught.value is failure
            else:
                page = await host.service_once(scope=correlation.request.scope, source="renderer")
                assert page.pending == (correlation.request.correlation_key,)
                assert len(page.failures) == 1
                assert page.failures[0].correlation_key == correlation.request.correlation_key
                assert page.failures[0].kind == "conflict"
            assert (await waits.inspect(correlation, context=CONTEXT)).pending_handoff
            assert len(provider.requests) == 1
        finally:
            await app.aclose()
            await waits.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_cancelled_live_writer_does_not_stop_later_waits(backend, tmp_path, request):
    from cayu.sessions.external_waits import ExternalWaitUnavailable

    async def scenario():
        async with stores(backend, tmp_path, request, [datetime.now(UTC)]) as (store, reopen):
            entered, release = asyncio.Event(), asyncio.Event()

            class Provider(ScriptedModelProvider):
                dispatches = 0

                @property
                def execution_profile_identity(self):
                    from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity

                    return ExecutionProfileBehaviorIdentity(
                        name="tests:external-host-live-provider",
                        behavior_version="1",
                        implementation_version="1",
                    )

                async def stream(self, request):
                    self.dispatches += 1
                    if self.dispatches == 2:
                        entered.set()
                        await release.wait()
                    async for event in super().stream(request):
                        yield event

            provider = Provider(
                [[ModelStreamEvent.completed({"finish_reason": "stop"})] for _ in range(3)]
            )
            waits = ExternalEventWaits(store=store, access_policy=Policy())
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="root", model="model"))
            adapter = SessionExternalWaitAdapter(app, waits)
            first = reservation().model_copy(update={"correlation_key": "a-live"})
            registrations = []
            for key in ("a-live", "b-ready"):
                correlation = await waits.reserve_correlation(
                    first.model_copy(update={"correlation_key": key}), context=CONTEXT
                )
                registered = registration(correlation)
                await waits.register(registered, context=CONTEXT)
                registrations.append(registered)
            live, ready = registrations
            await adapter.run_to_wait(
                RunRequest(agent_name="root", messages=[Message.text("user", "ready")]),
                ready,
                context=CONTEXT,
            )
            session_id = "live-" + uuid4().hex
            running = asyncio.create_task(
                adapter.run_to_wait(
                    RunRequest(
                        agent_name="root",
                        session_id=session_id,
                        messages=[Message.text("user", "hold")],
                    ),
                    live,
                    context=CONTEXT,
                )
            )
            await asyncio.wait_for(entered.wait(), 10)
            epoch = (await store.load(session_id)).run_epoch
            await waits.cancel(live.correlation, operation_key="cancel", context=CONTEXT)
            await waits.deliver(
                ExternalEventDelivery(
                    correlation=ready.correlation, delivery_id="done", payload_json="{}"
                ),
                context=CONTEXT,
            )
            other_store = reopen()
            other_waits = ExternalEventWaits(store=other_store, access_policy=Policy())
            other_app = CayuApp(session_store=other_store, enable_logging=False)
            other_app.register_provider(provider, default=True)
            other_app.register_agent(AgentSpec(name="root", model="model"))
            other = SessionExternalWaitAdapter(other_app, other_waits)
            stop = asyncio.Event()
            pages = []

            class Host(ExternalWaitHost):
                async def service_once(self, **kwargs):
                    page = await super().service_once(**kwargs)
                    pages.append(page)
                    stop.set()
                    return page

            try:
                await Host(other, context=CONTEXT).run(
                    scope=first.scope, source="renderer", stop=stop
                )
                page = pages[0]
                assert page.pending == ("a-live",) and page.settled == ("b-ready",), (
                    page,
                    provider.dispatches,
                )
                assert [(error.correlation_key, error.kind) for error in page.failures] == [
                    ("a-live", "conflict")
                ]
                assert (await store.load(session_id)).run_epoch == epoch
                assert (await store.inspect_session_execution(session_id)).state == "executing"
                assert provider.dispatches == 3
            finally:
                release.set()
                with pytest.raises(ExternalWaitUnavailable):
                    await running
                await app.aclose()
                await other_app.aclose()
                await waits.aclose()
                await other_waits.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_host_drains_pages_before_polling_interval(backend, tmp_path, request):
    from datetime import timedelta

    async def scenario():
        clock = [datetime.now(UTC)]
        async with stores(backend, tmp_path, request, clock) as (store, _):
            waits = ExternalEventWaits(store=store, access_policy=Policy())
            initial = reservation(deadline=clock[0] + timedelta(seconds=1))
            correlations = []
            for index in range(33):
                correlation = await waits.reserve_correlation(
                    initial.model_copy(update={"correlation_key": f"job-{index:02d}"}),
                    context=CONTEXT,
                )
                await waits.register(registration(correlation), context=CONTEXT)
                correlations.append(correlation)
            clock[0] += timedelta(seconds=2)
            app = CayuApp(session_store=store, enable_logging=False)
            stop = asyncio.Event()
            pages = []

            class Host(ExternalWaitHost):
                async def service_once(self, **kwargs):
                    page = await super().service_once(**kwargs)
                    pages.append(page)
                    if page.next_cursor is None:
                        stop.set()
                    return page

            try:
                await asyncio.wait_for(
                    Host(SessionExternalWaitAdapter(app, waits), context=CONTEXT).run(
                        scope=initial.scope, source="renderer", stop=stop, interval_s=60
                    ),
                    timeout=10,
                )
                assert [page.inspected for page in pages] == [32, 1]
                for correlation in correlations:
                    assert (
                        await waits.inspect(correlation, context=CONTEXT)
                    ).outcome.kind == "timeout"
            finally:
                await app.aclose()
                await waits.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure_type", [ContinuationConflict, ContinuationUnavailable])
def test_host_logs_persistent_row_failure_once(failure_type, monkeypatch, caplog):
    async def scenario():
        waits = ExternalEventWaits(store=InMemorySessionStore(), access_policy=Policy())
        app = CayuApp(session_store=waits.store, enable_logging=False)
        provider = ScriptedModelProvider([ModelStreamEvent.completed({"finish_reason": "stop"})])
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="root", model="model"))
        adapter = SessionExternalWaitAdapter(app, waits)
        correlation = await waits.reserve_correlation(reservation(), context=CONTEXT)
        registered = registration(correlation)
        await waits.register(registered, context=CONTEXT)
        try:
            await adapter.run_to_wait(
                RunRequest(agent_name="root", messages=[Message.text("user", "submit")]),
                registered,
                context=CONTEXT,
            )
            await waits.deliver(
                ExternalEventDelivery(
                    correlation=correlation, delivery_id="done", payload_json="{}"
                ),
                context=CONTEXT,
            )

            async def reject(*args, **kwargs):
                raise failure_type("Row changed after discovery")

            monkeypatch.setattr(SessionContinuationOwner, "service", reject)
            stop = asyncio.Event()
            pages = []

            class Host(ExternalWaitHost):
                async def service_once(self, **kwargs):
                    page = await super().service_once(**kwargs)
                    pages.append(page)
                    if len(pages) == 3:
                        stop.set()
                    return page

            with caplog.at_level(logging.WARNING, logger="cayu.external_wait_host"):
                await Host(adapter, context=CONTEXT).run(
                    scope=correlation.request.scope,
                    source="renderer",
                    stop=stop,
                    interval_s=0.01,
                )
            key = correlation.request.correlation_key
            assert [page.pending for page in pages] == [(key,)] * 3
            warnings = [record for record in caplog.records if key in record.getMessage()]
            if failure_type is ContinuationConflict:
                assert all(page.failures for page in pages)
                assert len(warnings) == 1
            else:
                assert not any(page.failures for page in pages)
                assert warnings == []
        finally:
            await app.aclose()
            await waits.aclose()

    asyncio.run(scenario())


def test_receiver_reports_changed_registration_as_row_conflict():
    from cayu.runtime._external_wait_receiver import ExternalWaitLatchReceiver

    async def scenario():
        waits = ExternalEventWaits(store=InMemorySessionStore(), access_policy=Policy())
        correlation = await waits.reserve_correlation(reservation(), context=CONTEXT)
        try:
            receiver = ExternalWaitLatchReceiver(waits, registration(correlation), CONTEXT)
            with pytest.raises(ContinuationConflict, match="unavailable or changed"):
                await receiver._authenticate(None)
        finally:
            await waits.aclose()

    asyncio.run(scenario())


def test_host_reports_preparation_orphaned_before_binding(monkeypatch):
    import cayu.external_wait_host as host_module

    async def scenario():
        waits = ExternalEventWaits(store=InMemorySessionStore(), access_policy=Policy())
        app = CayuApp(session_store=waits.store, enable_logging=False)
        provider = ScriptedModelProvider([ModelStreamEvent.completed({"finish_reason": "stop"})])
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="root", model="model"))
        adapter = SessionExternalWaitAdapter(app, waits)
        correlation = await waits.reserve_correlation(reservation(), context=CONTEXT)
        registered = registration(correlation)
        await waits.register(registered, context=CONTEXT)
        original = _ExternalExecutionToWait.prepare_initial

        async def crash_after_preparation(boundary, *args, **kwargs):
            await original(boundary, *args, **kwargs)
            raise RuntimeError("process died before native wait creation")

        monkeypatch.setattr(_ExternalExecutionToWait, "prepare_initial", crash_after_preparation)
        try:
            with pytest.raises(RuntimeError, match="process died"):
                await adapter.run_to_wait(
                    RunRequest(agent_name="root", messages=[Message.text("user", "submit")]),
                    registered,
                    context=CONTEXT,
                )
            await waits.deliver(
                ExternalEventDelivery(
                    correlation=correlation, delivery_id="done", payload_json="{}"
                ),
                context=CONTEXT,
            )
            host = ExternalWaitHost(adapter, context=CONTEXT)
            key = correlation.request.correlation_key
            page = await host.service_once(scope=correlation.request.scope, source="renderer")
            assert page.pending == (key,) and page.failures == ()

            later = time.time() + 120
            monkeypatch.setattr(host_module.time, "time", lambda: later)
            page = await host.service_once(scope=correlation.request.scope, source="renderer")
            assert page.pending == (key,)
            assert [(failure.correlation_key, failure.kind) for failure in page.failures] == [
                (key, "orphaned")
            ]
            assert len(provider.requests) == 0
        finally:
            await app.aclose()
            await waits.aclose()

    asyncio.run(scenario())
