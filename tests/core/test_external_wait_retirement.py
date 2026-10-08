"""Retirement is an exact terminal decision, not permission to discard handoffs."""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from tests.external_wait_support import CONTEXT, Policy, registration, reservation, stores

from cayu.external_waits import ExternalEventWaits
from cayu.sessions.external_waits import ExternalWaitConflict


class AdministrationPolicy(Policy):
    def authorize(self, context, *, scope, source, action):
        if source == "*" and action in {"retire", "cleanup"}:
            return not self.revoked and context == CONTEXT
        return super().authorize(context, scope=scope, source=source, action=action)


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_scope_retirement_and_bounded_pruning_survive_restart(backend, tmp_path, request):
    async def scenario():
        async with stores(backend, tmp_path, request, [datetime.now(UTC)]) as (store, reopen):
            policy = AdministrationPolicy()
            waits = ExternalEventWaits(store=store, access_policy=policy)
            first = reservation()
            correlations = [
                await waits.reserve_correlation(
                    first.model_copy(update={"correlation_key": f"job-{index}"}), context=CONTEXT
                )
                for index in range(3)
            ]
            for correlation in correlations[:2]:
                await waits.cancel(correlation, operation_key="cancel", context=CONTEXT)
            before = await waits.list(scope=first.scope, source="renderer", context=CONTEXT)
            with pytest.raises(ExternalWaitConflict, match="unresolved"):
                await waits.retire_scope(first.scope, operation_key="retire", context=CONTEXT)
            assert await waits.list(scope=first.scope, source="renderer", context=CONTEXT) == before
            await waits.cancel(correlations[2], operation_key="cancel", context=CONTEXT)
            receipt = await waits.retire_scope(first.scope, operation_key="retire", context=CONTEXT)
            assert receipt.correlation_count == 3
            restored = ExternalEventWaits(store=reopen(), access_policy=policy)
            assert (
                await restored.retire_scope(first.scope, operation_key="retire", context=CONTEXT)
                == receipt
            )
            with pytest.raises(ExternalWaitConflict):
                await restored.retire_scope(first.scope, operation_key="changed", context=CONTEXT)
            for limit in (False, 0, 257):
                with pytest.raises(ValueError):
                    await restored.prune_retired_scope(receipt, limit=limit, context=CONTEXT)
            policy.revoked = True
            with pytest.raises(PermissionError):
                await restored.prune_retired_scope(receipt, context=CONTEXT)
            policy.revoked = False
            altered = receipt.model_copy(update={"retired_at_ms": receipt.retired_at_ms + 1})
            with pytest.raises(ExternalWaitConflict):
                await restored.prune_retired_scope(altered, context=CONTEXT)
            first_page = await restored.prune_retired_scope(receipt, limit=1, context=CONTEXT)
            assert (first_page.removed, first_page.remaining) == (1, 2)
            restored = ExternalEventWaits(store=reopen(), access_policy=policy)
            final_page = await restored.prune_retired_scope(receipt, limit=256, context=CONTEXT)
            assert (final_page.removed, final_page.remaining) == (2, 0)
            assert (await restored.prune_retired_scope(receipt, context=CONTEXT)).removed == 0
            assert (
                await restored.retire_scope(first.scope, operation_key="retire", context=CONTEXT)
                == receipt
            )
            for correlation in correlations:
                with pytest.raises(ExternalWaitConflict, match="retired"):
                    await restored.reserve_correlation(correlation.request, context=CONTEXT)
            # A new generation is independent, never reusing old identities.
            new_request = first.model_copy(
                update={"scope": first.scope.model_copy(update={"generation": 2})}
            )
            assert (
                await restored.reserve_correlation(new_request, context=CONTEXT)
            ).request == new_request

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_retirement_preserves_selected_but_unconsumed_session_outcome(backend, tmp_path, request):
    from cayu import CayuApp
    from cayu.agents import AgentSpec
    from cayu.evals.testing import ScriptedModelProvider
    from cayu.messages import Message
    from cayu.providers.base import ModelStreamEvent
    from cayu.session_external_waits import SessionExternalWaitAdapter
    from cayu.sessions.base import RunRequest
    from cayu.sessions.external_waits import ExternalEventDelivery

    async def scenario():
        async with stores(backend, tmp_path, request, [datetime.now(UTC)]) as (store, reopen):
            waits = ExternalEventWaits(store=store, access_policy=AdministrationPolicy())
            correlation = await waits.reserve_correlation(reservation(), context=CONTEXT)
            registered = registration(correlation)
            await waits.register(registered, context=CONTEXT)
            provider = ScriptedModelProvider(
                [
                    [ModelStreamEvent.completed({"finish_reason": "stop"})],
                    [ModelStreamEvent.completed({"finish_reason": "stop"})],
                ]
            )
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="root", model="model"))
            adapter = SessionExternalWaitAdapter(app, waits)
            await adapter.run_to_wait(
                RunRequest(agent_name="root", messages=[Message.text("user", "Submit job")]),
                registered,
                context=CONTEXT,
            )
            await waits.deliver(
                ExternalEventDelivery(
                    correlation=correlation, delivery_id="result", payload_json="{}"
                ),
                context=CONTEXT,
            )
            before = await waits.inspect(correlation, context=CONTEXT)
            assert before.outcome.kind == "event" and before.pending_handoff
            second = ExternalEventWaits(store=reopen(), access_policy=AdministrationPolicy())
            with pytest.raises(ExternalWaitConflict, match="unresolved"):
                await second.retire_scope(
                    correlation.request.scope, operation_key="retire", context=CONTEXT
                )
            assert await waits.inspect(correlation, context=CONTEXT) == before
            await adapter.service_wait(registered, context=CONTEXT)
            assert len(provider.requests) == 2
            receipt = await second.retire_scope(
                correlation.request.scope, operation_key="retire", context=CONTEXT
            )
            assert (await second.prune_retired_scope(receipt, context=CONTEXT)).remaining == 0
            assert len(provider.requests) == 2
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("retirement_first", [False, True])
def test_scope_retirement_serializes_against_new_reservation(
    backend, retirement_first, tmp_path, request
):
    async def scenario():
        async with stores(backend, tmp_path, request, [datetime.now(UTC)]) as (store, reopen):
            first = ExternalEventWaits(store=store, access_policy=AdministrationPolicy())
            second = ExternalEventWaits(store=reopen(), access_policy=AdministrationPolicy())
            expected = reservation()
            if retirement_first:
                receipt = await first.retire_scope(
                    expected.scope, operation_key="retire", context=CONTEXT
                )
                assert receipt.correlation_count == 0
                with pytest.raises(ExternalWaitConflict, match="retired"):
                    await second.reserve_correlation(expected, context=CONTEXT)
            else:
                correlation = await first.reserve_correlation(expected, context=CONTEXT)
                with pytest.raises(ExternalWaitConflict, match="unresolved"):
                    await second.retire_scope(
                        expected.scope, operation_key="retire", context=CONTEXT
                    )
                await first.cancel(correlation, operation_key="cancel", context=CONTEXT)
                assert (
                    await second.retire_scope(
                        expected.scope, operation_key="retire", context=CONTEXT
                    )
                ).correlation_count == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("failure", ["lost_ack", "cancel"])
def test_retirement_acknowledgement_loss_has_exact_reopened_readback(
    backend, failure, tmp_path, request, monkeypatch
):
    async def scenario():
        async with stores(backend, tmp_path, request, [datetime.now(UTC)]) as (store, reopen):
            waits = ExternalEventWaits(store=store, access_policy=AdministrationPolicy())
            expected = reservation()
            original = store._retire_external_wait_scope
            committed = asyncio.Event()
            release = asyncio.Event()
            receipts = []

            async def lose_ack(command):
                receipts.append(await original(command))
                committed.set()
                if failure == "cancel":
                    await release.wait()
                raise RuntimeError("retirement acknowledgement lost")

            with monkeypatch.context() as patch:
                patch.setattr(store, "_retire_external_wait_scope", lose_ack)
                task = asyncio.create_task(
                    waits.retire_scope(expected.scope, operation_key="retire", context=CONTEXT)
                )
                await asyncio.wait_for(committed.wait(), 10)
                if failure == "cancel":
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                    assert task.cancelled() and task.cancelling() == 1
                else:
                    with pytest.raises(RuntimeError, match="acknowledgement lost"):
                        await task
                second = (
                    ExternalEventWaits(store=reopen(), access_policy=AdministrationPolicy())
                    if backend != "memory"
                    else waits
                )
                with pytest.raises(ExternalWaitConflict, match="retired"):
                    await second.reserve_correlation(expected, context=CONTEXT)
                release.set()
                assert await waits.aclose()
            del store._retire_external_wait_scope
            restored = ExternalEventWaits(store=reopen(), access_policy=AdministrationPolicy())
            assert (
                await restored.retire_scope(expected.scope, operation_key="retire", context=CONTEXT)
                == receipts[0]
            )

    asyncio.run(scenario())


def test_scope_administration_is_not_inferred_from_source_authority():
    async def scenario():
        from cayu.sessions.base import InMemorySessionStore

        store = InMemorySessionStore()
        waits = ExternalEventWaits(store=store, access_policy=Policy())
        expected = reservation()
        with pytest.raises(PermissionError):
            await waits.retire_scope(expected.scope, operation_key="retire", context=CONTEXT)
        await waits.reserve_correlation(expected, context=CONTEXT)

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_retirement_cannot_discard_pending_timer_publication(
    backend, tmp_path, request, monkeypatch
):
    from cayu.external_wait_scheduler import TaskStoreWaitScheduler
    from cayu.tasks.memory import InMemoryTaskStore

    async def scenario():
        async with stores(backend, tmp_path, request, [datetime.now(UTC)]) as (store, reopen):
            waits = ExternalEventWaits(store=store, access_policy=AdministrationPolicy())
            correlation = await waits.reserve_correlation(
                reservation(deadline=datetime.now(UTC) - timedelta(seconds=1)), context=CONTEXT
            )
            registered = registration(correlation)
            await waits.register(registered, context=CONTEXT)
            tasks = InMemoryTaskStore()
            scheduler = TaskStoreWaitScheduler(
                waits=waits, task_store=tasks, scheduler_id="scheduler"
            )
            original = waits._observe_operation

            async def lose_ack(operation, *, key, expectation):
                result = await original(operation, key=key, expectation=expectation)
                if key[0] == "external-timer":
                    raise RuntimeError("timer acknowledgement lost")
                return result

            with monkeypatch.context() as patch:
                patch.setattr(waits, "_observe_operation", lose_ack)
                with pytest.raises(RuntimeError, match="acknowledgement lost"):
                    await scheduler.schedule(registered, context=CONTEXT)
            restored = ExternalEventWaits(store=reopen(), access_policy=AdministrationPolicy())
            pending = await restored.inspect(correlation, context=CONTEXT)
            assert pending.outcome.kind == "timeout" and pending.pending_timer
            with pytest.raises(ExternalWaitConflict, match="unresolved"):
                await restored.retire_scope(
                    correlation.request.scope, operation_key="retire", context=CONTEXT
                )
            recovered_scheduler = TaskStoreWaitScheduler(
                waits=restored, task_store=tasks, scheduler_id="scheduler"
            )
            await recovered_scheduler.schedule(registered, context=CONTEXT)
            assert (
                await restored.retire_scope(
                    correlation.request.scope, operation_key="retire", context=CONTEXT
                )
            ).correlation_count == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_concurrent_reservation_and_retirement_have_one_native_winner(backend, tmp_path, request):
    async def scenario():
        async with stores(backend, tmp_path, request, [datetime.now(UTC)]) as (store, reopen):
            first = ExternalEventWaits(store=store, access_policy=AdministrationPolicy())
            second = ExternalEventWaits(store=reopen(), access_policy=AdministrationPolicy())
            expected = reservation()
            ready = asyncio.Event()

            async def reserve():
                await ready.wait()
                return await first.reserve_correlation(expected, context=CONTEXT)

            async def retire():
                await ready.wait()
                return await second.retire_scope(
                    expected.scope, operation_key="retire", context=CONTEXT
                )

            pending = [asyncio.create_task(reserve()), asyncio.create_task(retire())]
            ready.set()
            outcomes = await asyncio.gather(*pending, return_exceptions=True)
            assert sum(isinstance(value, ExternalWaitConflict) for value in outcomes) == 1
            if isinstance(outcomes[0], ExternalWaitConflict):
                assert outcomes[1].correlation_count == 0
                assert await first.lookup(expected, context=CONTEXT) is None
            else:
                assert (await second.lookup(expected, context=CONTEXT)).correlation == outcomes[0]

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_expiry_sweep_is_bounded_and_allows_retirement(backend, tmp_path, request):
    async def scenario():
        clock = [datetime.now(UTC)]
        async with stores(backend, tmp_path, request, clock) as (store, reopen):
            waits = ExternalEventWaits(store=store, access_policy=AdministrationPolicy())
            expected = reservation(early_event_retention_seconds=1)
            for index in range(2):
                await waits.reserve_correlation(
                    expected.model_copy(update={"correlation_key": f"job-{index}"}), context=CONTEXT
                )
            if backend == "postgres":
                await asyncio.sleep(1.05)
            else:
                clock[0] += timedelta(seconds=1)
            restored = ExternalEventWaits(store=reopen(), access_policy=AdministrationPolicy())
            first = await restored.observe_page(
                scope=expected.scope, source="renderer", context=CONTEXT, limit=1
            )
            assert len(first) == 1 and first[0].outcome.kind == "unavailable"
            with pytest.raises(ExternalWaitConflict, match="unresolved"):
                await restored.retire_scope(expected.scope, operation_key="retire", context=CONTEXT)
            last = await restored.observe_page(
                scope=expected.scope,
                source="renderer",
                context=CONTEXT,
                after=first[0].correlation.request.correlation_key,
                limit=1,
            )
            assert len(last) == 1 and last[0].outcome.kind == "unavailable"
            assert (
                await restored.retire_scope(expected.scope, operation_key="retire", context=CONTEXT)
            ).correlation_count == 2

    asyncio.run(scenario())
