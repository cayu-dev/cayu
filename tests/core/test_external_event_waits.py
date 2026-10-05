from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from tests.external_wait_support import CONTEXT, Policy, registration, reservation, stores

from cayu.external_waits import (
    ExternalEventWaits,
    ExternalWaitProjector,
)
from cayu.sessions.base import InMemorySessionStore
from cayu.sessions.external_waits import (
    ExternalEventDelivery,
    ExternalWaitCapacityExceeded,
    ExternalWaitConflict,
    ExternalWaitLimits,
    ExternalWaitUnavailable,
)
from cayu.vaults.redaction import SecretRedactor


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("shape", ["unicode", "collection"])
@pytest.mark.parametrize("headroom", [-1, 0, 1])
@pytest.mark.parametrize("projected", [False, True])
def test_compact_utf8_payload_limits_and_restart(
    backend, shape, headroom, projected, tmp_path, request
):
    import json

    value = {"text": "é" * 12000} if shape == "unicode" else [0] * 12000
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    limit = len(encoded.encode()) + headroom

    class Projector(ExternalWaitProjector):
        calls = 0

        def project(self, outcome):
            self.calls += 1
            return encoded

    async def scenario():
        async with stores(backend, tmp_path, request, [datetime.now(UTC)]) as (store, reopen):
            projector = Projector()
            limits = ExternalWaitLimits(
                payload_bytes=65536 if projected else limit, projection_bytes=limit
            )
            waits = ExternalEventWaits(
                store=store,
                access_policy=Policy(),
                limits=limits,
                projectors={("json", 1): projector} if projected else None,
            )
            correlation = await waits.reserve_correlation(reservation(), context=CONTEXT)
            registered = registration(correlation)
            await waits.register(registered, context=CONTEXT)
            delivery = ExternalEventDelivery(
                correlation=correlation,
                delivery_id="done",
                payload_json="{}" if projected else encoded,
            )
            before = await waits.inspect(correlation, context=CONTEXT)
            if headroom < 0 and not projected:
                with pytest.raises(ValueError):
                    await waits.deliver(delivery, context=CONTEXT)
                assert await waits.inspect(correlation, context=CONTEXT) == before
            else:
                receipt = await waits.deliver(delivery, context=CONTEXT)
                before = await waits.inspect(correlation, context=CONTEXT)
                if headroom < 0:
                    with pytest.raises(ExternalWaitUnavailable):
                        await waits.project(registered, context=CONTEXT)
                    assert await waits.inspect(correlation, context=CONTEXT) == before
                else:
                    result = await waits.project(registered, context=CONTEXT)
                    assert result.projection_json == encoded
            restored = ExternalEventWaits(
                store=reopen(),
                access_policy=Policy(),
                limits=limits,
                projectors={("json", 1): projector} if projected else None,
            )
            if headroom < 0:
                if projected:
                    with pytest.raises(ExternalWaitUnavailable):
                        await restored.project(registered, context=CONTEXT)
                else:
                    with pytest.raises(ValueError):
                        await restored.deliver(delivery, context=CONTEXT)
                assert await restored.inspect(correlation, context=CONTEXT) == before
            else:
                assert await restored.deliver(delivery, context=CONTEXT) == receipt
                assert await restored.project(registered, context=CONTEXT) == result
                assert projector.calls == int(projected)
            await restored.aclose()
            await waits.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("custom", [False, True])
def test_maximum_escaped_payload_and_projection_fit_reserved_record(
    backend, tmp_path, request, custom
):
    import json

    encoded = json.dumps('"' * 32767)
    assert len(encoded.encode()) == 65536

    class Projector(ExternalWaitProjector):
        def project(self, outcome):
            return encoded

    async def scenario():
        async with stores(backend, tmp_path, request, [datetime.now(UTC)]) as (store, reopen):
            waits = ExternalEventWaits(
                store=store,
                access_policy=Policy(),
                projectors={("json", 1): Projector()} if custom else None,
            )
            correlation = await waits.reserve_correlation(reservation(), context=CONTEXT)
            registered = registration(correlation)
            await waits.register(registered, context=CONTEXT)
            await waits.deliver(
                ExternalEventDelivery(
                    correlation=correlation, delivery_id="done", payload_json=encoded
                ),
                context=CONTEXT,
            )
            result = await waits.project(registered, context=CONTEXT)
            restored = ExternalEventWaits(store=reopen(), access_policy=Policy(), projectors={})
            assert (await restored.project(registered, context=CONTEXT)) == result
            assert result.projection_json == result.outcome.payload_json == encoded

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_projection_lost_ack_restart_replays_committed_input_without_projector(
    backend, tmp_path, request
):
    class Projector(ExternalWaitProjector):
        calls = 0

        def project(self, outcome):
            self.calls += 1
            return '{"answer":42}'

    async def scenario():
        async with stores(backend, tmp_path, request, [datetime.now(UTC)]) as (store, reopen):
            projector = Projector()
            waits = ExternalEventWaits(
                store=store, access_policy=Policy(), projectors={("answer", 1): projector}
            )
            correlation = await waits.reserve_correlation(reservation(), context=CONTEXT)
            registered = registration(correlation).model_copy(update={"projector_id": "answer"})
            await waits.register(registered, context=CONTEXT)
            await waits.deliver(
                ExternalEventDelivery(
                    correlation=correlation, delivery_id="done", payload_json="{}"
                ),
                context=CONTEXT,
            )
            original = store._mutate_external_wait

            async def lose_ack(command):
                await original(command)
                raise RuntimeError("lost projection acknowledgement")

            store._mutate_external_wait = lose_ack
            with pytest.raises(RuntimeError, match="lost projection acknowledgement"):
                await waits.project(registered, context=CONTEXT)
            del store._mutate_external_wait
            restored = ExternalEventWaits(store=reopen(), access_policy=Policy(), projectors={})
            projected = await restored.project(registered, context=CONTEXT)
            assert projected.projection_json == '{"answer":42}'
            assert projector.calls == 1
            assert await restored.project(registered, context=CONTEXT) == projected
            with pytest.raises(ExternalWaitConflict):
                await restored.project(
                    registered.model_copy(update={"projector_version": 2}), context=CONTEXT
                )
            assert await restored.inspect(correlation, context=CONTEXT) == projected

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_bad_or_unavailable_projector_never_publishes_partial_input(backend, tmp_path, request):
    class Projector(ExternalWaitProjector):
        def project(self, outcome):
            return "x" * 65537

    async def scenario():
        async with stores(backend, tmp_path, request, [datetime.now(UTC)]) as (store, reopen):
            waits = ExternalEventWaits(
                store=store, access_policy=Policy(), projectors={("json", 1): Projector()}
            )
            correlation = await waits.reserve_correlation(reservation(), context=CONTEXT)
            registered = registration(correlation)
            await waits.register(registered, context=CONTEXT)
            await waits.deliver(
                ExternalEventDelivery(
                    correlation=correlation, delivery_id="done", payload_json="{}"
                ),
                context=CONTEXT,
            )
            before = await waits.inspect(correlation, context=CONTEXT)
            with pytest.raises(ExternalWaitUnavailable):
                await waits.project(registered, context=CONTEXT)
            assert await waits.inspect(correlation, context=CONTEXT) == before
            restored = ExternalEventWaits(store=reopen(), access_policy=Policy(), projectors={})
            with pytest.raises(ExternalWaitUnavailable):
                await restored.project(registered, context=CONTEXT)
            assert await restored.inspect(correlation, context=CONTEXT) == before

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_public_early_delivery_registration_restart_and_exact_replay(backend, tmp_path, request):
    async def scenario():
        clock = [datetime.now(UTC)]
        async with stores(backend, tmp_path, request, clock) as (store, reopen):
            waits = ExternalEventWaits(store=store, access_policy=Policy())
            expected = reservation()
            correlation = await waits.reserve_correlation(expected, context=CONTEXT)
            delivery = ExternalEventDelivery(
                correlation=correlation, delivery_id="done", payload_json='{"url":"local-video"}'
            )
            receipt = await waits.deliver(delivery, context=CONTEXT)
            assert receipt.disposition == "accepted"
            assert (await waits.inspect(correlation, context=CONTEXT)).outcome is None
            restored = ExternalEventWaits(store=reopen(), access_policy=Policy())
            assert await restored.reserve_correlation(expected, context=CONTEXT) == correlation
            assert await restored.deliver(delivery, context=CONTEXT) == receipt
            registered = await restored.register(registration(correlation), context=CONTEXT)
            assert registered.outcome.kind == "event"
            assert registered.outcome.payload_json == delivery.payload_json
            assert await restored.register(registration(correlation), context=CONTEXT) == registered
            with pytest.raises(ExternalWaitConflict):
                await restored.deliver(
                    delivery.model_copy(update={"payload_json": '{"url":"changed"}'}),
                    context=CONTEXT,
                )
            assert await restored.inspect(correlation, context=CONTEXT) == registered

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_independent_owners_race_delivery_and_cancel(backend, tmp_path, request):
    async def scenario():
        clock = [datetime.now(UTC)]
        async with stores(backend, tmp_path, request, clock) as (store, reopen):
            first = ExternalEventWaits(store=store, access_policy=Policy())
            other = ExternalEventWaits(store=reopen(), access_policy=Policy())
            correlation = await first.reserve_correlation(reservation(), context=CONTEXT)
            await first.register(registration(correlation), context=CONTEXT)
            await asyncio.gather(
                first.deliver(
                    ExternalEventDelivery(
                        correlation=correlation, delivery_id="done", payload_json="{}"
                    ),
                    context=CONTEXT,
                ),
                other.cancel(correlation, operation_key="cancel", context=CONTEXT),
            )
            selected = await first.inspect(correlation, context=CONTEXT)
            assert selected.outcome.kind in {"event", "cancelled"}
            assert await other.observe(correlation, context=CONTEXT) == selected

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_owner_clock_deadline_equality(backend, offset, tmp_path, request):
    async def scenario():
        clock = [datetime.now(UTC).replace(microsecond=0)]
        deadline = clock[0] + timedelta(seconds=10)
        async with stores(backend, tmp_path, request, clock) as (store, _):
            waits = ExternalEventWaits(store=store, access_policy=Policy())
            correlation = await waits.reserve_correlation(
                reservation(deadline=deadline), context=CONTEXT
            )
            await waits.register(registration(correlation), context=CONTEXT)
            clock[0] = deadline + timedelta(milliseconds=offset)
            await waits.deliver(
                ExternalEventDelivery(
                    correlation=correlation, delivery_id="done", payload_json="{}"
                ),
                context=CONTEXT,
            )
            result = await waits.inspect(correlation, context=CONTEXT)
            assert result.outcome.kind == ("event" if offset < 0 else "timeout")

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_capacity_reserves_cleanup_and_refusal_does_not_mutate(backend, tmp_path, request):
    async def scenario():
        clock = [datetime.now(UTC)]
        async with stores(backend, tmp_path, request, clock) as (store, reopen):
            limits = ExternalWaitLimits(correlations=1, deliveries=1)
            waits = ExternalEventWaits(store=store, access_policy=Policy(), limits=limits)
            expected = reservation()
            correlation = await waits.reserve_correlation(expected, context=CONTEXT)
            before = await waits.inspect(correlation, context=CONTEXT)
            with pytest.raises(ExternalWaitCapacityExceeded):
                await waits.reserve_correlation(
                    expected.model_copy(update={"correlation_key": "other"}), context=CONTEXT
                )
            assert await waits.inspect(correlation, context=CONTEXT) == before
            restored = ExternalEventWaits(store=reopen(), access_policy=Policy(), limits=limits)
            cancelled = await restored.cancel(correlation, operation_key="cancel", context=CONTEXT)
            assert cancelled.outcome.kind == "cancelled"
            assert (
                await restored.cancel(correlation, operation_key="cancel", context=CONTEXT)
                == cancelled
            )

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_revocation_denies_current_read_and_delivery_but_other_authority_can_cleanup(
    backend, tmp_path, request
):
    async def scenario():
        clock = [datetime.now(UTC)]
        async with stores(backend, tmp_path, request, clock) as (store, reopen):
            policy = Policy()
            waits = ExternalEventWaits(store=store, access_policy=policy)
            correlation = await waits.reserve_correlation(reservation(), context=CONTEXT)
            policy.revoked = True
            with pytest.raises(PermissionError):
                await waits.inspect(correlation, context=CONTEXT)
            with pytest.raises(PermissionError):
                await waits.deliver(
                    ExternalEventDelivery(
                        correlation=correlation, delivery_id="done", payload_json="{}"
                    ),
                    context=CONTEXT,
                )
            administrator = ExternalEventWaits(store=reopen(), access_policy=Policy())
            assert (
                await administrator.cancel(correlation, operation_key="cleanup", context=CONTEXT)
            ).outcome.kind == "cancelled"

    asyncio.run(scenario())


def test_cancelled_observer_does_not_cancel_dispatched_store_write():
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        class Store(InMemorySessionStore):
            external_wait_version = 1

            async def _mutate_external_wait(self, command):
                entered.set()
                await release.wait()
                return await super()._mutate_external_wait(command)

        waits = ExternalEventWaits(store=Store(), access_policy=Policy())
        expected = reservation()
        observer = asyncio.create_task(waits.reserve_correlation(expected, context=CONTEXT))
        await entered.wait()
        observer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observer
        assert observer.cancelled() and observer.cancelling() == 1
        assert waits._owners.outstanding()
        release.set()
        correlation = await waits.reserve_correlation(expected, context=CONTEXT)
        assert (await waits.lookup(expected, context=CONTEXT)).correlation == correlation
        assert await waits.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_lost_reservation_acknowledgement_reconciles_without_new_identity(
    backend, tmp_path, request
):
    async def scenario():
        async with stores(backend, tmp_path, request, [datetime.now(UTC)]) as (store, reopen):
            waits = ExternalEventWaits(store=store, access_policy=Policy())
            original = store._mutate_external_wait
            committed = []

            async def commit_then_lose_ack(command):
                result = await original(command)
                committed.append(result)
                raise RuntimeError("lost acknowledgement")

            store._mutate_external_wait = commit_then_lose_ack
            expected = reservation()
            with pytest.raises(RuntimeError, match="lost acknowledgement"):
                await waits.reserve_correlation(expected, context=CONTEXT)
            del store._mutate_external_wait
            restored = ExternalEventWaits(store=reopen(), access_policy=Policy())
            retained = await restored.lookup(expected, context=CONTEXT)
            assert retained.correlation == committed[0].correlation
            assert (
                await restored.reserve_correlation(expected, context=CONTEXT)
                == retained.correlation
            )
            assert retained.revision == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_discovery_filters_source_before_bounded_pagination(backend, tmp_path, request):
    class BothSources(Policy):
        def authorize(self, context, *, scope, source, action):
            return context == CONTEXT and source in {"renderer", "other"}

    async def scenario():
        async with stores(backend, tmp_path, request, [datetime.now(UTC)]) as (store, reopen):
            waits = ExternalEventWaits(store=store, access_policy=BothSources())
            expected = reservation()
            for key, source in (
                ("a", "other"),
                ("b", "renderer"),
                ("c", "other"),
                ("d", "renderer"),
            ):
                await waits.reserve_correlation(
                    expected.model_copy(update={"correlation_key": key, "source": source}),
                    context=CONTEXT,
                )
            restored = ExternalEventWaits(store=reopen(), access_policy=Policy())
            first = await restored.list(
                scope=expected.scope, source="renderer", limit=1, context=CONTEXT
            )
            assert [item.correlation.request.correlation_key for item in first] == ["b"]
            second = await restored.list(
                scope=expected.scope, source="renderer", after="b", limit=1, context=CONTEXT
            )
            assert [item.correlation.request.correlation_key for item in second] == ["d"]

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_rejected_delivery_does_not_commit_lazy_expiry(backend, tmp_path, request):
    async def scenario():
        clock = [datetime.now(UTC)]
        async with stores(backend, tmp_path, request, clock) as (store, reopen):
            waits = ExternalEventWaits(store=store, access_policy=Policy())
            correlation = await waits.reserve_correlation(
                reservation(early_event_retention_seconds=1), context=CONTEXT
            )
            delivery = ExternalEventDelivery(
                correlation=correlation, delivery_id="once", payload_json="{}"
            )
            await waits.deliver(delivery, context=CONTEXT)
            before = await waits.inspect(correlation, context=CONTEXT)
            clock[0] += timedelta(seconds=2)
            restored = ExternalEventWaits(store=reopen(), access_policy=Policy())
            with pytest.raises(ExternalWaitConflict):
                await restored.deliver(
                    delivery.model_copy(update={"payload_json": "[]"}), context=CONTEXT
                )
            assert await restored.inspect(correlation, context=CONTEXT) == before
            expired = await restored.observe(correlation, context=CONTEXT)
            assert expired.outcome.kind == "unavailable"
            assert expired.outcome.payload_json is None
            assert await restored.observe(correlation, context=CONTEXT) == expired

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_early_event_beats_deadline_when_registered_after_deadline(backend, tmp_path, request):
    async def scenario():
        clock = [datetime.now(UTC)]
        async with stores(backend, tmp_path, request, clock) as (store, reopen):
            waits = ExternalEventWaits(store=store, access_policy=Policy())
            correlation = await waits.reserve_correlation(
                reservation(deadline=clock[0] + timedelta(seconds=1)), context=CONTEXT
            )
            await waits.deliver(
                ExternalEventDelivery(
                    correlation=correlation, delivery_id="done", payload_json="{}"
                ),
                context=CONTEXT,
            )
            clock[0] += timedelta(seconds=2)
            restored = ExternalEventWaits(store=reopen(), access_policy=Policy())
            result = await restored.register(registration(correlation), context=CONTEXT)
            assert result.outcome.kind == "event"
            assert await restored.observe(correlation, context=CONTEXT) == result

    asyncio.run(scenario())


def test_rejected_nested_input_does_not_emit_diagnostic_canary(capsys, caplog):
    import warnings

    canary = "external-wait-private-repr-canary"

    class Unsafe:
        def __repr__(self):
            return canary

        def __str__(self):
            return canary

    async def scenario():
        waits = ExternalEventWaits(store=InMemorySessionStore(), access_policy=Policy())
        valid = reservation()
        malformed = valid.model_copy(
            update={"scope": valid.scope.model_copy(update={"generation": Unsafe()})}
        )
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            with pytest.raises(ValueError) as failure:
                await waits.reserve_correlation(malformed, context=CONTEXT)
        assert not captured
        assert canary not in str(failure.value)
        assert canary not in repr(failure.value)
        assert failure.value.__context__ is None
        assert await waits.lookup(valid, context=CONTEXT) is None

    asyncio.run(scenario())
    emitted = capsys.readouterr()
    assert canary not in emitted.out + emitted.err + caplog.text


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize("column", ["source", "session_binding"])
def test_indexed_source_corruption_is_not_returned_as_authorized_record(
    backend, column, tmp_path, request
):
    async def scenario():
        async with stores(backend, tmp_path, request, [datetime.now(UTC)]) as (store, reopen):
            waits = ExternalEventWaits(store=store, access_policy=Policy())
            correlation = await waits.reserve_correlation(reservation(), context=CONTEXT)
            scope = correlation.request.scope
            assignment = (
                "source='other'"
                if column == "source"
                else "session_id='foreign',session_instance_id='foreign-incarnation'"
            )
            if backend == "sqlite":
                with store._connection:
                    store._connection.execute(
                        f"UPDATE cayu_external_waits SET {assignment} WHERE scope=? AND generation=?",
                        (scope.application_scope, scope.generation),
                    )
            else:
                async with store._connection() as connection:
                    await connection.execute(
                        f"UPDATE cayu_external_waits SET {assignment} WHERE scope=%s AND generation=%s",
                        (scope.application_scope, scope.generation),
                    )
            restored = ExternalEventWaits(store=reopen(), access_policy=Policy())
            with pytest.raises(ExternalWaitConflict, match="indexed identity"):
                await restored.inspect(correlation, context=CONTEXT)

    asyncio.run(scenario())


@pytest.mark.parametrize("caller_deadline", [False, True])
def test_expired_observer_retains_write_until_exact_reconciliation(caller_deadline):
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        class Store(InMemorySessionStore):
            external_wait_version = 1

            async def _mutate_external_wait(self, command):
                entered.set()
                await release.wait()
                return await super()._mutate_external_wait(command)

        waits = ExternalEventWaits(store=Store(), access_policy=Policy())
        expected = reservation()
        if not caller_deadline:
            waits._owners.observation_timeout = 0.01

        async def observe():
            if caller_deadline:
                async with asyncio.timeout(0.01):
                    return await waits.reserve_correlation(expected, context=CONTEXT)
            return await waits.reserve_correlation(expected, context=CONTEXT)

        observer = asyncio.create_task(observe())
        try:
            await entered.wait()
            with pytest.raises(TimeoutError if caller_deadline else ExternalWaitUnavailable):
                await observer
            assert not observer.cancelled() and observer.cancelling() == 0
            assert waits._owners.outstanding()
            assert await waits.lookup(expected, context=CONTEXT) is None
            release.set()
            correlation = await waits.reserve_correlation(expected, context=CONTEXT)
            assert (await waits.lookup(expected, context=CONTEXT)).correlation == correlation
        finally:
            release.set()
            assert await waits.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_secret_identifiers_reject_before_write_and_payload_replay_keeps_original_commitment(
    backend, tmp_path, request
):
    async def scenario():
        canary, other = "external-wait-secret-value", "different-private-value"
        async with stores(backend, tmp_path, request, [datetime.now(UTC)]) as (store, reopen):
            waits = ExternalEventWaits(
                store=store, access_policy=Policy(), redactor=SecretRedactor([canary, other])
            )
            correlation = await waits.reserve_correlation(reservation(), context=CONTEXT)
            before = await waits.inspect(correlation, context=CONTEXT)
            for field in ("operation_key", "projector_id"):
                with pytest.raises(ValueError, match="identities must not contain secrets"):
                    await waits.register(
                        registration(correlation).model_copy(update={field: canary}),
                        context=CONTEXT,
                    )
            with pytest.raises(ValueError, match="identities must not contain secrets"):
                await waits.cancel(correlation, operation_key=canary, context=CONTEXT)
            with pytest.raises(ValueError, match="identities must not contain secrets"):
                await waits.deliver(
                    ExternalEventDelivery(
                        correlation=correlation, delivery_id=canary, payload_json="{}"
                    ),
                    context=CONTEXT,
                )
            assert await waits.inspect(correlation, context=CONTEXT) == before
            delivery = ExternalEventDelivery(
                correlation=correlation,
                delivery_id="done",
                payload_json='{"result":"' + canary + '"}',
            )
            receipt = await waits.deliver(delivery, context=CONTEXT)
            with pytest.raises(ExternalWaitConflict):
                await waits.deliver(
                    delivery.model_copy(update={"payload_json": '{"result":"' + other + '"}'}),
                    context=CONTEXT,
                )
            restored = ExternalEventWaits(
                store=reopen(), access_policy=Policy(), redactor=SecretRedactor([canary, other])
            )
            assert await restored.deliver(delivery, context=CONTEXT) == receipt
            result = await restored.register(registration(correlation), context=CONTEXT)
            assert "REDACTED_SECRET" in result.outcome.payload_json
            assert canary not in result.model_dump_json()
            durable = await restored.store._read_external_wait(
                correlation.request.scope, correlation.request.correlation_key
            )
            assert canary not in durable.model_dump_json()

    asyncio.run(scenario())
