"""A pending page head cannot starve another durable clarification delivery."""

import asyncio

import pytest
from tests.core.test_clarification_public import (
    test_public_question_uses_real_assistant_export as public_question_journey,
)
from tests.core.test_participant_identity import CONTEXT

from cayu import (
    CollaborationHost,
    HostClarificationMaintenanceSource,
    HostOwnershipLimits,
    HostRegistration,
)
from cayu.collaboration.peer_content import _commitment

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("restart", [False, True])
async def test_pending_delivery_head_does_not_starve_settlement(
    backend, restart, tmp_path, request, monkeypatch
):
    async def exercise(app, *, delivery, context, application_for, collaboration_factory):
        peer = delivery.append
        occurrence = peer.occurrence.model_copy(update={"occurrence_id": "sibling-occurrence"})
        occurrence = occurrence.model_copy(
            update={
                "provenance_sha256": _commitment(
                    occurrence.model_dump(
                        mode="json", exclude={"schema_version", "provenance_sha256"}
                    ),
                    "peer_content.occurrence",
                ),
            }
        )
        key = peer.append_key.model_copy(update={"occurrence_id": occurrence.occurrence_id})
        sibling_peer = peer.model_copy(
            update={
                "operation_key": "sibling-peer",
                "occurrence": occurrence,
                "append_key": key,
                "attempt_key": peer.attempt_key.model_copy(update={"append_key": key}),
            }
        )
        sibling = delivery.model_copy(
            update={
                "operation": delivery.operation.model_copy(
                    update={"caller_key": "zz-sibling-delivery"}
                ),
                "append": sibling_peer,
            }
        )
        await app.prepare_clarification_delivery(sibling, context=context)
        # The receiving owner has a genuine exclusion, but the collaboration
        # acknowledgement is deliberately outstanding. The first delivery has
        # no receiving decision at all and must remain pending.
        await app.session_store.exclude_peer_content(sibling_peer, reason="withdrawn")
        page = await app.list_pending_clarification_deliveries(context=CONTEXT)
        assert [item.recovery.operation for item in page.items] == [
            delivery.operation,
            sibling.operation,
        ]
        assert page.next_cursor is None
        seen = []

        def track(reconcile):
            async def tracking(expected, **kwargs):
                result = await reconcile(expected, **kwargs)
                seen.append((expected.operation, result.status))
                return result

            return tracking

        monkeypatch.setattr(
            app._clarification_coordinator,
            "reconcile_delivery",
            track(app._clarification_coordinator.reconcile_delivery),
        )

        def host_for(current):
            return CollaborationHost(
                current,
                HostRegistration(
                    limits=HostOwnershipLimits(1, 1, 2, 262144),
                    producer_sources=(),
                    producer_rules=(),
                    clarification_maintenance_sources=(
                        HostClarificationMaintenanceSource("deliveries", CONTEXT),
                    ),
                    observation_timeout_s=0.1,
                ),
            )

        async def drive(host, predicate):
            async with asyncio.timeout(60):
                while not predicate():
                    await host.service_once()
                    assert host.inspect().failed == host.inspect().source_failures == 0
                    await asyncio.sleep(0.001)

        host = host_for(app)
        reopened_store = None
        try:
            if restart:
                # Reconstruct from the two durable obligations, without copying
                # any local cursor or task handle into the replacement host.
                async with asyncio.timeout(60):
                    while (await host.aclose()).pending:
                        await asyncio.sleep(0.001)
                reopened_store = collaboration_factory()
                reopened = application_for(reopened_store, app.session_store)
                await reopened.initialize_collaboration()
                monkeypatch.setattr(
                    reopened._clarification_coordinator,
                    "reconcile_delivery",
                    track(reopened._clarification_coordinator.reconcile_delivery),
                )
                host = host_for(reopened)
            await drive(host, lambda: (sibling.operation, "excluded") in seen)
            remaining = await app.list_pending_clarification_deliveries(context=CONTEXT)
            assert [item.recovery.operation for item in remaining.items] == [delivery.operation]
            assert await app.session_store.read_peer_content_attempt(peer) is None
            assert (delivery.operation, "pending") in seen
        finally:
            async with asyncio.timeout(60):
                while (await host.aclose()).pending:
                    await asyncio.sleep(0.001)
            if (
                reopened_store is not None
                and reopened_store is not app._participant_coordinator._store
            ):
                await reopened_store.close()

    await public_question_journey(
        backend,
        tmp_path,
        request,
        monkeypatch,
        temporary_service=False,
        public_reply=False,
        post_admission=False,
        side_session=False,
        delivery_prepared_driver=exercise,
        journey_ttl_ms=900_000,
    )
