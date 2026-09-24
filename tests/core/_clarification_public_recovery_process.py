"""Recover through public maintenance with no original instruction or provider."""

import asyncio
import json
import os
import sys

import pytest
from tests.core.test_participant_identity import CONTEXT, app, registration

from cayu.collaboration.participants import CollaborationInitialization


async def recover(value):
    assert os.getpid() != value["parent_pid"]
    if value["backend"] == "sqlite":
        from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore
        from cayu.storage.sqlite import SQLiteSessionStore

        source = SQLiteCollaborationStore(value["source_path"])
        sessions = SQLiteSessionStore(value["session_path"])
    else:
        from cayu.storage.collaboration_postgres import PostgresCollaborationStore
        from cayu.storage.postgres import PostgresSessionStore

        source = PostgresCollaborationStore(value["dsn"])
        sessions = PostgresSessionStore(value["dsn"])
    initial = CollaborationInitialization.model_validate(value["initial"])
    registered = registration(
        scope=initial.binding.application_scope, limits=initial.binding.limits
    )
    application = app(source, registered, session_store=sessions)
    try:
        assert await application.initialize_collaboration() == initial
        if value.get("kind") == "question":
            from cayu import ClarificationExpiryRequest

            expiry = ClarificationExpiryRequest.model_validate(value["expiry"])
            first = await application.expire_clarification_question(expiry, context=CONTEXT)
            assert first.status == "expired"
            assert await application.expire_clarification_question(expiry, context=CONTEXT) == first
            assert not (await application.list_due_clarification_questions(context=CONTEXT)).items
            assert (
                len(
                    (await application.list_pending_clarification_deliveries(context=CONTEXT)).items
                )
                == 1
            )
            return
        if value.get("kind", "service") == "delivery":
            from tests.core._clarification_delivery_recovery_flow import recover_delivery

            await recover_delivery(application, expected_status="appended")
            return
        page = await application.list_pending_clarification_services(context=CONTEXT, limit=1)
        assert len(page.items) == 1
        selector = page.items[0].recovery
        if value.get("kind") == "unresolved_service":
            from pathlib import Path

            from cayu.collaboration._contracts import CollaborationConflict
            from cayu.events import EventType

            before = await sessions.load(selector.session_id)
            events = await sessions.load_events(selector.session_id)
            first = await application.reconcile_clarification_service(selector, context=CONTEXT)
            assert first.state == "admitted" and first.released_session_status is None
            assert (
                await application.reconcile_clarification_service(selector, context=CONTEXT)
                == first
            )
            with pytest.raises(CollaborationConflict):
                await application.exclude_clarification_service(selector, context=CONTEXT)
            pending = await application.list_pending_clarification_services(context=CONTEXT)
            assert pending.items == page.items
            assert await sessions.load(selector.session_id) == before
            assert await sessions.load_events(selector.session_id) == events
            if value["backend"] == "sqlite":
                from cayu.storage.budget_ledger import SQLiteBudgetLedger

                ledger = SQLiteBudgetLedger(
                    Path(value["source_path"]).with_name("clarification-budget.sqlite")
                )
            else:
                from cayu.storage.postgres import PostgresBudgetLedger

                ledger = PostgresBudgetLedger(value["dsn"])
            try:
                reservations = [
                    await ledger.load_reservation(event.payload["reservation_id"])
                    for event in events
                    if event.type == EventType.BUDGET_RESERVED
                ]
                # The initial waiting turn settled; the killed service's
                # genuinely admitted dispatch remains charged and unresolved.
                assert len(reservations) == 2
                initial_reservation, active_reservation = reservations
                assert initial_reservation is not None and active_reservation is not None
                assert initial_reservation.status == "reconciled"
                assert active_reservation.status == "active"
                assert active_reservation.dispatch_id is not None
                assert active_reservation.dispatched_at is not None
                assert active_reservation.actual_amount is None
                assert initial_reservation.budget_limit_id == active_reservation.budget_limit_id
            finally:
                await ledger.close()
            return
        first = await application.reconcile_clarification_service(selector, context=CONTEXT)
        assert first.state == "returned" and first.released_session_status == "completed"
        assert await application.reconcile_clarification_service(selector, context=CONTEXT) == first
        assert not (await application.list_pending_clarification_services(context=CONTEXT)).items
    finally:
        await application._request_coordinator.close()
        await source.close()
        await sessions.close()


if __name__ == "__main__":
    asyncio.run(recover(json.load(sys.stdin)))
