"""Shared root accounting reaches actual temporary provider admission."""

from decimal import Decimal

from cayu.events import EventType


async def verify_shared_budget(app, source, target, binding, payloads, *, allowed):
    events = [
        *await app.session_store.load_events(source.id),
        *await app.session_store.load_events(target.id),
    ]
    reserved = [event for event in events if event.type == EventType.BUDGET_RESERVED]
    rejected = [event for event in events if event.type == EventType.BUDGET_RESERVATION_FAILED]
    limit_count = 1 + len(binding.ancestor_budget_ids)
    assert len(payloads) == 2 + allowed
    assert len(reserved) == len(payloads) * limit_count
    records = [
        await app.budget_ledger.load_reservation(event.payload["reservation_id"])
        for event in reserved
    ]
    limit_ids = {record.budget_limit_id for record in records}
    assert len(limit_ids) == limit_count
    assert all(record.status == "reconciled" for record in records)
    assert all(
        record.settlement_event_payload["budget_binding_authority_sha256"]
        == binding.authority_digest
        for record in records
    )
    for limit_id in limit_ids:
        # Root and ancestor reservations represent the same dispatches, not
        # additive spending. Each ceiling must independently see every call.
        members = [record for record in records if record.budget_limit_id == limit_id]
        assert len(members) == len(payloads)
        assert sum((record.actual_amount for record in members), Decimal(0)) == Decimal(
            "0.000015"
        ) * (2 + allowed)
    assert len(rejected) == (not allowed)
    if rejected:
        assert Decimal(rejected[0].payload["requested"]) == Decimal("0.002048")
