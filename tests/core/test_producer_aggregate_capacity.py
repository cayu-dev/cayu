"""Public producer admission counts retained and reserved namespace capacity once."""

from dataclasses import replace

import pytest
from tests.core import test_prepared_admission_public as preparations
from tests.core.test_participant_identity import CONTEXT, registration
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_producer_output_contracts import output_scenario

from cayu.collaboration._contracts import ExactNotFound
from cayu.collaboration.participants import CollaborationCapacityExceeded


@pytest.mark.anyio
@pytest.mark.parametrize("dimension", ["operations", "events", "retained_bytes"])
async def test_public_registration_aggregate_capacity_and_exact_replay(
    native_stores, monkeypatch, dimension
):
    setup = preparations.setup
    configured = registration()

    async def configured_setup(store):
        return await setup(store, reg=configured)

    monkeypatch.setattr(preparations, "setup", configured_setup)
    count, reserved, control = {
        "operations": ("operation_count", "reserved_operations", "control_operations"),
        "events": ("event_count", "reserved_events", "control_events"),
        "retained_bytes": ("retained_bytes", "reserved_bytes", "control_bytes"),
    }[dimension]

    async def scenario():
        values = await output_scenario(native_stores)
        monkeypatch.setattr(values[0]._request_coordinator._owners, "observation_timeout", 60)
        return values

    async def state(values):
        app, _, admission, _, _, initialized, _, _ = values
        store = native_stores[0]
        async with store._transaction(initialized.owner.application_scope, write=False) as tx:
            return (
                await store._anchor(tx, initialized, app._secret_redactor),
                await store._permit_state(tx, admission.prepared.recipient, app._secret_redactor),
            )

    async def register(values):
        app, resolver, _, _, _, _, command, execution = values
        return await app.register_producer_output(
            command, execution, context=resolver.recipient.context
        )

    # Calibrate against a genuine public registration, not a predicted helper
    # count. Each following case has an independent registered namespace with
    # the same work and a durably configured ceiling around this exact total.
    baseline = await scenario()
    await register(baseline)
    anchor, _ = await state(baseline)
    required = getattr(anchor, count) + getattr(anchor, reserved)
    for headroom in (-1, 0, 1):
        fresh = registration()
        limits = fresh.bootstrap.limits.model_copy(
            update={dimension: required + getattr(fresh.bootstrap.limits, control) + headroom}
        )
        configured = replace(fresh, bootstrap=fresh.bootstrap.model_copy(update={"limits": limits}))
        values = await scenario()
        app, _, _, provider, session, _, command, _ = values
        before = await state(values)
        if headroom < 0:
            with pytest.raises(CollaborationCapacityExceeded):
                await register(values)
            assert await state(values) == before
            found = await app.lookup_producer_registration(command, context=CONTEXT)
            assert isinstance(found, ExactNotFound)
        else:
            first = await register(values)
            after = await state(values)
            total = getattr(after[0], count) + getattr(after[0], reserved)
            assert total == required
            # Retained and reserved counts are each individually affordable.
            # The authoritative gate must compare their sum, including prior
            # admission work, rather than enforce two independent thresholds.
            assert 0 < getattr(after[0], count) < required
            assert 0 < getattr(after[0], reserved) < required
            assert await register(values) == first
            assert await state(values) == after
        assert not provider.requests
        assert (await app.session_store.load(session.id)).run_epoch == 0
