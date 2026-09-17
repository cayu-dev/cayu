"""Authenticated live obligation pagination across supported stores."""

from dataclasses import replace

import pytest
from tests.core import test_participant_identity as identity_tests
from tests.core.test_collaboration_permits import REDACTOR, Receiver, permit
from tests.core.test_participant_identity import CONTEXT, app, create, registration

from cayu.collaboration._contracts import CollaborationConflict, CollaborationContractError
from cayu.collaboration.access import CollaborationAccessDenied

pytestmark = pytest.mark.anyio
stores = identity_tests.stores


async def test_public_pending_inventory_is_filtered_ordered_and_reconstructible(stores):
    store = stores()
    reg = registration()
    application = app(store, reg)
    initialized = await application.initialize_collaboration()
    _, created = await create(application, initialized)
    ref = created.participants[0].reference
    expected = [permit(initialized, ref, key=f"permit-{number}") for number in range(5)]
    for value in expected:
        await store._register_permit(initialized, value, redactor=REDACTOR)
    for value in (expected[1], expected[3]):
        await store._settle_permit(initialized, value, reader=Receiver(value), redactor=REDACTOR)
    reconstructed = app(stores(), reg)
    await reconstructed.initialize_collaboration()
    for pending_only, positions in ((True, [1, 3, 5]), (False, [1, 2, 3, 4, 5])):
        cursor, observed = None, []
        for _ in range(5):
            page = await reconstructed.list_participant_obligations(
                ref,
                context=CONTEXT,
                pending_only=pending_only,
                limit=2,
                cursor=cursor,
            )
            observed.extend(value.position for value in page.obligations)
            for value in page.obligations:
                assert value.participant == ref
                assert (value.outcome is None) == (value.state == "pending")
            cursor = page.next_cursor
            if cursor is None:
                break
        assert observed == positions and cursor is None


async def test_obligation_cursor_binds_principal_participant_filter_and_current_access(stores):
    store = stores()
    policy = identity_tests.Policy()
    application = app(store, registration(policy=policy))
    initialized = await application.initialize_collaboration()
    _, created = await create(application, initialized)
    _, other = await create(application, initialized, key="other")
    ref, other_ref = created.participants[0].reference, other.participants[0].reference
    for number in range(2):
        await store._register_permit(
            initialized, permit(initialized, ref, key=f"p-{number}"), redactor=REDACTOR
        )
    policy.allowed = (ref,)
    page = await application.list_participant_obligations(ref, context=CONTEXT, limit=1)
    assert page.next_cursor is not None
    with pytest.raises(CollaborationAccessDenied):
        await application.list_participant_obligations(other_ref, context=CONTEXT)
    for field, value in (
        ("principal", "different"),
        ("participant", other_ref),
        ("scope", "other"),
    ):
        with pytest.raises(CollaborationConflict):
            await application.list_participant_obligations(
                ref,
                context=CONTEXT,
                cursor=page.next_cursor.model_copy(update={field: value}),
            )
    with pytest.raises(CollaborationConflict):
        await application.list_participant_obligations(
            ref, context=CONTEXT, pending_only=False, cursor=page.next_cursor
        )
    policy.denied.add("obligations")
    with pytest.raises(CollaborationAccessDenied):
        await application.list_participant_obligations(
            ref, context=CONTEXT, cursor=page.next_cursor
        )


@pytest.mark.parametrize(
    "field,value",
    [("limit", True), ("limit", 0), ("limit", 65), ("pending_only", 1), ("pending_only", "true")],
)
async def test_obligation_queries_validate_before_store_dispatch(stores, monkeypatch, field, value):
    store = stores()
    application = app(store, registration())
    initialized = await application.initialize_collaboration()
    _, created = await create(application, initialized)

    async def forbidden(*args, **kwargs):
        pytest.fail("Invalid input reached store")

    monkeypatch.setattr(store, "scan_obligations", forbidden)
    with pytest.raises(CollaborationContractError):
        await application.list_participant_obligations(
            created.participants[0].reference, context=CONTEXT, **{field: value}
        )


async def test_obligation_pages_shrink_to_byte_bound_without_skipping(stores):
    from cayu.collaboration._contracts import MAX_ENVELOPE_BYTES
    from cayu.collaboration._preparation import contract_bytes

    store = stores()
    reg = registration()
    reg = replace(
        reg,
        bootstrap=reg.bootstrap.model_copy(
            update={
                "application_scope": reg.bootstrap.application_scope + "s" * 480,
                "owner_name": "o" * 512,
                "limits": reg.bootstrap.limits.model_copy(
                    update={"retained_bytes": 16 * 1024 * 1024}
                ),
            }
        ),
    )
    application = app(store, reg)
    initialized = await application.initialize_collaboration()
    _, created = await create(application, initialized)
    ref = created.participants[0].reference
    for number in range(24):
        expected = permit(initialized, ref, key=f"p-{number}")
        await store._register_permit(initialized, expected, redactor=REDACTOR)
    cursor, positions, lengths = None, [], []
    for _ in range(16):
        page = await application.list_participant_obligations(ref, context=CONTEXT, cursor=cursor)
        assert len(contract_bytes(page, redactor=REDACTOR)) <= MAX_ENVELOPE_BYTES
        positions.extend(value.position for value in page.obligations)
        lengths.append(len(page.obligations))
        cursor = page.next_cursor
        if cursor is None:
            break
    assert positions == list(range(1, 25)) and cursor is None
    assert 0 < lengths[0] < 24
