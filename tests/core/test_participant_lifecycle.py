"""Public lifecycle semantics with real store histories and empty-frontier evidence."""

import asyncio

import pytest
from tests.core import test_participant_identity as identity_tests
from tests.core.test_participant_identity import CONTEXT, app, configuration, create, registration

from cayu.collaboration._contracts import CollaborationConflict
from cayu.collaboration.lifecycle import ParticipantLifecycleChange
from cayu.collaboration.participants import (
    ParticipantAliasChange,
    ParticipantConfigure,
)

pytestmark = pytest.mark.anyio
stores = identity_tests.stores


def change(initialized, participant, *, key, revision, state):
    return ParticipantLifecycleChange(
        operation=initialized.operation(key),
        participant=participant,
        expected_lifecycle_revision=revision,
        state=state,
    )


@pytest.mark.parametrize("closed", ["draining", "disabled"])
async def test_lifecycle_receipts_are_historical_and_reenable_is_new_authority(stores, closed):
    reg = registration()
    application = app(stores(), reg)
    initialized = await application.initialize_collaboration()
    create_request, created = await create(application, initialized, alias="reviewer")
    ref = created.participants[0].reference
    request = change(initialized, ref, key="close", revision=1, state=closed)
    closed_receipt = await application.change_participant_lifecycle(request, context=CONTEXT)
    assert closed_receipt.participant.lifecycle == closed
    inspected = await application.inspect_participant(ref, context=CONTEXT)
    assert inspected.participant == closed_receipt.participant
    assert inspected.settlement == "settled"
    assert inspected.issued_permit_frontier == inspected.outstanding_obligations == 0
    configured = await application.configure_participant(
        ParticipantConfigure(
            operation=initialized.operation("configure"),
            participant=ref,
            expected_configuration_revision=1,
            configuration=configuration(2),
        ),
        context=CONTEXT,
    )
    assert configured.participants[0].lifecycle == closed
    assert configured.participants[0].lifecycle_revision == 2
    enabled = await application.change_participant_lifecycle(
        change(
            initialized,
            ref,
            key="enable",
            revision=2,
            state="active",
        ),
        context=CONTEXT,
    )
    assert enabled.participant.admission_generation == 2
    assert enabled.participant.lifecycle_revision == 3
    assert enabled.participant.configuration_revision == 2
    reconstructed = app(stores(), reg)
    assert await reconstructed.initialize_collaboration() == initialized
    assert (
        await reconstructed.change_participant_lifecycle(request, context=CONTEXT) == closed_receipt
    )
    assert await reconstructed.create_participant(create_request, context=CONTEXT) == created
    assert (
        await reconstructed.inspect_participant(ref, context=CONTEXT)
    ).participant == enabled.participant
    alias = await reconstructed.resolve_participant_alias("reviewer", context=CONTEXT)
    assert alias.target == ref


async def test_retirement_cannot_be_reenabled_or_reconfigured_but_alias_can_be_removed(stores):
    application = app(stores(), registration())
    initialized = await application.initialize_collaboration()
    create_request, created = await create(application, initialized, alias="reviewer")
    ref = created.participants[0].reference
    retired = await application.change_participant_lifecycle(
        change(
            initialized,
            ref,
            key="retire",
            revision=1,
            state="retired",
        ),
        context=CONTEXT,
    )
    for state in ("active", "draining", "disabled", "retired"):
        with pytest.raises(CollaborationConflict):
            await application.change_participant_lifecycle(
                change(
                    initialized,
                    ref,
                    key=f"bad-{state}",
                    revision=2,
                    state=state,
                ),
                context=CONTEXT,
            )
    with pytest.raises(CollaborationConflict):
        await application.configure_participant(
            ParticipantConfigure(
                operation=initialized.operation("configure"),
                participant=ref,
                expected_configuration_revision=1,
                configuration=configuration(2),
            ),
            context=CONTEXT,
        )
    removed = await application.change_participant_alias(
        ParticipantAliasChange(
            operation=initialized.operation("remove"),
            alias="reviewer",
            expected_alias_revision=1,
            expected_target=ref,
            target=None,
        ),
        context=CONTEXT,
    )
    assert removed.participants[0] == retired.participant
    with pytest.raises(CollaborationConflict):
        await application.change_participant_alias(
            ParticipantAliasChange(
                operation=initialized.operation("rebind"),
                alias="reviewer",
                expected_alias_revision=2,
                expected_target=None,
                target=ref,
            ),
            context=CONTEXT,
        )
    assert await application.create_participant(create_request, context=CONTEXT) == created


async def test_concurrent_lifecycle_controls_have_one_winner(stores):
    reg = registration()
    applications = [app(stores(), reg), app(stores(), reg)]
    initialized = await applications[0].initialize_collaboration()
    await applications[1].initialize_collaboration()
    _, created = await create(applications[0], initialized)
    ref = created.participants[0].reference
    results = await asyncio.gather(
        *(
            application.change_participant_lifecycle(
                change(
                    initialized,
                    ref,
                    key=state,
                    revision=1,
                    state=state,
                ),
                context=CONTEXT,
            )
            for application, state in zip(applications, ("draining", "disabled"), strict=True)
        ),
        return_exceptions=True,
    )
    assert sum(isinstance(result, CollaborationConflict) for result in results) == 1
    inspected = await applications[0].inspect_participant(ref, context=CONTEXT)
    assert inspected.participant.lifecycle_revision == 2
    page = await applications[0].list_participant_events(context=CONTEXT)
    assert len(page.events) == 3
