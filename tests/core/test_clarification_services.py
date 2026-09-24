"""Native service/permit/lineage transaction, without provider dispatch."""

import asyncio

import pytest
from tests.core.test_participant_identity import stores as stores
from tests.core.test_temporary_continuation_permits import prepared

from cayu.collaboration._clarification_service_store import (
    discover_services_in_transaction,
    prepare_runtime_service_in_transaction,
    register_service_in_transaction,
)
from cayu.collaboration._contracts import CollaborationContractError
from cayu.collaboration._request_store import operation_key
from cayu.collaboration.clarifications import ClarificationDueCursor
from cayu.runtime._session_continuation import continuation_digest
from cayu.runtime._temporary_continuation import temporary_service_invocation_id
from cayu.vaults.redaction import SecretRedactor

pytestmark = pytest.mark.anyio
REDACTOR = SecretRedactor()


@pytest.mark.parametrize("registered_first", (False, True))
async def test_read_only_preparation_does_not_renew_authority(stores, registered_first):
    from tests.core.test_participant_identity import CONTEXT
    from tests.core.test_participant_lifecycle import change

    from cayu.collaboration._contracts import CollaborationConflict

    store = stores()
    application, initial, template = await prepared(store, open_question=True)
    async with store._transaction(initial.binding.application_scope, write=False) as tx:
        before = await store._anchor(tx, initial, REDACTOR)
        preparation = await prepare_runtime_service_in_transaction(
            store, tx, initial, template.dispatch, redactor=REDACTOR
        )
        assert await store._anchor(tx, initial, REDACTOR) == before
        assert (
            await tx.get(
                "clarification_services", operation_key(template.dispatch.intent.operation)
            )
            is None
        )
    assert (
        await store._lookup_registered_permit(
            initial, preparation.permit.operation, redactor=REDACTOR
        )
        is None
    )

    async def register():
        instance = stores()
        async with instance._transaction(initial.binding.application_scope, write=True) as tx:
            return await register_service_in_transaction(
                instance, tx, initial, preparation.dispatch, preparation.permit, redactor=REDACTOR
            )

    if registered_first:
        original = await register()
    await application.change_participant_lifecycle(
        change(
            initial,
            preparation.permit.intent.request.participant,
            key="disable-after-preparation",
            revision=1,
            state="disabled",
        ),
        context=CONTEXT,
    )
    if registered_first:
        assert await register() == original
        reopened = stores()
        async with reopened._transaction(initial.binding.application_scope, write=False) as tx:
            assert (
                await prepare_runtime_service_in_transaction(
                    reopened, tx, initial, template.dispatch, redactor=REDACTOR
                )
                == preparation
            )
    else:
        with pytest.raises(CollaborationConflict):
            await register()
        assert (
            await store._lookup_registered_permit(
                initial, preparation.permit.operation, redactor=REDACTOR
            )
            is None
        )


async def test_service_registration_charges_once_and_rolls_back_as_one_transaction(stores):
    store = stores()
    _, initial, admission = await prepared(store, open_question=True)

    async def register(instance, dispatch, permit, *, fail=False):
        async with instance._transaction(initial.binding.application_scope, write=True) as tx:
            result = await register_service_in_transaction(
                instance, tx, initial, dispatch, permit, redactor=REDACTOR
            )
            if fail:
                raise OSError("lost transaction before commit")
            return result

    with pytest.raises(OSError):
        await register(store, admission.dispatch, admission.permit, fail=True)
    assert (
        await store._lookup_registered_permit(
            initial, admission.permit.operation, redactor=REDACTOR
        )
        is None
    )
    record, competing = await asyncio.gather(
        register(store, admission.dispatch, admission.permit),
        register(stores(), admission.dispatch, admission.permit),
    )
    assert competing == record
    assert await register(stores(), admission.dispatch, admission.permit) == record
    reopened = stores()
    async with reopened._transaction(initial.binding.application_scope, write=False) as tx:
        assert await tx.scan_clarification_request_handoffs(
            record.dispatch.intent.question.request, family="clarification_services", limit=1
        ) == [record.model_dump(mode="json")]
        for limit in (0, True, 65):
            with pytest.raises(CollaborationContractError):
                await tx.scan_clarification_request_handoffs(
                    record.dispatch.intent.question.request,
                    family="clarification_services",
                    limit=limit,
                )
        assert await discover_services_in_transaction(
            reopened, tx, initial, after=None, limit=1, redactor=REDACTOR
        ) == (record,)
        assert (
            await discover_services_in_transaction(
                reopened,
                tx,
                initial,
                after=ClarificationDueCursor(
                    deadline_at_ms=admission.dispatch.intent.question.deadline_at_ms,
                    operation=admission.dispatch.intent.operation,
                ),
                limit=1,
                redactor=REDACTOR,
            )
            == ()
        )
    async with store._transaction(initial.binding.application_scope, write=False) as tx:
        lineage = await tx.get(
            "clarification_lineages", operation_key(admission.dispatch.intent.question.lineage)
        )
        assert lineage["usage"]["service_turns"] == 1
        assert lineage["usage"]["pending"] == 2  # question decision plus service obligation
        before = await store._anchor(tx, initial, REDACTOR)
    dispatch = admission.dispatch.model_copy(
        update={
            "intent": admission.dispatch.intent.model_copy(
                update={
                    "operation": initial.operation("second-service"),
                    "invocation_id": temporary_service_invocation_id(
                        initial.operation("second-service")
                    ),
                }
            )
        }
    )
    registration = admission.permit.intent.request.model_copy(
        update={
            "operation": initial.operation("second-permit"),
            "settlement_operation": initial.operation("second-settlement"),
            "source_operation": dispatch.intent.operation,
            "admission_commitment": continuation_digest(dispatch),
        }
    )
    permit = admission.permit.model_copy(
        update={
            "operation": registration.operation,
            "intent": admission.permit.intent.model_copy(update={"request": registration}),
        }
    )
    with pytest.raises(CollaborationContractError):
        await register(stores(), dispatch, permit)
    assert (
        await store._lookup_registered_permit(initial, permit.operation, redactor=REDACTOR) is None
    )
    async with store._transaction(initial.binding.application_scope, write=False) as tx:
        assert await store._anchor(tx, initial, REDACTOR) == before
    # Closing the question decision is not evidence that dispatched service is
    # quiescent. Its independent responsibility must remain discoverable.
    from cayu.collaboration._clarification_commands import (
        ClarificationCloseCommand,
        ClarificationOpenReceipt,
    )
    from cayu.collaboration._clarification_state import clarification_commitment
    from cayu.collaboration._clarification_store import close_in_transaction
    from cayu.collaboration._preparation import prepare_contract

    question = admission.dispatch.intent.question
    async with store._transaction(initial.binding.application_scope, write=True) as tx:
        opening = prepare_contract(
            ClarificationOpenReceipt,
            await tx.get("operations", operation_key(question.operation)),
            redactor=REDACTOR,
        )
        await close_in_transaction(
            store,
            tx,
            initial,
            ClarificationCloseCommand(
                operation=initial.operation("close-question"),
                expected=opening.command.expected,
                question=question.operation,
                question_sha256=clarification_commitment(question, REDACTOR),
                kind="cancelled",
                initiator=question.initiator,
            ),
            redactor=REDACTOR,
        )
    reopened = stores()
    async with reopened._transaction(initial.binding.application_scope, write=False) as tx:
        assert await discover_services_in_transaction(
            reopened, tx, initial, after=None, limit=1, redactor=REDACTOR
        ) == (record,)
        lineage = await tx.get("clarification_lineages", operation_key(question.lineage))
        assert lineage["usage"]["pending"] == 1
