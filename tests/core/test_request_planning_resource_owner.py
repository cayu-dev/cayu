"""Native resource adapter uses registered owners and exact permit expectations."""

import time
from hashlib import sha256

import pytest
from tests.artifacts.test_resource_transfer_templates import registered_template
from tests.artifacts.test_resources import make_store
from tests.core.test_collaboration_request_foundation import setup
from tests.core.test_participant_identity import CONTEXT, app, registration
from tests.core.test_prepared_admission_public import PreparationResolver

from cayu.collaboration._contracts import CollaborationContractError
from cayu.collaboration._planning_preflight import preflight_stage_terminal
from cayu.collaboration._planning_records import (
    RequestPlanningEvent,
    RequestPlanningStageIntent,
    RequestPlanningStageRecord,
)
from cayu.collaboration._planning_resource_types import (
    RequestResourceAcquisitionStageCommand,
    RequestResourceStageRelease,
    RequestResourceTransferStageCommand,
)
from cayu.collaboration._preparation import contract_bytes, prepare_contract
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.request_access import RequestRegistration
from cayu.collaboration.resource_preparation import RequestPlanningResource
from cayu.sessions._planning_resource_owner import NativePlanningResourceOwner
from cayu.vaults.redaction import SecretRedactor

pytestmark = pytest.mark.anyio


@pytest.fixture
def artifact_store(tmp_path):
    return make_store(tmp_path)


async def test_native_planning_resource_adapter_retains_exact_registered_owners(
    tmp_path, artifact_store
):
    store, artifact = artifact_store
    (
        source,
        _command,
        permit,
        destination,
        template,
        reader,
        resolver,
        source_ledger,
        ledger,
    ) = await registered_template(
        tmp_path, store, artifact, planning_deadline=int(time.time() * 1000) + 60_000
    )
    expected = RequestPlanningResource(
        acquisition_permit=permit,
        transfer=template,
        transfer_permit=reader._responsibilities[0][1],
    )
    application = app(
        ledger,
        registration(scope=destination.owner.application_scope),
        collaboration_requests=RequestRegistration(
            mandates=resolver, max_ttl_ms=60_000, resource_owners=(source, destination)
        ),
    )
    await application.initialize_collaboration()
    owner = NativePlanningResourceOwner(application)
    try:
        _, initialized, _, _, request, _ = await setup(
            ledger, reg=registration(scope=destination.owner.application_scope)
        )
        recipient = expected.transfer_permit.intent.request.participant
        request = request.model_copy(update={"target": recipient})
        request_resolver = PreparationResolver(request, recipient)
        accepting = app(
            ledger,
            registration(scope=destination.owner.application_scope),
            collaboration_requests=RequestRegistration(
                mandates=request_resolver, max_ttl_ms=60_000
            ),
        )
        await accepting.initialize_collaboration()
        accepted = await accepting.accept_collaboration_request(
            request, context=request_resolver.sender.context
        )
        acquisition_stage = RequestResourceAcquisitionStageCommand(
            operation=initialized.operation("resource-acquisition-stage"),
            expected=accepted.expected,
            resource=expected,
        )
        acquired = await owner.acquire(expected, context=CONTEXT)
        assert await owner.acquire(expected, context=CONTEXT) == acquired
        changed = expected.model_copy(
            update={
                "acquisition_permit": permit.model_copy(
                    update={
                        "intent": permit.intent.model_copy(
                            update={
                                "request": permit.intent.request.model_copy(
                                    update={"expected_lifecycle_revision": 2}
                                )
                            }
                        )
                    }
                )
            }
        )
        before = source._journal.path.read_bytes()
        with pytest.raises(CollaborationUnavailable):
            await owner.acquire(changed, context=CONTEXT)
        assert source._journal.path.read_bytes() == before
        transferred = await owner.transfer(expected, acquired.receipt, context=CONTEXT)
        assert transferred.preparation.permit == expected.transfer_permit
        material = await destination.read_material(transferred.reference)
        assert material.transfer == transferred.receipt
        assert material.preparation == transferred.preparation
        assert await owner.transfer(expected, acquired.receipt, context=CONTEXT) == transferred
        transfer_stage = RequestResourceTransferStageCommand(
            operation=initialized.operation("resource-transfer-stage"),
            expected=accepted.expected,
            resource=expected,
            acquisition=acquired.receipt,
        )
        source_before = source._journal.path.read_bytes()
        destination_before = destination._journal.path.read_bytes()
        with pytest.raises(CollaborationUnavailable):
            await owner.discharge_stage(transfer_stage.model_dump(mode="json"), context=CONTEXT)
        for invalid in (
            transfer_stage.model_copy(update={"operation": expected.transfer_permit.operation}),
            transfer_stage.model_copy(
                update={
                    "resource": expected.model_copy(
                        update={
                            "transfer_permit": expected.transfer_permit.model_copy(
                                update={
                                    "intent": expected.transfer_permit.intent.model_copy(
                                        update={
                                            "request": expected.transfer_permit.intent.request.model_copy(
                                                update={"expected_lifecycle_revision": 2}
                                            )
                                        }
                                    )
                                }
                            )
                        }
                    )
                }
            ),
            transfer_stage.model_copy(
                update={
                    "operation": transfer_stage.operation.model_copy(
                        update={"generation": transfer_stage.operation.generation + 1}
                    )
                }
            ),
        ):
            with pytest.raises(CollaborationContractError):
                await owner.discharge_stage(invalid, context=CONTEXT)
            assert source._journal.path.read_bytes() == source_before
            assert destination._journal.path.read_bytes() == destination_before
        # Durable command reconstruction does not authenticate caller-shaped
        # settlement: the registered owner still obtains native positive proof.
        for stage in (transfer_stage, acquisition_stage):
            reconstructed = type(stage).model_validate_json(stage.model_dump_json())
            before = await owner.discharge_stage(reconstructed, context=CONTEXT)
            assert before.receipt.receiving.proves_exclusion
            assert before.receipt.command == stage
            assert (
                RequestResourceStageRelease.model_validate_json(before.receipt.model_dump_json())
                == before.receipt
            )
            assert await owner.discharge_stage(reconstructed, context=CONTEXT) == before
            assert_stage_envelope(initialized, before.receipt)
        await store.delete(artifact.id)
    finally:
        await source.drain()
        await destination.drain()
        await source_ledger.close()
        await ledger.close()


def assert_stage_envelope(initialized, receipt, *, expected_plan=None, ordinal=1):
    """Schema/size evidence only; native stage retention is qualified separately."""
    redactor = SecretRedactor()
    command = receipt.command
    selected = command.expected.intent.selection
    intent = RequestPlanningStageIntent(
        operation=initialized.operation("local-stage"),
        plan=initialized.operation("resource-plan"),
        plan_sha256="a" * 64,
        ordinal=ordinal,
        command=command,
    )
    event = RequestPlanningEvent(
        id="a" * 32,
        sequence=1,
        operation=intent.operation,
        plan=intent.plan,
        request=selected.reference,
        type="plan_stage_retained",
        commitment=sha256(contract_bytes(intent, redactor=redactor)).hexdigest(),
        participants=tuple(
            dict.fromkeys((selected.sender.reference, selected.recipient.reference))
        ),
    )
    pending = RequestPlanningStageRecord(
        intent=intent,
        state="pending",
        registered_at_ms=selected.accepted_at_ms,
        settled_at_ms=None,
        receipt=None,
        registration_event=event,
        settlement_event=None,
        reserved_bytes=3 * 65536,
    )
    assert (
        preflight_stage_terminal(
            pending, initialized, ceiling=65536, redactor=redactor, expected_plan=expected_plan
        )
        <= 65536
    )
    terminal = pending.model_copy(
        update={
            "state": "settled",
            "settled_at_ms": selected.accepted_at_ms + 1,
            "receipt": receipt,
            "reserved_bytes": 0,
            "settlement_event": event.model_copy(
                update={
                    "id": "b" * 32,
                    "sequence": 2,
                    "type": "plan_stage_settled",
                    "commitment": sha256(contract_bytes(receipt, redactor=redactor)).hexdigest(),
                }
            ),
        }
    )
    assert RequestPlanningStageRecord.model_validate_json(terminal.model_dump_json()) == terminal
    wrong_mode = (
        "clarification_open" if receipt.command.mode == "request_admission" else "request_admission"
    )
    for mode in (None, True, "future_command", wrong_mode):
        document = terminal.model_dump(mode="json")
        document["receipt"]["command"]["mode"] = mode
        with pytest.raises(CollaborationContractError):
            prepare_contract(RequestPlanningStageRecord, document, redactor=redactor)
