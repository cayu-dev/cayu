"""Public service selection contracts; runtime acceptance is tested separately."""

import pytest
from tests.core.test_clarification_contracts import operation
from tests.core.test_clarification_deliveries import delivery_record
from tests.core.test_temporary_continuation_contracts import selection

from cayu import ClarificationServiceReceipt, ClarificationServiceRequest
from cayu.collaboration._contracts import CollaborationContractError
from cayu.collaboration._preparation import prepare_contract
from cayu.collaboration.clarifications import MAX_CLARIFICATION_TEXT_BYTES
from cayu.runtime._session_continuation import continuation_digest
from cayu.vaults.redaction import SecretRedactor


def service_request():
    selected = selection()
    return ClarificationServiceRequest(
        operation=operation("service"),
        initiator=selected.initiator,
        delivery=delivery_record().intent,
        ticket=selected.ticket,
        service_generation=1,
        parent_service=None,
        instruction="Answer the delivered clarification.",
    )


def test_public_selection_reconstructs_and_binds_host_instruction():
    request = service_request()
    assert ClarificationServiceRequest.model_validate_json(request.model_dump_json()) == request
    changed = request.model_copy(update={"instruction": "Give a shorter answer."})
    changed = prepare_contract(ClarificationServiceRequest, changed, redactor=SecretRedactor())
    assert continuation_digest(changed) != continuation_digest(request)
    assert ClarificationServiceReceipt.__name__ == "ClarificationServiceReceipt"


@pytest.mark.parametrize(
    "instruction",
    (
        "",
        " ",
        "x" * (MAX_CLARIFICATION_TEXT_BYTES + 1),
        "é" * (MAX_CLARIFICATION_TEXT_BYTES // 2 + 1),
    ),
)
def test_host_instruction_is_nonblank_and_byte_bounded(instruction):
    with pytest.raises(CollaborationContractError):
        prepare_contract(
            ClarificationServiceRequest,
            service_request().model_copy(update={"instruction": instruction}),
            redactor=SecretRedactor(),
        )


def test_exact_utf8_instruction_ceiling_is_accepted():
    request = service_request().model_copy(
        update={"instruction": "é" * (MAX_CLARIFICATION_TEXT_BYTES // 2)}
    )
    assert (
        prepare_contract(ClarificationServiceRequest, request, redactor=SecretRedactor()) == request
    )


@pytest.mark.anyio
@pytest.mark.parametrize("field", ("instruction", "service_generation", "delivery_context"))
async def test_public_service_rejects_mutated_input_without_diagnostic_leak(
    field, monkeypatch, capsys, caplog, recwarn
):
    from cayu import CayuApp
    from cayu.collaboration.exports import SessionExportAccessContext

    canary = "private-clarification-diagnostic-canary"

    class Hostile:
        def __repr__(self):
            raise AssertionError(canary)

        __str__ = __repr__

    app = CayuApp()
    value = service_request().model_copy(update={"instruction": canary})
    context = SessionExportAccessContext(principal="operator")
    delivery_context = None
    if field == "delivery_context":
        delivery_context = context.model_copy(update={"principal": Hostile()})
    else:
        value = value.model_copy(update={field: Hostile()})

    async def no_dependency(*args, **kwargs):
        pytest.fail("Rejected caller input reached a durable or authorization dependency")

    monkeypatch.setattr(app._clarification_coordinator.requests, "_dependency", no_dependency)
    with pytest.raises(ValueError) as caught:
        await app.service_clarification(value, context=context, delivery_context=delivery_context)
    output = capsys.readouterr()
    diagnostics = str(caught.value) + repr(caught.value) + caplog.text + output.out + output.err
    diagnostics += "".join(str(item.message) for item in recwarn)
    assert canary not in diagnostics
    assert not app._clarification_coordinator.requests._owners.pending
