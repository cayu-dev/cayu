"""Bounded maintenance representations, not receiving authorization tests."""

import pytest
from tests.core.test_clarification_service_api import service_request

from cayu import (
    ClarificationDeliveryRecovery,
    ClarificationDueQuestion,
    ClarificationDueQuestionPage,
    ClarificationExpiryRequest,
    ClarificationPendingDelivery,
    ClarificationPendingDeliveryPage,
    ClarificationPendingService,
    ClarificationPendingServicePage,
    ClarificationQuestionRecovery,
    ClarificationServiceRecovery,
)
from cayu.collaboration._clarification_deliveries import ClarificationDeliveryIntent
from cayu.collaboration._clarification_recovery import ServiceRecoveryInput
from cayu.collaboration._clarification_recovery_types import ClarificationPendingServiceQuery
from cayu.collaboration._clarification_service_api import ClarificationServiceRequest
from cayu.collaboration._contracts import CollaborationContractError
from cayu.collaboration._preparation import prepare_contract
from cayu.runtime._session_continuation import continuation_digest
from cayu.vaults.redaction import SecretRedactor


def recovery():
    request = service_request()
    return ClarificationServiceRecovery(
        operation=request.operation,
        session_id=request.ticket.session_id,
        session_instance_id=request.ticket.session_instance_id,
        selection_sha256=continuation_digest(request),
        dispatch_sha256="d" * 64,
    )


def test_question_expiry_is_exact_and_not_a_service_request():
    question = service_request().delivery.question
    selector = ClarificationQuestionRecovery(
        operation=question.operation,
        question_sha256="a" * 64,
        request_sha256="b" * 64,
    )
    expiry = ClarificationExpiryRequest(
        operation=question.operation.model_copy(update={"caller_key": "expiry"}), recovery=selector
    )
    assert ClarificationExpiryRequest.model_validate_json(expiry.model_dump_json()) == expiry
    for operation in (
        question.operation,
        expiry.operation.model_copy(update={"generation": expiry.operation.generation + 1}),
        expiry.operation.model_copy(update={"namespace_incarnation": "other"}),
        expiry.operation.model_copy(update={"application_scope": "other"}),
    ):
        with pytest.raises(CollaborationContractError):
            prepare_contract(
                ClarificationExpiryRequest,
                expiry.model_copy(update={"operation": operation}),
                redactor=SecretRedactor(),
            )
    with pytest.raises(CollaborationContractError):
        prepare_contract(ClarificationServiceRequest, selector, redactor=SecretRedactor())


@pytest.mark.parametrize("count", (0, 1, 63, 64, 65))
def test_question_due_page_is_bounded(count):
    question = service_request().delivery.question
    item = ClarificationDueQuestion(
        recovery=ClarificationQuestionRecovery(
            operation=question.operation, question_sha256="a" * 64, request_sha256="b" * 64
        ),
        deadline_at_ms=question.deadline_at_ms,
    )
    if count > 64:
        with pytest.raises(ValueError):
            ClarificationDueQuestionPage(items=(item,) * count, next_cursor=None)
    else:
        assert (
            len(ClarificationDueQuestionPage(items=(item,) * count, next_cursor=None).items)
            == count
        )


def test_delivery_recovery_is_not_an_executable_intent():
    intent = service_request().delivery
    selected = ClarificationDeliveryRecovery(
        operation=intent.operation, intent_sha256=continuation_digest(intent)
    )
    assert ClarificationDeliveryRecovery.model_validate_json(selected.model_dump_json()) == selected
    with pytest.raises(CollaborationContractError):
        prepare_contract(ClarificationDeliveryIntent, selected, redactor=SecretRedactor())
    assert "payload" not in selected.model_dump_json()


@pytest.mark.parametrize("count", (0, 1, 63, 64, 65))
def test_delivery_discovery_page_is_bounded(count):
    intent = service_request().delivery
    item = ClarificationPendingDelivery(
        recovery=ClarificationDeliveryRecovery(
            operation=intent.operation, intent_sha256=continuation_digest(intent)
        ),
        question=intent.question.operation,
        deadline_at_ms=intent.append.attempt_key.deadline_at_ms,
    )
    if count > 64:
        with pytest.raises(ValueError):
            ClarificationPendingDeliveryPage(items=(item,) * count, next_cursor=None)
    else:
        assert (
            len(ClarificationPendingDeliveryPage(items=(item,) * count, next_cursor=None).items)
            == count
        )


def test_recovery_selector_roundtrip_is_not_an_execution_request():
    selected = recovery()
    assert ClarificationServiceRecovery.model_validate_json(selected.model_dump_json()) == selected
    assert (
        ServiceRecoveryInput.model_validate_json(
            ServiceRecoveryInput(request=selected).model_dump_json()
        ).request
        == selected
    )
    with pytest.raises(CollaborationContractError):
        prepare_contract(ClarificationServiceRequest, selected, redactor=SecretRedactor())
    assert "instruction" not in selected.model_dump_json()


@pytest.mark.parametrize("limit", (1, 32, 64))
def test_discovery_limit_accepts_finite_boundaries(limit):
    assert ClarificationPendingServiceQuery(limit=limit).limit == limit


@pytest.mark.parametrize("limit", (0, 65, -1, True, False, "32", 32.0, None))
def test_discovery_limit_rejects_invalid_values(limit):
    with pytest.raises(ValueError):
        ClarificationPendingServiceQuery(limit=limit)


@pytest.mark.parametrize("count", (0, 1, 63, 64, 65))
def test_discovery_page_has_a_hard_item_ceiling(count):
    item = ClarificationPendingService(
        recovery=recovery(),
        question=service_request().delivery.question.operation,
        deadline_at_ms=1,
    )
    if count > 64:
        with pytest.raises(ValueError):
            ClarificationPendingServicePage(items=(item,) * count, next_cursor=None)
    else:
        assert (
            len(ClarificationPendingServicePage(items=(item,) * count, next_cursor=None).items)
            == count
        )


@pytest.mark.parametrize(
    "field,value",
    (
        ("kind", "future"),
        ("selection_sha256", None),
        ("selection_sha256", "x" * 64),
        ("dispatch_sha256", True),
        ("session_instance_id", False),
    ),
)
def test_recovery_selector_rejects_ambiguous_fields(field, value):
    with pytest.raises(CollaborationContractError):
        prepare_contract(
            ClarificationServiceRecovery,
            recovery().model_copy(update={field: value}),
            redactor=SecretRedactor(),
        )
