from __future__ import annotations

import warnings
from datetime import UTC, datetime, timedelta, timezone

import pytest

from cayu.runtime.tool_effects import (
    ToolEffectReceipt,
    copy_tool_effect_receipt,
    tool_effect_receipt_digest,
)


def _receipt(**changes) -> ToolEffectReceipt:
    fields = {
        "receipt_id": "deployment-1",
        "receipt_schema": "deployment",
        "receipt_schema_version": 1,
        "tool_call_id": "call-1",
        "idempotency_key": "key-1",
        "tool_name": "deploy",
        "outcome": "completed",
        "message": "Deployment observed.",
        "observed_at": datetime(2026, 9, 8, tzinfo=UTC),
        "source": "operator",
    }
    return ToolEffectReceipt(**(fields | changes))


def test_receipt_detaches_nested_values_and_canonicalizes_identity() -> None:
    structured = {"deployment": {"version": 2}}
    original = _receipt(structured=structured, resource_versions={"b": "2", "a": "1"})
    copied = copy_tool_effect_receipt(original)
    digest = tool_effect_receipt_digest(original)
    structured["deployment"]["version"] = 3
    assert original.structured == {"deployment": {"version": 2}}
    original.structured["deployment"]["version"] = 4
    assert copied.structured == {"deployment": {"version": 2}}
    assert digest == tool_effect_receipt_digest(copied)
    assert digest != tool_effect_receipt_digest(original)
    reordered = _receipt(structured=copied.structured, resource_versions={"a": "1", "b": "2"})
    assert tool_effect_receipt_digest(reordered) == digest


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": True},
        {"schema_version": 2},
        {"receipt_schema_version": True},
        {"tool_call_id": " "},
        {"tool_name": "é" * 129},
        {"message": "é" * 2049},
        {"outcome": "not_found"},
        {"source": "model"},
        {"observed_at": datetime(2026, 9, 8)},
        {"structured": {"raw": "x" * (64 * 1024)}},
        {"resource_versions": {str(n): "v" for n in range(33)}},
        {"integrity": {str(n): "v" for n in range(17)}},
        {"headers": {"authorization": "not allowed"}},
    ],
)
def test_receipt_rejects_invalid_or_unbounded_material(changes) -> None:
    with pytest.raises(ValueError):
        _receipt(**changes)


def test_observation_has_one_utc_representation() -> None:
    offset = timezone(timedelta(hours=2))
    local = _receipt(observed_at=datetime(2026, 9, 8, 2, tzinfo=offset))
    assert local.observed_at == datetime(2026, 9, 8, tzinfo=UTC)
    assert tool_effect_receipt_digest(local) == tool_effect_receipt_digest(_receipt())


def test_copy_does_not_serialize_or_format_mutated_hostile_values(capsys, caplog) -> None:
    class Hostile:
        def __repr__(self) -> str:
            pytest.fail("Rejected receipt value was formatted.")

        def __str__(self) -> str:
            pytest.fail("Rejected receipt value was formatted.")

    receipt = _receipt()
    unsafe = receipt.model_copy(update={"structured": {"value": Hostile()}})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError):
            copy_tool_effect_receipt(unsafe)
    assert caught == []
    assert capsys.readouterr() == ("", "")
    assert not caplog.records


def test_user_input_receipt_request_rejects_hostile_review_without_diagnostics(capsys, caplog):
    from cayu.runtime.human_review import HumanReviewContext, HumanReviewReference
    from cayu.runtime.tool_effects import ToolEffectReconciliationRequest
    from cayu.runtime.user_input import UserInputResponse

    class Hostile:
        def __repr__(self):
            return "private-review-canary"

        def __str__(self):
            return "private-review-canary"

    reference = HumanReviewReference(
        context=HumanReviewContext(recipient="operator", purpose="recovery"),
        policy_version="1",
        content_tag="a" * 64,
    )
    response = UserInputResponse(session_id="session", input_id="input", answer="yes")
    response.review_reference = reference.model_copy(
        update={"context": reference.context.model_copy(update={"recipient": Hostile()})}
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises((ValueError, TypeError)) as raised:
            ToolEffectReconciliationRequest(
                session_id="session",
                session_instance_id="instance",
                tool_round_id="round",
                tool_call_id="call",
                tool_name="deploy",
                idempotency_key="key",
                expected_run_epoch=1,
                expected_revision=1,
                lookup=True,
                user_input_response=response,
            )
    assert "private-review-canary" not in str(raised.value)
    assert caught == []
    assert capsys.readouterr() == ("", "")
    assert not caplog.records


def test_mutated_valid_sibling_is_checked_even_when_another_field_is_invalid() -> None:
    unsafe = _receipt().model_copy(update={"schema_version": True, "integrity": {"bad": object()}})
    with pytest.raises(ValueError):
        copy_tool_effect_receipt(unsafe)


@pytest.mark.parametrize(
    "changes",
    [
        {"receipt_id": "deployment-2"},
        {"receipt_schema": "another-schema"},
        {"receipt_schema_version": 2},
        {"tool_call_id": "call-2"},
        {"idempotency_key": "key-2"},
        {"tool_name": "another-tool"},
        {"external_system": "another-system"},
        {"outcome": "failed"},
        {"message": "Different evidence."},
        {"structured": {"version": 2}},
        {"resource_versions": {"resource": "2"}},
        {"observed_at": datetime(2026, 9, 9, tzinfo=UTC)},
        {"source": "adapter"},
        {"integrity": {"signature": "different"}},
    ],
)
def test_every_receipt_field_participates_in_content_identity(changes) -> None:
    assert tool_effect_receipt_digest(_receipt(**changes)) != tool_effect_receipt_digest(_receipt())


def test_individually_bounded_fields_cannot_exceed_total_envelope_bound() -> None:
    with pytest.raises(ValueError):
        _receipt(
            structured={"value": "x" * 60_000},
            resource_versions={str(n): "v" * 1024 for n in range(32)},
            integrity={str(n): "v" * 1024 for n in range(16)},
        )
