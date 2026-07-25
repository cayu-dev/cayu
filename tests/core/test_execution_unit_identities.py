from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic import ValidationError

from cayu.core import ToolCallPart, ToolResultPart
from cayu.runtime import (
    BudgetLimitIdentity,
    ModelAttemptIdentity,
    ModelStepIdentity,
    ToolRoundIdentity,
    copy_model_attempt_identity,
    copy_model_step_identity,
    copy_tool_round_identity,
    new_model_step_identity,
)


def test_model_execution_identities_are_opaque_distinct_and_linked() -> None:
    first_step = new_model_step_identity()
    second_step = new_model_step_identity()
    first_attempt = first_step.new_attempt()
    retry_attempt = first_step.new_attempt()
    tool_round = first_attempt.new_tool_round()

    assert first_step.model_step_id.startswith("mstep_")
    assert first_step.model_step_id != second_step.model_step_id
    assert first_attempt.model_attempt_id.startswith("matt_")
    assert first_attempt.model_attempt_id != retry_attempt.model_attempt_id
    assert first_attempt.model_step_id == retry_attempt.model_step_id == first_step.model_step_id
    assert tool_round.tool_round_id.startswith("tround_")
    assert tool_round.model_attempt_id == first_attempt.model_attempt_id
    assert first_attempt.payload() == {
        "model_step_id": first_step.model_step_id,
        "model_attempt_id": first_attempt.model_attempt_id,
    }
    assert tool_round.payload() == {
        "model_step_id": first_step.model_step_id,
        "model_attempt_id": first_attempt.model_attempt_id,
        "tool_round_id": tool_round.tool_round_id,
    }
    assert tool_round.matches_payload(tool_round.payload()) is True
    assert (
        tool_round.matches_payload(
            {
                **tool_round.payload(),
                "model_attempt_id": retry_attempt.model_attempt_id,
            }
        )
        is False
    )


@pytest.mark.parametrize(
    ("model", "field_name"),
    (
        (ModelStepIdentity, "model_step_id"),
        (ModelAttemptIdentity, "model_step_id"),
        (ModelAttemptIdentity, "model_attempt_id"),
        (ToolRoundIdentity, "tool_round_id"),
        (BudgetLimitIdentity, "budget_limit_id"),
    ),
)
def test_execution_identity_fields_reject_noncanonical_values(model, field_name: str) -> None:
    valid = {
        "model_step_id": f"mstep_{'a' * 32}",
        "model_attempt_id": f"matt_{'b' * 32}",
        "budget_limit_id": f"blim_{'c' * 64}",
        "tool_round_id": f"tround_{'d' * 32}",
    }
    for invalid in (
        "",
        " value ",
        "workload-secret-value",
        valid[field_name].upper(),
        valid[field_name][:-1],
    ):
        values = dict(valid)
        if model is ModelStepIdentity:
            values = {"model_step_id": values["model_step_id"]}
        elif model is BudgetLimitIdentity:
            values = {"budget_limit_id": values["budget_limit_id"]}
        values[field_name] = invalid
        with pytest.raises(ValidationError):
            model(**values)


def test_execution_identity_copies_revalidate_exact_runtime_types() -> None:
    step = new_model_step_identity()
    attempt = step.new_attempt()
    tool_round = attempt.new_tool_round()

    assert copy_model_step_identity(step) == step
    assert copy_model_step_identity(step) is not step
    assert copy_model_attempt_identity(attempt) == attempt
    assert copy_model_attempt_identity(attempt) is not attempt
    assert copy_tool_round_identity(tool_round) == tool_round
    assert copy_tool_round_identity(tool_round) is not tool_round

    with pytest.raises(TypeError):
        copy_model_step_identity(attempt)
    with pytest.raises(TypeError):
        copy_model_attempt_identity(cast("Any", step))
    with pytest.raises(TypeError):
        copy_tool_round_identity(cast("Any", attempt))


def test_execution_identity_models_are_frozen_and_forbid_extra_fields() -> None:
    step = new_model_step_identity()

    with pytest.raises(ValidationError):
        step.model_step_id = f"mstep_{'d' * 32}"
    with pytest.raises(ValidationError):
        ModelStepIdentity.model_validate(
            {
                "model_step_id": f"mstep_{'a' * 32}",
                "provider_supplied_identity": "spoofed",
            }
        )


@pytest.mark.parametrize("part_type", (ToolCallPart, ToolResultPart))
def test_transcript_tool_parts_share_the_canonical_tool_round_contract(part_type) -> None:
    identity = new_model_step_identity().new_attempt().new_tool_round()
    part = part_type(
        tool_call_id="call_1",
        tool_name="echo",
        **identity.payload(),
    )

    reconstructed = ToolRoundIdentity.model_validate(
        {
            "model_step_id": part.model_step_id,
            "model_attempt_id": part.model_attempt_id,
            "tool_round_id": part.tool_round_id,
        }
    )

    assert reconstructed == identity


@pytest.mark.parametrize("part_type", (ToolCallPart, ToolResultPart))
@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("model_step_id", "mstep_transcript_1"),
        ("model_attempt_id", "matt_transcript_1"),
        ("tool_round_id", "tround_transcript_1"),
    ),
)
def test_transcript_tool_parts_reject_noncanonical_execution_identities(
    part_type,
    field_name: str,
    invalid_value: str,
) -> None:
    identity_payload = new_model_step_identity().new_attempt().new_tool_round().payload()
    identity_payload[field_name] = invalid_value

    with pytest.raises(ValidationError, match="not a valid Cayu execution-unit identifier"):
        part_type(
            tool_call_id="call_1",
            tool_name="echo",
            **identity_payload,
        )
