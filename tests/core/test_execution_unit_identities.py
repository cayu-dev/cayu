from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic import ValidationError

from cayu.runtime import (
    BudgetLimitIdentity,
    ModelAttemptIdentity,
    ModelStepIdentity,
    copy_model_attempt_identity,
    copy_model_step_identity,
    new_model_step_identity,
)


def test_model_execution_identities_are_opaque_distinct_and_linked() -> None:
    first_step = new_model_step_identity()
    second_step = new_model_step_identity()
    first_attempt = first_step.new_attempt()
    retry_attempt = first_step.new_attempt()

    assert first_step.model_step_id.startswith("mstep_")
    assert first_step.model_step_id != second_step.model_step_id
    assert first_attempt.model_attempt_id.startswith("matt_")
    assert first_attempt.model_attempt_id != retry_attempt.model_attempt_id
    assert first_attempt.model_step_id == retry_attempt.model_step_id == first_step.model_step_id
    assert first_attempt.payload() == {
        "model_step_id": first_step.model_step_id,
        "model_attempt_id": first_attempt.model_attempt_id,
    }


@pytest.mark.parametrize(
    ("model", "field_name"),
    (
        (ModelStepIdentity, "model_step_id"),
        (ModelAttemptIdentity, "model_step_id"),
        (ModelAttemptIdentity, "model_attempt_id"),
        (BudgetLimitIdentity, "budget_limit_id"),
    ),
)
def test_execution_identity_fields_reject_noncanonical_values(model, field_name: str) -> None:
    valid = {
        "model_step_id": f"mstep_{'a' * 32}",
        "model_attempt_id": f"matt_{'b' * 32}",
        "budget_limit_id": f"blim_{'c' * 64}",
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

    assert copy_model_step_identity(step) == step
    assert copy_model_step_identity(step) is not step
    assert copy_model_attempt_identity(attempt) == attempt
    assert copy_model_attempt_identity(attempt) is not attempt

    with pytest.raises(TypeError):
        copy_model_step_identity(attempt)
    with pytest.raises(TypeError):
        copy_model_attempt_identity(cast("Any", step))


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
