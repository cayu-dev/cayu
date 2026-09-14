from __future__ import annotations

import warnings

import pytest

from cayu.tools.base import ToolSpec
from cayu.tools.inference import (
    AuxiliaryInferencePolicy,
    InferenceLimits,
    copy_auxiliary_inference_policy,
    copy_inference_limits,
)


def limits(**changes):
    return InferenceLimits(
        **{
            "max_input_tokens": 100,
            "max_output_tokens": 50,
            "timeout_seconds": 10,
            **changes,
        }
    )


@pytest.mark.parametrize("field", tuple(InferenceLimits.model_fields))
@pytest.mark.parametrize("value", [True, False, "1", 0, -1, None])
def test_limits_reject_invalid_scalar_fields(field, value):
    with pytest.raises(ValueError):
        limits(**{field: value})


@pytest.mark.parametrize("value", [float("inf"), float("nan"), 3601, 10**400])
def test_timeout_is_finite_and_bounded(value):
    with pytest.raises(ValueError):
        limits(timeout_seconds=value)


@pytest.mark.parametrize("field", tuple(InferenceLimits.model_fields))
def test_request_cannot_raise_any_application_limit(field):
    ceiling = limits()
    with pytest.raises(ValueError, match="application-owned"):
        limits(**{field: getattr(ceiling, field) + 1}).bounded_by(ceiling)
    copied = ceiling.bounded_by(ceiling)
    assert copied == ceiling
    assert copied is not ceiling


def test_policy_is_detached_canonical_and_reconstructable():
    original = limits()
    purposes = ["tool.summary", "tool.extract"]
    policy = AuxiliaryInferencePolicy(limits=original, purposes=purposes)
    purposes.clear()
    assert policy.limits is not original
    assert policy.purposes == ("tool.extract", "tool.summary")
    assert AuxiliaryInferencePolicy.model_validate_json(policy.model_dump_json()) == policy
    assert copy_auxiliary_inference_policy(policy) == policy


@pytest.mark.parametrize(
    "purposes", [[], ["tool.x", "tool.x"], ["x"] * 33, ["X"], ["x..y"], [True]]
)
def test_policy_rejects_ambiguous_purposes(purposes):
    with pytest.raises(ValueError):
        AuxiliaryInferencePolicy(limits=limits(), purposes=purposes)


def test_post_construction_mutation_is_revalidated_without_serializer_warnings(capsys, caplog):
    class Hostile:
        def __repr__(self):
            raise AssertionError("Untrusted repr was invoked")

        def __str__(self):
            raise AssertionError("Untrusted str was invoked")

    mutated = limits()
    object.__setattr__(mutated, "max_input_tokens", Hostile())
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with pytest.raises(ValueError) as failure:
            copy_inference_limits(mutated)
        assert "input_value" not in str(failure.value)
    assert not captured
    assert not caplog.records
    assert capsys.readouterr() == ("", "")


def test_declarations_cannot_carry_runtime_identity():
    with pytest.raises(ValueError):
        InferenceLimits(**limits().model_dump(), principal="forged")
    with pytest.raises(ValueError):
        AuxiliaryInferencePolicy(limits=limits(), purposes=["tool.summary"], provider="forged")


def test_tool_spec_copies_and_reconstructs_inference_declaration():
    policy = AuxiliaryInferencePolicy(limits=limits(), purposes=["tool.summary"])
    spec = ToolSpec(name="summarize", auxiliary_inference=policy)
    assert spec.auxiliary_inference == policy
    assert spec.auxiliary_inference is not policy
    assert spec.auxiliary_inference.limits is not policy.limits
    assert ToolSpec.model_validate_json(spec.model_dump_json()) == spec
    copied = spec.model_copy()
    assert copied == spec
    assert copied.auxiliary_inference is not spec.auxiliary_inference
    object.__setattr__(policy.limits, "max_output_tokens", 999)
    assert spec.auxiliary_inference.limits.max_output_tokens == 50


@pytest.mark.parametrize("value", [True, "enabled", [], {"purposes": ["tool.summary"]}])
def test_tool_spec_rejects_invalid_inference_declaration(value):
    with pytest.raises(ValueError):
        ToolSpec(name="summarize", auxiliary_inference=value)


def test_tool_spec_copy_revalidates_mutated_policy_before_serialization(capsys, caplog):
    class Hostile:
        def __repr__(self):
            raise AssertionError("Untrusted repr was invoked")

    spec = ToolSpec(
        name="summarize",
        auxiliary_inference=AuxiliaryInferencePolicy(limits=limits(), purposes=["tool.summary"]),
    )
    object.__setattr__(spec.auxiliary_inference.limits, "max_input_tokens", Hostile())
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with pytest.raises(ValueError):
            spec.model_copy()
    assert not captured
    assert not caplog.records
    assert capsys.readouterr() == ("", "")
