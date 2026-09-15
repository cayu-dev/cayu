from __future__ import annotations

import asyncio
import warnings
from dataclasses import replace

import pytest

from cayu import ModelFailoverPolicy, ModelTarget
from cayu.providers.base import ModelProviderError
from cayu.runtime._model_failover import (
    FailoverDisposition,
    FailoverObservation,
    FailoverSuppression,
    decide_model_failover,
)
from cayu.runtime.retry_policy import RetryPolicy, RetrySuppression, retry_decision
from cayu.sessions.base import copy_model_failover_policy


def test_failover_public_policy_detaches_ordered_targets():
    primary = ModelTarget(provider_name="primary", model="small")
    first = ModelTarget(provider_name="backup", model="large")
    second = {"provider_name": "backup", "model": "larger"}
    supplied = [first, second]
    policy = ModelFailoverPolicy.model_validate({"fallbacks": supplied})
    supplied.clear()
    second["model"] = "changed"
    object.__setattr__(first, "model", "changed")

    targets = policy.resolve_targets(primary)
    assert [(target.provider_name, target.model) for target in targets] == [
        ("primary", "small"),
        ("backup", "large"),
        ("backup", "larger"),
    ]
    assert targets[0] is not primary
    assert targets[1] is not policy.fallbacks[0]
    assert policy.max_total_attempts == 20
    assert ModelFailoverPolicy.model_validate_json(policy.model_dump_json()) == policy
    with pytest.raises(ValueError):
        policy.max_total_attempts = 80


@pytest.mark.parametrize("attempts", [1, 20, 80])
def test_failover_total_attempt_bounds(attempts):
    policy = ModelFailoverPolicy(
        fallbacks=(ModelTarget(provider_name="backup", model="model"),),
        max_total_attempts=attempts,
    )
    assert copy_model_failover_policy(policy).max_total_attempts == attempts


@pytest.mark.parametrize("attempts", [True, False, 0, -1, 81, 1.0, "2", None])
def test_failover_rejects_invalid_attempt_bounds(attempts):
    with pytest.raises(ValueError):
        ModelFailoverPolicy(
            fallbacks=(ModelTarget(provider_name="backup", model="model"),),
            max_total_attempts=attempts,
        )


@pytest.mark.parametrize("fallbacks", [[], (), None, "backup", {}, iter(())])
def test_failover_requires_bounded_explicit_sequence(fallbacks):
    with pytest.raises(ValueError):
        ModelFailoverPolicy(fallbacks=fallbacks)


def test_failover_rejects_duplicate_primary_and_fallbacks():
    target = ModelTarget(provider_name="backup", model="model")
    with pytest.raises(ValueError, match="distinct"):
        ModelFailoverPolicy(fallbacks=(target, target))
    policy = ModelFailoverPolicy(fallbacks=(target,))
    with pytest.raises(ValueError, match="primary"):
        policy.resolve_targets(target)


def test_failover_candidate_count_bound():
    targets = tuple(ModelTarget(provider_name="provider", model=f"model-{i}") for i in range(8))
    policy = ModelFailoverPolicy(fallbacks=targets[:7])
    assert len(policy.resolve_targets(targets[7])) == 8
    with pytest.raises(ValueError):
        ModelFailoverPolicy(fallbacks=targets)


@pytest.mark.parametrize(
    "value",
    [
        {"model": "model"},
        {"provider_name": "provider"},
        {"provider_name": "provider", "model": "model", "credentials": "invalid"},
        {"provider_name": True, "model": "model"},
        {"provider_name": "provider", "model": b"model"},
        {"provider_name": "provider", "model": " "},
        {"provider_name": " provider", "model": "model"},
        {"provider_name": "provider", "model": "\x00"},
        {"provider_name": "provider", "model": "\ud800"},
        {"provider_name": "provider", "model": "m" * 257},
        {"provider_name": "provider", "model": "é" * 129},
    ],
)
def test_failover_rejects_incomplete_or_unsafe_target(value):
    with pytest.raises(ValueError):
        ModelFailoverPolicy.model_validate({"fallbacks": [value]})


def test_failover_target_identity_byte_boundary():
    policy = ModelFailoverPolicy.model_validate(
        {"fallbacks": [{"provider_name": "p" * 256, "model": "é" * 128}]}
    )
    assert len(policy.fallbacks[0].model.encode()) == 256


def test_failover_revalidates_bypassed_values_without_diagnostic_disclosure(capsys, caplog):
    canary = "FAILOVER-SECRET-CANARY"

    class Hostile:
        def __repr__(self):
            return canary

        def __str__(self):
            return canary

    target = ModelTarget.model_construct(provider_name="backup", model=Hostile())
    policy = ModelFailoverPolicy.model_construct(fallbacks=(target,), max_total_attempts=2)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with pytest.raises(ValueError) as failure:
            copy_model_failover_policy(policy)
    assert canary not in str(failure.value)
    assert canary not in repr(failure.value)
    assert not captured
    assert canary not in caplog.text
    streams = capsys.readouterr()
    assert canary not in streams.out + streams.err


def test_failover_revalidates_mutated_frozen_policy():
    policy = ModelFailoverPolicy(fallbacks=(ModelTarget(provider_name="backup", model="model"),))
    object.__setattr__(policy, "max_total_attempts", True)
    with pytest.raises(ValueError):
        policy.resolve_targets(ModelTarget(provider_name="primary", model="model"))


def _decision_inputs():
    failure = ModelProviderError(
        "Service unavailable", provider="primary", status_code=503, retryable=True
    )
    return {
        "failure": failure,
        "provider_name": "primary",
        "retry": retry_decision(
            policy=RetryPolicy(max_attempts=1),
            attempt=1,
            error="Service unavailable",
            status_code=503,
            retryable=True,
        ),
        "observation": FailoverObservation(
            provider_name="primary",
            caller_cancelled=False,
            completion_observed=False,
            provider_effect_observed=False,
            provider_operation_owned=False,
            cleanup_settled=True,
        ),
        "candidate_index": 0,
        "candidate_count": 2,
        "attempts_used": 1,
        "max_total_attempts": 2,
    }


def test_failover_can_select_after_local_retry_is_disabled():
    assert (
        decide_model_failover(**_decision_inputs()).disposition is FailoverDisposition.SELECT_NEXT
    )


@pytest.mark.parametrize("field,value", [("attempts_used", 2), ("candidate_index", 1)])
def test_failover_exhaustion_does_not_grant_next_dispatch(field, value):
    inputs = _decision_inputs()
    inputs[field] = value
    assert decide_model_failover(**inputs).disposition is FailoverDisposition.EXHAUSTED


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("caller_cancelled", True, FailoverSuppression.CANCELLATION),
        ("completion_observed", True, FailoverSuppression.COMPLETION_OBSERVED),
        ("provider_effect_observed", True, FailoverSuppression.PROVIDER_EFFECT_OBSERVED),
        ("provider_operation_owned", True, FailoverSuppression.PROVIDER_OPERATION),
        ("cleanup_settled", False, FailoverSuppression.CLEANUP_UNSETTLED),
    ],
)
def test_failover_positive_safety_observations_override_retryable_status(field, value, reason):
    inputs = _decision_inputs()
    inputs["observation"] = replace(inputs["observation"], **{field: value})
    decision = decide_model_failover(**inputs)
    assert decision.disposition is FailoverDisposition.SUPPRESSED
    assert decision.suppression is reason


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504, 529])
def test_failover_recognizes_only_explicit_service_rejection(status):
    inputs = _decision_inputs()
    inputs["failure"] = ModelProviderError(
        "failure", provider="primary", status_code=status, retryable=True
    )
    inputs["retry"] = retry_decision(
        policy=RetryPolicy(max_attempts=1),
        attempt=1,
        error="failure",
        status_code=status,
        retryable=True,
    )
    assert decide_model_failover(**inputs).disposition is FailoverDisposition.SELECT_NEXT


@pytest.mark.parametrize(
    "status,retryable",
    [(None, True), (400, True), (401, True), (403, True), (429, False), (503, None)],
)
def test_failover_does_not_classify_from_error_text(status, retryable):
    inputs = _decision_inputs()
    inputs["failure"] = ModelProviderError(
        "HTTP 503 rate limit connection timeout",
        provider="primary",
        status_code=status,
        retryable=retryable,
    )
    assert decide_model_failover(**inputs).disposition is FailoverDisposition.SUPPRESSED


def test_failover_cannot_bypass_local_retry_schedule_or_suppression():
    inputs = _decision_inputs()
    inputs["retry"] = retry_decision(
        policy=RetryPolicy(max_attempts=5),
        attempt=1,
        error="failure",
        status_code=503,
        retryable=True,
    )
    assert decide_model_failover(**inputs).suppression is FailoverSuppression.RETRY_NOT_EXHAUSTED
    inputs["retry"] = retry_decision(
        policy=RetryPolicy(max_attempts=1),
        attempt=1,
        error="failure",
        status_code=503,
        retryable=True,
        suppression=RetrySuppression.AUTOMATIC_RETRY_DISABLED,
    )
    assert decide_model_failover(**inputs).suppression is FailoverSuppression.RETRY_SUPPRESSED


@pytest.mark.parametrize(
    "failure", [RuntimeError("HTTP 503"), asyncio.CancelledError(), GeneratorExit()]
)
def test_failover_does_not_unwrap_non_provider_failures(failure):
    inputs = _decision_inputs()
    failure.__cause__ = inputs["failure"]
    inputs["failure"] = failure
    assert decide_model_failover(**inputs).suppression is FailoverSuppression.NON_PROVIDER_FAILURE


def test_failover_does_not_treat_historical_cancel_cause_as_current_cancel():
    inputs = _decision_inputs()
    inputs["failure"].__cause__ = asyncio.CancelledError()
    assert decide_model_failover(**inputs).disposition is FailoverDisposition.SELECT_NEXT


def test_failover_rejects_wrong_provider_evidence():
    inputs = _decision_inputs()
    inputs["provider_name"] = "other"
    assert (
        decide_model_failover(**inputs).suppression
        is FailoverSuppression.PROVIDER_IDENTITY_MISMATCH
    )


def test_failover_uses_observed_execution_source_not_adapter_error_identity():
    inputs = _decision_inputs()
    inputs["failure"] = ModelProviderError(
        "Protocol error", provider="openai", status_code=503, retryable=True
    )
    assert decide_model_failover(**inputs).disposition is FailoverDisposition.SELECT_NEXT
    inputs["observation"] = replace(inputs["observation"], provider_name="another-registration")
    assert (
        decide_model_failover(**inputs).suppression
        is FailoverSuppression.PROVIDER_IDENTITY_MISMATCH
    )


@pytest.mark.parametrize(
    "field", ["candidate_index", "candidate_count", "attempts_used", "max_total_attempts"]
)
def test_failover_counter_booleans_are_not_authority(field):
    inputs = _decision_inputs()
    inputs[field] = True
    with pytest.raises(TypeError):
        decide_model_failover(**inputs)


def test_failover_revalidates_post_construction_observation_mutation():
    inputs = _decision_inputs()
    object.__setattr__(inputs["observation"], "cleanup_settled", 1)
    with pytest.raises(TypeError):
        decide_model_failover(**inputs)
