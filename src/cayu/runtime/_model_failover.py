"""Pure failover decisions; dispatch authority remains owned by the runtime/store.

An eligible decision is necessary, never sufficient, to enter another provider.
The owner must also confirm accounting, the exact durable predecessor, candidate
capabilities, profile identity and the successor's dispatch admission.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from cayu.providers.base import (
    ModelContextOverflowError,
    ModelProviderError,
    ModelStreamDeadlineError,
)
from cayu.runtime.retry_policy import RetryDecision, RetryDisposition


class FailoverDisposition(StrEnum):
    SELECT_NEXT = "select_next"
    EXHAUSTED = "exhausted"
    SUPPRESSED = "suppressed"


class FailoverSuppression(StrEnum):
    CANCELLATION = "cancellation"
    DEADLINE = "deadline"
    COMPLETION_OBSERVED = "completion_observed"
    PROVIDER_EFFECT_OBSERVED = "provider_effect_observed"
    PROVIDER_OPERATION = "provider_operation"
    CLEANUP_UNSETTLED = "cleanup_unsettled"
    NON_PROVIDER_FAILURE = "non_provider_failure"
    PROVIDER_IDENTITY_MISMATCH = "provider_identity_mismatch"
    NONRETRYABLE_OR_UNKNOWN = "nonretryable_or_unknown"
    INELIGIBLE_FAILURE = "ineligible_failure"
    RETRY_NOT_EXHAUSTED = "retry_not_exhausted"
    RETRY_SUPPRESSED = "retry_suppressed"


# Deliberately separate from generic retry's text/timeout/connection taxonomy.
# A typed service rejection can permit fallback; a silent transport cannot.
_FAILOVER_SERVICE_STATUS_CODES = frozenset({429, 500, 502, 503, 504, 529})


@dataclass(frozen=True, slots=True)
class FailoverObservation:
    """Explicit live observations, not inferences from missing durable events.

    ``provider_effect_observed`` is monotonic across the logical model step,
    including preceding local retries. ``cleanup_settled`` describes owned local
    operations; it is not a claim that all remote billing or work was aborted.
    """

    provider_name: str
    caller_cancelled: bool
    completion_observed: bool
    provider_effect_observed: bool
    provider_operation_owned: bool
    cleanup_settled: bool

    def __post_init__(self) -> None:
        if (
            type(self.provider_name) is not str
            or not self.provider_name
            or self.provider_name != self.provider_name.strip()
        ):
            raise ValueError("Failover observations require the execution provider identity.")
        for observed in (
            self.caller_cancelled,
            self.completion_observed,
            self.provider_effect_observed,
            self.provider_operation_owned,
            self.cleanup_settled,
        ):
            if type(observed) is not bool:
                raise TypeError("Failover observations require explicit boolean evidence.")


@dataclass(frozen=True, slots=True)
class FailoverDecision:
    disposition: FailoverDisposition
    suppression: FailoverSuppression | None = None

    def __post_init__(self) -> None:
        if type(self.disposition) is not FailoverDisposition:
            raise TypeError("Failover disposition must be typed.")
        if self.suppression is not None and type(self.suppression) is not FailoverSuppression:
            raise TypeError("Failover suppression must be typed.")
        if (self.disposition is FailoverDisposition.SUPPRESSED) != (self.suppression is not None):
            raise ValueError("Only suppressed failover decisions carry a suppression reason.")


def decide_model_failover(
    *,
    failure: BaseException,
    provider_name: str,
    retry: RetryDecision,
    observation: FailoverObservation,
    candidate_index: int,
    candidate_count: int,
    attempts_used: int,
    max_total_attempts: int,
) -> FailoverDecision:
    """Classify the current failure without inspecting error text or causal history."""

    if not isinstance(failure, BaseException):
        raise TypeError("Failover requires the observed failure.")
    if (
        type(provider_name) is not str
        or not provider_name
        or provider_name != provider_name.strip()
    ):
        raise ValueError("Failover requires the exact execution provider identity.")
    if type(retry) is not RetryDecision or type(observation) is not FailoverObservation:
        raise TypeError("Failover requires typed retry and observation evidence.")
    observation = replace(observation)
    if any(
        type(value) is not int
        for value in (candidate_index, candidate_count, attempts_used, max_total_attempts)
    ):
        raise TypeError("Failover counters must be integers.")
    if not (
        1 <= candidate_count <= 8
        and 0 <= candidate_index < candidate_count
        and 1 <= attempts_used <= max_total_attempts <= 80
    ):
        raise ValueError("Failover counters exceed the bounded plan.")

    def suppress(reason: FailoverSuppression) -> FailoverDecision:
        return FailoverDecision(FailoverDisposition.SUPPRESSED, reason)

    if observation.caller_cancelled:
        return suppress(FailoverSuppression.CANCELLATION)
    if isinstance(failure, ModelStreamDeadlineError):
        return suppress(FailoverSuppression.DEADLINE)
    if isinstance(failure, ModelContextOverflowError):
        return suppress(FailoverSuppression.INELIGIBLE_FAILURE)
    if observation.completion_observed:
        return suppress(FailoverSuppression.COMPLETION_OBSERVED)
    if observation.provider_effect_observed:
        return suppress(FailoverSuppression.PROVIDER_EFFECT_OBSERVED)
    if observation.provider_operation_owned:
        return suppress(FailoverSuppression.PROVIDER_OPERATION)
    if not observation.cleanup_settled:
        return suppress(FailoverSuppression.CLEANUP_UNSETTLED)
    if not isinstance(failure, ModelProviderError):
        return suppress(FailoverSuppression.NON_PROVIDER_FAILURE)
    # Adapter errors identify their protocol/backend (for example "openai"),
    # which need not equal the application's registration name. Only the live
    # attempt owner supplies execution identity; an error string cannot prove it.
    if observation.provider_name != provider_name:
        return suppress(FailoverSuppression.PROVIDER_IDENTITY_MISMATCH)
    if type(failure.provider) is not str or not failure.provider.strip():
        return suppress(FailoverSuppression.NONRETRYABLE_OR_UNKNOWN)
    if failure.retryable is not True:
        return suppress(FailoverSuppression.NONRETRYABLE_OR_UNKNOWN)
    if (
        type(failure.status_code) is not int
        or failure.status_code not in _FAILOVER_SERVICE_STATUS_CODES
    ):
        return suppress(FailoverSuppression.INELIGIBLE_FAILURE)
    if retry.suppression is not None:
        return suppress(FailoverSuppression.RETRY_SUPPRESSED)
    if retry.provider_retryable is not True or retry.status_code != failure.status_code:
        return suppress(FailoverSuppression.INELIGIBLE_FAILURE)
    # The chain ceiling governs local retries too. An otherwise retryable
    # attempt can exhaust the chain before its candidate's retry policy does.
    if attempts_used == max_total_attempts:
        return FailoverDecision(FailoverDisposition.EXHAUSTED)
    if (
        retry.retry is not False
        or retry.disposition is not RetryDisposition.CONFIGURED_ATTEMPT_EXHAUSTION
    ):
        return suppress(FailoverSuppression.RETRY_NOT_EXHAUSTED)
    if candidate_index + 1 == candidate_count:
        return FailoverDecision(FailoverDisposition.EXHAUSTED)
    return FailoverDecision(FailoverDisposition.SELECT_NEXT)
