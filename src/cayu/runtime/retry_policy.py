from __future__ import annotations

import random
import re
from enum import StrEnum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    field_validator,
)

DEFAULT_RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 504, 529)
_STATUS_CODE_PATTERNS = (
    re.compile(r"\bHTTP\s+(?:status\s+)?(?:code\s+)?([1-5][0-9][0-9])\b", re.IGNORECASE),
    re.compile(r"\bHTTP/[0-9.]+\s+([1-5][0-9][0-9])\b", re.IGNORECASE),
    re.compile(r"\bstatus(?:[_\s-]?code)?\s*[:=]?\s*([1-5][0-9][0-9])\b", re.IGNORECASE),
)
_TIMEOUT_PATTERNS = (
    "timeout",
    "timed out",
    "read timed out",
    "connect timed out",
    "stream idle timeout",
)
_CONNECTION_PATTERNS = (
    "connection reset",
    "connection aborted",
    "connection refused",
    "connection error",
    "network",
    "temporarily unavailable",
    "temporary failure",
)
_RATE_LIMIT_PATTERNS = (
    "rate limit",
    "too many requests",
)
_PERMANENT_ERROR_PATTERNS = (
    "insufficient_quota",
    "exceeded your current quota",
    "run out of credits",
    "out of credits",
    "monthly spend",
    "billing",
)


class RetryReason(StrEnum):
    HTTP_STATUS = "http_status"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    RATE_LIMIT = "rate_limit"


class RetryPolicy(BaseModel):
    """Retry controls for one provider model step.

    `max_attempts` includes the initial attempt. The default of 1 means retries
    are disabled.

    Frozen: policies are immutable value objects, so they can be shared across
    attempts and sessions without per-attempt defensive copies.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_attempts: StrictInt = Field(default=1, ge=1, le=10)
    initial_delay_s: StrictFloat = Field(default=0.5, ge=0.0, le=60.0)
    max_delay_s: StrictFloat = Field(default=10.0, ge=0.0, le=300.0)
    backoff_multiplier: StrictFloat = Field(default=2.0, ge=1.0, le=10.0)
    jitter_s: StrictFloat = Field(default=0.0, ge=0.0, le=60.0)
    retry_on_status_codes: tuple[StrictInt, ...] = Field(
        default=DEFAULT_RETRYABLE_STATUS_CODES,
        min_length=0,
        max_length=32,
    )
    retry_on_timeout: StrictBool = True
    retry_on_connection_error: StrictBool = True
    retry_on_rate_limit: StrictBool = True

    @field_validator("retry_on_status_codes")
    @classmethod
    def validate_status_codes(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        for status_code in value:
            if status_code < 100 or status_code > 599:
                raise ValueError("retry status codes must be between 100 and 599.")
        return value


class RetryDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    retry: StrictBool
    reason: RetryReason | None = None
    status_code: StrictInt | None = Field(default=None, ge=100, le=599)
    delay_seconds: StrictFloat = Field(default=0.0, ge=0.0)
    attempt: StrictInt = Field(ge=1)
    next_attempt: StrictInt | None = Field(default=None, ge=2)
    max_attempts: StrictInt = Field(ge=1)


def copy_retry_policy(policy: RetryPolicy | None) -> RetryPolicy:
    """Validate `policy` and return it unchanged (default policy for `None`).

    `RetryPolicy` is frozen with immutable field values, so sharing the
    instance is safe — the previous per-attempt field-by-field rebuild was a
    drift-prone no-op.
    """
    if policy is None:
        return RetryPolicy()
    if type(policy) is not RetryPolicy:
        raise TypeError("Retry policy must be a RetryPolicy instance.")
    return policy


def retry_decision(
    *,
    policy: RetryPolicy,
    attempt: int,
    error: str,
    status_code: int | None = None,
    retryable: bool | None = None,
    retry_after_s: float | None = None,
) -> RetryDecision:
    """Classify one failed model attempt into a retry decision.

    `status_code`, `retryable`, and `retry_after_s` are the typed fields a
    provider exposes on `ModelProviderError`. When supplied they take precedence
    over reparsing `error` text: `status_code` seeds classification directly,
    `retryable=False` forces a terminal decision, and `retry_after_s` overrides
    the computed backoff so a provider `Retry-After` directive is honored. They
    default to `None` so string-only callers keep the legacy regex behavior.
    """

    policy = copy_retry_policy(policy)
    if type(attempt) is not int:
        raise TypeError("attempt must be an integer.")
    if attempt < 1:
        raise ValueError("attempt must be greater than or equal to 1.")
    if type(error) is not str:
        raise TypeError("error must be a string.")
    if retry_after_s is not None:
        if type(retry_after_s) not in {int, float} or retry_after_s < 0:
            raise ValueError("retry_after_s must be a non-negative number.")
        retry_after_s = float(retry_after_s)

    reason, classified_status = classify_retryable_error(
        policy=policy,
        error=error,
        status_code=status_code,
        retryable=retryable,
    )
    can_retry = reason is not None and attempt < policy.max_attempts
    return RetryDecision(
        retry=can_retry,
        reason=reason,
        status_code=classified_status,
        delay_seconds=(
            _retry_delay(policy, attempt, retry_after_s=retry_after_s) if can_retry else 0.0
        ),
        attempt=attempt,
        next_attempt=attempt + 1 if can_retry else None,
        max_attempts=policy.max_attempts,
    )


def classify_retryable_error(
    *,
    policy: RetryPolicy,
    error: str,
    status_code: int | None = None,
    retryable: bool | None = None,
) -> tuple[RetryReason | None, int | None]:
    """Classify a failure into a retry reason and effective HTTP status code.

    A typed `status_code` (e.g. from `ModelProviderError.status_code`) is
    preferred over the regex scan of `error`. A typed `retryable` flag is the
    provider's own tri-state verdict: `False` short-circuits to a terminal
    decision, `True` is honored as a transient failure when neither the status
    code nor the message text already matched a configured pattern.
    """

    policy = copy_retry_policy(policy)
    if type(error) is not str:
        raise TypeError("error must be a string.")
    if status_code is not None and (
        type(status_code) is not int or status_code < 100 or status_code > 599
    ):
        raise ValueError("status_code must be a valid HTTP status code.")
    if retryable is not None and type(retryable) is not bool:
        raise TypeError("retryable must be a boolean.")
    normalized = error.lower()

    effective_status = status_code if status_code is not None else _http_status_code(error)

    # The provider already classified this failure as terminal; don't burn a
    # retry on an error it told us can never succeed as-is.
    if retryable is False:
        return None, effective_status

    if _is_permanent_provider_error(status_code=effective_status, normalized_error=normalized):
        return None, effective_status

    if effective_status is not None and effective_status in policy.retry_on_status_codes:
        return RetryReason.HTTP_STATUS, effective_status

    if policy.retry_on_rate_limit and any(
        pattern in normalized for pattern in _RATE_LIMIT_PATTERNS
    ):
        return RetryReason.RATE_LIMIT, effective_status

    if policy.retry_on_timeout and any(pattern in normalized for pattern in _TIMEOUT_PATTERNS):
        return RetryReason.TIMEOUT, effective_status

    if policy.retry_on_connection_error and any(
        pattern in normalized for pattern in _CONNECTION_PATTERNS
    ):
        return RetryReason.CONNECTION, effective_status

    # The provider explicitly flagged the failure retryable, but neither its
    # status code nor the message text matched a configured pattern. Honor the
    # typed signal as a transient (connection-class) failure rather than drop a
    # retry the provider asked for.
    if retryable is True and policy.retry_on_connection_error:
        return RetryReason.CONNECTION, effective_status

    return None, effective_status


def retry_event_payload(
    *,
    decision: RetryDecision,
    provider_name: str,
    model: str,
    step: int,
    error: str,
) -> dict[str, Any]:
    if type(decision) is not RetryDecision:
        raise TypeError("decision must be a RetryDecision.")
    return {
        "provider": provider_name,
        "model": model,
        "step": step,
        "attempt": decision.attempt,
        "next_attempt": decision.next_attempt,
        "max_attempts": decision.max_attempts,
        "delay_seconds": decision.delay_seconds,
        "reason": None if decision.reason is None else decision.reason.value,
        "status_code": decision.status_code,
        "error": error,
    }


def _http_status_code(error: str) -> int | None:
    for pattern in _STATUS_CODE_PATTERNS:
        match = pattern.search(error)
        if match is not None:
            return int(match.group(1))
    return None


def _is_permanent_provider_error(*, status_code: int | None, normalized_error: str) -> bool:
    if not any(pattern in normalized_error for pattern in _PERMANENT_ERROR_PATTERNS):
        return False
    if status_code is None:
        return True
    return 400 <= status_code < 500


def _retry_delay(
    policy: RetryPolicy,
    attempt: int,
    *,
    retry_after_s: float | None = None,
) -> float:
    if retry_after_s is not None:
        # A provider `Retry-After` directive is authoritative: waiting less just
        # earns another rejection. Still bound it by `max_delay_s` so a hostile
        # or buggy header can't stall the run indefinitely.
        return float(min(retry_after_s, policy.max_delay_s))
    base_delay = policy.initial_delay_s * (policy.backoff_multiplier ** (attempt - 1))
    delay = min(base_delay, policy.max_delay_s)
    if policy.jitter_s > 0:
        delay += random.uniform(0, policy.jitter_s)
    return float(min(delay, policy.max_delay_s))
