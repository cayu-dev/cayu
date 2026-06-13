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
    """

    model_config = ConfigDict(extra="forbid")

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
    model_config = ConfigDict(extra="forbid")

    retry: StrictBool
    reason: RetryReason | None = None
    status_code: StrictInt | None = Field(default=None, ge=100, le=599)
    delay_seconds: StrictFloat = Field(default=0.0, ge=0.0)
    attempt: StrictInt = Field(ge=1)
    next_attempt: StrictInt | None = Field(default=None, ge=2)
    max_attempts: StrictInt = Field(ge=1)


def copy_retry_policy(policy: RetryPolicy | None) -> RetryPolicy:
    if policy is None:
        return RetryPolicy()
    if type(policy) is not RetryPolicy:
        raise TypeError("Retry policy must be a RetryPolicy instance.")
    return RetryPolicy(
        max_attempts=policy.max_attempts,
        initial_delay_s=policy.initial_delay_s,
        max_delay_s=policy.max_delay_s,
        backoff_multiplier=policy.backoff_multiplier,
        jitter_s=policy.jitter_s,
        retry_on_status_codes=tuple(policy.retry_on_status_codes),
        retry_on_timeout=policy.retry_on_timeout,
        retry_on_connection_error=policy.retry_on_connection_error,
        retry_on_rate_limit=policy.retry_on_rate_limit,
    )


def retry_decision(
    *,
    policy: RetryPolicy,
    attempt: int,
    error: str,
) -> RetryDecision:
    policy = copy_retry_policy(policy)
    if type(attempt) is not int:
        raise TypeError("attempt must be an integer.")
    if attempt < 1:
        raise ValueError("attempt must be greater than or equal to 1.")
    if type(error) is not str:
        raise TypeError("error must be a string.")

    reason, status_code = classify_retryable_error(policy=policy, error=error)
    can_retry = reason is not None and attempt < policy.max_attempts
    return RetryDecision(
        retry=can_retry,
        reason=reason,
        status_code=status_code,
        delay_seconds=_retry_delay(policy, attempt) if can_retry else 0.0,
        attempt=attempt,
        next_attempt=attempt + 1 if can_retry else None,
        max_attempts=policy.max_attempts,
    )


def classify_retryable_error(
    *,
    policy: RetryPolicy,
    error: str,
) -> tuple[RetryReason | None, int | None]:
    policy = copy_retry_policy(policy)
    if type(error) is not str:
        raise TypeError("error must be a string.")
    normalized = error.lower()

    status_code = _http_status_code(error)
    if _is_permanent_provider_error(status_code=status_code, normalized_error=normalized):
        return None, status_code

    if status_code is not None and status_code in policy.retry_on_status_codes:
        return RetryReason.HTTP_STATUS, status_code

    if policy.retry_on_rate_limit and any(
        pattern in normalized for pattern in _RATE_LIMIT_PATTERNS
    ):
        return RetryReason.RATE_LIMIT, status_code

    if policy.retry_on_timeout and any(pattern in normalized for pattern in _TIMEOUT_PATTERNS):
        return RetryReason.TIMEOUT, status_code

    if policy.retry_on_connection_error and any(
        pattern in normalized for pattern in _CONNECTION_PATTERNS
    ):
        return RetryReason.CONNECTION, status_code

    return None, status_code


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


def _retry_delay(policy: RetryPolicy, attempt: int) -> float:
    base_delay = policy.initial_delay_s * (policy.backoff_multiplier ** (attempt - 1))
    delay = min(base_delay, policy.max_delay_s)
    if policy.jitter_s > 0:
        delay += random.uniform(0, policy.jitter_s)
    return float(min(delay, policy.max_delay_s))
