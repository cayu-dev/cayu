"""Provider retry decision values shared with durable event schemas."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt


class RetryReason(StrEnum):
    OUTPUT = "output"
    HTTP_STATUS = "http_status"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    RATE_LIMIT = "rate_limit"
    UNKNOWN_PROVIDER = "unknown_provider"


class RetryDisposition(StrEnum):
    RETRY_SCHEDULED = "retry_scheduled"
    PERMANENT_PROVIDER_ERROR = "permanent_provider_error"
    EXPLICIT_NONRETRYABLE = "explicit_nonretryable"
    UNKNOWN_PROVIDER_ATTEMPT_CAP = "unknown_provider_attempt_cap"
    CONFIGURED_ATTEMPT_EXHAUSTION = "configured_attempt_exhaustion"
    POLICY_DISALLOWED = "policy_disallowed"
    CLASSIFICATION_UNAVAILABLE = "classification_unavailable"
    SUPPRESSED = "suppressed"


class RetrySuppression(StrEnum):
    COMPLETION_OBSERVED = "completion_observed"
    PROVIDER_OPERATION = "provider_operation"
    PROVIDER_EFFECT_OBSERVED = "provider_effect_observed"
    CANCELLATION = "cancellation"
    DEADLINE = "deadline"
    AUTOMATIC_RETRY_DISABLED = "automatic_retry_disabled"


class RetryDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    retry: StrictBool
    disposition: RetryDisposition = RetryDisposition.CLASSIFICATION_UNAVAILABLE
    suppression: RetrySuppression | None = None
    provider_retryable: StrictBool | None = None
    reason: RetryReason | None = None
    status_code: StrictInt | None = Field(default=None, ge=100, le=599)
    delay_seconds: StrictFloat = Field(default=0.0, ge=0.0)
    attempt: StrictInt = Field(ge=1)
    next_attempt: StrictInt | None = Field(default=None, ge=2)
    max_attempts: StrictInt = Field(ge=1)
    effective_max_attempts: StrictInt = Field(ge=1)
