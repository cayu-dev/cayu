"""Immutable provider retry configuration shared by execution and saved state."""

from __future__ import annotations

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


class RetryPolicy(BaseModel):
    """Retry controls for one provider model step.

    `max_attempts` includes the initial attempt. The default permits five total
    attempts for classified transient provider failures. Unknown provider
    failures retain the stricter `max_unknown_attempts` ceiling.

    Frozen: policies are immutable value objects, so they can be shared across
    attempts and sessions without per-attempt defensive copies.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Ceilings fit provider rate-limit windows: a 429 may ask for several
    # minutes, and background work can afford many attempts. Raise max_delay_s
    # to honor a long Retry-After; defaults stay short for interactive runs.
    max_attempts: StrictInt = Field(default=5, ge=1, le=50)
    max_unknown_attempts: StrictInt = Field(default=2, ge=1, le=50)
    initial_delay_s: StrictFloat = Field(default=0.5, ge=0.0, le=600.0)
    max_delay_s: StrictFloat = Field(default=30.0, ge=0.0, le=3600.0)
    backoff_multiplier: StrictFloat = Field(default=2.0, ge=1.0, le=10.0)
    jitter_s: StrictFloat = Field(default=0.5, ge=0.0, le=600.0)
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
