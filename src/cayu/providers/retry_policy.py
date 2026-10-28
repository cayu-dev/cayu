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
    model_validator,
)

DEFAULT_RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 504, 529)
# Rate-limit and overload statuses: the ones `max_retry_elapsed_s` paces.
RATE_LIMIT_STATUS_CODES = (429, 503, 529)
DEFAULT_MAX_RETRY_ELAPSED_S = 300.0


class RetryPolicy(BaseModel):
    """Retry controls for one provider model step.

    `max_attempts` includes the initial attempt. The default permits five total
    attempts for classified transient provider failures; other failures get 7.5
    to 9.5 seconds of backoff in all. `cayu guide providers#provider-retries`
    shows how to wait out longer outages. Unknown provider failures retain the
    stricter `max_unknown_attempts` ceiling.

    `max_retry_elapsed_s` is a time budget for rate-limit and overload failures
    (HTTP 429, 503 and 529, or a provider rate-limit message), counted from the
    start of the model step's first attempt; 300 seconds by default, `None`
    turns it off. The first such retry uses ordinary backoff. Each later one
    waits at least its share of the time left, spread over the attempts left
    and growing by `backoff_multiplier`, so a sustained limit is waited out
    instead of exhausting the attempts in seconds. No retry waits longer than
    `max_delay_s` (60 seconds by default) or the time left, and once the budget
    is spent the step stops retrying them. `max_attempts` still bounds the
    attempts, and a zero `initial_delay_s` still never waits. The budget
    applies to agent model steps and automatic compaction.

    The default budget never cuts a policy's own backoff short: when
    `max_retry_elapsed_s` is not set and the attempts and delays can wait
    longer than 300 seconds in all (`max_attempts - 1` delays of
    `initial_delay_s` growing by `backoff_multiplier`, plus `jitter_s`, each at
    most `max_delay_s`), the budget is off and that backoff applies as written.
    Set `max_retry_elapsed_s` to keep a budget for such a policy.

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
    max_delay_s: StrictFloat = Field(default=60.0, ge=0.0, le=3600.0)
    max_retry_elapsed_s: StrictFloat | None = Field(
        default=DEFAULT_MAX_RETRY_ELAPSED_S, gt=0.0, le=86400.0
    )
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

    @model_validator(mode="after")
    def keep_own_backoff(self) -> RetryPolicy:
        # A budget the caller never set must not stop a longer backoff the
        # caller did choose. Resolve it here, not when retrying, so the saved
        # policy carries the budget that applies and reloads unchanged.
        if (
            "max_retry_elapsed_s" not in self.model_fields_set
            and _longest_backoff_s(self) > DEFAULT_MAX_RETRY_ELAPSED_S
        ):
            self.__dict__["max_retry_elapsed_s"] = None
        return self


def _longest_backoff_s(policy: RetryPolicy) -> float:
    """Longest total wait of `policy`'s ordinary backoff across its attempts.

    Ignores `Retry-After` and the rate-limit budget's pacing.
    """
    total = 0.0
    for attempt in range(1, policy.max_attempts):
        delay = policy.initial_delay_s * (policy.backoff_multiplier ** (attempt - 1))
        if delay > 0:
            delay += policy.jitter_s
        total += min(delay, policy.max_delay_s)
    return total


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
