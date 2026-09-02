"""Durable ownership for provider-operation cancellation side effects."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, StrictInt

from cayu._validation import copy_durable_json_object, require_clean_nonblank

PROVIDER_OPERATION_CANCELLATION_CLAIM_CHECKPOINT_KEY = (
    "__cayu_provider_operation_cancellation_claim_v1__"
)


class ProviderOperationCancellationClaim(BaseModel):
    """Exact stage, operation, and epoch allowed to mutate cancellation state."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    version: Literal[1] = 1
    claim_id: str
    stage_id: str
    run_epoch: StrictInt
    operation_id: str
    stream_protocol: str
    expires_at: datetime

    def model_post_init(self, __context: Any) -> None:
        del __context
        for field_name in ("claim_id", "stage_id", "operation_id", "stream_protocol"):
            object.__setattr__(
                self,
                field_name,
                require_clean_nonblank(getattr(self, field_name), field_name),
            )
        if self.run_epoch < 0:
            raise ValueError("run_epoch must be nonnegative.")
        if self.expires_at.tzinfo is None or self.expires_at.utcoffset() is None:
            raise ValueError("expires_at must be timezone-aware.")
        object.__setattr__(self, "expires_at", self.expires_at.astimezone(UTC))

    def same_owner(self, other: ProviderOperationCancellationClaim) -> bool:
        """Compare immutable authority while permitting lease renewal."""

        return (
            self.claim_id,
            self.stage_id,
            self.run_epoch,
            self.operation_id,
            self.stream_protocol,
        ) == (
            other.claim_id,
            other.stage_id,
            other.run_epoch,
            other.operation_id,
            other.stream_protocol,
        )

    def active_at(self, now: datetime) -> bool:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be timezone-aware.")
        return self.expires_at > now.astimezone(UTC)


def provider_operation_cancellation_claim_from_checkpoint(
    checkpoint: dict[str, Any] | None,
) -> ProviderOperationCancellationClaim | None:
    """Parse the active cancellation owner, failing closed on malformed evidence."""

    if checkpoint is None:
        return None
    raw = checkpoint.get(PROVIDER_OPERATION_CANCELLATION_CLAIM_CHECKPOINT_KEY)
    if raw is None:
        return None
    return ProviderOperationCancellationClaim.model_validate(raw)


def active_provider_operation_cancellation_claim_from_checkpoint(
    checkpoint: dict[str, Any] | None,
    *,
    now: datetime,
) -> ProviderOperationCancellationClaim | None:
    """Return only a currently leased cancellation owner."""

    claim = provider_operation_cancellation_claim_from_checkpoint(checkpoint)
    return claim if claim is not None and claim.active_at(now) else None


def checkpoint_with_provider_operation_cancellation_claim(
    checkpoint: dict[str, Any] | None,
    claim: ProviderOperationCancellationClaim,
    *,
    now: datetime,
) -> dict[str, Any]:
    """Acquire or exactly replay one active cancellation claim."""

    claim = ProviderOperationCancellationClaim.model_validate(claim)
    copied = {} if checkpoint is None else copy_durable_json_object(checkpoint, "checkpoint")
    existing = provider_operation_cancellation_claim_from_checkpoint(copied)
    if existing is not None:
        same_owner = existing.same_owner(claim)
        if (same_owner and not existing.active_at(now)) or (
            not same_owner and existing.active_at(now)
        ):
            raise RuntimeError("Provider-operation cancellation claim is no longer acquirable.")
    copied[PROVIDER_OPERATION_CANCELLATION_CLAIM_CHECKPOINT_KEY] = claim.model_dump(mode="json")
    return copied


def checkpoint_without_provider_operation_cancellation_claim(
    checkpoint: dict[str, Any] | None,
    claim: ProviderOperationCancellationClaim,
    *,
    now: datetime,
) -> dict[str, Any]:
    """Release only the exact active cancellation claim."""

    claim = ProviderOperationCancellationClaim.model_validate(claim)
    copied = {} if checkpoint is None else copy_durable_json_object(checkpoint, "checkpoint")
    existing = provider_operation_cancellation_claim_from_checkpoint(copied)
    if existing is None or not existing.same_owner(claim) or not existing.active_at(now):
        raise RuntimeError("Provider-operation cancellation ownership changed before release.")
    copied.pop(PROVIDER_OPERATION_CANCELLATION_CLAIM_CHECKPOINT_KEY, None)
    return copied


__all__ = [
    "PROVIDER_OPERATION_CANCELLATION_CLAIM_CHECKPOINT_KEY",
    "ProviderOperationCancellationClaim",
    "active_provider_operation_cancellation_claim_from_checkpoint",
    "checkpoint_with_provider_operation_cancellation_claim",
    "checkpoint_without_provider_operation_cancellation_claim",
    "provider_operation_cancellation_claim_from_checkpoint",
]
