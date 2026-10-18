"""Content-free declaration of the revision a person approved.

The runtime approval primitive is a binary decision on the call the model
proposed; it cannot see that a person changed the proposal before approving it.
An application that lets reviewers edit records the change in its own store and
declares it on the approval with two digests: the version the agent proposed and
the version the person approved. The runtime validates the declaration on the
request and publishes it as the approval event's own ``approval_revision`` field,
outside the bounded audit metadata, so unrelated metadata cannot drop it.
Supervision queries classify an approval whose digests differ as an edit. Only
digests cross this boundary, never the content they identify.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictStr, ValidationError, model_validator

APPROVAL_REVISION_METADATA_KEY = "cayu:approval_revision"
# Opaque digest tokens such as lowercase SHA-256 hex or ``sha256:<hex>``. The
# alphabet leaves no room for free text, so a declaration cannot carry content.
APPROVAL_REVISION_DIGEST_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:+=-]{0,127}$"


class ApprovalRevision(BaseModel):
    """The proposed and approved versions of one approved call, as digests."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    proposed_digest: StrictStr = Field(pattern=APPROVAL_REVISION_DIGEST_PATTERN)
    approved_digest: StrictStr = Field(pattern=APPROVAL_REVISION_DIGEST_PATTERN)

    @property
    def edited(self) -> bool:
        """Whether the person approved a different version than the agent proposed."""

        return self.approved_digest != self.proposed_digest


class ApprovalArgumentsRevision(BaseModel):
    """Runtime evidence that a person approved a call with changed arguments.

    ``changed_fields`` names only top-level properties the tool's schema
    declares; it never carries argument values. The digests identify the
    proposed and approved versions with HMAC-SHA256 under the session store's
    public-authority alias key ``digest_key_id``, so they cannot be confirmed
    by guessing; all three are ``None`` when the store has no alias keyring.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    changed_fields: tuple[StrictStr, ...] = Field(default=(), max_length=64)
    proposed_arguments_digest: StrictStr | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    approved_arguments_digest: StrictStr | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    digest_key_id: StrictStr | None = None

    @model_validator(mode="after")
    def validate_digests(self) -> ApprovalArgumentsRevision:
        present = {
            self.proposed_arguments_digest is not None,
            self.approved_arguments_digest is not None,
            self.digest_key_id is not None,
        }
        if len(present) != 1:
            raise ValueError("Argument digests and their key id are published together.")
        return self


def approval_revision_metadata(
    *, proposed_digest: str, approved_digest: str
) -> dict[str, dict[str, str]]:
    """Return the approval-metadata entry declaring one approved revision.

    Merge it into ``ToolApprovalRequest.metadata`` next to application keys.
    """

    revision = ApprovalRevision(proposed_digest=proposed_digest, approved_digest=approved_digest)
    return {APPROVAL_REVISION_METADATA_KEY: revision.model_dump(mode="json")}


def validate_approval_revision_metadata(metadata: dict[str, Any]) -> ApprovalRevision | None:
    """Validate a declaration on an approval request; ``None`` when there is none."""

    if APPROVAL_REVISION_METADATA_KEY not in metadata:
        return None
    try:
        return ApprovalRevision.model_validate(metadata[APPROVAL_REVISION_METADATA_KEY])
    except ValidationError:
        raise ValueError(
            f"metadata[{APPROVAL_REVISION_METADATA_KEY!r}] must hold exactly "
            "proposed_digest and approved_digest digest tokens."
        ) from None
