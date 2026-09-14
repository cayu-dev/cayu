"""Application-declared identities for behavior that cannot be inspected safely."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cayu._validation import require_durable_clean_nonblank

EXECUTION_PROFILE_IDENTITY_TEXT_MAX_CHARS = 256


class ExecutionProfileBehaviorIdentity(BaseModel):
    """Stable application declaration for one behavior-bearing component.

    ``name`` identifies the logical component. ``behavior_version`` changes when
    externally observable semantics change, while ``implementation_version``
    changes for every implementation deployment, even when the public contract
    remains the same. Values are fingerprint input only; profiles never persist
    these strings in ordinary runtime evidence.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    name: str = Field(max_length=EXECUTION_PROFILE_IDENTITY_TEXT_MAX_CHARS)
    behavior_version: str = Field(max_length=EXECUTION_PROFILE_IDENTITY_TEXT_MAX_CHARS)
    implementation_version: str = Field(max_length=EXECUTION_PROFILE_IDENTITY_TEXT_MAX_CHARS)

    @field_validator("name", "behavior_version", "implementation_version")
    @classmethod
    def validate_identity_text(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)


def copy_execution_profile_behavior_identity(
    identity: ExecutionProfileBehaviorIdentity | None,
) -> ExecutionProfileBehaviorIdentity | None:
    """Return an owned, revalidated declaration."""

    if identity is None:
        return None
    if type(identity) is not ExecutionProfileBehaviorIdentity:
        raise TypeError(
            "execution_profile_identity must be an ExecutionProfileBehaviorIdentity or None."
        )
    # Read the three declared fields directly.  Serializing a caller-owned model
    # is not a safe copy boundary: a model constructed without validation can
    # contain arbitrary objects, and Pydantic may include their representations
    # in serializer warnings before our registration boundary can redact them.
    return ExecutionProfileBehaviorIdentity(
        name=identity.name,
        behavior_version=identity.behavior_version,
        implementation_version=identity.implementation_version,
    )
