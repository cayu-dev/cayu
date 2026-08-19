"""Secret-safe validation for application-declared execution identities."""

from __future__ import annotations

from cayu.core.execution_identity import (
    ExecutionProfileBehaviorIdentity,
    copy_execution_profile_behavior_identity,
)
from cayu.vaults import SecretRedactor


def copy_secret_free_execution_profile_behavior_identity(
    identity: ExecutionProfileBehaviorIdentity | None,
    *,
    redactor: SecretRedactor,
    field_name: str,
) -> ExecutionProfileBehaviorIdentity | None:
    """Copy an identity only when none of its public hash inputs are secrets."""

    copied = copy_execution_profile_behavior_identity(identity)
    if copied is None:
        return None
    for identity_field in ("name", "behavior_version", "implementation_version"):
        value = getattr(copied, identity_field)
        if redactor.redact_text(value) != value:
            raise ValueError(
                f"{field_name}.{identity_field} contains a configured workload secret and "
                "cannot be used as public execution-profile authority."
            )
    return copied


__all__ = ["copy_secret_free_execution_profile_behavior_identity"]
