"""The Release a host deployed one Cayu application process as.

A host that deploys applications states which Release each process runs, so
supervision and session reads can tell which Release each run came from. Cayu
records that label wherever the process begins executing work and never acts on
it: the Release does not select behavior, admit work, or take part in
execution-profile comparison.
"""

from __future__ import annotations

import os
import re
import unicodedata
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, StrictStr, ValidationError, field_validator

from cayu._validation import require_durable_clean_nonblank

if TYPE_CHECKING:
    from cayu.vaults.redaction import SecretRedactor

RELEASE_ID_ENV = "CAYU_RELEASE_ID"
RELEASE_SNAPSHOT_ROOT_ENV = "CAYU_RELEASE_SNAPSHOT_ROOT"
RELEASE_ID_MAX_CHARS = 256
RELEASE_EVENT_PAYLOAD_KEY = "release"
_CONTROL_CHARACTER_CATEGORY = "Cc"
_LOWER_HEX_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_LOWER_HEX_SHA256 = re.compile(_LOWER_HEX_SHA256_PATTERN)


def validate_release_id(value: str, field_name: str) -> str:
    """Return a Release id: clean durable text without control characters, within its bound."""

    value = require_durable_clean_nonblank(value, field_name)
    if len(value) > RELEASE_ID_MAX_CHARS:
        raise ValueError(f"{field_name} cannot exceed {RELEASE_ID_MAX_CHARS} characters.")
    if any(unicodedata.category(character) == _CONTROL_CHARACTER_CATEGORY for character in value):
        raise ValueError(f"{field_name} cannot contain control characters.")
    return value


class ReleaseIdentity(BaseModel):
    """A Release id and, when the host recorded one, its AgentSnapshot root."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    id: StrictStr
    snapshot_root: StrictStr | None = Field(default=None, pattern=_LOWER_HEX_SHA256_PATTERN)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
        return validate_release_id(value, info.field_name)


def resolve_release_identity(
    configured: ReleaseIdentity | None,
    redactor: SecretRedactor,
    *,
    environ: Mapping[str, str] | None = None,
) -> ReleaseIdentity | None:
    """Return the Release the host states for this process, or ``None`` without one.

    The host states a Release in the environment, with ``CAYU_RELEASE_ID`` and,
    optionally, ``CAYU_RELEASE_SNAPSHOT_ROOT``, or in application configuration.
    ``environ`` defaults to the process environment, which is validated whenever
    it is set. When both state a Release they must be equal, so configuration
    can never replace the host's statement. A Release that names a workload
    secret is rejected. Errors never echo the values.
    """

    if configured is not None:
        if not isinstance(configured, ReleaseIdentity):
            raise TypeError("release_identity must be a ReleaseIdentity or None.")
        configured = ReleaseIdentity.model_validate(configured.model_dump(mode="python"))
    stated = _environment_release_identity(os.environ if environ is None else environ)
    if stated is not None and configured is not None and configured != stated:
        raise ValueError(
            f"release_identity conflicts with the Release that {RELEASE_ID_ENV} and "
            f"{RELEASE_SNAPSHOT_ROOT_ENV} state."
        )
    resolved = configured if stated is None else stated
    recorded = release_event_fields(resolved)
    if redactor.redact_json_values(recorded) != recorded:
        raise ValueError("The release identity contains a workload secret and cannot be recorded.")
    return resolved


def _environment_release_identity(environ: Mapping[str, str]) -> ReleaseIdentity | None:
    release_id = environ.get(RELEASE_ID_ENV)
    snapshot_root = environ.get(RELEASE_SNAPSHOT_ROOT_ENV)
    if release_id is None:
        if snapshot_root is not None:
            raise ValueError(f"{RELEASE_SNAPSHOT_ROOT_ENV} requires {RELEASE_ID_ENV}.")
        return None
    try:
        validate_release_id(release_id, RELEASE_ID_ENV)
    except ValueError as exc:
        raise ValueError(
            f"{RELEASE_ID_ENV} must be clean nonblank text without control characters, of "
            f"at most {RELEASE_ID_MAX_CHARS} characters."
        ) from exc
    if snapshot_root is not None and _LOWER_HEX_SHA256.fullmatch(snapshot_root) is None:
        raise ValueError(f"{RELEASE_SNAPSHOT_ROOT_ENV} must be a lowercase SHA-256 digest.")
    return ReleaseIdentity(id=release_id, snapshot_root=snapshot_root)


def release_event_fields(release_identity: ReleaseIdentity | None) -> dict[str, Any]:
    """Return the payload fields an event records its process's Release in."""

    if release_identity is None:
        return {}
    return {RELEASE_EVENT_PAYLOAD_KEY: release_identity.model_dump(mode="json")}


def recorded_release_identity(payload: Mapping[str, Any]) -> ReleaseIdentity | None:
    """Return the Release an event recorded, or ``None`` without a valid one.

    Only ``id`` and ``snapshot_root`` are read, so a record written by a later
    Cayu with more fields still yields its Release here.
    """

    recorded = payload.get(RELEASE_EVENT_PAYLOAD_KEY)
    if type(recorded) is not dict:
        return None
    try:
        return ReleaseIdentity.model_validate(
            {field: recorded[field] for field in ReleaseIdentity.model_fields if field in recorded}
        )
    except ValidationError:
        return None


__all__ = [
    "RELEASE_EVENT_PAYLOAD_KEY",
    "RELEASE_ID_ENV",
    "RELEASE_ID_MAX_CHARS",
    "RELEASE_SNAPSHOT_ROOT_ENV",
    "ReleaseIdentity",
    "recorded_release_identity",
    "release_event_fields",
    "resolve_release_identity",
    "validate_release_id",
]
