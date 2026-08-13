"""Durable, redacted execution-profile identities for session admission."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum
from hashlib import sha256
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from cayu._validation import canonical_durable_json_bytes, copy_durable_json_value
from cayu.core.events import Event, copy_event

EXECUTION_PROFILE_SCHEMA_VERSION = 1
EXECUTION_PROFILE_METADATA_KEY = "cayu:execution_profile"
_EXECUTION_PROFILE_RECORD_TYPE = "cayu.execution-profile"


class ExecutionProfileComponentClass(StrEnum):
    """Stable classes of execution authority represented by a profile."""

    RUNTIME = "runtime"
    PROVIDER_TARGET = "provider_target"
    DURABLE_SYSTEM_PROJECTION = "durable_system_projection"
    DIRECT_TOOLS = "direct_tools"


class ExecutionProfileIdentityStrength(StrEnum):
    """How strongly one component is identified."""

    APPLICATION_VERSIONED = "application_versioned"
    STRUCTURAL = "structural"
    UNAVAILABLE = "unavailable"


class ExecutionProfileIdentityAvailability(StrEnum):
    """Whether a component can be compared at the admission boundary."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class ExecutionProfileComponentIdentity(BaseModel):
    """Redacted identity for one typed execution-profile component."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    component_class: ExecutionProfileComponentClass
    strength: ExecutionProfileIdentityStrength
    availability: ExecutionProfileIdentityAvailability
    fingerprint: str | None

    @field_validator("fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("fingerprint must be a lowercase SHA-256 digest.")
        return value

    @model_validator(mode="after")
    def validate_availability(self) -> ExecutionProfileComponentIdentity:
        unavailable = self.availability is ExecutionProfileIdentityAvailability.UNAVAILABLE
        if unavailable != (self.fingerprint is None):
            raise ValueError("Unavailable components must omit their fingerprint.")
        if unavailable != (self.strength is ExecutionProfileIdentityStrength.UNAVAILABLE):
            raise ValueError("Unavailable components must use unavailable identity strength.")
        return self


class ExecutionProfileIdentity(BaseModel):
    """Versioned, redacted identity frozen before a session can execute."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal[1] = EXECUTION_PROFILE_SCHEMA_VERSION
    fingerprint: str
    components: tuple[ExecutionProfileComponentIdentity, ...]

    @field_validator("fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("fingerprint must be a lowercase SHA-256 digest.")
        return value

    @model_validator(mode="after")
    def validate_components(self) -> ExecutionProfileIdentity:
        classes = tuple(component.component_class for component in self.components)
        required_classes = tuple(sorted(ExecutionProfileComponentClass, key=str))
        if classes != required_classes:
            raise ValueError(
                "Execution-profile components must contain every required class exactly once "
                "in sorted order."
            )
        expected = _profile_fingerprint(self.components)
        if self.fingerprint != expected:
            raise ValueError("Execution-profile fingerprint does not match its components.")
        return self

    def component(
        self,
        component_class: ExecutionProfileComponentClass,
    ) -> ExecutionProfileComponentIdentity:
        for component in self.components:
            if component.component_class is component_class:
                return component
        raise KeyError(f"Execution profile has no {component_class.value} component.")


class ExecutionProfileMismatchError(RuntimeError):
    """Raised after durable evidence rejects a changed execution profile."""

    def __init__(
        self,
        *,
        session_id: str,
        expected_profile_fingerprint: str,
        candidate_profile_fingerprint: str,
        changed_component_classes: tuple[ExecutionProfileComponentClass, ...],
    ) -> None:
        self.session_id = session_id
        self.expected_profile_fingerprint = expected_profile_fingerprint
        self.candidate_profile_fingerprint = candidate_profile_fingerprint
        self.changed_component_classes = changed_component_classes
        changed = ", ".join(component.value for component in changed_component_classes)
        super().__init__(
            f"Session {session_id} execution profile changed in: {changed}. "
            "Start a new session or use an explicit profile-adoption flow."
        )


class ExecutionProfileRejectionResult(BaseModel):
    """Durable rejection event plus whether an exact prior write was replayed."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    event: Event
    replayed: bool = False

    @field_validator("event", mode="before")
    @classmethod
    def copy_rejection_event(cls, value: Event) -> Event:
        return copy_event(value)


def build_execution_profile_identity(
    *,
    runtime_name: str,
    runtime_version: str | None,
    provider_name: str,
    model: str,
    durable_system_prompt: str | None,
    direct_tools: Iterable[Mapping[str, Any]],
) -> ExecutionProfileIdentity:
    """Build a profile without retaining raw prompts, schemas, or tool names."""

    components = (
        _available_component(
            ExecutionProfileComponentClass.DIRECT_TOOLS,
            ExecutionProfileIdentityStrength.STRUCTURAL,
            list(direct_tools),
        ),
        _available_component(
            ExecutionProfileComponentClass.DURABLE_SYSTEM_PROJECTION,
            ExecutionProfileIdentityStrength.STRUCTURAL,
            {"system_prompt": durable_system_prompt},
        ),
        _available_component(
            ExecutionProfileComponentClass.PROVIDER_TARGET,
            ExecutionProfileIdentityStrength.STRUCTURAL,
            {"provider_name": provider_name, "model": model},
        ),
        (
            _available_component(
                ExecutionProfileComponentClass.RUNTIME,
                ExecutionProfileIdentityStrength.APPLICATION_VERSIONED,
                {"runtime_name": runtime_name, "runtime_version": runtime_version},
            )
            if runtime_version is not None
            else _unavailable_component(ExecutionProfileComponentClass.RUNTIME)
        ),
    )
    sorted_components = tuple(sorted(components, key=lambda component: component.component_class))
    return ExecutionProfileIdentity(
        fingerprint=_profile_fingerprint(sorted_components),
        components=sorted_components,
    )


def changed_execution_profile_components(
    expected: ExecutionProfileIdentity,
    candidate: ExecutionProfileIdentity,
) -> tuple[ExecutionProfileComponentClass, ...]:
    """Return bounded component classes that changed or cannot be verified."""

    expected_by_class = {item.component_class: item for item in expected.components}
    candidate_by_class = {item.component_class: item for item in candidate.components}
    classes = sorted(set(expected_by_class) | set(candidate_by_class), key=str)
    return tuple(
        component_class
        for component_class in classes
        if (
            expected_by_class.get(component_class) != candidate_by_class.get(component_class)
            or expected_by_class[component_class].availability
            is ExecutionProfileIdentityAvailability.UNAVAILABLE
            or candidate_by_class[component_class].availability
            is ExecutionProfileIdentityAvailability.UNAVAILABLE
        )
    )


def unavailable_execution_profile_components(
    profile: ExecutionProfileIdentity,
) -> tuple[ExecutionProfileComponentClass, ...]:
    """Return required component classes with no deterministic identity."""

    return tuple(
        component.component_class
        for component in profile.components
        if component.availability is ExecutionProfileIdentityAvailability.UNAVAILABLE
    )


def execution_profile_with_component(
    profile: ExecutionProfileIdentity,
    component: ExecutionProfileComponentIdentity,
) -> ExecutionProfileIdentity:
    """Return a profile with one durable component identity replaced."""

    by_class = {item.component_class: item for item in profile.components}
    by_class[component.component_class] = component
    components = tuple(sorted(by_class.values(), key=lambda item: item.component_class))
    return ExecutionProfileIdentity(
        fingerprint=_profile_fingerprint(components),
        components=components,
    )


def execution_profile_session_metadata(
    profile: ExecutionProfileIdentity,
) -> dict[str, Any]:
    """Return the bounded runtime-owned record stored with a new session."""

    dumped = profile.model_dump(mode="json")
    return {
        "record_type": _EXECUTION_PROFILE_RECORD_TYPE,
        "schema_version": EXECUTION_PROFILE_SCHEMA_VERSION,
        "baseline": dumped,
        "expected": dumped,
    }


def execution_profile_metadata_after_adoption(
    metadata: Mapping[str, Any],
    profile: ExecutionProfileIdentity,
) -> dict[str, Any]:
    """Advance the expected profile while retaining the immutable baseline."""

    copied = copy_durable_json_value(dict(metadata), "session.metadata")
    current = copied.get(EXECUTION_PROFILE_METADATA_KEY)
    if type(current) is not dict:
        raise ValueError("Session has no durable execution-profile identity.")
    # Validate the complete record before retaining its immutable baseline.
    execution_profile_from_session_metadata(copied)
    current["expected"] = profile.model_dump(mode="json")
    copied[EXECUTION_PROFILE_METADATA_KEY] = current
    return copied


def execution_profile_from_session_metadata(
    metadata: Mapping[str, Any],
) -> ExecutionProfileIdentity:
    """Load the current expected profile, failing closed on absent/malformed state."""

    raw = metadata.get(EXECUTION_PROFILE_METADATA_KEY)
    if type(raw) is not dict:
        raise ValueError("Session has no durable execution-profile identity.")
    if set(raw) != {"record_type", "schema_version", "baseline", "expected"}:
        raise ValueError("Session execution-profile metadata is malformed.")
    if (
        raw["record_type"] != _EXECUTION_PROFILE_RECORD_TYPE
        or raw["schema_version"] != EXECUTION_PROFILE_SCHEMA_VERSION
    ):
        raise ValueError("Session execution-profile metadata version is unsupported.")
    # Revalidate after every backend round trip. No raw component material is stored.
    return ExecutionProfileIdentity.model_validate(
        copy_durable_json_value(raw["expected"], "execution_profile.expected")
    )


def _available_component(
    component_class: ExecutionProfileComponentClass,
    strength: ExecutionProfileIdentityStrength,
    material: Any,
) -> ExecutionProfileComponentIdentity:
    fingerprint = sha256(
        canonical_durable_json_bytes(material, f"execution_profile.{component_class.value}")
    ).hexdigest()
    return ExecutionProfileComponentIdentity(
        component_class=component_class,
        strength=strength,
        availability=ExecutionProfileIdentityAvailability.AVAILABLE,
        fingerprint=fingerprint,
    )


def _unavailable_component(
    component_class: ExecutionProfileComponentClass,
) -> ExecutionProfileComponentIdentity:
    return ExecutionProfileComponentIdentity(
        component_class=component_class,
        strength=ExecutionProfileIdentityStrength.UNAVAILABLE,
        availability=ExecutionProfileIdentityAvailability.UNAVAILABLE,
        fingerprint=None,
    )


def _profile_fingerprint(
    components: tuple[ExecutionProfileComponentIdentity, ...],
) -> str:
    material = {
        "schema_version": EXECUTION_PROFILE_SCHEMA_VERSION,
        "components": [component.model_dump(mode="json") for component in components],
    }
    return sha256(canonical_durable_json_bytes(material, "execution_profile")).hexdigest()
