"""Pure contracts for selecting registered tools for one model step."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from functools import cached_property
from hashlib import sha256
from typing import Any, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_serializer,
    field_validator,
    model_validator,
)

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    canonical_durable_json_bytes,
    copy_durable_json_object,
    freeze_json_value,
    require_durable_clean_nonblank,
    require_durable_text,
    require_execution_unit_id,
    thaw_json_value,
)
from cayu.core.execution_identity import (
    ExecutionProfileBehaviorIdentity,
    copy_execution_profile_behavior_identity,
)
from cayu.core.tools import ToolEffect, ToolResult

TOOL_EXPOSURE_SCHEMA_VERSION = 1
TOOL_CAPABILITY_CEILING_SCHEMA_VERSION = 1
TOOL_EXPOSURE_PROFILE_ID_MAX_CHARS = 256
TOOL_EXPOSURE_MAX_REGISTERED_TOOLS = 10_000
TOOL_EXPOSURE_MAX_CATALOG_BYTES = 32 * 1024 * 1024
TOOL_EXPOSURE_METADATA_MAX_ENTRIES = 64
TOOL_EXPOSURE_METADATA_MAX_BYTES = 4 * 1024
ALL_REGISTERED_TOOLS_PROFILE_ID = "cayu:all-registered-tools:v1"
NOT_EXPOSED_IN_REQUEST_REASON = "not_exposed_in_request"
TOOL_CAPABILITY_CEILING_METADATA_KEY = "cayu:tool_capability_ceiling"


def _sha256_durable_json(value: Any, field_name: str) -> str:
    return sha256(canonical_durable_json_bytes(value, field_name)).hexdigest()


def _copy_bounded_sequence(
    value: object,
    *,
    field_name: str,
    max_items: int,
) -> tuple[Any, ...]:
    if isinstance(value, str | bytes | bytearray | Mapping):
        raise TypeError(f"{field_name} must be an iterable, not text or a mapping.")
    try:
        iterator = iter(cast("Iterable[Any]", value))
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable.") from exc
    copied: list[Any] = []
    for index, item in enumerate(iterator):
        if index >= max_items:
            raise ValueError(f"{field_name} cannot contain more than {max_items} items.")
        copied.append(item)
    return tuple(copied)


def _copy_bounded_metadata(value: object, field_name: str) -> dict[str, Any]:
    copied = copy_durable_json_object(value, field_name)
    if len(copied) > TOOL_EXPOSURE_METADATA_MAX_ENTRIES:
        raise ValueError(
            f"{field_name} cannot contain more than "
            f"{TOOL_EXPOSURE_METADATA_MAX_ENTRIES} top-level entries."
        )
    encoded = canonical_durable_json_bytes(copied, field_name)
    if len(encoded) > TOOL_EXPOSURE_METADATA_MAX_BYTES:
        raise ValueError(
            f"{field_name} cannot exceed {TOOL_EXPOSURE_METADATA_MAX_BYTES} canonical JSON bytes."
        )
    return copied


def _validate_profile_id(value: str, field_name: str = "profile_id") -> str:
    value = require_durable_clean_nonblank(value, field_name)
    if len(value) > TOOL_EXPOSURE_PROFILE_ID_MAX_CHARS:
        raise ValueError(
            f"{field_name} cannot exceed {TOOL_EXPOSURE_PROFILE_ID_MAX_CHARS} characters."
        )
    return value


def _validate_tool_name(value: str, field_name: str) -> str:
    return require_durable_clean_nonblank(value, field_name)


class RegisteredToolCapability(BaseModel):
    """Immutable, callable-free summary of one registered application tool.

    The summary is safe to pass to an application-owned exposure policy: it
    contains the declared model contract and execution classifications, but no
    live tool, policy, runner, environment, credential, or secret handle.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="never",
    )

    name: str
    description: str = ""
    input_schema: Mapping[str, Any] = Field(default_factory=dict)
    parallel_safe: StrictBool = True
    effect: ToolEffect = ToolEffect.EXTERNAL
    publishes_arguments: StrictBool = True
    workspace_mutation: StrictBool = False
    schema_fingerprint: str = ""
    definition_fingerprint: str = ""

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _validate_tool_name(value, "name")

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        return require_durable_text(value, "description")

    @field_validator("input_schema", mode="before")
    @classmethod
    def copy_input_schema(cls, value: object) -> dict[str, Any]:
        return copy_durable_json_object(value, "input_schema")

    @field_validator("input_schema")
    @classmethod
    def freeze_input_schema(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return freeze_json_value(dict(value))

    @field_serializer("input_schema")
    def serialize_input_schema(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return dict(thaw_json_value(value))

    @model_validator(mode="after")
    def validate_and_bind_fingerprints(self) -> RegisteredToolCapability:
        if self.workspace_mutation and self.parallel_safe:
            raise ValueError("Workspace-mutating tools must declare parallel_safe=False.")
        if self.workspace_mutation and self.effect is ToolEffect.NONE:
            raise ValueError("Workspace-mutating tools cannot declare ToolEffect.NONE.")
        schema_fingerprint = _sha256_durable_json(
            thaw_json_value(self.input_schema),
            "registered_tool_capability.input_schema",
        )
        if self.schema_fingerprint not in {"", schema_fingerprint}:
            raise ValueError("schema_fingerprint does not match input_schema.")
        definition_fingerprint = _sha256_durable_json(
            {
                "record_type": "cayu.registered-tool-capability",
                "schema_version": TOOL_EXPOSURE_SCHEMA_VERSION,
                "name": self.name,
                "description": self.description,
                "schema_fingerprint": schema_fingerprint,
                "parallel_safe": self.parallel_safe,
                "effect": self.effect.value,
                "publishes_arguments": self.publishes_arguments,
                "workspace_mutation": self.workspace_mutation,
            },
            "registered_tool_capability",
        )
        if self.definition_fingerprint not in {"", definition_fingerprint}:
            raise ValueError("definition_fingerprint does not match the capability.")
        object.__setattr__(self, "schema_fingerprint", schema_fingerprint)
        object.__setattr__(self, "definition_fingerprint", definition_fingerprint)
        return self

    def input_schema_copy(self) -> dict[str, Any]:
        """Return an owned ordinary-JSON copy for provider/profile projection."""

        return dict(thaw_json_value(self.input_schema))

    @cached_property
    def _canonical_size_bytes(self) -> int:
        return len(
            canonical_durable_json_bytes(
                {
                    "name": self.name,
                    "description": self.description,
                    "input_schema": self.input_schema_copy(),
                    "parallel_safe": self.parallel_safe,
                    "effect": self.effect.value,
                    "publishes_arguments": self.publishes_arguments,
                    "workspace_mutation": self.workspace_mutation,
                    "schema_fingerprint": self.schema_fingerprint,
                    "definition_fingerprint": self.definition_fingerprint,
                },
                "registered_tool_capability",
            )
        )


def _revalidate_registered_tool_capability(
    value: RegisteredToolCapability,
) -> RegisteredToolCapability:
    """Detach one caller-owned capability from undeclared model state."""

    return RegisteredToolCapability(
        name=value.name,
        description=value.description,
        input_schema=value.input_schema,
        parallel_safe=value.parallel_safe,
        effect=value.effect,
        publishes_arguments=value.publishes_arguments,
        workspace_mutation=value.workspace_mutation,
        schema_fingerprint=value.schema_fingerprint,
        definition_fingerprint=value.definition_fingerprint,
    )


class ToolCapabilityCeiling(BaseModel):
    """Immutable maximum application-tool set for one durable session.

    Callers name already registered tools. The runtime canonicalizes those names
    to registration order before the ceiling becomes session authority. Its
    fingerprint is safe identity evidence; it does not contain schemas, tool
    implementations, policy bodies, credentials, or arguments.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    schema_version: Literal[1] = TOOL_CAPABILITY_CEILING_SCHEMA_VERSION
    tool_names: tuple[str, ...] = ()
    fingerprint: str = ""

    @field_validator("tool_names", mode="before")
    @classmethod
    def copy_tool_names(cls, value: object) -> tuple[Any, ...]:
        return _copy_bounded_sequence(
            value,
            field_name="tool_names",
            max_items=TOOL_EXPOSURE_MAX_REGISTERED_TOOLS,
        )

    @field_validator("tool_names")
    @classmethod
    def validate_tool_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        names = tuple(
            _validate_tool_name(name, f"tool_names[{index}]") for index, name in enumerate(value)
        )
        if len(names) != len(set(names)):
            raise ValueError("tool_names must contain unique tool names.")
        return names

    @model_validator(mode="after")
    def validate_and_bind_fingerprint(self) -> ToolCapabilityCeiling:
        fingerprint = _sha256_durable_json(
            {
                "record_type": "cayu.tool-capability-ceiling",
                "schema_version": self.schema_version,
                "tool_names": list(self.tool_names),
            },
            "tool_capability_ceiling",
        )
        if self.fingerprint not in {"", fingerprint}:
            raise ValueError("fingerprint does not match the tool capability ceiling.")
        object.__setattr__(self, "fingerprint", fingerprint)
        return self


def copy_tool_capability_ceiling(
    ceiling: ToolCapabilityCeiling,
) -> ToolCapabilityCeiling:
    """Return a detached, fully revalidated durable ceiling."""

    if type(ceiling) is not ToolCapabilityCeiling:
        raise TypeError("ceiling must be a ToolCapabilityCeiling.")
    return ToolCapabilityCeiling.model_validate(ceiling.model_dump(mode="python", warnings=False))


def resolve_tool_capability_ceiling(
    requested: ToolCapabilityCeiling | None,
    registered_tools: tuple[RegisteredToolCapability, ...],
    *,
    maximum: ToolCapabilityCeiling | None = None,
) -> ToolCapabilityCeiling:
    """Canonicalize a new or narrowed ceiling against registered capabilities.

    ``maximum`` is the durable parent/session ceiling. A requested ceiling may
    only preserve or narrow it. Maximum names absent from a reconstructed
    catalog are removed; no newly registered or child-only name can enter the
    result.
    """

    capabilities = _copy_capability_sequence(registered_tools, "registered_tools")
    registered_names = tuple(capability.name for capability in capabilities)
    if len(registered_names) != len(set(registered_names)):
        raise ValueError("registered_tools must contain unique tool names.")
    registered_name_set = frozenset(registered_names)

    if requested is not None:
        requested = copy_tool_capability_ceiling(requested)
    if maximum is not None:
        maximum = copy_tool_capability_ceiling(maximum)
        maximum_name_set = frozenset(maximum.tool_names)
        maximum_names = tuple(name for name in maximum.tool_names if name in registered_name_set)
    else:
        maximum_names = registered_names
        maximum_name_set = registered_name_set

    selected_names = maximum_names if requested is None else requested.tool_names
    if any(name not in registered_name_set for name in selected_names):
        raise ValueError("The requested tool capability ceiling contains an unregistered tool.")
    if any(name not in maximum_name_set for name in selected_names):
        raise ValueError("A tool capability ceiling may be narrowed but never widened.")
    selected_name_set = frozenset(selected_names)
    canonical_names = tuple(name for name in registered_names if name in selected_name_set)
    return ToolCapabilityCeiling(tool_names=canonical_names)


def tool_capability_ceiling_from_session_metadata(
    metadata: Mapping[str, Any],
) -> ToolCapabilityCeiling:
    """Load copied durable ceiling authority from runtime-owned session metadata."""

    if not isinstance(metadata, Mapping):
        raise TypeError("session metadata must be a mapping.")
    raw = metadata.get(TOOL_CAPABILITY_CEILING_METADATA_KEY)
    if raw is None:
        raise ValueError("Session metadata has no durable tool capability ceiling.")
    return ToolCapabilityCeiling.model_validate(raw)


def session_metadata_with_tool_capability_ceiling(
    metadata: Mapping[str, Any],
    ceiling: ToolCapabilityCeiling,
) -> dict[str, Any]:
    """Copy metadata and set exact runtime-owned ceiling authority."""

    copied = copy_durable_json_object(metadata, "session.metadata")
    copied[TOOL_CAPABILITY_CEILING_METADATA_KEY] = copy_tool_capability_ceiling(ceiling).model_dump(
        mode="json"
    )
    return copy_durable_json_object(copied, "session.metadata")


def session_metadata_after_tool_capability_ceiling_narrowing(
    metadata: Mapping[str, Any],
    ceiling: ToolCapabilityCeiling,
) -> dict[str, Any]:
    """Atomically applicable metadata update that rejects every widening path."""

    candidate = copy_tool_capability_ceiling(ceiling)
    current = tool_capability_ceiling_from_session_metadata(metadata)
    if not frozenset(candidate.tool_names) <= frozenset(current.tool_names):
        raise ValueError("A tool capability ceiling may be narrowed but never widened.")
    return session_metadata_with_tool_capability_ceiling(metadata, candidate)


def _copy_capability_sequence(
    value: object,
    field_name: str,
) -> tuple[RegisteredToolCapability, ...]:
    """Copy capabilities while enforcing count and canonical-size limits eagerly."""

    if isinstance(value, str | bytes | bytearray | Mapping):
        raise TypeError(f"{field_name} must be an iterable, not text or a mapping.")
    try:
        iterator = iter(cast("Iterable[Any]", value))
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable.") from exc

    copied: list[RegisteredToolCapability] = []
    catalog_bytes = 2  # JSON array delimiters.
    for index, item in enumerate(iterator):
        if index >= TOOL_EXPOSURE_MAX_REGISTERED_TOOLS:
            raise ValueError(
                f"{field_name} cannot contain more than {TOOL_EXPOSURE_MAX_REGISTERED_TOOLS} items."
            )
        capability = (
            _revalidate_registered_tool_capability(item)
            if isinstance(item, RegisteredToolCapability)
            else RegisteredToolCapability.model_validate(item)
        )
        catalog_bytes += capability._canonical_size_bytes + (1 if copied else 0)
        if catalog_bytes > TOOL_EXPOSURE_MAX_CATALOG_BYTES:
            raise ValueError(
                f"{field_name} cannot exceed "
                f"{TOOL_EXPOSURE_MAX_CATALOG_BYTES} canonical JSON bytes in total."
            )
        copied.append(capability)
    return tuple(copied)


class ToolExposurePolicyRequest(BaseModel):
    """Immutable, bounded input supplied to a tool-exposure policy."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    session_id: str
    agent_name: str
    provider_name: str
    model: str
    step: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    transcript_cursor: StrictInt = Field(default=0, ge=0)
    registered_tools: tuple[RegisteredToolCapability, ...]
    capability_ceiling: tuple[str, ...]
    previous_profile_id: str | None = None
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "agent_name", "provider_name", "model")
    @classmethod
    def validate_identity_text(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("previous_profile_id")
    @classmethod
    def validate_previous_profile_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_profile_id(value, "previous_profile_id")

    @field_validator("registered_tools", mode="before")
    @classmethod
    def copy_registered_tools(
        cls,
        value: object,
    ) -> tuple[RegisteredToolCapability, ...]:
        return _copy_capability_sequence(value, "registered_tools")

    @field_validator("capability_ceiling", mode="before")
    @classmethod
    def copy_capability_ceiling(cls, value: object) -> tuple[Any, ...]:
        return _copy_bounded_sequence(
            value,
            field_name="capability_ceiling",
            max_items=TOOL_EXPOSURE_MAX_REGISTERED_TOOLS,
        )

    @field_validator("capability_ceiling")
    @classmethod
    def validate_capability_ceiling(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        names = tuple(
            _validate_tool_name(name, f"capability_ceiling[{index}]")
            for index, name in enumerate(value)
        )
        if len(names) != len(set(names)):
            raise ValueError("capability_ceiling must contain unique tool names.")
        return names

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: object) -> dict[str, Any]:
        return _copy_bounded_metadata(value, "metadata")

    @field_validator("metadata")
    @classmethod
    def freeze_metadata(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return freeze_json_value(dict(value))

    @field_serializer("metadata")
    def serialize_metadata(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return dict(thaw_json_value(value))

    @model_validator(mode="after")
    def validate_catalog(self) -> ToolExposurePolicyRequest:
        registered_names = tuple(tool.name for tool in self.registered_tools)
        if len(registered_names) != len(set(registered_names)):
            raise ValueError("registered_tools must contain unique tool names.")
        registered_name_set = set(registered_names)
        unknown = [name for name in self.capability_ceiling if name not in registered_name_set]
        if unknown:
            raise ValueError("capability_ceiling contains an unregistered tool name.")
        ceiling_set = set(self.capability_ceiling)
        canonical_ceiling = tuple(name for name in registered_names if name in ceiling_set)
        if self.capability_ceiling != canonical_ceiling:
            raise ValueError("capability_ceiling must preserve registered tool order.")
        return self

    @property
    def eligible_tools(self) -> tuple[RegisteredToolCapability, ...]:
        """Return ceiling-constrained capabilities in canonical registration order."""

        ceiling = set(self.capability_ceiling)
        return tuple(tool for tool in self.registered_tools if tool.name in ceiling)


class ToolExposureDecision(BaseModel):
    """A policy decision naming registered tools, never tool implementations."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    profile_id: str
    tool_names: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("profile_id")
    @classmethod
    def validate_profile_id(cls, value: str) -> str:
        return _validate_profile_id(value)

    @field_validator("tool_names", mode="before")
    @classmethod
    def copy_tool_names(cls, value: object) -> tuple[Any, ...]:
        return _copy_bounded_sequence(
            value,
            field_name="tool_names",
            max_items=TOOL_EXPOSURE_MAX_REGISTERED_TOOLS,
        )

    @field_validator("tool_names")
    @classmethod
    def validate_tool_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        names = tuple(
            _validate_tool_name(name, f"tool_names[{index}]") for index, name in enumerate(value)
        )
        if len(names) != len(set(names)):
            raise ValueError("tool_names must contain unique tool names.")
        return names

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: object) -> dict[str, Any]:
        return _copy_bounded_metadata(value, "metadata")

    @field_validator("metadata")
    @classmethod
    def freeze_metadata(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return freeze_json_value(dict(value))

    @field_serializer("metadata")
    def serialize_metadata(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return dict(thaw_json_value(value))


class ResolvedToolExposure(BaseModel):
    """Canonical, descriptor-bound result of validating one policy decision."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    schema_version: Literal[1] = TOOL_EXPOSURE_SCHEMA_VERSION
    profile_id: str
    tools: tuple[RegisteredToolCapability, ...]
    registered_count: StrictInt = Field(ge=0, le=TOOL_EXPOSURE_MAX_REGISTERED_TOOLS)
    ceiling_count: StrictInt = Field(ge=0, le=TOOL_EXPOSURE_MAX_REGISTERED_TOOLS)
    metadata: Mapping[str, Any] = Field(default_factory=dict)
    fingerprint: str = ""

    @field_validator("profile_id")
    @classmethod
    def validate_profile_id(cls, value: str) -> str:
        return _validate_profile_id(value)

    @field_validator("tools", mode="before")
    @classmethod
    def copy_tools(cls, value: object) -> tuple[RegisteredToolCapability, ...]:
        return _copy_capability_sequence(value, "tools")

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: object) -> dict[str, Any]:
        return _copy_bounded_metadata(value, "metadata")

    @field_validator("metadata")
    @classmethod
    def freeze_metadata(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return freeze_json_value(dict(value))

    @field_serializer("metadata")
    def serialize_metadata(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return dict(thaw_json_value(value))

    @model_validator(mode="after")
    def validate_and_bind_fingerprint(self) -> ResolvedToolExposure:
        names = tuple(tool.name for tool in self.tools)
        if len(names) != len(set(names)):
            raise ValueError("tools must contain unique tool names.")
        if self.ceiling_count > self.registered_count:
            raise ValueError("ceiling_count cannot exceed registered_count.")
        if len(self.tools) > self.ceiling_count:
            raise ValueError("Resolved tools cannot exceed the capability ceiling.")
        fingerprint = _sha256_durable_json(
            {
                "record_type": "cayu.resolved-tool-exposure",
                "schema_version": self.schema_version,
                "profile_id": self.profile_id,
                "tool_definition_fingerprints": [
                    tool.definition_fingerprint for tool in self.tools
                ],
            },
            "resolved_tool_exposure",
        )
        if self.fingerprint not in {"", fingerprint}:
            raise ValueError("fingerprint does not match the resolved exposure.")
        object.__setattr__(self, "fingerprint", fingerprint)
        return self

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(tool.name for tool in self.tools)


class ToolExposure(BaseModel):
    """Content-minimized public evidence for one prepared model-step tool projection.

    The record intentionally contains no separate tool-name list, definitions,
    policy metadata, arguments, or policy reasoning. ``profile_id`` is a public,
    application-selected non-secret label. The fingerprint binds the record to the
    exact ordered definitions retained privately by the runtime.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    schema_version: Literal[1] = TOOL_EXPOSURE_SCHEMA_VERSION
    execution_profile_fingerprint: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    profile_id: str
    exposure_fingerprint: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    registered_count: StrictInt = Field(ge=0, le=TOOL_EXPOSURE_MAX_REGISTERED_TOOLS)
    ceiling_count: StrictInt = Field(ge=0, le=TOOL_EXPOSURE_MAX_REGISTERED_TOOLS)
    exposed_count: StrictInt = Field(ge=0, le=TOOL_EXPOSURE_MAX_REGISTERED_TOOLS)
    profile_changed: StrictBool
    step: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    provider_name: str
    model: str
    model_step_id: str

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != TOOL_EXPOSURE_SCHEMA_VERSION:
            raise ValueError("Tool exposure schema_version must be the integer 1.")
        return value

    @field_validator("profile_id")
    @classmethod
    def validate_profile_id(cls, value: str) -> str:
        return _validate_profile_id(value)

    @field_validator("provider_name", "model")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("model_step_id")
    @classmethod
    def validate_model_step_id(cls, value: str) -> str:
        try:
            return require_execution_unit_id(value, "model_step_id")
        except ValueError:
            # Public events replace private execution-unit IDs with scoped linkage
            # aliases. Keep the evidence record usable on both sides of that boundary.
            from cayu.runtime._event_projection import (
                PRIVATE_EVENT_AUTHORITY,
                public_event_linkage_sequence,
            )

            if (
                value == PRIVATE_EVENT_AUTHORITY
                or public_event_linkage_sequence(
                    value,
                    field_name="model_step_id",
                )
                is not None
            ):
                return value
            raise

    @model_validator(mode="after")
    def validate_counts(self) -> ToolExposure:
        if self.ceiling_count > self.registered_count:
            raise ValueError("ceiling_count cannot exceed registered_count.")
        if self.exposed_count > self.ceiling_count:
            raise ValueError("exposed_count cannot exceed ceiling_count.")
        return self


def tool_exposure_record(
    exposure: ResolvedToolExposure,
    *,
    profile_changed: bool,
    step: int,
    provider_name: str,
    model: str,
    model_step_id: str,
    execution_profile_fingerprint: str,
) -> ToolExposure:
    """Project one frozen descriptor snapshot into content-minimized public evidence."""

    if type(exposure) is not ResolvedToolExposure:
        raise TypeError("exposure must be a ResolvedToolExposure.")
    if type(profile_changed) is not bool:
        raise TypeError("profile_changed must be a bool.")
    return ToolExposure(
        execution_profile_fingerprint=execution_profile_fingerprint,
        profile_id=exposure.profile_id,
        exposure_fingerprint=exposure.fingerprint,
        registered_count=exposure.registered_count,
        ceiling_count=exposure.ceiling_count,
        exposed_count=len(exposure.tools),
        profile_changed=profile_changed,
        step=step,
        provider_name=provider_name,
        model=model,
        model_step_id=model_step_id,
    )


class ResolvedToolExposureAuthority(BaseModel):
    """Compact durable authority derived from one resolved request snapshot."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    schema_version: Literal[1] = TOOL_EXPOSURE_SCHEMA_VERSION
    profile_id: str
    tool_names: tuple[str, ...]
    registered_count: StrictInt = Field(ge=0, le=TOOL_EXPOSURE_MAX_REGISTERED_TOOLS)
    ceiling_count: StrictInt = Field(ge=0, le=TOOL_EXPOSURE_MAX_REGISTERED_TOOLS)
    fingerprint: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )

    @field_validator("profile_id")
    @classmethod
    def validate_profile_id(cls, value: str) -> str:
        return _validate_profile_id(value)

    @field_validator("tool_names", mode="before")
    @classmethod
    def copy_tool_names(cls, value: object) -> tuple[Any, ...]:
        return _copy_bounded_sequence(
            value,
            field_name="tool_names",
            max_items=TOOL_EXPOSURE_MAX_REGISTERED_TOOLS,
        )

    @field_validator("tool_names")
    @classmethod
    def validate_tool_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        names = tuple(
            _validate_tool_name(name, f"tool_names[{index}]") for index, name in enumerate(value)
        )
        if len(names) != len(set(names)):
            raise ValueError("tool_names must contain unique tool names.")
        return names

    @model_validator(mode="after")
    def validate_counts(self) -> ResolvedToolExposureAuthority:
        if self.ceiling_count > self.registered_count:
            raise ValueError("ceiling_count cannot exceed registered_count.")
        if len(self.tool_names) > self.ceiling_count:
            raise ValueError("tool_names cannot exceed the capability ceiling.")
        return self


def resolved_tool_exposure_authority(
    exposure: ResolvedToolExposure,
) -> ResolvedToolExposureAuthority:
    """Project a resolved descriptor snapshot into compact durable authority."""

    if type(exposure) is not ResolvedToolExposure:
        raise TypeError("exposure must be a ResolvedToolExposure.")
    return ResolvedToolExposureAuthority(
        profile_id=exposure.profile_id,
        tool_names=exposure.tool_names,
        registered_count=exposure.registered_count,
        ceiling_count=exposure.ceiling_count,
        fingerprint=exposure.fingerprint,
    )


def copy_resolved_tool_exposure_authority(
    authority: ResolvedToolExposureAuthority,
) -> ResolvedToolExposureAuthority:
    """Return a detached, fully revalidated durable exposure authority."""

    if type(authority) is not ResolvedToolExposureAuthority:
        raise TypeError("authority must be a ResolvedToolExposureAuthority.")
    return ResolvedToolExposureAuthority.model_validate(authority.model_dump(mode="python"))


def validate_resolved_tool_exposure_authority(
    authority: ResolvedToolExposureAuthority,
    registered_tools: tuple[RegisteredToolCapability, ...],
) -> ResolvedToolExposureAuthority:
    """Bind compact durable authority back to the current registered catalog."""

    copied, _reconstructed = _bind_resolved_tool_exposure_authority(
        authority,
        registered_tools,
    )
    return copied


def resolved_tool_exposure_from_authority(
    authority: ResolvedToolExposureAuthority,
    registered_tools: tuple[RegisteredToolCapability, ...],
) -> ResolvedToolExposure:
    """Reconstruct one validated frozen snapshot from compact durable authority."""

    _copied, reconstructed = _bind_resolved_tool_exposure_authority(
        authority,
        registered_tools,
    )
    return reconstructed


def _bind_resolved_tool_exposure_authority(
    authority: ResolvedToolExposureAuthority,
    registered_tools: tuple[RegisteredToolCapability, ...],
) -> tuple[ResolvedToolExposureAuthority, ResolvedToolExposure]:
    """Return detached authority and its catalog-bound descriptor snapshot."""

    copied = copy_resolved_tool_exposure_authority(authority)
    capabilities = _copy_capability_sequence(registered_tools, "registered_tools")
    if copied.registered_count != len(capabilities):
        raise ValueError("Tool exposure authority conflicts with the registered tool count.")
    selected_names = frozenset(copied.tool_names)
    selected_tools = tuple(tool for tool in capabilities if tool.name in selected_names)
    if tuple(tool.name for tool in selected_tools) != copied.tool_names:
        raise ValueError("Tool exposure authority conflicts with registered tool order.")
    reconstructed = ResolvedToolExposure(
        profile_id=copied.profile_id,
        tools=selected_tools,
        registered_count=copied.registered_count,
        ceiling_count=copied.ceiling_count,
    )
    if reconstructed.fingerprint != copied.fingerprint:
        raise ValueError("Tool exposure authority conflicts with registered definitions.")
    return copied, reconstructed


def copy_resolved_tool_exposure(
    exposure: ResolvedToolExposure,
) -> ResolvedToolExposure:
    """Return a detached, fully revalidated exposure snapshot."""

    if type(exposure) is not ResolvedToolExposure:
        raise TypeError("exposure must be a ResolvedToolExposure.")
    return ResolvedToolExposure.model_validate(exposure.model_dump(mode="python"))


def unexposed_tool_result() -> ToolResult:
    """Build the fixed provider-facing failure for a call absent from a request."""

    return ToolResult(
        content="Tool unavailable for this model request.",
        is_error=True,
    )


class ToolExposurePolicy(ABC):
    """Pure application policy selecting a subset of registered tools.

    Implementations receive only copied immutable records. They must be local,
    deterministic, and side-effect-free: selection is part of preparing one
    model request, not an authorization or model-routing hook.
    """

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity | None:
        """Return a stable application declaration, or ``None`` when non-portable."""

        return None

    @abstractmethod
    def select(self, request: ToolExposurePolicyRequest) -> ToolExposureDecision:
        """Select registered tool names for one model step."""


class AllRegisteredToolsExposurePolicy(ToolExposurePolicy):
    """Compatibility policy exposing every tool inside the effective ceiling."""

    def _execution_profile_material(self) -> dict[str, object]:
        return {"profile_id": ALL_REGISTERED_TOOLS_PROFILE_ID}

    def select(self, request: ToolExposurePolicyRequest) -> ToolExposureDecision:
        return ToolExposureDecision(
            profile_id=ALL_REGISTERED_TOOLS_PROFILE_ID,
            tool_names=tuple(tool.name for tool in request.eligible_tools),
        )


class StaticToolExposurePolicy(ToolExposurePolicy):
    """Expose one deterministic named allow-list, including an empty list."""

    def __init__(self, *, profile_id: str, tools: Iterable[str]) -> None:
        tool_names = cast(
            "tuple[str, ...]",
            _copy_bounded_sequence(
                tools,
                field_name="tools",
                max_items=TOOL_EXPOSURE_MAX_REGISTERED_TOOLS,
            ),
        )
        validated = ToolExposureDecision(profile_id=profile_id, tool_names=tool_names)
        self._exposure = ToolExposureDecision(
            profile_id=validated.profile_id,
            tool_names=tuple(sorted(validated.tool_names)),
        )

    @property
    def profile_id(self) -> str:
        return self._exposure.profile_id

    @property
    def tools(self) -> tuple[str, ...]:
        return self._exposure.tool_names

    def _execution_profile_material(self) -> dict[str, object]:
        return {"profile_id": self.profile_id, "tools": list(self.tools)}

    def select(self, request: ToolExposurePolicyRequest) -> ToolExposureDecision:
        return self._exposure


def _copy_tool_exposure_policy_request(
    request: ToolExposurePolicyRequest,
) -> ToolExposurePolicyRequest:
    """Return a declared-field-only policy request with revalidated capabilities."""

    return ToolExposurePolicyRequest(
        session_id=request.session_id,
        agent_name=request.agent_name,
        provider_name=request.provider_name,
        model=request.model,
        step=request.step,
        transcript_cursor=request.transcript_cursor,
        registered_tools=request.registered_tools,
        capability_ceiling=request.capability_ceiling,
        previous_profile_id=request.previous_profile_id,
        metadata=thaw_json_value(request.metadata),
    )


def _policy_request_declared_state(request: ToolExposurePolicyRequest) -> tuple[Any, ...]:
    """Capture bounded declared state whose mutation would invalidate selection."""

    return (
        request.session_id,
        request.agent_name,
        request.provider_name,
        request.model,
        request.step,
        request.transcript_cursor,
        tuple((tool.name, tool.definition_fingerprint) for tool in request.registered_tools),
        request.capability_ceiling,
        request.previous_profile_id,
        canonical_durable_json_bytes(
            thaw_json_value(request.metadata),
            "tool_exposure_policy_request.metadata",
        ),
    )


def resolve_tool_exposure(
    policy: ToolExposurePolicy,
    request: ToolExposurePolicyRequest,
) -> ResolvedToolExposure:
    """Run and validate a pure policy against registered and ceiling state."""

    if not isinstance(policy, ToolExposurePolicy):
        raise TypeError("policy must be a ToolExposurePolicy.")
    if type(request) is not ToolExposurePolicyRequest:
        raise TypeError("request must be a ToolExposurePolicyRequest.")

    policy_request = _copy_tool_exposure_policy_request(request)
    request_state = _policy_request_declared_state(policy_request)
    registered_tools = policy_request.registered_tools
    registered_names = tuple(tool.name for tool in registered_tools)
    registered_name_set = frozenset(registered_names)
    registered_definition_fingerprints = {
        tool.name: tool.definition_fingerprint for tool in registered_tools
    }
    capability_ceiling = policy_request.capability_ceiling
    ceiling = frozenset(capability_ceiling)

    identity_before = copy_execution_profile_behavior_identity(policy.execution_profile_identity)
    decision = policy.select(policy_request)
    identity_after = copy_execution_profile_behavior_identity(policy.execution_profile_identity)
    if identity_before != identity_after:
        raise RuntimeError("Tool exposure policy identity changed during selection.")
    try:
        request_state_after = _policy_request_declared_state(policy_request)
    except Exception as exc:
        raise RuntimeError("Tool exposure policy mutated its request during selection.") from exc
    if request_state_after != request_state:
        raise RuntimeError("Tool exposure policy mutated its request during selection.")
    if type(decision) is not ToolExposureDecision:
        raise TypeError("Tool exposure policies must return ToolExposureDecision instances.")
    copied_decision = ToolExposureDecision(
        profile_id=decision.profile_id,
        tool_names=decision.tool_names,
        metadata=thaw_json_value(decision.metadata),
    )

    if any(name not in registered_name_set for name in copied_decision.tool_names):
        raise ValueError("Tool exposure selected an unregistered tool.")
    if any(name not in ceiling for name in copied_decision.tool_names):
        raise ValueError("Tool exposure selected a tool outside the capability ceiling.")
    selected = set(copied_decision.tool_names)
    selected_tools = tuple(
        tool
        for registered_name, tool in zip(registered_names, registered_tools, strict=True)
        if registered_name in selected
    )
    expected_definition_fingerprints = tuple(
        registered_definition_fingerprints[registered_name]
        for registered_name in registered_names
        if registered_name in selected
    )
    try:
        resolved = ResolvedToolExposure(
            profile_id=copied_decision.profile_id,
            tools=selected_tools,
            registered_count=len(registered_tools),
            ceiling_count=len(capability_ceiling),
            metadata=thaw_json_value(copied_decision.metadata),
        )
    except Exception as exc:
        raise RuntimeError("Tool exposure policy mutated a capability during selection.") from exc
    if (
        tuple(tool.definition_fingerprint for tool in resolved.tools)
        != expected_definition_fingerprints
    ):
        raise RuntimeError("Tool exposure policy mutated a capability during selection.")
    return resolved
