"""Canonical provider-independent tool catalogue contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import cached_property
from hashlib import sha256
from types import MappingProxyType
from typing import Any, Literal, cast
from urllib.parse import quote, unquote

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
    thaw_json_value,
)
from cayu.core.execution_identity import (
    ExecutionProfileBehaviorIdentity,
    copy_execution_profile_behavior_identity,
)
from cayu.core.isolated_tools import ISOLATED_TOOL_PROTOCOL_NAME
from cayu.core.tools import ToolEffect

TOOL_DESCRIPTOR_SCHEMA_VERSION = 2
TOOL_CATALOGUE_SCHEMA_VERSION = 1
TOOL_CATALOGUE_MAX_TOOLS = 10_000
TOOL_CATALOGUE_MAX_BYTES = 32 * 1024 * 1024
TOOL_ID_MAX_CHARS = 512
TOOL_PROVENANCE_TEXT_MAX_CHARS = 512

CALL_TOOL_NAME = "call_tool"
SEARCH_TOOLS_NAME = "search_tools"
STRUCTURED_OUTPUT_TOOL_NAME = "__cayu_submit_structured_output"
FRAMEWORK_TOOL_NAMES = frozenset(
    {
        CALL_TOOL_NAME,
        SEARCH_TOOLS_NAME,
        STRUCTURED_OUTPUT_TOOL_NAME,
    }
)

_SHA256_IDENTIFIER_PATTERN = r"^sha256:[0-9a-f]{64}$"
_MCP_SOURCE_TOOL_IDENTITY_VERSION = 1
_TOOL_ID_SAFE_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"


def _sha256_identifier(value: object, field_name: str) -> str:
    digest = sha256(canonical_durable_json_bytes(value, field_name)).hexdigest()
    return f"sha256:{digest}"


def validate_tool_catalogue_revision(
    value: str,
    field_name: str = "catalogue_revision",
) -> str:
    """Validate one deterministic catalogue revision reference."""

    value = require_durable_clean_nonblank(value, field_name)
    if (
        len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{field_name} must be a SHA-256 catalogue revision.")
    return value


def validate_tool_descriptor_version(
    value: str,
    field_name: str = "descriptor_version",
) -> str:
    """Validate one deterministic descriptor-version reference."""

    value = require_durable_clean_nonblank(value, field_name)
    if (
        len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{field_name} must be a SHA-256 descriptor version.")
    return value


def validate_canonical_tool_id(
    value: str,
    field_name: str = "tool_id",
) -> str:
    """Validate one canonical native or MCP tool identity reference."""

    value = require_durable_clean_nonblank(value, field_name)
    if len(value) > TOOL_ID_MAX_CHARS:
        raise ValueError(f"{field_name} cannot exceed {TOOL_ID_MAX_CHARS} characters.")
    if value.startswith("cayu:"):
        encoded_name = value.removeprefix("cayu:")
        try:
            name = unquote(encoded_name, errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{field_name} must contain a canonical Cayu tool identity.") from exc
        if (
            not encoded_name
            or quote(
                require_durable_clean_nonblank(name, field_name),
                safe=_TOOL_ID_SAFE_CHARS,
            )
            != encoded_name
        ):
            raise ValueError(f"{field_name} must contain a canonical Cayu tool identity.")
        return value
    if value.startswith("mcp:"):
        parts = value.split(":")
        if (
            len(parts) == 3
            and all(len(part) == 64 for part in parts[1:])
            and all(character in "0123456789abcdef" for part in parts[1:] for character in part)
        ):
            return value
    raise ValueError(f"{field_name} must contain a canonical Cayu or MCP tool identity.")


def _copy_bounded_items(value: object, field_name: str) -> tuple[Any, ...]:
    if isinstance(value, str | bytes | bytearray | Mapping | BaseModel):
        raise TypeError(f"{field_name} must be an iterable, not text, a mapping, or a model.")
    try:
        iterator = iter(cast("Iterable[Any]", value))
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable.") from exc
    copied: list[Any] = []
    for index, item in enumerate(iterator):
        if index >= TOOL_CATALOGUE_MAX_TOOLS:
            raise ValueError(
                f"{field_name} cannot contain more than {TOOL_CATALOGUE_MAX_TOOLS} items."
            )
        copied.append(item)
    return tuple(copied)


def _validate_bounded_text(value: str, field_name: str) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    if len(value) > TOOL_PROVENANCE_TEXT_MAX_CHARS:
        raise ValueError(f"{field_name} cannot exceed {TOOL_PROVENANCE_TEXT_MAX_CHARS} characters.")
    return value


def validate_application_tool_name(value: str) -> str:
    """Validate one registered application name against Cayu's framework namespace."""

    name = require_durable_clean_nonblank(value, "tool.name")
    if name in FRAMEWORK_TOOL_NAMES:
        raise ValueError(f"Tool name is reserved by the Cayu framework: {name}")
    return name


class ToolDescriptorProvenance(BaseModel):
    """Callable-free origin evidence for one canonical descriptor."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    kind: Literal["cayu", "mcp"] = "cayu"
    source_id: str | None = None
    source_tool_fingerprint: str | None = Field(
        default=None,
        pattern=_SHA256_IDENTIFIER_PATTERN,
    )
    source_contract_fingerprint: str | None = Field(
        default=None,
        pattern=_SHA256_IDENTIFIER_PATTERN,
    )

    @field_validator("source_id")
    @classmethod
    def validate_optional_text(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _validate_bounded_text(value, info.field_name)

    @model_validator(mode="after")
    def validate_shape(self) -> ToolDescriptorProvenance:
        values = (
            self.source_id,
            self.source_tool_fingerprint,
            self.source_contract_fingerprint,
        )
        if self.kind == "cayu" and any(value is not None for value in values):
            raise ValueError("Cayu tool provenance cannot contain MCP source fields.")
        if self.kind == "mcp" and any(value is None for value in values):
            raise ValueError("MCP tool provenance requires complete source identity evidence.")
        if self.kind == "mcp" and (
            self.source_id is None
            or not self.source_id.startswith("sha256:")
            or len(self.source_id) != 71
            or any(character not in "0123456789abcdef" for character in self.source_id[7:])
        ):
            raise ValueError("MCP source_id must be an authoritative SHA-256 identity.")
        return self


def copy_tool_descriptor_provenance(
    provenance: ToolDescriptorProvenance,
) -> ToolDescriptorProvenance:
    """Return a detached, fully revalidated provenance record."""

    if type(provenance) is not ToolDescriptorProvenance:
        raise TypeError("provenance must be a ToolDescriptorProvenance.")
    return ToolDescriptorProvenance.model_validate(provenance.model_dump(mode="python"))


def canonical_tool_id(*, name: str, provenance: ToolDescriptorProvenance) -> str:
    """Return one stable collision-free ID from admitted tool provenance."""

    name = require_durable_clean_nonblank(name, "name")
    provenance = copy_tool_descriptor_provenance(provenance)
    if provenance.kind == "cayu":
        value = f"cayu:{quote(name, safe=_TOOL_ID_SAFE_CHARS)}"
    else:
        if provenance.source_id is None or provenance.source_tool_fingerprint is None:
            raise AssertionError("Validated MCP provenance is incomplete.")
        source_id = provenance.source_id.removeprefix("sha256:")
        source_tool = provenance.source_tool_fingerprint.removeprefix("sha256:")
        value = f"mcp:{source_id}:{source_tool}"
    return validate_canonical_tool_id(value)


def mcp_source_tool_fingerprint(source_tool_name: str) -> str:
    """Return bounded non-disclosing identity for one public MCP source name."""

    source_tool_name = require_durable_clean_nonblank(source_tool_name, "source_tool_name")
    return _sha256_identifier(
        {
            "record_type": "cayu.mcp-source-tool",
            "identity_version": _MCP_SOURCE_TOOL_IDENTITY_VERSION,
            "name": source_tool_name,
        },
        "mcp_source_tool",
    )


class ToolExecutionContract(BaseModel):
    """Callable-free evidence for the boundary that executes one tool."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    boundary: Literal["in_process", "posix_process"] = "in_process"
    timeout_strength: Literal[
        "none",
        "cooperative_in_process",
        "hard_process_deadline",
    ] = "none"
    sandboxed: StrictBool = False
    adapter_identity: ExecutionProfileBehaviorIdentity | None = None
    adapter_configuration_sha256: str | None = Field(
        default=None,
        pattern=_SHA256_IDENTIFIER_PATTERN,
    )
    hard_deadline_seconds: float | None = None
    protocol: str | None = None
    protocol_version: StrictInt | None = None
    max_terminal_payload_bytes: StrictInt | None = Field(
        default=None,
        ge=64 * 1024,
        le=MAX_DURABLE_JSON_INTEGER,
    )

    @field_validator("adapter_identity", mode="before")
    @classmethod
    def copy_adapter_identity(cls, value: object) -> object:
        if isinstance(value, ExecutionProfileBehaviorIdentity):
            copied = copy_execution_profile_behavior_identity(value)
            if copied is None:  # pragma: no cover - narrowed by isinstance
                raise AssertionError("Adapter identity copy unexpectedly returned None.")
            return copied.model_dump(mode="python")
        return value

    @field_validator("protocol")
    @classmethod
    def validate_optional_protocol(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_bounded_text(value, "protocol")

    @field_validator("hard_deadline_seconds", mode="before")
    @classmethod
    def validate_optional_deadline(cls, value: object) -> object:
        if value is None:
            return None
        if type(value) not in {int, float}:
            raise TypeError("hard_deadline_seconds must be numeric or None.")
        normalized = float(cast("int | float", value))
        if not 0 < normalized <= 24 * 60 * 60:
            raise ValueError("hard_deadline_seconds must be positive and at most 24 hours.")
        return normalized

    @model_validator(mode="after")
    def validate_shape(self) -> ToolExecutionContract:
        if self.sandboxed:
            raise ValueError("The host process boundary cannot claim sandboxing.")
        adapter_fields = (
            self.adapter_identity,
            self.adapter_configuration_sha256,
            self.hard_deadline_seconds,
            self.protocol,
            self.protocol_version,
        )
        if self.boundary == "in_process":
            if self.timeout_strength not in {"none", "cooperative_in_process"} or any(
                value is not None for value in adapter_fields
            ):
                raise ValueError("In-process tools cannot claim process-adapter authority.")
        elif (
            self.timeout_strength != "hard_process_deadline"
            or any(value is None for value in adapter_fields)
            or self.protocol != ISOLATED_TOOL_PROTOCOL_NAME
            or self.protocol_version != 1
        ):
            raise ValueError("POSIX process tools require complete hard-deadline evidence.")
        return self


def copy_tool_execution_contract(contract: ToolExecutionContract) -> ToolExecutionContract:
    if type(contract) is not ToolExecutionContract:
        raise TypeError("execution_contract must be a ToolExecutionContract.")
    return ToolExecutionContract.model_validate(contract.model_dump(mode="python"))


class ToolDescriptor(BaseModel):
    """Immutable canonical contract for one runtime-admitted tool."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    schema_version: Literal[2] = TOOL_DESCRIPTOR_SCHEMA_VERSION
    tool_id: str = ""
    name: str
    description: str = ""
    input_schema: Mapping[str, Any] = Field(default_factory=dict)
    parallel_safe: StrictBool = True
    effect: ToolEffect = ToolEffect.EXTERNAL
    publishes_arguments: StrictBool = True
    workspace_mutation: StrictBool = False
    execution_contract: ToolExecutionContract = Field(default_factory=ToolExecutionContract)
    provenance: ToolDescriptorProvenance = Field(default_factory=ToolDescriptorProvenance)
    schema_fingerprint: str = Field(default="", pattern=r"^(?:|sha256:[0-9a-f]{64})$")
    version: str = Field(default="", pattern=r"^(?:|sha256:[0-9a-f]{64})$")

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != TOOL_DESCRIPTOR_SCHEMA_VERSION:
            raise ValueError("Tool descriptor schema_version must be the integer 2.")
        return value

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "name")

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

    @field_validator("provenance", mode="before")
    @classmethod
    def copy_provenance(cls, value: object) -> object:
        if isinstance(value, ToolDescriptorProvenance):
            return value.model_dump(mode="python")
        return value

    @model_validator(mode="after")
    def validate_and_bind_identity(self) -> ToolDescriptor:
        if self.workspace_mutation and self.parallel_safe:
            raise ValueError("Workspace-mutating tools must declare parallel_safe=False.")
        if self.workspace_mutation and self.effect is ToolEffect.NONE:
            raise ValueError("Workspace-mutating tools cannot declare ToolEffect.NONE.")
        expected_tool_id = canonical_tool_id(name=self.name, provenance=self.provenance)
        if self.tool_id not in {"", expected_tool_id}:
            raise ValueError("tool_id does not match the descriptor provenance.")
        schema_fingerprint = _sha256_identifier(
            thaw_json_value(self.input_schema),
            "tool_descriptor.input_schema",
        )
        if self.schema_fingerprint not in {"", schema_fingerprint}:
            raise ValueError("schema_fingerprint does not match input_schema.")
        version = _sha256_identifier(
            {
                "record_type": "cayu.tool-descriptor",
                "schema_version": self.schema_version,
                "tool_id": expected_tool_id,
                "name": self.name,
                "description": self.description,
                "schema_fingerprint": schema_fingerprint,
                "parallel_safe": self.parallel_safe,
                "effect": self.effect.value,
                "publishes_arguments": self.publishes_arguments,
                "workspace_mutation": self.workspace_mutation,
                "execution_contract": self.execution_contract.model_dump(mode="json"),
                "provenance": self.provenance.model_dump(mode="json"),
            },
            "tool_descriptor",
        )
        if self.version not in {"", version}:
            raise ValueError("version does not match the tool descriptor.")
        object.__setattr__(self, "tool_id", expected_tool_id)
        object.__setattr__(self, "schema_fingerprint", schema_fingerprint)
        object.__setattr__(self, "version", version)
        return self

    def input_schema_copy(self) -> dict[str, Any]:
        """Return an owned ordinary-JSON schema copy."""

        return dict(thaw_json_value(self.input_schema))

    def exposure_capability_material(self) -> dict[str, Any]:
        """Return the existing callable-free exposure-policy projection."""

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema_copy(),
            "parallel_safe": self.parallel_safe,
            "effect": self.effect,
            "publishes_arguments": self.publishes_arguments,
            "workspace_mutation": self.workspace_mutation,
            "execution_contract": self.execution_contract.model_dump(mode="json"),
        }

    def execution_profile_material(self) -> dict[str, Any]:
        """Return compact material for the durable ordered direct-tools component."""

        return {
            "tool_id": self.tool_id,
            "descriptor_version": self.version,
        }


def build_tool_descriptor(
    *,
    name: str,
    description: str,
    input_schema: Mapping[str, Any],
    parallel_safe: bool,
    effect: ToolEffect,
    publishes_arguments: bool,
    workspace_mutation: bool,
    execution_contract: ToolExecutionContract | None = None,
    provenance: ToolDescriptorProvenance | None = None,
) -> ToolDescriptor:
    """Build one descriptor from copied runtime-admitted registration state."""

    return ToolDescriptor(
        name=name,
        description=description,
        input_schema=input_schema,
        parallel_safe=parallel_safe,
        effect=effect,
        publishes_arguments=publishes_arguments,
        workspace_mutation=workspace_mutation,
        execution_contract=(
            ToolExecutionContract()
            if execution_contract is None
            else copy_tool_execution_contract(execution_contract)
        ),
        provenance=(ToolDescriptorProvenance() if provenance is None else provenance),
    )


def copy_tool_descriptor(descriptor: ToolDescriptor) -> ToolDescriptor:
    """Return a detached, fully revalidated descriptor."""

    if type(descriptor) is not ToolDescriptor:
        raise TypeError("descriptor must be a ToolDescriptor.")
    return ToolDescriptor.model_validate(descriptor.model_dump(mode="python", warnings=False))


def _copy_descriptor_sequence(value: object) -> tuple[ToolDescriptor, ...]:
    copied: list[ToolDescriptor] = []
    total_bytes = 2
    for item in _copy_bounded_items(value, "descriptors"):
        descriptor = (
            copy_tool_descriptor(item)
            if isinstance(item, ToolDescriptor)
            else ToolDescriptor.model_validate(item)
        )
        total_bytes += len(
            canonical_durable_json_bytes(
                descriptor.model_dump(mode="json"),
                "tool_catalogue.descriptor",
            )
        ) + (1 if copied else 0)
        if total_bytes > TOOL_CATALOGUE_MAX_BYTES:
            raise ValueError(
                f"descriptors cannot exceed {TOOL_CATALOGUE_MAX_BYTES} canonical JSON bytes."
            )
        copied.append(descriptor)
    return tuple(copied)


def _validated_catalogue_revision(
    descriptors: tuple[ToolDescriptor, ...],
    *,
    descriptor_count: int,
    supplied_revision: str,
) -> str:
    tool_ids = tuple(descriptor.tool_id for descriptor in descriptors)
    names = tuple(descriptor.name for descriptor in descriptors)
    if len(tool_ids) != len(set(tool_ids)):
        raise ValueError("Tool catalogue descriptors must have unique tool_id values.")
    if len(names) != len(set(names)):
        raise ValueError("Tool catalogue descriptors must have unique model-visible names.")
    if descriptors != tuple(sorted(descriptors, key=lambda item: item.tool_id)):
        raise ValueError("Tool catalogue descriptors must preserve canonical tool_id order.")
    if descriptor_count != len(descriptors):
        raise ValueError("descriptor_count does not match descriptors.")
    revision = _sha256_identifier(
        {
            "record_type": "cayu.tool-catalogue",
            "schema_version": TOOL_CATALOGUE_SCHEMA_VERSION,
            "descriptors": [
                {"tool_id": item.tool_id, "version": item.version} for item in descriptors
            ],
        },
        "tool_catalogue",
    )
    if supplied_revision not in {"", revision}:
        raise ValueError("revision does not match the tool catalogue.")
    return revision


class ToolCatalogSnapshot(BaseModel):
    """Immutable, deterministically ordered canonical tool catalogue."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    schema_version: Literal[1] = TOOL_CATALOGUE_SCHEMA_VERSION
    descriptors: tuple[ToolDescriptor, ...] = ()
    descriptor_count: StrictInt = Field(ge=0, le=TOOL_CATALOGUE_MAX_TOOLS)
    revision: str = Field(default="", pattern=r"^(?:|sha256:[0-9a-f]{64})$")

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != TOOL_CATALOGUE_SCHEMA_VERSION:
            raise ValueError("Tool catalogue schema_version must be the integer 1.")
        return value

    @field_validator("descriptors", mode="before")
    @classmethod
    def copy_descriptors(cls, value: object) -> tuple[ToolDescriptor, ...]:
        return _copy_descriptor_sequence(value)

    @model_validator(mode="after")
    def validate_and_bind_revision(self) -> ToolCatalogSnapshot:
        revision = _validated_catalogue_revision(
            self.descriptors,
            descriptor_count=self.descriptor_count,
            supplied_revision=self.revision,
        )
        object.__setattr__(self, "revision", revision)
        return self

    def descriptor_for_name(self, name: str) -> ToolDescriptor:
        """Return one copied descriptor by its model-visible registered name."""

        name = require_durable_clean_nonblank(name, "name")
        return copy_tool_descriptor(self._descriptor_indexes[0][name])

    def descriptor_for_id(self, tool_id: str) -> ToolDescriptor:
        """Return one copied descriptor by its canonical tool identity."""

        tool_id = require_durable_clean_nonblank(tool_id, "tool_id")
        return copy_tool_descriptor(self._descriptor_indexes[1][tool_id])

    @cached_property
    def _descriptor_indexes(
        self,
    ) -> tuple[Mapping[str, ToolDescriptor], Mapping[str, ToolDescriptor]]:
        return (
            MappingProxyType({item.name: item for item in self.descriptors}),
            MappingProxyType({item.tool_id: item for item in self.descriptors}),
        )


def build_tool_catalog_snapshot(descriptors: Iterable[ToolDescriptor]) -> ToolCatalogSnapshot:
    """Copy and canonicalize an admitted descriptor collection."""

    copied = _copy_descriptor_sequence(descriptors)
    ordered = tuple(sorted(copied, key=lambda item: item.tool_id))
    revision = _validated_catalogue_revision(
        ordered,
        descriptor_count=len(ordered),
        supplied_revision="",
    )
    # Descriptors were detached and fully revalidated above. Constructing the
    # outer record directly avoids a second deep schema copy/hash at registration.
    return ToolCatalogSnapshot.model_construct(
        schema_version=TOOL_CATALOGUE_SCHEMA_VERSION,
        descriptors=ordered,
        descriptor_count=len(ordered),
        revision=revision,
    )


def copy_tool_catalog_snapshot(snapshot: ToolCatalogSnapshot) -> ToolCatalogSnapshot:
    """Return a detached, fully revalidated catalogue snapshot."""

    if type(snapshot) is not ToolCatalogSnapshot:
        raise TypeError("snapshot must be a ToolCatalogSnapshot.")
    return ToolCatalogSnapshot.model_validate(snapshot.model_dump(mode="python", warnings=False))


def tool_catalogue_descriptors_within_ceiling(
    snapshot: ToolCatalogSnapshot,
    tool_names: Iterable[str],
) -> tuple[ToolDescriptor, ...]:
    """Return the canonical catalogue view admitted by a durable name ceiling."""

    snapshot = copy_tool_catalog_snapshot(snapshot)
    if isinstance(tool_names, str | bytes | bytearray | Mapping):
        raise TypeError("tool_names must be an iterable of registered names.")
    copied_names: list[str] = []
    for index, name in enumerate(tool_names):
        if index >= TOOL_CATALOGUE_MAX_TOOLS:
            raise ValueError(
                f"tool_names cannot contain more than {TOOL_CATALOGUE_MAX_TOOLS} items."
            )
        copied_names.append(require_durable_clean_nonblank(name, f"tool_names[{index}]"))
    names = tuple(copied_names)
    if len(names) != len(set(names)):
        raise ValueError("tool_names must contain unique values.")
    registered_names = frozenset(item.name for item in snapshot.descriptors)
    if any(name not in registered_names for name in names):
        raise ValueError("The catalogue view contains an unregistered tool name.")
    selected = frozenset(names)
    return tuple(
        copy_tool_descriptor(item) for item in snapshot.descriptors if item.name in selected
    )
