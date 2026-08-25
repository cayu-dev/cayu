"""Provider-neutral, durable discovery for large application tool catalogues."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Any, Literal, cast
from weakref import ReferenceType, ref

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
    canonical_durable_json_bytes,
    copy_durable_json_object,
    freeze_json_value,
    require_durable_clean_nonblank,
    require_durable_text,
    thaw_json_value,
)
from cayu.core.events import Event, EventType, event_with_runtime_payload_authority
from cayu.core.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.core.tools import (
    DurableToolOperationConflict,
    Tool,
    ToolContext,
    ToolEffect,
    ToolResult,
    ToolSpec,
    _runtime_tool_invocation_authority,
)
from cayu.runtime.tool_catalogue import (
    SEARCH_TOOLS_NAME,
    ToolCatalogSnapshot,
    ToolDescriptor,
    validate_canonical_tool_id,
    validate_tool_catalogue_revision,
    validate_tool_descriptor_version,
)
from cayu.runtime.tool_exposure import ToolCapabilityCeiling, copy_tool_capability_ceiling
from cayu.runtime.tool_grants import (
    TARGETED_TOOL_DIGEST_PATTERN,
    TARGETED_TOOL_REFERENCE_MAX_BYTES,
    ResolvedTargetedToolInvocation,
    TargetedToolUseRejectionReason,
    targeted_tool_view_generation_id,
    tool_reference_use_id,
)

TOOL_DISCOVERY_SCHEMA_VERSION = 1
TOOL_DISCOVERY_VIEW_OPERATION_KEY = "cayu:tool-discovery-view:v1"
TOOL_DISCOVERY_ONLY_PROFILE_ID = "cayu:tool-discovery-only:v1"
TOOL_DISCOVERY_MAX_QUERY_CHARS = 256
TOOL_DISCOVERY_MAX_RESULTS = 8
TOOL_DISCOVERY_DEFAULT_RESULTS = 5
TOOL_DISCOVERY_MAX_GRANTS = 256
TOOL_DISCOVERY_MAX_SCAN_COUNT = 10_000
TOOL_DISCOVERY_MAX_SCHEMA_BYTES = 64 * 1024
TOOL_DISCOVERY_MAX_DESCRIPTION_CHARS = 4_096
TOOL_DISCOVERY_MAX_RESULT_BYTES = 256 * 1024
TOOL_DISCOVERY_SCOPE_ID_MAX_BYTES = 2_048
TOOL_DISCOVERY_MAX_SCHEMA_SEARCH_NODES = 4_096
TOOL_DISCOVERY_MAX_SCHEMA_SEARCH_TERMS = 4_096
TOOL_DISCOVERY_MAX_WRITE_ATTEMPTS = 8
TOOL_DISCOVERY_INSPECTION_MAX_GRANTS = TOOL_DISCOVERY_MAX_GRANTS
TOOL_DISCOVERY_REFERENCE_PREFIX = "cayu_tool_v1_"
TOOL_DISCOVERY_REFERENCE_PATTERN = rf"^{TOOL_DISCOVERY_REFERENCE_PREFIX}[0-9a-f]{{64}}$"
_WORD_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


class ToolDiscoveryMode(StrEnum):
    """Application-selected provider-neutral tool discovery mode."""

    SEARCH_TOOLS = SEARCH_TOOLS_NAME


class ToolDiscoveryViewInconsistentError(RuntimeError):
    """Raised when durable view state conflicts with current session authority."""


class ToolDiscoveryViewNotEnabledError(RuntimeError):
    """Raised when a session has no configured discovery view."""


def copy_tool_discovery_mode(
    value: object,
    field_name: str = "tool_discovery_mode",
) -> ToolDiscoveryMode:
    """Return one exact discovery mode without accepting truthy substitutes."""

    if type(value) is ToolDiscoveryMode:
        return value
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a ToolDiscoveryMode or string.")
    try:
        return ToolDiscoveryMode(value)
    except ValueError:
        supported = ", ".join(mode.value for mode in ToolDiscoveryMode)
        raise ValueError(f"{field_name} must be one of: {supported}.") from None


def search_tools_spec() -> dict[str, Any]:
    """Return the detached stable provider-facing search definition."""

    return {
        "name": SEARCH_TOOLS_NAME,
        "description": (
            "Search the current session's registered tool catalogue. Returns bounded "
            "matching tool definitions and opaque references; invoke a returned tool by "
            "calling call_tool with its exact tool_ref. Already visible tools are omitted."
        ),
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": TOOL_DISCOVERY_MAX_QUERY_CHARS,
                    "description": (
                        "Words describing the capability to find. Use * to list the "
                        "highest-ranked currently hidden tools."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": TOOL_DISCOVERY_MAX_RESULTS,
                    "default": TOOL_DISCOVERY_DEFAULT_RESULTS,
                },
            },
            "required": ["query"],
        },
    }


def tool_discovery_execution_profile_material() -> dict[str, Any]:
    """Bind the portable search protocol and stable tool schema into a profile."""

    return {
        "kind": "cayu:portable-tool-discovery",
        "schema_version": TOOL_DISCOVERY_SCHEMA_VERSION,
        "mode": ToolDiscoveryMode.SEARCH_TOOLS.value,
        "view_operation_key": TOOL_DISCOVERY_VIEW_OPERATION_KEY,
        "max_results": TOOL_DISCOVERY_MAX_RESULTS,
        "max_grants": TOOL_DISCOVERY_MAX_GRANTS,
        "max_scan_count": TOOL_DISCOVERY_MAX_SCAN_COUNT,
        "max_schema_bytes": TOOL_DISCOVERY_MAX_SCHEMA_BYTES,
        "max_description_chars": TOOL_DISCOVERY_MAX_DESCRIPTION_CHARS,
        "max_result_bytes": TOOL_DISCOVERY_MAX_RESULT_BYTES,
        "max_schema_search_nodes": TOOL_DISCOVERY_MAX_SCHEMA_SEARCH_NODES,
        "max_schema_search_terms": TOOL_DISCOVERY_MAX_SCHEMA_SEARCH_TERMS,
        "execution": {
            "behavior_name": "cayu:search-tools",
            "behavior_version": "1",
            "implementation_version": "1",
            "effect": ToolEffect.IDEMPOTENT.value,
            "parallel_safe": False,
            "publishes_arguments": False,
        },
        "tool_spec_sha256": (
            "sha256:"
            + sha256(
                canonical_durable_json_bytes(search_tools_spec(), "search_tools.tool_spec")
            ).hexdigest()
        ),
    }


def _sha256_identity(value: object, field_name: str) -> str:
    return f"sha256:{sha256(canonical_durable_json_bytes(value, field_name)).hexdigest()}"


def _bounded_utf8_text(value: str, field_name: str, *, max_bytes: int) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    if len(value.encode("utf-8")) > max_bytes:
        raise ValueError(f"{field_name} cannot exceed {max_bytes} UTF-8 bytes.")
    return value


def _utc_datetime(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware.")
    return value.astimezone(UTC)


def _bounded_description(value: str) -> str:
    value = require_durable_text(value, "description")
    if len(value) <= TOOL_DISCOVERY_MAX_DESCRIPTION_CHARS:
        return value
    return f"{value[: TOOL_DISCOVERY_MAX_DESCRIPTION_CHARS - 1]}…"


def tool_discovery_generation_id(*, session_id: str, root_invocation_id: str) -> str:
    """Return the branch-local generation shared with targeted tool views."""

    return targeted_tool_view_generation_id(
        session_id=session_id,
        root_invocation_id=root_invocation_id,
    )


class ToolDiscoveryGrantRecord(BaseModel):
    """One persistent, branch-local addressability grant created by search."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    schema_version: Literal[1] = TOOL_DISCOVERY_SCHEMA_VERSION
    grant_id: str = Field(pattern=TARGETED_TOOL_DIGEST_PATTERN)
    tool_ref: str = Field(pattern=TOOL_DISCOVERY_REFERENCE_PATTERN)
    tool_id: str
    tool_name: str
    catalogue_revision: str
    descriptor_version: str
    schema_fingerprint: str = Field(pattern=TARGETED_TOOL_DIGEST_PATTERN)
    origin_query_sha256: str = Field(pattern=TARGETED_TOOL_DIGEST_PATTERN)
    origin_model_step_id: str
    created_at: datetime
    discovered_revision: StrictInt = Field(ge=1, le=TOOL_DISCOVERY_MAX_GRANTS)

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != TOOL_DISCOVERY_SCHEMA_VERSION:
            raise ValueError("Tool discovery grant schema_version must be the integer 1.")
        return value

    @field_validator("tool_ref")
    @classmethod
    def validate_tool_ref(cls, value: str) -> str:
        return _bounded_utf8_text(
            value,
            "tool_ref",
            max_bytes=TARGETED_TOOL_REFERENCE_MAX_BYTES,
        )

    @field_validator("tool_name", "origin_model_step_id")
    @classmethod
    def validate_bounded_text(cls, value: str, info) -> str:
        return _bounded_utf8_text(
            value,
            info.field_name,
            max_bytes=TOOL_DISCOVERY_SCOPE_ID_MAX_BYTES,
        )

    @field_validator("tool_id")
    @classmethod
    def validate_tool_id(cls, value: str) -> str:
        return validate_canonical_tool_id(value)

    @field_validator("catalogue_revision")
    @classmethod
    def validate_catalogue_revision(cls, value: str) -> str:
        return validate_tool_catalogue_revision(value)

    @field_validator("descriptor_version")
    @classmethod
    def validate_descriptor_version(cls, value: str) -> str:
        return validate_tool_descriptor_version(value)

    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        return _utc_datetime(value, "created_at")


class ToolDiscoveryViewState(BaseModel):
    """Versioned durable tool view for one session branch."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    schema_version: Literal[1] = TOOL_DISCOVERY_SCHEMA_VERSION
    session_id: str
    generation_id: str = Field(pattern=TARGETED_TOOL_DIGEST_PATTERN)
    agent_name: str
    catalogue_revision: str
    ceiling_fingerprint: str = Field(pattern=TARGETED_TOOL_DIGEST_PATTERN)
    revision: StrictInt = Field(ge=0, le=TOOL_DISCOVERY_MAX_GRANTS)
    grants: tuple[ToolDiscoveryGrantRecord, ...] = ()

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != TOOL_DISCOVERY_SCHEMA_VERSION:
            raise ValueError("Tool discovery view schema_version must be the integer 1.")
        return value

    @field_validator("session_id", "generation_id", "agent_name")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _bounded_utf8_text(
            value,
            info.field_name,
            max_bytes=TOOL_DISCOVERY_SCOPE_ID_MAX_BYTES,
        )

    @field_validator("catalogue_revision")
    @classmethod
    def validate_catalogue_revision(cls, value: str) -> str:
        return validate_tool_catalogue_revision(value)

    @field_validator("grants", mode="before")
    @classmethod
    def bound_grants(cls, value: object) -> object:
        if not isinstance(value, tuple | list):
            raise TypeError("Tool discovery grants must be a sequence.")
        if len(value) > TOOL_DISCOVERY_MAX_GRANTS:
            raise ValueError(
                f"Tool discovery grants cannot exceed {TOOL_DISCOVERY_MAX_GRANTS} records."
            )
        return value

    @model_validator(mode="after")
    def validate_grants(self) -> ToolDiscoveryViewState:
        if self.grants != tuple(sorted(self.grants, key=lambda item: item.tool_id)):
            raise ValueError("Tool discovery grants must preserve canonical tool-id order.")
        grant_ids = tuple(grant.grant_id for grant in self.grants)
        refs = tuple(grant.tool_ref for grant in self.grants)
        tool_ids = tuple(grant.tool_id for grant in self.grants)
        if any(len(values) != len(set(values)) for values in (grant_ids, refs, tool_ids)):
            raise ValueError("Tool discovery grants must have unique identities.")
        if (self.revision == 0) != (not self.grants):
            raise ValueError("Tool discovery revision must be zero exactly for an empty view.")
        if self.grants and max(grant.discovered_revision for grant in self.grants) != self.revision:
            raise ValueError("Tool discovery revision must identify the latest grant creation.")
        for grant in self.grants:
            expected_grant_id = tool_discovery_grant_id(
                session_id=self.session_id,
                generation_id=self.generation_id,
                agent_name=self.agent_name,
                catalogue_revision=self.catalogue_revision,
                ceiling_fingerprint=self.ceiling_fingerprint,
                descriptor=grant,
            )
            if grant.grant_id != expected_grant_id:
                raise ValueError("Tool discovery grant_id conflicts with its view authority.")
            if grant.tool_ref != tool_discovery_reference(expected_grant_id):
                raise ValueError("Tool discovery reference conflicts with its grant identity.")
            if grant.catalogue_revision != self.catalogue_revision:
                raise ValueError("Tool discovery grant has stale catalogue authority.")
            if grant.discovered_revision > self.revision:
                raise ValueError("Tool discovery grant revision exceeds its view revision.")
        return self

    def record_for_reference(self, tool_ref: str) -> ToolDiscoveryGrantRecord | None:
        """Resolve an exact opaque reference inside this view."""

        tool_ref = require_durable_clean_nonblank(tool_ref, "tool_ref")
        return next((grant for grant in self.grants if grant.tool_ref == tool_ref), None)


class ToolDiscoveryGrantInspection(BaseModel):
    """Content-minimized discovery grant state safe for control-plane reads."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    schema_version: Literal[1] = TOOL_DISCOVERY_SCHEMA_VERSION
    tool_id: str
    tool_name: str
    descriptor_version: str
    schema_fingerprint: str = Field(pattern=TARGETED_TOOL_DIGEST_PATTERN)
    created_at: datetime
    discovered_revision: StrictInt = Field(ge=1, le=TOOL_DISCOVERY_MAX_GRANTS)

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != TOOL_DISCOVERY_SCHEMA_VERSION:
            raise ValueError("Tool discovery inspection schema_version must be the integer 1.")
        return value

    @field_validator("tool_id")
    @classmethod
    def validate_tool_id(cls, value: str) -> str:
        return validate_canonical_tool_id(value)

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, value: str) -> str:
        return _bounded_utf8_text(
            value,
            "tool_name",
            max_bytes=TOOL_DISCOVERY_SCOPE_ID_MAX_BYTES,
        )

    @field_validator("descriptor_version")
    @classmethod
    def validate_descriptor_version(cls, value: str) -> str:
        return validate_tool_descriptor_version(value)

    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        return _utc_datetime(value, "created_at")


class ToolDiscoveryViewInspection(BaseModel):
    """Bounded current-view projection without schemas or callable references."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    schema_version: Literal[1] = TOOL_DISCOVERY_SCHEMA_VERSION
    session_id: str
    generation_id: str = Field(pattern=TARGETED_TOOL_DIGEST_PATTERN)
    agent_name: str
    catalogue_revision: str
    ceiling_fingerprint: str = Field(pattern=TARGETED_TOOL_DIGEST_PATTERN)
    revision: StrictInt = Field(ge=0, le=TOOL_DISCOVERY_MAX_GRANTS)
    grant_count: StrictInt = Field(ge=0, le=TOOL_DISCOVERY_MAX_GRANTS)
    grants: tuple[ToolDiscoveryGrantInspection, ...] = ()
    grants_truncated: StrictBool

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != TOOL_DISCOVERY_SCHEMA_VERSION:
            raise ValueError("Tool discovery inspection schema_version must be the integer 1.")
        return value

    @field_validator("session_id", "generation_id", "agent_name")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _bounded_utf8_text(
            value,
            info.field_name,
            max_bytes=TOOL_DISCOVERY_SCOPE_ID_MAX_BYTES,
        )

    @field_validator("catalogue_revision")
    @classmethod
    def validate_catalogue_revision(cls, value: str) -> str:
        return validate_tool_catalogue_revision(value)

    @field_validator("grants", mode="before")
    @classmethod
    def bound_grants(cls, value: object) -> object:
        if not isinstance(value, tuple | list):
            raise TypeError("Tool discovery inspection grants must be a sequence.")
        if len(value) > TOOL_DISCOVERY_INSPECTION_MAX_GRANTS:
            raise ValueError(
                "Tool discovery inspection grants cannot exceed "
                f"{TOOL_DISCOVERY_INSPECTION_MAX_GRANTS} records."
            )
        return value

    @model_validator(mode="after")
    def validate_projection(self) -> ToolDiscoveryViewInspection:
        if self.grants != tuple(sorted(self.grants, key=lambda item: item.tool_id)):
            raise ValueError("Tool discovery inspection grants must preserve canonical order.")
        if len({grant.tool_id for grant in self.grants}) != len(self.grants):
            raise ValueError("Tool discovery inspection grants must have unique tool ids.")
        if self.grant_count < len(self.grants):
            raise ValueError("grant_count cannot be smaller than the inspected grant count.")
        if self.grants_truncated != (self.grant_count > len(self.grants)):
            raise ValueError("grants_truncated must match the bounded inspection result.")
        if (self.revision == 0) != (self.grant_count == 0):
            raise ValueError("Tool discovery revision must be zero exactly for an empty view.")
        if any(grant.discovered_revision > self.revision for grant in self.grants):
            raise ValueError("An inspected grant revision cannot exceed the view revision.")
        return self


def tool_discovery_view_inspection(
    state: ToolDiscoveryViewState,
    *,
    session_id: str,
    limit: int = TOOL_DISCOVERY_INSPECTION_MAX_GRANTS,
) -> ToolDiscoveryViewInspection:
    """Project one validated view without reference- or schema-bearing authority."""

    if type(state) is not ToolDiscoveryViewState:
        raise TypeError("state must be a ToolDiscoveryViewState.")
    if type(limit) is not int or not 1 <= limit <= TOOL_DISCOVERY_INSPECTION_MAX_GRANTS:
        raise ValueError(
            f"limit must be an integer from 1 through {TOOL_DISCOVERY_INSPECTION_MAX_GRANTS}."
        )
    state = ToolDiscoveryViewState.model_validate(state.model_dump(mode="python"))
    projected_grants = tuple(
        ToolDiscoveryGrantInspection(
            tool_id=grant.tool_id,
            tool_name=grant.tool_name,
            descriptor_version=grant.descriptor_version,
            schema_fingerprint=grant.schema_fingerprint,
            created_at=grant.created_at,
            discovered_revision=grant.discovered_revision,
        )
        for grant in state.grants[:limit]
    )
    return ToolDiscoveryViewInspection(
        session_id=session_id,
        generation_id=state.generation_id,
        agent_name=state.agent_name,
        catalogue_revision=state.catalogue_revision,
        ceiling_fingerprint=state.ceiling_fingerprint,
        revision=state.revision,
        grant_count=len(state.grants),
        grants=projected_grants,
        grants_truncated=len(projected_grants) < len(state.grants),
    )


def tool_discovery_grant_id(
    *,
    session_id: str,
    generation_id: str,
    agent_name: str,
    catalogue_revision: str,
    ceiling_fingerprint: str,
    descriptor: ToolDescriptor | ToolDiscoveryGrantRecord,
) -> str:
    """Return one deterministic persistent grant identity."""

    if isinstance(descriptor, ToolDescriptor):
        tool_name = descriptor.name
        descriptor_version = descriptor.version
    else:
        tool_name = descriptor.tool_name
        descriptor_version = descriptor.descriptor_version
    return _sha256_identity(
        {
            "record_type": "cayu.tool-discovery-grant",
            "schema_version": TOOL_DISCOVERY_SCHEMA_VERSION,
            "session_id": require_durable_clean_nonblank(session_id, "session_id"),
            "generation_id": require_durable_clean_nonblank(generation_id, "generation_id"),
            "agent_name": require_durable_clean_nonblank(agent_name, "agent_name"),
            "catalogue_revision": validate_tool_catalogue_revision(catalogue_revision),
            "ceiling_fingerprint": require_durable_clean_nonblank(
                ceiling_fingerprint,
                "ceiling_fingerprint",
            ),
            "tool_id": descriptor.tool_id,
            "tool_name": tool_name,
            "descriptor_version": descriptor_version,
            "schema_fingerprint": descriptor.schema_fingerprint,
        },
        "tool_discovery_grant",
    )


def tool_discovery_reference(grant_id: str) -> str:
    """Project a grant identity as an opaque, non-authorizing model reference."""

    grant_id = require_durable_clean_nonblank(grant_id, "grant_id")
    if not re.fullmatch(TARGETED_TOOL_DIGEST_PATTERN, grant_id):
        raise ValueError("grant_id must be a SHA-256 identity.")
    digest = sha256(
        canonical_durable_json_bytes(
            {
                "record_type": "cayu.tool-discovery-reference",
                "schema_version": TOOL_DISCOVERY_SCHEMA_VERSION,
                "grant_id": grant_id,
            },
            "tool_discovery_reference",
        )
    ).hexdigest()
    return f"{TOOL_DISCOVERY_REFERENCE_PREFIX}{digest}"


def tool_discovery_reference_rejection_reason(
    tool_ref: str,
) -> TargetedToolUseRejectionReason:
    """Classify one unresolved discovery-shaped reference without resolving it."""

    tool_ref = require_durable_clean_nonblank(tool_ref, "tool_ref")
    if (
        tool_ref.startswith(TOOL_DISCOVERY_REFERENCE_PREFIX)
        and re.fullmatch(
            TOOL_DISCOVERY_REFERENCE_PATTERN,
            tool_ref,
        )
        is None
    ):
        return TargetedToolUseRejectionReason.MALFORMED
    return TargetedToolUseRejectionReason.UNKNOWN


def empty_tool_discovery_view(
    *,
    session_id: str,
    generation_id: str,
    agent_name: str,
    catalogue: ToolCatalogSnapshot,
    ceiling: ToolCapabilityCeiling,
) -> ToolDiscoveryViewState:
    """Build the current empty branch-local view authority."""

    if type(catalogue) is not ToolCatalogSnapshot:
        raise TypeError("catalogue must be a ToolCatalogSnapshot.")
    ceiling = copy_tool_capability_ceiling(ceiling)
    return ToolDiscoveryViewState(
        session_id=session_id,
        generation_id=generation_id,
        agent_name=agent_name,
        catalogue_revision=catalogue.revision,
        ceiling_fingerprint=f"sha256:{ceiling.fingerprint}",
        revision=0,
    )


def initial_tool_discovery_operation_records(
    *,
    session_id: str,
    root_invocation_id: str,
    agent_name: str,
    catalogue: ToolCatalogSnapshot,
    ceiling: ToolCapabilityCeiling,
) -> dict[str, dict[str, Any]]:
    """Return the typed empty view committed with one new session branch."""

    view = empty_tool_discovery_view(
        session_id=session_id,
        generation_id=tool_discovery_generation_id(
            session_id=session_id,
            root_invocation_id=root_invocation_id,
        ),
        agent_name=agent_name,
        catalogue=catalogue,
        ceiling=ceiling,
    )
    return {TOOL_DISCOVERY_VIEW_OPERATION_KEY: view.model_dump(mode="json")}


def current_tool_discovery_view(
    raw: dict[str, Any] | None,
    *,
    session_id: str,
    generation_id: str,
    agent_name: str,
    catalogue: ToolCatalogSnapshot,
    ceiling: ToolCapabilityCeiling,
) -> ToolDiscoveryViewState:
    """Load exact current authority and reject missing, stale, or foreign state."""

    empty = empty_tool_discovery_view(
        session_id=session_id,
        generation_id=generation_id,
        agent_name=agent_name,
        catalogue=catalogue,
        ceiling=ceiling,
    )
    if raw is None:
        raise ValueError("Tool discovery view is not initialized.")
    state = ToolDiscoveryViewState.model_validate(raw)
    if (
        state.session_id != empty.session_id
        or state.generation_id != empty.generation_id
        or state.agent_name != empty.agent_name
        or state.catalogue_revision != empty.catalogue_revision
        or state.ceiling_fingerprint != empty.ceiling_fingerprint
    ):
        raise ValueError("Tool discovery view conflicts with current session authority.")
    return state


def tool_discovery_record_matches_descriptor(
    record: ToolDiscoveryGrantRecord,
    descriptor: ToolDescriptor,
) -> bool:
    """Return whether a current descriptor exactly retains one discovery grant."""

    if type(record) is not ToolDiscoveryGrantRecord or type(descriptor) is not ToolDescriptor:
        return False
    return (
        record.tool_id == descriptor.tool_id
        and record.tool_name == descriptor.name
        and record.descriptor_version == descriptor.version
        and record.schema_fingerprint == descriptor.schema_fingerprint
    )


def resolved_discovered_tool_invocation(
    *,
    record: ToolDiscoveryGrantRecord,
    session_id: str,
    interaction_id: str,
    model_step_id: str,
    outer_tool_call_id: str,
    arguments_sha256: str,
    invocation_id: str,
) -> ResolvedTargetedToolInvocation:
    """Project persistent discovery authority into existing dual-identity evidence."""

    if type(record) is not ToolDiscoveryGrantRecord:
        raise TypeError("record must be a ToolDiscoveryGrantRecord.")
    use_id = tool_reference_use_id(
        grant_id=record.grant_id,
        session_id=session_id,
        interaction_id=interaction_id,
        model_step_id=model_step_id,
        outer_tool_call_id=outer_tool_call_id,
        arguments_sha256=arguments_sha256,
        invocation_id=invocation_id,
    )
    return ResolvedTargetedToolInvocation(
        dispatch_kind="gateway",
        model_tool_name="call_tool",
        tool_ref=record.tool_ref,
        grant_id=record.grant_id,
        use_id=use_id,
        session_id=session_id,
        interaction_id=interaction_id,
        tool_id=record.tool_id,
        effective_tool_name=record.tool_name,
        catalogue_revision=record.catalogue_revision,
        descriptor_version=record.descriptor_version,
        schema_fingerprint=record.schema_fingerprint,
        model_step_id=model_step_id,
        outer_tool_call_id=outer_tool_call_id,
        arguments_sha256=arguments_sha256,
        invocation_id=invocation_id,
    )


def discovered_tool_rejection_event(
    *,
    session_id: str,
    interaction_id: str,
    agent_name: str,
    environment_name: str | None,
    model_step_id: str,
    outer_tool_call_id: str,
    arguments_sha256: str,
    reason: TargetedToolUseRejectionReason,
    timestamp: datetime,
) -> Event:
    """Build privacy-safe rejection evidence for one discovery reference."""

    if type(reason) is not TargetedToolUseRejectionReason:
        raise TypeError("reason must be a TargetedToolUseRejectionReason.")
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware.")
    payload = {
        "schema_version": TOOL_DISCOVERY_SCHEMA_VERSION,
        "authority_kind": "tool_discovery",
        "outcome": "rejected",
        "rejection_reason": reason.value,
        "model_step_id": require_durable_clean_nonblank(model_step_id, "model_step_id"),
        "outer_tool_call_id": require_durable_clean_nonblank(
            outer_tool_call_id,
            "outer_tool_call_id",
        ),
        "arguments_sha256": require_durable_clean_nonblank(
            arguments_sha256,
            "arguments_sha256",
        ),
    }
    rejection_id = _sha256_identity(
        {
            "record_type": "cayu.tool-discovery-rejection",
            **payload,
            "session_id": session_id,
            "interaction_id": interaction_id,
        },
        "tool_discovery_rejection",
    )
    payload["rejection_id"] = rejection_id
    return event_with_runtime_payload_authority(
        Event(
            id=f"{rejection_id}:tool-discovery-rejected",
            type=EventType.TARGETED_TOOL_REFERENCE_REJECTED,
            session_id=session_id,
            interaction_id=interaction_id,
            agent_name=agent_name,
            environment_name=environment_name,
            timestamp=timestamp.astimezone(UTC),
            payload=payload,
        ),
        "arguments_sha256",
        "rejection_id",
    )


class ToolDiscoverySearchMatch(BaseModel):
    """Bounded model-facing descriptor returned by ``search_tools``."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    tool_ref: str = Field(pattern=TOOL_DISCOVERY_REFERENCE_PATTERN)
    tool_id: str
    name: str
    description: str
    input_schema: Mapping[str, Any]
    descriptor_version: str
    schema_fingerprint: str = Field(pattern=TARGETED_TOOL_DIGEST_PATTERN)
    readiness: Literal["registered"] = "registered"

    @field_validator("tool_ref")
    @classmethod
    def validate_tool_ref(cls, value: str) -> str:
        return _bounded_utf8_text(
            value,
            "tool_ref",
            max_bytes=TARGETED_TOOL_REFERENCE_MAX_BYTES,
        )

    @field_validator("tool_id")
    @classmethod
    def validate_tool_id(cls, value: str) -> str:
        return validate_canonical_tool_id(value)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _bounded_utf8_text(
            value,
            "name",
            max_bytes=TOOL_DISCOVERY_SCOPE_ID_MAX_BYTES,
        )

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        value = require_durable_text(value, "description")
        if len(value) > TOOL_DISCOVERY_MAX_DESCRIPTION_CHARS:
            raise ValueError(
                f"description cannot exceed {TOOL_DISCOVERY_MAX_DESCRIPTION_CHARS} characters."
            )
        return value

    @field_validator("input_schema", mode="before")
    @classmethod
    def copy_schema(cls, value: object) -> dict[str, Any]:
        copied = copy_durable_json_object(value, "input_schema")
        if (
            len(canonical_durable_json_bytes(copied, "input_schema"))
            > TOOL_DISCOVERY_MAX_SCHEMA_BYTES
        ):
            raise ValueError(f"input_schema cannot exceed {TOOL_DISCOVERY_MAX_SCHEMA_BYTES} bytes.")
        return copied

    @field_validator("input_schema")
    @classmethod
    def freeze_schema(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return freeze_json_value(dict(value))

    @field_serializer("input_schema")
    def serialize_schema(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return dict(thaw_json_value(value))

    @field_validator("descriptor_version")
    @classmethod
    def validate_descriptor_version(cls, value: str) -> str:
        return validate_tool_descriptor_version(value)

    @model_validator(mode="after")
    def validate_schema_fingerprint(self) -> ToolDiscoverySearchMatch:
        expected = _sha256_identity(
            thaw_json_value(self.input_schema),
            "tool_discovery.input_schema",
        )
        if self.schema_fingerprint != expected:
            raise ValueError("schema_fingerprint conflicts with input_schema.")
        return self


class ToolDiscoverySearchResult(BaseModel):
    """Deterministic result and durable view revision from one catalogue search."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    schema_version: Literal[1] = TOOL_DISCOVERY_SCHEMA_VERSION
    query: str
    matches: tuple[ToolDiscoverySearchMatch, ...]
    view_revision: StrictInt = Field(ge=0)
    truncated: StrictBool = False

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != TOOL_DISCOVERY_SCHEMA_VERSION:
            raise ValueError("Tool discovery result schema_version must be the integer 1.")
        return value

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        return _normalize_search_query(value)

    @field_validator("matches", mode="before")
    @classmethod
    def bound_matches(cls, value: object) -> object:
        if not isinstance(value, tuple | list):
            raise TypeError("Tool discovery matches must be a sequence.")
        if len(value) > TOOL_DISCOVERY_MAX_RESULTS:
            raise ValueError(
                f"Tool discovery matches cannot exceed {TOOL_DISCOVERY_MAX_RESULTS} records."
            )
        return value

    @model_validator(mode="after")
    def bound_serialized_result(self) -> ToolDiscoverySearchResult:
        if (
            len(
                canonical_durable_json_bytes(
                    self.model_dump(mode="json"),
                    "search_tools.result",
                )
            )
            > TOOL_DISCOVERY_MAX_RESULT_BYTES
        ):
            raise ValueError(
                f"Tool discovery result cannot exceed {TOOL_DISCOVERY_MAX_RESULT_BYTES} bytes."
            )
        return self


def minimized_tool_discovery_result(result: ToolResult) -> ToolResult:
    """Return bounded public evidence without schemas, descriptions, or references."""

    if type(result) is not ToolResult:
        raise TypeError("result must be a ToolResult.")
    try:
        discovery_result = ToolDiscoverySearchResult.model_validate(
            result.model_dump(mode="python")["structured"]
        )
    except (TypeError, ValueError):
        return ToolResult(
            content="Tool discovery result withheld from public event history.",
            is_error=result.is_error,
        )
    return ToolResult(
        content=(
            f"Tool discovery returned {len(discovery_result.matches)} model-visible result(s)."
        ),
        structured={
            "schema_version": discovery_result.schema_version,
            "match_count": len(discovery_result.matches),
            "view_revision": discovery_result.view_revision,
            "truncated": discovery_result.truncated,
        },
        is_error=result.is_error,
    )


def _normalized_words(value: str) -> tuple[str, ...]:
    return tuple(_WORD_PATTERN.findall(value.casefold()))


def _normalize_search_query(value: object) -> str:
    if type(value) is not str:
        raise TypeError("query must be a string.")
    value = require_durable_clean_nonblank(value, "query")
    if len(value) > TOOL_DISCOVERY_MAX_QUERY_CHARS:
        raise ValueError(f"query cannot exceed {TOOL_DISCOVERY_MAX_QUERY_CHARS} characters.")
    if value == "*":
        return value
    normalized = " ".join(value.casefold().split())
    if len(normalized) > TOOL_DISCOVERY_MAX_QUERY_CHARS:
        raise ValueError(
            f"normalized query cannot exceed {TOOL_DISCOVERY_MAX_QUERY_CHARS} characters."
        )
    if not _normalized_words(normalized):
        raise ValueError("query must contain at least one letter or number.")
    return normalized


def _schema_property_words(schema: Mapping[str, Any]) -> frozenset[str]:
    """Return a bounded token index over JSON Schema property names."""

    words: set[str] = set()
    pending: list[object] = [schema]
    visited = 0
    examined_terms = 0
    while (
        pending
        and visited < TOOL_DISCOVERY_MAX_SCHEMA_SEARCH_NODES
        and examined_terms < TOOL_DISCOVERY_MAX_SCHEMA_SEARCH_TERMS
    ):
        current = pending.pop()
        visited += 1
        if isinstance(current, Mapping):
            current_mapping = cast("Mapping[str, object]", current)
            properties = current_mapping.get("properties")
            if isinstance(properties, Mapping):
                properties_mapping = cast("Mapping[str, object]", properties)
                for name in sorted(properties_mapping):
                    if examined_terms >= TOOL_DISCOVERY_MAX_SCHEMA_SEARCH_TERMS:
                        break
                    examined_terms += 1
                    if type(name) is str:
                        words.update(_normalized_words(name))
            pending.extend(current_mapping[key] for key in sorted(current_mapping, reverse=True))
        elif isinstance(current, Sequence) and not isinstance(current, str | bytes | bytearray):
            pending.extend(reversed(current))
    return frozenset(words)


def _descriptor_score(
    descriptor: ToolDescriptor,
    *,
    normalized_query: str,
    query_text: str,
    query_words: frozenset[str],
) -> int:
    if normalized_query == "*":
        return 1
    name_text = " ".join(_normalized_words(descriptor.name))
    name_words = frozenset(name_text.split())
    description_words = frozenset(_normalized_words(descriptor.description))
    property_words = _schema_property_words(descriptor.input_schema)
    score = 0
    if normalized_query == descriptor.tool_id.casefold():
        score += 2_000
    if query_text == name_text:
        score += 2_000
    name_overlap = len(query_words & name_words)
    description_overlap = len(query_words & description_words)
    property_overlap = len(query_words & property_words)
    score += name_overlap * 120 + description_overlap * 20 + property_overlap * 40
    if query_words <= name_words:
        score += 600
    elif query_words <= description_words:
        score += 80
    elif query_words <= property_words:
        score += 100
    return score


def search_tool_descriptors(
    query: str,
    *,
    catalogue: ToolCatalogSnapshot,
    ceiling: ToolCapabilityCeiling,
    excluded_names: Iterable[str] = (),
) -> tuple[ToolDescriptor, ...]:
    """Rank current hidden descriptors with deterministic, model-free search."""

    query = _normalize_search_query(query)
    if type(catalogue) is not ToolCatalogSnapshot:
        raise TypeError("catalogue must be a ToolCatalogSnapshot.")
    ceiling = copy_tool_capability_ceiling(ceiling)
    excluded = frozenset(
        require_durable_clean_nonblank(name, "excluded_names") for name in excluded_names
    )
    ceiling_names = frozenset(ceiling.tool_names)
    descriptors = catalogue.descriptors
    if len(descriptors) > TOOL_DISCOVERY_MAX_SCAN_COUNT:
        raise ValueError(
            f"Tool discovery cannot scan more than {TOOL_DISCOVERY_MAX_SCAN_COUNT} descriptors."
        )
    eligible = tuple(
        descriptor
        for descriptor in descriptors
        if descriptor.name in ceiling_names and descriptor.name not in excluded
    )
    if query == "*":
        return eligible
    query_text = " ".join(_normalized_words(query))
    query_words = frozenset(query_text.split())
    ranked = [
        (
            _descriptor_score(
                descriptor,
                normalized_query=query,
                query_text=query_text,
                query_words=query_words,
            ),
            descriptor,
        )
        for descriptor in eligible
    ]
    return tuple(
        descriptor
        for score, descriptor in sorted(
            (item for item in ranked if item[0] > 0),
            key=lambda item: (-item[0], item[1].tool_id),
        )
    )


@dataclass(frozen=True, slots=True)
class _RuntimeToolDiscoveryAuthority:
    session_id: str
    generation_id: str
    agent_name: str
    catalogue: ToolCatalogSnapshot
    ceiling: ToolCapabilityCeiling
    directly_exposed_names: frozenset[str]
    model_step_id: str
    created_at: datetime


_RUNTIME_TOOL_DISCOVERY_AUTHORITIES: dict[
    int,
    tuple[ReferenceType[Any], _RuntimeToolDiscoveryAuthority],
] = {}


def _bind_runtime_tool_discovery_authority(
    context: ToolContext,
    *,
    generation_id: str,
    catalogue: ToolCatalogSnapshot,
    ceiling: ToolCapabilityCeiling,
    directly_exposed_names: Iterable[str],
    model_step_id: str,
    created_at: datetime,
) -> None:
    """Bind private search authority to one Cayu-created tool context."""

    if type(context) is not ToolContext:
        raise TypeError("Runtime tool discovery authority requires a ToolContext.")
    if context.agent_name is None:
        raise RuntimeError("Runtime tool discovery authority requires an agent identity.")
    context_id = id(context)
    existing = _RUNTIME_TOOL_DISCOVERY_AUTHORITIES.get(context_id)
    if existing is not None and existing[0]() is context:
        raise RuntimeError("Runtime tool discovery authority is already bound.")
    authority = _RuntimeToolDiscoveryAuthority(
        session_id=context.session_id,
        generation_id=require_durable_clean_nonblank(generation_id, "generation_id"),
        agent_name=context.agent_name,
        catalogue=catalogue,
        ceiling=copy_tool_capability_ceiling(ceiling),
        directly_exposed_names=frozenset(directly_exposed_names),
        model_step_id=_bounded_utf8_text(
            model_step_id,
            "model_step_id",
            max_bytes=TOOL_DISCOVERY_SCOPE_ID_MAX_BYTES,
        ),
        created_at=_utc_datetime(created_at, "created_at"),
    )

    def discard(expired: ReferenceType[Any]) -> None:
        registered = _RUNTIME_TOOL_DISCOVERY_AUTHORITIES.get(context_id)
        if registered is not None and registered[0] is expired:
            _RUNTIME_TOOL_DISCOVERY_AUTHORITIES.pop(context_id, None)

    context_reference = ref(context, discard)
    _RUNTIME_TOOL_DISCOVERY_AUTHORITIES[context_id] = (context_reference, authority)


def _runtime_tool_discovery_authority(
    context: ToolContext,
) -> _RuntimeToolDiscoveryAuthority | None:
    registered = _RUNTIME_TOOL_DISCOVERY_AUTHORITIES.get(id(context))
    if registered is None or registered[0]() is not context:
        return None
    return registered[1]


def _tool_discovery_search_transition(
    state: ToolDiscoveryViewState,
    *,
    ranked: tuple[ToolDescriptor, ...],
    query: str,
    limit: int,
    model_step_id: str,
    created_at: datetime,
) -> tuple[ToolDiscoveryViewState, ToolDiscoverySearchResult]:
    """Build one deterministic view transition and its exact private result."""

    existing_by_tool_id = {grant.tool_id: grant for grant in state.grants}
    next_revision = state.revision + 1
    new_grants: list[ToolDiscoveryGrantRecord] = []
    matches: list[ToolDiscoverySearchMatch] = []
    truncated = len(ranked) > limit
    for descriptor in ranked:
        if len(matches) >= limit:
            break
        descriptor_schema = descriptor.input_schema_copy()
        if (
            len(canonical_durable_json_bytes(descriptor_schema, "search_tools.input_schema"))
            > TOOL_DISCOVERY_MAX_SCHEMA_BYTES
        ):
            truncated = True
            continue
        grant = existing_by_tool_id.get(descriptor.tool_id)
        if grant is None:
            if len(state.grants) + len(new_grants) >= TOOL_DISCOVERY_MAX_GRANTS:
                truncated = True
                continue
            grant_id = tool_discovery_grant_id(
                session_id=state.session_id,
                generation_id=state.generation_id,
                agent_name=state.agent_name,
                catalogue_revision=state.catalogue_revision,
                ceiling_fingerprint=state.ceiling_fingerprint,
                descriptor=descriptor,
            )
            grant = ToolDiscoveryGrantRecord(
                grant_id=grant_id,
                tool_ref=tool_discovery_reference(grant_id),
                tool_id=descriptor.tool_id,
                tool_name=descriptor.name,
                catalogue_revision=state.catalogue_revision,
                descriptor_version=descriptor.version,
                schema_fingerprint=descriptor.schema_fingerprint,
                origin_query_sha256=_sha256_identity(
                    {"query": query},
                    "tool_discovery_query",
                ),
                origin_model_step_id=model_step_id,
                created_at=created_at,
                discovered_revision=next_revision,
            )
            new_grants.append(grant)
        candidate = ToolDiscoverySearchMatch(
            tool_ref=grant.tool_ref,
            tool_id=descriptor.tool_id,
            name=descriptor.name,
            description=_bounded_description(descriptor.description),
            input_schema=descriptor_schema,
            descriptor_version=descriptor.version,
            schema_fingerprint=descriptor.schema_fingerprint,
        )
        candidate_matches = (*matches, candidate)
        candidate_payload = {
            "schema_version": TOOL_DISCOVERY_SCHEMA_VERSION,
            "query": query,
            "matches": [match.model_dump(mode="json") for match in candidate_matches],
            # Reserve the largest possible V1 revision representation so adding a
            # later match cannot make an earlier size decision optimistic.
            "view_revision": TOOL_DISCOVERY_MAX_GRANTS,
            "truncated": truncated,
        }
        if (
            len(canonical_durable_json_bytes(candidate_payload, "search_tools.result"))
            > TOOL_DISCOVERY_MAX_RESULT_BYTES
        ):
            if grant in new_grants:
                new_grants.remove(grant)
            truncated = True
            continue
        matches.append(candidate)

    desired_state = (
        ToolDiscoveryViewState.model_validate(
            {
                **state.model_dump(mode="python"),
                "revision": next_revision,
                "grants": tuple(
                    sorted((*state.grants, *new_grants), key=lambda item: item.tool_id)
                ),
            }
        )
        if new_grants
        else state
    )
    return desired_state, ToolDiscoverySearchResult(
        query=query,
        matches=tuple(matches),
        view_revision=desired_state.revision,
        truncated=truncated,
    )


class SearchToolsTool(Tool):
    """Cayu-owned stable discovery tool executed through the ordinary tool path."""

    spec = ToolSpec(
        name=SEARCH_TOOLS_NAME,
        description=search_tools_spec()["description"],
        input_schema=search_tools_spec()["input_schema"],
        parallel_safe=False,
        effect=ToolEffect.IDEMPOTENT,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="cayu:search-tools",
            behavior_version="1",
            implementation_version="1",
        ),
    )

    @property
    def _publish_arguments(self) -> bool:
        return False

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        discovery = _runtime_tool_discovery_authority(ctx)
        durable = _runtime_tool_invocation_authority(ctx)
        if discovery is None or durable is None:
            return ToolResult(
                content="Tool discovery runtime authority is unavailable.",
                is_error=True,
            )
        try:
            query = _normalize_search_query(args.get("query"))
            raw_limit = args.get("limit", TOOL_DISCOVERY_DEFAULT_RESULTS)
            if type(raw_limit) is not int or not 1 <= raw_limit <= TOOL_DISCOVERY_MAX_RESULTS:
                raise ValueError(
                    f"limit must be an integer from 1 to {TOOL_DISCOVERY_MAX_RESULTS}."
                )
        except (TypeError, ValueError) as exc:
            return ToolResult(content=str(exc), is_error=True)

        ranked = search_tool_descriptors(
            query,
            catalogue=discovery.catalogue,
            ceiling=discovery.ceiling,
            excluded_names=discovery.directly_exposed_names,
        )
        result: ToolDiscoverySearchResult | None = None
        for attempt in range(TOOL_DISCOVERY_MAX_WRITE_ATTEMPTS):
            raw = await durable.load_durable_operation(TOOL_DISCOVERY_VIEW_OPERATION_KEY)
            state = current_tool_discovery_view(
                raw,
                session_id=discovery.session_id,
                generation_id=discovery.generation_id,
                agent_name=discovery.agent_name,
                catalogue=discovery.catalogue,
                ceiling=discovery.ceiling,
            )
            desired_state, candidate_result = _tool_discovery_search_transition(
                state,
                ranked=ranked,
                query=query,
                limit=raw_limit,
                model_step_id=discovery.model_step_id,
                created_at=discovery.created_at,
            )
            desired_raw = desired_state.model_dump(mode="json")
            if raw == desired_raw:
                result = candidate_result
                break
            try:
                await durable.compare_and_set_durable_operation(
                    TOOL_DISCOVERY_VIEW_OPERATION_KEY,
                    raw,
                    desired_raw,
                    {},
                )
            except DurableToolOperationConflict:
                if attempt + 1 == TOOL_DISCOVERY_MAX_WRITE_ATTEMPTS:
                    raise
                continue
            result = candidate_result
            break
        if result is None:  # pragma: no cover - loop always returns or raises
            raise RuntimeError("Tool discovery search did not reach a durable outcome.")
        payload = result.model_dump(mode="json")
        return ToolResult(
            content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            structured=payload,
        )


__all__ = [
    "TOOL_DISCOVERY_DEFAULT_RESULTS",
    "TOOL_DISCOVERY_INSPECTION_MAX_GRANTS",
    "TOOL_DISCOVERY_MAX_DESCRIPTION_CHARS",
    "TOOL_DISCOVERY_MAX_GRANTS",
    "TOOL_DISCOVERY_MAX_QUERY_CHARS",
    "TOOL_DISCOVERY_MAX_RESULTS",
    "TOOL_DISCOVERY_MAX_RESULT_BYTES",
    "TOOL_DISCOVERY_MAX_SCAN_COUNT",
    "TOOL_DISCOVERY_MAX_SCHEMA_BYTES",
    "TOOL_DISCOVERY_MAX_WRITE_ATTEMPTS",
    "TOOL_DISCOVERY_ONLY_PROFILE_ID",
    "TOOL_DISCOVERY_REFERENCE_PATTERN",
    "TOOL_DISCOVERY_REFERENCE_PREFIX",
    "TOOL_DISCOVERY_SCHEMA_VERSION",
    "TOOL_DISCOVERY_VIEW_OPERATION_KEY",
    "SearchToolsTool",
    "ToolDiscoveryGrantInspection",
    "ToolDiscoveryGrantRecord",
    "ToolDiscoveryMode",
    "ToolDiscoverySearchMatch",
    "ToolDiscoverySearchResult",
    "ToolDiscoveryViewInconsistentError",
    "ToolDiscoveryViewInspection",
    "ToolDiscoveryViewNotEnabledError",
    "ToolDiscoveryViewState",
    "copy_tool_discovery_mode",
    "current_tool_discovery_view",
    "discovered_tool_rejection_event",
    "empty_tool_discovery_view",
    "initial_tool_discovery_operation_records",
    "minimized_tool_discovery_result",
    "resolved_discovered_tool_invocation",
    "search_tool_descriptors",
    "search_tools_spec",
    "tool_discovery_execution_profile_material",
    "tool_discovery_generation_id",
    "tool_discovery_grant_id",
    "tool_discovery_record_matches_descriptor",
    "tool_discovery_reference",
    "tool_discovery_reference_rejection_reason",
    "tool_discovery_view_inspection",
]
