"""Shared hosted tool-discovery projection, validation and grant preparation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from hashlib import sha256
from typing import Any, cast

from cayu._validation import (
    canonical_durable_json_bytes,
    copy_json_value,
    require_durable_clean_nonblank,
)
from cayu.context.structured_output import require_secret_free_json_schema_keys
from cayu.providers.base import (
    OPENAI_HOSTED_TOOL_SEARCH_PROTOCOL,
    TOOL_DISCOVERY_PROJECTION_MAX_TOOLS,
    ModelStreamEvent,
    ToolDiscoveryProjectionRequest,
)
from cayu.runtime import _runtime_records as runtime_records
from cayu.sessions.base import (
    RuntimePublicationOperationRecordMutation,
    SessionStore,
    runtime_publication_operation_record_value_digest,
)
from cayu.sessions.records import Session
from cayu.tools.catalogue import ToolDescriptor
from cayu.tools.discovery import (
    TOOL_DISCOVERY_VIEW_OPERATION_KEY,
    _tool_discovery_definition_for_descriptor,
    current_tool_discovery_view,
    hosted_tool_discovery_transition,
    tool_discovery_generation_id,
)
from cayu.tools.exposure import tool_capability_ceiling_from_session_metadata
from cayu.vaults.redaction import SecretRedactor


def _hosted_tool_discovery_projection(
    *,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    excluded_tool_names: Iterable[str],
    redactor: SecretRedactor,
) -> ToolDiscoveryProjectionRequest:
    """Build the exact bounded, secret-free hosted candidate projection."""

    excluded = frozenset(excluded_tool_names)
    if any(type(name) is not str for name in excluded):
        raise TypeError("Excluded tool names must be strings.")
    ceiling_names = frozenset(
        tool_capability_ceiling_from_session_metadata(session.metadata).tool_names
    )
    descriptors = tuple(
        descriptor
        for descriptor in registered_agent.tool_catalogue.descriptors
        if descriptor.name in ceiling_names and descriptor.name not in excluded
    )
    if len(descriptors) > TOOL_DISCOVERY_PROJECTION_MAX_TOOLS:
        raise ValueError(
            "OpenAI hosted Tool Search candidate count exceeds the bounded "
            f"maximum of {TOOL_DISCOVERY_PROJECTION_MAX_TOOLS}; reduce the session "
            "tool ceiling or use portable search_tools."
        )
    candidates = tuple(
        sorted(
            (_tool_discovery_definition_for_descriptor(descriptor) for descriptor in descriptors),
            key=lambda tool: cast("str", tool["name"]),
        )
    )
    redacted_candidates = tuple(
        _redacted_provider_tool_definitions(
            candidates,
            redactor=redactor,
            field_name="tool_discovery_candidates",
        )
    )
    return ToolDiscoveryProjectionRequest(
        protocol=OPENAI_HOSTED_TOOL_SEARCH_PROTOCOL,
        candidate_tools=redacted_candidates,
        generation_id=tool_discovery_generation_id(
            session_id=session.id,
            root_invocation_id=session.invocation.root_invocation_id,
        ),
    )


def _hosted_tool_discovery_projection_digest(
    projection: ToolDiscoveryProjectionRequest,
) -> str:
    if projection.protocol != OPENAI_HOSTED_TOOL_SEARCH_PROTOCOL:
        raise ValueError("Hosted Tool Search authority requires the hosted protocol.")
    return sha256(
        canonical_durable_json_bytes(
            projection.model_dump(mode="json"),
            "hosted_tool_discovery_projection",
        )
    ).hexdigest()


def _hosted_tool_name_sha256(name: str) -> str:
    return sha256(
        require_durable_clean_nonblank(name, "hosted tool name").encode("utf-8")
    ).hexdigest()


async def _hosted_tool_discovery_publication_authority(
    *,
    session_store: SessionStore,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    projection: ToolDiscoveryProjectionRequest | None,
    stream_event: ModelStreamEvent,
    tool_calls: list[runtime_records.ToolCallRequest],
    model_step_id: str,
    created_at: datetime,
) -> tuple[tuple[RuntimePublicationOperationRecordMutation, ...], dict[str, str]]:
    """Validate a hosted selection and prepare its atomic durable grant update."""

    result = stream_event.tool_discovery_result
    if projection is None or projection.protocol != OPENAI_HOSTED_TOOL_SEARCH_PROTOCOL:
        if result is not None:
            raise ValueError("Provider returned hosted Tool Search evidence unexpectedly.")
        return (), {}

    candidates_by_name = {cast("str", tool["name"]): tool for tool in projection.candidate_tools}
    candidate_call_names = {call.name for call in tool_calls if call.name in candidates_by_name}
    replay_loaded_names = frozenset(projection.loaded_tool_names)
    if result is None:
        if candidate_call_names - replay_loaded_names:
            raise ValueError(
                "Provider called a deferred function without hosted Tool Search evidence."
            )
        return (), {}

    loaded_names = result.loaded_tool_names
    loaded_name_set = frozenset(loaded_names)
    if candidate_call_names - loaded_name_set - replay_loaded_names:
        raise ValueError("Provider called a deferred function outside the loaded subset.")
    selected_descriptors: list[ToolDescriptor] = []
    ceiling_names = frozenset(
        tool_capability_ceiling_from_session_metadata(session.metadata).tool_names
    )
    for loaded in result.loaded_tools:
        name = cast("str", loaded["name"])
        candidate = candidates_by_name.get(name)
        if candidate is None:
            raise ValueError("Provider loaded a function outside the requested catalogue.")
        if loaded != candidate:
            raise ValueError("Provider altered a hosted Tool Search function definition.")
        descriptor = registered_agent.tool_catalogue.descriptor_for_name(name)
        if descriptor.name not in ceiling_names:
            raise ValueError("Provider loaded a function outside the session tool ceiling.")
        selected_descriptors.append(descriptor)

    if not selected_descriptors:
        return (), {}

    raw_view = await session_store.load_session_operation(
        session.id,
        TOOL_DISCOVERY_VIEW_OPERATION_KEY,
    )
    view = current_tool_discovery_view(
        raw_view,
        session_id=session.id,
        generation_id=tool_discovery_generation_id(
            session_id=session.id,
            root_invocation_id=session.invocation.root_invocation_id,
        ),
        agent_name=registered_agent.spec.name,
        catalogue=registered_agent.tool_catalogue,
        ceiling=tool_capability_ceiling_from_session_metadata(session.metadata),
    )
    desired_view, grant_ids = hosted_tool_discovery_transition(
        view,
        descriptors=selected_descriptors,
        model_step_id=model_step_id,
        created_at=created_at,
    )
    newly_loaded_grant_ids = {
        name: grant_id for name, grant_id in grant_ids.items() if name not in replay_loaded_names
    }
    if desired_view == view:
        return (), newly_loaded_grant_ids
    if raw_view is None:  # pragma: no cover - current_tool_discovery_view rejects this
        raise RuntimeError("Hosted Tool Search lost its source discovery view.")
    mutation = RuntimePublicationOperationRecordMutation(
        key=TOOL_DISCOVERY_VIEW_OPERATION_KEY,
        expected_value_digest=runtime_publication_operation_record_value_digest(raw_view),
        value=desired_view.model_dump(mode="json"),
    )
    return (
        (mutation,),
        newly_loaded_grant_ids,
    )


def _redacted_provider_tool_definitions(
    tools: Iterable[Mapping[str, Any]],
    *,
    redactor: SecretRedactor,
    field_name: str,
) -> list[dict[str, Any]]:
    """Validate executable tool authority and redact its provider-visible values."""

    copied = copy_json_value(list(tools), field_name)
    if type(copied) is not list or any(type(tool) is not dict for tool in copied):
        raise AssertionError("Provider tool definitions must be a list of objects.")
    copied_tools = cast("list[dict[str, Any]]", copied)
    for index, tool in enumerate(copied_tools):
        tool_name = tool.get("name")
        if type(tool_name) is not str:
            raise AssertionError("A provider tool name must be a string.")
        if redactor.redact_text(tool_name) != tool_name:
            raise ValueError(
                f"{field_name}[{index}].name contains a workload secret and cannot "
                "be sent as provider execution authority."
            )
        redactor.require_no_secret_keys(
            {
                "name": tool_name,
                "description": tool.get("description"),
                "input_schema": None,
            },
            field_name=f"{field_name}[{index}]",
            preserve_keys={"name", "description", "input_schema"},
            match_short_substrings=True,
        )
        input_schema = tool.get("input_schema")
        if type(input_schema) is not dict:
            raise AssertionError("A provider tool input_schema must be an object.")
        require_secret_free_json_schema_keys(
            input_schema,
            redactor=redactor,
            field_name=f"{field_name}[{index}].input_schema",
        )
    redacted = redactor.redact_json_values(
        copied_tools,
        preserve_string_fields={"name"},
    )
    if type(redacted) is not list or any(type(tool) is not dict for tool in redacted):
        raise AssertionError("Provider tool redaction returned a non-list of objects.")
    return redacted
