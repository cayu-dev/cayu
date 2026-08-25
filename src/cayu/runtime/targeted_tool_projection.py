"""Targeted-tool delivery selection and cache-stable native projection markers."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from hashlib import sha256
from typing import Any

from cayu._validation import canonical_durable_json_bytes, require_durable_clean_nonblank
from cayu.core.messages import Message, MessageRole, ProviderStatePart
from cayu.providers.base import (
    OPENAI_ADDITIONAL_TOOLS_PROTOCOL,
    TARGETED_TOOL_PROJECTION_MARKER_TYPE,
    ModelProvider,
    TargetedToolProjectionRequest,
)
from cayu.runtime.tool_catalogue import ToolCatalogSnapshot
from cayu.runtime.tool_grants import (
    TARGETED_TOOL_GRANT_MAX_REQUESTS,
    PreparedTargetedToolGrant,
    TargetedToolGrantRecord,
    copy_targeted_tool_grant_record,
    persisted_targeted_tool_grant_batch_fingerprint,
    prepared_targeted_tool_grant_batch_fingerprint,
)


class TargetedToolMode(StrEnum):
    """Application-selected delivery policy for interaction-scoped tool grants."""

    CALL_TOOL = "call_tool"
    OPENAI_ADDITIONAL_TOOLS = "openai_additional_tools"
    OPENAI_ADDITIONAL_TOOLS_OR_CALL_TOOL = "openai_additional_tools_or_call_tool"


class TargetedToolProjectionKind(StrEnum):
    """One concrete provider projection selected for an invocation."""

    CALL_TOOL = "call_tool"
    OPENAI_ADDITIONAL_TOOLS = "openai_additional_tools"


def copy_targeted_tool_mode(
    value: object,
    field_name: str = "targeted_tool_mode",
) -> TargetedToolMode:
    """Return one exact targeted-tool delivery policy."""

    if type(value) is TargetedToolMode:
        return value
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a TargetedToolMode or string.")
    try:
        return TargetedToolMode(value)
    except ValueError:
        supported = ", ".join(mode.value for mode in TargetedToolMode)
        raise ValueError(f"{field_name} must be one of: {supported}.") from None


def resolve_targeted_tool_projection(
    mode: TargetedToolMode | None,
    *,
    provider: ModelProvider,
    model: str,
) -> TargetedToolProjectionKind | None:
    """Resolve one explicit delivery policy without model-name inference."""

    if mode is None:
        return None
    mode = copy_targeted_tool_mode(mode)
    model = require_durable_clean_nonblank(model, "model")
    if mode is TargetedToolMode.CALL_TOOL:
        return TargetedToolProjectionKind.CALL_TOOL
    supported = provider.supports_targeted_tool_projection(
        model=model,
        protocol=OPENAI_ADDITIONAL_TOOLS_PROTOCOL,
    )
    if type(supported) is not bool:
        raise TypeError("Provider targeted-tool capability checks must return a bool.")
    if supported:
        return TargetedToolProjectionKind.OPENAI_ADDITIONAL_TOOLS
    if mode is TargetedToolMode.OPENAI_ADDITIONAL_TOOLS_OR_CALL_TOOL:
        return TargetedToolProjectionKind.CALL_TOOL
    provider.preflight_targeted_tool_projection(
        model=model,
        protocol=OPENAI_ADDITIONAL_TOOLS_PROTOCOL,
    )
    raise RuntimeError("Provider targeted-tool preflight returned without establishing support.")


def _marker_id(
    *,
    interaction_id: str,
    generation_id: str,
    batch_fingerprint: str,
) -> str:
    material = {
        "record_type": TARGETED_TOOL_PROJECTION_MARKER_TYPE,
        "schema_version": 1,
        "interaction_id": require_durable_clean_nonblank(interaction_id, "interaction_id"),
        "generation_id": require_durable_clean_nonblank(generation_id, "generation_id"),
        "batch_fingerprint": require_durable_clean_nonblank(
            batch_fingerprint,
            "batch_fingerprint",
        ),
    }
    return f"sha256:{sha256(canonical_durable_json_bytes(material, 'targeted marker')).hexdigest()}"


def targeted_tool_projection_marker_message(
    prepared: tuple[PreparedTargetedToolGrant, ...],
    *,
    interaction_id: str,
    generation_id: str,
) -> Message:
    """Build the tiny runtime-owned marker committed at grant acquisition."""

    if not prepared or len(prepared) > TARGETED_TOOL_GRANT_MAX_REQUESTS:
        raise ValueError("A targeted projection marker requires a bounded grant batch.")
    if any(type(grant) is not PreparedTargetedToolGrant for grant in prepared):
        raise TypeError("prepared must contain exact PreparedTargetedToolGrant values.")
    marker_id = _marker_id(
        interaction_id=interaction_id,
        generation_id=generation_id,
        batch_fingerprint=prepared_targeted_tool_grant_batch_fingerprint(prepared),
    )
    return Message(
        role=MessageRole.ASSISTANT,
        content=(
            ProviderStatePart(
                provider="openai",
                state={
                    "type": TARGETED_TOOL_PROJECTION_MARKER_TYPE,
                    "protocol": OPENAI_ADDITIONAL_TOOLS_PROTOCOL,
                    "marker_id": marker_id,
                },
            ),
        ),
    )


def openai_targeted_tool_projection(
    records: Iterable[TargetedToolGrantRecord],
    *,
    catalogue: ToolCatalogSnapshot,
) -> tuple[TargetedToolProjectionRequest, dict[str, str]]:
    """Project one admitted native snapshot and its private grant mapping.

    Once sent, an OpenAI ``additional_tools`` item remains part of stateless replay
    even if its grant later expires, is revoked, or is consumed. Those lifecycle
    changes are enforced when a call resolves; they never rewrite the historical
    provider item or its cache boundary.
    """

    if type(catalogue) is not ToolCatalogSnapshot:
        raise TypeError("catalogue must be a ToolCatalogSnapshot.")
    copied = tuple(copy_targeted_tool_grant_record(record) for record in records)
    if not copied or len(copied) > TARGETED_TOOL_GRANT_MAX_REQUESTS:
        raise ValueError("Native targeted projection requires a bounded grant batch.")
    interaction_ids = {record.interaction_id for record in copied}
    generation_ids = {record.generation_id for record in copied}
    if len(interaction_ids) != 1 or len(generation_ids) != 1:
        raise ValueError("Native targeted projection records must share one authority scope.")
    tools: list[dict[str, Any]] = []
    grant_ids_by_name: dict[str, str] = {}
    for record in copied:
        if record.catalogue_revision != catalogue.revision:
            raise ValueError("Native targeted projection has stale catalogue authority.")
        try:
            descriptor = catalogue.descriptor_for_id(record.tool_id)
        except KeyError:
            raise ValueError("Native targeted projection names an unknown descriptor.") from None
        if (
            descriptor.name != record.tool_name
            or descriptor.version != record.descriptor_version
            or descriptor.schema_fingerprint != record.schema_fingerprint
        ):
            raise ValueError("Native targeted projection has stale descriptor authority.")
        if descriptor.name in grant_ids_by_name:
            raise ValueError("Native targeted projection repeats a tool name.")
        grant_ids_by_name[descriptor.name] = record.grant_id
        tools.append(
            {
                "name": descriptor.name,
                "description": descriptor.description,
                "input_schema": descriptor.input_schema_copy(),
            }
        )
    tools.sort(key=lambda tool: str(tool["name"]))
    marker_id = persisted_targeted_tool_projection_marker_id(copied)
    return (
        TargetedToolProjectionRequest(marker_id=marker_id, tools=tuple(tools)),
        grant_ids_by_name,
    )


def persisted_targeted_tool_projection_marker_id(
    records: Iterable[TargetedToolGrantRecord],
) -> str:
    """Derive the acquisition marker identity from one admitted durable batch."""

    copied = tuple(copy_targeted_tool_grant_record(record) for record in records)
    if not copied or len(copied) > TARGETED_TOOL_GRANT_MAX_REQUESTS:
        raise ValueError("A targeted projection marker requires a bounded grant batch.")
    interaction_ids = {record.interaction_id for record in copied}
    generation_ids = {record.generation_id for record in copied}
    if len(interaction_ids) != 1 or len(generation_ids) != 1:
        raise ValueError("Targeted projection records must share one authority scope.")
    return _marker_id(
        interaction_id=next(iter(interaction_ids)),
        generation_id=next(iter(generation_ids)),
        batch_fingerprint=persisted_targeted_tool_grant_batch_fingerprint(copied),
    )


def targeted_tool_projection_marker_id(message: Message) -> str | None:
    """Return the validated marker identity from one runtime control message."""

    if type(message) is not Message or message.role is not MessageRole.ASSISTANT:
        return None
    if len(message.content) != 1 or type(message.content[0]) is not ProviderStatePart:
        return None
    part = message.content[0]
    if part.provider != "openai" or part.state.get("type") != TARGETED_TOOL_PROJECTION_MARKER_TYPE:
        return None
    if set(part.state) != {"type", "protocol", "marker_id"}:
        raise ValueError("Targeted projection marker must use the canonical shape.")
    if part.state.get("protocol") != OPENAI_ADDITIONAL_TOOLS_PROTOCOL:
        raise ValueError("Targeted projection marker uses an unsupported protocol.")
    marker_id = part.state.get("marker_id")
    if (
        type(marker_id) is not str
        or len(marker_id) != 71
        or not marker_id.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in marker_id[7:])
    ):
        raise ValueError("Targeted projection marker has an invalid identity.")
    return marker_id
