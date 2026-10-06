"""Runtime attestation for Cayu-owned shared-artifact tool results."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
from typing import Any, Literal, cast

from cayu._validation import canonical_durable_json_bytes, copy_durable_json_value
from cayu.events import (
    Event,
    event_payload_authority_is_runtime_generated,
    event_with_runtime_payload_authority,
)
from cayu.tools import _shared_artifact_result_schema as shared_artifact_result_schema
from cayu.tools.base import ToolResult
from cayu.tools.shared_artifacts import (
    MATERIALIZE_SHARED_ARTIFACT_TOOL_NAME,
    PUBLISH_WORKSPACE_ARTIFACT_TOOL_NAME,
    MaterializeSharedArtifactTool,
    PublishWorkspaceArtifactTool,
    SharedArtifactMaterializationReceipt,
    SharedArtifactPublicationReceipt,
    SharedArtifactRef,
)

_AUTHORITY_PREFIX = "cayu.shared-artifact-result.v1"
_ResultKind = Literal["publication", "materialization"]


def attest_runtime_shared_artifact_result(
    event: Event,
    result: ToolResult,
    *,
    tool: object | None,
) -> Event:
    """Attest an exact success tuple produced by one shipped Runtime tool."""

    kind = _runtime_tool_kind(tool)
    if kind is None:
        return event
    controls = _controls(result.structured, kind=kind)
    if controls is None:
        return event
    marker = _marker(kind, controls)
    payload = dict(event.payload)
    payload[shared_artifact_result_schema.SHARED_ARTIFACT_RESULT_AUTHORITY_FIELD] = marker
    return event_with_runtime_payload_authority(
        event.model_copy(update={"payload": payload}),
        shared_artifact_result_schema.SHARED_ARTIFACT_RESULT_AUTHORITY_FIELD,
    )


def restore_attested_tool_result(
    event: Event,
    *,
    original: ToolResult,
    redacted: ToolResult,
) -> tuple[Event, ToolResult]:
    controls = _attested_controls(event, original.structured, trust_persisted=False)
    if controls is None:
        return _without_authority(event), redacted
    return event, _result_with_controls(redacted, controls)


def preserve_attested_controls_across_hook(
    event: Event,
    *,
    original: ToolResult,
    replacement: ToolResult,
) -> ToolResult:
    controls = _attested_controls(event, original.structured, trust_persisted=False)
    return replacement if controls is None else _result_with_controls(replacement, controls)


def restore_persisted_shared_artifact_result_authority(event: Event) -> Event:
    """Restore private marker provenance after a staged terminal is loaded."""

    if shared_artifact_result_schema.SHARED_ARTIFACT_RESULT_AUTHORITY_FIELD not in event.payload:
        return event
    raw_result = event.payload.get("result")
    controls = (
        None
        if type(raw_result) is not dict
        else _attested_controls(
            event,
            raw_result.get("structured"),
            trust_persisted=True,
        )
    )
    if controls is None:
        raise ValueError("Persisted shared-artifact result authority is malformed.")
    return event_with_runtime_payload_authority(
        event,
        shared_artifact_result_schema.SHARED_ARTIFACT_RESULT_AUTHORITY_FIELD,
    )


def restore_attested_event_result(
    event: Event,
    *,
    redacted_payload: dict[str, Any],
    trust_persisted: bool,
    reject_malformed: bool,
) -> None:
    if shared_artifact_result_schema.SHARED_ARTIFACT_RESULT_AUTHORITY_FIELD not in event.payload:
        return
    source_result = event.payload.get("result")
    target_result = redacted_payload.get("result")
    controls = (
        None
        if type(source_result) is not dict
        else _attested_controls(
            event,
            source_result.get("structured"),
            trust_persisted=trust_persisted,
        )
    )
    target_structured = None if type(target_result) is not dict else target_result.get("structured")
    if controls is None or type(target_structured) is not dict:
        if reject_malformed:
            raise ValueError("Shared-artifact result authority is malformed.")
        return
    _overlay(target_structured, controls)


def persisted_shared_artifact_control_paths(
    value: object,
) -> frozenset[tuple[str, ...]]:
    """Return exact staged-event paths covered by one valid persisted marker."""

    if type(value) is not dict:
        return frozenset()
    try:
        event = Event.model_validate(value)
    except (TypeError, ValueError):
        return frozenset()
    raw_result = event.payload.get("result")
    structured = None if type(raw_result) is not dict else raw_result.get("structured")
    controls = _attested_controls(event, structured, trust_persisted=True)
    if controls is None:
        return frozenset()
    return frozenset(
        {
            ("payload", shared_artifact_result_schema.SHARED_ARTIFACT_RESULT_AUTHORITY_FIELD),
            *(
                ("payload", "result", "structured", *path)
                for path in _string_control_paths(controls)
            ),
        }
    )


def _runtime_tool_kind(tool: object | None) -> _ResultKind | None:
    if type(tool) is PublishWorkspaceArtifactTool:
        return "publication"
    if type(tool) is MaterializeSharedArtifactTool:
        return "materialization"
    return None


def _controls(structured: object, *, kind: _ResultKind) -> dict[str, Any] | None:
    source = _json_object(structured)
    if source is None:
        return None
    expected_keys = {
        "shared_artifact_kind",
        "opaque_ref",
        "shared_artifact_ref",
        "publication_receipt" if kind == "publication" else "materialization_receipt",
        "recovered_from_durable_receipt",
    }
    if set(source) != expected_keys or source.get("shared_artifact_kind") != kind:
        return None
    if type(source.get("recovered_from_durable_receipt")) is not bool:
        return None
    try:
        reference = SharedArtifactRef.model_validate(source.get("shared_artifact_ref"))
        if source.get("opaque_ref") != reference.to_opaque_ref():
            return None
        if kind == "publication":
            receipt: SharedArtifactPublicationReceipt | SharedArtifactMaterializationReceipt = (
                SharedArtifactPublicationReceipt.model_validate(source.get("publication_receipt"))
            )
        else:
            receipt = SharedArtifactMaterializationReceipt.model_validate(
                source.get("materialization_receipt")
            )
    except (TypeError, ValueError):
        return None
    if receipt.reference != reference:
        return None
    return source


def _attested_controls(
    event: Event,
    structured: object,
    *,
    trust_persisted: bool,
) -> dict[str, Any] | None:
    marker = event.payload.get(shared_artifact_result_schema.SHARED_ARTIFACT_RESULT_AUTHORITY_FIELD)
    if type(marker) is not str:
        return None
    parts = marker.split(":")
    if (
        len(parts) != 3
        or parts[0] != _AUTHORITY_PREFIX
        or parts[1] not in {"publication", "materialization"}
        or len(parts[2]) != 64
        or any(character not in "0123456789abcdef" for character in parts[2])
    ):
        return None
    kind = cast("_ResultKind", parts[1])
    if not trust_persisted and not event_payload_authority_is_runtime_generated(
        event,
        field_name=shared_artifact_result_schema.SHARED_ARTIFACT_RESULT_AUTHORITY_FIELD,
        value=marker,
    ):
        return None
    expected_tool_name = (
        PUBLISH_WORKSPACE_ARTIFACT_TOOL_NAME
        if kind == "publication"
        else MATERIALIZE_SHARED_ARTIFACT_TOOL_NAME
    )
    if event.tool_name != expected_tool_name:
        return None
    controls = _controls(structured, kind=kind)
    return controls if controls is not None and marker == _marker(kind, controls) else None


def _marker(kind: _ResultKind, controls: Mapping[str, Any]) -> str:
    digest = sha256(
        b"cayu.shared-artifact-result.v1\0"
        + kind.encode("ascii")
        + b"\0"
        + canonical_durable_json_bytes(controls, "shared_artifact_result_controls")
    ).hexdigest()
    return f"{_AUTHORITY_PREFIX}:{kind}:{digest}"


def _json_object(value: object) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    copied = copy_durable_json_value(value, "shared_artifact_result")
    return copied if type(copied) is dict else None


def _without_authority(event: Event) -> Event:
    if shared_artifact_result_schema.SHARED_ARTIFACT_RESULT_AUTHORITY_FIELD not in event.payload:
        return event
    payload = dict(event.payload)
    payload.pop(shared_artifact_result_schema.SHARED_ARTIFACT_RESULT_AUTHORITY_FIELD, None)
    return event.model_copy(update={"payload": payload})


def _result_with_controls(result: ToolResult, controls: Mapping[str, Any]) -> ToolResult:
    structured = _json_object(result.structured) or {}
    _overlay(structured, controls)
    return ToolResult(
        content=result.content,
        structured=structured,
        artifacts=result.artifacts,
        is_error=result.is_error,
    )


def _overlay(target: dict[str, Any], controls: Mapping[str, Any]) -> None:
    for key, value in controls.items():
        target[key] = copy_durable_json_value(value, key)


def _string_control_paths(
    value: object,
    path: tuple[str, ...] = (),
) -> set[tuple[str, ...]]:
    if type(value) is str:
        return {path}
    if type(value) is list:
        return {item for value_item in value for item in _string_control_paths(value_item, path)}
    if type(value) is not dict:
        return set()
    observed: set[tuple[str, ...]] = set()
    for key, item in value.items():
        if type(key) is str:
            observed.update(_string_control_paths(item, (*path, key)))
    return observed


__all__ = [
    "attest_runtime_shared_artifact_result",
    "persisted_shared_artifact_control_paths",
    "preserve_attested_controls_across_hook",
    "restore_attested_event_result",
    "restore_attested_tool_result",
    "restore_persisted_shared_artifact_result_authority",
]
