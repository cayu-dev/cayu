"""Environment handles and targeted authority shared by invocation components."""

from __future__ import annotations

from typing import Any

from cayu._validation import (
    require_clean_nonblank,
    require_unicode_scalar_text,
)
from cayu.artifacts.local import LocalArtifactStore
from cayu.events import (
    Event,
    event_retains_runtime_payload_authority,
    event_with_runtime_payload_authority,
)
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._event_projection import PRIVATE_EVENT_AUTHORITY
from cayu.tools import _argument_publication as tool_argument_publication
from cayu.vaults.redaction import SecretRedactor
from cayu.workspaces.local import LocalWorkspace


def _published_argument_presence(tool_call, registered_tool):
    return tool_argument_publication.publish_argument_presence(
        tool_call.argument_presence,
        None if registered_tool is None else registered_tool.schema,
    )


def _environment_name(
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> str | None:
    return None if registered_environment is None else registered_environment.spec.name


def _workspace_id(
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> str | None:
    if registered_environment is None or registered_environment.environment.workspace is None:
        return None
    workspace_id = getattr(registered_environment.environment.workspace, "id", None)
    if workspace_id is None:
        return None
    return require_clean_nonblank(workspace_id, "workspace.id")


def _workspace(registered_environment: runtime_records.RegisteredEnvironment | None) -> Any:
    if registered_environment is None:
        return None
    return registered_environment.environment.workspace


def _artifact_store_id(
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> str | None:
    if registered_environment is None or registered_environment.environment.artifact_store is None:
        return None
    artifact_store_id = getattr(registered_environment.environment.artifact_store, "id", None)
    if artifact_store_id is None:
        return None
    artifact_store_id = require_clean_nonblank(artifact_store_id, "artifact_store.id")
    return require_unicode_scalar_text(artifact_store_id, "artifact_store.id")


def _artifact_store(registered_environment: runtime_records.RegisteredEnvironment | None) -> Any:
    if registered_environment is None:
        return None
    return registered_environment.environment.artifact_store


def _workspace_receipt_artifact_store(
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> tuple[Any, str]:
    """Exclude local stores whose receipt writes would mutate the observed workspace."""

    artifact_store = _artifact_store(registered_environment)
    workspace = _workspace(registered_environment)
    if registered_environment is not None and registered_environment.bound_workspace is not None:
        workspace = registered_environment.bound_workspace.workspace
    if isinstance(artifact_store, LocalArtifactStore) and isinstance(workspace, LocalWorkspace):
        try:
            artifact_store.root.relative_to(workspace.root)
        except ValueError:
            pass
        else:
            return None, "manifest_artifact_store_inside_workspace"
    return artifact_store, "manifest_artifact_store_unavailable"


def _runner(registered_environment: runtime_records.RegisteredEnvironment | None) -> Any:
    if registered_environment is None:
        return None
    return registered_environment.environment.runner


def _knowledge_store(
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> Any:
    if registered_environment is None:
        return None
    return registered_environment.environment.knowledge_store


def _knowledge_access_scope(
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> Any:
    if registered_environment is None:
        return None
    return registered_environment.environment.knowledge_access_scope


def _mcp_servers(
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> tuple[Any, ...]:
    if registered_environment is None:
        return ()
    return registered_environment.environment.mcp_servers


def _targeted_tool_invocation_payload(
    tool_call: runtime_records.ToolCallRequest,
) -> dict[str, Any]:
    invocation = tool_call.targeted_tool_invocation
    if invocation is None:
        return {}
    payload = {
        "dispatch_kind": invocation.dispatch_kind,
        "model_tool_name": invocation.model_tool_name,
        "grant_id": invocation.grant_id,
        "use_id": invocation.use_id,
        "effective_tool_id": invocation.tool_id,
        "catalogue_revision": invocation.catalogue_revision,
        "descriptor_version": invocation.descriptor_version,
        "schema_fingerprint": invocation.schema_fingerprint,
        "arguments_sha256": invocation.arguments_sha256,
        "invocation_id": invocation.invocation_id,
    }
    if tuple(payload) != _TARGETED_TOOL_INVOCATION_PAYLOAD_FIELDS:
        raise AssertionError("Targeted tool invocation payload fields drifted.")
    return payload


def _event_with_targeted_tool_invocation_authority(
    event: Event,
    tool_call: runtime_records.ToolCallRequest,
) -> Event:
    """Attest exact targeted-tool linkage copied from one resolved invocation."""

    if tool_call.targeted_tool_invocation is None:
        return event
    return event_with_runtime_payload_authority(
        event,
        *_TARGETED_TOOL_INVOCATION_PAYLOAD_FIELDS,
    )


def _restore_targeted_tool_invocation_event_authority(
    event: Event,
    tool_call: runtime_records.ToolCallRequest,
    *,
    redactor: SecretRedactor | None = None,
) -> Event:
    """Rebind one terminal returned through a durable checkpoint callback."""

    invocation = tool_call.targeted_tool_invocation
    if invocation is None:
        return event
    if event.tool_name != tool_call.name or event.payload.get("tool_call_id") != tool_call.id:
        raise RuntimeError("Targeted tool terminal conflicts with its resolved invocation.")
    expected = _targeted_tool_invocation_payload(tool_call)
    for field_name, expected_value in expected.items():
        observed = event.payload.get(field_name)
        retained_authority = event_retains_runtime_payload_authority(
            event,
            field_name=field_name,
            value=expected_value,
        )
        redacted_attested_authority = retained_authority and (
            observed == PRIVATE_EVENT_AUTHORITY
            or (redactor is not None and observed == redactor.redact_json(expected_value))
        )
        if (
            field_name in event.payload
            and observed != expected_value
            and not redacted_attested_authority
        ):
            raise RuntimeError(
                "Targeted tool terminal conflicts with its durable invocation authority."
            )
    rebound = event.model_copy(update={"payload": {**event.payload, **expected}})
    return _event_with_targeted_tool_invocation_authority(rebound, tool_call)


_TARGETED_TOOL_INVOCATION_PAYLOAD_FIELDS = (
    "dispatch_kind",
    "model_tool_name",
    "grant_id",
    "use_id",
    "effective_tool_id",
    "catalogue_revision",
    "descriptor_version",
    "schema_fingerprint",
    "arguments_sha256",
    "invocation_id",
)
