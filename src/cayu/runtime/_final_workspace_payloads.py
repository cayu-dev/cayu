"""Durable payloads describing a session's final workspace state.

Pure builders used by the environment lifecycle when it records the final
workspace revision, snapshot, source publication and final Git receipt.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Any

from cayu._coding_product_authority import (
    CODING_PRODUCT_FINAL_GIT_RECEIPT_SCHEMA,
    is_final_git_result_envelope,
)
from cayu._validation import canonical_durable_json_bytes, copy_durable_metadata, copy_json_value
from cayu.environments.bindings import (
    SyncBinding,
    WorkspaceBinding,
    WorkspaceSnapshot,
    _runtime_owned_workspace_observer_name,
)
from cayu.environments.docker_coding import DockerCodingWorkspaceBinding
from cayu.runtime import _invocation_secrets as invocation_secrets
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime.public_authority import PublicAuthorityAliasCodec
from cayu.vaults.redaction import SecretRedactor
from cayu.workspaces.observation_recovery import _project_workspace_observation_authority
from cayu.workspaces.revisions import (
    WorkspaceMutationAttributionConfidence,
    WorkspaceRevisionDeltaStatus,
    WorkspaceRevisionObservation,
    WorkspaceRevisionObservationStatus,
)


def _final_workspace_revision_payload(
    observation: WorkspaceRevisionObservation,
    *,
    registered_environment: runtime_records.RegisteredEnvironment | None,
    session_id: str,
    redactor: SecretRedactor,
    public_authority_alias_codec: PublicAuthorityAliasCodec | None,
    finalization_delta: dict[str, Any] | None,
) -> dict[str, Any]:
    if (
        invocation_secrets.registered_environment_secret_resolution_scope(registered_environment)
        != "static"
    ):
        # Invocation-resolved secret registries deliberately remain in-process
        # and invocation-scoped. Terminal finalization can run after that scope
        # has been discarded or in a fresh recovery process, so arbitrary
        # observer text cannot be proven safe for a dynamic environment. Keep a
        # fixed, content-free view instead of allowing branch/revision fields to
        # bypass the tool-round publication quarantine.
        binding = (
            None if registered_environment is None else registered_environment.environment.binding
        )
        runtime_owned_observer = (
            None if binding is None else _runtime_owned_workspace_observer_name(binding)
        )
        try:
            projected_authority = _project_workspace_observation_authority(
                session_id=session_id,
                configured_workspace_id=observation.identity.workspace_id,
                configured_observer=observation.identity.observer,
                configured_artifact_store_id=None,
                observer_is_runtime_owned=(runtime_owned_observer == observation.identity.observer),
                secret_resolution_scope="dynamic",
                redactor=redactor,
                public_authority_alias_codec=public_authority_alias_codec,
            )
            workspace_id = projected_authority.workspace_id
            observer = projected_authority.observer
        except (RuntimeError, ValueError):
            # A store without a durable alias keyring cannot authenticate raw
            # dynamic identities after the invocation secret scope is gone.
            # Finalization remains available, but publishes fixed evidence.
            workspace_id = "workspace-authority-unavailable"
            observer = (
                observation.identity.observer
                if runtime_owned_observer == observation.identity.observer
                else "workspace-observer-unavailable"
            )
        payload: dict[str, Any] = {
            "workspace_id": workspace_id,
            "observer": observer,
            "status": WorkspaceRevisionObservationStatus.TRUNCATED.value,
            "revision": None,
            "head_revision": None,
            "branch": None,
            "path_scope": observation.path_scope,
            "total_paths": observation.total_paths,
            "detail_code": "final_revision_secret_scope_unavailable",
        }
        if finalization_delta is not None:
            payload["finalization_delta"] = {
                "attribution_confidence": (
                    WorkspaceMutationAttributionConfidence.UNATTRIBUTED_FINALIZATION_CHANGE.value
                ),
                "status": WorkspaceRevisionDeltaStatus.TRUNCATED.value,
                "before_revision": None,
                "after_revision": None,
                "paths": [],
                "retained_paths": 0,
                "total_paths": 0,
                "truncated": True,
                "head_changed": False,
                "branch_changed": False,
                "detail_code": "finalization_delta_secret_scope_unavailable",
            }
        return payload
    payload = {
        "workspace_id": observation.identity.workspace_id,
        "observer": observation.identity.observer,
        "status": observation.status.value,
        "revision": observation.revision,
        "head_revision": observation.head_revision,
        "branch": observation.branch,
        "path_scope": observation.path_scope,
        "total_paths": observation.total_paths,
        "detail_code": observation.detail_code,
    }
    if finalization_delta is not None:
        payload["finalization_delta"] = copy_json_value(
            finalization_delta,
            "finalization_delta",
        )
    return payload


def _final_workspace_snapshot_payload(
    snapshot: WorkspaceSnapshot | None,
    *,
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> dict[str, Any] | None:
    if (
        invocation_secrets.registered_environment_secret_resolution_scope(registered_environment)
        != "static"
    ):
        # Binding snapshots may repeat observer-derived branch, revision, path,
        # or extension metadata. They share the terminal workspace-evidence
        # boundary and therefore require the same positive static-scope proof as
        # the final revision projection.
        return None
    return _workspace_snapshot_payload(snapshot)


def _source_publication_receipt_payload(
    snapshot: WorkspaceSnapshot | None,
    *,
    binding: WorkspaceBinding,
    destination_workspace_id: object,
    workload_workspace_id: object,
    expected_destination_workspace_id: str | None,
    expected_workload_workspace_id: str | None,
) -> dict[str, object] | None:
    """Publish the fixed, secret-free settlement fields owned by ``SyncBinding``."""

    if (
        not isinstance(binding, SyncBinding)
        or snapshot is None
        or snapshot.source != "sync"
        or expected_destination_workspace_id is None
        or expected_workload_workspace_id is None
        or snapshot.workspace_id != expected_destination_workspace_id
        or snapshot.metadata.get("target_workspace_id") != expected_workload_workspace_id
    ):
        return None
    metadata = snapshot.metadata
    copied_files = metadata.get("copied_files")
    copied_bytes = metadata.get("copied_bytes")
    deleted_files = metadata.get("deleted_files")
    outcome = metadata.get("outcome")
    source_conflict_policy = metadata.get("source_conflict_policy")
    sync_back = metadata.get("sync_back")
    delete_missing = metadata.get("delete_missing")
    if (
        type(destination_workspace_id) is not str
        or not destination_workspace_id
        or type(workload_workspace_id) is not str
        or not workload_workspace_id
        or type(copied_files) is not int
        or copied_files < 0
        or type(copied_bytes) is not int
        or copied_bytes < 0
        or type(deleted_files) is not int
        or deleted_files < 0
        or type(outcome) is not str
        or type(source_conflict_policy) is not str
        or type(sync_back) is not str
        or type(delete_missing) is not bool
    ):
        return None
    snapshot_payload = {
        "schema": "cayu.source_publication_snapshot.v1",
        "destination_workspace_id": destination_workspace_id,
        "workload_workspace_id": workload_workspace_id,
        "source": "sync",
        "outcome": outcome,
        "source_conflict_policy": source_conflict_policy,
        "sync_back": sync_back,
        "delete_missing": delete_missing,
        "copied_files": copied_files,
        "copied_bytes": copied_bytes,
        "deleted_files": deleted_files,
    }
    snapshot_sha256 = (
        "sha256:"
        + sha256(
            canonical_durable_json_bytes(snapshot_payload, "source_publication_snapshot")
        ).hexdigest()
    )
    receipt = {
        "schema": "cayu.source_publication_receipt.v1",
        "snapshot_sha256": snapshot_sha256,
        "destination_workspace_id": destination_workspace_id,
        "workload_workspace_id": workload_workspace_id,
        "outcome": outcome,
        "source_conflict_policy": source_conflict_policy,
        "sync_back": sync_back,
        "delete_missing": delete_missing,
        "copied_files": copied_files,
        "copied_bytes": copied_bytes,
        "deleted_files": deleted_files,
    }
    return {
        **receipt,
        "receipt_sha256": "sha256:"
        + sha256(canonical_durable_json_bytes(receipt, "source_publication_receipt")).hexdigest(),
    }


def _final_git_receipt_payload(
    snapshot: WorkspaceSnapshot | None,
    *,
    binding: WorkspaceBinding,
    destination_workspace_id: object,
    workload_workspace_id: object,
    expected_destination_workspace_id: str | None,
    expected_workload_workspace_id: str | None,
    registered_environment: runtime_records.RegisteredEnvironment | None = None,
) -> dict[str, object] | None:
    """Publish digest-bound final Git evidence from the Docker sync boundary."""

    if (
        invocation_secrets.registered_environment_secret_resolution_scope(registered_environment)
        != "static"
    ):
        # Git evidence includes arbitrary workspace text and shares the terminal
        # snapshot quarantine after invocation-local secret scopes are discarded.
        return None
    if (
        type(binding) is not DockerCodingWorkspaceBinding
        or snapshot is None
        or snapshot.source != "sync"
        or expected_destination_workspace_id is None
        or expected_workload_workspace_id is None
        or snapshot.workspace_id != expected_destination_workspace_id
        or snapshot.metadata.get("target_workspace_id") != expected_workload_workspace_id
    ):
        return None
    evidence = snapshot.metadata.get("final_git_evidence")
    if type(evidence) is not dict or set(evidence) != {
        "request_fingerprint",
        "source_workspace_id",
        "baseline_revision",
        "workspace_revision",
        "status",
        "summary",
        "diff",
    }:
        return None
    request_fingerprint = evidence.get("request_fingerprint")
    source_workspace_id = evidence.get("source_workspace_id")
    baseline_revision = evidence.get("baseline_revision")
    workspace_revision = evidence.get("workspace_revision")
    status = evidence.get("status")
    summary = evidence.get("summary")
    diff = evidence.get("diff")
    if (
        type(destination_workspace_id) is not str
        or not destination_workspace_id
        or type(workload_workspace_id) is not str
        or not workload_workspace_id
        or type(request_fingerprint) is not str
        or not request_fingerprint
        or source_workspace_id != expected_destination_workspace_id
        or type(baseline_revision) is not str
        or not baseline_revision
        or type(workspace_revision) is not str
        or not workspace_revision
        or type(status) is not dict
        or set(status) != {"structured"}
        or type(status.get("structured")) is not dict
        or not is_final_git_result_envelope(status.get("structured"), mode="status")
        or type(summary) is not dict
        or set(summary) != {"structured"}
        or type(summary.get("structured")) is not dict
        or not is_final_git_result_envelope(summary.get("structured"), mode="summary")
        or type(diff) is not dict
        or set(diff) != {"content", "structured"}
        or type(diff.get("content")) is not str
        or type(diff.get("structured")) is not dict
        or not is_final_git_result_envelope(diff.get("structured"), mode="diff")
    ):
        return None
    receipt = {
        "schema": CODING_PRODUCT_FINAL_GIT_RECEIPT_SCHEMA,
        "request_fingerprint": request_fingerprint,
        "destination_workspace_id": destination_workspace_id,
        "workload_workspace_id": workload_workspace_id,
        "baseline_revision": baseline_revision,
        "workspace_revision": workspace_revision,
        "status": copy_json_value(status, "final_git_status"),
        "summary": copy_json_value(summary, "final_git_summary"),
        "diff": copy_json_value(diff, "final_git_diff"),
    }
    return {
        **receipt,
        "receipt_sha256": "sha256:"
        + sha256(canonical_durable_json_bytes(receipt, "final_git_receipt")).hexdigest(),
    }


def _workspace_snapshot_payload(snapshot: WorkspaceSnapshot | None) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    return {
        "snapshot_id": snapshot.snapshot_id,
        "workspace_id": snapshot.workspace_id,
        "version": snapshot.version,
        "source": snapshot.source,
        "metadata": copy_durable_metadata(snapshot.metadata, "metadata"),
    }
