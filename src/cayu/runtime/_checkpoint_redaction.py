from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from cayu._validation import copy_json_value
from cayu.runtime._shared_artifact_results import persisted_shared_artifact_control_paths
from cayu.runtime._web_access_results import persisted_web_access_control_paths
from cayu.runtime.checkpoints import (
    ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
    AUTOMATIC_RECALL_CHECKPOINT_KEY,
    CHECKPOINT_SCHEMA_VERSION_KEY,
    COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY,
    INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
    INVOCATION_LIFECYCLE_RECEIPT_LEDGER_RECORD_TYPE,
    INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
    SETTLED_INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
)
from cayu.runtime.execution_profiles import (
    EXECUTION_PROFILE_METADATA_KEY,
    ExecutionProfileIdentity,
    execution_profile_from_session_metadata,
)
from cayu.runtime.sessions import (
    RUNTIME_BUILD_PROVENANCE_METADATA_KEY,
    runtime_build_provenance_from_session_metadata,
)
from cayu.runtime.structured_output import json_schema_contains_secret
from cayu.runtime.tool_catalogue import CALL_TOOL_NAME
from cayu.runtime.tool_exposure import (
    TOOL_CAPABILITY_CEILING_METADATA_KEY,
    tool_capability_ceiling_from_session_metadata,
)
from cayu.runtime.tool_grants import ResolvedTargetedToolInvocation, validate_targeted_tool_digest
from cayu.vaults import SecretRedactor

_DURABLE_STRUCTURE_STRING_FIELDS = frozenset(
    {
        "agent_name",
        "after_observation_id",
        "approval_id",
        "artifact_id",
        "assistant_message_state",
        "before_observation_id",
        "before_state",
        "binding_generation_id",
        "catalogue_revision",
        "workspace_id",
        "observer",
        "observer_authority",
        "artifact_store_id",
        "covered_tool_call_ids",
        "decision",
        "environment_name",
        "evidence_kind",
        "after_state",
        "delta_state",
        "event_id",
        "expires_at",
        "hooks_state",
        "id",
        "input_id",
        "interaction_id",
        "model_attempt_id",
        "model_step_id",
        "mutation_event_id",
        "name",
        "policy_decision",
        "policy_evidence",
        "profile_id",
        "receipt_id",
        "phase",
        "role",
        "round_id",
        "secret_resolution_scope",
        "session_instance_id",
        "source_interaction_id",
        "source_model_step_id",
        "state",
        "strategy",
        "task_id",
        "timestamp",
        "tool_call_id",
        "tool_name",
        "tool_outcome_event_id",
        "tool_round_id",
        "type",
        "workflow_name",
        "window_id",
        "record_type",
        "resolution_stage",
        "execution_state",
        "component_class",
        "strength",
        "availability",
    }
)
_DURABLE_ENUM_STRING_FIELDS = frozenset(
    {
        "assistant_message_state",
        "channel",
        "channels",
        "decision",
        "policy_decision",
        "policy_evidence",
        "policy_state",
        "role",
        "secret_resolution_scope",
        "source",
        "state",
        "status",
        "strategy",
        "type",
        "hooks_state",
        "failure_code",
        "mode",
        "notice",
        "phase",
        "record_type",
        "resolution_stage",
        "execution_state",
        "selection_reason",
        "before_state",
        "after_state",
        "delta_state",
        "observer_authority",
    }
)
_DURABLE_SHA256_STRING_FIELDS = frozenset(
    {
        "contribution_sha256",
        "configuration_sha256",
        "answer_request_digest",
        "content_hash",
        "manifest_sha256",
        "policy_sha256",
        "pause_digest",
        "projection_sha256",
        "receipt_document_sha256",
        "receipt_manifest_binding_hmac_sha256",
        "resolution_request_digest",
        "situation_sha256",
        "source_root_digest",
        "target_root_digest",
        "user_message_sha256",
        "user_text_sha256",
        "fingerprint",
        "execution_profile_fingerprint",
        "exposure_fingerprint",
        "sha256",
        "mutation_event_digest",
        "tool_outcome_event_digest",
        "ticket",
    }
)
_DURABLE_STRUCTURE_KEYS = (_DURABLE_STRUCTURE_STRING_FIELDS | _DURABLE_SHA256_STRING_FIELDS) | {
    "approval_close_intent",
    ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
    INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
    INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
    SETTLED_INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
    "components",
    "assistant_publication",
    "approval_resolution_intent",
    "user_input_resolution_intent",
    "user_input_supersession_intent",
    "pending_tool_approval",
    "pending_tool_round",
    "pending_user_input",
    "action",
    "active_taint_labels",
    "active_operation_id",
    "aliases",
    "allow_unpriced",
    "arguments",
    "artifacts",
    "anchor_transcript_index",
    AUTOMATIC_RECALL_CHECKPOINT_KEY,
    "as_of",
    "backoff_multiplier",
    "batch",
    "budget_limits",
    "cache_read_input_per_million",
    "cache_write_1h_per_million",
    "cache_write_5m_per_million",
    "cache_write_input_per_million",
    "cache_write_ttls",
    "contextual_pricing_requirements",
    "candidate_count",
    "candidates",
    "chunk_id",
    "chunk_index",
    "compact_notice",
    "context_compaction",
    "created_at",
    "currency",
    "deferred_messages",
    "dimensions",
    "duration_seconds",
    "entry_id",
    "effort",
    "effective_from",
    "effective_max_attempts",
    "effective_through",
    "enabled",
    "errors",
    "event",
    "generated_at",
    "generation",
    "include_in_transcript",
    "incomplete_session_recovery_claim",
    "initial_delay_s",
    "interrupt_payload",
    "input_per_million",
    "jitter_s",
    "json_schema",
    "key",
    "kind",
    "label",
    "runtime_authored_user_message",
    "runtime_authored_anchors",
    "runtime_build_provenance",
    "limits",
    "match",
    "match_prefixes",
    "message",
    "max_attempts",
    "max_cache_read_input_tokens",
    "max_cache_write_input_tokens",
    "max_delay_s",
    "max_elapsed_seconds",
    "max_estimated_cost",
    "max_input_tokens",
    "max_output_tokens",
    "max_retries",
    "max_steps",
    "max_tokens",
    "max_tool_calls",
    "max_total_tokens",
    "max_unknown_attempts",
    "model_step",
    "metadata",
    "min_input_tokens",
    "min_total_tokens",
    "model",
    "namespace",
    "options",
    "output_per_million",
    "path",
    "period",
    "policy_context_version",
    "policy_state",
    "payload",
    "price_book_version",
    "prices",
    "pricing",
    "pricing_context",
    "pricing_model",
    "provenance",
    "provider_name",
    "profile",
    "projected_bytes",
    "records",
    "question",
    "quarantined_assistant_message",
    "reason",
    "rank",
    "repair_prompt",
    "requires_cache_write_ttls",
    "reservation",
    "reserved_output_tokens",
    "resource_id",
    "resource_mappings",
    "retry_on_connection_error",
    "retry_on_rate_limit",
    "retry_on_status_codes",
    "retry_on_timeout",
    "retry_policy",
    "retry_request",
    "scope",
    "schema_version",
    "score",
    "score_kind",
    "score_normalized",
    "size_bytes",
    "schedules",
    "session_operations",
    "session_id",
    "source",
    "source_id",
    "source_type",
    "source_uri",
    "sources",
    "standard",
    "structured_output",
    "structured_output_attempt",
    "structured_output_validation",
    "summary",
    "title",
    "thinking",
    "timezone",
    "tool_exposure",
    "tool_names",
    "registered_count",
    "ceiling_count",
    "tool_calls",
    "targeted_tool_grant_id",
    "targeted_tool_invocation",
    "targeted_tool_rejection",
    "tool_ref",
    "grant_id",
    "use_id",
    "dispatch_kind",
    "model_tool_name",
    "effective_tool_name",
    "outer_tool_call_id",
    "arguments_sha256",
    "invocation_id",
    "rejection_event_id",
    "usage_triggered_context",
    "url",
    "version",
    "window",
    "environment_factory_allocation_owner",
    "environment_factory_reconnect",
    "pending_interruption_cascade",
    "pending_session_interrupt",
    "attempt_count",
    "attempt_id",
    "abandoned_at",
    "claim_expires_at",
    "claim_id",
    "claim_run_epoch",
    "claimed_at",
    "compacted_transcript_cursor",
    "current_attempt_id",
    "estimated_context_input_tokens",
    "estimated_context_window_tokens",
    "estimated_delta_input_tokens",
    "event_ids",
    "exhausted",
    "failure_recorded",
    "instruction_digest",
    "instruction_present",
    "injected_bytes",
    "last_input_tokens",
    "last_total_tokens",
    "last_transcript_cursor",
    "operation_id",
    "manifest",
    "manifest_truncated",
    "progress",
    "provider_count_context_window_tokens",
    "provider_count_input_tokens",
    "request_digest",
    "request_metadata",
    "source_run_epoch",
    "run_epoch",
    "source_transcript_cursor",
    "status",
    "staged_terminals",
    "trigger_estimated_context_tokens",
    "turns",
    "truncated",
    "updated_at",
    "valid",
    "excerpt",
    "channel",
    "channels",
    "coverage_truncated",
    "focus",
    "failure_code",
    "fused_rank",
    "identity",
    "index_version",
    "items",
    "locator_json",
    "matches",
    "mode",
    "notice",
    "offer",
    "omitted_item_count",
    "projection",
    "record_id",
    "representation",
    "required",
    "revision",
    "selection_reason",
    "text_complete",
}
_DURABLE_SUBAGENT_ROOTS = frozenset(
    {"durable_subagent_submission_seeds", "durable_subagent_submissions"}
)
_DURABLE_SUBAGENT_STRUCTURE_KEYS = frozenset(
    {
        "agent_alias",
        "authority",
        "causal_budget_id",
        "child_execution_profile",
        "child_model",
        "child_provider_name",
        "child_runtime_name",
        "child_runtime_version",
        "child_session_id",
        "content",
        "dispatch_id",
        "effective_arguments",
        "failure_code",
        "idempotency_key",
        "interaction_started_event_id",
        "invocation_origin",
        "labels",
        "messages",
        "parent_run_epoch",
        "parent_session_id",
        "parent_task_id",
        "outcome",
        "queue_task_id",
        "queue_task_type",
        "request",
        "spawn_fingerprint",
        "task_id",
        "task_worker_id",
        "text",
    }
)
_DURABLE_SUBAGENT_SHA256_STRING_FIELDS = frozenset(
    {
        "effective_arguments_sha256",
        "parent_execution_profile_fingerprint",
        "parent_session_instance_fingerprint",
        "request_sha256",
        "receipt_sha256",
        "seed_sha256",
        "submission_sha256",
    }
)
_DURABLE_UNTRUSTED_CONTAINERS = frozenset(
    {
        "arguments",
        "dimensions",
        "effective_arguments",
        "environment_factory_allocation_owner",
        "environment_factory_reconnect",
        "input_schema",
        "json_schema",
        "metadata",
        "options",
        "output",
        "payload",
        "provider_options",
        "request_metadata",
        "structured",
        "task_error_payload",
        "terminal_payload",
        "turn_completed_payload",
    }
)
# ``records`` is a map from caller-generated operation IDs to one typed runtime
# record. Its dynamic key is untrusted, but the record below it resumes the
# runtime schema. Other dynamic maps, notably environment reconnect metadata,
# contain arbitrary extension data and must remain untrusted at every depth.
_DURABLE_SINGLE_LEVEL_TYPED_MAPS = frozenset(
    {
        "durable_subagent_submission_seeds",
        "durable_subagent_submissions",
        "records",
        "workspace_observations",
    }
)
_QUARANTINED_ASSISTANT_MESSAGE_KEYS = frozenset({"role", "content"})
_QUARANTINED_ASSISTANT_MESSAGE_PART_KEYS = frozenset(
    {
        "type",
        "text",
        "tool_call_id",
        "tool_name",
        "arguments",
        "tool_round_id",
        "model_step_id",
        "model_attempt_id",
        "provider",
        "state",
        "provider_state",
    }
)
_QUARANTINED_ASSISTANT_MESSAGE_UNTRUSTED_CONTAINERS = frozenset(
    {"arguments", "provider_state", "state"}
)
_DURABLE_ROOT_STRUCTURE_KEYS = frozenset(
    {
        ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
        INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
        INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
        SETTLED_INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
        CHECKPOINT_SCHEMA_VERSION_KEY,
        AUTOMATIC_RECALL_CHECKPOINT_KEY,
        COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY,
        "context_compaction",
        "durable_subagent_submission_seeds",
        "durable_subagent_submissions",
        "environment_factory_allocation_owner",
        "environment_factory_reconnect",
        "incomplete_session_recovery_claim",
        "runtime_authored_user_message",
        "pending_interruption_cascade",
        "pending_session_interrupt",
        "approval_resolution_intent",
        "user_input_resolution_intent",
        "pending_tool_approval",
        "pending_tool_round",
        "pending_user_input",
        "session_operations",
        "usage_triggered_context",
        "workspace_observations",
    }
)
_ACTIVE_INVOCATION_PROFILE_ROOT_IDENTITY_PATHS = frozenset(
    {
        (ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY, "record_type"),
        (ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY, "session_id"),
        (ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY, "interaction_id"),
    }
)
_ACTIVE_INVOCATION_PROFILE_COMPONENT_IDENTITY_FIELDS = frozenset(
    {
        "component_class",
        "strength",
        "availability",
    }
)
_RUNTIME_BUILD_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "recipe",
        "origin",
        "availability",
        "strength",
        "fingerprint",
        "artifact_kind",
        "artifact_digest",
        "source_revision",
        "detail_code",
    }
)
_WORKSPACE_OBSERVATION_IDENTITY_FIELDS = frozenset(
    {
        "record_type",
        "session_id",
        "interaction_id",
        "window_id",
        "binding_generation_id",
        "workspace_id",
        "observer",
        "artifact_store_id",
        "agent_name",
        "environment_name",
        "tool_name",
        "tool_call_id",
        "model_step_id",
        "model_attempt_id",
        "tool_round_id",
        "before_observation_id",
        "tool_outcome_event_id",
        "after_observation_id",
        "mutation_event_id",
    }
)
_PENDING_TOOL_ROUND_EXECUTION_IDENTITY_FIELDS = frozenset(
    {
        "interaction_id",
        "model_attempt_id",
        "model_step_id",
        "source_model_step_id",
        "tool_round_id",
    }
)
_FORK_RUNTIME_SESSION_STATUS_PATHS = frozenset(
    {
        (
            INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
            "receipts",
            "result_session",
            "metadata",
            "cayu:fork_execution_profile",
            "source_status",
        ),
        (
            INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
            "receipts",
            "result_session",
            "metadata",
            "status",
        ),
    }
)
_FORK_RUNTIME_SESSION_STATUS_VALUES = frozenset(
    {"pending", "running", "interrupting", "completed", "failed", "interrupted"}
)
_COMPLETION_RESULT_EVENT_PUBLICATION_ID_PREFIX = "completion-result-publication:v1:"
_COMPLETION_RESULT_EVENT_PUBLICATION_OWNER_ID_PREFIX = "completion-result-owner:v1:"
_INVOCATION_TERMINAL_DECISION_ROOTS = frozenset(
    {
        INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
        SETTLED_INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
    }
)
_INVOCATION_TERMINAL_DECISION_FIELDS = frozenset(
    {
        "record_type",
        "schema_version",
        "decision_id",
        "outcome",
        "session_id",
        "session_instance_id",
        "run_epoch",
        "profile_interaction_id",
        "interaction_id",
        "execution_profile_fingerprint",
        "interaction_event_id",
        "predecessor_interaction_event_id",
        "terminal_event_id",
        "observed_at",
        "terminal_payload",
        "interruption_request_id",
        "task_id",
        "runtime_task_failure_id",
        "task_terminalization_request_sha256",
        "task_error_payload",
        "turn_completed_payload",
        "record_digest",
    }
)
_INVOCATION_TERMINAL_DECISION_UNTRUSTED_FIELDS = frozenset(
    {"terminal_payload", "task_error_payload", "turn_completed_payload"}
)


def require_secret_free_durable_object(
    value: dict[str, Any],
    *,
    redactor: SecretRedactor,
    field_name: str,
    schema_root: str | None = None,
) -> dict[str, Any]:
    """Copy one checkpoint payload and reject secrets in values or data-owned keys."""

    if schema_root is not None and schema_root not in _DURABLE_ROOT_STRUCTURE_KEYS:
        raise ValueError("schema_root must identify a runtime-owned checkpoint root.")
    copied = copy_json_value(value, field_name)
    if type(copied) is not dict:
        raise AssertionError(f"{field_name} copy returned a non-object.")
    if durable_value_contains_secret(
        copied,
        redactor=redactor,
        path=(() if schema_root is None else (schema_root,)),
    ):
        copied.clear()
        raise ValueError(
            f"{field_name} contains a workload secret; refusing a durable checkpoint "
            "that could change later tool execution."
        )
    return copied


def durable_value_contains_secret(
    value: Any,
    *,
    redactor: SecretRedactor,
    path: tuple[str, ...] = (),
    _trusted_web_control_paths: frozenset[tuple[str, ...]] = frozenset(),
    _trusted_targeted_tool_references: frozenset[tuple[tuple[str, ...], str]] | None = None,
    _trusted_lifecycle_receipt_values: frozenset[tuple[tuple[str, ...], str]] | None = None,
    _trusted_lifecycle_receipt_keys: frozenset[tuple[tuple[str, ...], str]] | None = None,
    _trusted_terminal_decision_values: frozenset[tuple[tuple[str, ...], str]] | None = None,
) -> bool:
    """Return whether a checkpoint tree contains secret text outside schema-owned keys."""

    if _trusted_targeted_tool_references is None:
        _trusted_targeted_tool_references = _targeted_tool_reference_authority(value, path=path)
    if _trusted_lifecycle_receipt_values is None or _trusted_lifecycle_receipt_keys is None:
        (
            _trusted_lifecycle_receipt_values,
            _trusted_lifecycle_receipt_keys,
        ) = _invocation_lifecycle_receipt_metadata_authority(value, path=path)
    if _trusted_terminal_decision_values is None:
        _trusted_terminal_decision_values = _invocation_terminal_decision_authority(
            value,
            path=path,
        )
    if type(value) is str:
        if (
            (path, value) in _trusted_targeted_tool_references
            or (
                path,
                value,
            )
            in _trusted_lifecycle_receipt_values
            or (
                path,
                value,
            )
            in _trusted_terminal_decision_values
        ):
            return False
        if path in _trusted_web_control_paths:
            # Exact closed controls are runtime protocol, not copied workload
            # text. The complete persisted attestation was validated before
            # this path was admitted; neighboring data fields remain subject
            # to the workload-secret boundary.
            return False
        if (
            _is_active_invocation_profile_identity_path(path)
            or _is_automatic_recall_evidence_identity_path(path)
            or _is_completion_result_event_publication_identity(path, value)
            or _is_workspace_observation_identity_path(path)
            or _is_pending_tool_round_execution_identity_path(path)
            or _is_tool_exposure_authority_identity_path(path)
        ):
            # These checkpoint roots are runtime-owned typed authority. Active
            # profiles cross their dedicated admission boundary; workspace
            # observations and result-publication reservations are admitted
            # only through their runtime-owned state machines. Configured
            # dynamic identities are already field-scoped keyed aliases;
            # static identities were checked, and exact runtime identities
            # retain structural provenance.
            return False
        if (
            path in _FORK_RUNTIME_SESSION_STATUS_PATHS
            and value in _FORK_RUNTIME_SESSION_STATUS_VALUES
        ):
            # Fork preparation injects these reserved metadata records only after
            # public metadata redaction. A validated lifecycle ledger authenticates
            # the exact duplicated result session, and these fields are closed
            # SessionStatus values. Neighboring fork-authority fields remain subject
            # to workload-secret rejection.
            return False
        if path and path[-1] in _DURABLE_ENUM_STRING_FIELDS and _path_has_typed_schema(path[:-1]):
            # Typed model validation owns these finite protocol values. A
            # credential that happens to equal "deny", "planned", or another
            # enum literal must not erase or reject the protocol decision.
            return False
        if (
            path
            and (
                path[-1] in _DURABLE_SHA256_STRING_FIELDS or _is_durable_subagent_sha256_path(path)
            )
            and _path_has_typed_schema(path[:-1])
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        ):
            # This runtime-generated commitment contains no copied secret text.
            # The owning typed model validates the digest again before it can
            # become authority, so coincidental substring matches are safe.
            return False
        return redactor.redact_text(value) != value
    if value is None or type(value) in {bool, int, float}:
        return False
    if type(value) is list:
        return any(
            durable_value_contains_secret(
                item,
                redactor=redactor,
                path=path,
                _trusted_web_control_paths=_trusted_web_control_paths,
                _trusted_targeted_tool_references=_trusted_targeted_tool_references,
                _trusted_lifecycle_receipt_values=_trusted_lifecycle_receipt_values,
                _trusted_lifecycle_receipt_keys=_trusted_lifecycle_receipt_keys,
                _trusted_terminal_decision_values=_trusted_terminal_decision_values,
            )
            for item in value
        )
    if type(value) is dict:
        trusted_web_control_paths = _trusted_web_control_paths
        if _is_staged_terminal_event(path):
            relative_paths = persisted_web_access_control_paths(
                value
            ) | persisted_shared_artifact_control_paths(value)
            if relative_paths:
                trusted_web_control_paths = frozenset(
                    (*path, *relative_path) for relative_path in relative_paths
                )
        if path and path[-1] == "json_schema" and _path_has_typed_schema(path[:-1]):
            return json_schema_contains_secret(
                value,
                redactor=redactor,
                field_name="checkpoint.json_schema",
            )
        fixed_tool_exposure_reason = (
            _is_staged_terminal_event_payload(path)
            and value.get("blocked_by") == "tool_exposure"
            and value.get("reason") == "not_exposed_in_request"
        )
        for key, item in value.items():
            structural_key = (
                (
                    key in (_DURABLE_ROOT_STRUCTURE_KEYS if not path else _DURABLE_STRUCTURE_KEYS)
                    or _is_durable_subagent_structural_key(path, key)
                    or _is_completion_result_event_publication_structural_key(path, key)
                    or _is_invocation_lifecycle_receipt_structural_key(path, key)
                    or _is_invocation_terminal_decision_structural_key(path, key)
                    or _is_active_invocation_build_provenance_structural_key(path, key)
                    or (path, key) in _trusted_lifecycle_receipt_keys
                    or _is_quarantined_assistant_message_structural_key(path, key)
                    or _is_staged_terminal_event_payload(path)
                )
                and _path_has_typed_schema(path)
                and (not path or path[-1] not in _DURABLE_SINGLE_LEVEL_TYPED_MAPS)
            )
            if not structural_key:
                try:
                    redactor.require_no_secret_keys(
                        {key: None},
                        field_name="checkpoint",
                        match_short_substrings=True,
                    )
                except ValueError:
                    return True
            if fixed_tool_exposure_reason and key == "reason":
                continue
            if durable_value_contains_secret(
                item,
                redactor=redactor,
                path=(*path, key),
                _trusted_web_control_paths=trusted_web_control_paths,
                _trusted_targeted_tool_references=_trusted_targeted_tool_references,
                _trusted_lifecycle_receipt_values=_trusted_lifecycle_receipt_values,
                _trusted_lifecycle_receipt_keys=_trusted_lifecycle_receipt_keys,
                _trusted_terminal_decision_values=_trusted_terminal_decision_values,
            ):
                return True
        return False
    raise AssertionError("Durable checkpoint contains non-JSON-compatible data.")


def _targeted_tool_reference_authority(
    value: Any,
    *,
    path: tuple[str, ...],
) -> frozenset[tuple[tuple[str, ...], str]]:
    """Recognize exact runtime-selected gateway references in typed checkpoints."""

    candidates: list[tuple[str, dict[str, Any]]] = []
    if not path and type(value) is dict:
        for root in ("pending_tool_round", "pending_tool_approval", "pending_user_input"):
            candidate = value.get(root)
            if type(candidate) is dict:
                candidates.append((root, candidate))
    elif (
        len(path) == 1
        and path[0]
        in {
            "pending_tool_round",
            "pending_tool_approval",
            "pending_user_input",
        }
        and type(value) is dict
    ):
        candidates.append((path[0], value))

    trusted: set[tuple[tuple[str, ...], str]] = set()
    for root, candidate in candidates:
        calls = candidate.get("tool_calls")
        if type(calls) is not list:
            continue
        for call in calls:
            if type(call) is not dict:
                continue
            grant_id = call.get("targeted_tool_grant_id")
            if type(grant_id) is not str:
                continue
            try:
                validate_targeted_tool_digest(grant_id, "targeted_tool_grant_id")
            except ValueError:
                continue
            if (
                call.get("tool_name") != CALL_TOOL_NAME
                and call.get("model_tool_name") != CALL_TOOL_NAME
            ):
                continue
            trusted.add(((root, "tool_calls", "targeted_tool_grant_id"), grant_id))
            arguments = call.get("arguments")
            if type(arguments) is dict:
                tool_ref = arguments.get("tool_ref")
                if type(tool_ref) is str:
                    trusted.add(((root, "tool_calls", "arguments", "tool_ref"), tool_ref))
            invocation = call.get("targeted_tool_invocation")
            if type(invocation) is not dict:
                continue
            try:
                resolved = ResolvedTargetedToolInvocation.model_validate(invocation)
            except ValueError:
                continue
            if resolved.grant_id != grant_id:
                continue
            for field_name in (
                "arguments_sha256",
                "catalogue_revision",
                "descriptor_version",
                "grant_id",
                "invocation_id",
                "schema_fingerprint",
                "tool_ref",
                "use_id",
            ):
                trusted.add(
                    (
                        (root, "tool_calls", "targeted_tool_invocation", field_name),
                        getattr(resolved, field_name),
                    )
                )
    return frozenset(trusted)


def _invocation_terminal_decision_authority(
    value: Any,
    *,
    path: tuple[str, ...],
) -> frozenset[tuple[tuple[str, ...], str]]:
    """Authenticate top-level controls and identities of a complete decision."""

    candidates: list[tuple[str, dict[str, Any]]] = []
    if not path and type(value) is dict:
        for root in _INVOCATION_TERMINAL_DECISION_ROOTS:
            candidate = value.get(root)
            if type(candidate) is dict:
                candidates.append((root, candidate))
    elif len(path) == 1 and path[0] in _INVOCATION_TERMINAL_DECISION_ROOTS and type(value) is dict:
        candidates.append((path[0], value))
    if not candidates:
        return frozenset()

    from cayu.runtime._invocation_terminal_decision import InvocationTerminalDecision

    trusted: set[tuple[tuple[str, ...], str]] = set()
    for root, candidate in candidates:
        try:
            decision = InvocationTerminalDecision.model_validate(candidate)
        except (TypeError, ValueError):
            continue
        projected = decision.model_dump(mode="json")
        for field_name, field_value in projected.items():
            if (
                field_name not in _INVOCATION_TERMINAL_DECISION_UNTRUSTED_FIELDS
                and type(field_value) is str
            ):
                trusted.add(((root, field_name), field_value))
    return frozenset(trusted)


def _invocation_lifecycle_receipt_metadata_authority(
    value: Any,
    *,
    path: tuple[str, ...],
) -> tuple[
    frozenset[tuple[tuple[str, ...], str]],
    frozenset[tuple[tuple[str, ...], str]],
]:
    """Authenticate duplicated runtime metadata inside command receipts."""

    if path or type(value) is not dict:
        return frozenset(), frozenset()
    raw_ledger = value.get(INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY)
    if type(raw_ledger) is not dict or raw_ledger.get("record_type") != (
        INVOCATION_LIFECYCLE_RECEIPT_LEDGER_RECORD_TYPE
    ):
        return frozenset(), frozenset()
    try:
        from cayu.runtime._invocation_lifecycle import (
            _invocation_lifecycle_receipt_ledger_from_checkpoint,
        )

        ledger = _invocation_lifecycle_receipt_ledger_from_checkpoint(
            {INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY: raw_ledger}
        )
    except (ImportError, RuntimeError, TypeError, ValueError):
        return frozenset(), frozenset()

    trusted_values: set[tuple[tuple[str, ...], str]] = set()
    trusted_keys: set[tuple[tuple[str, ...], str]] = set()

    def retain_projection(projected: Any, projected_path: tuple[str, ...]) -> None:
        if type(projected) is str:
            trusted_values.add((projected_path, projected))
            return
        if type(projected) is list:
            for item in projected:
                retain_projection(item, projected_path)
            return
        if type(projected) is dict:
            for key, item in projected.items():
                trusted_keys.add((projected_path, key))
                retain_projection(item, (*projected_path, key))

    def retain_authenticated_receipt_identities(
        projected: Any,
        projected_path: tuple[str, ...],
    ) -> None:
        if type(projected) is str:
            if _is_invocation_lifecycle_receipt_identity_path(projected_path):
                trusted_values.add((projected_path, projected))
            return
        if type(projected) is list:
            for item in projected:
                retain_authenticated_receipt_identities(item, projected_path)
            return
        if type(projected) is dict:
            for key, item in projected.items():
                item_path = (*projected_path, key)
                if _is_invocation_lifecycle_receipt_identity_path(item_path):
                    trusted_keys.add((projected_path, key))
                retain_authenticated_receipt_identities(item, item_path)

    # Identity-shaped text is trusted only after the complete ledger, every
    # receipt, and their content digests have authenticated it. A malformed
    # ledger therefore contributes no path exemptions at all.
    retain_authenticated_receipt_identities(
        ledger.model_dump(mode="json"),
        (INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,),
    )

    for receipt in ledger.receipts:
        raw_metadata = receipt.result_session.metadata
        profile_path = (
            INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
            "receipts",
            "result_session",
            "metadata",
        )
        raw_provenance = raw_metadata.get(RUNTIME_BUILD_PROVENANCE_METADATA_KEY)
        if raw_provenance is not None:
            try:
                provenance = runtime_build_provenance_from_session_metadata(raw_metadata)
            except (TypeError, ValueError):
                return frozenset(), frozenset()
            trusted_keys.add((profile_path, RUNTIME_BUILD_PROVENANCE_METADATA_KEY))
            retain_projection(
                provenance.model_dump(mode="json"),
                (*profile_path, RUNTIME_BUILD_PROVENANCE_METADATA_KEY),
            )
        if EXECUTION_PROFILE_METADATA_KEY not in raw_metadata:
            continue
        raw_profile_record = raw_metadata[EXECUTION_PROFILE_METADATA_KEY]
        if type(raw_profile_record) is not dict:
            return frozenset(), frozenset()
        try:
            if set(raw_profile_record) != {"record_type", "schema_version", "baseline", "expected"}:
                return frozenset(), frozenset()
            execution_profile_from_session_metadata(raw_metadata)
            baseline = ExecutionProfileIdentity.model_validate(raw_profile_record["baseline"])
            expected = ExecutionProfileIdentity.model_validate(raw_profile_record["expected"])
        except (TypeError, ValueError):
            return frozenset(), frozenset()
        trusted_keys.add((profile_path, EXECUTION_PROFILE_METADATA_KEY))
        record_path = (*profile_path, EXECUTION_PROFILE_METADATA_KEY)
        for key in ("record_type", "schema_version", "baseline", "expected"):
            trusted_keys.add((record_path, key))
        retain_projection(raw_profile_record["record_type"], (*record_path, "record_type"))
        retain_projection(baseline.model_dump(mode="json"), (*record_path, "baseline"))
        retain_projection(expected.model_dump(mode="json"), (*record_path, "expected"))

        raw_ceiling = raw_metadata.get(TOOL_CAPABILITY_CEILING_METADATA_KEY)
        if raw_ceiling is not None:
            try:
                tool_capability_ceiling_from_session_metadata(raw_metadata)
            except (TypeError, ValueError):
                return frozenset(), frozenset()
            # The key is runtime-owned and its value was fingerprint-validated.
            # Tool names remain subject to redaction: unlike the profile's closed
            # enum identities, they can carry caller-selected text.
            trusted_keys.add((profile_path, TOOL_CAPABILITY_CEILING_METADATA_KEY))
    return frozenset(trusted_values), frozenset(trusted_keys)


def _is_active_invocation_profile_identity_path(path: tuple[str, ...]) -> bool:
    if path in _ACTIVE_INVOCATION_PROFILE_ROOT_IDENTITY_PATHS:
        return True
    if (
        len(path) == 4
        and path[:3]
        == (
            ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
            "profile",
            "runtime_build_provenance",
        )
        and path[-1] in _RUNTIME_BUILD_PROVENANCE_FIELDS
    ):
        return True
    return (
        len(path) == 4
        and path[:3]
        == (
            ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
            "profile",
            "components",
        )
        and path[-1] in _ACTIVE_INVOCATION_PROFILE_COMPONENT_IDENTITY_FIELDS
    )


def _is_active_invocation_build_provenance_structural_key(
    path: tuple[str, ...],
    key: str,
) -> bool:
    if path == (ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY, "profile"):
        return key == "runtime_build_provenance"
    return (
        path
        == (
            ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
            "profile",
            "runtime_build_provenance",
        )
        and key in _RUNTIME_BUILD_PROVENANCE_FIELDS
    )


def _is_invocation_lifecycle_receipt_identity_path(path: tuple[str, ...]) -> bool:
    if len(path) == 2 and path[0] == INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY:
        return path[1] in {
            "record_type",
            "release_capacity_command_identity",
            "record_sha256",
        }
    if len(path) == 3 and path[:2] == (
        INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
        "receipts",
    ):
        return path[2] in {
            "record_type",
            "kind",
            "command_identity",
            "command_sha256",
            "session_id",
            "session_instance_id",
            "record_sha256",
        }
    if len(path) == 4 and path[:3] == (
        INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
        "receipts",
        "result_session",
    ):
        return path[3] in {
            "id",
            "instance_id",
            "agent_name",
            "environment_name",
            "workflow_name",
            "provider_name",
            "model",
            "parent_session_id",
            "causal_budget_id",
            "runtime_name",
            "runtime_version",
            "status",
            "created_at",
            "updated_at",
            "last_activity_at",
        }
    if len(path) == 5 and path[:4] == (
        INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
        "receipts",
        "result_session",
        "invocation",
    ):
        return path[4] in {
            "root_invocation_id",
            "root_session_id",
            "source",
        }
    if len(path) == 5 and path[:4] == (
        INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
        "receipts",
        "active_profile",
        "profile",
    ):
        return path[4] == "runtime_build_provenance"
    return (
        len(path) >= 4
        and path[:3]
        == (
            INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
            "receipts",
            "active_profile",
        )
        and _is_active_invocation_profile_identity_path(
            (ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY, *path[3:])
        )
    )


def _is_invocation_lifecycle_receipt_structural_key(
    path: tuple[str, ...],
    key: str,
) -> bool:
    if path == (INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,):
        return key in {
            "record_type",
            "schema_version",
            "receipts",
            "release_capacity_command_identity",
            "record_sha256",
        }
    return path == (
        INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
        "receipts",
    ) and key in {
        "record_type",
        "schema_version",
        "kind",
        "command_identity",
        "command_sha256",
        "session_id",
        "session_instance_id",
        "result_session",
        "active_profile",
        "record_sha256",
    }


def _is_invocation_terminal_decision_structural_key(
    path: tuple[str, ...],
    key: str,
) -> bool:
    return (
        len(path) == 1
        and path[0] in _INVOCATION_TERMINAL_DECISION_ROOTS
        and key in _INVOCATION_TERMINAL_DECISION_FIELDS
    )


def _is_workspace_observation_identity_path(path: tuple[str, ...]) -> bool:
    return (
        len(path) == 3
        and path[0] == "workspace_observations"
        and path[2] in _WORKSPACE_OBSERVATION_IDENTITY_FIELDS
    ) or (
        len(path) == 4
        and path[0] == "workspace_observations"
        and path[2] == "artifacts"
        and path[3] == "artifact_id"
    )


def _is_pending_tool_round_execution_identity_path(path: tuple[str, ...]) -> bool:
    """Return whether a typed pending round owns this runtime execution ID."""

    return (
        len(path) >= 2
        and path[0] == "pending_tool_round"
        and path[-1] in _PENDING_TOOL_ROUND_EXECUTION_IDENTITY_FIELDS
        and _path_has_typed_schema(path[:-1])
    )


def _is_tool_exposure_authority_identity_path(path: tuple[str, ...]) -> bool:
    """Recognize exact typed exposure fields that must survive secret collisions."""

    return (
        len(path) == 3
        and path[0] in {"pending_tool_round", "pending_user_input"}
        and path[1] == "tool_exposure"
        and path[2] in {"catalogue_revision", "profile_id", "tool_names"}
        and _path_has_typed_schema(path[:-1])
    )


def _path_has_typed_schema(path: tuple[str, ...]) -> bool:
    """Return whether `path` remains inside a known runtime-owned checkpoint shape."""

    if _is_completion_result_event_publication_schema_path(path):
        return True
    if path and path[0] == INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY:
        return True
    if path and path[0] in _INVOCATION_TERMINAL_DECISION_ROOTS:
        return len(path) == 1 or (
            len(path) == 2
            and path[1] in _INVOCATION_TERMINAL_DECISION_FIELDS
            and path[1] not in _INVOCATION_TERMINAL_DECISION_UNTRUSTED_FIELDS
        )
    if path and path[0] not in _DURABLE_ROOT_STRUCTURE_KEYS:
        return False
    for index, part in enumerate(path):
        if index == 0 and part in _DURABLE_SUBAGENT_ROOTS:
            continue
        if (
            part in _DURABLE_UNTRUSTED_CONTAINERS
            and not (part == "payload" and _is_staged_terminal_event_payload(path[: index + 1]))
        ) or (
            _is_quarantined_assistant_message_part_key(path[:index], part)
            and part in _QUARANTINED_ASSISTANT_MESSAGE_UNTRUSTED_CONTAINERS
        ):
            return False
        if part in _DURABLE_STRUCTURE_KEYS:
            continue
        if _is_durable_subagent_structural_key(path[:index], part):
            continue
        if _is_quarantined_assistant_message_structural_key(path[:index], part):
            continue
        if index > 0 and path[index - 1] in _DURABLE_SINGLE_LEVEL_TYPED_MAPS:
            continue
        return False
    return True


def _is_completion_result_event_publication_id(value: str) -> bool:
    return (
        len(value) == len(_COMPLETION_RESULT_EVENT_PUBLICATION_ID_PREFIX) + 64
        and value.startswith(_COMPLETION_RESULT_EVENT_PUBLICATION_ID_PREFIX)
        and all(
            character in "0123456789abcdef"
            for character in value.removeprefix(_COMPLETION_RESULT_EVENT_PUBLICATION_ID_PREFIX)
        )
    )


def _is_completion_result_event_publication_schema_path(path: tuple[str, ...]) -> bool:
    if path in {
        (COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY,),
        (COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY, "reservations"),
    }:
        return True
    if (
        len(path) == 3
        and path[:2] == (COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY, "reservations")
        and _is_completion_result_event_publication_id(path[2])
    ):
        return True
    if (
        len(path) == 4
        and path[:2] == (COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY, "reservations")
        and _is_completion_result_event_publication_id(path[2])
        and path[3] == "owners"
    ):
        return True
    return (
        len(path) == 5
        and path[:2] == (COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY, "reservations")
        and _is_completion_result_event_publication_id(path[2])
        and path[3] == "owners"
        and _is_completion_result_event_publication_owner_id(path[4])
    )


def _is_completion_result_event_publication_owner_id(value: str) -> bool:
    return (
        len(value) == len(_COMPLETION_RESULT_EVENT_PUBLICATION_OWNER_ID_PREFIX) + 64
        and value.startswith(_COMPLETION_RESULT_EVENT_PUBLICATION_OWNER_ID_PREFIX)
        and all(
            character in "0123456789abcdef"
            for character in value.removeprefix(
                _COMPLETION_RESULT_EVENT_PUBLICATION_OWNER_ID_PREFIX
            )
        )
    )


def _is_completion_result_event_publication_structural_key(
    path: tuple[str, ...],
    key: str,
) -> bool:
    if path == (COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY,):
        return key in {"schema_version", "reservations"}
    if path == (
        COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY,
        "reservations",
    ):
        return _is_completion_result_event_publication_id(key)
    return (
        (
            len(path) == 3
            and path[:2] == (COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY, "reservations")
            and _is_completion_result_event_publication_id(path[2])
            and key in {"schema_version", "publication_id", "authority_sha256", "owners"}
        )
        or (
            len(path) == 4
            and path[:2] == (COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY, "reservations")
            and _is_completion_result_event_publication_id(path[2])
            and path[3] == "owners"
            and _is_completion_result_event_publication_owner_id(key)
        )
        or (
            len(path) == 5
            and path[:2] == (COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY, "reservations")
            and _is_completion_result_event_publication_id(path[2])
            and path[3] == "owners"
            and _is_completion_result_event_publication_owner_id(path[4])
            and key in {"schema_version", "owner_id", "expires_at"}
        )
    )


def _is_completion_result_event_publication_identity(
    path: tuple[str, ...],
    value: str,
) -> bool:
    if (
        len(path) == 4
        and path[:2] == (COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY, "reservations")
        and _is_completion_result_event_publication_id(path[2])
        and path[3] == "publication_id"
    ):
        return value == path[2]
    if (
        len(path) == 4
        and path[:2] == (COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY, "reservations")
        and _is_completion_result_event_publication_id(path[2])
        and path[3] == "authority_sha256"
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
        and path[2] == f"{_COMPLETION_RESULT_EVENT_PUBLICATION_ID_PREFIX}{value}"
    ):
        return True
    if (
        len(path) == 6
        and path[:2] == (COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY, "reservations")
        and _is_completion_result_event_publication_id(path[2])
        and path[3] == "owners"
        and _is_completion_result_event_publication_owner_id(path[4])
        and path[5] == "expires_at"
    ):
        try:
            expiry = datetime.fromisoformat(value)
        except ValueError:
            return False
        return (
            expiry.tzinfo is not None
            and expiry.utcoffset() is not None
            and expiry.astimezone(UTC).isoformat() == value
        )
    return (
        len(path) == 6
        and path[:2] == (COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY, "reservations")
        and _is_completion_result_event_publication_id(path[2])
        and path[3] == "owners"
        and _is_completion_result_event_publication_owner_id(path[4])
        and path[5] == "owner_id"
        and value == path[4]
    )


def _is_automatic_recall_evidence_identity_path(path: tuple[str, ...]) -> bool:
    return path == (AUTOMATIC_RECALL_CHECKPOINT_KEY, "receipt_id")


def _is_durable_subagent_structural_key(path: tuple[str, ...], key: str) -> bool:
    return (
        bool(path)
        and path[0] in _DURABLE_SUBAGENT_ROOTS
        and (
            key in _DURABLE_SUBAGENT_STRUCTURE_KEYS or key in _DURABLE_SUBAGENT_SHA256_STRING_FIELDS
        )
    )


def _is_durable_subagent_sha256_path(path: tuple[str, ...]) -> bool:
    return (
        len(path) >= 3
        and path[0] in _DURABLE_SUBAGENT_ROOTS
        and path[-1] in _DURABLE_SUBAGENT_SHA256_STRING_FIELDS
    )


def _is_staged_terminal_event_payload(path: tuple[str, ...]) -> bool:
    return (
        len(path) >= 4
        and path[-4]
        in {
            "pending_tool_round",
            "pending_user_input",
        }
        and path[-3:]
        == (
            "staged_terminals",
            "event",
            "payload",
        )
    )


def _is_staged_terminal_event(path: tuple[str, ...]) -> bool:
    return (
        len(path) == 3
        and path[0] in {"pending_tool_round", "pending_user_input"}
        and path[1:] == ("staged_terminals", "event")
    )


def _is_quarantined_assistant_message_structural_key(
    path: tuple[str, ...],
    key: str,
) -> bool:
    if path and (
        path[-1] == "quarantined_assistant_message"
        or (len(path) >= 2 and path[-2:] == ("assistant_publication", "message"))
    ):
        return key in _QUARANTINED_ASSISTANT_MESSAGE_KEYS
    return _is_quarantined_assistant_message_part_key(path, key)


def _is_quarantined_assistant_message_part_key(
    path: tuple[str, ...],
    key: str,
) -> bool:
    return (
        len(path) >= 2
        and (
            path[-2:] == ("quarantined_assistant_message", "content")
            or (len(path) >= 3 and path[-3:] == ("assistant_publication", "message", "content"))
        )
        and key in _QUARANTINED_ASSISTANT_MESSAGE_PART_KEYS
    )
