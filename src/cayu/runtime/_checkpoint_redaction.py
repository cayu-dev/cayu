from __future__ import annotations

from typing import Any

from cayu._validation import copy_json_value
from cayu.runtime._web_access_results import persisted_web_access_control_paths
from cayu.runtime.checkpoints import (
    ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
    CHECKPOINT_SCHEMA_VERSION_KEY,
)
from cayu.runtime.structured_output import json_schema_contains_secret
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
        "phase",
        "role",
        "round_id",
        "secret_resolution_scope",
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
        "component_class",
        "strength",
        "availability",
    }
)
_DURABLE_ENUM_STRING_FIELDS = frozenset(
    {
        "assistant_message_state",
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
        "phase",
        "before_state",
        "after_state",
        "delta_state",
        "observer_authority",
    }
)
_DURABLE_SHA256_STRING_FIELDS = frozenset(
    {
        "resolution_request_digest",
        "user_message_sha256",
        "fingerprint",
        "execution_profile_fingerprint",
        "exposure_fingerprint",
        "sha256",
        "mutation_event_digest",
        "tool_outcome_event_digest",
    }
)
_DURABLE_STRUCTURE_KEYS = (_DURABLE_STRUCTURE_STRING_FIELDS | _DURABLE_SHA256_STRING_FIELDS) | {
    "approval_close_intent",
    ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
    "components",
    "assistant_publication",
    "approval_resolution_intent",
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
    "knowledge_injection",
    "runtime_authored_user_message",
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
        CHECKPOINT_SCHEMA_VERSION_KEY,
        "context_compaction",
        "durable_subagent_submission_seeds",
        "durable_subagent_submissions",
        "environment_factory_allocation_owner",
        "environment_factory_reconnect",
        "incomplete_session_recovery_claim",
        "knowledge_injection",
        "runtime_authored_user_message",
        "pending_interruption_cascade",
        "pending_session_interrupt",
        "approval_resolution_intent",
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
        "model_attempt_id",
        "model_step_id",
        "source_model_step_id",
        "tool_round_id",
    }
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
) -> bool:
    """Return whether a checkpoint tree contains secret text outside schema-owned keys."""

    if type(value) is str:
        if path in _trusted_web_control_paths:
            # Exact closed controls are runtime protocol, not copied workload
            # text. The complete persisted attestation was validated before
            # this path was admitted; neighboring data fields remain subject
            # to the workload-secret boundary.
            return False
        if (
            _is_active_invocation_profile_identity_path(path)
            or _is_workspace_observation_identity_path(path)
            or _is_pending_tool_round_execution_identity_path(path)
            or _is_tool_exposure_authority_identity_path(path)
        ):
            # Both checkpoint roots are runtime-owned typed authority. Active
            # profiles cross their dedicated admission boundary; workspace
            # observation roots can be created only with a matching private
            # intent admission. Configured dynamic identities are already
            # field-scoped keyed aliases; static identities were checked, and
            # exact runtime observer identities retain structural provenance.
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
            )
            for item in value
        )
    if type(value) is dict:
        trusted_web_control_paths = _trusted_web_control_paths
        if _is_staged_terminal_event(path):
            relative_paths = persisted_web_access_control_paths(value)
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
            ):
                return True
        return False
    raise AssertionError("Durable checkpoint contains non-JSON-compatible data.")


def _is_active_invocation_profile_identity_path(path: tuple[str, ...]) -> bool:
    if path in _ACTIVE_INVOCATION_PROFILE_ROOT_IDENTITY_PATHS:
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
        and path[2] in {"profile_id", "tool_names"}
        and _path_has_typed_schema(path[:-1])
    )


def _path_has_typed_schema(path: tuple[str, ...]) -> bool:
    """Return whether `path` remains inside a known runtime-owned checkpoint shape."""

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
