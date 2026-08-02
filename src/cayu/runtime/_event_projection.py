from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from datetime import datetime
from hmac import compare_digest
from typing import Any, cast

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    collision_safe_json_object,
    copy_durable_json_value,
)
from cayu.core.events import (
    Event,
    EventType,
    copy_event,
    event_envelope_authority_is_runtime_generated,
    event_id_is_runtime_generated,
    event_nested_payload_authority_is_runtime_generated,
    event_payload_authority_is_runtime_generated,
)
from cayu.core.tools import (
    _COMMAND_POLICY_DENIAL_SOURCE,
    _POLICY_DENIAL_TRUNCATION_MARKER,
    _TOOL_POLICY_DENIAL_SOURCE,
    ToolEffect,
)
from cayu.core.workflows import WORKFLOW_ATTEMPT_EVENT_TYPE
from cayu.providers.base import ModelFinishReason
from cayu.runtime import _tool_results as tool_results
from cayu.runtime._tool_identity import tool_idempotency_key
from cayu.runtime.model_steps import StepClassificationType
from cayu.runtime.public_authority import (
    PUBLIC_AUTHORITY_ALIAS_PREFIX,
    PublicAuthorityAliasCodec,
    parse_public_authority_alias,
)
from cayu.runtime.tool_result_projection import (
    _TOOL_RESULT_PROJECTION_PROVENANCE_PATH,
    reestimate_tool_result_projection_tokens,
)
from cayu.vaults.redaction import SecretRedactor

PUBLIC_EVENT_ID_PREFIX = "cayu_event_"
PUBLIC_EVENT_LINKAGE_SEPARATOR = ":"
PUBLIC_EVENT_ENVELOPE_ALIAS_PREFIX = PUBLIC_AUTHORITY_ALIAS_PREFIX
REDACTED_CUSTOM_EVENT_TYPE = "custom.redacted"
PRIVATE_EVENT_AUTHORITY = "[PRIVATE_EVENT_AUTHORITY]"
_ENVELOPE_ALIAS_FIELD_BY_NESTED_PATH: Mapping[tuple[str, ...], str] = {
    ("interaction_ids", "*"): "interaction_id",
}


@dataclass(frozen=True, slots=True)
class EventPayloadPolicy:
    """Structure and authority owned by one exact runtime event type."""

    owned_keys: frozenset[str] = frozenset()
    owned_nested_paths: frozenset[tuple[str, ...]] = frozenset()
    authority_keys: frozenset[str] = frozenset()
    public_authority_keys: frozenset[str] = frozenset()
    aliased_authority_keys: frozenset[str] = frozenset()
    nested_authority_paths: frozenset[tuple[str, ...]] = frozenset()
    aliased_nested_authority_paths: frozenset[tuple[str, ...]] = frozenset()
    envelope_aliased_nested_authority_paths: frozenset[tuple[str, ...]] = frozenset()
    untrusted_container_keys: frozenset[str] = frozenset()
    untrusted_container_paths: frozenset[tuple[str, ...]] = frozenset()

    def __post_init__(self) -> None:
        if not self.authority_keys <= self.owned_keys:
            raise ValueError("Event authority keys must also be owned keys.")
        if not self.public_authority_keys <= self.authority_keys:
            raise ValueError("Public event authority keys must also be authority keys.")
        if not self.aliased_authority_keys <= self.authority_keys:
            raise ValueError("Aliased event authority keys must also be authority keys.")
        if self.public_authority_keys & self.aliased_authority_keys:
            raise ValueError("Event authority cannot be both public and aliased.")
        if not self.aliased_nested_authority_paths <= self.nested_authority_paths:
            raise ValueError("Aliased nested event authority must also be nested authority.")
        if not self.envelope_aliased_nested_authority_paths <= self.nested_authority_paths:
            raise ValueError(
                "Envelope-aliased nested event authority must also be nested authority."
            )
        if any(len(path) < 2 for path in self.nested_authority_paths):
            raise ValueError("Nested event authority paths must contain at least two keys.")
        if not self.untrusted_container_keys <= self.owned_keys:
            raise ValueError("Untrusted event containers must also be owned keys.")
        if any(len(path) < 2 for path in self.untrusted_container_paths):
            raise ValueError("Nested untrusted event containers require at least two keys.")
        if not self.untrusted_container_paths <= self.owned_nested_paths:
            raise ValueError("Nested untrusted event containers must be owned schema paths.")
        if any(len(path) < 2 for path in self.owned_nested_paths):
            raise ValueError("Nested event schema paths must contain at least two keys.")
        if any(path[0] not in self.owned_keys for path in self.owned_nested_paths):
            raise ValueError("Nested event schema paths require an owned top-level key.")


@dataclass(frozen=True, slots=True)
class _ToolEventBoundary:
    controls: dict[str, Any]
    projection_references: dict[int, dict[str, Any]]
    malformed: bool = False


_MODEL_EXECUTION_AUTHORITY_KEYS = frozenset(
    {
        "model_attempt_id",
        "model_step_id",
        "tool_round_id",
    }
)
_TOOL_LINKAGE_AUTHORITY_KEYS = frozenset(
    {
        "approval_id",
        "idempotency_key",
        "input_id",
        "model_attempt_id",
        "model_step_id",
        "task_id",
        "tool_call_id",
        "tool_round_id",
    }
)
_TOOL_EVENT_TYPES = frozenset(
    {
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.TOOL_CALL_FAILED,
        EventType.TOOL_CALL_BLOCKED,
        EventType.TOOL_CALL_APPROVAL_REQUESTED,
        EventType.TOOL_CALL_APPROVED,
        EventType.TOOL_CALL_APPROVAL_DENIED,
        EventType.TOOL_CALL_APPROVAL_EXPIRED,
    }
)
_INTERACTION_STATUS_BY_EVENT = {
    EventType.INTERACTION_STARTED: "active",
    EventType.INTERACTION_RESUMED: "active",
    EventType.INTERACTION_PAUSED: "paused",
    EventType.INTERACTION_COMPLETED: "completed",
    EventType.INTERACTION_FAILED: "failed",
    EventType.INTERACTION_INTERRUPTED: "interrupted",
}
_INTERRUPTION_TYPES = frozenset(
    {
        "limit_reached",
        "operator_requested",
        "runtime_interrupted",
        "tool_approval_required",
        "user_input_required",
    }
)
_POLICY_DENIAL_DECISIONS = {
    _TOOL_POLICY_DENIAL_SOURCE: frozenset({"deny"}),
    _COMMAND_POLICY_DENIAL_SOURCE: frozenset(
        {
            "deny",
            "require_command_approval",
        }
    ),
}
_POLICY_DENIAL_ERRORS = frozenset(
    {
        "command_approval_required",
        "command_denied",
    }
)
_NON_NEGATIVE_INTEGER_CONTROL_KEYS = frozenset({"attempt", "max_attempts", "next_attempt", "step"})
_POSITIVE_INTEGER_CONTROL_KEYS = frozenset(
    {
        "accepted_event_sequence",
        "event_sequence",
        "start_event_sequence",
    }
)
_TERMINAL_CONTROL_KEYS = frozenset(
    {
        "terminal_outcome",
        "tool_effect",
        "outcome_unknown",
        "manual_reconciliation_required",
        "durable_value_error_code",
        "durable_value_error_path",
    }
)

_SESSION_STATUS_VALUES = frozenset(
    {"pending", "running", "interrupting", "completed", "failed", "interrupted"}
)
_BUDGET_SCOPE_VALUES = frozenset({"app", "agent", "causal", "session", "run"})
_BUDGET_ACTION_VALUES = frozenset({"interrupt", "notify"})
_BUDGET_SETTLEMENT_KIND_VALUES = frozenset({"completed", "conservative", "released"})
_BUDGET_RESERVATION_STATUS_VALUES = frozenset({"active", "reconciled", "released"})
_PRICING_MATCH_VALUES = frozenset({"exact", "prefix", "resource_mapping"})
_MCP_STATUS_VALUES = frozenset(
    {
        "first_seen",
        "changed",
        "unchanged",
        "not_evaluated",
        "history_conflict",
        "history_unavailable",
        "fenced",
    }
)
_MCP_OUTCOME_VALUES = frozenset({"accepted", "blocked", "batch_blocked", "fenced"})
_TASK_STATUS_VALUES = frozenset(
    {
        "pending",
        "claimed",
        "running",
        "paused",
        "blocked",
        "needs_attention",
        "completed",
        "failed",
        "cancelled",
    }
)
_SESSION_CHECKPOINT_VALUES = frozenset(
    {
        "context_compaction",
        "pending_tool_approval",
        "pending_user_input",
        "usage_triggered_context",
    }
)
_DECLARED_FIXED_CONTROLS: Mapping[
    EventType,
    Mapping[tuple[str, ...], frozenset[Any]],
] = {
    EventType.MODEL_STARTED: {
        ("purpose",): frozenset({"context_compaction"}),
    },
    **{
        event_type: {
            ("status",): _MCP_STATUS_VALUES,
            ("outcome",): _MCP_OUTCOME_VALUES,
            ("policy", "action"): frozenset({"allow", "alert", "block"}),
            ("policy", "status"): frozenset({"first_seen", "changed", "unchanged"}),
        }
        for event_type in {
            EventType.MCP_MANIFEST_CHECKED,
            EventType.MCP_MANIFEST_BLOCKED,
        }
    },
    **{
        event_type: {
            ("action",): _BUDGET_ACTION_VALUES,
            ("scope",): _BUDGET_SCOPE_VALUES,
        }
        for event_type in {
            EventType.BUDGET_CHECKED,
            EventType.BUDGET_LIMIT_REACHED,
            EventType.BUDGET_RESERVED,
            EventType.BUDGET_RESERVATION_FAILED,
        }
    },
    EventType.BUDGET_RECONCILED: {
        ("settlement_kind",): _BUDGET_SETTLEMENT_KIND_VALUES,
        ("status",): _BUDGET_RESERVATION_STATUS_VALUES,
        ("pricing", "match"): _PRICING_MATCH_VALUES,
    },
    EventType.BUDGET_RESERVATION_RELEASED: {
        ("settlement_kind",): _BUDGET_SETTLEMENT_KIND_VALUES,
        ("status",): _BUDGET_RESERVATION_STATUS_VALUES,
        ("pricing", "match"): _PRICING_MATCH_VALUES,
    },
    EventType.MODEL_COMPLETED: {
        ("purpose",): frozenset({"context_compaction"}),
        ("budget_settlements", "*", "settlement_kind"): _BUDGET_SETTLEMENT_KIND_VALUES,
        ("budget_settlements", "*", "status"): _BUDGET_RESERVATION_STATUS_VALUES,
        ("budget_settlements", "*", "pricing", "match"): _PRICING_MATCH_VALUES,
    },
    **{
        event_type: {
            ("outcome",): frozenset({"completed", "failed", "interrupted"}),
            ("terminal_outcome",): frozenset({"completed", "failed", "interrupted"}),
            ("factory_allocation_action",): frozenset({"preserve"}),
        }
        for event_type in {
            EventType.ENVIRONMENT_BINDING_FINALIZE_STARTED,
            EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED,
            EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED,
        }
    },
    EventType.WORKFLOW_STEP_STARTED: {
        ("kind",): frozenset({"gated_loop"}),
    },
    EventType.WORKFLOW_STEP_COMPLETED: {
        ("kind",): frozenset({"gated_loop"}),
        ("outcome",): frozenset({"pass", "fail"}),
        ("passed",): frozenset({True, False}),
    },
    EventType.SERVER_MUTATION_ACCEPTED: {
        ("accepted_event_publication_uncertain",): frozenset({True, False}),
    },
    EventType.TURN_COMPLETED: {("status",): _SESSION_STATUS_VALUES},
    EventType.SESSION_CHECKPOINTED: {("checkpoint",): _SESSION_CHECKPOINT_VALUES},
    EventType.SESSION_MESSAGE_QUEUED: {("delivery_mode",): frozenset({"next_turn", "on_idle"})},
    EventType.SESSION_MESSAGE_DELIVERED: {("delivery_mode",): frozenset({"next_turn", "on_idle"})},
    **{
        event_type: {("task_status",): _TASK_STATUS_VALUES}
        for event_type in {
            EventType.TASK_CREATED,
            EventType.TASK_STARTED,
            EventType.TASK_COMPLETED,
            EventType.TASK_FAILED,
            EventType.TASK_CANCELLED,
        }
    },
    EventType.CREDENTIAL_MODE_SELECTED: {
        ("credential_mode",): frozenset({"raw_env", "trusted_tool", "virtual_egress"})
    },
    **{
        event_type: {("strategy",): frozenset({"native", "tool"})}
        for event_type in {
            EventType.STRUCTURED_OUTPUT_VALIDATED,
            EventType.STRUCTURED_OUTPUT_VALIDATING,
            EventType.STRUCTURED_OUTPUT_FAILED,
            EventType.STRUCTURED_OUTPUT_RETRY,
        }
    },
    **{
        event_type: {
            ("checkpoint",): frozenset({"context_compaction"}),
            ("coverage_mode",): frozenset(
                {"pending", "full", "partial_prefix", "no_progress", "failed"}
            ),
            ("chunk_mode",): frozenset(
                {
                    "pending",
                    "failed",
                    "single_request",
                    "message_prefix",
                    "hierarchical_atomic_unit",
                    "digest_prefix",
                    "digest_capacity_exhausted",
                    "provider_native_exact",
                    "custom",
                }
            ),
            ("bounded_input",): frozenset({True, False}),
            ("compaction_failed",): frozenset({True, False}),
        }
        for event_type in {
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.CONTEXT_COMPACTION_COMPLETED,
            EventType.CONTEXT_COMPACTION_FAILED,
        }
    },
}
_EXTENSIBLE_FIXED_CONTROLS = frozenset(
    {
        (EventType.SESSION_CHECKPOINTED, ("checkpoint",)),
    }
)


def _policy(
    *owned_keys: str,
    owned_nested_paths: Collection[tuple[str, ...]] = (),
    authority_keys: Collection[str] = (),
    public_authority_keys: Collection[str] = (),
    aliased_authority_keys: Collection[str] = (),
    nested_authority_paths: Collection[tuple[str, ...]] = (),
    aliased_nested_authority_paths: Collection[tuple[str, ...]] = (),
    envelope_aliased_nested_authority_paths: Collection[tuple[str, ...]] = (),
    untrusted_container_keys: Collection[str] = (),
    untrusted_container_paths: Collection[tuple[str, ...]] = (),
) -> EventPayloadPolicy:
    authority = frozenset(authority_keys)
    nested_authority = frozenset(nested_authority_paths)
    untrusted = frozenset(untrusted_container_keys)
    nested_untrusted = frozenset(untrusted_container_paths)
    return EventPayloadPolicy(
        owned_keys=frozenset(owned_keys) | authority | untrusted,
        # A nested authority field is necessarily part of the owning event's
        # schema. Keeping this implication in the policy constructor prevents
        # exact linkage keys inside otherwise-untrusted containers from being
        # rejected or renamed under short-secret key collisions.
        owned_nested_paths=(frozenset(owned_nested_paths) | nested_authority | nested_untrusted),
        authority_keys=authority,
        public_authority_keys=frozenset(public_authority_keys),
        aliased_authority_keys=frozenset(aliased_authority_keys),
        nested_authority_paths=nested_authority,
        aliased_nested_authority_paths=frozenset(aliased_nested_authority_paths),
        envelope_aliased_nested_authority_paths=frozenset(envelope_aliased_nested_authority_paths),
        untrusted_container_keys=untrusted,
        untrusted_container_paths=nested_untrusted,
    )


def _keys(value: str) -> tuple[str, ...]:
    return tuple(value.split())


def _observed_policy(
    keys: str,
    *,
    owned_nested_paths: Collection[tuple[str, ...]] = (),
    authority_keys: Collection[str] = (),
    public_authority_keys: Collection[str] = (),
    aliased_authority_keys: Collection[str] = (),
    nested_authority_paths: Collection[tuple[str, ...]] = (),
    aliased_nested_authority_paths: Collection[tuple[str, ...]] = (),
    envelope_aliased_nested_authority_paths: Collection[tuple[str, ...]] = (),
    untrusted_container_keys: Collection[str] = (),
    untrusted_container_paths: Collection[tuple[str, ...]] = (),
) -> EventPayloadPolicy:
    """Build one exact policy from the audited producer-key inventory."""

    owned = _keys(keys)
    explicit_authority = set(authority_keys)
    explicit_authority.update(key for key in owned if key.endswith("_id"))
    explicit_authority.update(
        key
        for key in owned
        if key
        in {
            "idempotency_key",
            "ordering_key",
        }
    )
    return _policy(
        *owned,
        owned_nested_paths=owned_nested_paths,
        authority_keys=explicit_authority,
        public_authority_keys=public_authority_keys,
        aliased_authority_keys=aliased_authority_keys,
        nested_authority_paths=nested_authority_paths,
        aliased_nested_authority_paths=aliased_nested_authority_paths,
        envelope_aliased_nested_authority_paths=envelope_aliased_nested_authority_paths,
        untrusted_container_keys=untrusted_container_keys,
        untrusted_container_paths=untrusted_container_paths,
    )


_AGGREGATE_USAGE_NESTED_PATHS = frozenset(
    {
        ("token_usage", "input_tokens"),
        ("token_usage", "output_tokens"),
        ("token_usage", "total_tokens"),
        ("token_usage", "reasoning_output_tokens"),
        ("token_usage", "cache"),
        ("token_usage", "cache", "read_tokens"),
        ("token_usage", "cache", "write_tokens"),
        ("token_usage", "cache", "write_5m_tokens"),
        ("token_usage", "cache", "write_1h_tokens"),
        ("token_usage", "cache", "write_unknown_ttl_tokens"),
        ("token_usage", "cache", "cached_input_tokens"),
        ("token_usage", "cache", "uncached_input_tokens"),
    }
)
_CACHE_USAGE_FIELD_NAMES = frozenset(
    {
        "read_tokens",
        "write_tokens",
        "write_5m_tokens",
        "write_1h_tokens",
        "write_unknown_ttl_tokens",
        "cached_input_tokens",
        "uncached_input_tokens",
    }
)
_USAGE_METRICS_FIELD_NAMES = frozenset(
    {
        "provider_name",
        "requested_model",
        "model",
        "billing_identity",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "reasoning_output_tokens",
        "cache",
    }
)
_BILLING_IDENTITY_FIELD_NAMES = frozenset(
    {
        "provider_name",
        "resource_id",
        "request_evidence",
        "completion_evidence",
        "pricing_contexts",
    }
)


def _billing_identity_schema_paths(prefix: tuple[str, ...]) -> frozenset[tuple[str, ...]]:
    return frozenset(
        {
            *((*prefix, field_name) for field_name in _BILLING_IDENTITY_FIELD_NAMES),
            (*prefix, "pricing_contexts", "*", "dimensions"),
        }
    )


_MODEL_USAGE_METRICS_NESTED_PATHS = frozenset(
    {
        *(("usage_metrics", field_name) for field_name in _USAGE_METRICS_FIELD_NAMES),
        *(("usage_metrics", "cache", field_name) for field_name in _CACHE_USAGE_FIELD_NAMES),
    }
) | _billing_identity_schema_paths(("usage_metrics", "billing_identity"))
_MODEL_BILLING_IDENTITY_NESTED_PATHS = _billing_identity_schema_paths(("billing_identity",))
_MODEL_ACCOUNTING_UNTRUSTED_CONTAINER_PATHS = frozenset(
    {
        ("billing_identity", "request_evidence"),
        ("billing_identity", "completion_evidence"),
        ("billing_identity", "pricing_contexts", "*", "dimensions"),
        ("usage_metrics", "billing_identity", "request_evidence"),
        ("usage_metrics", "billing_identity", "completion_evidence"),
        (
            "usage_metrics",
            "billing_identity",
            "pricing_contexts",
            "*",
            "dimensions",
        ),
    }
)
_MODEL_CONTEXT_PRESSURE_NESTED_PATHS = frozenset(
    {
        ("context_pressure", "estimated_tool_schema_input_tokens"),
        ("context_pressure", "estimated_structured_output_input_tokens"),
        ("context_pressure", "estimated_request_options_input_tokens"),
        ("context_pressure", "estimated_request_overhead_input_tokens"),
    }
)
_MODEL_COMPLETION_NESTED_PATHS = frozenset(
    {
        ("completion", "finish_reason"),
        ("completion", "raw_finish_reason"),
        ("completion", "status"),
        ("completion", "end_turn"),
    }
)
_BUDGET_RECONCILIATION_FIELD_NAMES = frozenset(
    {
        "reservation_id",
        "settlement_id",
        "settlement_kind",
        "budget_limit_id",
        "model_step_id",
        "model_attempt_id",
        "status",
        "reserved_amount",
        "actual_amount",
        "released_amount",
        "reason",
        "settled_at_unix_us",
        "pricing",
        "billing_identity",
    }
)
_BUDGET_PRICING_FIELD_NAMES = frozenset(
    {
        "provider_name",
        "model",
        "match",
        "provenance",
        "effective_from",
        "effective_through",
        "tier_max_input_tokens",
    }
)


def _budget_reconciliation_schema_paths(
    prefix: tuple[str, ...],
) -> frozenset[tuple[str, ...]]:
    pricing_prefix = (*prefix, "pricing")
    return frozenset(
        {
            *(
                ((*prefix, field_name) for field_name in _BUDGET_RECONCILIATION_FIELD_NAMES)
                if prefix
                else ()
            ),
            *((*pricing_prefix, field_name) for field_name in _BUDGET_PRICING_FIELD_NAMES),
            (*pricing_prefix, "provenance", "source"),
            (*pricing_prefix, "provenance", "url"),
            (*pricing_prefix, "provenance", "as_of"),
        }
    ) | _billing_identity_schema_paths((*prefix, "billing_identity"))


_BUDGET_RECONCILIATION_NESTED_PATHS = _budget_reconciliation_schema_paths(())
_MODEL_BUDGET_SETTLEMENT_NESTED_PATHS = _budget_reconciliation_schema_paths(
    ("budget_settlements", "*"),
)
_MODEL_BUDGET_SETTLEMENT_AUTHORITY_PATHS = frozenset(
    {
        ("budget_settlements", "*", field_name)
        for field_name in {
            "reservation_id",
            "settlement_id",
            "budget_limit_id",
            "model_step_id",
            "model_attempt_id",
        }
    }
)
_BUDGET_RECONCILIATION_UNTRUSTED_PATHS = frozenset(
    {
        ("billing_identity", "request_evidence"),
        ("billing_identity", "completion_evidence"),
        ("billing_identity", "pricing_contexts", "*", "dimensions"),
    }
)
_MODEL_BUDGET_SETTLEMENT_UNTRUSTED_PATHS = frozenset(
    {("budget_settlements", "*", *path) for path in _BUDGET_RECONCILIATION_UNTRUSTED_PATHS}
)
_TOOL_RESULT_NESTED_PATHS = frozenset(
    {
        ("result", "content"),
        ("result", "structured"),
        ("result", "artifacts"),
        ("result", "is_error"),
        ("result", "structured", "terminal_outcome"),
        ("result", "structured", "tool_effect"),
        ("result", "structured", "outcome_unknown"),
        ("result", "structured", "manual_reconciliation_required"),
        ("result", "structured", "durable_value_error_code"),
        ("result", "structured", "durable_value_error_path"),
    }
)
_TOOL_RESULT_PROJECTION_RECORD_FIELDS = frozenset(
    {
        "artifact_id",
        "artifact_sha256",
        "failure_type",
        "logical_identity_sha256",
        "original_bytes",
        "original_token_estimate",
        "policy_id",
        "projected_bytes",
        "projected_token_estimate",
        "schema_version",
        "status",
        "token_estimation_method",
        "tool_call_id_sha256",
    }
)
_TOOL_RESULT_PROJECTION_RECORD_NESTED_PATHS = frozenset(
    ("tool_result_projection", field_name) for field_name in _TOOL_RESULT_PROJECTION_RECORD_FIELDS
)
_TOOL_EVENT_NESTED_PATHS = _TOOL_RESULT_NESTED_PATHS | _TOOL_RESULT_PROJECTION_RECORD_NESTED_PATHS
_TOOL_DENIAL_RESULT_NESTED_PATHS = _TOOL_RESULT_NESTED_PATHS | {
    ("result", "structured", "decision"),
    ("result", "structured", "error"),
    ("result", "structured", "reason"),
}
_TOOL_PROJECTED_DENIAL_RESULT_NESTED_PATHS = (
    _TOOL_DENIAL_RESULT_NESTED_PATHS | _TOOL_RESULT_PROJECTION_RECORD_NESTED_PATHS
)
_TOOL_RESULT_NESTED_AUTHORITY_PATHS = frozenset(
    {("result", "structured", field_name) for field_name in _TOOL_LINKAGE_AUTHORITY_KEYS}
)
_ACTIONABLE_NESTED_AUTHORITY_FIELD_NAMES = frozenset(
    {"approval_id", "input_id", "tool_call_id", "tool_round_id"}
)
_RESOLUTION_ACTOR_NESTED_FIELD_NAMES = frozenset({"source", "subject", "tenant"})
_PENDING_TOOL_CALL_FIELD_NAMES = frozenset(
    {
        "active_taint_labels",
        "arguments",
        "metadata",
        "policy_decision",
        "policy_evidence",
        "reason",
        "tool_call_id",
        "tool_name",
    }
)
_PENDING_APPROVAL_FIELD_NAMES = frozenset(
    {
        "agent_name",
        "approval_id",
        "arguments",
        "budget_limits",
        "environment_name",
        "expires_at",
        "limits",
        "max_steps",
        "metadata",
        "model_attempt_id",
        "model_step_id",
        "reason",
        "retry_policy",
        "structured_output",
        "task_id",
        "thinking",
        "tool_call_id",
        "tool_calls",
        "tool_name",
        "tool_round_id",
        "workspace_id",
    }
)
_PENDING_USER_INPUT_FIELD_NAMES = frozenset(
    {
        "agent_name",
        "arguments",
        "budget_limits",
        "environment_name",
        "input_id",
        "limits",
        "max_steps",
        "model_attempt_id",
        "model_step_id",
        "options",
        "question",
        "retry_policy",
        "structured_output",
        "task_id",
        "thinking",
        "tool_call_id",
        "tool_calls",
        "tool_name",
        "tool_round_id",
        "workspace_id",
    }
)


def _pause_schema_paths(
    container_name: str,
    field_names: Collection[str],
) -> frozenset[tuple[str, ...]]:
    """Return the audited fixed keys of one typed pause payload."""

    return frozenset(
        {
            *((container_name, field_name) for field_name in field_names),
            *(
                (container_name, "tool_calls", "*", field_name)
                for field_name in _PENDING_TOOL_CALL_FIELD_NAMES
            ),
        }
    )


_APPROVAL_NESTED_SCHEMA_PATHS = _pause_schema_paths(
    "approval",
    _PENDING_APPROVAL_FIELD_NAMES,
)
_USER_INPUT_NESTED_SCHEMA_PATHS = _pause_schema_paths(
    "user_input",
    _PENDING_USER_INPUT_FIELD_NAMES,
)
_APPROVAL_NESTED_AUTHORITY_PATHS = frozenset(
    {
        ("approval", field_name)
        for field_name in {
            "approval_id",
            "tool_round_id",
            "model_step_id",
            "model_attempt_id",
            "tool_call_id",
            "workspace_id",
            "task_id",
        }
    }
    | {
        ("approval", "tool_calls", "*", "tool_call_id"),
    }
)
_USER_INPUT_NESTED_AUTHORITY_PATHS = frozenset(
    {
        ("user_input", field_name)
        for field_name in {
            "input_id",
            "tool_round_id",
            "model_step_id",
            "model_attempt_id",
            "tool_call_id",
            "workspace_id",
            "task_id",
        }
    }
    | {
        ("user_input", "tool_calls", "*", "tool_call_id"),
    }
)
_TOOL_CALL_LIST_NESTED_AUTHORITY_PATHS = frozenset({("tool_calls", "*", "tool_call_id")})


def _actionable_nested_authority_paths(
    paths: Collection[tuple[str, ...]],
) -> frozenset[tuple[str, ...]]:
    return frozenset(path for path in paths if path[-1] in _ACTIONABLE_NESTED_AUTHORITY_FIELD_NAMES)


def _resolution_actor_nested_paths(*container_names: str) -> frozenset[tuple[str, ...]]:
    """Return the exact schema-owned leaves of typed resolution actors."""

    return frozenset(
        (container_name, field_name)
        for container_name in container_names
        for field_name in _RESOLUTION_ACTOR_NESTED_FIELD_NAMES
    )


def public_event_id(sequence: int) -> str:
    if type(sequence) is not int or not 1 <= sequence <= MAX_DURABLE_JSON_INTEGER:
        raise ValueError(f"sequence must be an integer between 1 and {MAX_DURABLE_JSON_INTEGER}.")
    return f"{PUBLIC_EVENT_ID_PREFIX}{sequence}"


def public_event_sequence(value: str) -> int | None:
    if type(value) is not str or not value.startswith(PUBLIC_EVENT_ID_PREFIX):
        return None
    suffix = value.removeprefix(PUBLIC_EVENT_ID_PREFIX)
    if not suffix or suffix.startswith("0") or not suffix.isascii() or not suffix.isdecimal():
        return None
    # Avoid both Python's arbitrary-size integer work and its digit-limit
    # exception on untrusted aliases before constructing a durable query.
    maximum = str(MAX_DURABLE_JSON_INTEGER)
    if len(suffix) > len(maximum) or (len(suffix) == len(maximum) and suffix > maximum):
        return None
    sequence = int(suffix)
    return sequence if sequence >= 1 else None


def public_event_linkage_id(sequence: int, field_name: str) -> str:
    """Return a stable presentation alias for one record-owned linkage field."""

    if type(field_name) is not str or not field_name or not field_name.isidentifier():
        raise ValueError("field_name must be a non-empty identifier.")
    return f"{public_event_id(sequence)}{PUBLIC_EVENT_LINKAGE_SEPARATOR}{field_name}"


def public_event_linkage_sequence(value: str, *, field_name: str) -> int | None:
    """Parse an exact field-scoped linkage alias without accepting lookalikes."""

    if type(value) is not str or type(field_name) is not str:
        return None
    suffix = f"{PUBLIC_EVENT_LINKAGE_SEPARATOR}{field_name}"
    if not value.endswith(suffix):
        return None
    return public_event_sequence(value[: -len(suffix)])


def public_event_envelope_alias(
    value: str,
    *,
    field_name: str,
    codec: PublicAuthorityAliasCodec,
    session_id: str | None = None,
) -> str:
    """Return a stable non-authoritative alias for private envelope identity."""

    if type(value) is not str or not value:
        raise ValueError("value must be a non-empty string.")
    if field_name not in {"session_id", "interaction_id"}:
        raise ValueError("field_name must be session_id or interaction_id.")
    if not isinstance(codec, PublicAuthorityAliasCodec):
        raise TypeError("codec must be a PublicAuthorityAliasCodec.")
    if field_name == "session_id" and session_id is not None:
        raise ValueError("Session aliases must not have a session scope.")
    if field_name == "interaction_id" and session_id is None:
        raise ValueError("Interaction aliases require a private session scope.")
    return codec.encode(value, field_name=field_name, session_id=session_id)


def public_event_envelope_alias_field(value: str) -> str | None:
    """Return the field owned by a syntactically valid envelope alias."""

    parsed = parse_public_authority_alias(value)
    if parsed is None or parsed.field_name not in {"session_id", "interaction_id"}:
        return None
    return parsed.field_name


def _require_public_authority_alias_codec(
    codec: PublicAuthorityAliasCodec | None,
) -> PublicAuthorityAliasCodec:
    if not isinstance(codec, PublicAuthorityAliasCodec):
        raise RuntimeError("Secret-bearing public authority requires a configured alias keyring.")
    return codec


def private_event_linkage_value(event: Event, *, field_name: str) -> str | None:
    """Resolve one public alias from schema-owned private durable authority.

    Singular top-level and nested authority take precedence over repeated list
    evidence. Conflicting or malformed legacy values fail closed because one
    field-scoped public alias must never choose between multiple authorities.
    """

    if type(event) is not Event:
        raise TypeError("event must be an Event.")
    if type(field_name) is not str or not field_name or not field_name.isidentifier():
        raise ValueError("field_name must be a non-empty identifier.")
    policy = event_payload_policy(event.type)
    singular: list[str] = []
    repeated: list[str] = []

    if (
        field_name in _ACTIONABLE_NESTED_AUTHORITY_FIELD_NAMES
        and field_name in policy.aliased_authority_keys
        and field_name in event.payload
        and not _collect_private_linkage_value(event.payload[field_name], singular)
    ):
        return None

    paths = sorted(
        (path for path in policy.aliased_nested_authority_paths if path[-1] == field_name),
        key=lambda path: ("*" in path, len(path), path),
    )
    for path in paths:
        target = repeated if "*" in path else singular
        for value in _values_at_schema_path(event.payload, path):
            if not _collect_private_linkage_value(value, target):
                return None

    candidates = singular or repeated
    unique = set(candidates)
    if len(unique) != 1:
        return None
    return candidates[0]


def _resolvable_alias_fields(
    event: Event,
    *,
    policy: EventPayloadPolicy,
) -> frozenset[str]:
    """Return only aliases backed by one valid private durable value."""

    field_names = {
        *policy.aliased_authority_keys,
        *(path[-1] for path in policy.aliased_nested_authority_paths),
    }
    return frozenset(
        field_name
        for field_name in field_names
        if private_event_linkage_value(event, field_name=field_name) is not None
    )


def _collect_private_linkage_value(value: Any, target: list[str]) -> bool:
    if value is None:
        return True
    if type(value) is not str or not value.strip():
        return False
    target.append(value)
    return True


def _values_at_schema_path(value: Any, path: tuple[str, ...]) -> list[Any]:
    values = [value]
    for component in path:
        selected: list[Any] = []
        for candidate in values:
            if component == "*":
                if type(candidate) is list:
                    selected.extend(candidate)
                continue
            if type(candidate) is dict and component in candidate:
                selected.append(candidate[component])
        values = selected
        if not values:
            break
    return values


def _event_policies() -> dict[EventType, EventPayloadPolicy]:
    """Return the explicit policy registry.

    Every enum member is present even when the event owns no fixed payload
    structure. The assignments below are intentionally per exact type; shared
    policy objects only reduce repetition and do not grant keys to custom or
    unrelated events.
    """

    policies = {event_type: _policy() for event_type in EventType}

    interaction_summary = _policy(
        "active_duration_ms",
        "completed_at",
        "model_step_count",
        "models",
        "pending_action_kind",
        "provider_names",
        "result_transcript_end",
        "result_transcript_start",
        "source_transcript_end",
        "source_transcript_start",
        "start_event_id",
        "start_event_sequence",
        "started_at",
        "status",
        "token_usage",
        "tool_call_count",
        "wall_duration_ms",
        owned_nested_paths=_AGGREGATE_USAGE_NESTED_PATHS,
        authority_keys={"start_event_id"},
    )
    for event_type in (
        EventType.INTERACTION_STARTED,
        EventType.INTERACTION_RESUMED,
        EventType.INTERACTION_PAUSED,
        EventType.INTERACTION_COMPLETED,
        EventType.INTERACTION_FAILED,
        EventType.INTERACTION_INTERRUPTED,
    ):
        policies[event_type] = interaction_summary

    model_started = _policy(
        "attempt",
        "max_attempts",
        "model",
        "model_attempt_id",
        "model_step_id",
        "provider",
        "purpose",
        "step",
        authority_keys=_MODEL_EXECUTION_AUTHORITY_KEYS,
    )
    policies[EventType.MODEL_STARTED] = model_started
    model_delta = _policy(
        "attempt",
        "delta",
        "max_attempts",
        "model_attempt_id",
        "model_step_id",
        "step",
        authority_keys=_MODEL_EXECUTION_AUTHORITY_KEYS,
    )
    policies[EventType.MODEL_TEXT_DELTA] = model_delta
    policies[EventType.MODEL_THINKING_DELTA] = model_delta
    policies[EventType.MODEL_COMPLETED] = _policy(
        "actor",
        "attempt",
        "attempt_id",
        "bedrock_usage",
        "billing_identity",
        "budget_settlements",
        "compaction_outcome",
        "compactor",
        "completion",
        "completion_error",
        "completion_outcome",
        "context_overflow",
        "context_pressure",
        "details",
        "end_turn",
        "error",
        "error_type",
        "finish_reason",
        "id",
        "incomplete_details",
        "instruction_digest",
        "instruction_present",
        "max_attempts",
        "metadata",
        "mode",
        "model",
        "model_attempt_id",
        "model_step_id",
        "operation_id",
        "provider_debug",
        "provider_name",
        "purpose",
        "reason",
        "rejected_usage_evidence",
        "request_id",
        "requested_model",
        "source_run_epoch",
        "source_transcript_cursor",
        "state",
        "status",
        "step",
        "step_classification",
        "stop_reason",
        "stop_sequence",
        "tool_round_id",
        "transcript_cursor",
        "usage",
        "usage_metrics",
        "usage_metrics_rejected",
        "usage_normalization_failed",
        "usage_unavailable_reason",
        owned_nested_paths=(
            _MODEL_COMPLETION_NESTED_PATHS
            | _MODEL_USAGE_METRICS_NESTED_PATHS
            | _MODEL_BILLING_IDENTITY_NESTED_PATHS
            | _MODEL_BUDGET_SETTLEMENT_NESTED_PATHS
            | _MODEL_CONTEXT_PRESSURE_NESTED_PATHS
            | _resolution_actor_nested_paths("actor")
            | {
                ("step_classification", "type"),
                ("step_classification", "reason"),
            }
        ),
        authority_keys=_MODEL_EXECUTION_AUTHORITY_KEYS,
        nested_authority_paths=_MODEL_BUDGET_SETTLEMENT_AUTHORITY_PATHS,
        untrusted_container_keys={
            "details",
            "incomplete_details",
            "metadata",
            "provider_debug",
        },
        untrusted_container_paths=(
            _MODEL_ACCOUNTING_UNTRUSTED_CONTAINER_PATHS | _MODEL_BUDGET_SETTLEMENT_UNTRUSTED_PATHS
        ),
    )
    model_failure_keys = (
        "attempt",
        "context_overflow",
        "error",
        "error_code",
        "error_type",
        "max_attempts",
        "model",
        "model_attempt_id",
        "model_step_id",
        "provider",
        "provider_error_code",
        "provider_error_type",
        "request_id",
        "retry_after_s",
        "retryable",
        "stage",
        "status_code",
        "step",
    )
    policies[EventType.MODEL_ERROR] = _policy(
        *model_failure_keys,
        authority_keys=_MODEL_EXECUTION_AUTHORITY_KEYS,
    )
    policies[EventType.MODEL_RETRY] = _policy(
        *model_failure_keys,
        "delay_s",
        "delay_seconds",
        "next_attempt",
        "reason",
        authority_keys=_MODEL_EXECUTION_AUTHORITY_KEYS,
    )
    policies[EventType.MODEL_ATTEMPT_DISCARDED] = _policy(
        "attempt",
        "max_attempts",
        "model",
        "model_attempt_id",
        "model_step_id",
        "next_attempt",
        "provider",
        "reason",
        "status_code",
        "step",
        authority_keys=_MODEL_EXECUTION_AUTHORITY_KEYS,
    )

    tool_common = {
        "approval",
        "approval_id",
        "approval_metadata_truncated",
        "arguments",
        "effect",
        "effective_arguments",
        "expired",
        "idempotency_key",
        "input_id",
        "metadata",
        "metadata_truncated",
        "model_attempt_id",
        "model_step_id",
        "reason",
        "recovered",
        "requested_decision",
        "resolution_reason",
        "resolution_request_digest",
        "resolved_by",
        "result",
        "short_circuited_by",
        "structured_output_validation",
        "task_id",
        "tool_call_id",
        "tool_call_metadata_truncated",
        "tool_name",
        "tool_round_id",
    }
    tool_terminal = tool_common | {
        "durable_value_error_code",
        "durable_value_error_path",
        "manual_reconciliation_required",
        "manual_recovery",
        "outcome_unknown",
        "registration_state",
        "terminal_outcome",
        "tool_effect",
    }
    tool_actor_paths = _resolution_actor_nested_paths("resolved_by")
    policies[EventType.TOOL_CALL_STARTED] = _policy(
        *tool_common,
        owned_nested_paths=_TOOL_RESULT_NESTED_PATHS | tool_actor_paths,
        authority_keys=_TOOL_LINKAGE_AUTHORITY_KEYS,
        aliased_authority_keys={
            "approval_id",
            "input_id",
            "tool_call_id",
            "tool_round_id",
        },
        nested_authority_paths=_TOOL_RESULT_NESTED_AUTHORITY_PATHS,
        aliased_nested_authority_paths=_actionable_nested_authority_paths(
            _TOOL_RESULT_NESTED_AUTHORITY_PATHS
        ),
        untrusted_container_keys={
            "approval",
            "arguments",
            "effective_arguments",
            "metadata",
            "result",
            "structured_output_validation",
        },
    )
    for event_type in (EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED):
        policies[event_type] = _policy(
            *tool_terminal,
            "abnormal_termination",
            "limit",
            "reason",
            "resolved_by",
            "tool_result_projection",
            owned_nested_paths=_TOOL_EVENT_NESTED_PATHS | tool_actor_paths,
            authority_keys=_TOOL_LINKAGE_AUTHORITY_KEYS,
            aliased_authority_keys={
                "approval_id",
                "input_id",
                "tool_call_id",
                "tool_round_id",
            },
            nested_authority_paths=_TOOL_RESULT_NESTED_AUTHORITY_PATHS,
            aliased_nested_authority_paths=_actionable_nested_authority_paths(
                _TOOL_RESULT_NESTED_AUTHORITY_PATHS
            ),
            untrusted_container_keys={
                "approval",
                "arguments",
                "effective_arguments",
                "manual_recovery",
                "metadata",
                "result",
                "structured_output_validation",
            },
        )
    policies[EventType.TOOL_CALL_BLOCKED] = _policy(
        *tool_common,
        "blocked_by",
        "decision",
        "denied_by",
        "reason",
        "tool_result_projection",
        owned_nested_paths=_TOOL_PROJECTED_DENIAL_RESULT_NESTED_PATHS | tool_actor_paths,
        authority_keys=_TOOL_LINKAGE_AUTHORITY_KEYS,
        aliased_authority_keys={
            "approval_id",
            "input_id",
            "tool_call_id",
            "tool_round_id",
        },
        nested_authority_paths=_TOOL_RESULT_NESTED_AUTHORITY_PATHS,
        aliased_nested_authority_paths=_actionable_nested_authority_paths(
            _TOOL_RESULT_NESTED_AUTHORITY_PATHS
        ),
        untrusted_container_keys={
            "approval",
            "arguments",
            "effective_arguments",
            "metadata",
            "result",
        },
    )
    policies[EventType.TOOL_CALL_APPROVAL_REQUESTED] = _policy(
        *tool_common,
        "approval_required",
        "expires_at",
        "reason",
        "recovered",
        owned_nested_paths=(
            _TOOL_RESULT_NESTED_PATHS | _APPROVAL_NESTED_SCHEMA_PATHS | tool_actor_paths
        ),
        authority_keys=_TOOL_LINKAGE_AUTHORITY_KEYS,
        aliased_authority_keys={
            "approval_id",
            "input_id",
            "tool_call_id",
            "tool_round_id",
        },
        nested_authority_paths=(
            _TOOL_RESULT_NESTED_AUTHORITY_PATHS | _APPROVAL_NESTED_AUTHORITY_PATHS
        ),
        aliased_nested_authority_paths=_actionable_nested_authority_paths(
            _TOOL_RESULT_NESTED_AUTHORITY_PATHS | _APPROVAL_NESTED_AUTHORITY_PATHS
        ),
        untrusted_container_keys={"approval", "arguments", "metadata", "result"},
    )
    policies[EventType.TOOL_CALL_APPROVED] = _policy(
        "approval_id",
        "reason",
        "resolved_by",
        "tool_call_id",
        "tool_round_id",
        owned_nested_paths=tool_actor_paths,
        authority_keys={
            "approval_id",
            "model_attempt_id",
            "model_step_id",
            "tool_call_id",
            "tool_round_id",
        },
        aliased_authority_keys={"approval_id", "tool_call_id", "tool_round_id"},
    )
    policies[EventType.TOOL_CALL_APPROVAL_DENIED] = _policy(
        *tool_terminal,
        "approval_required",
        "expired",
        "reason",
        "resolved_by",
        owned_nested_paths=_TOOL_DENIAL_RESULT_NESTED_PATHS | tool_actor_paths,
        authority_keys=_TOOL_LINKAGE_AUTHORITY_KEYS,
        aliased_authority_keys={
            "approval_id",
            "input_id",
            "tool_call_id",
            "tool_round_id",
        },
        nested_authority_paths=_TOOL_RESULT_NESTED_AUTHORITY_PATHS,
        aliased_nested_authority_paths=_actionable_nested_authority_paths(
            _TOOL_RESULT_NESTED_AUTHORITY_PATHS
        ),
    )
    policies[EventType.TOOL_CALL_APPROVAL_EXPIRED] = _policy(
        "approval_id",
        "expires_at",
        "requested_decision",
        "resolved_by",
        "tool_call_id",
        "tool_round_id",
        "triggered_by",
        owned_nested_paths=(tool_actor_paths | _resolution_actor_nested_paths("triggered_by")),
        authority_keys={
            "approval_id",
            "model_attempt_id",
            "model_step_id",
            "tool_call_id",
            "tool_round_id",
        },
        aliased_authority_keys={"approval_id", "tool_call_id", "tool_round_id"},
    )

    policies[EventType.SERVER_MUTATION_ACCEPTED] = _observed_policy(
        "accepted_event_id accepted_event_publication_uncertain accepted_event_sequence "
        "accepted_event_type mutation_id mutation_kind",
        public_authority_keys={"mutation_id"},
    )
    terminal_finalization_keys = (
        "binding_finalize_error binding_finalize_publication_error environment_factory_release"
    )
    terminal_finalization_containers = {
        "binding_finalize_error",
        "binding_finalize_publication_error",
        "environment_factory_release",
    }
    policies[EventType.SESSION_STARTED] = _observed_policy(
        "agent_name parent_session_id traceparent tracestate"
    )
    policies[EventType.SESSION_RESUMED] = _observed_policy(
        "agent_name appended_messages approval_id decision dispatch_id expired input_id "
        "interruption_type model_attempt_id model_step_id parent_session_id resolved_by "
        "task_id tool_call_id tool_round_id traceparent tracestate",
        owned_nested_paths=_resolution_actor_nested_paths("resolved_by"),
    )
    policies[EventType.SESSION_COMPLETED] = _observed_policy(
        terminal_finalization_keys,
        authority_keys={"session_run_operation_id"},
        untrusted_container_keys=terminal_finalization_containers,
    )
    policies[EventType.SESSION_FAILED] = _observed_policy(
        "approval_id binding_cleanup durable_value_error_code durable_value_error_path "
        "error error_type interruption_type "
        "manual_recovery_required model_attempt_id model_step_id tool_call_id "
        f"tool_name tool_round_id {terminal_finalization_keys}",
        authority_keys={"session_run_operation_id"},
        aliased_authority_keys={"approval_id", "tool_call_id", "tool_round_id"},
        untrusted_container_keys={"binding_cleanup"} | terminal_finalization_containers,
    )
    policies[EventType.SESSION_INTERRUPTED] = _observed_policy(
        "abandoned actual approval approval_close_intent approval_id "
        "approval_metadata_truncated cost_summary durable_value_error_code "
        "durable_value_error_path error error_type input_id interruption_request_id "
        "interruption_type limit manual_recovery_persisted "
        "manual_recovery_persistence_unknown manual_recovery_required "
        "manual_recovery_stale_live_failure maximum message metadata model_attempt_id "
        "model_step_id persistence_reconciliation_error_type policy_metadata reason "
        "recovered requested_by resolved_by tool_call_id tool_call_metadata_truncated "
        "session_run_operation_id tool_evidence_conflict tool_name tool_round_id "
        "usage_summary user_input " + terminal_finalization_keys,
        owned_nested_paths=_resolution_actor_nested_paths(
            "requested_by",
            "resolved_by",
        )
        | _APPROVAL_NESTED_SCHEMA_PATHS
        | _USER_INPUT_NESTED_SCHEMA_PATHS,
        aliased_authority_keys={
            "approval_id",
            "input_id",
            "tool_call_id",
            "tool_round_id",
        },
        authority_keys={"session_run_operation_id"},
        nested_authority_paths=(
            _APPROVAL_NESTED_AUTHORITY_PATHS | _USER_INPUT_NESTED_AUTHORITY_PATHS
        ),
        aliased_nested_authority_paths=_actionable_nested_authority_paths(
            _APPROVAL_NESTED_AUTHORITY_PATHS | _USER_INPUT_NESTED_AUTHORITY_PATHS
        ),
        untrusted_container_keys={
            "approval",
            "metadata",
            "policy_metadata",
            "user_input",
        }
        | terminal_finalization_containers,
    )
    policies[EventType.SESSION_INTERRUPTION_CASCADE_RETRY_REQUESTED] = _observed_policy(
        "attempt_id interruption_type previous_generation retry_metadata retry_reason "
        "retry_request_id retry_requested_by",
        owned_nested_paths=_resolution_actor_nested_paths("retry_requested_by"),
        untrusted_container_keys={"retry_metadata"},
    )
    policies[EventType.SESSION_INTERRUPTION_CASCADE_COMPLETED] = _observed_policy(
        "attempt_id descendant_count generation interruption_type retry_metadata "
        "retry_reason retry_request_id retry_requested_by",
        owned_nested_paths=_resolution_actor_nested_paths("retry_requested_by"),
        untrusted_container_keys={"retry_metadata"},
    )
    policies[EventType.SESSION_INTERRUPTION_CASCADE_FAILED] = _observed_policy(
        "attempt_id failure_count failures failures_truncated generation interruption_type",
        untrusted_container_keys={"failures"},
    )
    policies[EventType.SESSION_AWAITING_USER_INPUT] = _observed_policy(
        "input_id model_attempt_id model_step_id options question tool_call_id tool_calls "
        "tool_round_id",
        aliased_authority_keys={"input_id", "tool_call_id", "tool_round_id"},
        nested_authority_paths=_TOOL_CALL_LIST_NESTED_AUTHORITY_PATHS,
        aliased_nested_authority_paths=_TOOL_CALL_LIST_NESTED_AUTHORITY_PATHS,
        untrusted_container_keys={"options", "tool_calls"},
    )
    policies[EventType.SESSION_CHECKPOINTED] = _observed_policy(
        "actor approval_id attempt_id calls checkpoint cleared compacted_transcript_cursor "
        "compactor estimated_context_input_tokens estimated_context_window_tokens "
        "estimated_delta_input_tokens input_id instruction_digest instruction_present "
        "last_input_tokens last_total_tokens last_transcript_cursor min_input_tokens "
        "min_total_tokens mode model_attempt_id model_step_id "
        "newly_compacted_message_count operation_id "
        "previous_compacted_transcript_cursor provider_count_context_window_tokens "
        "provider_count_input_tokens reason recent_message_count request_id "
        "reserved_output_tokens result_transcript_cursor source_run_epoch "
        "source_transcript_cursor tool_call_id tool_round_id "
        "trigger_estimated_context_tokens",
        owned_nested_paths=_resolution_actor_nested_paths("actor"),
    )
    policies[EventType.SESSION_FORKED] = _observed_policy(
        "agent_name causal_budget_id copy_checkpoint environment_name inherited_taint_labels "
        "model parent_session_id provider_name source_session_id source_status "
        "transcript_cursor"
    )
    policies[EventType.SESSION_LIMIT_REACHED] = _observed_policy(
        "actual cost_summary limit maximum message reason usage_summary"
    )
    message_policy = _observed_policy(
        "accepted_run_epoch accepted_transcript_cursor actor delivery_mode ordering_key "
        "queue_id run_epoch transcript_cursor",
        owned_nested_paths=_resolution_actor_nested_paths("actor"),
    )
    policies[EventType.SESSION_MESSAGE_QUEUED] = message_policy
    policies[EventType.SESSION_MESSAGE_DELIVERED] = message_policy
    policies[EventType.SESSION_RUN_FENCED] = _observed_policy(
        "inactive_before metadata previous_run_epoch reason run_epoch",
        untrusted_container_keys={"metadata"},
    )
    policies[EventType.TURN_COMPLETED] = _observed_policy(
        "duration_ms interaction_ids models provider_names status step_count token_usage "
        "tool_call_count",
        owned_nested_paths=_AGGREGATE_USAGE_NESTED_PATHS,
        nested_authority_paths={("interaction_ids", "*")},
        envelope_aliased_nested_authority_paths={("interaction_ids", "*")},
    )

    budget_common = (
        "accepted action actor actual attempt_id budget_limit_id compactor cost_summary "
        "currency instruction_digest instruction_present key limit_reached maximum message "
        "mode model_attempt_id model_step_id model_steps operation_id reason request_id "
        "requested scope source_run_epoch source_transcript_cursor unpriced_model_steps "
        "window window_details"
    )
    for event_type in (
        EventType.BUDGET_CHECKED,
        EventType.BUDGET_LIMIT_REACHED,
        EventType.BUDGET_RESERVATION_FAILED,
    ):
        policies[event_type] = _observed_policy(
            budget_common,
            owned_nested_paths=_resolution_actor_nested_paths("actor"),
        )
    policies[EventType.BUDGET_RESERVED] = _observed_policy(
        budget_common
        + " agent_name billing_identity model provider_name reservation_id session_id",
        owned_nested_paths=(
            _resolution_actor_nested_paths("actor")
            | _billing_identity_schema_paths(("billing_identity",))
        ),
        untrusted_container_paths={
            ("billing_identity", "request_evidence"),
            ("billing_identity", "completion_evidence"),
            ("billing_identity", "pricing_contexts", "*", "dimensions"),
        },
    )
    settlement_policy = _observed_policy(
        "actor actual_amount attempt_id billing_identity budget_limit_id compactor "
        "instruction_digest instruction_present interaction_id mode model_attempt_id "
        "model_step_id operation_id pricing reason released_amount request_id reservation_id "
        "reserved_amount settled_at_unix_us settlement_id settlement_kind source_run_epoch "
        "source_transcript_cursor status",
        owned_nested_paths=(
            _resolution_actor_nested_paths("actor") | _BUDGET_RECONCILIATION_NESTED_PATHS
        ),
        untrusted_container_paths=_BUDGET_RECONCILIATION_UNTRUSTED_PATHS,
    )
    policies[EventType.BUDGET_RECONCILED] = settlement_policy
    policies[EventType.BUDGET_RESERVATION_RELEASED] = settlement_policy

    policies[EventType.CREDENTIAL_PROXY_CHECKED] = _observed_policy(
        "action allowed approval_id credential destination idempotency_key input_id metadata "
        "model_attempt_id model_step_id reason result_metadata tool_call_id tool_round_id",
        untrusted_container_keys={"metadata", "result_metadata"},
    )
    policies[EventType.CREDENTIAL_MODE_SELECTED] = _observed_policy(
        "approved_destination_count credential_mode grant_count runner_kind"
    )
    policies[EventType.EGRESS_GRANT_MINTED] = _observed_policy("grant_id")
    policies[EventType.EGRESS_GRANT_REVOKED] = _observed_policy("grant_id outcome")
    egress_request_policy = _observed_policy(
        "action allowed credential destination grant_id metadata reason request_id",
        untrusted_container_keys={"metadata"},
    )
    policies[EventType.EGRESS_REQUEST_AUTHORIZED] = egress_request_policy
    policies[EventType.EGRESS_REQUEST_DENIED] = egress_request_policy

    mcp_policy = _observed_policy(
        "advertised_tool_count change_classes diff history_key manifest_hash "
        "manifest_identity outcome policy previous reason server_hash source_manifest_hash "
        "status tool_count",
        owned_nested_paths={
            ("policy", "action"),
            ("policy", "status"),
            ("policy", "matched_changes"),
            ("policy", "reason"),
        },
        untrusted_container_keys={"diff", "previous"},
    )
    policies[EventType.MCP_MANIFEST_CHECKED] = mcp_policy
    policies[EventType.MCP_MANIFEST_BLOCKED] = mcp_policy

    task_policy = _observed_policy(
        "assigned_agent_name parent_task_id task_id task_session_id task_status task_type"
    )
    for event_type in (
        EventType.TASK_CREATED,
        EventType.TASK_STARTED,
        EventType.TASK_COMPLETED,
        EventType.TASK_FAILED,
        EventType.TASK_CANCELLED,
    ):
        policies[event_type] = task_policy

    structured_policy = _observed_policy(
        "attempt errors max_retries model_attempt_id model_step_id name output step strategy "
        "tool_round_id valid",
        untrusted_container_keys={"errors", "output"},
    )
    for event_type in (
        EventType.STRUCTURED_OUTPUT_VALIDATED,
        EventType.STRUCTURED_OUTPUT_VALIDATING,
        EventType.STRUCTURED_OUTPUT_FAILED,
        EventType.STRUCTURED_OUTPUT_RETRY,
    ):
        policies[event_type] = structured_policy

    compaction_policy = _observed_policy(
        "actor attempt_id bounded_input checkpoint chunk_count chunk_mode "
        "compacted_transcript_cursor compaction_failed compactor coverage_mode error_type "
        "instruction_digest instruction_present mode model_step_id "
        "newly_compacted_message_count operation_id previous_compacted_transcript_cursor "
        "reason recent_message_count represented_message_count represented_source_end "
        "represented_source_start request_id requested_source_end requested_source_start "
        "result_transcript_cursor source_run_epoch source_transcript_cursor summary_chars",
        owned_nested_paths=_resolution_actor_nested_paths("actor"),
    )
    for event_type in (
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.CONTEXT_COMPACTION_COMPLETED,
        EventType.CONTEXT_COMPACTION_FAILED,
    ):
        policies[event_type] = compaction_policy

    count_policy = _observed_policy(
        "attempt count durable_value_error_code durable_value_path error error_type "
        "max_attempts messages model model_attempt_id model_step_id observation_id options "
        "provider step tools",
        untrusted_container_keys={"messages", "options", "tools"},
    )
    policies[EventType.CONTEXT_COUNTED] = count_policy
    policies[EventType.CONTEXT_COUNT_FAILED] = count_policy
    pressure_policy = _observed_policy(
        "attempt estimate max_attempts messages model model_attempt_id model_step_id "
        "observation_id options provider step tools",
        untrusted_container_keys={"messages", "options", "tools"},
    )
    policies[EventType.CONTEXT_PRESSURE_ESTIMATED] = pressure_policy
    reconciliation_policy = _observed_policy(
        "actual_input_tokens attempt delta_tokens max_attempts model model_attempt_id "
        "model_step_id observation_id pre_call_count pre_call_estimate provider reconciled "
        "relative_error step"
    )
    policies[EventType.CONTEXT_COUNT_RECONCILED] = reconciliation_policy
    policies[EventType.CONTEXT_PRESSURE_RECONCILED] = reconciliation_policy
    overflow_policy = _observed_policy(
        "error error_type model_attempt_id model_step_id original_message_count phase policy "
        "provider provider_error_code recovery_message_count status_code step"
    )
    policies[EventType.CONTEXT_OVERFLOW_DETECTED] = overflow_policy
    policies[EventType.CONTEXT_OVERFLOW_RECOVERING] = overflow_policy
    policies[EventType.CONTEXT_OVERFLOW_FAILED] = overflow_policy

    knowledge_policy = _observed_policy(
        "durable_value_error_code durable_value_error_path error error_type hit_count "
        "injected_bytes model_step_id policy query query_chars sources tool_call_id "
        "total_hits_known truncated",
        untrusted_container_keys={"sources"},
    )
    for event_type in (
        EventType.KNOWLEDGE_SEARCH_STARTED,
        EventType.KNOWLEDGE_SEARCH_COMPLETED,
        EventType.KNOWLEDGE_SEARCH_FAILED,
        EventType.KNOWLEDGE_INJECTED,
    ):
        policies[event_type] = knowledge_policy

    binding_policy = _observed_policy(
        "binding_cleanup binding_type bound_metadata bound_path bound_snapshot "
        "bound_workspace_id configured_workspace_id environment_factory_release error "
        "error_type factory_allocation_action failures final_snapshot has_bound_runner "
        "has_configured_runner outcome source_workspace_id terminal_outcome",
        untrusted_container_keys={
            "binding_cleanup",
            "bound_metadata",
            "environment_factory_release",
            "failures",
        },
    )
    for event_type in (
        EventType.ENVIRONMENT_BINDING_STARTED,
        EventType.ENVIRONMENT_BINDING_COMPLETED,
        EventType.ENVIRONMENT_BINDING_FAILED,
        EventType.ENVIRONMENT_BINDING_FINALIZE_STARTED,
        EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED,
        EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED,
    ):
        policies[event_type] = binding_policy
    factory_policy = _observed_policy(
        "allocation_id causal_budget_id durable_value_error_code durable_value_error_path "
        "environment_factory_release environment_name error error_type factory_type labels "
        "parent_session_id reconnect_metadata requested_environment_name result_metadata",
        untrusted_container_keys={
            "environment_factory_release",
            "labels",
            "reconnect_metadata",
            "result_metadata",
        },
    )
    for event_type in (
        EventType.ENVIRONMENT_FACTORY_STARTED,
        EventType.ENVIRONMENT_FACTORY_COMPLETED,
        EventType.ENVIRONMENT_FACTORY_FAILED,
    ):
        policies[event_type] = factory_policy

    hook_policy = _observed_policy(
        "actions durable_value_error_code durable_value_error_path error error_type hook_name "
        "phase scope terminal_event_id terminal_event_type tool_call_id tool_name",
        untrusted_container_keys={"actions"},
    )
    for event_type in (
        EventType.HOOK_STARTED,
        EventType.HOOK_COMPLETED,
        EventType.HOOK_FAILED,
    ):
        policies[event_type] = hook_policy

    workflow_policy = _observed_policy(
        "agent attempt_id child_session_id detail gate has_output item_key kind n outcome "
        "passed step_id total workflow",
        untrusted_container_keys={"detail"},
    )
    for event_type in (
        EventType.WORKFLOW_STARTED,
        EventType.WORKFLOW_STEP_STARTED,
        EventType.WORKFLOW_STEP_COMPLETED,
        EventType.WORKFLOW_COMPLETED,
    ):
        policies[event_type] = workflow_policy

    policies[EventType.RUNTIME_SINK_FAILED] = _observed_policy(
        "error error_type event_id event_sequence event_type sink"
    )
    policies[EventType.MEMORY_SEARCH] = _observed_policy(
        "hit_count query results truncated",
        untrusted_container_keys={"results"},
    )
    runner_policy = _observed_policy(
        "adapter command duration_ms error error_type exit_code timed_out",
        untrusted_container_keys={"command"},
    )
    policies[EventType.RUNNER_EXEC_STARTED] = runner_policy
    policies[EventType.RUNNER_EXEC_COMPLETED] = runner_policy

    return policies


EVENT_PAYLOAD_POLICIES: Mapping[EventType, EventPayloadPolicy] = _event_policies()
_INTERNAL_EVENT_PAYLOAD_POLICIES: Mapping[str, EventPayloadPolicy] = {
    WORKFLOW_ATTEMPT_EVENT_TYPE: _observed_policy("attempt_id"),
}

if set(EVENT_PAYLOAD_POLICIES) != set(EventType):
    raise AssertionError("Every built-in event type must have an exact payload policy.")
for _control_event_type, _control_specs in _DECLARED_FIXED_CONTROLS.items():
    _control_policy = EVENT_PAYLOAD_POLICIES[_control_event_type]
    for _control_path in _control_specs:
        if len(_control_path) == 1:
            if _control_path[0] not in _control_policy.owned_keys:
                raise AssertionError(
                    f"Fixed control {_control_event_type}.{_control_path[0]} is not schema-owned."
                )
        elif _control_path not in _control_policy.owned_nested_paths:
            raise AssertionError(
                f"Fixed control {_control_event_type}.{'.'.join(_control_path)} "
                "is not schema-owned."
            )


def event_payload_policy(event_type: EventType | str) -> EventPayloadPolicy:
    if isinstance(event_type, EventType):
        return EVENT_PAYLOAD_POLICIES[event_type]
    return _INTERNAL_EVENT_PAYLOAD_POLICIES.get(event_type, EventPayloadPolicy())


def prepare_new_runtime_event(event: Event, *, redactor: SecretRedactor) -> Event:
    """Validate and redact one event before its first durable append."""

    return _prepare_runtime_event(
        event,
        redactor=redactor,
        validate_budget_payload=True,
    )


def prepare_budget_settlement_event_template(
    event: Event,
    *,
    redactor: SecretRedactor,
) -> Event:
    """Prepare non-durable causal metadata for a future ledger settlement.

    A reservation must retain redacted event metadata before the ledger can
    produce the terminal accounting fields. The ledger later merges its exact
    reconciliation payload into this template, and that resulting event still
    crosses :func:`prepare_new_runtime_event` before its first durable append.
    """

    if type(event) is not Event or event.type not in {
        EventType.BUDGET_RECONCILED,
        EventType.BUDGET_RESERVATION_RELEASED,
    }:
        raise ValueError("Budget settlement templates require a terminal budget event type.")
    return _prepare_runtime_event(
        event,
        redactor=redactor,
        validate_budget_payload=False,
    )


def _prepare_runtime_event(
    event: Event,
    *,
    redactor: SecretRedactor,
    validate_budget_payload: bool,
) -> Event:
    """Apply the common event-owned new-value projection policy."""

    _validate_inputs(event, redactor)
    _validate_new_envelope_authority(event, redactor=redactor)
    policy = event_payload_policy(event.type)
    _validate_fixed_field_types(event, policy=policy)
    if validate_budget_payload:
        _validate_budget_payload_schema(event)
    tool_event_boundary = _recognized_tool_event_boundary(
        event,
        reject_malformed=True,
        trust_persisted_projection=False,
    )
    projection_references = (
        {} if tool_event_boundary is None else tool_event_boundary.projection_references
    )
    _require_no_secret_payload_keys(
        event.payload,
        policy=policy,
        redactor=redactor,
        projection_references=projection_references,
    )
    _reject_secret_authority_values(
        event,
        policy.authority_keys,
        redactor=redactor,
    )
    _reject_secret_nested_authority_values(
        event,
        policy=policy,
        redactor=redactor,
    )
    controls = _recognized_controls(
        event,
        tool_event_boundary=tool_event_boundary,
    )
    redacted_payload = _redact_payload(
        event.payload,
        policy=policy,
        redactor=redactor,
        projection_references=projection_references,
    )
    _restore_runtime_payload_authority(
        event,
        policy=policy,
        redacted_payload=redacted_payload,
    )
    _restore_runtime_nested_payload_authority(
        event,
        policy=policy,
        redacted_payload=redacted_payload,
    )
    _restore_policy_denial_truncation_markers(
        event,
        redacted_payload=redacted_payload,
        redactor=redactor,
    )
    _restore_runtime_tool_result_projection(
        event,
        redacted_payload=redacted_payload,
        references=projection_references,
        redactor=redactor,
    )
    redacted_payload.update(_top_level_controls(controls))
    _restore_nested_controls(
        event,
        redacted_payload=redacted_payload,
        controls=controls,
    )
    _restore_declared_fixed_controls(
        event,
        redacted_payload=redacted_payload,
        reject_malformed=True,
    )
    _synchronize_runtime_tool_result_projection_record(
        redacted_payload,
        controls=controls,
    )
    return _copy_projected_event(
        event,
        event_type=event.type,
        event_id=event.id,
        payload=redacted_payload,
        redactor=redactor,
        redact_session_id=False,
        public_sequence=None,
    )


def project_runtime_event(
    event: Event,
    *,
    sequence: int,
    redactor: SecretRedactor,
    public_authority_alias_codec: PublicAuthorityAliasCodec | None = None,
) -> Event:
    """Project an untrusted or legacy record without granting projection authority."""

    return _project_runtime_event(
        event,
        sequence=sequence,
        redactor=redactor,
        public_authority_alias_codec=public_authority_alias_codec,
        trust_persisted_projection=False,
    )


def project_persisted_runtime_event(
    event: Event,
    *,
    sequence: int,
    redactor: SecretRedactor,
    public_authority_alias_codec: PublicAuthorityAliasCodec | None = None,
) -> Event:
    """Project one durable record for an external consumer.

    The caller retains the original record for claims, accounting, cursor
    advancement, and terminal-lineage decisions.
    """

    return _project_runtime_event(
        event,
        sequence=sequence,
        redactor=redactor,
        public_authority_alias_codec=public_authority_alias_codec,
        trust_persisted_projection=True,
    )


def _project_runtime_event(
    event: Event,
    *,
    sequence: int,
    redactor: SecretRedactor,
    public_authority_alias_codec: PublicAuthorityAliasCodec | None,
    trust_persisted_projection: bool,
) -> Event:
    """Apply the shared public projection with an internal persisted-record capability."""

    _validate_inputs(event, redactor)
    policy = event_payload_policy(event.type)
    tool_event_boundary = _recognized_tool_event_boundary(
        event,
        reject_malformed=False,
        trust_persisted_projection=trust_persisted_projection,
    )
    projection_references = (
        {} if tool_event_boundary is None else tool_event_boundary.projection_references
    )
    controls = _recognized_controls(
        event,
        reject_malformed=False,
        tool_event_boundary=tool_event_boundary,
    )
    resolvable_alias_fields = _resolvable_alias_fields(event, policy=policy)
    redacted_payload = _redact_payload(
        event.payload,
        policy=policy,
        redactor=redactor,
        authority_alias_sequence=sequence,
        resolvable_alias_fields=resolvable_alias_fields,
        public_authority_alias_codec=public_authority_alias_codec,
        envelope_alias_session_id=event.session_id,
        projection_references=projection_references,
    )
    _restore_policy_denial_truncation_markers(
        event,
        redacted_payload=redacted_payload,
        redactor=redactor,
    )
    _remove_malformed_public_controls(
        event,
        policy=policy,
        redacted_payload=redacted_payload,
        controls=controls,
    )
    _restore_runtime_tool_result_projection(
        event,
        redacted_payload=redacted_payload,
        references=projection_references,
        redactor=redactor,
    )
    for key in policy.authority_keys:
        if key not in redacted_payload or redacted_payload[key] is None:
            continue
        if key in policy.public_authority_keys:
            continue
        redacted_payload[key] = (
            public_event_linkage_id(sequence, key)
            if key in resolvable_alias_fields
            else PRIVATE_EVENT_AUTHORITY
        )
    redacted_payload.update(_public_authority_aliases(event, event_sequence=sequence))
    # Only fixed validated discriminators are restored during exposure.
    redacted_payload.update(
        {
            key: value
            for key, value in _top_level_controls(controls).items()
            if key not in policy.authority_keys
        }
    )
    _restore_nested_controls(
        event,
        redacted_payload=redacted_payload,
        controls=controls,
        restore_authority=False,
    )
    _restore_declared_fixed_controls(
        event,
        redacted_payload=redacted_payload,
        reject_malformed=False,
    )
    _synchronize_runtime_tool_result_projection_record(
        redacted_payload,
        controls=controls,
    )
    event_type: EventType | str = event.type
    if not isinstance(event_type, EventType) and redactor.redact_text(str(event_type)) != str(
        event_type
    ):
        event_type = REDACTED_CUSTOM_EVENT_TYPE
    return _copy_projected_event(
        event,
        event_type=event_type,
        event_id=public_event_id(sequence),
        payload=redacted_payload,
        redactor=redactor,
        redact_session_id=True,
        public_sequence=sequence,
        public_authority_alias_codec=public_authority_alias_codec,
    )


def _validate_inputs(event: Event, redactor: SecretRedactor) -> None:
    if type(event) is not Event:
        raise TypeError("Runtime events must be Event instances.")
    if not isinstance(redactor, SecretRedactor):
        raise TypeError("redactor must be a SecretRedactor.")


def _top_level_controls(controls: Mapping[str, Any]) -> dict[str, Any]:
    """Return scalar control keys; dotted keys address validated nested leaves."""

    return {key: value for key, value in controls.items() if "." not in key}


def _validate_new_envelope_authority(event: Event, *, redactor: SecretRedactor) -> None:
    authority_fields = [("session_id", event.session_id, False)]
    if event.id.startswith(PUBLIC_EVENT_ID_PREFIX):
        raise ValueError("New event IDs must not use Cayu's reserved public alias namespace.")
    authority_fields.append(("event_id", event.id, event_id_is_runtime_generated(event)))
    if event.interaction_id is not None:
        authority_fields.append(("interaction_id", event.interaction_id, False))
    if not isinstance(event.type, EventType):
        authority_fields.append(("event_type", str(event.type), False))
    for field_name, value, generated_event_id in authority_fields:
        envelope_field = "session_id" if field_name == "session_id" else field_name
        runtime_generated = generated_event_id or (
            envelope_field in {"session_id", "interaction_id"}
            and event_envelope_authority_is_runtime_generated(
                event,
                field_name=envelope_field,
                value=value,
            )
        )
        if runtime_generated and not redactor.is_exact_secret(value):
            continue
        if redactor.redact_text(value) != value:
            raise ValueError(
                f"event.{field_name} contains a workload secret and cannot be "
                "used as durable event authority."
            )


def _reject_secret_authority_values(
    event: Event,
    authority_keys: Collection[str],
    *,
    redactor: SecretRedactor,
) -> None:
    payload = event.payload
    for field_name in authority_keys:
        if field_name not in payload:
            continue
        value = payload.get(field_name)
        if value is None:
            continue
        if type(value) is not str or not value.strip():
            raise TypeError(f"event.payload.{field_name} must be null or a non-empty string.")
        if _declared_fixed_control_is_valid(event, (field_name,), value):
            continue
        if field_name == "idempotency_key":
            if not _matches_runtime_tool_idempotency_key(event, value):
                raise ValueError(
                    "event.payload.idempotency_key does not match the runtime-owned "
                    "tool execution identity."
                )
            if redactor.is_exact_secret(value):
                raise ValueError(
                    "event.payload.idempotency_key contains a workload secret and cannot "
                    "be used as durable event authority."
                )
            continue
        if redactor.is_exact_secret(value):
            raise ValueError(
                f"event.payload.{field_name} contains a workload secret and cannot "
                "be used as durable event authority."
            )
        if event_payload_authority_is_runtime_generated(
            event,
            field_name=field_name,
            value=value,
        ):
            continue
        if redactor.redact_text(value) != value:
            raise ValueError(
                f"event.payload.{field_name} contains a workload secret and cannot "
                "be used as durable event authority."
            )


def _matches_runtime_tool_idempotency_key(event: Event, value: str) -> bool:
    """Verify content-addressed tool authority from the exact owning event."""

    tool_call_id = event.payload.get("tool_call_id")
    if type(tool_call_id) is not str or not tool_call_id.strip():
        return False
    optional: dict[str, str | None] = {}
    for payload_field, argument_name in (
        ("tool_round_id", "tool_round_id"),
        ("approval_id", "approval_id"),
        ("input_id", "pause_id"),
    ):
        candidate = event.payload.get(payload_field)
        if candidate is not None and (type(candidate) is not str or not candidate.strip()):
            return False
        optional[argument_name] = candidate
    expected = tool_idempotency_key(
        session_id=event.session_id,
        tool_call_id=tool_call_id,
        tool_round_id=optional["tool_round_id"],
        approval_id=optional["approval_id"],
        pause_id=optional["pause_id"],
    )
    return compare_digest(
        value.encode("utf-8", "surrogatepass"),
        expected.encode("utf-8", "surrogatepass"),
    )


def _restore_runtime_payload_authority(
    event: Event,
    *,
    policy: EventPayloadPolicy,
    redacted_payload: dict[str, Any],
) -> None:
    """Preserve only exact, privately attested runtime-owned authority values."""

    for field_name in policy.authority_keys:
        value = event.payload.get(field_name)
        if type(value) is not str:
            continue
        if field_name == "idempotency_key" and _matches_runtime_tool_idempotency_key(
            event,
            value,
        ):
            redacted_payload[field_name] = value
            continue
        if event_payload_authority_is_runtime_generated(
            event,
            field_name=field_name,
            value=value,
        ):
            redacted_payload[field_name] = value


def _restore_runtime_nested_payload_authority(
    event: Event,
    *,
    policy: EventPayloadPolicy,
    redacted_payload: dict[str, Any],
) -> None:
    """Restore only exact nested linkage attested by its runtime producer."""

    def restore(source: Any, projected: Any, path: tuple[str, ...]) -> Any:
        if _path_matches_any(path, policy.nested_authority_paths):
            if type(source) is str and event_nested_payload_authority_is_runtime_generated(
                event,
                path=path,
                value=source,
            ):
                return source
            return projected
        if type(source) is dict and type(projected) is dict:
            for key, child in source.items():
                if key in projected:
                    projected[key] = restore(child, projected[key], (*path, key))
        elif type(source) is list and type(projected) is list:
            for index, (child, projected_child) in enumerate(zip(source, projected, strict=False)):
                projected[index] = restore(child, projected_child, (*path, "*"))
        return projected

    restore(event.payload, redacted_payload, ())


def _reject_secret_nested_authority_values(
    event: Event,
    *,
    policy: EventPayloadPolicy,
    redactor: SecretRedactor,
) -> None:
    def visit(value: Any, path: tuple[str, ...]) -> None:
        if _path_matches_any(path, policy.nested_authority_paths):
            if value is None:
                return
            if type(value) is not str or not value.strip():
                raise TypeError(
                    f"event.payload.{'.'.join(path)} must be null or a non-empty string."
                )
            if _declared_fixed_control_is_valid(event, path, value):
                return
            if path[-1] == "idempotency_key":
                if not _matches_runtime_tool_idempotency_key(event, value):
                    raise ValueError(
                        f"event.payload.{'.'.join(path)} does not match the runtime-owned "
                        "tool execution identity."
                    )
                if redactor.is_exact_secret(value):
                    raise ValueError(
                        f"event.payload.{'.'.join(path)} contains a workload secret and "
                        "cannot be used as durable event authority."
                    )
                return
            if redactor.is_exact_secret(value):
                raise ValueError(
                    f"event.payload.{'.'.join(path)} contains a workload secret and "
                    "cannot be used as durable event authority."
                )
            if event_nested_payload_authority_is_runtime_generated(
                event,
                path=path,
                value=value,
            ):
                return
            if redactor.redact_text(value) != value:
                raise ValueError(
                    f"event.payload.{'.'.join(path)} contains a workload secret and "
                    "cannot be used as durable event authority."
                )
            return
        if type(value) is dict:
            for key, child in cast("dict[str, Any]", value).items():
                visit(child, (*path, key))
        elif type(value) is list:
            for child in value:
                visit(child, (*path, "*"))

    visit(event.payload, ())


def _declared_fixed_control_is_valid(
    event: Event,
    path: tuple[str, ...],
    value: Any,
) -> bool:
    event_type = event.type
    if not isinstance(event_type, EventType):
        return False
    allowed_values = _DECLARED_FIXED_CONTROLS.get(event_type, {}).get(path)
    if allowed_values is None:
        return False
    return any(type(value) is type(allowed) and value == allowed for allowed in allowed_values)


def _validate_fixed_field_types(event: Event, *, policy: EventPayloadPolicy) -> None:
    """Validate fixed scalar controls before they can borrow schema ownership."""

    for field_name in _NON_NEGATIVE_INTEGER_CONTROL_KEYS:
        if field_name not in policy.owned_keys or field_name not in event.payload:
            continue
        value = event.payload[field_name]
        if type(value) is not int or value < 0:
            raise TypeError(f"event.payload.{field_name} must be a non-negative integer.")
    for field_name in _POSITIVE_INTEGER_CONTROL_KEYS:
        if field_name not in policy.owned_keys or field_name not in event.payload:
            continue
        value = event.payload[field_name]
        if field_name == "start_event_sequence" and value is None:
            continue
        if type(value) is not int or value < 1:
            raise TypeError(f"event.payload.{field_name} must be a positive integer.")


def _validate_budget_payload_schema(event: Event) -> None:
    """Validate exact typed accounting containers before preserving schema keys."""

    if event.type == EventType.BUDGET_RESERVED:
        identity = event.payload.get("billing_identity")
        if identity is not None:
            from cayu.core.billing import BillingIdentity

            BillingIdentity.model_validate(identity)
        return
    if event.type in {
        EventType.BUDGET_RECONCILED,
        EventType.BUDGET_RESERVATION_RELEASED,
    }:
        from cayu.runtime.budgets import budget_reconciliation_from_payload

        settlement = {
            field_name: event.payload.get(field_name)
            for field_name in _BUDGET_RECONCILIATION_FIELD_NAMES
        }
        budget_reconciliation_from_payload(settlement)
        return
    if event.type != EventType.MODEL_COMPLETED or "budget_settlements" not in event.payload:
        return
    settlements = event.payload["budget_settlements"]
    if type(settlements) is not list:
        raise TypeError("event.payload.budget_settlements must be a list.")
    from cayu.runtime.budgets import budget_reconciliation_from_payload

    for settlement in settlements:
        budget_reconciliation_from_payload(settlement)


def _redact_payload(
    payload: dict[str, Any],
    *,
    policy: EventPayloadPolicy,
    redactor: SecretRedactor,
    authority_alias_sequence: int | None = None,
    resolvable_alias_fields: Collection[str] = (),
    public_authority_alias_codec: PublicAuthorityAliasCodec | None = None,
    envelope_alias_session_id: str | None = None,
    projection_references: Mapping[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    strict_projection_references = projection_references or {}

    def redact(
        value: Any,
        *,
        inside_untrusted: bool,
        path: tuple[str, ...],
    ) -> Any:
        if authority_alias_sequence is not None and _path_matches_any(
            path,
            policy.nested_authority_paths,
        ):
            if value is None:
                return None
            envelope_alias_field = _ENVELOPE_ALIAS_FIELD_BY_NESTED_PATH.get(path)
            if (
                envelope_alias_field is not None
                and path in policy.envelope_aliased_nested_authority_paths
                and type(value) is str
                and value.strip()
            ):
                if redactor.redact_text(value) == value:
                    return value
                return public_event_envelope_alias(
                    value,
                    field_name=envelope_alias_field,
                    codec=_require_public_authority_alias_codec(public_authority_alias_codec),
                    session_id=(
                        envelope_alias_session_id
                        if envelope_alias_field == "interaction_id"
                        else None
                    ),
                )
            return (
                public_event_linkage_id(authority_alias_sequence, path[-1])
                if _path_matches_any(
                    path,
                    policy.aliased_nested_authority_paths,
                )
                and path[-1] in resolvable_alias_fields
                else PRIVATE_EVENT_AUTHORITY
            )
        if type(value) is dict:
            items: list[tuple[str, Any]] = []
            for key, child in cast("dict[str, Any]", value).items():
                child_path = (*path, key)
                owned = _policy_owns_path(policy, child_path) and (
                    not inside_untrusted or _path_matches_any(child_path, policy.owned_nested_paths)
                )
                public_key = key if owned else redactor.redact_text(key)
                items.append(
                    (
                        public_key,
                        redact(
                            child,
                            inside_untrusted=(
                                inside_untrusted
                                or not owned
                                or _policy_marks_untrusted(policy, child_path)
                            ),
                            path=child_path,
                        ),
                    )
                )
            return collision_safe_json_object(items, preserve_input_order=True)
        if type(value) is list:
            projected_items: list[Any] = []
            for index, item in enumerate(value):
                strict_reference = (
                    strict_projection_references.get(index)
                    if path == ("result", "artifacts")
                    else None
                )
                if type(item) is dict and item == strict_reference:
                    projected_items.append(
                        copy_durable_json_value(
                            strict_reference,
                            "tool_result_projection_reference",
                        )
                    )
                    continue
                projected_items.append(
                    redact(
                        item,
                        inside_untrusted=inside_untrusted,
                        path=(*path, "*"),
                    )
                )
            return projected_items
        if type(value) is str:
            return redactor.redact_text(value)
        return value

    projected = redact(payload, inside_untrusted=False, path=())
    if type(projected) is not dict:
        raise AssertionError("Event payload projection returned a non-object.")
    return cast("dict[str, Any]", projected)


def _require_no_secret_payload_keys(
    payload: dict[str, Any],
    *,
    policy: EventPayloadPolicy,
    redactor: SecretRedactor,
    projection_references: Mapping[int, dict[str, Any]] | None = None,
) -> None:
    """Reject secret-bearing keys outside one exact event schema."""

    strict_projection_references = projection_references or {}

    def visit(
        value: Any,
        *,
        inside_untrusted: bool,
        path: tuple[str, ...],
    ) -> None:
        if type(value) is dict:
            for key, child in cast("dict[str, Any]", value).items():
                child_path = (*path, key)
                owned = _policy_owns_path(policy, child_path) and (
                    not inside_untrusted or _path_matches_any(child_path, policy.owned_nested_paths)
                )
                if not owned and redactor.redact_text(key) != key:
                    public_path = redactor.redact_text(".".join(child_path))
                    raise ValueError(
                        "event.payload contains a workload secret in an object key "
                        f"at {public_path!r}; refusing to publish it."
                    )
                visit(
                    child,
                    inside_untrusted=(
                        inside_untrusted or not owned or _policy_marks_untrusted(policy, child_path)
                    ),
                    path=child_path,
                )
            return
        if type(value) is list:
            for index, child in enumerate(value):
                strict_reference = (
                    strict_projection_references.get(index)
                    if path == ("result", "artifacts")
                    else None
                )
                if type(child) is dict and child == strict_reference:
                    continue
                visit(
                    child,
                    inside_untrusted=inside_untrusted,
                    path=(*path, "*"),
                )

    visit(payload, inside_untrusted=False, path=())


def _policy_owns_path(
    policy: EventPayloadPolicy,
    path: tuple[str, ...],
) -> bool:
    if len(path) == 1:
        return path[0] in policy.owned_keys
    return _path_matches_any(path, policy.owned_nested_paths)


def _path_matches_any(
    path: tuple[str, ...],
    patterns: Collection[tuple[str, ...]],
) -> bool:
    return any(
        len(path) == len(pattern)
        and all(
            expected == "*" or expected == actual
            for actual, expected in zip(path, pattern, strict=True)
        )
        for pattern in patterns
    )


def _policy_marks_untrusted(
    policy: EventPayloadPolicy,
    path: tuple[str, ...],
) -> bool:
    return (len(path) == 1 and path[0] in policy.untrusted_container_keys) or _path_matches_any(
        path, policy.untrusted_container_paths
    )


def _recognized_controls(
    event: Event,
    *,
    tool_event_boundary: _ToolEventBoundary | None,
    reject_malformed: bool = True,
) -> dict[str, Any]:
    event_type = event.type
    expected_interaction_status = (
        _INTERACTION_STATUS_BY_EVENT.get(event_type) if isinstance(event_type, EventType) else None
    )
    if expected_interaction_status is not None:
        status = event.payload.get("status")
        if status == expected_interaction_status:
            controls: dict[str, Any] = {"status": status}
            for field_name in ("started_at", "completed_at"):
                value = event.payload.get(field_name)
                if type(value) is not str:
                    continue
                try:
                    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                except ValueError:
                    continue
                if parsed.tzinfo is not None and parsed.utcoffset() is not None:
                    controls[field_name] = value
            return controls
        if status is not None and reject_malformed:
            raise ValueError(f"Invalid {event.type} status control: {status!r}.")
        return {}
    if event.type in _TOOL_EVENT_TYPES:
        try:
            if tool_event_boundary is None:
                raise AssertionError("Tool event controls require a parsed boundary.")
            if tool_event_boundary.malformed:
                return {}
            controls = dict(tool_event_boundary.controls)
            # Tool names are descriptive data. Linkage controls are restored
            # only after the strict new-write authority check, while public
            # projection filters them below.
            controls.pop("tool_name", None)
            effect = event.payload.get("effect")
            if effect is not None:
                if type(effect) is not str or effect not in {item.value for item in ToolEffect}:
                    raise ValueError("Invalid runtime tool effect control.")
                controls["effect"] = effect
            if event.type == EventType.TOOL_CALL_BLOCKED:
                _recognize_policy_block_controls(
                    event,
                    controls=controls,
                )
            registration_state = event.payload.get("registration_state")
            if registration_state is not None:
                if registration_state != "unregistered_at_policy_plan":
                    raise ValueError("Invalid tool registration_state control.")
                controls["registration_state"] = registration_state
            return controls
        except (TypeError, ValueError):
            if reject_malformed:
                raise
            return {}
    if event.type == EventType.MODEL_COMPLETED:
        controls: dict[str, Any] = {}
        classification = event.payload.get("step_classification")
        classification_type = classification.get("type") if type(classification) is dict else None
        if type(classification_type) is str and classification_type in {
            item.value for item in StepClassificationType
        }:
            controls["step_classification.type"] = classification_type
        elif classification is not None and reject_malformed:
            raise ValueError("Invalid model step_classification control.")
        completion = event.payload.get("completion")
        if completion is not None:
            if type(completion) is not dict:
                if reject_malformed:
                    raise ValueError("Invalid model completion control.")
                return controls
            finish_reason = completion.get("finish_reason")
            end_turn = completion.get("end_turn")
            if type(finish_reason) is not str or finish_reason not in {
                item.value for item in ModelFinishReason
            }:
                if reject_malformed:
                    raise ValueError("Invalid model completion finish_reason control.")
            else:
                controls["completion.finish_reason"] = finish_reason
            if "end_turn" in completion:
                if end_turn is not None and type(end_turn) is not bool:
                    if reject_malformed:
                        raise ValueError("Invalid model completion end_turn control.")
                else:
                    controls["completion.end_turn"] = end_turn
        return controls
    if event.type in {EventType.SESSION_RESUMED, EventType.SESSION_INTERRUPTED}:
        interruption_type = event.payload.get("interruption_type")
        if interruption_type is None:
            return {}
        if type(interruption_type) is str and interruption_type in _INTERRUPTION_TYPES:
            return {"interruption_type": interruption_type}
        if reject_malformed:
            raise ValueError("Invalid session interruption_type control.")
        return {}
    if event.type == EventType.STRUCTURED_OUTPUT_VALIDATING:
        strategy = event.payload.get("strategy")
        if strategy in {"native", "tool"}:
            return {"strategy": strategy}
        if strategy is not None and reject_malformed:
            raise ValueError("Invalid structured-output strategy control.")
    return {}


def _recognized_tool_event_boundary(
    event: Event,
    *,
    reject_malformed: bool,
    trust_persisted_projection: bool,
) -> _ToolEventBoundary | None:
    """Parse runtime tool controls and projection references exactly once."""

    if event.type not in _TOOL_EVENT_TYPES:
        return None
    try:
        controls, references = tool_results.runtime_tool_event_boundary_controls(
            event.payload,
            include_terminal_controls=event.type
            in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED},
        )
    except (TypeError, ValueError):
        if reject_malformed:
            raise
        return _ToolEventBoundary(
            controls={},
            projection_references={},
            malformed=True,
        )
    if "tool_result_projection" not in event_payload_policy(event.type).owned_keys or (
        not trust_persisted_projection
        and not _tool_result_projection_has_runtime_provenance(
            event,
            controls=controls,
        )
    ):
        controls.pop("tool_result_projection", None)
        references = {}
    return _ToolEventBoundary(
        controls=controls,
        projection_references=references,
    )


def _tool_result_projection_has_runtime_provenance(
    event: Event,
    *,
    controls: Mapping[str, Any],
) -> bool:
    """Require in-process attestation before projection data bypasses new-write redaction."""

    record = controls.get("tool_result_projection")
    if type(record) is not dict:
        return False
    policy_id = record.get("policy_id")
    return type(policy_id) is str and event_nested_payload_authority_is_runtime_generated(
        event,
        path=_TOOL_RESULT_PROJECTION_PROVENANCE_PATH,
        value=policy_id,
    )


def _remove_malformed_public_controls(
    event: Event,
    *,
    policy: EventPayloadPolicy,
    redacted_payload: dict[str, Any],
    controls: Mapping[str, Any],
) -> None:
    """Keep malformed legacy controls observable only as non-authoritative data.

    New events reject these shapes before persistence. Legacy records still
    need to be listable and replayable, but a wrong-type or future value must
    not retain the canonical field that downstream consumers recognize as
    protocol authority.
    """

    for field_name in _NON_NEGATIVE_INTEGER_CONTROL_KEYS:
        if field_name not in policy.owned_keys or field_name not in event.payload:
            continue
        value = event.payload[field_name]
        if type(value) is not int or value < 0:
            redacted_payload.pop(field_name, None)
    for field_name in _POSITIVE_INTEGER_CONTROL_KEYS:
        if field_name not in policy.owned_keys or field_name not in event.payload:
            continue
        value = event.payload[field_name]
        if field_name == "start_event_sequence" and value is None:
            continue
        if type(value) is not int or value < 1:
            redacted_payload.pop(field_name, None)

    event_type = event.type
    expected_status = (
        _INTERACTION_STATUS_BY_EVENT.get(event_type) if isinstance(event_type, EventType) else None
    )
    if (
        expected_status is not None
        and "status" in event.payload
        and controls.get("status") != expected_status
    ):
        redacted_payload.pop("status", None)

    if event.type in _TOOL_EVENT_TYPES and "effect" in event.payload and "effect" not in controls:
        redacted_payload.pop("effect", None)

    if (
        event.type == EventType.STRUCTURED_OUTPUT_VALIDATING
        and "strategy" in event.payload
        and "strategy" not in controls
    ):
        redacted_payload.pop("strategy", None)

    if event.type == EventType.MODEL_COMPLETED:
        original_classification = event.payload.get("step_classification")
        projected_classification = redacted_payload.get("step_classification")
        if "step_classification.type" not in controls:
            if type(projected_classification) is dict:
                projected_classification.pop("type", None)
            elif original_classification is not None:
                redacted_payload.pop("step_classification", None)
        original_completion = event.payload.get("completion")
        projected_completion = redacted_payload.get("completion")
        if type(projected_completion) is dict:
            for field_name in ("finish_reason", "end_turn"):
                if (
                    f"completion.{field_name}" not in controls
                    and type(original_completion) is dict
                    and field_name in original_completion
                ):
                    projected_completion.pop(field_name, None)
        elif original_completion is not None:
            redacted_payload.pop("completion", None)

    if (
        event.type in {EventType.SESSION_RESUMED, EventType.SESSION_INTERRUPTED}
        and "interruption_type" in event.payload
        and "interruption_type" not in controls
    ):
        redacted_payload.pop("interruption_type", None)

    if event.type == EventType.TOOL_CALL_BLOCKED:
        for field_name in (
            "blocked_by",
            "decision",
            "denied_by",
            "requested_decision",
        ):
            if field_name in event.payload and field_name not in controls:
                redacted_payload.pop(field_name, None)
        result = redacted_payload.get("result")
        structured = result.get("structured") if type(result) is dict else None
        original_result = event.payload.get("result")
        original_structured = (
            original_result.get("structured") if type(original_result) is dict else None
        )
        if type(structured) is dict and type(original_structured) is dict:
            for field_name in ("decision", "error"):
                if (
                    f"result.structured.{field_name}" not in controls
                    and field_name in original_structured
                ):
                    structured.pop(field_name, None)

    if (
        event.type in _TOOL_EVENT_TYPES
        and "registration_state" in event.payload
        and "registration_state" not in controls
    ):
        redacted_payload.pop("registration_state", None)

    if event.type not in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}:
        return
    for field_name in _TERMINAL_CONTROL_KEYS:
        if field_name not in controls:
            redacted_payload.pop(field_name, None)
    result = redacted_payload.get("result")
    if type(result) is not dict:
        return
    structured = result.get("structured")
    if type(structured) is not dict:
        return
    for field_name in _TERMINAL_CONTROL_KEYS:
        if field_name not in controls:
            structured.pop(field_name, None)


def _public_authority_aliases(
    event: Event,
    *,
    event_sequence: int,
) -> dict[str, str]:
    """Derive public aliases only from positive durable sequence evidence."""

    if event.type == EventType.SERVER_MUTATION_ACCEPTED:
        sequence = event.payload.get("accepted_event_sequence")
        if type(sequence) is int and sequence >= 1:
            return {"accepted_event_id": public_event_id(sequence)}
    if event.type == EventType.RUNTIME_SINK_FAILED:
        sequence = event.payload.get("event_sequence")
        if type(sequence) is int and sequence >= 1:
            return {"event_id": public_event_id(sequence)}
    if event.type in _INTERACTION_STATUS_BY_EVENT:
        sequence = event.payload.get("start_event_sequence")
        if event.type == EventType.INTERACTION_STARTED and sequence is None:
            sequence = event_sequence
        if type(sequence) is int and sequence >= 1:
            return {"start_event_id": public_event_id(sequence)}
    return {}


def _restore_runtime_tool_result_projection(
    event: Event,
    *,
    redacted_payload: dict[str, Any],
    references: Mapping[int, dict[str, Any]],
    redactor: SecretRedactor,
) -> None:
    """Rebuild model-facing text while preserving one validated runtime reference."""

    if not references:
        return
    original_result = event.payload.get("result")
    redacted_result = redacted_payload.get("result")
    if type(original_result) is not dict or type(redacted_result) is not dict:
        raise AssertionError("Validated projection lost its result object.")
    projected_content, projected_artifacts = tool_results.project_runtime_tool_result_for_boundary(
        original_content=original_result.get("content"),
        redacted_artifacts=redacted_result.get("artifacts"),
        references=dict(references),
        redactor=redactor,
    )
    redacted_result["content"] = projected_content
    redacted_result["artifacts"] = projected_artifacts


def _synchronize_runtime_tool_result_projection_record(
    payload: dict[str, Any],
    *,
    controls: Mapping[str, Any],
) -> None:
    """Keep validated projection evidence aligned with boundary-redacted content."""

    if type(controls.get("tool_result_projection")) is not dict:
        return
    result = payload.get("result")
    record = payload.get("tool_result_projection")
    if type(result) is not dict or type(record) is not dict:
        raise AssertionError("Validated projection lost its result or evidence record.")
    content = result.get("content")
    if type(content) is not str:
        raise AssertionError("Validated projection lost its content.")
    record["projected_bytes"] = len(content.encode("utf-8"))
    (
        record["projected_token_estimate"],
        record["token_estimation_method"],
    ) = reestimate_tool_result_projection_tokens(
        content,
        token_estimation_method=record.get("token_estimation_method"),
    )


def _restore_policy_denial_truncation_markers(
    event: Event,
    *,
    redacted_payload: dict[str, Any],
    redactor: SecretRedactor,
) -> None:
    if event.type != EventType.TOOL_CALL_BLOCKED or "denied_by" not in event.payload:
        return
    reason = event.payload.get("reason")
    if type(reason) is str:
        redacted_payload["reason"] = _redact_policy_denial_text(
            reason,
            redactor=redactor,
        )
    original_result = event.payload.get("result")
    projected_result = redacted_payload.get("result")
    if type(original_result) is not dict or type(projected_result) is not dict:
        return
    content = original_result.get("content")
    if type(content) is str:
        projected_result["content"] = _redact_policy_denial_text(
            content,
            redactor=redactor,
        )
    original_structured = original_result.get("structured")
    projected_structured = projected_result.get("structured")
    if type(original_structured) is not dict or type(projected_structured) is not dict:
        return
    structured_reason = original_structured.get("reason")
    if type(structured_reason) is str:
        projected_structured["reason"] = _redact_policy_denial_text(
            structured_reason,
            redactor=redactor,
        )


def _redact_policy_denial_text(value: str, *, redactor: SecretRedactor) -> str:
    if not value.endswith(_POLICY_DENIAL_TRUNCATION_MARKER):
        return redactor.redact_text(value)
    prefix = value[: -len(_POLICY_DENIAL_TRUNCATION_MARKER)]
    return redactor.redact_text(prefix) + _POLICY_DENIAL_TRUNCATION_MARKER


def _restore_nested_controls(
    event: Event,
    *,
    redacted_payload: dict[str, Any],
    controls: dict[str, Any],
    restore_authority: bool = True,
) -> None:
    classification_type = controls.get("step_classification.type")
    if event.type == EventType.MODEL_COMPLETED and type(classification_type) is str:
        classification = redacted_payload.get("step_classification")
        if type(classification) is dict:
            classification["type"] = classification_type

    if event.type == EventType.MODEL_COMPLETED:
        completion = redacted_payload.get("completion")
        if type(completion) is dict:
            for field_name in ("finish_reason", "end_turn"):
                control_key = f"completion.{field_name}"
                if control_key in controls:
                    completion[field_name] = controls[control_key]

    if event.type == EventType.TOOL_CALL_BLOCKED:
        result = redacted_payload.get("result")
        structured = result.get("structured") if type(result) is dict else None
        if type(structured) is dict:
            for field_name in ("decision", "error"):
                control_key = f"result.structured.{field_name}"
                if control_key in controls:
                    structured[field_name] = controls[control_key]

    terminal_controls = {
        key: value for key, value in controls.items() if key in _TERMINAL_CONTROL_KEYS
    }
    if not restore_authority:
        terminal_controls = {
            key: value
            for key, value in terminal_controls.items()
            if key not in event_payload_policy(event.type).authority_keys
        }
    if not terminal_controls:
        return
    result = redacted_payload.get("result")
    if type(result) is not dict:
        return
    structured = result.get("structured")
    if type(structured) is dict:
        structured.update(terminal_controls)


def _restore_declared_fixed_controls(
    event: Event,
    *,
    redacted_payload: dict[str, Any],
    reject_malformed: bool,
) -> None:
    """Restore only literal controls proven by an exact event-type schema."""

    event_type = event.type
    if not isinstance(event_type, EventType):
        return
    specs = _DECLARED_FIXED_CONTROLS.get(event_type, {})
    for path, allowed_values in specs.items():
        allowed_types = {type(value) for value in allowed_values}
        _restore_fixed_control_path(
            original=event.payload,
            projected=redacted_payload,
            path=path,
            allowed_values=allowed_values,
            allowed_types=allowed_types,
            public_path="event.payload." + ".".join(path),
            reject_malformed=reject_malformed,
            preserve_unknown=(event_type, path) in _EXTENSIBLE_FIXED_CONTROLS,
        )


def _restore_fixed_control_path(
    *,
    original: Any,
    projected: Any,
    path: tuple[str, ...],
    allowed_values: Collection[Any],
    allowed_types: Collection[type[Any]],
    public_path: str,
    reject_malformed: bool,
    preserve_unknown: bool,
) -> None:
    if not path:
        raise AssertionError("Fixed control path must not be empty.")
    field_name, *remaining = path
    if field_name == "*":
        if type(original) is not list or type(projected) is not list:
            if original is not None and reject_malformed:
                raise ValueError(f"{public_path} has an invalid container.")
            return
        for original_item, projected_item in zip(original, projected, strict=True):
            _restore_fixed_control_path(
                original=original_item,
                projected=projected_item,
                path=tuple(remaining),
                allowed_values=allowed_values,
                allowed_types=allowed_types,
                public_path=public_path,
                reject_malformed=reject_malformed,
                preserve_unknown=preserve_unknown,
            )
        return
    if type(original) is not dict or type(projected) is not dict or field_name not in original:
        return
    if remaining:
        child = original[field_name]
        expected_container = list if remaining[0] == "*" else dict
        if child is not None and type(child) is not expected_container:
            if reject_malformed:
                raise ValueError(f"{public_path} has an invalid container.")
            projected.pop(field_name, None)
            return
        _restore_fixed_control_path(
            original=child,
            projected=projected.get(field_name),
            path=tuple(remaining),
            allowed_values=allowed_values,
            allowed_types=allowed_types,
            public_path=public_path,
            reject_malformed=reject_malformed,
            preserve_unknown=preserve_unknown,
        )
        return
    value = original[field_name]
    if type(value) in allowed_types and value in allowed_values:
        projected[field_name] = value
        return
    if preserve_unknown:
        if type(value) in allowed_types:
            return
        if reject_malformed:
            raise TypeError(f"{public_path} has an invalid type.")
        projected.pop(field_name, None)
        return
    if reject_malformed:
        raise ValueError(f"Invalid fixed control at {public_path}.")
    projected.pop(field_name, None)


def _recognize_policy_block_controls(
    event: Event,
    *,
    controls: dict[str, Any],
) -> None:
    denied_by = event.payload.get("denied_by")
    blocked_by = event.payload.get("blocked_by")
    decision = event.payload.get("decision")
    requested_decision = event.payload.get("requested_decision")
    if denied_by is not None:
        allowed_decisions = _POLICY_DENIAL_DECISIONS.get(denied_by)
        if denied_by == _TOOL_POLICY_DENIAL_SOURCE and blocked_by == "tool_policy_reauthorization":
            allowed_decisions = frozenset({"deny", "require_approval"})
        if allowed_decisions is None or decision not in allowed_decisions:
            raise ValueError("Invalid policy-denial classification controls.")
        if blocked_by is not None:
            if blocked_by != "tool_policy_reauthorization":
                raise ValueError("Invalid policy-denial blocked_by control.")
            controls["blocked_by"] = blocked_by
        controls["denied_by"] = denied_by
        controls["decision"] = decision
        result = event.payload.get("result")
        structured = result.get("structured") if type(result) is dict else None
        if type(structured) is dict:
            structured_decision = structured.get("decision")
            if structured_decision is not None:
                if structured_decision != decision:
                    raise ValueError("Policy-denial result decision conflicts with its event.")
                controls["result.structured.decision"] = structured_decision
            structured_error = structured.get("error")
            if structured_error is not None:
                if denied_by != _COMMAND_POLICY_DENIAL_SOURCE:
                    raise ValueError("Only command-policy denials own an error control.")
                if structured_error not in _POLICY_DENIAL_ERRORS:
                    raise ValueError("Invalid command-policy denial error control.")
                controls["result.structured.error"] = structured_error
        return
    if blocked_by == "policy_evaluation_ambiguous":
        if decision != "ambiguous" or requested_decision not in {"approve", "deny"}:
            raise ValueError("Invalid ambiguous policy-evaluation controls.")
        controls["blocked_by"] = blocked_by
        controls["decision"] = decision
        controls["requested_decision"] = requested_decision
        return
    if blocked_by in {"before_tool_call_hook", "tool_policy_reauthorization"}:
        if decision is not None or requested_decision is not None:
            raise ValueError("Hook-origin tool blocks cannot assert a policy decision.")
        controls["blocked_by"] = blocked_by


def _copy_projected_event(
    event: Event,
    *,
    event_type: EventType | str,
    event_id: str,
    payload: dict[str, Any],
    redactor: SecretRedactor,
    redact_session_id: bool,
    public_sequence: int | None,
    public_authority_alias_codec: PublicAuthorityAliasCodec | None = None,
) -> Event:
    if redact_session_id and (type(public_sequence) is not int or public_sequence < 1):
        raise ValueError("Public event projection requires a positive durable sequence.")
    projected_session_id = event.session_id
    projected_interaction_id = event.interaction_id
    if redact_session_id:
        if redactor.redact_text(event.session_id) != event.session_id:
            projected_session_id = public_event_envelope_alias(
                event.session_id,
                field_name="session_id",
                codec=_require_public_authority_alias_codec(public_authority_alias_codec),
            )
        if (
            event.interaction_id is not None
            and redactor.redact_text(event.interaction_id) != event.interaction_id
        ):
            projected_interaction_id = public_event_envelope_alias(
                event.interaction_id,
                field_name="interaction_id",
                codec=_require_public_authority_alias_codec(public_authority_alias_codec),
                session_id=event.session_id,
            )
    projected = copy_event(event).model_copy(
        update={
            "type": event_type,
            "session_id": projected_session_id,
            "interaction_id": projected_interaction_id,
            "id": event_id,
            "agent_name": (
                None if event.agent_name is None else redactor.redact_text(event.agent_name)
            ),
            "environment_name": (
                None
                if event.environment_name is None
                else redactor.redact_text(event.environment_name)
            ),
            "workflow_name": (
                None if event.workflow_name is None else redactor.redact_text(event.workflow_name)
            ),
            "tool_name": (
                None if event.tool_name is None else redactor.redact_text(event.tool_name)
            ),
            "payload": payload,
        },
        deep=True,
    )
    if not redact_session_id:
        return projected
    # Public Event objects must not retain raw authority in Pydantic private
    # attributes. A model dump is insufficient protection because third-party
    # sinks receive the live object and may inspect ``__pydantic_private__``.
    return Event.model_validate(projected.model_dump(mode="python"))
