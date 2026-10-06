"""Shared event payload schemas and durable linkage evidence."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, replace
from typing import Any

from cayu.artifacts.settlement import ArtifactWriteSettlementEvidence
from cayu.egress.authority import EgressAuthorityTransitionState
from cayu.events import SESSION_EXPORT_EVENT_FIELDS, SESSION_EXPORT_EVENT_TYPES, Event, EventType
from cayu.providers._retry_decision import RetryDecision
from cayu.tools._shared_artifact_result_schema import (
    SHARED_ARTIFACT_RESULT_AUTHORITY_FIELD,
    SHARED_ARTIFACT_RESULT_EVENT_SCHEMA_PATHS,
)
from cayu.tools._web_access_result_schema import (
    WEB_ACCESS_RESULT_AUTHORITY_FIELD,
    WEB_ACCESS_RESULT_EVENT_SCHEMA_PATHS,
)
from cayu.workflows.base import WORKFLOW_ATTEMPT_EVENT_TYPE
from cayu.workspaces._revision_records import (
    _WORKSPACE_PATH_REVISION_AUTHORITY_FIELDS,
    _WORKSPACE_PATH_REVISION_DELTA_AUTHORITY_FIELDS,
    _WORKSPACE_PATH_REVISION_DELTA_FIELDS,
    _WORKSPACE_PATH_REVISION_FIELDS,
)

_MODEL_POLICY_PATHS = frozenset(
    [
        ("model_policy", key)
        for key in (
            "scope",
            "incarnation_id",
            "incarnation_epoch",
            "installation_id",
            "installation_seq",
            "action",
            "snapshot",
            "model",
            "provider_name",
        )
    ]
    + [
        ("model_policy", "scope", key)
        for key in (
            "organization_id",
            "cloud_agent_id",
            "application_id",
            "instance_id",
            "integration_id",
            "credential_family_id",
            "inference_key_id",
        )
    ]
    + [
        ("model_policy", "snapshot", key)
        for key in ("snapshot_id", "effective_revision", "config_sha256")
    ]
)


@dataclass(frozen=True, slots=True)
class EventPayloadPolicy:
    """Structure and authority owned by one exact runtime event type."""

    owned_keys: frozenset[str] = frozenset()
    owned_nested_paths: frozenset[tuple[str, ...]] = frozenset()
    authority_keys: frozenset[str] = frozenset()
    internal_authority_keys: frozenset[str] = frozenset()
    internal_keys: frozenset[str] = frozenset()
    exact_internal_keys: frozenset[str] = frozenset()
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
        if not self.internal_authority_keys <= self.authority_keys:
            raise ValueError("Internal event authority keys must also be authority keys.")
        if not self.internal_keys <= self.owned_keys:
            raise ValueError("Internal event keys must also be owned keys.")
        if not self.exact_internal_keys <= self.internal_keys:
            raise ValueError("Exact internal event keys must also be internal keys.")
        if self.internal_keys & self.authority_keys:
            raise ValueError("Internal event keys and authority keys must be disjoint.")
        if self.internal_authority_keys & (
            self.public_authority_keys | self.aliased_authority_keys
        ):
            raise ValueError("Internal event authority cannot be public or aliased.")
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


_MODEL_EXECUTION_AUTHORITY_KEYS = frozenset(
    {
        "execution_profile_fingerprint",
        "model_attempt_id",
        "model_step_id",
        "tool_round_id",
    }
)


_TOOL_LINKAGE_AUTHORITY_KEYS = frozenset(
    {
        "approval_id",
        "execution_profile_fingerprint",
        "idempotency_key",
        "input_id",
        "model_attempt_id",
        "model_step_id",
        "task_id",
        "tool_call_id",
        "tool_round_id",
    }
)


_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS = frozenset({"execution_profile_fingerprint"})


_TOOL_TERMINAL_TIMING_KEYS = frozenset(
    {
        "tool_effect_completed_at",
        "tool_terminal_staged_at",
        "tool_terminal_publication_started_at",
    }
)


_TOOL_EXPOSURE_PUBLIC_AUTHORITY_KEYS = frozenset({"exposure_fingerprint", "profile_id"})


_TOOL_EXPOSURE_RECORD_PUBLIC_AUTHORITY_KEYS = _TOOL_EXPOSURE_PUBLIC_AUTHORITY_KEYS | {
    "catalogue_revision"
}


_TARGETED_TOOL_GRANT_PUBLIC_AUTHORITY_KEYS = frozenset(
    {
        "arguments_sha256",
        "catalogue_revision",
        "descriptor_version",
        "generation_id",
        "grant_id",
        "rejection_id",
        "request_id",
        "schema_fingerprint",
        "tool_id",
        "use_id",
    }
)


_TARGETED_TOOL_INVOCATION_PUBLIC_AUTHORITY_KEYS = frozenset(
    {
        "arguments_sha256",
        "catalogue_revision",
        "descriptor_version",
        "dispatch_kind",
        "effective_tool_id",
        "grant_id",
        "invocation_id",
        "model_tool_name",
        "schema_fingerprint",
        "use_id",
    }
)


_EGRESS_AUTHORITY_EVENT_STATES = {
    EventType.EGRESS_AUTHORITY_REQUESTED: EgressAuthorityTransitionState.AUTHORIZED.value,
    EventType.EGRESS_AUTHORITY_AUTHORIZED: EgressAuthorityTransitionState.AUTHORIZED.value,
    EventType.EGRESS_AUTHORITY_INSTALLING: EgressAuthorityTransitionState.INSTALLING.value,
    EventType.EGRESS_AUTHORITY_ACTIVATED: EgressAuthorityTransitionState.ACTIVE.value,
    EventType.EGRESS_AUTHORITY_REFUSED: EgressAuthorityTransitionState.REFUSED.value,
    EventType.EGRESS_AUTHORITY_AMBIGUOUS: EgressAuthorityTransitionState.AMBIGUOUS.value,
}


_EGRESS_AUTHORITY_EVENT_TYPES = frozenset(_EGRESS_AUTHORITY_EVENT_STATES)


def _policy(
    *owned_keys: str,
    owned_nested_paths: Collection[tuple[str, ...]] = (),
    authority_keys: Collection[str] = (),
    internal_authority_keys: Collection[str] = (),
    internal_keys: Collection[str] = (),
    exact_internal_keys: Collection[str] = (),
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
        internal_authority_keys=frozenset(internal_authority_keys),
        internal_keys=frozenset(internal_keys),
        exact_internal_keys=frozenset(exact_internal_keys),
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
    internal_authority_keys: Collection[str] = (),
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
        internal_authority_keys=internal_authority_keys,
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
        ("input_coverage", "transcript_cursor"),
        ("input_coverage", "message_count"),
        ("input_coverage", "messages_sha256"),
        ("context_pressure", "estimated_tool_schema_input_tokens"),
        ("context_pressure", "estimated_structured_output_input_tokens"),
        ("context_pressure", "estimated_request_options_input_tokens"),
        ("context_pressure", "estimated_request_overhead_input_tokens"),
    }
)


_REQUEST_SIZE_FIELD_NAMES = frozenset({"characters", "utf8_bytes", "canonical_json_bytes"})


_REQUEST_FINGERPRINT_FIELD_NAMES = frozenset(
    {
        "availability",
        "value",
        "algorithm",
        "key_id",
        "canonicalization_version",
        "unavailable_reason",
    }
)


_PROMPT_CONTRIBUTION_MANIFEST_NESTED_PATHS = frozenset(
    {
        ("prompt_contribution_manifest", "schema_version"),
        ("prompt_contribution_manifest", "system"),
        ("prompt_contribution_manifest", "system", "count"),
        ("prompt_contribution_manifest", "system", "size"),
        ("prompt_contribution_manifest", "system_fingerprint"),
        ("prompt_contribution_manifest", "contributions"),
        ("prompt_contribution_manifest", "contributions", "*", "kind"),
        ("prompt_contribution_manifest", "contributions", "*", "size"),
        ("prompt_contribution_manifest", "contributions", "*", "fingerprint"),
    }
    | {
        ("prompt_contribution_manifest", "system", "size", field_name)
        for field_name in _REQUEST_SIZE_FIELD_NAMES
    }
    | {
        ("prompt_contribution_manifest", "system_fingerprint", field_name)
        for field_name in _REQUEST_FINGERPRINT_FIELD_NAMES
    }
    | {
        ("prompt_contribution_manifest", "contributions", "*", "size", field_name)
        for field_name in _REQUEST_SIZE_FIELD_NAMES
    }
    | {
        (
            "prompt_contribution_manifest",
            "contributions",
            "*",
            "fingerprint",
            field_name,
        )
        for field_name in _REQUEST_FINGERPRINT_FIELD_NAMES
    }
)


_REQUEST_FOOTPRINT_NESTED_PATHS = frozenset(
    {
        ("total", "count"),
        ("total", "size"),
        ("messages", "count"),
        ("messages", "system"),
        ("messages", "system", "count"),
        ("messages", "system", "size"),
        ("messages", "groups"),
        ("messages", "groups", "*", "role"),
        ("messages", "groups", "*", "part_type"),
        ("messages", "groups", "*", "count"),
        ("messages", "groups", "*", "size"),
        ("messages", "size"),
        ("tools", "count"),
        ("tools", "size"),
        ("attachments", "count"),
        ("attachments", "source_bytes"),
        ("attachments", "groups"),
        ("attachments", "groups", "*", "kind"),
        ("attachments", "groups", "*", "count"),
        ("attachments", "groups", "*", "source_bytes"),
        ("options", "known_categories"),
        ("options", "unknown_count"),
        ("options", "size"),
        ("prompt_contributions", "availability"),
        ("prompt_contributions", "contributions"),
        ("prompt_contributions", "contributions", "*", "kind"),
        ("prompt_contributions", "contributions", "*", "size"),
        ("prompt_contributions", "contributions", "*", "fingerprint"),
        ("prompt_contributions", "unavailable_reason"),
        ("structured_output", "count"),
        ("structured_output", "size"),
        ("cache_breakpoints", "*", "kind"),
        ("cache_breakpoints", "*", "ttl"),
        ("cache_breakpoints", "*", "fingerprint"),
        ("tool_exposure", "profile_id"),
        ("tool_exposure", "exposure_fingerprint"),
        ("tool_exposure", "registered_count"),
        ("tool_exposure", "ceiling_count"),
        ("tool_exposure", "exposed_count"),
        ("tool_exposure", "profile_changed"),
        ("targeted_tool_grants", "schema_version"),
        ("targeted_tool_grants", "projection"),
        ("targeted_tool_grants", "native_marker_id"),
        ("targeted_tool_grants", "generation_id"),
        ("targeted_tool_grants", "catalogue_revision"),
        ("targeted_tool_grants", "grant_count"),
        ("targeted_tool_grants", "grant_ids"),
        ("targeted_tool_grants", "tool_ids"),
        ("targeted_tool_grants", "max_calls"),
        ("targeted_tool_grants", "used_calls"),
        ("targeted_tool_grants", "remaining_calls"),
        ("targeted_tool_grants", "direct_tool_prefix_changed"),
        ("tool_discovery_view", "generation_id"),
        ("tool_discovery_view", "revision"),
        ("tool_discovery_view", "catalogue_revision"),
        ("tool_discovery_view", "ceiling_fingerprint"),
        ("tool_discovery_view", "grant_count"),
        ("tool_discovery_projection", "protocol"),
        ("tool_discovery_projection", "candidate_count"),
        ("tool_discovery_projection", "loaded_count"),
        ("tool_discovery_projection", "generation_id"),
    }
    | {
        (*prefix, field_name)
        for prefix in (
            ("total", "size"),
            ("messages", "system", "size"),
            ("messages", "groups", "*", "size"),
            ("messages", "size"),
            ("tools", "size"),
            ("options", "size"),
            ("structured_output", "size"),
        )
        for field_name in _REQUEST_SIZE_FIELD_NAMES
    }
    | {
        ("component_tokens", field_name)
        for field_name in {
            "method",
            "confidence",
            "total_input_tokens",
            "system_message_input_tokens",
            "non_system_message_input_tokens",
            "tool_schema_input_tokens",
            "structured_output_input_tokens",
            "attachment_input_tokens",
            "request_options_input_tokens",
        }
    }
    | {
        ("context_pressure", field_name)
        for field_name in {
            "method",
            "confidence",
            "observed_context_input_tokens",
            "estimated_delta_input_tokens",
            "estimated_message_input_tokens",
            "estimated_tool_schema_input_tokens",
            "estimated_structured_output_input_tokens",
            "estimated_request_options_input_tokens",
            "estimated_request_overhead_input_tokens",
            "previous_request_overhead_input_tokens",
            "estimated_request_overhead_delta_tokens",
            "estimated_attachment_input_tokens",
            "estimated_context_input_tokens",
            "reserved_output_tokens",
            "estimated_context_window_tokens",
            "provider_count_input_tokens",
            "provider_count_context_window_tokens",
            "anchor_transcript_cursor",
            "current_transcript_cursor",
            "estimated_message_count",
            "chars_per_token",
            "json_chars_per_token",
            "binary_bytes_per_token",
        }
    }
    | {
        ("fingerprints", fingerprint_name)
        for fingerprint_name in {
            "provider_neutral_request",
            "provider_wire_request",
            "system",
            "tool_manifest",
            "conversation_prefix",
        }
    }
    | {
        ("fingerprints", fingerprint_name, field_name)
        for fingerprint_name in {
            "provider_neutral_request",
            "provider_wire_request",
            "system",
            "tool_manifest",
            "conversation_prefix",
        }
        for field_name in _REQUEST_FINGERPRINT_FIELD_NAMES
    }
    | {
        ("cache_breakpoints", "*", "fingerprint", field_name)
        for field_name in _REQUEST_FINGERPRINT_FIELD_NAMES
    }
    | {
        ("prompt_contributions", "contributions", "*", "size", field_name)
        for field_name in _REQUEST_SIZE_FIELD_NAMES
    }
    | {
        ("prompt_contributions", "contributions", "*", "fingerprint", field_name)
        for field_name in _REQUEST_FINGERPRINT_FIELD_NAMES
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
        ("result", "structured", "durable_value_error_limit"),
        ("result", "structured", "durable_value_error_observed_lower_bound"),
        ("result", "structured", "isolated_tool_failure_code"),
        ("result", "structured", "isolated_tool_cleanup_failure_code"),
        ("result", "structured", "tool_execution_boundary"),
        ("result", "structured", "tool_timeout_strength"),
    }
)


_TOOL_RESULT_PROJECTION_RECORD_FIELDS = frozenset(
    {
        "artifact_id",
        "artifact_sha256",
        "artifact_write_settlement",
        "failure_type",
        "store_id_bytes",
        "store_id_max_bytes",
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


_ARTIFACT_WRITE_SETTLEMENT_RECORD_FIELDS = frozenset(ArtifactWriteSettlementEvidence.model_fields)


_TOOL_RESULT_PROJECTION_RECORD_NESTED_PATHS = (
    frozenset(
        ("tool_result_projection", field_name)
        for field_name in _TOOL_RESULT_PROJECTION_RECORD_FIELDS
    )
    | frozenset(
        ("tool_result_projection", "artifact_write_settlement", field_name)
        for field_name in _ARTIFACT_WRITE_SETTLEMENT_RECORD_FIELDS
    )
    | {
        ("tool_result_projection", "artifact_write_settlement", "failure_codes", "*"),
    }
)


_TOOL_EVENT_NESTED_PATHS = (
    _TOOL_RESULT_NESTED_PATHS
    | _TOOL_RESULT_PROJECTION_RECORD_NESTED_PATHS
    | WEB_ACCESS_RESULT_EVENT_SCHEMA_PATHS
    | SHARED_ARTIFACT_RESULT_EVENT_SCHEMA_PATHS
)


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
        "arguments_state",
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
        "arguments_state",
        "budget_limits",
        "environment_name",
        "execution_profile_fingerprint",
        "expires_at",
        "limits",
        "max_steps",
        "metadata",
        "model_attempt_id",
        "model_step_id",
        "publish_arguments",
        "reason",
        "retry_policy",
        "secret_resolution_scope",
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
        "arguments_state",
        "assistant_message_state",
        "assistant_publication",
        "budget_limits",
        "environment_name",
        "input_id",
        "limits",
        "max_steps",
        "model_attempt_id",
        "model_step",
        "model_step_id",
        "options",
        "question",
        "quarantined_assistant_message",
        "retry_policy",
        "schema_version",
        "source_run_epoch",
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


_USER_INPUT_SUPERSESSION_FIELD_NAMES = frozenset(
    {
        "schema_version",
        "session_id",
        "session_instance_id",
        "source_interaction_id",
        "source_run_epoch",
        "input_id",
        "tool_call_id",
        "tool_round_id",
        "model_step_id",
        "model_attempt_id",
        "execution_profile_fingerprint",
        "pause_digest",
        "state",
        "claim_run_epoch",
        "resolution_request_digest",
    }
)


_USER_INPUT_SUPERSESSION_NESTED_SCHEMA_PATHS = frozenset(
    ("user_input_supersession_intent", field_name)
    for field_name in _USER_INPUT_SUPERSESSION_FIELD_NAMES
)


_USER_INPUT_SUPERSESSION_NESTED_AUTHORITY_PATHS = frozenset(
    ("user_input_supersession_intent", field_name)
    for field_name in {
        "session_id",
        "session_instance_id",
        "source_interaction_id",
        "input_id",
        "tool_call_id",
        "tool_round_id",
        "model_step_id",
        "model_attempt_id",
        "execution_profile_fingerprint",
        "pause_digest",
        "state",
        "resolution_request_digest",
    }
)


_AMBIGUOUS_USER_INPUT_SUPERSESSION_FIELD_NAMES = frozenset(
    {
        "schema_version",
        "session_id",
        "session_instance_id",
        "source_checkpoint_digest",
        "state",
    }
)


_AMBIGUOUS_USER_INPUT_SUPERSESSION_NESTED_SCHEMA_PATHS = frozenset(
    ("ambiguous_user_input_supersession_intent", field_name)
    for field_name in _AMBIGUOUS_USER_INPUT_SUPERSESSION_FIELD_NAMES
)


_AMBIGUOUS_USER_INPUT_SUPERSESSION_NESTED_AUTHORITY_PATHS = frozenset(
    ("ambiguous_user_input_supersession_intent", field_name)
    for field_name in {
        "session_id",
        "session_instance_id",
        "source_checkpoint_digest",
        "state",
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


def _egress_authority_event_nested_paths() -> frozenset[tuple[str, ...]]:
    """Return every schema-owned path in bounded transition evidence."""

    paths: set[tuple[str, ...]] = set()
    for container_name in ("from_authority", "to_authority"):
        paths.update(
            (container_name, field_name)
            for field_name in {
                "schema_version",
                "generation",
                "fingerprint",
                "authority_source",
                "authority_scope",
                "policy_version",
                "runner_kind",
                "cutover_strategy",
                "comparison_available",
                "policies",
                "bindings",
            }
        )
        paths.update(
            (container_name, "policies", "*", field_name)
            for field_name in {
                "name",
                "kind",
                "allowed_destinations",
                "operations",
                "denied_path_prefixes",
                "comparison_available",
            }
        )
        paths.update(
            {
                (container_name, "policies", "*", "allowed_destinations", "*"),
                (container_name, "policies", "*", "denied_path_prefixes", "*"),
            }
        )
        paths.update(
            (container_name, "policies", "*", "operations", "*", field_name)
            for field_name in {"method", "path", "match"}
        )
        paths.update(
            (container_name, "bindings", "*", field_name)
            for field_name in {
                "destination",
                "policy_name",
                "credential_kind",
                "credential_authority_fingerprint",
            }
        )
    paths.update(("actor", field_name) for field_name in _RESOLUTION_ACTOR_NESTED_FIELD_NAMES)
    paths.update(
        ("receipt", field_name)
        for field_name in {
            "record_type",
            "schema_version",
            "state",
            "from_fingerprint",
            "to_fingerprint",
            "from_generation",
            "to_generation",
            "runner_kind",
            "strategy",
            "environment_fingerprint",
            "same_allocation",
            "workspace_continuity_verified",
            "old_authority_revoked",
            "old_path_closed",
            "backend_verified",
            "fingerprint",
        }
    )
    return frozenset(paths)


_EGRESS_AUTHORITY_EVENT_NESTED_PATHS = _egress_authority_event_nested_paths()


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

    egress_authority_transition = _observed_policy(
        "adapter_strategy actor authorization_reason classification environment_fingerprint "
        "from_authority policy_identity reason receipt revision schema_version state "
        "source_environment_fingerprint to_authority transition_fingerprint transition_id",
        owned_nested_paths=_EGRESS_AUTHORITY_EVENT_NESTED_PATHS,
        authority_keys={
            "environment_fingerprint",
            "source_environment_fingerprint",
            "transition_fingerprint",
            "transition_id",
        },
        untrusted_container_keys={"actor", "from_authority", "receipt", "to_authority"},
    )
    for event_type in _EGRESS_AUTHORITY_EVENT_TYPES:
        policies[event_type] = egress_authority_transition

    interaction_summary = _policy(
        "model_policy",
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
        "targeted_tool_grant_batch_fingerprint",
        "targeted_tool_grant_count",
        "token_usage",
        "tool_call_count",
        "wall_duration_ms",
        owned_nested_paths=_AGGREGATE_USAGE_NESTED_PATHS | _MODEL_POLICY_PATHS,
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

    model_started = _observed_policy(
        "actor attempt attempt_id compactor execution_profile_fingerprint instruction_digest instruction_present max_attempts "
        "file_attachment_attestations mode model model_attempt_id model_step_id operation_id provider purpose reason request_id "
        "source_run_epoch source_transcript_cursor step",
        owned_nested_paths=_resolution_actor_nested_paths("actor"),
        authority_keys=(_MODEL_EXECUTION_AUTHORITY_KEYS | {"file_attachment_attestations"}),
        internal_authority_keys={"file_attachment_attestations"},
        public_authority_keys=_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
    )
    policies[EventType.MODEL_STARTED] = model_started
    auxiliary_attribution_paths = {
        ("auxiliary_inference", "operation_id"),
        ("auxiliary_inference", "tool_call_id"),
        ("auxiliary_inference", "parent", "model_step_id"),
        ("auxiliary_inference", "parent", "model_attempt_id"),
        ("auxiliary_inference", "parent", "tool_round_id"),
    }
    auxiliary_start = _observed_policy(
        "attempt auxiliary_inference provider_name requested_model",
        authority_keys=_MODEL_EXECUTION_AUTHORITY_KEYS,
        public_authority_keys=_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
        owned_nested_paths={
            ("auxiliary_inference", "purpose"),
            ("auxiliary_inference", "parent"),
        },
        nested_authority_paths=auxiliary_attribution_paths,
    )
    policies[EventType.MODEL_AUXILIARY_ATTEMPT_STARTED] = auxiliary_start
    policies[EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED] = replace(
        auxiliary_start,
        owned_keys=auxiliary_start.owned_keys
        | frozenset(
            {
                "provider",
                "model",
                "auxiliary_outcome",
                "usage_status",
                "usage_metrics",
                "billing_identity",
                "budget_settlements",
                "provider_error",
                "retry_decision",
            }
        ),
        owned_nested_paths=(
            auxiliary_start.owned_nested_paths
            | _MODEL_USAGE_METRICS_NESTED_PATHS
            | _MODEL_BILLING_IDENTITY_NESTED_PATHS
            | _MODEL_BUDGET_SETTLEMENT_NESTED_PATHS
            | {("retry_decision", key) for key in RetryDecision.model_fields}
        ),
        nested_authority_paths=(
            auxiliary_start.nested_authority_paths | _MODEL_BUDGET_SETTLEMENT_AUTHORITY_PATHS
        ),
        untrusted_container_keys=frozenset({"provider_error", "retry_decision"}),
        untrusted_container_paths=(
            _MODEL_ACCOUNTING_UNTRUSTED_CONTAINER_PATHS | _MODEL_BUDGET_SETTLEMENT_UNTRUSTED_PATHS
        ),
    )
    policies[EventType.REQUEST_FOOTPRINT_RECORDED] = _observed_policy(
        "attempt attempt_id attachments cache_breakpoints component_tokens context_pressure execution_profile_fingerprint "
        "fingerprints max_attempts messages model model_attempt_id model_step_id observation_id "
        "operation_id options provider_name prompt_contributions request_variant schema_version "
        "step structured_output targeted_native_item_active targeted_native_item_message_index "
        "targeted_tool_grants tool_discovery_projection tool_discovery_view tool_exposure tools total",
        owned_nested_paths=_REQUEST_FOOTPRINT_NESTED_PATHS,
        authority_keys={"execution_profile_fingerprint"},
        public_authority_keys=_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
    )
    policies[EventType.TOOL_EXPOSURE_RECORDED] = _observed_policy(
        "catalogue_revision ceiling_count execution_profile_fingerprint exposed_count "
        "exposure_fingerprint model model_step_id profile_changed profile_id provider_name "
        "registered_count schema_version step",
        authority_keys=(
            _MODEL_EXECUTION_AUTHORITY_KEYS | _TOOL_EXPOSURE_RECORD_PUBLIC_AUTHORITY_KEYS
        ),
        public_authority_keys=_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS
        | _TOOL_EXPOSURE_RECORD_PUBLIC_AUTHORITY_KEYS,
        aliased_authority_keys={"model_step_id"},
    )
    targeted_grant_lifecycle = _observed_policy(
        "arguments_sha256 catalogue_revision descriptor_version expires_at generation_id "
        "grant_id invocation_id issued_at max_calls model_step_id origin outer_tool_call_id "
        "outcome rejection_id rejection_reason remaining_calls request_id schema_fingerprint schema_version "
        "tool_id use_id used_calls",
        authority_keys=_TARGETED_TOOL_GRANT_PUBLIC_AUTHORITY_KEYS,
        public_authority_keys=_TARGETED_TOOL_GRANT_PUBLIC_AUTHORITY_KEYS,
        aliased_authority_keys={"invocation_id", "model_step_id", "outer_tool_call_id"},
    )
    for event_type in {
        EventType.TARGETED_TOOL_GRANT_ISSUED,
        EventType.TARGETED_TOOL_GRANT_REUSED,
        EventType.TARGETED_TOOL_GRANT_RECONSTRUCTED,
        EventType.TARGETED_TOOL_GRANT_EXPIRED,
        EventType.TARGETED_TOOL_GRANT_REVOKED,
        EventType.TARGETED_TOOL_REFERENCE_CONSUMED,
        EventType.TARGETED_TOOL_REFERENCE_REJOINED,
        EventType.TARGETED_TOOL_REFERENCE_REJECTED,
    }:
        policies[event_type] = targeted_grant_lifecycle
    policies[EventType.TARGETED_TOOL_GRANT_FORK_RESET] = _observed_policy(
        "inherited_grant_count inherited_reference_count schema_version "
        "source_interaction_id source_session_id"
    )
    model_delta = _policy(
        "attempt",
        "delta",
        "max_attempts",
        "model_attempt_id",
        "model_step_id",
        "step",
        "provider_operation_progress",
        authority_keys=_MODEL_EXECUTION_AUTHORITY_KEYS,
        public_authority_keys=_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
        internal_keys={"provider_operation_progress"},
        exact_internal_keys={"provider_operation_progress"},
    )
    policies[EventType.MODEL_TEXT_DELTA] = model_delta
    policies[EventType.MODEL_THINKING_DELTA] = model_delta
    policies[EventType.MODEL_HOSTED_TOOL_CALL] = _observed_policy(
        "action attempt call_id max_attempts model model_attempt_id model_step_id "
        "provider_name provider_operation_id source_count status step tool_type",
        owned_nested_paths={
            ("action", "type"),
            ("action", "query"),
            ("action", "queries"),
            ("action", "queries", "*"),
            ("action", "url"),
            ("action", "pattern"),
            ("action", "sources"),
            ("action", "sources", "*", "type"),
            ("action", "sources", "*", "url"),
            ("action", "sources", "*", "title"),
        },
        authority_keys=_MODEL_EXECUTION_AUTHORITY_KEYS | {"call_id", "provider_operation_id"},
        untrusted_container_keys={"action"},
    )
    policies[EventType.MODEL_CITATION] = _observed_policy(
        "attempt citation_type end_index max_attempts model model_attempt_id model_step_id "
        "provenance provider_operation_id start_index step title url",
        owned_nested_paths={
            ("provenance", "hosted_tool"),
            ("provenance", "provider_name"),
            ("provenance", "untrusted_external_evidence"),
        },
        authority_keys=_MODEL_EXECUTION_AUTHORITY_KEYS | {"provider_operation_id"},
        untrusted_container_keys={"provenance"},
    )
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
        "input_coverage",
        "max_attempts",
        "metadata",
        "mode",
        "model",
        "model_attempt_id",
        "model_step_id",
        "operation_id",
        "provider_debug",
        "provider_name",
        "provider_operation_progress",
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
        public_authority_keys=_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
        internal_keys={"provider_operation_progress"},
        exact_internal_keys={"provider_operation_progress"},
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
        "provider_name",
        "requested_model",
        "attempt",
        "context_overflow",
        "error",
        "error_code",
        "error_type",
        "retry",
        "retry_disposition",
        "retry_suppression",
        "provider_retryable",
        "effective_max_attempts",
        "max_attempts",
        "model",
        "model_attempt_id",
        "model_step_id",
        "provider",
        "provider_error_code",
        "provider_error_type",
        "provider_rejection_reason",
        "provider_rejection_parameter",
        "provider_rejection_explanation",
        "provider_rejection_unavailable_reason",
        "provider_rejection_request_id_state",
        "provider_api_classification_reason",
        "provider_api_classification_origin",
        "provider_protocol_reason",
        "provider_protocol_stage",
        "provider_protocol_field",
        "provider_protocol_citation_condition",
        "provider_protocol_citation_start_kind",
        "provider_protocol_citation_end_kind",
        "provider_protocol_citation_text_length",
        "provider_protocol_citation_text_offset",
        "provider_protocol_citation_start_index",
        "provider_protocol_citation_end_index",
        "provider_protocol_citation_start_index_status",
        "provider_protocol_citation_end_index_status",
        "provider_protocol_source_index",
        "provider_protocol_source_type_kind",
        "provider_protocol_source_supported_types",
        "provider_protocol_source_type_value_status",
        "provider_protocol_source_type_value",
        "provider_deadline_kind",
        "provider_deadline_timeout_s",
        "provider_effect_outcome",
        "provider_last_progress_at",
        "provider_whitespace_since_progress",
        "provider_semantic_idle_elapsed_s",
        "provider_excluded_semantic_pause_s",
        "provider_last_progress_elapsed_s",
        "provider_last_progress_kind",
        "provider_recovery_disposition",
        "provider_stream_elapsed_s",
        "request_id",
        "retry_after_s",
        "retryable",
        "stage",
        "status_code",
        "step",
        "stream_cleanup_failed",
    )
    auxiliary_terminal = policies[EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED]
    policies[EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED] = replace(
        auxiliary_terminal,
        owned_nested_paths=auxiliary_terminal.owned_nested_paths
        | {("provider_error", key) for key in model_failure_keys},
    )
    policies[EventType.MODEL_HTTP_CLEANUP] = _policy(
        "model_attempt_id",
        "model_step_id",
        "source_run_epoch",
        "provider",
        "local_http_cleanup",
        "provider_effect_outcome",
        "provider_deadline_kind",
        "provider_deadline_timeout_s",
        "provider_stream_elapsed_s",
        "provider_last_progress_at",
        "provider_last_progress_elapsed_s",
        "provider_last_progress_kind",
        "provider_whitespace_since_progress",
        "provider_semantic_idle_elapsed_s",
        "provider_excluded_semantic_pause_s",
        authority_keys=_MODEL_EXECUTION_AUTHORITY_KEYS,
        public_authority_keys=_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
    )
    policies[EventType.MODEL_ERROR] = _policy(
        *model_failure_keys,
        "purpose",
        "execution_admission",
        "provider_operation_progress",
        "reason",
        authority_keys=_MODEL_EXECUTION_AUTHORITY_KEYS,
        public_authority_keys=_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
        internal_keys={"provider_operation_progress"},
        exact_internal_keys={"provider_operation_progress"},
        untrusted_container_keys={"execution_admission"},
    )
    policies[EventType.MODEL_RETRY] = _policy(
        *model_failure_keys,
        "delay_s",
        "delay_seconds",
        "next_attempt",
        "reason",
        authority_keys=_MODEL_EXECUTION_AUTHORITY_KEYS,
        public_authority_keys=_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
    )
    policies[EventType.MODEL_ATTEMPT_DISCARDED] = _policy(
        "attempt",
        "retry",
        "retry_disposition",
        "retry_suppression",
        "provider_retryable",
        "effective_max_attempts",
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
        public_authority_keys=_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
    )
    policies[EventType.MODEL_FAILOVER_SELECTED] = _policy(
        "schema_version",
        "route_id",
        "route_generation",
        "stage_id",
        "model_step_id",
        "model_attempt_id",
        "provider",
        "model",
        "configured_provider",
        "configured_model",
        "candidate_index",
        "candidate_count",
        "attempts_used",
        "max_total_attempts",
        "previous_provider",
        "previous_model",
        "previous_stage_id",
        "reason",
        "status_code",
        authority_keys=_MODEL_EXECUTION_AUTHORITY_KEYS,
        public_authority_keys=_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
    )
    policies[EventType.MODEL_FAILOVER_EXHAUSTED] = _policy(
        "schema_version",
        "route_id",
        "route_generation",
        "stage_id",
        "model_step_id",
        "model_attempt_id",
        "provider",
        "model",
        "configured_provider",
        "configured_model",
        "candidate_index",
        "candidate_count",
        "attempts_used",
        "max_total_attempts",
        "reason",
        authority_keys=_MODEL_EXECUTION_AUTHORITY_KEYS,
        public_authority_keys=_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
    )
    provider_operation_starting_keys = (
        "attempt",
        "max_attempts",
        "model",
        "model_attempt_id",
        "model_step_id",
        "provider",
        "source_run_epoch",
        "start_id",
        "step",
    )
    policies[EventType.PROVIDER_OPERATION_STARTING] = _policy(
        *provider_operation_starting_keys,
        authority_keys=_MODEL_EXECUTION_AUTHORITY_KEYS | {"start_id"},
        public_authority_keys=_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
        internal_authority_keys={"start_id"},
    )
    policies[EventType.PROVIDER_OPERATION_STARTED] = _policy(
        "attempt",
        "max_attempts",
        "model",
        "model_attempt_id",
        "model_step_id",
        "operation_id",
        "provider",
        "recovery_metadata",
        "source_run_epoch",
        "start_id",
        "state_version",
        "status",
        "step",
        "stream_protocol",
        authority_keys=_MODEL_EXECUTION_AUTHORITY_KEYS
        | {"operation_id", "start_id", "stream_protocol"},
        internal_authority_keys={"start_id"},
        public_authority_keys={
            "operation_id",
            "stream_protocol",
            *_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
        },
        internal_keys={"recovery_metadata"},
        exact_internal_keys={"recovery_metadata"},
    )
    policies[EventType.PROVIDER_OPERATION_PROGRESS] = _policy(
        "attempt",
        "max_attempts",
        "model_attempt_id",
        "model_step_id",
        "operation_id",
        "provider",
        "provider_operation_progress",
        "step",
        "stream_protocol",
        authority_keys=_MODEL_EXECUTION_AUTHORITY_KEYS | {"operation_id", "stream_protocol"},
        public_authority_keys={
            "operation_id",
            "stream_protocol",
            *_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
        },
        internal_keys={"provider_operation_progress"},
        exact_internal_keys={"provider_operation_progress"},
        nested_authority_paths={
            (
                "provider_operation_progress",
                "stream_event",
                "payload",
                "arguments",
                "tool_ref",
            )
        },
    )
    provider_operation_recovery_keys = (
        "attempt",
        "max_attempts",
        "model",
        "model_attempt_id",
        "model_step_id",
        "operation_id",
        "provider",
        "run_epoch",
        "source_run_epoch",
        "status",
        "step",
        "stream_protocol",
    )
    policies[EventType.PROVIDER_OPERATION_RECONNECT_SCHEDULED] = _policy(
        *provider_operation_recovery_keys,
        authority_keys=_MODEL_EXECUTION_AUTHORITY_KEYS | {"operation_id", "stream_protocol"},
        public_authority_keys={
            "operation_id",
            "stream_protocol",
            *_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
        },
    )
    policies[EventType.PROVIDER_OPERATION_RECONNECT_STARTED] = _policy(
        *provider_operation_recovery_keys,
        authority_keys=_MODEL_EXECUTION_AUTHORITY_KEYS | {"operation_id", "stream_protocol"},
        public_authority_keys={
            "operation_id",
            "stream_protocol",
            *_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
        },
    )
    policies[EventType.PROVIDER_OPERATION_RECONCILED] = _policy(
        *provider_operation_recovery_keys,
        authority_keys=_MODEL_EXECUTION_AUTHORITY_KEYS | {"operation_id", "stream_protocol"},
        public_authority_keys={
            "operation_id",
            "stream_protocol",
            *_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
        },
    )
    provider_operation_cancellation_keys = (
        *provider_operation_recovery_keys,
        "cancellation_status",
        "error_type",
        "provider_status",
    )
    policies[EventType.PROVIDER_OPERATION_CANCEL_REQUESTED] = _policy(
        *provider_operation_cancellation_keys,
        authority_keys=_MODEL_EXECUTION_AUTHORITY_KEYS | {"operation_id", "stream_protocol"},
        public_authority_keys={
            "operation_id",
            "stream_protocol",
            *_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
        },
    )
    policies[EventType.PROVIDER_OPERATION_CANCEL_RESOLVED] = _policy(
        *provider_operation_cancellation_keys,
        authority_keys=_MODEL_EXECUTION_AUTHORITY_KEYS | {"operation_id", "stream_protocol"},
        public_authority_keys={
            "operation_id",
            "stream_protocol",
            *_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
        },
    )
    provider_cleanup_failure_paths = {
        ("provider_cleanup_failure", field_name)
        for field_name in {
            "durable_value_error_code",
            "durable_value_error_path",
            "durable_value_error_limit",
            "durable_value_error_observed_lower_bound",
            "error",
            "error_type",
            "phase",
        }
    }
    policies[EventType.PROVIDER_OPERATION_RECOVERY_REQUIRED] = _policy(
        *provider_operation_recovery_keys,
        *(key for key in model_failure_keys if key.startswith("provider_protocol_")),
        "idempotent_start_recovery",
        "provider_cleanup_failure",
        "recovery_reason",
        "start_id",
        owned_nested_paths=provider_cleanup_failure_paths,
        authority_keys=_MODEL_EXECUTION_AUTHORITY_KEYS
        | {"operation_id", "start_id", "stream_protocol"},
        internal_authority_keys={"start_id"},
        public_authority_keys={
            "operation_id",
            "stream_protocol",
            *_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
        },
        untrusted_container_keys={"provider_cleanup_failure"},
    )
    policies[EventType.PROVIDER_OPERATION_RESOLVED] = _policy(
        *provider_operation_recovery_keys,
        "duplicate_request_risk",
        "metadata",
        "reason",
        "recovery_reason",
        "resolution_action",
        "resolution_id",
        "resolved_by",
        "stage_id",
        owned_nested_paths=_resolution_actor_nested_paths("resolved_by"),
        authority_keys=_MODEL_EXECUTION_AUTHORITY_KEYS
        | {
            "operation_id",
            "resolution_id",
            "stage_id",
            "stream_protocol",
        },
        internal_authority_keys={"resolution_id", "stage_id"},
        public_authority_keys={
            "operation_id",
            "stream_protocol",
            *_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
        },
        untrusted_container_keys={"metadata"},
    )

    tool_common = {
        "approval",
        "approval_id",
        "approval_metadata_truncated",
        "arguments",
        "arguments_state",
        "arguments_sha256",
        "catalogue_revision",
        "descriptor_version",
        "dispatch_kind",
        "effect",
        "effective_tool_id",
        "execution_profile_fingerprint",
        "effective_arguments",
        "expired",
        "idempotency_key",
        "input_id",
        "invocation_id",
        "grant_id",
        "metadata",
        "metadata_truncated",
        "model_attempt_id",
        "model_step_id",
        "model_tool_name",
        "reason",
        "recovered",
        "requested_decision",
        "resolution_reason",
        "resolution_request_digest",
        "resolved_by",
        "result",
        "short_circuited_by",
        "schema_fingerprint",
        "structured_output_validation",
        "task_id",
        "tool_call_id",
        "tool_call_metadata_truncated",
        "tool_name",
        "tool_round_id",
        "use_id",
        "workspace_mutation_capture_detail_code",
        "workspace_mutation_capture_status",
    }
    tool_terminal = tool_common | {
        "effect_reconciled",
        "reconciliation_state",
        "receipt_id",
        "receipt_evidence",
        "arguments_exact",
        "durable_value_error_code",
        "durable_value_error_path",
        "durable_value_error_limit",
        "durable_value_error_observed_lower_bound",
        "isolated_tool_failure_code",
        "isolated_tool_cleanup_failure_code",
        "manual_reconciliation_required",
        "manual_recovery",
        "outcome_unknown",
        "registration_state",
        "terminal_outcome",
        "failure_evidence",
        "tool_effect",
        "tool_execution_boundary",
        "tool_timeout_strength",
        WEB_ACCESS_RESULT_AUTHORITY_FIELD,
        SHARED_ARTIFACT_RESULT_AUTHORITY_FIELD,
        *_TOOL_TERMINAL_TIMING_KEYS,
    }
    tool_actor_paths = _resolution_actor_nested_paths("resolved_by")
    policies[EventType.TOOL_EFFECT_OUTCOME_UNKNOWN] = _policy(
        "schema_version",
        "failure_evidence",
        "unverified_output",
        "state",
        "record_revision",
        "intent_digest",
        "dispatch_id",
        "model_step_id",
        "model_attempt_id",
        "tool_round_id",
        "tool_call_id",
        "approval_id",
        authority_keys=_MODEL_EXECUTION_AUTHORITY_KEYS | {"tool_call_id", "approval_id"},
        aliased_authority_keys={"approval_id", "tool_call_id", "tool_round_id"},
        untrusted_container_keys={"unverified_output"},
    )
    policies[EventType.TOOL_EFFECT_CLEANUP_OBSERVED] = _policy(
        "schema_version",
        "artifacts",
        "truncated",
        "scope_digest",
        "model_step_id",
        "model_attempt_id",
        "tool_round_id",
        "tool_call_id",
        authority_keys=_MODEL_EXECUTION_AUTHORITY_KEYS | {"tool_call_id"},
        aliased_authority_keys={"tool_call_id", "tool_round_id"},
    )
    policies[EventType.TOOL_EFFECT_RECONCILIATION_STARTED] = _policy(
        "schema_version",
        "request_digest",
        "intent_digest",
        "dispatch_id",
        "expected_revision",
        "expected_run_epoch",
        "lookup",
        "model_step_id",
        "model_attempt_id",
        "tool_round_id",
        "tool_call_id",
        "approval_id",
        "execution_profile_fingerprint",
        authority_keys=_MODEL_EXECUTION_AUTHORITY_KEYS | {"tool_call_id", "approval_id"},
        aliased_authority_keys={"approval_id", "tool_call_id", "tool_round_id"},
    )
    receipt_evidence_paths = {
        ("receipt_evidence", name)
        for name in (
            "schema_version",
            "receipt_id",
            "receipt_schema",
            "receipt_schema_version",
            "outcome",
            "source",
            "observed_at",
            "receipt_digest",
            "integrity",
            "resource_versions",
        )
    }
    policies[EventType.TOOL_EFFECT_RECEIPT_VALIDATED] = _policy(
        "schema_version",
        "request_digest",
        "receipt_evidence",
        "execution_profile_fingerprint",
        "model_step_id",
        "model_attempt_id",
        "tool_round_id",
        "tool_call_id",
        "approval_id",
        owned_nested_paths=receipt_evidence_paths,
        authority_keys=_MODEL_EXECUTION_AUTHORITY_KEYS | {"tool_call_id", "approval_id"},
        aliased_authority_keys={"approval_id", "tool_call_id", "tool_round_id"},
    )
    policies[EventType.TOOL_EFFECT_RECONCILIATION_OBSERVED] = _policy(
        "approval_id",
        "schema_version",
        "request_digest",
        "resource_versions",
        "result",
        "execution_profile_fingerprint",
        "model_step_id",
        "model_attempt_id",
        "tool_round_id",
        "tool_call_id",
        "idempotency_key",
        owned_nested_paths={
            ("result", name)
            for name in ("outcome", "observation", "retryable", "receipt", "resource_versions")
        },
        authority_keys=_MODEL_EXECUTION_AUTHORITY_KEYS
        | {"approval_id", "tool_call_id", "idempotency_key", "request_digest"},
        public_authority_keys=_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS | {"request_digest"},
        aliased_authority_keys={"approval_id", "tool_call_id", "tool_round_id"},
        untrusted_container_keys={"result", "resource_versions"},
    )
    # Validator rejection carries the same untrusted observation envelope.
    # A late-dispatch loser instead carries only the store-owned identity digests.
    observation_policy = policies[EventType.TOOL_EFFECT_RECONCILIATION_OBSERVED]
    policies[EventType.TOOL_EFFECT_RECONCILIATION_CONFLICT] = replace(
        observation_policy,
        owned_keys=observation_policy.owned_keys
        | {
            "kind",
            "intent_digest",
            "dispatch_digest",
            "winner_digest",
            "source_run_epoch",
            "attempt_digest",
            "selection_digest",
        },
    )
    policies[EventType.TOOL_CALL_STARTED] = _policy(
        *tool_common,
        owned_nested_paths=_TOOL_RESULT_NESTED_PATHS | tool_actor_paths,
        authority_keys=(
            _TOOL_LINKAGE_AUTHORITY_KEYS | _TARGETED_TOOL_INVOCATION_PUBLIC_AUTHORITY_KEYS
        ),
        public_authority_keys=(
            _EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS
            | _TARGETED_TOOL_INVOCATION_PUBLIC_AUTHORITY_KEYS
        ),
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
            owned_nested_paths=_TOOL_EVENT_NESTED_PATHS | tool_actor_paths | receipt_evidence_paths,
            authority_keys={
                *_TOOL_LINKAGE_AUTHORITY_KEYS,
                *_TARGETED_TOOL_INVOCATION_PUBLIC_AUTHORITY_KEYS,
                "resolution_request_digest",
                WEB_ACCESS_RESULT_AUTHORITY_FIELD,
                SHARED_ARTIFACT_RESULT_AUTHORITY_FIELD,
                *_TOOL_TERMINAL_TIMING_KEYS,
            },
            internal_authority_keys={
                WEB_ACCESS_RESULT_AUTHORITY_FIELD,
                SHARED_ARTIFACT_RESULT_AUTHORITY_FIELD,
            },
            public_authority_keys=(
                _EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS
                | _TARGETED_TOOL_INVOCATION_PUBLIC_AUTHORITY_KEYS
                | _TOOL_TERMINAL_TIMING_KEYS
            ),
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
    workspace_observation_keys = (
        "binding_generation_id branch detail_code execution_profile_fingerprint head_revision "
        "model_attempt_id model_step model_step_id observer "
        "manifest_artifact_id manifest_artifact_sha256 manifest_artifact_size_bytes path_scope paths phase "
        "revision session_run_epoch status tool_call_id tool_round_id total_paths window_id "
        "workspace_id artifact_store_id"
    )
    workspace_path_owned_paths = {
        ("paths", "*", field_name) for field_name in _WORKSPACE_PATH_REVISION_FIELDS
    }
    workspace_path_authority_paths = {
        ("paths", "*", field_name) for field_name in _WORKSPACE_PATH_REVISION_AUTHORITY_FIELDS
    }
    workspace_delta_owned_paths = {
        ("paths", "*", field_name) for field_name in _WORKSPACE_PATH_REVISION_DELTA_FIELDS
    }
    workspace_delta_authority_paths = {
        ("paths", "*", field_name) for field_name in _WORKSPACE_PATH_REVISION_DELTA_AUTHORITY_FIELDS
    }
    workspace_attribution_owned_paths = (
        {
            ("attribution", field_name)
            for field_name in {
                "confidence",
                "detail_code",
                "direct_reconciliation",
                "overlap_detected",
                "writer_isolation",
            }
        }
        | {("writer_isolation", phase) for phase in {"before", "after"}}
        | {
            ("writer_isolation", phase, field_name)
            for phase in {"before", "after"}
            for field_name in {"status", "mechanism", "generation", "detail_code"}
        }
        | {
            ("direct_mutations", field_name)
            for field_name in {
                "operations",
                "retained_operations",
                "total_operations",
                "truncated",
            }
        }
        | {
            ("direct_mutations", "operations", "*"),
        }
        | {
            ("direct_mutations", "operations", "*", field_name)
            for field_name in {
                "sequence",
                "method",
                "path_sha256",
                "result_valid",
                "result_operation",
                "result_evidence_sha256",
            }
        }
        | {
            ("pre_window_change", field_name)
            for field_name in {
                "attribution_confidence",
                "status",
                "before_revision",
                "after_revision",
                "paths",
                "retained_paths",
                "total_paths",
                "truncated",
                "head_changed",
                "branch_changed",
                "detail_code",
            }
        }
        | {
            ("pre_window_change", "paths", "*", field_name)
            for field_name in {"path_sha256", "change"}
        }
    )
    workspace_attribution_authority_paths = (
        {
            ("attribution", field_name)
            for field_name in {
                "confidence",
                "detail_code",
                "direct_reconciliation",
                "writer_isolation",
            }
        }
        | {("writer_isolation", phase, "status") for phase in {"before", "after"}}
        | {
            ("direct_mutations", "operations", "*", field_name)
            for field_name in {
                "method",
                "path_sha256",
                "result_operation",
                "result_evidence_sha256",
            }
        }
        | {("pre_window_change", field_name) for field_name in {"attribution_confidence", "status"}}
        | {("pre_window_change", "paths", "*", "change")}
    )
    policies[EventType.WORKSPACE_REVISION_OBSERVED] = _observed_policy(
        workspace_observation_keys,
        owned_nested_paths=workspace_path_owned_paths,
        authority_keys={
            "execution_profile_fingerprint",
            "manifest_artifact_sha256",
            "observer",
        },
        public_authority_keys={
            *_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
            "manifest_artifact_sha256",
            "observer",
        },
        aliased_authority_keys={
            "model_attempt_id",
            "model_step_id",
            "binding_generation_id",
            "manifest_artifact_id",
            "tool_call_id",
            "tool_round_id",
            "window_id",
            "workspace_id",
            "artifact_store_id",
        },
        nested_authority_paths=workspace_path_authority_paths,
        untrusted_container_keys={"paths"},
    )
    policies[EventType.WORKSPACE_MUTATION_RECORDED] = _observed_policy(
        "after_observation_id after_revision before_observation_id before_revision binding_generation_id "
        "attribution direct_mutations pre_window_change writer_isolation "
        "branch_changed detail_code execution_profile_fingerprint head_changed model_attempt_id model_step model_step_id "
        "manifest_artifact_id manifest_artifact_sha256 manifest_artifact_size_bytes observer paths "
        "recovery_run_epoch session_run_epoch status tool_call_id tool_outcome_event_digest tool_outcome_event_id "
        "tool_round_id total_paths window_id workspace_id artifact_store_id",
        owned_nested_paths=workspace_delta_owned_paths | workspace_attribution_owned_paths,
        authority_keys={
            "execution_profile_fingerprint",
            "manifest_artifact_sha256",
            "observer",
            "tool_outcome_event_digest",
        },
        public_authority_keys={
            *_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
            "manifest_artifact_sha256",
            "observer",
            "tool_outcome_event_digest",
        },
        aliased_authority_keys={
            "after_observation_id",
            "before_observation_id",
            "binding_generation_id",
            "model_attempt_id",
            "model_step_id",
            "manifest_artifact_id",
            "tool_call_id",
            "tool_outcome_event_id",
            "tool_round_id",
            "window_id",
            "workspace_id",
            "artifact_store_id",
        },
        nested_authority_paths=(
            workspace_delta_authority_paths | workspace_attribution_authority_paths
        ),
        untrusted_container_keys={"direct_mutations", "paths", "writer_isolation"},
    )
    policies[EventType.WORKSPACE_OBSERVATION_FINALIZED] = _observed_policy(
        "after_observation_id before_observation_id binding_generation_id branch_changed detail_code "
        "attribution "
        "execution_profile_fingerprint failed_artifact_count head_changed "
        "model_attempt_id model_step model_step_id mutation_event_id paths recovery_run_epoch "
        "referenced_artifact_count "
        "revision_after_artifact_id revision_after_artifact_sha256 "
        "revision_after_artifact_size_bytes revision_after_artifact_state "
        "revision_before_artifact_id revision_before_artifact_sha256 "
        "revision_before_artifact_size_bytes revision_before_artifact_state "
        "revision_delta_artifact_id revision_delta_artifact_sha256 "
        "revision_delta_artifact_size_bytes revision_delta_artifact_state "
        "session_run_epoch status tool_call_id tool_outcome_event_digest tool_outcome_event_id "
        "mutation_event_digest "
        "tool_round_id total_paths window_id workspace_id observer artifact_store_id",
        owned_nested_paths={
            ("attribution", "confidence"),
            ("attribution", "writer_isolation"),
            ("attribution", "overlap_detected"),
            ("attribution", "direct_reconciliation"),
            ("attribution", "detail_code"),
        },
        authority_keys={
            "execution_profile_fingerprint",
            "mutation_event_digest",
            "observer",
            "revision_after_artifact_sha256",
            "revision_before_artifact_sha256",
            "revision_delta_artifact_sha256",
            "tool_outcome_event_digest",
        },
        public_authority_keys={
            *_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
            "mutation_event_digest",
            "observer",
            "revision_after_artifact_sha256",
            "revision_before_artifact_sha256",
            "revision_delta_artifact_sha256",
            "tool_outcome_event_digest",
        },
        aliased_authority_keys={
            "after_observation_id",
            "before_observation_id",
            "binding_generation_id",
            "model_attempt_id",
            "model_step_id",
            "mutation_event_id",
            "revision_after_artifact_id",
            "revision_before_artifact_id",
            "revision_delta_artifact_id",
            "tool_call_id",
            "tool_outcome_event_id",
            "tool_round_id",
            "window_id",
            "workspace_id",
            "artifact_store_id",
        },
        untrusted_container_keys={"paths"},
    )
    policies[EventType.TOOL_CALL_BLOCKED] = _policy(
        *tool_common,
        *_TOOL_TERMINAL_TIMING_KEYS,
        "blocked_by",
        "decision",
        "denied_by",
        "exposure_fingerprint",
        "profile_id",
        "reason",
        "tool_result_projection",
        owned_nested_paths=_TOOL_PROJECTED_DENIAL_RESULT_NESTED_PATHS | tool_actor_paths,
        authority_keys=(
            _TOOL_LINKAGE_AUTHORITY_KEYS
            | _TOOL_EXPOSURE_PUBLIC_AUTHORITY_KEYS
            | _TARGETED_TOOL_INVOCATION_PUBLIC_AUTHORITY_KEYS
            | _TOOL_TERMINAL_TIMING_KEYS
        ),
        public_authority_keys=(
            _EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS
            | _TOOL_EXPOSURE_PUBLIC_AUTHORITY_KEYS
            | _TARGETED_TOOL_INVOCATION_PUBLIC_AUTHORITY_KEYS
            | _TOOL_TERMINAL_TIMING_KEYS
        ),
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
        authority_keys=(
            _TOOL_LINKAGE_AUTHORITY_KEYS | _TARGETED_TOOL_INVOCATION_PUBLIC_AUTHORITY_KEYS
        ),
        public_authority_keys=(
            _EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS
            | _TARGETED_TOOL_INVOCATION_PUBLIC_AUTHORITY_KEYS
        ),
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
        "execution_profile_fingerprint",
        "reason",
        "resolved_by",
        "tool_call_id",
        "tool_round_id",
        owned_nested_paths=tool_actor_paths,
        authority_keys={
            "approval_id",
            "execution_profile_fingerprint",
            "model_attempt_id",
            "model_step_id",
            "tool_call_id",
            "tool_round_id",
        },
        public_authority_keys=_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
        aliased_authority_keys={"approval_id", "tool_call_id", "tool_round_id"},
    )
    policies[EventType.TOOL_CALL_APPROVAL_DENIED] = _policy(
        *tool_terminal,
        "approval_required",
        "expired",
        "reason",
        "resolved_by",
        owned_nested_paths=_TOOL_DENIAL_RESULT_NESTED_PATHS | tool_actor_paths,
        authority_keys=(
            _TOOL_LINKAGE_AUTHORITY_KEYS
            | _TARGETED_TOOL_INVOCATION_PUBLIC_AUTHORITY_KEYS
            | _TOOL_TERMINAL_TIMING_KEYS
        ),
        public_authority_keys=(
            _EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS
            | _TARGETED_TOOL_INVOCATION_PUBLIC_AUTHORITY_KEYS
            | _TOOL_TERMINAL_TIMING_KEYS
        ),
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
        "execution_profile_fingerprint",
        "expires_at",
        "requested_decision",
        "resolved_by",
        "tool_call_id",
        "tool_round_id",
        "triggered_by",
        owned_nested_paths=(tool_actor_paths | _resolution_actor_nested_paths("triggered_by")),
        authority_keys={
            "approval_id",
            "execution_profile_fingerprint",
            "model_attempt_id",
            "model_step_id",
            "tool_call_id",
            "tool_round_id",
        },
        public_authority_keys=_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
        aliased_authority_keys={"approval_id", "tool_call_id", "tool_round_id"},
    )

    policies[EventType.SERVER_MUTATION_ACCEPTED] = _observed_policy(
        "accepted_event_id accepted_event_publication_uncertain accepted_event_sequence "
        "accepted_event_type mutation_id mutation_kind",
        public_authority_keys={"mutation_id"},
    )
    terminal_finalization_keys = (
        "binding_finalize_error binding_finalize_publication_error environment_factory_release "
        "failure_evidence final_revision"
    )
    terminal_finalization_containers = {
        "binding_finalize_error",
        "binding_finalize_publication_error",
        "environment_factory_release",
        "final_revision",
    }
    terminal_finalization_owned_paths = (
        {
            ("final_revision", "status"),
            ("final_revision", "path_scope"),
            ("final_revision", "finalization_delta"),
        }
        | {
            ("final_revision", "finalization_delta", field_name)
            for field_name in {
                "attribution_confidence",
                "status",
                "before_revision",
                "after_revision",
                "paths",
                "retained_paths",
                "total_paths",
                "truncated",
                "head_changed",
                "branch_changed",
                "detail_code",
            }
        }
        | {
            ("final_revision", "finalization_delta", "paths", "*"),
            ("final_revision", "finalization_delta", "paths", "*", "path_sha256"),
            ("final_revision", "finalization_delta", "paths", "*", "change"),
        }
    )
    failure_evidence_owned_paths = {
        ("failure_evidence", key)
        for key in (
            "classification",
            "deadline",
            "deadline_phase",
            "exception_types",
            "run_epoch",
            "secondary_failures",
            "session_id",
            "settlement",
            "terminal_event_id",
            "truncated",
        )
    } | {("failure_evidence", "deadline", key) for key in ("expires_at", "source", "scope")}
    for event_type in (EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED):
        policies[event_type] = replace(
            policies[event_type],
            owned_nested_paths=policies[event_type].owned_nested_paths
            | frozenset(failure_evidence_owned_paths),
        )
    policies[EventType.TOOL_EFFECT_OUTCOME_UNKNOWN] = replace(
        policies[EventType.TOOL_EFFECT_OUTCOME_UNKNOWN],
        owned_nested_paths=frozenset(failure_evidence_owned_paths),
    )
    policies[EventType.SESSION_STARTED] = _observed_policy(
        "agent_name input_contract model_policy parent_session_id prompt_contribution_manifest run_epoch "
        "traceparent tracestate",
        owned_nested_paths=_PROMPT_CONTRIBUTION_MANIFEST_NESTED_PATHS | _MODEL_POLICY_PATHS,
        authority_keys={"input_contract"},
        internal_authority_keys={"input_contract"},
    )
    policies[EventType.SESSION_RESUMED] = _observed_policy(
        "agent_name appended_messages approval_id decision dispatch_id execution_profile_fingerprint expired input_contract input_id "
        "interruption_type model_attempt_id model_step_id parent_session_id resolved_by "
        "run_epoch task_id tool_call_id tool_round_id traceparent tracestate",
        owned_nested_paths=_resolution_actor_nested_paths("resolved_by"),
        authority_keys={"execution_profile_fingerprint", "input_contract"},
        internal_authority_keys={"input_contract"},
        public_authority_keys=_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
    )
    policies[EventType.SESSION_COMPLETED] = _observed_policy(
        terminal_finalization_keys,
        owned_nested_paths=terminal_finalization_owned_paths | failure_evidence_owned_paths,
        authority_keys={"session_run_operation_id"},
        untrusted_container_keys=terminal_finalization_containers,
    )
    policies[EventType.SESSION_FAILED] = _observed_policy(
        "approval_id binding_cleanup durable_value_error_code durable_value_error_path "
        "compaction_failure error error_type interaction_transition_failures interruption_type "
        "manual_recovery_required model_attempt_id model_step_id tool_call_id "
        f"tool_name tool_round_id {terminal_finalization_keys}",
        owned_nested_paths=terminal_finalization_owned_paths | failure_evidence_owned_paths,
        authority_keys={"session_run_operation_id"},
        aliased_authority_keys={"approval_id", "tool_call_id", "tool_round_id"},
        untrusted_container_keys={
            "binding_cleanup",
            "compaction_failure",
            "interaction_transition_failures",
        }
        | terminal_finalization_containers,
    )
    policies[EventType.SESSION_INTERRUPTED] = _observed_policy(
        "abandoned actual action_id action_kind child_session_id status approval approval_close_intent approval_id "
        "approval_metadata_truncated cost_summary durable_value_error_code "
        "durable_value_error_path error error_type execution_profile_fingerprint input_id interruption_request_id "
        "interaction_transition_failures interruption_type limit manual_recovery_persisted "
        "manual_recovery_persistence_unknown manual_recovery_required "
        "manual_recovery_stale_live_failure maximum message metadata model_attempt_id "
        "model_step_id pause_digest persistence_reconciliation_error_type policy_metadata reason "
        "provider_cancellation_failures "
        "recovered requested_by resolved_by tool_call_id tool_call_metadata_truncated "
        "session_run_operation_id tool_evidence_conflict tool_name tool_round_id "
        "source_run_epoch usage_summary user_input user_input_supersession_intent "
        "ambiguous_user_input_supersession_intent " + terminal_finalization_keys,
        owned_nested_paths=terminal_finalization_owned_paths
        | failure_evidence_owned_paths
        | _resolution_actor_nested_paths("requested_by", "resolved_by")
        | _APPROVAL_NESTED_SCHEMA_PATHS
        | _USER_INPUT_NESTED_SCHEMA_PATHS
        | _USER_INPUT_SUPERSESSION_NESTED_SCHEMA_PATHS
        | _AMBIGUOUS_USER_INPUT_SUPERSESSION_NESTED_SCHEMA_PATHS,
        aliased_authority_keys={
            "child_session_id",
            "action_id",
            "approval_id",
            "input_id",
            "tool_call_id",
            "tool_round_id",
        },
        authority_keys={"execution_profile_fingerprint", "session_run_operation_id"},
        public_authority_keys=_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
        nested_authority_paths=(
            _APPROVAL_NESTED_AUTHORITY_PATHS
            | _USER_INPUT_NESTED_AUTHORITY_PATHS
            | _USER_INPUT_SUPERSESSION_NESTED_AUTHORITY_PATHS
            | _AMBIGUOUS_USER_INPUT_SUPERSESSION_NESTED_AUTHORITY_PATHS
        ),
        aliased_nested_authority_paths=_actionable_nested_authority_paths(
            _APPROVAL_NESTED_AUTHORITY_PATHS | _USER_INPUT_NESTED_AUTHORITY_PATHS
        ),
        untrusted_container_keys={
            "approval",
            "interaction_transition_failures",
            "metadata",
            "policy_metadata",
            "provider_cancellation_failures",
            "user_input",
        }
        | terminal_finalization_containers,
    )
    policies[EventType.SESSION_DELEGATED_ACTION_UPDATED] = _observed_policy(
        "action_id action_kind child_session_id status interruption_type "
        "model_attempt_id model_step_id tool_call_id tool_round_id",
        aliased_authority_keys={"child_session_id", "action_id", "tool_call_id", "tool_round_id"},
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
        "execution_profile_fingerprint input_id model_attempt_id model_step_id options pause_digest "
        "question source_run_epoch tool_call_id tool_calls tool_round_id",
        authority_keys={"execution_profile_fingerprint"},
        public_authority_keys=_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
        aliased_authority_keys={"input_id", "tool_call_id", "tool_round_id"},
        owned_nested_paths={
            ("tool_calls", "*", "arguments_state"),
        },
        nested_authority_paths=_TOOL_CALL_LIST_NESTED_AUTHORITY_PATHS,
        aliased_nested_authority_paths=_TOOL_CALL_LIST_NESTED_AUTHORITY_PATHS,
        untrusted_container_keys={"options", "tool_calls"},
    )
    policies[EventType.SESSION_CHECKPOINTED] = _observed_policy(
        "actor approval_id attempt_id calls checkpoint cleared compacted_transcript_cursor "
        "compaction_model_calls_unrepresented compactor estimated_context_input_tokens estimated_context_window_tokens "
        "estimated_delta_input_tokens execution_profile_fingerprint input_id instruction_digest instruction_present "
        "last_input_tokens last_total_tokens last_transcript_cursor min_input_tokens "
        "min_total_tokens mode model_attempt_id model_step_id "
        "newly_compacted_message_count operation_id "
        "pause_digest previous_compacted_transcript_cursor provider_count_context_window_tokens "
        "provider_count_input_tokens reason recent_message_count request_id "
        "reserved_output_tokens resolution_request_digest result_transcript_cursor source_run_epoch "
        "source_transcript_cursor tool_call_id tool_round_id "
        "transition trigger_estimated_context_tokens",
        authority_keys={
            "execution_profile_fingerprint",
            "pause_digest",
            "resolution_request_digest",
        },
        public_authority_keys=_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
        owned_nested_paths=_resolution_actor_nested_paths("actor"),
    )
    # Public digests are metadata, not export receipts or disclosure authority.
    for event_type in SESSION_EXPORT_EVENT_TYPES:
        policies[event_type] = _policy(
            *SESSION_EXPORT_EVENT_FIELDS,
            authority_keys=SESSION_EXPORT_EVENT_FIELDS,
            public_authority_keys=SESSION_EXPORT_EVENT_FIELDS,
        )
    policies[EventType.SESSION_FORKED] = _policy(
        "agent_name",
        "causal_budget_id",
        "copy_checkpoint",
        "environment_name",
        "execution_profile_selection",
        "fork_request_sha256",
        "inherited_taint_labels",
        "initial_dispatch_id",
        "model",
        "model_failover_candidate_index",
        "parent_session_id",
        "provider_name",
        "selected_profile_fingerprint",
        "source_profile_fingerprint",
        "source_environment_allocation_owners",
        "source_run_epoch",
        "source_session_id",
        "source_status",
        "source_transcript_cursor",
        "system_prompt_policy",
        "transcript_cursor",
        "workspace_lineage",
        owned_nested_paths={
            ("source_environment_allocation_owners", "*", "environment_name"),
            ("source_environment_allocation_owners", "*", "owner_session_id"),
            ("workspace_lineage", "status"),
            ("workspace_lineage", "source_workspace_revision"),
            ("workspace_lineage", "detail_code"),
        },
        authority_keys={
            "causal_budget_id",
            "fork_request_sha256",
            "initial_invocation_profile_fingerprint",
            "initial_invocation_request_sha256",
            "parent_session_id",
            "selected_profile_fingerprint",
            "source_profile_fingerprint",
            "source_session_id",
            "source_instance_fingerprint",
            "source_transcript_sha256",
            "source_checkpoint_sha256",
            "source_execution_profile_fingerprint",
            "initial_dispatch_id",
        },
        internal_authority_keys={
            "fork_request_sha256",
            "initial_invocation_profile_fingerprint",
            "initial_invocation_request_sha256",
            "selected_profile_fingerprint",
            "source_profile_fingerprint",
        },
        internal_keys={"source_environment_allocation_owners"},
        public_authority_keys={
            "initial_dispatch_id",
            "source_instance_fingerprint",
            "source_transcript_sha256",
            "source_checkpoint_sha256",
            "source_execution_profile_fingerprint",
        },
        nested_authority_paths={
            ("source_environment_allocation_owners", "*", "environment_name"),
            ("source_environment_allocation_owners", "*", "owner_session_id"),
        },
    )
    policies[EventType.SESSION_LIMIT_REACHED] = _observed_policy(
        "actual cost_summary limit maximum message reason usage_summary"
    )
    message_source_paths = {
        ("source", key)
        for key in (
            "session_id",
            "session_instance_id",
            "run_epoch",
            "transcript_cursor",
            "transcript_sha256",
            "checkpoint_sha256",
        )
    }
    message_policy = _observed_policy(
        "accepted_run_epoch accepted_transcript_cursor actor delivery_mode ordering_key "
        "input_contract queue_id run_epoch transcript_cursor source",
        owned_nested_paths=_resolution_actor_nested_paths("actor") | message_source_paths,
        authority_keys={"input_contract"},
        internal_authority_keys={"input_contract"},
        nested_authority_paths={("source", "session_id")},
        envelope_aliased_nested_authority_paths={("source", "session_id")},
    )
    policies[EventType.SESSION_MESSAGE_QUEUED] = message_policy
    policies[EventType.SESSION_MESSAGE_DELIVERED] = message_policy
    for event_type in (
        EventType.SESSION_MESSAGE_WITHDRAWN,
        EventType.SESSION_MESSAGE_QUARANTINED,
        EventType.SESSION_MESSAGE_STALE,
        EventType.SESSION_MESSAGE_EXPIRED,
    ):
        policies[event_type] = _observed_policy(
            "queue_id ordering_key actor run_epoch transcript_cursor status delivery_mode source",
            owned_nested_paths=_resolution_actor_nested_paths("actor") | message_source_paths,
            nested_authority_paths={("source", "session_id")},
            envelope_aliased_nested_authority_paths={("source", "session_id")},
        )
    profile_paths = {
        (profile_key, path)
        for profile_key in ("expected_profile", "candidate_profile")
        for path in ("schema_version", "fingerprint")
    } | {
        (profile_key, "components", "*", path)
        for profile_key in ("expected_profile", "candidate_profile")
        for path in ("component_class", "strength", "availability", "fingerprint")
    }
    profile_decision_policy = _observed_policy(
        "actor adoption_request_fingerprint authority_decision candidate_profile "
        "changed_component_classes decision "
        "candidate_profile_fingerprint expected_profile expected_profile_fingerprint "
        "idempotency_identity policy_identity policy_reason reason",
        owned_nested_paths=(
            profile_paths
            | _resolution_actor_nested_paths("actor")
            | {("changed_component_classes", "*")}
        ),
        authority_keys={
            "adoption_request_fingerprint",
            "authority_decision",
            "decision",
            "idempotency_identity",
            "policy_identity",
            "candidate_profile_fingerprint",
            "expected_profile_fingerprint",
        },
        internal_authority_keys={"adoption_request_fingerprint"},
        public_authority_keys={
            "authority_decision",
            "decision",
            "idempotency_identity",
            "policy_identity",
            "candidate_profile_fingerprint",
            "expected_profile_fingerprint",
        },
        untrusted_container_keys={"actor", "candidate_profile", "expected_profile"},
    )
    policies[EventType.SESSION_EXECUTION_PROFILE_DECIDED] = profile_decision_policy
    policies[EventType.SESSION_EXECUTION_PROFILE_REJECTED] = profile_decision_policy
    policies[EventType.SESSION_MODEL_SWITCHED] = _observed_policy(
        "cache_state_dropped full_transcript_projection model_changed "
        "provider_changed provider_state_parts_dropped source_model "
        "source_provider_name source_transcript_cursor target_model target_provider_name "
        "thinking_parts_dropped",
        authority_keys={
            "source_model",
            "source_provider_name",
            "target_model",
            "target_provider_name",
        },
        public_authority_keys={
            "source_model",
            "source_provider_name",
            "target_model",
            "target_provider_name",
        },
    )
    policies[EventType.SESSION_RUN_FENCED] = _observed_policy(
        "inactive_for_seconds metadata previous_run_epoch reason run_epoch",
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
        "currency execution_profile_fingerprint instruction_digest instruction_present key limit_reached maximum message "
        "mode model_attempt_id model_step_id model_steps operation_id reason request_id "
        "requested scope source_run_epoch source_transcript_cursor unpriced_model_steps "
        "unpriced_auxiliary_attempts "
        "window window_details"
    )
    for event_type in (
        EventType.BUDGET_CHECKED,
        EventType.BUDGET_LIMIT_REACHED,
        EventType.BUDGET_RESERVATION_FAILED,
    ):
        policies[event_type] = _observed_policy(
            budget_common,
            authority_keys={"execution_profile_fingerprint"},
            public_authority_keys=_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
            owned_nested_paths=_resolution_actor_nested_paths("actor"),
        )
    policies[EventType.BUDGET_RESERVED] = _observed_policy(
        budget_common
        + " agent_name billing_identity model provider_name reservation_id session_id",
        authority_keys={"execution_profile_fingerprint"},
        public_authority_keys=_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
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
        "actor actual_amount attempt_id billing_identity budget_limit_id compactor execution_profile_fingerprint "
        "instruction_digest instruction_present interaction_id mode model_attempt_id "
        "model_step_id operation_id pricing reason released_amount request_id reservation_id "
        "reserved_amount session_instance_id settled_at_unix_us settlement_id settlement_kind source_run_epoch "
        "source_transcript_cursor status",
        owned_nested_paths=(
            _resolution_actor_nested_paths("actor") | _BUDGET_RECONCILIATION_NESTED_PATHS
        ),
        authority_keys={"execution_profile_fingerprint"},
        public_authority_keys=_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
        untrusted_container_paths=_BUDGET_RECONCILIATION_UNTRUSTED_PATHS,
    )
    policies[EventType.BUDGET_RECONCILED] = settlement_policy
    policies[EventType.BUDGET_RESERVATION_RELEASED] = settlement_policy

    policies[EventType.CREDENTIAL_PROXY_CHECKED] = _observed_policy(
        "action allowed approval_id credential destination execution_profile_fingerprint idempotency_key input_id metadata "
        "model_attempt_id model_step_id reason result_metadata tool_call_id tool_round_id",
        authority_keys={"execution_profile_fingerprint"},
        public_authority_keys=_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
        untrusted_container_keys={"metadata", "result_metadata"},
    )
    policies[EventType.CREDENTIAL_MODE_SELECTED] = _observed_policy(
        "approved_destination_count credential_mode execution_profile_fingerprint grant_count runner_kind",
        authority_keys={"execution_profile_fingerprint"},
        public_authority_keys=_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
    )
    policies[EventType.EGRESS_GRANT_MINTED] = _observed_policy(
        "execution_profile_fingerprint grant_id",
        authority_keys={"execution_profile_fingerprint"},
        public_authority_keys=_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
    )
    policies[EventType.EGRESS_GRANT_REVOKED] = _observed_policy(
        "execution_profile_fingerprint grant_id outcome",
        authority_keys={"execution_profile_fingerprint"},
        public_authority_keys=_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
    )
    egress_request_policy = _observed_policy(
        "action allowed authorization_kind credential destination execution_profile_fingerprint "
        "grant_id metadata method path policy_name reason request_id status_code",
        authority_keys={"execution_profile_fingerprint"},
        public_authority_keys=_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
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
    policies[EventType.TASK_COMPLETION_RESULT_RESOLVED] = _observed_policy(
        "application_request_sha256 contract_fingerprint contract_id "
        "decision_id resolver_configuration_fingerprint resolver_id resolver_version "
        "result_digest result_kind result_reference_id task_id",
        authority_keys={
            "application_request_sha256",
            "contract_fingerprint",
            "contract_id",
            "decision_id",
            "resolver_configuration_fingerprint",
            "resolver_id",
            "resolver_version",
            "result_digest",
            "result_kind",
            "result_reference_id",
            "task_id",
        },
    )
    policies[EventType.TASK_INTERRUPTED_HANDOFF] = _observed_policy(
        "attempt handoff_id handoff_status session_run_epoch task_id",
        authority_keys={"handoff_id", "task_id"},
    )

    structured_policy = _observed_policy(
        "attempt errors execution_profile_fingerprint max_retries model_attempt_id model_step_id "
        "name output step strategy tool_round_id valid",
        authority_keys={"execution_profile_fingerprint"},
        public_authority_keys=_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
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
        "elapsed_ms execution_profile_fingerprint "
        "instruction_digest instruction_present mode model_step_id "
        "newly_compacted_message_count operation_id previous_compacted_transcript_cursor "
        "phase provider_dispatch_disposition provider_error_code provider_error_type "
        "provider_retryable reason recent_message_count recovery_action "
        "represented_message_count represented_source_end retry_disposition retryable "
        "represented_source_start request_id requested_source_end requested_source_start "
        "result_transcript_cursor source_run_epoch source_transcript_cursor status_code "
        "summary_chars retained_target retained_target_enforced retained_target_met "
        "estimated_context_input_tokens estimated_context_window_tokens "
        "estimated_window_within_trigger",
        authority_keys={"execution_profile_fingerprint"},
        public_authority_keys=_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
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
        "execution_profile_fingerprint "
        "max_attempts messages model model_attempt_id model_step_id observation_id options "
        "provider step tools",
        authority_keys={"execution_profile_fingerprint"},
        public_authority_keys=_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
        untrusted_container_keys={"messages", "options", "tools"},
    )
    policies[EventType.CONTEXT_COUNTED] = count_policy
    policies[EventType.CONTEXT_COUNT_FAILED] = count_policy
    pressure_policy = _observed_policy(
        "attempt estimate execution_profile_fingerprint max_attempts messages model model_attempt_id model_step_id "
        "observation_id options provider step tools",
        authority_keys={"execution_profile_fingerprint"},
        public_authority_keys=_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
        untrusted_container_keys={"messages", "options", "tools"},
    )
    policies[EventType.CONTEXT_PRESSURE_ESTIMATED] = pressure_policy
    reconciliation_policy = _observed_policy(
        "actual_input_tokens attempt delta_tokens execution_profile_fingerprint max_attempts model model_attempt_id "
        "model_step_id observation_id pre_call_count pre_call_estimate provider reconciled "
        "relative_error step",
        authority_keys={"execution_profile_fingerprint"},
        public_authority_keys=_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
    )
    policies[EventType.CONTEXT_COUNT_RECONCILED] = reconciliation_policy
    policies[EventType.CONTEXT_PRESSURE_RECONCILED] = reconciliation_policy
    overflow_policy = _observed_policy(
        "error error_type execution_profile_fingerprint model_attempt_id model_step_id "
        "original_message_count phase policy provider provider_error_code "
        "recovery_message_count status_code step",
        authority_keys={"execution_profile_fingerprint"},
        public_authority_keys=_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
    )
    policies[EventType.CONTEXT_OVERFLOW_DETECTED] = overflow_policy
    policies[EventType.CONTEXT_OVERFLOW_RECOVERING] = overflow_policy
    policies[EventType.CONTEXT_OVERFLOW_FAILED] = overflow_policy

    automatic_recall_policy = _observed_policy(
        "admission_truncated anchor_transcript_index configuration_sha256 contribution_sha256 duration_seconds "
        "error_type evaluated_candidate_count execution_profile_fingerprint focused_item_count "
        "manifest_sha256 model_step_id offered_item_count policy_sha256 projected_bytes "
        "recall_candidate_count recall_truncated silent_item_count situation_sha256 source_names "
        "source_statuses",
        authority_keys={"execution_profile_fingerprint"},
        public_authority_keys=_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
        untrusted_container_keys={"source_names", "source_statuses"},
    )
    for event_type in (
        EventType.AUTOMATIC_RECALL_STARTED,
        EventType.AUTOMATIC_RECALL_COMPLETED,
        EventType.AUTOMATIC_RECALL_FAILED,
        EventType.AUTOMATIC_RECALL_ADMITTED,
    ):
        policies[event_type] = automatic_recall_policy

    binding_policy = _observed_policy(
        "binding_cleanup binding_generation_id binding_type bound_metadata bound_path bound_snapshot "
        "bound_workspace_id configured_workspace_id environment_factory_release error "
        "error_type execution_profile_fingerprint factory_allocation_action failures final_git_receipt final_revision final_snapshot has_bound_runner "
        "has_configured_runner outcome source_publication_receipt source_workspace_id terminal_outcome",
        owned_nested_paths=(
            terminal_finalization_owned_paths
            | {
                ("source_publication_receipt", field_name)
                for field_name in {
                    "schema",
                    "receipt_sha256",
                    "snapshot_sha256",
                    "destination_workspace_id",
                    "workload_workspace_id",
                    "outcome",
                    "source_conflict_policy",
                    "sync_back",
                    "delete_missing",
                    "copied_files",
                    "copied_bytes",
                    "deleted_files",
                }
            }
            | {
                ("final_git_receipt", field_name)
                for field_name in {
                    "schema",
                    "receipt_sha256",
                    "request_fingerprint",
                    "destination_workspace_id",
                    "workload_workspace_id",
                    "baseline_revision",
                    "workspace_revision",
                }
            }
        ),
        authority_keys={"execution_profile_fingerprint"},
        public_authority_keys=_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
        untrusted_container_keys={
            "binding_cleanup",
            "bound_metadata",
            "environment_factory_release",
            "failures",
            "final_git_receipt",
            "final_revision",
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
        "environment_factory_release environment_name error error_type execution_profile_fingerprint factory_type labels "
        "parent_session_id reconnect_metadata requested_environment_name result_metadata",
        owned_nested_paths={
            ("reconnect_metadata", "allocation_fingerprint"),
        },
        authority_keys={"execution_profile_fingerprint"},
        public_authority_keys=_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
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
    materialization_policy = _observed_policy(
        "binding_generation_id elapsed_ms environment_name execution_profile_fingerprint mode "
        "reason trigger_tool_call_id trigger_tool_name",
        authority_keys={"binding_generation_id", "execution_profile_fingerprint"},
        public_authority_keys=_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
    )
    for event_type in (
        EventType.ENVIRONMENT_DEFERRED,
        EventType.ENVIRONMENT_MATERIALIZATION_STARTED,
        EventType.ENVIRONMENT_MATERIALIZATION_COMPLETED,
        EventType.ENVIRONMENT_MATERIALIZATION_FAILED,
    ):
        policies[event_type] = materialization_policy
    policies[EventType.ENVIRONMENT_LIFECYCLE_TRANSITION] = _observed_policy(
        "binding_generation_id candidate evidence_schema evidence_states evidence_valid_until "
        "executable_evidence_states execution_profile_fingerprint outcome ownership phase "
        "refusal_capabilities refusal_codes refusal_executable_sha256 release_action "
        "schema_version",
        authority_keys={"binding_generation_id", "execution_profile_fingerprint"},
        public_authority_keys=_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
        untrusted_container_keys={
            "evidence_states",
            "executable_evidence_states",
            "refusal_capabilities",
            "refusal_codes",
            "refusal_executable_sha256",
        },
    )

    hook_policy = _observed_policy(
        "actions durable_value_error_code durable_value_error_path error error_type execution_profile_fingerprint hook_index "
        "hook_invocation_id hook_name phase scope terminal_event_id terminal_event_type tool_call_id tool_name",
        authority_keys={"execution_profile_fingerprint", "hook_invocation_id"},
        public_authority_keys={
            *_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
            "hook_invocation_id",
        },
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
    policies[EventType.RUNTIME_INTERACTION_TRANSITION_ACKNOWLEDGEMENT_FAILED] = _observed_policy(
        "interaction_transition_failures transition_event_type",
        untrusted_container_keys={"interaction_transition_failures"},
    )
    policies[EventType.MEMORY_SEARCH] = _observed_policy(
        "hit_count query results truncated",
        untrusted_container_keys={"results"},
    )
    runner_policy = _observed_policy(
        "adapter approval_id cancelled command duration_ms error error_type execution_id execution_profile_fingerprint "
        "exit_code idempotency_key input_id model_attempt_id model_step_id timed_out tool_call_id "
        "tool_round_id",
        authority_keys=_TOOL_LINKAGE_AUTHORITY_KEYS,
        public_authority_keys=_EXECUTION_PROFILE_PUBLIC_AUTHORITY_KEYS,
        untrusted_container_keys={"command"},
    )
    policies[EventType.RUNNER_EXEC_STARTED] = runner_policy
    policies[EventType.RUNNER_EXEC_COMPLETED] = runner_policy

    # These counters are fixed runtime schema, not caller-selected object keys.
    # Register each diagnostic consumer without granting authority to its values
    # or to arbitrary sibling fields in the containing summary.
    for event_type, policy in tuple(policies.items()):
        accounting_paths = set()
        if "cost_summary" in policy.owned_keys:
            accounting_paths.update(
                ("cost_summary", key)
                for key in ("auxiliary_attempts", "unpriced_auxiliary_attempts")
            )
        if "usage_summary" in policy.owned_keys:
            accounting_paths.add(("usage_summary", "unmeasured_model_attempts"))
        if accounting_paths:
            policies[event_type] = replace(
                policy, owned_nested_paths=policy.owned_nested_paths | accounting_paths
            )

    return policies


EVENT_PAYLOAD_POLICIES: Mapping[EventType, EventPayloadPolicy] = _event_policies()


_INTERNAL_EVENT_PAYLOAD_POLICIES: Mapping[str, EventPayloadPolicy] = {
    WORKFLOW_ATTEMPT_EVENT_TYPE: _observed_policy("attempt_id"),
}


def event_payload_policy(event_type: EventType | str) -> EventPayloadPolicy:
    if isinstance(event_type, EventType):
        return EVENT_PAYLOAD_POLICIES[event_type]
    return _INTERNAL_EVENT_PAYLOAD_POLICIES.get(event_type, EventPayloadPolicy())


if set(EVENT_PAYLOAD_POLICIES) != set(EventType):
    raise AssertionError("Every built-in event type must have an exact payload policy.")
