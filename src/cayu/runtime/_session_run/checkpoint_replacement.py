"""Checkpoint replacement that keeps runtime-owned state."""

from __future__ import annotations

from typing import Any

from cayu._validation import copy_durable_record, copy_json_value
from cayu.runtime import _prompt_transition, _session_operation_state
from cayu.runtime._durable_subagents import (
    DURABLE_SUBAGENT_SUBMISSION_SEEDS_CHECKPOINT_KEY,
    DURABLE_SUBAGENT_SUBMISSIONS_CHECKPOINT_KEY,
)
from cayu.runtime._interruption_coordinator import (
    _PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY,
    _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY,
)
from cayu.runtime._shared_allocation_records import SHARED_ALLOCATIONS_CHECKPOINT_KEY
from cayu.runtime.provider_operations import (
    _PROVIDER_OPERATION_FALLBACK_ORDINALS_CHECKPOINT_KEY,
    _PROVIDER_OPERATION_PENDING_DISPOSITION_CHECKPOINT_KEY,
)
from cayu.sessions._checkpoint_publication import INITIAL_TRANSCRIPT_PENDING_CHECKPOINT_KEY
from cayu.sessions._terminal_evidence import (
    _QUEUED_DISPATCH_TERMINAL_RECEIPTS_CHECKPOINT_KEY,
    _SESSION_RUN_OPERATION_CHECKPOINT_KEY,
)
from cayu.sessions.checkpoints import (
    ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
    CHECKPOINT_SCHEMA_VERSION_KEY,
    COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
    INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
    INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
    SETTLED_INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
)
from cayu.sessions.records import Session
from cayu.sessions.recovery import _INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY


def _replace_checkpoint_preserving_runtime_state(
    checkpoint: dict[str, Any],
):
    replacement = copy_durable_record(checkpoint, "checkpoint")

    def transform(_session: Session, current: dict[str, Any] | None) -> dict[str, Any]:
        updated = copy_durable_record(replacement, "checkpoint")
        updated[CHECKPOINT_SCHEMA_VERSION_KEY] = CURRENT_CHECKPOINT_SCHEMA_VERSION
        for key, field_name in (
            ("workspace_checkpoints", "workspace_checkpoints"),
            (
                ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
                "active_invocation_execution_profile",
            ),
            (
                INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
                "invocation_lifecycle_receipt",
            ),
            (
                INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
                "invocation_terminal_decision",
            ),
            (
                SETTLED_INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
                "settled_invocation_terminal_decision",
            ),
            (
                INITIAL_TRANSCRIPT_PENDING_CHECKPOINT_KEY,
                "initial_transcript_pending",
            ),
            (_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY, "pending_session_interrupt"),
            (_PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY, "pending_interruption_cascade"),
            (_session_operation_state._SESSION_OPERATIONS_CHECKPOINT_KEY, "session_operations"),
            (_SESSION_RUN_OPERATION_CHECKPOINT_KEY, "session_run_operation"),
            (
                _QUEUED_DISPATCH_TERMINAL_RECEIPTS_CHECKPOINT_KEY,
                "queued_dispatch_terminal_receipts",
            ),
            (
                _prompt_transition._PROMPT_ANATOMY_TRANSITION_INTENTS_CHECKPOINT_KEY,
                "prompt_anatomy_transition_intents",
            ),
            (
                _INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY,
                "incomplete_session_recovery_claim",
            ),
            (
                DURABLE_SUBAGENT_SUBMISSION_SEEDS_CHECKPOINT_KEY,
                "durable_subagent_submission_seeds",
            ),
            (
                DURABLE_SUBAGENT_SUBMISSIONS_CHECKPOINT_KEY,
                "durable_subagent_submissions",
            ),
            (
                _PROVIDER_OPERATION_FALLBACK_ORDINALS_CHECKPOINT_KEY,
                "provider_operation_fallback_dispatch_ordinals",
            ),
            (
                _PROVIDER_OPERATION_PENDING_DISPOSITION_CHECKPOINT_KEY,
                "provider_operation_pending_resolution_disposition",
            ),
            (
                COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY,
                "completion_result_event_publications",
            ),
            (
                "recovery_plan_execution",
                "recovery_plan_execution",
            ),
            # An agent session's shared allocations, which its subagents and a
            # resumed run in another process change while this run works.
            (SHARED_ALLOCATIONS_CHECKPOINT_KEY, "shared_environment_allocations"),
        ):
            updated.pop(key, None)
            if current is not None and key in current:
                updated[key] = copy_json_value(current[key], field_name)
        return updated

    return transform
