"""Versioned local qualification registry; selectors are repository-owned identities."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    name: str
    invariant: str
    boundary: str
    selectors: tuple[str, ...]


def core(file, *names):
    path = f"tests/core/test_{file}.py"
    return tuple(f"{path}::test_{name}" for name in names) if names else (path,)


SCENARIOS = (
    Scenario(
        "provider-failures",
        "Retryable failures honor bounded backoff while invalid completion stays explicit",
        "provider completion/error normalization before retry dispatch",
        core(
            "openai_provider",
            "runtime_honors_bounded_retry_after_from_successful_http_stream",
            "openai_protocol_failure_uses_bounded_unknown_retry_path",
            "openai_server_completion_validation_uses_bounded_unknown_retry_path",
        ),
    ),
    Scenario(
        "negative-controls",
        "Seeded authority loss, blind retry, semantic-stall acceptance, hot polling and leaks fail the same assertions",
        "claim, dispatch, stream progress and cleanup",
        ("tests/qualification/test_negative_controls.py",),
    ),
    Scenario(
        "capacity",
        "Concurrent durable tasks finish once and release claims and tool work",
        "task claim to session and task terminal publication",
        ("tests/qualification/test_capacity.py",),
    ),
    Scenario(
        "publication",
        "Pre-commit rollback and committed acknowledgement loss retain authority",
        "native SessionStore publication transaction",
        core("session_operation_fault_harness"),
    ),
    Scenario(
        "tool-publication",
        "Tool terminal evidence agrees with transcript after publication faults",
        "tool result terminal publication",
        core("tool_round_publication_failure_matrix"),
    ),
    Scenario(
        "replay",
        "Completed contract replay performs no provider or external tool effects",
        "recorded completed-session contract",
        ("tests/evals/test_runtime_contract_replay.py",),
    ),
    Scenario(
        "dynamic-tools",
        "Rejected dynamic references rediscover without leaking tool authority",
        "durable targeted-tool grant",
        core("dynamic_tool_reference_recovery"),
    ),
    Scenario(
        "provider-history",
        "Provider-native history survives redaction without protocol corruption",
        "redacted checkpoint and provider request",
        core("explicit_compaction_transcript_redaction"),
    ),
    Scenario(
        "compaction",
        "Long and wide tool rounds compact atomically within the effective request bound",
        "compaction checkpoint publication",
        core(
            "explicit_session_compaction",
            "size_based_compaction_uses_configured_attachment_retention",
            "size_based_compaction_compacts_an_oversized_latest_tool_round_atomically",
            "size_based_compaction_checkpoint_survives_sqlite_restart",
            "size_based_compaction_runs_through_a_real_long_tool_loop",
            "size_based_compaction_continues_after_a_wide_oversized_tool_round",
        ),
    ),
    Scenario(
        "provider-deadlines",
        "Byte activity cannot mask semantic stalls or extend absolute lifetime",
        "provider dispatch and durable error classification",
        core("provider_stream_deadlines"),
    ),
    Scenario(
        "provider-recovery",
        "Interrupted and ambiguous completions remain explicit without blind redispatch",
        "model completion record",
        core("model_completion_recovery"),
    ),
    Scenario(
        "fresh-process",
        "SIGKILL recovery reconstructs the app and respects persisted tool and task authority",
        "model dispatch, tool execution, approval, task claim and attachment",
        (
            "tests/recovery/test_sigkill_recovery.py",
            *core(
                "approval_from_event", "pending_approval_payload_is_stable_at_terminal_publication"
            ),
        ),
    ),
    Scenario(
        "checkpoint-crash",
        "Checkpoint and event publication remain recoverable after process loss",
        "workspace checkpoint publication",
        ("tests/runtime/test_workspace_checkpoint_publication.py",),
    ),
    Scenario(
        "worker-authority",
        "Stale workers cannot dispatch and cancellation retains ownership until settlement",
        "claim readback, lease renewal and task terminalization",
        core(
            "task_worker",
            "task_worker_revalidates_before_dispatch_after_delayed_claim_acknowledgement",
            "task_worker_stops_handler_when_heartbeat_stalls_past_lease",
            "task_worker_cancellation_fences_opaque_handler_until_natural_settlement",
            "run_task_worker_reconciles_failure_terminalization_acknowledgement_loss",
            "run_task_worker_hands_interrupted_session_to_reconstructed_control_plane",
            "remote_interrupt_wins_linked_task_failure_and_preserves_queued_turn",
        ),
    ),
    Scenario(
        "environments",
        "Cancellation and exact cleanup retries retain resources until quiescence",
        "environment bind, finalize and release receipt",
        core("environment_lifecycle")
        + core(
            "invocation_lifecycle_commands",
            "invocation_context_preserves_exact_live_authority_references",
        ),
    ),
    Scenario(
        "environment-contention",
        "Conflicting environment allocations retain one authoritative owner",
        "environment allocation claim",
        ("tests/environments/test_environment_binding.py",),
    ),
    Scenario(
        "operator-recovery",
        "Bounded inspection and recovery reject stale plans and converge cleanup",
        "recovery plan selection and fenced execution",
        core("recovery_plans") + core("recovery_cleanup"),
    ),
)

POSTGRES_SCENARIOS = (
    Scenario(
        "postgres-cleanup",
        "Owned databases are removed after normal, abrupt and forced pytest shutdown",
        "parent-owned database creation and verified deletion after process termination",
        ("tests/qualification/test_postgres_cleanup.py",),
    ),
    Scenario(
        "postgres-authority",
        "PostgreSQL publication faults and interrupted handoffs reconcile exactly",
        "native PostgreSQL commit and claim release",
        core("postgres_session_store", "postgres_session_operation_fault_conformance")
        + core("postgres_task_store", "postgres_interrupted_task_handoff_faults_reconcile_exactly"),
    ),
)
