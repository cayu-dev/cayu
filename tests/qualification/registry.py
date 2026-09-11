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


_MAINTENANCE_APPLICATION = "tests/qualification/test_repository_maintenance_application.py"


def _maintenance_module(name, invariant, boundary):
    return Scenario(
        f"repository-maintenance-{name.replace('_', '-')}",
        invariant,
        boundary,
        (f"tests/qualification/test_repository_maintenance_{name}.py",),
    )


# Each scenario keeps the runner's existing 300-second default ownership bound.
# Full journeys are independently selected: the combined application suite takes
# more than that limit even when every individual test passes.
MAINTENANCE_SCENARIOS = (
    Scenario(
        "repository-maintenance-corpus",
        "The fixed seed fails independent acceptance and test weakening cannot pass it",
        "pinned seed bytes, ordered probe responses and declared toolchain",
        tuple(
            f"tests/qualification/test_repository_maintenance_{name}.py"
            for name in ("case", "probe", "toolchain")
        ),
    ),
    Scenario(
        "repository-maintenance-application-contract",
        "Generated construction retains canonical layout and exact admission controls",
        "emitted source, image readback and public factory preparation",
        tuple(
            f"{_MAINTENANCE_APPLICATION}::{name}"
            for name in (
                "test_consumer_uses_public_imports_and_canonical_layout",
                "test_consumer_declares_server_runtime_dependency_without_service_preset",
                "test_image_builder_captures_probe_and_rejects_post_capture_changes",
                "test_generated_image_readback_constructs_target_specific_profile",
                "test_public_factory_constructs_and_prepares_the_bounded_workflow",
            )
        ),
    ),
    *(
        Scenario(
            f"repository-maintenance-journey-{name}",
            "The selected bounded journey independently checks publication, authority and cleanup",
            f"generated workflow and persistent or in-memory owners: {name}; controlled providers only",
            (
                f"{_MAINTENANCE_APPLICATION}::test_public_workflow_seals_verifies_and_replays[{node}]",
            ),
        )
        for name, node in (
            ("memory-normal", "memory-False-False-False-False-False-False"),
            ("memory-rejected", "memory-True-False-False-False-False-False"),
            ("sqlite-normal", "sqlite-False-False-False-False-False-False"),
            ("sqlite-rejected", "sqlite-True-False-False-False-False-False"),
            *(
                (name, name)
                for name in (
                    "sqlite-lost-push-ack",
                    "sqlite-lost-github-ack",
                    "memory-managed-worker",
                    "sqlite-managed-worker",
                    "sqlite-managed-lost-push-ack",
                    "sqlite-managed-lost-github-ack",
                    "sqlite-approval-process-restart",
                    "sqlite-managed-worker-rejected",
                    "memory-workflow-eval",
                    "sqlite-workflow-eval",
                    "sqlite-workflow-eval-repeated",
                    "sqlite-workflow-eval-dirty-source",
                    "memory-workflow-eval-rejected",
                    "sqlite-workflow-eval-rejected",
                )
            ),
        )
    ),
    *(
        _maintenance_module(*row)
        for row in (
            (
                "identity",
                "Reserved identities are immutable and phase-specific",
                "identity validation and copying",
            ),
            (
                "runs",
                "Reservations atomically bind the exact tenant and request",
                "SQLite/PostgreSQL reservation and readback",
            ),
            (
                "intake",
                "Exact intake cannot grant a conflicting task",
                "reservation to Runtime task creation",
            ),
            (
                "budget",
                "Bounded priced policy is required before work",
                "application budget validation",
            ),
            (
                "deadline",
                "Original elapsed authority survives handoffs",
                "reservation deadline through workflow admission",
            ),
            (
                "eval",
                "Repeated coding-stage evaluation retains trial resources and fixed contracts",
                "emitted native corpus target, per-trial quiescence and context cleanup",
            ),
            (
                "request",
                "Accepted configuration binds later execution",
                "configuration capture and reconstruction",
            ),
            (
                "worker",
                "Rejected claims settle without coding dispatch",
                "native worker to emitted coding handler",
            ),
            (
                "http",
                "Product access cannot bypass tenant authority",
                "HTTP intake and resource lookup",
            ),
            (
                "operator",
                "Only authorized operators discover allocated references",
                "operator HTTP resource projection",
            ),
            (
                "operator_tasks",
                "Task observations do not grant recovery authority",
                "four reserved task projections",
            ),
            (
                "delivery_view",
                "Native effects and task outcomes remain distinct",
                "Git/GitHub historical projection",
            ),
            (
                "cost",
                "Native cohort accounting retains unknown charges",
                "causal budget reporting projection",
            ),
            (
                "final",
                "Final output requires exact verified delivery evidence",
                "PR URL, acceptance and cost projection",
            ),
            (
                "results",
                "Only completed verified coding grants a handoff",
                "durable task and product readback",
            ),
            (
                "git_configuration",
                "Git host configuration retains credential boundaries",
                "host broker construction",
            ),
            (
                "delivery_configuration",
                "Delivery requests require explicit native authority",
                "Git request configuration",
            ),
            (
                "git_intake",
                "Git preparation retains exact request identity",
                "verified coding to preparation task",
            ),
            (
                "git_approval",
                "Git consent binds the reviewed tree and destination",
                "pending preparation to approved task",
            ),
            (
                "git_worker",
                "Git ownership remains held until dispatched work settles",
                "Runtime claim through native broker",
            ),
            (
                "git_results",
                "Only exact pushed evidence grants the downstream handoff",
                "native Git result reconstruction",
            ),
            (
                "github_intake",
                "PR consent is separate from Git consent",
                "Git result to GitHub task",
            ),
            (
                "github_results",
                "PR result requires current exact head evidence",
                "native GitHub result readback",
            ),
            ("github_http", "Only operator authority can approve PR work", "GitHub approval HTTP"),
            (
                "github_worker",
                "PR work retains task authority during cleanup",
                "Runtime claim through connector lifetime",
            ),
            (
                "github_host",
                "Host repository configuration excludes embedded credentials",
                "GitHub native host configuration",
            ),
            (
                "git_http",
                "Git approval HTTP cannot replace actor or tree authority",
                "operator review and approval routes",
            ),
            (
                "github",
                "Connector observation preserves native poll and cleanup ownership",
                "run, poll, seal and close",
            ),
            (
                "deployment",
                "Application roles share validated durable dependencies",
                "generated factory and native store construction",
            ),
            (
                "schema",
                "Schema preparation delegates to its explicit owner",
                "reservation schema CLI",
            ),
            (
                "compose",
                "Deployment assets preserve role and resource boundaries",
                "emitted Compose and image contracts",
            ),
            (
                "coding_loss",
                "Process loss cannot grant redispatch or false completion",
                "generated SQLite SIGKILL and registered recovery",
            ),
            (
                "incidents",
                "Runbook commands match supported recovery entrances",
                "emitted guide and CLI parser",
            ),
            (
                "configuration",
                "Canonical factory requires explicit shared policy",
                "configured application factory",
            ),
            (
                "access_configuration",
                "Access configuration separates product and operator principals",
                "configuration parser to HTTP authentication",
            ),
            (
                "roles",
                "Named roles preserve stop and cancellation ownership",
                "CLI signals and retained worker lifetime",
            ),
            (
                "asgi",
                "Request settlement precedes dependency release",
                "ASGI admission seal and shutdown",
            ),
            (
                "api",
                "API construction retains startup and cleanup authority",
                "generated API factory and host lifetime",
            ),
            (
                "shutdown",
                "Dependencies close only after positive owned drains",
                "retained deployment shutdown task",
            ),
            (
                "eval_lineage",
                "Evaluation preserves workflow-root identity",
                "coding workflow and eval lineage",
            ),
            (
                "policy",
                "Declared bounds do not expand repository authority",
                "maintenance policy construction",
            ),
            (
                "acceptance",
                "Agent-authored evidence cannot replace the independent oracle",
                "sealed product acceptance",
            ),
        )
    ),
    Scenario(
        "repository-maintenance-harness-ownership",
        "Process harness does not delete caller-owned application stores",
        "helper process and backend cleanup ownership",
        ("tests/recovery/test_worker_harness_ownership.py",),
    ),
    Scenario(
        "repository-maintenance-subagent-drain",
        "Background children remain owned until positive settlement",
        "public subagent registry drain and application lifetime",
        ("tests/core/test_subagent_registry_drain.py",),
    ),
)


SCENARIOS = (
    Scenario(
        "github-delivery-lifecycle",
        "Sealed delivery retains opaque mutations until local quiescence is observed",
        "public connector shutdown through provider and artifact settlement",
        core("github_delivery_lifecycle"),
    ),
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
        "coding-product-recovery",
        "Settled coding products retain exact authority and recover publication without redispatch",
        "retained source and invocation evidence to exact product publication and generated workflow",
        (
            *core("coding_product_recovery"),
            *core("coding_product_lineage"),
            "tests/cli/test_scaffold_coding_recovery.py",
        ),
    ),
    *MAINTENANCE_SCENARIOS,
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
        "docker-allocation",
        "Allocation retries converge and released immutable owners remain terminal",
        "immutable admission, container acknowledgement and allocation reaping",
        (
            "tests/environments/test_docker_coding.py",
            "tests/test_immutable_inputs.py",
            "tests/test_filesystem_lock.py",
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


DOCKER_SCENARIOS = (
    Scenario(
        "docker-allocation-live",
        "Interrupted tasks continue once in a successor allocation after exact predecessor disposal",
        "model-step handoff, Docker disposal, fresh-process reconstruction and immutable references",
        (
            "tests/environments/test_docker_coding_live.py::test_real_docker_reallocate_after_confirmed_disposal",
            "tests/environments/test_docker_allocation_lifetime_live.py",
        ),
    ),
)
