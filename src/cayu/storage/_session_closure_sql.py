"""Content-free bounded SQL preflight for native closure record snapshots."""

# Foreign-key discovery is not ownership evidence. Only these schema-owned
# task dependencies may be explicitly removed by task closure. Unknown
# restrictive references must retain their database-enforced protection.
TASK_CLOSURE_DEPENDENCIES = frozenset(
    {
        ("cayu_work_attempts", "task_id"),
        ("cayu_completion_proposals", "task_id"),
        ("cayu_completion_decisions", "task_id"),
        ("cayu_completion_decision_application_receipts", "task_id"),
        ("cayu_completion_verifier_profiles", "task_id"),
        ("cayu_work_attempt_admissions", "task_id"),
        ("cayu_local_execution_attempts", "task_id"),
        ("cayu_work_attempt_preparation_holds", "task_id"),
        ("cayu_work_attempt_lifecycle_receipts", "task_id"),
    }
)


def closure_size_statement(*, postgres: bool) -> tuple[str, int]:
    parameter = "%s" if postgres else "?"
    sources = (
        ("cayu_recall_receipts", "session_id", ("receipt_json",)),
        ("cayu_context_exposures", "session_id", ("exposure_json",)),
        (
            "cayu_recall_item_exposures AS item JOIN cayu_context_exposures AS exposure "
            "ON exposure.exposure_id = item.exposure_id",
            "exposure.session_id",
            ("item.item_json",),
        ),
        (
            "cayu_deferred_interaction_inputs",
            "session_id",
            ("interaction_id", "source_messages" if postgres else "source_messages_json"),
        ),
        ("cayu_targeted_tool_grants", "session_id", ("record" if postgres else "record_json",)),
        ("cayu_targeted_tool_grant_uses", "session_id", ("record" if postgres else "record_json",)),
        (
            "cayu_sessions",
            "id",
            (
                "id",
                "instance_id",
                "agent_name",
                "provider_name",
                "model",
                "parent_session_id",
                "causal_budget_id",
                "runtime_name",
                "runtime_version",
                "environment_name",
                "status",
                "invocation" if postgres else "invocation_json",
                "metadata" if postgres else "metadata_json",
            ),
        ),
        ("cayu_session_labels", "session_id", ("key", "value")),
        (
            "cayu_events",
            "session_id",
            ("event",)
            if postgres
            else (
                "payload_json",
                "event_id",
                "session_id",
                "interaction_id",
                "agent_name",
                "environment_name",
                "workflow_name",
                "tool_name",
                "timestamp",
                "event_type",
            ),
        ),
        (
            "cayu_transcript_messages",
            "session_id",
            ("message" if postgres else "message_json", "interaction_id"),
        ),
        ("cayu_checkpoints", "session_id", ("state" if postgres else "state_json",)),
        (
            "cayu_session_message_queue",
            "session_id",
            (
                "queue_id",
                "idempotency_key",
                "content",
                "message_json",
                "conditions_json",
                "terminal_json",
                "requested_by" if postgres else "requested_by_json",
                "delivery_mode",
                "status",
                "accepted_event_id",
                "delivered_event_id",
            ),
        ),
        (
            "cayu_session_operations",
            "session_id",
            ("idempotency_key", "record" if postgres else "record_json"),
        ),
        (
            "cayu_persisted_event_side_effects",
            "session_id",
            ("event_id", "status", "claim_id", "last_error"),
        ),
        (
            "cayu_session_message_deliveries",
            "session_id",
            (
                "delivery_id",
                "interaction_id",
                "interaction_started_event" if postgres else "interaction_started_event_json",
                "queue_ids" if postgres else "queue_ids_json",
                "events" if postgres else "events_json",
            ),
        ),
    )
    selects = []
    for index, (table, identity, columns) in enumerate(sources):
        sizes = (
            [f"COALESCE(octet_length({column}::text), 0)" for column in columns]
            if postgres
            else [f"COALESCE(length(CAST({column} AS BLOB)), 0)" for column in columns]
        )
        selects.append(
            f"SELECT size FROM (SELECT {' + '.join(sizes)} AS size FROM {table} "
            f"WHERE {identity} = {parameter} LIMIT {parameter}) AS source_{index}"
        )
    return (
        "SELECT COUNT(*), COALESCE(SUM(size), 0) FROM ("
        + " UNION ALL ".join(selects)
        + ") AS closure_sizes",
        len(sources),
    )
