"""Postgres SessionStore parity + concurrency tests.

These mirror the SQLite/InMemory conformance assertions in
``test_sqlite_session_store.py`` and ``test_session_store_queries.py`` so the
identical behavioral contract is proven against a real Dockerized Postgres.
They skip automatically when Docker is unavailable (see ``conftest.py``).
"""

from __future__ import annotations

import asyncio
import base64
import json
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import SecretStr, ValidationError
from tests.core.checkpoint_schema_conformance import (
    assert_assistant_publication_checkpoint_conformance,
    assert_current_checkpoint_publication_upgrade_conformance,
    assert_future_checkpoint_rejection_conformance,
    assert_reserved_checkpoint_key_migration_conformance,
    assert_runtime_publication_rejects_invocation_authority_mutation,
    assert_versionless_noop_transform_stamps_conformance,
    assert_versionless_pending_continuation_fails_closed_conformance,
)
from tests.core.pending_action_conformance import assert_pending_action_store_conformance
from tests.core.session_topology_conformance import (
    assert_session_topology_store_conformance,
)
from tests.core.test_fork_groups import assert_viable_fork_group_store_conformance
from tests.core.test_provider_operation_offline_recovery import (
    assert_budgeted_offline_provider_operation_recovery,
    assert_offline_provider_operation_recovery,
    assert_offline_provider_operation_reuses_run_limit_accounting,
    assert_pending_provider_operation_later_completes,
    assert_provider_resolution_process_loss_recovery,
    assert_terminal_session_fails_closed_with_active_provider_operation,
    stage_provider_resolution_process_loss,
)
from tests.core.tool_result_projection_conformance import (
    assert_tool_result_projection_recovery_conformance,
    assert_tool_result_projection_session_store_conformance,
)
from tests.workspaces.test_durable_local_workspace_branches import (
    assert_durable_workspace_branch_store_conformance,
)

from cayu import ExecutionProfileComponentClass, LocalArtifactStore
from cayu.core import Event, EventType, Message
from cayu.providers import ProviderOperationStatus
from cayu.runtime import (
    EventOrder,
    EventQuery,
    InvocationOriginClaim,
    InvocationOriginTrust,
    RunLimits,
    RunRequest,
    Session,
    SessionDebugState,
    SessionExecutionSource,
    SessionIdentity,
    SessionLineageQuery,
    SessionOrder,
    SessionQuery,
    SessionRunFenced,
    SessionStatus,
    SessionTopologyCycle,
    SessionTopologyQuery,
    TranscriptQuery,
)
from cayu.runtime.provider_operations import ProviderOperationResolutionAction
from cayu.runtime.public_authority import (
    PublicAuthorityAliasCodec,
    PublicAuthorityAliasKeyring,
)
from cayu.runtime.sessions import (
    EventQueryResultTooLarge,
    PendingActionKind,
    PendingActionQuery,
    _McpManifestBaselineEvidenceInvalid,
    fork_session_invocation,
)

pytestmark = pytest.mark.usefixtures("postgres_dsn")

_TABLES = (
    "cayu_knowledge_embeddings",
    "cayu_knowledge_index_readiness_current",
    "cayu_knowledge_index_readiness_events",
    "cayu_budget_settlements",
    "cayu_budget_reservations",
    "cayu_task_terminalization_receipts",
    "cayu_knowledge_change_acknowledgements",
    "cayu_knowledge_change_consumers",
    "cayu_knowledge_change_labels",
    "cayu_knowledge_change_audiences",
    "cayu_knowledge_changes",
    "cayu_knowledge_evidence",
    "cayu_knowledge_publication_receipts",
    "cayu_knowledge_labels",
    "cayu_knowledge_aspects",
    "cayu_knowledge_impact_targets",
    "cayu_knowledge_chunks",
    "cayu_knowledge_revisions",
    "cayu_knowledge_entries",
    "cayu_event_watcher_state",
    "cayu_budget_reservation_identities",
    "cayu_events",
    "cayu_session_labels",
    "cayu_public_authority_aliases",
    "cayu_public_authority_alias_keys",
    "cayu_public_authority_alias_config",
    "cayu_transcript_search_configuration",
    "cayu_transcript_messages",
    "cayu_session_message_queue",
    "cayu_persisted_event_side_effects",
    "cayu_mcp_manifest_baselines",
    "cayu_checkpoints",
    "cayu_session_operations",
    "cayu_tasks",
    "cayu_sessions",
    "cayu_eval_baseline_mutations",
    "cayu_eval_baselines",
    "cayu_eval_result_records",
    "cayu_eval_results",
    "cayu_eval_runs",
    "cayu_eval_cases",
    "cayu_eval_suites",
    "cayu_eval_corpora",
    "cayu_schema_migrations",
)


def _identity() -> SessionIdentity:
    return SessionIdentity(provider_name="fake", model="fake-model")


def _public_authority_codec(
    *,
    active_key_id: str = "primary",
    key_byte: int = 7,
) -> PublicAuthorityAliasCodec:
    encoded_key = base64.urlsafe_b64encode(bytes([key_byte]) * 32).decode().rstrip("=")
    return PublicAuthorityAliasCodec(
        PublicAuthorityAliasKeyring(
            active_key_id=active_key_id,
            keys={active_key_id: SecretStr(encoded_key)},
        )
    )


def _tool_round_identity_payload() -> dict[str, str]:
    return {
        "model_step_id": f"mstep_{'1' * 32}",
        "model_attempt_id": f"matt_{'2' * 32}",
        "tool_round_id": f"tround_{'3' * 32}",
    }


async def _truncate(dsn: str) -> None:
    import psycopg

    async with await psycopg.AsyncConnection.connect(dsn) as conn:
        async with conn.cursor() as cur:
            for table in _TABLES:
                await cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        await conn.commit()


def _new_store(dsn: str):
    from cayu import PostgresSessionStore
    from cayu.storage.migrations import SchemaMode

    # Tests own a throwaway database and (re)create the schema each run.
    return PostgresSessionStore(dsn, min_size=1, max_size=4, schema_mode=SchemaMode.CREATE)


def _run(dsn: str, coro_factory) -> object:
    async def runner():
        await _truncate(dsn)
        store = _new_store(dsn)
        try:
            return await coro_factory(store)
        finally:
            await store.close()

    return asyncio.run(runner())


def test_postgres_viable_fork_group_store_conformance(postgres_dsn: str) -> None:
    _run(postgres_dsn, assert_viable_fork_group_store_conformance)


def test_postgres_session_store_preserves_projected_tool_results(
    postgres_dsn: str,
    tmp_path,
) -> None:
    async def ops(store) -> None:
        await assert_tool_result_projection_session_store_conformance(
            store,
            LocalArtifactStore(tmp_path / "postgres-projection-artifacts"),
            session_id="sess_projection_postgres",
        )
        await assert_tool_result_projection_recovery_conformance(
            store,
            LocalArtifactStore(tmp_path / "postgres-projection-recovery-artifacts"),
            session_id="sess_projection_recovery_postgres",
        )

    _run(postgres_dsn, ops)


def test_postgres_pending_action_store_conformance(postgres_dsn: str) -> None:
    _run(postgres_dsn, assert_pending_action_store_conformance)


def test_postgres_durable_workspace_branch_conformance(postgres_dsn: str, tmp_path) -> None:
    async def ops(store):
        await assert_durable_workspace_branch_store_conformance(
            store,
            tmp_path / "postgres-durable-workspace",
        )

    _run(postgres_dsn, ops)


def test_postgres_offline_provider_operation_recovery(postgres_dsn: str) -> None:
    _run(postgres_dsn, assert_offline_provider_operation_recovery)


@pytest.mark.parametrize(
    ("action", "after_status_transition"),
    [
        (ProviderOperationResolutionAction.FALLBACK_RETRY, False),
        (ProviderOperationResolutionAction.FALLBACK_RETRY, True),
        (ProviderOperationResolutionAction.FAIL, False),
        (ProviderOperationResolutionAction.FAIL, True),
    ],
)
def test_postgres_provider_resolution_process_loss_finishes_disposition(
    postgres_dsn: str,
    action: ProviderOperationResolutionAction,
    after_status_transition: bool,
) -> None:
    async def scenario() -> None:
        await _truncate(postgres_dsn)
        store = _new_store(postgres_dsn)
        try:
            session_id, provider = await stage_provider_resolution_process_loss(
                store,
                action=action,
                after_status_transition=after_status_transition,
            )
        finally:
            await store.close()

        reopened = _new_store(postgres_dsn)
        try:
            await assert_provider_resolution_process_loss_recovery(
                reopened,
                session_id=session_id,
                provider=provider,
                action=action,
            )
        finally:
            await reopened.close()

    asyncio.run(scenario())


def test_postgres_terminal_session_fails_closed_with_active_provider_operation(
    postgres_dsn: str,
) -> None:
    _run(postgres_dsn, assert_terminal_session_fails_closed_with_active_provider_operation)


def test_postgres_budgeted_offline_provider_operation_recovery(postgres_dsn: str) -> None:
    async def ops(store) -> None:
        from cayu import PostgresBudgetLedger
        from cayu.storage.migrations import SchemaMode

        ledger = PostgresBudgetLedger(
            postgres_dsn,
            min_size=1,
            max_size=4,
            schema_mode=SchemaMode.CREATE,
            reservation_ttl_seconds=None,
        )
        try:
            await assert_budgeted_offline_provider_operation_recovery(store, ledger)
        finally:
            await ledger.close()

    _run(postgres_dsn, ops)


@pytest.mark.parametrize("limit_kind", ["tokens", "tools", "elapsed", "cost"])
def test_postgres_offline_provider_operation_reuses_run_limit_accounting(
    postgres_dsn: str,
    limit_kind: str,
) -> None:
    async def ops(store) -> None:
        await assert_offline_provider_operation_reuses_run_limit_accounting(
            store,
            limit_kind=limit_kind,
        )

    _run(postgres_dsn, ops)


@pytest.mark.parametrize("override_kind", ["changed_limits", "same_budget_limits"])
def test_postgres_offline_field_restatement_obeys_frozen_profile(
    postgres_dsn: str,
    override_kind: str,
) -> None:
    async def ops(store) -> None:
        await assert_offline_provider_operation_reuses_run_limit_accounting(
            store,
            limit_kind="cost" if override_kind == "changed_limits" else "tokens",
            approval_limits=(
                RunLimits(max_total_tokens=1_000, scope="run")
                if override_kind == "changed_limits"
                else None
            ),
            approval_budget_limits=(() if override_kind == "same_budget_limits" else None),
            expected_profile_change=(
                ExecutionProfileComponentClass.FINALIZATION
                if override_kind == "changed_limits"
                else None
            ),
        )

    _run(postgres_dsn, ops)


@pytest.mark.parametrize(
    "initial_status",
    [ProviderOperationStatus.QUEUED, ProviderOperationStatus.IN_PROGRESS],
)
def test_postgres_pending_provider_operation_later_completes(
    postgres_dsn: str,
    initial_status: ProviderOperationStatus,
) -> None:
    async def ops(store) -> None:
        await assert_pending_provider_operation_later_completes(
            store,
            initial_status=initial_status,
        )

    _run(postgres_dsn, ops)


def test_postgres_session_topology_store_conformance(postgres_dsn: str) -> None:
    _run(postgres_dsn, assert_session_topology_store_conformance)


def test_postgres_session_lineage_paginates_tied_unicode_identifiers(
    postgres_dsn: str,
) -> None:
    async def ops(store) -> None:
        import psycopg

        await store.create(
            RunRequest(agent_name="agent", session_id="unicode-parent", messages=[]),
            identity=_identity(),
        )
        child_ids = ("unicode-a", "unicode-ä", "unicode-z", "unicode-Z")
        for child_id in child_ids:
            await store.create(
                RunRequest(
                    agent_name="agent",
                    session_id=child_id,
                    parent_session_id="unicode-parent",
                    messages=[],
                ),
                identity=_identity(),
            )

        tied_at = datetime(2026, 1, 1, tzinfo=UTC)
        async with await psycopg.AsyncConnection.connect(postgres_dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE cayu_sessions SET created_at = %s WHERE id = ANY(%s)",
                    (tied_at, list(child_ids)),
                )
            await conn.commit()

        complete = await store.query_session_lineage(
            SessionLineageQuery(parent_session_id="unicode-parent", limit=4)
        )
        assert [child.id for child in complete.children] == sorted(child_ids)

        observed: list[str] = []
        cursor = None
        while True:
            page = await store.query_session_lineage(
                SessionLineageQuery(
                    parent_session_id="unicode-parent",
                    cursor=cursor,
                    limit=1,
                )
            )
            observed.extend(child.id for child in page.children)
            if not page.has_more:
                break
            assert page.next_cursor is not None
            assert page.next_cursor != cursor
            cursor = page.next_cursor

        assert observed == [child.id for child in complete.children]
        assert set(observed) == set(child_ids)

    _run(postgres_dsn, ops)


def test_postgres_public_authority_aliases_are_indexed_and_durable(
    postgres_dsn: str,
) -> None:
    async def runner() -> None:
        from cayu import PostgresSessionStore
        from cayu.storage.migrations import SchemaMode

        await _truncate(postgres_dsn)
        codec = _public_authority_codec()
        store = PostgresSessionStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
            public_authority_alias_codec=codec,
        )
        session_id = "session-private-authority"
        interaction_id = "interaction-private-authority"
        event_interaction_id = "event-written-interaction"
        transcript_interaction_id = "transcript-written-interaction"
        nested_interaction_id = "nested-written-interaction"
        session_alias = codec.encode(session_id, field_name="session_id")
        interaction_alias = codec.encode(
            interaction_id,
            field_name="interaction_id",
            session_id=session_id,
        )
        try:
            assert store.public_authority_alias_codec is codec
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            await store.register_public_authority_alias(
                interaction_alias,
                field_name="interaction_id",
                private_value=interaction_id,
                scope_session_id=session_id,
            )
            await store.append_events(
                session_id,
                [
                    Event(
                        type=EventType.TURN_COMPLETED,
                        session_id=session_id,
                        interaction_id=event_interaction_id,
                        payload={"interaction_ids": [nested_interaction_id]},
                    )
                ],
            )
            await store.append_transcript_messages(
                session_id,
                [Message.text("user", "indexed")],
                interaction_id=transcript_interaction_id,
            )
            assert (
                await store.resolve_public_authority_alias(
                    session_alias,
                    field_name="session_id",
                )
                == session_id
            )
            assert (
                await store.resolve_public_authority_alias(
                    interaction_alias,
                    field_name="interaction_id",
                    scope_session_id=session_id,
                )
                == interaction_id
            )
            assert (
                await store.resolve_public_authority_alias(
                    interaction_alias,
                    field_name="interaction_id",
                    scope_session_id="another-session",
                )
                is None
            )
            for indexed_interaction_id in (
                event_interaction_id,
                transcript_interaction_id,
                nested_interaction_id,
            ):
                indexed_alias = codec.encode(
                    indexed_interaction_id,
                    field_name="interaction_id",
                    session_id=session_id,
                )
                assert (
                    await store.resolve_public_authority_alias(
                        indexed_alias,
                        field_name="interaction_id",
                        scope_session_id=session_id,
                    )
                    == indexed_interaction_id
                )
        finally:
            await store.close()

        reopened = PostgresSessionStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.VALIDATE,
            public_authority_alias_codec=codec,
        )
        try:
            assert (
                await reopened.resolve_public_authority_alias(
                    session_alias,
                    field_name="session_id",
                )
                == session_id
            )
            assert (
                await reopened.resolve_public_authority_alias(
                    interaction_alias,
                    field_name="interaction_id",
                    scope_session_id=session_id,
                )
                == interaction_id
            )
        finally:
            await reopened.close()

        secondary_key = base64.urlsafe_b64encode(bytes([11]) * 32).decode().rstrip("=")
        rotated = codec.rotated(
            active_key_id="secondary",
            key=SecretStr(secondary_key),
        )
        stale_store = PostgresSessionStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.VALIDATE,
            public_authority_alias_codec=codec,
        )
        assert await stale_store.load(session_id) is not None
        rotated_store = PostgresSessionStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.VALIDATE,
            public_authority_alias_codec=rotated,
        )
        rotated_session_alias = rotated.encode(session_id, field_name="session_id")
        try:
            assert (
                await rotated_store.resolve_public_authority_alias(
                    rotated_session_alias,
                    field_name="session_id",
                )
                == session_id
            )
            assert (
                await rotated_store.resolve_public_authority_alias(
                    session_alias,
                    field_name="session_id",
                )
                == session_id
            )
            with pytest.raises(RuntimeError, match="configuration is stale"):
                await stale_store.load(session_id)
        finally:
            await stale_store.close()
            await rotated_store.close()

        retired = rotated.rotated(
            active_key_id="secondary",
            key=SecretStr(secondary_key),
            retire_key_ids=("primary",),
        )
        retired_store = PostgresSessionStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.VALIDATE,
            public_authority_alias_codec=retired,
        )
        try:
            assert (
                await retired_store.resolve_public_authority_alias(
                    session_alias,
                    field_name="session_id",
                )
                is None
            )
            assert (
                await retired_store.resolve_public_authority_alias(
                    rotated_session_alias,
                    field_name="session_id",
                )
                == session_id
            )
        finally:
            await retired_store.close()

        codec_less = PostgresSessionStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.VALIDATE,
        )
        try:
            with pytest.raises(RuntimeError, match="configure.*alias keyring"):
                await codec_less.ensure_schema()
        finally:
            await codec_less.close()

        reused_key_id = _public_authority_codec(key_byte=19)
        mismatched = PostgresSessionStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.VALIDATE,
            public_authority_alias_codec=reused_key_id,
        )
        try:
            with pytest.raises(RuntimeError, match="different key material"):
                await mismatched.ensure_schema()
        finally:
            await mismatched.close()

    asyncio.run(runner())


def test_postgres_public_authority_alias_registration_fails_closed_without_codec(
    postgres_dsn: str,
) -> None:
    codec = _public_authority_codec()

    async def ops(store) -> None:
        alias = codec.encode("private-session", field_name="session_id")
        with pytest.raises(ValueError, match="store-configured provenance"):
            await store.register_public_authority_alias(
                alias,
                field_name="session_id",
                private_value="private-session",
            )

    _run(postgres_dsn, ops)


def test_postgres_already_open_codec_less_writer_is_fenced_after_key_initialization(
    postgres_dsn: str,
) -> None:
    async def runner() -> None:
        from cayu import PostgresSessionStore
        from cayu.storage.migrations import SchemaMode

        await _truncate(postgres_dsn)
        stale = PostgresSessionStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        keyed = PostgresSessionStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.VALIDATE,
            public_authority_alias_codec=_public_authority_codec(),
        )
        try:
            await stale.ensure_schema()
            await keyed.ensure_schema()
            with pytest.raises(RuntimeError, match="aliases require.*keyring"):
                await stale.create(
                    RunRequest(
                        agent_name="assistant",
                        session_id="must-not-commit",
                        messages=[],
                    ),
                    identity=_identity(),
                )
            assert await keyed.load("must-not-commit") is None
        finally:
            await stale.close()
            await keyed.close()

    asyncio.run(runner())


def test_postgres_public_authority_alias_startup_backfills_every_identity_source(
    postgres_dsn: str,
) -> None:
    async def runner() -> None:
        from cayu import PostgresSessionStore
        from cayu.storage.migrations import SchemaMode

        await _truncate(postgres_dsn)
        session_id = "legacy-session"
        event_interaction_id = "event-only-interaction"
        transcript_interaction_id = "transcript-only-interaction"
        nested_interaction_id = "nested-turn-interaction"
        initial = PostgresSessionStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            await initial.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            await initial.append_events(
                session_id,
                [
                    Event(
                        type=EventType.TURN_COMPLETED,
                        session_id=session_id,
                        interaction_id=event_interaction_id,
                        payload={"interaction_ids": [nested_interaction_id]},
                    )
                ],
            )
            await initial.append_transcript_messages(
                session_id,
                [Message.text("user", "legacy")],
                interaction_id=transcript_interaction_id,
            )
        finally:
            await initial.close()

        codec = _public_authority_codec()
        backfilling = PostgresSessionStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.VALIDATE,
            public_authority_alias_codec=codec,
        )
        try:
            session_alias = codec.encode(session_id, field_name="session_id")
            assert (
                await backfilling.resolve_public_authority_alias(
                    session_alias,
                    field_name="session_id",
                )
                == session_id
            )
            for interaction_id in (
                event_interaction_id,
                transcript_interaction_id,
                nested_interaction_id,
            ):
                alias = codec.encode(
                    interaction_id,
                    field_name="interaction_id",
                    session_id=session_id,
                )
                assert (
                    await backfilling.resolve_public_authority_alias(
                        alias,
                        field_name="interaction_id",
                        scope_session_id=session_id,
                    )
                    == interaction_id
                )
        finally:
            await backfilling.close()

    asyncio.run(runner())


def test_postgres_checkpoint_schema_runtime_conformance(postgres_dsn: str) -> None:
    async def exercise(store) -> None:
        await assert_versionless_pending_continuation_fails_closed_conformance(
            store,
            session_id="sess-postgres-versionless-checkpoint",
        )
        await assert_versionless_noop_transform_stamps_conformance(
            store,
            session_id="sess-postgres-versionless-noop-transform",
        )
        await assert_future_checkpoint_rejection_conformance(
            store,
            session_id="sess-postgres-future-checkpoint",
        )
        await assert_reserved_checkpoint_key_migration_conformance(
            store,
            session_id="sess-postgres-reserved-key-migration",
        )
        await assert_current_checkpoint_publication_upgrade_conformance(
            store,
            session_id_prefix="sess-postgres-current-publication",
        )
        await assert_runtime_publication_rejects_invocation_authority_mutation(
            store,
            session_id_prefix="sess-postgres-invocation-authority-publication",
        )
        await assert_assistant_publication_checkpoint_conformance(
            store,
            session_id="sess-postgres-assistant-publication",
        )

    _run(postgres_dsn, exercise)


def test_postgres_bounded_event_query_rejects_payload_bytes_before_return(
    postgres_dsn: str,
) -> None:
    async def ops(store) -> None:
        await store.create(
            RunRequest(
                agent_name="agent",
                session_id="bounded-event-session",
                messages=[Message.text("user", "test")],
            ),
            identity=_identity(),
        )
        await store.append_events(
            "bounded-event-session",
            [
                Event(
                    id="large-event",
                    type=EventType.SESSION_STARTED,
                    session_id="bounded-event-session",
                    payload={"irrelevant": "x" * 4096},
                )
            ],
        )
        query = EventQuery(session_id="bounded-event-session", limit=1)

        with pytest.raises(EventQueryResultTooLarge):
            await store.query_events_bounded(query, max_bytes=1024)

        records = await store.query_events_bounded(query, max_bytes=8192)
        assert [record.event.id for record in records] == ["large-event"]

    _run(postgres_dsn, ops)


def test_postgres_session_topology_rejects_durable_cycle(postgres_dsn: str) -> None:
    async def ops(store) -> None:
        await assert_session_topology_store_conformance(store)
        import psycopg

        async with await psycopg.AsyncConnection.connect(postgres_dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE cayu_sessions SET parent_session_id = %s WHERE id = %s",
                    ("topology-focus", "topology-root"),
                )
            await conn.commit()

        with pytest.raises(SessionTopologyCycle):
            await store.query_session_topology(
                SessionTopologyQuery(focus_session_id="topology-focus")
            )

    _run(postgres_dsn, ops)


def test_postgres_session_topology_rejects_expanded_branch_cycle(
    postgres_dsn: str,
) -> None:
    async def ops(store) -> None:
        await assert_session_topology_store_conformance(store)
        import psycopg

        async with await psycopg.AsyncConnection.connect(postgres_dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE cayu_sessions SET parent_session_id = id WHERE id = %s",
                    ("topology-root-sibling",),
                )
            await conn.commit()

        with pytest.raises(SessionTopologyCycle):
            await store.query_session_topology(
                SessionTopologyQuery(
                    focus_session_id="topology-focus",
                    expanded_parent_ids=("topology-root-sibling",),
                )
            )

    _run(postgres_dsn, ops)


def test_postgres_session_topology_child_query_uses_composite_index(
    postgres_dsn: str,
) -> None:
    async def ops(store) -> None:
        import psycopg

        await store.create(
            RunRequest(
                agent_name="parent",
                session_id="topology-plan-parent",
                messages=[Message.text("user", "parent")],
            ),
            identity=_identity(),
        )
        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                """
                INSERT INTO cayu_sessions (
                    id, agent_name, provider_name, model, parent_session_id,
                    causal_budget_id, runtime_name, runtime_version,
                    environment_name, status, created_at, updated_at,
                    last_activity_at, run_epoch, event_seq, invocation, metadata
                )
                SELECT
                    'topology-plan-child-' || lpad(value::text, 6, '0'),
                    'child', 'fake', 'fake-model', 'topology-plan-parent',
                    'topology-plan-budget', 'cayu', NULL, NULL, 'pending',
                    TIMESTAMPTZ '2026-01-01T00:00:00Z',
                    TIMESTAMPTZ '2026-01-01T00:00:00Z',
                    TIMESTAMPTZ '2026-01-01T00:00:00Z',
                    0,
                    0,
                    jsonb_build_object(
                        'schema_version', 1,
                        'origin', jsonb_build_object('trust', 'unattributed'),
                        'root_invocation_id',
                        'f055bedc-62cf-4fa4-979a-d0378ca93131',
                        'root_session_id', 'topology-plan-parent',
                        'source', 'subagent'
                    ),
                    '{}'::jsonb
                FROM generate_series(0, 99999) AS value
                """
            )
            await conn.commit()
            await cur.execute("SET LOCAL enable_seqscan = off")
            await cur.execute(
                """
                EXPLAIN (ANALYZE, COSTS OFF, FORMAT JSON)
                WITH requested_branches AS (
                    SELECT parent_session_id, cursor_created_at, cursor_id,
                           branch_order
                    FROM unnest(
                        %s::text[],
                        %s::timestamptz[],
                        %s::text[]
                    ) WITH ORDINALITY AS requested(
                        parent_session_id,
                        cursor_created_at,
                        cursor_id,
                        branch_order
                    )
                )
                SELECT child.*
                FROM requested_branches AS requested
                CROSS JOIN LATERAL (
                    SELECT id, parent_session_id, created_at
                    FROM cayu_sessions
                    WHERE cayu_sessions.parent_session_id =
                          requested.parent_session_id
                      AND (
                          requested.cursor_created_at IS NULL
                          OR cayu_sessions.created_at > requested.cursor_created_at
                          OR (
                              cayu_sessions.created_at = requested.cursor_created_at
                              AND cayu_sessions.id COLLATE "C" >
                                  requested.cursor_id COLLATE "C"
                          )
                      )
                    ORDER BY cayu_sessions.created_at ASC,
                             cayu_sessions.id COLLATE "C" ASC
                    LIMIT %s
                ) AS child
                ORDER BY requested.branch_order ASC,
                         child.created_at ASC,
                         child.id COLLATE "C" ASC
                """,
                (["topology-plan-parent"], [None], [None], 26),
            )
            plan_document = (await cur.fetchone())[0][0]["Plan"]

        def plan_nodes(node):
            yield node
            for child in node.get("Plans", []):
                yield from plan_nodes(child)

        index_nodes = [
            node
            for node in plan_nodes(plan_document)
            if node.get("Index Name") == "idx_cayu_sessions_parent_created_id"
        ]
        assert index_nodes
        assert all(node["Actual Rows"] <= 26 for node in index_nodes)

    _run(postgres_dsn, ops)


def test_postgres_manifest_baseline_validation_redacts_corrupt_jsonb(postgres_dsn: str) -> None:
    async def ops(store):
        import psycopg

        secret = "raw-secret-postgres-baseline"
        history_key = "sha256:" + "1" * 64
        await store.load_mcp_manifest_baselines(())
        async with await psycopg.AsyncConnection.connect(postgres_dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO cayu_mcp_manifest_baselines (
                        history_key, generation, baseline, updated_at
                    )
                    VALUES (%s, 1, jsonb_build_object('manifest_hash', %s::text), %s)
                    """,
                    (history_key, secret, datetime.now(UTC)),
                )
            await conn.commit()

        with pytest.raises(_McpManifestBaselineEvidenceInvalid) as raised:
            await store.load_mcp_manifest_baselines((history_key,))
        assert str(raised.value) == "Stored MCP manifest baseline evidence is invalid."
        assert raised.value.__cause__ is None
        assert secret not in str(raised.value)

    _run(postgres_dsn, ops)


def test_postgres_session_store_queries_checkpoint_backed_pending_actions(postgres_dsn):
    async def ops(store):
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="pending_pg",
                messages=[Message.text("user", "hello")],
            ),
            identity=_identity(),
        )
        approval_payload = {
            "approval_id": "approval_pg",
            **_tool_round_identity_payload(),
            "tool_call_id": "call_pg",
            "tool_name": "deploy",
            "reason": "latest request",
            "arguments": {"service": "api"},
            "agent_name": "assistant",
            "publish_arguments": True,
            "tool_calls": [
                {
                    "tool_call_id": "call_pg",
                    "tool_name": "deploy",
                    "arguments": {"service": "api"},
                    "policy_decision": None,
                    "reason": None,
                    "metadata": {},
                    "active_taint_labels": [],
                }
            ],
        }
        approval_event_payload = dict(approval_payload)
        approval_event_payload.pop("publish_arguments")
        await store.append_events(
            session.id,
            [
                Event(
                    id=f"unrelated_{index}",
                    type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                    session_id=session.id,
                    payload={
                        **_tool_round_identity_payload(),
                        "approval_id": f"unrelated_approval_{index}",
                        "tool_call_id": f"unrelated_call_{index}",
                        "approval": {
                            "approval_id": f"unrelated_approval_{index}",
                            **_tool_round_identity_payload(),
                            "tool_call_id": f"unrelated_call_{index}",
                            "tool_name": "unrelated",
                        },
                    },
                )
                for index in range(500)
            ]
            + [
                Event(
                    id="approval_pg",
                    type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                    session_id=session.id,
                    agent_name="assistant",
                    tool_name="deploy",
                    payload={
                        **_tool_round_identity_payload(),
                        "approval_id": "approval_pg",
                        "tool_call_id": "call_pg",
                        "approval": approval_event_payload,
                    },
                ),
                Event(
                    id="approval_pg_latest",
                    type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                    session_id=session.id,
                    agent_name="assistant",
                    tool_name="deploy",
                    payload={
                        **_tool_round_identity_payload(),
                        "approval_id": "approval_pg",
                        "tool_call_id": "call_pg",
                        "approval": approval_event_payload,
                    },
                ),
            ],
        )
        await store.checkpoint(
            session.id,
            {
                "pending_tool_approval": approval_payload,
            },
        )
        await store.update_status(session.id, SessionStatus.INTERRUPTED)

        result = await store.query_pending_actions(
            PendingActionQuery(kind=PendingActionKind.TOOL_APPROVAL, q="deploy")
        )
        assert result.has_more is False
        assert result.inspected_candidate_count == 1
        assert len(result.actions) == 1
        action = result.actions[0]
        assert action.session.id == session.id
        assert action.event.event.id == "approval_pg_latest"
        # This legacy fixture has no positive secret-scope evidence, so its
        # argument-derived policy reason is not safe for public projection.
        assert action.detail == "Approval required"
        assert action.arguments is None

        user_session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="pending_input_pg",
                messages=[Message.text("user", "hello")],
            ),
            identity=_identity(),
        )
        await store.append_event(
            user_session.id,
            Event(
                id="input_pg",
                type=EventType.SESSION_AWAITING_USER_INPUT,
                session_id=user_session.id,
                payload={
                    **_tool_round_identity_payload(),
                    "input_id": "input_pg",
                    "tool_call_id": "call_input_pg",
                    "question": "Deploy now?",
                    "options": ["yes", "no"],
                },
            ),
        )
        await store.checkpoint(
            user_session.id,
            {
                "pending_user_input": {
                    "input_id": "input_pg",
                    **_tool_round_identity_payload(),
                    "tool_call_id": "call_input_pg",
                    "tool_name": "ask_user",
                    "question": "Deploy now?",
                    "options": ["yes", "no"],
                    "arguments": {},
                    "agent_name": "assistant",
                    "tool_calls": [
                        {
                            "tool_call_id": "call_input_pg",
                            "tool_name": "ask_user",
                            "arguments": {},
                            "policy_decision": None,
                            "reason": None,
                            "metadata": {},
                            "active_taint_labels": [],
                        }
                    ],
                }
            },
        )
        await store.update_status(user_session.id, SessionStatus.INTERRUPTED)

        round_session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="pending_round_pg",
                messages=[Message.text("user", "hello")],
            ),
            identity=_identity(),
        )
        await store.append_events(
            round_session.id,
            [
                Event(
                    id="round_started_pg",
                    type=EventType.TOOL_CALL_STARTED,
                    session_id=round_session.id,
                    tool_name="charge",
                    payload={
                        **_tool_round_identity_payload(),
                        "tool_call_id": "call_round_pg",
                    },
                ),
                Event(
                    id="round_failed_pg",
                    type=EventType.SESSION_FAILED,
                    session_id=round_session.id,
                    payload={
                        **_tool_round_identity_payload(),
                        "manual_recovery_required": True,
                        "tool_call_id": "call_round_pg",
                    },
                ),
            ],
        )
        await store.checkpoint(
            round_session.id,
            {
                "pending_tool_round": {
                    **_tool_round_identity_payload(),
                    "agent_name": "assistant",
                    "tool_calls": [
                        {
                            "tool_call_id": "call_round_pg",
                            "tool_name": "charge",
                            "arguments": {"amount": 42},
                            "policy_decision": None,
                            "reason": None,
                            "metadata": {},
                            "active_taint_labels": [],
                        }
                    ],
                }
            },
        )
        await store.update_status(round_session.id, SessionStatus.FAILED)

        all_actions = await store.query_pending_actions(PendingActionQuery())
        actions_by_session = {pending.session.id: pending for pending in all_actions.actions}
        assert actions_by_session["pending_input_pg"].kind == PendingActionKind.USER_INPUT
        assert actions_by_session["pending_round_pg"].kind == PendingActionKind.MANUAL_RECOVERY
        assert actions_by_session["pending_round_pg"].tool_call_id == "call_round_pg"

        long_approval_id = "approval_" + "x" * 10_000
        long_id_session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="pending_long_identifier_pg",
                messages=[Message.text("user", "hello")],
            ),
            identity=_identity(),
        )
        await store.append_event(
            long_id_session.id,
            Event(
                id="long_identifier_approval_event",
                type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                session_id=long_id_session.id,
                agent_name="assistant",
                tool_name="deploy",
                payload={
                    **_tool_round_identity_payload(),
                    "approval_id": long_approval_id,
                    "tool_call_id": "long_identifier_call",
                    "approval": {
                        "approval_id": long_approval_id,
                        **_tool_round_identity_payload(),
                        "tool_call_id": "long_identifier_call",
                        "tool_name": "deploy",
                        "arguments": {},
                        "agent_name": "assistant",
                        "tool_calls": [
                            {
                                "tool_call_id": "long_identifier_call",
                                "tool_name": "deploy",
                                "arguments": {},
                                "policy_decision": None,
                                "reason": None,
                                "metadata": {},
                                "active_taint_labels": [],
                            }
                        ],
                    },
                },
            ),
        )
        await store.checkpoint(
            long_id_session.id,
            {
                "pending_tool_approval": {
                    "approval_id": long_approval_id,
                    **_tool_round_identity_payload(),
                    "tool_call_id": "long_identifier_call",
                    "tool_name": "deploy",
                    "arguments": {},
                    "agent_name": "assistant",
                    "publish_arguments": True,
                    "tool_calls": [
                        {
                            "tool_call_id": "long_identifier_call",
                            "tool_name": "deploy",
                            "arguments": {},
                            "policy_decision": None,
                            "reason": None,
                            "metadata": {},
                            "active_taint_labels": [],
                        }
                    ],
                }
            },
        )
        await store.update_status(long_id_session.id, SessionStatus.INTERRUPTED)
        long_identifier_result = await store.query_pending_actions(
            PendingActionQuery(session_id=long_id_session.id)
        )
        assert len(long_identifier_result.actions) == 1
        assert long_identifier_result.actions[0].approval_id == long_approval_id

        malformed = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="pending_malformed_pg",
                messages=[Message.text("user", "hello")],
            ),
            identity=_identity(),
        )
        await store.checkpoint(
            malformed.id,
            {"pending_tool_round": {"tool_calls": "not-an-array"}},
        )
        await store.update_status(malformed.id, SessionStatus.INTERRUPTED)
        malformed_result = await store.query_pending_actions(
            PendingActionQuery(session_id=malformed.id)
        )
        assert malformed_result.actions == []
        assert [issue.code for issue in malformed_result.issues] == ["source_invalid"]

        import psycopg

        async with await psycopg.AsyncConnection.connect(postgres_dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE cayu_events SET payload = jsonb_build_object("
                    "'unrelated_result', repeat('x', 1048576)) "
                    "WHERE session_id = %s AND event_id = 'approval_pg_latest'",
                    (session.id,),
                )
            await conn.commit()
        projection_only = await store.query_pending_actions(
            PendingActionQuery(session_id=session.id)
        )
        assert len(projection_only.actions) == 1
        assert projection_only.actions[0].approval_id == "approval_pg"

        concurrent_listing, _ = await asyncio.gather(
            store.query_pending_actions(PendingActionQuery(session_id=session.id)),
            store.checkpoint(session.id, {}),
        )
        assert len(concurrent_listing.actions) in {0, 1}
        after_concurrent_resolution = await store.query_pending_actions(
            PendingActionQuery(session_id=session.id)
        )
        assert after_concurrent_resolution.actions == []

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as conn,
            conn.cursor() as cur,
        ):
            await cur.execute("ANALYZE cayu_events")
            await cur.execute("SET LOCAL enable_seqscan = off")
            await cur.execute(
                """
                EXPLAIN (COSTS OFF)
                SELECT session_id
                FROM cayu_checkpoints
                WHERE pending_action_flags <> 0
                """
            )
            plan = "\n".join(row[0] for row in await cur.fetchall())
            assert "idx_cayu_checkpoints_pending_control_action" in plan
            await cur.execute(
                """
                EXPLAIN (COSTS OFF)
                SELECT sequence
                FROM cayu_events
                WHERE session_id = 'pending_pg'
                  AND (
                      event_type = 'session.resumed'
                      OR event_type = 'session.completed'
                      OR event_type = 'session.failed'
                  )
                ORDER BY sequence DESC
                """
            )
            event_plan = "\n".join(row[0] for row in await cur.fetchall())
            assert "idx_cayu_events_pending_action_barrier" in event_plan
            await cur.execute(
                """
                EXPLAIN (COSTS OFF)
                WITH action_ids(session_id, action_id) AS (
                    VALUES ('pending_pg'::text, 'approval_pg'::text)
                ),
                action_types(event_type) AS (
                    VALUES ('tool.call.approval_requested'::text)
                )
                SELECT matched.sequence
                FROM action_ids
                CROSS JOIN action_types
                CROSS JOIN LATERAL (
                    SELECT event.sequence
                    FROM cayu_events AS event
                    WHERE event.session_id = action_ids.session_id
                      AND event.event_type = action_types.event_type
                      AND event.event_type IN (
                        'tool.call.approval_requested',
                        'session.awaiting_user_input',
                        'session.interrupted',
                        'tool.call.started',
                        'tool.call.completed',
                        'tool.call.failed',
                        'tool.call.blocked',
                        'tool.call.approval_denied'
                      )
                      AND event.pending_action_lookup_key IS NOT NULL
                      AND event.pending_action_lookup_key = encode(sha256(convert_to(
                          action_ids.action_id,
                          'UTF8'
                      )), 'hex')
                    ORDER BY event.sequence DESC
                    LIMIT 1
                ) AS matched
                """
            )
            lookup_plan = "\n".join(row[0] for row in await cur.fetchall())
            assert "idx_cayu_events_pending_action_lookup" in lookup_plan
            await cur.execute(
                """
                EXPLAIN (COSTS OFF)
                WITH action_ids(session_id, action_id) AS (
                    VALUES ('pending_pg'::text, 'call_pg'::text)
                )
                SELECT event.sequence
                FROM action_ids
                JOIN cayu_events AS event
                  ON event.session_id = action_ids.session_id
                 AND event.pending_action_lookup_key = encode(sha256(convert_to(
                     action_ids.action_id,
                     'UTF8'
                 )), 'hex')
                WHERE event.event_type IN (
                    'tool.call.approval_requested',
                    'session.awaiting_user_input',
                    'session.interrupted',
                    'tool.call.started',
                    'tool.call.completed',
                    'tool.call.failed',
                    'tool.call.blocked',
                    'tool.call.approval_denied'
                )
                  AND event.event_type IN (
                    'tool.call.started',
                    'tool.call.completed',
                    'tool.call.failed',
                    'tool.call.blocked',
                    'tool.call.approval_denied'
                )
                  AND event.pending_action_lookup_key IS NOT NULL
                """
            )
            ledger_plan = "\n".join(row[0] for row in await cur.fetchall())
            assert "idx_cayu_events_pending_action_lookup" in ledger_plan

    _run(postgres_dsn, ops)


def test_postgres_session_store_persists_sessions_events_and_checkpoints(postgres_dsn):
    async def ops(store):
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_pg",
                environment_name="local-dev",
                messages=[Message.text("user", "hi")],
                metadata={"project_id": 123},
            ),
            identity=SessionIdentity(
                provider_name="anthropic",
                model="claude-test",
                runtime_name="cayu",
                runtime_version="test-version",
            ),
        )
        assert session.status == SessionStatus.PENDING
        assert session.provider_name == "anthropic"
        assert session.model == "claude-test"
        assert session.runtime_version == "test-version"

        await store.update_status("sess_pg", SessionStatus.RUNNING)
        await store.append_event(
            "sess_pg",
            Event(
                type=EventType.SESSION_STARTED,
                session_id="sess_pg",
                agent_name="assistant",
                environment_name="local-dev",
                payload={"step": 1},
            ),
        )
        await store.append_event(
            "sess_pg",
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id="sess_pg",
                agent_name="assistant",
                environment_name="local-dev",
                payload={"finish_reason": "stop"},
            ),
        )
        await store.append_transcript_messages(
            "sess_pg",
            [Message.text("user", "hi"), Message.text("assistant", "hello")],
        )
        await store.checkpoint(
            "sess_pg",
            {
                "messages": [{"role": "user", "content": "hi"}],
                "step": 1,
                "pending_interruption_cascade": {
                    "attempt_id": "attempt_pg",
                    "interrupt_payload": {
                        "reason": "operator stop",
                        "metadata": {"large": "x" * 100_000},
                    },
                    "generation": 0,
                    "failure_recorded": False,
                    "created_at": "2026-07-11T00:00:00+00:00",
                },
            },
        )

        loaded = await store.load("sess_pg")
        events = await store.load_events("sess_pg")
        transcript = await store.load_transcript("sess_pg")
        checkpoint = await store.load_checkpoint("sess_pg")
        state = await store.load_state("sess_pg")
        marker = await store.load_interruption_cascade_marker("sess_pg")

        assert loaded is not None
        assert loaded.agent_name == "assistant"
        assert loaded.environment_name == "local-dev"
        assert loaded.status == SessionStatus.RUNNING
        assert loaded.metadata == {"project_id": 123}
        assert [event.type for event in events] == [
            EventType.SESSION_STARTED,
            EventType.MODEL_COMPLETED,
        ]
        assert [event.payload for event in events] == [
            {"step": 1},
            {"finish_reason": "stop"},
        ]
        assert [message.role for message in transcript] == ["user", "assistant"]
        assert [message.content[0].text for message in transcript] == ["hi", "hello"]
        assert checkpoint == {
            "messages": [{"role": "user", "content": "hi"}],
            "step": 1,
            "pending_interruption_cascade": {
                "attempt_id": "attempt_pg",
                "interrupt_payload": {
                    "reason": "operator stop",
                    "metadata": {"large": "x" * 100_000},
                },
                "generation": 0,
                "failure_recorded": False,
                "created_at": "2026-07-11T00:00:00+00:00",
            },
        }
        assert state is not None
        assert state.id == "sess_pg"
        assert state.status == SessionStatus.RUNNING
        assert not hasattr(state, "metadata")
        assert marker == {
            "attempt_id": "attempt_pg",
            "interrupt_payload": {},
            "generation": 0,
            "failure_recorded": False,
            "created_at": "2026-07-11T00:00:00+00:00",
        }
        assert len(repr(marker)) < 500
        assert await store.load_interruption_cascade_marker("missing") is None
        assert await store.load_state("missing") is None

        await store.checkpoint("sess_pg", {"pending_interruption_cascade": None})
        assert await store.load_interruption_cascade_marker("sess_pg") is None

    _run(postgres_dsn, ops)


def test_postgres_session_store_atomically_appends_transcript_and_transforms_checkpoint(
    postgres_dsn,
):
    async def ops(store):
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_atomic",
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )
        await store.checkpoint("sess_atomic", {"pending_tool_approval": {"approval_id": "a1"}})
        await store.append_transcript_messages_and_transform_checkpoint(
            "sess_atomic",
            [Message.text("assistant", "done")],
            lambda _session, _checkpoint: {"closed": True},
        )
        transcript = await store.load_transcript("sess_atomic")
        checkpoint = await store.load_checkpoint("sess_atomic")
        assert [message.role for message in transcript] == ["assistant"]
        assert transcript[0].content[0].text == "done"
        assert checkpoint == {"closed": True}

    _run(postgres_dsn, ops)


def test_postgres_session_store_atomically_transitions_status_and_checkpoint(postgres_dsn):
    async def ops(store):
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_status_ck",
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )
        session = await store.transition_status_and_checkpoint(
            "sess_status_ck",
            from_statuses={SessionStatus.PENDING},
            to_status=SessionStatus.INTERRUPTING,
            checkpoint_transform=lambda _s, ck: {
                **({} if ck is None else ck),
                "pending_session_interrupt": {"reason": "operator stop"},
            },
        )
        checkpoint = await store.load_checkpoint("sess_status_ck")
        assert session.status == SessionStatus.INTERRUPTING
        assert checkpoint == {"pending_session_interrupt": {"reason": "operator stop"}}

    _run(postgres_dsn, ops)


def test_postgres_session_store_rejects_stale_atomic_status_checkpoint_transition(postgres_dsn):
    async def ops(store):
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_stale",
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )
        # Move it out of PENDING so the guarded transition can no longer match.
        await store.update_status("sess_stale", SessionStatus.RUNNING)

        with pytest.raises(ValueError, match="Session status transition not allowed"):
            await store.transition_status_and_checkpoint(
                "sess_stale",
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.INTERRUPTING,
                checkpoint_transform=lambda _s, ck: {
                    **({} if ck is None else ck),
                    "pending_session_interrupt": {"reason": "operator stop"},
                },
            )

        session = await store.load("sess_stale")
        checkpoint = await store.load_checkpoint("sess_stale")
        assert session is not None
        assert session.status == SessionStatus.RUNNING
        # Failed transition must NOT have written a checkpoint (atomic rollback).
        assert checkpoint is None

    _run(postgres_dsn, ops)


def test_postgres_session_store_fences_stale_run_writes(postgres_dsn):
    async def ops(store):
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_pg_fenced",
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )
        await store.update_status("sess_pg_fenced", SessionStatus.COMPLETED)
        running = asyncio.Event()
        release_stale_writer = asyncio.Event()

        async def stale_writer() -> None:
            claimed = await store.transition_status(
                "sess_pg_fenced",
                from_statuses={SessionStatus.COMPLETED},
                to_status=SessionStatus.RUNNING,
            )
            assert claimed.run_epoch == 1
            running.set()
            await release_stale_writer.wait()
            with pytest.raises(SessionRunFenced, match="no longer owns"):
                await store.append_event(
                    "sess_pg_fenced",
                    Event(
                        id="event_from_stale_pg_run",
                        type=EventType.MODEL_COMPLETED,
                        session_id="sess_pg_fenced",
                        agent_name="assistant",
                    ),
                )
            with pytest.raises(SessionRunFenced, match="no longer owns"):
                await store.append_transcript_messages(
                    "sess_pg_fenced",
                    [Message.text("assistant", "late answer")],
                )
            with pytest.raises(SessionRunFenced, match="no longer owns"):
                await store.checkpoint("sess_pg_fenced", {"late": True})
            with pytest.raises(SessionRunFenced, match="no longer owns"):
                await store.transition_status(
                    "sess_pg_fenced",
                    from_statuses={SessionStatus.RUNNING},
                    to_status=SessionStatus.COMPLETED,
                )

        task = asyncio.create_task(stale_writer())
        await running.wait()
        fenced = await store.fence_stalled_run(
            "sess_pg_fenced",
            statuses={SessionStatus.RUNNING},
            inactive_before=datetime.now(UTC) + timedelta(seconds=1),
        )
        assert fenced is not None
        assert fenced.run_epoch == 2
        recovered = await store.transition_status(
            "sess_pg_fenced",
            from_statuses={SessionStatus.RUNNING},
            to_status=SessionStatus.INTERRUPTED,
        )
        assert recovered.status == SessionStatus.INTERRUPTED
        await store.release_run_fence("sess_pg_fenced")

        release_stale_writer.set()
        await task
        assert await store.load_events("sess_pg_fenced") == []

    _run(postgres_dsn, ops)


def test_postgres_session_store_transforms_current_checkpoint_during_fork(postgres_dsn):
    async def ops(store):
        source = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_fork_ck_source",
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )
        await store.update_status(source.id, SessionStatus.COMPLETED)
        await store.checkpoint(source.id, {"version": 2})
        await store.append_transcript_messages(
            source.id,
            [Message.text("user", "first request"), Message.text("assistant", "first answer")],
        )

        fork = await store.create_fork(
            source_session_id=source.id,
            fork=Session(
                id="sess_fork_ck_child",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
                parent_session_id=source.id,
                invocation=fork_session_invocation(source),
                status=SessionStatus.COMPLETED,
            ),
            source_statuses={SessionStatus.COMPLETED},
            expected_source_run_epoch=source.run_epoch,
            transcript_cursor=None,
            checkpoint_transform=lambda _s, ck: {"copied_version": ck["version"] if ck else None},
        )

        assert fork.parent_session_id == source.id
        assert fork.status == SessionStatus.COMPLETED
        assert await store.load_checkpoint("sess_fork_ck_child") == {"copied_version": 2}
        transcript = await store.load_transcript("sess_fork_ck_child")
        assert [m.content[0].text for m in transcript] == ["first request", "first answer"]
        children = (await store.list_sessions(SessionQuery(parent_session_id=source.id))).sessions
        assert [s.id for s in children] == ["sess_fork_ck_child"]

    _run(postgres_dsn, ops)


def test_postgres_session_store_persists_run_request_parent_session_id(postgres_dsn):
    async def ops(store):
        parent = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_pg_run_parent",
                invocation_origin=InvocationOriginClaim(
                    subject="application-user",
                    tenant="customer-a",
                ),
                messages=[Message.text("user", "parent")],
            ),
            identity=_identity(),
        )
        child = await store.create(
            RunRequest(
                agent_name="reviewer",
                session_id="sess_pg_run_child",
                parent_session_id="sess_pg_run_parent",
                causal_budget_id="job_pg_run_parent",
                messages=[Message.text("user", "child")],
            ),
            identity=_identity(),
        )

        assert child.parent_session_id == "sess_pg_run_parent"
        assert parent.invocation.origin.trust is InvocationOriginTrust.HOST_ASSERTED
        assert child.invocation.origin == parent.invocation.origin
        assert child.invocation.root_session_id == parent.id
        assert child.invocation.source is SessionExecutionSource.SDK_RUN
        loaded = await store.load("sess_pg_run_child")
        assert loaded is not None
        assert loaded.parent_session_id == "sess_pg_run_parent"
        assert loaded.causal_budget_id == "job_pg_run_parent"
        assert loaded.invocation == child.invocation
        children = (
            await store.list_sessions(SessionQuery(parent_session_id="sess_pg_run_parent"))
        ).sessions
        assert [session.id for session in children] == ["sess_pg_run_child"]

    _run(postgres_dsn, ops)


def test_postgres_child_creation_locks_parent_against_delete_and_reuse(
    postgres_dsn,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def ops(store):
        parent = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_pg_locked_parent",
                messages=[],
                invocation_origin=InvocationOriginClaim(subject="original-user"),
            ),
            identity=_identity(),
        )
        parent_locked = asyncio.Event()
        release_parent = asyncio.Event()
        original_load = store._load_for_key_share

        async def load_then_pause(cur, session_id: str):
            loaded = await original_load(cur, session_id)
            parent_locked.set()
            await release_parent.wait()
            return loaded

        monkeypatch.setattr(store, "_load_for_key_share", load_then_pause)
        child_task = asyncio.create_task(
            store.create(
                RunRequest(
                    agent_name="reviewer",
                    session_id="sess_pg_locked_child",
                    parent_session_id=parent.id,
                    messages=[],
                ),
                identity=_identity(),
            )
        )
        await asyncio.wait_for(parent_locked.wait(), timeout=2.0)
        delete_task = asyncio.create_task(store.delete_session(parent.id))
        await asyncio.sleep(0.1)
        assert delete_task.done() is False

        release_parent.set()
        child = await asyncio.wait_for(child_task, timeout=2.0)
        await asyncio.wait_for(delete_task, timeout=2.0)
        replacement = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=parent.id,
                messages=[],
                invocation_origin=InvocationOriginClaim(subject="replacement-user"),
            ),
            identity=_identity(),
        )
        loaded_child = await store.load(child.id)

        assert loaded_child is not None
        assert loaded_child.parent_session_id is None
        assert loaded_child.invocation == child.invocation
        assert loaded_child.invocation.root_invocation_id == parent.invocation.root_invocation_id
        assert (
            loaded_child.invocation.root_invocation_id != replacement.invocation.root_invocation_id
        )

    _run(postgres_dsn, ops)


def test_postgres_session_store_rejects_missing_parent_session_id(postgres_dsn):
    async def ops(store):
        with pytest.raises(ValueError, match="Parent session not found"):
            await store.create(
                RunRequest(
                    agent_name="reviewer",
                    session_id="sess_pg_missing_parent_child",
                    parent_session_id="sess_pg_missing_parent",
                    messages=[Message.text("user", "child")],
                ),
                identity=_identity(),
            )
        assert await store.load("sess_pg_missing_parent_child") is None

    _run(postgres_dsn, ops)


def test_postgres_session_store_rejects_self_parent_session_id(postgres_dsn):
    async def ops(store):
        with pytest.raises(ValueError, match="own parent"):
            await store.create(
                RunRequest(
                    agent_name="reviewer",
                    session_id="sess_pg_self_parent",
                    parent_session_id="sess_pg_self_parent",
                    messages=[Message.text("user", "child")],
                ),
                identity=_identity(),
            )
        assert await store.load("sess_pg_self_parent") is None

    _run(postgres_dsn, ops)


def test_postgres_session_store_fork_honors_transcript_cursor(postgres_dsn):
    async def ops(store):
        source = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_fork_cursor_source",
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )
        await store.update_status(source.id, SessionStatus.COMPLETED)
        await store.append_transcript_messages(
            source.id,
            [
                Message.text("user", "m1"),
                Message.text("assistant", "m2"),
                Message.text("user", "m3"),
            ],
        )
        await store.create_fork(
            source_session_id=source.id,
            fork=Session(
                id="sess_fork_cursor_child",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
                parent_session_id=source.id,
                invocation=fork_session_invocation(source),
                status=SessionStatus.COMPLETED,
            ),
            source_statuses={SessionStatus.COMPLETED},
            expected_source_run_epoch=source.run_epoch,
            transcript_cursor=2,
            checkpoint_transform=None,
        )
        transcript = await store.load_transcript("sess_fork_cursor_child")
        assert [m.content[0].text for m in transcript] == ["m1", "m2"]

        with pytest.raises(ValueError, match="transcript_cursor is greater"):
            await store.create_fork(
                source_session_id=source.id,
                fork=Session(
                    id="sess_fork_cursor_overflow",
                    agent_name="assistant",
                    provider_name="fake",
                    model="fake-model",
                    parent_session_id=source.id,
                    invocation=fork_session_invocation(source),
                    status=SessionStatus.COMPLETED,
                ),
                source_statuses={SessionStatus.COMPLETED},
                expected_source_run_epoch=source.run_epoch,
                transcript_cursor=99,
                checkpoint_transform=None,
            )

    _run(postgres_dsn, ops)


def test_postgres_session_store_rejects_fork_status_and_provider_mismatch(postgres_dsn):
    async def ops(store):
        source = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_fork_mismatch_source",
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )
        await store.update_status(source.id, SessionStatus.COMPLETED)

        with pytest.raises(ValueError, match="Fork status must match"):
            await store.create_fork(
                source_session_id=source.id,
                fork=Session(
                    id="sess_fork_status_child",
                    agent_name="assistant",
                    provider_name="fake",
                    model="fake-model",
                    parent_session_id=source.id,
                    invocation=fork_session_invocation(source),
                    status=SessionStatus.RUNNING,
                ),
                source_statuses={SessionStatus.COMPLETED},
                expected_source_run_epoch=source.run_epoch,
                transcript_cursor=None,
                checkpoint_transform=None,
            )

        with pytest.raises(ValueError, match="Fork provider_name must match"):
            await store.create_fork(
                source_session_id=source.id,
                fork=Session(
                    id="sess_fork_provider_child",
                    agent_name="assistant",
                    provider_name="other",
                    model="fake-model",
                    parent_session_id=source.id,
                    invocation=fork_session_invocation(source),
                    status=SessionStatus.COMPLETED,
                ),
                source_statuses={SessionStatus.COMPLETED},
                expected_source_run_epoch=source.run_epoch,
                transcript_cursor=None,
                checkpoint_transform=None,
            )

    _run(postgres_dsn, ops)


def test_postgres_session_store_rejects_duplicate_sessions_and_mismatched_events(postgres_dsn):
    async def ops(store):
        request = RunRequest(
            agent_name="assistant",
            session_id="sess_duplicate",
            messages=[Message.text("user", "hi")],
        )
        await store.create(request, identity=_identity())

        with pytest.raises(ValueError, match="Session already exists"):
            await store.create(request, identity=_identity())

        with pytest.raises(ValueError, match="Event session_id"):
            await store.append_event(
                "sess_duplicate",
                Event(type=EventType.SESSION_STARTED, session_id="other_session"),
            )

        event = Event(id="event_dup", type=EventType.SESSION_STARTED, session_id="sess_duplicate")
        await store.append_event("sess_duplicate", event)
        with pytest.raises(ValueError, match="Event already exists"):
            await store.append_event("sess_duplicate", event)

        with pytest.raises(KeyError, match="Session not found"):
            await store.load_events("missing_session")

    _run(postgres_dsn, ops)


def test_postgres_session_store_lists_sessions_with_filters_and_pagination(postgres_dsn):
    async def ops(store):
        await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_builder_1",
                environment_name="local",
                messages=[Message.text("user", "build")],
            ),
            identity=_identity(),
        )
        await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_builder_2",
                environment_name="hosted",
                messages=[Message.text("user", "build again")],
            ),
            identity=_identity(),
        )
        await store.create(
            RunRequest(
                agent_name="reviewer",
                session_id="sess_reviewer",
                environment_name="hosted",
                messages=[Message.text("user", "review")],
            ),
            identity=_identity(),
        )
        await store.create(
            RunRequest(
                agent_name="operator",
                session_id="sess_openai_operator",
                environment_name="sandbox",
                labels={"marker": "literal%token", "workflow": "pr-review"},
                messages=[Message.text("user", "review PR")],
            ),
            identity=SessionIdentity(provider_name="openai", model="gpt-5.5"),
        )
        await store.update_status("sess_builder_1", SessionStatus.RUNNING)
        await store.update_status("sess_builder_2", SessionStatus.COMPLETED)

        builder_sessions = (
            await store.list_sessions(
                SessionQuery(agent_name="builder", order_by=SessionOrder.CREATED_AT_ASC)
            )
        ).sessions
        hosted_sessions = (
            await store.list_sessions(
                SessionQuery(environment_name="hosted", order_by=SessionOrder.CREATED_AT_ASC)
            )
        ).sessions
        completed_sessions = (
            await store.list_sessions(SessionQuery(status=SessionStatus.COMPLETED))
        ).sessions
        paged_sessions = (
            await store.list_sessions(
                SessionQuery(limit=1, offset=1, order_by=SessionOrder.CREATED_AT_ASC)
            )
        ).sessions
        openai_sessions = (
            await store.list_sessions(
                SessionQuery(provider_name="openai", order_by=SessionOrder.CREATED_AT_ASC)
            )
        ).sessions
        model_sessions = (
            await store.list_sessions(
                SessionQuery(model="gpt-5.5", order_by=SessionOrder.CREATED_AT_ASC)
            )
        ).sessions
        query_by_agent_sessions = (
            await store.list_sessions(SessionQuery(q="OPER", order_by=SessionOrder.CREATED_AT_ASC))
        ).sessions
        query_by_model_sessions = (
            await store.list_sessions(SessionQuery(q="gpt-5", order_by=SessionOrder.CREATED_AT_ASC))
        ).sessions
        query_by_label_sessions = (
            await store.list_sessions(
                SessionQuery(q="pr-review", order_by=SessionOrder.CREATED_AT_ASC)
            )
        ).sessions
        query_by_literal_percent_sessions = (
            await store.list_sessions(SessionQuery(q="%", order_by=SessionOrder.CREATED_AT_ASC))
        ).sessions

        assert [s.id for s in builder_sessions] == ["sess_builder_1", "sess_builder_2"]
        assert [s.id for s in hosted_sessions] == ["sess_builder_2", "sess_reviewer"]
        assert [s.id for s in completed_sessions] == ["sess_builder_2"]
        assert [s.id for s in paged_sessions] == ["sess_builder_2"]
        assert [s.id for s in openai_sessions] == ["sess_openai_operator"]
        assert [s.id for s in model_sessions] == ["sess_openai_operator"]
        assert [s.id for s in query_by_agent_sessions] == ["sess_openai_operator"]
        assert [s.id for s in query_by_model_sessions] == ["sess_openai_operator"]
        assert [s.id for s in query_by_label_sessions] == ["sess_openai_operator"]
        assert [s.id for s in query_by_literal_percent_sessions] == ["sess_openai_operator"]

    _run(postgres_dsn, ops)


def test_postgres_session_store_filters_sessions_by_last_activity(postgres_dsn):
    async def ops(store):
        for session_id in ("sess_pg_activity_stale", "sess_pg_activity_recent"):
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", session_id)],
                ),
                identity=_identity(),
            )
        inactive_before = datetime.now(UTC)
        await asyncio.sleep(0.001)
        await store.append_event(
            "sess_pg_activity_recent",
            Event(
                id="event_pg_recent_activity",
                type=EventType.SESSION_STARTED,
                session_id="sess_pg_activity_recent",
            ),
        )

        sessions = (
            await store.list_sessions(
                SessionQuery(
                    last_activity_before=inactive_before,
                    order_by=SessionOrder.CREATED_AT_ASC,
                )
            )
        ).sessions

        assert [session.id for session in sessions] == ["sess_pg_activity_stale"]

    _run(postgres_dsn, ops)


def test_postgres_session_store_preserves_and_filters_session_labels(postgres_dsn):
    async def ops(store):
        created = await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_pg_labels_invoice",
                labels={
                    "owner": "org_123",
                    "project": "ap_q2",
                    "workflow": "invoice-ingestion",
                },
                messages=[Message.text("user", "ingest invoice")],
            ),
            identity=_identity(),
        )
        await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_pg_labels_research",
                labels={"owner": "org_123", "project": "research"},
                messages=[Message.text("user", "research")],
            ),
            identity=_identity(),
        )
        await store.create(
            RunRequest(
                agent_name="reviewer",
                session_id="sess_pg_labels_other_owner",
                labels={"owner": "org_999", "project": "ap_q2"},
                messages=[Message.text("user", "review")],
            ),
            identity=_identity(),
        )

        loaded = await store.load(created.id)
        owner_sessions = (
            await store.list_sessions(
                SessionQuery(labels={"owner": "org_123"}, order_by=SessionOrder.CREATED_AT_ASC)
            )
        ).sessions
        exact_sessions = (
            await store.list_sessions(
                SessionQuery(
                    labels={"owner": "org_123", "project": "ap_q2"},
                    order_by=SessionOrder.CREATED_AT_ASC,
                )
            )
        ).sessions
        missing_sessions = (
            await store.list_sessions(SessionQuery(labels={"owner": "missing"}))
        ).sessions
        exists_sessions = (
            await store.list_sessions(
                SessionQuery(
                    label_selectors=[{"key": "workflow", "operator": "exists"}],
                    order_by=SessionOrder.CREATED_AT_ASC,
                )
            )
        ).sessions
        in_sessions = (
            await store.list_sessions(
                SessionQuery(
                    label_selectors=[
                        {"key": "project", "operator": "in", "values": ["ap_q2", "research"]}
                    ],
                    order_by=SessionOrder.CREATED_AT_ASC,
                )
            )
        ).sessions
        not_in_sessions = (
            await store.list_sessions(
                SessionQuery(
                    labels={"owner": "org_123"},
                    label_selectors=[
                        {"key": "project", "operator": "not_in", "values": ["research"]}
                    ],
                    order_by=SessionOrder.CREATED_AT_ASC,
                )
            )
        ).sessions
        not_exists_sessions = (
            await store.list_sessions(
                SessionQuery(
                    label_selectors=[{"key": "owner", "operator": "not_exists"}],
                    order_by=SessionOrder.CREATED_AT_ASC,
                )
            )
        ).sessions

        assert loaded is not None
        assert loaded.labels == {
            "owner": "org_123",
            "project": "ap_q2",
            "workflow": "invoice-ingestion",
        }
        assert [session.id for session in owner_sessions] == [
            "sess_pg_labels_invoice",
            "sess_pg_labels_research",
        ]
        assert [session.id for session in exact_sessions] == ["sess_pg_labels_invoice"]
        assert missing_sessions == []
        assert [session.id for session in exists_sessions] == ["sess_pg_labels_invoice"]
        assert [session.id for session in in_sessions] == [
            "sess_pg_labels_invoice",
            "sess_pg_labels_research",
            "sess_pg_labels_other_owner",
        ]
        assert [session.id for session in not_in_sessions] == ["sess_pg_labels_invoice"]
        assert [session.id for session in not_exists_sessions] == []

    _run(postgres_dsn, ops)


def test_postgres_session_store_query_events_with_filters_cursors_and_batching(postgres_dsn):
    async def ops(store):
        await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_builder",
                environment_name="local",
                messages=[Message.text("user", "build")],
            ),
            identity=_identity(),
        )
        await store.create(
            RunRequest(
                agent_name="reviewer",
                session_id="sess_reviewer",
                environment_name="hosted",
                messages=[Message.text("user", "review")],
            ),
            identity=_identity(),
        )
        await store.append_events(
            "sess_builder",
            [
                Event(
                    id="event_1",
                    type=EventType.SESSION_STARTED,
                    session_id="sess_builder",
                    agent_name="builder",
                    environment_name="local",
                    timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
                ),
                Event(
                    id="event_2",
                    type=EventType.TOOL_CALL_COMPLETED,
                    session_id="sess_builder",
                    agent_name="builder",
                    environment_name="local",
                    tool_name="read_file",
                    timestamp=datetime(2026, 1, 1, 12, 5, tzinfo=UTC),
                ),
                Event(
                    id="event_3",
                    type=EventType.MODEL_COMPLETED,
                    session_id="sess_builder",
                    agent_name="builder",
                    environment_name="local",
                    timestamp=datetime(2026, 1, 1, 12, 10, tzinfo=UTC),
                    payload={"finish_reason": "stop"},
                ),
            ],
        )
        await store.append_event(
            "sess_reviewer",
            Event(
                id="event_4",
                type=EventType.SESSION_STARTED,
                session_id="sess_reviewer",
                agent_name="reviewer",
                environment_name="hosted",
                timestamp=datetime(2026, 1, 1, 12, 15, tzinfo=UTC),
            ),
        )

        all_records = await store.query_events(EventQuery(limit=10))
        desc_records = await store.query_events(
            EventQuery(order_by=EventOrder.SEQUENCE_DESC, limit=2)
        )
        since_records = await store.query_events(
            EventQuery(since=datetime(2026, 1, 1, 12, 5, tzinfo=UTC), limit=10)
        )
        until_records = await store.query_events(
            EventQuery(until=datetime(2026, 1, 1, 12, 10, tzinfo=UTC), limit=10)
        )
        window_records = await store.query_events(
            EventQuery(
                since=datetime(2026, 1, 1, 12, 5, tzinfo=UTC),
                until=datetime(2026, 1, 1, 12, 15, tzinfo=UTC),
                limit=10,
            )
        )
        builder_records = await store.query_events(EventQuery(session_id="sess_builder"))
        session_ids_records = await store.query_events(
            EventQuery(session_ids=("sess_reviewer", "sess_builder"), limit=10)
        )
        read_file_records = await store.query_events(EventQuery(tool_name="read_file"))
        event_id_records = await store.query_events(
            EventQuery(session_id="sess_builder", event_id="event_2")
        )
        started_records = await store.query_events(EventQuery(event_type=EventType.SESSION_STARTED))
        excluded_records = await store.query_events(
            EventQuery(
                exclude_event_types=(EventType.MODEL_COMPLETED,),
                order_by=EventOrder.SEQUENCE_DESC,
                limit=2,
            )
        )
        multi_type_records = await store.query_events(
            EventQuery(
                event_types=(EventType.SESSION_STARTED, EventType.TOOL_CALL_COMPLETED),
                limit=10,
            )
        )
        cursor_records = await store.query_events(
            EventQuery(after_sequence=all_records[1].sequence, limit=10)
        )
        backward_records = await store.query_events(
            EventQuery(
                session_id="sess_builder",
                before_sequence=all_records[3].sequence,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=2,
            )
        )
        bounded_records = await store.query_events(
            EventQuery(
                session_id="sess_builder",
                after_sequence=all_records[0].sequence,
                before_sequence=all_records[3].sequence,
                limit=10,
            )
        )

        assert [r.sequence for r in all_records] == [1, 2, 3, 4]
        assert [r.sequence for r in desc_records] == [4, 3]
        assert [r.event.id for r in since_records] == ["event_2", "event_3", "event_4"]
        assert [r.event.id for r in until_records] == ["event_1", "event_2"]
        assert [r.event.id for r in window_records] == ["event_2", "event_3"]
        assert [r.event.id for r in builder_records] == ["event_1", "event_2", "event_3"]
        assert [r.event.id for r in session_ids_records] == [
            "event_1",
            "event_2",
            "event_3",
            "event_4",
        ]
        assert [r.event.id for r in read_file_records] == ["event_2"]
        assert [r.event.id for r in event_id_records] == ["event_2"]
        assert [r.event.id for r in started_records] == ["event_1", "event_4"]
        assert [r.event.id for r in excluded_records] == ["event_4", "event_2"]
        assert [r.event.id for r in multi_type_records] == ["event_1", "event_2", "event_4"]
        assert [r.event.id for r in cursor_records] == ["event_3", "event_4"]
        assert [r.event.id for r in backward_records] == ["event_3", "event_2"]
        assert [r.event.id for r in bounded_records] == ["event_2", "event_3"]

        # A batch containing a duplicate event id must roll back atomically.
        with pytest.raises(ValueError, match="Event already exists"):
            await store.append_events(
                "sess_builder",
                [
                    Event(
                        id="event_new_rolled_back",
                        type=EventType.MODEL_STARTED,
                        session_id="sess_builder",
                    ),
                    Event(
                        id="event_2",
                        type=EventType.MODEL_COMPLETED,
                        session_id="sess_builder",
                    ),
                ],
            )

        records_after = await store.query_events(EventQuery(limit=10))
        assert [r.event.id for r in records_after] == [
            "event_1",
            "event_2",
            "event_3",
            "event_4",
        ]

    _run(postgres_dsn, ops)


def test_postgres_session_store_append_and_load_transcript_messages(postgres_dsn):
    async def ops(store):
        await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_transcript",
                messages=[Message.text("user", "build")],
            ),
            identity=_identity(),
        )
        user_message = Message.text("user", "build")
        assistant_message = Message.tool_call(
            tool_call_id="call_1",
            tool_name="read_file",
            arguments={"path": "README.md"},
        )
        tool_message = Message.tool_result(
            tool_call_id="call_1",
            tool_name="read_file",
            content="contents",
            structured={"bytes": 8},
        )

        await store.append_transcript_messages("sess_transcript", [user_message, assistant_message])
        # Messages are frozen: stored transcripts cannot be corrupted through
        # references the caller still holds.
        with pytest.raises(ValidationError):
            user_message.content[0].text = "mutated"  # type: ignore[misc]
        await store.append_transcript_messages("sess_transcript", [tool_message])

        transcript = await store.load_transcript("sess_transcript")
        assert [m.role for m in transcript] == ["user", "assistant", "tool"]
        assert transcript[0].content[0].text == "build"
        assert transcript[1].content[0].tool_name == "read_file"
        assert transcript[2].content[0].structured == {"bytes": 8}

        with pytest.raises(KeyError, match="Session not found"):
            await store.append_transcript_messages("missing_session", [Message.text("user", "hi")])
        with pytest.raises(KeyError, match="Session not found"):
            await store.load_transcript("missing_session")

    _run(postgres_dsn, ops)


def test_postgres_session_store_transition_status_atomically(postgres_dsn):
    async def ops(store):
        await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_transition",
                messages=[Message.text("user", "build")],
            ),
            identity=_identity(),
        )
        await store.update_status("sess_transition", SessionStatus.COMPLETED)

        transitioned = await store.transition_status(
            "sess_transition",
            from_statuses={SessionStatus.COMPLETED},
            to_status=SessionStatus.RUNNING,
        )
        assert transitioned.status == SessionStatus.RUNNING

        with pytest.raises(ValueError, match="transition not allowed"):
            await store.transition_status(
                "sess_transition",
                from_statuses={SessionStatus.COMPLETED},
                to_status=SessionStatus.RUNNING,
            )

        loaded = await store.load("sess_transition")
        assert loaded is not None
        assert loaded.status == SessionStatus.RUNNING

    _run(postgres_dsn, ops)


def test_postgres_session_store_concurrent_appends_keep_contiguous_order(postgres_dsn):
    """Concurrent append batches must produce a contiguous per-session order."""

    async def ops(store):
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_concurrent",
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )

        async def append_batch(prefix: str) -> None:
            await store.append_events(
                "sess_concurrent",
                [
                    Event(
                        id=f"{prefix}_{i}",
                        type=EventType.MODEL_TEXT_DELTA,
                        session_id="sess_concurrent",
                        payload={"i": i},
                    )
                    for i in range(10)
                ],
            )

        await asyncio.gather(*(append_batch(f"w{w}") for w in range(5)))

        events = await store.load_events("sess_concurrent")
        assert len(events) == 50
        assert len({e.id for e in events}) == 50

        # The global query cursor (sequence) must be a contiguous 1..50 range.
        records = await store.query_events(EventQuery(session_id="sess_concurrent", limit=1000))
        assert [r.sequence for r in records] == list(range(1, 51))

        # Per-session order is dense and unique under concurrency.
        import psycopg

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                "SELECT session_order FROM cayu_events WHERE session_id = %s "
                "ORDER BY session_order ASC",
                ("sess_concurrent",),
            )
            orders = [row[0] for row in await cur.fetchall()]
        assert orders == list(range(1, 51))

    _run(postgres_dsn, ops)


def test_postgres_session_store_append_advances_event_seq_counter(postgres_dsn):
    """append_events must advance cayu_sessions.event_seq to the last order."""

    async def ops(store):
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_counter",
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )

        import psycopg

        async def read_counter() -> int:
            async with (
                await psycopg.AsyncConnection.connect(postgres_dsn) as conn,
                conn.cursor() as cur,
            ):
                await cur.execute(
                    "SELECT event_seq FROM cayu_sessions WHERE id = %s",
                    ("sess_counter",),
                )
                row = await cur.fetchone()
            assert row is not None
            return row[0]

        # A freshly created session starts at 0 (no events reserved yet).
        assert await read_counter() == 0

        await store.append_events(
            "sess_counter",
            [
                Event(
                    id=f"a_{i}",
                    type=EventType.MODEL_TEXT_DELTA,
                    session_id="sess_counter",
                    payload={"i": i},
                )
                for i in range(3)
            ],
        )
        assert await read_counter() == 3

        await store.append_event(
            "sess_counter",
            Event(
                id="a_last",
                type=EventType.MODEL_COMPLETED,
                session_id="sess_counter",
                payload={"finish_reason": "stop"},
            ),
        )
        assert await read_counter() == 4

        # An empty batch neither advances the counter nor raises for a live session.
        await store.append_events("sess_counter", [])
        assert await read_counter() == 4

        # The counter tracks the highest stored session_order exactly.
        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                "SELECT MAX(session_order) FROM cayu_events WHERE session_id = %s",
                ("sess_counter",),
            )
            max_order = (await cur.fetchone())[0]
        assert max_order == 4

        # A missing session still raises, even for an empty batch.
        with pytest.raises(KeyError):
            await store.append_events("missing_counter", [])

    _run(postgres_dsn, ops)


def test_postgres_session_store_failed_checkpoint_transition_is_transactional(postgres_dsn):
    """If the transform raises, neither status nor checkpoint may change."""

    async def ops(store):
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_tx",
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )
        await store.checkpoint("sess_tx", {"existing": True})

        def boom(_s, _ck):
            raise RuntimeError("transform failure")

        with pytest.raises(RuntimeError, match="transform failure"):
            await store.transition_status_and_checkpoint(
                "sess_tx",
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.INTERRUPTING,
                checkpoint_transform=boom,
            )

        loaded = await store.load("sess_tx")
        assert loaded is not None
        assert loaded.status == SessionStatus.PENDING
        assert await store.load_checkpoint("sess_tx") == {"existing": True}

    _run(postgres_dsn, ops)


def test_postgres_session_store_summarize_events(postgres_dsn):
    async def ops(store):
        await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_builder",
                environment_name="local",
                messages=[Message.text("user", "build")],
            ),
            identity=_identity(),
        )
        await store.append_events(
            "sess_builder",
            [
                Event(
                    id="event_1",
                    type=EventType.SESSION_STARTED,
                    session_id="sess_builder",
                ),
                Event(
                    id="event_2",
                    type=EventType.TOOL_CALL_COMPLETED,
                    session_id="sess_builder",
                    tool_name="read_file",
                ),
                Event(
                    id="event_3",
                    type=EventType.MODEL_COMPLETED,
                    session_id="sess_builder",
                    payload={"finish_reason": "stop"},
                ),
            ],
        )

        summary = await store.summarize_events("sess_builder")
        assert summary.session_id == "sess_builder"
        assert summary.total_events == 3
        assert summary.counts_by_type == {
            "model.completed": 1,
            "session.started": 1,
            "tool.call.completed": 1,
        }
        assert summary.latest_event is not None
        assert summary.latest_event.event.id == "event_3"

        with pytest.raises(KeyError, match="Session not found"):
            await store.summarize_events("missing_session")

    _run(postgres_dsn, ops)


def test_postgres_session_store_batches_large_event_session_id_queries(postgres_dsn, monkeypatch):
    import cayu.storage.postgres as postgres_module

    monkeypatch.setattr(postgres_module, "_EVENT_QUERY_SESSION_IDS_BATCH_SIZE", 2)

    async def ops(store):
        for index in range(5):
            session_id = f"sess_batch_{index}"
            await store.create(
                RunRequest(
                    agent_name="builder",
                    session_id=session_id,
                    environment_name="local",
                    messages=[Message.text("user", f"batch {index}")],
                ),
                identity=_identity(),
            )
            await store.append_event(
                session_id,
                Event(
                    id=f"event_batch_{index}",
                    type=EventType.SESSION_STARTED,
                    session_id=session_id,
                    interaction_id=("batch-target" if index % 2 == 0 else "batch-other"),
                    agent_name="builder",
                    environment_name="local",
                    workflow_name="maintenance" if index % 2 == 0 else "other",
                    timestamp=datetime(2026, 1, 1, 12, index, tzinfo=UTC),
                ),
            )

        session_ids = (
            "sess_batch_4",
            "sess_batch_0",
            "sess_batch_2",
            "sess_batch_1",
            "sess_batch_3",
        )
        records = await store.query_events(EventQuery(session_ids=session_ids, limit=10))
        limited_records = await store.query_events(EventQuery(session_ids=session_ids, limit=3))
        cursor_records = await store.query_events(
            EventQuery(
                session_ids=session_ids,
                after_sequence=records[1].sequence,
                limit=10,
            )
        )
        filtered_records = await store.query_events(
            EventQuery(
                session_ids=session_ids,
                interaction_id="batch-target",
                workflow_name="maintenance",
                event_type=EventType.SESSION_STARTED,
                limit=10,
            )
        )

        assert [record.event.id for record in records] == [
            "event_batch_0",
            "event_batch_1",
            "event_batch_2",
            "event_batch_3",
            "event_batch_4",
        ]
        assert [record.event.id for record in limited_records] == [
            "event_batch_0",
            "event_batch_1",
            "event_batch_2",
        ]
        assert [record.event.id for record in cursor_records] == [
            "event_batch_2",
            "event_batch_3",
            "event_batch_4",
        ]
        assert [record.event.id for record in filtered_records] == [
            "event_batch_0",
            "event_batch_2",
            "event_batch_4",
        ]

    _run(postgres_dsn, ops)


def test_postgres_session_store_summarize_outcome_from_terminal_and_retry_events(postgres_dsn):
    async def ops(store):
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_outcome",
                messages=[Message.text("user", "retry then stop")],
            ),
            identity=_identity(),
        )
        await store.update_status("sess_outcome", SessionStatus.INTERRUPTED)
        await store.append_events(
            "sess_outcome",
            [
                Event(
                    id="event_retry",
                    type=EventType.MODEL_RETRY,
                    session_id="sess_outcome",
                    payload={
                        "provider": "fake",
                        "model": "fake-model",
                        "step": 1,
                        "attempt": 1,
                        "next_attempt": 2,
                        "max_attempts": 2,
                        "reason": "http_status",
                        "status_code": 429,
                        "delay_seconds": 0.0,
                        "error": "rate limited",
                    },
                ),
                Event(
                    id="event_interrupted",
                    type=EventType.SESSION_INTERRUPTED,
                    session_id="sess_outcome",
                    payload={
                        "interruption_type": "limit_reached",
                        "limit": "total_tokens",
                        "actual": 12,
                        "maximum": 10,
                        "message": "Run limit reached.",
                    },
                ),
                Event(
                    id="event_hook_after_terminal",
                    type=EventType.HOOK_COMPLETED,
                    session_id="sess_outcome",
                    payload={"hook": "after_session_interrupted"},
                ),
            ],
        )

        outcome = await store.summarize_outcome("sess_outcome")
        assert outcome.status == SessionStatus.INTERRUPTED
        assert outcome.reason == "limit_reached"
        assert outcome.details == {
            "interruption_type": "limit_reached",
            "limit": "total_tokens",
            "maximum": 10,
            "actual": 12,
            "message": "Run limit reached.",
        }
        assert outcome.retry == {
            "provider": "fake",
            "model": "fake-model",
            "step": 1,
            "attempt": 1,
            "next_attempt": 2,
            "max_attempts": 2,
            "delay_seconds": 0.0,
            "reason": "http_status",
            "status_code": 429,
        }
        assert outcome.terminal_event is not None
        assert outcome.terminal_event.event.id == "event_interrupted"
        assert outcome.latest_retry_event is not None
        assert outcome.latest_retry_event.event.id == "event_retry"

        with pytest.raises(KeyError, match="Session not found"):
            await store.summarize_outcome("missing_session")

    _run(postgres_dsn, ops)


def test_postgres_session_store_summarize_outcome_scopes_to_latest_lifecycle(postgres_dsn):
    """Terminal and retry events before the latest start/resume must be ignored.

    This exercises the COALESCE(MAX(sequence)) lifecycle subquery directly: a clean
    resume after an earlier completion + retry should surface only the post-resume
    terminal event and no stale retry.
    """

    async def ops(store):
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_resume",
                messages=[Message.text("user", "first")],
            ),
            identity=_identity(),
        )
        await store.update_status("sess_resume", SessionStatus.COMPLETED)
        await store.append_events(
            "sess_resume",
            [
                Event(id="event_started", type=EventType.SESSION_STARTED, session_id="sess_resume"),
                Event(
                    id="event_retry_old",
                    type=EventType.MODEL_RETRY,
                    session_id="sess_resume",
                    payload={
                        "provider": "fake",
                        "model": "fake-model",
                        "step": 1,
                        "attempt": 1,
                        "next_attempt": 2,
                        "max_attempts": 2,
                        "reason": "timeout",
                        "delay_seconds": 0.0,
                    },
                ),
                Event(
                    id="event_completed_first",
                    type=EventType.SESSION_COMPLETED,
                    session_id="sess_resume",
                ),
                Event(id="event_resumed", type=EventType.SESSION_RESUMED, session_id="sess_resume"),
                Event(
                    id="event_completed_after_resume",
                    type=EventType.SESSION_COMPLETED,
                    session_id="sess_resume",
                ),
            ],
        )

        outcome = await store.summarize_outcome("sess_resume")
        assert outcome.status == SessionStatus.COMPLETED
        assert outcome.reason == "completed"
        assert outcome.retry is None
        assert outcome.latest_retry_event is None
        assert outcome.terminal_event is not None
        assert outcome.terminal_event.event.id == "event_completed_after_resume"

    _run(postgres_dsn, ops)


def test_postgres_session_store_query_transcript_pagination_and_role_filter(postgres_dsn):
    async def ops(store):
        await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_transcript",
                messages=[Message.text("user", "build")],
            ),
            identity=_identity(),
        )
        user_message = Message.text("user", "build")
        assistant_message = Message.tool_call(
            tool_call_id="call_1",
            tool_name="read_file",
            arguments={"path": "README.md"},
        )
        tool_message = Message.tool_result(
            tool_call_id="call_1",
            tool_name="read_file",
            content="contents",
            structured={"bytes": 8},
        )
        await store.append_transcript_messages(
            "sess_transcript",
            [user_message, assistant_message, tool_message],
        )

        # Stable, gap-free index across the full transcript; offset/limit paginate it.
        page = await store.query_transcript(
            TranscriptQuery(session_id="sess_transcript", offset=1, limit=1)
        )
        assert page.total_records == 3
        assert [record.index for record in page.records] == [1]
        assert page.records[0].message.content[0].tool_name == "read_file"

        # Role filter keeps the original full-transcript index (0), not a re-counted one.
        user_page = await store.query_transcript(
            TranscriptQuery(session_id="sess_transcript", role="user", limit=10)
        )
        assert user_page.total_records == 1
        assert [record.index for record in user_page.records] == [0]
        assert user_page.records[0].message.content[0].text == "build"

        with pytest.raises(KeyError, match="Session not found"):
            await store.query_transcript(TranscriptQuery(session_id="missing_session"))

    _run(postgres_dsn, ops)


def _lifecycle_request(
    session_id: str,
    *,
    parent: str | None = None,
    labels: dict[str, str] | None = None,
    metadata: dict[str, object] | None = None,
) -> RunRequest:
    return RunRequest(
        agent_name="assistant",
        session_id=session_id,
        parent_session_id=parent,
        labels=labels or {},
        metadata=metadata or {},
        messages=[Message.text("user", "hi")],
    )


def test_postgres_session_store_delete_session_cascades_and_is_idempotent(postgres_dsn):
    async def ops(store):
        await store.create(_lifecycle_request("sess_keep"), identity=_identity())
        await store.create(
            _lifecycle_request("sess_drop", labels={"team": "drop"}), identity=_identity()
        )
        await store.append_events(
            "sess_drop",
            [Event(type=EventType.SESSION_STARTED, session_id="sess_drop", agent_name="assistant")],
        )
        await store.append_transcript_messages("sess_drop", [Message.text("assistant", "bye")])
        await store.checkpoint("sess_drop", {"cursor": 1})

        await store.delete_session("sess_drop")

        assert await store.load("sess_drop") is None
        assert await store.query_events(EventQuery(session_id="sess_drop")) == []
        assert await store.load_checkpoint("sess_drop") is None
        assert (await store.list_sessions(SessionQuery(labels={"team": "drop"}))).sessions == []
        assert await store.load("sess_keep") is not None
        await store.create(_lifecycle_request("sess_drop"), identity=_identity())
        assert await store.load("sess_drop") is not None
        await store.delete_session("sess_never_existed")

    _run(postgres_dsn, ops)


def test_postgres_session_store_delete_rejects_in_flight_sessions(postgres_dsn):
    async def ops(store):
        for index, status in enumerate((SessionStatus.RUNNING, SessionStatus.INTERRUPTING)):
            session_id = f"sess_inflight_{index}"
            await store.create(_lifecycle_request(session_id), identity=_identity())
            await store.update_status(session_id, status)
            with pytest.raises(ValueError, match="interrupt it first"):
                await store.delete_session(session_id)
            assert await store.load(session_id) is not None

    _run(postgres_dsn, ops)


def test_postgres_session_store_delete_rechecks_status_after_waiting_for_row_lock(postgres_dsn):
    async def ops(store):
        import psycopg

        session_id = "sess_delete_race"
        await store.create(_lifecycle_request(session_id), identity=_identity())
        async with await psycopg.AsyncConnection.connect(postgres_dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE cayu_sessions SET status = %s WHERE id = %s",
                    (str(SessionStatus.RUNNING), session_id),
                )
                delete_task = asyncio.create_task(store.delete_session(session_id))
                await asyncio.sleep(0.1)
                assert delete_task.done() is False
            await conn.commit()

        with pytest.raises(ValueError, match="interrupt it first"):
            await asyncio.wait_for(delete_task, timeout=2.0)
        loaded = await store.load(session_id)
        assert loaded is not None
        assert loaded.status == SessionStatus.RUNNING

    _run(postgres_dsn, ops)


def test_postgres_session_store_delete_parent_nulls_child_parent(postgres_dsn):
    async def ops(store):
        await store.create(_lifecycle_request("sess_parent"), identity=_identity())
        await store.create(
            _lifecycle_request("sess_child", parent="sess_parent"), identity=_identity()
        )
        await store.delete_session("sess_parent")
        child = await store.load("sess_child")
        assert child is not None
        assert child.parent_session_id is None

    _run(postgres_dsn, ops)


def test_postgres_session_store_update_labels_replaces_and_filters(postgres_dsn):
    async def ops(store):
        created = await store.create(
            _lifecycle_request("sess_labeled", labels={"team": "research", "stage": "draft"}),
            identity=_identity(),
        )
        await store.update_status("sess_labeled", SessionStatus.COMPLETED)
        before_edit = await store.load("sess_labeled")
        assert before_edit is not None
        updated = await store.update_labels("sess_labeled", {"stage": "review"})
        assert updated.labels == {"stage": "review"}
        assert updated.created_at == created.created_at
        assert updated.updated_at >= before_edit.updated_at
        assert updated.last_activity_at == before_edit.last_activity_at
        assert updated.status == SessionStatus.COMPLETED
        reloaded = await store.load("sess_labeled")
        assert reloaded is not None
        assert reloaded.labels == {"stage": "review"}
        matched = (await store.list_sessions(SessionQuery(labels={"stage": "review"}))).sessions
        assert [session.id for session in matched] == ["sess_labeled"]
        stale = (await store.list_sessions(SessionQuery(labels={"team": "research"}))).sessions
        assert stale == []
        cleared = await store.update_labels("sess_labeled", {})
        assert cleared.labels == {}
        assert cleared.last_activity_at == before_edit.last_activity_at
        assert (await store.list_sessions(SessionQuery(labels={"stage": "review"}))).sessions == []
        with pytest.raises(ValueError, match="reserved"):
            await store.update_labels("sess_labeled", {"cayu:internal": "x"})
        with pytest.raises(KeyError, match="Session not found"):
            await store.update_labels("sess_missing", {"k": "v"})

    _run(postgres_dsn, ops)


def test_postgres_session_store_update_metadata_replaces(postgres_dsn):
    async def ops(store):
        await store.create(
            _lifecycle_request(
                "sess_meta",
                metadata={
                    "a": 1,
                    "keep": False,
                    "subagent": {"mode": "background"},
                    "cayu:taint_labels": ["untrusted"],
                },
            ),
            identity=_identity(),
        )
        await store.update_status("sess_meta", SessionStatus.COMPLETED)
        before_edit = await store.load("sess_meta")
        assert before_edit is not None
        updated = await store.update_metadata("sess_meta", {"b": [1, 2]})
        assert updated.metadata == {
            "b": [1, 2],
            "subagent": {"mode": "background"},
            "cayu:taint_labels": ["untrusted"],
        }
        assert updated.updated_at >= before_edit.updated_at
        assert updated.last_activity_at == before_edit.last_activity_at
        assert updated.status == SessionStatus.COMPLETED
        reloaded = await store.load("sess_meta")
        assert reloaded is not None
        assert reloaded.metadata == updated.metadata
        with pytest.raises(ValueError, match="runtime-owned"):
            await store.update_metadata("sess_meta", {"subagent": {}})
        with pytest.raises(ValueError, match="runtime-owned"):
            await store.update_metadata("sess_meta", {"cayu:taint_labels": []})
        with pytest.raises(ValueError, match="NUL"):
            await store.update_metadata("sess_meta", {"value": "not\x00durable"})
        assert (await store.load("sess_meta")).metadata == updated.metadata
        with pytest.raises(KeyError, match="Session not found"):
            await store.update_metadata("sess_missing", {"k": "v"})

    _run(postgres_dsn, ops)


def test_postgres_session_store_cursor_pagination_is_stable_across_orders(postgres_dsn):
    async def ops(store):
        for index in range(5):
            await store.create(_lifecycle_request(f"sess_{index}"), identity=_identity())
        for order in SessionOrder:
            full = (await store.list_sessions(SessionQuery(order_by=order, limit=100))).sessions
            expected_ids = [session.id for session in full]
            collected: list[str] = []
            cursor: str | None = None
            while True:
                page = await store.list_sessions(
                    SessionQuery(order_by=order, limit=2, cursor=cursor, include_total_count=True)
                )
                assert page.total_count == len(expected_ids)
                collected.extend(session.id for session in page.sessions)
                if page.next_cursor is None:
                    break
                cursor = page.next_cursor
            assert collected == expected_ids, order

    _run(postgres_dsn, ops)


def test_postgres_session_store_filters_debug_state_before_pagination(postgres_dsn):
    async def create(
        store,
        session_id: str,
        status: SessionStatus,
        events: list[Event],
    ) -> None:
        await store.create(_lifecycle_request(session_id), identity=_identity())
        await store.update_status(session_id, status)
        await store.append_events(session_id, events)

    async def ops(store):
        await create(
            store,
            "pg_debug_normal",
            SessionStatus.COMPLETED,
            [Event(type=EventType.SESSION_COMPLETED, session_id="pg_debug_normal")],
        )
        await create(
            store,
            "pg_debug_tool_failed",
            SessionStatus.COMPLETED,
            [
                Event(type=EventType.TOOL_CALL_FAILED, session_id="pg_debug_tool_failed"),
                Event(type=EventType.SESSION_COMPLETED, session_id="pg_debug_tool_failed"),
            ],
        )
        await create(
            store,
            "pg_debug_tool_blocked",
            SessionStatus.COMPLETED,
            [
                Event(type=EventType.TOOL_CALL_BLOCKED, session_id="pg_debug_tool_blocked"),
                Event(type=EventType.SESSION_COMPLETED, session_id="pg_debug_tool_blocked"),
            ],
        )
        await create(
            store,
            "pg_debug_failed",
            SessionStatus.FAILED,
            [Event(type=EventType.SESSION_FAILED, session_id="pg_debug_failed")],
        )
        await create(
            store,
            "pg_debug_interrupted",
            SessionStatus.INTERRUPTED,
            [Event(type=EventType.SESSION_INTERRUPTED, session_id="pg_debug_interrupted")],
        )

        tool_issues = await store.list_sessions(
            SessionQuery(
                debug_state=SessionDebugState.TOOL_ISSUE,
                order_by=SessionOrder.CREATED_AT_ASC,
            )
        )
        assert [session.id for session in tool_issues.sessions] == [
            "pg_debug_tool_failed",
            "pg_debug_tool_blocked",
        ]

        needs_attention = await store.list_sessions(
            SessionQuery(
                debug_state=SessionDebugState.NEEDS_ATTENTION,
                order_by=SessionOrder.CREATED_AT_ASC,
                limit=3,
                include_total_count=True,
            )
        )
        assert [session.id for session in needs_attention.sessions] == [
            "pg_debug_tool_failed",
            "pg_debug_tool_blocked",
            "pg_debug_failed",
        ]
        assert needs_attention.total_count == 4
        assert needs_attention.next_cursor is not None

        next_page = await store.list_sessions(
            SessionQuery(
                debug_state=SessionDebugState.NEEDS_ATTENTION,
                order_by=SessionOrder.CREATED_AT_ASC,
                limit=3,
                cursor=needs_attention.next_cursor,
                include_total_count=True,
            )
        )
        assert [session.id for session in next_page.sessions] == ["pg_debug_interrupted"]
        assert next_page.total_count == 4
        assert next_page.next_cursor is None

        assert [
            session.id
            for session in (
                await store.list_sessions(
                    SessionQuery(
                        debug_state=SessionDebugState.SESSION_FAILURE,
                        order_by=SessionOrder.CREATED_AT_ASC,
                    )
                )
            ).sessions
        ] == ["pg_debug_failed"]
        assert [
            session.id
            for session in (
                await store.list_sessions(
                    SessionQuery(
                        debug_state=SessionDebugState.INTERRUPTION,
                        order_by=SessionOrder.CREATED_AT_ASC,
                    )
                )
            ).sessions
        ] == ["pg_debug_interrupted"]

    _run(postgres_dsn, ops)


def test_postgres_session_store_cursor_survives_concurrent_insert(postgres_dsn):
    async def ops(store):
        for index in range(4):
            await store.create(_lifecycle_request(f"sess_{index}"), identity=_identity())
        order = SessionOrder.CREATED_AT_ASC
        first = await store.list_sessions(SessionQuery(order_by=order, limit=2))
        seen = [session.id for session in first.sessions]
        await store.create(_lifecycle_request("sess_inserted"), identity=_identity())
        cursor = first.next_cursor
        while cursor is not None:
            page = await store.list_sessions(SessionQuery(order_by=order, limit=2, cursor=cursor))
            seen.extend(session.id for session in page.sessions)
            cursor = page.next_cursor
        assert len(seen) == len(set(seen)), seen
        assert {"sess_0", "sess_1", "sess_2", "sess_3"} <= set(seen)

    _run(postgres_dsn, ops)


def test_postgres_session_store_rejects_invalid_cursor(postgres_dsn):
    async def ops(store):
        await store.create(_lifecycle_request("sess_only"), identity=_identity())
        with pytest.raises(ValueError, match="[Cc]ursor"):
            await store.list_sessions(SessionQuery(cursor="!!!not-a-cursor"))
        forged = base64.urlsafe_b64encode(
            json.dumps(
                {
                    "version": 1,
                    "sort_value": "not-a-timestamp",
                    "session_id_b64": base64.urlsafe_b64encode(b"sess_only").decode("ascii"),
                },
                separators=(",", ":"),
            ).encode()
        ).decode("ascii")
        with pytest.raises(ValueError, match="[Cc]ursor"):
            await store.list_sessions(SessionQuery(cursor=forged))

    _run(postgres_dsn, ops)


def test_postgres_session_store_cursor_pagination_empty_result(postgres_dsn):
    async def ops(store):
        await store.create(_lifecycle_request("sess_only"), identity=_identity())
        page = await store.list_sessions(
            SessionQuery(labels={"absent": "1"}, limit=2, include_total_count=True)
        )
        assert page.sessions == []
        assert page.next_cursor is None
        assert page.total_count == 0

    _run(postgres_dsn, ops)
