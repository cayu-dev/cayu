"""Eval-store retention for SQLite and Postgres: selection, protections and audit."""

from __future__ import annotations

import asyncio
import contextlib
import json
import sqlite3
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from tests.evals.test_corpus_execution import _corpus, _provider, _target

from cayu.evals.execution import run_corpus_suite
from cayu.evals.store import (
    EvalBaselineKey,
    EvalBaselineUpdate,
    EvalRunInvocation,
    EvalRunRequest,
    EvalRunStatus,
)
from cayu.storage.retention import (
    EvalRetentionPolicy,
    RetentionAuditState,
    RetentionMode,
    RetentionProtection,
)
from cayu.vaults.redaction import SecretRedactor

_NO_SECRETS = SecretRedactor()


@dataclass
class EvalHarness:
    store: Any
    corpus: Any
    result: Any
    execute: Callable[..., Awaitable[None]]
    postgres: bool
    snapshot_path: Any = None

    def request(
        self, run_id: str, digit: str, *, invocation: EvalRunInvocation | None = None
    ) -> EvalRunRequest:
        suite = self.corpus.suites[0]
        return EvalRunRequest(
            run_id=run_id,
            idempotency_key="sha256:" + digit * 64,
            corpus_revision=self.corpus.revision,
            target_key=self.corpus.target_key,
            suite_id=suite.id,
            suite_revision=suite.revision,
            max_concurrency=1,
            invocation=EvalRunInvocation() if invocation is None else invocation,
        )

    async def completed(self, run_id: str, digit: str) -> None:
        await self.store.admit_run(self.request(run_id, digit), redact_json=_NO_SECRETS.redact_json)
        claimed = await self.store.claim_run()
        assert claimed is not None
        await self.store.publish_result(
            claimed.claim, self.result, redact_json=_NO_SECRETS.redact_json
        )

    async def cancelled(
        self, run_id: str, digit: str, *, invocation: EvalRunInvocation | None = None
    ) -> None:
        await self.store.admit_run(
            self.request(run_id, digit, invocation=invocation),
            redact_json=_NO_SECRETS.redact_json,
        )
        assert (await self.store.request_cancel(run_id)).status is EvalRunStatus.CANCELLED

    async def queued(self, run_id: str, digit: str) -> None:
        await self.store.admit_run(self.request(run_id, digit), redact_json=_NO_SECRETS.redact_json)

    async def age(self, days: int = 3) -> None:
        past = datetime.now(UTC) - timedelta(days=days)
        value = past if self.postgres else past.isoformat()
        await self.execute(
            "UPDATE cayu_eval_runs SET created_at = ?, updated_at = ?, finished_at = ?, "
            "started_at = CASE WHEN started_at IS NULL THEN NULL ELSE ? END, "
            "cancel_requested_at = CASE WHEN cancel_requested_at IS NULL THEN NULL ELSE ? END "
            "WHERE finished_at IS NOT NULL",
            (value, value, value, value, value),
        )


def _policy(**overrides: Any) -> EvalRetentionPolicy:
    return EvalRetentionPolicy(older_than=timedelta(days=1), **overrides)


def _ids(report) -> list[str]:
    return [item.item_id for item in report.items]


def _reasons(report) -> dict[str, set[RetentionProtection]]:
    return {item.item_id: set(item.protections) for item in report.protected}


async def assert_dry_run_matches_apply_and_deletes_runs(harness: EvalHarness) -> None:
    store = harness.store
    await harness.completed("done", "1")
    await harness.cancelled("stopped", "2")
    await harness.queued("waiting", "3")
    fresh = await _policy_report(store, dry_run=True)
    assert fresh.items == (), "runs that just finished are not old enough"
    await harness.age()
    planned = await _policy_report(store, dry_run=True)
    assert sorted(_ids(planned)) == ["done", "stopped"]
    done = next(item for item in planned.items if item.item_id == "done")
    assert done.counts == {
        "result_records_removed": 1,
        "results_removed": 1,
        "runs_removed": 1,
        "trial_checkpoints_removed": 0,
    }
    applied = await _policy_report(store, dry_run=False)
    assert [(i.item_id, i.counts, i.bytes) for i in applied.items] == [
        (i.item_id, i.counts, i.bytes) for i in planned.items
    ]
    assert await store.load_run("done") is None
    assert await store.load_run("stopped") is None
    assert await store.load_result_by_revision(harness.result.revision) is None
    assert (await store.load_run("waiting")).status is EvalRunStatus.QUEUED
    record = await store.load_retention_audit(applied.audit_id)
    assert record.state is RetentionAuditState.COMPLETED
    assert record.store_kind == "evals"
    assert sorted(entry.item_id for entry in record.entries) == ["done", "stopped"]
    [listed] = await store.list_retention_audits(item_id="done")
    assert listed.audit_id == applied.audit_id


async def _policy_report(store, *, dry_run: bool, **overrides):
    return await store.apply_retention_policy(_policy(dry_run=dry_run, **overrides))


async def assert_baseline_protects(harness: EvalHarness) -> None:
    await harness.completed("chosen", "1")
    suite = harness.corpus.suites[0]
    await harness.store.set_baseline(
        EvalBaselineUpdate(
            key=EvalBaselineKey(
                target_key=harness.corpus.target_key,
                corpus_revision=harness.corpus.revision,
                suite_id=suite.id,
            ),
            result_revision=harness.result.revision,
            expected_generation=0,
            operation_id="sha256:" + "e" * 64,
            actor_id="retention-test",
        ),
        redact_json=_NO_SECRETS.redact_json,
    )
    await harness.age()
    for dry_run in (True, False):
        report = await _policy_report(harness.store, dry_run=dry_run)
        assert _reasons(report) == {"chosen": {RetentionProtection.BASELINE}}
    assert await harness.store.load_run("chosen") is not None


async def assert_campaign_evidence_protects(harness: EvalHarness) -> None:
    await harness.cancelled(
        "campaign", "1", invocation=EvalRunInvocation(retain_trial_checkpoints=True)
    )
    await harness.age()
    report = await _policy_report(harness.store, dry_run=False)
    assert _reasons(report) == {"campaign": {RetentionProtection.CAMPAIGN_EVIDENCE}}
    assert await harness.store.load_run("campaign") is not None


async def assert_retry_lineage_protects(harness: EvalHarness) -> None:
    await harness.cancelled("source", "1")
    await harness.queued("retry", "2")
    await harness.execute(
        "UPDATE cayu_eval_runs SET invocation_json = ? WHERE run_id = 'retry'",
        (json.dumps({"schema_version": 1, "retry_of": {"run_id": "source"}}),),
    )
    await harness.age()
    report = await _policy_report(harness.store, dry_run=False)
    assert _reasons(report) == {"source": {RetentionProtection.EVAL_REFERENCE}}


async def assert_caller_protected_runs(harness: EvalHarness) -> None:
    await harness.cancelled("named", "1")
    await harness.age()
    report = await harness.store.apply_retention_policy(
        _policy(dry_run=False), protected_ids=["named"]
    )
    assert _reasons(report) == {"named": {RetentionProtection.CALLER_PROTECTED}}


async def assert_snapshot_pin_protects(harness: EvalHarness) -> None:
    if harness.snapshot_path is None:
        pytest.skip("The agent snapshot store has no tables in this backend's database.")
    from cayu.snapshots.base import SQLiteAgentSnapshotStore

    await harness.cancelled("pinned", "1")
    await harness.age()
    SQLiteAgentSnapshotStore(harness.snapshot_path)
    await harness.execute(
        "INSERT INTO cayu_agent_snapshot_bindings (binding_id, snapshot_root, "
        "authority_scope_fingerprint, binding_document, snapshot_document, "
        "put_receipt_document) VALUES ('b', 'r', 'f', '{}', ?, '{}')",
        (json.dumps({"evaluation": {"run_id": "pinned"}}),),
    )
    await harness.execute(
        "INSERT INTO cayu_agent_snapshot_pins (pin_id, snapshot_root, binding_id, document, "
        "released) VALUES ('p', 'r', 'b', '{}', 0)"
    )
    report = await _policy_report(harness.store, dry_run=False)
    assert _reasons(report) == {"pinned": {RetentionProtection.SNAPSHOT_PIN}}


async def assert_session_references_are_reported(harness: EvalHarness) -> None:
    await harness.queued("progressing", "1")
    await harness.execute(
        "UPDATE cayu_eval_runs SET scenario_progress_json = ? WHERE run_id = 'progressing'",
        (json.dumps({"trials": [{"session_id": "trial-session"}]}),),
    )
    assert "trial-session" in await harness.store.retention_session_references()


SCENARIOS = (
    assert_dry_run_matches_apply_and_deletes_runs,
    assert_baseline_protects,
    assert_campaign_evidence_protects,
    assert_retry_lineage_protects,
    assert_caller_protected_runs,
    assert_snapshot_pin_protects,
    assert_session_references_are_reported,
)


async def _scenario_inputs():
    corpus = _corpus(trials=1)
    result = await run_corpus_suite(
        _target(_provider(trials=1)), corpus, corpus.suites[0].id, max_concurrency=1
    )
    return corpus, result


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda scenario: scenario.__name__)
def test_sqlite_eval_retention(tmp_path, scenario) -> None:
    async def run() -> None:
        from cayu.storage.evals_sqlite import SQLiteEvalStore

        path = tmp_path / "evals.sqlite"
        corpus, result = await _scenario_inputs()
        store = SQLiteEvalStore(path)

        async def execute(sql, parameters=()):
            with contextlib.closing(sqlite3.connect(path, timeout=5)) as connection:
                connection.execute(sql, parameters)
                connection.commit()

        try:
            await store.save_corpus(corpus, redact_json=_NO_SECRETS.redact_json)
            await scenario(EvalHarness(store, corpus, result, execute, False, path))
        finally:
            await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda scenario: scenario.__name__)
def test_postgres_eval_retention(postgres_dsn, scenario) -> None:
    async def run() -> None:
        import psycopg
        from psycopg import sql

        from cayu.storage.evals_postgres import PostgresEvalStore
        from cayu.storage.migrations import SchemaMode

        async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
            rows = await (
                await connection.execute(
                    "SELECT tablename FROM pg_tables WHERE schemaname = current_schema() "
                    "AND tablename LIKE 'cayu_%%'"
                )
            ).fetchall()
            for (table,) in rows:
                await connection.execute(
                    sql.SQL("DROP TABLE {} CASCADE").format(sql.Identifier(table))
                )
            await connection.commit()
        corpus, result = await _scenario_inputs()
        store = PostgresEvalStore(
            postgres_dsn, min_size=1, max_size=2, schema_mode=SchemaMode.MIGRATE
        )

        async def execute(statement, parameters=()):
            async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
                await connection.execute(statement.replace("?", "%s"), parameters)
                await connection.commit()

        try:
            await store.save_corpus(corpus, redact_json=_NO_SECRETS.redact_json)
            await scenario(EvalHarness(store, corpus, result, execute, True))
        finally:
            await store.close()

    asyncio.run(run())


def test_eval_retention_policy_contract() -> None:
    with pytest.raises(ValueError, match="terminal eval runs"):
        EvalRetentionPolicy(older_than=timedelta(days=1), statuses={"running"})
    with pytest.raises(ValueError, match="only mode='delete'"):
        EvalRetentionPolicy(older_than=timedelta(days=1), mode=RetentionMode.COMPACT)
    assert EvalRetentionPolicy(older_than=timedelta(days=1)).statuses == {
        "completed",
        "failed",
        "cancelled",
    }


def test_in_memory_eval_store_declares_no_retention() -> None:
    from cayu.evals.store import InMemoryEvalStore
    from cayu.storage.evals_postgres import PostgresEvalStore
    from cayu.storage.evals_sqlite import SQLiteEvalStore

    assert InMemoryEvalStore.supports_storage_retention is False
    assert SQLiteEvalStore.supports_storage_retention is True
    assert PostgresEvalStore.supports_storage_retention is True
