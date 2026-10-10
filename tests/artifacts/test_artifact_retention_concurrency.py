"""Late references and publication fences for artifact retention."""

from __future__ import annotations

import asyncio
import contextlib
import json
import sqlite3
from datetime import timedelta

import pytest
from tests.core.session_retention_conformance import RetentionHarness, create_retention_session

from cayu.artifacts.base import ArtifactScope
from cayu.artifacts.local import LocalArtifactStore
from cayu.artifacts.retention import apply_artifact_retention_policy
from cayu.messages import Message
from cayu.runtime.storage_retention import apply_storage_retention
from cayu.snapshots.base import SQLiteAgentSnapshotStore
from cayu.storage.evals_sqlite import SQLiteEvalStore
from cayu.storage.retention import (
    ArtifactRetentionPolicy,
    RetentionPhase,
    RetentionProtection,
    StorageRetentionPolicy,
)
from cayu.storage.sqlite import SQLiteSessionStore


def _policy():
    return ArtifactRetentionPolicy(older_than=timedelta(microseconds=1), dry_run=False)


async def _put(artifacts, name):
    return await artifacts.put_bytes(
        b"evidence", filename=name, scope=ArtifactScope.ENVIRONMENT, environment_name="workspace"
    )


async def _append_reference(sessions, artifact_id):
    await sessions.append_transcript_messages(
        "reader",
        [
            Message.tool_result(
                tool_call_id="call-1",
                tool_name="read_file",
                content="see the attachment",
                artifacts=[{"artifact_id": artifact_id}],
            )
        ],
    )


async def _assert_late_reference(sessions, artifacts):
    await create_retention_session(RetentionHarness(store=sessions), "reader")
    artifact = await _put(artifacts, "late.txt")

    async def progress(update):
        if update.phase is RetentionPhase.PLANNED:
            assert update.planned_items == 1
            await _append_reference(sessions, artifact.id)
            assert artifact.id in await sessions.retention_artifact_references()

    report = await apply_storage_retention(
        StorageRetentionPolicy(artifacts=_policy(), dry_run=False),
        session_store=sessions,
        artifact_stores=[artifacts],
        progress=progress,
    )
    assert not report.errors
    [result] = report.reports
    assert result.items == ()
    assert result.protected[0].protections == (RetentionProtection.SESSION_REFERENCE,)
    assert [item.id for item in (await artifacts.list(limit=None)).artifacts] == [artifact.id]
    assert (await sessions.load_retention_audit(result.audit_id)).entries == ()


def test_late_transcript_reference_is_kept(sqlite_resources, tmp_path):
    async def run():
        async with sqlite_resources as resources:
            sessions = resources.own(SQLiteSessionStore(resources.path()))
            await _assert_late_reference(sessions, LocalArtifactStore(tmp_path / "artifacts"))

    asyncio.run(run())


def _assert_sqlite_writer_fenced(path):
    with (
        contextlib.closing(sqlite3.connect(path, timeout=0)) as connection,
        pytest.raises(sqlite3.OperationalError, match="locked"),
    ):
        connection.execute("BEGIN IMMEDIATE")


@pytest.mark.parametrize("same_database", [False, True])
@pytest.mark.parametrize("source", ["eval", "snapshot"])
@pytest.mark.parametrize("plural", [False, True])
def test_late_external_references_and_database_fences(
    sqlite_resources, tmp_path, same_database, source, plural
):
    async def run():
        async with sqlite_resources as resources:
            session_path = resources.path()
            source_path = session_path if same_database else resources.path("references.sqlite")
            sessions = resources.own(SQLiteSessionStore(session_path))
            evals = resources.own(SQLiteEvalStore(source_path)) if source == "eval" else None
            snapshots = [SQLiteAgentSnapshotStore(source_path)] if source == "snapshot" else []
            checked = []

            class CheckingArtifacts(LocalArtifactStore):
                async def delete(self, artifact_id):
                    # Both databases exclude a writer all the way into the
                    # real delete, even when sources share the session DB.
                    _assert_sqlite_writer_fenced(session_path)
                    _assert_sqlite_writer_fenced(source_path)
                    checked.append(artifact_id)
                    await super().delete(artifact_id)

            artifacts = CheckingArtifacts(tmp_path / "artifacts")
            named = await _put(artifacts, "named.txt")
            free = await _put(artifacts, "free.txt")

            async def progress(update):
                if update.phase is RetentionPhase.PLANNED:
                    assert update.planned_items == 2
                    # Seed only reference rows, as in the retention conformance
                    # suite. No production lock or query is patched.
                    with contextlib.closing(sqlite3.connect(source_path)) as connection:
                        document = json.dumps(
                            {"artifact_ids": [named.id]} if plural else {"artifact_id": named.id}
                        )
                        if source == "eval":
                            connection.execute(
                                "INSERT INTO cayu_eval_run_trial_checkpoints "
                                "(run_id, case_id, trial_number, checkpoint_json, document_bytes) "
                                "VALUES ('run', 'case', 1, ?, ?)",
                                (document, len(document)),
                            )
                        else:
                            connection.execute(
                                "INSERT INTO cayu_agent_snapshot_bindings (binding_id, "
                                "snapshot_root, authority_scope_fingerprint, binding_document, "
                                "snapshot_document, put_receipt_document) "
                                "VALUES ('b', 'r', 'f', '{}', ?, '{}')",
                                (document,),
                            )
                            connection.execute(
                                "INSERT INTO cayu_agent_snapshot_pins (pin_id, snapshot_root, "
                                "binding_id, document, released) VALUES ('p', 'r', 'b', '{}', 0)"
                            )
                        connection.commit()
                elif update.phase is RetentionPhase.BATCH:
                    # The fence ends before progress callbacks and audits.
                    with contextlib.closing(sqlite3.connect(source_path, timeout=0)) as connection:
                        connection.execute("BEGIN IMMEDIATE")
                        connection.rollback()

            report = await apply_storage_retention(
                StorageRetentionPolicy(artifacts=_policy(), dry_run=False),
                session_store=sessions,
                eval_store=evals,
                snapshot_stores=snapshots,
                artifact_stores=[artifacts],
                progress=progress,
            )
            assert not report.errors
            [result] = report.reports
            assert [item.item_id for item in result.items] == [free.id]
            assert checked == [free.id]
            assert result.protected[0].item_id == named.id
            expected = (
                RetentionProtection.EVAL_REFERENCE
                if source == "eval"
                else RetentionProtection.SNAPSHOT_PIN
            )
            assert result.protected[0].protections == (expected,)

    asyncio.run(run())


def test_cancellation_keeps_fence_until_delete_and_audit_settle(sqlite_resources, tmp_path):
    async def run():
        async with sqlite_resources as resources:
            path = resources.path()
            sessions = resources.own(SQLiteSessionStore(path))
            entered, release = asyncio.Event(), asyncio.Event()

            class PausedArtifacts(LocalArtifactStore):
                async def delete(self, artifact_id):
                    entered.set()
                    await release.wait()
                    _assert_sqlite_writer_fenced(path)
                    await super().delete(artifact_id)

            artifacts = PausedArtifacts(tmp_path / "artifacts")
            artifact = await _put(artifacts, "cancelled.txt")
            task = resources.task(
                apply_artifact_retention_policy(
                    artifacts, _policy(), session_store=sessions, audit=sessions
                )
            )
            try:
                await asyncio.wait_for(entered.wait(), timeout=5)
                task.cancel()
                with pytest.raises(TimeoutError):
                    await asyncio.wait_for(asyncio.shield(task), timeout=0.05)
                _assert_sqlite_writer_fenced(path)
            finally:
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await task
            assert (await artifacts.list(limit=None)).artifacts == ()
            [record] = await sessions.list_retention_audits()
            audit = await sessions.load_retention_audit(record.audit_id)
            assert [entry.item_id for entry in audit.entries] == [artifact.id]
            with contextlib.closing(sqlite3.connect(path, timeout=0)) as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.rollback()

    asyncio.run(run())


def test_unknown_reference_source_fails_closed(sqlite_resources, tmp_path):
    async def run():
        async with sqlite_resources as resources:
            sessions = resources.own(SQLiteSessionStore(resources.path()))
            artifacts = LocalArtifactStore(tmp_path / "artifacts")
            artifact = await _put(artifacts, "opaque.txt")
            report = await apply_artifact_retention_policy(
                artifacts,
                _policy(),
                session_store=sessions,
                snapshot_stores=[object()],
                audit=sessions,
            )
            assert report.items == ()
            assert report.protected[0].item_id == artifact.id
            assert report.protected[0].protections == (RetentionProtection.ERASURE_GUARD,)

    asyncio.run(run())


def test_postgres_late_reference_and_publication_fence(postgres_dsn, tmp_path):
    async def run():
        import psycopg

        from cayu.storage.evals_postgres import PostgresEvalStore
        from cayu.storage.migrations import SchemaMode
        from cayu.storage.postgres import PostgresSessionStore

        # The guard must not borrow a second connection while holding the
        # first, and two store objects for this DB must share one guard.
        sessions = PostgresSessionStore(
            postgres_dsn, min_size=1, max_size=1, schema_mode=SchemaMode.CREATE
        )
        evals = PostgresEvalStore(postgres_dsn, min_size=1, max_size=1)
        try:
            await _assert_late_reference(sessions, LocalArtifactStore(tmp_path / "late"))
            checked = []

            class CheckingArtifacts(LocalArtifactStore):
                async def delete(self, artifact_id):
                    async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
                        await connection.execute("SET LOCAL lock_timeout = '100ms'")
                        with pytest.raises(psycopg.errors.LockNotAvailable):
                            await connection.execute(
                                "UPDATE cayu_transcript_messages SET message = message"
                            )
                    checked.append(artifact_id)
                    await super().delete(artifact_id)

            artifacts = CheckingArtifacts(tmp_path / "fenced")
            artifact = await _put(artifacts, "free.txt")
            report = await asyncio.wait_for(
                apply_artifact_retention_policy(
                    artifacts, _policy(), session_store=sessions, eval_store=evals, audit=sessions
                ),
                timeout=10,
            )
            assert [item.item_id for item in report.items] == checked == [artifact.id]
        finally:
            await evals.close()
            await sessions.close()

    asyncio.run(run())
