"""Artifact-store retention: age and size selection, every protection, and audit."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path

import pytest
from tests.core.session_retention_conformance import RetentionHarness, create_retention_session

from cayu.artifacts.base import ArtifactScope
from cayu.artifacts.local import LocalArtifactStore
from cayu.artifacts.retention import apply_artifact_retention_policy
from cayu.messages import Message
from cayu.storage.retention import (
    ArtifactRetentionPolicy,
    RetentionAuditState,
    RetentionMode,
    RetentionProtection,
)
from cayu.storage.sqlite import SQLiteSessionStore

_AGED = timedelta(milliseconds=1)


def _policy(**overrides) -> ArtifactRetentionPolicy:
    overrides.setdefault("older_than", _AGED)
    return ArtifactRetentionPolicy(**overrides)


async def _put(store: LocalArtifactStore, name: str, size: int = 10, *, session_id=None):
    if session_id is None:
        return await store.put_bytes(
            b"x" * size,
            filename=name,
            scope=ArtifactScope.ENVIRONMENT,
            environment_name="workspace",
        )
    return await store.put_bytes(
        b"x" * size, filename=name, scope=ArtifactScope.SESSION, session_id=session_id
    )


def _reasons(report) -> dict[str, set[RetentionProtection]]:
    return {item.item_id: set(item.protections) for item in report.protected}


def _run(tmp_path: Path, scenario) -> None:
    async def run() -> None:
        artifacts = LocalArtifactStore(tmp_path / "artifacts")
        sessions = SQLiteSessionStore(tmp_path / "sessions.db")
        try:
            await scenario(artifacts, sessions)
        finally:
            await sessions.close()

    asyncio.run(run())


def test_dry_run_matches_apply_and_audits_through_the_session_store(tmp_path) -> None:
    async def scenario(artifacts, sessions) -> None:
        old = await _put(artifacts, "old.txt", 64)
        small = await _put(artifacts, "small.txt", 1)
        await asyncio.sleep(0.01)
        assert (
            await apply_artifact_retention_policy(
                artifacts, _policy(older_than=timedelta(days=1)), session_store=sessions
            )
        ).items == ()
        planned = await apply_artifact_retention_policy(
            artifacts, _policy(min_size_bytes=8), session_store=sessions
        )
        assert [item.item_id for item in planned.items] == [old.id]
        applied = await apply_artifact_retention_policy(
            artifacts,
            _policy(min_size_bytes=8, dry_run=False),
            session_store=sessions,
            audit=sessions,
        )
        assert [(i.item_id, i.counts, i.bytes) for i in applied.items] == [
            (old.id, {"artifacts_removed": 1}, 64)
        ]
        remaining = await artifacts.list(limit=None)
        assert [item.id for item in remaining.artifacts] == [small.id]
        record = await sessions.load_retention_audit(applied.audit_id)
        assert record.store_kind == "artifacts"
        assert record.state is RetentionAuditState.COMPLETED
        assert record.policy["store_id"] == artifacts.id
        assert [entry.item_id for entry in record.entries] == [old.id]
        [listed] = await sessions.list_retention_audits(store_kind="artifacts")
        assert listed.audit_id == applied.audit_id

    _run(tmp_path, scenario)


def test_pins_protect_and_deletion_enforces_them(tmp_path) -> None:
    async def scenario(artifacts, sessions) -> None:
        pinned = await _put(artifacts, "pinned.txt")
        await artifacts.pin(pinned.id, owner="workspace-checkpoint")
        await asyncio.sleep(0.01)
        report = await apply_artifact_retention_policy(
            artifacts, _policy(dry_run=False), session_store=sessions, audit=sessions
        )
        assert _reasons(report) == {pinned.id: {RetentionProtection.PINNED}}
        await artifacts.release_pin(pinned.id, owner="workspace-checkpoint")
        report = await apply_artifact_retention_policy(
            artifacts, _policy(dry_run=False), session_store=sessions, audit=sessions
        )
        assert [item.item_id for item in report.items] == [pinned.id]

    _run(tmp_path, scenario)


def test_a_store_that_cannot_report_pins_keeps_everything(tmp_path) -> None:
    class Opaque(LocalArtifactStore):
        async def has_retention_pins(self, artifact_id: str) -> bool | None:
            return None

    async def run() -> None:
        artifacts = Opaque(tmp_path / "artifacts")
        await _put(artifacts, "unknown.txt")
        await asyncio.sleep(0.01)
        report = await apply_artifact_retention_policy(artifacts, _policy())
        [item] = report.protected
        assert item.protections == (RetentionProtection.PINNED,)
        assert "cannot report durable pins" in item.detail

    asyncio.run(run())


def test_owning_session_protects_until_it_is_deleted(tmp_path) -> None:
    async def scenario(artifacts, sessions) -> None:
        await create_retention_session(RetentionHarness(store=sessions), "owner")
        owned = await _put(artifacts, "owned.txt", session_id="owner")
        await asyncio.sleep(0.01)
        report = await apply_artifact_retention_policy(artifacts, _policy(), session_store=sessions)
        assert _reasons(report) == {owned.id: {RetentionProtection.SESSION_REFERENCE}}
        unchecked = await apply_artifact_retention_policy(artifacts, _policy())
        assert _reasons(unchecked) == {owned.id: {RetentionProtection.SESSION_REFERENCE}}
        await sessions.delete_session("owner")
        report = await apply_artifact_retention_policy(artifacts, _policy(), session_store=sessions)
        assert [item.item_id for item in report.items] == [owned.id]

    _run(tmp_path, scenario)


def test_transcript_references_are_reported_by_the_session_store(tmp_path) -> None:
    async def scenario(artifacts, sessions) -> None:
        referenced = await _put(artifacts, "attached.txt")
        await create_retention_session(RetentionHarness(store=sessions), "reader")
        await sessions.append_transcript_messages(
            "reader",
            [
                Message.tool_result(
                    tool_call_id="call-1",
                    tool_name="read_file",
                    content="see the attachment",
                    artifacts=[{"artifact_id": referenced.id}],
                )
            ],
        )
        assert referenced.id in await sessions.retention_artifact_references()

    _run(tmp_path, scenario)


@pytest.mark.parametrize(
    "protection",
    [
        RetentionProtection.EVAL_REFERENCE,
        RetentionProtection.KNOWLEDGE_EVIDENCE,
        RetentionProtection.SNAPSHOT_PIN,
        RetentionProtection.SESSION_REFERENCE,
    ],
)
def test_references_from_other_stores_protect(tmp_path, protection) -> None:
    async def scenario(artifacts, sessions) -> None:
        named = await _put(artifacts, "named.txt")
        await asyncio.sleep(0.01)
        report = await apply_artifact_retention_policy(
            artifacts,
            _policy(dry_run=False),
            session_store=sessions,
            references={protection: [named.id]},
            audit=sessions,
        )
        assert _reasons(report) == {named.id: {protection}}

    _run(tmp_path, scenario)


def test_caller_protection_and_store_size_target(tmp_path) -> None:
    async def scenario(artifacts, sessions) -> None:
        first = await _put(artifacts, "first.txt", 100)
        second = await _put(artifacts, "second.txt", 100)
        third = await _put(artifacts, "third.txt", 100)
        await asyncio.sleep(0.01)
        report = await apply_artifact_retention_policy(
            artifacts,
            _policy(target_total_bytes=250),
            session_store=sessions,
            protected_artifact_ids=[first.id],
        )
        assert _reasons(report) == {first.id: {RetentionProtection.CALLER_PROTECTED}}
        # Deleting the second artifact reaches the 250-byte target before the third.
        assert [item.item_id for item in report.items] == [second.id]
        assert report.deferred_count == 1
        assert third.id not in [item.item_id for item in report.items]

    _run(tmp_path, scenario)


def test_artifact_policy_contract(tmp_path) -> None:
    with pytest.raises(ValueError, match="only mode='delete'"):
        ArtifactRetentionPolicy(older_than=timedelta(days=1), mode=RetentionMode.COMPACT)
    with pytest.raises(ValueError, match="scopes"):
        ArtifactRetentionPolicy(older_than=timedelta(days=1), scopes={"bucket"})

    async def run() -> None:
        with pytest.raises(ValueError, match="audit sink"):
            await apply_artifact_retention_policy(
                LocalArtifactStore(tmp_path / "artifacts"),
                ArtifactRetentionPolicy(older_than=timedelta(days=1), dry_run=False),
            )

    asyncio.run(run())
