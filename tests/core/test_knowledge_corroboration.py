"""`remember_knowledge(same_as=...)` records the calling session as another source."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from cayu.cli import main
from cayu.knowledge.corroboration import (
    KNOWLEDGE_CORROBORATION_METADATA_KEY,
    MAX_KNOWLEDGE_CORROBORATIONS,
    KnowledgeCorroboration,
    KnowledgeCorroborationRefused,
    corroborated_knowledge_revision,
    knowledge_corroborations,
    knowledge_source_count,
)
from cayu.knowledge.records import (
    KnowledgeEntry,
    KnowledgeEvidenceRole,
    KnowledgeStatus,
)
from cayu.knowledge.scopes import KnowledgeAccessScope
from cayu.knowledge.search import KnowledgeListQuery
from cayu.storage import migrations as schema_migrations
from cayu.storage.knowledge_memory import InMemoryKnowledgeStore
from cayu.storage.knowledge_review import KnowledgeReviewWorkflow
from cayu.tools.base import ToolContext
from cayu.tools.knowledge import RememberKnowledgePolicy, RememberKnowledgeTool

_SCOPE = KnowledgeAccessScope.privileged()
_FACT = "Pull request work runs under one repository lock."
_BACKENDS = ["memory", "sqlite", "postgres"]


def _store(backend: str, tmp_path: Path, request: pytest.FixtureRequest):
    if backend == "memory":
        return InMemoryKnowledgeStore(access_scope=_SCOPE)
    if backend == "sqlite":
        from cayu.storage.knowledge_sqlite import SQLiteKnowledgeStore

        return SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite", access_scope=_SCOPE)
    from cayu.storage.knowledge_postgres import PostgresKnowledgeStore

    return PostgresKnowledgeStore(
        request.getfixturevalue("postgres_test_dsn"),
        access_scope=_SCOPE,
        min_size=1,
        max_size=2,
        schema_mode=schema_migrations.SchemaMode.CREATE,
    )


async def _close(store: Any) -> None:
    close = getattr(store, "close", None)
    if close is not None:
        await close()


def _ctx(store: Any, session_id: str, **kwargs: Any) -> ToolContext:
    return ToolContext(
        session_id=session_id, agent_name="assistant", knowledge_store=store, **kwargs
    )


async def _propose(store: Any, session_id: str = "session_a", **kwargs: Any) -> str:
    result = await RememberKnowledgeTool(**kwargs).run(_ctx(store, session_id), {"text": _FACT})
    assert result.is_error is False, result.structured
    return result.structured["entry"]["entry_id"]


async def _entry_count(store: Any) -> int:
    page = await store.list_entries(
        KnowledgeListQuery(
            statuses=[KnowledgeStatus.PENDING, KnowledgeStatus.ACTIVE], limit=25, max_bytes=50_000
        )
    )
    return len(page.entries)


async def _keep(store: Any, entry_id: str) -> None:
    await KnowledgeReviewWorkflow(store, access_scope=_SCOPE).approve(
        entry_id,
        operation_id=f"keep-{entry_id}",
        reviewer_identity="reviewer",
        reviewer_version="1",
    )


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", _BACKENDS)
def test_corroborating_a_pending_proposal_records_the_session_as_a_source(
    backend: str, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    async def run():
        store = _store(backend, tmp_path, request)
        try:
            entry_id = await _propose(store)
            result = await RememberKnowledgeTool().run(
                _ctx(store, "session_b"),
                {
                    "text": "One lock guards pull request work.",
                    "same_as": entry_id,
                    "reason": "Saw the same lock in the PR worker.",
                },
            )
            entry = await store.get_entry(entry_id)
            evidence = await store.read_evidence(entry_id, max_records=10)
            count = await _entry_count(store)
            # Keeping it carries the sources with it.
            await _keep(store, entry_id)
            kept = await store.get_entry(entry_id)
            kept_evidence = await store.read_evidence(entry_id, max_records=10)
        finally:
            await _close(store)
        return entry_id, result, entry, evidence, count, kept, kept_evidence

    entry_id, result, entry, evidence, count, kept, kept_evidence = asyncio.run(run())

    assert result.is_error is False, result.structured
    assert result.structured["corroborated"] is True
    assert result.structured["already_recorded"] is False
    assert result.structured["source_count"] == 2
    assert result.structured["entry"] == {
        "entry_id": entry_id,
        "revision": 2,
        "status": "pending",
    }
    # No new entry, and the corroborating text is not merged in.
    assert count == 1
    assert entry.text == _FACT
    assert entry.revision == 2
    assert entry.status is KnowledgeStatus.PENDING
    assert entry.source_id == "session_a"
    assert knowledge_source_count(entry) == 2
    [source] = knowledge_corroborations(entry)
    assert source.session_id == "session_b"
    assert source.reason == "Saw the same lock in the PR worker."
    assert entry.metadata[KNOWLEDGE_CORROBORATION_METADATA_KEY]["source_count"] == 2
    [supporting] = evidence.evidence
    assert supporting.role is KnowledgeEvidenceRole.SUPPORTING
    assert supporting.source_id == "session_b"
    assert supporting.source_uri == "cayu://sessions/session_b"
    assert supporting.entry_revision == 2
    assert supporting.metadata["reason"] == "Saw the same lock in the PR worker."

    assert kept.status is KnowledgeStatus.ACTIVE
    assert knowledge_source_count(kept) == 2
    assert [item.source_id for item in kept_evidence.evidence] == ["session_b"]


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", _BACKENDS)
def test_corroborating_a_kept_entry_needs_the_policy_to_allow_it(
    backend: str, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    async def run():
        store = _store(backend, tmp_path, request)
        try:
            entry_id = await _propose(store)
            await _keep(store, entry_id)
            refused = await RememberKnowledgeTool().run(
                _ctx(store, "session_b"), {"text": _FACT, "same_as": entry_id}
            )
            unchanged = await store.get_entry(entry_id)
            allowed = await RememberKnowledgeTool(
                policy=RememberKnowledgePolicy(corroborate_kept=True)
            ).run(_ctx(store, "session_b"), {"text": _FACT, "same_as": entry_id})
            entry = await store.get_entry(entry_id)
        finally:
            await _close(store)
        return refused, unchanged, allowed, entry

    refused, unchanged, allowed, entry = asyncio.run(run())

    assert refused.is_error is True
    assert refused.structured["error"] == "corroboration_refused"
    assert refused.structured["reason"] == "target_not_pending"
    assert unchanged.revision == 2
    assert allowed.is_error is False, allowed.structured
    assert allowed.structured["source_count"] == 2
    assert entry.status is KnowledgeStatus.ACTIVE
    assert entry.revision == 3
    assert entry.text == _FACT
    assert knowledge_source_count(entry) == 2


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", _BACKENDS)
def test_a_retry_or_repeat_never_counts_a_session_twice(
    backend: str, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    async def run():
        store = _store(backend, tmp_path, request)
        try:
            entry_id = await _propose(store)
            tool = RememberKnowledgeTool()
            args = {"text": _FACT, "same_as": entry_id}
            first = await tool.run(_ctx(store, "session_b", idempotency_key="call-1"), args)
            # The same call delivered again, and a later call with another key.
            replay = await tool.run(_ctx(store, "session_b", idempotency_key="call-1"), args)
            again = await tool.run(_ctx(store, "session_b", idempotency_key="call-2"), args)
            # The session that wrote the entry is already one of its sources.
            origin = await tool.run(_ctx(store, "session_a"), args)
            entry = await store.get_entry(entry_id)
        finally:
            await _close(store)
        return first, replay, again, origin, entry

    first, replay, again, origin, entry = asyncio.run(run())

    assert first.structured["written"] is True
    for repeated in (replay, again, origin):
        assert repeated.is_error is False, repeated.structured
        assert repeated.structured["already_recorded"] is True
        assert repeated.structured["written"] is False
        assert repeated.structured["source_count"] == 2
    assert entry.revision == 2
    assert knowledge_source_count(entry) == 2


def test_an_entry_whose_text_has_trailing_whitespace_can_be_corroborated() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_SCOPE)
        first = await RememberKnowledgeTool().run(_ctx(store, "session_a"), {"text": _FACT + "\n"})
        entry_id = first.structured["entry"]["entry_id"]
        return await RememberKnowledgeTool().run(
            _ctx(store, "session_b"), {"text": _FACT, "same_as": entry_id}
        )

    result = asyncio.run(run())

    assert result.is_error is False, result.structured
    assert result.structured["source_count"] == 2


def test_a_long_entry_keeps_every_chunk() -> None:
    text = " ".join(
        f"Step {index} of the release checklist is done by hand." for index in range(300)
    )

    async def run():
        store = InMemoryKnowledgeStore(access_scope=_SCOPE)
        first = await RememberKnowledgeTool().run(_ctx(store, "session_a"), {"text": text})
        entry_id = first.structured["entry"]["entry_id"]
        before = await store.read_chunks(entry_id, max_chunks=100, max_bytes=100_000)
        result = await RememberKnowledgeTool().run(
            _ctx(store, "session_b"), {"text": "Same checklist.", "same_as": entry_id}
        )
        after = await store.read_chunks(entry_id, max_chunks=100, max_bytes=100_000)
        return before, result, after

    before, result, after = asyncio.run(run())

    assert len(before) > 1
    assert result.structured["source_count"] == 2
    assert [chunk.text for chunk in after] == [chunk.text for chunk in before]
    assert {chunk.entry_revision for chunk in after} == {2}


class _LostAcknowledgementStore(InMemoryKnowledgeStore):
    """Commits a publication, then loses its acknowledgement once."""

    def __init__(self) -> None:
        super().__init__(access_scope=_SCOPE)
        self.lose_next = False

    async def publish_entry_revision(self, entry, chunks, **kwargs):
        receipt = await super().publish_entry_revision(entry, chunks, **kwargs)
        if self.lose_next:
            self.lose_next = False
            raise RuntimeError("connection reset after commit")
        return receipt


def test_a_lost_acknowledgement_does_not_double_count() -> None:
    async def run():
        store = _LostAcknowledgementStore()
        entry_id = await _propose(store)
        store.lose_next = True
        tool = RememberKnowledgeTool()
        args = {"text": _FACT, "same_as": entry_id}
        first = await tool.run(_ctx(store, "session_b"), args)
        retry = await tool.run(_ctx(store, "session_b"), args)
        return first, retry, await store.get_entry(entry_id)

    first, retry, entry = asyncio.run(run())

    assert first.is_error is False, first.structured
    assert first.structured["post_write_error"] == "publication_acknowledgement_lost"
    assert first.structured["source_count"] == 2
    assert retry.structured["already_recorded"] is True
    assert entry.revision == 2
    assert knowledge_source_count(entry) == 2


class _RacingStore(InMemoryKnowledgeStore):
    """Lets another session corroborate the entry just before the first publication."""

    def __init__(self) -> None:
        super().__init__(access_scope=_SCOPE)
        self.race: Any = None

    async def publish_entry_revision(self, entry, chunks, **kwargs):
        race, self.race = self.race, None
        if race is not None:
            await race()
        return await super().publish_entry_revision(entry, chunks, **kwargs)


def test_a_concurrent_corroboration_makes_the_loser_reread_and_retry() -> None:
    async def run():
        store = _RacingStore()
        entry_id = await _propose(store)
        args = {"text": _FACT, "same_as": entry_id}

        async def race() -> None:
            winner = await RememberKnowledgeTool().run(_ctx(store, "session_c"), args)
            assert winner.structured["source_count"] == 2

        store.race = race
        result = await RememberKnowledgeTool().run(_ctx(store, "session_b"), args)
        return result, await store.get_entry(entry_id)

    result, entry = asyncio.run(run())

    assert result.is_error is False, result.structured
    assert result.structured["source_count"] == 3
    assert entry.revision == 3
    assert [item.session_id for item in knowledge_corroborations(entry)] == [
        "session_c",
        "session_b",
    ]


def test_a_target_outside_the_callers_scope_is_refused(tmp_path: Path) -> None:
    from cayu.storage.knowledge_sqlite import SQLiteKnowledgeStore

    path = tmp_path / "knowledge.sqlite"

    async def run():
        store = SQLiteKnowledgeStore(path, access_scope=_SCOPE)
        # The same database, seen by a session allowed only the default namespace.
        narrow = SQLiteKnowledgeStore(
            path,
            access_scope=KnowledgeAccessScope(
                allowed_namespaces=["default"],
                allowed_statuses=[KnowledgeStatus.ACTIVE, KnowledgeStatus.PENDING],
            ),
        )
        try:
            await store.create_entry(
                KnowledgeEntry(
                    id="elsewhere",
                    namespace="other",
                    text=_FACT,
                    status=KnowledgeStatus.PENDING,
                )
            )
            module_a = RememberKnowledgePolicy(copy_session_labels=("module",))
            entry_id = await _propose_labelled(store, module_a, "a")
            hidden = await RememberKnowledgeTool().run(
                _ctx(narrow, "session_b"), {"text": _FACT, "same_as": "elsewhere"}
            )
            other_module = await RememberKnowledgeTool(policy=module_a).run(
                _ctx(store, "session_b", session_labels={"module": "b"}),
                {"text": _FACT, "same_as": entry_id},
            )
            missing = await RememberKnowledgeTool().run(
                _ctx(store, "session_b"), {"text": _FACT, "same_as": "no-such-entry"}
            )
            return (
                hidden,
                other_module,
                missing,
                await store.get_entry(entry_id),
                await store.get_entry("elsewhere"),
            )
        finally:
            await narrow.close()
            await store.close()

    hidden, other_module, missing, entry, elsewhere = asyncio.run(run())

    assert hidden.is_error is True
    assert hidden.structured["reason"] == "target_not_found"
    assert other_module.is_error is True
    assert other_module.structured["reason"] == "target_out_of_scope"
    assert missing.structured["reason"] == "target_not_found"
    assert "Nothing was written" in missing.content
    assert entry.revision == 1
    assert elsewhere.revision == 1


async def _propose_labelled(store: Any, policy: RememberKnowledgePolicy, module: str) -> str:
    result = await RememberKnowledgeTool(policy=policy).run(
        _ctx(store, "session_a", session_labels={"module": module}), {"text": _FACT}
    )
    assert result.is_error is False, result.structured
    return result.structured["entry"]["entry_id"]


def test_archived_targets_are_refused_and_reason_needs_same_as() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_SCOPE)
        entry_id = await _propose(store)
        await KnowledgeReviewWorkflow(store, access_scope=_SCOPE).reject_reviewed(
            entry_id,
            expected_revision=1,
            operation_id="drop-it",
            reviewer_identity="reviewer",
            reviewer_version="1",
            reason="Duplicate.",
        )
        archived = await RememberKnowledgeTool().run(
            _ctx(store, "session_b"), {"text": _FACT, "same_as": entry_id}
        )
        lone_reason = await RememberKnowledgeTool().run(
            _ctx(store, "session_b"), {"text": "Another fact.", "reason": "Because."}
        )
        return archived, lone_reason, await _entry_count(store)

    archived, lone_reason, count = asyncio.run(run())

    assert archived.structured["reason"] == "target_not_live"
    assert lone_reason.is_error is True
    assert "`reason` requires `same_as`" in lone_reason.content
    assert count == 0


def test_cayu_knowledge_show_prints_the_source_count_and_sources(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "knowledge.sqlite"

    async def seed() -> str:
        from cayu.storage.knowledge_sqlite import SQLiteKnowledgeStore

        store = SQLiteKnowledgeStore(path, access_scope=_SCOPE)
        try:
            entry_id = await _propose(store)
            for session_id in ("session_b", "session_c"):
                result = await RememberKnowledgeTool().run(
                    _ctx(store, session_id),
                    {"text": _FACT, "same_as": entry_id, "reason": f"Seen in {session_id}."},
                )
                assert result.is_error is False, result.structured
        finally:
            await store.close()
        return entry_id

    entry_id = asyncio.run(seed())

    assert main(["knowledge", "show", entry_id, "--sqlite", str(path), "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)["entry"]
    assert shown["source_count"] == 3
    assert [source["session_id"] for source in shown["corroborations"]] == [
        "session_b",
        "session_c",
    ]
    assert shown["corroborations"][0]["reason"] == "Seen in session_b."

    assert main(["knowledge", "show", entry_id, "--sqlite", str(path)]) == 0
    table = capsys.readouterr().out
    assert "source_count" in table and "3" in table
    assert "corroborated_by" in table and "Seen in session_c." in table


def test_source_count_ignores_a_summary_cayu_did_not_write_and_stops_at_the_limit() -> None:
    entry = KnowledgeEntry(
        id="noted",
        text=_FACT,
        metadata={KNOWLEDGE_CORROBORATION_METADATA_KEY: {"source_count": 9}},
    )
    assert knowledge_source_count(entry) == 1
    corroboration = KnowledgeCorroboration(session_id="s", recorded_at=datetime.now(UTC))
    with pytest.raises(KnowledgeCorroborationRefused) as refused:
        corroborated_knowledge_revision(entry, [], [], corroboration=corroboration)
    assert refused.value.code == "malformed_sources"

    full = KnowledgeEntry(id="full", text=_FACT)
    for index in range(MAX_KNOWLEDGE_CORROBORATIONS):
        full, _, _ = corroborated_knowledge_revision(
            full,
            [],
            [],
            corroboration=KnowledgeCorroboration(
                session_id=f"session_{index}", recorded_at=datetime.now(UTC)
            ),
        )
    assert knowledge_source_count(full) == MAX_KNOWLEDGE_CORROBORATIONS + 1
    with pytest.raises(KnowledgeCorroborationRefused) as refused:
        corroborated_knowledge_revision(full, [], [], corroboration=corroboration)
    assert refused.value.code == "source_limit_reached"
