"""``find_knowledge``: a read-only lookup over kept entries and pending proposals."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence

import pytest

from cayu._knowledge_tool_supply import is_knowledge_tool
from cayu.knowledge.records import KnowledgeEntry, KnowledgeStatus, KnowledgeVisibility
from cayu.knowledge.scopes import KnowledgeAccessScope
from cayu.knowledge.similarity import (
    KnowledgeComparable,
    KnowledgeSimilarity,
    KnowledgeSimilarityScore,
)
from cayu.storage.knowledge_sqlite import SQLiteKnowledgeStore
from cayu.storage.memory import InMemoryKnowledgeStore
from cayu.tools import FindKnowledgeTool
from cayu.tools.base import ToolContext, ToolEffect
from cayu.tools.knowledge import RememberKnowledgePolicy
from cayu.vaults.redaction import SecretRedactor

_POLICY = RememberKnowledgePolicy(copy_session_labels=("module", "repository"))
_SESSION_LABELS = {"module": "tasks/worker.py", "repository": "example/app"}
_FACT = (
    "The task worker renews its lease only after re-reading the task row, so a reclaimed "
    "task never gets a heartbeat from its previous worker."
)
_PROPOSAL_BODY = (
    "Task worker lease renewal re-reads the task row first; a reclaimed task never "
    "receives a heartbeat from the previous worker process."
)
_DRAFT = {
    "title": "Worker lease renewal re-reads the task",
    "text": "The worker re-reads the task row before renewing its lease, so a reclaimed "
    "task gets no heartbeat from the previous worker.",
}


def _entry(entry_id: str, text: str, **fields) -> KnowledgeEntry:
    fields.setdefault("labels", {**_SESSION_LABELS, "tenant": "a"})
    return KnowledgeEntry(id=entry_id, text=text, **fields)


_ENTRIES = (
    _entry("kept", _FACT, title="Lease renewal re-reads the task row"),
    _entry(
        "proposal",
        _PROPOSAL_BODY,
        title="Worker re-reads the task before lease renewal",
        status=KnowledgeStatus.PENDING,
        metadata={"cayu:corroboration": {"source_count": 3, "sources": []}},
    ),
    _entry("unrelated", "Branch storage keeps directory snapshots immutable across rollback."),
    _entry("archived", _FACT + " Archived.", status=KnowledgeStatus.ARCHIVED),
    _entry(
        "other-module",
        _FACT,
        labels={**_SESSION_LABELS, "module": "tasks/dispatch.py", "tenant": "a"},
    ),
    _entry("no-module", _FACT, labels={"repository": "example/app", "tenant": "a"}),
    _entry("other-namespace", _FACT, namespace="elsewhere"),
    _entry("user-visible", _FACT, visibility=KnowledgeVisibility.USER),
    _entry("other-tenant", _FACT, labels={**_SESSION_LABELS, "tenant": "b"}),
)

_SCOPE = KnowledgeAccessScope(
    allowed_namespaces=["default", "elsewhere"],
    allowed_visibilities=[KnowledgeVisibility.GLOBAL, KnowledgeVisibility.USER],
    allowed_statuses=[KnowledgeStatus.ACTIVE, KnowledgeStatus.PENDING, KnowledgeStatus.ARCHIVED],
    required_labels={"tenant": "a"},
)


_SCOPE_ALL = KnowledgeAccessScope(allowed_namespaces=["default"])


def _context(store, scope: KnowledgeAccessScope = _SCOPE, **fields) -> ToolContext:
    return ToolContext(
        session_id="hunter-2",
        session_labels=_SESSION_LABELS,
        knowledge_store=store,
        knowledge_access_scope=scope,
        **fields,
    )


async def _seed(store) -> None:
    privileged = KnowledgeAccessScope.privileged()
    for entry in _ENTRIES:
        await store.create_entry(entry, access_scope=privileged)


def _open_store(backend: str, tmp_path, request):
    if backend == "memory":
        return InMemoryKnowledgeStore()
    if backend == "sqlite":
        return SQLiteKnowledgeStore(tmp_path / "knowledge.db")
    from cayu.storage.knowledge_postgres import PostgresKnowledgeStore
    from cayu.storage.migrations import SchemaMode

    return PostgresKnowledgeStore(
        request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_find_knowledge_matches_kept_and_pending_entries_in_the_session_scope(
    backend, tmp_path, request
):
    store = _open_store(backend, tmp_path, request)

    async def run():
        try:
            await _seed(store)
            tool = FindKnowledgeTool(policy=_POLICY)
            found = await tool.run(_context(store), dict(_DRAFT))
            # Another tenant's scope reaches only its own entry, never tenant `a`'s.
            tenant_b = KnowledgeAccessScope(
                allowed_namespaces=["default"], required_labels={"tenant": "b"}
            )
            tenant_found = await tool.run(_context(store, tenant_b), dict(_DRAFT))
            # A scope that can't read proposals finds only kept entries.
            kept_only = KnowledgeAccessScope(
                allowed_namespaces=["default"],
                allowed_statuses=[KnowledgeStatus.ACTIVE],
                required_labels={"tenant": "a"},
            )
            kept_found = await tool.run(_context(store, kept_only), dict(_DRAFT))
            return found, tenant_found, kept_found
        finally:
            close = getattr(store, "close", None)
            if close is not None:
                await close()

    found, tenant_found, kept_found = asyncio.run(run())
    assert not found.is_error, found
    rows = found.structured["matches"]
    assert {row["entry_id"] for row in rows} == {"kept", "proposal"}
    by_id = {row["entry_id"]: row for row in rows}
    assert by_id["kept"] == {
        "entry_id": "kept",
        "title": "Lease renewal re-reads the task row",
        "status": "kept",
        "kind": "fact",
        "sources": 1,
        "score": by_id["kept"]["score"],
        "read_with": "read_knowledge",
    }
    assert by_id["proposal"]["status"] == "unreviewed proposal"
    assert by_id["proposal"]["sources"] == 3
    assert "read_with" not in by_id["proposal"]
    # The proposal's title is shown, never its body.
    rendered = json.dumps(found.model_dump(mode="json"))
    assert "Worker re-reads the task before lease renewal" in rendered
    assert _PROPOSAL_BODY not in rendered and "previous worker process" not in rendered
    assert _FACT not in rendered
    assert found.structured["candidates_truncated"] is False
    assert "proposal (unreviewed proposal, fact, 3 sources" in found.content

    assert [row["entry_id"] for row in tenant_found.structured["matches"]] == ["other-tenant"]
    assert [row["entry_id"] for row in kept_found.structured["matches"]] == ["kept"]


def test_find_knowledge_is_a_read_with_no_effect():
    tool = FindKnowledgeTool()
    assert tool.spec.name == "find_knowledge"
    assert tool.spec.effect is ToolEffect.NONE
    assert tool.spec.parallel_safe
    assert tool._publish_arguments is False
    assert tool.retain_arguments_for_model is True
    assert is_knowledge_tool(tool)
    assert tool._execution_profile_material() is not None
    assert FindKnowledgeTool(policy=_POLICY)._execution_profile_material() is None


def test_find_knowledge_without_a_store_reports_it():
    result = asyncio.run(FindKnowledgeTool().run(ToolContext(session_id="s"), {"text": "x"}))
    assert result.is_error and result.structured == {"error": "missing_knowledge_store"}


def test_find_knowledge_filters_by_kind_and_bounds_its_limit():
    store = InMemoryKnowledgeStore()

    async def run():
        await _seed(store)
        await store.create_entry(
            _entry("warning", _FACT, kind="warning"), access_scope=KnowledgeAccessScope.privileged()
        )
        tool = FindKnowledgeTool(policy=_POLICY)
        warnings = await tool.run(_context(store), {**_DRAFT, "kind": "warning"})
        one = await tool.run(_context(store), {**_DRAFT, "limit": 1})
        too_many = await tool.run(_context(store), {**_DRAFT, "limit": 11})
        return warnings, one, too_many

    warnings, one, too_many = asyncio.run(run())
    assert [row["entry_id"] for row in warnings.structured["matches"]] == ["warning"]
    assert len(one.structured["matches"]) == 1
    assert too_many.is_error


def test_find_knowledge_compares_at_most_max_candidates():
    store = InMemoryKnowledgeStore()

    async def run():
        privileged = KnowledgeAccessScope.privileged()
        for index in range(7):
            await store.create_entry(_entry(f"copy-{index}", _FACT), access_scope=privileged)
        return await FindKnowledgeTool(policy=_POLICY, max_candidates=3).run(
            _context(store), dict(_DRAFT)
        )

    result = asyncio.run(run())
    assert result.structured["compared"] == 3
    assert result.structured["candidates_truncated"] is True
    assert "Only the first 3 entries" in result.content


def test_find_knowledge_redacts_titles():
    secret = "title-secret-canary-ABCDEFGH"
    store = InMemoryKnowledgeStore()

    async def run():
        await store.create_entry(
            _entry("kept", _FACT, title=f"Lease renewal {secret}"),
            access_scope=KnowledgeAccessScope.privileged(),
        )
        return await FindKnowledgeTool(policy=_POLICY).run(
            _context(store, invocation_secret_redactor=lambda: SecretRedactor(secret)),
            dict(_DRAFT),
        )

    result = asyncio.run(run())
    assert result.structured["matches"]
    assert secret not in json.dumps(result.model_dump(mode="json"))


class _Recording(KnowledgeSimilarity):
    def __init__(self) -> None:
        super().__init__()
        self.subjects: list[KnowledgeComparable] = []

    async def score(
        self, subject: KnowledgeComparable, candidates: Sequence[KnowledgeComparable]
    ) -> list[KnowledgeSimilarityScore]:
        self.subjects.append(subject)
        return [
            KnowledgeSimilarityScore(score=1.0, similar="Branch" in candidate.text)
            for candidate in candidates
        ]


def test_find_knowledge_uses_the_similarity_part_it_is_given():
    store = InMemoryKnowledgeStore()
    part = _Recording()

    async def run():
        await _seed(store)
        return await FindKnowledgeTool(policy=_POLICY, similarity=part).run(
            _context(store), dict(_DRAFT)
        )

    result = asyncio.run(run())
    assert [row["entry_id"] for row in result.structured["matches"]] == ["unrelated"]
    assert dict(part.subjects[0].labels) == _SESSION_LABELS
    with pytest.raises(TypeError, match="KnowledgeSimilarity"):
        FindKnowledgeTool(similarity=object())  # type: ignore[arg-type]


def test_entries_outside_the_similarity_scope_do_not_use_up_the_cap():
    store = InMemoryKnowledgeStore()
    policy = RememberKnowledgePolicy(copy_session_labels=("module",))

    async def run():
        privileged = KnowledgeAccessScope.privileged()
        for index in range(5):
            await store.create_entry(
                _entry(f"other-{index}", _FACT, labels={"module": f"m{index}.py"}, importance=1.0),
                access_scope=privileged,
            )
        await store.create_entry(_entry("unlabelled", _FACT, labels={}), access_scope=privileged)
        # The session has no module label, so only the unlabelled note is in its scope.
        return await FindKnowledgeTool(policy=policy, max_candidates=5).run(
            ToolContext(session_id="s", knowledge_store=store, knowledge_access_scope=_SCOPE_ALL),
            dict(_DRAFT),
        )

    result = asyncio.run(run())
    assert [row["entry_id"] for row in result.structured["matches"]] == ["unlabelled"]
    assert result.structured["compared"] == 1
    assert result.structured["candidates_truncated"] is False
