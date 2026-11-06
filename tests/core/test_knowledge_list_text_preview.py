"""A knowledge listing's `text_preview` is an excerpt of the entry's text, not its title."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from cayu.knowledge._retrieval_results import (
    KNOWLEDGE_LIST_EXCERPT_CHARS,
    _knowledge_list_preview,
    _knowledge_text_excerpt,
)
from cayu.knowledge.records import KnowledgeEntry
from cayu.knowledge.scopes import KnowledgeAccessScope
from cayu.knowledge.search import KnowledgeListQuery
from cayu.storage import migrations as schema_migrations
from cayu.storage.knowledge_indexer import KnowledgeIndexer, KnowledgeIndexRequest
from cayu.storage.knowledge_memory import InMemoryKnowledgeStore
from cayu.tools.base import ToolContext
from cayu.tools.knowledge import ListKnowledgeTool
from cayu.vaults.redaction import SecretRedactor

_ACCESS_SCOPE = KnowledgeAccessScope.privileged()
_LONG_TEXT = (
    "Retry failed invoice exports three times with a five minute backoff, then page the "
    "billing on-call. Exports that fail on a closed accounting period must not be retried "
    "at all: they need the controller to reopen the period first, which the runbook "
    "describes in its second section."
)
_SHORT_TEXT = "Retry failed payroll exports once."


def test_excerpt_keeps_short_text_whole() -> None:
    assert _knowledge_text_excerpt("  A short note.\n") == "A short note."


def test_excerpt_cuts_long_text_on_a_word_boundary_with_an_ellipsis() -> None:
    excerpt = _knowledge_text_excerpt(_LONG_TEXT)

    assert len(_LONG_TEXT) > KNOWLEDGE_LIST_EXCERPT_CHARS
    assert len(excerpt) <= KNOWLEDGE_LIST_EXCERPT_CHARS
    assert excerpt.endswith("…")
    head = excerpt.removesuffix("…")
    assert _LONG_TEXT.startswith(head)
    # The cut falls between words, not inside one.
    assert _LONG_TEXT[len(head)].isspace()


def test_excerpt_cuts_inside_a_word_too_long_to_break_elsewhere() -> None:
    excerpt = _knowledge_text_excerpt("x" * 500)

    assert excerpt == "x" * (KNOWLEDGE_LIST_EXCERPT_CHARS - 1) + "…"


def test_excerpt_keeps_a_word_that_ends_exactly_at_the_cut() -> None:
    text = "a" * 9 + " tail end"

    assert _knowledge_text_excerpt(text, 10) == "a" * 9 + "…"


def test_list_preview_reports_completeness_and_budget_cuts_separately() -> None:
    assert _knowledge_list_preview(_SHORT_TEXT, 10_000) == (_SHORT_TEXT, True, False)
    preview, complete, budget_cut = _knowledge_list_preview(_LONG_TEXT, 10_000)
    assert preview == _knowledge_text_excerpt(_LONG_TEXT, ellipsis=False)
    assert (complete, budget_cut) == (False, False)
    assert _knowledge_list_preview(_SHORT_TEXT, 5) == ("Retry", False, True)


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_store_list_previews_the_text_of_entries_with_similar_titles(
    backend: str, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    async def run() -> None:
        store = _store(backend, tmp_path, dsn)
        try:
            await store.create_entry(
                KnowledgeEntry(
                    id="invoice-retries",
                    title="Retry policy",
                    text=_LONG_TEXT,
                    importance=0.9,
                )
            )
            await store.create_entry(
                KnowledgeEntry(
                    id="payroll-retries",
                    title="Retry policy",
                    text=_SHORT_TEXT,
                    importance=0.5,
                )
            )
            listed = await store.list_entries(KnowledgeListQuery(limit=10, max_bytes=10_000))
            budgeted = await store.list_entries(KnowledgeListQuery(limit=10, max_bytes=10))
        finally:
            close = getattr(store, "close", None)
            if close is not None:
                await close()

        long_item, short_item = listed.entries
        assert long_item.entry.id == "invoice-retries"
        # Unredacted store previews carry no ellipsis; the completeness bit marks the cut.
        assert long_item.text_preview == _knowledge_text_excerpt(_LONG_TEXT, ellipsis=False)
        assert _LONG_TEXT.startswith(long_item.text_preview)
        assert long_item.text_preview.startswith("Retry failed invoice exports")
        assert long_item.text_preview_complete is False
        assert short_item.text_preview == _SHORT_TEXT
        assert short_item.text_preview_complete is True
        # An excerpt is the listing's shape, not a cut the caller must page past.
        assert listed.truncated is False

        [budget_item] = budgeted.entries
        assert budget_item.text_preview == "Retry fail"
        assert budget_item.text_preview_complete is False
        assert budgeted.truncated is True

    asyncio.run(run())


def test_list_knowledge_withholds_a_secret_the_excerpt_cut_splits() -> None:
    secret = "sk_live_Q7mZ4pR9tV2xW8yB3nC6dF1gH5jK0lM4sT6uV8wX0yZ2aB4cD6eF8gH0iJ2kL4mN6"
    # No whitespace after character 100, so the 200-character cut lands inside the secret.
    text = "config=" + "a" * 150 + secret + " tail"

    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        await KnowledgeIndexer(store).index_text(KnowledgeIndexRequest(entry_id="cfg", text=text))
        return await ListKnowledgeTool().run(
            ToolContext(
                session_id="session_1",
                knowledge_store=store,
                invocation_secret_redactor=lambda: SecretRedactor(secret),
            ),
            {"preview_bytes": 4096, "max_bytes": 10_000},
        )

    result = asyncio.run(run())

    rendered = repr(result.model_dump(mode="json"))
    assert secret[:16] not in rendered
    assert result.structured["entries"][0]["text_preview_truncated"] is True


def _store(backend: str, tmp_path: Path, dsn: str | None):
    if backend == "memory":
        from cayu.storage.knowledge_memory import InMemoryKnowledgeStore

        return InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
    if backend == "sqlite":
        from cayu.storage.knowledge_sqlite import SQLiteKnowledgeStore

        return SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite", access_scope=_ACCESS_SCOPE)
    from cayu.storage.knowledge_postgres import PostgresKnowledgeStore

    assert dsn is not None
    return PostgresKnowledgeStore(
        dsn,
        access_scope=_ACCESS_SCOPE,
        min_size=1,
        max_size=2,
        schema_mode=schema_migrations.SchemaMode.CREATE,
    )
