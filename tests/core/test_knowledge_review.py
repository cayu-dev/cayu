from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from cayu.storage import (
    InMemoryKnowledgeStore,
    KnowledgeEntry,
    KnowledgeListQuery,
    KnowledgeQuery,
    KnowledgeReviewWorkflow,
    KnowledgeStatus,
)


class ScopeDriftKnowledgeStore(InMemoryKnowledgeStore):
    async def get_entry(self, entry_id: str) -> KnowledgeEntry | None:
        entry = await super().get_entry(entry_id)
        if entry is not None and entry.id == "pending_git":
            await self.put_entry(entry.model_copy(update={"labels": {"project": "other"}}))
        return entry


def test_review_workflow_lists_pending_entries_in_scope() -> None:
    async def run():
        store = InMemoryKnowledgeStore(
            [
                KnowledgeEntry(
                    id="pending_git",
                    text="Remote sandbox Git pushes should use a brokered credential proxy.",
                    namespace="project:cayu",
                    labels={"project": "cayu", "tenant": "trusted"},
                    kind="procedure",
                    status=KnowledgeStatus.PENDING,
                    aspects=["git"],
                ),
                KnowledgeEntry(
                    id="pending_other_tenant",
                    text="This pending entry belongs to another tenant.",
                    namespace="project:cayu",
                    labels={"project": "cayu", "tenant": "other"},
                    status=KnowledgeStatus.PENDING,
                ),
                KnowledgeEntry(
                    id="active_git",
                    text="Active entries should not appear in pending review.",
                    namespace="project:cayu",
                    labels={"project": "cayu", "tenant": "trusted"},
                    status=KnowledgeStatus.ACTIVE,
                ),
            ]
        )
        workflow = KnowledgeReviewWorkflow(
            store,
            namespace="project:cayu",
            labels={"project": "cayu", "tenant": "trusted"},
        )
        return await workflow.list_pending(aspects=["git"], limit=10)

    result = asyncio.run(run())

    assert [item.entry.id for item in result.entries] == ["pending_git"]
    assert result.query.statuses == [KnowledgeStatus.PENDING]
    assert result.query.namespace == "project:cayu"
    assert result.query.labels == {"project": "cayu", "tenant": "trusted"}


def test_review_workflow_approves_pending_entry() -> None:
    async def run():
        store = InMemoryKnowledgeStore(
            [
                KnowledgeEntry(
                    id="pending_git",
                    text="Remote sandbox Git pushes should use a brokered credential proxy.",
                    namespace="project:cayu",
                    labels={"project": "cayu"},
                    status=KnowledgeStatus.PENDING,
                )
            ]
        )
        workflow = KnowledgeReviewWorkflow(store, namespace="project:cayu")
        approved = await workflow.approve("pending_git")
        default_search = await store.search(
            KnowledgeQuery(text="brokered credential proxy", namespace="project:cayu")
        )
        return approved, default_search

    approved, default_search = asyncio.run(run())

    assert approved.status is KnowledgeStatus.ACTIVE
    assert [hit.entry.id for hit in default_search.hits] == ["pending_git"]


def test_review_workflow_rejects_pending_entry_as_archived() -> None:
    async def run():
        store = InMemoryKnowledgeStore(
            [
                KnowledgeEntry(
                    id="pending_bad",
                    text="This proposed fact should not be recalled.",
                    namespace="project:cayu",
                    status=KnowledgeStatus.PENDING,
                )
            ]
        )
        workflow = KnowledgeReviewWorkflow(store, namespace="project:cayu")
        rejected = await workflow.reject("pending_bad")
        active_search = await store.search(
            KnowledgeQuery(text="proposed fact", namespace="project:cayu")
        )
        archived_list = await store.list_entries(
            KnowledgeListQuery(
                namespace="project:cayu",
                statuses=[KnowledgeStatus.ARCHIVED],
            )
        )
        return rejected, active_search, archived_list

    rejected, active_search, archived_list = asyncio.run(run())

    assert rejected.status is KnowledgeStatus.ARCHIVED
    assert active_search.hits == []
    assert [item.entry.id for item in archived_list.entries] == ["pending_bad"]


def test_review_workflow_refuses_non_pending_entries() -> None:
    async def run():
        store = InMemoryKnowledgeStore(
            [
                KnowledgeEntry(
                    id="active_git",
                    text="Already active knowledge.",
                    namespace="project:cayu",
                    status=KnowledgeStatus.ACTIVE,
                )
            ]
        )
        workflow = KnowledgeReviewWorkflow(store, namespace="project:cayu")
        await workflow.approve("active_git")

    with pytest.raises(ValueError, match="not 'pending'"):
        asyncio.run(run())


def test_review_workflow_refuses_entries_outside_scope() -> None:
    async def run():
        store = InMemoryKnowledgeStore(
            [
                KnowledgeEntry(
                    id="pending_other",
                    text="Other project knowledge.",
                    namespace="project:other",
                    labels={"project": "other"},
                    status=KnowledgeStatus.PENDING,
                )
            ]
        )
        workflow = KnowledgeReviewWorkflow(
            store,
            namespace="project:cayu",
            labels={"project": "cayu"},
        )
        await workflow.approve("pending_other")

    with pytest.raises(PermissionError, match="outside review namespace"):
        asyncio.run(run())


def test_review_workflow_rechecks_scope_during_status_transition() -> None:
    async def run():
        store = ScopeDriftKnowledgeStore(
            [
                KnowledgeEntry(
                    id="pending_git",
                    text="Remote sandbox Git pushes should use a brokered credential proxy.",
                    namespace="project:cayu",
                    labels={"project": "cayu"},
                    status=KnowledgeStatus.PENDING,
                )
            ]
        )
        workflow = KnowledgeReviewWorkflow(
            store,
            namespace="project:cayu",
            labels={"project": "cayu"},
        )
        await workflow.approve("pending_git")

    with pytest.raises(ValueError, match="expected labels"):
        asyncio.run(run())


def test_review_workflow_rejects_conflicting_query_scope() -> None:
    workflow = KnowledgeReviewWorkflow(
        InMemoryKnowledgeStore(),
        namespace="project:cayu",
        labels={"project": "cayu"},
    )

    async def conflicting_namespace():
        return await workflow.list_pending(namespace="project:other")

    async def conflicting_label():
        return await workflow.list_pending(labels={"project": "other"})

    with pytest.raises(ValueError, match="conflicts with review namespace"):
        asyncio.run(conflicting_namespace())
    with pytest.raises(ValueError, match="conflicts with review label"):
        asyncio.run(conflicting_label())


def test_review_workflow_validates_store_contract() -> None:
    with pytest.raises(TypeError, match="review store methods"):
        KnowledgeReviewWorkflow(cast("Any", object()))
