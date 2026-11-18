"""Work admitted before `MembershipAccessPolicy(knowledge=True)`: why it has no knowledge, and
how an operator grants it some.

Cayu stores the grant each piece of work was admitted with and never widens it on its own, so
a conversation admitted while the policy granted no knowledge keeps none after knowledge is
turned on. Its knowledge tools say so instead of coming back empty, and an operator's recorded
grant admits it to the knowledge of the organizations it was admitted in.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from cayu import CayuApp
from cayu._resource_access_binding import ResourceExecutionBinding as _Binding
from cayu.knowledge.access import ScopedKnowledgeAccess
from cayu.memberships import (
    InMemoryMembershipDirectory,
    KnowledgeGrant,
    Membership,
    MembershipAccessPolicy,
    admitted_organizations,
    conversation_labels,
)
from cayu.memberships.policy import DEFAULT_AUTHORITY
from cayu.resource_access import (
    binding_sha256,
    current_decision,
    encode_grant,
    execution_access,
    work_admitted_without_knowledge,
)
from cayu.sessions import InMemorySessionStore
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import RunRequest
from cayu.storage import (
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeEntry,
    KnowledgeListQuery,
    KnowledgeQuery,
    KnowledgeStatus,
)
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tools import (
    ListKnowledgeTool,
    RememberKnowledgePolicy,
    RememberKnowledgeTool,
    SearchKnowledgeTool,
    ToolContext,
)

NAMESPACE = "project:acme-agent:agent:assistant"
IDENTITY = SessionIdentity(provider_name="fake", model="fake")


def _scope() -> KnowledgeAccessScope:
    return KnowledgeAccessScope(
        allowed_namespaces=[NAMESPACE],
        allowed_statuses=[KnowledgeStatus.ACTIVE, KnowledgeStatus.PENDING],
        session_label_keys=("organization",),
    )


def _directory() -> InMemoryMembershipDirectory:
    return InMemoryMembershipDirectory({"alice": [Membership("acme")]})


async def _binding(policy: MembershipAccessPolicy, subject: str) -> _Binding:
    decision = current_decision(await policy.resolve(subject))
    return _Binding(
        authority=policy.authority,
        subject=subject,
        admitted_json=encode_grant(decision.grant),
        policy_revision=decision.revision,
    )


def _labels(subject: str = "alice", organization: str = "acme") -> dict[str, str]:
    return conversation_labels(
        subject=subject, membership=Membership(organization), sharing="organization"
    )


async def _seed(store: InMemoryKnowledgeStore) -> None:
    for organization in ("acme", "globex"):
        await store.create_entry(
            KnowledgeEntry(
                id=f"{organization}-invoices",
                namespace=NAMESPACE,
                text=f"{organization} invoices over ten thousand need a second approver.",
                status=KnowledgeStatus.ACTIVE,
                labels={"organization": organization},
            )
        )


async def _tools(store, binding, policy, *, labels=None):
    """What `search_knowledge`, `list_knowledge` and `remember_knowledge` return inside one
    of alice's acme conversations run under ``binding``."""

    labels = _labels() if labels is None else labels
    context = ToolContext(
        session_id="alice-acme",
        session_labels=labels,
        knowledge_store=store,
        idempotency_key="remember-alice-acme",
    )
    async with execution_access(binding, policy, labels):
        searched = await SearchKnowledgeTool().run(
            context, {"query": "invoices approver", "namespace": NAMESPACE}
        )
        listed = await ListKnowledgeTool().run(context, {})
        remembered = await RememberKnowledgeTool(
            policy=RememberKnowledgePolicy(
                default_namespace=NAMESPACE, copy_session_labels=("organization",)
            )
        ).run(context, {"text": "Acme closes its books on the fifth."})
    return searched, listed, remembered


def test_knowledge_tools_say_the_work_was_admitted_before_knowledge() -> None:
    async def run() -> None:
        store = InMemoryKnowledgeStore(access_scope=_scope())
        await _seed(store)
        directory = _directory()
        before = await _binding(MembershipAccessPolicy(directory), "alice")
        policy = MembershipAccessPolicy(directory, knowledge=True)

        for result in await _tools(store, before, policy):
            assert result.is_error
            assert result.structured == {"error": "knowledge_not_admitted"}
            assert "admitted before its access policy granted knowledge" in result.content

        # Recall reads the store directly: it stays empty rather than failing the turn.
        async with execution_access(before, policy, _labels()):
            hits = await store.search(KnowledgeQuery(text="invoices", namespace=NAMESPACE))
        assert hits.hits == []

    asyncio.run(run())


def test_legitimately_empty_knowledge_reads_are_not_reported() -> None:
    async def run() -> None:
        store = InMemoryKnowledgeStore(access_scope=_scope())
        directory = _directory()
        # Admitted with knowledge, and nothing stored yet.
        policy = MembershipAccessPolicy(directory, knowledge=True)
        searched, listed, _ = await _tools(store, await _binding(policy, "alice"), policy)
        assert not searched.is_error and searched.content == "No knowledge results found."
        assert not listed.is_error
        # A policy that grants no knowledge at all.
        without = MembershipAccessPolicy(directory)
        searched, _, _ = await _tools(store, await _binding(without, "alice"), without)
        assert searched.structured.get("error") != "knowledge_not_admitted"

    asyncio.run(run())


def test_an_operator_grant_admits_earlier_work_to_its_organizations_knowledge() -> None:
    async def run() -> None:
        store = InMemoryKnowledgeStore(access_scope=_scope())
        await _seed(store)
        directory = _directory()
        before = await _binding(MembershipAccessPolicy(directory), "alice")
        policy = MembershipAccessPolicy(directory, knowledge=True)
        assert await policy.knowledge_admission(before) is None

        recorded = await directory.knowledge_grants.record(
            [
                KnowledgeGrant(
                    binding_sha256=binding_sha256(before),
                    authority=before.authority,
                    subject="alice",
                    organizations=tuple(admitted_organizations(before)),
                    sessions=1,
                    tasks=0,
                    granted_at=datetime.now(UTC),
                )
            ]
        )
        assert len(recorded) == 1

        searched, listed, remembered = await _tools(store, before, policy)
        assert not searched.is_error
        assert [hit["entry_id"] for hit in searched.structured["hits"]] == ["acme-invoices"]
        assert not listed.is_error
        assert not remembered.is_error, remembered.content

        # Alice joins globex later: the earlier work still gets only what it was admitted
        # in, while a new admission gets both.
        directory.set_memberships("alice", [Membership("acme"), Membership("globex")])
        old = ScopedKnowledgeAccess(store, binding=before, policy=policy)
        listing = await old.list_entries(KnowledgeListQuery(namespace=NAMESPACE, limit=100))
        assert {item.entry.labels["organization"] for item in listing.entries} == {"acme"}
        fresh = ScopedKnowledgeAccess(store, binding=await _binding(policy, "alice"), policy=policy)
        listing = await fresh.list_entries(KnowledgeListQuery(namespace=NAMESPACE, limit=100))
        assert {item.entry.labels["organization"] for item in listing.entries} == {
            "acme",
            "globex",
        }

        # A grant recorded for one admission doesn't cover another.
        later = await _binding(MembershipAccessPolicy(directory), "alice")
        assert binding_sha256(later) != binding_sha256(before)
        assert await policy.knowledge_admission(later) is None

    asyncio.run(run())


def test_the_work_admitted_without_knowledge_is_listed_by_admission() -> None:
    async def run() -> None:
        directory = InMemoryMembershipDirectory(
            {"alice": [Membership("acme")], "dave": [Membership("globex")]}
        )
        sessions = InMemorySessionStore()
        tasks = InMemoryTaskStore()

        async def start(app: CayuApp, subject: str, organization: str) -> str:
            access = await app.access(subject)
            created = await access.sessions.create(
                RunRequest(
                    agent_name="assistant", messages=[], labels=_labels(subject, organization)
                ),
                identity=IDENTITY,
            )
            return created.id

        before = CayuApp(
            session_store=sessions,
            task_store=tasks,
            resource_access_policy=MembershipAccessPolicy(directory),
        )
        after = CayuApp(
            session_store=sessions,
            task_store=tasks,
            resource_access_policy=MembershipAccessPolicy(directory, knowledge=True),
        )
        old_alice = await start(before, "alice", "acme")
        old_dave = await start(before, "dave", "globex")
        await start(after, "alice", "acme")

        work = await work_admitted_without_knowledge(sessions, tasks, authority=DEFAULT_AUTHORITY)
        assert [(item.binding.subject, item.session_ids) for item in work] == [
            ("alice", (old_alice,)),
            ("dave", (old_dave,)),
        ]
        assert admitted_organizations(work[0].binding) == ["acme"]
        assert work[0].binding_sha256 == binding_sha256(work[0].binding)
        only = await work_admitted_without_knowledge(
            sessions, tasks, authority=DEFAULT_AUTHORITY, subject="dave"
        )
        assert [item.session_ids for item in only] == [(old_dave,)]
        assert await work_admitted_without_knowledge(sessions, tasks, authority="other") == ()

    asyncio.run(run())
