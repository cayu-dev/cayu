"""Knowledge scoped by organization: what one organization's agent remembers stays its own.

`MembershipAccessPolicy(knowledge=True)` grants each member the knowledge of their
organizations, and a knowledge scope with `session_label_keys=("organization",)` pins every
knowledge operation inside a conversation to that conversation's organization. Both are
enforced by the store, so these tests read the store directly, the way recall, lookup and the
pending entries recall offers do.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from cayu._resource_access_binding import ResourceExecutionBinding
from cayu.knowledge._revision_rules import _default_chunk_for_entry
from cayu.knowledge.access import ScopedKnowledgeAccess, session_knowledge_labels
from cayu.knowledge.scopes import knowledge_access_scope_sha256
from cayu.memberships import (
    InMemoryMembershipDirectory,
    Membership,
    MembershipAccessPolicy,
    conversation_labels,
)
from cayu.resource_access import (
    ResourceAccessDenied,
    current_decision,
    encode_grant,
    execution_access,
)
from cayu.storage import (
    InMemoryKnowledgeStore,
    KnowledgeAccessDenied,
    KnowledgeAccessScope,
    KnowledgeEntry,
    KnowledgeListQuery,
    KnowledgeQuery,
    KnowledgeReviewWorkflow,
    KnowledgeStatus,
)
from cayu.storage.knowledge_sqlite import SQLiteKnowledgeStore
from cayu.tools import RememberKnowledgePolicy, RememberKnowledgeTool, ToolContext

NAMESPACE = "project:acme-agent:agent:assistant"
NOTE = "Invoices over ten thousand need a second approver."
ALICE = Membership("acme")
CAROL = Membership("acme")
DAVE = Membership("globex")


def _scope(*, session_label_keys: tuple[str, ...] = ("organization",)) -> KnowledgeAccessScope:
    return KnowledgeAccessScope(
        allowed_namespaces=[NAMESPACE],
        allowed_statuses=[KnowledgeStatus.ACTIVE, KnowledgeStatus.PENDING],
        session_label_keys=session_label_keys,
    )


BACKENDS = ["memory", "sqlite", "postgres"]


def _store(backend: str, request: pytest.FixtureRequest, scope: KnowledgeAccessScope):
    if backend == "memory":
        return InMemoryKnowledgeStore(access_scope=scope)
    if backend == "sqlite":
        path = request.getfixturevalue("tmp_path") / "knowledge.db"
        return SQLiteKnowledgeStore(path, access_scope=scope)
    from cayu.storage.knowledge_postgres import PostgresKnowledgeStore
    from cayu.storage.migrations import SchemaMode

    return PostgresKnowledgeStore(
        request.getfixturevalue("postgres_dsn"),
        schema_mode=SchemaMode.CREATE,
        access_scope=scope,
    )


async def _close(store) -> None:
    if not isinstance(store, InMemoryKnowledgeStore):
        await store.close()


def _directory() -> InMemoryMembershipDirectory:
    return InMemoryMembershipDirectory(
        {
            "alice": [ALICE],
            "carol": [CAROL],
            "dave": [DAVE],
            # Belongs to both organizations.
            "erin": [Membership("acme"), Membership("globex")],
        }
    )


async def _binding(policy: MembershipAccessPolicy, subject: str) -> ResourceExecutionBinding:
    decision = current_decision(await policy.resolve(subject))
    return ResourceExecutionBinding(
        authority=policy.authority,
        subject=subject,
        admitted_json=encode_grant(decision.grant),
        policy_revision=decision.revision,
    )


def _labels(subject: str, organization: str) -> dict[str, str]:
    return conversation_labels(
        subject=subject, membership=Membership(organization), sharing="organization"
    )


async def _remember(store, policy, subject, organization, *, copy=("organization",)):
    tool = RememberKnowledgeTool(
        policy=RememberKnowledgePolicy(default_namespace=NAMESPACE, copy_session_labels=copy)
    )
    labels = _labels(subject, organization)
    async with execution_access(await _binding(policy, subject), policy, labels):
        return await tool.run(
            ToolContext(
                session_id=f"{subject}-{organization}",
                session_labels=labels,
                knowledge_store=store,
                idempotency_key=f"remember-{subject}-{organization}-{copy}",
            ),
            {"text": NOTE, "labels": {"organization": "globex"}},
        )


async def _seen(store, policy, subject, organization, entry_id) -> dict[str, set[str]]:
    """What recall, lookup and the pending listing return inside one of `subject`'s
    conversations in `organization`."""

    async with execution_access(
        await _binding(policy, subject), policy, _labels(subject, organization)
    ):
        recalled = await store.search(
            KnowledgeQuery(
                text="invoices second approver",
                namespace=NAMESPACE,
                statuses=[KnowledgeStatus.ACTIVE, KnowledgeStatus.PENDING],
            )
        )
        pending = await store.list_entries(
            KnowledgeListQuery(statuses=[KnowledgeStatus.PENDING], limit=100)
        )
        looked_up = await store.get_entry(entry_id)
    return {
        "recalled": {hit.entry.id for hit in recalled.hits},
        "pending": {item.entry.id for item in pending.entries},
        "looked_up": set() if looked_up is None else {looked_up.id},
    }


@pytest.mark.parametrize("backend", BACKENDS)
def test_one_organizations_remembered_entry_never_reaches_another(
    backend: str, request: pytest.FixtureRequest
) -> None:
    async def run() -> None:
        policy = MembershipAccessPolicy(_directory(), knowledge=True)
        store = _store(backend, request, _scope())
        try:
            result = await _remember(store, policy, "alice", "acme")
            assert result.structured is not None and not result.is_error, result
            entry_id = result.structured["entry"]["entry_id"]
            entry = await store.get_entry(entry_id)
            # Labelled from the conversation, whatever the model asked for.
            assert entry is not None and entry.labels == {"organization": "acme"}
            assert entry.status is KnowledgeStatus.PENDING

            mine = {"recalled", "pending", "looked_up"}
            everything = {key: {entry_id} for key in mine}
            nothing = {key: set() for key in mine}
            # Another member of the same organization recalls it.
            assert await _seen(store, policy, "carol", "acme", entry_id) == everything
            # A member of another organization never does, in any form.
            assert await _seen(store, policy, "dave", "globex", entry_id) == nothing
            # A member of both sees it only in the organization's own conversations, so it
            # can't land in a conversation the other organization's members read.
            assert await _seen(store, policy, "erin", "acme", entry_id) == everything
            assert await _seen(store, policy, "erin", "globex", entry_id) == nothing

            # Outside a conversation (a member reading knowledge in the web app), the grant
            # alone decides.
            for subject, expected in (("dave", set()), ("carol", {entry_id})):
                handle = ScopedKnowledgeAccess(
                    store, binding=await _binding(policy, subject), policy=policy
                )
                listed = await handle.list_entries(
                    KnowledgeListQuery(statuses=[KnowledgeStatus.PENDING], limit=100)
                )
                assert {item.entry.id for item in listed.entries} == expected
                assert (await handle.get_entry(entry_id) is not None) == bool(expected)

            # Curation is per organization: operators work through one organization's queue.
            queue = await store.list_entries(
                KnowledgeListQuery(
                    statuses=[KnowledgeStatus.PENDING], labels={"organization": "globex"}
                )
            )
            assert queue.entries == []
        finally:
            await _close(store)

    asyncio.run(run())


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_tool_that_forgets_the_organization_cannot_write_or_leak(
    backend: str, request: pytest.FixtureRequest
) -> None:
    async def run() -> None:
        policy = MembershipAccessPolicy(_directory(), knowledge=True)
        store = _store(backend, request, _scope())
        try:
            result = await _remember(store, policy, "alice", "acme", copy=())
            assert result.is_error, result
            assert (await store.list_entries(KnowledgeListQuery(limit=100))).entries == []
            labels = _labels("alice", "acme")
            async with execution_access(await _binding(policy, "alice"), policy, labels):
                # Nor straight into the store with another organization's label.
                with pytest.raises(PermissionError):
                    await store.create_entry(
                        KnowledgeEntry(
                            id="foreign",
                            namespace=NAMESPACE,
                            text=NOTE,
                            labels={"organization": "globex"},
                        )
                    )
                with pytest.raises(PermissionError):
                    await store.create_entry(
                        KnowledgeEntry(id="bare", namespace=NAMESPACE, text=NOTE)
                    )
        finally:
            await _close(store)

    asyncio.run(run())


def test_a_conversation_without_an_organization_gets_no_knowledge() -> None:
    async def run() -> None:
        policy = MembershipAccessPolicy(_directory(), knowledge=True)
        store = InMemoryKnowledgeStore(access_scope=_scope())
        binding = await _binding(policy, "alice")
        # Admission refuses such a conversation already; the store refuses it on its own too.
        async with execution_access(binding, policy, _labels("alice", "acme")):
            from cayu.resource_access import _active

            active = _active.get()
            token = _active.set(type(active)(active.binding, active.policy, ()))
            try:
                with pytest.raises(ResourceAccessDenied):
                    await store.list_entries(KnowledgeListQuery())
            finally:
                _active.reset(token)

    asyncio.run(run())


def test_without_knowledge_the_policy_grants_what_it_always_did() -> None:
    async def run() -> None:
        directory = _directory()
        default = await MembershipAccessPolicy(directory).resolve("erin")
        enabled = await MembershipAccessPolicy(directory, knowledge=True).resolve("erin")
        assert encode_grant(default.grant) != encode_grant(enabled.grant)
        # Off by default: knowledge is denied, and every other family is unchanged.
        assert default.grant.knowledge.read == ()
        for family in ("sessions", "tasks", "artifacts"):
            assert getattr(default.grant, family) == getattr(enabled.grant, family)
        assert enabled.grant.knowledge.delete == ()
        assert enabled.grant.knowledge.protected_keys() == {"organization"}

    asyncio.run(run())


@pytest.mark.parametrize("backend", BACKENDS)
def test_knowledge_written_outside_a_conversation_needs_the_organization(
    backend: str, request: pytest.FixtureRequest
) -> None:
    """Operator code, seeding and curation run outside any binding. An active or pending
    entry they write without `organization` would reach no conversation, so it's refused,
    with what to add; entries already stored without one can still be fixed or retired."""

    async def run() -> None:
        store = _store(backend, request, _scope())
        try:
            with pytest.raises(KnowledgeAccessDenied) as refused:
                await store.create_entry(
                    KnowledgeEntry(
                        id="bare", namespace=NAMESPACE, text=NOTE, status=KnowledgeStatus.ACTIVE
                    )
                )
            assert "'organization'" in str(refused.value)
            assert 'labels={"organization": "<organization id>"}' in str(refused.value)
            assert await store.get_entry("bare") is None

            kept = await store.create_entry(
                KnowledgeEntry(
                    id="kept",
                    namespace=NAMESPACE,
                    text=NOTE,
                    status=KnowledgeStatus.ACTIVE,
                    labels={"organization": "acme"},
                )
            )
            # A revision can't drop it either, through either write path.
            unlabelled = kept.model_copy(update={"revision": 2, "labels": {}})
            with pytest.raises(KnowledgeAccessDenied):
                await store.append_entry_revision(unlabelled, expected_revision=1)
            with pytest.raises(KnowledgeAccessDenied):
                await store.publish_entry_revision(
                    unlabelled,
                    [_default_chunk_for_entry(unlabelled)],
                    operation_id="drop-label",
                    expected_revision=1,
                )
            current = await store.get_entry("kept")
            assert current is not None and current.revision == 1

            # remember_knowledge outside a conversation with no organization is refused too.
            result = await RememberKnowledgeTool(
                policy=RememberKnowledgePolicy(
                    default_namespace=NAMESPACE, copy_session_labels=("organization",)
                )
            ).run(
                ToolContext(session_id="operator", knowledge_store=store, idempotency_key="op"),
                {"text": NOTE},
            )
            assert result.is_error, result
            listed = await store.list_entries(KnowledgeListQuery(limit=100))
            assert {item.entry.id for item in listed.entries} == {"kept"}
        finally:
            await _close(store)

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_an_entry_stored_without_the_organization_can_be_labelled_or_retired(
    backend: str, tmp_path: Path
) -> None:
    """Entries stored before the rule: revising, keeping or activating one without the label
    is refused, labelling or retiring it isn't."""

    legacy = [
        KnowledgeEntry(id=entry_id, namespace=NAMESPACE, text=NOTE, status=status)
        for entry_id, status in (
            ("legacy", KnowledgeStatus.ACTIVE),
            ("stale", KnowledgeStatus.ACTIVE),
            ("proposal", KnowledgeStatus.PENDING),
            ("dropped", KnowledgeStatus.PENDING),
        )
    ]

    async def open_store():
        if backend == "memory":
            return InMemoryKnowledgeStore(legacy, access_scope=_scope())
        path = tmp_path / "knowledge.db"
        plain = SQLiteKnowledgeStore(path, access_scope=_scope(session_label_keys=()))
        try:
            for entry in legacy:
                await plain.create_entry(entry)
        finally:
            await plain.close()
        return SQLiteKnowledgeStore(path, access_scope=_scope())

    async def run() -> None:
        store = await open_store()
        try:
            current = await store.get_entry("legacy")
            assert current is not None
            with pytest.raises(KnowledgeAccessDenied):
                await store.append_entry_revision(
                    current.model_copy(update={"revision": 2, "text": "Revised."}),
                    expected_revision=1,
                )
            labelled = await store.append_entry_revision(
                current.model_copy(update={"revision": 2, "labels": {"organization": "acme"}}),
                expected_revision=1,
            )
            assert labelled.labels == {"organization": "acme"}

            # Keeping an unlabelled proposal would publish it to no one.
            with pytest.raises(KnowledgeAccessDenied):
                await KnowledgeReviewWorkflow(store, namespace=NAMESPACE).approve(
                    "proposal",
                    operation_id="keep-proposal",
                    reviewer_identity="operator",
                    reviewer_version="1",
                )
            with pytest.raises(KnowledgeAccessDenied):
                await store.transition_entry_status(
                    "proposal",
                    expected_revision=1,
                    from_status=KnowledgeStatus.PENDING,
                    to_status=KnowledgeStatus.ACTIVE,
                )
            proposal = await store.get_entry("proposal")
            assert proposal is not None and proposal.status is KnowledgeStatus.PENDING

            # Dropping or archiving one still works.
            dropped = await store.transition_entry_status(
                "dropped",
                expected_revision=1,
                from_status=KnowledgeStatus.PENDING,
                to_status=KnowledgeStatus.ARCHIVED,
            )
            assert dropped.status is KnowledgeStatus.ARCHIVED
            stale = await store.get_entry("stale")
            assert stale is not None
            archived = await store.append_entry_revision(
                stale.model_copy(update={"revision": 2, "status": KnowledgeStatus.ARCHIVED}),
                expected_revision=1,
            )
            assert archived.status is KnowledgeStatus.ARCHIVED
        finally:
            await _close(store)

    asyncio.run(run())


def test_projects_without_organizations_read_and_write_as_before(tmp_path: Path) -> None:
    """A scope without `session_label_keys` is unchanged, and outside a binding one with
    them narrows no reads."""

    plain = KnowledgeAccessScope(
        allowed_namespaces=[NAMESPACE],
        allowed_statuses=[KnowledgeStatus.ACTIVE, KnowledgeStatus.PENDING],
    )
    assert "session_label_keys" not in plain.model_dump()
    # The same identity the scope always had.
    assert knowledge_access_scope_sha256(plain) == knowledge_access_scope_sha256(
        KnowledgeAccessScope.model_validate(
            {
                "allowed_namespaces": [NAMESPACE],
                "allowed_statuses": ["active", "pending"],
            }
        )
    )

    async def run(scope: KnowledgeAccessScope) -> set[str]:
        store = SQLiteKnowledgeStore(
            tmp_path / f"{len(scope.session_label_keys)}.db", access_scope=scope
        )
        try:
            tool = RememberKnowledgeTool(
                policy=RememberKnowledgePolicy(default_namespace=NAMESPACE)
            )
            result = await tool.run(
                ToolContext(session_id="s", knowledge_store=store, idempotency_key="one"),
                {"text": NOTE},
            )
            assert not result.is_error, result
            hits = await store.search(
                KnowledgeQuery(
                    text="invoices",
                    namespace=NAMESPACE,
                    statuses=[KnowledgeStatus.ACTIVE, KnowledgeStatus.PENDING],
                )
            )
            return {hit.entry.id for hit in hits.hits}
        finally:
            await store.close()

    plain_hits = asyncio.run(run(plain))
    assert len(plain_hits) == 1


async def _seed_both_organizations(store) -> None:
    for organization in ("acme", "globex"):
        await store.create_entry(
            KnowledgeEntry(
                id=f"{organization}-invoices",
                namespace=NAMESPACE,
                text=f"{organization}: {NOTE}",
                status=KnowledgeStatus.ACTIVE,
                labels={"organization": organization},
            )
        )


async def _readable(store) -> set[str]:
    """The entries a search, a listing and a lookup each return here, together. Only this
    test's entries count: the PostgreSQL database is shared by the module's tests."""

    ours = {"acme-invoices", "globex-invoices", "acme-own"}
    hits = await store.search(KnowledgeQuery(text="invoices second approver", namespace=NAMESPACE))
    listed = await store.list_entries(KnowledgeListQuery(limit=100))
    seen = {hit.entry.id for hit in hits.hits} & ours
    assert seen == {item.entry.id for item in listed.entries} & ours
    for entry_id in ("acme-invoices", "globex-invoices"):
        assert (await store.get_entry(entry_id) is not None) == (entry_id in seen)
    return seen


@pytest.mark.parametrize("backend", BACKENDS)
def test_work_for_an_organization_without_a_member_reads_and_writes_only_its_own(
    backend: str, request: pytest.FixtureRequest
) -> None:
    """A schedule, an operator task or an `app.run` session labelled with an organization is
    that organization's work even with no member binding: its knowledge is pinned to the
    label, as a member's conversation is. Work with no organization label stays
    operator-wide."""

    async def run() -> None:
        store = _store(backend, request, _scope())
        try:
            await _seed_both_organizations(store)
            both = {"acme-invoices", "globex-invoices"}
            # Operators, curation and maintenance see every organization.
            assert await _readable(store) == both
            with session_knowledge_labels({"purpose": "maintenance"}):
                assert await _readable(store) == both

            with session_knowledge_labels({"organization": "acme", "purpose": "digest"}):
                assert await _readable(store) == {"acme-invoices"}
                # Nor can it write into another organization's knowledge.
                with pytest.raises(PermissionError):
                    await store.create_entry(
                        KnowledgeEntry(
                            id="foreign",
                            namespace=NAMESPACE,
                            text=NOTE,
                            labels={"organization": "globex"},
                        )
                    )
                own = await store.create_entry(
                    KnowledgeEntry(
                        id="acme-own",
                        namespace=NAMESPACE,
                        text=NOTE,
                        labels={"organization": "acme"},
                    )
                )
                assert own.labels == {"organization": "acme"}
                # Its own pending notes can be kept and its entries deleted.
                await store.create_entry(
                    KnowledgeEntry(
                        id="acme-proposal",
                        namespace=NAMESPACE,
                        text=NOTE,
                        status=KnowledgeStatus.PENDING,
                        labels={"organization": "acme"},
                    )
                )
                kept = await KnowledgeReviewWorkflow(store, namespace=NAMESPACE).approve(
                    "acme-proposal",
                    operation_id="keep-acme-proposal",
                    reviewer_identity="operator",
                    reviewer_version="1",
                )
                assert kept.entry.status is KnowledgeStatus.ACTIVE
                assert await store.delete_entry("acme-proposal", expected_revision=2) is not None
                assert await store.get_entry("acme-proposal") is None
                # A run started inside it without the label, such as a child session, is
                # still acme's work; one labelled for another organization gets nothing.
                with session_knowledge_labels({}):
                    assert await _readable(store) == {"acme-invoices", "acme-own"}
                with (
                    session_knowledge_labels({"organization": "globex"}),
                    pytest.raises(ResourceAccessDenied),
                ):
                    await store.list_entries(KnowledgeListQuery(limit=100))

            with session_knowledge_labels({"organization": "globex"}):
                assert await _readable(store) == {"globex-invoices"}
            assert await _readable(store) == both | {"acme-own"}
        finally:
            await _close(store)

    asyncio.run(run())


def test_a_scope_naming_another_organization_is_refused_in_an_organizations_work() -> None:
    async def run() -> None:
        store = InMemoryKnowledgeStore(
            [
                KnowledgeEntry(
                    id=f"{organization}-invoices",
                    namespace=NAMESPACE,
                    text=NOTE,
                    labels={"organization": organization},
                )
                for organization in ("acme", "globex")
            ]
        )
        globex = KnowledgeAccessScope(
            allowed_namespaces=[NAMESPACE],
            required_labels={"organization": "globex"},
            session_label_keys=("organization",),
        )
        query = KnowledgeListQuery(limit=100)
        listed = await store.list_entries(query, access_scope=globex)
        assert {item.entry.id for item in listed.entries} == {"globex-invoices"}
        with (
            session_knowledge_labels({"organization": "acme"}),
            pytest.raises(ResourceAccessDenied),
        ):
            await store.list_entries(query, access_scope=globex)

    asyncio.run(run())


def test_a_member_handle_used_in_an_organizations_work_is_pinned_to_it() -> None:
    async def run() -> None:
        policy = MembershipAccessPolicy(_directory(), knowledge=True)
        store = InMemoryKnowledgeStore(access_scope=_scope())
        await _seed_both_organizations(store)
        # erin belongs to both organizations; outside any session the grant alone decides.
        handle = ScopedKnowledgeAccess(store, binding=await _binding(policy, "erin"), policy=policy)
        query = KnowledgeListQuery(limit=100)
        listed = await handle.list_entries(query)
        assert {item.entry.id for item in listed.entries} == {"acme-invoices", "globex-invoices"}
        with session_knowledge_labels({"organization": "globex"}):
            listed = await handle.list_entries(query)
        assert {item.entry.id for item in listed.entries} == {"globex-invoices"}

    asyncio.run(run())


def test_an_app_run_labelled_with_an_organization_gets_only_its_knowledge() -> None:
    """The runtime pins a tool's knowledge to its session's labels without a binding."""

    from cayu import AgentSpec, CayuApp, Message
    from cayu.evals import ScriptedModelProvider
    from cayu.providers import ModelStreamEvent
    from cayu.sessions.memory import InMemorySessionStore
    from cayu.sessions.requests import RunRequest
    from cayu.tools import Tool, ToolResult, ToolSpec

    store = InMemoryKnowledgeStore(access_scope=_scope())
    seen: list[set[str]] = []

    class LooksUp(Tool):
        """A tool that reads the store itself and filters nothing."""

        spec = ToolSpec(
            name="look_up",
            description="Look up invoice knowledge.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            seen.append(await _readable(store))
            return ToolResult(content="ok")

    async def run(labels: dict[str, str]) -> set[str]:
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(id="call_1", name="look_up", arguments={}),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[LooksUp()])
        try:
            async for _event in app.run(
                RunRequest(
                    agent_name="assistant", labels=labels, messages=[Message.text("user", "go")]
                )
            ):
                pass
        finally:
            await app.aclose()
        return seen.pop()

    async def scenario() -> None:
        await _seed_both_organizations(store)
        assert await run(_labels("carol", "acme")) == {"acme-invoices"}
        assert await run(_labels("dave", "globex")) == {"globex-invoices"}
        assert await run({}) == {"acme-invoices", "globex-invoices"}

    asyncio.run(scenario())
