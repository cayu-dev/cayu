"""MembershipAccessPolicy: each person sees only what their current memberships allow.

These are the policy tests generated organizations projects used to carry in
`tests/test_organizations.py`; they run on the in-memory directory, without a database.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from cayu import AgentSpec, CayuApp, Message, ResumeRequest
from cayu.evals import ScriptedModelProvider
from cayu.memberships import (
    SHARING_LABELS,
    AccessSnapshot,
    InMemoryMembershipDirectory,
    Membership,
    MembershipAccessPolicy,
    MembershipDirectory,
    Sharing,
    conversation_labels,
)
from cayu.providers import ModelStreamEvent
from cayu.resource_access import ResourceAccessDenied, ResourceAccessPolicy
from cayu.sessions import InMemorySessionStore
from cayu.sessions.access import SessionAccessDenied
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import RunRequest

IDENTITY = SessionIdentity(provider_name="fake", model="fake")
ALICE = Membership("acme", "acme/finance")
CAROL = Membership("acme", "acme/finance")
BOB = Membership("acme", "acme/legal")
DAVE = Membership("globex", "globex/support")


def _directory() -> InMemoryMembershipDirectory:
    return InMemoryMembershipDirectory(
        {"alice": [ALICE], "carol": [CAROL], "bob": [BOB], "dave": [DAVE]}
    )


def _app(directory: MembershipDirectory) -> CayuApp:
    return CayuApp(
        session_store=InMemorySessionStore(),
        resource_access_policy=MembershipAccessPolicy(directory),
    )


async def _start(app: CayuApp, subject: str, membership: Membership, sharing: Sharing) -> str:
    labels = conversation_labels(subject=subject, membership=membership, sharing=sharing)
    access = await app.access(subject)
    created = await access.sessions.create(
        RunRequest(agent_name="assistant", messages=[], labels=labels), identity=IDENTITY
    )
    return created.id


async def _visible(app: CayuApp, subject: str) -> set[str]:
    access = await app.access(subject)
    return {item.id for item in (await access.sessions.list_sessions()).sessions}


async def _can_read(app: CayuApp, subject: str, session_id: str) -> bool:
    access = await app.access(subject)
    try:
        await access.sessions.load(session_id)
    except SessionAccessDenied:
        return False
    return True


def test_the_policy_keeps_the_authority_generated_projects_use() -> None:
    policy = MembershipAccessPolicy(_directory())
    assert isinstance(policy, ResourceAccessPolicy)
    assert policy.authority == "b2b-membership-v1"
    assert MembershipAccessPolicy(_directory(), authority="acme-v2").authority == "acme-v2"
    for options in (
        {"authority": " "},
        {"decision_lifetime": timedelta(0)},
        {"max_memberships": 0},
    ):
        with pytest.raises(ValueError):
            MembershipAccessPolicy(_directory(), **options)


def test_a_decision_lasts_the_decision_lifetime_and_carries_the_revision() -> None:
    async def run() -> None:
        directory = _directory()
        directory.set_memberships("alice", [ALICE])
        before = datetime.now(UTC)
        decision = await MembershipAccessPolicy(
            directory, decision_lifetime=timedelta(seconds=5)
        ).resolve("alice")
        assert decision.revision == 1
        assert decision.valid_until is not None
        assert before + timedelta(seconds=5) <= decision.valid_until
        assert decision.valid_until <= datetime.now(UTC) + timedelta(seconds=5)
        assert set(decision.grant.sessions.protected_label_keys) == set(SHARING_LABELS)
        # Nobody's memberships grant nothing.
        empty = await MembershipAccessPolicy(directory).resolve("nobody")
        assert empty.grant.sessions.read == () and empty.valid_until is None

    asyncio.run(run())


def test_each_person_sees_their_own_their_department_and_their_organization() -> None:
    async def run() -> None:
        app = _app(_directory())
        personal = await _start(app, "alice", ALICE, "personal")
        department = await _start(app, "alice", ALICE, "department")
        organization = await _start(app, "alice", ALICE, "organization")

        assert await _visible(app, "alice") == {personal, department, organization}
        # Same department.
        assert await _visible(app, "carol") == {department, organization}
        # Same organization, another department.
        assert await _visible(app, "bob") == {organization}
        # Another organization sees nothing, and a known ID reads as missing.
        assert await _visible(app, "dave") == set()
        assert not await _can_read(app, "dave", organization)

    asyncio.run(run())


def test_nobody_creates_outside_their_memberships_or_for_someone_else() -> None:
    async def run() -> None:
        app = _app(_directory())
        dave = await app.access("dave")
        for labels in (
            # Into another organization.
            conversation_labels(subject="dave", membership=ALICE, sharing="organization"),
            # Into a department of their organization they don't belong to.
            conversation_labels(subject="bob", membership=ALICE, sharing="department"),
            # As if someone else owned it.
            conversation_labels(subject="alice", membership=DAVE, sharing="personal"),
        ):
            with pytest.raises(SessionAccessDenied):
                await dave.sessions.create(
                    RunRequest(agent_name="assistant", messages=[], labels=labels),
                    identity=IDENTITY,
                )
        # A department of their own, labelled with another organization or none.
        alice = await app.access("alice")
        mislabelled = conversation_labels(subject="alice", membership=ALICE, sharing="department")
        for labels in (
            {**mislabelled, "organization": "globex"},
            {key: value for key, value in mislabelled.items() if key != "organization"},
        ):
            with pytest.raises(SessionAccessDenied):
                await alice.sessions.create(
                    RunRequest(agent_name="assistant", messages=[], labels=labels),
                    identity=IDENTITY,
                )
        # Not even one they could read, shared with their department or organization.
        for sharing in ("department", "organization"):
            labels = conversation_labels(subject="carol", membership=CAROL, sharing=sharing)
            with pytest.raises(SessionAccessDenied):
                await alice.sessions.create(
                    RunRequest(agent_name="assistant", messages=[], labels=labels),
                    identity=IDENTITY,
                )

    asyncio.run(run())


def test_sharing_labels_cant_be_changed_and_only_the_owner_changes_a_conversation() -> None:
    async def run() -> None:
        app = _app(_directory())
        shared = await _start(app, "alice", ALICE, "department")
        alice = await app.access("alice")
        loaded = await alice.sessions.load(shared)
        with pytest.raises(SessionAccessDenied):
            await alice.sessions.update_labels(shared, {**loaded.labels, "sharing": "organization"})
        carol = await app.access("carol")
        with pytest.raises(SessionAccessDenied):
            await carol.sessions.update_labels(shared, {**loaded.labels, "topic": "budget"})

    asyncio.run(run())


def test_a_removed_member_loses_access_at_their_next_step() -> None:
    async def run() -> None:
        directory = _directory()
        app = _app(directory)
        shared = await _start(app, "alice", ALICE, "department")
        carol = await app.access("carol")
        assert (await carol.sessions.load(shared)).id == shared

        directory.set_memberships("carol", [])

        # The handle she already holds is rechecked against current membership.
        with pytest.raises(SessionAccessDenied):
            await carol.sessions.load(shared)

    asyncio.run(run())


def test_only_the_owner_continues_a_shared_conversation() -> None:
    provider = ScriptedModelProvider(
        [
            ModelStreamEvent.text_delta("Done."),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
        * 2
    )

    async def start(app: CayuApp, subject: str, labels: dict[str, str]) -> str:
        access = await app.access(subject)
        request = RunRequest(
            agent_name="assistant",
            messages=[Message.text("user", "Summarize the budget")],
            labels=labels,
        )
        async for event in access.run(request):
            session_id = event.session_id
        return session_id

    async def resume(app: CayuApp, subject: str, session_id: str) -> None:
        access = await app.access(subject)
        request = ResumeRequest(
            session_id=session_id, messages=[Message.text("user", "And next month?")]
        )
        async for _ in access.resume(request):
            pass

    async def run() -> None:
        app = _app(_directory())
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="scripted-model"))
        labels = conversation_labels(subject="alice", membership=ALICE, sharing="department")
        shared = await start(app, "alice", labels)
        # Carol reads it, but continuing it would act as Alice.
        assert await _can_read(app, "carol", shared)
        with pytest.raises(ResourceAccessDenied):
            await resume(app, "carol", shared)
        await resume(app, "alice", shared)

    asyncio.run(run())


def test_a_department_conversation_belongs_to_its_department_alone() -> None:
    async def run() -> None:
        # Mallory belongs to Acme's finance department and to Globex.
        mallory = [Membership("acme", "acme/finance"), Membership("globex")]
        directory = InMemoryMembershipDirectory({"mallory": mallory, "dave": [DAVE]})
        app = _app(directory)
        # Labelled with Globex, but shared with Acme's finance department.
        labels = {
            "organization": "globex",
            "department": "acme/finance",
            "owner": "mallory",
            "sharing": "department",
        }
        access = await app.access("mallory")
        created = await access.sessions.create(
            RunRequest(agent_name="assistant", messages=[], labels=labels), identity=IDENTITY
        )
        # Globex's members don't see it: only the department decides.
        assert not await _can_read(app, "dave", created.id)

        directory.set_memberships("mallory", [Membership("globex")])

        # Out of the department, its Globex label keeps nothing for her: she can't
        # read it, nor change or continue it as its owner.
        assert not await _can_read(app, "mallory", created.id)
        with pytest.raises(SessionAccessDenied):
            await access.sessions.update_labels(created.id, {**labels, "topic": "q3"})

    asyncio.run(run())


def test_a_membership_of_another_organizations_department_is_refused() -> None:
    class Directory:
        async def current(self, subject: str) -> AccessSnapshot:
            # As a table without the department foreign key could report it.
            return AccessSnapshot((Membership("globex", "acme/finance"),), revision=3)

    async def run() -> None:
        with pytest.raises(ValueError, match="isn't one of 'globex'"):
            await MembershipAccessPolicy(Directory()).resolve("mallory")
        # Nor one whose organization is only a prefix of the department's.
        prefix = InMemoryMembershipDirectory({"erin": [Membership("acme", "acmesoft/legal")]})
        with pytest.raises(ValueError, match="department"):
            await MembershipAccessPolicy(prefix).resolve("erin")

    asyncio.run(run())


def test_the_most_memberships_a_person_may_have_still_fit_one_grant() -> None:
    # The longest IDs the tables hold: a 64-character department of a 62-character
    # organization, whose IDs it starts with.
    longest = [Membership(f"org{index:0>59}", f"org{index:0>59}/d") for index in range(16)]

    async def run() -> None:
        app = _app(InMemoryMembershipDirectory({"erin": longest}))
        await app.access("erin")
        too_many = InMemoryMembershipDirectory({"erin": [*longest, Membership("one-more")]})
        with pytest.raises(ValueError, match="memberships"):
            await MembershipAccessPolicy(too_many).resolve("erin")
        # A project that allows fewer refuses sooner.
        with pytest.raises(ValueError, match="more than 2 memberships"):
            await MembershipAccessPolicy(
                InMemoryMembershipDirectory({"erin": longest[:3]}), max_memberships=2
            ).resolve("erin")

    asyncio.run(run())


def test_department_sharing_needs_a_department() -> None:
    with pytest.raises(ValueError, match="department"):
        conversation_labels(subject="erin", membership=Membership("acme"), sharing="department")
    assert conversation_labels(
        subject="erin", membership=Membership("acme", "acme/legal"), sharing="department"
    ) == {
        "organization": "acme",
        "department": "acme/legal",
        "owner": "erin",
        "sharing": "department",
    }
