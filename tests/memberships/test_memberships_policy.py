"""MembershipAccessPolicy: each person sees only what their current memberships allow.

These are the policy tests generated organizations projects used to carry in
`tests/test_organizations.py`; they run on the in-memory directory, without a database.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
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
from cayu.resource_access import (
    ResourceAccessDecision,
    ResourceAccessDenied,
    ResourceAccessPolicy,
)
from cayu.sessions import (
    InMemorySessionStore,
    SessionAccessRule,
    SessionAccessSelector,
)
from cayu.sessions.access import SessionAccessDenied
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import RunRequest

IDENTITY = SessionIdentity(provider_name="fake", model="fake")
ACME = Membership("acme")
GLOBEX = Membership("globex")


def _directory() -> InMemoryMembershipDirectory:
    return InMemoryMembershipDirectory({"alice": [ACME], "carol": [ACME], "dave": [GLOBEX]})


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
        directory.set_memberships("alice", [ACME])
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


def test_each_person_sees_their_own_and_their_organizations() -> None:
    async def run() -> None:
        app = _app(_directory())
        personal = await _start(app, "alice", ACME, "personal")
        organization = await _start(app, "alice", ACME, "organization")

        assert await _visible(app, "alice") == {personal, organization}
        # Same organization.
        assert await _visible(app, "carol") == {organization}
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
            conversation_labels(subject="dave", membership=ACME, sharing="organization"),
            # As if someone else owned it.
            conversation_labels(subject="alice", membership=GLOBEX, sharing="personal"),
        ):
            with pytest.raises(SessionAccessDenied):
                await dave.sessions.create(
                    RunRequest(agent_name="assistant", messages=[], labels=labels),
                    identity=IDENTITY,
                )
        alice = await app.access("alice")
        own = conversation_labels(subject="alice", membership=ACME, sharing="organization")
        for labels in (
            # Labelled with another organization, or none.
            {**own, "organization": "globex"},
            {key: value for key, value in own.items() if key != "organization"},
            # Shared in a way the policy doesn't grant, or not said.
            {**own, "sharing": "team"},
            {key: value for key, value in own.items() if key != "sharing"},
            # Not even one she could read, shared with her organization by carol.
            conversation_labels(subject="carol", membership=ACME, sharing="organization"),
        ):
            with pytest.raises(SessionAccessDenied):
                await alice.sessions.create(
                    RunRequest(agent_name="assistant", messages=[], labels=labels),
                    identity=IDENTITY,
                )

    asyncio.run(run())


def test_sharing_labels_cant_be_changed_and_only_the_owner_changes_a_conversation() -> None:
    async def run() -> None:
        app = _app(_directory())
        shared = await _start(app, "alice", ACME, "organization")
        alice = await app.access("alice")
        loaded = await alice.sessions.load(shared)
        for key, value in (("sharing", "personal"), ("owner", "carol"), ("organization", "globex")):
            with pytest.raises(SessionAccessDenied):
                await alice.sessions.update_labels(shared, {**loaded.labels, key: value})
        carol = await app.access("carol")
        with pytest.raises(SessionAccessDenied):
            await carol.sessions.update_labels(shared, {**loaded.labels, "topic": "budget"})

    asyncio.run(run())


def test_a_removed_member_loses_access_at_their_next_step() -> None:
    async def run() -> None:
        directory = _directory()
        app = _app(directory)
        shared = await _start(app, "alice", ACME, "organization")
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
        labels = conversation_labels(subject="alice", membership=ACME, sharing="organization")
        shared = await start(app, "alice", labels)
        # Carol reads it, but continuing it would act as Alice.
        assert await _can_read(app, "carol", shared)
        with pytest.raises(ResourceAccessDenied):
            await resume(app, "carol", shared)
        await resume(app, "alice", shared)

    asyncio.run(run())


def test_a_conversation_shared_in_a_way_the_policy_doesnt_know_is_its_owners_alone() -> None:
    """Projects generated by Cayu 0.16.1 to 0.20.0 could share a conversation with a
    department. The policy grants nothing on departments, so such a conversation narrows to
    its owner: nobody else sees it, and the owner still reads and continues it."""

    async def run() -> None:
        directory = InMemoryMembershipDirectory(
            {"alice": [ACME], "carol": [ACME], "dave": [GLOBEX]}
        )
        app = _app(directory)
        labels = {
            "organization": "acme",
            "department": "acme/finance",
            "owner": "alice",
            "sharing": "department",
        }
        # Started before the project moved to the library, which would refuse to create it.
        created = await app.session_store.create(
            RunRequest(agent_name="assistant", messages=[], labels=labels), identity=IDENTITY
        )
        # Carol was in that department; neither she nor anyone else in acme or globex sees it.
        assert await _visible(app, "carol") == set()
        assert not await _can_read(app, "carol", created.id)
        assert not await _can_read(app, "dave", created.id)
        # Its owner does, and may still change it.
        assert await _visible(app, "alice") == {created.id}
        alice = await app.access("alice")
        await alice.sessions.update_labels(created.id, {**labels, "topic": "q3"})
        # Only inside her organizations: out of acme, it is gone for her too.
        directory.set_memberships("alice", [GLOBEX])
        assert not await _can_read(app, "alice", created.id)

    asyncio.run(run())


def test_the_most_memberships_a_person_may_have_still_fit_one_grant() -> None:
    # The longest organization IDs the tables hold.
    longest = [Membership(f"org{index:0>61}") for index in range(16)]

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


def test_a_conversation_is_shared_personally_or_with_the_organization() -> None:
    assert conversation_labels(subject="erin", membership=ACME, sharing="organization") == {
        "organization": "acme",
        "owner": "erin",
        "sharing": "organization",
    }
    assert set(SHARING_LABELS) == {"organization", "owner", "sharing"}
    with pytest.raises(ValueError, match="personal"):
        conversation_labels(subject="erin", membership=ACME, sharing="team")  # type: ignore[arg-type]


def _rule(**labels: list[str]) -> SessionAccessRule:
    return SessionAccessRule(
        selectors=tuple(
            SessionAccessSelector(key=key, values=tuple(values)) for key, values in labels.items()
        )
    )


class _Read:
    """A directory that answers with one snapshot already read."""

    def __init__(self, snapshot: AccessSnapshot) -> None:
        self._snapshot = snapshot

    async def current(self, subject: str) -> AccessSnapshot:
        return self._snapshot


class TeamAccessPolicy(ResourceAccessPolicy):
    """The example in docs/memberships.md: the library's grant, plus conversations shared
    with a team, which is this application's own grouping."""

    def __init__(self, directory: MembershipDirectory, teams) -> None:
        self._directory = directory
        # The application's own: `async teams(subject) -> [(organization, team), ...]`.
        self._teams = teams

    @property
    def authority(self) -> str:
        return "acme-teams-v1"

    async def resolve(self, subject: str) -> ResourceAccessDecision:
        # One read of the memberships, for the library's grant and the teams alike.
        snapshot = await self._directory.current(subject)
        decision = await MembershipAccessPolicy(_Read(snapshot), authority=self.authority).resolve(
            subject
        )
        organizations = {item.organization for item in snapshot.memberships}
        # Only teams of organizations the person still belongs to, each rule naming its
        # organization, so a team of the same name elsewhere never matches.
        teams = [(org, team) for org, team in await self._teams(subject) if org in organizations]
        if not teams:
            return decision
        read = tuple(_rule(organization=[o], team=[t], sharing=["team"]) for o, t in teams)
        create = tuple(
            _rule(organization=[o], owner=[subject], team=[t], sharing=["team"]) for o, t in teams
        )
        grant = decision.grant
        added = {
            family: replace(scope, read=(*scope.read, *read), create=(*scope.create, *create))
            for family, scope in (
                ("sessions", grant.sessions),
                ("tasks", grant.tasks),
                ("artifacts", grant.artifacts),
            )
            if scope is not None
        }
        return replace(decision, grant=replace(grant, **added))


def test_an_application_with_groups_of_its_own_writes_its_policy_over_the_same_directory() -> None:
    # Acme and globex each have a team called finance.
    teams = {
        "alice": [("acme", "finance")],
        "carol": [("acme", "finance")],
        "dave": [("globex", "finance")],
        "mallory": [("acme", "finance")],
    }

    async def teams_of(subject: str) -> list[tuple[str, str]]:
        return teams.get(subject, [])

    async def run() -> None:
        directory = InMemoryMembershipDirectory(
            {
                "alice": [ACME],
                "carol": [ACME],
                "bob": [ACME],
                "dave": [GLOBEX],
                # Removed from acme; the team row the application forgot grants nothing.
                "mallory": [GLOBEX],
            }
        )
        app = CayuApp(
            session_store=InMemorySessionStore(),
            resource_access_policy=TeamAccessPolicy(directory, teams_of),
        )
        alice = await app.access("alice")
        labels = {
            **conversation_labels(subject="alice", membership=ACME, sharing="personal"),
            "sharing": "team",
            "team": "finance",
        }
        created = await alice.sessions.create(
            RunRequest(agent_name="assistant", messages=[], labels=labels), identity=IDENTITY
        )
        assert await _can_read(app, "carol", created.id)
        for outsider in ("bob", "dave", "mallory"):
            assert not await _can_read(app, outsider, created.id), outsider
        # The team label is protected like the library's own.
        with pytest.raises(SessionAccessDenied):
            await alice.sessions.update_labels(created.id, {**labels, "team": "legal"})
        # Nobody starts a team conversation in another organization, or for a team of
        # theirs elsewhere.
        dave = await app.access("dave")
        for organization in ("acme", "globex"):
            forged = {**labels, "organization": organization, "owner": "dave"}
            if organization == "globex":
                forged["team"] = "legal"
            with pytest.raises(SessionAccessDenied):
                await dave.sessions.create(
                    RunRequest(agent_name="assistant", messages=[], labels=forged),
                    identity=IDENTITY,
                )

    asyncio.run(run())
