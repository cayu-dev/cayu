"""Files for the ``b2b`` preset of ``cayu new``.

The preset starts from the agent preset's files and the app-database convention, and adds
what an application serving several businesses needs: organizations, departments and
memberships in its own tables, and a policy that turns a person's current membership into
Cayu resource-access grants. The application decides who belongs where; Cayu enforces the
result on every read, listing and background step.
"""

from __future__ import annotations

from cayu.cli.scaffold_plan import ApplicationPlan

_MODELS_APPEND = '''

class Organization(Base):
    """A business this application serves. Its ID has no '/', which separates it from a
    department's own part in a department ID."""

    __tablename__ = "organizations"
    __table_args__ = (CheckConstraint("strpos(id, '/') = 0"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))


class Department(Base):
    """A department of one organization; its ID starts with the organization's, as in
    `acme/finance`, so it can never name another organization's department."""

    __tablename__ = "departments"
    __table_args__ = (
        # The target of a membership's (organization, department) reference.
        UniqueConstraint("organization_id", "id"),
        CheckConstraint("starts_with(id, organization_id || '/')"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    organization_id: Mapped[str] = mapped_column(
        ForeignKey(f"{SCHEMA}.organizations.id", ondelete="CASCADE")
    )
    name: Mapped[str] = mapped_column(String(200))


class Membership(Base):
    """One person's membership of an organization, and of at most one of its departments.

    The department must belong to the same organization; remove or move its members before
    deleting a department.
    """

    __tablename__ = "memberships"
    __table_args__ = (
        UniqueConstraint("subject", "organization_id"),
        ForeignKeyConstraint(
            ["organization_id", "department_id"],
            [f"{SCHEMA}.departments.organization_id", f"{SCHEMA}.departments.id"],
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    subject: Mapped[str] = mapped_column(String(200))
    organization_id: Mapped[str] = mapped_column(
        ForeignKey(f"{SCHEMA}.organizations.id", ondelete="CASCADE")
    )
    department_id: Mapped[str | None] = mapped_column(String(64))


class AccessRevision(Base):
    """Raised on every membership change, so Cayu rechecks access it admitted before it."""

    __tablename__ = "access_revision"

    id: Mapped[int] = mapped_column(primary_key=True)
    revision: Mapped[int] = mapped_column(BigInteger)
'''

_MIGRATION = '''"""Organizations, departments and memberships.

Revision ID: 0002
Revises: 0001
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "organizations",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.CheckConstraint("strpos(id, '/') = 0"),
        schema="app",
    )
    op.create_table(
        "departments",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column(
            "organization_id",
            sa.String(length=64),
            sa.ForeignKey("app.organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.UniqueConstraint("organization_id", "id"),
        sa.CheckConstraint("starts_with(id, organization_id || '/')"),
        schema="app",
    )
    op.create_table(
        "memberships",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("subject", sa.String(length=200), nullable=False),
        sa.Column(
            "organization_id",
            sa.String(length=64),
            sa.ForeignKey("app.organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("department_id", sa.String(length=64), nullable=True),
        sa.UniqueConstraint("subject", "organization_id"),
        sa.ForeignKeyConstraint(
            ["organization_id", "department_id"],
            ["app.departments.organization_id", "app.departments.id"],
        ),
        schema="app",
    )
    op.create_table(
        "access_revision",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("revision", sa.BigInteger(), nullable=False),
        schema="app",
    )


def downgrade() -> None:
    for table in ("access_revision", "memberships", "departments", "organizations"):
        op.drop_table(table, schema="app")
'''

_ORGANIZATIONS_PY = '''"""Organizations, departments and memberships: who belongs where, owned by this application.

Cayu keeps no membership of its own. `policies/access.py` turns the current memberships a
`MembershipDirectory` reports into Cayu resource-access grants, and Cayu asks again at every
protected step, so a membership change grants or revokes access at the person's next step. The application's tables hold them
(`database/memberships.py`); `InMemoryMembershipDirectory` serves tests.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Membership:
    organization: str
    department: str | None = None


@dataclass(frozen=True, slots=True)
class AccessSnapshot:
    """A person's current memberships, read together with the revision they belong to."""

    memberships: tuple[Membership, ...]
    revision: int


class MembershipDirectory(Protocol):
    async def current(self, subject: str) -> AccessSnapshot: ...


class InMemoryMembershipDirectory:
    """Memberships kept in memory, for tests and for trying the policy without PostgreSQL."""

    def __init__(
        self, memberships: dict[str, Iterable[Membership]] | None = None
    ) -> None:
        self._memberships = {
            subject: tuple(items) for subject, items in (memberships or {}).items()
        }
        self._revision = 0

    async def current(self, subject: str) -> AccessSnapshot:
        return AccessSnapshot(self._memberships.get(subject, ()), self._revision)

    def set_memberships(self, subject: str, memberships: Iterable[Membership]) -> None:
        self._memberships[subject] = tuple(memberships)
        self._revision += 1
'''

_MEMBERSHIPS_PY = '''"""Memberships in the application's own tables.

Change them through `add_member` and `remove_member`: each raises the access revision, so
Cayu refuses any decision read before the change.
"""

from __future__ import annotations

import asyncio

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from database import models
from database.session import session
from domain.organizations import AccessSnapshot, Membership

# The single row of `access_revision`.
_REVISION_ROW = 1


class DatabaseMembershipDirectory:
    """Reads memberships when Cayu asks; it connects on first use, never at startup."""

    async def current(self, subject: str) -> AccessSnapshot:
        return await asyncio.to_thread(self._current, subject)

    def _current(self, subject: str) -> AccessSnapshot:
        with session() as db:
            # One snapshot, so the memberships and their revision agree.
            db.connection(execution_options={"isolation_level": "REPEATABLE READ"})
            rows = db.execute(
                select(
                    models.Membership.organization_id, models.Membership.department_id
                )
                .where(models.Membership.subject == subject)
                .order_by(models.Membership.organization_id)
            ).all()
            revision = db.scalar(
                select(models.AccessRevision.revision).where(
                    models.AccessRevision.id == _REVISION_ROW
                )
            )
        return AccessSnapshot(
            memberships=tuple(
                Membership(organization, department)
                for organization, department in rows
            ),
            revision=revision or 0,
        )


def add_member(
    db: Session, *, subject: str, organization: str, department: str | None = None
) -> None:
    """Add `subject` to `organization`, or move them to `department` if already a member."""

    db.execute(
        insert(models.Membership)
        .values(subject=subject, organization_id=organization, department_id=department)
        .on_conflict_do_update(
            index_elements=[
                models.Membership.subject,
                models.Membership.organization_id,
            ],
            set_={"department_id": department},
        )
    )
    _raise_revision(db)


def remove_member(db: Session, *, subject: str, organization: str) -> None:
    """Remove `subject` from `organization`; their next protected step there is denied."""

    db.execute(
        delete(models.Membership).where(
            models.Membership.subject == subject,
            models.Membership.organization_id == organization,
        )
    )
    _raise_revision(db)


def _raise_revision(db: Session) -> None:
    db.execute(
        insert(models.AccessRevision)
        .values(id=_REVISION_ROW, revision=1)
        .on_conflict_do_update(
            index_elements=[models.AccessRevision.id],
            set_={"revision": models.AccessRevision.revision + 1},
        )
    )
'''

_SHARING_PY = '''"""How a new conversation is shared: the labels Cayu enforces, from verified membership.

Never take these labels from a request. Build them here from the signed-in person and a
membership read from `domain/organizations.py`; Cayu refuses any the person may not create.
"""

from __future__ import annotations

from typing import Literal

from domain.organizations import Membership

Sharing = Literal["personal", "department", "organization"]


def conversation_labels(
    *, subject: str, membership: Membership, sharing: Sharing
) -> dict[str, str]:
    labels = {
        "organization": membership.organization,
        "owner": subject,
        "sharing": sharing,
    }
    if sharing == "department":
        if membership.department is None:
            raise ValueError("Only a member of a department can share with it.")
        labels["department"] = membership.department
    return labels
'''

_ACCESS_POLICY_PY = '''"""Who may see and continue which conversations, from current membership.

Each person sees their own personal conversations, those shared with a department they
belong to, and those shared with an organization they belong to, and nothing of any other
organization. Only the owner continues, changes or deletes a conversation: seeing a shared
one doesn't let anyone else run it. Cayu applies this to sessions, tasks and artifacts on
every read, listing and background step, asking this policy again each time, so a removed
member loses access at their next step. See `uv run --no-sync cayu guide applications`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from cayu.resource_access import (
    ResourceAccessDecision,
    ResourceAccessGrant,
    ResourceAccessPolicy,
)
from cayu.sessions import SessionAccessRule, SessionAccessScope, SessionAccessSelector

from database.memberships import DatabaseMembershipDirectory
from domain.organizations import MembershipDirectory

# The longest Cayu may use one decision. It asks the policy again at every protected step.
DECISION_LIFETIME = timedelta(seconds=60)
# Cayu stores each admitted grant with the work it admits, up to 64 KiB. With the longest IDs
# (64 characters) a person's grant fits up to 20 memberships; this leaves room to spare.
MAX_MEMBERSHIPS = 16
# The labels that decide sharing; nobody may change them on an existing conversation.
SHARING_LABELS = ("organization", "department", "owner", "sharing")


def _rule(**labels: list[str]) -> SessionAccessRule:
    return SessionAccessRule(
        selectors=tuple(
            SessionAccessSelector(key=key, values=tuple(values))
            for key, values in labels.items()
        )
    )


class MembershipAccessPolicy(ResourceAccessPolicy):
    authority = "b2b-membership-v1"

    def __init__(self, directory: MembershipDirectory) -> None:
        self._directory = directory

    async def resolve(self, subject: str) -> ResourceAccessDecision:
        snapshot = await self._directory.current(subject)
        organizations = sorted({item.organization for item in snapshot.memberships})
        if not organizations:
            return ResourceAccessDecision(
                ResourceAccessGrant(), revision=snapshot.revision
            )
        if len(snapshot.memberships) > MAX_MEMBERSHIPS:
            raise ValueError(
                f"{subject!r} has more than {MAX_MEMBERSHIPS} memberships, more than one grant holds."
            )
        departments = sorted(
            {
                item.department
                for item in snapshot.memberships
                if item.department is not None
            }
        )
        owner = [subject]
        visible = [
            _rule(organization=organizations, owner=owner, sharing=["personal"]),
            _rule(organization=organizations, sharing=["organization"]),
        ]
        # A new conversation is always the person's own, inside their memberships.
        creatable = [
            _rule(
                organization=organizations,
                owner=owner,
                sharing=["personal", "organization"],
            )
        ]
        own = list(creatable)
        if departments:
            # A department conversation is decided by its department alone: a department ID
            # names its organization (`acme/finance`), so whatever organization label it
            # carries grants nothing.
            visible.append(_rule(department=departments, sharing=["department"]))
            creatable.append(
                _rule(
                    organization=organizations,
                    department=departments,
                    owner=owner,
                    sharing=["department"],
                )
            )
            own.append(
                _rule(department=departments, owner=owner, sharing=["department"])
            )
        scope = SessionAccessScope(
            read=tuple(visible),
            create=tuple(creatable),
            execute=tuple(own),
            modify=tuple(own),
            delete=tuple(own),
            update_labels=tuple(own),
            protected_label_keys=SHARING_LABELS,
        )
        return ResourceAccessDecision(
            ResourceAccessGrant(sessions=scope, tasks=scope, artifacts=scope),
            revision=snapshot.revision,
            valid_until=datetime.now(UTC) + DECISION_LIFETIME,
        )


def build_access_policy() -> MembershipAccessPolicy:
    return MembershipAccessPolicy(DatabaseMembershipDirectory())
'''

_DEMO_SEED_PY = '''"""Demonstration organizations and people: `uv run --no-sync python -m operations.demo_seed`.

Two organizations share this one database and agent, for local development and tests only:
it refuses a database that already holds other organizations. Running it again adds no
duplicates.
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from database import models
from database.memberships import add_member
from database.session import session

DEMO_ORGANIZATIONS = {"acme": "Acme", "globex": "Globex"}
DEMO_DEPARTMENTS = {
    "acme/finance": ("acme", "Finance"),
    "acme/legal": ("acme", "Legal"),
    "globex/support": ("globex", "Support"),
}
# (subject, organization, department)
DEMO_MEMBERS = (
    ("alice", "acme", "acme/finance"),
    ("carol", "acme", "acme/finance"),
    ("bob", "acme", "acme/legal"),
    ("dave", "globex", "globex/support"),
)


def seed(db: Session) -> None:
    others = db.scalar(
        select(func.count())
        .select_from(models.Organization)
        .where(models.Organization.id.not_in(DEMO_ORGANIZATIONS))
    )
    if others:
        raise RuntimeError(
            "This database holds real organizations; seed demonstration data elsewhere."
        )
    for identifier, name in DEMO_ORGANIZATIONS.items():
        db.execute(
            insert(models.Organization)
            .values(id=identifier, name=name)
            .on_conflict_do_nothing()
        )
    for identifier, (organization, name) in DEMO_DEPARTMENTS.items():
        db.execute(
            insert(models.Department)
            .values(id=identifier, organization_id=organization, name=name)
            .on_conflict_do_nothing()
        )
    for subject, organization, department in DEMO_MEMBERS:
        add_member(
            db, subject=subject, organization=organization, department=department
        )


def main() -> None:
    with session() as db:
        seed(db)
        db.commit()
    print(
        f"Seeded {len(DEMO_ORGANIZATIONS)} organizations and {len(DEMO_MEMBERS)} members."
    )


if __name__ == "__main__":
    main()
'''

_CONFTEST_PY = '''"""Fixtures shared by the PostgreSQL tests."""

from test_app_database import database  # noqa: F401  (a fresh PostgreSQL database per test)
'''

_TEST_ORGANIZATIONS_PY = '''"""Organizations share one database and agent; each person sees only what membership allows.

The policy tests run without a database. The PostgreSQL ones (`CAYU_TEST_POSTGRES_DSN`) read
memberships from the application's own tables.
"""

from __future__ import annotations

import asyncio

import pytest
from alembic import command
from sqlalchemy.exc import IntegrityError
from test_app_database import ALEMBIC, requires_postgres

from app import build_app
from cayu import CayuApp, Message, ResumeRequest
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.resource_access import ResourceAccessDenied
from cayu.sessions import InMemorySessionStore
from cayu.sessions.access import SessionAccessDenied
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import RunRequest
from cayu.tasks import InMemoryTaskStore
from database import models
from database.memberships import DatabaseMembershipDirectory, add_member, remove_member
from database.session import session
from domain.organizations import (
    InMemoryMembershipDirectory,
    Membership,
    MembershipDirectory,
)
from domain.sharing import Sharing, conversation_labels
from operations.demo_seed import seed
from policies.access import MAX_MEMBERSHIPS, MembershipAccessPolicy

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


async def _start(
    app: CayuApp, subject: str, membership: Membership, sharing: Sharing
) -> str:
    labels = conversation_labels(
        subject=subject, membership=membership, sharing=sharing
    )
    access = await app.access(subject)
    created = await access.sessions.create(
        RunRequest(agent_name="assistant", messages=[], labels=labels),
        identity=IDENTITY,
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
            conversation_labels(
                subject="dave", membership=ALICE, sharing="organization"
            ),
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
        mislabelled = conversation_labels(
            subject="alice", membership=ALICE, sharing="department"
        )
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
            labels = conversation_labels(
                subject="carol", membership=CAROL, sharing=sharing
            )
            with pytest.raises(SessionAccessDenied):
                await alice.sessions.create(
                    RunRequest(agent_name="assistant", messages=[], labels=labels),
                    identity=IDENTITY,
                )

    asyncio.run(run())


def test_sharing_labels_cant_be_changed_and_only_the_owner_changes_a_conversation() -> (
    None
):
    async def run() -> None:
        app = _app(_directory())
        shared = await _start(app, "alice", ALICE, "department")
        alice = await app.access("alice")
        loaded = await alice.sessions.load(shared)
        with pytest.raises(SessionAccessDenied):
            await alice.sessions.update_labels(
                shared, {**loaded.labels, "sharing": "organization"}
            )
        carol = await app.access("carol")
        with pytest.raises(SessionAccessDenied):
            await carol.sessions.update_labels(
                shared, {**loaded.labels, "topic": "budget"}
            )

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

    async def start(app, subject: str, labels: dict[str, str]) -> str:
        access = await app.access(subject)
        request = RunRequest(
            agent_name="__AGENT_NAME__",
            messages=[Message.text("user", "Summarize the budget")],
            labels=labels,
        )
        async for event in access.run(request):
            session_id = event.session_id
        return session_id

    async def resume(app, subject: str, session_id: str) -> None:
        access = await app.access(subject)
        request = ResumeRequest(
            session_id=session_id, messages=[Message.text("user", "And next month?")]
        )
        async for _ in access.resume(request):
            pass

    async def run() -> None:
        async with build_app(
            provider=provider,
            session_store=InMemorySessionStore(),
            task_store=InMemoryTaskStore(),
            access_policy=MembershipAccessPolicy(_directory()),
        ) as app:
            labels = conversation_labels(
                subject="alice", membership=ALICE, sharing="department"
            )
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
            RunRequest(agent_name="assistant", messages=[], labels=labels),
            identity=IDENTITY,
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


def test_the_most_memberships_a_person_may_have_still_fit_one_grant() -> None:
    longest = [
        Membership(f"org{index:0>61}", f"dep{index:0>61}")
        for index in range(MAX_MEMBERSHIPS)
    ]

    async def run() -> None:
        app = _app(InMemoryMembershipDirectory({"erin": longest}))
        await app.access("erin")
        too_many = InMemoryMembershipDirectory(
            {"erin": [*longest, Membership("one-more")]}
        )
        with pytest.raises(ValueError, match="memberships"):
            await MembershipAccessPolicy(too_many).resolve("erin")

    asyncio.run(run())


def test_department_sharing_needs_a_department() -> None:
    with pytest.raises(ValueError, match="department"):
        conversation_labels(
            subject="erin", membership=Membership("acme"), sharing="department"
        )


@requires_postgres
def test_memberships_and_revocation_come_from_the_applications_tables(database) -> None:
    async def run() -> None:
        app = _app(DatabaseMembershipDirectory())
        shared = await _start(app, "alice", ALICE, "department")
        assert await _can_read(app, "carol", shared)
        assert not await _can_read(app, "bob", shared)
        assert not await _can_read(app, "dave", shared)

        with session() as db:
            remove_member(db, subject="carol", organization="acme")
            db.commit()

        assert not await _can_read(app, "carol", shared)

    command.upgrade(ALEMBIC, "head")
    with session() as db:
        seed(db)
        seed(db)
        db.commit()
    asyncio.run(run())


@requires_postgres
def test_a_membership_cant_name_another_organizations_department(database) -> None:
    command.upgrade(ALEMBIC, "head")
    with session() as db:
        seed(db)
        db.commit()
    with session() as db, pytest.raises(IntegrityError):
        add_member(
            db, subject="mallory", organization="globex", department="acme/finance"
        )
        db.commit()
    # A department ID always starts with its organization's, so none is ever reused across them.
    with session() as db, pytest.raises(IntegrityError):
        db.add(
            models.Department(id="finance", organization_id="globex", name="Finance")
        )
        db.commit()
    # Nor can an organization's ID look like another's department prefix.
    with session() as db, pytest.raises(IntegrityError):
        db.add(models.Organization(id="acme/eu", name="Acme EU"))
        db.commit()
'''


_GUIDANCE = """
## Organizations and sharing

Several businesses share this one deployment, database and agent. The application decides
who belongs where; Cayu enforces it.

| Where | What it owns |
| --- | --- |
| `database/models.py`, `migrations/` | Organizations, departments (each in one organization, its ID starting with the organization's, as in `acme/finance`) and memberships, in the `app` schema |
| `database/memberships.py` | Reading memberships for Cayu; `add_member` and `remove_member`, the only way to change them |
| `policies/access.py` | Turning a person's current memberships into Cayu resource-access grants |
| `domain/sharing.py` | The labels a new conversation is created with: personal, department or organization |

- Start a conversation through `await app.access(subject)` with the labels
  `conversation_labels(...)` returns for the signed-in person. Never take a subject, an
  organization or labels from the request itself.
- Everyone sees their own conversations, their department's and their organization's, and
  nothing of another organization. Only the owner continues, changes or deletes one.
- Cayu asks the policy again at every protected step, so removing a member takes effect at
  their next step, in background work too.
- `uv run --no-sync alembic upgrade head`, then `uv run --no-sync python -m
  operations.demo_seed`, gives a development database two demonstration organizations.
  `tests/test_organizations.py` shows each rule.
"""


def b2b_project_files(*, files: dict[str, str], plan: ApplicationPlan) -> dict[str, str]:
    """The b2b additions on top of the agent preset's and app-database's files."""

    models = files["database/models.py"].replace(
        "from sqlalchemy import DateTime, MetaData, String, func\n",
        "from sqlalchemy import (\n    BigInteger,\n    CheckConstraint,\n    DateTime,\n    ForeignKey,\n"
        "    ForeignKeyConstraint,\n    MetaData,\n    String,\n    UniqueConstraint,\n"
        "    func,\n)\n",
    )
    app = (
        files["app.py"]
        .replace(
            "from memory.context import build_context_policy\n",
            "from memory.context import build_context_policy\n"
            "from policies.access import build_access_policy\n",
        )
        .replace(
            "from cayu.model_policy import configured_model_policy\n",
            "from cayu.model_policy import configured_model_policy\n"
            "from cayu.resource_access import ResourceAccessPolicy\n",
        )
        .replace(
            "    artifact_store: ArtifactStore | None = None,\n) -> CayuApp:",
            "    artifact_store: ArtifactStore | None = None,\n"
            "    access_policy: ResourceAccessPolicy | None = None,\n) -> CayuApp:",
        )
        .replace(
            "        enable_logging=runtime.enable_logging,\n",
            "        enable_logging=runtime.enable_logging,\n"
            "        resource_access_policy=(\n"
            "            access_policy if access_policy is not None else build_access_policy()\n"
            "        ),\n",
        )
    )
    return {
        "README.md": files["README.md"] + _GUIDANCE,
        "AGENTS.md": files["AGENTS.md"] + _GUIDANCE,
        "app.py": app,
        "database/models.py": models + _MODELS_APPEND,
        "migrations/versions/0002_organizations.py": _MIGRATION,
        "domain/organizations.py": _ORGANIZATIONS_PY,
        "database/memberships.py": _MEMBERSHIPS_PY,
        "domain/sharing.py": _SHARING_PY,
        "policies/access.py": _ACCESS_POLICY_PY,
        "operations/demo_seed.py": _DEMO_SEED_PY,
        "tests/conftest.py": _CONFTEST_PY,
        "tests/test_organizations.py": _TEST_ORGANIZATIONS_PY.replace(
            "__AGENT_NAME__", plan.agent_name
        ),
    }
