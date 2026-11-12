"""Files for the ``b2b`` preset of ``cayu new``.

The preset starts from the agent preset's files and the app-database convention, and adds
what an application serving several businesses needs: organizations, departments and
memberships in its own tables, and a policy that turns a person's current membership into
Cayu resource-access grants. The application decides who belongs where; Cayu enforces the
result on every read, listing and background step.
"""

from __future__ import annotations

import html
from importlib.resources import files as package_files

from cayu._operator_credentials import (
    ENVIRONMENT_OPERATOR_AUTH_TARGET,
    ENVIRONMENT_OPERATOR_ONLY_AUTH_TARGET,
)
from cayu.cli.scaffold_app_routes import app_route_files
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
        CheckConstraint("role in ('member', 'admin')"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    subject: Mapped[str] = mapped_column(String(200))
    organization_id: Mapped[str] = mapped_column(
        ForeignKey(f"{SCHEMA}.organizations.id", ondelete="CASCADE")
    )
    department_id: Mapped[str | None] = mapped_column(String(64))
    # `admin` may add, move and remove the organization's members. It is this application's
    # decision only: it never widens what Cayu lets anyone see.
    role: Mapped[str] = mapped_column(String(16), server_default="member")


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
from typing import Literal, Protocol


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


def app_role(roles: Iterable[str]) -> Literal["admin", "member"]:
    """The role the web app shows a person: `admin` when they are an admin of any of their
    organizations, else `member`.

    It only words the profile menu and hides what they can't do; every change is still
    checked against the one organization it touches (`role` on that membership).
    """

    return "admin" if "admin" in roles else "member"
'''

_MEMBERSHIPS_PY = '''"""Memberships in the application's own tables.

Change them through `add_member` and `remove_member`: each raises the access revision, so
Cayu refuses any decision read before the change.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

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


@dataclass(frozen=True, slots=True)
class MemberRecord:
    subject: str
    organization: str
    department: str | None
    role: str


def add_member(
    db: Session,
    *,
    subject: str,
    organization: str,
    department: str | None = None,
    role: str = "member",
) -> None:
    """Add `subject` to `organization`, or move them to `department` and `role` if already
    a member."""

    db.execute(
        insert(models.Membership)
        .values(
            subject=subject,
            organization_id=organization,
            department_id=department,
            role=role,
        )
        .on_conflict_do_update(
            index_elements=[
                models.Membership.subject,
                models.Membership.organization_id,
            ],
            set_={"department_id": department, "role": role},
        )
    )
    _raise_revision(db)


def admins_of(db: Session, organization: str, *, lock: bool = False) -> set[str]:
    """The organization's admins; `lock=True` holds their rows until the transaction ends."""

    query = select(models.Membership.subject).where(
        models.Membership.organization_id == organization,
        models.Membership.role == "admin",
    )
    if lock:
        query = query.with_for_update()
    return set(db.scalars(query))


def memberships_of(db: Session, subject: str) -> list[MemberRecord]:
    return _records(db, models.Membership.subject == subject)


def members_of(db: Session, organization: str) -> list[MemberRecord]:
    return _records(db, models.Membership.organization_id == organization)


def _records(db: Session, condition) -> list[MemberRecord]:
    rows = db.execute(
        select(
            models.Membership.subject,
            models.Membership.organization_id,
            models.Membership.department_id,
            models.Membership.role,
        )
        .where(condition)
        .order_by(models.Membership.organization_id, models.Membership.subject)
    ).all()
    return [MemberRecord(*row) for row in rows]


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
# (subject, organization, department, role)
DEMO_MEMBERS = (
    ("alice", "acme", "acme/finance", "admin"),
    ("carol", "acme", "acme/finance", "member"),
    ("bob", "acme", "acme/legal", "member"),
    ("dave", "globex", "globex/support", "admin"),
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
    for subject, organization, department, role in DEMO_MEMBERS:
        add_member(
            db,
            subject=subject,
            organization=organization,
            department=department,
            role=role,
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


_ROLES_MIGRATION = '''"""Membership roles: `member` or `admin`.

Revision ID: 0003
Revises: 0002
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "memberships",
        sa.Column(
            "role", sa.String(length=16), server_default="member", nullable=False
        ),
        schema="app",
    )
    op.create_check_constraint(
        "memberships_role_check",
        "memberships",
        "role in ('member', 'admin')",
        schema="app",
    )


def downgrade() -> None:
    op.drop_constraint(
        "memberships_role_check", "memberships", schema="app", type_="check"
    )
    op.drop_column("memberships", "role", schema="app")
'''

_SETTINGS_APPEND = """


def configured_sign_in() -> str:
    \"\"\"How people sign in to the application's routes outside `cayu serve --dev`: `cloud`,
    Cayu Cloud's sign-in (the default), or `oidc`, a bearer token your identity provider
    issues (CAYU_OIDC_ISSUER and CAYU_OIDC_AUDIENCE).\"\"\"

    value = (os.environ.get("APP_SIGN_IN") or "cloud").strip()
    if value not in {"cloud", "oidc"}:
        raise ValueError("APP_SIGN_IN must be `cloud` or `oidc`.")
    return value
"""

_SIGN_IN_PY = '''"""How a request becomes a verified person: the one sign-in seam every route uses.

`build_authenticate` returns `authenticate(request) -> subject`.

- `cayu serve --dev` signs in the demonstration people of `operations/demo_seed.py` with a
  signed cookie, and nobody else. Nothing else serves it.
- Otherwise `APP_SIGN_IN` selects Cayu Cloud's sign-in (`cloud`) or a bearer token from your
  identity provider (`oidc`). The subject is that provider's ID for the person, so memberships
  are keyed on it. An organization claim is never trusted: membership comes from this
  application's tables.

Until a production sign-in is configured, every route answers 503 and lets nobody in.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
from collections.abc import Awaitable, Callable

from fastapi import HTTPException, Request

from cayu.server.auth import AuthConfigurationError
from cayu.server.oidc import OidcBearerAuth
from cayu.server.proxy_identity import ProxyIdentityAuth
from configuration.settings import configured_sign_in
from operations.demo_seed import DEMO_MEMBERS

logger = logging.getLogger(__name__)

Authenticate = Callable[[Request], Awaitable[str]]
DEMO_COOKIE = "demo_session"
DEMO_SUBJECTS = frozenset(member[0] for member in DEMO_MEMBERS)


class DemoSignIn:
    """Development only: a signed cookie naming one demonstration person. Its key lives in
    this process alone, so a restart signs everyone out."""

    def __init__(self) -> None:
        self._key = secrets.token_bytes(32)

    def cookie(self, subject: str) -> str:
        if subject not in DEMO_SUBJECTS:
            raise HTTPException(404, "No such demonstration person.")
        return f"{subject}.{self._signature(subject)}"

    async def __call__(self, request: Request) -> str:
        subject, _, signature = request.cookies.get(DEMO_COOKIE, "").partition(".")
        if subject in DEMO_SUBJECTS and hmac.compare_digest(
            signature, self._signature(subject)
        ):
            return subject
        raise HTTPException(401, "Sign in first.")

    def _signature(self, subject: str) -> str:
        return hmac.new(self._key, subject.encode(), hashlib.sha256).hexdigest()


def build_authenticate(*, development: bool) -> Authenticate:
    if development:
        return DemoSignIn()
    try:
        if configured_sign_in() == "oidc":
            return _oidc(OidcBearerAuth.from_environment())
        return _cloud(ProxyIdentityAuth.from_environment())
    except AuthConfigurationError as exc:
        return _unavailable(str(exc))


def _oidc(verifier: OidcBearerAuth) -> Authenticate:
    async def authenticate(request: Request) -> str:
        return (await verifier(request)).subject

    return authenticate


def _cloud(proxy: ProxyIdentityAuth) -> Authenticate:
    # Only the identity Cayu Cloud's router signed, never its operator fallback.
    person = proxy.product_dependency()

    async def authenticate(request: Request) -> str:
        return (await person(request)).subject_id

    return authenticate


def _unavailable(reason: str) -> Authenticate:
    # The reason names settings, so it goes to the log, not to whoever is calling.
    logger.warning("The application's sign-in isn't configured: %s", reason)

    async def authenticate(request: Request) -> str:
        raise HTTPException(503, "Sign-in isn't configured.")

    return authenticate
'''

_HTTP_PY = '''"""THE AGENT'S OWN. The application's HTTP API, which `cayu serve` serves under /app
([tool.cayu.serve] routers in pyproject.toml).

- /app/resources/...: Cayu's scoped routes for conversations, tasks and runs. Each answers
  only what the signed-in person may see; a foreign ID looks missing.
- /app/me and /app/conversation-labels: who is signed in, their memberships, and the labels a
  new conversation they share takes. The active organization a client shows is only a filter
  and a default: it never grants anything.
- /app/organizations/{organization}/members: the members, which an organization's admins
  add, move and remove.
- /app/whoami, /app/settings, /app/overview, /app/task-runs and /app/inbox: the routes every
  Agent's web app reads (integrations/http/generated/), given this application's people and
  roles (`person` below), its overview metrics (`overview_metrics.py`) and the settings it
  reads from its environment (`fixed_settings.py`).
- /app/dev/sign-in and /app/dev/people: the demonstration login and sign-out, served
  only by `cayu serve --dev`.
"""

from __future__ import annotations

import asyncio
import os
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError

from cayu import CayuApp
from cayu.server.resource_access import create_resource_router
from database.memberships import (
    MemberRecord,
    add_member,
    admins_of,
    members_of,
    memberships_of,
    remove_member,
)
from database.session import session
from domain.organizations import Membership, app_role
from domain.sharing import Sharing, conversation_labels
from integrations.http.fixed_settings import fixed_settings
from integrations.http.generated import Person, build_generated_router
from integrations.http.overview_metrics import overview_metrics
from integrations.sign_in import (
    DEMO_COOKIE,
    DEMO_SUBJECTS,
    DemoSignIn,
    build_authenticate,
)


# The length of `memberships.subject`.
MEMBER_SUBJECT_LENGTH = 200
# The agent the web interface starts conversations with.
AGENT_NAME = "__AGENT_NAME__"
# The Agent's application in Cayu Cloud when Cloud doesn't say (CAYU_CLOUD_APPLICATION_ID):
# the slug `cayu cloud init` makes from the project's name.
CLOUD_APPLICATION = "__CLOUD_APPLICATION__"


class DemoSignInRequest(BaseModel):
    subject: str


class LabelsRequest(BaseModel):
    organization: str
    sharing: Sharing


class MemberRequest(BaseModel):
    department: str | None = None
    role: Literal["member", "admin"] = "member"


def build_router(app: CayuApp, *, development: bool) -> APIRouter:
    authenticate = build_authenticate(development=development)
    router = APIRouter(prefix="/app")
    router.include_router(create_resource_router(app, authenticate=authenticate))

    async def signed_in(request: Request) -> str:
        return await authenticate(request)

    async def person(request: Request) -> Person:
        """The signed-in person for the generated routes. Their role is `admin` when they
        administer any of their organizations; changing members is still checked per
        organization.

        They decide Cayu's inbox items through `/app/inbox` only under `cayu serve --dev`,
        where Cayu's API is open. Deployed, `OPERATOR_ONLY_AUTH` refuses the members Cayu
        Cloud signs in, so the Agent's operators decide on the Developer dashboard; Cayu
        refuses or allows either way."""

        subject = await authenticate(request)
        role = app_role(record.role for record in await _read(memberships_of, subject))
        return Person(
            subject=subject,
            role=role,
            may_decide=development,
            may_change=role == "admin",
        )

    router.include_router(
        build_generated_router(
            app,
            person=person,
            application=cloud_application(),
            overview_metrics=overview_metrics,
            fixed_settings=fixed_settings,
            agent_name=AGENT_NAME,
        )
    )

    @router.get("/me")
    async def me(subject: str = Depends(signed_in)) -> dict:
        records = await _read(memberships_of, subject)
        return {
            "subject": subject,
            "agent_name": AGENT_NAME,
            "memberships": [_member(item) for item in records],
        }

    @router.post("/conversation-labels")
    async def labels(body: LabelsRequest, subject: str = Depends(signed_in)) -> dict:
        record = await _membership(subject, body.organization)
        if record is None:
            raise HTTPException(403, "You aren't a member of that organization.")
        membership = Membership(record.organization, record.department)
        try:
            return conversation_labels(
                subject=subject, membership=membership, sharing=body.sharing
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    @router.get("/organizations/{organization}/members")
    async def members(
        organization: str, subject: str = Depends(signed_in)
    ) -> list[dict]:
        if await _membership(subject, organization) is None:
            raise HTTPException(404, "Organization unavailable.")
        return [_member(item) for item in await _read(members_of, organization)]

    # A member ID is the identity provider's opaque subject, which may contain '/'.
    @router.put("/organizations/{organization}/members/{member:path}")
    async def put_member(
        organization: str,
        member: str,
        body: MemberRequest,
        subject: str = Depends(signed_in),
    ) -> dict:
        await _require_admin(subject, organization)
        _require_subject(member)
        try:
            await _change_members(
                subject,
                add_member,
                subject=member,
                organization=organization,
                department=body.department,
                role=body.role,
            )
        except IntegrityError:
            raise HTTPException(
                422, "The department must be one of this organization's."
            ) from None
        return {"subject": member, "organization": organization, **body.model_dump()}

    @router.delete(
        "/organizations/{organization}/members/{member:path}", status_code=204
    )
    async def delete_member(
        organization: str, member: str, subject: str = Depends(signed_in)
    ) -> Response:
        await _require_admin(subject, organization)
        _require_subject(member)
        await _change_members(
            subject,
            remove_member,
            subject=member,
            organization=organization,
        )
        return Response(status_code=204)

    if isinstance(authenticate, DemoSignIn):
        demo = authenticate

        @router.get("/dev/people")
        async def demo_people() -> list[str]:
            return sorted(DEMO_SUBJECTS)

        @router.post("/dev/sign-in", status_code=204)
        async def demo_sign_in(body: DemoSignInRequest) -> Response:
            response = Response(status_code=204)
            response.set_cookie(
                DEMO_COOKIE, demo.cookie(body.subject), httponly=True, samesite="strict"
            )
            return response

        @router.delete("/dev/sign-in", status_code=204)
        async def demo_sign_out() -> Response:
            response = Response(status_code=204)
            response.delete_cookie(DEMO_COOKIE, httponly=True, samesite="strict")
            return response

    return router


def cloud_application() -> str:
    """The application the Settings page's `cayu cloud settings` commands name: the one
    Cayu Cloud runs this Agent as, else the one `cayu cloud init` would create. Never the
    specialist's name, which can differ from both."""

    return os.environ.get("CAYU_CLOUD_APPLICATION_ID", "").strip() or CLOUD_APPLICATION


def _member(record: MemberRecord) -> dict:
    return {
        "subject": record.subject,
        "organization": record.organization,
        "department": record.department,
        "role": record.role,
    }


async def _membership(subject: str, organization: str) -> MemberRecord | None:
    records = await _read(memberships_of, subject)
    return next((item for item in records if item.organization == organization), None)


async def _require_admin(subject: str, organization: str) -> None:
    record = await _membership(subject, organization)
    if record is None:
        raise HTTPException(404, "Organization unavailable.")
    if record.role != "admin":
        raise HTTPException(403, "Only the organization's admins manage its members.")


async def _read(query, value):
    def read():
        with session() as db:
            return query(db, value)

    return await asyncio.to_thread(read)


def _require_subject(member: str) -> None:
    if not member or member != member.strip() or len(member) > MEMBER_SUBJECT_LENGTH:
        raise HTTPException(
            422,
            f"A member is a non-blank ID of at most {MEMBER_SUBJECT_LENGTH} characters.",
        )
    # A URL can't carry these segments as they are: clients resolve them first.
    if {".", ".."} & set(member.split("/")):
        raise HTTPException(422, "A member ID has no '.' or '..' path segments.")


async def _change_members(caller: str, change, **arguments) -> None:
    """Change the organization's members while its admins' rows are locked, so the caller
    is still an admin when it happens and at least one admin remains afterwards."""

    organization = arguments["organization"]

    def write() -> None:
        with session() as db:
            if caller not in admins_of(db, organization, lock=True):
                raise HTTPException(
                    403, "Only the organization's admins manage its members."
                )
            change(db, **arguments)
            if not admins_of(db, organization):
                db.rollback()
                raise HTTPException(409, "An organization keeps at least one admin.")
            db.commit()

    await asyncio.to_thread(write)
'''


_FIXED_SETTINGS_PY = '''"""THE AGENT'S OWN. The settings this Agent reads from its environment when a process starts.

The web app's Settings page lists them under "Need a redeploy", next to the managed
settings Cayu Cloud changes while the Agent runs. Secrets (API keys, the database URL) are
left out: the page would show their values.
"""

from __future__ import annotations

from configuration.settings import (
    configured_model,
    configured_provider_choice,
    configured_sign_in,
)
from integrations.http.generated import FixedSetting


def fixed_settings() -> tuple[FixedSetting, ...]:
    return (
        FixedSetting(
            name="provider",
            env_var="CAYU_PROVIDER",
            label="Model provider",
            help="The provider the Agent's model calls go to. Unset, the Agent has none.",
            why="Chosen when each process builds the app; a change needs a redeploy.",
            current=configured_provider_choice,
        ),
        FixedSetting(
            name="model",
            env_var="CAYU_MODEL",
            label="Model",
            help="The default model, until Cayu Cloud binds the primary model slot.",
            why="The agent is built with its model when each process starts; a change needs a redeploy.",
            current=configured_model,
        ),
        FixedSetting(
            name="sign_in",
            env_var="APP_SIGN_IN",
            label="Sign-in",
            help="How people sign in to /app outside `cayu serve --dev`: cloud (Cayu Cloud's sign-in) or oidc (a bearer token).",
            why="Read when the routes are built; a change needs a redeploy.",
            current=configured_sign_in,
        ),
    )
'''


_OVERVIEW_METRICS_PY = '''"""THE AGENT'S OWN. The metrics on the web app's Overview, for the signed-in person.

Several organizations share this Agent, so every number counts only what the person may
see: their organizations and their members from the application's tables, and the
conversations and tasks Cayu's resource-access policy (`policies/access.py`) lets them read.
The generated `GET /app/overview` (integrations/http/generated/overview.py) calls
`overview_metrics`.
"""

from __future__ import annotations

import asyncio

from cayu.sessions import SessionQuery
from cayu.tasks import TaskQuery

from database.memberships import members_of, memberships_of
from database.session import session
from domain.organizations import Membership
from integrations.http.generated import Metric, OverviewContext

# Tasks counted at most; more shows as this many and a plus.
TASK_COUNT_LIMIT = 1000


async def overview_metrics(context: OverviewContext) -> list[Metric]:
    subject = context.person.subject
    organizations, members = await asyncio.to_thread(_people, subject)
    access = await context.app.access(subject)
    visible = await access.sessions.list_sessions(SessionQuery(limit=1, include_total_count=True))
    own = await access.sessions.list_sessions(
        SessionQuery(limit=1, include_total_count=True, labels={"owner": subject})
    )
    tasks = (
        []
        if context.app.task_store is None
        else await access.tasks.list(TaskQuery(limit=TASK_COUNT_LIMIT))
    )
    return [
        Metric(
            label="Organizations",
            value=str(len(organizations)),
            description=", ".join(_where(item) for item in organizations) or "No organization has added you yet.",
        ),
        Metric(
            label="Members",
            value=str(members),
            description="People in your organizations, you included.",
        ),
        Metric(
            label="Conversations",
            value=str(visible.total_count or 0),
            description=(
                f"{own.total_count or 0} started by you, the rest shared with your department "
                "or organization."
            ),
        ),
        Metric(
            label="Tasks",
            value=f"{len(tasks)}{'+' if len(tasks) == TASK_COUNT_LIMIT else ''}",
            description="Background work you can see. This Agent queues none of its own yet.",
        ),
    ]


def _people(subject: str) -> tuple[list[Membership], int]:
    """The person's memberships, and how many people their organizations have together."""

    with session() as db:
        records = memberships_of(db, subject)
        people = {
            member.subject
            for record in records
            for member in members_of(db, record.organization)
        }
    return [Membership(record.organization, record.department) for record in records], len(people)


def _where(membership: Membership) -> str:
    return membership.department or membership.organization
'''


_TEST_GENERATED_ROUTES_PY = '''"""The generated web-app routes: whoami, settings, overview, task runs and the inbox.

These run without a database, on a stand-in person; `tests/test_http_api.py` checks them
with this application's own people and metrics.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from cayu import CayuApp
from cayu.managed_settings import (
    ManagedSettings,
    ManagedSettingsChannel,
    ManagedSettingsConflict,
    ManagedSettingsSnapshot,
)
from cayu.server import ServerConfig, create_server
from cayu.server.auth import AuthContext
from cayu.server.config import OpenAccess
from cayu.sessions import InMemorySessionStore
from cayu.tasks import InMemoryTaskStore
from cayu.tasks.creation import TaskCreate
from cayu.tasks.worker import run_task_worker
from domain.organizations import InMemoryMembershipDirectory, Membership
from domain.sharing import conversation_labels
from integrations.http.generated import (
    FixedSetting,
    Metric,
    OverviewContext,
    Person,
    build_generated_router,
)
from integrations.http.generated.task_runs import TaskRun, list_task_runs
from policies.access import MembershipAccessPolicy

ALICE = Membership("acme", "acme/finance")
PEOPLE = {
    "alice": Person(subject="alice", role="admin", may_decide=False, may_change=True),
    "bob": Person(subject="bob", role="member", may_decide=False, may_change=False),
}


async def _person(request: Request) -> Person:
    found = PEOPLE.get(request.headers.get("x-person", ""))
    if found is None:
        raise HTTPException(401, "Sign in first.")
    return found


async def _metrics(context: OverviewContext) -> list[Metric]:
    return [Metric(label="Seen by", value=context.person.subject, description=None)]


def _fixed() -> list[FixedSetting]:
    return [
        FixedSetting(
            name="region",
            env_var="TEST_REGION",
            label="Region",
            help="Where it runs.",
            why="Read at start.",
            current=lambda: "eu",
        )
    ]


def _routes(app: CayuApp):
    return build_generated_router(
        app,
        person=_person,
        application="example-agent",
        overview_metrics=_metrics,
        fixed_settings=_fixed,
        agent_name="example-agent",
        inbox_page="/inbox",
    )


def _client(app: CayuApp, *, server: FastAPI | None = None) -> TestClient:
    server = server or FastAPI()
    server.include_router(_routes(app), prefix="/app")
    return TestClient(server)


def _app(**stores: Any) -> CayuApp:
    directory = InMemoryMembershipDirectory({"alice": [ALICE], "bob": [Membership("globex")]})
    return CayuApp(
        session_store=InMemorySessionStore(),
        resource_access_policy=MembershipAccessPolicy(directory),
        **stores,
    )


def test_every_route_needs_a_signed_in_person() -> None:
    with _client(_app(task_store=InMemoryTaskStore())) as client:
        for path in ("whoami", "settings", "overview", "task-runs", "inbox", "inbox/counts"):
            assert client.get(f"/app/{path}").status_code == 401, path
        assert client.post("/app/inbox/item/settle", json={}).status_code == 401


def test_whoami_says_who_is_signed_in_and_what_they_may_do() -> None:
    with _client(_app()) as client:
        answer = client.get("/app/whoami", headers={"x-person": "alice"}).json()
    assert answer == {
        "person": {"subject": "alice", "name": None, "email": None},
        "role": "admin",
        "may_decide": False,
        "may_change": True,
    }


def test_settings_list_cayus_settings_with_their_release_values() -> None:
    with _client(_app()) as client:
        view = client.get("/app/settings", headers={"x-person": "bob"}).json()
    rows = {row["key"]: row for row in view["settings"]}
    steps = rows["config.run.max_steps"]
    assert steps["value"] == 1000 and steps["source"] == "default" and steps["kind"] == "int"
    assert steps["group"] == "Runs"
    assert steps["change"] == (
        "cayu cloud settings set --application example-agent config.run.max_steps=1000"
    )
    assert steps["reset"] == (
        "cayu cloud settings unset --application example-agent config.run.max_steps"
    )
    unset = rows["config.run.limits.max_tool_calls"]
    assert unset["value"] is None and unset["error"] is None and unset["group"] == "Run limits"
    assert rows["config.run.thinking.effort"]["kind"] == "choice"
    assert "high" in rows["config.run.thinking.effort"]["choices"]
    assert view["managed"] is False
    assert view["cloud"] == {"state": None, "status": None}
    assert view["read_only"] == [
        {
            "name": "region",
            "env_var": "TEST_REGION",
            "label": "Region",
            "help": "Where it runs.",
            "value": "eu",
            "source": "default",
            "needs": "redeploy",
            "why": "Read at start.",
            "warning": None,
        }
    ]


class _Cloud(ManagedSettingsChannel):
    """Cayu Cloud's answer, as a test stands it in."""

    def __init__(self, values: Mapping[str, Any], conflicts=()) -> None:
        self._snapshot = ManagedSettingsSnapshot(
            contract="cayu.managed-settings-values/v1",
            application_id="app",
            release_id="release",
            schema_digest="digest",
            revision=1,
            valid_until=datetime.now(UTC) + timedelta(minutes=5),
            values=dict(values),
            conflicts=tuple(conflicts),
        )

    @property
    def release_id(self) -> str:
        return "release"

    async def read(self) -> ManagedSettingsSnapshot:
        return self._snapshot

    async def report(self, report: Mapping[str, Any]) -> None:
        return None

    def read_at_build(self) -> ManagedSettingsSnapshot:
        return self._snapshot


def test_settings_say_which_values_come_from_cayu_cloud_and_which_it_refused() -> None:
    cloud = _Cloud(
        {"config.run.max_steps": 40, "config.run.retry_policy.max_attempts": 99},
        conflicts=[
            ManagedSettingsConflict(key="config.run.retry_policy.max_attempts", reason="too many"),
            # Refused outright: Cloud leaves it out of the values it applies.
            ManagedSettingsConflict(key="config.run.limits.max_tool_calls", reason="out of range"),
        ],
    )
    app = _app(managed_settings=ManagedSettings(cloud))
    with _client(app) as client:
        view = client.get("/app/settings", headers={"x-person": "bob"}).json()
    rows = {row["key"]: row for row in view["settings"]}
    assert view["managed"] is True
    assert view["cloud"]["state"] == "applied"
    assert rows["config.run.max_steps"]["value"] == 40
    assert rows["config.run.max_steps"]["source"] == "managed"
    attempts = rows["config.run.retry_policy.max_attempts"]
    assert attempts["value"] == 5 and attempts["source"] == "default"
    assert attempts["error"] == "The value set in Cayu Cloud is refused: too many."
    tool_calls = rows["config.run.limits.max_tool_calls"]
    assert tool_calls["source"] == "default"
    assert tool_calls["error"] == "The value set in Cayu Cloud is refused: out of range."


def test_the_overview_has_the_agents_metrics_and_cayus_inbox_counts() -> None:
    app = _app()
    server = create_server(app, config=ServerConfig(access=OpenAccess()))
    with _client(app, server=server) as client:
        answer = client.get("/app/overview", headers={"x-person": "bob"}).json()
    assert answer["metrics"] == [{"label": "Seen by", "value": "bob", "description": None}]
    assert answer["inbox"]["total"] == 0 and answer["inbox"]["unavailable"] is None
    datetime.fromisoformat(answer["as_of"])


def test_the_overview_says_when_cayus_inbox_is_closed_to_the_person() -> None:
    async def operators_only(request: Request) -> AuthContext:
        raise HTTPException(401, "Operators only.")

    app = _app()
    server = create_server(app, config=ServerConfig.protected(operators_only))
    with _client(app, server=server) as client:
        response = client.get("/app/overview", headers={"x-person": "bob"})
    assert response.status_code == 200
    inbox = response.json()["inbox"]
    assert inbox["by_kind"] == [] and "operators" in inbox["unavailable"]


def test_the_overview_keeps_its_metrics_when_cayus_inbox_fails() -> None:
    server = FastAPI()

    @server.get("/api/inbox/counts")
    async def broken() -> dict[str, Any]:
        raise ConnectionError("database down")

    with _client(_app(), server=server) as client:
        response = client.get("/app/overview", headers={"x-person": "bob"})
    assert response.status_code == 200
    answer = response.json()
    assert answer["metrics"] == [{"label": "Seen by", "value": "bob", "description": None}]
    assert answer["inbox"]["unavailable"] == "Cayu's inbox answered 500."


def _queue(app: CayuApp, client: TestClient, subject: str, membership: Membership, **task: Any) -> str:
    async def create() -> str:
        access = await app.access(subject)
        labels = conversation_labels(subject=subject, membership=membership, sharing="personal")
        created = await access.tasks.create(TaskCreate(**task), labels=labels)
        return created.id

    return client.portal.call(create)


def test_task_runs_group_the_persons_tasks_by_run() -> None:
    store = InMemoryTaskStore()
    app = _app(task_store=store)
    with _client(app) as client:
        first = _queue(app, client, "alice", ALICE, type="report", title="March", metadata={"run": "march"})
        _queue(app, client, "alice", ALICE, type="report", title="March", metadata={"run": "march"})
        alone = _queue(app, client, "alice", ALICE, type="report", title="April")
        _queue(app, client, "bob", Membership("globex"), type="report", title="Globex's own")

        async def work() -> None:
            async def handler(app, task, worker_id) -> None:
                if task.id == first:
                    raise RuntimeError("lost the printer")
                await store.complete_task(
                    task.id, {}, worker_id=worker_id, claim_generation=task.claim_generation
                )

            await run_task_worker(app, store, handler, worker_id="worker", max_tasks=3)

        client.portal.call(work)
        runs = client.get("/app/task-runs", headers={"x-person": "alice"}).json()

    by_id = {run["id"]: run for run in runs}
    assert set(by_id) == {"march", alone}
    assert by_id["march"]["tasks"] == 2 and by_id["march"]["status"] == "partly_failed"
    assert by_id["march"]["title"] == "March" and by_id["march"]["ended_at"] is not None
    assert by_id[alone]["status"] == "completed" and by_id[alone]["tasks"] == 1


def test_a_run_with_open_work_beyond_the_window_is_never_reported_ended() -> None:
    store = InMemoryTaskStore()
    app = _app(task_store=store)
    later = datetime.now(UTC) + timedelta(days=1)
    with _client(app) as client:
        # March's first task waits for tomorrow; its second, and April's, finish now.
        _queue(app, client, "alice", ALICE, type="report", title="March", metadata={"run": "march"}, available_at=later)
        _queue(app, client, "alice", ALICE, type="report", title="March", metadata={"run": "march"})
        _queue(app, client, "alice", ALICE, type="report", title="April", metadata={"run": "april"})

        async def work() -> list[TaskRun]:
            async def handler(app, task, worker_id) -> None:
                await store.complete_task(
                    task.id, {}, worker_id=worker_id, claim_generation=task.claim_generation
                )

            await run_task_worker(app, store, handler, worker_id="worker", max_tasks=2)
            # Only the newest two tasks fit the window; March's open task is older.
            return await list_task_runs(app, "alice", window=2)

        runs = {run.id: run for run in client.portal.call(work)}
    assert runs["march"].status == "scheduled" and runs["march"].ended_at is None
    assert runs["march"].tasks == 2 and runs["march"].partial is True
    # Not every task was read, so no run is reported as ended.
    assert runs["april"].status == "partial" and runs["april"].ended_at is None


def test_a_run_whose_open_task_is_missed_is_not_reported_ended() -> None:
    store = InMemoryTaskStore()
    app = _app(task_store=store)
    later = datetime.now(UTC) + timedelta(days=1)
    with _client(app) as client:
        _queue(app, client, "alice", ALICE, type="report", title="March", metadata={"run": "march"}, available_at=later)
        _queue(app, client, "alice", ALICE, type="report", title="March", metadata={"run": "march"})
        _queue(app, client, "alice", ALICE, type="report", title="April", metadata={"run": "april"})

        async def work() -> list[TaskRun]:
            async def handler(app, task, worker_id) -> None:
                await store.complete_task(
                    task.id, {}, worker_id=worker_id, claim_generation=task.claim_generation
                )

            await run_task_worker(app, store, handler, worker_id="worker", max_tasks=2)
            # From here on, every read of one status misses the open tasks, as a task that
            # moves from pending to running between two reads would.
            listed = store.list_tasks

            async def moving(query=None, **kwargs):
                if query is not None and query.status is not None:
                    return []
                return await listed(query, **kwargs)

            store.list_tasks = moving
            return await list_task_runs(app, "alice", window=2)

        runs = {run.id: run for run in client.portal.call(work)}
    # March's open task was never read; its finished sibling alone doesn't end the run.
    assert runs["march"].status == "partial" and runs["march"].ended_at is None
    assert runs["march"].tasks == 1 and runs["march"].partial is True


@pytest.mark.parametrize("limit", [0, 501])
def test_task_runs_refuse_a_limit_out_of_range(limit: int) -> None:
    with _client(_app(task_store=InMemoryTaskStore())) as client:
        response = client.get(f"/app/task-runs?limit={limit}", headers={"x-person": "alice"})
    assert response.status_code == 422


def _cayu_inbox_stand_in(seen: list[tuple[str, str, dict[str, Any]]]) -> FastAPI:
    """A server whose `/api/inbox` routes record what they were asked, standing in for Cayu's."""

    server = FastAPI()
    items = {"ours": "example-agent", "theirs": "other-agent", "for-carol": "example-agent"}

    @server.get("/api/inbox")
    async def listed(request: Request) -> dict[str, Any]:
        seen.append(("GET", "/inbox", dict(request.query_params)))
        return {"items": [], "next_cursor": None, "has_more": False, "issues": []}

    @server.get("/api/inbox/counts")
    async def counted(request: Request) -> dict[str, Any]:
        seen.append(("GET", "/inbox/counts", dict(request.query_params)))
        return {"complete": True, "total_count": 0, "counts": {}}

    @server.get("/api/inbox/{item_id}")
    async def item(item_id: str) -> dict[str, Any]:
        if item_id == "forbidden":
            raise HTTPException(
                403, {"code": "role_forbidden", "message": "Your role can't use this."}
            )
        if item_id not in items:
            raise HTTPException(404, "Pending action not found.")
        return {"id": item_id, "status": "pending", "agent_name": items[item_id]}

    @server.post("/api/inbox/{item_id}/settle")
    async def settle(item_id: str, request: Request) -> dict[str, Any]:
        if item_id == "for-carol":
            raise HTTPException(403, "Human-review recipient does not match authentication.")
        seen.append(("POST", f"/inbox/{item_id}/settle", await request.json()))
        return {"replayed": False, "item": {"id": item_id, "status": "settled"}}

    return server


def test_the_inbox_is_cayus_for_this_agent_only() -> None:
    seen: list[tuple[str, str, dict[str, Any]]] = []
    app = _app()
    with _client(app, server=_cayu_inbox_stand_in(seen)) as client:
        bob = {"x-person": "bob"}
        assert client.get("/app/inbox?kind=approval&agent_name=other", headers=bob).status_code == 200
        assert client.get("/app/inbox/counts?q=invoice", headers=bob).status_code == 200
        decision = {"decision": "approve", "reason": None}
        settled = client.post("/app/inbox/ours/settle", json=decision, headers=bob)
        theirs = client.post("/app/inbox/theirs/settle", json=decision, headers=bob)
        missing = client.post("/app/inbox/missing/settle", json=decision, headers=bob)
    assert seen == [
        ("GET", "/inbox", {"limit": "50", "kind": "approval", "agent_name": "example-agent"}),
        ("GET", "/inbox/counts", {"q": "invoice", "agent_name": "example-agent"}),
        ("POST", "/inbox/ours/settle", decision),
    ]
    assert settled.json()["item"]["status"] == "settled"
    assert (theirs.status_code, missing.status_code) == (404, 404)


def test_the_inbox_keeps_cayus_reason_unless_it_refuses_the_person_outright() -> None:
    with _client(_app(), server=_cayu_inbox_stand_in([])) as client:
        bob = {"x-person": "bob"}
        decision = {"decision": "approve"}
        not_theirs = client.post("/app/inbox/for-carol/settle", json=decision, headers=bob)
        role = client.post("/app/inbox/forbidden/settle", json=decision, headers=bob)
    assert not_theirs.status_code == 403
    assert not_theirs.json()["detail"] == "Human-review recipient does not match authentication."
    assert role.status_code == 403 and "operators" in role.json()["detail"]


def test_the_overview_counts_only_this_agents_inbox() -> None:
    seen: list[tuple[str, str, dict[str, Any]]] = []
    with _client(_app(), server=_cayu_inbox_stand_in(seen)) as client:
        answer = client.get("/app/overview", headers={"x-person": "bob"})
    assert answer.status_code == 200
    assert seen == [("GET", "/inbox/counts", {"agent_name": "example-agent"})]


def test_the_inbox_says_when_cayu_refuses_the_person() -> None:
    async def operators_only(request: Request) -> AuthContext:
        raise HTTPException(401, "Operators only.")

    app = _app()
    server = create_server(app, config=ServerConfig.protected(operators_only))
    with _client(app, server=server) as client:
        bob = {"x-person": "bob"}
        answers = [
            client.get("/app/inbox", headers=bob),
            client.get("/app/inbox/counts", headers=bob),
            client.post("/app/inbox/any/settle", json={"decision": "approve"}, headers=bob),
        ]
    # 403, not Cayu's 401, which the app would take for an expired sign-in.
    assert [answer.status_code for answer in answers] == [403, 403, 403]
    assert all("operators" in answer.json()["detail"] for answer in answers)


def test_the_inbox_answers_from_cayus_own_routes() -> None:
    app = _app()
    server = create_server(app, config=ServerConfig(access=OpenAccess()))
    with _client(app, server=server) as client:
        answer = client.get("/app/inbox/counts", headers={"x-person": "bob"})
    assert answer.status_code == 200 and answer.json()["complete"] is True
'''


_TEST_HTTP_API_PY = '''"""The application's HTTP API: sign-in, scoped conversations and member management.

The sign-in and conversation tests run without a database; member management reads and
writes the application's tables (`CAYU_TEST_POSTGRES_DSN`).
"""

from __future__ import annotations

import re

from pathlib import Path

from alembic import command
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_app_database import ALEMBIC, requires_postgres

from cayu import CayuApp
from cayu.server import serve_web_app
from cayu.sessions import InMemorySessionStore
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import RunRequest
from database.memberships import DatabaseMembershipDirectory
from database.session import session
from domain.organizations import InMemoryMembershipDirectory, Membership
from domain.sharing import conversation_labels
from integrations.http import AGENT_NAME, build_router
from operations.demo_seed import seed
from policies.access import MembershipAccessPolicy

ALICE = Membership("acme", "acme/finance")
IDENTITY = SessionIdentity(provider_name="fake", model="fake")
# The built web app, `[tool.cayu.serve] ui` in pyproject.toml.
WEB = Path(__file__).resolve().parents[1] / "web" / "static"


def _client(app: CayuApp, *, development: bool = True) -> TestClient:
    server = FastAPI()
    server.include_router(build_router(app, development=development))
    # As `cayu serve` serves it: last, for every path no route answers.
    serve_web_app(server, WEB)
    return TestClient(server)


def _app(directory) -> CayuApp:
    return CayuApp(
        session_store=InMemorySessionStore(),
        resource_access_policy=MembershipAccessPolicy(directory),
    )


def _sign_in(client: TestClient, subject: str) -> None:
    client.cookies.clear()
    assert client.post("/app/dev/sign-in", json={"subject": subject}).status_code == 204


def _start(client: TestClient, app: CayuApp, sharing: str) -> str:
    async def create() -> str:
        access = await app.access("alice")
        labels = conversation_labels(subject="alice", membership=ALICE, sharing=sharing)
        created = await access.sessions.create(
            RunRequest(agent_name="assistant", messages=[], labels=labels),
            identity=IDENTITY,
        )
        return created.id

    return client.portal.call(create)


def _listed(client: TestClient) -> set[str]:
    response = client.post("/app/resources/sessions/query", json={})
    assert response.status_code == 200, response.text
    return {item["id"] for item in response.json()["sessions"]}


def test_the_demonstration_login_signs_in_only_its_people() -> None:
    app = _app(InMemoryMembershipDirectory({"alice": [ALICE]}))
    with _client(app) as client:
        assert client.post("/app/resources/sessions/query", json={}).status_code == 401
        assert (
            client.post("/app/dev/sign-in", json={"subject": "mallory"}).status_code
            == 404
        )
        client.cookies.set("demo_session", "alice.forged")
        assert client.post("/app/resources/sessions/query", json={}).status_code == 401
        _sign_in(client, "alice")
        assert client.post("/app/resources/sessions/query", json={}).status_code == 200


def test_without_development_there_is_no_demonstration_login(monkeypatch) -> None:
    for name in (
        "APP_SIGN_IN",
        "CAYU_PROXY_IDENTITY_ISSUER",
        "CAYU_PROXY_IDENTITY_AUDIENCE",
        "CAYU_PROXY_IDENTITY_JWKS_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    app = _app(InMemoryMembershipDirectory({"alice": [ALICE]}))
    with _client(app, development=False) as client:
        assert (
            client.post("/app/dev/sign-in", json={"subject": "alice"}).status_code
            == 404
        )
        # Until a production sign-in is configured, nobody gets in.
        assert client.post("/app/resources/sessions/query", json={}).status_code == 503


def test_the_web_app_is_served_at_the_root_with_only_its_built_files() -> None:
    app = _app(InMemoryMembershipDirectory({"alice": [ALICE]}))
    with _client(app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert page.headers["cache-control"] == "no-cache"
        assert page.headers["content-security-policy"] == "frame-ancestors 'none'"
        # No images from elsewhere, which a reply could use to send what it read to
        # another site.
        assert "img-src 'self' data:" in page.text
        (script,) = re.findall(r'src="/(assets/[^"]+\\.js)"', page.text)
        asset = client.get(f"/{script}")
        assert asset.status_code == 200
        assert "immutable" in asset.headers["cache-control"]
        assert client.get("/assets/missing.js").status_code == 404
        assert client.get("/assets/nested/missing.js").status_code == 404
        # The app routes in the browser: its pages answer with the page itself.
        deep = client.get("/conversations/some-id")
        assert deep.status_code == 200 and deep.text == page.text
        # The API's own paths are never the page.
        assert client.get("/app/no-such-route").status_code == 404
        # Its demonstration picker lists the seeded people only under `cayu serve --dev`.
        assert "alice" in client.get("/app/dev/people").json()
        _sign_in(client, "alice")
        assert client.delete("/app/dev/sign-in").status_code == 204
        assert client.post("/app/resources/sessions/query", json={}).status_code == 401
    with _client(app, development=False) as client:
        assert client.get("/").status_code == 200
        assert client.get("/app/dev/people").status_code == 404


def test_conversations_over_http_show_each_person_only_their_own() -> None:
    directory = InMemoryMembershipDirectory(
        {
            "alice": [ALICE],
            "carol": [Membership("acme", "acme/finance")],
            "bob": [Membership("acme", "acme/legal")],
            "dave": [Membership("globex", "globex/support")],
        }
    )
    app = _app(directory)
    with _client(app) as client:
        personal = _start(client, app, "personal")
        department = _start(client, app, "department")
        _sign_in(client, "carol")
        assert _listed(client) == {department}
        _sign_in(client, "bob")
        assert _listed(client) == set()
        _sign_in(client, "dave")
        assert client.get(f"/app/resources/sessions/{department}").status_code == 404
        _sign_in(client, "alice")
        assert _listed(client) == {personal, department}


@requires_postgres
def test_admins_manage_members_and_a_removed_member_loses_access_at_once(
    database,
) -> None:
    command.upgrade(ALEMBIC, "head")
    with session() as db:
        seed(db)
        db.commit()
    app = _app(DatabaseMembershipDirectory())
    with _client(app) as client:
        shared = _start(client, app, "department")

        _sign_in(client, "carol")
        me = client.get("/app/me").json()
        assert me["agent_name"] == AGENT_NAME
        assert me["memberships"] == [
            {
                "subject": "carol",
                "organization": "acme",
                "department": "acme/finance",
                "role": "member",
            }
        ]
        labels = client.post(
            "/app/conversation-labels",
            json={"organization": "acme", "sharing": "department"},
        ).json()
        assert labels["department"] == "acme/finance"
        assert labels["owner"] == "carol"
        assert (
            client.post(
                "/app/conversation-labels",
                json={"organization": "globex", "sharing": "organization"},
            ).status_code
            == 403
        )
        assert client.get(f"/app/resources/sessions/{shared}").status_code == 200
        # A member reads the member list but doesn't change it.
        assert len(client.get("/app/organizations/acme/members").json()) == 3
        assert client.delete("/app/organizations/acme/members/bob").status_code == 403

        _sign_in(client, "dave")
        assert client.get("/app/organizations/acme/members").status_code == 404
        assert client.delete("/app/organizations/acme/members/carol").status_code == 404

        _sign_in(client, "alice")
        moved = client.put(
            "/app/organizations/acme/members/bob",
            json={"department": "globex/support"},
        )
        assert moved.status_code == 422
        too_long = "x" * 201
        assert (
            client.delete(f"/app/organizations/acme/members/{too_long}").status_code
            == 422
        )
        # A '..' segment, which a browser would resolve to another member, is refused.
        dotted = "/app/organizations/acme/members/staff/%2E%2E/bob"
        assert client.delete(dotted).status_code == 422
        # Acme's only admin can't leave it without one.
        assert (
            client.put(
                "/app/organizations/acme/members/alice",
                json={"department": "acme/finance", "role": "member"},
            ).status_code
            == 409
        )
        assert client.delete("/app/organizations/acme/members/alice").status_code == 409
        # An identity provider's subject may contain '/'.
        assert (
            client.put(
                "/app/organizations/acme/members/staff/erin",
                json={"department": "acme/legal"},
            ).status_code
            == 200
        )
        assert (
            client.delete("/app/organizations/acme/members/staff/erin").status_code
            == 204
        )
        assert client.delete("/app/organizations/acme/members/carol").status_code == 204

        _sign_in(client, "carol")
        assert client.get(f"/app/resources/sessions/{shared}").status_code == 404


@requires_postgres
def test_whoami_and_the_overview_answer_for_the_signed_in_person(database) -> None:
    command.upgrade(ALEMBIC, "head")
    with session() as db:
        seed(db)
        db.commit()
    app = _app(DatabaseMembershipDirectory())
    with _client(app) as client:
        assert client.get("/app/whoami").status_code == 401
        _start(client, app, "personal")
        _start(client, app, "department")

        # An admin of any of their organizations is `admin`; others are members. Under
        # `cayu serve --dev`, Cayu's inbox is open, so they may decide in the app's Inbox.
        _sign_in(client, "alice")
        alice = client.get("/app/whoami").json()
        assert (alice["role"], alice["may_change"], alice["may_decide"]) == ("admin", True, True)
        metrics = {
            item["label"]: item["value"]
            for item in client.get("/app/overview").json()["metrics"]
        }
        assert metrics == {
            "Organizations": "1",
            "Members": "3",
            "Conversations": "2",
            "Tasks": "0",
        }

        _sign_in(client, "carol")
        carol = client.get("/app/whoami").json()
        assert (carol["role"], carol["may_change"]) == ("member", False)
        conversations = next(
            item
            for item in client.get("/app/overview").json()["metrics"]
            if item["label"] == "Conversations"
        )
        # Carol sees Alice's department conversation, not her personal one.
        assert conversations["value"] == "1"
        assert conversations["description"].startswith("0 started by you")

        _sign_in(client, "dave")
        metrics = {
            item["label"]: item["value"]
            for item in client.get("/app/overview").json()["metrics"]
        }
        assert (metrics["Members"], metrics["Conversations"]) == ("1", "0")
'''

_TEST_BACKGROUND_REVOCATION_PY = '''"""Removing a member stops their background work at its next step, also after a restart."""

from __future__ import annotations

import asyncio

import pytest

from cayu import CayuApp, Message
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.resource_access import ResourceAccessDenied
from cayu.sessions import InMemorySessionStore
from cayu.sessions.access import SessionAccessDenied
from cayu.sessions.requests import RunRequest
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import InMemoryTaskStore
from cayu.tasks.creation import TaskCreate
from cayu.tasks.worker import run_task_worker
from app import build_app
from domain.organizations import InMemoryMembershipDirectory, Membership
from domain.sharing import conversation_labels
from policies.access import MembershipAccessPolicy

ALICE = Membership("acme", "acme/finance")


async def _queue_report(app: CayuApp) -> str:
    access = await app.access("alice")
    labels = conversation_labels(
        subject="alice", membership=ALICE, sharing="department"
    )
    task = await access.tasks.create(
        TaskCreate(type="report", title="Quarterly report"), labels=labels
    )
    return task.id


async def _work(app: CayuApp, store) -> list[str]:
    handled: list[str] = []

    async def handler(app, task, worker_id) -> None:
        handled.append(task.id)

    await run_task_worker(app, store, handler, worker_id="worker", max_tasks=1)
    return handled


def test_a_removed_members_queued_task_never_runs() -> None:
    async def run() -> None:
        directory = InMemoryMembershipDirectory({"alice": [ALICE]})
        store = InMemoryTaskStore()
        app = CayuApp(
            task_store=store, resource_access_policy=MembershipAccessPolicy(directory)
        )
        task_id = await _queue_report(app)

        directory.set_memberships("alice", [])

        assert await _work(app, store) == []
        assert (await store.load_task(task_id)).status == "failed"

    asyncio.run(run())


def test_the_task_stays_refused_after_a_restart(tmp_path) -> None:
    directory = InMemoryMembershipDirectory({"alice": [ALICE]})
    path = tmp_path / "tasks.db"

    async def before_restart() -> str:
        store = SQLiteTaskStore(path)
        try:
            app = CayuApp(
                task_store=store,
                resource_access_policy=MembershipAccessPolicy(directory),
            )
            return await _queue_report(app)
        finally:
            await store.close()

    async def after_restart(task_id: str) -> None:
        store = SQLiteTaskStore(path)
        try:
            app = CayuApp(
                task_store=store,
                resource_access_policy=MembershipAccessPolicy(directory),
            )
            assert await _work(app, store) == []
            assert (await store.load_task(task_id)).status == "failed"
        finally:
            await store.close()

    task_id = asyncio.run(before_restart())
    directory.set_memberships("alice", [])
    asyncio.run(after_restart(task_id))


def test_a_running_conversation_stops_at_its_next_step() -> None:
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.text_delta("Working on it."),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ]
    )

    async def run() -> None:
        directory = InMemoryMembershipDirectory({"alice": [ALICE]})
        async with build_app(
            provider=provider,
            session_store=InMemorySessionStore(),
            task_store=InMemoryTaskStore(),
            access_policy=MembershipAccessPolicy(directory),
        ) as app:
            access = await app.access("alice")
            labels = conversation_labels(
                subject="alice", membership=ALICE, sharing="department"
            )
            events = access.run(
                RunRequest(
                    agent_name="__AGENT_NAME__",
                    messages=[Message.text("user", "Draft the report")],
                    labels=labels,
                )
            )
            await anext(events)

            directory.set_memberships("alice", [])

            with pytest.raises((ResourceAccessDenied, SessionAccessDenied)):
                async for _ in events:
                    pass
        # The model was never asked: the next step was refused first.
        assert provider.requests == []

    asyncio.run(run())
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
| `integrations/sign_in.py` | The one place a request becomes a verified person |
| `integrations/http/` | The API `cayu serve` serves under `/app` (`[tool.cayu.serve] routers`) |
| `integrations/http/generated/` | The routes every Agent's web app reads: `/app/whoami`, `/app/settings`, `/app/overview`, `/app/task-runs`, `/app/inbox` |
| `web/` | The web app at `/`: its source, and the committed build in `web/static` that `cayu serve` serves (`[tool.cayu.serve] ui`; see "The web app" below) |

- Start a conversation through `await app.access(subject)` with the labels
  `conversation_labels(...)` returns for the signed-in person. Never take a subject, an
  organization or labels from the request itself.
- Everyone sees their own conversations, their department's and their organization's, and
  nothing of another organization. Only the owner continues, changes or deletes one.
- Cayu asks the policy again at every protected step, so removing a member takes effect at
  their next step, in background work too.
- The Developer dashboard reads every conversation, so it is for operators only:
  `[tool.cayu.serve].auth` is `OPERATOR_ONLY_AUTH`, which refuses the people Cayu Cloud signs
  in as members. They reach their data only through `/app`.
- Organization admins (`role = "admin"` on a membership) add, move and remove members
  through `/app/organizations/{organization}/members`. That role is the application's own
  decision and never widens what Cayu lets anyone see.
- The web interface signs people in with a cookie: Cayu Cloud's sign-in, or the
  demonstration login. With `APP_SIGN_IN=oidc`, the API is for clients that send their own
  bearer token, and the interface can't sign anyone in.
- Sign-in: `cayu serve --dev` offers a demonstration login at `POST /app/dev/sign-in` for
  the people `operations/demo_seed.py` adds, and nobody else. Outside development,
  `APP_SIGN_IN` selects Cayu Cloud's sign-in (`cloud`, the default) or an OIDC bearer token
  (`oidc`, with CAYU_OIDC_ISSUER and CAYU_OIDC_AUDIENCE); memberships are keyed on the ID
  that sign-in gives each person. Until one is configured, the API answers 503.
- An organization a client shows as active is only a filter and a default for new
  conversations (`POST /app/conversation-labels`); it never grants anything.
- The web app calls only `/app`. After changing `web/src`, run `npm --prefix web ci`
  and `npm --prefix web run build`, and commit `web/static`; running and deploying need no
  Node. `npm --prefix web run check:build` fails when `web/static` isn't the build of
  `web/src`. `npm --prefix web run dev` reloads it live against `cayu serve --dev`.
- With `CAYU_DATABASE_URL` set, `uv run --no-sync cayu storage migrate`, `uv run --no-sync
  alembic upgrade head` and `uv run --no-sync python -m operations.demo_seed` give a
  development database two demonstration organizations; `uv run --no-sync cayu serve --dev`
  then serves the app at http://127.0.0.1:8000/. `uv run --no-sync cayu guide
  b2b` walks through it.
  `tests/test_organizations.py`, `tests/test_http_api.py` and
  `tests/test_background_revocation.py` show each rule.

## The web app

`web/` is one React app, served at `/` and calling only `/app`. Each file says in its
first doc comment whether it is `GENERATED.` (the generator owns it and may rewrite it) or
`THE AGENT'S OWN.` (yours).

| Path | What it is | Whose |
| --- | --- | --- |
| `web/src/main.tsx`, `web/src/app/` | The shell: browser routes at `/`, the area pattern, the layout with its navigation and profile menu, the status pages, the API requester | Generated |
| `web/src/api/` | Typed clients for the generated routes: `whoami`, `settings`, `overview` | Generated |
| `web/src/generated/` | The Overview and Settings areas | Generated |
| `web/src/blocks/task-runs.tsx` | TaskRuns: an Agent's Cayu tasks, one row per run (`GET /app/task-runs`). This Agent queues no tasks yet, so no area uses it | A Cayu building block |
| `web/src/agent/` | This Agent's own areas (Conversations, Members), its Overview sections, its sign-in page for `cayu serve --dev`, and `index.tsx`, which lists its areas | The Agent's |
| `web/src/kit/shadcn/` | The component kit: stock shadcn (Base UI, base-nova) and the parts built from it | Generated |
| `integrations/http/generated/` | The routes the generated pages read | Generated |
| `integrations/http/overview_metrics.py`, `fixed_settings.py` | This Agent's Overview metrics and the settings it reads at start-up | The Agent's |

The app imports every component from `@kit/...`; `tsconfig.json` and `vite.config.ts` point
`@kit` at `src/kit/shadcn`. Another kit is a folder next to it with the same entries; point
both at it, and no page changes.

The navigation's Runtime group links to Cayu's Inbox (`/cayu/inbox`), where approvals and
questions are decided, and to the Developer dashboard (`/cayu/`); the Overview's "Waiting for
a person" links there too. Both are for the Agent's operators: deployed, `OPERATOR_ONLY_AUTH`
refuses the people Cayu Cloud signs in as members. `/app/inbox` serves this Agent's inbox as
the signed-in person, for an Inbox page of the app's own.
"""


# The page's title in the template, and in its build: the app shows it as the Agent's name.
_WEB_TITLE = "<title>Agent</title>"


def _web_files(name: str) -> dict[str, str]:
    """The web app for `web/`: its source, which a project rebuilds with
    `npm --prefix web run build`, and the built files in `web/static` it serves.

    Both pages are titled with the project's name. Vite copies the title as it is, and the
    page isn't named by its content, so a fresh build still matches `web/static`."""

    found: dict[str, str] = {}

    def walk(node, prefix: str) -> None:
        for child in sorted(node.iterdir(), key=lambda item: item.name):
            if child.is_dir():
                walk(child, f"{prefix}{child.name}/")
            else:
                found[f"web/{prefix}{child.name}"] = child.read_text(encoding="utf-8")

    walk(package_files("cayu.cli").joinpath("templates", "b2b_web"), "")
    title = f"<title>{html.escape(name)}</title>"
    for page in ("web/index.html", "web/static/index.html"):
        if found[page].count(_WEB_TITLE) != 1:
            raise RuntimeError(f"{page} in the b2b web template has no {_WEB_TITLE}")
        found[page] = found[page].replace(_WEB_TITLE, title)
    return found


# The agent preset's routing rows this preset answers differently.
_AGENTS_ROWS = {
    "| A web UI that shows progress or live results | `mount_cayu(...)` and `connect()` from "
    "the served `client.js`. Do not poll on a fixed timer. | `cayu guide app-ui` |": (
        "| A web UI for the organizations' people | `web/`, served at `/`, calling only "
        "`/app`: add an area in `web/src/agent/` and list it in `web/src/agent/index.tsx`; leave "
        '`GENERATED.` files to the generator (README "The web app"). Not `client.js`: it calls '
        "the operator API, which people signed in as members can't use. | `cayu guide b2b` |"
    ),
    "| A public or multi-user product | `cayu new NAME --preset service` | "
    "`cayu guide references#server` |": (
        "| Another organization, department or member | `add_member` in "
        "`database/memberships.py`, or an admin on the Members page | `cayu guide b2b` |"
    ),
}


# This preset is already the multi-user application that paragraph sends people to.
_MULTI_USER_ADVICE = (
    "If the requested application is public or multi-user, regenerate with\n"
    "`cayu new NAME --preset service` or adopt Cayu's maintained service contract;\n"
    "do not improvise product authorization around raw Cayu routes."
)
_B2B_MULTI_USER = (
    "This application is multi-user: its people reach their data only through `/app`, which\n"
    "authorizes each request with `app.access(subject)`; do not add product routes over raw\n"
    "Cayu routes."
)


def _cloud_application(name: str) -> str:
    """The application slug `cayu cloud init` makes from the project's name, or the name
    itself when it makes none (init then asks for another name)."""

    from cayu.cli._cloud_api import CloudApiError
    from cayu.cli._cloud_project import _application_slug

    try:
        return _application_slug(name)
    except CloudApiError:
        return name


def _agents_md(agents: str) -> str:
    for row, replacement in _AGENTS_ROWS.items():
        agents = agents.replace(row, replacement)
    return agents.replace(_MULTI_USER_ADVICE, _B2B_MULTI_USER)


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
            "            enable_logging=runtime.enable_logging,\n",
            "            enable_logging=runtime.enable_logging,\n"
            "            resource_access_policy=(\n"
            "                access_policy if access_policy is not None else build_access_policy()\n"
            "            ),\n",
        )
    )
    return {
        # The Developer dashboard login the b2b preset uses (see pyproject.toml below).
        "README.md": files["README.md"].replace(
            ENVIRONMENT_OPERATOR_AUTH_TARGET, ENVIRONMENT_OPERATOR_ONLY_AUTH_TARGET
        )
        + _GUIDANCE,
        "AGENTS.md": _agents_md(files["AGENTS.md"]) + _GUIDANCE,
        # Installed packages and build output of web/ and of a Developer dashboard ejected
        # with `cayu dashboard eject`, which writes no .gitignore of its own.
        ".gitignore": files[".gitignore"] + "node_modules/\ndist/\n",
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
        # The application's own users sign in through the same proxy as members, so the
        # Developer dashboard (which reads every conversation) is for operators only.
        "pyproject.toml": files["pyproject.toml"].replace(
            f'[tool.cayu.serve]\nauth = "{ENVIRONMENT_OPERATOR_AUTH_TARGET}"\n',
            f'[tool.cayu.serve]\nauth = "{ENVIRONMENT_OPERATOR_ONLY_AUTH_TARGET}"\n'
            'routers = ["integrations.http:build_router"]\n'
            'ui = "web/static"\n',
        ),
        "configuration/settings.py": files["configuration/settings.py"] + _SETTINGS_APPEND,
        "migrations/versions/0003_membership_roles.py": _ROLES_MIGRATION,
        "integrations/sign_in.py": _SIGN_IN_PY,
        "integrations/http/__init__.py": _HTTP_PY.replace(
            "__AGENT_NAME__", plan.agent_name
        ).replace("__CLOUD_APPLICATION__", _cloud_application(plan.name)),
        **app_route_files(),
        "integrations/http/fixed_settings.py": _FIXED_SETTINGS_PY,
        "integrations/http/overview_metrics.py": _OVERVIEW_METRICS_PY,
        **_web_files(plan.name),
        "tests/test_http_api.py": _TEST_HTTP_API_PY,
        "tests/test_generated_routes.py": _TEST_GENERATED_ROUTES_PY,
        "tests/test_background_revocation.py": _TEST_BACKGROUND_REVOCATION_PY.replace(
            "__AGENT_NAME__", plan.agent_name
        ),
    }
