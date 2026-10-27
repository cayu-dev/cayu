"""Files for the ``b2b`` preset of ``cayu new``.

The preset starts from the agent preset's files and the app-database convention, and adds
what an application serving several businesses needs: organizations, departments and
memberships in its own tables, and a policy that turns a person's current membership into
Cayu resource-access grants. The application decides who belongs where; Cayu enforces the
result on every read, listing and background step.
"""

from __future__ import annotations

from importlib.resources import files as package_files

from cayu._operator_credentials import (
    ENVIRONMENT_OPERATOR_AUTH_TARGET,
    ENVIRONMENT_OPERATOR_ONLY_AUTH_TARGET,
)
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

_HTTP_PY = '''"""The application's HTTP API, which `cayu serve` serves under /app
([tool.cayu.serve] routers in pyproject.toml).

- /app/resources/...: Cayu's scoped routes for conversations, tasks and runs. Each answers
  only what the signed-in person may see; a foreign ID looks missing.
- /app/me and /app/conversation-labels: who is signed in, their memberships, and the labels a
  new conversation they share takes. The active organization a client shows is only a filter
  and a default: it never grants anything.
- /app/organizations/{organization}/members: the members, which an organization's admins
  add, move and remove.
- /app/ui/: the web interface (integrations/web.py).
- /app/dev/sign-in and /app/dev/people: the demonstration login and sign-out, served
  only by `cayu serve --dev`.
"""

from __future__ import annotations

import asyncio
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
from domain.organizations import Membership
from domain.sharing import Sharing, conversation_labels
from integrations.sign_in import (
    DEMO_COOKIE,
    DEMO_SUBJECTS,
    DemoSignIn,
    build_authenticate,
)
from integrations.web import build_web_router


# The length of `memberships.subject`.
MEMBER_SUBJECT_LENGTH = 200
# The agent the web interface starts conversations with.
AGENT_NAME = "__AGENT_NAME__"


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
    router.include_router(build_web_router())

    async def signed_in(request: Request) -> str:
        return await authenticate(request)

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


_WEB_PY = """\"\"\"The web interface, served at /app/ui/ from web/static.

`web/` holds its source; `npm --prefix web run build` rebuilds web/static, which is committed,
so running and deploying the application needs no Node. Only the built files are served: the
page itself and the files its build names in web/static/assets.
\"\"\"

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, RedirectResponse


def _static() -> Path:
    return Path(__file__).resolve().parent.parent / "web" / "static"


def build_web_router() -> APIRouter:
    router = APIRouter(prefix="/ui", include_in_schema=False)

    @router.get("")
    def root() -> RedirectResponse:
        return RedirectResponse("ui/")

    @router.get("/")
    def page() -> FileResponse:
        # Always checked again: it names the current build's hashed files.
        return FileResponse(
            _static() / "index.html",
            headers={
                "Cache-Control": "no-cache",
                # Never inside another site's frame, where its buttons could be clicked
                # for the signed-in person, and no images from elsewhere, which a reply
                # could use to send what it read to another site.
                "Content-Security-Policy": "frame-ancestors 'none'; img-src 'self' data:",
            },
        )

    @router.get("/assets/{name}")
    def asset(name: str) -> FileResponse:
        assets = _static() / "assets"
        if name not in {entry.name for entry in assets.iterdir() if entry.is_file()}:
            raise HTTPException(404, "Not found.")
        # Each build names its files by their content, so they never change.
        return FileResponse(
            assets / name,
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    return router
"""


_TEST_HTTP_API_PY = '''"""The application's HTTP API: sign-in, scoped conversations and member management.

The sign-in and conversation tests run without a database; member management reads and
writes the application's tables (`CAYU_TEST_POSTGRES_DSN`).
"""

from __future__ import annotations

import re

from alembic import command
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_app_database import ALEMBIC, requires_postgres

from cayu import CayuApp
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


def _client(app: CayuApp, *, development: bool = True) -> TestClient:
    server = FastAPI()
    server.include_router(build_router(app, development=development))
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


def test_the_web_interface_is_served_with_only_its_built_files() -> None:
    app = _app(InMemoryMembershipDirectory({"alice": [ALICE]}))
    with _client(app) as client:
        page = client.get("/app/ui/")
        assert page.status_code == 200
        assert page.headers["cache-control"] == "no-cache"
        assert (
            page.headers["content-security-policy"]
            == "frame-ancestors 'none'; img-src 'self' data:"
        )
        (script,) = re.findall(r'src="\\./(assets/[^"]+\\.js)"', page.text)
        asset = client.get(f"/app/ui/{script}")
        assert asset.status_code == 200
        assert "immutable" in asset.headers["cache-control"]
        assert client.get("/app/ui/assets/missing.js").status_code == 404
        assert client.get("/app/ui/assets/..%2Findex.html").status_code == 404
        # Its demonstration picker lists the seeded people only under `cayu serve --dev`.
        assert "alice" in client.get("/app/dev/people").json()
        _sign_in(client, "alice")
        assert client.delete("/app/dev/sign-in").status_code == 204
        assert client.post("/app/resources/sessions/query", json={}).status_code == 401
    with _client(app, development=False) as client:
        assert client.get("/app/ui/").status_code == 200
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
| `integrations/http.py` | The API `cayu serve` serves under `/app` (`[tool.cayu.serve] routers`) |
| `web/`, `integrations/web.py` | The web interface at `/app/ui/`: its source and the committed build in `web/static` |

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
- The web interface calls only `/app`. After changing `web/src`, run `npm --prefix web ci`
  and `npm --prefix web run build`, and commit `web/static`; running and deploying need no
  Node. `npm --prefix web run dev` reloads it live against `cayu serve --dev`.
- With `CAYU_DATABASE_URL` set, `uv run --no-sync cayu storage migrate`, `uv run --no-sync
  alembic upgrade head` and `uv run --no-sync python -m operations.demo_seed` give a
  development database two demonstration organizations; `uv run --no-sync cayu serve --dev`
  then serves the interface at http://127.0.0.1:8000/app/ui/. `uv run --no-sync cayu guide
  b2b` walks through it.
  `tests/test_organizations.py`, `tests/test_http_api.py` and
  `tests/test_background_revocation.py` show each rule.
"""


def _web_files() -> dict[str, str]:
    """The web interface for `web/`: its source, which a project rebuilds with
    `npm --prefix web run build`, and the built files in `web/static` it serves."""

    found: dict[str, str] = {}

    def walk(node, prefix: str) -> None:
        for child in sorted(node.iterdir(), key=lambda item: item.name):
            if child.is_dir():
                walk(child, f"{prefix}{child.name}/")
            else:
                found[f"web/{prefix}{child.name}"] = child.read_text(encoding="utf-8")

    walk(package_files("cayu.cli").joinpath("templates", "b2b_web"), "")
    return found


# The agent preset's routing rows this preset answers differently.
_AGENTS_ROWS = {
    "| A web UI that shows progress or live results | `mount_cayu(...)` and `connect()` from "
    "the served `client.js`. Do not poll on a fixed timer. | `cayu guide app-ui` |": (
        "| A web UI for the organizations' people | `web/`, served at `/app/ui/`, calling only "
        "`/app`. Not `client.js`: it calls the operator API, which people signed in as members "
        "can't use. | `cayu guide b2b` |"
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
            "        enable_logging=runtime.enable_logging,\n",
            "        enable_logging=runtime.enable_logging,\n"
            "        resource_access_policy=(\n"
            "            access_policy if access_policy is not None else build_access_policy()\n"
            "        ),\n",
        )
    )
    return {
        # The Developer dashboard login the b2b preset uses (see pyproject.toml below).
        "README.md": files["README.md"].replace(
            ENVIRONMENT_OPERATOR_AUTH_TARGET, ENVIRONMENT_OPERATOR_ONLY_AUTH_TARGET
        )
        + _GUIDANCE,
        "AGENTS.md": _agents_md(files["AGENTS.md"]) + _GUIDANCE,
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
            'routers = ["integrations.http:build_router"]\n',
        ),
        "configuration/settings.py": files["configuration/settings.py"] + _SETTINGS_APPEND,
        "migrations/versions/0003_membership_roles.py": _ROLES_MIGRATION,
        "integrations/sign_in.py": _SIGN_IN_PY,
        "integrations/http.py": _HTTP_PY.replace("__AGENT_NAME__", plan.agent_name),
        "integrations/web.py": _WEB_PY,
        **_web_files(),
        "tests/test_http_api.py": _TEST_HTTP_API_PY,
        "tests/test_background_revocation.py": _TEST_BACKGROUND_REVOCATION_PY.replace(
            "__AGENT_NAME__", plan.agent_name
        ),
    }
