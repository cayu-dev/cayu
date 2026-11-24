"""The `app-database` capability: application tables in their own PostgreSQL schema.

Cayu's own stores keep their `cayu_*` tables in the database's default schema and migrate
them with `cayu storage migrate`. This capability generates the application's side of the
same database: SQLAlchemy models in the `app` schema and an Alembic environment that only
ever sees that schema, so it can neither alter nor drop Cayu's tables. Cayu itself keeps no
SQLAlchemy or Alembic dependency; they live in the generated project.
"""

from __future__ import annotations

APP_DATABASE = "app-database"

_PACKAGE_INIT = '"""Application-owned tables and the connection that reads and writes them."""\n'

_MODELS = '''"""Application-owned tables, in the `app` PostgreSQL schema beside Cayu's `cayu_*` tables.

Add a table here, then generate and review its migration:
`uv run alembic revision --autogenerate -m "describe the change"`.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, MetaData, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

SCHEMA = "app"


class Base(DeclarativeBase):
    metadata = MetaData(schema=SCHEMA)


class Note(Base):
    """An example table; replace it with the application's own records."""

    __tablename__ = "notes"

    id: Mapped[int] = mapped_column(primary_key=True)
    body: Mapped[str] = mapped_column(String(2000))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
'''

_SESSION = '''"""The application's database connection, opened on first use, never at import or startup.

Cayu Cloud smoke-tests a release without database credentials, so nothing here may connect
while the application starts. Migrations run as a deploy step, never from the app.

These sessions are synchronous, and routes, tools and task handlers are async. A query blocks
the thread that runs it, and a session keeps its connection from its first query until it
commits, rolls back or closes. From async code:

- `await run_in_thread(work)` runs `work(db)` in a worker thread, in one transaction, and
  closes the session before it returns. Waiting for a connection then never blocks the event
  loop.
- Never hold a session across an `await` of slow work (a model call, a file read or upload).
  Copy the plain values you need, end the transaction, then await. With the pool's five
  connections held across awaits, the next query blocks the event loop while it waits for a
  connection, so none of the five can resume to give theirs back, and every request hangs.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from functools import cache
from typing import TypeVar

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from configuration.settings import configured_app_database_url

T = TypeVar("T")


def database_url(*, direct: bool = False) -> str:
    """`direct=True`, for migrations, prefers the server address behind a pooling proxy."""

    url = configured_app_database_url(direct=direct)
    if url is None:
        raise RuntimeError(
            "CAYU_DATABASE_URL is not set: application tables live in PostgreSQL. "
            "Set it to the database Cayu uses, then run `uv run alembic upgrade head`."
        )
    scheme, _, rest = url.partition("://")
    # The psycopg 3 driver Cayu already installs.
    return f"postgresql+psycopg://{rest}" if scheme in ("postgresql", "postgres") else url


@cache
def engine() -> Engine:
    return create_engine(
        database_url(),
        # No more connections than Cayu's own pool takes by default (CAYU_DATABASE_POOL_MAX).
        pool_size=5,
        max_overflow=0,
        pool_pre_ping=True,
        # Behind a transaction-pooling proxy such as PgBouncer, a statement prepared on one
        # server connection is missing on the next.
        connect_args={"prepare_threshold": None},
    )


def session() -> Session:
    """A new session: use it as a context manager and commit what it changes."""

    return Session(engine(), expire_on_commit=False)


async def run_in_thread(work: Callable[[Session], T]) -> T:
    """Run `work(db)` in a worker thread with a new session, committing when it returns.

    The session is closed before this returns: return plain values or loaded rows, never
    the session itself. `work` is synchronous, so it can't await while it holds a connection.
    A thread can't be stopped, so when the caller is cancelled this still waits for the
    transaction to end before the cancellation goes on: no database work outlives its caller.
    """

    def unit() -> T:
        with session() as db, db.begin():
            return work(db)

    worker = asyncio.ensure_future(asyncio.to_thread(unit))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        while not worker.done():
            try:
                await asyncio.wait({worker})
            except asyncio.CancelledError:
                pass
        if not worker.cancelled():
            worker.exception()  # Retrieved, so asyncio doesn't log it as unhandled.
        raise
'''

_LOCKS = '''"""Transaction-scoped locks, for application code that must not race with itself.

Two requests that create the same row, such as one with an id derived from its input, can
both find it missing and both insert it. Claim, then recheck: take the lock on the key the
row is created under, then read again before writing.

    with session() as db, db.begin():
        lock(db, f"invoice:{invoice_id}")
        if db.get(Invoice, invoice_id) is None:
            db.add(Invoice(id=invoice_id, ...))

A second caller waits at `lock` until the first commits or rolls back, then sees its row.
The lock is released when the transaction ends; there is no unlock. Take several locks in
one order everywhere (sorted keys, say), or two callers can each wait for the other's.
Keys are hashed into PostgreSQL's advisory-lock space under an `app:` prefix, so they
never meet the locks Cayu takes in the same database.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.orm import Session


def _key(key: str) -> dict[str, str]:
    if not key:
        raise ValueError("a lock key must not be empty")
    return {"key": f"app:{key}"}


def lock(db: Session, key: str) -> None:
    """Wait for, then hold, the lock on `key` until `db`'s transaction ends."""

    db.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"), _key(key))


def try_lock(db: Session, key: str) -> bool:
    """Take the lock on `key` if no other transaction holds it; never waits."""

    return bool(
        db.execute(
            text("SELECT pg_try_advisory_xact_lock(hashtextextended(:key, 0))"), _key(key)
        ).scalar()
    )
'''

_ALEMBIC_INI = """# Migrations for the application's own tables, in the `app` schema.
# Cayu's tables are migrated by `cayu storage migrate`, never by Alembic.
[alembic]
script_location = %(here)s/migrations
prepend_sys_path = .
path_separator = os
file_template = %%(rev)s_%%(slug)s
"""

_ENV = '''"""Migrations for the `app` schema only; Cayu's tables in the default schema stay unseen.

Nothing here changes `search_path`: Cayu finds its tables through it.
"""

from __future__ import annotations

from alembic import context
from sqlalchemy import create_engine, pool, text

from database.models import SCHEMA, Base
from database.session import database_url

# Serialises concurrent application migrations of one database (two deploys, a developer).
# A session-level lock: it is released when the migration's connection closes.
_MIGRATION_LOCK = int.from_bytes(b"cayu-app", "big")


def include_name(name, type_, parent_names):
    # The default schema, where Cayu's tables live, arrives as None and is left out.
    if type_ == "schema":
        return name == SCHEMA
    return True


def _configure(**options) -> None:
    context.configure(
        target_metadata=Base.metadata,
        include_schemas=True,
        include_name=include_name,
        version_table_schema=SCHEMA,
        **options,
    )


def run_migrations_offline() -> None:
    # Named parameters, so offline SQL has no DBAPI percent escaping ('100%', not '100%%').
    _configure(dialect_name="postgresql", literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.execute(f\'CREATE SCHEMA IF NOT EXISTS "{SCHEMA}"\')
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(database_url(direct=True), poolclass=pool.NullPool)
    with engine.connect() as connection:
        connection.execute(text("SELECT pg_advisory_lock(:key)"), {"key": _MIGRATION_LOCK})
        # Create the schema only when it's missing: a least-privilege role may use a schema
        # provisioned for it without the right to create one.
        missing = connection.execute(
            text("SELECT to_regnamespace(:schema) IS NULL"), {"schema": SCHEMA}
        ).scalar()
        if missing:
            connection.execute(text(f\'CREATE SCHEMA "{SCHEMA}"\'))
        # Checked once the schema exists (`current_schema()` skips missing ones); a refusal
        # rolls the new schema back.
        if connection.execute(text("SELECT current_schema()")).scalar() == SCHEMA:
            raise RuntimeError(
                f"The database role's default schema is {SCHEMA!r}, so Cayu's tables would land "
                "beside the application's. Use a role with another name."
            )
        connection.commit()
        _configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
'''

_SCRIPT_MAKO = '''"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
${imports if imports else ""}
revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
'''

_FIRST_REVISION = '''"""Create the application's first table.

Revision ID: 0001
Revises:
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "notes",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("body", sa.String(length=2000), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        schema="app",
    )


def downgrade() -> None:
    op.drop_table("notes", schema="app")
'''

_TEST = '''"""Application tables: hermetic checks always, PostgreSQL checks with a test server.

Set `CAYU_TEST_POSTGRES_DSN` to a server where you may create databases; each test
creates its own and drops it afterwards. Without it the PostgreSQL checks are skipped, unless
`CAYU_REQUIRE_POSTGRES_TESTS=1` (as in the generated CI), which makes them fail instead.
Tests of the application's own behavior take the `migrated_database` fixture: Cayu's tables
and the application's, built by their migrations rather than `create_all`; other test modules
import it from here (`from test_app_database import migrated_database, requires_postgres`).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time

import pytest
from alembic import command
from alembic.config import Config

from cayu.cli import main as cayu
from cayu.testing.postgres import scratch_database
from database import session
from database.locks import lock, try_lock
from database.models import SCHEMA, Base, Note

ALEMBIC = Config("alembic.ini")
SERVER = os.environ.get("CAYU_TEST_POSTGRES_DSN", "")
# Where a skipped check would hide a missing database, such as CI, they fail instead.
REQUIRED = os.environ.get("CAYU_REQUIRE_POSTGRES_TESTS", "").strip() not in ("", "0")
SKIPPED = "CAYU_TEST_POSTGRES_DSN is not set"
requires_postgres = pytest.mark.skipif(not SERVER and not REQUIRED, reason=SKIPPED)


def test_every_table_lives_in_the_app_schema() -> None:
    assert Base.metadata.tables
    assert {table.schema for table in Base.metadata.tables.values()} == {SCHEMA}


def test_a_cancelled_caller_waits_for_its_transaction_to_end(monkeypatch) -> None:
    events = []

    class Recorded:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            events.append("closed")

        def begin(self):
            return contextlib.nullcontext()

    monkeypatch.setattr(session, "session", Recorded)

    def work(db) -> None:
        time.sleep(0.2)
        events.append("committed")

    async def caller() -> None:
        task = asyncio.create_task(session.run_in_thread(work))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        events.append("caller stopped")

    asyncio.run(caller())
    assert events == ["committed", "closed", "caller stopped"]


def test_nothing_connects_until_a_session_needs_the_database(monkeypatch) -> None:
    monkeypatch.delenv("CAYU_DATABASE_URL", raising=False)
    monkeypatch.delenv("CAYU_DATABASE_DIRECT_URL", raising=False)
    session.engine.cache_clear()
    with pytest.raises(RuntimeError, match="CAYU_DATABASE_URL is not set"):
        session.engine()


def _scratch(monkeypatch, *, migrated: bool):
    if not SERVER:
        pytest.fail(
            "CAYU_REQUIRE_POSTGRES_TESTS is set, so this test needs PostgreSQL: set "
            "CAYU_TEST_POSTGRES_DSN to a server where it may create databases."
        )
    with scratch_database(
        SERVER, cayu_migrations=migrated, app_migrations=migrated, prefix="app_test"
    ) as url:
        monkeypatch.setenv("CAYU_DATABASE_URL", url)
        monkeypatch.delenv("CAYU_DATABASE_DIRECT_URL", raising=False)
        session.engine.cache_clear()
        try:
            yield url
        finally:
            session.engine().dispose()
            session.engine.cache_clear()


@pytest.fixture
def database(monkeypatch):
    """An empty scratch database, for the tests of the migrations themselves."""

    yield from _scratch(monkeypatch, migrated=False)


@pytest.fixture
def migrated_database(monkeypatch):
    """A scratch database migrated as a deploy migrates one: Cayu's tables, then the app's.

    Use it for tests that read and write the application's tables, so they run on the schema
    the migrations build, not on `Base.metadata.create_all`. Another test module takes it with
    `from test_app_database import migrated_database, requires_postgres  # noqa: F401`.
    """

    yield from _scratch(monkeypatch, migrated=True)


def _app_migrate() -> None:
    command.upgrade(ALEMBIC, "head")


def _cayu_migrate() -> None:
    assert cayu(["storage", "migrate", "--json"]) == 0


def _tables(schema: str) -> set[str]:
    with session.engine().connect() as connection:
        rows = connection.exec_driver_sql(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = %s",
            (schema,),
        )
        return {row[0] for row in rows}


@requires_postgres
def test_migrations_create_the_app_tables_on_an_empty_database(database) -> None:
    _app_migrate()
    assert {"alembic_version", *(t.name for t in Base.metadata.tables.values())} <= _tables(SCHEMA)
    with session.session() as db:
        db.add(Note(body="first"))
        db.commit()
        assert db.query(Note).count() == 1


@requires_postgres
@pytest.mark.parametrize("cayu_first", [True, False])
def test_cayu_and_application_migrations_share_one_database_in_either_order(
    database, cayu_first: bool
) -> None:
    steps = (_cayu_migrate, _app_migrate) if cayu_first else (_app_migrate, _cayu_migrate)
    for step in steps:
        step()
    assert "cayu_schema_migrations" in _tables("public")
    assert "alembic_version" in _tables(SCHEMA)
    assert not {name for name in _tables(SCHEMA) if name.startswith("cayu_")}


@requires_postgres
def test_more_async_callers_than_connections_all_finish(migrated_database) -> None:
    callers = 3 * session.engine().pool.size()

    async def handler(n: int) -> int:
        await session.run_in_thread(lambda db: db.add(Note(body=f"note {n}")))
        # Slow async work, such as a model call, with no session open.
        await asyncio.sleep(0.05)
        return await session.run_in_thread(lambda db: db.query(Note).count())

    async def all_handlers() -> list[int]:
        return await asyncio.wait_for(
            asyncio.gather(*(handler(n) for n in range(callers))), timeout=20
        )

    assert max(asyncio.run(all_handlers())) == callers


@requires_postgres
def test_a_lock_holds_until_its_transaction_ends(migrated_database) -> None:
    with session.session() as first, first.begin():
        lock(first, "note:1")
        with session.session() as second, second.begin():
            assert not try_lock(second, "note:1")
            assert try_lock(second, "note:2")
    with session.session() as third, third.begin():
        assert try_lock(third, "note:1")


@requires_postgres
def test_autogenerate_sees_no_changes_and_never_touches_cayu_tables(database) -> None:
    _cayu_migrate()
    _app_migrate()
    # Fails on any autogenerate operation, such as dropping a Cayu table it shouldn't see.
    command.check(ALEMBIC)

'''

# Replaces the default "share it with your own table prefix" line in the storage guidance.
APP_DATABASE_TABLES = """- Application-owned records, such as orders, cases, members, and sync cursors,
  live in the same PostgreSQL database in the `app` schema: models in
  `database/models.py`, Alembic migrations in `migrations/`. Cayu's `cayu_*`
  tables stay in the default schema and Alembic never sees them. Never change
  `search_path`.
- Add a table: define its model, run
  `uv run alembic revision --autogenerate -m "describe the change"`, review the
  file in `migrations/versions/`, then `uv run alembic upgrade head`. A new local
  database also needs `uv run cayu storage migrate` first. The application
  connects on first use and never migrates itself. Details:
  `cayu guide applications#application-tables`.
- `database/session.py` is synchronous. From async code, `await run_in_thread(work)`
  runs the database work in a worker thread in one transaction; never hold a session
  across an `await` of slow work. Copy plain values and end the transaction first, or
  five held connections hang every request.
- Code that creates rows under a deterministic key claims, then rechecks: inside the
  transaction, `lock(db, key)` from `database/locks.py`, then read again before
  writing. The lock ends with the transaction.
- Tests that read and write these tables take the `migrated_database` fixture in
  `tests/test_app_database.py` (another test module imports it:
  `from test_app_database import migrated_database, requires_postgres  # noqa: F401`):
  a scratch database built by both migrations. Evals run
  against one with `uv run --no-sync cayu dev postgres scratch -- uv run --no-sync cayu
  eval run`.
- The tests that read and write these tables need a PostgreSQL server where they
  may create databases (`CAYU_TEST_POSTGRES_DSN`), and are skipped without one. Run
  `eval "$(uv run --no-sync cayu dev postgres up --export)"` before `uv run --no-sync
  pytest` to start a throwaway one, the one the generated CI uses, and set the variable;
  `uv run --no-sync cayu dev postgres down` removes it and its data.
"""


_SETTINGS_APPEND = '''

def configured_app_database_url(*, direct: bool = False) -> str | None:
    """The application tables' database: `CAYU_DATABASE_URL`, the one Cayu uses.

    Migrations pass `direct=True` to prefer `CAYU_DATABASE_DIRECT_URL`, the server
    address behind a transaction-pooling proxy, for their session-level lock.
    """

    # The same variables, in the same order, as `cayu storage migrate` reads.
    names = ("CAYU_DATABASE_DIRECT_URL", "CAYU_DATABASE_URL") if direct else ("CAYU_DATABASE_URL",)
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None
'''


def app_database_files(settings: str) -> dict[str, str]:
    """The files `--with app-database` adds or extends; `settings` is the project's settings.py."""

    return {
        "configuration/settings.py": settings.rstrip() + "\n" + _SETTINGS_APPEND,
        "alembic.ini": _ALEMBIC_INI,
        "database/__init__.py": _PACKAGE_INIT,
        "database/models.py": _MODELS,
        "database/session.py": _SESSION,
        "database/locks.py": _LOCKS,
        "migrations/env.py": _ENV,
        "migrations/script.py.mako": _SCRIPT_MAKO,
        "migrations/versions/0001_app_tables.py": _FIRST_REVISION,
        "tests/test_app_database.py": _TEST,
    }
