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
"""

from __future__ import annotations

from functools import cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from configuration.settings import configured_app_database_url


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
creates its own and drops it afterwards.
"""

from __future__ import annotations

import os
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.engine import URL

from cayu.cli import main as cayu
from database import session
from database.models import SCHEMA, Base, Note

ALEMBIC = Config("alembic.ini")
SERVER = os.environ.get("CAYU_TEST_POSTGRES_DSN", "")
requires_postgres = pytest.mark.skipif(not SERVER, reason="CAYU_TEST_POSTGRES_DSN is not set")


def test_every_table_lives_in_the_app_schema() -> None:
    assert Base.metadata.tables
    assert {table.schema for table in Base.metadata.tables.values()} == {SCHEMA}


def test_nothing_connects_until_a_session_needs_the_database(monkeypatch) -> None:
    monkeypatch.delenv("CAYU_DATABASE_URL", raising=False)
    monkeypatch.delenv("CAYU_DATABASE_DIRECT_URL", raising=False)
    session.engine.cache_clear()
    with pytest.raises(RuntimeError, match="CAYU_DATABASE_URL is not set"):
        session.engine()


@pytest.fixture
def database(monkeypatch):
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    name = f"app_test_{uuid4().hex}"
    with psycopg.connect(SERVER, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    parts = conninfo_to_dict(make_conninfo(SERVER, dbname=name))
    host = parts.pop("host", None)
    socket = host is not None and host.startswith("/")
    url = URL.create(
        "postgresql",
        username=parts.pop("user", None),
        password=parts.pop("password", None),
        host=None if socket else host,
        port=int(parts.pop("port")) if "port" in parts else None,
        database=parts.pop("dbname"),
        query={**parts, **({"host": host} if socket else {})},
    ).render_as_string(hide_password=False)
    monkeypatch.setenv("CAYU_DATABASE_URL", url)
    monkeypatch.delenv("CAYU_DATABASE_DIRECT_URL", raising=False)
    session.engine.cache_clear()
    try:
        yield url
    finally:
        session.engine().dispose()
        session.engine.cache_clear()
        with psycopg.connect(SERVER, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name))
            )


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
        "migrations/env.py": _ENV,
        "migrations/script.py.mako": _SCRIPT_MAKO,
        "migrations/versions/0001_app_tables.py": _FIRST_REVISION,
        "tests/test_app_database.py": _TEST,
    }
