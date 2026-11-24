"""A scratch PostgreSQL database for tests and evals, migrated the way a deploy migrates one.

`scratch_database()` creates a database with a random name on a server where the caller may
create databases (`CAYU_TEST_POSTGRES_DSN` by default, which `cayu dev postgres up --export`
sets), runs `cayu storage migrate` and, when the project has an `alembic.ini`, its Alembic
migrations to head, yields the new database's URL, and drops the database afterwards. Both
migrations run as subprocesses with `CAYU_DATABASE_URL` set to the scratch database, so
nothing in the calling process changes and the application's tables come from its
migrations, never from `create_all`. `cayu dev postgres scratch -- COMMAND` runs a command,
such as `cayu eval run`, against one.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote, urlencode
from uuid import uuid4

from cayu.storage.targets import DATABASE_DIRECT_URL_ENV, DATABASE_URL_ENV

#: The variable naming a server where tests may create databases.
TEST_SERVER_VARIABLE = "CAYU_TEST_POSTGRES_DSN"
_DATABASE_VARIABLES = (DATABASE_URL_ENV, DATABASE_DIRECT_URL_ENV)


class ScratchDatabaseError(RuntimeError):
    """A scratch database could not be created or migrated."""


def database_environment(url: str, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """`base` (default: this process's environment) with `CAYU_DATABASE_URL` set to `url`.

    `CAYU_DATABASE_DIRECT_URL` is removed, so nothing prefers another database.
    """

    environment = {
        key: value
        for key, value in (os.environ if base is None else base).items()
        if key not in _DATABASE_VARIABLES
    }
    environment[DATABASE_URL_ENV] = url
    return environment


@contextmanager
def scratch_database(
    server: str | None = None,
    *,
    project: Path | str | None = None,
    cayu_migrations: bool = True,
    app_migrations: bool = True,
    prefix: str = "cayu_scratch",
    timeout: float = 300.0,
) -> Iterator[str]:
    """Create, migrate, yield and finally drop a scratch database; yields its URL.

    `server` defaults to `CAYU_TEST_POSTGRES_DSN`. `project` (default: the working directory)
    is where the migrations run; Alembic runs only when it has an `alembic.ini`. Pass
    `cayu_migrations=False` or `app_migrations=False` to leave either set of tables out, for
    example to test the migrations themselves on an empty database.
    """

    import psycopg
    from psycopg import sql

    server = (server if server is not None else os.environ.get(TEST_SERVER_VARIABLE, "")).strip()
    if not server:
        raise ScratchDatabaseError(
            f"{TEST_SERVER_VARIABLE} is not set: name a PostgreSQL server where databases may "
            "be created, or start one with `cayu dev postgres up --export`."
        )
    if not prefix.isidentifier() or prefix.lower() != prefix:
        raise ScratchDatabaseError("prefix must be a lower-case identifier")
    name = f"{prefix}_{uuid4().hex}"
    root = Path.cwd() if project is None else Path(project)
    with psycopg.connect(server, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    try:
        url = database_url(server, name)
        if cayu_migrations:
            _run(root, url, ("-m", "cayu", "storage", "migrate", "--json"), timeout)
        if app_migrations and (root / "alembic.ini").is_file():
            _run(root, url, ("-m", "alembic", "upgrade", "head"), timeout)
        yield url
    finally:
        with psycopg.connect(server, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name))
            )


def database_url(server: str, name: str) -> str:
    """A `postgresql://` URL for database `name` on `server` (a URL or a conninfo string)."""

    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    parts = {
        key: str(value)
        for key, value in conninfo_to_dict(make_conninfo(server, dbname=name)).items()
        if value is not None
    }
    user = parts.pop("user", None)
    password = parts.pop("password", None)
    host = parts.pop("host", None)
    port = parts.pop("port", None)
    database = parts.pop("dbname")
    if host is not None and (host.startswith("/") or "," in host):
        # A socket directory, or several hosts, only fit in the query.
        parts["host"] = host
        host = None
    credentials = ""
    if user is not None:
        credentials = quote(user, safe="")
        if password is not None:
            credentials += ":" + quote(password, safe="")
        credentials += "@"
    address = ""
    if host is not None:
        # An IPv6 address goes in brackets, so its colons aren't read as the port's.
        address = f"[{host}]" if ":" in host else host
        if port is not None:
            address += f":{port}"
    elif port is not None:
        parts["port"] = port
    # `%20`, not `+`: libpq decodes only percent escapes.
    query = f"?{urlencode(parts, quote_via=quote)}" if parts else ""
    return f"postgresql://{credentials}{address}/{quote(database, safe='')}{query}"


def _run(root: Path, url: str, arguments: tuple[str, ...], timeout: float) -> None:
    result = subprocess.run(
        [sys.executable, *arguments],
        cwd=root,
        env=database_environment(url),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        command = " ".join(arguments[1:])
        detail = (result.stderr or result.stdout).strip()[-2000:]
        raise ScratchDatabaseError(f"`{command}` failed on the scratch database:\n{detail}")
