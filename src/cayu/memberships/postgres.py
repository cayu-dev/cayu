"""Memberships in the application's own PostgreSQL tables.

:class:`PostgresMemberships` reads and writes two tables the application creates and
migrates itself, in its own schema (``app`` by default). Their shape is the contract; it is
exactly what ``cayu new --with organizations`` (and the b2b preset) generates in migrations
0002 and 0003::

    CREATE TABLE app.memberships (
        id serial PRIMARY KEY,
        subject varchar(200) NOT NULL,
        organization_id varchar(64) NOT NULL REFERENCES app.organizations (id) ON DELETE CASCADE,
        department_id varchar(64),
        role varchar(16) NOT NULL DEFAULT 'member',
        UNIQUE (subject, organization_id),
        FOREIGN KEY (organization_id, department_id)
            REFERENCES app.departments (organization_id, id),
        CONSTRAINT memberships_role_check CHECK (role in ('member', 'admin'))
    );
    CREATE TABLE app.access_revision (
        id serial PRIMARY KEY,  -- one row, id 1
        revision bigint NOT NULL
    );

The organizations and departments tables, and any columns the application adds to them, stay
the application's own. :func:`check_membership_tables` checks what the store relies on.

Every change raises the access revision in the same transaction, and :meth:`current` reads a
person's memberships and the revision in one REPEATABLE READ snapshot, so Cayu refuses any
decision read before a change. No change leaves an organization that has an admin without
one.

It runs on any DB-API connection to PostgreSQL that uses the ``%s`` parameter style:
``connect`` is called for each operation and returns a connection with autocommit off and no
transaction open, which the store commits or rolls back and then closes. ``functools.partial(psycopg.connect,
url)`` opens one each time; a pool's checkout, such as SQLAlchemy's ``engine.raw_connection``,
reuses the application's own pool. Calls run in a worker thread.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import Callable, Iterable
from typing import Any

from cayu.memberships.changes import (
    LastAdmin,
    MemberChange,
    NotAnAdmin,
    RemoveMember,
    SetMember,
    UnknownDepartment,
)
from cayu.memberships.directory import ROLES, AccessSnapshot, Member, Membership, Role

# The single row of `access_revision`.
_REVISION_ROW = 1
_SCHEMA_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,62}")
# The lengths of the contract's columns.
SUBJECT_LENGTH = 200
ORGANIZATION_LENGTH = 64
DEPARTMENT_LENGTH = 64
_FOREIGN_KEY_VIOLATION = "23503"


class MembershipTablesInvalid(RuntimeError):
    """The membership tables aren't the shape :class:`PostgresMemberships` relies on."""

    def __init__(self, problems: Iterable[str]) -> None:
        self.problems = tuple(problems)
        super().__init__(
            "The membership tables aren't the shape cayu.memberships needs:\n- "
            + "\n- ".join(self.problems)
        )


def _schema(schema: str) -> str:
    if not isinstance(schema, str) or _SCHEMA_NAME.fullmatch(schema) is None:
        raise ValueError("schema must be a plain PostgreSQL identifier, such as 'app'.")
    return schema


def _identifier(name: str, *, field: str, limit: int) -> str:
    if not isinstance(name, str) or not name or name != name.strip() or len(name) > limit:
        raise ValueError(f"{field} must be a non-blank ID of at most {limit} characters.")
    return name


def _checked(subject: str, organization: str, department: str | None, role: str) -> None:
    _identifier(subject, field="A member's subject", limit=SUBJECT_LENGTH)
    _identifier(organization, field="An organization", limit=ORGANIZATION_LENGTH)
    if "/" in organization:
        raise ValueError("An organization ID has no '/'.")
    if department is not None:
        _identifier(department, field="A department", limit=DEPARTMENT_LENGTH)
        if not department.startswith(f"{organization}/"):
            raise UnknownDepartment(
                f"{department!r} isn't a department of {organization!r}: a department ID "
                "starts with its organization's ID and '/'."
            )
    if role not in ROLES:
        raise ValueError(f"A role is one of {', '.join(ROLES)}.")


class PostgresMemberships:
    """The application's memberships, read as Cayu's directory and changed in place.

    ``connect`` returns a new DB-API connection to the application's database for each
    operation (see the module documentation); ``schema`` is where the tables live.
    """

    def __init__(self, connect: Callable[[], Any], *, schema: str = "app") -> None:
        if not callable(connect):
            raise TypeError("connect must be a callable that returns a database connection.")
        self._connect = connect
        self._schema = _schema(schema)
        self._memberships = f'"{self._schema}".memberships'
        self._revision = f'"{self._schema}".access_revision'

    @property
    def schema(self) -> str:
        return self._schema

    async def current(self, subject: str) -> AccessSnapshot:
        """``subject``'s memberships and the revision they belong to, from one snapshot."""

        return await asyncio.to_thread(self._current, subject)

    async def of_subject(self, subject: str) -> tuple[Member, ...]:
        """Every organization ``subject`` belongs to, ordered by organization."""

        return await asyncio.to_thread(self._members, "subject", subject)

    async def of_organization(self, organization: str) -> tuple[Member, ...]:
        """``organization``'s members, ordered by subject."""

        return await asyncio.to_thread(self._members, "organization_id", organization)

    async def add(
        self,
        subject: str,
        organization: str,
        *,
        department: str | None = None,
        role: Role = "member",
    ) -> None:
        """Add ``subject`` to ``organization``, or move them to ``department`` and ``role``
        if already a member.

        Raises :class:`UnknownDepartment` for an organization that doesn't exist or a
        department that isn't its own, and :class:`LastAdmin` when it would demote the
        organization's last admin.
        """

        change = SetMember(subject, department, role)
        await asyncio.to_thread(self._change, organization, change, None)

    async def remove(self, subject: str, organization: str) -> bool:
        """Remove ``subject`` from ``organization``; whether they were a member.

        Their next protected step there is refused. Raises :class:`LastAdmin` when they are
        the organization's last admin.
        """

        return await asyncio.to_thread(self._change, organization, RemoveMember(subject), None)

    async def change_as_admin(self, caller: str, organization: str, change: MemberChange) -> bool:
        """Make ``change`` to ``organization``'s members on behalf of ``caller``.

        In one transaction it locks the organization's admins' rows, checks ``caller`` is
        one of them (:class:`NotAnAdmin` otherwise), makes the change, keeps at least one
        admin (:class:`LastAdmin`) and raises the access revision. Returns whether a member
        was added, moved or removed.
        """

        _identifier(caller, field="The caller's subject", limit=SUBJECT_LENGTH)
        return await asyncio.to_thread(self._change, organization, change, caller)

    def _current(self, subject: str) -> AccessSnapshot:
        def read(cursor: Any) -> AccessSnapshot:
            # One snapshot, so the memberships and their revision agree.
            cursor.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            cursor.execute(
                f"SELECT organization_id, department_id FROM {self._memberships} "
                "WHERE subject = %s ORDER BY organization_id",
                (subject,),
            )
            rows = cursor.fetchall()
            cursor.execute(f"SELECT revision FROM {self._revision} WHERE id = %s", (_REVISION_ROW,))
            revision = cursor.fetchone()
            return AccessSnapshot(
                memberships=tuple(
                    Membership(organization, department) for organization, department in rows
                ),
                revision=0 if revision is None or revision[0] is None else int(revision[0]),
            )

        return self._transaction(read)

    def _members(self, column: str, value: str) -> tuple[Member, ...]:
        def read(cursor: Any) -> tuple[Member, ...]:
            cursor.execute(
                f"SELECT subject, organization_id, department_id, role FROM {self._memberships} "
                f"WHERE {column} = %s ORDER BY organization_id, subject",
                (value,),
            )
            return tuple(Member(*row) for row in cursor.fetchall())

        return self._transaction(read)

    def _change(self, organization: str, change: MemberChange, caller: str | None) -> bool:
        if isinstance(change, SetMember):
            _checked(change.subject, organization, change.department, change.role)
        elif isinstance(change, RemoveMember):
            _identifier(change.subject, field="A member's subject", limit=SUBJECT_LENGTH)
            _identifier(organization, field="An organization", limit=ORGANIZATION_LENGTH)
        else:
            raise TypeError("A membership change is a SetMember or a RemoveMember.")

        def write(cursor: Any) -> bool:
            # Holding the admins' rows until the end keeps the caller an admin while the
            # change happens, and two changes from losing the last admin between them.
            admins = self._admins(cursor, organization, lock=True)
            if caller is not None and caller not in admins:
                raise NotAnAdmin("Only the organization's admins manage its members.")
            if isinstance(change, SetMember):
                cursor.execute(
                    f"INSERT INTO {self._memberships} "
                    "(subject, organization_id, department_id, role) VALUES (%s, %s, %s, %s) "
                    "ON CONFLICT (subject, organization_id) "
                    "DO UPDATE SET department_id = EXCLUDED.department_id, role = EXCLUDED.role",
                    (change.subject, organization, change.department, change.role),
                )
                changed = True
            else:
                cursor.execute(
                    f"DELETE FROM {self._memberships} WHERE subject = %s AND organization_id = %s",
                    (change.subject, organization),
                )
                changed = cursor.rowcount > 0
            if admins and not self._admins(cursor, organization, lock=False):
                raise LastAdmin("An organization keeps at least one admin.")
            cursor.execute(
                f"INSERT INTO {self._revision} (id, revision) VALUES (%s, 1) "
                f"ON CONFLICT (id) DO UPDATE SET revision = {self._revision}.revision + 1",
                (_REVISION_ROW,),
            )
            return changed

        try:
            return self._transaction(write, commit=True)
        except Exception as exc:
            # psycopg 3 names the SQLSTATE `sqlstate`, psycopg2 `pgcode`.
            code = getattr(exc, "sqlstate", None) or getattr(exc, "pgcode", None)
            if code == _FOREIGN_KEY_VIOLATION:
                raise UnknownDepartment(
                    f"{organization!r} doesn't exist, or the department isn't one of its own."
                ) from None
            raise

    def _admins(self, cursor: Any, organization: str, *, lock: bool) -> set[str]:
        cursor.execute(
            f"SELECT subject FROM {self._memberships} "
            "WHERE organization_id = %s AND role = 'admin'" + (" FOR UPDATE" if lock else ""),
            (organization,),
        )
        return {row[0] for row in cursor.fetchall()}

    def _transaction(self, work: Callable[[Any], Any], *, commit: bool = False) -> Any:
        connection = self._connect()
        if getattr(connection, "autocommit", False):
            connection.close()
            # Each statement would commit on its own: a refused change would stay made.
            raise ValueError("PostgresMemberships needs connections with autocommit off.")
        try:
            cursor = connection.cursor()
            try:
                result = work(cursor)
            finally:
                cursor.close()
            if commit:
                connection.commit()
            else:
                connection.rollback()
            return result
        except BaseException:
            # A broken connection can't roll back; closing it is all that's left to do.
            with contextlib.suppress(Exception):
                connection.rollback()
            raise
        finally:
            connection.close()


_TEXT_TYPES = frozenset({"varchar", "text", "bpchar"})
_INTEGER_TYPES = frozenset({"int2", "int4", "int8"})
# (table, column): (the types it may have, whether it is NOT NULL)
_COLUMNS: dict[tuple[str, str], tuple[frozenset[str], bool]] = {
    ("memberships", "subject"): (_TEXT_TYPES, True),
    ("memberships", "organization_id"): (_TEXT_TYPES, True),
    ("memberships", "department_id"): (_TEXT_TYPES, False),
    ("memberships", "role"): (_TEXT_TYPES, True),
    ("access_revision", "id"): (_INTEGER_TYPES, True),
    ("access_revision", "revision"): (frozenset({"int8"}), True),
}
_COLUMNS_SQL = """
SELECT c.relname, a.attname, t.typname, a.attnotnull
FROM pg_attribute a
JOIN pg_class c ON c.oid = a.attrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
JOIN pg_type t ON t.oid = a.atttypid
WHERE n.nspname = %s AND c.relname IN ('memberships', 'access_revision')
  AND c.relkind IN ('r', 'p') AND a.attnum > 0 AND NOT a.attisdropped
"""
_UNIQUE_SQL = """
SELECT c.relname, ARRAY(
    SELECT a.attname::text FROM unnest(i.indkey::int2[]) AS k(attnum)
    JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = k.attnum
)
FROM pg_index i
JOIN pg_class c ON c.oid = i.indrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = %s AND c.relname IN ('memberships', 'access_revision')
  AND i.indisunique AND i.indpred IS NULL AND i.indexprs IS NULL
"""
_CONSTRAINTS_SQL = """
SELECT con.contype::text,
    ARRAY(
        SELECT a.attname::text FROM unnest(con.conkey) WITH ORDINALITY AS k(attnum, ord)
        JOIN pg_attribute a ON a.attrelid = con.conrelid AND a.attnum = k.attnum
        ORDER BY k.ord
    ),
    rn.nspname, rc.relname,
    ARRAY(
        SELECT a.attname::text FROM unnest(con.confkey) WITH ORDINALITY AS k(attnum, ord)
        JOIN pg_attribute a ON a.attrelid = con.confrelid AND a.attnum = k.attnum
        ORDER BY k.ord
    ),
    pg_get_constraintdef(con.oid)
FROM pg_constraint con
JOIN pg_class c ON c.oid = con.conrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
LEFT JOIN pg_class rc ON rc.oid = con.confrelid
LEFT JOIN pg_namespace rn ON rn.oid = rc.relnamespace
WHERE n.nspname = %s AND c.relname = 'memberships' AND con.contype IN ('f', 'c')
"""


def check_membership_tables(connection: Any, *, schema: str = "app") -> None:
    """Check that ``schema``'s membership tables are the shape :class:`PostgresMemberships`
    relies on, and raise :class:`MembershipTablesInvalid` naming each difference.

    It checks the columns the store reads and writes, the unique (subject, organization)
    and access-revision keys its upserts need, the foreign key that keeps a membership's
    department inside its organization, and the check that keeps roles to ``member`` and
    ``admin``. ``connection`` is a DB-API connection to the application's database; the
    check only reads the catalog and leaves its transaction to the caller.
    """

    schema = _schema(schema)
    cursor = connection.cursor()
    try:
        problems = _table_problems(cursor, schema)
    finally:
        cursor.close()
    if problems:
        raise MembershipTablesInvalid(problems)


def _table_problems(cursor: Any, schema: str) -> list[str]:
    cursor.execute(_COLUMNS_SQL, (schema,))
    columns = {
        (table, column): (kind, not_null) for table, column, kind, not_null in cursor.fetchall()
    }
    tables = {table for table, _ in columns}
    problems = [
        f"There is no table {schema}.{table}."
        for table in ("memberships", "access_revision")
        if table not in tables
    ]
    for (table, column), (kinds, not_null) in _COLUMNS.items():
        if table not in tables:
            continue
        found = columns.get((table, column))
        if found is None:
            problems.append(f"{schema}.{table} has no column {column}.")
        elif found[0] not in kinds:
            problems.append(
                f"{schema}.{table}.{column} is {found[0]}; it must be one of "
                f"{', '.join(sorted(kinds))}."
            )
        elif found[1] != not_null:
            problems.append(
                f"{schema}.{table}.{column} must be {'NOT NULL' if not_null else 'nullable'}."
            )
    cursor.execute(_UNIQUE_SQL, (schema,))
    unique = {(table, frozenset(names)) for table, names in cursor.fetchall()}
    for table, key in (
        ("memberships", frozenset({"subject", "organization_id"})),
        ("access_revision", frozenset({"id"})),
    ):
        if table in tables and (table, key) not in unique:
            problems.append(f"{schema}.{table} has no unique key on ({', '.join(sorted(key))}).")
    if "memberships" not in tables:
        return problems
    cursor.execute(_CONSTRAINTS_SQL, (schema,))
    department_key = role_check = False
    for kind, own, target_schema, target, referenced, definition in cursor.fetchall():
        if kind == "f":
            department_key = department_key or (
                (target_schema, target) == (schema, "departments")
                and set(zip(own, referenced, strict=False))
                == {("organization_id", "organization_id"), ("department_id", "id")}
            )
        elif list(own) == ["role"]:
            role_check = role_check or ("'member'" in definition and "'admin'" in definition)
    if not department_key:
        problems.append(
            f"{schema}.memberships has no foreign key (organization_id, department_id) "
            f"referencing {schema}.departments (organization_id, id)."
        )
    if not role_check:
        problems.append(f"{schema}.memberships has no check keeping role to 'member' and 'admin'.")
    return problems
