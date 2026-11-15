"""The membership tables of the documented contract, for the tests that need PostgreSQL."""

from __future__ import annotations

import psycopg

# The tables generated migrations 0002 and 0003 create, as SQL: what an application without
# Alembic would run.
CONTRACT = """
CREATE SCHEMA {schema};
CREATE TABLE {schema}.organizations (
    id varchar(64) PRIMARY KEY,
    name varchar(200) NOT NULL,
    CHECK (strpos(id, '/') = 0)
);
CREATE TABLE {schema}.memberships (
    id serial PRIMARY KEY,
    subject varchar(200) NOT NULL,
    organization_id varchar(64) NOT NULL
        REFERENCES {schema}.organizations (id) ON DELETE CASCADE,
    UNIQUE (subject, organization_id)
);
CREATE TABLE {schema}.access_revision (
    id serial PRIMARY KEY,
    revision bigint NOT NULL
);
ALTER TABLE {schema}.memberships ADD COLUMN role varchar(16) DEFAULT 'member' NOT NULL;
ALTER TABLE {schema}.memberships
    ADD CONSTRAINT memberships_role_check CHECK (role in ('member', 'admin'));
"""
# The tables Cayu 0.16.1 to 0.20.0 generated, which kept departments. Those projects keep
# them; the library must work on them as they are.
OLDER_CONTRACT = """
CREATE SCHEMA {schema};
CREATE TABLE {schema}.organizations (
    id varchar(64) PRIMARY KEY,
    name varchar(200) NOT NULL,
    CHECK (strpos(id, '/') = 0)
);
CREATE TABLE {schema}.departments (
    id varchar(64) PRIMARY KEY,
    organization_id varchar(64) NOT NULL
        REFERENCES {schema}.organizations (id) ON DELETE CASCADE,
    name varchar(200) NOT NULL,
    UNIQUE (organization_id, id),
    CHECK (starts_with(id, organization_id || '/'))
);
CREATE TABLE {schema}.memberships (
    id serial PRIMARY KEY,
    subject varchar(200) NOT NULL,
    organization_id varchar(64) NOT NULL
        REFERENCES {schema}.organizations (id) ON DELETE CASCADE,
    department_id varchar(64),
    UNIQUE (subject, organization_id),
    FOREIGN KEY (organization_id, department_id)
        REFERENCES {schema}.departments (organization_id, id)
);
CREATE TABLE {schema}.access_revision (
    id serial PRIMARY KEY,
    revision bigint NOT NULL
);
ALTER TABLE {schema}.memberships ADD COLUMN role varchar(16) DEFAULT 'member' NOT NULL;
ALTER TABLE {schema}.memberships
    ADD CONSTRAINT memberships_role_check CHECK (role in ('member', 'admin'));
"""
SEED = """
INSERT INTO {schema}.organizations VALUES ('acme', 'Acme'), ('globex', 'Globex');
"""
OLDER_SEED = """
INSERT INTO {schema}.departments VALUES ('acme/finance', 'acme', 'Finance');
"""


def _execute(dsn: str, sql: str) -> None:
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(sql)


def _create(dsn: str, schema: str = "app", *, older: bool = False) -> None:
    _execute(dsn, f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    _execute(dsn, (OLDER_CONTRACT if older else CONTRACT).format(schema=schema))
    _execute(dsn, SEED.format(schema=schema))
    if older:
        _execute(dsn, OLDER_SEED.format(schema=schema))
