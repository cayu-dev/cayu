"""Native clarification schema qualification, independent of runtime capability."""

import sqlite3
from contextlib import closing

import pytest

from cayu.storage._collaboration_schema import (
    SQLITE_COLLABORATION_CLARIFICATION_DDL,
    SQLITE_COLLABORATION_DDL,
    SQLITE_COLLABORATION_LIFECYCLE_DDL,
    SQLITE_COLLABORATION_REQUEST_DDL,
    validate_sqlite_collaboration_schema,
)
from cayu.storage.migrations import SchemaError


def _install(connection):
    for ddl in (
        SQLITE_COLLABORATION_DDL,
        SQLITE_COLLABORATION_LIFECYCLE_DDL,
        SQLITE_COLLABORATION_REQUEST_DDL,
        SQLITE_COLLABORATION_CLARIFICATION_DDL,
    ):
        connection.executescript(ddl)


def _validate(connection):
    validate_sqlite_collaboration_schema(
        connection, lifecycle=True, requests=True, clarifications=True
    )


def test_clarification_schema_survives_reopen(tmp_path):
    path = tmp_path / "clarifications.sqlite"
    with closing(sqlite3.connect(path)) as connection:
        _install(connection)
        _validate(connection)
    with closing(sqlite3.connect(path)) as connection:
        _validate(connection)
        # Reinstallation cannot alter identities or accidentally duplicate indexes.
        _install(connection)
        _validate(connection)


@pytest.mark.parametrize("family", ["questions", "inputs", "lineages", "services", "deliveries"])
def test_clarification_schema_rejects_missing_table(family):
    with closing(sqlite3.connect(":memory:")) as connection:
        _install(connection)
        connection.execute(f"DROP TABLE cayu_collaboration_clarification_{family}")
        with pytest.raises(SchemaError):
            _validate(connection)


@pytest.mark.parametrize(
    "name",
    [
        "due",
        "request",
        "service_pending",
        "delivery_pending",
        "service_request",
        "delivery_request",
        "lineage_question",
        "service_request_pending",
        "delivery_request_pending",
    ],
)
def test_clarification_schema_rejects_divergent_index(name):
    with closing(sqlite3.connect(":memory:")) as connection:
        _install(connection)
        index = f"cayu_collaboration_clarification_{name}_idx"
        connection.execute(f"DROP INDEX {index}")
        connection.execute(
            f"CREATE INDEX {index} ON cayu_collaboration_clarification_questions(scope)"
        )
        with pytest.raises(SchemaError):
            _validate(connection)


def test_pre_clarification_schema_is_not_misreported_as_qualified():
    with closing(sqlite3.connect(":memory:")) as connection:
        for ddl in (
            SQLITE_COLLABORATION_DDL,
            SQLITE_COLLABORATION_LIFECYCLE_DDL,
            SQLITE_COLLABORATION_REQUEST_DDL,
        ):
            connection.executescript(ddl)
        validate_sqlite_collaboration_schema(connection, lifecycle=True, requests=True)
        with pytest.raises(SchemaError):
            _validate(connection)


def test_clarification_schema_requires_native_request_pruning_progress():
    with closing(sqlite3.connect(":memory:")) as connection:
        _install(connection)
        connection.execute("DROP TABLE cayu_collaboration_request_pruning")
        with pytest.raises(SchemaError):
            _validate(connection)
