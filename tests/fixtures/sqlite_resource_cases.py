"""Intentional failures, invoked only by the resource-scope subprocess regression."""

import asyncio
import sqlite3


def test_assertion_failure(sqlite_resources):
    async def scenario():
        async with sqlite_resources as scope:
            connection = scope.own(sqlite3.connect(scope.path()), kind="connection")
            scope.own(connection.cursor(), kind="cursor")
            raise AssertionError("controlled test-body failure")

    asyncio.run(scenario())


def test_missing_context(sqlite_resources):
    pass


def test_following_test_has_no_leftovers(sqlite_resources, tmp_path):
    # The new fixture root is the only scope root permitted in the test session.
    roots = list(tmp_path.parent.rglob("sqlite-scope-*"))
    assert roots == [sqlite_resources.root]

    async def scenario():
        async with sqlite_resources as scope:
            connection = scope.own(sqlite3.connect(scope.path()), kind="connection")
            cursor = scope.own(connection.cursor(), kind="cursor")
            assert cursor.execute("SELECT 1").fetchone() == (1,)

    asyncio.run(scenario())
