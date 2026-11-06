"""Run the alias configuration check in the same round trip as an operation's first statements.

The session store checks the public authority alias configuration before every
operation, so a worker whose keyring went stale fails its next read or write.
Sent as its own autocommit statement, the check costs a round trip of its own.
Here it is pipelined after the operation's first statements, and its result is
verified before theirs reach the caller. A stale worker still fails the
operation; whatever its first statements wrote is in the open transaction,
which rolls back with the failure. (Psycopg still sends BEGIN on its own: it
syncs after BEGIN in pipeline mode.)

On an autocommit connection, only for reads: there the first statements and the
check share one implicit transaction, and a write would commit before the check
is verified.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from cayu.storage._phase_timing import timed_postgres_connection

Statement = tuple[Any, Any, Any]


class AliasCheckedConnection:
    """A connection whose first statement carries the alias configuration check."""

    def __init__(
        self,
        connection: Any,
        *,
        query: str,
        verify: Callable[[tuple[Any, ...] | None], None],
    ) -> None:
        self._connection = connection
        self._query = query
        self._verify = verify
        self._pending = True

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)

    def cursor(self, *args: Any, **kwargs: Any) -> AliasCheckedCursor:
        # A named cursor declares a server-side cursor, which a pipeline can't hold.
        named = bool(args) or kwargs.get("name") is not None
        return AliasCheckedCursor(self._connection.cursor(*args, **kwargs), self, named=named)

    async def execute(self, query: Any, parameters: Any = None, **kwargs: Any) -> Any:
        return await self.cursor().execute(query, parameters, **kwargs)

    async def _run_first(self, statements: Sequence[Callable[[], Awaitable[Any]]]) -> None:
        async with self._connection.cursor() as checker:
            async with self._connection.pipeline():
                # After the operation's statements, so its first statement still
                # takes a repeatable-read transaction's snapshot.
                for statement in statements:
                    await statement()
                await checker.execute(self._query)
            self._verify(await checker.fetchone())
        # Only a verified check is paid: after a failed first statement (and a
        # rollback the caller recovers from) the next statement checks again.
        self._pending = False

    async def _run_check_alone(self) -> None:
        await self._run_first(())


class AliasCheckedCursor:
    def __init__(self, cursor: Any, owner: AliasCheckedConnection, *, named: bool) -> None:
        self._cursor = cursor
        self._owner = owner
        self._named = named

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cursor, name)

    async def __aenter__(self) -> AliasCheckedCursor:
        await self._cursor.__aenter__()
        return self

    async def __aexit__(self, *arguments: Any) -> Any:
        return await self._cursor.__aexit__(*arguments)

    def __aiter__(self) -> Any:
        return self._cursor.__aiter__()

    async def execute(
        self, query: Any, parameters: Any = None, **kwargs: Any
    ) -> AliasCheckedCursor:
        if self._owner._pending and not _is_transaction_setting(query):
            if self._named:
                await self._owner._run_check_alone()
            else:
                await self._owner._run_first(
                    (lambda: self._cursor.execute(query, parameters, **kwargs),)
                )
                return self
        await self._cursor.execute(query, parameters, **kwargs)
        return self

    async def executemany(self, query: Any, parameters: Any, **kwargs: Any) -> AliasCheckedCursor:
        if self._owner._pending:
            await self._owner._run_check_alone()
        await self._cursor.executemany(query, parameters, **kwargs)
        return self


async def execute_pipelined(statements: Sequence[Statement]) -> None:
    """Execute (cursor, query, parameters) statements in one round trip.

    The server runs them one after the other, each with its own statement
    snapshot. A pending alias check on the first cursor's connection rides along.
    """

    first = statements[0][0]
    runs = tuple(
        (
            lambda cursor=cursor, query=query, parameters=parameters: _raw(cursor).execute(
                query, parameters
            )
        )
        for cursor, query, parameters in statements
    )
    if isinstance(first, AliasCheckedCursor) and first._owner._pending:
        await first._owner._run_first(runs)
        return
    async with timed_postgres_connection(first.connection).pipeline():
        for run in runs:
            await run()


def _is_transaction_setting(query: Any) -> bool:
    # SET TRANSACTION must come before any query, so the check waits for the next statement.
    return type(query) is str and query.lstrip()[:4].upper() == "SET "


def _raw(cursor: Any) -> Any:
    return cursor._cursor if isinstance(cursor, AliasCheckedCursor) else cursor
