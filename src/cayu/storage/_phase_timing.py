"""Store-boundary measurements; no SQL text, bound values or extra queries escape."""

from __future__ import annotations

import asyncio
from time import monotonic
from typing import Any, cast

from cayu.runtime._phase_timing import current_store_counters


def _verb(query):
    # Runtime store SQL is plain static text. Do not stringify driver composables
    # or arbitrary extension objects just to observe them.
    return query.lstrip().split(None, 1)[0].upper() if type(query) is str and query.strip() else ""


def _write(query):
    return _verb(query) in {"INSERT", "UPDATE", "DELETE", "REPLACE"}


def _bytes(value):
    if type(value) is str:
        return len(value.encode("utf-8"))
    if isinstance(value, bytes | bytearray | memoryview):
        return value.nbytes if isinstance(value, memoryview) else len(value)
    if value is None:
        return 0
    if type(value) is bool:
        return 1
    if type(value) is int or type(value) is float:
        return 8
    return 0


def _bound_bytes(parameters):
    values = parameters.values() if type(parameters) is dict else parameters
    return (
        sum(_bytes(value) for value in values)
        if isinstance(values, (list, tuple))
        else (sum(_bytes(value) for value in values) if type(parameters) is dict else 0)
    )


class TimedStoreLock(asyncio.Lock):
    async def acquire(self):
        counters = current_store_counters()
        if counters is None:
            return await super().acquire()
        started = monotonic()
        try:
            return await super().acquire()
        finally:
            counters.add(store_lock_wait_seconds=max(0, monotonic() - started))


class TimedStoreReadQueue(asyncio.LifoQueue):
    async def get(self):
        counters = current_store_counters()
        if counters is None:
            return await super().get()
        started = monotonic()
        try:
            return await super().get()
        finally:
            counters.add(store_lock_wait_seconds=max(0, monotonic() - started))


class SQLiteTimingCursor:
    def __init__(self, cursor, connection):
        self._cursor = cursor
        self._connection = connection

    def __getattr__(self, name):
        return getattr(self._cursor, name)

    def execute(self, query, parameters=()):
        self._connection._execute(self._cursor.execute, query, parameters)
        return self

    def executemany(self, query, parameters):
        self._connection._execute_many(self._cursor.executemany, query, parameters)
        return self

    def _fetch(self, function, *args):
        counters = current_store_counters()
        if counters is None:
            return function(*args)
        started = monotonic()
        try:
            return function(*args)
        finally:
            counters.add(store_execution_seconds=max(0, monotonic() - started))

    def fetchone(self):
        return self._fetch(self._cursor.fetchone)

    def fetchall(self):
        return self._fetch(self._cursor.fetchall)

    def fetchmany(self, *args):
        return self._fetch(self._cursor.fetchmany, *args)

    def __iter__(self):
        return self

    def __next__(self):
        return self._fetch(next, self._cursor)


class SQLiteTimingConnection:
    def __init__(self, connection):
        object.__setattr__(self, "_connection", connection)

    def __getattr__(self, name):
        return getattr(self._connection, name)

    def __setattr__(self, name, value):
        setattr(self._connection, name, value)

    def _execute(self, function, query, parameters=()):
        counters = current_store_counters()
        if counters is None:
            return function(query, parameters)
        before = self._connection.in_transaction
        started = monotonic()
        try:
            result = function(query, parameters)
        finally:
            elapsed = max(0, monotonic() - started)
            lock_acquisition = _verb(query) == "BEGIN" and "IMMEDIATE" in query.upper()
            counters.add(
                store_execution_seconds=elapsed,
                store_lock_wait_seconds=elapsed if lock_acquisition else 0,
            )
        if not before and (
            _verb(query) in {"SELECT", "WITH", "BEGIN"} or self._connection.in_transaction
        ):
            counters.add(store_transaction_count=1)
        if _write(query):
            counters.add(store_bytes_written=_bound_bytes(parameters))
        return result

    def _execute_many(self, function, query, parameters):
        counters = current_store_counters()
        if counters is None:
            return function(query, parameters)
        written = 0

        def counted():
            nonlocal written
            for row in parameters:
                written += _bound_bytes(row)
                yield row

        before = self._connection.in_transaction
        started = monotonic()
        try:
            return function(query, counted())
        finally:
            counters.add(
                store_execution_seconds=max(0, monotonic() - started),
                store_transaction_count=int(not before and self._connection.in_transaction),
                store_bytes_written=written if _write(query) else 0,
            )

    def execute(self, query, parameters=()):
        if current_store_counters() is None:
            # Outside a measured phase, and always when timing is disabled,
            # return the driver's cursor so rows are not iterated in Python.
            return self._connection.execute(query, parameters)
        return SQLiteTimingCursor(self._execute(self._connection.execute, query, parameters), self)

    def executemany(self, query, parameters):
        if current_store_counters() is None:
            return self._connection.executemany(query, parameters)
        return SQLiteTimingCursor(
            self._execute_many(self._connection.executemany, query, parameters), self
        )

    def cursor(self, *args, **kwargs):
        return SQLiteTimingCursor(self._connection.cursor(*args, **kwargs), self)

    def commit(self):
        counters = current_store_counters()
        if counters is None or not self._connection.in_transaction:
            return self._connection.commit()
        started = monotonic()
        try:
            return self._connection.commit()
        finally:
            counters.add(store_commit_seconds=max(0, monotonic() - started))

    def rollback(self):
        return self._connection.rollback()

    def __enter__(self):
        self._connection.__enter__()
        return self

    def __exit__(self, *arguments):
        counters = current_store_counters()
        if counters is None or not self._connection.in_transaction:
            return self._connection.__exit__(*arguments)
        started = monotonic()
        try:
            return self._connection.__exit__(*arguments)
        finally:
            if arguments[0] is None:
                counters.add(store_commit_seconds=max(0, monotonic() - started))


def timed_sqlite_connection(connection):
    # The proxy delegates all driver behavior; it does not replace the driver's
    # transaction implementation, row factory, authorizer or cancellation handle.
    import sqlite3

    return cast("sqlite3.Connection", SQLiteTimingConnection(connection))


def _postgres_idle(connection):
    return getattr(getattr(connection, "info", None), "transaction_status", None) == 0


def _postgres_parameters(cursor, parameters, counters):
    """Count actual adapted JSON bytes once, preserving the driver's dumps function."""
    from psycopg.adapt import PyFormat
    from psycopg.types.json import Json, Jsonb

    written = 0

    def adapt(value):
        nonlocal written
        if isinstance(value, Json | Jsonb):
            dumps = (
                value.dumps
                or cursor.adapters.get_dumper(type(value), PyFormat.AUTO)(type(value), cursor).dumps
            )

            def counted(obj):
                data = dumps(obj)
                if isinstance(data, str):
                    data = data.encode()
                counters.add(store_bytes_written=len(data))
                return data

            return type(value)(value.obj, dumps=counted)
        written += _bytes(value)
        return value

    if isinstance(parameters, (tuple, list)):
        result = tuple(adapt(value) for value in parameters)
    elif type(parameters) is dict:
        result = {key: adapt(value) for key, value in parameters.items()}
    else:
        return parameters
    counters.add(store_bytes_written=written)
    return result


class PostgresTimingCursor:
    def __init__(self, cursor, connection):
        self._cursor = cursor
        self._connection = connection

    def __getattr__(self, name):
        return getattr(self._cursor, name)

    async def __aenter__(self):
        await self._cursor.__aenter__()
        return self

    async def __aexit__(self, *arguments):
        return await self._cursor.__aexit__(*arguments)

    async def execute(self, query, parameters=None, **kwargs):
        counters = current_store_counters()
        if counters is None:
            await self._cursor.execute(query, parameters, **kwargs)
            return self
        before = _postgres_idle(self._connection)
        if _write(query):
            parameters = _postgres_parameters(self._cursor, parameters, counters)
        started = monotonic()
        try:
            await self._cursor.execute(query, parameters, **kwargs)
            return self
        finally:
            elapsed = max(0, monotonic() - started)
            lock_acquisition = type(query) is str and "FOR UPDATE" in query.upper()
            counters.add(
                store_execution_seconds=elapsed,
                store_lock_wait_seconds=elapsed if lock_acquisition else 0,
                store_transaction_count=int(before and not _postgres_idle(self._connection)),
            )

    async def executemany(self, query, parameters, **kwargs):
        counters = current_store_counters()
        if counters is None:
            await self._cursor.executemany(query, parameters, **kwargs)
            return self
        before = _postgres_idle(self._connection)
        values = (
            (_postgres_parameters(self._cursor, row, counters) for row in parameters)
            if _write(query)
            else parameters
        )
        started = monotonic()
        try:
            await self._cursor.executemany(query, values, **kwargs)
            return self
        finally:
            counters.add(
                store_execution_seconds=max(0, monotonic() - started),
                store_transaction_count=int(before and not _postgres_idle(self._connection)),
            )

    async def _fetch(self, function, *args):
        counters = current_store_counters()
        if counters is None:
            return await function(*args)
        started = monotonic()
        try:
            return await function(*args)
        finally:
            counters.add(store_execution_seconds=max(0, monotonic() - started))

    async def fetchone(self):
        return await self._fetch(self._cursor.fetchone)

    async def fetchall(self):
        return await self._fetch(self._cursor.fetchall)

    async def fetchmany(self, *args):
        return await self._fetch(self._cursor.fetchmany, *args)

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self._fetch(anext, self._cursor)


class PostgresTimingConnection:
    def __init__(self, connection):
        self._connection = connection

    def __getattr__(self, name):
        return getattr(self._connection, name)

    def cursor(self, *args, **kwargs):
        return PostgresTimingCursor(self._connection.cursor(*args, **kwargs), self._connection)

    async def execute(self, query, parameters=None, **kwargs):
        return await self.cursor().execute(query, parameters, **kwargs)

    def transaction(self, *args, **kwargs):
        return PostgresTimingTransaction(
            self._connection.transaction(*args, **kwargs), self._connection
        )

    async def commit(self):
        counters = current_store_counters()
        if counters is None or _postgres_idle(self._connection):
            return await self._connection.commit()
        started = monotonic()
        try:
            return await self._connection.commit()
        finally:
            counters.add(store_commit_seconds=max(0, monotonic() - started))


class PostgresTimingTransaction:
    def __init__(self, scope, connection):
        self._scope = scope
        self._connection = connection
        self._outer = False

    async def __aenter__(self):
        counters = current_store_counters()
        if counters is None:
            return await self._scope.__aenter__()
        self._outer = _postgres_idle(self._connection)
        started = monotonic()
        try:
            return await self._scope.__aenter__()
        finally:
            counters.add(
                store_execution_seconds=max(0, monotonic() - started),
                store_transaction_count=int(self._outer and not _postgres_idle(self._connection)),
            )

    async def __aexit__(self, *arguments):
        counters = current_store_counters()
        if counters is None:
            return await self._scope.__aexit__(*arguments)
        started = monotonic()
        try:
            return await self._scope.__aexit__(*arguments)
        finally:
            metric = (
                "store_commit_seconds"
                if self._outer and arguments[0] is None
                else "store_execution_seconds"
            )
            counters.add(**{metric: max(0, monotonic() - started)})


class PostgresTimingScope:
    """Measure pool acquisition and implicit commit without reconfiguring a pool."""

    def __init__(self, scope, *, raw=False):
        self._scope = scope
        self._raw = raw
        self._connection: Any = None

    async def __aenter__(self):
        counters = current_store_counters()
        if counters is None:
            self._connection = await self._scope.__aenter__()
            return self._connection
        started = monotonic()
        try:
            self._connection = await self._scope.__aenter__()
            return self._connection if self._raw else PostgresTimingConnection(self._connection)
        finally:
            counters.add(store_lock_wait_seconds=max(0, monotonic() - started))

    async def __aexit__(self, *arguments):
        counters = current_store_counters()
        if counters is None or _postgres_idle(self._connection) or arguments[0] is not None:
            return await self._scope.__aexit__(*arguments)
        started = monotonic()
        try:
            return await self._scope.__aexit__(*arguments)
        finally:
            counters.add(store_commit_seconds=max(0, monotonic() - started))


def timed_postgres_connection(connection):
    return connection if current_store_counters() is None else PostgresTimingConnection(connection)
