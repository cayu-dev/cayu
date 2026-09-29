"""Environment-selected session, task, and knowledge stores for applications."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cayu.storage._diagnostic_inspection import current_diagnostic_store_inspection
from cayu.storage.targets import (
    DATABASE_DIRECT_URL_ENV,
    SessionStoreBackend,
    SessionStoreTargetError,
    application_store_target,
    configured_database_direct_url,
    configured_database_pool_max,
    parse_database_url,
    require_sqlite_store_allowed,
)

if TYPE_CHECKING:
    from cayu.knowledge.scopes import KnowledgeAccessScope
    from cayu.runtime.public_authority import PublicAuthorityAliasCodec
    from cayu.sessions.base import SessionStore
    from cayu.storage.memory import KnowledgeStore
    from cayu.tasks.base import TaskStore


class ApplicationStores:
    """The stores selected by :func:`open_application_stores` and their resources.

    The object owns every store it built and, for PostgreSQL, the shared
    connection pool. ``close()`` closes the stores and then the pool, at most
    once. Cayu's server does not close application stores; long-lived processes
    keep them for their lifetime, and tests or scripts close them explicitly.
    """

    __slots__ = (
        "_backend",
        "_close_lock",
        "_closed",
        "_knowledge_store",
        "_pool",
        "_pool_max_size",
        "_session_store",
        "_task_admission_listener",
        "_task_store",
    )

    def __init__(
        self,
        *,
        backend: SessionStoreBackend,
        session_store: SessionStore,
        task_store: TaskStore | None,
        knowledge_store: KnowledgeStore | None,
        pool: Any | None = None,
        pool_max_size: int | None = None,
        task_admission_listener: bool = False,
    ) -> None:
        self._backend = backend
        self._session_store = session_store
        self._task_store = task_store
        self._knowledge_store = knowledge_store
        self._pool = pool
        self._pool_max_size = pool_max_size
        self._task_admission_listener = task_admission_listener
        self._close_lock = threading.Lock()
        self._closed = False

    def __repr__(self) -> str:
        return (
            "ApplicationStores("
            f"backend={self._backend.value!r}, "
            f"tasks={self._task_store is not None!r}, "
            f"knowledge={self._knowledge_store is not None!r}, "
            f"pool_max_size={self._pool_max_size!r})"
        )

    @property
    def backend(self) -> SessionStoreBackend:
        return self._backend

    @property
    def session_store(self) -> SessionStore:
        return self._session_store

    @property
    def task_store(self) -> TaskStore | None:
        return self._task_store

    @property
    def knowledge_store(self) -> KnowledgeStore | None:
        return self._knowledge_store

    @property
    def pool_max_size(self) -> int | None:
        """Maximum pooled PostgreSQL connections shared by the stores, if any."""

        return self._pool_max_size

    @property
    def connection_budget(self) -> int:
        """Most PostgreSQL connections these stores open (pool plus listener)."""

        if self._backend is SessionStoreBackend.SQLITE:
            return 0
        pooled = self._pool_max_size or 0
        return pooled + (1 if self._task_admission_listener else 0)

    async def close(self) -> None:
        """Close the stores, then the shared pool; later calls do nothing."""

        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        errors: list[Exception] = []
        # The task store first: it owns the dedicated LISTEN connection.
        for store in (self._task_store, self._knowledge_store, self._session_store):
            close = getattr(store, "close", None)
            if close is None:
                continue
            try:
                await close()
            except Exception as exc:
                errors.append(exc)
        if self._pool is not None:
            try:
                await self._pool.close()
            except Exception as exc:
                errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup("Application stores did not close cleanly.", errors)

    async def __aenter__(self) -> ApplicationStores:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()


def open_application_stores(
    database_url: str | None,
    *,
    sqlite_path: str | os.PathLike[str],
    tasks: bool = True,
    knowledge_scope: KnowledgeAccessScope | None = None,
    public_authority_alias_codec: PublicAuthorityAliasCodec | None = None,
    pool_max_size: int | None = None,
    direct_database_url: str | None = None,
) -> ApplicationStores:
    """Build the application's session, task, and knowledge stores.

    ``database_url`` is normally ``configured_database_url()``, the value of
    ``CAYU_DATABASE_URL``. A ``postgres://`` or ``postgresql://`` URL builds the
    PostgreSQL stores on one shared connection pool of at most ``pool_max_size``
    connections (default ``CAYU_DATABASE_POOL_MAX``, else 5). They validate the
    schema at first use; migrations are an explicit deploy step
    (``cayu storage migrate``). The task store keeps one dedicated ``LISTEN``
    connection outside the pool for task-admission wakeups. It connects to
    ``direct_database_url`` (default ``CAYU_DATABASE_DIRECT_URL``, else
    ``database_url``), because ``LISTEN`` does not survive transaction pooling.

    ``None`` builds SQLite stores in ``sqlite_path``, which must be absolute, with
    SQLite's local-development schema behavior. An absolute ``sqlite:///`` URL
    selects that file instead, matching the CLI's store resolution. Any other URL
    scheme raises :class:`~cayu.storage.targets.SessionStoreTargetError`.

    ``tasks`` and ``knowledge_scope`` choose which optional stores exist. Nothing
    connects during construction.
    """

    if type(tasks) is not bool:
        raise TypeError("tasks must be a bool.")
    target = application_store_target(database_url, sqlite_path=sqlite_path)
    if target.backend is SessionStoreBackend.SQLITE:
        assert target.sqlite_path is not None
        return _open_sqlite_stores(
            target.sqlite_path,
            tasks=tasks,
            knowledge_scope=knowledge_scope,
            public_authority_alias_codec=public_authority_alias_codec,
        )
    assert target.postgres_dsn is not None
    if pool_max_size is None:
        pool_max_size = configured_database_pool_max()
    elif type(pool_max_size) is not int or pool_max_size < 1:
        raise ValueError("pool_max_size must be a positive integer.")
    if direct_database_url is None:
        direct_database_url = configured_database_direct_url()
    listener_dsn = target.postgres_dsn
    if direct_database_url is not None:
        direct = parse_database_url(
            direct_database_url, source=f"environment:{DATABASE_DIRECT_URL_ENV}"
        )
        if direct.backend is not SessionStoreBackend.POSTGRES or direct.postgres_dsn is None:
            raise SessionStoreTargetError(f"{DATABASE_DIRECT_URL_ENV} must contain a Postgres URL.")
        listener_dsn = direct.postgres_dsn
    return _open_postgres_stores(
        target.postgres_dsn,
        listener_dsn=listener_dsn,
        pool_max_size=pool_max_size,
        tasks=tasks,
        knowledge_scope=knowledge_scope,
        public_authority_alias_codec=public_authority_alias_codec,
    )


def _open_sqlite_stores(
    path: Path,
    *,
    tasks: bool,
    knowledge_scope: KnowledgeAccessScope | None,
    public_authority_alias_codec: PublicAuthorityAliasCodec | None,
) -> ApplicationStores:
    from cayu.storage.knowledge_sqlite import SQLiteKnowledgeStore
    from cayu.storage.sqlite import SQLiteSessionStore
    from cayu.storage.tasks_sqlite import SQLiteTaskStore

    # Fail before opening any file when the deployment requires PostgreSQL.
    require_sqlite_store_allowed("The application's SQLite stores")
    session_store = SQLiteSessionStore(
        path,
        public_authority_alias_codec=public_authority_alias_codec,
    )
    task_store = SQLiteTaskStore(path) if tasks else None
    knowledge_store = (
        SQLiteKnowledgeStore(path, access_scope=knowledge_scope)
        if knowledge_scope is not None
        else None
    )
    return ApplicationStores(
        backend=SessionStoreBackend.SQLITE,
        session_store=session_store,
        task_store=task_store,
        knowledge_store=knowledge_store,
    )


def _open_postgres_stores(
    dsn: str,
    *,
    listener_dsn: str,
    pool_max_size: int,
    tasks: bool,
    knowledge_scope: KnowledgeAccessScope | None,
    public_authority_alias_codec: PublicAuthorityAliasCodec | None,
) -> ApplicationStores:
    try:
        from psycopg_pool import AsyncConnectionPool

        from cayu.storage.migrations import SchemaMode
        from cayu.storage.postgres import (
            PostgresKnowledgeStore,
            PostgresSessionStore,
            PostgresTaskStore,
        )
    except ModuleNotFoundError as exc:
        if (exc.name or "").partition(".")[0] not in {"psycopg", "psycopg_pool"}:
            raise
        raise SessionStoreTargetError(
            'PostgreSQL application stores require the postgres extra. Install "cayu[postgres]".'
        ) from exc

    if current_diagnostic_store_inspection() is not None:
        # Diagnostic inspection opens read-only stores, which must own their pools.
        session_store = PostgresSessionStore(
            dsn,
            max_size=pool_max_size,
            public_authority_alias_codec=public_authority_alias_codec,
        )
        task_store = PostgresTaskStore(dsn, max_size=pool_max_size) if tasks else None
        knowledge_store = (
            PostgresKnowledgeStore(dsn, max_size=pool_max_size, access_scope=knowledge_scope)
            if knowledge_scope is not None
            else None
        )
        return ApplicationStores(
            backend=SessionStoreBackend.POSTGRES,
            session_store=session_store,
            task_store=task_store,
            knowledge_store=knowledge_store,
        )

    pool = AsyncConnectionPool(
        dsn,
        min_size=1,
        max_size=pool_max_size,
        open=False,
        # Disable server-side prepared statements so pooled traffic works behind a
        # transaction-pooling proxy. A connect option, not a configure callback,
        # keeps the pool acceptable to the task store's mutation boundary.
        kwargs={"prepare_threshold": None},
    )
    session_store = PostgresSessionStore(
        pool=pool,
        schema_mode=SchemaMode.VALIDATE,
        public_authority_alias_codec=public_authority_alias_codec,
    )
    task_store = (
        PostgresTaskStore(
            pool=pool,
            schema_mode=SchemaMode.VALIDATE,
            task_admission_listener_conninfo=listener_dsn,
        )
        if tasks
        else None
    )
    knowledge_store = (
        PostgresKnowledgeStore(
            pool=pool,
            schema_mode=SchemaMode.VALIDATE,
            access_scope=knowledge_scope,
        )
        if knowledge_scope is not None
        else None
    )
    return ApplicationStores(
        backend=SessionStoreBackend.POSTGRES,
        session_store=session_store,
        task_store=task_store,
        knowledge_store=knowledge_store,
        pool=pool,
        pool_max_size=pool_max_size,
        task_admission_listener=task_store is not None,
    )


__all__ = ["ApplicationStores", "open_application_stores"]
