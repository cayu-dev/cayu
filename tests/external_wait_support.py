"""Shared external-wait scenarios use the same native backend construction."""

from contextlib import asynccontextmanager
from uuid import uuid4

from cayu.external_waits import ExternalWaitAccessPolicy, ExternalWaitContext
from cayu.sessions.base import InMemorySessionStore
from cayu.sessions.external_waits import (
    ExternalCorrelationRequest,
    ExternalWaitRegistration,
    ExternalWaitScope,
)
from cayu.storage.migrations import SchemaMode
from cayu.storage.postgres import PostgresSessionStore
from cayu.storage.sqlite import SQLiteSessionStore


class Policy(ExternalWaitAccessPolicy):
    revoked = False

    def authorize(self, context, *, scope, source, action):
        return not self.revoked and context.principal == "host" and source == "renderer"


CONTEXT = ExternalWaitContext(principal="host")


@asynccontextmanager
async def stores(backend, tmp_path, request, clock, *, public_authority_alias_codec=None):
    opened = []
    memory = InMemorySessionStore(
        ownership_clock=lambda: clock[0], public_authority_alias_codec=public_authority_alias_codec
    )
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    def reopen():
        if backend == "memory":
            return memory
        if backend == "sqlite":
            store = SQLiteSessionStore(
                tmp_path / "waits.sqlite",
                ownership_clock=lambda: clock[0],
                public_authority_alias_codec=public_authority_alias_codec,
            )
        else:
            store = PostgresSessionStore(
                dsn,
                schema_mode=SchemaMode.CREATE,
                min_size=1,
                max_size=2,
                public_authority_alias_codec=public_authority_alias_codec,
            )
        opened.append(store)
        return store

    try:
        yield reopen(), reopen
    finally:
        for store in opened:
            await store.close()


def reservation(**kwargs):
    return ExternalCorrelationRequest(
        scope=ExternalWaitScope(application_scope=uuid4().hex, generation=1),
        source="renderer",
        correlation_key="job",
        **kwargs,
    )


def registration(correlation):
    return ExternalWaitRegistration(
        correlation=correlation, operation_key="wait", projector_id="json", projector_version=1
    )
