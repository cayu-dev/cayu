"""Fresh-process public export recovery after commit without acknowledgement."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from tests.core.test_session_exports import CONTEXT, Harness, published
from tests.core.test_targeted_tool_grants import _codec

from cayu.storage import PostgresSessionStore, SQLiteSessionStore
from cayu.storage.migrations import SchemaMode

_CHILD = r"""
import asyncio
import os
import sys

from cayu.events import EventType
from cayu.storage import PostgresSessionStore, SQLiteSessionStore
from cayu.storage.migrations import SchemaMode
from tests.core.test_session_exports import CONTEXT, Harness
from tests.core.test_targeted_tool_grants import _codec

kind, session_id = sys.argv[1:]
locator = os.environ["CAYU_EXPORT_PROCESS_STORE"]
base = SQLiteSessionStore if kind == "sqlite" else PostgresSessionStore

class ExitAfterCommit(base):
    # Re-attest the test wrapper: native ownership and commit run unchanged.
    # The process exits only after the real backend acknowledges durable commit,
    # before the export coordinator or its public caller gets that acknowledgement.
    session_export_version = 1

    async def publish_session_operation_guarded_with_store_time(self, *args, **kwargs):
        result = await super().publish_session_operation_guarded_with_store_time(*args, **kwargs)
        if any(event.type == EventType.SESSION_EXPORT_PUBLISHED for event in kwargs["events"]):
            os._exit(23)
        return result

def factory(store_type=None):
    options = {"public_authority_alias_codec": _codec()}
    if kind == "postgres":
        options["schema_mode"] = SchemaMode.CREATE
    return ExitAfterCommit(locator, **options)

async def run():
    case = Harness(factory)
    case.session_id = session_id
    app, store, _, _ = case.app()
    await case.create(store)
    request = await case.request(app)
    await app.export_session(request, context=CONTEXT)
    raise AssertionError("Child unexpectedly received the export acknowledgement")

asyncio.run(run())
"""


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_fresh_process_replay_after_durable_commit_without_acknowledgement(
    backend, tmp_path, request
):
    locator = (
        request.getfixturevalue("postgres_dsn")
        if backend == "postgres"
        else str(tmp_path / "process-export.sqlite")
    )
    session_id = f"process-export-{uuid4().hex}"
    repository = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["CAYU_EXPORT_PROCESS_STORE"] = locator
    environment["PYTHONPATH"] = os.pathsep.join((str(repository / "src"), str(repository)))
    child = subprocess.run(
        [sys.executable, "-c", _CHILD, backend, session_id],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert child.returncode == 23, child.stderr
    assert child.stdout == ""

    def factory(store_type=None):
        if backend == "sqlite":
            return SQLiteSessionStore(locator, public_authority_alias_codec=_codec())
        return PostgresSessionStore(
            locator, schema_mode=SchemaMode.CREATE, public_authority_alias_codec=_codec()
        )

    async def reconcile():
        case = Harness(factory)
        case.session_id = session_id
        try:
            # No old projector exists in this process's application registration.
            app, store, _, _ = case.app(projectors=())
            export_request = await case.request(app)
            observed = await app.lookup_session_export(export_request, context=CONTEXT)
            assert observed.status == "match"
            receipt = await app.export_session(export_request, context=CONTEXT)
            assert receipt == observed.receipt
            assert await app.export_session(export_request, context=CONTEXT) == receipt
            assert await app.read_session_export(export_request, context=CONTEXT) == {"count": 5}
            assert [event.id for event in await published(store, session_id)] == [receipt.event_id]
        finally:
            await case.close()

    asyncio.run(reconcile())
