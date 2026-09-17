"""Fresh-process public export recovery after commit without acknowledgement."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from tests.core.test_session_export_mandates import Resolver
from tests.core.test_session_exports import CONTEXT, Harness, published
from tests.core.test_targeted_tool_grants import _codec

from cayu.collaboration.exports import SessionExportRequest, SessionExportUnavailable
from cayu.storage import PostgresSessionStore, SQLiteSessionStore
from cayu.storage.migrations import SchemaMode

_CHILD = r"""
import asyncio
import os
import sys
from pathlib import Path

from cayu.events import EventType
from cayu.storage import PostgresSessionStore, SQLiteSessionStore
from cayu.storage.migrations import SchemaMode
from tests.core.test_session_exports import CONTEXT, Harness
from tests.core.test_targeted_tool_grants import _codec
from tests.core.test_session_export_mandates import Resolver
from tests.core.test_session_export_content_release import ReviewOwner, reviewed_request

kind, session_id, mode = sys.argv[1:]
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
    context = CONTEXT
    if mode == "mandate":
        resolver = Resolver(request)
        app, _, _, _ = case.app(mandates=resolver)
        context = resolver.context
    elif mode == "reviewed_prose":
        owner = ReviewOwner()
        app, _, _, _ = case.app(release_readers=(owner,))
        request, _ = await reviewed_request(case, app, store, owner)
    # Caller intent is saved separately before dispatch. It is not a receipt,
    # private owner state, or permission to expose the result after restart.
    Path(os.environ["CAYU_EXPORT_PROCESS_REQUEST"]).write_text(request.model_dump_json())
    await app.export_session(request, context=context)
    raise AssertionError("Child unexpectedly received the export acknowledgement")

asyncio.run(run())
"""


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize("mode", ["deterministic", "mandate", "reviewed_prose"])
def test_fresh_process_replay_after_durable_commit_without_acknowledgement(
    backend, tmp_path, request, mode
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
    intent_path = tmp_path / "expected-export.json"
    environment["CAYU_EXPORT_PROCESS_REQUEST"] = str(intent_path)
    environment["PYTHONPATH"] = os.pathsep.join((str(repository / "src"), str(repository)))
    child = subprocess.run(
        [sys.executable, "-c", _CHILD, backend, session_id, mode],
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
            export_request = SessionExportRequest.model_validate_json(intent_path.read_text())
            resolver = Resolver(export_request) if mode == "mandate" else None
            context = CONTEXT if resolver is None else resolver.context
            app, store, _, _ = case.app(projectors=(), mandates=resolver)
            observed = await app.lookup_session_export(export_request, context=context)
            assert observed.status == "match"
            receipt = await app.export_session(export_request, context=context)
            assert receipt == observed.receipt
            assert await app.export_session(export_request, context=context) == receipt
            if mode == "reviewed_prose":
                assert receipt.expected.intent.release_receipt is not None
                # Historical receipt recovery needs no live review implementation;
                # missing current release authority still refuses payload exposure.
                with pytest.raises(SessionExportUnavailable):
                    await app.read_session_export(export_request, context=context)
            else:
                assert await app.read_session_export(export_request, context=context) == {
                    "count": 5
                }
            assert [event.id for event in await published(store, session_id)] == [receipt.event_id]
        finally:
            await case.close()

    asyncio.run(reconcile())
