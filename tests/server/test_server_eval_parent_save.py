"""Parent-aware authored saves preserve only positively owned case source links."""

from __future__ import annotations

import json
import os

import pytest
from fastapi.testclient import TestClient
from tests.server.test_server_eval_session_additions import (
    _destination,
    _review,
)
from tests.server.test_server_eval_session_additions import (
    addition_environment as addition_environment,
)
from tests.server.test_server_evaluation_promotion import _AUTH_HEADERS, _SESSION_ID

from cayu.evals.read_views import EvalSuiteViewIdentity
from cayu.evals.suite_authoring import (
    EvalSuiteDraftV3,
    eval_suite_document_from_json,
)


@pytest.fixture
def parent_save_postgres_dsn(request):
    """Each case needs a fresh catalog; reopen within a case retains that catalog."""
    import psycopg
    from psycopg import sql

    dsn = request.getfixturevalue("postgres_dsn")
    # The shared test DSN belongs to this module/worker, not to one parameter
    # case. Fixed immutable suite identities must not replay a previous test.
    with psycopg.connect(dsn) as connection:
        tables = connection.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = current_schema() "
            "AND starts_with(tablename, 'cayu_eval_') ORDER BY tablename"
        ).fetchall()
        if tables:
            connection.execute(
                sql.SQL("TRUNCATE {} CASCADE").format(
                    sql.SQL(", ").join(sql.Identifier(row[0]) for row in tables)
                )
            )
    return dsn


def test_parent_save_request_contract_keeps_parent_optional_and_strict(addition_environment):
    server, _, _ = addition_environment()
    schema = server.openapi()["components"]["schemas"]["EvalSuiteSaveRequest"]
    assert "parent_suite_revision" in schema["properties"]
    assert "parent_suite_revision" not in schema["required"]


def _linked_parent(client):
    selector, _ = _destination(client, "authored")
    body, plan = _review(client, selector)
    saved = client.post(
        f"/api/evals/sessions/{_SESSION_ID}/additions",
        headers={**_AUTH_HEADERS, "Idempotency-Key": "parent-save-source"},
        json=body,
    )
    assert saved.status_code == 201, saved.text
    identity = EvalSuiteViewIdentity.model_validate(plan["destination"])
    response = client.get(f"/api/evals/suites/{identity.revision}", headers=_AUTH_HEADERS)
    assert response.status_code == 200, response.text
    return response.json(), plan["case_id"]


def _edited(client, parent, **changes):
    draft = EvalSuiteDraftV3.from_document(eval_suite_document_from_json(json.dumps(parent)))
    draft = draft.model_copy(update={"name": "Edited suite", **changes})
    response = client.post(
        "/api/evals/suites/preview",
        headers=_AUTH_HEADERS,
        json={"draft": draft.model_dump(mode="json")},
    )
    assert response.status_code == 200, response.text
    return response.json()["suite"]


def _save(client, child, parent=None):
    body = {"suite": child, "expected_suite_revision": child["revision"]}
    if parent is not None:
        body["parent_suite_revision"] = parent["revision"]
    return client.post("/api/evals/suites", headers=_AUTH_HEADERS, json=body)


def _source(client, document, case_id):
    identity = EvalSuiteViewIdentity(
        source="authored", target_key=document["target_key"], revision=document["revision"]
    )
    return client.get(
        f"/api/evals/suite-views/{identity.id}/cases/{case_id}/source", headers=_AUTH_HEADERS
    )


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_parent_save_preserves_exact_links_and_replay_but_portable_copy_cannot(
    addition_environment, request, backend
):
    dsn = None
    if backend == "postgres":
        required = os.environ.get("CAYU_REQUIRE_POSTGRES_TESTS", "").lower() in {"1", "true", "yes"}
        if not os.environ.get("CAYU_TEST_POSTGRES_DSN") and not required:
            pytest.skip("PostgreSQL qualification is delegated to the required CI lane.")
        dsn = request.getfixturevalue("parent_save_postgres_dsn")
    server, _, _ = addition_environment(backend, postgres_dsn=dsn)
    with TestClient(server) as client:
        parent, case_id = _linked_parent(client)
        child = _edited(client, parent)
        parent_case = next(case for case in parent["cases"] if case["id"] == case_id)
        assert next(case for case in child["cases"] if case["id"] == case_id) == parent_case
        saved = _save(client, child, parent)
        assert saved.status_code == 201, saved.text
        assert _source(client, child, case_id).json() == _source(client, parent, case_id).json()
        assert _source(client, child, case_id).status_code == 200
        assert _save(client, child, parent).json() == saved.json()
        assert _save(client, child).status_code == 409
        assert _save(client, child, child).json() == saved.json()  # Read-only unchanged save.

        grandchild = _edited(client, child, name="Edited again")
        assert _save(client, grandchild, child).status_code == 201
        assert (
            _source(client, grandchild, case_id).json() == _source(client, parent, case_id).json()
        )
        assert _save(client, grandchild, parent).status_code == 409

        imported = _edited(client, parent, name="Portable copy")
        assert _save(client, imported).status_code == 201
        assert _source(client, imported, case_id).json() == {
            "session_id": None,
            "state": "unavailable",
        }
        assert _save(client, imported, parent).status_code == 409
        changed_draft = EvalSuiteDraftV3.from_document(
            eval_suite_document_from_json(json.dumps(parent))
        )
        cases = tuple(
            case.model_copy(update={"name": "Changed case"}) if case.id == case_id else case
            for case in changed_draft.cases
        )
        changed = _edited(client, parent, name="Changed member", cases=cases)
        assert _save(client, changed, parent).status_code == 201
        assert _source(client, changed, case_id).json() == {
            "session_id": None,
            "state": "unavailable",
        }


@pytest.mark.parametrize("bad_parent", ["missing", "wrong-suite", "wrong-type"])
def test_parent_save_rejects_unproven_parent_without_publishing(addition_environment, bad_parent):
    server, _, _ = addition_environment()
    with TestClient(server) as client:
        parent, _ = _linked_parent(client)
        child = _edited(client, parent)
        if bad_parent == "wrong-suite":
            other = _edited(client, parent, id="different-suite")
            assert _save(client, other).status_code == 201
            revision = other["revision"]
        else:
            revision = "sha256:" + "0" * 64 if bad_parent == "missing" else True
        response = _save(client, child, {"revision": revision})
        assert response.status_code == (422 if bad_parent == "wrong-type" else 409)
        assert (
            client.get(f"/api/evals/suites/{child['revision']}", headers=_AUTH_HEADERS).status_code
            == 404
        )


def test_parent_save_rolls_back_late_link_failure_and_replays_after_restart(
    addition_environment, monkeypatch, tmp_path
):
    from cayu.storage.evals_sqlite import SQLiteEvalStore

    server, _, store = addition_environment()
    with TestClient(server) as client:
        parent, case_id = _linked_parent(client)
        parent_selector = EvalSuiteViewIdentity(
            source="authored", target_key=parent["target_key"], revision=parent["revision"]
        ).id
        body, plan = _review(client, parent_selector, case_id="second-captured")
        added = client.post(
            f"/api/evals/sessions/{_SESSION_ID}/additions",
            headers={**_AUTH_HEADERS, "Idempotency-Key": "second-source"},
            json=body,
        )
        assert added.status_code == 201, added.text
        parent = client.get(
            f"/api/evals/suites/{plan['destination']['revision']}", headers=_AUTH_HEADERS
        ).json()
        child = _edited(client, parent)
        original = store._connection

        class LateLinkFailure:
            inserted = 0

            def __getattr__(self, name):
                return getattr(original, name)

            def execute(self, sql, *args, **kwargs):
                result = original.execute(sql, *args, **kwargs)
                if sql.startswith("INSERT INTO cayu_eval_case_addition_sources"):
                    self.inserted += 1
                    if self.inserted == 2:
                        raise RuntimeError("after child source insertion")
                return result

        monkeypatch.setattr(store, "_connection", LateLinkFailure())
        with pytest.raises(RuntimeError, match="after child source insertion"):
            _save(client, child, parent)
        assert not original.in_transaction
        assert (
            client.get(f"/api/evals/suites/{child['revision']}", headers=_AUTH_HEADERS).status_code
            == 404
        )
        assert _source(client, parent, case_id).status_code == 200
        selector = EvalSuiteViewIdentity(
            source="authored", target_key=child["target_key"], revision=child["revision"]
        ).id
        assert (
            original.execute(
                "SELECT COUNT(*) FROM cayu_eval_case_addition_sources WHERE destination_selector = ?",
                (selector,),
            ).fetchone()[0]
            == 0
        )
        monkeypatch.setattr(store, "_connection", original)
        saved = _save(client, child, parent)
        assert saved.status_code == 201, saved.text

    reopened = SQLiteEvalStore(tmp_path / "additions-0.db")
    restarted, _, _ = addition_environment(store_override=reopened)
    with TestClient(restarted) as client:
        assert _save(client, child, parent).json() == saved.json()
        assert _save(client, child).status_code == 409
        assert _source(client, child, case_id).status_code == 200
        assert _source(client, child, "second-captured").json()["state"] == "available"


def test_parent_save_cancellation_keeps_writer_fenced_until_real_dispatch_settles(
    addition_environment, monkeypatch
):
    import asyncio
    import threading

    import httpx

    from cayu.storage.evals_sqlite import SQLiteEvalStore

    server, _, store = addition_environment()
    with TestClient(server) as client:
        parent, case_id = _linked_parent(client)
        child = _edited(client, parent)
        entered, release = threading.Event(), threading.Event()
        original = SQLiteEvalStore._save_prepared_authored_suite_in_transaction

        def held_write(self, *args, **kwargs):
            entry = original(self, *args, **kwargs)
            entered.set()
            assert release.wait(10)
            return entry

        monkeypatch.setattr(
            SQLiteEvalStore, "_save_prepared_authored_suite_in_transaction", held_write
        )
        body = {
            "suite": child,
            "expected_suite_revision": child["revision"],
            "parent_suite_revision": parent["revision"],
        }

        async def exercise():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as http:
                owner = asyncio.create_task(
                    http.post("/api/evals/suites", headers=_AUTH_HEADERS, json=body)
                )
                competitor = None
                try:
                    assert await asyncio.to_thread(entered.wait, 5)
                    assert owner.cancel()
                    await asyncio.sleep(0)
                    assert owner.cancelling() == 1 and not owner.done()
                    competitor = asyncio.create_task(
                        http.post(
                            "/api/evals/suites",
                            headers=_AUTH_HEADERS,
                            json={**body, "parent_suite_revision": None},
                        )
                    )
                    await asyncio.sleep(0.05)
                    assert store._lock.locked() and not competitor.done()
                    release.set()
                    with pytest.raises(asyncio.CancelledError):
                        await owner
                    assert owner.cancelled() and owner.cancelling() == 1
                    assert (await competitor).status_code == 409
                    assert (
                        await http.post("/api/evals/suites", headers=_AUTH_HEADERS, json=body)
                    ).status_code == 201
                finally:
                    release.set()
                    await asyncio.gather(
                        owner, *([] if competitor is None else [competitor]), return_exceptions=True
                    )

        # The lifespan's eval run coordinator polls this store on the client's
        # loop. Racing requests from a second loop could bind the store lock
        # to that loop and fail the competitor instead of fencing it.
        client.portal.call(exercise)
        assert _source(client, child, case_id).status_code == 200


def test_memory_parent_save_uses_positive_publication_and_restores_partial_mutation(
    addition_environment, monkeypatch
):
    import asyncio

    from cayu.evals.store import EvalAuthoredSuiteConflict, InMemoryEvalStore
    from cayu.storage.evals_sqlite import SQLiteEvalStore

    server, app, durable = addition_environment()
    captured = []
    original = SQLiteEvalStore.publish_session_addition

    async def observe(self, prepared, record, **kwargs):
        captured.append((prepared, record))
        return await original(self, prepared, record, **kwargs)

    monkeypatch.setattr(SQLiteEvalStore, "publish_session_addition", observe)
    with TestClient(server) as client:
        parent_json, case_id = _linked_parent(client)
        child_json = _edited(client, parent_json)

        async def exercise():
            memory = InMemoryEvalStore()
            prepared, record = captured[0]
            initial = await durable.load_authored_suite(prepared.plan.parent.revision)
            await memory.save_authored_suite(initial, redact_json=app.redact_json)
            await memory.publish_session_addition(prepared, record, redact_json=app.redact_json)
            parent = eval_suite_document_from_json(json.dumps(parent_json))
            child = eval_suite_document_from_json(json.dumps(child_json))
            selector = EvalSuiteViewIdentity(
                source="authored", target_key=child.target_key, revision=child.revision
            ).id

            class LateFailure(dict):
                def setdefault(self, key, default=None):
                    super().setdefault(key, default)
                    raise RuntimeError("after memory child association")

            memory._case_addition_sources = LateFailure(memory._case_addition_sources)
            with pytest.raises(RuntimeError, match="after memory child association"):
                await memory.save_authored_suite(
                    child, parent_suite_revision=parent.revision, redact_json=app.redact_json
                )
            assert await memory.load_authored_suite(child.revision) is None
            assert await memory.load_case_addition(selector, case_id) is None
            assert child.revision not in memory._authored_suite_parents
            memory._case_addition_sources = dict(memory._case_addition_sources)
            saved = await memory.save_authored_suite(
                child, parent_suite_revision=parent.revision, redact_json=app.redact_json
            )
            assert await memory.load_case_addition(selector, case_id) == record
            # Exact replay must not republish private associations.
            memory._case_addition_sources = LateFailure(memory._case_addition_sources)
            assert (
                await memory.save_authored_suite(
                    child, parent_suite_revision=parent.revision, redact_json=app.redact_json
                )
                == saved
            )
            with pytest.raises(EvalAuthoredSuiteConflict):
                await memory.save_authored_suite(child, redact_json=app.redact_json)
            # Identical portable parent/case values do not recreate the private
            # association that the real HTTP publication established above.
            imported = InMemoryEvalStore()
            await imported.save_authored_suite(parent, redact_json=app.redact_json)
            await imported.save_authored_suite(
                child, parent_suite_revision=parent.revision, redact_json=app.redact_json
            )
            assert await imported.load_case_addition(selector, case_id) is None

        asyncio.run(exercise())


@pytest.mark.parametrize("cancel_owner", [False, True])
def test_parent_save_failed_rollback_fences_writer_and_restart_recovers_exactly(
    addition_environment, monkeypatch, tmp_path, cancel_owner
):
    import asyncio
    import threading

    import httpx

    from cayu.storage.evals_sqlite import SQLiteEvalStore

    server, _, store = addition_environment()
    entered, release = threading.Event(), threading.Event()
    primary, rollback_error = (
        RuntimeError("parent commit failed"),
        RuntimeError("parent rollback failed"),
    )

    class FaultedConnection:
        closed = False
        armed = False

        def __init__(self, connection):
            self.connection = connection

        def __getattr__(self, name):
            return getattr(self.connection, name)

        def commit(self):
            binding = self.connection.execute(
                "SELECT parent_revision FROM cayu_eval_authored_suites WHERE revision = ?",
                (child["revision"],),
            ).fetchone()
            if binding is None:
                return self.connection.commit()
            assert binding[0] == parent["revision"]
            self.armed = True
            assert (
                self.connection.execute(
                    "SELECT COUNT(*) FROM cayu_eval_case_addition_sources WHERE destination_selector = ?",
                    (selector,),
                ).fetchone()[0]
                == 1
            )
            entered.set()
            assert release.wait(10)
            raise primary

        def rollback(self):
            if not self.armed:
                return self.connection.rollback()
            assert self.connection.in_transaction
            raise rollback_error

        def close(self):
            self.connection.close()
            self.closed = True

    with TestClient(server) as client:
        parent, case_id = _linked_parent(client)
        child = _edited(client, parent)
        selector = EvalSuiteViewIdentity(
            source="authored", target_key=child["target_key"], revision=child["revision"]
        ).id
        faulted = FaultedConnection(store._connection)
        monkeypatch.setattr(store, "_connection", faulted)
        body = {
            "suite": child,
            "expected_suite_revision": child["revision"],
            "parent_suite_revision": parent["revision"],
        }

        async def exercise():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as http:
                owner = asyncio.create_task(
                    http.post("/api/evals/suites", headers=_AUTH_HEADERS, json=body)
                )
                try:
                    assert await asyncio.to_thread(entered.wait, 5)
                    if cancel_owner:
                        assert owner.cancel()
                        await asyncio.sleep(0)
                        assert owner.cancelling() == 1 and not owner.done()
                    assert store._lock.locked()
                    release.set()
                    try:
                        await owner
                    except asyncio.CancelledError as error:
                        assert cancel_owner and owner.cancelled() and owner.cancelling() == 1
                        propagated = error.__cause__
                    except BaseException as error:
                        assert not cancel_owner
                        propagated = error
                    else:
                        pytest.fail("failed publication cannot return success")
                    assert isinstance(propagated, BaseExceptionGroup)
                    assert list(propagated.exceptions) == [primary, rollback_error]
                    assert faulted.closed and not store._lock.locked()
                finally:
                    release.set()
                    await asyncio.gather(owner, return_exceptions=True)

        asyncio.run(exercise())
        failed = TestClient(server, raise_server_exceptions=False)
        assert _save(failed, child, parent).status_code == 500

    reopened = SQLiteEvalStore(tmp_path / "additions-0.db")
    restarted, _, _ = addition_environment(store_override=reopened)
    with TestClient(restarted) as client:
        assert (
            client.get(f"/api/evals/suites/{child['revision']}", headers=_AUTH_HEADERS).status_code
            == 404
        )
        assert _source(client, parent, case_id).json()["state"] == "available"
        assert _save(client, child, parent).status_code == 201
        assert _source(client, child, case_id).json()["state"] == "available"


def test_parent_binding_schema_migrates_existing_documents_and_rejects_missing_column(
    addition_environment, tmp_path
):
    import sqlite3

    from cayu.storage.evals_postgres import _POSTGRES_EVAL_MIN_REQUIRED_REVISION
    from cayu.storage.evals_sqlite import _SQLITE_EVAL_MIN_REQUIRED_REVISION, SQLiteEvalStore
    from cayu.storage.migrations import RevisionKind, SchemaMode, SchemaTooOld, revision

    assert revision(131).kind is RevisionKind.ADDITIVE
    assert _SQLITE_EVAL_MIN_REQUIRED_REVISION == _POSTGRES_EVAL_MIN_REQUIRED_REVISION == 132
    server, _, _ = addition_environment()
    with TestClient(server) as client:
        parent, case_id = _linked_parent(client)
        child = _edited(client, parent)
    path = tmp_path / "additions-0.db"
    with sqlite3.connect(path) as connection:
        connection.execute("ALTER TABLE cayu_eval_authored_suites DROP COLUMN parent_revision")
        connection.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 131")
        connection.execute("PRAGMA user_version = 130")
    with pytest.raises(SchemaTooOld, match="requires >= 132"):
        SQLiteEvalStore(path, schema_mode=SchemaMode.VALIDATE)
    migrated = SQLiteEvalStore(path, schema_mode=SchemaMode.MIGRATE)
    restarted, _, _ = addition_environment(store_override=migrated)
    with TestClient(restarted) as client:
        assert _source(client, parent, case_id).json()["state"] == "available"
        assert _save(client, child, parent).status_code == 201
        assert _source(client, child, case_id).json()["state"] == "available"
    with sqlite3.connect(path) as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE cayu_eval_authored_suites SET parent_revision = ? WHERE revision = ?",
                ("unvalidated", child["revision"]),
            )
        connection.execute("ALTER TABLE cayu_eval_authored_suites DROP COLUMN parent_revision")
    with pytest.raises(RuntimeError, match="authored"):
        SQLiteEvalStore(path, schema_mode=SchemaMode.VALIDATE)


@pytest.mark.parametrize("backend", ["postgres"])
def test_postgres_parent_save_late_failure_is_atomic_and_reopens_exactly(
    addition_environment, request, monkeypatch, backend
):
    required = os.environ.get("CAYU_REQUIRE_POSTGRES_TESTS", "").lower() in {"1", "true", "yes"}
    if not os.environ.get("CAYU_TEST_POSTGRES_DSN") and not required:
        pytest.skip("PostgreSQL qualification is delegated to the required CI lane.")
    dsn = request.getfixturevalue("parent_save_postgres_dsn")
    from psycopg import AsyncCursor

    server, _, _ = addition_environment(backend, postgres_dsn=dsn)
    original = AsyncCursor.execute
    fired = False
    with TestClient(server) as client:
        parent, case_id = _linked_parent(client)
        child = _edited(client, parent)

        async def late_link(cursor, query, *args, **kwargs):
            nonlocal fired
            value = await original(cursor, query, *args, **kwargs)
            if (
                not fired
                and isinstance(query, str)
                and query.startswith("INSERT INTO cayu_eval_case_addition_sources")
            ):
                fired = True
                raise RuntimeError("after postgres child association")
            return value

        monkeypatch.setattr(AsyncCursor, "execute", late_link)
        with pytest.raises(RuntimeError, match="after postgres child association"):
            _save(client, child, parent)
        assert fired
        assert (
            client.get(f"/api/evals/suites/{child['revision']}", headers=_AUTH_HEADERS).status_code
            == 404
        )
        assert _source(client, parent, case_id).json()["state"] == "available"
        saved = _save(client, child, parent)
        assert saved.status_code == 201, saved.text

    restarted, _, _ = addition_environment(backend, postgres_dsn=dsn)
    with TestClient(restarted) as client:
        assert _save(client, child, parent).json() == saved.json()
        assert _save(client, child).status_code == 409
        assert _source(client, child, case_id).json()["state"] == "available"


@pytest.mark.parametrize(
    ("backend", "fail_rollback"), [("sqlite", False), ("postgres", False), ("postgres", True)]
)
def test_addition_reports_parent_binding_collision_without_publishing_receipt(
    addition_environment, request, monkeypatch, backend, fail_rollback
):
    import httpx

    from cayu.evals.store import EvalAuthoredSuiteConflict

    dsn = request.getfixturevalue("parent_save_postgres_dsn") if backend == "postgres" else None
    server, _, store = addition_environment(backend, postgres_dsn=dsn)
    original = type(store).publish_session_addition
    observed, failed_connections, returned = [], [], []
    rollback_error = RuntimeError("addition conflict rollback failed")
    if fail_rollback:
        from psycopg import AsyncConnection, AsyncCursor

        original_execute, original_rollback = AsyncCursor.execute, AsyncConnection.rollback
        original_return = store._pool.putconn

        async def track_receipt(cursor, query, params=None, **kwargs):
            result = await original_execute(cursor, query, params, **kwargs)
            if (
                observed
                and isinstance(query, str)
                and query.startswith("INSERT INTO cayu_eval_session_additions")
                and params[0] == observed[-1].receipt.operation_id
            ):
                failed_connections.append(cursor.connection)
            return result

        async def fail_owned_rollback(connection):
            if any(connection is failed for failed in failed_connections):
                raise rollback_error
            return await original_rollback(connection)

        async def track_return(connection):
            if any(connection is failed for failed in failed_connections):
                returned.append(connection)
            await original_return(connection)

        monkeypatch.setattr(AsyncCursor, "execute", track_receipt)
        monkeypatch.setattr(AsyncConnection, "rollback", fail_owned_rollback)
        monkeypatch.setattr(store._pool, "putconn", track_return)

    async def interleave_authored_save(self, prepared, record, **kwargs):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server), base_url="http://test"
        ) as http:
            saved = await http.post(
                "/api/evals/suites",
                headers=_AUTH_HEADERS,
                json={
                    "suite": prepared.authored.model_dump(mode="json"),
                    "expected_suite_revision": prepared.authored.revision,
                    "parent_suite_revision": prepared.plan.parent.revision,
                },
            )
            assert saved.status_code == 201, saved.text
        observed.append(record)
        return await original(self, prepared, record, **kwargs)

    with TestClient(server) as client:
        selector, _ = _destination(client, "authored")
        body, plan = _review(client, selector)
        monkeypatch.setattr(type(store), "publish_session_addition", interleave_authored_save)

        def publish():
            return client.post(
                f"/api/evals/sessions/{_SESSION_ID}/additions",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "colliding-parent"},
                json=body,
            )

        if fail_rollback:
            with pytest.raises(BaseExceptionGroup) as captured:
                publish()
            assert len(captured.value.exceptions) == 2
            assert isinstance(captured.value.exceptions[0], EvalAuthoredSuiteConflict)
            assert captured.value.exceptions[1] is rollback_error
            assert len(failed_connections) == 1 and failed_connections[0].closed
            assert sum(connection is failed_connections[0] for connection in returned) == 1
        else:
            response = publish()
            assert response.status_code == 409, response.text
        assert len(observed) == 1
        receipt = client.get(
            f"/api/evals/additions/{observed[0].receipt.operation_id}", headers=_AUTH_HEADERS
        )
        assert receipt.status_code == 404
        document = client.get(
            f"/api/evals/suites/{plan['destination']['revision']}", headers=_AUTH_HEADERS
        ).json()
        assert _source(client, document, plan["case_id"]).json()["state"] == "unavailable"


@pytest.mark.parametrize("backend", ["postgres"])
@pytest.mark.parametrize("cancel_owner", [False, True])
@pytest.mark.parametrize("health_check", [False, True])
def test_postgres_parent_rollback_failure_preserves_outcome_and_discards_connection(
    addition_environment, request, monkeypatch, backend, cancel_owner, health_check
):
    import asyncio
    from contextlib import ExitStack

    import httpx
    from psycopg import AsyncConnection

    from cayu.storage.evals_postgres import PostgresEvalStore

    dsn = request.getfixturevalue("parent_save_postgres_dsn")
    external_pool = external_store = None
    if health_check:
        from psycopg_pool import AsyncConnectionPool

        from cayu.storage.migrations import SchemaMode

        external_pool = AsyncConnectionPool(
            dsn, min_size=1, max_size=2, open=False, check=AsyncConnectionPool.check_connection
        )
        external_store = PostgresEvalStore(pool=external_pool, schema_mode=SchemaMode.MIGRATE)
    server, _, store = addition_environment(
        backend, postgres_dsn=dsn, store_override=external_store
    )
    primary = RuntimeError("parent publication commit failed")
    rollback_error = RuntimeError("parent publication rollback failed")
    original_write = PostgresEvalStore._save_prepared_authored_suite_in_transaction
    original_commit, original_rollback = AsyncConnection.commit, AsyncConnection.rollback
    original_return = store._pool.putconn
    failed_connections, returned = [], []

    with TestClient(server) as client, ExitStack() as cleanup:
        if external_pool is not None:
            cleanup.callback(client.portal.call, external_pool.close)
        parent, case_id = _linked_parent(client)
        child = _edited(client, parent)
        body = {
            "suite": child,
            "expected_suite_revision": child["revision"],
            "parent_suite_revision": parent["revision"],
        }

        async def exercise():
            committed, rollback_started, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
            armed = None

            async def track_write(self, cur, validated, *args, **kwargs):
                nonlocal armed
                value = await original_write(self, cur, validated, *args, **kwargs)
                if validated.revision == child["revision"] and not failed_connections:
                    armed = cur.connection
                return value

            async def fail_commit(connection):
                if connection is not armed or failed_connections:
                    return await original_commit(connection)
                failed_connections.append(connection)
                committed.set()
                if cancel_owner:
                    await asyncio.Event().wait()
                raise primary

            async def fail_rollback(connection):
                if connection is not armed:
                    return await original_rollback(connection)
                rollback_started.set()
                await release.wait()
                raise rollback_error

            async def track_return(connection):
                if any(connection is failed for failed in failed_connections):
                    returned.append(connection)
                await original_return(connection)

            with monkeypatch.context() as faults:
                faults.setattr(
                    PostgresEvalStore, "_save_prepared_authored_suite_in_transaction", track_write
                )
                faults.setattr(AsyncConnection, "commit", fail_commit)
                faults.setattr(AsyncConnection, "rollback", fail_rollback)
                faults.setattr(store._pool, "putconn", track_return)
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=server), base_url="http://test"
                ) as http:
                    owner = asyncio.create_task(
                        http.post("/api/evals/suites", headers=_AUTH_HEADERS, json=body)
                    )
                    competitor = None
                    try:
                        await asyncio.wait_for(committed.wait(), 5)
                        if cancel_owner:
                            assert owner.cancel()
                            assert owner.cancelling() == 1
                        await asyncio.wait_for(rollback_started.wait(), 5)
                        assert not owner.done() and not failed_connections[0].closed
                        competitor = asyncio.create_task(
                            http.post("/api/evals/suites", headers=_AUTH_HEADERS, json=body)
                        )
                        await asyncio.sleep(0.05)
                        assert not competitor.done()  # Unique child identity remains locked.
                        release.set()
                        try:
                            await owner
                        except asyncio.CancelledError as error:
                            assert cancel_owner and owner.cancelled() and owner.cancelling() == 1
                            failure = error.__cause__
                            assert isinstance(failure, BaseExceptionGroup)
                            assert failure.exceptions == (rollback_error,)
                        except BaseExceptionGroup as failure:
                            assert not cancel_owner and not owner.cancelled()
                            assert failure.exceptions == (primary, rollback_error)
                        else:
                            pytest.fail("failed publication cannot return success")
                        assert failed_connections[0].closed
                        assert (
                            sum(connection is failed_connections[0] for connection in returned) == 1
                        )
                        retry = await asyncio.wait_for(competitor, 5)
                        assert retry.status_code == 201, retry.text
                    finally:
                        release.set()
                        await asyncio.gather(
                            owner,
                            *([] if competitor is None else [competitor]),
                            return_exceptions=True,
                        )

        # PostgreSQL pools belong to the server's event loop.
        client.portal.call(exercise)
        assert _source(client, child, case_id).json()["state"] == "available"
        assert _save(client, child, parent).status_code == 201
        assert _save(client, child).status_code == 409
    restarted, _, _ = addition_environment(backend, postgres_dsn=dsn)
    with TestClient(restarted) as client:
        assert _save(client, child, parent).status_code == 201
        assert _source(client, child, case_id).json()["state"] == "available"


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_readding_unchanged_case_to_parent_bound_suite_preserves_noop_receipt(
    addition_environment, request, monkeypatch, backend
):
    import asyncio

    from cayu.evals.store import InMemoryEvalStore

    dsn = request.getfixturevalue("parent_save_postgres_dsn") if backend == "postgres" else None
    server, _, store = addition_environment(backend, postgres_dsn=dsn)
    publications = []
    original = type(store).publish_session_addition

    async def observe(self, prepared, record, **kwargs):
        publications.append((prepared, record))
        return await original(self, prepared, record, **kwargs)

    monkeypatch.setattr(type(store), "publish_session_addition", observe)
    with TestClient(server) as client:
        parent, case_id = _linked_parent(client)
        child = _edited(client, parent)
        assert _save(client, child, parent).status_code == 201
        selector = EvalSuiteViewIdentity(
            source="authored", target_key=child["target_key"], revision=child["revision"]
        ).id
        body, plan = _review(client, selector, case_id=case_id)
        assert plan["membership_changed"] is False
        assert plan["parent"] == plan["destination"]
        source = _source(client, child, case_id).json()
        response = client.post(
            f"/api/evals/sessions/{_SESSION_ID}/additions",
            headers={**_AUTH_HEADERS, "Idempotency-Key": "unchanged-parent-case"},
            json=body,
        )
        assert response.status_code == 201, response.text
        assert (
            client.post(
                f"/api/evals/sessions/{_SESSION_ID}/additions",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "unchanged-parent-case"},
                json=body,
            ).json()
            == response.json()
        )
        assert _source(client, child, case_id).json() == source
        assert _save(client, child).status_code == 409  # No rebinding.

        if backend == "sqlite":
            # Production memory stores have no HTTP project context. Use the
            # exact trusted handoffs produced by the real HTTP path instead.
            seed = client.get(
                f"/api/evals/suites/{publications[0][0].plan.parent.revision}",
                headers=_AUTH_HEADERS,
            ).json()

            async def memory_noop():
                memory = InMemoryEvalStore()
                await memory.save_authored_suite(
                    eval_suite_document_from_json(json.dumps(seed)), redact_json=lambda value: value
                )
                await memory.publish_session_addition(
                    *publications[0], redact_json=lambda value: value
                )
                await memory.save_authored_suite(
                    eval_suite_document_from_json(json.dumps(child)),
                    parent_suite_revision=parent["revision"],
                    redact_json=lambda value: value,
                )
                result = await memory.publish_session_addition(
                    *publications[1], redact_json=lambda value: value
                )
                assert result.receipt.model_dump(mode="json") == response.json()
                assert (
                    await memory.publish_session_addition(
                        *publications[1], redact_json=lambda value: value
                    )
                    == result
                )
                assert await memory.load_case_addition(selector, case_id) is not None
                assert memory._authored_suite_parents[child["revision"]] == parent["revision"]

            asyncio.run(memory_noop())
