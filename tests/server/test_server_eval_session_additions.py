"""Session additions traverse real HTTP publication and immutable destination reads."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from tests.server.test_server_captured_evaluations import (
    captured_route_server as captured_route_server,
)
from tests.server.test_server_eval_suite_authoring import _draft
from tests.server.test_server_evaluation_promotion import _AUTH_HEADERS, _SESSION_ID

from cayu.evals.read_views import EvalSuiteViewIdentity


def _destination(client: TestClient, source: str) -> tuple[str, dict]:
    response = client.post(
        f"/api/evals/sessions/{_SESSION_ID}/evaluation/preview",
        headers=_AUTH_HEADERS,
        json={},
    )
    assert response.status_code == 200, response.text
    candidate = response.json()["candidate"]
    if source == "authored":
        draft = _draft().model_copy(update={"target_key": candidate["target_key"]})
        preview = client.post(
            "/api/evals/suites/preview",
            headers=_AUTH_HEADERS,
            json={"draft": draft.model_dump(mode="json")},
        )
        assert preview.status_code == 200, preview.text
        document = preview.json()["suite"]
        saved = client.post(
            "/api/evals/suites",
            headers=_AUTH_HEADERS,
            json={"suite": document, "expected_suite_revision": document["revision"]},
        )
        assert saved.status_code == 201, saved.text
        identity = EvalSuiteViewIdentity(
            source="authored", target_key=candidate["target_key"], revision=document["revision"]
        )
    else:
        saved = client.post(
            f"/api/evals/sessions/{_SESSION_ID}/evaluation/save",
            headers=_AUTH_HEADERS,
            json={"candidate": candidate, "expected_candidate_revision": candidate["revision"]},
        )
        assert saved.status_code == 201, saved.text
        identity = EvalSuiteViewIdentity(
            source="corpus",
            target_key=candidate["target_key"],
            revision=saved.json()["record"]["corpus_revision"],
            suite_id=candidate["suite"]["id"],
        )
    parent = client.get(f"/api/evals/suite-views/{identity.id}", headers=_AUTH_HEADERS)
    assert parent.status_code == 200, parent.text
    return identity.id, parent.json()


@pytest.mark.parametrize("source", ["authored", "corpus"])
@pytest.mark.parametrize("mode", ["case", "score"])
def test_existing_suite_addition_preview_preserves_exact_parent(
    captured_route_server, source: str, mode: str
) -> None:
    with TestClient(captured_route_server()) as client:
        selector, parent = _destination(client, source)
        response = client.post(
            f"/api/evals/sessions/{_SESSION_ID}/additions/preview",
            headers=_AUTH_HEADERS,
            json={
                "mode": mode,
                "destination": {"kind": "existing", "selector": selector},
                "case_id": "captured-addition",
                "case_name": "Captured addition",
            },
        )
        assert response.status_code == 200, response.text
        # A preview is not membership or history publication.
        unchanged = client.get(f"/api/evals/suite-views/{selector}", headers=_AUTH_HEADERS)
        assert unchanged.status_code == 200, unchanged.text
        assert unchanged.json() == parent


@pytest.mark.parametrize("source", ["authored", "corpus"])
@pytest.mark.parametrize("mode", ["case", "score"])
def test_additions_publish_one_exact_receipt_and_preserve_execution_scope(
    captured_route_server,
    tmp_path,
    source: str,
    mode: str,
) -> None:
    import asyncio

    from cayu.storage.evals_sqlite import SQLiteEvalStore

    with TestClient(captured_route_server()) as client:
        selector, parent = _destination(client, source)
        request = {
            "mode": mode,
            "destination": {"kind": "existing", "selector": selector},
            "case_id": "captured-addition",
            "case_name": "Captured addition",
        }
        preview = client.post(
            f"/api/evals/sessions/{_SESSION_ID}/additions/preview",
            headers=_AUTH_HEADERS,
            json=request,
        )
        assert preview.status_code == 200, preview.text
        plan = preview.json()
        body = {
            **request,
            "expected_plan_revision": plan["revision"],
            "expected_source_revision": plan["source_revision"],
        }
        headers = {**_AUTH_HEADERS, "Idempotency-Key": "add-one"}
        url = f"/api/evals/sessions/{_SESSION_ID}/additions"
        committed = client.post(url, headers=headers, json=body)
        assert committed.status_code == 201, committed.text
        receipt = committed.json()
        assert receipt["plan"] == plan
        assert client.post(url, headers=headers, json=body).json() == receipt
        conflicting = client.post(url, headers=headers, json={**body, "case_name": "Changed"})
        assert conflicting.status_code == 409
        recovered = client.get(
            f"/api/evals/additions/{receipt['operation_id']}", headers=_AUTH_HEADERS
        )
        assert recovered.status_code == 200, recovered.text
        assert recovered.json() == receipt
        unchanged = client.get(f"/api/evals/suite-views/{selector}", headers=_AUTH_HEADERS)
        assert unchanged.status_code == 200
        assert unchanged.json()["cases"] == parent["cases"]
        resulting = EvalSuiteViewIdentity.model_validate(plan["destination"])
        detail = client.get(f"/api/evals/suite-views/{resulting.id}", headers=_AUTH_HEADERS)
        assert detail.status_code == 200, detail.text
        if mode == "case":
            assert resulting.id != selector
            assert detail.json()["cases"]["total"] == parent["cases"]["total"] + 1
            assert plan["membership_changed"] is True
            assert plan["execution_case_count"] is None
        else:
            assert resulting.id == selector
            assert plan["membership_changed"] is False
            assert plan["execution_case_count"] == 1
            captured = client.get(
                f"/api/evals/results/{plan['captured_result_revision']}", headers=_AUTH_HEADERS
            )
            assert captured.status_code == 200, captured.text
            assert captured.json()["result"]["score"]["case_id"] == "captured-addition"

        async def reconstruct():
            reopened = SQLiteEvalStore(tmp_path / "routes-0.db")
            try:
                stored = await reopened.load_session_addition(receipt["operation_id"])
                assert stored is not None
                assert stored.receipt.model_dump(mode="json") == receipt
                assert stored.private_session_id == _SESSION_ID
                assert stored.request_material["request"] == {**body, "assertions": None}
            finally:
                await reopened.close()

        asyncio.run(reconstruct())


def _review(client, selector, mode="case", **changes):
    request = {
        "mode": mode,
        "destination": {"kind": "existing", "selector": selector},
        "case_id": "captured-addition",
        "case_name": "Captured addition",
        **changes,
    }
    preview = client.post(
        f"/api/evals/sessions/{_SESSION_ID}/additions/preview",
        headers=_AUTH_HEADERS,
        json=request,
    )
    assert preview.status_code == 200, preview.text
    plan = preview.json()
    return {
        **request,
        "expected_plan_revision": plan["revision"],
        "expected_source_revision": plan["source_revision"],
    }, plan


@pytest.mark.parametrize("source", ["authored", "corpus"])
def test_scored_attachment_is_explicit_deduplicated_and_has_a_server_owned_source(
    captured_route_server,
    source,
):
    from cayu.evals.read_views import run_view_id

    with TestClient(captured_route_server()) as client:
        selector, parent = _destination(client, source)
        body, plan = _review(client, selector, mode="score")
        url = f"/api/evals/sessions/{_SESSION_ID}/additions"
        for key in ("first", "second"):
            saved = client.post(url, headers={**_AUTH_HEADERS, "Idempotency-Key": key}, json=body)
            assert saved.status_code == 201, saved.text
        detail = client.get(f"/api/evals/suite-views/{selector}", headers=_AUTH_HEADERS)
        assert detail.status_code == 200, detail.text
        attached = [
            item
            for item in detail.json()["runs"]["items"]
            if item["result_revision"] == plan["captured_result_revision"]
        ]
        assert len(attached) == 1
        run = attached[0]
        assert run["suite"] != plan["destination"]
        assert run["placement"]["destination"] == plan["destination"]
        assert run["placement"]["execution_case_count"] == 1
        assert run["baseline"] is None
        assert detail.json()["cases"] == parent["cases"]
        result_selector = run_view_id("result", plan["captured_result_revision"])
        placed = client.get(
            f"/api/evals/run-views/{result_selector}",
            headers=_AUTH_HEADERS,
            params={"placement": selector},
        )
        assert placed.status_code == 200, placed.text
        assert placed.json()["run"]["placement"] == run["placement"]
        actual = EvalSuiteViewIdentity.model_validate(run["suite"])
        catalog = client.get("/api/evals/suite-views", headers=_AUTH_HEADERS)
        assert any(item["id"] == actual.id for item in catalog.json()["items"])
        linked = client.get(
            f"/api/evals/suite-views/{selector}/cases/captured-addition/source",
            headers=_AUTH_HEADERS,
            params={"result_revision": plan["captured_result_revision"]},
        )
        assert linked.status_code == 200, linked.text
        assert linked.json() == {"state": "available", "session_id": _SESSION_ID}
        absent = client.get(
            f"/api/evals/suite-views/{selector}/cases/forged/source",
            headers=_AUTH_HEADERS,
            params={"result_revision": plan["captured_result_revision"]},
        )
        assert absent.json() == {"state": "unavailable", "session_id": None}


@pytest.mark.parametrize(
    "changed",
    [
        "session_id",
        "case_id",
        "case_name",
        "mode",
        "expected_plan_revision",
        "expected_source_revision",
        "destination",
        "assertions",
    ],
)
def test_exact_operation_replay_rejects_each_changed_request_field(captured_route_server, changed):
    with TestClient(captured_route_server()) as client:
        selector, _ = _destination(client, "authored")
        body, _ = _review(client, selector)
        headers = {**_AUTH_HEADERS, "Idempotency-Key": "same-key"}
        url = f"/api/evals/sessions/{_SESSION_ID}/additions"
        saved = client.post(url, headers=headers, json=body)
        assert saved.status_code == 201, saved.text
        values = {
            "case_id": "other-case",
            "case_name": "Other name",
            "mode": "score",
            "expected_plan_revision": "sha256:" + "f" * 64,
            "expected_source_revision": "sha256:" + "e" * 64,
            "destination": {"kind": "new", "suite_id": "other", "name": "Other"},
            "assertions": [{"kind": "root_status", "id": "finished", "expected": "completed"}],
        }
        rejected = (
            client.post("/api/evals/sessions/another-source/additions", headers=headers, json=body)
            if changed == "session_id"
            else client.post(url, headers=headers, json={**body, changed: values[changed]})
        )
        assert rejected.status_code == 409, rejected.text
        assert client.post(url, headers=headers, json=body).json() == saved.json()


@pytest.mark.parametrize("mode", ["case", "score"])
def test_transaction_failure_after_document_write_publishes_nothing_and_retry_succeeds(
    captured_route_server,
    monkeypatch,
    mode,
):
    from cayu.storage.evals_sqlite import SQLiteEvalStore

    with TestClient(captured_route_server(), raise_server_exceptions=False) as client:
        selector, parent = _destination(client, "authored")
        body, plan = _review(client, selector, mode=mode)
        catalog = client.get("/api/evals/suite-views", headers=_AUTH_HEADERS).json()
        method = (
            "_save_prepared_authored_suite_in_transaction"
            if mode == "case"
            else "_save_result_in_transaction"
        )
        original = getattr(SQLiteEvalStore, method)
        fired = False

        def fail_after_write(self, *args, **kwargs):
            nonlocal fired
            value = original(self, *args, **kwargs) if mode == "case" else original(*args, **kwargs)
            if not fired:
                fired = True
                raise RuntimeError("injected failure after real durable statement")
            return value

        monkeypatch.setattr(SQLiteEvalStore, method, fail_after_write)
        url = f"/api/evals/sessions/{_SESSION_ID}/additions"
        headers = {**_AUTH_HEADERS, "Idempotency-Key": "rollback-then-retry"}
        failed = client.post(url, headers=headers, json=body)
        assert failed.status_code == 500
        assert fired, "failure injection must follow the actual write"
        assert client.get("/api/evals/suite-views", headers=_AUTH_HEADERS).json() == catalog
        assert (
            client.get(f"/api/evals/suite-views/{selector}", headers=_AUTH_HEADERS).json() == parent
        )
        retried = client.post(url, headers=headers, json=body)
        assert retried.status_code == 201, retried.text
        assert retried.json()["plan"] == plan
        assert client.post(url, headers=headers, json=body).json() == retried.json()


def test_committed_response_loss_recovers_receipt_without_recapturing_changed_source(
    captured_route_server,
    monkeypatch,
):
    from cayu.server._captured_evaluation_source import CapturedEvaluationSource
    from cayu.storage.evals_sqlite import SQLiteEvalStore

    with TestClient(captured_route_server(), raise_server_exceptions=False) as client:
        selector, _ = _destination(client, "authored")
        body, _ = _review(client, selector, mode="score")
        original = SQLiteEvalStore.publish_session_addition
        committed = []

        async def lose_response(self, *args, **kwargs):
            record = await original(self, *args, **kwargs)
            committed.append(record)
            raise RuntimeError("committed response was lost")

        monkeypatch.setattr(SQLiteEvalStore, "publish_session_addition", lose_response)
        url = f"/api/evals/sessions/{_SESSION_ID}/additions"
        headers = {**_AUTH_HEADERS, "Idempotency-Key": "lost-response"}
        assert client.post(url, headers=headers, json=body).status_code == 500
        assert len(committed) == 1

        async def source_changed(*args, **kwargs):
            raise AssertionError("replay must not recapture mutable source")

        monkeypatch.setattr(CapturedEvaluationSource, "load", source_changed)
        replayed = client.post(url, headers=headers, json=body)
        assert replayed.status_code == 201, replayed.text
        assert replayed.json() == committed[0].receipt.model_dump(mode="json")
        assert len(committed) == 1


@pytest.mark.parametrize("mode", ["case", "score"])
def test_new_suite_addition_keeps_mode_specific_membership(captured_route_server, mode):
    with TestClient(captured_route_server()) as client:
        request = {
            "mode": mode,
            "destination": {"kind": "new", "suite_id": "new-suite", "name": "New suite"},
            "case_id": "captured-case",
            "case_name": "Captured case",
        }
        preview = client.post(
            f"/api/evals/sessions/{_SESSION_ID}/additions/preview",
            headers=_AUTH_HEADERS,
            json=request,
        )
        assert preview.status_code == 200, preview.text
        plan = preview.json()
        saved = client.post(
            f"/api/evals/sessions/{_SESSION_ID}/additions",
            headers={**_AUTH_HEADERS, "Idempotency-Key": "new-suite"},
            json={
                **request,
                "expected_plan_revision": plan["revision"],
                "expected_source_revision": plan["source_revision"],
            },
        )
        assert saved.status_code == 201, saved.text
        identity = EvalSuiteViewIdentity.model_validate(plan["destination"])
        assert identity.source == ("authored" if mode == "case" else "corpus")
        detail = client.get(f"/api/evals/suite-views/{identity.id}", headers=_AUTH_HEADERS)
        assert detail.status_code == 200, detail.text
        assert detail.json()["cases"]["total"] == 1


@pytest.fixture
def addition_environment(tmp_path):
    import asyncio
    from pathlib import Path

    from tests.server.test_server_evaluation_promotion import _authenticate, _seed_app

    from cayu.evals.store import InMemoryEvalStore
    from cayu.project_control_plane import (
        ProjectControlPlaneAccess,
        _create_project_control_plane_context,
    )
    from cayu.server import DashboardConfig, ServerConfig, create_server
    from cayu.storage.evals_sqlite import SQLiteEvalStore

    contexts = []

    def build(
        backend="sqlite",
        auth=_authenticate,
        postgres_dsn=None,
        store_override=None,
        app_override=None,
    ):
        app = app_override or asyncio.run(_seed_app())
        store = store_override or (
            InMemoryEvalStore()
            if backend == "memory"
            else SQLiteEvalStore(tmp_path / f"additions-{len(contexts)}.db")
        )
        if backend == "postgres" and store_override is None:
            from cayu.storage.evals_postgres import PostgresEvalStore
            from cayu.storage.migrations import SchemaMode

            asyncio.run(store.close())
            store = PostgresEvalStore(postgres_dsn, schema_mode=SchemaMode.MIGRATE)
        context = _create_project_control_plane_context(
            project_root=Path(__file__).resolve().parents[2],
            project_id="additions-contract",
            eval_store=store,
            store_backend=backend,
            store_source="project",
            access=ProjectControlPlaneAccess.AUTHENTICATED_PRODUCTION,
        )
        contexts.append(context)
        server = create_server(
            app,
            config=ServerConfig.protected(auth, dashboard=DashboardConfig(enabled=False)),
            project_context=context,
        )
        return server, app, store

    yield build
    for context in contexts:
        asyncio.run(context.close())


@pytest.mark.parametrize("backend", ["sqlite"])
def test_source_deletion_removes_link_but_exact_receipt_survives(addition_environment, backend):

    server, app, _ = addition_environment(backend)
    with TestClient(server) as client:
        selector, _ = _destination(client, "authored")
        body, plan = _review(client, selector)
        headers = {**_AUTH_HEADERS, "Idempotency-Key": "survive-source-deletion"}
        url = f"/api/evals/sessions/{_SESSION_ID}/additions"
        saved = client.post(url, headers=headers, json=body)
        assert saved.status_code == 201, saved.text
        destination = EvalSuiteViewIdentity.model_validate(plan["destination"])
        source_url = f"/api/evals/suite-views/{destination.id}/cases/{plan['case_id']}/source"
        assert client.get(source_url, headers=_AUTH_HEADERS).json()["state"] == "available"
        assert client.portal is not None
        client.portal.call(app.session_store.delete_session, _SESSION_ID)
        assert client.get(source_url, headers=_AUTH_HEADERS).json() == {
            "state": "unavailable",
            "session_id": None,
        }
        assert client.post(url, headers=headers, json=body).json() == saved.json()


def test_real_request_cancellation_keeps_dispatched_sqlite_publication_fenced(
    addition_environment,
    monkeypatch,
):
    import asyncio
    import threading

    import httpx

    from cayu.storage.evals_sqlite import SQLiteEvalStore

    server, _, store = addition_environment()
    with TestClient(server) as client:
        selector, _ = _destination(client, "authored")
        body, plan = _review(client, selector)
        entered, release = threading.Event(), threading.Event()
        original = SQLiteEvalStore._save_prepared_authored_suite_in_transaction

        def held_write(self, *args, **kwargs):
            value = original(self, *args, **kwargs)
            entered.set()
            assert release.wait(10), "test must release dispatched worker"
            return value

        monkeypatch.setattr(
            SQLiteEvalStore, "_save_prepared_authored_suite_in_transaction", held_write
        )

        async def exercise():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as async_client:
                url = f"/api/evals/sessions/{_SESSION_ID}/additions"
                headers = {**_AUTH_HEADERS, "Idempotency-Key": "cancel-in-flight"}
                owner = asyncio.create_task(async_client.post(url, headers=headers, json=body))
                competitor = None
                try:
                    assert await asyncio.to_thread(entered.wait, 5)
                    assert owner.cancel()
                    await asyncio.sleep(0)
                    assert owner.cancelling() == 1 and not owner.done()
                    competitor = asyncio.create_task(
                        async_client.post(
                            url, headers=headers, json={**body, "case_name": "Competing mutation"}
                        )
                    )
                    await asyncio.sleep(0.05)
                    assert not competitor.done()
                    assert store._lock.locked()
                    release.set()
                    with pytest.raises(asyncio.CancelledError):
                        await owner
                    assert owner.cancelled() and owner.cancelling() == 1
                    rejected = await competitor
                    assert rejected.status_code == 409, rejected.text
                    replayed = await async_client.post(url, headers=headers, json=body)
                    assert replayed.status_code == 201, replayed.text
                    assert replayed.json()["plan"] == plan
                    # The eval run coordinator may hold the lock on its own schedule; a lock
                    # the cancelled publication leaked would never come free.
                    await asyncio.wait_for(store._lock.acquire(), timeout=10)
                    store._lock.release()
                finally:
                    release.set()
                    await asyncio.gather(
                        owner, *([] if competitor is None else [competitor]), return_exceptions=True
                    )

        # Run on the server's event loop: its eval run coordinator shares this store's lock.
        client.portal.call(exercise)


def test_member_and_unknown_roles_cannot_publish_or_resolve_private_sources(addition_environment):
    from tests.server.test_server_evaluation_promotion import _authenticate

    from cayu.server import AuthContext, ControlPlaneRoles

    class RoleAuth:
        control_plane_roles = ControlPlaneRoles()
        role = "admin"
        subject = "operator"
        tenant = "tenant-a"
        calls = 0

        def __call__(self, request):
            self.calls += 1
            _authenticate(request)
            context = AuthContext(
                subject=self.subject, tenant=self.tenant, claims={"role": self.role}
            )
            self.control_plane_roles.require(request, context)
            return context

    auth = RoleAuth()
    server, _, _ = addition_environment(auth=auth)
    with TestClient(server) as client:
        selector, _ = _destination(client, "authored")
        body, plan = _review(client, selector)
        url = f"/api/evals/sessions/{_SESSION_ID}/additions"
        headers = {**_AUTH_HEADERS, "Idempotency-Key": "role-bound"}
        auth.calls = 0
        saved = client.post(url, headers=headers, json=body)
        assert saved.status_code == 201, saved.text
        assert auth.calls == 1
        operation_url = f"/api/evals/additions/{saved.json()['operation_id']}"
        destination = EvalSuiteViewIdentity.model_validate(plan["destination"])
        source_url = f"/api/evals/suite-views/{destination.id}/cases/{plan['case_id']}/source"
        for role in ("member", "unknown-future-role"):
            auth.role = role
            for path in (url, url + "/preview"):
                rejected = client.post(path, headers=headers, json={"malformed": True})
                assert rejected.status_code == 403, rejected.text
            for path in (operation_url, source_url):
                rejected = client.get(path, headers=_AUTH_HEADERS)
                assert rejected.status_code == 403, rejected.text
        auth.role = "admin"
        auth.tenant = "tenant-b"
        assert client.get(operation_url, headers=_AUTH_HEADERS).status_code == 404
        auth.tenant = "tenant-a"
        auth.subject = "another-operator"
        assert client.get(operation_url, headers=_AUTH_HEADERS).status_code == 404
        assert client.get(operation_url).status_code == 401


def test_identical_portable_case_provenance_never_creates_an_operational_source_link(
    captured_route_server,
):
    from cayu.evals.suite_authoring import EvalSuiteDocumentV3, EvalSuiteDraftV3

    with TestClient(captured_route_server()) as client:
        selector, _ = _destination(client, "authored")
        body, plan = _review(client, selector)
        saved = client.post(
            f"/api/evals/sessions/{_SESSION_ID}/additions",
            headers={**_AUTH_HEADERS, "Idempotency-Key": "trusted-source"},
            json=body,
        )
        assert saved.status_code == 201, saved.text
        destination = EvalSuiteViewIdentity.model_validate(plan["destination"])
        document_response = client.get(
            f"/api/evals/suites/{destination.revision}", headers=_AUTH_HEADERS
        )
        assert document_response.status_code == 200, document_response.text
        # Parent is V1 here; create a current V3 import preserving the diagnostic
        # source of every case while changing only the destination identity.
        material = document_response.json()
        from cayu.evals.suite_authoring import EvalCaseDefinitionV1, EvalCaseDraftV2

        draft = EvalSuiteDraftV3(
            id="imported-copy",
            name="Imported copy",
            target_key=destination.target_key,
            cases=tuple(
                EvalCaseDraftV2.from_case(EvalCaseDefinitionV1.model_validate(case))
                for case in material["cases"]
            ),
        )
        preview = client.post(
            "/api/evals/suites/preview",
            headers=_AUTH_HEADERS,
            json={"draft": draft.model_dump(mode="json")},
        )
        assert preview.status_code == 200, preview.text
        imported = preview.json()["suite"]
        imported_document = EvalSuiteDocumentV3.model_validate(imported)
        trusted_source = next(
            case["source"] for case in material["cases"] if case["id"] == plan["case_id"]
        )
        copied_source = next(
            case.source for case in imported_document.cases if case.id == plan["case_id"]
        )
        assert copied_source is not None
        assert copied_source.model_dump(mode="json") == trusted_source
        published = client.post(
            "/api/evals/suites",
            headers=_AUTH_HEADERS,
            json={"suite": imported, "expected_suite_revision": imported["revision"]},
        )
        assert published.status_code == 201, published.text
        copied = EvalSuiteViewIdentity(
            source="authored", target_key=destination.target_key, revision=imported["revision"]
        )
        linked = client.get(
            f"/api/evals/suite-views/{destination.id}/cases/{plan['case_id']}/source",
            headers=_AUTH_HEADERS,
        )
        unlinked = client.get(
            f"/api/evals/suite-views/{copied.id}/cases/{plan['case_id']}/source",
            headers=_AUTH_HEADERS,
        )
        assert linked.json()["state"] == "available"
        assert unlinked.json() == {"state": "unavailable", "session_id": None}


def test_memory_publication_rolls_back_late_failure_and_replays_exactly(
    addition_environment, monkeypatch
):
    from cayu.evals.read_views import EvalViewQuery
    from cayu.evals.store import InMemoryEvalStore
    from cayu.storage.evals_sqlite import SQLiteEvalStore

    server, app, durable = addition_environment()
    captured = []
    original = SQLiteEvalStore.publish_session_addition

    async def observe(self, prepared, record, **kwargs):
        captured.append((prepared, record))
        return await original(self, prepared, record, **kwargs)

    monkeypatch.setattr(SQLiteEvalStore, "publish_session_addition", observe)
    with TestClient(server) as client:
        selector, _ = _destination(client, "authored")
        body, _ = _review(client, selector, mode="score")
        saved = client.post(
            f"/api/evals/sessions/{_SESSION_ID}/additions",
            headers={**_AUTH_HEADERS, "Idempotency-Key": "memory-conformance"},
            json=body,
        )
        assert saved.status_code == 201, saved.text
        prepared, record = captured[0]

        async def exercise():
            memory = InMemoryEvalStore()
            parent = await durable.load_authored_suite(prepared.plan.parent.revision)
            await memory.save_authored_suite(parent, redact_json=app.redact_json)
            original_save = memory._save_result_document

            def fail_late(*args, **kwargs):
                original_save(*args, **kwargs)
                raise RuntimeError("late memory publication")

            monkeypatch.setattr(memory, "_save_result_document", fail_late)
            with pytest.raises(RuntimeError, match="late memory"):
                await memory.publish_session_addition(prepared, record, redact_json=app.redact_json)
            assert await memory.load_session_addition(record.receipt.operation_id) is None
            assert await memory.load_result_record(prepared.captured.revision) is None
            assert await memory.load_corpus(prepared.corpus.revision) is None
            monkeypatch.setattr(memory, "_save_result_document", original_save)
            first = await memory.publish_session_addition(
                prepared, record, redact_json=app.redact_json
            )
            assert (
                await memory.publish_session_addition(prepared, record, redact_json=app.redact_json)
                == first
            )
            history = await memory.list_view_index(
                EvalViewQuery(target_key=parent.target_key, suite=prepared.plan.destination)
            )
            assert len(history.items) == 1
            await memory.close()

        # Run on the server's event loop: its eval run coordinator shares this store's lock.
        client.portal.call(exercise)


@pytest.mark.parametrize("backend", ["postgres"])
def test_postgres_additions_publish_and_recover_through_http(
    addition_environment, request, backend, monkeypatch
):
    import os

    # Local runs do not provision PostgreSQL. The required CI lane owns its
    # disposable service/container and must fail rather than silently skip.
    required = os.environ.get("CAYU_REQUIRE_POSTGRES_TESTS", "").lower() in {"1", "true", "yes"}
    if not os.environ.get("CAYU_TEST_POSTGRES_DSN") and not required:
        pytest.skip("PostgreSQL qualification is delegated to the required CI lane.")
    dsn = request.getfixturevalue("postgres_dsn")
    server, _, _ = addition_environment(backend, postgres_dsn=dsn)
    from cayu.storage.evals_postgres import PostgresEvalStore

    with TestClient(server, raise_server_exceptions=False) as client:
        for destination_source in ("authored", "corpus"):
            selector, _ = _destination(client, destination_source)
            for mode in ("case", "score"):
                body, plan = _review(client, selector, mode=mode)
                headers = {**_AUTH_HEADERS, "Idempotency-Key": f"pg-{destination_source}-{mode}"}
                url = f"/api/evals/sessions/{_SESSION_ID}/additions"
                catalog = client.get("/api/evals/suite-views", headers=_AUTH_HEADERS).json()
                method = (
                    "_save_prepared_authored_suite_in_transaction"
                    if mode == "case" and destination_source == "authored"
                    else "_save_result_in_transaction"
                    if mode == "score"
                    else "_save_prepared_corpus_in_transaction"
                )
                original = getattr(PostgresEvalStore, method)
                fired = False

                async def fail_after_write(
                    self, *args, _original=original, _method=method, **kwargs
                ):
                    nonlocal fired
                    if _method == "_save_prepared_authored_suite_in_transaction":
                        await _original(self, *args, **kwargs)
                    else:
                        await _original(*args, **kwargs)
                    fired = True
                    raise RuntimeError("injected PostgreSQL failure after durable statement")

                with monkeypatch.context() as patch:
                    patch.setattr(PostgresEvalStore, method, fail_after_write)
                    failed = client.post(url, headers=headers, json=body)
                    assert failed.status_code == 500, failed.text
                    assert fired
                    assert (
                        client.get("/api/evals/suite-views", headers=_AUTH_HEADERS).json()
                        == catalog
                    )
                saved = client.post(url, headers=headers, json=body)
                assert saved.status_code == 201, saved.text
                assert client.post(url, headers=headers, json=body).json() == saved.json()
                destination = EvalSuiteViewIdentity.model_validate(plan["destination"])
                source_link = client.get(
                    f"/api/evals/suite-views/{destination.id}/cases/{plan['case_id']}/source",
                    headers=_AUTH_HEADERS,
                    params={"result_revision": plan["captured_result_revision"]}
                    if mode == "score"
                    else {},
                )
                assert source_link.status_code == 200, source_link.text
                assert source_link.json() == {"state": "available", "session_id": _SESSION_ID}
                receipt = client.get(
                    f"/api/evals/additions/{saved.json()['operation_id']}", headers=_AUTH_HEADERS
                )
                assert receipt.json()["plan"] == plan
                detail = client.get(f"/api/evals/suite-views/{selector}", headers=_AUTH_HEADERS)
                assert detail.status_code == 200, detail.text


def test_concurrent_additions_fork_exact_parent_and_duplicate_keys_join(addition_environment):
    import asyncio

    import httpx

    server, _, _ = addition_environment()
    with TestClient(server) as client:
        selector, parent = _destination(client, "authored")
        first, _ = _review(client, selector, case_id="first-case", case_name="First case")
        second, _ = _review(client, selector, case_id="second-case", case_name="Second case")

        async def exercise():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as async_client:
                url = f"/api/evals/sessions/{_SESSION_ID}/additions"
                replies = await asyncio.gather(
                    *[
                        async_client.post(
                            url, headers={**_AUTH_HEADERS, "Idempotency-Key": key}, json=body
                        )
                        for key, body in (("first", first), ("second", second), ("first", first))
                    ]
                )
                assert all(reply.status_code == 201 for reply in replies), [
                    reply.text for reply in replies
                ]
                assert replies[0].json() == replies[2].json()
                return [reply.json()["plan"] for reply in replies[:2]]

        # Run on the server's event loop: its eval run coordinator shares this store's lock.
        plans = client.portal.call(exercise)
        assert plans[0]["destination"] != plans[1]["destination"]
        assert (
            client.get(f"/api/evals/suite-views/{selector}", headers=_AUTH_HEADERS).json() == parent
        )
        for plan in plans:
            destination = EvalSuiteViewIdentity.model_validate(plan["destination"])
            detail = client.get(f"/api/evals/suite-views/{destination.id}", headers=_AUTH_HEADERS)
            assert detail.status_code == 200, detail.text
            assert detail.json()["cases"]["total"] == parent["cases"]["total"] + 1
            assert any(case["id"] == plan["case_id"] for case in detail.json()["cases"]["items"])


def test_stale_and_forged_requests_reject_without_publication_and_exact_existing_case_is_noop(
    captured_route_server,
):
    with TestClient(captured_route_server()) as client:
        selector, parent = _destination(client, "authored")
        body, plan = _review(client, selector)
        url = f"/api/evals/sessions/{_SESSION_ID}/additions"
        headers = {**_AUTH_HEADERS, "Idempotency-Key": "validate-before-write"}
        for field in ("expected_plan_revision", "expected_source_revision"):
            rejected = client.post(url, headers=headers, json={**body, field: "sha256:" + "f" * 64})
            assert rejected.status_code == 409, rejected.text
        for change in ({"private_session_id": _SESSION_ID}, {"case_id": True}, {"mode": "future"}):
            rejected = client.post(url, headers=headers, json={**body, **change})
            assert rejected.status_code == 422, rejected.text
        assert (
            client.get(f"/api/evals/suite-views/{selector}", headers=_AUTH_HEADERS).json() == parent
        )
        saved = client.post(url, headers=headers, json=body)
        assert saved.status_code == 201, saved.text
        destination = EvalSuiteViewIdentity.model_validate(plan["destination"])
        repeated, repeated_plan = _review(client, destination.id)
        assert repeated_plan["membership_changed"] is False
        assert repeated_plan["destination"] == plan["destination"]
        noop = client.post(
            url, headers={**_AUTH_HEADERS, "Idempotency-Key": "existing-case"}, json=repeated
        )
        assert noop.status_code == 201, noop.text
        conflict = client.post(
            url + "/preview",
            headers=_AUTH_HEADERS,
            json={
                key: value
                for key, value in {**repeated, "case_name": "Different case"}.items()
                if not key.startswith("expected_")
            },
        )
        assert conflict.status_code == 409, conflict.text


def test_preview_binds_server_source_identity_even_when_portable_evidence_is_identical(
    addition_environment, monkeypatch
):
    import asyncio

    from cayu import Message, RunRequest

    server, app, _ = addition_environment()

    async def second_identical_source():
        async for _ in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="second-private-source",
                messages=[Message.text("user", "promote this completed run")],
            )
        ):
            pass

    asyncio.run(second_identical_source())
    with TestClient(server) as client:
        selector, parent = _destination(client, "authored")
        body, original_plan = _review(client, selector)
        original = type(app)._resolve_public_session_id

        async def changed_resolution(self, public_session_id):
            if self is app and public_session_id == _SESSION_ID:
                return "second-private-source"
            return await original(self, public_session_id)

        monkeypatch.setattr(type(app), "_resolve_public_session_id", changed_resolution)
        url = f"/api/evals/sessions/{_SESSION_ID}/additions"
        updated = client.post(
            url + "/preview",
            headers=_AUTH_HEADERS,
            json={key: value for key, value in body.items() if not key.startswith("expected_")},
        )
        assert updated.status_code == 200, updated.text
        assert updated.json()["source_revision"] == original_plan["source_revision"]
        rejected = client.post(
            url,
            headers={**_AUTH_HEADERS, "Idempotency-Key": "changed-source-authority"},
            json=body,
        )
        assert rejected.status_code == 409, rejected.text
        assert (
            client.get(f"/api/evals/suite-views/{selector}", headers=_AUTH_HEADERS).json() == parent
        )
        assert updated.json()["revision"] != original_plan["revision"]
        assert "second-private-source" not in updated.text


@pytest.mark.parametrize("cancel_owner", [False, True])
@pytest.mark.parametrize("close_failure", [False, True])
def test_failed_sqlite_settlement_never_replays_uncommitted_receipt(
    addition_environment, monkeypatch, tmp_path, cancel_owner, close_failure
):
    import asyncio
    import threading

    import httpx

    from cayu.evals.session_additions import addition_operation_id
    from cayu.storage.evals_sqlite import SQLiteEvalStore

    server, app, store = addition_environment()
    entered, release = threading.Event(), threading.Event()
    primary = RuntimeError("publication commit failed")
    rollback_error = RuntimeError("publication rollback failed")
    close_error = RuntimeError("publication close acknowledgement failed")

    class FaultedConnection:
        closed = False
        armed = False

        def __init__(self, connection):
            self.connection = connection

        def __getattr__(self, name):
            return getattr(self.connection, name)

        def commit(self):
            receipts = self.connection.execute(
                "SELECT COUNT(*) FROM cayu_eval_session_additions"
            ).fetchone()[0]
            if receipts == 0:
                # The server's eval run coordinator polls through the same
                # connection; on a slow runner its claim can commit here first.
                # Fault only the addition publication.
                self.connection.commit()
                return
            # The real receipt INSERT and document writes have already executed.
            assert receipts == 1
            self.armed = True
            entered.set()
            assert release.wait(30)
            raise primary

        def rollback(self):
            if not self.armed:
                self.connection.rollback()
                return
            assert self.connection.in_transaction
            raise rollback_error

        def close(self):
            self.connection.close()
            first = not self.closed
            self.closed = True
            if close_failure and first:
                raise close_error

    def leaves(error):
        if isinstance(error, BaseExceptionGroup):
            return [leaf for child in error.exceptions for leaf in leaves(child)]
        return [error]

    with TestClient(server) as client:
        selector, _ = _destination(client, "authored")
        body, plan = _review(client, selector)
        faulted = FaultedConnection(store._connection)
        monkeypatch.setattr(store, "_connection", faulted)
        key = "failed-settlement"
        operation_id = addition_operation_id("eval-operator", None, key)

        async def exercise():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as http:
                owner = asyncio.create_task(
                    http.post(
                        f"/api/evals/sessions/{_SESSION_ID}/additions",
                        headers={**_AUTH_HEADERS, "Idempotency-Key": key},
                        json=body,
                    )
                )
                try:
                    # Reaching the injected commit is a scheduling barrier, not a latency limit.
                    assert await asyncio.to_thread(entered.wait, 30)
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
                        pytest.fail("publication must fail")
                    assert faulted.closed, "uncertain writer must be fenced before reuse"
                    assert leaves(propagated) == [primary, rollback_error] + (
                        [close_error] if close_failure else []
                    )
                    # The server's eval run coordinator takes this lock on its own schedule,
                    # so it may hold it right now; a lock the failed publication leaked would
                    # never come free.
                    await asyncio.wait_for(store._lock.acquire(), timeout=10)
                    store._lock.release()
                finally:
                    release.set()
                    await asyncio.gather(owner, return_exceptions=True)

        # Run on the server's event loop: its eval run coordinator shares this store's lock.
        client.portal.call(exercise)
        failed_client = TestClient(server, raise_server_exceptions=False)
        assert (
            failed_client.get(
                f"/api/evals/additions/{operation_id}", headers=_AUTH_HEADERS
            ).status_code
            == 500
        )
        assert (
            failed_client.post(
                f"/api/evals/sessions/{_SESSION_ID}/additions",
                headers={**_AUTH_HEADERS, "Idempotency-Key": key},
                json=body,
            ).status_code
            == 500
        )
        reopened = SQLiteEvalStore(tmp_path / "additions-0.db")

        async def inspect_reopened():
            assert await reopened.load_session_addition(operation_id) is None
            destination = EvalSuiteViewIdentity.model_validate(plan["destination"])
            assert await reopened.load_authored_suite(destination.revision) is None

        asyncio.run(inspect_reopened())
        restarted, _, _ = addition_environment(store_override=reopened, app_override=app)
        recovered = TestClient(restarted)
        headers = {**_AUTH_HEADERS, "Idempotency-Key": key}
        url = f"/api/evals/sessions/{_SESSION_ID}/additions"
        saved = recovered.post(url, headers=headers, json=body)
        assert saved.status_code == 201, saved.text
        assert saved.json()["plan"] == plan
        assert recovered.post(url, headers=headers, json=body).json() == saved.json()
        assert (
            recovered.get(f"/api/evals/additions/{operation_id}", headers=_AUTH_HEADERS).json()
            == saved.json()
        )


@pytest.mark.parametrize("source", ["authored", "corpus"])
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_sequential_children_preserve_positive_case_sources(
    addition_environment, request, backend, source, monkeypatch, tmp_path
):
    import asyncio
    import os

    dsn = None
    if backend == "postgres":
        required = os.environ.get("CAYU_REQUIRE_POSTGRES_TESTS", "").lower() in {"1", "true", "yes"}
        if not os.environ.get("CAYU_TEST_POSTGRES_DSN") and not required:
            pytest.skip("PostgreSQL qualification is delegated to the required CI lane.")
        dsn = request.getfixturevalue("postgres_dsn")
    server, app, store = addition_environment(backend, postgres_dsn=dsn)
    with TestClient(server) as client:
        parent, _ = _destination(client, source)
        url = f"/api/evals/sessions/{_SESSION_ID}/additions"
        previous = parent
        receipts = []
        for name in ("case-a", "case-b"):
            body, plan = _review(client, previous, case_id=name)
            headers = {**_AUTH_HEADERS, "Idempotency-Key": f"sequential-{source}-{name}"}
            if name == "case-b":
                fired = False
                if backend == "sqlite":
                    original_connection = store._connection

                    class LateAssociationFailure:
                        def __init__(self, connection):
                            self.connection = connection

                        def __getattr__(self, attribute):
                            return getattr(self.connection, attribute)

                        def execute(self, sql, *args, **kwargs):
                            nonlocal fired
                            result = self.connection.execute(sql, *args, **kwargs)
                            if not fired and sql.startswith(
                                "INSERT INTO cayu_eval_case_addition_sources"
                            ):
                                fired = True
                                raise RuntimeError("after inherited private association write")
                            return result

                    monkeypatch.setattr(
                        store, "_connection", LateAssociationFailure(original_connection)
                    )
                else:
                    from psycopg import AsyncCursor

                    original_execute = AsyncCursor.execute

                    async def late_association(
                        cursor, query, *args, execute=original_execute, **kwargs
                    ):
                        nonlocal fired
                        result = await execute(cursor, query, *args, **kwargs)
                        if (
                            not fired
                            and isinstance(query, str)
                            and query.startswith("INSERT INTO cayu_eval_case_addition_sources")
                        ):
                            fired = True
                            raise RuntimeError("after inherited private association write")
                        return result

                    monkeypatch.setattr(AsyncCursor, "execute", late_association)
                with pytest.raises(RuntimeError, match="after inherited private association"):
                    client.post(url, headers=headers, json=body)
                assert fired
                child = EvalSuiteViewIdentity.model_validate(plan["destination"])
                assert client.portal is not None
                assert client.portal.call(store.load_case_addition, child.id, "case-a") is None
                assert (
                    client.get(
                        f"/api/evals/suite-views/{child.id}", headers=_AUTH_HEADERS
                    ).status_code
                    == 404
                )
                assert (
                    client.get(
                        f"/api/evals/suite-views/{previous}/cases/case-a/source",
                        headers=_AUTH_HEADERS,
                    ).json()["state"]
                    == "available"
                )
                if backend == "sqlite":
                    monkeypatch.setattr(store, "_connection", original_connection)
                else:
                    monkeypatch.setattr(AsyncCursor, "execute", original_execute)
            published = client.post(url, headers=headers, json=body)
            assert published.status_code == 201, published.text
            assert client.post(url, headers=headers, json=body).json() == published.json()
            previous = EvalSuiteViewIdentity.model_validate(plan["destination"]).id
            receipts.append((previous, published.json()))
        for selector, case_id in (
            (receipts[0][0], "case-a"),
            (previous, "case-a"),
            (previous, "case-b"),
        ):
            linked = client.get(
                f"/api/evals/suite-views/{selector}/cases/{case_id}/source", headers=_AUTH_HEADERS
            )
            assert linked.status_code == 200, linked.text
            assert linked.json() == {"state": "available", "session_id": _SESSION_ID}

        async def reconstruct():
            if backend == "sqlite":
                from cayu.storage.evals_sqlite import SQLiteEvalStore

                reopened = SQLiteEvalStore(tmp_path / "additions-0.db")
            else:
                from cayu.storage.evals_postgres import PostgresEvalStore

                reopened = PostgresEvalStore(dsn)
            try:
                for case_id in ("case-a", "case-b"):
                    record = await reopened.load_case_addition(previous, case_id)
                    assert record is not None
                    assert record.receipt.plan.case_id == case_id
                    assert record.public_session_id == _SESSION_ID
                original_record = await reopened.load_session_addition(
                    receipts[0][1]["operation_id"]
                )
                assert original_record is not None
                assert original_record.receipt.model_dump(mode="json") == receipts[0][1]
            finally:
                await reopened.close()

        asyncio.run(reconstruct())
        # Live operational resolution, not copied portable provenance, still owns links.
        from cayu import CayuApp

        original_resolve = CayuApp._resolve_public_session_id

        async def remapped(self, session_id):
            if self is app and session_id == _SESSION_ID:
                return "wrong-private-source"
            return await original_resolve(self, session_id)

        monkeypatch.setattr(CayuApp, "_resolve_public_session_id", remapped)
        link = f"/api/evals/suite-views/{previous}/cases/case-a/source"
        assert client.get(link, headers=_AUTH_HEADERS).json()["state"] == "unavailable"
        monkeypatch.setattr(CayuApp, "_resolve_public_session_id", original_resolve)
        assert client.portal is not None
        client.portal.call(app.session_store.delete_session, _SESSION_ID)
        assert client.get(link, headers=_AUTH_HEADERS).json()["state"] == "unavailable"


@pytest.mark.parametrize("source", ["authored", "corpus"])
def test_memory_child_associations_rollback_and_replay_exactly(
    addition_environment, monkeypatch, source
):
    from cayu.evals.store import InMemoryEvalStore
    from cayu.storage.evals_sqlite import SQLiteEvalStore

    server, app, durable = addition_environment()
    captured = []
    original = SQLiteEvalStore.publish_session_addition

    async def observe(self, prepared, record, **kwargs):
        captured.append((prepared, record))
        return await original(self, prepared, record, **kwargs)

    monkeypatch.setattr(SQLiteEvalStore, "publish_session_addition", observe)
    with TestClient(server) as client:
        selector, _ = _destination(client, source)
        for case_id in ("case-a", "case-b"):
            body, plan = _review(client, selector, case_id=case_id)
            saved = client.post(
                f"/api/evals/sessions/{_SESSION_ID}/additions",
                headers={**_AUTH_HEADERS, "Idempotency-Key": case_id},
                json=body,
            )
            assert saved.status_code == 201, saved.text
            selector = EvalSuiteViewIdentity.model_validate(plan["destination"]).id

        async def exercise():
            memory = InMemoryEvalStore()
            initial = captured[0][0].plan.parent
            assert initial is not None
            if source == "authored":
                document = await durable.load_authored_suite(initial.revision)
                await memory.save_authored_suite(document, redact_json=app.redact_json)
            else:
                document = await durable.load_corpus(initial.revision)
                await memory.save_corpus(document, redact_json=app.redact_json)
            prepared_a, record_a = captured[0]
            await memory.publish_session_addition(prepared_a, record_a, redact_json=app.redact_json)
            prepared_b, record_b = captured[1]

            class LateFailure(dict):
                def setdefault(self, key, default=None):
                    super().setdefault(key, default)
                    raise RuntimeError("after inherited memory association write")

            memory._case_addition_sources = LateFailure(memory._case_addition_sources)
            with pytest.raises(RuntimeError, match="after inherited memory association"):
                await memory.publish_session_addition(
                    prepared_b, record_b, redact_json=app.redact_json
                )
            assert await memory.load_case_addition(prepared_b.plan.destination.id, "case-a") is None
            assert await memory.load_session_addition(record_b.receipt.operation_id) is None
            assert (
                await memory.load_case_addition(prepared_a.plan.destination.id, "case-a")
                == record_a
            )
            committed = await memory.publish_session_addition(
                prepared_b, record_b, redact_json=app.redact_json
            )
            assert (
                await memory.publish_session_addition(
                    prepared_b, record_b, redact_json=app.redact_json
                )
                == committed
            )
            assert (
                await memory.load_case_addition(prepared_b.plan.destination.id, "case-a")
                == record_a
            )
            assert (
                await memory.load_case_addition(prepared_b.plan.destination.id, "case-b")
                == record_b
            )
            await memory.close()

        # Run on the server's event loop: its eval run coordinator shares this store's lock.
        client.portal.call(exercise)


def test_corpus_child_preserves_other_suites_positive_sources(addition_environment):

    from cayu.evals.corpus import EvalCaseSpec, EvalCorpusDocument, EvalSuiteSpec

    server, app, store = addition_environment()
    with TestClient(server) as client:
        original_selector, _ = _destination(client, "corpus")
        from cayu.evals.read_views import decode_suite_view

        identity = decode_suite_view(original_selector)
        assert client.portal is not None
        corpus = client.portal.call(store.load_corpus, identity.revision)
        other_suite = EvalSuiteSpec.create(id="other-suite", name="Other suite")
        original_case = corpus.cases[0]
        other_case = EvalCaseSpec.create(
            id="other-original",
            suite_id=other_suite.id,
            name=original_case.name,
            source=original_case.source,
            input=original_case.input,
            assertions=original_case.assertions,
        )
        expanded = EvalCorpusDocument.create(
            target_key=corpus.target_key,
            evidence_policy=corpus.evidence_policy,
            pricing_profile=corpus.pricing_profile,
            suites=(*corpus.suites, other_suite),
            cases=(*corpus.cases, other_case),
        )
        from functools import partial

        client.portal.call(partial(store.save_corpus, expanded, redact_json=app.redact_json))
        first_selector = EvalSuiteViewIdentity(
            source="corpus",
            target_key=expanded.target_key,
            revision=expanded.revision,
            suite_id=identity.suite_id,
        ).id
        url = f"/api/evals/sessions/{_SESSION_ID}/additions"
        first_body, first_plan = _review(client, first_selector, case_id="case-a")
        first = client.post(
            url, headers={**_AUTH_HEADERS, "Idempotency-Key": "first-suite"}, json=first_body
        )
        assert first.status_code == 201, first.text
        first_destination = EvalSuiteViewIdentity.model_validate(first_plan["destination"])
        other_selector = EvalSuiteViewIdentity(
            source="corpus",
            target_key=expanded.target_key,
            revision=first_destination.revision,
            suite_id=other_suite.id,
        ).id
        second_body, second_plan = _review(client, other_selector, case_id="case-b")
        second = client.post(
            url, headers={**_AUTH_HEADERS, "Idempotency-Key": "other-suite"}, json=second_body
        )
        assert second.status_code == 201, second.text
        child = EvalSuiteViewIdentity.model_validate(second_plan["destination"])
        unchanged_suite = EvalSuiteViewIdentity(
            source="corpus",
            target_key=expanded.target_key,
            revision=child.revision,
            suite_id=identity.suite_id,
        )
        linked = client.get(
            f"/api/evals/suite-views/{unchanged_suite.id}/cases/case-a/source",
            headers=_AUTH_HEADERS,
        )
        assert linked.json() == {"state": "available", "session_id": _SESSION_ID}


@pytest.mark.parametrize("destination_kind", ["new", "v2", "v3"])
def test_authored_addition_preserves_structured_judge_assertions_through_http(
    addition_environment, destination_kind
):

    from cayu.evals.corpus import StructuredRubricCriterionV1
    from cayu.evals.suite_authoring import (
        EvalCaseDraftV2,
        EvalSuiteDraftV2,
        EvalSuiteDraftV3,
        PublicJudgeReferenceDraftV1,
        StructuredModelJudgeAssertionDraftV1,
        StructuredRubricDraftV1,
    )

    server, _, store = addition_environment()
    assertion = StructuredModelJudgeAssertionDraftV1(
        id="quality",
        judge_profile_key="quality-judge",
        judge_profile_revision="sha256:" + "a" * 64,
        rubric=StructuredRubricDraftV1(
            id="answer-quality",
            criteria=(
                StructuredRubricCriterionV1(
                    id="correct", name="Correctness", description="Answer is correct", weight="1"
                ),
            ),
        ),
        reference=PublicJudgeReferenceDraftV1(
            id="answer-reference", expected_answer="Expected answer"
        ),
    ).compile()
    with TestClient(server) as client:
        selector, _ = _destination(client, "authored")
        from cayu.evals.read_views import decode_suite_view

        if destination_kind == "new":
            destination = {"kind": "new", "suite_id": "judge-cases", "name": "Judge cases"}
        else:
            draft_type = EvalSuiteDraftV2 if destination_kind == "v2" else EvalSuiteDraftV3
            initial = _draft()
            draft = draft_type(
                id="judge-parent",
                name="Judge parent",
                target_key=decode_suite_view(selector).target_key,
                cases=tuple(
                    EvalCaseDraftV2.model_validate(case.model_dump(mode="python"))
                    for case in initial.cases
                ),
            )
            preview = client.post(
                "/api/evals/suites/preview",
                headers=_AUTH_HEADERS,
                json={"draft": draft.model_dump(mode="json")},
            )
            assert preview.status_code == 200, preview.text
            document = preview.json()["suite"]
            saved = client.post(
                "/api/evals/suites",
                headers=_AUTH_HEADERS,
                json={"suite": document, "expected_suite_revision": document["revision"]},
            )
            assert saved.status_code == 201, saved.text
            destination = {
                "kind": "existing",
                "selector": EvalSuiteViewIdentity(
                    source="authored",
                    target_key=document["target_key"],
                    revision=document["revision"],
                ).id,
            }
        body, plan = _review(
            client,
            selector,
            destination=destination,
            assertions=[assertion.model_dump(mode="json")],
        )
        saved = client.post(
            f"/api/evals/sessions/{_SESSION_ID}/additions",
            headers={**_AUTH_HEADERS, "Idempotency-Key": "structured-judge"},
            json=body,
        )
        assert saved.status_code == 201, saved.text
        assert client.portal is not None
        document = client.portal.call(store.load_authored_suite, plan["destination"]["revision"])
        assert document is not None
        case = next(case for case in document.cases if case.id == plan["case_id"])
        assert case.assertions == (assertion,)
        assert case.assertions[0].rubric.revision == assertion.rubric.revision
        assert assertion.reference is not None
        assert case.assertions[0].reference is not None
        assert case.assertions[0].reference.revision == assertion.reference.revision


def test_exact_replay_distinguishes_boolean_and_numeric_assertion_values(
    addition_environment, monkeypatch
):
    import copy

    from cayu.evals.session_additions import (
        EvalAdditionConflict,
        EvalAdditionRecord,
        addition_revision,
    )
    from cayu.storage.evals_sqlite import SQLiteEvalStore

    server, _, store = addition_environment()
    captured = []
    original = SQLiteEvalStore.publish_session_addition

    async def observe(self, prepared, record, **kwargs):
        captured.append((prepared, record, kwargs))
        return await original(self, prepared, record, **kwargs)

    monkeypatch.setattr(SQLiteEvalStore, "publish_session_addition", observe)
    with TestClient(server) as client:
        selector, _ = _destination(client, "authored")
        body, _ = _review(
            client,
            selector,
            assertions=[
                {
                    "id": "arguments",
                    "kind": "tool_arguments_contain",
                    "tool_name": "example",
                    "expected_subset": {"enabled": True},
                }
            ],
        )
        headers = {**_AUTH_HEADERS, "Idempotency-Key": "json-types"}
        url = f"/api/evals/sessions/{_SESSION_ID}/additions"
        saved = client.post(url, headers=headers, json=body)
        assert saved.status_code == 201, saved.text
        assert client.post(url, headers=headers, json=body).json() == saved.json()
        changed = copy.deepcopy(body)
        changed["assertions"][0]["expected_subset"]["enabled"] = 1
        rejected = client.post(url, headers=headers, json=changed)
        assert rejected.status_code == 409, rejected.text
        assert client.post(url, headers=headers, json=body).json() == saved.json()

        # The typed internal store entrance enforces the same exact JSON identity.
        prepared, record, kwargs = captured[0]
        material = record.model_dump(mode="json")
        material["request_material"]["request"]["assertions"][0]["expected_subset"]["enabled"] = 1
        material["receipt"]["request_revision"] = addition_revision(material["request_material"])
        changed_record = EvalAdditionRecord.model_validate(material)
        with pytest.raises(EvalAdditionConflict):
            from functools import partial

            assert client.portal is not None
            client.portal.call(
                partial(store.publish_session_addition, prepared, changed_record, **kwargs)
            )
