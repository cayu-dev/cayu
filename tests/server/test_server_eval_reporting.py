from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from uuid import uuid4

import httpx
import pytest
from tests.cli.test_evals_cli import build_slow_eval_plan
from tests.server.test_server_captured_evaluations import (
    captured_route_server as captured_route_server,
)
from tests.server.test_server_evals import _AUTH_HEADERS, _provider, _server, _target

from cayu.evals.runner import run_eval_plan
from cayu.storage.evals_sqlite import SQLiteEvalStore

_TOKEN = "reporting-test-credential-0000000000000001"
_ROTATED_TOKEN = "reporting-test-credential-0000000000000002"
_REPORT_HEADERS = {"Authorization": f"Bearer {_TOKEN}"}


def _configure(monkeypatch, target_key):
    monkeypatch.setenv(
        "CAYU_EVAL_REPORTERS",
        json.dumps(
            [
                {
                    "publisher_id": "github:example/reviewer",
                    "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
                    "target_keys": [target_key],
                }
                for token in (_TOKEN, _ROTATED_TOKEN)
            ]
        ),
    )


async def _payload(target_key):
    run = await run_eval_plan(build_slow_eval_plan())
    document = run.model_dump(mode="json")
    evidence = {
        key: document[key]
        for key in ("suite_id", "status", "score", "started_at", "completed_at", "duration_ms")
    }
    evidence["cases"] = [
        {
            **{key: case[key] for key in ("case_id", "status", "score", "trial_policy")},
            "trials": [
                {
                    **{
                        key: trial.get(key)
                        for key in (
                            "trial_number",
                            "status",
                            "score",
                            "final_output",
                            "structured_output",
                            "error",
                            "unavailable_reason",
                        )
                    },
                    "assertions": [
                        {
                            key: check[key]
                            for key in ("name", "outcome", "score", "threshold", "message")
                        }
                        for check in trial["assertions"]
                    ],
                }
                for trial in case["trials"]
            ],
        }
        for case in document["cases"]
    ]
    return {
        "schema_version": 1,
        "report_id": str(uuid4()),
        "target_key": target_key,
        "program_key": "evals.reviewer:build",
        "declared_suite_revision": None,
        "origin": {
            "kind": "ci",
            "provider": "github",
            "repository": "example/reviewer",
            "branch": "feature/checks",
            "commit": "a" * 40,
            "pr_head_commit": "b" * 40,
            "pull_request": 42,
            "run_url": "https://github.com/example/reviewer/actions/runs/123",
        },
        "evidence": {"kind": "direct", "run": evidence},
    }


def test_http_report_persists_external_unknown_revision_and_replays_after_restart(
    tmp_path, monkeypatch
):
    async def exercise():
        target = _target(_provider())
        _configure(monkeypatch, target.key)
        payload = await _payload(target.key)
        database = tmp_path / "reports.db"
        receipt = None
        for attempt in range(2):
            store = SQLiteEvalStore(database)
            try:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=_server(target, store)),
                    base_url="http://test",
                ) as client:
                    response = await client.post(
                        "/api/evals/reports", json=payload, headers=_REPORT_HEADERS
                    )
                    assert response.status_code == 200, response.text
                    if attempt == 0:
                        receipt = response.json()
                    else:
                        assert response.json() == receipt
                    assert receipt["publisher_id"] == "github:example/reviewer"
                    catalog = await client.get("/api/evals/suite-views", headers=_AUTH_HEADERS)
                    assert catalog.status_code == 200, catalog.text
                    rows = catalog.json()["items"]
                    assert len(rows) == 1
                    series = rows[0]
                    assert series["source"] == "reported"
                    assert "revision" not in series["identity"]
                    assert series["identity"]["program_key"] == payload["program_key"]
                    detail = await client.get(
                        f"/api/evals/run-views/{receipt['run_key']}", headers=_AUTH_HEADERS
                    )
                    assert detail.status_code == 200, detail.text
                    body = detail.json()
                    assert body["run"]["declared_suite_revision"] is None
                    assert body["run"]["origin"]["commit"] == "a" * 40
                    assert body["run"]["origin"]["pr_head_commit"] == "b" * 40
                    assert body["comparison"]["state"] == "unavailable"
                    assert body["comparison"]["unavailable_reasons"] == [
                        "external_report_comparison_unavailable"
                    ]
                    assert body["results"]["total"] == 1
            finally:
                await store.close()

    asyncio.run(exercise())


def test_report_token_is_not_operator_authority_and_rotation_preserves_exact_identity(
    tmp_path, monkeypatch
):
    async def exercise():
        target = _target(_provider())
        _configure(monkeypatch, target.key)
        store = SQLiteEvalStore(tmp_path / "reports.db")
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=_server(target, store)), base_url="http://test"
            ) as client:
                payload = await _payload(target.key)
                created = await client.post(
                    "/api/evals/reports", json=payload, headers=_REPORT_HEADERS
                )
                assert created.status_code == 200, created.text
                rotated = await client.post(
                    "/api/evals/reports",
                    json=payload,
                    headers={"Authorization": f"Bearer {_ROTATED_TOKEN}"},
                )
                assert rotated.status_code == 200, rotated.text
                assert rotated.json() == created.json()
                for path in ("/api/agents", "/api/evals/suite-views", "/api/sessions"):
                    response = await client.get(path, headers=_REPORT_HEADERS)
                    assert response.status_code == 401, (path, response.text)
                denied = await client.post(
                    "/api/evals/reports", json=payload, headers=_AUTH_HEADERS
                )
                assert denied.status_code == 401
                changed = copy.deepcopy(payload)
                changed["origin"]["branch"] = "another-branch"
                conflict = await client.post(
                    "/api/evals/reports", json=changed, headers=_REPORT_HEADERS
                )
                assert conflict.status_code == 409, conflict.text
                foreign = copy.deepcopy(payload)
                foreign["target_key"] = "another-specialist"
                rejected = await client.post(
                    "/api/evals/reports", json=foreign, headers=_REPORT_HEADERS
                )
                assert rejected.status_code == 403, rejected.text
        finally:
            await store.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "authorization",
    [None, "Bearer invalid", "Bearer " + "x" * 8193],
    ids=["missing", "unknown", "oversize"],
)
def test_reporting_auth_rejects_before_parsing_private_body(tmp_path, monkeypatch, authorization):
    async def exercise():
        target = _target(_provider())
        _configure(monkeypatch, target.key)
        store = SQLiteEvalStore(tmp_path / "reports.db")
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=_server(target, store)), base_url="http://test"
            ) as client:
                response = await client.post(
                    "/api/evals/reports",
                    content=b'{"private": "REPORT-PRIVATE-CANARY"',
                    headers={} if authorization is None else {"Authorization": authorization},
                )
                assert response.status_code == 401, response.text
                assert "REPORT-PRIVATE-CANARY" not in response.text
        finally:
            await store.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "mode", ["bool_version", "duplicate_key", "credential_echo", "unknown_field"]
)
def test_http_report_rejects_unsafe_evidence_without_mutation(
    tmp_path, monkeypatch, capsys, caplog, mode
):
    async def exercise():
        target = _target(_provider())
        _configure(monkeypatch, target.key)
        payload = await _payload(target.key)
        if mode == "bool_version":
            payload["schema_version"] = True
        elif mode == "credential_echo":
            payload["evidence"]["run"]["cases"][0]["trials"][0]["final_output"] = _TOKEN
        elif mode == "unknown_field":
            payload["private"] = "REPORT-REJECTED-PRIVATE-CANARY"
        raw = json.dumps(payload)
        if mode == "duplicate_key":
            raw = raw[:-1] + ',"target_key":"REPORT-REJECTED-PRIVATE-CANARY"}'
        store = SQLiteEvalStore(tmp_path / "reports.db")
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=_server(target, store)), base_url="http://test"
            ) as client:
                response = await client.post(
                    "/api/evals/reports",
                    content=raw,
                    headers={**_REPORT_HEADERS, "Content-Type": "application/json"},
                )
                assert response.status_code == 422, response.text
                catalog = await client.get("/api/evals/suite-views", headers=_AUTH_HEADERS)
                assert catalog.status_code == 200 and catalog.json()["items"] == []
                captured = capsys.readouterr()
                diagnostic = response.text + captured.out + captured.err + caplog.text
                assert (
                    _TOKEN not in diagnostic and "REPORT-REJECTED-PRIVATE-CANARY" not in diagnostic
                )
        finally:
            await store.close()

    asyncio.run(exercise())


def test_http_report_long_series_identity_has_bounded_paging_and_download(tmp_path, monkeypatch):
    async def exercise():
        target = _target(_provider())
        _configure(monkeypatch, target.key)
        payload = await _payload(target.key)
        payload["program_key"] = "😀" * 256
        payload["evidence"]["run"]["suite_id"] = "😀" * 256
        store = SQLiteEvalStore(tmp_path / "reports.db")
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=_server(target, store)), base_url="http://test"
            ) as client:
                receipts = []
                for _ in range(2):
                    payload["report_id"] = str(uuid4())
                    published = await client.post(
                        "/api/evals/reports", json=payload, headers=_REPORT_HEADERS
                    )
                    assert published.status_code == 200, published.text
                    receipts.append(published.json())
                catalog = await client.get("/api/evals/suite-views", headers=_AUTH_HEADERS)
                assert catalog.status_code == 200, catalog.text
                series = catalog.json()["items"][0]
                assert len(series["id"]) < 1024
                detail = await client.get(
                    f"/api/evals/suite-views/{series['id']}",
                    params={"runs_limit": 1},
                    headers=_AUTH_HEADERS,
                )
                assert detail.status_code == 200, detail.text
                cursor = detail.json()["runs"]["next_cursor"]
                assert cursor and len(cursor) < 1024
                second = await client.get(
                    f"/api/evals/suite-views/{series['id']}",
                    params={"runs_limit": 1, "runs_cursor": cursor},
                    headers=_AUTH_HEADERS,
                )
                assert second.status_code == 200, second.text
                assert second.json()["runs"]["items"][0]["id"] == receipts[0]["run_key"]
                download = await client.get(
                    f"/api/evals/reports/{receipts[1]['run_key']}/download", headers=_AUTH_HEADERS
                )
                assert download.status_code == 200, download.text
                from cayu.evals.reported import EvalReport

                assert download.json() == EvalReport.model_validate(payload).model_dump(mode="json")
                assert "attachment" in download.headers["content-disposition"]
        finally:
            await store.close()

    asyncio.run(exercise())


def test_http_report_exact_corpus_association_keeps_both_comparison_directions_unavailable(
    tmp_path, monkeypatch
):
    from tests.evals.test_corpus_execution import _corpus
    from tests.evals.test_eval_store import _request

    from cayu.evals.execution import run_corpus_suite
    from cayu.evals.read_views import EvalSuiteViewIdentity, run_view_id

    async def exercise():
        target = _target(_provider())
        _configure(monkeypatch, target.key)
        corpus = _corpus(target_key=target.key, trials=1)
        result = await run_corpus_suite(target, corpus, corpus.suites[0].id, max_concurrency=1)
        store = SQLiteEvalStore(tmp_path / "reports.db")
        payload = {
            "schema_version": 1,
            "report_id": str(uuid4()),
            "target_key": target.key,
            "program_key": "corpus-ci",
            "evidence": {"kind": "corpus", "result": result.model_dump(mode="json")},
            "association": {"corpus_revision": corpus.revision, "suite_id": corpus.suites[0].id},
        }
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=_server(target, store)), base_url="http://test"
            ) as client:
                absent = await client.post(
                    "/api/evals/reports", json=payload, headers=_REPORT_HEADERS
                )
                assert absent.status_code == 409, absent.text
                await store.save_corpus(corpus, redact_json=target.app.redact_json)
                from tests.server.test_server_eval_suite_authoring import _draft

                from cayu.evals.store import EvalRunInvocation
                from cayu.evals.suite_authoring import compile_eval_suite_authoring_draft

                authored = compile_eval_suite_authoring_draft(
                    _draft().model_copy(
                        update={"target_key": target.key, "id": corpus.suites[0].id}
                    )
                )
                await store.save_authored_suite(authored, redact_json=target.app.redact_json)
                await store.admit_run(
                    _request(
                        corpus, run_id="authored-report-source", key="authored-report-source"
                    ).model_copy(
                        update={
                            "invocation": EvalRunInvocation(
                                authored_suite_revision=authored.revision,
                                authored_suite_selection_revision="sha256:" + "a" * 64,
                            )
                        }
                    ),
                    redact_json=target.app.redact_json,
                )
                before = await client.get("/api/evals/suite-views", headers=_AUTH_HEADERS)
                assert [item["source"] for item in before.json()["items"]] == ["authored"]
                created = await client.post(
                    "/api/evals/reports", json=payload, headers=_REPORT_HEADERS
                )
                assert created.status_code == 200, created.text
                external = created.json()["run_key"]
                # An imported corpus document does not create a native result record.
                assert await store.load_result_record(result.revision) is None
                baseline = await client.post(
                    f"/api/evals/results/{result.revision}/baseline",
                    json={
                        "result_revision": result.revision,
                        "expected_generation": 0,
                        "operation_id": "sha256:" + "9" * 64,
                    },
                    headers=_AUTH_HEADERS,
                )
                assert baseline.status_code == 404, baseline.text
                identity = EvalSuiteViewIdentity(
                    source="corpus",
                    target_key=target.key,
                    revision=corpus.revision,
                    suite_id=corpus.suites[0].id,
                )
                history = await client.get(
                    f"/api/evals/suite-views/{identity.id}", headers=_AUTH_HEADERS
                )
                assert history.status_code == 200, history.text
                assert history.json()["runs"]["items"][0]["id"] == external
                cursor, entries = None, []
                while True:
                    catalog = await client.get(
                        "/api/evals/suite-views",
                        headers=_AUTH_HEADERS,
                        params={"limit": 1, **({"cursor": cursor} if cursor else {})},
                    )
                    page = catalog.json()
                    entries.extend(page["items"])
                    if not page["has_more"]:
                        break
                    cursor = page["next_cursor"]
                combined = next((item for item in entries if item["id"] == identity.id), None)
                assert combined is not None, entries
                assert combined["last_run"]["id"] == external
                authored_identity = EvalSuiteViewIdentity(
                    source="authored", target_key=target.key, revision=authored.revision
                )
                authored_history = await client.get(
                    f"/api/evals/suite-views/{authored_identity.id}", headers=_AUTH_HEADERS
                )
                assert [
                    item["native_run_id"] for item in authored_history.json()["runs"]["items"]
                ] == ["authored-report-source"]
                await store.request_cancel("authored-report-source")
                await store.admit_run(
                    _request(corpus, run_id="native", key="native"),
                    redact_json=target.app.redact_json,
                )
                lease = await store.claim_run(lease_seconds=300)
                await store.publish_result(lease.claim, result, redact_json=target.app.redact_json)
                native = run_view_id("run", "native")
                catalog = (await client.get("/api/evals/suite-views", headers=_AUTH_HEADERS)).json()
                reported_series = next(
                    item["id"] for item in catalog["items"] if item["source"] == "reported"
                )
                rejected_placement = await client.get(
                    f"/api/evals/run-views/{native}",
                    params={"placement": reported_series},
                    headers=_AUTH_HEADERS,
                )
                assert rejected_placement.status_code == 409, rejected_placement.text
                for placement in (reported_series, identity.id):
                    retained = await client.get(
                        f"/api/evals/run-views/{external}",
                        params={"placement": placement},
                        headers=_AUTH_HEADERS,
                    )
                    assert retained.status_code == 200, retained.text
                    assert retained.json()["run"]["source"] == "reported"
                for current, previous in (
                    (native, external),
                    (external, native),
                    (external, external),
                ):
                    detail = await client.get(
                        f"/api/evals/run-views/{current}",
                        params={"compared_with": previous},
                        headers=_AUTH_HEADERS,
                    )
                    assert detail.status_code == 200, detail.text
                    assert detail.json()["comparison"]["state"] == "unavailable"
                    assert detail.json()["comparison"]["unavailable_reasons"] == [
                        "external_report_comparison_unavailable"
                    ]
                    assert detail.json()["comparison"]["regressions"] == []
        finally:
            await store.close()

    asyncio.run(exercise())


def test_http_report_cancel_after_commit_keeps_owner_and_exact_retry(tmp_path, monkeypatch):
    import threading

    async def exercise():
        target = _target(_provider())
        _configure(monkeypatch, target.key)
        payload = await _payload(target.key)
        store = SQLiteEvalStore(tmp_path / "reports.db")
        committed = threading.Event()
        release = threading.Event()
        original = store._run

        async def held_run(operation, **kwargs):
            def held(connection):
                value = operation(connection)
                committed.set()
                assert release.wait(5), "Test must release the committed worker"
                return value

            return await original(held, **kwargs)

        monkeypatch.setattr(store, "_run", held_run)
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=_server(target, store)), base_url="http://test"
            ) as client:
                owner = asyncio.create_task(
                    client.post("/api/evals/reports", json=payload, headers=_REPORT_HEADERS)
                )
                assert await asyncio.to_thread(committed.wait, 2), (
                    "Publication must reach the real durable owner"
                )
                owner.cancel()
                await asyncio.sleep(0)
                owner.cancel()
                assert owner.cancelling() == 2 and not owner.done()
                retry = asyncio.create_task(
                    client.post("/api/evals/reports", json=payload, headers=_REPORT_HEADERS)
                )
                await asyncio.sleep(0)
                assert not retry.done()
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await owner
                assert owner.cancelled() and owner.cancelling() == 2
                recovered = await retry
                assert recovered.status_code == 200, recovered.text
                repeated = await client.post(
                    "/api/evals/reports", json=payload, headers=_REPORT_HEADERS
                )
                assert repeated.json() == recovered.json()
                catalog = await client.get("/api/evals/suite-views", headers=_AUTH_HEADERS)
                assert len(catalog.json()["items"]) == 1
        finally:
            release.set()
            await store.close()

    asyncio.run(exercise())


def test_http_report_exact_text_byte_bound_and_request_one_over(tmp_path, monkeypatch):
    async def exercise():
        target = _target(_provider())
        _configure(monkeypatch, target.key)
        payload = await _payload(target.key)
        trial = payload["evidence"]["run"]["cases"][0]["trials"][0]
        trial["final_output"] = "😀" * ((64 << 10) // 4)
        store = SQLiteEvalStore(tmp_path / "reports.db")
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=_server(target, store)), base_url="http://test"
            ) as client:
                response = await client.post(
                    "/api/evals/reports", json=payload, headers=_REPORT_HEADERS
                )
                assert response.status_code == 200, response.text
                payload["report_id"] = str(uuid4())
                trial["final_output"] += "a"
                one_over = await client.post(
                    "/api/evals/reports", json=payload, headers=_REPORT_HEADERS
                )
                assert one_over.status_code == 422, one_over.text
                oversized = await client.post(
                    "/api/evals/reports",
                    content=b"{}",
                    headers={**_REPORT_HEADERS, "Content-Length": str((40 << 20) + (64 << 10) + 1)},
                )
                assert oversized.status_code == 413, oversized.text
        finally:
            await store.close()

    asyncio.run(exercise())


def test_reported_series_cannot_receive_session_additions_or_source_links(captured_route_server):
    from fastapi.testclient import TestClient
    from tests.server.test_server_evaluation_promotion import _SESSION_ID

    from cayu.evals.reported import EvalReportedSeriesSelector

    selector = EvalReportedSeriesSelector(series_id="a" * 64).id
    with TestClient(captured_route_server()) as client:
        response = client.post(
            f"/api/evals/sessions/{_SESSION_ID}/additions/preview",
            headers=_AUTH_HEADERS,
            json={
                "mode": "case",
                "destination": {"kind": "existing", "selector": selector},
                "case_id": "captured-addition",
                "case_name": "Captured addition",
            },
        )
        assert response.status_code == 422, response.text
        source = client.get(
            f"/api/evals/suite-views/{selector}/cases/captured-addition/source",
            headers=_AUTH_HEADERS,
        )
        assert source.status_code == 422, source.text


def test_member_can_inspect_report_but_never_download_or_publish_with_operator_auth(
    tmp_path, monkeypatch
):
    from tests.server.test_server_evals import _authenticate, _evals_config

    from cayu.server import (
        AuthContext,
        ControlPlaneRoles,
        DashboardConfig,
        ServerConfig,
        create_server,
    )

    class MemberAuth:
        control_plane_roles = ControlPlaneRoles()

        def __call__(self, request):
            _authenticate(request)
            context = AuthContext(subject="viewer", claims={"role": "member"})
            self.control_plane_roles.require(request, context)
            return context

    async def exercise():
        target = _target(_provider())
        _configure(monkeypatch, target.key)
        store = SQLiteEvalStore(tmp_path / "member-report.db")
        try:
            server = create_server(
                target.app,
                config=ServerConfig.protected(
                    MemberAuth(),
                    dashboard=DashboardConfig(enabled=False),
                    evals=_evals_config(target, store),
                ),
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                payload = await _payload(target.key)
                accepted = await client.post(
                    "/api/evals/reports", headers=_REPORT_HEADERS, json=payload
                )
                assert accepted.status_code == 200, accepted.text
                key = accepted.json()["run_key"]
                catalog = await client.get("/api/evals/suite-views", headers=_AUTH_HEADERS)
                assert catalog.status_code == 200, catalog.text
                series = catalog.json()["items"][0]["id"]
                for path in (f"/api/evals/suite-views/{series}", f"/api/evals/run-views/{key}"):
                    response = await client.get(path, headers=_AUTH_HEADERS)
                    assert response.status_code == 200, response.text
                assert (
                    await client.get(f"/api/evals/reports/{key}/download", headers=_AUTH_HEADERS)
                ).status_code == 403
                assert (
                    await client.post("/api/evals/reports", headers=_AUTH_HEADERS, json=payload)
                ).status_code == 401
        finally:
            await store.close()

    asyncio.run(exercise())


def test_report_large_case_remains_readable_with_exact_download(tmp_path, monkeypatch):
    async def exercise():
        target = _target(_provider())
        _configure(monkeypatch, target.key)
        payload = await _payload(target.key)
        run = payload["evidence"]["run"]
        large = copy.deepcopy(run["cases"][0])
        large["case_id"] = "zz-large-case"
        large["trials"][0]["assertions"] = [
            {
                "name": f"assertion-{i}",
                "outcome": "passed",
                "score": 1.0,
                "message": "x" * (64 << 10),
            }
            for i in range(129)
        ]
        run["cases"].append(large)
        path = tmp_path / "large.db"
        receipt = None
        for attempt in range(2):
            store = SQLiteEvalStore(path)
            try:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=_server(target, store)),
                    base_url="http://test",
                ) as client:
                    if attempt == 0:
                        response = await client.post(
                            "/api/evals/reports", json=payload, headers=_REPORT_HEADERS
                        )
                        assert response.status_code == 200, response.text
                        receipt = response.json()
                    url = f"/api/evals/run-views/{receipt['run_key']}"
                    response = await client.get(
                        url,
                        params={"cases_limit": 2, "max_result_bytes": 8388608},
                        headers=_AUTH_HEADERS,
                    )
                    assert response.status_code == 200, response.text
                    detail = response.json()
                    assert detail["run"]["cases"]["total"] == 2
                    small, big = detail["results"]["items"]
                    assert small["evidence"] is not None
                    assert big["evidence"] is None and big["evidence_omitted"] is True
                    assert big["status"] == "passed" and big["trial_count"] == 1
                    for selected in ("failing", "flaky", "improved", "regressed"):
                        filtered = await client.get(
                            url, params={"case_filter": selected}, headers=_AUTH_HEADERS
                        )
                        assert filtered.status_code == 200
                        assert filtered.json()["results"]["total"] == 0
                        assert filtered.json()["run"]["cases"]["total"] == 2
                    first = (
                        await client.get(url, params={"cases_limit": 1}, headers=_AUTH_HEADERS)
                    ).json()
                    last = await client.get(
                        url,
                        params={"cases_limit": 1, "cases_cursor": first["results"]["next_cursor"]},
                        headers=_AUTH_HEADERS,
                    )
                    assert last.status_code == 200
                    assert last.json()["results"]["items"][0]["id"] == "zz-large-case"
                    downloaded = await client.get(
                        "/api/" + detail["run"]["download_url"], headers=_AUTH_HEADERS
                    )
                    assert downloaded.status_code == 200
                    from cayu.evals.reported import EvalReport

                    assert downloaded.json() == EvalReport.model_validate(payload).model_dump(
                        mode="json"
                    )
            finally:
                await store.close()

    asyncio.run(exercise())


def test_report_large_corpus_case_remains_readable(tmp_path, monkeypatch):
    from tests.evals.test_corpus_execution import _corpus

    from cayu.evals.execution import run_corpus_suite

    async def exercise():
        target = _target(_provider())
        _configure(monkeypatch, target.key)
        corpus = _corpus(target_key=target.key, trials=5)
        result = await run_corpus_suite(target, corpus, corpus.suites[0].id, max_concurrency=1)
        document = result.model_dump(mode="json")
        for trial in document["run"]["cases"][0]["trials"]:
            text = "x" * (16 << 10)
            trial["output"] = {
                "schema_version": 1,
                "text": text,
                "evidence_state": "complete",
                "preview_truncated": False,
                "retained_chars": len(text),
                "retained_bytes": len(text),
                "retained_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
        from cayu.evals.corpus import _content_revision

        document["run"]["revision"] = _content_revision(document["run"], "published run")
        document["revision"] = _content_revision(document, "execution result")
        payload = {
            "report_id": "large-corpus-report",
            "target_key": target.key,
            "program_key": "corpus-ci",
            "evidence": {"kind": "corpus", "result": document},
        }
        store = SQLiteEvalStore(tmp_path / "large-corpus.db")
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=_server(target, store)), base_url="http://test"
            ) as client:
                uploaded = await client.post(
                    "/api/evals/reports", json=payload, headers=_REPORT_HEADERS
                )
                assert uploaded.status_code == 200, uploaded.text
                receipt = uploaded.json()
                response = await client.get(
                    f"/api/evals/run-views/{receipt['run_key']}", headers=_AUTH_HEADERS
                )
                assert response.status_code == 200, response.text
                case = response.json()["results"]["items"][0]
                assert case["evidence_omitted"] and case["evidence"] is None
                downloaded = await client.get(
                    "/api/" + response.json()["run"]["download_url"], headers=_AUTH_HEADERS
                )
                assert downloaded.status_code == 200
                assert downloaded.json()["evidence"]["result"] == document
        finally:
            await store.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_http_report_rejects_nonfinite_structured_output(tmp_path, monkeypatch, literal):
    async def exercise():
        target = _target(_provider())
        _configure(monkeypatch, target.key)
        payload = await _payload(target.key)
        payload["evidence"]["run"]["cases"][0]["trials"][0]["structured_output"] = {
            "value": "NONFINITE_PLACEHOLDER"
        }
        store = SQLiteEvalStore(tmp_path / "http-finite.db")
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=_server(target, store)), base_url="http://test"
            ) as client:
                content = json.dumps(payload).replace('"NONFINITE_PLACEHOLDER"', literal)
                response = await client.post(
                    "/api/evals/reports",
                    content=content,
                    headers={**_REPORT_HEADERS, "Content-Type": "application/json"},
                )
                assert response.status_code == 422
                catalog = await client.get("/api/evals/suite-views", headers=_AUTH_HEADERS)
                assert catalog.json()["items"] == []
        finally:
            await store.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("kind", ["unicode", "unavailable", "large", "nodes", "structured"])
def test_report_public_read_boundaries(tmp_path, monkeypatch, kind):
    async def exercise():
        target = _target(_provider())
        _configure(monkeypatch, target.key)
        payload = await _payload(target.key)
        run = payload["evidence"]["run"]
        original = run["cases"][0]
        cases = []
        for index in range(2):
            case = copy.deepcopy(original)
            case["case_id"] = ("例" * 200 if kind == "unicode" else "case-") + str(index)
            if kind == "unavailable":
                case["status"] = "unavailable"
                case["score"] = None
                case["trials"][0].update(
                    status="unavailable",
                    score=None,
                    unavailable_reason="provider unavailable",
                    assertions=[],
                )
            if kind == "large":
                case["trials"][0]["assertions"] = [
                    {
                        "name": f"check-{i}",
                        "outcome": "passed",
                        "score": 1.0,
                        "message": "x" * (64 << 10),
                    }
                    for i in range(129)
                ]
            if kind == "nodes":
                case["trials"][0]["structured_output"] = [0] * 130_000
            if kind == "structured" and index == 0:
                case["trials"][0]["structured_output"] = "s" * (17 << 20)
            cases.append(case)
        run["cases"] = cases
        if kind == "unavailable":
            run.update(status="unavailable", score=None)
        store = SQLiteEvalStore(tmp_path / "read-boundaries.db")
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=_server(target, store)), base_url="http://test"
            ) as client:
                published = await client.post(
                    "/api/evals/reports", json=payload, headers=_REPORT_HEADERS
                )
                assert published.status_code == 200, published.text
                url = "/api/evals/run-views/" + published.json()["run_key"]
                params = {"cases_limit": 1}
                if kind == "unavailable":
                    params["case_filter"] = "failing"
                response = await client.get(url, params=params, headers=_AUTH_HEADERS)
                assert response.status_code == 200, response.text
                first = response.json()["results"]
                assert first["total"] == 2
                assert first["items"][0]["id"] == cases[0]["case_id"]
                cursor = first["next_cursor"]
                assert cursor and len(cursor) <= 1024
                params["cases_cursor"] = cursor
                response = await client.get(url, params=params, headers=_AUTH_HEADERS)
                assert response.status_code == 200, response.text
                assert response.json()["results"]["items"][0]["id"] == cases[1]["case_id"]
                assert response.json()["results"]["has_more"] is False
                params["case_filter"] = "flaky"
                assert (
                    await client.get(url, params=params, headers=_AUTH_HEADERS)
                ).status_code == 422
        finally:
            await store.close()

    asyncio.run(exercise())


def _trial_summary_payloads(base):
    def check(outcome, score, name="check", threshold=None):
        return {"name": name, "outcome": outcome, "score": score, "threshold": threshold}

    passed = [check("passed", 1.0)]
    failed = [check("failed", 0.0)]
    errored = [check("error", None)]
    unavailable = [check("unavailable", None)]
    rows = [
        ("passed", "passed", 1.0, passed, None, None, True),
        ("graded-failed", "failed", 0.25, [check("failed", 0.25, threshold=0.5)], None, None, True),
        (
            "mixed-failed",
            "failed",
            0.5,
            [*passed, check("failed", 0.0, "second")],
            None,
            None,
            True,
        ),
        ("error", "error", None, errored, "assertion failed", None, True),
        ("unavailable", "unavailable", None, unavailable, None, "input missing", True),
        ("workflow-error", "error", None, unavailable, "workflow capture failed", None, True),
        ("empty-error", "error", None, [], "execution failed", None, True),
        ("empty-unavailable", "unavailable", None, [], None, "capture unavailable", True),
        ("empty-skipped", "skipped", 0.0, [], None, None, True),
        ("false-pass", "passed", 1.0, failed, None, None, False),
        ("false-failure", "failed", 0.0, passed, None, None, False),
        ("wrong-score", "passed", 0.5, passed, None, None, False),
        ("empty-pass", "passed", 1.0, [], None, None, False),
        ("empty-failed", "failed", 0.0, [], None, None, False),
        ("empty-skipped-score", "skipped", 1.0, [], None, None, False),
        ("scored-error", "error", 1.0, errored, "error", None, False),
        ("scored-unavailable", "unavailable", 1.0, unavailable, None, "missing", False),
        ("hidden-error", "unavailable", None, errored, None, "missing", False),
        ("error-passed-checks", "error", None, passed, "error", None, False),
        ("missing-error", "error", None, errored, None, None, False),
        ("missing-reason", "unavailable", None, unavailable, None, None, False),
        ("passed-error", "passed", 1.0, passed, "contradiction", None, False),
    ]
    for name, status, score, checks, error, reason, valid in rows:
        payload = copy.deepcopy(base)
        payload["report_id"] = name
        run = payload["evidence"]["run"]
        case = run["cases"][0]
        trial = case["trials"][0]
        trial.update(
            status=status, score=score, assertions=checks, error=error, unavailable_reason=reason
        )
        case.update(status=status, score=score, error=error, unavailable_reason=reason)
        run.update(status=status, score=score)
        yield name, payload, valid

    from cayu.evals.trial_policy import EvalSuiteTrialPolicyV1

    for conflict in (False, True):
        payload = copy.deepcopy(base)
        payload["report_id"] = "case-contract-conflict" if conflict else "minimum-pass-policy"
        run = payload["evidence"]["run"]
        case = run["cases"][0]
        first = case["trials"][0]
        first.update(status="passed", score=1.0, assertions=passed)
        second = copy.deepcopy(first)
        second.update(trial_number=2)
        second["assertions"] = [check("passed", 1.0, "different")] if conflict else failed
        if not conflict:
            second.update(status="failed", score=0.0)
        case["trials"] = [first, second]
        case["trial_policy"] = EvalSuiteTrialPolicyV1.create(
            trial_count=2, minimum_passed_trials=1, max_concurrency=1
        ).model_dump(mode="json")
        case.update(status="passed", score=1.0 if conflict else 0.5)
        run.update(status=case["status"], score=case["score"])
        yield payload["report_id"], payload, not conflict
        if conflict:
            payload = copy.deepcopy(payload)
            payload["report_id"] = "case-contract-error"
            case = payload["evidence"]["run"]["cases"][0]
            case.update(status="error", score=None, error="Assertion contracts differ")
            payload["evidence"]["run"].update(status="error", score=None)
            yield payload["report_id"], payload, True


def test_http_report_trial_summary_matches_retained_checks_before_publication(
    tmp_path, monkeypatch
):
    async def exercise():
        target = _target(_provider())
        _configure(monkeypatch, target.key)
        base = await _payload(target.key)
        store = SQLiteEvalStore(tmp_path / "trial-summaries.db")
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=_server(target, store)), base_url="http://test"
            ) as client:
                variants = list(_trial_summary_payloads(base))
                for name, payload, valid in sorted(variants, key=lambda item: item[2]):
                    response = await client.post(
                        "/api/evals/reports", json=payload, headers=_REPORT_HEADERS
                    )
                    assert response.status_code == (200 if valid else 422), (name, response.text)
                    if not valid:
                        catalog = await client.get("/api/evals/suite-views", headers=_AUTH_HEADERS)
                        assert catalog.json()["items"] == [], name
                    else:
                        detail = await client.get(
                            "/api/evals/run-views/" + response.json()["run_key"],
                            params={"case_filter": "failing"},
                            headers=_AUTH_HEADERS,
                        )
                        assert detail.status_code == 200, detail.text
                        expected = payload["evidence"]["run"]["status"] in {
                            "failed",
                            "error",
                            "unavailable",
                        }
                        assert detail.json()["results"]["total"] == int(expected), name
        finally:
            await store.close()

    asyncio.run(exercise())
