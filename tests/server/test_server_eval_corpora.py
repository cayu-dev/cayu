from __future__ import annotations

import asyncio

import httpx
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi import HTTPException, Request
from fastapi.routing import APIRoute
from tests.server.test_server_evals import (
    _AUTH_HEADERS,
    _authenticate,
    _corpus,
    _evals_config,
    _provider,
    _server,
    _target,
)

from cayu.evals.corpus import EvalCaseSpec, EvalCorpusDocument, EvalSuiteSpec, eval_corpus_to_json
from cayu.evals.store import (
    EvalCorpusConflict,
    EvalStorePublicationRejected,
    EvalStoreResultTooLarge,
)
from cayu.server import DashboardConfig, ServerApiConfig, ServerConfig, create_server
from cayu.server import routes as routes_module
from cayu.storage.evals_sqlite import SQLiteEvalStore
from cayu.vaults import SecretRedactor


@pytest.mark.parametrize("prefix", ["/api", "/custom/v2"])
def test_corpus_routes_preserve_auth_bounds_catalog_pages_and_downloads(sqlite_resources, prefix):
    async def exercise():
        async with sqlite_resources as resources:
            provider = _provider()
            target = _target(provider)
            store = resources.own(SQLiteEvalStore(resources.path()))
            calls = []

            def authenticate(request: Request):
                calls.append(request.url.path)
                return _authenticate(request)

            server = create_server(
                target.app,
                config=ServerConfig.protected(
                    authenticate,
                    api=ServerApiConfig(path=prefix),
                    dashboard=DashboardConfig(enabled=False),
                    evals=_evals_config(target, store),
                ),
            )
            root = f"{prefix}/evals/corpora"
            paths = [
                root,
                root,
                f"{root}/{{corpus_revision}}",
                f"{root}/{{corpus_revision}}/download",
                f"{root}/{{corpus_revision}}/suites",
                f"{root}/{{corpus_revision}}/suites/{{suite_id}}/cases",
            ]
            routes = [route for route in server.routes if isinstance(route, APIRoute)]
            corpus_routes = [route for route in routes if route.path in paths]
            assert [route.path for route in corpus_routes] == paths
            dependency = (
                next(route for route in routes if route.path == f"{prefix}/sessions")
                .dependencies[0]
                .dependency
            )
            for route in corpus_routes:
                assert isinstance(route, routes_module._BoundedEvalsRoute)
                assert route.dependencies[0].dependency is dependency
                assert type(route).preparse_auth is authenticate

            template = _corpus()
            suites = tuple(
                EvalSuiteSpec.create(
                    id=f"suite-{index}",
                    name=f"Suite {index}",
                    trial_request=template.suites[0].trial_request,
                )
                for index in range(2)
            )
            case = template.cases[0]
            corpus = EvalCorpusDocument.create(
                target_key=template.target_key,
                evidence_policy=template.evidence_policy,
                suites=suites,
                cases=tuple(
                    EvalCaseSpec.create(
                        id=f"{suite.id}-case-{index}",
                        suite_id=suite.id,
                        name=f"Case {index}",
                        source=case.source,
                        input=case.input,
                        assertions=case.assertions,
                    )
                    for suite in suites
                    for index in range(2)
                ),
            )
            detail = f"{root}/{corpus.revision}"
            requests = [
                ("POST", root, corpus.model_dump(mode="json"), 201),
                ("GET", root, None, 200),
                ("GET", detail, None, 200),
                ("GET", f"{detail}/download", None, 200),
                ("GET", f"{detail}/suites", None, 200),
                ("GET", f"{detail}/suites/{suites[0].id}/cases", None, 200),
            ]
            async with (
                server.router.lifespan_context(server),
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=server), base_url="http://test"
                ) as client,
            ):
                for method, path, payload, status in requests:
                    calls.clear()
                    denied = await client.request(
                        method,
                        path,
                        content=b"private invalid input",
                        headers={"Content-Length": str(corpus_routes[0].max_request_bytes + 1)},
                    )
                    assert denied.status_code == 401
                    assert denied.json() == {"detail": "unauthorized"}
                    assert denied.headers["cache-control"] == "private, no-store"
                    assert calls == [path]
                    calls.clear()
                    response = await client.request(
                        method, path, headers=_AUTH_HEADERS, json=payload
                    )
                    assert response.status_code == status
                    assert response.headers["cache-control"] == "private, no-store"
                    assert calls == [path]
                    if path.endswith("/download"):
                        assert response.content == eval_corpus_to_json(corpus).encode("utf-8")
                        assert response.headers["content-disposition"] == (
                            f'attachment; filename="{corpus.target_key}-'
                            f'{corpus.revision[7:19]}.eval.json"'
                        )
                    elif path == detail:
                        assert response.json() == corpus.model_dump(mode="json")

                # A second corpus gives the catalog a real continuation cursor.
                imported = await client.post(
                    root, headers=_AUTH_HEADERS, json=template.model_dump(mode="json")
                )
                assert imported.status_code == 201
                for path, field, expected in (
                    (root, "revision", {corpus.revision, template.revision}),
                    (f"{detail}/suites", "id", {suite.id for suite in suites}),
                    (
                        f"{detail}/suites/{suites[0].id}/cases",
                        "id",
                        {item.id for item in corpus.cases if item.suite_id == suites[0].id},
                    ),
                ):
                    first = await client.get(path, headers=_AUTH_HEADERS, params={"limit": 1})
                    assert first.status_code == 200
                    page = first.json()
                    assert len(page["items"]) == 1
                    assert page["next_cursor"] is not None
                    second = await client.get(
                        path,
                        headers=_AUTH_HEADERS,
                        params={"limit": 1, "cursor": page["next_cursor"]},
                    )
                    assert second.status_code == 200
                    assert second.json()["next_cursor"] is None
                    assert {
                        item[field] for item in page["items"] + second.json()["items"]
                    } == expected
                    for params, error_detail in (
                        ({"limit": 0}, "Invalid Evals request."),
                        ({"max_result_bytes": 1023}, "Invalid Evals request."),
                        ({"cursor": "invalid"}, "Invalid Evals query."),
                    ):
                        invalid = await client.get(path, headers=_AUTH_HEADERS, params=params)
                        assert invalid.status_code == 422
                        assert invalid.json() == {"detail": error_detail}
                        assert invalid.headers["cache-control"] == "private, no-store"

                for path, detail_message in (
                    (f"{root}?target_key=unpublished", "Eval target not found."),
                    (f"{detail}/suites/missing/cases", "Eval suite not found."),
                ):
                    missing = await client.get(path, headers=_AUTH_HEADERS)
                    assert missing.status_code == 404
                    assert missing.json() == {"detail": detail_message}

                def deny_access():
                    raise HTTPException(status_code=403, detail="corpus access revoked")

                server.dependency_overrides[dependency] = deny_access
                for method, path, payload, _ in requests:
                    denied = await client.request(method, path, headers=_AUTH_HEADERS, json=payload)
                    assert denied.status_code == 403
                    assert denied.json() == {"detail": "corpus access revoked"}
                    assert denied.headers["cache-control"] == "private, no-store"
            assert provider.requests == []

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["invalid", "oversized", "hidden", "missing"])
def test_corpus_reads_preserve_private_errors_for_catalog_and_run_creation(
    sqlite_resources, monkeypatch, failure
):
    async def exercise():
        async with sqlite_resources as resources:
            provider = _provider()
            target = _target(provider)
            store = resources.own(SQLiteEvalStore(resources.path()))
            reads = []

            async def load(revision):
                reads.append(revision)
                if failure == "invalid":
                    raise ValueError("private storage input")
                if failure == "oversized":
                    raise EvalStoreResultTooLarge(1024)
                return _corpus(target_key="unpublished") if failure == "hidden" else None

            async def unexpected_admission(*args, **kwargs):
                pytest.fail("An unreadable corpus must not admit a run.")

            monkeypatch.setattr(store, "load_corpus", load)
            monkeypatch.setattr(store, "admit_run", unexpected_admission)
            server = _server(target, store)
            revision = "sha256:" + "0" * 64
            root = f"/api/evals/corpora/{revision}"
            expected = {
                "invalid": (422, "Invalid Evals query."),
                "oversized": (413, "Eval corpus exceeds the server byte limit."),
                "hidden": (404, "Eval corpus not found."),
                "missing": (404, "Eval corpus not found."),
            }[failure]
            async with (
                server.router.lifespan_context(server),
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=server), base_url="http://test"
                ) as client,
            ):
                for method, path, payload in (
                    ("GET", root, None),
                    ("GET", f"{root}/download", None),
                    ("GET", f"{root}/suites", None),
                    ("GET", f"{root}/suites/refund-regressions/cases", None),
                    (
                        "POST",
                        "/api/evals/runs",
                        {
                            "corpus_revision": revision,
                            "suite_id": "refund-regressions",
                            "expected_execution_profile_revision": revision,
                        },
                    ),
                ):
                    response = await client.request(
                        method,
                        path,
                        json=payload,
                        headers={**_AUTH_HEADERS, "Idempotency-Key": "corpus-private-read"},
                    )
                    assert response.status_code == expected[0]
                    assert response.json() == {"detail": expected[1]}
                    assert response.headers["cache-control"] == "private, no-store"
            assert reads == [revision] * 5
            assert provider.requests == []

    asyncio.run(exercise())


@pytest.mark.parametrize("catalog", ["corpora", "suites", "cases"])
@pytest.mark.parametrize("failure", ["invalid", "oversized"])
def test_corpus_catalogs_preserve_private_store_errors(
    sqlite_resources, monkeypatch, catalog, failure
):
    async def exercise():
        async with sqlite_resources as resources:
            provider = _provider()
            target = _target(provider)
            store = resources.own(SQLiteEvalStore(resources.path()))
            corpus = _corpus()
            await store.save_corpus(corpus, redact_json=target.app.redact_json)
            queries = []

            async def list_page(query):
                queries.append(query)
                if failure == "oversized":
                    raise EvalStoreResultTooLarge(2048)
                raise ValueError("private catalog details")

            monkeypatch.setattr(store, f"list_{catalog}", list_page)
            root = "/api/evals/corpora"
            path, detail = {
                "corpora": (root, "Eval catalog page exceeds the requested byte limit."),
                "suites": (
                    f"{root}/{corpus.revision}/suites",
                    "Eval suite page exceeds the requested byte limit.",
                ),
                "cases": (
                    f"{root}/{corpus.revision}/suites/{corpus.suites[0].id}/cases",
                    "Eval case page exceeds the requested byte limit.",
                ),
            }[catalog]
            server = _server(target, store)
            async with (
                server.router.lifespan_context(server),
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=server), base_url="http://test"
                ) as client,
            ):
                response = await client.get(
                    path, headers=_AUTH_HEADERS, params={"limit": 2, "max_result_bytes": 2048}
                )
                assert response.status_code == (413 if failure == "oversized" else 422)
                assert response.json() == {
                    "detail": detail if failure == "oversized" else "Invalid Evals query."
                }
                assert response.headers["cache-control"] == "private, no-store"
            assert len(queries) == 1
            assert queries[0].limit == 2
            assert queries[0].max_result_bytes == 2048
            assert provider.requests == []

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["conflict", "unsafe", "oversized"])
def test_corpus_import_preserves_publication_errors_and_redactor(
    sqlite_resources, monkeypatch, failure
):
    async def exercise():
        async with sqlite_resources as resources:
            provider = _provider()
            target = _target(provider)
            store = resources.own(SQLiteEvalStore(resources.path()))
            corpus = _corpus()
            writes = []

            async def save(value, *, redact_json):
                assert value == corpus
                assert redact_json == target.app.redact_json
                writes.append(value.revision)
                if failure == "oversized":
                    raise EvalStoreResultTooLarge(1024)
                error = (
                    EvalCorpusConflict if failure == "conflict" else EvalStorePublicationRejected
                )
                raise error("private publication details")

            monkeypatch.setattr(store, "save_corpus", save)
            server = _server(target, store)
            expected = {
                "conflict": (409, "Eval corpus revision conflicts with stored content."),
                "unsafe": (422, "Eval corpus contains unsafe public data."),
                "oversized": (413, "Eval corpus exceeds the server byte limit."),
            }[failure]
            async with (
                server.router.lifespan_context(server),
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=server), base_url="http://test"
                ) as client,
            ):
                response = await client.post(
                    "/api/evals/corpora", headers=_AUTH_HEADERS, json=corpus.model_dump(mode="json")
                )
                assert response.status_code == expected[0]
                assert response.json() == {"detail": expected[1]}
                assert response.headers["cache-control"] == "private, no-store"
            assert writes == [corpus.revision]
            assert provider.requests == []

    asyncio.run(exercise())


def test_corpus_import_rejects_workload_secrets_without_persistence(sqlite_resources):
    async def exercise():
        async with sqlite_resources as resources:
            provider = _provider()
            secret = "corpus-fixture-private-credential"
            target = _target(provider, secret_redactor=SecretRedactor(secret))
            store = resources.own(SQLiteEvalStore(resources.path()))
            corpus = _corpus(input_text=f"Refund order using {secret}.")
            server = _server(target, store)
            async with (
                server.router.lifespan_context(server),
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=server), base_url="http://test"
                ) as client,
            ):
                response = await client.post(
                    "/api/evals/corpora", headers=_AUTH_HEADERS, json=corpus.model_dump(mode="json")
                )
                assert response.status_code == 422
                assert response.json() == {"detail": "Eval corpus contains unsafe public data."}
                assert response.headers["cache-control"] == "private, no-store"
            assert await store.load_corpus(corpus.revision) is None
            assert provider.requests == []

    asyncio.run(exercise())
