"""Filtered session pages preserve the native cursor and shared state evidence."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from tests.core.session_store_conformance.support import (
    _close_store,
    _open_store,
    _reopen_store,
    conformance_postgres_dsn,
    session_store_case,
)
from tests.core.test_cloud_reporting import _app, _drain, _start

from cayu import Message
from cayu.server import ServerConfig, create_server
from cayu.sessions import ForkSessionRequest, RunRequest
from cayu.sessions._run_identity import eval_trial_scope

__all__ = ["conformance_postgres_dsn", "session_store_case"]


def server(app):
    return create_server(app, config=ServerConfig.local_development())


def client_for(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=server(app)), base_url="http://test")


@pytest.mark.store_semantics
def test_filtered_pages_continue_through_empty_windows_after_reopen(session_store_case):
    async def run():
        store = await _open_store(session_store_case)
        try:
            app = _app(store)
            for name, specialist in [
                ("one", "writer"),
                ("two", "asker"),
                ("three", "writer"),
                ("four", "approver"),
            ]:
                await _start(app, name, specialist)
            params = [
                ("include", "overview"),
                ("overview_state", "waiting_input"),
                ("overview_state", "waiting_approval"),
                ("include_eval", "false"),
                ("include_forks", "false"),
                ("limit", "1"),
                ("order_by", "created_at_asc"),
            ]
            cursor = None
            pages = []
            for index in range(4):
                if index == 2:
                    store = await _reopen_store(session_store_case, store)
                    app = _app(store)
                async with client_for(app) as client:
                    response = await client.get(
                        "/api/sessions",
                        params=params + ([] if cursor is None else [("cursor", cursor)]),
                    )
                assert response.status_code == 200, response.text
                body = response.json()
                assert body["total_count"] is None
                assert [item["id"] for item in body["sessions"]] == [
                    item["session_id"] for item in body["overviews"]
                ]
                pages.append([item["id"] for item in body["sessions"]])
                cursor = body["next_cursor"]
                assert (cursor is not None) == (index < 3)
            assert pages == [[], ["two"], [], ["four"]]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_filtered_cursor_binds_every_query_field_and_normalizes_state_order():
    async def run():
        app = _app()
        await _start(app, "one", "writer")
        await _start(app, "two", "asker")
        base = {
            "include": "overview",
            "overview_state": ["completed", "waiting_input"],
            "include_eval": "false",
            "include_forks": "false",
            "limit": "1",
            "order_by": "created_at_asc",
        }
        async with client_for(app) as client:
            first = await client.get("/api/sessions", params=base)
            assert first.status_code == 200
            cursor = first.json()["next_cursor"]
            assert cursor
            assert first.json()["total_count"] is None
            changes = {
                "q": "writer",
                "status": "completed",
                "debug_state": "tool_issue",
                "agent_name": "writer",
                "provider_name": "other",
                "model": "other",
                "environment_name": "other",
                "parent_session_id": "one",
                "causal_budget_id": "one",
                "order_by": "updated_at_desc",
                "limit": "2",
                "label": "tenant=acme",
                "label_selector": "!archived",
                "overview_state": "completed",
                "include_eval": "true",
                "include_forks": "true",
                "include": ["overview", "usage"],
            }
            for key, value in changes.items():
                response = await client.get(
                    "/api/sessions", params={**base, "cursor": cursor, key: value}
                )
                assert response.status_code == 422, (key, response.text)
            reordered = await client.get(
                "/api/sessions",
                params={
                    **base,
                    "cursor": cursor,
                    "overview_state": ["waiting_input", "completed", "completed"],
                },
            )
            assert reordered.status_code == 200, reordered.text
            assert [row["id"] for row in reordered.json()["sessions"]] == ["two"]
            unfiltered = await client.get("/api/sessions", params={"cursor": cursor})
            assert unfiltered.status_code == 422
            native = await client.get("/api/sessions", params={"limit": 1})
            assert native.json()["total_count"] == 2
            wrong_mode = await client.get(
                "/api/sessions", params={**base, "cursor": native.json()["next_cursor"]}
            )
            assert wrong_mode.status_code == 422
        await app.aclose()

    asyncio.run(run())


def test_overview_filter_validation_is_bounded_and_requires_overview():
    async def run():
        app = _app()
        async with client_for(app) as client:
            for params in [
                {"overview_state": "completed"},
                {"include_eval": "false"},
                {"include_forks": "false"},
                {"include": "overview", "overview_state": "completed", "offset": "1"},
                {"include": "overview", "include_eval": "false", "ids": "one"},
                {"include": "overview", "overview_state": "future_state"},
            ]:
                response = await client.get("/api/sessions", params=params)
                assert response.status_code == 422, response.text
            for cursor in ["CURSOR_SECRET_CANARY", "CURSOR_SECRET_CANARY" + "x" * 9000]:
                response = await client.get(
                    "/api/sessions",
                    params={"include": "overview", "include_eval": "false", "cursor": cursor},
                )
                assert response.status_code == 422
                assert "CURSOR_SECRET_CANARY" not in response.text
        await app.aclose()

    asyncio.run(run())


def test_eval_and_fork_filters_use_runtime_evidence_and_keep_ordinary_children():
    async def run():
        app = _app()
        await _start(app, "root", "writer")
        async with eval_trial_scope():
            await _start(app, "trial", "writer")
        await _drain(
            app.fork_session(ForkSessionRequest(source_session_id="root", session_id="fork"))
        )
        for name, parent in [("child", "root"), ("fork-child", "fork")]:
            await _drain(
                app.run(
                    RunRequest(
                        session_id=name,
                        parent_session_id=parent,
                        agent_name="writer",
                        messages=[Message.text("user", "go")],
                    )
                )
            )
        async with client_for(app) as client:
            for evals, forks, expected in [
                (True, True, {"root", "trial", "fork", "child", "fork-child"}),
                (False, True, {"root", "fork", "child", "fork-child"}),
                (True, False, {"root", "trial", "child"}),
                (False, False, {"root", "child"}),
            ]:
                response = await client.get(
                    "/api/sessions",
                    params={
                        "include": "overview",
                        "include_eval": str(evals).lower(),
                        "include_forks": str(forks).lower(),
                    },
                )
                assert response.status_code == 200, response.text
                assert {row["id"] for row in response.json()["sessions"]} == expected
        await app.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("opaque", ['"' * 4096, "界" * 1365], ids=["escaped", "unicode"])
def test_filtered_cursor_preserves_maximum_opaque_store_cursor(monkeypatch, opaque):
    async def run():
        app = _app()
        original = app.session_store.list_sessions
        seen = []

        async def opaque_page(query=None, **kwargs):
            seen.append(query.cursor)
            page = await original(query.model_copy(update={"cursor": None}), **kwargs)
            return page.model_copy(update={"next_cursor": opaque if query.cursor is None else None})

        monkeypatch.setattr(app.session_store, "list_sessions", opaque_page)
        async with client_for(app) as client:
            params = {"include": "overview", "include_eval": "false"}
            first = await client.get("/api/sessions", params=params)
            assert first.status_code == 200, first.text
            second = await client.get(
                "/api/sessions", params={**params, "cursor": first.json()["next_cursor"]}
            )
            assert second.status_code == 200, second.text
            assert second.json()["next_cursor"] is None
            assert seen == [None, opaque]
        await app.aclose()

    asyncio.run(run())


def test_deeply_nested_cursor_is_rejected_without_parser_failure():
    import base64

    async def run():
        app = _app()
        cursor = "sessionquery_" + base64.urlsafe_b64encode(b"[" * 1100 + b"]" * 1100).decode()
        async with client_for(app) as client:
            response = await client.get(
                "/api/sessions",
                params={"include": "overview", "include_eval": "false", "cursor": cursor},
            )
            assert response.status_code == 422
        await app.aclose()

    asyncio.run(run())


def test_filtered_candidate_window_is_bounded_and_skips_excluded_enrichment(monkeypatch):
    async def run():
        app = _app()
        await _start(app, "excluded", "writer")
        await _start(app, "included", "asker")
        store = app.session_store
        original_list = store.list_sessions
        original_operation = store.load_session_operation
        windows = []

        async def counted_list(query=None, **kwargs):
            if query.parent_session_id is None:
                windows.append(query)
            return await original_list(query, **kwargs)

        async def forbidden_enrichment(session_id, operation_id):
            assert session_id != "excluded", "Excluded rows must not require browser enrichment"
            return await original_operation(session_id, operation_id)

        monkeypatch.setattr(store, "list_sessions", counted_list)
        monkeypatch.setattr(store, "load_session_operation", forbidden_enrichment)
        async with client_for(app) as client:
            params = {
                "include": ["overview", "usage", "execution_profile"],
                "overview_state": "waiting_input",
                "order_by": "created_at_asc",
                "limit": 1,
            }
            first = await client.get("/api/sessions", params=params)
            assert first.status_code == 200, first.text
            assert first.json()["sessions"] == []
            assert first.json()["overviews"] == []
            assert first.json()["usage"] == []
            assert first.json()["execution_profiles"] == []
            assert len(windows) == 1
            assert windows[0].limit == 1
            assert not windows[0].include_total_count
            second = await client.get(
                "/api/sessions", params={**params, "cursor": first.json()["next_cursor"]}
            )
            assert second.status_code == 200, second.text
            assert [row["id"] for row in second.json()["sessions"]] == ["included"]
            assert len(second.json()["overviews"]) == len(second.json()["usage"]) == 1
            assert len(second.json()["execution_profiles"]) == 1
            assert len(windows) == 2
            assert second.json()["next_cursor"] is None
        await app.aclose()

    asyncio.run(run())


def test_incomplete_pending_evidence_cannot_be_filtered_into_success(monkeypatch):
    from cayu.sessions.pending_action_contracts import PendingActionIssue, PendingActionSession

    async def run():
        app = _app()
        await _start(app, "completed", "writer")
        store = app.session_store
        original = store.query_pending_actions

        async def incomplete(query=None, **kwargs):
            page = await original(query, **kwargs)
            session = await store.load(query.session_id)
            issue = PendingActionIssue.source_invalid(PendingActionSession.from_session(session))
            return page.model_copy(update={"issues": [issue]})

        monkeypatch.setattr(store, "query_pending_actions", incomplete)
        async with client_for(app) as client:
            response = await client.get(
                "/api/sessions",
                params={"include": "overview", "overview_state": "waiting_input"},
            )
            assert response.status_code == 503, response.text
            assert "sessions" not in response.json()
        await app.aclose()

    asyncio.run(run())


def test_filtered_http_read_cancellation_propagates_without_retry(monkeypatch):
    async def run():
        app = _app()
        await _start(app, "one", "writer")
        entered = asyncio.Event()
        released = asyncio.Event()
        calls = 0

        async def blocked_read(query=None, **kwargs):
            nonlocal calls
            calls += 1
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                released.set()

        monkeypatch.setattr(app.session_store, "list_sessions", blocked_read)
        async with client_for(app) as client:
            task = asyncio.create_task(
                client.get("/api/sessions", params={"include": "overview", "include_eval": "false"})
            )
            await asyncio.wait_for(entered.wait(), 3)
            task.cancel()
            assert task.cancelling() == 1
            with pytest.raises(asyncio.CancelledError):
                await task
            assert task.cancelled()
            assert released.is_set()
            assert calls == 1
            assert (await app.session_store.load("one")).status.value == "completed"
        await app.aclose()

    asyncio.run(run())


def test_http_title_search_intersects_raw_status_and_overview_state():
    async def run():
        app = _app()
        for name, specialist in [("finished", "writer"), ("waiting", "asker")]:
            await _drain(
                app.run(
                    RunRequest(
                        session_id=name,
                        agent_name=specialist,
                        title="Quarterly analysis",
                        messages=[Message.text("user", "go")],
                    )
                )
            )
        async with client_for(app) as client:
            params = {
                "include": "overview",
                "q": "QUARTERLY",
                "overview_state": "waiting_input",
            }
            response = await client.get("/api/sessions", params=params)
            assert response.status_code == 200, response.text
            assert [row["id"] for row in response.json()["sessions"]] == ["waiting"]
            assert response.json()["overviews"][0]["title"] == "Quarterly analysis"
            incompatible = await client.get("/api/sessions", params={**params, "status": "failed"})
            assert incompatible.status_code == 200
            assert incompatible.json()["sessions"] == []
            assert incompatible.json()["total_count"] is None
        await app.aclose()

    asyncio.run(run())


def test_unfiltered_cursor_still_uses_native_query_validation():
    async def run():
        app = _app()
        await _start(app, "one", "writer")
        await _start(app, "two", "writer")
        async with client_for(app) as client:
            first = await client.get("/api/sessions", params={"limit": 1})
            cursor = first.json()["next_cursor"]
            for params in [{"cursor": cursor, "offset": 1}, {"cursor": "x" * 4097}]:
                response = await client.get("/api/sessions", params=params)
                assert response.status_code == 422, response.text
        await app.aclose()

    asyncio.run(run())


def test_cursor_envelope_rejects_invalid_shapes_before_store_read(monkeypatch, caplog, capsys):
    import base64
    import json

    async def run():
        app = _app()
        await _start(app, "one", "writer")
        await _start(app, "two", "writer")
        params = {"include": "overview", "include_eval": "false", "limit": 1}
        async with client_for(app) as client:
            first = await client.get("/api/sessions", params=params)
            cursor = first.json()["next_cursor"]
            document = json.loads(base64.urlsafe_b64decode(cursor.removeprefix("sessionquery_")))

            async def forbidden_read(*args, **kwargs):
                raise AssertionError("Invalid cursor must be rejected before native list dispatch")

            monkeypatch.setattr(app.session_store, "list_sessions", forbidden_read)
            caplog.clear()
            capsys.readouterr()
            malformed = [
                {**document, "version": True},
                {**document, "version": 2},
                {**document, "after": {"secret": "CURSOR_SECRET_CANARY"}},
                {**document, "after": "CURSOR_SECRET_CANARY" + "x" * 4096},
                {**document, "after": "CURSOR_SECRET_CANARY\ud800"},
                {**document, "unexpected": "CURSOR_SECRET_CANARY"},
                [document],
            ]
            encoded = [
                "sessionquery_" + base64.urlsafe_b64encode(json.dumps(item).encode()).decode()
                for item in malformed
            ]
            encoded.append("sessionquery_" + "A" * 34_000)
            for token in encoded:
                response = await client.get("/api/sessions", params={**params, "cursor": token})
                assert response.status_code == 422, response.text
                assert "CURSOR_SECRET_CANARY" not in response.text
            captured = capsys.readouterr()
            assert "CURSOR_SECRET_CANARY" not in caplog.text + captured.out + captured.err
        await app.aclose()

    asyncio.run(run())


@pytest.mark.store_semantics
def test_nested_native_cursor_rejected_in_filtered_and_unfiltered_http(
    session_store_case, caplog, capsys
):
    import base64
    import json
    import warnings

    async def run():
        store = await _open_store(session_store_case)
        try:
            app = _app(store)
            await _start(app, "one", "writer")
            await _start(app, "two", "writer")
            params = {"include": "overview", "include_eval": "false", "limit": 1}
            inner = base64.urlsafe_b64encode(
                b"[" * 1100 + b'"CURSOR_SECRET_CANARY"' + b"]" * 1100
            ).decode()
            async with client_for(app) as client:
                first = await client.get("/api/sessions", params=params)
                assert first.status_code == 200
                document = json.loads(
                    base64.urlsafe_b64decode(
                        first.json()["next_cursor"].removeprefix("sessionquery_")
                    )
                )
                document["after"] = inner
                outer = (
                    "sessionquery_"
                    + base64.urlsafe_b64encode(json.dumps(document).encode()).decode()
                )
                caplog.clear()
                capsys.readouterr()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    for query in [{**params, "cursor": outer}, {"cursor": inner}]:
                        response = await client.get("/api/sessions", params=query)
                        assert response.status_code == 422, response.text
                        assert "CURSOR_SECRET_CANARY" not in response.text
                captured = capsys.readouterr()
                diagnostic = caplog.text + captured.out + captured.err
                diagnostic += "".join(str(item.message) for item in caught)
                assert "CURSOR_SECRET_CANARY" not in diagnostic
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.store_semantics
def test_public_title_search_uses_current_redaction_before_membership_and_pagination(
    session_store_case,
):
    import base64
    import json

    from cayu.vaults.redaction import SecretRedactor

    async def run():
        store = await _open_store(session_store_case)
        secret = "SEARCH_CANARY_hidden_segment_79"
        try:
            # Store under an earlier policy, then read with a newly configured secret.
            app = _app(store)
            for name, title in [("one", f"Audit {secret}"), ("two", "Ordinary invoice")]:
                await _drain(
                    app.run(
                        RunRequest(
                            session_id=name,
                            agent_name="writer",
                            title=title,
                            messages=[Message.text("user", "go")],
                        )
                    )
                )
            assert (await store.load("one")).metadata["cayu:title"] == f"Audit {secret}"
            store = await _reopen_store(session_store_case, store)
            app = _app(store, secret_redactor=SecretRedactor([secret]))
            async with client_for(app) as client:
                for method, path, extra in [
                    ("GET", "/api/sessions", {}),
                    ("GET", "/api/sessions", {"include": "overview"}),
                    ("POST", "/api/sessions/summary", {}),
                ]:
                    params = {"limit": 1, "order_by": "created_at_asc", **extra}
                    positions = []
                    for guess in ["hidden_segment", "not_in_any_title"]:
                        first = await client.request(method, path, params={**params, "q": guess})
                        assert first.status_code == 200, first.text
                        assert first.json()["sessions"] == []
                        assert first.json()["total_count"] is None
                        cursor = first.json()["next_cursor"]
                        assert cursor
                        document = json.loads(
                            base64.urlsafe_b64decode(cursor.removeprefix("sessionquery_"))
                        )
                        positions.append(document["after"])
                        second = await client.request(
                            method, path, params={**params, "q": guess, "cursor": cursor}
                        )
                        assert second.status_code == 200, second.text
                        assert second.json()["sessions"] == []
                        assert second.json()["total_count"] is None
                        assert second.json()["next_cursor"] is None
                        assert secret not in first.text + second.text
                    assert positions[0] == positions[1]
                    # Search the visible prefix of a redacted title and a normal title.
                    for query, expected in [("Audit", "one"), ("invoice", "two")]:
                        response = await client.request(
                            method, path, params={**params, "q": query, "limit": 2}
                        )
                        assert response.status_code == 200, response.text
                        rows = response.json()["sessions"]
                        ids = [
                            row["id"] if method == "GET" else row["session"]["id"] for row in rows
                        ]
                        assert ids == [expected]
                        assert secret not in response.text
                    conflict = await client.request(
                        method, path, params={**params, "q": "Audit", "offset": 1}
                    )
                    assert conflict.status_code == 422
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_public_text_query_binds_cursor_and_scans_one_native_page(monkeypatch):
    async def run():
        app = _app()
        for name in ["one", "two", "three"]:
            await _start(app, name, "writer")
        original = app.session_store.list_sessions
        queries = []

        async def observed(query=None, **kwargs):
            queries.append(query)
            return await original(query, **kwargs)

        monkeypatch.setattr(app.session_store, "list_sessions", observed)
        async with client_for(app) as client:
            params = {"q": "writer", "limit": 1, "order_by": "created_at_asc"}
            first = await client.get("/api/sessions", params=params)
            assert first.status_code == 200, first.text
            assert len(queries) == 1
            assert queries[0].q is None
            assert queries[0].limit == 1
            assert not queries[0].include_total_count
            assert first.json()["total_count"] is None
            cursor = first.json()["next_cursor"]
            for update in [{"q": "three"}, {"limit": 2}, {"status": "completed"}]:
                changed = await client.get(
                    "/api/sessions", params={**params, "cursor": cursor, **update}
                )
                assert changed.status_code == 422
            crossed = await client.post(
                "/api/sessions/summary", params={**params, "cursor": cursor}
            )
            assert crossed.status_code == 422
            assert len(queries) == 1
            second = await client.get("/api/sessions", params={**params, "cursor": cursor})
            assert second.status_code == 200
            assert [row["id"] for row in second.json()["sessions"]] == ["two"]
            assert len(queries) == 2
        await app.aclose()

    asyncio.run(run())


def test_public_search_uses_valid_display_title_and_redacted_labels():
    from cayu.vaults.redaction import SecretRedactor

    async def run():
        app = _app()
        await _drain(
            app.run(
                RunRequest(
                    session_id="one",
                    agent_name="writer",
                    title="Visible prefix " + "x" * 236 + "short",
                    labels={"customer": "label-secret-value"},
                    messages=[Message.text("user", "go")],
                )
            )
        )
        secured = _app(
            app.session_store, secret_redactor=SecretRedactor(["short", "label-secret-value"])
        )
        async with client_for(secured) as client:
            display = await client.get("/api/sessions", params={"include": "overview"})
            assert display.status_code == 200
            # Redaction expansion exceeds the current display-title bound.
            assert display.json()["overviews"][0]["title"] is None
            for query in ["Visible prefix", "secret-value"]:
                response = await client.get("/api/sessions", params={"q": query})
                assert response.status_code == 200
                assert response.json()["sessions"] == []
                assert response.json()["total_count"] is None
            visible_label_key = await client.get("/api/sessions", params={"q": "customer"})
            assert [row["id"] for row in visible_label_key.json()["sessions"]] == ["one"]
        await secured.aclose()

    asyncio.run(run())
