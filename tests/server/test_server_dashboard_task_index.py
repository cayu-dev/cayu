"""Task relations and cancellation through the public control plane."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")
from fastapi.testclient import TestClient

from cayu import CayuApp
from cayu.server import ServerConfig, create_server
from cayu.tasks import InMemoryTaskStore, TaskCreate
from cayu.tasks.graphs import TaskGraphCreate, TaskGraphNode


@pytest.fixture
def sqlite_task_store():
    from cayu.storage import SQLiteTaskStore

    stores = []

    def create(path):
        store = SQLiteTaskStore(path)
        stores.append(store)
        return store

    yield create
    for store in stores:
        asyncio.run(store.close())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_task_relations_are_public_on_list_and_detail(backend, tmp_path, sqlite_task_store):
    if backend == "sqlite":
        store = sqlite_task_store(tmp_path / "tasks.sqlite")
    else:
        store = InMemoryTaskStore()
    asyncio.run(
        store.create_task_graph(
            TaskGraphCreate(
                graph_id="closing",
                nodes=(
                    TaskGraphNode(task=TaskCreate(task_id="first", type="report")),
                    TaskGraphNode(
                        task=TaskCreate(task_id="second", type="report"),
                        prerequisite_task_ids=("first",),
                    ),
                ),
            )
        )
    )
    with TestClient(
        create_server(
            CayuApp(task_store=store, enable_logging=False), config=ServerConfig.local_development()
        )
    ) as client:
        rows = client.get("/api/tasks").json()
        second = next(row for row in rows if row["id"] == "second")
        assert second.get("graph_id") == "closing"
        assert second.get("prerequisite_task_ids") == ["first"]
        detail = client.get("/api/tasks/second").json()
        assert detail["graph_id"] == "closing"
        assert detail["prerequisite_task_ids"] == ["first"]


@pytest.mark.parametrize("operation", ["schedule", "cancel"])
def test_general_task_cancel_is_full_access_and_member_can_read_schedule(operation):
    pytest.importorskip("jwt")
    from cryptography.hazmat.primitives.asymmetric import ec
    from tests.server.test_proxy_identity import FakeJwks, _as, _auth, _token

    key = ec.generate_private_key(ec.SECP256R1())
    store = InMemoryTaskStore()
    asyncio.run(store.create_task(TaskCreate(task_id="pending", type="report")))
    app = CayuApp(task_store=store, enable_logging=False)
    with TestClient(
        create_server(app, config=ServerConfig.protected(_auth(FakeJwks(key))))
    ) as client:
        member = _as(_token(key, role="member"))
        admin = _as(_token(key, role="admin"))
        if operation == "schedule":
            history = client.get("/api/tasks/pending/schedule/events", headers=member)
            assert history.status_code == 200, history.text
            return
        assert client.post("/api/tasks/pending/cancel", headers=member).status_code == 403
        assert asyncio.run(store.load_task("pending")).status.value == "pending"
        cancelled = client.post("/api/tasks/pending/cancel", headers=admin)
        assert cancelled.status_code == 200, cancelled.text
        assert cancelled.json()["status"] == "cancelled"
        assert asyncio.run(store.load_task("pending")).status.value == "cancelled"


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_task_index_counts_are_complete_and_search_matches_only_public_text(
    backend, tmp_path, sqlite_task_store
):
    from cayu.vaults import SecretRedactor

    if backend == "sqlite":
        store = sqlite_task_store(tmp_path / "index.sqlite")
    else:
        store = InMemoryTaskStore()

    async def seed():
        for index in range(1003):
            await store.create_task(
                TaskCreate(task_id=f"visible-{index:04}", type="report", title="Visible report")
            )
        await store.create_task(TaskCreate(task_id="internal", type="cayu.housekeeping"))
        await store.create_task(
            TaskCreate(task_id="secret", type="report", title="private-search-canary")
        )
        await store.cancel_task("visible-0000")

    asyncio.run(seed())
    app = CayuApp(
        task_store=store,
        secret_redactor=SecretRedactor("private-search-canary"),
        enable_logging=False,
    )
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        response = client.get(
            "/api/tasks", params={"include": "counts", "include_internal": "false", "limit": 2}
        )
        assert response.status_code == 200, response.text
        page = response.json()
        assert isinstance(page, dict), "counts require an explicit page envelope"
        assert len(page["items"]) == 2
        assert page["total_count"] == "1004"
        assert page["counts_by_status"]["pending"] == "1003"
        assert page["counts_by_status"]["cancelled"] == "1"
        assert page["next_cursor"]
        hidden = client.get(
            "/api/tasks", params={"include": "counts", "q": "private-search-canary"}
        ).json()
        assert hidden["total_count"] == "0"
        assert hidden["items"] == []
        visible = client.get(
            "/api/tasks", params={"include": "counts", "q": "Visible report", "status": "cancelled"}
        ).json()
        assert visible["total_count"] == "1"
        assert visible["counts_by_status"]["pending"] == "1002"
        assert visible["counts_by_status"]["cancelled"] == "1"
        assert [item["id"] for item in visible["items"]] == ["visible-0000"]
        wrong_filter = client.get(
            "/api/tasks",
            params={"include": "counts", "include_internal": "true", "cursor": page["next_cursor"]},
        )
        assert wrong_filter.status_code == 422


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_task_dependents_page_and_keyset_survive_new_rows_before_previous_page(
    backend, tmp_path, sqlite_task_store
):
    if backend == "sqlite":
        store = sqlite_task_store(tmp_path / "relations.sqlite")
    else:
        store = InMemoryTaskStore()

    async def seed():
        await store.create_task_graph(
            TaskGraphCreate(
                graph_id="graph",
                nodes=(
                    TaskGraphNode(task=TaskCreate(task_id="root", type="report")),
                    TaskGraphNode(
                        task=TaskCreate(task_id="child-a", type="report"),
                        prerequisite_task_ids=("root",),
                    ),
                    TaskGraphNode(
                        task=TaskCreate(task_id="child-b", type="report"),
                        prerequisite_task_ids=("root",),
                    ),
                ),
            )
        )
        await store.create_task(TaskCreate(task_id="unrelated", type="report"))

    asyncio.run(seed())
    with TestClient(
        create_server(
            CayuApp(task_store=store, enable_logging=False), config=ServerConfig.local_development()
        )
    ) as client:
        first = client.get("/api/tasks/root/dependents", params={"limit": 1})
        assert first.status_code == 200, first.text
        page = first.json()
        assert page["total_count"] == "2"
        assert page["items"][0]["id"] == "child-a"
        second = client.get(
            "/api/tasks/root/dependents", params={"limit": 1, "cursor": page["next_cursor"]}
        ).json()
        assert [row["id"] for row in second["items"]] == ["child-b"]
        assert second["next_cursor"] is None
        assert (
            client.get(
                "/api/tasks/unrelated/dependents", params={"cursor": page["next_cursor"]}
            ).status_code
            == 422
        )
        assert client.get("/api/tasks/missing/dependents").status_code == 404
        # New records before the previous page must not shift its continuation.
        listed = client.get(
            "/api/tasks", params={"include": "counts", "order_by": "created_at_desc", "limit": 1}
        ).json()
        assert listed["items"][0]["id"] == "unrelated"
        expected_next = client.get(
            "/api/tasks", params={"order_by": "created_at_desc", "limit": 2}
        ).json()[1]["id"]
        asyncio.run(store.create_task(TaskCreate(task_id="newer", type="report")))
        continued = client.get(
            "/api/tasks",
            params={
                "include": "counts",
                "order_by": "created_at_desc",
                "limit": 1,
                "cursor": listed["next_cursor"],
            },
        ).json()
        assert continued["items"][0]["id"] == expected_next


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_http_cancel_retains_live_worker_ownership_until_handler_drains(
    backend, tmp_path, sqlite_task_store
):
    import httpx

    from cayu.tasks.worker import run_task_worker

    async def scenario():
        if backend == "sqlite":
            store = sqlite_task_store(tmp_path / "worker.sqlite")
        else:
            store = InMemoryTaskStore()
        await store.create_task(TaskCreate(task_id="owned", type="report"))
        app = CayuApp(task_store=store, enable_logging=False)
        entered, release = asyncio.Event(), asyncio.Event()

        async def handler(_app, _task, _worker):
            entered.set()
            await release.wait()

        worker = asyncio.create_task(
            run_task_worker(
                app, store, handler, worker_id="live-owner", poll_interval_s=0.01, max_tasks=1
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 5)
            server = create_server(app, config=ServerConfig.local_development())
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                response = await client.post("/api/tasks/owned/cancel")
                assert response.status_code == 200, response.text
                requested = await store.load_task("owned")
                assert requested.worker_id == "live-owner"
                assert requested.status.value in {"claimed", "running"}
                assert requested.status_reason == "cancellation_requested"
                assert await store.claim_task("competitor") is None
                assert not worker.done()
                release.set()
                await asyncio.wait_for(worker, 5)
                settled = await store.load_task("owned")
                assert settled.status.value == "cancelled"
                assert settled.worker_id is None and settled.lease_expires_at is None
        finally:
            release.set()
            if not worker.done():
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)

    asyncio.run(scenario())


def test_sqlite_http_index_cancellation_keeps_connection_fenced(
    monkeypatch, tmp_path, sqlite_task_store
):
    import threading

    import httpx

    async def scenario():
        store = sqlite_task_store(tmp_path / "cancel-read.sqlite")
        await store.create_task(TaskCreate(task_id="barrier-row", type="report", title="visible"))
        app = CayuApp(task_store=store, enable_logging=False)
        server = create_server(app, config=ServerConfig.local_development())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server), base_url="http://test"
        ) as client:
            initial = await client.get("/api/tasks", params={"include": "counts"})
            assert isinstance(initial.json(), dict)
            entered, release = asyncio.Event(), threading.Event()
            loop = asyncio.get_running_loop()
            redact = app.redact_json

            def pause_projection(value):
                if value == "barrier-row":
                    loop.call_soon_threadsafe(entered.set)
                    if not release.wait(5):
                        raise TimeoutError("test projection barrier timed out")
                return redact(value)

            monkeypatch.setattr(app, "redact_json", pause_projection)
            read = asyncio.create_task(
                client.get("/api/tasks", params={"include": "counts", "q": "visible"})
            )
            competing = None
            try:
                await asyncio.wait_for(entered.wait(), 5)
                read.cancel()
                await asyncio.sleep(0)
                read.cancel()
                assert read.cancelling() == 2
                competing = asyncio.create_task(
                    store.create_task(TaskCreate(task_id="later", type="report"))
                )
                await asyncio.sleep(0.02)
                assert not read.done()
                assert not competing.done()
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await read
                assert read.cancelled() and read.cancelling() == 2
                await asyncio.wait_for(competing, 5)
                assert (await store.load_task("barrier-row")).status.value == "pending"
                assert (await store.load_task("later")).status.value == "pending"
            finally:
                release.set()
                await asyncio.gather(
                    read, *([] if competing is None else [competing]), return_exceptions=True
                )

    asyncio.run(scenario())


def test_task_index_unsupported_and_failed_sources_never_return_partial_counts(capsys, caplog):
    from cayu.tasks.store import TaskStore

    class Unsupported(InMemoryTaskStore):
        query_task_index = getattr(TaskStore, "query_task_index", None)

    class Failed(InMemoryTaskStore):
        async def query_task_index(self, query, *, matches):
            await super().query_task_index(query, matches=matches)
            raise ValueError("private-source-canary")

    for store, expected in [(Unsupported(), 501), (Failed(), 503)]:
        asyncio.run(store.create_task(TaskCreate(type="report")))
        with TestClient(
            create_server(
                CayuApp(task_store=store, enable_logging=False),
                config=ServerConfig.local_development(),
            )
        ) as client:
            response = client.get("/api/tasks", params={"include": "counts"})
            assert response.status_code == expected
            assert "private-source-canary" not in response.text
            assert "counts_by_status" not in response.json()
            if expected == 501:
                capability = client.get("/api/contract").json()["capabilities"]["surfaces"][
                    "task_index"
                ]
                assert capability["read"] == {"enabled": False, "unavailable_reason": "unsupported"}

    captured = capsys.readouterr()
    assert "private-source-canary" not in captured.out + captured.err + caplog.text


def test_postgres_task_index_and_cancellation_through_http(postgres_dsn):
    """CI owns PostgreSQL startup; this test is never selected locally."""
    from uuid import uuid4

    import httpx

    from cayu.storage.migrations import SchemaMode
    from cayu.storage.tasks_postgres import PostgresTaskStore

    async def scenario():
        store = PostgresTaskStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        prefix = uuid4().hex
        try:
            for index in range(4):
                await store.create_task(
                    TaskCreate(task_id=f"{prefix}-{index}", type=prefix, title="public report")
                )
            await store.create_task_graph(
                TaskGraphCreate(
                    graph_id=prefix + "-graph",
                    nodes=(
                        TaskGraphNode(task=TaskCreate(task_id=prefix + "-root", type=prefix)),
                        TaskGraphNode(
                            task=TaskCreate(task_id=prefix + "-child", type=prefix),
                            prerequisite_task_ids=(prefix + "-root",),
                        ),
                    ),
                )
            )
            app = CayuApp(task_store=store, enable_logging=False)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(
                    app=create_server(app, config=ServerConfig.local_development())
                ),
                base_url="http://test",
            ) as client:
                dependents = await client.get(f"/api/tasks/{prefix}-root/dependents")
                assert dependents.status_code == 200, dependents.text
                assert dependents.json()["total_count"] == "1"
                assert dependents.json()["items"][0]["prerequisite_task_ids"] == [prefix + "-root"]
                cancelled = await client.post(f"/api/tasks/{prefix}-0/cancel")
                assert cancelled.status_code == 200
                response = await client.get(
                    "/api/tasks",
                    params={
                        "include": "counts",
                        "type": prefix,
                        "q": "public",
                        "limit": 1,
                        "order_by": "created_at_asc",
                    },
                )
                assert response.status_code == 200, response.text
                page = response.json()
                assert page["total_count"] == "4"
                assert page["counts_by_status"]["pending"] == "3"
                assert page["counts_by_status"]["cancelled"] == "1"
                second = await client.get(
                    "/api/tasks",
                    params={
                        "include": "counts",
                        "type": prefix,
                        "q": "public",
                        "limit": 1,
                        "order_by": "created_at_asc",
                        "cursor": page["next_cursor"],
                    },
                )
                assert second.status_code == 200, second.text
                assert second.json()["items"][0]["id"] == f"{prefix}-1"
        finally:
            await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_member_schedule_history_is_redacted_and_general_cancel_preserves_revision_fence(
    backend, tmp_path, sqlite_task_store
):
    from datetime import UTC, datetime, timedelta

    from cryptography.hazmat.primitives.asymmetric import ec
    from tests.server.test_proxy_identity import FakeJwks, _as, _auth, _token

    from cayu.tasks import TaskSchedulePolicy
    from cayu.vaults import SecretRedactor

    if backend == "sqlite":
        store = sqlite_task_store(tmp_path / "schedule.sqlite")
    else:
        store = InMemoryTaskStore()
    app = CayuApp(
        task_store=store, enable_logging=False, secret_redactor=SecretRedactor("operation-canary")
    )
    asyncio.run(
        app.create_task(
            TaskCreate(
                task_id="scheduled",
                type="report",
                available_at=datetime.now(UTC) + timedelta(hours=1),
                schedule_policy=TaskSchedulePolicy(),
            )
        )
    )
    key = ec.generate_private_key(ec.SECP256R1())
    with TestClient(
        create_server(app, config=ServerConfig.protected(_auth(FakeJwks(key))))
    ) as client:
        member, admin = _as(_token(key, role="member")), _as(_token(key, role="admin"))
        denied_general = client.post("/api/tasks/scheduled/cancel", headers=admin)
        assert denied_general.status_code == 409, denied_general.text
        task = asyncio.run(store.load_task("scheduled"))
        assert task.status.value == "pending" and task.schedule.revision == 1
        body = {"task_id": "scheduled", "operation_id": "operation-canary", "expected_revision": 1}
        assert (
            client.post("/api/tasks/schedule/cancel", headers=member, json=body).status_code == 403
        )
        assert (
            client.post("/api/tasks/schedule/cancel", headers=admin, json=body).status_code == 422
        )
        # The trusted runtime can have recorded an opaque operation identity;
        # member publication must redact it while raw public input stays denied.
        from cayu.tasks import TaskScheduleCancelRequest

        asyncio.run(store.cancel_scheduled_task(TaskScheduleCancelRequest(**body)))
        history = client.get("/api/tasks/scheduled/schedule/events", headers=member)
        assert history.status_code == 200, history.text
        assert len(history.json()) == 2
        assert history.json()[-1]["type"] == "task.schedule_cancelled"
        assert "operation-canary" not in history.text


def test_member_task_index_filters_and_cursor_identity_preserve_read_only_access():
    from cryptography.hazmat.primitives.asymmetric import ec
    from tests.server.test_proxy_identity import FakeJwks, _as, _auth, _token

    from cayu.vaults import SecretRedactor

    key = ec.generate_private_key(ec.SECP256R1())
    store = InMemoryTaskStore()

    async def seed():
        await store.create_task(TaskCreate(task_id="public-a", type="report", title="visible"))
        await store.create_task(TaskCreate(task_id="public-b", type="report", title="visible"))
        await store.create_task(
            TaskCreate(task_id="private", type="report", title="private-title-canary")
        )
        await store.pause_task("public-b")

    asyncio.run(seed())
    app = CayuApp(
        task_store=store,
        enable_logging=False,
        secret_redactor=SecretRedactor("private-title-canary"),
    )
    with TestClient(
        create_server(app, config=ServerConfig.protected(_auth(FakeJwks(key))))
    ) as client:
        member = _as(_token(key, role="member"))
        first = client.get(
            "/api/tasks", headers=member, params={"include": "counts", "limit": 1}
        ).json()
        assert first["total_count"] == "3"
        assert first["next_cursor"]
        foreign = _as(_token(key, role="member", sub="another-person"))
        assert (
            client.get(
                "/api/tasks",
                headers=foreign,
                params={"include": "counts", "cursor": first["next_cursor"]},
            ).status_code
            == 422
        )
        filtered = client.get(
            "/api/tasks",
            headers=member,
            params=[("include", "counts"), ("statuses", "paused"), ("statuses", "cancelled")],
        ).json()
        assert filtered["total_count"] == "1"
        assert filtered["counts_by_status"]["pending"] == "2"
        assert [row["id"] for row in filtered["items"]] == ["public-b"]
        # The existing bare-list consumer also loses the raw-title oracle.
        assert (
            client.get("/api/tasks", headers=member, params={"q": "private-title-canary"}).json()
            == []
        )
        assert (
            len(
                client.get(
                    "/api/tasks", headers=member, params={"q": "visible", "limit": 1, "offset": 1}
                ).json()
            )
            == 1
        )
        assert (
            client.get(
                "/api/tasks", headers=member, params={"include": "counts", "offset": 1}
            ).status_code
            == 422
        )
        assert client.post("/api/tasks/public-a/cancel", headers=member).status_code == 403


@pytest.mark.parametrize("envelope", [True, False])
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("field", ["type", "session_id", "parent_task_id", "assigned_agent_name"])
def test_public_task_scalar_filters_match_projection_not_raw_values(
    backend, field, envelope, tmp_path, request
):
    from uuid import uuid4

    import httpx

    from cayu.vaults import SecretRedactor

    prefix = uuid4().hex
    secrets = [prefix + "-private-one", prefix + "-private-two"]
    if backend == "memory":
        store = InMemoryTaskStore()
    elif backend == "sqlite":
        from cayu.storage import SQLiteTaskStore

        store = SQLiteTaskStore(tmp_path / "scalar.sqlite")
    else:
        from cayu.storage.migrations import SchemaMode
        from cayu.storage.tasks_postgres import PostgresTaskStore

        store = PostgresTaskStore(
            request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
        )

    async def scenario():
        try:
            if field == "parent_task_id":
                for secret in secrets:
                    await store.create_task(TaskCreate(task_id=secret, type="parent"))
            nodes = [TaskGraphNode(task=TaskCreate(task_id=prefix, type="root"))]
            for index, secret in enumerate(secrets):
                values = {"type": "report", field: secret}
                nodes.append(
                    TaskGraphNode(
                        task=TaskCreate(task_id=f"{prefix}-{index}", title=prefix, **values),
                        prerequisite_task_ids=(prefix,),
                    )
                )
            await store.create_task_graph(TaskGraphCreate(graph_id=prefix, nodes=tuple(nodes)))
            app = CayuApp(
                task_store=store, secret_redactor=SecretRedactor(secrets), enable_logging=False
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(
                    app=create_server(app, config=ServerConfig.local_development())
                ),
                base_url="http://test",
            ) as client:
                detail = (await client.get(f"/api/tasks/{prefix}-0")).json()
                displayed = detail[field]
                assert displayed not in secrets
                dependent = (await client.get(f"/api/tasks/{prefix}/dependents")).json()
                assert len(dependent["items"]) == 2
                assert {item[field] for item in dependent["items"]} == {displayed}
                for value in [*secrets, displayed]:
                    # Omit q to verify scalar filtering independently of text search.
                    params = {field: value}
                    if envelope:
                        params["include"] = "counts"
                    response = await client.get("/api/tasks", params=params)
                    assert response.status_code == 200, response.text
                    body = response.json()
                    items = body["items"] if envelope else body
                    expected = {f"{prefix}-0", f"{prefix}-1"} if value == displayed else set()
                    assert {item["id"] for item in items} == expected
                    if envelope:
                        assert body["total_count"] == str(len(expected))
                plain = await client.get("/api/tasks", params={"type": "root"})
                assert prefix in {item["id"] for item in plain.json()}
        finally:
            if backend != "memory":
                await store.close()

    asyncio.run(scenario())


def test_custom_store_bare_filters_keep_public_matching_without_native_index():
    from cayu.tasks.store import TaskStore
    from cayu.vaults import SecretRedactor

    class ListOnlyStore(InMemoryTaskStore):
        query_task_index = TaskStore.query_task_index

    store = ListOnlyStore()

    async def seed():
        for index in range(260):
            await store.create_task(TaskCreate(task_id=f"skip-{index:03}", type="other"))
        for index in range(3):
            await store.create_task(
                TaskCreate(
                    task_id=f"match-{index}",
                    type="private-type-canary",
                    assigned_agent_name="specialist",
                    title="Public title",
                )
            )

    asyncio.run(seed())
    app = CayuApp(
        task_store=store,
        secret_redactor=SecretRedactor("private-type-canary"),
        enable_logging=False,
    )
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        displayed = client.get("/api/tasks/match-0").json()["type"]
        for params in [
            {"type": displayed},
            {"assigned_agent_name": "specialist"},
            {"q": "Public title"},
        ]:
            response = client.get(
                "/api/tasks",
                params={**params, "order_by": "created_at_asc", "limit": 1, "offset": 1},
            )
            assert response.status_code == 200, response.text
            assert [item["id"] for item in response.json()] == ["match-1"]
        assert client.get("/api/tasks", params={"type": "private-type-canary"}).json() == []
        assert client.get("/api/tasks", params={"include": "counts"}).status_code == 501


@pytest.mark.parametrize("action", ["dependents", "cancel"])
def test_task_control_paths_validate_blank_and_normalize_padded_ids(action):
    store = InMemoryTaskStore()
    asyncio.run(store.create_task(TaskCreate(task_id="task", type="report")))
    with TestClient(
        create_server(
            CayuApp(task_store=store, enable_logging=False), config=ServerConfig.local_development()
        ),
        raise_server_exceptions=False,
    ) as client:
        request = client.post if action == "cancel" else client.get
        assert request(f"/api/tasks/%20/{action}").status_code == 422
        assert request(f"/api/tasks/task%20/{action}").status_code == 200


def test_postgres_task_index_tied_timestamp_pages_use_codepoint_order(postgres_dsn):
    """Remote CI checks a real ICU database, not a default-C approximation."""
    from datetime import UTC, datetime
    from uuid import uuid4

    import httpx
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    from cayu.storage.migrations import SchemaMode
    from cayu.storage.tasks_postgres import PostgresTaskStore

    database = "task_collation_" + uuid4().hex
    with psycopg.connect(postgres_dsn, autocommit=True) as connection:
        connection.execute(
            sql.SQL(
                "CREATE DATABASE {} TEMPLATE template0 ENCODING 'UTF8' "
                "LOCALE_PROVIDER icu ICU_LOCALE 'en-US'"
            ).format(sql.Identifier(database))
        )
    dsn = make_conninfo(postgres_dsn, dbname=database)

    async def scenario():
        instant = datetime(2026, 10, 24, tzinfo=UTC)
        store = PostgresTaskStore(dsn, schema_mode=SchemaMode.CREATE)
        try:
            with psycopg.connect(dsn) as connection:
                assert connection.execute("SELECT 'a'::text < 'B'::text").fetchone() == (True,)
            identities = ["a", "B", "é", "中"]
            for identity in identities:
                await store.create_task(TaskCreate(task_id=identity, type="report"))
            # Creation timestamps belong to the creation owner, not the lease clock.
            with psycopg.connect(dsn) as connection:
                connection.execute(
                    "UPDATE cayu_tasks SET created_at = %s, updated_at = %s",
                    (instant, instant),
                )
                assert connection.execute(
                    "SELECT count(DISTINCT created_at), count(DISTINCT updated_at), "
                    "min(created_at), min(updated_at) FROM cayu_tasks"
                ).fetchone() == (1, 1, instant, instant)
            app = CayuApp(task_store=store, enable_logging=False)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(
                    app=create_server(app, config=ServerConfig.local_development())
                ),
                base_url="http://test",
            ) as client:
                for order in [
                    "created_at_asc",
                    "created_at_desc",
                    "updated_at_asc",
                    "updated_at_desc",
                ]:
                    cursor = None
                    collected = []
                    for _ in identities:
                        params = {"include": "counts", "limit": 1, "order_by": order}
                        if cursor is not None:
                            params["cursor"] = cursor
                        response = await client.get("/api/tasks", params=params)
                        assert response.status_code == 200, response.text
                        page = response.json()
                        assert page["total_count"] == "4"
                        collected.extend(item["id"] for item in page["items"])
                        cursor = page["next_cursor"]
                    assert collected == sorted(identities)
                    assert cursor is None
        finally:
            await store.close()

    try:
        asyncio.run(scenario())
    finally:
        with psycopg.connect(postgres_dsn, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database))
            )


@pytest.mark.parametrize("mode", ["scan_budget", "overlap"])
def test_custom_task_list_refuses_unbounded_or_shifted_partial_results(mode):
    from cayu.tasks.store import TaskStore

    class CustomStore(InMemoryTaskStore):
        query_task_index = TaskStore.query_task_index
        calls = 0

        async def list_tasks(self, query):
            self.calls += 1
            size = 8192 if mode == "scan_budget" else 512
            start = query.offset
            rows = [
                template.model_copy(update={"id": f"row-{index}"})
                for index in range(start, min(start + query.limit, size))
            ]
            if mode == "overlap" and start == 256:
                rows[0] = template.model_copy(update={"id": "row-0"})
            return rows

    store = CustomStore()
    template = asyncio.run(store.create_task(TaskCreate(type="report", title="visible")))
    with TestClient(
        create_server(
            CayuApp(task_store=store, enable_logging=False), config=ServerConfig.local_development()
        )
    ) as client:
        response = client.get(
            "/api/tasks",
            params={"q": "absent" if mode == "scan_budget" else "visible", "limit": 1000},
        )
        assert response.status_code == 503, response.text
        assert "items" not in response.json()
        assert store.calls <= (16 if mode == "scan_budget" else 2)
