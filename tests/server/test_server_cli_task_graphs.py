"""Declared HTTP graphs preserve atomic admission and exact actor-bound replay."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from cayu import CayuApp
from cayu.server import ServerConfig, create_server
from cayu.tasks import InMemoryTaskStore, TaskTypeSpec


def app_for(store):
    app = CayuApp(task_store=store, enable_logging=False)
    app.register_task_type(
        TaskTypeSpec(
            name="report",
            input_schema={
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "additionalProperties": False,
            },
        )
    )
    return app


def graph():
    return {
        "operation_id": "graph-one",
        "nodes": [
            {"key": "first", "type": "report", "input": {"topic": "first"}},
            {"key": "second", "type": "report", "input": {}, "prerequisite_keys": ["first"]},
        ],
    }


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_exact_graph_replay_and_http_provenance(tmp_path, backend):
    from cayu.storage import SQLiteTaskStore

    store = InMemoryTaskStore() if backend == "memory" else SQLiteTaskStore(tmp_path / "tasks.db")
    with TestClient(
        create_server(app_for(store), config=ServerConfig.local_development())
    ) as client:
        body = graph()
        response = client.post("/api/task-graphs", json=body)
        assert response.status_code == 200, response.text
        receipt = response.json()
        assert set(receipt["task_ids"]) == {"first", "second"}
        # Ordering is not authority: nodes and prerequisites are canonical sets.
        body["nodes"].reverse()
        assert client.post("/api/task-graphs", json=body).json() == receipt
        rows = {r["id"]: r for r in client.get("/api/tasks").json()}
        assert rows[receipt["task_ids"]["first"]]["status"] == "pending"
        assert rows[receipt["task_ids"]["second"]]["status"] == "waiting_dependencies"
        for task_id in receipt["task_ids"].values():
            detail = client.get(f"/api/tasks/{task_id}").json()
            assert detail["invocation"]["source"] == "http_task"
        for changed in ("input", "prerequisites", "key"):
            request = graph()
            if changed == "input":
                request["nodes"][1]["input"] = {"topic": "changed"}
            elif changed == "prerequisites":
                request["nodes"][1]["prerequisite_keys"] = []
            else:
                request["nodes"][1]["key"] = "third"
            assert client.post("/api/task-graphs", json=request).status_code == 409
        assert len(client.get("/api/tasks").json()) == 2
        capability = client.get("/api/contract").json()["capabilities"]["surfaces"][
            "task_graph_creation"
        ]
        assert capability["mutate"]["enabled"] is True
    if backend == "sqlite":
        asyncio.run(store.close())
        reopened = SQLiteTaskStore(tmp_path / "tasks.db")
        with TestClient(
            create_server(app_for(reopened), config=ServerConfig.local_development())
        ) as client:
            assert client.post("/api/task-graphs", json=graph()).json() == receipt
            # Registration rejects replacing declarations, so restart with a changed one.
            changed_app = CayuApp(task_store=reopened, enable_logging=False)
            changed_app.register_task_type(
                TaskTypeSpec(
                    name="report", description="Changed contract", input_schema={"type": "object"}
                )
            )
            with TestClient(
                create_server(changed_app, config=ServerConfig.local_development())
            ) as changed_client:
                assert changed_client.post("/api/task-graphs", json=graph()).status_code == 409
        asyncio.run(reopened.close())


@pytest.mark.parametrize("invalid", ["type", "input", "cycle", "duplicate", "spoof", "empty"])
def test_invalid_graph_never_publishes_a_valid_prefix(invalid):
    with TestClient(
        create_server(app_for(InMemoryTaskStore()), config=ServerConfig.local_development())
    ) as client:
        request = graph()
        if invalid == "type":
            request["nodes"][1]["type"] = "missing"
        elif invalid == "input":
            request["nodes"][1]["input"] = {"topic": True}
        elif invalid == "cycle":
            request["nodes"][0]["prerequisite_keys"] = ["second"]
        elif invalid == "duplicate":
            request["nodes"][1]["key"] = "first"
        elif invalid == "spoof":
            request["nodes"][1]["assigned_agent_name"] = "other"
        else:
            request["nodes"] = []
        assert client.post("/api/task-graphs", json=request).status_code == 422
        assert client.get("/api/tasks").json() == []


def test_graph_requires_full_access_and_retains_verified_actor():
    from cryptography.hazmat.primitives.asymmetric import ec
    from tests.server.test_proxy_identity import FakeJwks, _as, _auth, _token

    key = ec.generate_private_key(ec.SECP256R1())
    with TestClient(
        create_server(
            app_for(InMemoryTaskStore()), config=ServerConfig.protected(_auth(FakeJwks(key)))
        )
    ) as client:
        assert client.post("/api/task-graphs", json=graph()).status_code == 401
        assert (
            client.post(
                "/api/task-graphs", json=graph(), headers=_as(_token(key, role="member"))
            ).status_code
            == 403
        )
        admin = _as(_token(key, role="admin"))
        response = client.post("/api/task-graphs", json=graph(), headers=admin)
        assert response.status_code == 200, response.text
        item = client.get(
            "/api/tasks/" + response.json()["task_ids"]["first"], headers=admin
        ).json()
        assert item["invocation"]["origin"]["trust"] == "server_verified"
        assert item["invocation"]["origin"]["subject"] == "person-1"
        other = client.post(
            "/api/task-graphs", json=graph(), headers=_as(_token(key, role="admin", sub="person-2"))
        )
        assert other.status_code == 200
        assert other.json()["graph_id"] != response.json()["graph_id"]
        assert len(client.get("/api/tasks", headers=admin).json()) == 4


def test_graph_missing_store_and_unsupported_store_fail_closed():
    class Unsupported(InMemoryTaskStore):
        @property
        def supports_task_graphs(self):
            return False

    for store in (None, Unsupported()):
        with TestClient(
            create_server(app_for(store), config=ServerConfig.local_development())
        ) as client:
            response = client.post("/api/task-graphs", json=graph())
            assert response.status_code == 501


def test_graph_acknowledgement_loss_replays_one_atomic_graph():
    class LostAcknowledgement(InMemoryTaskStore):
        lose_next = True

        async def create_task_graph(self, request):
            receipt = await super().create_task_graph(request)
            if self.lose_next:
                self.lose_next = False
                raise RuntimeError("lost acknowledgement")
            return receipt

    with TestClient(
        create_server(app_for(LostAcknowledgement()), config=ServerConfig.local_development()),
        raise_server_exceptions=False,
    ) as client:
        assert client.post("/api/task-graphs", json=graph()).status_code == 500
        response = client.post("/api/task-graphs", json=graph())
        assert response.status_code == 200, response.text
        assert len(client.get("/api/tasks").json()) == 2


def test_postgres_graph_http_exact_replay_and_atomic_invalid_node(request):
    import os

    if not os.environ.get("CAYU_TEST_POSTGRES_DSN"):
        pytest.skip("Postgres HTTP parity runs in CI; local Docker is disabled.")
    import httpx

    from cayu.storage.migrations import SchemaMode
    from cayu.storage.tasks_postgres import PostgresTaskStore

    dsn = request.getfixturevalue("postgres_dsn")

    async def scenario():
        store = PostgresTaskStore(dsn, schema_mode=SchemaMode.CREATE)
        app = app_for(store)
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(
                    app=create_server(app, config=ServerConfig.local_development())
                ),
                base_url="http://testserver",
            ) as client:
                invalid = graph()
                invalid["nodes"][1]["type"] = "missing"
                assert (await client.post("/api/task-graphs", json=invalid)).status_code == 422
                assert (await client.get("/api/tasks")).json() == []
                first = await client.post("/api/task-graphs", json=graph())
                assert first.status_code == 200, first.text
                assert (await client.post("/api/task-graphs", json=graph())).json() == first.json()
                assert len((await client.get("/api/tasks")).json()) == 2
        finally:
            await app.aclose()
            await store.close()

    asyncio.run(scenario())


def test_graph_node_bound_and_boolean_identity_have_no_partial_admission():
    from cayu.tasks.graphs import TASK_GRAPH_MAX_NODES

    with TestClient(
        create_server(app_for(InMemoryTaskStore()), config=ServerConfig.local_development())
    ) as client:
        body = {
            "operation_id": "max-nodes",
            "nodes": [
                {"key": f"node-{index}", "type": "report", "input": {}}
                for index in range(TASK_GRAPH_MAX_NODES)
            ],
        }
        response = client.post("/api/task-graphs", json=body)
        assert response.status_code == 200, response.text
        assert len(response.json()["task_ids"]) == TASK_GRAPH_MAX_NODES
        body["operation_id"] = "too-many"
        body["nodes"].append({"key": "extra", "type": "report", "input": {}})
        assert client.post("/api/task-graphs", json=body).status_code == 422
        invalid = graph()
        invalid["operation_id"] = True
        assert client.post("/api/task-graphs", json=invalid).status_code == 422
        assert client.post("/api/task-graphs", json=graph()).status_code == 200


def test_graph_node_keys_cannot_bypass_public_identity_redaction(capsys, caplog):
    import warnings

    from cayu.vaults import SecretRedactor

    secret = "graph-key-secret-canary"
    app = CayuApp(
        task_store=InMemoryTaskStore(), secret_redactor=SecretRedactor(secret), enable_logging=False
    )
    app.register_task_type(TaskTypeSpec(name="report", input_schema={"type": "object"}))
    with (
        warnings.catch_warnings(record=True) as captured,
        TestClient(create_server(app, config=ServerConfig.local_development())) as client,
    ):
        body = {
            "operation_id": "redaction",
            "nodes": [{"key": secret, "type": "report", "input": {}}],
        }
        response = client.post("/api/task-graphs", json=body)
        assert response.status_code == 422
        assert client.get("/api/tasks").json() == []
    output = capsys.readouterr()
    assert secret not in response.text + output.out + output.err + caplog.text + str(captured)


def test_http_graph_cancellation_after_commit_replays_without_republication():
    import httpx

    async def scenario():
        committed, release = asyncio.Event(), asyncio.Event()

        class DelayedAcknowledgement(InMemoryTaskStore):
            first = True

            async def create_task_graph(self, request):
                receipt = await super().create_task_graph(request)
                if self.first:
                    self.first = False
                    committed.set()
                    await release.wait()
                return receipt

        store = DelayedAcknowledgement()
        app = app_for(store)
        server = create_server(app, config=ServerConfig.local_development())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server), base_url="http://testserver"
        ) as client:
            caller = asyncio.create_task(client.post("/api/task-graphs", json=graph()))
            try:
                await asyncio.wait_for(committed.wait(), 5)
                caller.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await caller
                assert caller.cancelled() and caller.cancelling() == 1
                replay = await client.post("/api/task-graphs", json=graph())
                assert replay.status_code == 200
                graph_id = replay.json()["graph_id"]
                before = await app.list_task_graph_events(graph_id)
                release.set()
                repeated = await client.post("/api/task-graphs", json=graph())
                assert repeated.json() == replay.json()
                assert await app.list_task_graph_events(graph_id) == before
                assert len((await client.get("/api/tasks")).json()) == 2
            finally:
                release.set()
                if not caller.done():
                    caller.cancel()
                await asyncio.gather(caller, return_exceptions=True)
                await app.aclose()

    asyncio.run(scenario())
