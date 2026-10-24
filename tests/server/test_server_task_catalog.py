"""Public task catalog and declared creation acceptance."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi.testclient import TestClient

from cayu import CayuApp
from cayu.server import ServerConfig, create_server
from cayu.sessions.memory import InMemorySessionStore
from cayu.tasks import InMemoryTaskStore


def test_task_catalog_without_declarations_is_explicitly_empty() -> None:
    app = CayuApp(
        session_store=InMemorySessionStore(),
        task_store=InMemoryTaskStore(),
        enable_logging=False,
    )
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        response = client.get("/api/task-types")
        assert response.status_code == 200
        assert response.json() == {"task_types": []}


def test_task_catalog_declaration_is_detached_from_caller_and_reader() -> None:
    from cayu.tasks import TaskTypeSpec

    app = CayuApp(
        session_store=InMemorySessionStore(), task_store=InMemoryTaskStore(), enable_logging=False
    )
    declaration = TaskTypeSpec(
        name="report",
        description="Build a report",
        agent_name=None,
        input_schema={"type": "object", "properties": {"topic": {"type": "string"}}},
    )
    app.register_task_type(declaration)
    declaration.input_schema["properties"]["topic"]["type"] = "number"
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        response = client.get("/api/task-types")
        assert response.status_code == 200
        assert response.json() == {
            "task_types": [
                {
                    "name": "report",
                    "description": "Build a report",
                    "agent_name": None,
                    "input_schema": {"type": "object", "properties": {"topic": {"type": "string"}}},
                }
            ]
        }


def test_task_catalog_authenticates_member_read() -> None:
    pytest.importorskip("jwt")
    from cryptography.hazmat.primitives.asymmetric import ec
    from tests.server.test_proxy_identity import FakeJwks, _as, _auth, _token

    key = ec.generate_private_key(ec.SECP256R1())
    app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
    with TestClient(
        create_server(app, config=ServerConfig.protected(_auth(FakeJwks(key))))
    ) as client:
        assert client.get("/api/task-types").status_code == 401
        response = client.get("/api/task-types", headers=_as(_token(key, role="member")))
        assert response.status_code == 200
        assert response.json() == {"task_types": []}


def test_task_create_rejects_undeclared_type_without_mutation() -> None:
    store = InMemoryTaskStore()
    with TestClient(
        create_server(CayuApp(task_store=store), config=ServerConfig.local_development())
    ) as client:
        response = client.post(
            "/api/tasks", json={"operation_id": "one", "type": "missing", "input": {}}
        )
        assert response.status_code == 422
        assert client.get("/api/tasks").json() == []


def test_declared_task_create_and_replay_use_real_store_and_exact_receipt() -> None:
    from cayu.tasks import TaskTypeSpec

    store = InMemoryTaskStore()
    app = CayuApp(task_store=store, enable_logging=False)
    app.register_task_type(TaskTypeSpec(name="report", input_schema={"type": "object"}))
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        body = {"operation_id": "create-one", "type": "report", "input": {"answer": True}}
        response = client.post("/api/tasks", json=body)
        assert response.status_code == 200, response.text
        first = response.json()["task_id"]
        assert client.post("/api/tasks", json=body).json() == {"task_id": first}
        changed = {**body, "input": {"answer": 1}}
        assert client.post("/api/tasks", json=changed).status_code == 409
        selected = client.get(f"/api/tasks/{first}/replay").json()
        assert selected["available"] is True
        replay = {"operation_id": "replay-one", "type": "report", "source": selected["source"]}
        response = client.post(f"/api/tasks/{first}/replay", json=replay)
        assert response.status_code == 200, response.text
        second = response.json()["task_id"]
        assert second != first
        assert client.post(f"/api/tasks/{first}/replay", json=replay).json() == {"task_id": second}
        task = client.get(f"/api/tasks/{second}").json()
        assert task["input"] == {"answer": True}
        assert task["session_id"] is None
        assert task["invocation"]["source"] == "http_task"
        assert len(client.get("/api/tasks").json()) == 2


def test_task_create_acknowledgement_loss_is_retryable_without_duplicate_work() -> None:
    from cayu.tasks import TaskTypeSpec

    class LostAcknowledgement(InMemoryTaskStore):
        lose_next = True

        async def create_task_operation(self, request, operation):
            task = await super().create_task_operation(request, operation)
            if self.lose_next:
                self.lose_next = False
                raise RuntimeError("simulated acknowledgement loss")
            return task

    app = CayuApp(task_store=LostAcknowledgement(), enable_logging=False)
    app.register_task_type(TaskTypeSpec(name="report", input_schema={"type": "object"}))
    with TestClient(
        create_server(app, config=ServerConfig.local_development()), raise_server_exceptions=False
    ) as client:
        body = {"operation_id": "retained", "type": "report", "input": {}}
        assert client.post("/api/tasks", json=body).status_code == 500
        response = client.post("/api/tasks", json=body)
        assert response.status_code == 200
        tasks = client.get("/api/tasks").json()
        assert len(tasks) == 1
        assert tasks[0]["id"] == response.json()["task_id"]


def test_task_replay_preserves_private_input_and_member_cannot_override_or_create(
    capsys, caplog
) -> None:
    import asyncio

    from cryptography.hazmat.primitives.asymmetric import ec
    from tests.server.test_proxy_identity import FakeJwks, _as, _auth, _token

    from cayu.tasks import TaskTypeSpec
    from cayu.vaults import SecretRedactor

    secret = "task-private-input-canary"
    store = InMemoryTaskStore()
    app = CayuApp(task_store=store, secret_redactor=SecretRedactor(secret), enable_logging=False)
    app.register_task_type(TaskTypeSpec(name="report", input_schema={"type": "object"}))
    key = ec.generate_private_key(ec.SECP256R1())
    admin = _as(_token(key, role="admin"))
    member = _as(_token(key, role="member"))
    with TestClient(
        create_server(app, config=ServerConfig.protected(_auth(FakeJwks(key))))
    ) as client:
        body = {"operation_id": "private-one", "type": "report", "input": {"value": secret}}
        assert client.post("/api/tasks", json=body, headers=member).status_code == 403
        response = client.post("/api/tasks", json=body, headers=admin)
        assert response.status_code == 200, response.text
        original = response.json()["task_id"]
        detail = client.get(f"/api/tasks/{original}", headers=member)
        selected = client.get(f"/api/tasks/{original}/replay", headers=member)
        assert selected.status_code == 200
        assert secret not in detail.text + selected.text
        replay = {
            "operation_id": "private-replay",
            "type": "report",
            "source": selected.json()["source"],
        }
        assert (
            client.post(f"/api/tasks/{original}/replay", json=replay, headers=member).status_code
            == 403
        )
        refused = client.post(
            f"/api/tasks/{original}/replay",
            json={**replay, "input": {"value": secret}},
            headers=admin,
        )
        assert refused.status_code == 422
        assert secret not in refused.text
        response = client.post(f"/api/tasks/{original}/replay", json=replay, headers=admin)
        assert response.status_code == 200, response.text
        replayed = asyncio.run(store.load_task(response.json()["task_id"]))
        assert replayed.input == {"value": secret}
        assert replayed.invocation.origin.trust.value == "server_verified"
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err + caplog.text


@pytest.mark.parametrize(
    "field",
    [
        "task_id",
        "created_at",
        "root_invocation_id",
        "restart_count",
        "session_id",
        "session_instance_id",
    ],
)
def test_task_replay_rejects_changed_source_identity_without_creation(field) -> None:
    from cayu.tasks import TaskTypeSpec

    app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
    app.register_task_type(TaskTypeSpec(name="report", input_schema={"type": "object"}))
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        response = client.post(
            "/api/tasks", json={"operation_id": "source", "type": "report", "input": {}}
        )
        task_id = response.json()["task_id"]
        selected = client.get(f"/api/tasks/{task_id}/replay").json()["source"]
        selected[field] = (
            1
            if field == "restart_count"
            else "2001-01-01T00:00:00Z"
            if field == "created_at"
            else "different-identity"
        )
        response = client.post(
            f"/api/tasks/{task_id}/replay",
            json={"operation_id": "changed", "type": "report", "source": selected},
        )
        assert response.status_code == 409, response.text
        assert len(client.get("/api/tasks").json()) == 1


def test_session_task_replay_requires_exact_native_binding_not_metadata() -> None:
    import asyncio

    from tests.core.task_invocation_fixtures import stored_session_invocation

    from cayu import Message
    from cayu.sessions.records import SessionIdentity
    from cayu.sessions.requests import RunRequest, run_request_with_task_invocation
    from cayu.tasks import TaskCreate

    sessions = InMemorySessionStore()
    tasks = InMemoryTaskStore()
    app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)

    async def seed() -> None:
        await tasks.create_task(TaskCreate(task_id="task-origin", type="report"))
        snapshot = await tasks.load_invocation_snapshot("task-origin")
        run = run_request_with_task_invocation(
            RunRequest(
                agent_name="worker",
                session_id="task-session",
                task_id="task-origin",
                messages=[Message.text("user", "report")],
            ),
            snapshot,
        )
        await sessions.create(run, identity=SessionIdentity(provider_name="fake", model="fake"))
        binding = await stored_session_invocation(sessions, "task-session")
        await tasks.start_task("task-origin", session_id="task-session", session_invocation=binding)
        await sessions.create(
            RunRequest(
                agent_name="worker",
                session_id="direct-session",
                messages=[Message.text("user", "report")],
                metadata={"task_id": "task-origin"},
            ),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )

    asyncio.run(seed())
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        response = client.get("/api/sessions/task-session/task-replay")
        assert response.status_code == 200, response.text
        assert response.json()["unavailable_reason"] == "task_type_not_declared"
        from cayu.tasks import TaskTypeSpec

        app.register_task_type(TaskTypeSpec(name="report", input_schema={"type": "object"}))
        response = client.get("/api/sessions/task-session/task-replay")
        assert response.json()["available"] is True
        assert response.json()["source"]["task_id"] == "task-origin"
        assert response.json()["source"]["session_instance_id"] is not None
        forged = client.get("/api/sessions/direct-session/task-replay")
        assert forged.json()["available"] is False
        assert forged.json()["unavailable_reason"] == "not_task_session"


@pytest.mark.parametrize("field", ["operation_id", "type"])
def test_declared_creation_rejects_blank_identity_without_mutation(field) -> None:
    app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
    body = {"operation_id": "operation", "type": "report", "input": {}}
    body[field] = "  "
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        assert client.post("/api/tasks", json=body).status_code == 422
        assert client.get("/api/tasks").json() == []


def test_task_creation_capability_requires_native_exact_support() -> None:
    class Unsupported(InMemoryTaskStore):
        supports_task_creation_operations = False

    for store, supported in [(InMemoryTaskStore(), True), (Unsupported(), False)]:
        app = CayuApp(task_store=store, enable_logging=False)
        with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
            surface = client.get("/api/contract").json()["capabilities"]["surfaces"][
                "task_creation"
            ]
            assert surface["configured"] is True
            assert surface["read"]["enabled"] is supported
            assert surface["mutate"]["enabled"] is supported
            if not supported:
                assert (
                    client.post(
                        "/api/tasks", json={"operation_id": "one", "type": "report", "input": {}}
                    ).status_code
                    == 501
                )


@pytest.mark.parametrize("case", ["schema", "specialist"])
def test_declared_creation_unavailable_contract_does_not_create_work(case) -> None:
    from cayu.tasks import TaskTypeSpec

    app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
    app.register_task_type(
        TaskTypeSpec(
            name="report",
            agent_name="missing" if case == "specialist" else None,
            input_schema={"type": "object", "required": ["topic"]},
        )
    )
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        response = client.post(
            "/api/tasks", json={"operation_id": "invalid", "type": "report", "input": {}}
        )
        assert response.status_code == 422
        assert client.get("/api/tasks").json() == []


def test_session_replay_source_deleted_after_selection_is_unavailable() -> None:
    import asyncio

    import httpx
    from tests.core.task_invocation_fixtures import stored_session_invocation

    from cayu import Message
    from cayu.sessions.records import SessionIdentity
    from cayu.sessions.requests import RunRequest, run_request_with_task_invocation
    from cayu.tasks import TaskCreate

    async def scenario() -> None:
        selected = asyncio.Event()
        release = asyncio.Event()

        class SelectionBarrier(InMemoryTaskStore):
            async def task_replay_sources_for_session(self, *args):
                candidates = await super().task_replay_sources_for_session(*args)
                selected.set()
                await release.wait()
                return candidates

        sessions = InMemorySessionStore()
        tasks = SelectionBarrier()
        app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
        await tasks.create_task(TaskCreate(task_id="race-task", type="report"))
        snapshot = await tasks.load_invocation_snapshot("race-task")
        request = run_request_with_task_invocation(
            RunRequest(
                agent_name="worker",
                session_id="race-session",
                task_id="race-task",
                messages=[Message.text("user", "report")],
            ),
            snapshot,
        )
        await sessions.create(request, identity=SessionIdentity(provider_name="fake", model="fake"))
        binding = await stored_session_invocation(sessions, "race-session")
        await tasks.start_task("race-task", session_id="race-session", session_invocation=binding)
        server = create_server(app, config=ServerConfig.local_development())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server), base_url="http://test"
        ) as client:
            inspection = asyncio.create_task(client.get("/api/sessions/race-session/task-replay"))
            await asyncio.wait_for(selected.wait(), timeout=5)
            await tasks.complete_task("race-task", {})
            await tasks.delete_session_tasks("race-session", task_ids=("race-task",), policy=None)
            assert await sessions.load("race-session") is not None
            release.set()
            response = await inspection
            assert response.status_code == 200, response.text
            assert response.json() == {
                "available": False,
                "unavailable_reason": "task_source_unavailable",
                "source": None,
            }
            assert (await client.get("/api/tasks/race-task/replay")).status_code == 404
        await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("literal", [False, True])
def test_task_schema_allows_reference_named_data_through_creation(literal) -> None:
    from cayu.tasks import TaskTypeSpec

    value = {"$ref": "https://example.invalid/data", "$schema": "ordinary data"}
    schema = (
        {"const": value}
        if literal
        else {
            "type": "object",
            "properties": {"$ref": {"type": "string"}, "$schema": {"type": "string"}},
            "required": ["$ref", "$schema"],
        }
    )
    app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
    app.register_task_type(TaskTypeSpec(name="document", input_schema=schema))
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        response = client.post(
            "/api/tasks", json={"operation_id": "document", "type": "document", "input": value}
        )
        assert response.status_code == 200, response.text
        assert client.get(f"/api/tasks/{response.json()['task_id']}").json()["input"] == value
