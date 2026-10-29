"""Environment details are declarations plus exact retained-session activity."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi.testclient import TestClient
from tests.core.test_cloud_reporting import _app
from tests.core.test_environment_activity import seed_environment_sessions

from cayu.artifacts.local import LocalArtifactStore
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.mcp.base import McpServerSpec
from cayu.server import ServerConfig, create_server


def test_environment_details_http_and_exact_lookup(tmp_path):
    app = _app()
    app.register_environment(
        Environment(
            EnvironmentSpec(name="dev"),
            artifact_store=LocalArtifactStore(tmp_path / "files"),
            mcp_servers=[McpServerSpec(name="lookup", command=["not-executed"])],
        )
    )
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        assert client.portal is not None
        active = client.portal.call(seed_environment_sessions, app.session_store, "http")
        response = client.get("/api/environments")
        assert response.status_code == 200, response.text
        row = response.json()["environments"][0]
        assert row["mcp_server_names"] == ["lookup"]
        assert row["artifact_store_kind"] == "local_disk"
        assert row["session_count"] == "2"
        assert row["last_activity_at"] == active.last_activity_at.isoformat().replace("+00:00", "Z")
        assert row["activity_as_of"] is not None
        detail = client.get("/api/environments/dev")
        assert detail.status_code == 200
        assert detail.json()["environments"][0]["session_count"] == "2"
        assert client.get("/api/environments/missing").status_code == 404


def test_environment_unknown_storage_kind_and_no_factory_materialization():
    from tests.server.test_server_inventory import CountingArtifactStore

    from cayu.artifacts.aws_s3 import S3ArtifactStore
    from cayu.environments.factory import EnvironmentFactory

    class Factory(EnvironmentFactory):
        async def create(self, request):
            raise AssertionError("inspection must not materialize a factory")

    class PretendS3ArtifactStore(CountingArtifactStore):
        id = "custom"

    app = _app()
    app.register_environment_factory(
        EnvironmentSpec(name="factory"),
        Factory(),
        artifact_store=S3ArtifactStore("fake-bucket", client=object()),
    )
    app.register_environment(
        Environment(EnvironmentSpec(name="custom"), artifact_store=PretendS3ArtifactStore())
    )
    app.register_environment(Environment(EnvironmentSpec(name="none")))
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        response = client.get("/api/environments")
        assert response.status_code == 200, response.text
        rows = {row["name"]: row for row in response.json()["environments"]}
        assert rows["factory"]["artifact_store_kind"] == "s3"
        assert rows["factory"]["is_factory"] is True
        assert rows["custom"]["artifact_store_kind"] == "unknown"
        assert rows["none"]["artifact_store_kind"] is None
        assert all(row["session_count"] == "0" for row in rows.values())


def test_environment_activity_unsupported_is_unknown_and_failure_does_not_publish_partial(
    monkeypatch,
):
    from cayu.sessions.memory import InMemorySessionStore

    class Unsupported(InMemorySessionStore):
        supports_environment_activity = False

        async def aggregate_environment_activity(self, query):
            raise AssertionError("unsupported must not dispatch")

    store = Unsupported()
    app = _app(store)
    app.register_environment(Environment(EnvironmentSpec(name="dev")))
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        row = client.get("/api/environments").json()["environments"][0]
        assert row["session_count"] is None and row["activity_as_of"] is None
        assert (
            client.get("/api/contract").json()["capabilities"]["surfaces"]["environment_details"][
                "read"
            ]["enabled"]
            is False
        )
    supported = _app()
    for i in range(501):
        supported.register_environment(Environment(EnvironmentSpec(name=f"env-{i}")))
    calls = []
    original = supported.session_store.aggregate_environment_activity

    async def broken_second_batch(query):
        calls.append(len(query.environment_names))
        if len(calls) == 2:
            raise RuntimeError("private-error-canary")
        return await original(query)

    monkeypatch.setattr(
        supported.session_store, "aggregate_environment_activity", broken_second_batch
    )
    with TestClient(create_server(supported, config=ServerConfig.local_development())) as client:
        response = client.get("/api/environments")
        assert response.status_code == 503
        assert calls == [500, 1]
        assert "private-error-canary" not in response.text
        assert "environments" not in response.json()


@pytest.mark.parametrize("malformation", ["missing", "foreign", "duplicate", "boolean"])
def test_environment_activity_rejects_inconsistent_custom_results(monkeypatch, malformation):
    app = _app()
    for name in ("dev", "prod"):
        app.register_environment(Environment(EnvironmentSpec(name=name)))
    original = app.session_store.aggregate_environment_activity

    async def malformed(query):
        value = await original(query)
        if malformation == "missing":
            value.environments = ()
        elif malformation == "foreign":
            value.environments[0].environment_name = "other"
        elif malformation == "duplicate":
            value.environments = (value.environments[0], value.environments[0])
        else:
            value.environments[0].session_count = False
        return value

    monkeypatch.setattr(app.session_store, "aggregate_environment_activity", malformed)
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        response = client.get("/api/environments")
        assert response.status_code == 500
        assert response.json() == {
            "detail": "The session store returned inconsistent environment activity."
        }


@pytest.mark.parametrize("role", ["member", "developer", "admin"])
def test_environment_details_preserve_roles_and_encoded_names(role):
    from fastapi import HTTPException

    from cayu.server import AuthenticatedAccess
    from cayu.server.auth import AuthContext
    from cayu.server.proxy_identity import ControlPlaneRoles

    roles = ControlPlaneRoles()
    roles.add_api_path("/api")

    class Auth:
        control_plane_roles = roles

        def __call__(self, request):
            if request.headers.get("authorization") != "Bearer test-only":
                raise HTTPException(401, "Authentication required")
            context = AuthContext(subject="reader", claims={"role": role})
            roles.require(request, context)
            return context

    app = _app()
    app.register_environment(Environment(EnvironmentSpec(name="dev/team é%")))
    with TestClient(
        create_server(app, config=ServerConfig(access=AuthenticatedAccess(dependency=Auth())))
    ) as client:
        assert client.get("/api/environments").status_code == 401
        headers = {"authorization": "Bearer test-only"}
        response = client.get("/api/environments/dev%2Fteam%20%C3%A9%25", headers=headers)
        assert response.status_code == 200, response.text
        assert response.json()["environments"][0]["name"] == "dev/team é%"
        assert response.headers["cache-control"] == "private, no-store"
        assert client.get("/api/environments", headers=headers).status_code == 200
        if role == "member":
            assert client.get("/api/system/diagnostics", headers=headers).status_code == 403


def test_environment_mcp_redaction_and_unknown_metadata_do_not_disclose_canaries(capsys, caplog):
    import warnings

    from tests.server.test_server_inventory import CountingArtifactStore

    from cayu.vaults.redaction import SecretRedactor

    canary = "private-environment-canary"

    class Opaque:
        def __repr__(self):
            return canary

        __str__ = __repr__

    class UnknownStore(CountingArtifactStore):
        storage_kind = Opaque()

    app = _app(secret_redactor=SecretRedactor([canary]))
    app.register_environment(
        Environment(
            EnvironmentSpec(name="dev"),
            artifact_store=UnknownStore(),
            mcp_servers=[McpServerSpec(name=canary, command=[canary], env={"SECRET": canary})],
        )
    )
    with (
        warnings.catch_warnings(record=True) as caught,
        TestClient(create_server(app, config=ServerConfig.local_development())) as client,
    ):
        response = client.get("/api/environments")
        assert response.status_code == 200, response.text
        row = response.json()["environments"][0]
        assert row["artifact_store_kind"] == "unknown"
        assert row["mcp_server_count"] == 1
        assert len(row["mcp_server_names"]) == 1
        assert canary not in response.text
    output = capsys.readouterr()
    assert canary not in output.out + output.err + caplog.text + "".join(
        str(w.message) for w in caught
    )


def test_environment_counts_include_real_eval_and_fork_sessions():
    from tests.core.test_cloud_reporting import _drain, _start

    from cayu.sessions import ForkSessionRequest
    from cayu.sessions._run_identity import eval_trial_scope

    app = _app()
    app.register_environment(Environment(EnvironmentSpec(name="dev")), default=True)

    async def seed():
        await _start(app, "normal", "writer")
        async with eval_trial_scope():
            await _start(app, "trial", "writer")
        await _drain(
            app.fork_session(ForkSessionRequest(source_session_id="normal", session_id="fork"))
        )

    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        assert client.portal is not None
        client.portal.call(seed)
        response = client.get("/api/environments")
        assert response.status_code == 200, response.text
        assert response.json()["environments"][0]["session_count"] == "3"


def test_environment_empty_catalog_does_not_query_activity(monkeypatch):
    app = _app()

    async def unexpected(query):
        raise AssertionError("Empty inventory must not read session state")

    monkeypatch.setattr(app.session_store, "aggregate_environment_activity", unexpected)
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        response = client.get("/api/environments")
        assert response.status_code == 200
        assert response.json() == {"environments": [], "total_count": 0}


@pytest.mark.parametrize("capability", [1, "yes", None])
def test_environment_activity_capability_requires_positive_boolean(capability, monkeypatch):
    app = _app()
    monkeypatch.setattr(app.session_store, "supports_environment_activity", capability)
    with pytest.raises(TypeError, match="environment_activity_supported must be a bool"):
        create_server(app, config=ServerConfig.local_development())
