from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, ValidationError

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

from cayu import CayuApp
from cayu.runtime.sessions import SessionStatus
from cayu.server import (
    AuthenticatedAccess,
    BasicAuth,
    CorsConfig,
    DashboardConfig,
    DocsConfig,
    OpenAccess,
    ServerApiConfig,
    ServerConfig,
    ServerLifecycleConfig,
    create_server,
    mount_cayu,
    mount_dashboard,
)


def _auth(_request):
    return {"subject": "operator"}


def _header_auth(request: Request):
    if request.headers.get("Authorization") != "Bearer valid":
        raise HTTPException(status_code=401, detail="unauthorized")
    return {"subject": "operator"}


def test_server_config_requires_an_explicit_access_policy() -> None:
    with pytest.raises(ValidationError, match="access"):
        ServerConfig()

    assert ServerConfig(access=OpenAccess()).deployment_name == "development"
    with pytest.raises(ValidationError, match="deployment_name"):
        ServerConfig(deployment_name="", access=OpenAccess())


@pytest.mark.parametrize(
    "deployment_name",
    ["development", "production", "preprod-eu", "alice-local"],
)
def test_deployment_name_is_arbitrary_metadata_only(deployment_name: str) -> None:
    config = ServerConfig(deployment_name=deployment_name, access=OpenAccess())

    summary = config.safe_summary()

    assert summary["deployment_name"] == deployment_name
    summary.pop("deployment_name")
    baseline = ServerConfig(access=OpenAccess()).safe_summary()
    baseline.pop("deployment_name")
    assert summary == baseline


def test_local_development_profile_selects_every_relaxed_policy_explicitly() -> None:
    config = ServerConfig.local_development(deployment_name="developer-laptop")

    assert config.deployment_name == "developer-laptop"
    assert isinstance(config.access, OpenAccess)
    assert config.docs.enabled is True
    assert config.cors.allowed_origins == ("http://localhost:5173",)


def test_protected_profile_preserves_custom_auth_without_serializing_it() -> None:
    config = ServerConfig.protected(_auth, deployment_name="production-eu")

    assert isinstance(config.access, AuthenticatedAccess)
    assert config.access.dependency is _auth
    assert "_auth" not in repr(config)
    assert "dependency" not in config.model_dump()["access"]
    assert "dependency" not in config.model_dump_json()
    assert config.safe_summary()["access"] == "authenticated"


def test_dashboard_may_use_a_distinct_explicit_access_policy() -> None:
    config = ServerConfig(
        access=AuthenticatedAccess(dependency=_auth),
        dashboard=DashboardConfig(access=OpenAccess()),
    )

    assert config.safe_summary()["dashboard"]["access"] == "open"


def test_dashboard_runtime_config_is_owned_immutable_and_serializable() -> None:
    source = {"theme": {"name": "dark"}, "features": ["tasks"]}
    config = ServerConfig(
        access=OpenAccess(),
        dashboard=DashboardConfig(runtime_config=source),
    )
    source["theme"]["name"] = "changed"
    source["features"].append("unsafe")

    assert config.dashboard.runtime_config["theme"]["name"] == "dark"
    assert list(config.dashboard.runtime_config["features"]) == ["tasks"]
    with pytest.raises(TypeError):
        config.dashboard.runtime_config["new"] = True
    dumped = json.loads(config.model_dump_json())
    assert dumped["dashboard"]["runtime_config"] == {
        "theme": {"name": "dark"},
        "features": ["tasks"],
    }


def test_dashboard_runtime_config_resolves_pydantic_models_before_freezing() -> None:
    class RuntimeValue(BaseModel):
        name: str

    config = DashboardConfig(runtime_config={"value": RuntimeValue(name="demo")})

    assert dict(config.runtime_config["value"]) == {"name": "demo"}
    with pytest.raises(ValidationError, match="JSON-serializable"):
        DashboardConfig(runtime_config={"value": object()})


def test_dashboard_runtime_config_rejects_cyclic_values() -> None:
    runtime_config = {}
    runtime_config["self"] = runtime_config

    with pytest.raises(ValidationError, match="JSON-serializable"):
        DashboardConfig(runtime_config=runtime_config)


@pytest.mark.parametrize("directory", ["", " "])
def test_dashboard_directory_rejects_blank_strings(directory: str) -> None:
    with pytest.raises(ValidationError, match="dashboard.directory"):
        DashboardConfig(directory=directory)


def test_server_paths_are_normalized_and_conflicts_fail_during_validation() -> None:
    config = ServerConfig(
        access=OpenAccess(),
        api=ServerApiConfig(path="cayu/api/"),
        dashboard=DashboardConfig(path="cayu"),
    )

    assert config.api.path == "/cayu/api"
    assert config.dashboard.path == "/cayu"

    with pytest.raises(ValidationError, match="must not live inside"):
        ServerConfig(
            access=OpenAccess(),
            api=ServerApiConfig(path="/api"),
            dashboard=DashboardConfig(path="/api/dashboard"),
        )


@pytest.mark.parametrize("field_name", ["deployment_name", "title"])
def test_server_identity_rejects_non_scalar_text(field_name: str) -> None:
    with pytest.raises(ValidationError, match="Unicode surrogate"):
        ServerConfig(
            access=OpenAccess(),
            **{field_name: f"bad{chr(0xD800)}value"},
        )


def test_server_identity_with_unicode_scalar_text_serializes_and_serves_openapi() -> None:
    config = ServerConfig(
        deployment_name="production-Бишкек",
        title="Cayu Бишкек",
        access=OpenAccess(),
        docs=DocsConfig(enabled=True),
    )

    assert json.loads(config.model_dump_json())["title"] == "Cayu Бишкек"
    response = TestClient(create_server(CayuApp(), config=config)).get("/openapi.json")
    assert response.status_code == 200
    assert response.json()["info"]["title"] == "Cayu Бишкек"


@pytest.mark.parametrize("config_model", [ServerApiConfig, DashboardConfig])
@pytest.mark.parametrize(
    "path",
    [
        "/control/../public",
        "/control/./private",
        "/control//private",
        "//control",
        "/control//",
        "/control\\private",
        "/control/\x1fprivate",
        "/control/%2fprivate",
        "/control/%5Cprivate",
        "/control/%2e%2e/public",
        "/control/%41dmin",
        "/control/%20private",
        "/control/%00private",
    ],
)
def test_server_mount_paths_reject_noncanonical_forms(config_model, path: str) -> None:
    with pytest.raises(ValidationError, match="path"):
        config_model(path=path)


def test_dashboard_requires_the_control_plane_api() -> None:
    with pytest.raises(ValidationError, match="dashboard.enabled requires api.enabled"):
        ServerConfig(access=OpenAccess(), api=ServerApiConfig(enabled=False))

    config = ServerConfig(
        access=OpenAccess(),
        api=ServerApiConfig(enabled=False),
        dashboard=DashboardConfig(enabled=False),
    )
    assert config.api.enabled is False


@pytest.mark.parametrize(
    "dashboard_path",
    ["/docs", "/docs/oauth2-redirect", "/openapi.json", "/redoc"],
)
def test_dashboard_rejects_generated_documentation_route_conflicts(
    dashboard_path: str,
) -> None:
    with pytest.raises(ValidationError, match="generated documentation route"):
        ServerConfig(
            access=OpenAccess(),
            dashboard=DashboardConfig(path=dashboard_path),
            docs=DocsConfig(enabled=True),
        )


@pytest.mark.parametrize(
    "policy",
    [
        {"api": ServerApiConfig(enabled=False)},
        {
            "api": ServerApiConfig(path="/api"),
            "dashboard": DashboardConfig(path="/api/dashboard"),
        },
        {
            "dashboard": DashboardConfig(path="/docs"),
            "docs": DocsConfig(enabled=True),
        },
    ],
    ids=["disabled-api", "nested-dashboard", "documentation-route"],
)
def test_server_relationship_errors_redact_resolved_authentication(policy) -> None:
    secret = "top-secret"
    access = AuthenticatedAccess(dependency=BasicAuth(username="operator", password=secret))

    with pytest.raises(ValidationError) as exc_info:
        ServerConfig(access=access, **policy)

    errors = exc_info.value.errors()
    assert all(error["input"] is None for error in errors)
    retained_errors = [error["ctx"]["error"] for error in errors]
    assert all(error.__traceback__ is None for error in retained_errors)
    assert secret not in str(exc_info.value)
    assert secret not in repr(errors)


def test_cors_and_lifecycle_configuration_are_finite_and_immutable() -> None:
    config = ServerConfig(
        access=OpenAccess(),
        cors=CorsConfig(
            allowed_origins=["https://control.example.com"],
            allow_methods=["GET", "POST"],
        ),
        lifecycle=ServerLifecycleConfig(
            replay_idle_timeout_s=12,
            startup_recovery_statuses={SessionStatus.INTERRUPTING},
            recovery_inactive_after_seconds=20,
            event_side_effect_startup_timeout_seconds=4,
            interruption_shutdown_grace_seconds=3,
        ),
    )

    assert config.cors.allowed_origins == ("https://control.example.com",)
    assert config.cors.allow_methods == ("GET", "POST")
    assert config.safe_summary()["cors"]["allow_methods"] == ["GET", "POST"]
    assert config.lifecycle.startup_recovery_statuses == frozenset({SessionStatus.INTERRUPTING})
    with pytest.raises(ValidationError, match="finite positive"):
        ServerLifecycleConfig(replay_idle_timeout_s=float("inf"))
    with pytest.raises(ValidationError, match="non-negative integer"):
        ServerLifecycleConfig(recovery_inactive_after_seconds=True)
    with pytest.raises(ValidationError, match="must not be empty"):
        ServerLifecycleConfig(startup_recovery_statuses=set())
    with pytest.raises(ValidationError, match="unsupported recovery status"):
        ServerLifecycleConfig(startup_recovery_statuses={SessionStatus.COMPLETED})
    with pytest.raises(ValidationError, match="cannot be combined"):
        CorsConfig(allowed_origins=("*",), allow_credentials=True)


@pytest.mark.parametrize(
    ("field_name", "entry", "error"),
    [
        ("allow_methods", "GET\r\nX-Evil: yes", "valid HTTP tokens"),
        ("allow_methods", "GET /admin", "valid HTTP tokens"),
        ("allow_headers", "X-Trace: injected", "valid HTTP tokens"),
        ("allow_headers", "X Trace", "valid HTTP tokens"),
        ("allowed_origins", "https://example.com\r\nX-Evil: yes", "control characters"),
    ],
)
def test_cors_rejects_values_that_cannot_cross_http_boundaries(
    field_name: str,
    entry: str,
    error: str,
) -> None:
    with pytest.raises(ValidationError, match=error):
        CorsConfig(**{field_name: [entry]})


def test_cors_headers_cross_a_real_h11_response_boundary() -> None:
    import socket
    import threading
    import time

    import httpx
    import uvicorn

    config = ServerConfig(
        access=AuthenticatedAccess(
            dependency=BasicAuth(
                username="operator",
                password="secret-password",
                realm='Operations "blue" \\ realm',
            )
        ),
        cors=CorsConfig(
            allowed_origins=("https://control.example.com",),
            allow_methods=("GET", "PATCH"),
            allow_headers=("X-Trace-Id",),
        ),
    )
    server_app = create_server(CayuApp(), config=config)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    host, port = listener.getsockname()
    server = uvicorn.Server(
        uvicorn.Config(
            server_app,
            http="h11",
            lifespan="off",
            log_level="critical",
        )
    )
    thread = threading.Thread(
        target=server.run,
        kwargs={"sockets": [listener]},
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)

    try:
        assert server.started
        with httpx.Client(timeout=5, trust_env=False) as client:
            response = client.options(
                f"http://{host}:{port}/api/health",
                headers={
                    "Origin": "https://control.example.com",
                    "Access-Control-Request-Method": "PATCH",
                    "Access-Control-Request-Headers": "X-Trace-Id",
                },
            )
            assert response.status_code == 200
            assert response.headers["access-control-allow-methods"] == "GET, PATCH"
            assert "x-trace-id" in response.headers["access-control-allow-headers"].lower()
            denied = client.get(f"http://{host}:{port}/api/sessions")
            assert denied.status_code == 401
            assert (
                denied.headers["www-authenticate"]
                == 'Basic realm="Operations \\"blue\\" \\\\ realm"'
            )
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        listener.close()
    assert not thread.is_alive()


def test_server_config_rejects_unknown_or_implicit_policy_values() -> None:
    with pytest.raises(ValidationError, match="callable auth dependency") as exc_info:
        AuthenticatedAccess(dependency="token")
    assert "token" not in str(exc_info.value)

    invalid_dependency = {"token": "top-secret"}
    with pytest.raises(ValidationError, match="callable auth dependency") as exc_info:
        AuthenticatedAccess(dependency=invalid_dependency)
    assert exc_info.value.errors()[0]["input"] is None
    assert "top-secret" not in repr(exc_info.value.errors())

    with pytest.raises(ValidationError, match="callable auth dependency") as exc_info:
        ServerConfig(
            access={
                "kind": "authenticated",
                "dependency": invalid_dependency,
            }
        )
    assert exc_info.value.errors()[0]["input"] is None
    assert "top-secret" not in repr(exc_info.value.errors())

    with pytest.raises(ValidationError):
        DocsConfig(enabled=1)
    with pytest.raises(ValidationError):
        CorsConfig(allowed_origins="https://control.example.com")
    with pytest.raises(ValidationError):
        ServerConfig(access={"kind": "unknown"})


def test_access_policy_validation_redacts_every_invalid_shape() -> None:
    secret = "top-secret"
    invalid_access_values = [
        lambda: OpenAccess(password={"token": secret}),
        lambda: AuthenticatedAccess(
            dependency=_auth,
            password={"token": secret},
        ),
        lambda: DashboardConfig(
            access={
                "kind": "authenticated",
                "dependency": _auth,
                "password": {"token": secret},
            }
        ),
        lambda: DashboardConfig(
            access=AuthenticatedAccess(dependency=_auth),
            password={"token": secret},
        ),
        lambda: ServerConfig(
            access={
                "kind": "authenticated",
                "dependency": _auth,
                "password": {"token": secret},
            }
        ),
        lambda: ServerConfig(
            access={
                "kind": secret,
                "dependency": {"token": secret},
            }
        ),
        lambda: ServerConfig(
            dashboard={
                "access": {
                    "kind": secret,
                    "dependency": {"token": secret},
                }
            }
        ),
    ]

    for construct in invalid_access_values:
        with pytest.raises(ValidationError) as exc_info:
            construct()
        errors = exc_info.value.errors()
        assert all(error["input"] is None for error in errors)
        assert secret not in str(exc_info.value)
        assert secret not in repr(errors)


def test_create_server_uses_resolved_config_as_its_stable_source_of_truth() -> None:
    config = ServerConfig.local_development(deployment_name="qa-eu")
    server = create_server(CayuApp(), config=config)
    client = TestClient(server)

    assert server.state.cayu_server_config is config
    assert server.state.cayu_server_config_summary == config.safe_summary()
    assert client.get("/api/health").json() == {"ok": True}
    assert client.get("/openapi.json").status_code == 200
    cors = client.options(
        "/api/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert cors.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_protected_config_preserves_custom_auth_and_disables_docs_by_default() -> None:
    config = ServerConfig.protected(_header_auth, deployment_name="production-eu")
    client = TestClient(create_server(CayuApp(), config=config))

    assert client.get("/api/sessions").status_code == 401
    assert client.get("/api/sessions", headers={"Authorization": "Bearer valid"}).status_code == 200
    assert client.get("/openapi.json").status_code == 404


def test_dashboard_access_can_be_stricter_than_open_api_access() -> None:
    config = ServerConfig(
        access=OpenAccess(),
        dashboard=DashboardConfig(
            access=AuthenticatedAccess(dependency=_header_auth),
        ),
    )
    client = TestClient(create_server(CayuApp(), config=config))

    assert client.get("/api/health").status_code == 200
    assert client.get("/cayu/").status_code == 401
    assert client.get("/cayu/", headers={"Authorization": "Bearer valid"}).status_code == 200


def test_api_and_dashboard_can_be_disabled_explicitly() -> None:
    config = ServerConfig(
        access=OpenAccess(),
        api=ServerApiConfig(enabled=False),
        dashboard=DashboardConfig(enabled=False),
    )
    client = TestClient(create_server(CayuApp(), config=config))

    assert client.get("/api/health").status_code == 404
    assert client.get("/cayu/").status_code == 404


def test_deployment_name_does_not_change_server_behavior() -> None:
    statuses = []
    for name in ("development", "production", "preprod-eu", "alice-local"):
        config = ServerConfig(
            deployment_name=name,
            access=OpenAccess(),
            docs=DocsConfig(enabled=True),
        )
        client = TestClient(create_server(CayuApp(), config=config))
        statuses.append(
            (
                client.get("/api/health").status_code,
                client.get("/openapi.json").status_code,
                client.get("/cayu/").status_code,
            )
        )

    assert statuses == [(200, 200, 200)] * 4


def test_create_server_accepts_only_a_resolved_server_config() -> None:
    with pytest.raises(TypeError, match="resolved ServerConfig"):
        create_server(CayuApp(), config={"access": {"kind": "open"}})


def test_create_server_rejects_removed_or_unknown_fastapi_options() -> None:
    config = ServerConfig(access=OpenAccess())

    with pytest.raises(TypeError, match="unexpected keyword argument 'auth'"):
        create_server(CayuApp(), config=config, auth=_auth)
    with pytest.raises(ValueError, match="Unsupported or policy-bearing FastAPI options: auth"):
        create_server(CayuApp(), config=config, fastapi_options={"auth": _auth})
    with pytest.raises(ValueError, match="policy-bearing FastAPI options: openapi_url"):
        create_server(CayuApp(), config=config, fastapi_options={"openapi_url": None})
    with pytest.raises(ValueError, match="policy-bearing FastAPI options: debug"):
        create_server(CayuApp(), config=config, fastapi_options={"debug": True})


@pytest.mark.parametrize(
    ("option_name", "value"),
    [
        ("routes", []),
        ("strict_content_type", False),
        ("swagger_ui_oauth2_redirect_url", "/cayu"),
    ],
)
def test_create_server_rejects_fastapi_options_that_can_change_cayu_policy(
    option_name: str,
    value,
) -> None:
    with pytest.raises(ValueError, match=f"policy-bearing FastAPI options: {option_name}"):
        create_server(
            CayuApp(),
            config=ServerConfig(access=OpenAccess()),
            fastapi_options={option_name: value},
        )


def test_create_server_applies_valid_non_policy_fastapi_options() -> None:
    server = create_server(
        CayuApp(),
        config=ServerConfig(access=OpenAccess()),
        fastapi_options={"root_path": "/proxy"},
    )

    assert server.root_path == "/proxy"


@pytest.mark.parametrize("directory_state", ["missing", "empty"])
def test_create_server_fails_when_enabled_dashboard_assets_are_unavailable(
    tmp_path,
    directory_state: str,
) -> None:
    dashboard_directory = tmp_path / "dashboard"
    if directory_state == "empty":
        dashboard_directory.mkdir()
    config = ServerConfig(
        access=OpenAccess(),
        dashboard=DashboardConfig(directory=dashboard_directory),
    )

    with pytest.raises(RuntimeError, match="Dashboard assets are unavailable") as exc_info:
        create_server(CayuApp(), config=config)

    assert str(dashboard_directory) in str(exc_info.value)
    assert "index.html" in str(exc_info.value)


def test_mount_cayu_fails_when_requested_dashboard_assets_are_unavailable(tmp_path) -> None:
    from fastapi import FastAPI

    server = FastAPI()
    routes_before = list(server.routes)
    lifespan_before = server.router.lifespan_context
    with pytest.raises(RuntimeError, match="Set dashboard=False"):
        mount_cayu(
            server,
            CayuApp(),
            dashboard_dir=tmp_path / "missing-dashboard",
            access=OpenAccess(),
        )

    assert server.routes == routes_before
    assert server.router.lifespan_context is lifespan_before


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"path": ""}, "path"),
        ({"path": "https://example.com/control"}, "path"),
        ({"path": "/control "}, "path"),
        ({"replay_idle_timeout_s": 0}, "replay_idle_timeout_s"),
        ({"replay_idle_timeout_s": float("inf")}, "replay_idle_timeout_s"),
        ({"replay_idle_timeout_s": float("nan")}, "replay_idle_timeout_s"),
    ],
)
def test_mount_cayu_validation_failure_does_not_modify_host_application(
    kwargs,
    error: str,
) -> None:
    from fastapi import FastAPI

    server = FastAPI()
    routes_before = list(server.routes)
    lifespan_before = server.router.lifespan_context

    with pytest.raises(ValueError, match=error):
        mount_cayu(
            server,
            CayuApp(),
            dashboard=False,
            access=OpenAccess(),
            **kwargs,
        )

    assert server.routes == routes_before
    assert server.router.lifespan_context is lifespan_before


@pytest.mark.parametrize("adapter", ["mount_cayu", "mount_dashboard"])
@pytest.mark.parametrize("invalid_config_kind", ["cyclic", "non_object"])
def test_dashboard_mounts_reject_invalid_runtime_config_before_modifying_host(
    adapter: str,
    invalid_config_kind: str,
) -> None:
    from fastapi import FastAPI

    if invalid_config_kind == "cyclic":
        runtime_config = {}
        runtime_config["self"] = runtime_config
    else:
        runtime_config = object()
    server = FastAPI()
    routes_before = list(server.routes)
    lifespan_before = server.router.lifespan_context

    with pytest.raises(ValueError, match="dashboard_config"):
        if adapter == "mount_cayu":
            mount_cayu(
                server,
                CayuApp(),
                access=OpenAccess(),
                dashboard_config=runtime_config,
            )
        else:
            mount_dashboard(server, dashboard_config=runtime_config)

    assert server.routes == routes_before
    assert server.router.lifespan_context is lifespan_before


def test_dashboard_mount_owns_runtime_config_before_browser_injection() -> None:
    from fastapi import FastAPI

    runtime_config = {"theme": {"name": "dark"}, "features": ["tasks"]}
    server = FastAPI()
    assert mount_dashboard(server, dashboard_config=runtime_config) is True

    runtime_config["theme"]["name"] = "changed"
    runtime_config["features"].append("unsafe")
    response = TestClient(server).get("/cayu/")

    assert response.status_code == 200
    marker = "window.__CAYU_DASHBOARD_CONFIG__="
    injected_config = json.loads(response.text.split(marker, 1)[1].split(";</script>", 1)[0])
    assert injected_config["theme"] == {"name": "dark"}
    assert injected_config["features"] == ["tasks"]


@pytest.mark.parametrize(
    "dashboard_path",
    ["https://example.com/control", " /control", "/control "],
)
def test_mount_dashboard_rejects_non_path_or_padded_mount_values(dashboard_path: str) -> None:
    from fastapi import FastAPI

    server = FastAPI()
    routes_before = list(server.routes)

    with pytest.raises(ValueError, match="dashboard_path"):
        mount_dashboard(server, dashboard_path=dashboard_path)

    assert server.routes == routes_before


def test_dashboard_mounting_helpers_reject_blank_directory_strings() -> None:
    from fastapi import FastAPI

    with pytest.raises(ValueError, match="dashboard_dir"):
        mount_dashboard(FastAPI(), dashboard_dir="")
    with pytest.raises(ValueError, match="dashboard_dir"):
        mount_cayu(
            FastAPI(),
            CayuApp(),
            dashboard_dir="",
            access=OpenAccess(),
        )


def test_mount_cayu_accepts_explicit_open_or_authenticated_access() -> None:
    from fastapi import FastAPI

    open_server = FastAPI()
    mount_cayu(open_server, CayuApp(), path="/control", access=OpenAccess())
    assert TestClient(open_server).get("/control/api/health").status_code == 200

    protected_server = FastAPI()
    mount_cayu(
        protected_server,
        CayuApp(),
        path="/control",
        access=AuthenticatedAccess(dependency=_header_auth),
    )
    protected = TestClient(protected_server)
    assert protected.get("/control/api/sessions").status_code == 401
    assert (
        protected.get(
            "/control/api/sessions",
            headers={"Authorization": "Bearer valid"},
        ).status_code
        == 200
    )


def test_mount_cayu_rejects_an_unresolved_access_policy() -> None:
    from fastapi import FastAPI

    with pytest.raises(TypeError, match="OpenAccess or AuthenticatedAccess"):
        mount_cayu(FastAPI(), CayuApp(), access={"kind": "open"})
