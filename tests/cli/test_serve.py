from __future__ import annotations

import sys
from base64 import b64encode
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

import cayu.cli.serve as serve_cli
from cayu.cli import main
from cayu.cli.scaffold import project_files
from cayu.runtime.sessions import SessionStatus


def test_serve_missing_project_guidance_matches_supported_cli(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)

    assert main(["serve", "--dev"]) == 1

    error = capsys.readouterr().err
    assert 'Add [tool.cayu] factory = "module:build_app" to pyproject.toml' in error
    assert "cayu serve module:build_app" not in error


def test_serve_fails_closed_before_building_an_unauthenticated_app(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.cayu]\nfactory = "must_not_build:build_app"\n',
        encoding="utf-8",
    )
    (tmp_path / "must_not_build.py").write_text(
        """def build_app():
    raise AssertionError("secure default must fail before factory invocation")
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("must_not_build", None)

    assert main(["serve"]) == 1

    error = capsys.readouterr().err
    assert "Refusing to start an unauthenticated server" in error
    assert "secure default" not in error
    assert "must_not_build" not in sys.modules


def test_serve_uses_the_maintained_service_factory_for_explicit_local_development(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    for relative, content in project_files("service", template="service").items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    launched: dict[str, Any] = {}
    uvicorn = ModuleType("uvicorn")
    uvicorn.run = lambda server, *, host, port: launched.update(  # type: ignore[attr-defined]
        server=server,
        host=host,
        port=port,
    )
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    monkeypatch.chdir(tmp_path)

    assert main(["serve", "--dev", "--port", "8123"]) == 0

    assert launched["host"] == "127.0.0.1"
    assert launched["port"] == 8123
    manifest = launched["server"].state.cayu_public_service_manifest
    assert manifest.mode == "development"
    assert manifest.product_access == "development"
    assert manifest.operator_access == "open"
    service = launched["server"].state.cayu_public_service
    assert service.project_control_plane_context_attached is True
    summary = launched["server"].state.cayu_project_control_plane_summary
    assert summary["project_id"] == "service"
    assert summary["application_release_id"] == f"manifest-{summary['app_manifest_fingerprint']}"
    assert summary["access"] == "trusted_local_development"
    assert summary["eval_store"] == {
        "configured": True,
        "backend": "sqlite",
        "source": "project",
    }
    with TestClient(launched["server"]) as client:
        readiness = client.get("/cayu/api/contract").json()["capabilities"]["evals_readiness"]
    assert readiness["catalog_read"] == {
        "state": "ready",
        "reason_code": None,
    }
    assert readiness["captured_result_persistence"] == {
        "state": "ready",
        "reason_code": None,
    }
    output = capsys.readouterr().out
    assert "Cayu product service: http://127.0.0.1:8123/api/operations" in output
    assert "Cayu operator control plane: http://127.0.0.1:8123/cayu/" in output


def test_serve_reports_custom_product_and_control_plane_paths(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """[tool.cayu]
factory = "unused:build_app"
service_factory = "unused:build_service"
""",
        encoding="utf-8",
    )
    service = SimpleNamespace(
        asgi_app=object(),
        manifest=SimpleNamespace(
            product_api_path="/product/v1",
            control_plane_path="/operators",
        ),
    )
    monkeypatch.setattr(
        serve_cli,
        "build_project_service",
        lambda _target, *, mode, command, project_context: service,
    )
    launched: dict[str, Any] = {}
    uvicorn = ModuleType("uvicorn")
    uvicorn.run = lambda server, *, host, port: launched.update(  # type: ignore[attr-defined]
        server=server,
        host=host,
        port=port,
    )
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    monkeypatch.chdir(tmp_path)

    assert main(["serve", "--dev", "--host", "::1", "--port", "8124"]) == 0

    assert launched["server"] is service.asgi_app
    assert launched["host"] == "::1"
    output = capsys.readouterr().out
    assert "Cayu product service: http://[::1]:8124/product/v1/operations" in output
    assert "Cayu operator control plane: http://[::1]:8124/operators/" in output


def test_serve_refuses_unsafe_production_service_before_starting_listener(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    for relative, content in project_files("service", template="service").items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    uvicorn = ModuleType("uvicorn")
    uvicorn.run = lambda *args, **kwargs: pytest.fail("unsafe service must not listen")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    monkeypatch.delenv("PRODUCT_AUTH_TOKENS_JSON", raising=False)
    monkeypatch.delenv("CAYU_OPERATOR_BEARER_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)

    assert main(["serve", "--host", "0.0.0.0"]) == 1

    error = capsys.readouterr().err
    assert "Refusing to start an unsafe public service" in error
    assert "PUBLIC_SERVICE_PRODUCT_ACCESS_UNSAFE" in error
    assert "PUBLIC_SERVICE_OPERATOR_ACCESS_UNSAFE" in error


def test_serve_starts_supported_production_service_on_one_listener(
    tmp_path: Path,
    monkeypatch,
) -> None:
    for relative, content in project_files("service", template="service").items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    launched: dict[str, Any] = {}
    uvicorn = ModuleType("uvicorn")
    uvicorn.run = lambda server, *, host, port: launched.update(  # type: ignore[attr-defined]
        server=server,
        host=host,
        port=port,
    )
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    monkeypatch.setenv(
        "PRODUCT_AUTH_TOKENS_JSON",
        '{"customer-token":{"tenant_id":"tenant-a","subject_id":"alice"}}',
    )
    monkeypatch.setenv("CAYU_OPERATOR_BEARER_TOKEN", "operator-token")
    monkeypatch.chdir(tmp_path)

    assert main(["serve", "--host", "0.0.0.0", "--port", "9000"]) == 0

    assert launched["host"] == "0.0.0.0"
    assert launched["port"] == 9000
    manifest = launched["server"].state.cayu_public_service_manifest
    assert manifest.mode == "production"
    assert manifest.product_access == "authenticated"
    assert manifest.operator_access == "authenticated"


def test_serve_discovers_project_and_runs_one_local_process(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    project = tmp_path / "project"
    nested = project / "agents" / "reviewer"
    nested.mkdir(parents=True)
    (project / "pyproject.toml").write_text(
        '[tool.cayu]\nfactory = "serve_project:build_app"\n',
        encoding="utf-8",
    )
    (project / "serve_project.py").write_text(
        """from pathlib import Path

from cayu import CayuApp

build_count = 0


def build_app():
    global build_count
    build_count += 1
    with Path("factory-log.txt").open("a", encoding="utf-8") as log:
        log.write(f"{Path.cwd()}\\n")
    app = CayuApp(enable_logging=False)
    return app
""",
        encoding="utf-8",
    )
    launched: dict[str, Any] = {}
    uvicorn = ModuleType("uvicorn")

    def run(server: Any, *, host: str, port: int) -> None:
        launched.update(server=server, host=host, port=port)

    uvicorn.run = run  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    monkeypatch.chdir(nested)
    sys.modules.pop("serve_project", None)

    assert (
        main(
            [
                "serve",
                "--dev",
                "--port",
                "9123",
            ]
        )
        == 0
    )

    server = launched["server"]
    assert launched["host"] == "127.0.0.1"
    assert launched["port"] == 9123
    assert server.state.cayu_server_config.access.kind == "open"
    assert server.state.cayu_server_config.lifecycle.startup_recovery_statuses is None
    assert (project / "factory-log.txt").read_text(encoding="utf-8") == f"{project}\n"
    assert "serve_project" not in sys.modules
    assert "Cayu control plane: http://127.0.0.1:9123/cayu/" in capsys.readouterr().out


def test_serve_rejects_unauthenticated_dev_on_a_non_loopback_host_before_building(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.cayu]\nfactory = "must_not_build:build_app"\n',
        encoding="utf-8",
    )
    (tmp_path / "must_not_build.py").write_text(
        """def build_app():
    raise AssertionError("unsafe dev bind must fail before factory invocation")
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("must_not_build", None)

    assert main(["serve", "--dev", "--host", "0.0.0.0"]) == 1

    error = capsys.readouterr().err
    assert "Refusing to expose an unauthenticated control plane" in error
    assert "loopback host" in error
    assert "unsafe dev bind" not in error
    assert "must_not_build" not in sys.modules


def test_serve_loads_configured_auth_for_the_control_plane_on_a_public_bind(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """[tool.cayu]
factory = "protected_project:build_app"

[tool.cayu.serve]
auth = "protected_project:AUTH"
""",
        encoding="utf-8",
    )
    (tmp_path / "protected_project.py").write_text(
        """from cayu import CayuApp
from cayu.server import BasicAuth

AUTH = BasicAuth(username="operator", password="secret-password")


def build_app():
    return CayuApp(enable_logging=False)
""",
        encoding="utf-8",
    )
    launched: dict[str, Any] = {}
    uvicorn = ModuleType("uvicorn")
    uvicorn.run = lambda server, *, host, port: launched.update(  # type: ignore[attr-defined]
        server=server,
        host=host,
        port=port,
    )
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("protected_project", None)

    assert main(["serve", "--host", "0.0.0.0"]) == 0

    server = launched["server"]
    assert launched["host"] == "0.0.0.0"
    assert server.state.cayu_server_config.access.kind == "authenticated"
    with TestClient(server) as client:
        assert client.get("/api/health").json() == {"ok": True}
        assert client.get("/api/sessions").status_code == 401
        credentials = b64encode(b"operator:secret-password").decode()
        assert (
            client.get(
                "/api/sessions",
                headers={"Authorization": f"Basic {credentials}"},
            ).status_code
            == 200
        )


def test_serve_applies_only_explicit_bounded_startup_recovery(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """[tool.cayu]
factory = "recovery_project:build_app"

[tool.cayu.serve]
auth = "recovery_project:AUTH"
startup_recovery_statuses = ["pending", "running"]
recovery_inactive_after_seconds = 900
""",
        encoding="utf-8",
    )
    (tmp_path / "recovery_project.py").write_text(
        """from cayu import CayuApp
from cayu.server import BasicAuth

AUTH = BasicAuth(username="operator", password="secret-password")


def build_app():
    return CayuApp(enable_logging=False)
""",
        encoding="utf-8",
    )
    launched: dict[str, Any] = {}
    uvicorn = ModuleType("uvicorn")
    uvicorn.run = lambda server, *, host, port: launched.update(  # type: ignore[attr-defined]
        server=server,
    )
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("recovery_project", None)

    assert main(["serve"]) == 0

    lifecycle = launched["server"].state.cayu_server_config.lifecycle
    assert lifecycle.startup_recovery_statuses == frozenset(
        {SessionStatus.PENDING, SessionStatus.RUNNING}
    )
    assert lifecycle.recovery_inactive_after_seconds == 900


def test_serve_rejects_recovery_statuses_without_an_inactivity_threshold(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """[tool.cayu]
factory = "recovery_project:build_app"

[tool.cayu.serve]
startup_recovery_statuses = ["pending"]
""",
        encoding="utf-8",
    )
    (tmp_path / "recovery_project.py").write_text(
        """def build_app():
    raise AssertionError("invalid recovery configuration must fail before app construction")
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("recovery_project", None)

    assert main(["serve", "--dev"]) == 1
    error = capsys.readouterr().err
    assert "startup_recovery_statuses requires recovery_inactive_after_seconds" in error
    assert "invalid recovery configuration" not in error


def test_serve_reports_project_factory_system_exit_as_startup_error(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.cayu]\nfactory = "exiting_factory:build_app"\n',
        encoding="utf-8",
    )
    (tmp_path / "exiting_factory.py").write_text(
        "def build_app():\n    raise SystemExit(7)\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("exiting_factory", None)

    assert main(["serve", "--dev"]) == 1
    assert "Serve project startup raised SystemExit with status 7" in capsys.readouterr().err


def test_serve_reports_auth_import_system_exit_before_factory_construction(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """[tool.cayu]
factory = "auth_exit_app:build_app"

[tool.cayu.serve]
auth = "exiting_auth:AUTH"
""",
        encoding="utf-8",
    )
    (tmp_path / "auth_exit_app.py").write_text(
        """from pathlib import Path


def build_app():
    Path("factory-ran.txt").write_text("ran\\n", encoding="utf-8")
""",
        encoding="utf-8",
    )
    (tmp_path / "exiting_auth.py").write_text(
        "raise SystemExit(8)\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("auth_exit_app", None)
    sys.modules.pop("exiting_auth", None)

    assert main(["serve"]) == 1
    assert "Serve project startup raised SystemExit with status 8" in capsys.readouterr().err
    assert not (tmp_path / "factory-ran.txt").exists()


def test_serve_reports_uvicorn_startup_failures(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.cayu]\nfactory = "failed_server:build_app"\n',
        encoding="utf-8",
    )
    (tmp_path / "failed_server.py").write_text(
        """from cayu import CayuApp


def build_app():
    return CayuApp(enable_logging=False)
""",
        encoding="utf-8",
    )
    uvicorn = ModuleType("uvicorn")

    def fail_to_start(server: Any, *, host: str, port: int) -> None:
        del server, host, port
        raise OSError("address already in use")

    uvicorn.run = fail_to_start  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("failed_server", None)

    assert main(["serve", "--dev"]) == 1
    assert "address already in use" in capsys.readouterr().err
