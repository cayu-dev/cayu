"""The local deploy check `cayu cloud deploy` and `cayu cloud init` run for public services.

A service scaffold reads its access configuration from the environment and falls back to
placeholder access. `cayu serve` refuses to start that on Cloud, so the deploy must refuse
to upload it before anything leaves the machine.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from cayu.cli import cloud as cloud_cli
from cayu.cli._cloud_deploy_check import run_cloud_deploy_check

_SERVICE_MODULE = """
import os

from fastapi import HTTPException, Request

from cayu import AgentSpec, CayuApp, ScriptedModelProvider, SQLiteSessionStore, SQLiteTaskStore
from cayu.server import (
    AuthenticatedAccess,
    AuthenticatedProductAccess,
    PlaceholderOperatorAccess,
    PlaceholderProductAccess,
    ServiceIdentityStoreKind,
    create_agent_service,
)

print("factory output that must not reach the CLI's JSON stdout")


class Store:
    category = ServiceIdentityStoreKind.DURABLE

    async def reserve(self, **kwargs):
        raise AssertionError

    async def find(self, **kwargs):
        raise AssertionError

    async def find_by_session_id(self, **kwargs):
        raise AssertionError

    async def claim_execution(self, **kwargs):
        raise AssertionError

    async def heartbeat_execution(self, **kwargs):
        raise AssertionError

    async def release_execution(self, **kwargs):
        raise AssertionError

    async def record_result_receipt(self, **kwargs):
        raise AssertionError

    async def record_recovery_status(self, **kwargs):
        raise AssertionError

    async def finish(self, **kwargs):
        raise AssertionError


async def authenticate(request: Request):
    raise HTTPException(status_code=401)


def build_app():
    app = CayuApp(
        session_store=SQLiteSessionStore("runtime.db"),
        task_store=SQLiteTaskStore("runtime.db"),
        enable_logging=False,
    )
    app.register_provider(ScriptedModelProvider([]), default=True)
    app.register_agent(AgentSpec(name="agent", model="scripted-model"))
    return app


def build_service(*, mode, project_context=None):
    configured = os.environ.get("CLOUD_CHECK_TEST_ACCESS") == "configured"
    return create_agent_service(
        build_app(),
        agent_name="agent",
        mode=mode,
        project_context=project_context,
        product_access=(
            AuthenticatedProductAccess(dependency=authenticate)
            if configured
            else PlaceholderProductAccess()
        ),
        operator_access=(
            AuthenticatedAccess(dependency=authenticate)
            if configured
            else PlaceholderOperatorAccess()
        ),
        product_store=Store(),
    )
"""


def _service_project(root: Path, *, module: str = _SERVICE_MODULE) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(
        """[project]
name = "check-service"
version = "0.1.0"
dependencies = ["cayu[server]"]

[tool.cayu]
factory = "cloud_check_service:build_app"
service_factory = "cloud_check_service:build_service"
""",
        encoding="utf-8",
    )
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (root / "cloud_check_service.py").write_text(module, encoding="utf-8")
    return root


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("CLOUD_CHECK_TEST_ACCESS", raising=False)
    monkeypatch.setenv("CAYU_CLOUD_EVIDENCE_DIR", str(tmp_path / "evidence"))
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "cloud-config.json"))
    sys.modules.pop("cloud_check_service", None)
    yield
    sys.modules.pop("cloud_check_service", None)


def test_placeholder_access_fails_with_the_check_diagnostics_and_anchors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    check = run_cloud_deploy_check(_service_project(tmp_path / "svc"), serves_web=True)

    assert check.status == "failed"
    assert check.codes == (
        "PUBLIC_SERVICE_OPERATOR_ACCESS_UNSAFE",
        "PUBLIC_SERVICE_PRODUCT_ACCESS_UNSAFE",
    )
    blocking = check.public_dict()["blocking"]
    assert blocking[1] == {
        "code": "PUBLIC_SERVICE_PRODUCT_ACCESS_UNSAFE",
        "documentation_anchor": "cayu guide diagnostics#public-service-product-access-unsafe",
        "hint": "Configure AuthenticatedProductAccess with trusted tenant resolution.",
        "message": "The product API does not use configured production authentication.",
        "path": "service.product_access",
        "severity": "error",
    }
    assert capsys.readouterr().out == ""


def test_configured_access_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUD_CHECK_TEST_ACCESS", "configured")

    check = run_cloud_deploy_check(_service_project(tmp_path / "svc"), serves_web=True)

    assert check.status == "passed"
    assert check.public_dict() == {
        "command": "cayu check --deploy --fail-on warning --json",
        "status": "passed",
    }


def test_only_a_served_public_service_in_the_uploaded_directory_is_checked(
    tmp_path: Path,
) -> None:
    project = _service_project(tmp_path / "svc")
    assert run_cloud_deploy_check(project, serves_web=False).status == "not_applicable"

    nested = project / "nested"
    nested.mkdir()
    assert run_cloud_deploy_check(nested, serves_web=True).status == "not_applicable"

    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "pyproject.toml").write_text(
        '[tool.cayu]\nfactory = "cloud_check_service:build_app"\n', encoding="utf-8"
    )
    assert run_cloud_deploy_check(plain, serves_web=True).status == "not_applicable"


def test_a_project_that_cannot_boot_here_is_unavailable_not_failed(tmp_path: Path) -> None:
    project = _service_project(
        tmp_path / "svc", module="import dependency_missing_from_this_environment\n"
    )

    check = run_cloud_deploy_check(project, serves_web=True)

    assert check.status == "unavailable"
    assert check.reason is not None
    assert "dependency_missing_from_this_environment" in check.reason


def _cloud(arguments: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, Any]:
    code = cloud_cli.run_cloud_cli(arguments)
    return code, json.loads(capsys.readouterr().out)


def _initialized(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    project = _service_project(tmp_path / "svc")
    code, _ = _cloud(["init", str(project)], capsys)
    assert code == 0
    return project


def test_deploy_refuses_before_any_cloud_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project = _initialized(tmp_path, capsys)

    def no_cloud(*args: object, **kwargs: object) -> object:
        raise AssertionError("the deploy check must refuse before contacting Cayu Cloud")

    monkeypatch.setattr(cloud_cli, "_cloud_client", no_cloud)

    code, payload = _cloud(["deploy", str(project)], capsys)

    assert code == 2
    error = payload["error"]
    assert error["category"] == "deploy_check_failed"
    assert "PUBLIC_SERVICE_PRODUCT_ACCESS_UNSAFE" in error["message"]
    assert "Nothing was uploaded" in error["message"]
    assert "--skip-deploy-check" in error["message"]
    assert [item["code"] for item in error["deploy_check"]["blocking"]] == [
        "PUBLIC_SERVICE_OPERATOR_ACCESS_UNSAFE",
        "PUBLIC_SERVICE_PRODUCT_ACCESS_UNSAFE",
    ]
    assert error["deploy_check"]["command"] == "cayu check --deploy --fail-on warning --json"
    assert "environment" in error["hint"]


@pytest.mark.parametrize(
    "arguments,environment,expected",
    [
        (["--skip-deploy-check"], None, {"status": "skipped"}),
        (
            [],
            "configured",
            {"command": "cayu check --deploy --fail-on warning --json", "status": "passed"},
        ),
    ],
)
def test_deploy_proceeds_when_the_check_passes_or_is_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    environment: str | None,
    expected: dict[str, object],
) -> None:
    project = _initialized(tmp_path, capsys)
    if environment is not None:
        monkeypatch.setenv("CLOUD_CHECK_TEST_ACCESS", environment)
    deployed: dict[str, object] = {}
    monkeypatch.setattr(cloud_cli, "_cloud_client", lambda *args, **kwargs: object())

    def deploy(arguments: object, **kwargs: object) -> dict[str, object]:
        deployed.update(kwargs)
        return {"operation": "deploy", "result": {}}

    monkeypatch.setattr(cloud_cli, "_deploy", deploy)

    code, _ = _cloud(["deploy", str(project), *arguments], capsys)

    assert code == 0
    assert deployed["deploy_check"] == expected


def test_init_mentions_the_check_for_a_public_service_with_placeholder_access(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = _service_project(tmp_path / "svc")

    code, payload = _cloud(["init", str(project)], capsys)

    assert code == 0
    result = payload["result"]
    assert result["runtime"] == "web"
    assert result["deploy_check"]["status"] == "failed"
    (warning,) = result["warnings"]
    assert "PUBLIC_SERVICE_PRODUCT_ACCESS_UNSAFE" in warning
    assert "access is still a placeholder" in warning
    assert "`cayu check --deploy --fail-on warning --json`" in warning


def test_init_stays_quiet_when_the_public_service_is_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("CLOUD_CHECK_TEST_ACCESS", "configured")
    project = _service_project(tmp_path / "svc")

    code, payload = _cloud(["init", str(project)], capsys)

    assert code == 0
    assert payload["result"]["deploy_check"]["status"] == "passed"
    assert "warnings" not in payload["result"]


@pytest.mark.parametrize("configured", [False, True])
def test_source_only_report_is_unavailable_not_passed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, configured: bool
) -> None:
    if configured:
        monkeypatch.setenv("CLOUD_CHECK_TEST_ACCESS", "configured")
    project = _service_project(tmp_path / "svc")
    with (project / "pyproject.toml").open("a") as stream:
        stream.write("\n[tool.cayu.scaffold]\nconvention = 999\n")

    check = run_cloud_deploy_check(project, serves_web=True)

    assert check.status == "unavailable"
    assert "SCAFFOLD_CONVENTION_UNSUPPORTED" in (check.reason or "")
    assert "cloud_check_service" not in sys.modules


def test_missing_readme_does_not_hide_unsafe_service_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cayu.cli import main

    monkeypatch.delenv("PRODUCT_AUTH_TOKENS_JSON", raising=False)
    monkeypatch.delenv("CAYU_OPERATOR_BEARER_TOKEN", raising=False)
    assert main(["new", "service", "--preset", "service", "--dir", str(tmp_path)]) == 0
    project = tmp_path / "service"
    (project / "README.md").unlink()

    check = run_cloud_deploy_check(project, serves_web=True)

    assert check.status == "failed"
    assert set(check.codes) >= {
        "PUBLIC_SERVICE_OPERATOR_ACCESS_UNSAFE",
        "PUBLIC_SERVICE_PRODUCT_ACCESS_UNSAFE",
    }
