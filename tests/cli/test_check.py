from __future__ import annotations

import json
import sys
from pathlib import Path

from cayu.cli import main


def test_deploy_check_reports_supported_public_service_posture(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """[tool.cayu]
factory = "service_project:build_app"
service_factory = "service_project:build_service"
""",
        encoding="utf-8",
    )
    (tmp_path / "service_project.py").write_text(
        """from fastapi import HTTPException, Request

from cayu import AgentSpec, CayuApp, ScriptedModelProvider, SQLiteSessionStore, SQLiteTaskStore
from cayu.server import (
    AuthenticatedAccess,
    AuthenticatedProductAccess,
    BasicAuth,
    ServiceIdentityStoreKind,
    create_agent_service,
)


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


async def product_auth(request: Request):
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


def build_service(*, mode):
    return create_agent_service(
        build_app(),
        agent_name="agent",
        mode=mode,
        product_access=AuthenticatedProductAccess(dependency=product_auth),
        operator_access=AuthenticatedAccess(
            dependency=BasicAuth(username="operator", password="secret-password")
        ),
        product_store=Store(),
    )
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("service_project", None)

    assert main(["check", "--deploy", "--fail-on", "warning", "--json"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["diagnostics"] == []
    assert report["service_evidence"] == {
        "application_security": "generated_suite_required",
        "configuration": "supported",
        "control_plane_access": "verified_authenticated",
        "host_owned_behavior": "unverified_outside_contract",
        "security_verification_command": "pytest -q tests/test_public_service_security.py",
        "service_contract": "verified_maintained",
    }


def test_deploy_check_emits_stable_diagnostics_for_unsafe_service_manifest(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """[tool.cayu]
factory = "unsafe_service:build_app"
service_factory = "unsafe_service:build_service"
""",
        encoding="utf-8",
    )
    (tmp_path / "unsafe_service.py").write_text(
        """from cayu import AgentSpec, CayuApp, ScriptedModelProvider
from cayu.server import DevelopmentProductAccess, OpenAccess, ServiceIdentityStoreKind, create_agent_service


class Store:
    category = ServiceIdentityStoreKind.DEVELOPMENT

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


async def dev_auth(request):
    return {"tenant_id": "dev", "subject_id": "dev"}


def build_app():
    app = CayuApp(enable_logging=False)
    app.register_provider(ScriptedModelProvider([]), default=True)
    app.register_agent(AgentSpec(name="agent", model="scripted-model"))
    return app


def build_service(*, mode):
    return create_agent_service(
        build_app(),
        agent_name="agent",
        mode="development",
        product_access=DevelopmentProductAccess(dependency=dev_auth),
        operator_access=OpenAccess(),
        product_store=Store(),
    )
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("unsafe_service", None)

    assert main(["check", "--deploy", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert [item["code"] for item in report["diagnostics"]] == [
        "PUBLIC_SERVICE_DEVELOPMENT_MODE",
        "PUBLIC_SERVICE_IDENTITY_STORE_NOT_DURABLE",
        "PUBLIC_SERVICE_OPERATOR_ACCESS_UNSAFE",
        "PUBLIC_SERVICE_PRODUCT_ACCESS_UNSAFE",
        "PUBLIC_SERVICE_SESSION_STORE_NOT_DURABLE",
        "PUBLIC_SERVICE_TASK_STORE_REQUIRED",
    ]
    assert report["service_evidence"]["configuration"] == "unsupported"


def test_deploy_check_rejects_arbitrary_asgi_service_factory_as_unverified(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """[tool.cayu]
factory = "host_app:build_app"
service_factory = "host_app:build_service"
""",
        encoding="utf-8",
    )
    (tmp_path / "host_app.py").write_text(
        """from fastapi import FastAPI
from cayu import CayuApp


def build_app():
    return CayuApp(enable_logging=False)


def build_service(*, mode):
    del mode
    return FastAPI()
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("host_app", None)

    assert main(["check", "--deploy", "--json"]) == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["code"] == "PROJECT_CHECK_FAILED"
    assert "arbitrary ASGI apps are unverified" in error["message"]


def test_deploy_check_rejects_a_tampered_auth_dependency_as_unverified(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """[tool.cayu]
factory = "tampered_service:build_app"
service_factory = "tampered_service:build_service"
""",
        encoding="utf-8",
    )
    (tmp_path / "tampered_service.py").write_text(
        """from cayu import AgentSpec, CayuApp, InMemoryTaskStore, ScriptedModelProvider
from cayu.server import (
    AuthenticatedAccess,
    AuthenticatedProductAccess,
    ServiceIdentityStoreKind,
    create_agent_service,
)


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


async def auth(request):
    return {"tenant_id": "tenant", "subject_id": "subject"}


def build_app():
    app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
    app.register_provider(ScriptedModelProvider([]), default=True)
    app.register_agent(AgentSpec(name="agent", model="scripted-model"))
    return app


def build_service(*, mode):
    service = create_agent_service(
        build_app(),
        agent_name="agent",
        mode=mode,
        product_access=AuthenticatedProductAccess(dependency=auth),
        operator_access=AuthenticatedAccess(dependency=auth),
        product_store=Store(),
    )

    operation_route = next(
        route
        for route in service.asgi_app.routes
        if getattr(route, "path", None) == "/api/operations"
        and "POST" in (getattr(route, "methods", None) or set())
    )
    operation_route.dependant.dependencies[0].call = lambda: {
        "tenant_id": "bypassed",
        "subject_id": "anonymous",
    }

    return service
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("tampered_service", None)

    assert main(["check", "--deploy", "--json"]) == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["code"] == "PROJECT_CHECK_FAILED"
    assert "unmodified result of create_agent_service" in error["message"]


def test_check_json_reports_actionable_provider_and_policy_failures(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.cayu]\nfactory = "broken_project:build_app"\n',
        encoding="utf-8",
    )
    (tmp_path / "broken_project.py").write_text(
        """from cayu import AgentSpec, CayuApp, Tool, ToolEffect, ToolResult, ToolSpec


class SendTool(Tool):
    spec = ToolSpec(
        name="send",
        effect=ToolEffect.EXTERNAL,
        input_schema={"type": "object", "additionalProperties": False},
    )

    async def run(self, ctx, args):
        return ToolResult(content="sent")


def build_app():
    app = CayuApp(enable_logging=False)
    app.register_agent(
        AgentSpec(name="sender", model="missing-model", provider_name="missing"),
        tools=[SendTool()],
    )
    return app
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("broken_project", None)

    assert main(["check"]) == 1

    report = json.loads(capsys.readouterr().out)
    assert report["schema_version"] == "2"
    assert report["manifest_fingerprint"]
    assert report["service_evidence"] == {
        "application_security": "not_evaluated",
        "configuration": "not_applicable",
        "control_plane_access": "not_evaluated",
        "host_owned_behavior": "unverified_outside_contract",
        "security_verification_command": "pytest -q tests/test_public_service_security.py",
        "service_contract": "not_declared",
    }
    assert [item["code"] for item in report["diagnostics"]] == [
        "AGENT_PROVIDER_NOT_FOUND",
        "EXTERNAL_TOOL_UNGUARDED",
    ]
    provider_finding = report["diagnostics"][0]
    assert provider_finding["path"] == "agents.sender.configured_provider"
    assert provider_finding["parameters"] == {"agent": "sender", "provider": "missing"}
    assert provider_finding["hint"] == "Register provider 'missing' or change sender.provider_name."
    assert provider_finding["documentation_anchor"].endswith("#agent-provider-not-found")

    destination = tmp_path / "check.json"
    assert main(["check", "--output", str(destination)]) == 1
    assert json.loads(destination.read_text(encoding="utf-8"))["diagnostics"]


def test_check_json_distinguishes_factory_failure_from_findings(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.cayu]\nfactory = "failed_project:build_app"\n',
        encoding="utf-8",
    )
    (tmp_path / "failed_project.py").write_text(
        'def build_app():\n    raise RuntimeError("boot exploded")\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("failed_project", None)

    assert main(["check", "--json"]) == 2

    error = json.loads(capsys.readouterr().out)["error"]
    assert error == {
        "code": "PROJECT_CHECK_FAILED",
        "message": "Application factory failed (RuntimeError): boot exploded",
    }
