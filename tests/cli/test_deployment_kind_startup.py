"""Hosting declarations cross real project factories before diagnostic capture."""

import sys
from types import ModuleType

import pytest
from fastapi.testclient import TestClient

from cayu.cli import main
from cayu.cli.scaffold import project_files


@pytest.mark.parametrize(
    "forward_context,declared,requested,expected",
    [
        (True, None, None, "embedded"),
        (True, "cloud", None, "cloud"),
        (True, "embedded", None, "embedded"),
        (True, "cloud", "cloud", "cloud"),
        (True, "embedded", "embedded", "embedded"),
        (True, "embedded", "cloud", None),
        (True, None, "cloud", "cloud"),
        (False, "cloud", "cloud", "cloud"),
        (False, None, None, "embedded"),
        (False, None, "cloud", None),
        (True, "local", "cloud", None),
    ],
)
def test_serve_declared_kind_is_verified_from_the_returned_service(
    tmp_path, monkeypatch, capsys, forward_context, declared, requested, expected
):
    for relative, content in project_files("service", preset="service").items():
        if relative.endswith("service.py"):
            if not forward_context:
                content = content.replace(
                    "    project_context: ProjectControlPlaneContext | None = None,\n", ""
                )
                content = content.replace("        project_context=project_context,\n", "")
            if declared is not None:
                content = content.replace(
                    "        agent_name=",
                    f"        deployment_kind={declared!r},\n        agent_name=",
                    1,
                )
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    observed = []
    uvicorn = ModuleType("uvicorn")

    def run(server, **kwargs):
        service = server.state.cayu_public_service
        with TestClient(server) as client:
            response = client.get("/cayu/api/system/diagnostics")
            assert response.status_code == 200
            observed.append(response.json()["deployment"]["kind"])
            assert service.deployment_kind == observed[-1]
            assert service._support_bundle_system_diagnostics().deployment.kind == observed[-1]

    uvicorn.run = run
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    monkeypatch.chdir(tmp_path)
    arguments = ["serve", "--dev"]
    if requested is not None:
        arguments += ["--deployment-kind", requested]
    assert main(arguments) == (0 if expected else 1)
    assert observed == ([expected] if expected else [])
    if expected is None:
        error = capsys.readouterr().err
        assert "deployment" in error and ("match" in error or "conflict" in error)


def test_cloud_init_declares_hosting_in_its_generated_command(tmp_path):
    from cayu.cli._cloud_project import CloudProjectManifest

    for relative, content in project_files("kind-agent", preset="service").items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    assert main(["cloud", "init", str(tmp_path)]) == 0
    manifest = CloudProjectManifest.load(tmp_path / "cayu-cloud.toml")
    assert "--deployment-kind cloud" in manifest.web.command
    assert manifest.entrypoint == manifest.web.command


@pytest.mark.parametrize("settled,raises_cleanup", [(False, False), (True, False), (True, True)])
def test_mismatching_factory_reports_actual_shutdown_outcome(
    tmp_path, monkeypatch, capsys, caplog, settled, raises_cleanup
):
    from cayu.applications import CayuApp
    from cayu.cli import serve

    class OwnedResource:
        closed = False

        async def close(self):
            self.closed = True

        def __repr__(self):
            return "private-resource-canary"

    owned = OwnedResource()
    state = ModuleType("deployment_test_state")
    state.owned = owned
    monkeypatch.setitem(sys.modules, "deployment_test_state", state)
    for relative, content in project_files("service", preset="service").items():
        if relative == "app.py":
            content = content.replace(
                "    app = CayuApp(\n",
                "    from deployment_test_state import owned\n"
                "    app = CayuApp(\n        owned_resources=(owned,),\n",
            )
        if relative.endswith("service.py"):
            content = content.replace(
                "    project_context: ProjectControlPlaneContext | None = None,\n", ""
            ).replace("        project_context=project_context,\n", "")
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    original_drain = CayuApp.drain_background_interruptions

    async def drain(app, *, timeout_s):
        assert await original_drain(app, timeout_s=timeout_s)
        return settled

    monkeypatch.setattr(CayuApp, "drain_background_interruptions", drain)
    original_build = serve.build_project_service
    applications = []

    def capture(*args, **kwargs):
        service = original_build(*args, **kwargs)
        applications.append(service.cayu_app)
        return service

    monkeypatch.setattr(serve, "build_project_service", capture)
    cleanup_error = RuntimeError("private-cleanup-canary")
    caught = []
    if raises_cleanup:
        original_close = CayuApp.aclose

        async def close_then_raise(app, **kwargs):
            await original_close(app, **kwargs)
            raise cleanup_error

        monkeypatch.setattr(CayuApp, "aclose", close_then_raise)
        original_assemble = serve._build_serve_app

        def capture_rejection(*args, **kwargs):
            try:
                return original_assemble(*args, **kwargs)
            except ExceptionGroup as error:
                caught.append(error)
                raise

        monkeypatch.setattr(serve, "_build_serve_app", capture_rejection)
    uvicorn = ModuleType("uvicorn")
    uvicorn.run = lambda *args, **kwargs: pytest.fail("Mismatching service must not start")
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    monkeypatch.chdir(tmp_path)
    assert main(["serve", "--dev", "--deployment-kind", "cloud"]) == 1
    assert len(applications) == 1
    outcome = applications[0].shutdown_outcome
    assert outcome is not None and outcome.settled is settled
    assert outcome.owned_resources == ("released" if settled else "retained")
    assert owned.closed is settled
    output = capsys.readouterr()
    if raises_cleanup:
        assert len(caught) == 1 and len(caught[0].exceptions) == 2
        mismatch, cleanup = caught[0].exceptions
        assert isinstance(mismatch, serve.ServeError)
        assert "deployment kind does not match" in str(mismatch)
        assert cleanup is cleanup_error
    else:
        assert "deployment kind does not match" in output.err
    assert "private-cleanup-canary" not in output.out + output.err + caplog.text
    if settled:
        assert "warning: application shutdown" not in output.err
    else:
        assert "warning: application shutdown incomplete" in output.err
        assert "still_draining" in output.err
    assert "private-resource-canary" not in output.out + output.err + caplog.text
