from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import stat
import subprocess
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from cayu.cli import _cloud_project as cloud_project
from cayu.cli import cloud as cloud_cli
from cayu.cli import main
from cayu.cli._cloud_api import CloudApiClient, CloudApiError


def _write_ready_context(
    path: Path,
    *,
    api_key_file: Path,
    api_url: str = "https://cloud.example.test",
) -> None:
    path.write_text(
        json.dumps(
            {
                "api_key_file": str(api_key_file),
                "api_url": api_url,
                "deployment_id": "cloud-test",
                "region": "us-west-2",
                "schema_version": 1,
                "status": "ready",
            }
        )
    )


def test_core_cli_owns_cloud_namespace(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["--help"])

    assert raised.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "cloud" in help_text
    assert "Manage Cayu Cloud." in help_text


def test_cloud_help_exposes_first_party_customer_commands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["cloud", "--help"])

    assert raised.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    for command in (
        "applications",
        "context",
        "deploy",
        "deployment",
        "doctor",
        "env",
        "evidence",
        "init",
        "runtimes",
    ):
        assert command in help_text
    assert "--api-url" not in help_text
    assert " run " not in f" {help_text} "
    assert " runs " not in f" {help_text} "

    with pytest.raises(SystemExit) as deploy_help:
        main(["cloud", "deploy", "--help"])

    assert deploy_help.value.code == 0
    deploy_help_text = " ".join(capsys.readouterr().out.split())
    assert "Create or update this application slug" in deploy_help_text


def test_core_cli_cloud_parse_errors_are_machine_readable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["cloud", "env", "set"]) == 2

    streams = capsys.readouterr()
    assert streams.err == ""
    assert json.loads(streams.out) == {
        "error": {
            "category": "invalid_input",
            "message": "the following arguments are required: assignment, --application",
        },
        "ok": False,
    }


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_cloud_wait_options_reject_nonfinite_values_before_authentication(
    value: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unexpected_client(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("invalid wait input reached Cloud authentication")

    monkeypatch.setattr(cloud_cli, "_cloud_client", unexpected_client)

    assert (
        main(
            [
                "cloud",
                "deployment",
                "wait",
                "dep_one",
                "--application",
                "research-agent",
                "--poll-seconds",
                value,
            ]
        )
        == 2
    )

    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is False
    assert output["error"]["category"] == "invalid_input"


@pytest.mark.parametrize(
    "application",
    [
        "../source-bundles/uploads",
        "nested/application",
        "research agent",
        "research?agent",
        "research#agent",
    ],
)
def test_cloud_deploy_rejects_noncanonical_application_before_authentication(
    application: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unexpected_client(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("invalid application slug reached Cloud authentication")

    monkeypatch.setattr(cloud_cli, "_cloud_client", unexpected_client)

    assert main(["cloud", "deploy", ".", "--application", application]) == 2

    assert json.loads(capsys.readouterr().out) == {
        "error": {
            "category": "invalid_input",
            "message": (
                "Application must be a canonical lowercase Cayu Cloud slug "
                "containing only letters, numbers, and hyphens."
            ),
        },
        "ok": False,
    }


@pytest.mark.parametrize(
    "command",
    [
        ["deploy", ".", "--poll-seconds", "2", "--wait-seconds", "1"],
        [
            "service",
            "destroy",
            "--application",
            "research-agent",
            "--poll-seconds",
            "2",
            "--wait-seconds",
            "1",
        ],
    ],
)
def test_cloud_wait_options_are_ordered_before_authentication_or_mutation(
    command: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unexpected_client(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("invalid wait input reached Cloud authentication")

    monkeypatch.setattr(cloud_cli, "_cloud_client", unexpected_client)

    assert main(["cloud", *command]) == 2

    assert json.loads(capsys.readouterr().out) == {
        "error": {
            "category": "invalid_wait",
            "message": "poll-seconds and wait-seconds must be positive and ordered.",
        },
        "ok": False,
    }


def test_cloud_env_secret_reads_value_from_file_and_never_returns_it(tmp_path: Path) -> None:
    class Client:
        def __init__(self) -> None:
            self.requests: list[tuple[str, str, dict[str, object]]] = []

        def request(self, method: str, path: str, **kwargs: object) -> dict[str, object]:
            self.requests.append((method, path, kwargs))
            if path == "/v1/applications":
                return {"items": [{"id": "app_voice", "name": "Voice Agent"}]}
            return {
                "configuration": {"name": "VAPI_API_KEY", "secret": True, "value": None},
                "service": {"status": "deploying"},
            }

    value_file = tmp_path / "vapi-key"
    value_file.write_text("private-vapi-value\n")
    client = Client()

    result = cloud_cli._environment(
        SimpleNamespace(
            application="Voice Agent",
            assignment="VAPI_API_KEY",
            environment_command="set",
            secret=True,
            value_file=value_file,
        ),
        client=client,
    )

    assert client.requests[-1] == (
        "PUT",
        "/v1/applications/app_voice/environment/VAPI_API_KEY",
        {"payload": {"secret": True, "value": "private-vapi-value"}},
    )
    assert result == {
        "operation": "env.set",
        "result": {
            "configuration": {"name": "VAPI_API_KEY", "secret": True, "value": None},
            "service": {"status": "deploying"},
        },
    }
    assert "private-vapi-value" not in json.dumps(result)


def test_cloud_env_plain_assignment_list_and_unset_use_agent_scope() -> None:
    class Client:
        def __init__(self) -> None:
            self.requests: list[tuple[str, str, dict[str, object]]] = []

        def request(self, method: str, path: str, **kwargs: object) -> dict[str, object]:
            self.requests.append((method, path, kwargs))
            if path == "/v1/applications":
                return {"items": [{"id": "app_voice", "name": "Voice Agent"}]}
            if method == "GET":
                return {"items": [{"name": "MODE", "secret": False, "value": "demo"}]}
            if method == "DELETE":
                return {"configuration": None, "service": {"status": "deploying"}}
            return {
                "configuration": {"name": "MODE", "secret": False, "value": "demo"},
                "service": {"status": "deploying"},
            }

    client = Client()

    set_result = cloud_cli._environment(
        SimpleNamespace(
            application="app_voice",
            assignment="MODE=demo",
            environment_command="set",
            secret=False,
            value_file=None,
        ),
        client=client,
    )
    list_result = cloud_cli._environment(
        SimpleNamespace(application="app_voice", environment_command="list"),
        client=client,
    )
    unset_result = cloud_cli._environment(
        SimpleNamespace(
            application="app_voice",
            environment_command="unset",
            name="MODE",
        ),
        client=client,
    )

    assert set_result["result"]["configuration"] == {
        "name": "MODE",
        "secret": False,
        "value": "demo",
    }
    assert list_result["result"]["items"][0]["name"] == "MODE"
    assert unset_result["result"] == {
        "configuration": None,
        "service": {"status": "deploying"},
    }
    assert client.requests[-1][:2] == (
        "DELETE",
        "/v1/applications/app_voice/environment/MODE",
    )


def test_cloud_env_rejects_secret_material_in_argv() -> None:
    with pytest.raises(cloud_cli.CloudCommandError, match="value-file"):
        cloud_cli._environment(
            SimpleNamespace(
                application="app_voice",
                assignment="VAPI_API_KEY=private",
                environment_command="set",
                secret=True,
                value_file=None,
            ),
            client=object(),
        )


def test_cloud_init_generates_web_manifest_from_cayu_project(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "research-agent"
version = "1.2.3"

[tool.cayu]
factory = "research_agent:build_app"

[tool.cayu.serve]
access = "authenticated"
""".strip()
        + "\n"
    )

    assert main(["cloud", "init", str(tmp_path)]) == 0

    output = json.loads(capsys.readouterr().out)
    manifest_path = tmp_path / "cayu-cloud.toml"
    manifest = cloud_project.CloudProjectManifest.load(manifest_path)
    assert output["operation"] == "init"
    assert output["result"] == {
        "application": "research-agent",
        "manifest": str(manifest_path),
        "name": "Research Agent",
        "runtime": "web",
    }
    assert manifest.entrypoint == "cayu serve --host 0.0.0.0 --port 8000"
    assert manifest.web == cloud_project.CloudWebProcess(
        command="cayu serve --host 0.0.0.0 --port 8000",
        port=8000,
    )
    assert manifest.worker is None


def test_cloud_init_generates_worker_manifest_from_single_console_script(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "mail-agent"
version = "0.4.0"

[project.scripts]
mail-agent = "mail_agent.cli:main"
""".strip()
        + "\n"
    )

    assert main(["cloud", "init", str(tmp_path)]) == 0

    output = json.loads(capsys.readouterr().out)
    manifest = cloud_project.CloudProjectManifest.load(tmp_path / "cayu-cloud.toml")
    assert output["result"]["runtime"] == "worker"
    assert manifest.entrypoint == "mail-agent"
    assert manifest.worker == cloud_project.CloudProcess(command="mail-agent")
    assert manifest.version == "0.4.0"


def test_cloud_init_refuses_to_replace_existing_manifest_without_force(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "agent"\nversion = "0.1.0"\n')
    manifest_path = tmp_path / "cayu-cloud.toml"
    manifest_path.write_text("keep me\n")

    assert main(["cloud", "init", str(tmp_path)]) == 2

    output = json.loads(capsys.readouterr().out)
    assert output["error"]["category"] == "manifest_exists"
    assert manifest_path.read_text() == "keep me\n"


def test_cloud_init_validates_generated_manifest_before_writing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "agent"\nversion = "not a valid version!"\n'
    )

    assert main(["cloud", "init", str(tmp_path)]) == 2

    output = json.loads(capsys.readouterr().out)
    assert output["error"]["category"] == "manifest_invalid"
    assert not (tmp_path / "cayu-cloud.toml").exists()


def test_deployment_promote_selects_the_release_with_an_optimistic_revision() -> None:
    class Client:
        def __init__(self) -> None:
            self.requests: list[tuple[str, str, dict[str, object]]] = []

        def request(self, method: str, path: str, **kwargs: object) -> dict[str, object]:
            self.requests.append((method, path, kwargs))
            if path == "/v1/applications":
                return {"items": [{"id": "app_research", "name": "Research Agent"}]}
            if path == "/v1/applications/app_research":
                return {"id": "app_research", "revision": 7}
            return {"id": "app_research", "current_deployment_id": "dep_next", "revision": 8}

    client = Client()
    result = cloud_cli._deployment(
        SimpleNamespace(
            application="Research Agent",
            deployment_command="promote",
            deployment_id="dep_next",
        ),
        client=client,
    )

    assert result["operation"] == "deployment.promote"
    assert client.requests[-1] == (
        "POST",
        "/v1/applications/app_research/deployments/dep_next/promote",
        {"payload": {"expected_application_revision": 7}},
    )


def test_service_destroy_waits_until_the_application_is_stopped() -> None:
    class Client:
        def __init__(self) -> None:
            self.statuses = iter(("deleting", "stopped"))
            self.requests: list[tuple[str, str]] = []

        def request(self, method: str, path: str, **kwargs: object) -> dict[str, object]:
            del kwargs
            self.requests.append((method, path))
            if path == "/v1/applications":
                return {
                    "items": [
                        {
                            "id": "app_research",
                            "name": "Research Agent",
                        }
                    ]
                }
            return {
                "application_id": "app_research",
                "status": next(self.statuses),
            }

    client = Client()
    clock = iter((0.0, 1.0))
    result = cloud_cli._service(
        SimpleNamespace(
            application="Research Agent",
            poll_seconds=1.0,
            service_command="destroy",
            wait_seconds=10.0,
        ),
        client=client,
        sleep=lambda _: None,
        monotonic=lambda: next(clock),
    )

    assert result["result"]["status"] == "stopped"
    assert client.requests[-2:] == [
        ("DELETE", "/v1/applications/app_research/service"),
        ("GET", "/v1/applications/app_research/service"),
    ]


@pytest.mark.parametrize("action", ["sleep", "wake"])
def test_service_sleep_and_wake_use_the_agent_web_service_endpoints(action: str) -> None:
    class Client:
        def __init__(self) -> None:
            self.requests: list[tuple[str, str]] = []

        def request(self, method: str, path: str, **kwargs: object) -> dict[str, object]:
            del kwargs
            self.requests.append((method, path))
            if path == "/v1/applications":
                return {"items": [{"id": "app_research", "name": "Research Agent"}]}
            return {"application_id": "app_research", "status": "starting"}

    client = Client()
    result = cloud_cli._service(
        SimpleNamespace(
            application="Research Agent",
            service_command=action,
        ),
        client=client,
    )

    assert result["operation"] == f"service.{action}"
    assert client.requests[-1] == (
        "POST",
        f"/v1/applications/app_research/service/{action}",
    )


def test_cloud_without_selected_context_returns_machine_readable_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "missing.json"))

    assert main(["cloud", "doctor"]) == 2

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error"]["category"] == "login_required"


def test_cloud_context_use_persists_only_private_absolute_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    api_key_file = tmp_path / "api-key"
    api_key_file.write_text("customer-secret-material\n")
    context_path = tmp_path / "cloud-context.json"
    _write_ready_context(context_path, api_key_file=Path("api-key"))
    config_path = tmp_path / "config.json"
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(config_path))
    monkeypatch.chdir(tmp_path.parent)

    assert main(["cloud", "context", "use", str(context_path)]) == 0

    output = json.loads(capsys.readouterr().out)
    persisted = json.loads(config_path.read_text())
    assert output["result"]["status"] == "active"
    assert persisted == {
        "active_context": str(context_path.resolve()),
        "schema_version": 1,
    }
    assert stat.S_IMODE(config_path.stat().st_mode) == 0o600
    assert "customer-secret-material" not in config_path.read_text()


def test_cloud_context_show_and_clear_are_first_party(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    api_key_file = tmp_path / "api-key"
    api_key_file.write_text("customer-secret-material\n")
    context_path = tmp_path / "cloud-context.json"
    _write_ready_context(context_path, api_key_file=Path("api-key"))
    config_path = tmp_path / "config.json"
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(config_path))
    monkeypatch.chdir(tmp_path.parent)
    assert main(["cloud", "context", "use", str(context_path)]) == 0
    capsys.readouterr()

    assert main(["cloud", "context", "show"]) == 0
    shown = capsys.readouterr().out
    assert json.loads(shown)["result"]["status"] == "active"
    assert "customer-secret-material" not in shown

    assert main(["cloud", "context", "clear"]) == 0
    cleared = json.loads(capsys.readouterr().out)
    assert cleared["result"] == {"cleared": True, "status": "inactive"}
    assert not config_path.exists()

    assert main(["cloud", "context", "clear"]) == 0
    repeated = json.loads(capsys.readouterr().out)
    assert repeated["result"] == {"cleared": False, "status": "inactive"}


def test_cloud_doctor_authenticates_through_the_core_package(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    requests: list[tuple[str, str | None]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requests.append((self.path, self.headers.get("Authorization")))
            payload = json.dumps({"items": [{"id": "app_one"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        api_key_file = tmp_path / "api-key"
        api_key_file.write_text("customer-secret-material\n")
        context_path = tmp_path / "cloud-context.json"
        _write_ready_context(
            context_path,
            api_key_file=Path("api-key"),
            api_url=f"http://127.0.0.1:{server.server_port}",
        )
        monkeypatch.chdir(tmp_path.parent)

        assert main(["cloud", "--context", str(context_path), "doctor"]) == 0
    finally:
        server.shutdown()
        thread.join()

    rendered = capsys.readouterr().out
    output = json.loads(rendered)
    assert output["result"]["api_reachable"] is True
    assert output["result"]["application_count"] == 1
    assert requests == [("/v1/applications", "Bearer customer-secret-material")]
    assert "customer-secret-material" not in rendered


def test_cloud_deploy_is_implemented_by_the_core_package(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    requests: list[tuple[str, str, dict[str, object] | None]] = []
    idempotency_keys: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requests.append(("GET", self.path, None))
            payload = json.dumps(
                {
                    "items": [
                        {
                            "id": "legacy-research",
                            "name": "Research Agent",
                            "revision": 1,
                        }
                    ]
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_PUT(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length)) if length else None
            requests.append(("PUT", self.path, body))
            payload = json.dumps(
                {"id": "research-agent", "name": "Research Agent", "revision": 1}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length)) if length else None
            requests.append(("POST", self.path, body))
            idempotency_keys.append(self.headers["Idempotency-Key"])
            payload = json.dumps({"id": "dep_one", "status": "queued"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            return

    gh = tmp_path / "gh"
    long_version = "v" * 128
    gh.write_text(
        f"""#!/bin/sh
cat <<'EOF'
schema_version = 1
application = "research-agent"
name = "Research Agent"
version = "{long_version}"
entrypoint = "python app.py"
capabilities = ["network"]
cpu_millis = 1000
memory_mb = 1024
timeout_seconds = 600
environment = "python"
compatibility = "cayu>=0.1"
policy_version = "v1"
EOF
"""
    )
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        api_key_file = tmp_path / "api-key"
        api_key_file.write_text("customer-secret-material\n")
        context_path = tmp_path / "cloud-context.json"
        _write_ready_context(
            context_path,
            api_key_file=api_key_file,
            api_url=f"http://127.0.0.1:{server.server_port}",
        )
        revision = "a" * 40
        unavailable_evidence = tmp_path / "unavailable-evidence"
        unavailable_evidence.write_text("not a directory\n")

        assert (
            main(
                [
                    "cloud",
                    "--context",
                    str(context_path),
                    "--evidence-dir",
                    str(unavailable_evidence),
                    "deploy",
                    "https://github.com/example/research-agent",
                    "--revision",
                    revision,
                    "--no-wait",
                    "--no-promote",
                ]
            )
            == 2
        )
        failure = json.loads(capsys.readouterr().out)
        assert failure["error"]["category"] == "local_state_unavailable"
        assert requests == []

        assert (
            main(
                [
                    "cloud",
                    "--context",
                    str(context_path),
                    "--evidence-dir",
                    str(tmp_path / "evidence"),
                    "deploy",
                    "https://github.com/example/research-agent",
                    "--revision",
                    revision,
                    "--no-wait",
                    "--no-promote",
                ]
            )
            == 0
        )
        rendered = capsys.readouterr().out
        output = json.loads(rendered)

        assert (
            main(
                [
                    "cloud",
                    "--context",
                    str(context_path),
                    "--evidence-dir",
                    str(tmp_path / "second-evidence"),
                    "deploy",
                    "https://github.com/example/research-agent",
                    "--revision",
                    "b" * 40,
                    "--no-wait",
                    "--no-promote",
                ]
            )
            == 0
        )
        capsys.readouterr()
    finally:
        server.shutdown()
        thread.join()

    assert output["operation"] == "deploy"
    assert output["result"]["deployment"]["id"] == "dep_one"
    assert requests[0] == ("GET", "/v1/applications", None)
    assert requests[1] == (
        "PUT",
        "/v1/applications/research-agent",
        {"name": "Research Agent"},
    )
    assert requests[2][0:2] == (
        "POST",
        "/v1/applications/research-agent/deployments",
    )
    assert requests[2][2]["manifest"]["source"] == {
        "repository": "https://github.com/example/research-agent",
        "revision": revision,
    }
    evidence_files = list((tmp_path / "evidence").glob("*.json"))
    assert len(evidence_files) == 1
    assert "customer-secret-material" not in evidence_files[0].read_text()
    assert len(idempotency_keys) == 2
    assert idempotency_keys[0] != idempotency_keys[1]
    assert all(len(key) <= 128 for key in idempotency_keys)


def test_local_project_bundle_is_deterministic_and_contains_the_working_directory(
    tmp_path: Path,
) -> None:
    (tmp_path / "cayu-cloud.toml").write_text(
        """
schema_version = 1
application = "research-agent"
name = "Research Agent"
version = "1.0.0"
entrypoint = "python agent.py"
capabilities = ["model.generate"]
cpu_millis = 1000
memory_mb = 1024
timeout_seconds = 600
environment = "python"
compatibility = "cayu>=0.1"
policy_version = "v1"
"""
    )
    (tmp_path / "agent.py").write_text("print('patched locally')\n")
    (tmp_path / "notes.txt").write_text("uncommitted patch output\n")
    (tmp_path / ".env").write_text("SECRET=must-not-upload\n")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "agent.pyc").write_bytes(b"cache")

    first = cloud_project.resolve_project(str(tmp_path), manifest_path=None, revision=None)
    second = cloud_project.resolve_project(str(tmp_path), manifest_path=None, revision=None)

    assert first.bundle is not None
    assert first.bundle == second.bundle
    assert first.content_digest == "sha256:" + hashlib.sha256(first.bundle).hexdigest()
    assert first.repository.startswith("cayu-cloud://source-bundles/sha256/")
    assert first.revision == first.content_digest.removeprefix("sha256:")[:40]
    with tarfile.open(fileobj=io.BytesIO(first.bundle), mode="r:gz") as archive:
        names = archive.getnames()
        assert names == ["source/agent.py", "source/cayu-cloud.toml", "source/notes.txt"]
        extracted = archive.extractfile("source/agent.py")
        assert extracted is not None
        assert extracted.read() == b"print('patched locally')\n"
    assert b"must-not-upload" not in first.bundle


def test_local_project_bundle_reflects_deleted_tracked_files(tmp_path: Path) -> None:
    (tmp_path / "cayu-cloud.toml").write_text(
        """
schema_version = 1
application = "research-agent"
name = "Research Agent"
version = "1.0.0"
entrypoint = "python agent.py"
capabilities = ["model.generate"]
cpu_millis = 1000
memory_mb = 1024
timeout_seconds = 600
environment = "python"
compatibility = "cayu>=0.1"
policy_version = "v1"
"""
    )
    (tmp_path / "agent.py").write_text("print('local')\n")
    deleted = tmp_path / "removed.py"
    deleted.write_text("raise RuntimeError('old code')\n")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    deleted.unlink()

    project = cloud_project.resolve_project(str(tmp_path), manifest_path=None, revision=None)

    assert project.bundle is not None
    with tarfile.open(fileobj=io.BytesIO(project.bundle), mode="r:gz") as archive:
        assert archive.getnames() == ["source/agent.py", "source/cayu-cloud.toml"]


def test_local_project_bundle_fails_closed_when_git_listing_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "cayu-cloud.toml").write_text(
        """
schema_version = 1
application = "research-agent"
name = "Research Agent"
version = "1.0.0"
entrypoint = "python agent.py"
capabilities = ["model.generate"]
cpu_millis = 1000
memory_mb = 1024
timeout_seconds = 600
environment = "python"
compatibility = "cayu>=0.1"
policy_version = "v1"
"""
    )
    (tmp_path / "agent.py").write_text("print('local')\n")
    ignored_secret = tmp_path / "private-token.txt"
    ignored_secret.write_text("ignored-credential-canary\n")
    (tmp_path / ".gitignore").write_text("private-token.txt\n")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)

    def unavailable(*_args: object, **_kwargs: object) -> SimpleNamespace:
        raise FileNotFoundError("git is unavailable")

    monkeypatch.setattr(cloud_project.subprocess, "run", unavailable)

    with pytest.raises(CloudApiError) as raised:
        cloud_project.resolve_project(str(tmp_path), manifest_path=None, revision=None)

    assert raised.value.category == "source_git_failed"
    assert "ignored-credential-canary" not in str(raised.value)


def test_cloud_deploy_uploads_a_local_bundle_before_creating_the_release(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    requests: list[tuple[str, str, object]] = []

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, payload: dict[str, object]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            requests.append(("GET", self.path, None))
            self._reply(
                {
                    "items": [
                        {
                            "id": "research-agent",
                            "name": "Research Agent",
                            "revision": 1,
                        }
                    ]
                }
            )

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length)) if length else None
            requests.append(("POST", self.path, body))
            if self.path == "/v1/source-bundles/uploads":
                assert isinstance(body, dict)
                digest = str(body["content_digest"])
                self._reply(
                    {
                        "content_digest": digest,
                        "repository": (
                            "cayu-cloud://source-bundles/sha256/" + digest.removeprefix("sha256:")
                        ),
                        "revision": digest.removeprefix("sha256:")[:40],
                        "size_bytes": body["size_bytes"],
                        "upload_url": f"http://127.0.0.1:{self.server.server_port}/upload",
                    }
                )
            else:
                self._reply({"id": "dep_local", "status": "queued"})

        def do_PUT(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            requests.append(("PUT", self.path, body))
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            return

    (tmp_path / "cayu-cloud.toml").write_text(
        """
schema_version = 1
application = "research-agent"
name = "Research Agent"
version = "1.0.0"
entrypoint = "python agent.py"
capabilities = ["model.generate"]
cpu_millis = 1000
memory_mb = 1024
timeout_seconds = 600
environment = "python"
compatibility = "cayu>=0.1"
policy_version = "v1"
"""
    )
    (tmp_path / "agent.py").write_text("print('local')\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        api_key_file = tmp_path / "api-key"
        api_key_file.write_text("customer-secret-material\n")
        context_path = tmp_path / "cloud-context.json"
        _write_ready_context(
            context_path,
            api_key_file=api_key_file,
            api_url=f"http://127.0.0.1:{server.server_port}",
        )

        assert (
            main(
                [
                    "cloud",
                    "--context",
                    str(context_path),
                    "--evidence-dir",
                    str(tmp_path / "evidence"),
                    "deploy",
                    str(tmp_path),
                    "--no-wait",
                    "--no-promote",
                ]
            )
            == 0
        )
    finally:
        server.shutdown()
        thread.join()

    output = json.loads(capsys.readouterr().out)
    upload_request = requests[1]
    uploaded = requests[2]
    deployment_request = requests[3]
    assert upload_request[0:2] == ("POST", "/v1/source-bundles/uploads")
    assert uploaded[0:2] == ("PUT", "/upload")
    assert isinstance(uploaded[2], bytes)
    assert upload_request[2]["content_digest"] == (
        "sha256:" + hashlib.sha256(uploaded[2]).hexdigest()
    )
    assert deployment_request[0:2] == (
        "POST",
        "/v1/applications/research-agent/deployments",
    )
    assert deployment_request[2]["manifest"]["source"]["repository"].startswith(
        "cayu-cloud://source-bundles/sha256/"
    )
    assert output["result"]["source"]["kind"] == "local_bundle"


def test_cloud_manifest_v2_describes_the_complete_application_runtime() -> None:
    manifest = cloud_project.CloudProjectManifest.loads(
        """
schema_version = 2
application = "domain-research"
name = "Domain Research"
version = "1.0.0"
entrypoint = "vacancy 'demo' --target 3"
capabilities = ["model.generate", "domains.resolve"]
cpu_millis = 512
memory_mb = 1024
timeout_seconds = 600
environment = "python"
compatibility = "cayu>=0.1"
policy_version = "v1"

[web]
command = "python -m vacancy"
port = 8000
idle_timeout_seconds = 60

[worker]
command = "sh -c 'while true; do vacancy --runs; sleep 60; done'"

[env]
CONTROL_PLANE = "on"
VACANCY_DB_PATH = "/data/domains.db"

[[schedules]]
name = "every-minute"
command = "vacancy --runs"
expression = "rate(1 minute)"
"""
    )

    assert manifest.runtime_payload() == {
        "environment": {
            "CONTROL_PLANE": "on",
            "VACANCY_DB_PATH": "/data/domains.db",
        },
        "schedules": [
            {
                "command": "vacancy --runs",
                "expression": "rate(1 minute)",
                "name": "every-minute",
            }
        ],
        "web": {
            "command": "python -m vacancy",
            "idle_timeout_seconds": 60,
            "port": 8000,
        },
        "worker": {"command": "sh -c 'while true; do vacancy --runs; sleep 60; done'"},
    }


def test_cloud_deploy_starts_the_complete_application_after_promotion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    requests: list[tuple[str, str, dict[str, object] | None]] = []

    class Handler(BaseHTTPRequestHandler):
        break_evidence_directory: Path | None = None
        deployment_manifest: dict[str, object] | None = None

        def _reply(self, payload: dict[str, object]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            requests.append(("GET", self.path, None))
            if self.path == "/v1/applications":
                self._reply(
                    {
                        "items": [
                            {
                                "current_deployment_id": None,
                                "id": "domain-research",
                                "name": "Domain Research",
                                "revision": 1,
                            }
                        ]
                    }
                )
            elif "/runtime-artifacts/" in self.path:
                self._reply({"id": "rta_two", "provider": "e2b"})
            else:
                self._reply(
                    {
                        "id": "dep_two",
                        "manifest": self.deployment_manifest,
                        "runtime_artifact_id": "rta_two",
                        "status": "promoted",
                    }
                )

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length)) if length else None
            requests.append(("POST", self.path, body))
            if self.path.endswith("/promote"):
                self._reply(
                    {
                        "current_deployment_id": "dep_two",
                        "id": "domain-research",
                        "name": "Domain Research",
                        "revision": 2,
                    }
                )
            else:
                assert isinstance(body, dict)
                manifest = body.get("manifest")
                assert isinstance(manifest, dict)
                Handler.deployment_manifest = manifest
                self._reply(
                    {
                        "id": "dep_two",
                        "manifest": manifest,
                        "runtime_artifact_id": "rta_two",
                        "status": "smoke_tested",
                    }
                )

        def do_PUT(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length)) if length else None
            requests.append(("PUT", self.path, body))
            if Handler.break_evidence_directory is not None:
                shutil.rmtree(Handler.break_evidence_directory)
                Handler.break_evidence_directory.write_text("not a directory\n")
            self._reply(
                {
                    "application_id": "domain-research",
                    "application_url": "https://agent.example.test",
                    "cayu_url": "https://agent.example.test/cayu",
                    "deployment_id": "dep_two",
                    "schedules": ["every-minute"],
                    "status": "running",
                    "web_service": "agent-web",
                    "worker_service": "agent-worker",
                }
            )

        def log_message(self, format: str, *args: object) -> None:
            return

    gh = tmp_path / "gh"
    gh.write_text(
        """#!/bin/sh
cat <<'EOF'
schema_version = 2
application = "domain-research"
name = "Domain Research"
version = "1.0.0"
entrypoint = "vacancy demo --target 3"
capabilities = ["model.generate", "domains.resolve"]
cpu_millis = 512
memory_mb = 1024
timeout_seconds = 600
environment = "python"
compatibility = "cayu>=0.1"
policy_version = "v1"
[env]
AUTH_TOKEN = "disabled"
MODE = "private-runtime-value"
[web]
command = "python -m vacancy"
port = 8000
[worker]
command = "sh -c 'while true; do vacancy --runs; sleep 60; done'"
[[schedules]]
name = "every-minute"
command = "vacancy --runs"
expression = "rate(1 minute)"
EOF
"""
    )
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        api_key_file = tmp_path / "api-key"
        api_key_file.write_text("customer-secret-material\n")
        context_path = tmp_path / "cloud-context.json"
        _write_ready_context(
            context_path,
            api_key_file=api_key_file,
            api_url=f"http://127.0.0.1:{server.server_port}",
        )
        assert (
            main(
                [
                    "cloud",
                    "--context",
                    str(context_path),
                    "--evidence-dir",
                    str(tmp_path / "evidence"),
                    "deploy",
                    "https://github.com/example/domain-research",
                    "--revision",
                    "a" * 40,
                    "--no-wait",
                ]
            )
            == 0
        )
        output = json.loads(capsys.readouterr().out)

        late_evidence = tmp_path / "late-evidence"
        Handler.break_evidence_directory = late_evidence
        assert (
            main(
                [
                    "cloud",
                    "--context",
                    str(context_path),
                    "--evidence-dir",
                    str(late_evidence),
                    "deploy",
                    "https://github.com/example/domain-research",
                    "--revision",
                    "b" * 40,
                    "--no-wait",
                ]
            )
            == 0
        )
        late_output = json.loads(capsys.readouterr().out)
    finally:
        server.shutdown()
        thread.join()

    assert output["result"]["service"]["application_url"] == "https://agent.example.test"
    assert late_output["result"]["service"]["application_url"] == ("https://agent.example.test")
    assert late_output["evidence_id"] is None
    assert late_output["evidence"] == {
        "category": "local_state_unavailable",
        "message": "Deployment succeeded, but local evidence could not be recorded.",
        "status": "unavailable",
    }
    service_request = next(
        item
        for item in requests
        if item[0:2]
        == (
            "PUT",
            "/v1/applications/domain-research/service",
        )
    )
    assert service_request[2] is None
    deployment_request = next(
        item
        for item in requests
        if item[0:2]
        == (
            "POST",
            "/v1/applications/domain-research/deployments",
        )
    )
    runtime = deployment_request[2]["manifest"]["runtime"]
    assert runtime["web"] == {
        "command": "python -m vacancy",
        "port": 8000,
    }
    assert runtime["schedules"][0]["expression"] == "rate(1 minute)"
    output_environment = output["result"]["deployment"]["manifest"]["runtime"]["environment"]
    assert output_environment == {
        "AUTH_TOKEN": "[redacted]",
        "MODE": "[redacted]",
    }
    assert "disabled" not in json.dumps(output)
    evidence_files = list((tmp_path / "evidence").glob("*.json"))
    assert len(evidence_files) == 1
    evidence = json.loads(evidence_files[0].read_text())
    evidence_environment = evidence["result"]["deployment"]["manifest"]["runtime"]["environment"]
    assert evidence_environment == {
        "AUTH_TOKEN": "[redacted]",
        "MODE": "[redacted]",
    }
    assert "disabled" not in evidence_files[0].read_text()


def test_cloud_inspection_commands_are_implemented_by_the_core_package(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/v1/applications":
                result: dict[str, object] = {
                    "items": [{"id": "app_research", "name": "Research Agent"}]
                }
            elif self.path.endswith("/deployments/dep_one"):
                result = {"id": "dep_one", "status": "promoted"}
            elif self.path.endswith("/runtime-artifacts"):
                result = {"items": [{"id": "rta_one", "status": "ready"}]}
            else:
                self.send_error(404)
                return
            payload = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        api_key_file = tmp_path / "api-key"
        api_key_file.write_text("customer-secret-material\n")
        context_path = tmp_path / "cloud-context.json"
        _write_ready_context(
            context_path,
            api_key_file=api_key_file,
            api_url=f"http://127.0.0.1:{server.server_port}",
        )
        common = ["cloud", "--context", str(context_path)]
        cases = (
            (["applications", "list"], "applications.list"),
            (
                [
                    "deployment",
                    "status",
                    "dep_one",
                    "--application",
                    "research-agent",
                ],
                "deployment.status",
            ),
            (
                ["runtimes", "list", "--application", "research-agent"],
                "runtimes.list",
            ),
            (
                [
                    "--evidence-dir",
                    str(tmp_path / "evidence"),
                    "evidence",
                    "list",
                ],
                "evidence.list",
            ),
        )
        for arguments, expected_operation in cases:
            assert main([*common, *arguments]) == 0
            assert json.loads(capsys.readouterr().out)["operation"] == expected_operation
    finally:
        server.shutdown()
        thread.join()


def test_cloud_api_key_uses_production_endpoint_without_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    requests: list[tuple[str, str, str]] = []

    def request(
        client: CloudApiClient,
        method: str,
        path: str,
        **_kwargs: object,
    ) -> dict[str, object]:
        requests.append((client.api_url, method, path))
        return {"items": []}

    monkeypatch.delenv("CAYU_CLOUD_API_URL", raising=False)
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "missing.json"))
    monkeypatch.setenv("CAYU_CLOUD_API_KEY", "customer-secret-material")
    monkeypatch.setattr(CloudApiClient, "request", request)

    assert main(["cloud", "applications", "list"]) == 0

    assert requests == [
        ("https://cloud.cayu.dev", "GET", "/v1/applications"),
    ]
    assert "customer-secret-material" not in capsys.readouterr().out


def test_cloud_context_endpoint_precedes_environment_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    requests: list[tuple[str, str, str]] = []

    def request(
        client: CloudApiClient,
        method: str,
        path: str,
        **_kwargs: object,
    ) -> dict[str, object]:
        requests.append((client.api_url, method, path))
        return {"items": []}

    api_key_file = tmp_path / "api-key"
    api_key_file.write_text("customer-secret-material\n")
    context_path = tmp_path / "cloud-context.json"
    _write_ready_context(
        context_path,
        api_key_file=api_key_file,
        api_url="https://context-cloud.example.test",
    )
    monkeypatch.setenv("CAYU_CLOUD_API_URL", "https://environment-cloud.example.test")
    monkeypatch.setattr(CloudApiClient, "request", request)

    assert (
        main(
            [
                "cloud",
                "--context",
                str(context_path),
                "applications",
                "list",
            ]
        )
        == 0
    )

    assert requests == [
        ("https://context-cloud.example.test", "GET", "/v1/applications"),
    ]
    assert "customer-secret-material" not in capsys.readouterr().out


@pytest.mark.parametrize("environment_key_file", [None, ""])
def test_cloud_persisted_context_rejects_environment_endpoint_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    environment_key_file: str | None,
) -> None:
    requests: list[tuple[str, str, str]] = []

    def request(
        client: CloudApiClient,
        method: str,
        path: str,
        **_kwargs: object,
    ) -> dict[str, object]:
        requests.append((client.api_url, method, path))
        return {"items": []}

    api_key_file = tmp_path / "api-key"
    api_key_file.write_text("context-secret-canary\n")
    context_path = tmp_path / "cloud-context.json"
    _write_ready_context(
        context_path,
        api_key_file=api_key_file,
        api_url="https://context-cloud.example.test",
    )
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "config.json"))
    monkeypatch.delenv("CAYU_CLOUD_API_KEY", raising=False)
    if environment_key_file is None:
        monkeypatch.delenv("CAYU_CLOUD_API_KEY_FILE", raising=False)
    else:
        monkeypatch.setenv("CAYU_CLOUD_API_KEY_FILE", environment_key_file)
    monkeypatch.setattr(CloudApiClient, "request", request)

    assert main(["cloud", "context", "use", str(context_path)]) == 0
    capsys.readouterr()
    monkeypatch.setenv("CAYU_CLOUD_API_URL", "https://other-cloud.example.test")

    assert main(["cloud", "applications", "list"]) == 2

    streams = capsys.readouterr()
    assert streams.err == ""
    assert requests == []
    assert json.loads(streams.out) == {
        "error": {
            "category": "context_api_mismatch",
            "message": (
                "CAYU_CLOUD_API_URL differs from the persisted context; provide "
                "an explicit API key for that Cayu Cloud or select a matching context."
            ),
        },
        "ok": False,
    }
    assert "context-secret-canary" not in streams.out


def test_cloud_explicit_api_credentials_do_not_require_a_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            payload = json.dumps({"items": []}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)
        monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "missing.json"))
        monkeypatch.setenv("CAYU_CLOUD_API_KEY", "customer-secret-material")
        monkeypatch.setenv(
            "CAYU_CLOUD_API_URL",
            f"http://127.0.0.1:{server.server_port}",
        )
        assert main(["cloud", "applications", "list"]) == 0
    finally:
        server.shutdown()
        thread.join()

    rendered = capsys.readouterr().out
    assert json.loads(rendered)["operation"] == "applications.list"
    assert "customer-secret-material" not in rendered


def test_cloud_api_does_not_forward_credentials_across_redirects(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    redirected_requests: list[str | None] = []

    class RedirectTargetHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            redirected_requests.append(self.headers.get("Authorization"))
            payload = json.dumps({"items": []}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            return

    target = ThreadingHTTPServer(("127.0.0.1", 0), RedirectTargetHandler)

    class RedirectHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(302)
            self.send_header(
                "Location",
                f"http://127.0.0.1:{target.server_port}/v1/applications",
            )
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            return

    redirect = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
    target_thread = threading.Thread(target=target.serve_forever, daemon=True)
    redirect_thread = threading.Thread(target=redirect.serve_forever, daemon=True)
    target_thread.start()
    redirect_thread.start()
    try:
        api_key_file = tmp_path / "api-key"
        api_key_file.write_text("customer-secret-material\n")
        context_path = tmp_path / "cloud-context.json"
        _write_ready_context(
            context_path,
            api_key_file=api_key_file,
            api_url=f"http://127.0.0.1:{redirect.server_port}",
        )

        assert main(["cloud", "--context", str(context_path), "doctor"]) == 2
    finally:
        redirect.shutdown()
        target.shutdown()
        redirect_thread.join()
        target_thread.join()

    rendered = capsys.readouterr().out
    assert json.loads(rendered)["error"]["category"] == "api_request_rejected"
    assert redirected_requests == []
    assert "customer-secret-material" not in rendered


def test_cloud_api_does_not_echo_credentials_from_error_responses(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            payload = json.dumps({"detail": "invalid token customer-secret-material"}).encode()
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        api_key_file = tmp_path / "api-key"
        api_key_file.write_text("customer-secret-material\n")
        context_path = tmp_path / "cloud-context.json"
        _write_ready_context(
            context_path,
            api_key_file=api_key_file,
            api_url=f"http://127.0.0.1:{server.server_port}",
        )

        assert main(["cloud", "--context", str(context_path), "doctor"]) == 2
    finally:
        server.shutdown()
        thread.join()

    rendered = capsys.readouterr().out
    assert json.loads(rendered)["error"] == {
        "category": "api_request_rejected",
        "message": "Cayu Cloud API returned HTTP 401.",
    }
    assert "customer-secret-material" not in rendered


def test_cloud_source_upload_reports_only_safe_object_store_error_fields() -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_PUT(self) -> None:
            payload = b"""<?xml version="1.0" encoding="UTF-8"?>
<Error>
  <Code>AccessDenied</Code>
  <Message>Denied for https://bucket.example/upload?signature=private-signed-query</Message>
  <RequestId>private-provider-request-id</RequestId>
</Error>"""
            self.send_response(403)
            self.send_header("Content-Type", "application/xml")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    signed_secret = "private-signed-query"
    try:
        client = CloudApiClient(
            api_url=f"http://127.0.0.1:{server.server_port}",
            api_key="customer-secret-material",
        )
        with pytest.raises(CloudApiError) as raised:
            client.upload_bytes(
                f"http://127.0.0.1:{server.server_port}/upload?signature={signed_secret}",
                b"source",
            )
    finally:
        server.shutdown()
        thread.join()

    assert raised.value.category == "source_upload_rejected"
    assert str(raised.value) == "Local source bundle upload returned HTTP 403: AccessDenied."
    assert "private-provider-request-id" not in str(raised.value)
    assert signed_secret not in str(raised.value)


def test_cloud_api_requires_https_outside_loopback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "missing.json"))
    monkeypatch.setenv("CAYU_CLOUD_API_KEY", "customer-secret-material")
    monkeypatch.setenv("CAYU_CLOUD_API_URL", "http://cloud.example.test")

    assert main(["cloud", "applications", "list"]) == 2

    rendered = capsys.readouterr().out
    assert json.loads(rendered)["error"] == {
        "category": "invalid_input",
        "message": "Cayu Cloud API URL must use HTTPS outside loopback.",
    }
    assert "customer-secret-material" not in rendered


def test_cloud_source_commands_are_noninteractive_and_drop_cloud_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((command, kwargs))
        if "ls-files" in command:
            return SimpleNamespace(stdout=b"")
        return SimpleNamespace(stdout="manifest")

    monkeypatch.setenv("CAYU_CLOUD_API_KEY", "customer-secret-material")
    monkeypatch.setenv("CAYU_CLOUD_API_KEY_FILE", "/private/cloud-key")
    monkeypatch.setattr(cloud_project.subprocess, "run", run)

    cloud_project._git_project_files(tmp_path)
    cloud_project._github_file(
        "https://github.com/example/project",
        revision="a" * 40,
        path="cayu-cloud.toml",
    )

    assert len(calls) == 3
    git_environment = calls[0][1]["env"]
    gh_auth_environment = calls[1][1]["env"]
    gh_api_environment = calls[2][1]["env"]
    assert isinstance(git_environment, dict)
    assert isinstance(gh_auth_environment, dict)
    assert isinstance(gh_api_environment, dict)
    assert calls[0][1]["stdin"] is subprocess.DEVNULL
    assert calls[1][1]["stdin"] is subprocess.DEVNULL
    assert calls[2][1]["stdin"] is subprocess.DEVNULL
    assert git_environment["GIT_TERMINAL_PROMPT"] == "0"
    assert git_environment["GCM_INTERACTIVE"] == "Never"
    assert git_environment["GIT_SSH_COMMAND"] == "ssh -oBatchMode=yes"
    assert gh_auth_environment["GH_PROMPT_DISABLED"] == "1"
    assert gh_api_environment["GH_PROMPT_DISABLED"] == "1"
    for environment in (git_environment, gh_auth_environment, gh_api_environment):
        assert "CAYU_CLOUD_API_KEY" not in environment
        assert "CAYU_CLOUD_API_KEY_FILE" not in environment


def test_cloud_github_file_preserves_public_access_without_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    requests: list[tuple[str, dict[str, str], dict[str, str]]] = []

    def run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(command)
        if command[1:3] == ["auth", "status"]:
            raise subprocess.CalledProcessError(1, command)
        raise AssertionError("logged-out public access must not invoke `gh api`")

    class Client:
        def __init__(self, *, follow_redirects: bool, timeout: float) -> None:
            assert follow_redirects is False
            assert timeout == 30.0

        def __enter__(self) -> Client:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def get(
            self,
            url: str,
            *,
            headers: dict[str, str],
            params: dict[str, str],
        ) -> SimpleNamespace:
            requests.append((url, headers, params))
            return SimpleNamespace(status_code=200, content=b"public manifest")

    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setattr(cloud_project.subprocess, "run", run)
    monkeypatch.setattr(
        cloud_project,
        "httpx",
        SimpleNamespace(Client=Client, RequestError=OSError),
        raising=False,
    )

    assert (
        cloud_project._github_file(
            "https://github.com/example/public-project",
            revision="a" * 40,
            path="cayu-cloud.toml",
        )
        == "public manifest"
    )
    assert calls[0] == ["gh", "auth", "status", "--hostname", "github.com"]
    assert len(calls) == 1
    assert requests == [
        (
            "https://api.github.com/repos/example/public-project/contents/cayu-cloud.toml",
            {"Accept": "application/vnd.github.raw+json"},
            {"ref": "a" * 40},
        )
    ]


@pytest.mark.parametrize("failure", ["transport", "service"])
def test_cloud_github_file_distinguishes_anonymous_source_unavailability_from_auth(
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        raise subprocess.CalledProcessError(1, command)

    class Client:
        def __init__(self, *, follow_redirects: bool, timeout: float) -> None:
            assert follow_redirects is False
            assert timeout == 30.0

        def __enter__(self) -> Client:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def get(
            self,
            _url: str,
            *,
            headers: dict[str, str],
            params: dict[str, str],
        ) -> SimpleNamespace:
            assert "Authorization" not in headers
            assert params == {"ref": "a" * 40}
            if failure == "transport":
                raise OSError("network unavailable")
            return SimpleNamespace(status_code=503, content=b"provider detail")

    monkeypatch.setattr(cloud_project.subprocess, "run", run)
    monkeypatch.setattr(
        cloud_project,
        "httpx",
        SimpleNamespace(Client=Client, RequestError=OSError),
        raising=False,
    )

    with pytest.raises(CloudApiError) as raised:
        cloud_project._github_file(
            "https://github.com/example/public-project",
            revision="a" * 40,
            path="cayu-cloud.toml",
        )

    assert raised.value.category == "source_unavailable"
    assert str(raised.value) == ("GitHub was unavailable while resolving the deployment source.")
    assert "provider detail" not in str(raised.value)


@pytest.mark.parametrize(
    "repository",
    [
        "https://github.com/owner/re po",
        "https://github.com/owner/%2e%2e",
        "https://github.com/owner/repo%2Fextra",
    ],
)
def test_cloud_remote_source_rejects_noncanonical_repository_components(
    repository: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("invalid repository reached source resolution")

    monkeypatch.setattr(cloud_project.subprocess, "run", unexpected)

    with pytest.raises(CloudApiError) as raised:
        cloud_project.resolve_project(
            repository,
            manifest_path=None,
            revision="a" * 40,
        )

    assert raised.value.category == "source_repository_invalid"


def test_cloud_github_file_accepts_explicit_gh_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((command, kwargs))
        if command[1:3] == ["auth", "status"]:
            return SimpleNamespace(stdout="")
        return SimpleNamespace(stdout="private manifest")

    monkeypatch.setenv("GH_TOKEN", "github_pat_private_test_value")
    monkeypatch.setattr(cloud_project.subprocess, "run", run)

    assert (
        cloud_project._github_file(
            "https://github.com/example/private-project",
            revision="b" * 40,
            path="cayu-cloud.toml",
        )
        == "private manifest"
    )
    assert len(calls) == 2
    for _, kwargs in calls:
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        assert environment["GH_TOKEN"] == "github_pat_private_test_value"


def test_cloud_github_file_keeps_manifest_error_when_authentication_is_usable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        if command[1:3] == ["auth", "status"]:
            return SimpleNamespace(stdout="")
        raise subprocess.CalledProcessError(1, command, stderr="authenticated 404")

    monkeypatch.setattr(cloud_project.subprocess, "run", run)

    with pytest.raises(cloud_project.CloudApiError) as raised:
        cloud_project._github_file(
            "https://github.com/example/private-project",
            revision="c" * 40,
            path="cayu-cloud.toml",
        )

    assert raised.value.category == "manifest_unavailable"


def test_cloud_deploy_reports_redacted_source_authentication_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "github_pat_must_never_be_rendered"
    api_key_file = tmp_path / "api-key"
    api_key_file.write_text("customer-secret-material\n")
    context_path = tmp_path / "cloud-context.json"
    _write_ready_context(context_path, api_key_file=api_key_file)
    isolated_home = tmp_path / "isolated-home"
    isolated_home.mkdir()
    monkeypatch.setenv("HOME", str(isolated_home))
    monkeypatch.setenv("GH_CONFIG_DIR", str(tmp_path / "gh-config"))
    monkeypatch.delenv("GH_TOKEN", raising=False)
    anonymous_requests: list[tuple[str, dict[str, str]]] = []

    class Client:
        def __init__(self, *, follow_redirects: bool, timeout: float) -> None:
            assert follow_redirects is False
            assert timeout == 30.0

        def __enter__(self) -> Client:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def get(
            self,
            url: str,
            *,
            headers: dict[str, str],
            params: dict[str, str],
        ) -> SimpleNamespace:
            del params
            anonymous_requests.append((url, headers))
            return SimpleNamespace(status_code=404, content=b"")

    def run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        raise subprocess.CalledProcessError(1, command, stderr=f"credential={secret}")

    monkeypatch.setattr(cloud_project.subprocess, "run", run)
    monkeypatch.setattr(cloud_project.httpx, "Client", Client)

    assert (
        main(
            [
                "cloud",
                "--context",
                str(context_path),
                "deploy",
                "https://github.com/example/private-project",
                "--revision",
                "d" * 40,
            ]
        )
        == 2
    )

    rendered = capsys.readouterr().out
    assert json.loads(rendered)["error"] == {
        "category": "source_auth_unavailable",
        "message": (
            "GitHub authentication is unavailable. Run `gh auth login` or set "
            "`GH_TOKEN` for noninteractive use."
        ),
    }
    assert secret not in rendered
    assert len(anonymous_requests) == 2
    assert all("Authorization" not in headers for _, headers in anonymous_requests)


def test_cloud_local_state_failure_remains_machine_readable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    api_key_file = tmp_path / "api-key"
    api_key_file.write_text("customer-secret-material\n")
    context_path = tmp_path / "cloud-context.json"
    _write_ready_context(context_path, api_key_file=api_key_file)
    monkeypatch.setattr(
        cloud_cli,
        "_write_private_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError()),
    )

    assert main(["cloud", "context", "use", str(context_path)]) == 2

    rendered = capsys.readouterr().out
    assert json.loads(rendered)["error"] == {
        "category": "local_state_unavailable",
        "message": "Could not update local Cayu Cloud state.",
    }
    assert "customer-secret-material" not in rendered


def test_cloud_structurally_invalid_api_response_remains_machine_readable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def invalid_response(_arguments: object) -> dict[str, object]:
        raise KeyError("required_api_field")

    monkeypatch.setattr(cloud_cli, "_execute", invalid_response)

    assert main(["cloud", "doctor"]) == 2

    assert json.loads(capsys.readouterr().out)["error"] == {
        "category": "api_response_invalid",
        "message": "Cayu Cloud API response is missing required data.",
    }
