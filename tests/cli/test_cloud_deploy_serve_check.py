"""The deploy check refuses a `cayu serve` web process that could not start on Cloud.

`cayu serve` outside `--dev` needs `cayu[server]` installed and a configured auth
target. The check reads project files and never needs the server extra here, and it
accepts any auth dependency, not only Basic auth.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from cayu.cli import cloud as cloud_cli
from cayu.cli._cloud_deploy_check import check_serve_can_start, run_cloud_deploy_check
from cayu.cli._serve_readiness import GENERATED_AUTH_MODULE
from cayu.cli.scaffold import project_files

_SERVE = "cayu serve --host 0.0.0.0 --port 8000"
_TARGET = "cayu.server.environment_auth:OPERATOR_BASIC_AUTH"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.delenv("CAYU_OPERATOR_USERNAME", raising=False)
    monkeypatch.delenv("CAYU_OPERATOR_PASSWORD", raising=False)
    monkeypatch.setenv("CAYU_CLOUD_EVIDENCE_DIR", str(tmp_path / "evidence"))
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "cloud-config.json"))
    for name in ("serve_check_auth", "cayu.server.environment_auth"):
        sys.modules.pop(name, None)
    yield
    sys.modules.pop("serve_check_auth", None)


def _project(
    root: Path,
    *,
    dependencies: str = '["cayu[postgres,server]==0.8.2"]',
    serve: str | None = f'auth = "{_TARGET}"',
    lock_extras: tuple[str, ...] | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    text = (
        f'[project]\nname = "serve-check-agent"\nversion = "0.1.0"\n'
        f"dependencies = {dependencies}\n\n"
        '[tool.cayu]\nfactory = "app:build_app"\n'
    )
    if serve is not None:
        text += f"\n[tool.cayu.serve]\n{serve}\n"
    (root / "pyproject.toml").write_text(text, encoding="utf-8")
    lock = "version = 1\n"
    if lock_extras is not None:
        lock += (
            '\n[[package]]\nname = "serve-check-agent"\nversion = "0.1.0"\n'
            'source = { virtual = "." }\n'
            f'dependencies = [{{ name = "cayu", extra = {json.dumps(list(lock_extras))} }}]\n'
        )
    (root / "uv.lock").write_text(lock, encoding="utf-8")
    return root


def _codes(root: Path) -> tuple[str, tuple[str, ...]]:
    check = run_cloud_deploy_check(root, serves_web=True, web_command=_SERVE)
    return check.status, check.codes


@pytest.mark.parametrize("preset", ("agent", "coding"))
def test_a_fresh_scaffold_passes_without_local_operator_credentials(
    tmp_path: Path, preset: str
) -> None:
    for relative, content in project_files("smoke-agent", preset=preset).items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    assert _codes(tmp_path) == ("passed", ())
    # The environment target is accepted by name; Cloud supplies its credentials.
    assert "cayu.server.environment_auth" not in sys.modules


def test_missing_auth_target_blocks_with_the_edit_to_make(tmp_path: Path) -> None:
    project = _project(tmp_path, serve=None)

    check = run_cloud_deploy_check(project, serves_web=True, web_command=_SERVE)

    assert check.status == "failed"
    (finding,) = cast("list[dict[str, Any]]", check.public_dict()["blocking"])
    assert finding["code"] == "SERVE_AUTH_MISSING"
    assert finding["path"] == "pyproject.toml:[tool.cayu.serve].auth"
    assert _TARGET in finding["hint"]


def test_missing_server_extra_blocks_even_when_installed_here(tmp_path: Path) -> None:
    project = _project(tmp_path, dependencies='["cayu[postgres]==0.8.2"]')

    check = run_cloud_deploy_check(project, serves_web=True, web_command=_SERVE)

    assert check.codes == ("SERVE_SERVER_EXTRA_MISSING",)
    hint = check.blocking[0].hint or ""
    assert 'change "cayu[postgres]==0.8.2" to "cayu[postgres,server]==0.8.2"' in hint
    assert "uv lock" in hint


@pytest.mark.parametrize("extra", ("all", "server-settings", "server_settings", "oidc"))
def test_extras_that_include_server_dependencies_are_servable(tmp_path: Path, extra: str) -> None:
    project = _project(
        tmp_path,
        dependencies=json.dumps([f"cayu[{extra}]==0.8.2"]),
        lock_extras=(extra,),
    )

    assert _codes(project) == ("passed", ())


@pytest.mark.parametrize("extra", ("all", "server-settings", "oidc"))
def test_server_supersets_still_require_server_dependencies_in_the_lock(
    tmp_path: Path, extra: str
) -> None:
    project = _project(
        tmp_path,
        dependencies=json.dumps([f"cayu[{extra}]==0.8.2"]),
        lock_extras=("postgres",),
    )

    assert _codes(project) == ("failed", ("SERVE_LOCK_MISSING_SERVER_EXTRA",))


@pytest.mark.parametrize(
    ("extras", "expected"),
    [
        (("postgres",), ("failed", ("SERVE_LOCK_MISSING_SERVER_EXTRA",))),
        (("postgres", "server"), ("passed", ())),
        (("all",), ("passed", ())),
        (("server-settings",), ("passed", ())),
        (("server_settings",), ("passed", ())),
        (("oidc",), ("passed", ())),
    ],
)
def test_the_lock_cloud_installs_from_must_include_the_server_extra(
    tmp_path: Path, extras: tuple[str, ...], expected: tuple[str, tuple[str, ...]]
) -> None:
    assert _codes(_project(tmp_path, lock_extras=extras)) == expected


@pytest.mark.parametrize(
    ("module", "expected"),
    [
        # Any request -> AuthContext dependency is accepted, not only BasicAuth.
        (
            "from cayu.server import AuthContext\n\n"
            "def AUTH(request):\n    return AuthContext(subject='oidc-user')\n",
            ("passed", ()),
        ),
        ("OTHER = None\n", ("failed", ("SERVE_AUTH_TARGET_UNRESOLVABLE",))),
        ("AUTH = 'not callable'\n", ("failed", ("SERVE_AUTH_TARGET_UNRESOLVABLE",))),
        # Credentials only Cloud has make the target unavailable here, not failed.
        ("import os\nAUTH = os.environ['OIDC_ISSUER_ONLY_ON_CLOUD']\n", ("unavailable", ())),
        # So does a built-in OIDC target whose variables are set only on Cloud.
        (
            "from cayu.server import OidcBearerAuth\n\n"
            "AUTH = OidcBearerAuth.from_environment('OIDC_ISSUER_ONLY_ON_CLOUD', "
            "'OIDC_AUDIENCE_ONLY_ON_CLOUD')\n",
            ("unavailable", ()),
        ),
    ],
)
def test_a_custom_auth_target_must_resolve_to_a_callable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module: str, expected: tuple
) -> None:
    monkeypatch.delenv("OIDC_ISSUER_ONLY_ON_CLOUD", raising=False)
    project = _project(tmp_path, serve='auth = "serve_check_auth:AUTH"')
    (project / "serve_check_auth.py").write_text(module, encoding="utf-8")

    assert _codes(project) == expected
    assert "serve_check_auth" not in sys.modules


def test_a_target_from_a_package_missing_here_is_unavailable_not_failed(tmp_path: Path) -> None:
    project = _project(tmp_path, serve='auth = "company_identity_not_installed:AUTH"')

    check = run_cloud_deploy_check(project, serves_web=True, web_command=_SERVE)

    assert check.status == "unavailable"
    assert "company_identity_not_installed" in (check.reason or "")


def test_only_a_cayu_serve_web_process_is_checked(tmp_path: Path) -> None:
    project = _project(tmp_path, serve=None, dependencies='["httpx"]')

    for command in ("python -m app", None):
        check = run_cloud_deploy_check(project, serves_web=True, web_command=command)
        assert check.status == "not_applicable"
    assert (
        run_cloud_deploy_check(project, serves_web=False, web_command=_SERVE).status
        == "not_applicable"
    )


def _cloud(arguments: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, Any]:
    code = cloud_cli.run_cloud_cli(arguments)
    return code, json.loads(capsys.readouterr().out)


def _write_manifest(project: Path, command: str = _SERVE) -> None:
    (project / "cayu-cloud.toml").write_text(
        "\n".join(
            (
                "schema_version = 2",
                'application = "serve-check-agent"',
                'name = "Serve Check Agent"',
                'version = "0.1.0"',
                f'entrypoint = "{_SERVE}"',
                'capabilities = ["model.generate"]',
                "cpu_millis = 1000",
                "memory_mb = 2048",
                "timeout_seconds = 900",
                'environment = "python"',
                'compatibility = "cayu>=0.1,<1"',
                'policy_version = "cayu-egress-v1"',
                "",
                "[web]",
                f"command = {json.dumps(command)}",
                "port = 8000",
                "",
            )
        ),
        encoding="utf-8",
    )


def test_deploy_refuses_a_web_process_that_cannot_start_before_any_cloud_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project = _project(tmp_path / "agent", serve=None)
    _write_manifest(project)

    def no_cloud(*args: object, **kwargs: object) -> object:
        raise AssertionError("the deploy check must refuse before contacting Cayu Cloud")

    monkeypatch.setattr(cloud_cli, "_cloud_client", no_cloud)

    code, payload = _cloud(["deploy", str(project)], capsys)

    assert code == 2
    error = payload["error"]
    assert error["category"] == "deploy_check_failed"
    assert "`cayu serve` web process could not start" in error["message"]
    assert _TARGET in error["message"]
    assert "Nothing was uploaded" in error["message"]
    assert [item["code"] for item in error["deploy_check"]["blocking"]] == ["SERVE_AUTH_MISSING"]


def test_deploy_proceeds_for_a_servable_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project = _project(tmp_path / "agent", lock_extras=("postgres", "server"))
    _write_manifest(project)
    deployed: dict[str, object] = {}
    monkeypatch.setattr(cloud_cli, "_cloud_client", lambda *args, **kwargs: object())

    def deploy(arguments: object, **kwargs: object) -> dict[str, object]:
        deployed.update(kwargs)
        return {"operation": "deploy", "result": {}}

    monkeypatch.setattr(cloud_cli, "_deploy", deploy)

    code, _ = _cloud(["deploy", str(project)], capsys)

    assert code == 0
    assert deployed["deploy_check"] == {
        "command": "cayu check --deploy --fail-on warning --json",
        "status": "passed",
    }


def test_init_warns_to_relock_after_adding_the_server_extra(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = _project(
        tmp_path / "agent",
        dependencies='["cayu[postgres]==0.8.1"]',
        serve=None,
        lock_extras=("postgres",),
    )

    code, payload = _cloud(["init", str(project)], capsys)

    assert code == 0
    result = payload["result"]
    assert result["serve"]["next_steps"] == ["uv lock"]
    assert [item["code"] for item in result["deploy_check"]["blocking"]] == [
        "SERVE_LOCK_MISSING_SERVER_EXTRA"
    ]
    (warning,) = result["warnings"]
    assert "Run `uv lock`" in warning


_AUTH_MODULE = (
    "from cayu.server import AuthContext\n\ndef AUTH(request):\n"
    "    return AuthContext(subject='operator')\n"
)


def _web_codes(root: Path, command: str) -> tuple[str, tuple[str, ...]]:
    check = run_cloud_deploy_check(root, serves_web=True, web_command=command)
    return check.status, check.codes


@pytest.mark.parametrize(
    "command",
    [
        "cayu serve --host 0.0.0.0 --auth serve_check_auth:AUTH",
        "cayu serve --host 0.0.0.0 --auth 'serve_check_auth:AUTH'",
        'cayu serve --host 0.0.0.0 --port 8000 --auth="serve_check_auth:AUTH"',
        '"cayu" serve --auth serve_check_auth:AUTH --host 0.0.0.0',
    ],
)
def test_an_auth_override_in_the_web_command_is_the_effective_target(
    tmp_path: Path, command: str
) -> None:
    project = _project(tmp_path, serve=None)
    (project / "serve_check_auth.py").write_text(_AUTH_MODULE, encoding="utf-8")

    assert _web_codes(project, command) == ("passed", ())


def test_a_broken_override_fails_even_with_a_valid_configured_target(tmp_path: Path) -> None:
    project = _project(tmp_path)  # [tool.cayu.serve].auth names the ready-made target.
    (project / "serve_check_auth.py").write_text(_AUTH_MODULE, encoding="utf-8")

    check = run_cloud_deploy_check(
        project,
        serves_web=True,
        web_command="cayu serve --host 0.0.0.0 --auth serve_check_auth:MISSING",
    )

    assert check.status == "failed"
    (finding,) = check.blocking
    assert finding.code == "SERVE_AUTH_TARGET_UNRESOLVABLE"
    assert finding.path == "cayu-cloud.toml:[web].command"
    assert "serve_check_auth:MISSING" in finding.message
    assert "--auth in the [web] command" in (finding.hint or "")


@pytest.mark.parametrize(
    ("command", "message"),
    [
        ("cayu serve --dev --host 0.0.0.0 --port 8000", "non-loopback host"),
        ("cayu serve --dev --host 127.0.0.1", "cannot route requests"),
    ],
)
def test_dev_mode_blocks_a_cloud_web_process(tmp_path: Path, command: str, message: str) -> None:
    project = _project(tmp_path)

    check = run_cloud_deploy_check(project, serves_web=True, web_command=command)

    assert check.codes == ("SERVE_DEV_MODE",)
    assert message in check.blocking[0].message
    assert check.blocking[0].path == "cayu-cloud.toml:[web].command"


@pytest.mark.parametrize(
    ("command", "message"),
    [
        ("cayu serve --dev --auth serve_check_auth:AUTH", "not allowed with argument"),
        ("cayu serve --port 99999", "port must be between"),
        ("cayu serve --unknown", "unrecognized arguments"),
        ("cayu serve --auth 'serve_check_auth:AUTH", "quotation"),
    ],
)
def test_arguments_cayu_serve_would_reject_block(
    tmp_path: Path, command: str, message: str
) -> None:
    project = _project(tmp_path)

    check = run_cloud_deploy_check(project, serves_web=True, web_command=command)

    assert check.codes == ("SERVE_COMMAND_INVALID",)
    assert message in check.blocking[0].message


@pytest.mark.parametrize(
    "command",
    [_SERVE, "cayu serve --host 0.0.0.0 --auth cayu.server.environment_auth:OPERATOR_BASIC_AUTH"],
)
def test_the_ready_made_target_needs_a_runtime_that_ships_it(tmp_path: Path, command: str) -> None:
    project = _project(
        tmp_path,
        dependencies='["cayu[postgres,server]==0.8.1"]',
        serve=None if "--auth" in command else f'auth = "{_TARGET}"',
    )

    check = run_cloud_deploy_check(project, serves_web=True, web_command=command)

    assert check.codes == ("SERVE_AUTH_TARGET_UNRESOLVABLE",)
    finding = check.blocking[0]
    assert "allows cayu 0.8.1 or older" in finding.message
    assert 'change "cayu[postgres,server]==0.8.1" to "cayu[postgres,server]>0.8.1"' in (
        finding.hint or ""
    )
    assert "server_auth.py" in (finding.hint or "")
    assert "cayu.server.environment_auth" not in sys.modules


def test_the_module_init_generates_is_accepted_without_local_credentials(tmp_path: Path) -> None:
    project = _project(
        tmp_path,
        dependencies='["cayu[postgres,server]==0.8.1"]',
        serve=f'auth = "{GENERATED_AUTH_MODULE.target}"',
    )
    (project / GENERATED_AUTH_MODULE.path).write_text(GENERATED_AUTH_MODULE.content)

    assert _codes(project) == ("passed", ())
    assert "server_auth" not in sys.modules

    # An edited module is imported like any custom target.
    (project / GENERATED_AUTH_MODULE.path).write_text("OPERATOR_BASIC_AUTH = None\n")
    sys.modules.pop("server_auth", None)
    assert _codes(project) == ("failed", ("SERVE_AUTH_TARGET_UNRESOLVABLE",))
    sys.modules.pop("server_auth", None)


def test_a_service_factory_refuses_an_auth_override(tmp_path: Path) -> None:
    project = _project(tmp_path, serve=None)
    pyproject = project / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text().replace(
            'factory = "app:build_app"\n',
            'factory = "app:build_app"\nservice_factory = "service:build_service"\n',
        )
    )

    findings, _ = check_serve_can_start(
        project, "cayu serve --host 0.0.0.0 --auth serve_check_auth:AUTH"
    )

    assert [item.code for item in findings] == ["SERVE_AUTH_WITH_SERVICE_FACTORY"]
    assert findings[0].path == "cayu-cloud.toml:[web].command"


def test_deploy_uploads_a_project_protected_only_by_the_command_line_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project = _project(tmp_path / "agent", serve=None, lock_extras=("postgres", "server"))
    (project / "serve_check_auth.py").write_text(_AUTH_MODULE, encoding="utf-8")
    _write_manifest(project, "cayu serve --host 0.0.0.0 --port 8000 --auth serve_check_auth:AUTH")
    deployed: dict[str, object] = {}
    monkeypatch.setattr(cloud_cli, "_cloud_client", lambda *args, **kwargs: object())

    def deploy(arguments: object, **kwargs: object) -> dict[str, object]:
        deployed.update(kwargs)
        return {"operation": "deploy", "result": {}}

    monkeypatch.setattr(cloud_cli, "_deploy", deploy)

    code, _ = _cloud(["deploy", str(project)], capsys)

    assert code == 0
    assert deployed["deploy_check"] == {
        "command": "cayu check --deploy --fail-on warning --json",
        "status": "passed",
    }
