"""The deploy check refuses a `cayu serve` web process that could not start on Cloud.

`cayu serve` outside `--dev` needs `cayu[server]` installed and a configured auth
target. The check reads project files and never needs the server extra here, and it
accepts any auth dependency, not only Basic auth.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from cayu.cli import _cloud_deploy_check as cloud_deploy_check
from cayu.cli import cloud as cloud_cli
from cayu.cli import main, scaffold
from cayu.cli._cloud_deploy_check import check_serve_can_start, run_cloud_deploy_check
from cayu.cli._serve_readiness import GENERATED_AUTH_MODULE
from cayu.cli.scaffold import project_files

_SERVE = "cayu serve --host 0.0.0.0 --port 8000"
_TARGET = "cayu.server.environment_auth:OPERATOR_BASIC_AUTH"
_MODULE_FILE = "cayu/server/environment_auth.py"
_WHEEL = "vendor/cayu-0.16.3-py3-none-any.whl"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.delenv("CAYU_OPERATOR_USERNAME", raising=False)
    monkeypatch.delenv("CAYU_OPERATOR_PASSWORD", raising=False)
    monkeypatch.setenv("CAYU_CLOUD_EVIDENCE_DIR", str(tmp_path / "evidence"))
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "cloud-config.json"))
    project_modules = ("serve_check_auth", "deploy_check_app", "missing_deploy_auth")
    for name in (*project_modules, "cayu.server.environment_auth"):
        sys.modules.pop(name, None)
    yield
    for name in project_modules:
        sys.modules.pop(name, None)


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


def _write_scaffold(root: Path, *, preset: str) -> None:
    for relative, content in project_files("smoke-agent", preset=preset).items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


@pytest.mark.parametrize("preset", ("agent", "coding"))
def test_a_fresh_scaffold_passes_without_local_operator_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preset: str
) -> None:
    # A release install pins the release it scaffolds with.
    monkeypatch.setattr(scaffold, "_cayu_source_checkout", lambda: None)
    monkeypatch.setattr(scaffold, "_installed_cayu_version", lambda: "0.16.3")
    _write_scaffold(tmp_path, preset=preset)

    assert _codes(tmp_path) == ("passed", ())
    # The environment target is accepted by name; Cloud supplies its credentials.
    assert "cayu.server.environment_auth" not in sys.modules


def test_a_scaffold_from_a_source_checkout_points_outside_the_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "cayu"
    (checkout / "src" / "cayu" / "server").mkdir(parents=True)
    (checkout / "src" / _MODULE_FILE).write_text("")
    monkeypatch.setattr(
        scaffold,
        "_cayu_source_checkout",
        lambda: scaffold._CayuSourceCheckout(root=checkout, version="0.16.3"),
    )
    project = tmp_path / "agent"
    _write_scaffold(project, preset="agent")

    check = run_cloud_deploy_check(project, serves_web=True, web_command=_SERVE)

    # Cloud receives only the project directory, so the checkout is not there.
    assert check.codes == ("SERVE_AUTH_TARGET_UNRESOLVABLE",)
    assert (
        f"the cayu path source {json.dumps(str(checkout))} in [tool.uv.sources] is an "
        "absolute path, which exists only on this machine"
    ) in check.blocking[0].message


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


def _checkable_project(root: Path) -> Path:
    project = _project(root, dependencies='["cayu[server]>=0.16.3"]', lock_extras=("server",))
    (project / "deploy_check_app.py").write_text(
        "from cayu import AgentSpec, CayuApp\n"
        "from cayu.evals import ScriptedModelProvider\n"
        "def build_app():\n"
        "    app = CayuApp(enable_logging=False)\n"
        "    app.register_provider(ScriptedModelProvider([]), default=True)\n"
        "    app.register_agent(AgentSpec(name='agent', model='scripted-model'))\n"
        "    return app\n",
        encoding="utf-8",
    )
    pyproject = project / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text().replace("app:build_app", "deploy_check_app:build_app"),
        encoding="utf-8",
    )
    return project


@pytest.mark.parametrize("absolute_source", [True, False])
def test_check_deploy_agrees_with_init_and_deploy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    absolute_source: bool,
) -> None:
    project = _checkable_project(tmp_path / "agent")
    if absolute_source:
        with (project / "pyproject.toml").open("a") as stream:
            stream.write(
                "\n[tool.uv.sources]\n"
                f"cayu = {{ path = {json.dumps(str(tmp_path / 'cayu'))}, editable = true }}\n"
            )
    monkeypatch.chdir(project)

    code, initialized = _cloud(["init", ".", "--force"], capsys)
    assert code == 0
    init_check = initialized["result"]["deploy_check"]
    assert init_check["status"] == ("failed" if absolute_source else "passed")

    check_code = main(["check", "--deploy", "--fail-on", "warning", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert check_code == (1 if absolute_source else 0), report
    assert "error" not in report
    assert [
        {key: diagnostic[key] for key in expected}
        for diagnostic, expected in zip(
            report["diagnostics"], init_check.get("blocking", []), strict=True
        )
    ] == init_check.get("blocking", [])

    def no_cloud(*args: object, **kwargs: object) -> object:
        raise AssertionError("the rejected deploy must not contact Cloud")

    deployed: dict[str, object] = {}

    def deploy(arguments: object, **kwargs: object) -> dict[str, object]:
        deployed.update(kwargs)
        return {"operation": "deploy", "result": {}}

    monkeypatch.setattr(
        cloud_cli, "_cloud_client", no_cloud if absolute_source else lambda *a, **kw: object()
    )
    monkeypatch.setattr(cloud_cli, "_deploy", deploy)
    code, payload = _cloud(["deploy", "."], capsys)
    if absolute_source:
        assert code == 2
        assert payload["error"]["category"] == "deploy_check_failed"
        assert payload["error"]["deploy_check"] == init_check
        assert deployed == {}
    else:
        assert code == 0
        assert deployed["deploy_check"] == init_check


@pytest.mark.parametrize(
    "case,expected_code",
    [
        ("extra", "SERVE_SERVER_EXTRA_MISSING"),
        ("lock", "SERVE_LOCK_MISSING_SERVER_EXTRA"),
        ("auth", "SERVE_AUTH_MISSING"),
        ("override", "SERVE_AUTH_TARGET_UNRESOLVABLE"),
        ("reload", "SERVE_RELOAD_MODE"),
        ("dev", "SERVE_DEV_MODE"),
        ("arguments", "SERVE_COMMAND_INVALID"),
    ],
)
def test_check_deploy_includes_sibling_serve_findings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    case: str,
    expected_code: str,
) -> None:
    project = _checkable_project(tmp_path / "agent")
    pyproject = project / "pyproject.toml"
    if case == "extra":
        pyproject.write_text(pyproject.read_text().replace("cayu[server]", "cayu"))
    elif case == "lock":
        lock = project / "uv.lock"
        lock.write_text(lock.read_text().replace('extra = ["server"]', "extra = []"))
    elif case == "auth":
        pyproject.write_text(pyproject.read_text().replace(f'auth = "{_TARGET}"', ""))
    elif case == "override":
        (project / "missing_deploy_auth.py").write_text("authenticate = None\n")
    command = _SERVE + {
        "override": " --auth missing_deploy_auth:authenticate",
        "reload": " --reload",
        "dev": " --dev",
        "arguments": " --unknown-option",
    }.get(case, "")
    _write_manifest(project, command)
    monkeypatch.chdir(project)

    def no_cloud(*args: object, **kwargs: object) -> object:
        raise AssertionError("invalid deployment must not contact Cloud")

    monkeypatch.setattr(cloud_cli, "_cloud_client", no_cloud)
    code, payload = _cloud(["deploy", "."], capsys)
    assert code == 2
    blocking = payload["error"]["deploy_check"]["blocking"]
    assert [finding["code"] for finding in blocking] == [expected_code]
    assert main(["check", "--deploy", "--json"]) == 1
    diagnostics = json.loads(capsys.readouterr().out)["diagnostics"]
    assert [
        {key: item[key] for key in expected}
        for item, expected in zip(diagnostics, blocking, strict=True)
    ] == blocking


@pytest.mark.parametrize("scope", ["ordinary", "tag", "other-web", "worker", "absent"])
def test_serve_deploy_findings_respect_check_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    scope: str,
) -> None:
    project = _checkable_project(tmp_path / "agent")
    _write_manifest(project, "uvicorn custom:app" if scope == "other-web" else _SERVE)
    pyproject = project / "pyproject.toml"
    pyproject.write_text(pyproject.read_text().replace(f'auth = "{_TARGET}"', ""))
    manifest = project / "cayu-cloud.toml"
    if scope == "worker":
        manifest.write_text(
            manifest.read_text().replace("[web]", "[worker]").replace("port = 8000", "")
        )
    elif scope == "absent":
        manifest.unlink()
    monkeypatch.chdir(project)
    arguments = ["check", "--json"]
    if scope != "ordinary":
        arguments.append("--deploy")
    if scope == "tag":
        arguments.extend(["--tag", "security"])
    code = main(arguments)
    report = json.loads(capsys.readouterr().out)
    assert "error" not in report
    assert all(not item["code"].startswith("SERVE_") for item in report["diagnostics"])
    if scope != "ordinary":
        assert code == 0


def test_check_deploy_preserves_scaffold_import_safety_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project = tmp_path / "agent"
    project.mkdir()
    _write_scaffold(project, preset="agent")
    app = project / "app.py"
    app.write_text(
        'from pathlib import Path\nPath("unsafe-auth-import.txt").touch()\n'
        + app.read_text().replace("from __future__ import annotations", "")
    )
    _write_manifest(project, _SERVE + " --auth app:build_app")
    monkeypatch.chdir(project)
    sys.modules.pop("app", None)

    assert main(["check", "--deploy", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert "SCAFFOLD_IMPORT_SIDE_EFFECT" in [item["code"] for item in report["diagnostics"]]
    assert not (project / "unsafe-auth-import.txt").exists()


def test_check_deploy_uses_the_web_command_auth_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project = _checkable_project(tmp_path / "agent")
    pyproject = project / "pyproject.toml"
    pyproject.write_text(pyproject.read_text().replace(_TARGET, "unused_auth:authenticate"))
    _write_manifest(project, _SERVE + f" --auth {_TARGET}")
    monkeypatch.chdir(project)

    assert main(["check", "--deploy", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["diagnostics"] == []


@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize("unsafe", [False, True])
def test_check_deploy_checks_custom_auth_source_before_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    override: bool,
    unsafe: bool,
) -> None:
    project = tmp_path / "agent"
    project.mkdir()
    _write_scaffold(project, preset="agent")
    package = project / "auth_extension"
    package.mkdir()
    (package / "__init__.py").touch()
    (package / "unsafe_auth.py").write_text(
        ('from pathlib import Path\nPath("unsafe-auth-import.txt").touch()\n' if unsafe else "")
        + "def authenticate(request):\n    return None\n"
    )
    command = _SERVE
    if override:
        command += " --auth auth_extension.unsafe_auth:authenticate"
    else:
        pyproject = project / "pyproject.toml"
        pyproject.write_text(
            pyproject.read_text().replace(_TARGET, "auth_extension.unsafe_auth:authenticate")
        )
    _write_manifest(project, command)
    monkeypatch.chdir(project)
    sys.modules.pop("auth_extension", None)
    sys.modules.pop("auth_extension.unsafe_auth", None)
    resolved: list[str] = []
    load_target = cloud_deploy_check.load_target

    def observe_load(target: str, **kwargs: Any) -> Any:
        resolved.append(target)
        return load_target(target, **kwargs)

    monkeypatch.setattr(cloud_deploy_check, "load_target", observe_load)

    # The scaffold has no private memory key file; the check lends the factory a
    # throwaway key, so only the unsafe auth source fails it.
    assert main(["check", "--deploy", "--json"]) == (1 if unsafe else 0)
    report = json.loads(capsys.readouterr().out)
    assert not (project / "unsafe-auth-import.txt").exists()
    assert (
        "SCAFFOLD_IMPORT_SIDE_EFFECT" in [item["code"] for item in report["diagnostics"]]
    ) is unsafe
    assert resolved == ([] if unsafe else ["auth_extension.unsafe_auth:authenticate"])


@pytest.mark.parametrize("problem", ["manifest", "auth-unavailable"])
def test_check_deploy_does_not_claim_success_when_preflight_cannot_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    problem: str,
) -> None:
    project = _checkable_project(tmp_path / "agent")
    _write_manifest(project, _SERVE + " --auth absent_cloud_dependency:authenticate")
    if problem == "manifest":
        (project / "cayu-cloud.toml").write_text("[malformed")
    monkeypatch.chdir(project)

    assert main(["check", "--deploy", "--json"]) == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["code"] == "PROJECT_CHECK_FAILED"
    assert (
        "Invalid Cayu Cloud TOML" if problem == "manifest" else "module was not found"
    ) in error["message"]


@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize("unsafe", [False, True])
@pytest.mark.parametrize(
    "parent,payload_file,missing",
    [
        ("auth_extension", "auth_extension/__init__.py", "missing"),
        ("auth_extension", "auth_extension/__init__.py", "missing.deep"),
        ("auth_extension.nested", "auth_extension/__init__.py", "missing"),
        ("auth_extension.nested", "auth_extension/nested/__init__.py", "missing"),
        ("auth_extension.namespace.deep", "auth_extension/namespace/deep/__init__.py", "missing"),
        ("auth_extension.shadow", "auth_extension/shadow.py", "missing"),
    ],
)
def test_check_deploy_scans_parents_of_missing_auth_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    override: bool,
    unsafe: bool,
    parent: str,
    payload_file: str,
    missing: str,
) -> None:
    project = tmp_path / "agent"
    project.mkdir()
    _write_scaffold(project, preset="agent")
    directory = project
    for part in parent.split("."):
        if part == "shadow":
            break
        directory /= part
        directory.mkdir()
        if part != "namespace":
            (directory / "__init__.py").touch()
    (project / payload_file).write_text(
        'from pathlib import Path\nPath("unsafe-parent-import.txt").touch()\n'
        if unsafe
        else "SAFE_VALUE = 1\n"
    )
    target = f"{parent}.{missing}:authenticate"
    command = _SERVE
    if override:
        command += f" --auth {target}"
    else:
        pyproject = project / "pyproject.toml"
        pyproject.write_text(pyproject.read_text().replace(_TARGET, target))
    _write_manifest(project, command)
    monkeypatch.chdir(project)

    code = main(["check", "--deploy", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert not (project / "unsafe-parent-import.txt").exists()
    assert code == 1, report
    findings = report["diagnostics"]
    if unsafe:
        assert any(
            item["code"] == "SCAFFOLD_IMPORT_SIDE_EFFECT"
            and item["path"].startswith(f"{payload_file}:")
            for item in findings
        )
        assert not any(item["code"] == "SERVE_AUTH_TARGET_UNRESOLVABLE" for item in findings)
    else:
        assert any(item["code"] == "SERVE_AUTH_TARGET_UNRESOLVABLE" for item in findings)
        assert not any(item["code"] == "SCAFFOLD_IMPORT_SIDE_EFFECT" for item in findings)


@pytest.mark.parametrize("state", ["absent", "valid-link", "dangling-link"])
def test_check_deploy_distinguishes_absent_and_unreadable_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    state: str,
) -> None:
    project = _checkable_project(tmp_path / "agent")
    if state != "absent":
        _write_manifest(project)
        manifest = project / "cayu-cloud.toml"
        configuration = project / "config"
        configuration.mkdir()
        target = configuration / "production.toml"
        manifest.rename(target)
        manifest.symlink_to(Path("config/production.toml"))
        if state == "dangling-link":
            target.unlink()
    monkeypatch.chdir(project)

    code = main(["check", "--deploy", "--json"])
    report = json.loads(capsys.readouterr().out)
    if state == "dangling-link":
        assert code == 2, report
        assert report["error"]["code"] == "PROJECT_CHECK_FAILED"
        assert "Could not read Cayu Cloud manifest" in report["error"]["message"]
    else:
        assert code == 0, report
        assert report["diagnostics"] == []


@pytest.mark.parametrize(
    "entrance",
    ["auth-override", "explicit", "doctor", "auth-config", "factory-config", "service-config"],
)
@pytest.mark.parametrize("unsafe", [False, True])
def test_source_gate_preserves_raw_and_configured_target_identity(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    entrance: str,
    unsafe: bool,
) -> None:
    assert main(["new", "smoke-agent", "--dir", str(tmp_path), "--preset", "agent"]) == 0
    capsys.readouterr()
    project = tmp_path / "smoke-agent"
    configured = entrance.endswith("-config")
    selected = "target_extension" if configured else " target_extension"
    for name in ("target_extension", " target_extension"):
        package = project / name
        package.mkdir()
        (package / "__init__.py").write_text(
            'from pathlib import Path\nPath("unexpected-parent-import.txt").touch()\n'
            if (name == selected) == unsafe
            else "SAFE_VALUE = 1\n"
        )
    raw_target = " target_extension.missing:authenticate "
    pyproject = project / "pyproject.toml"
    command = ["check", "--json"]
    if entrance == "auth-override":
        _write_manifest(project, _SERVE + f" --auth {json.dumps(raw_target)}")
        command.append("--deploy")
    elif entrance == "auth-config":
        pyproject.write_text(pyproject.read_text().replace(_TARGET, raw_target))
        _write_manifest(project)
        command.append("--deploy")
    elif entrance in {"factory-config", "service-config"}:
        setting = (
            f"factory = {json.dumps(raw_target)}"
            if entrance == "factory-config"
            else f'factory = "app:build_app"\nservice_factory = {json.dumps(raw_target)}'
        )
        pyproject.write_text(pyproject.read_text().replace('factory = "app:build_app"', setting))
    elif entrance == "explicit":
        command.append(raw_target)
    else:
        command = ["doctor", raw_target, "--bundle", str(tmp_path / "support.zip"), "--json"]
    # Separate processes retain Python's real unusual-name import semantics,
    # without a preceding test's sys.modules cache substituting another project.
    result = subprocess.run(
        [sys.executable, "-m", "cayu", *command],
        cwd=project,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    report = json.loads(result.stdout)
    assert not (project / "unexpected-parent-import.txt").exists()
    if entrance == "doctor":
        assert result.returncode == 2, report
        with zipfile.ZipFile(tmp_path / "support.zip") as archive:
            bundle = json.loads(archive.read("report.json"))
        reasons = {item.get("reason_code") for item in bundle["collectors"]}
        assert ("source_validation_failed" in reasons) is unsafe
    else:
        blockers = [
            item
            for item in report.get("diagnostics", [])
            if item["code"] == "SCAFFOLD_IMPORT_SIDE_EFFECT"
        ]
        assert bool(blockers) is unsafe, report
        if unsafe:
            assert result.returncode == 1
            assert all(item["path"].startswith(f"{selected}/__init__.py:") for item in blockers)
        else:
            assert result.returncode == (1 if entrance == "auth-config" else 2), report


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


def _vendored_project(root: Path, source: str) -> Path:
    """The project shape that bundles a Cayu build and installs it by path."""

    project = _project(root, dependencies='["cayu[postgres,server]>=0.16.1"]')
    pyproject = project / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text() + f"\n[tool.uv.sources]\ncayu = {{ path = {json.dumps(source)} }}\n",
        encoding="utf-8",
    )
    return project


def _wheel(path: Path, *members: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as wheel:
        for member in members:
            wheel.writestr(member, "")


def test_a_vendored_wheel_that_ships_the_module_passes(tmp_path: Path) -> None:
    project = _vendored_project(tmp_path, _WHEEL)
    _wheel(project / _WHEEL, "cayu/__init__.py", _MODULE_FILE)

    assert _codes(project) == ("passed", ())
    assert "cayu.server.environment_auth" not in sys.modules


def test_a_vendored_wheel_without_the_module_names_the_path_source(tmp_path: Path) -> None:
    project = _vendored_project(tmp_path, _WHEEL)
    _wheel(project / _WHEEL, "cayu/__init__.py", "cayu/server/app.py")

    check = run_cloud_deploy_check(project, serves_web=True, web_command=_SERVE)

    assert check.codes == ("SERVE_AUTH_TARGET_UNRESOLVABLE",)
    finding = check.blocking[0]
    assert finding.message == (
        f"`cayu serve` cannot load its auth target {_TARGET}: the cayu path source "
        f'"{_WHEEL}" in [tool.uv.sources] does not contain cayu.server.environment_auth.'
    )
    hint = finding.hint or ""
    assert hint.startswith(
        "Point the cayu path source in [tool.uv.sources] at a cayu wheel, source "
        "distribution, or source tree inside the project that contains "
        "cayu.server.environment_auth"
    )
    assert "server_auth.py" in hint
    assert "0.8.1 or older" not in finding.message + hint


def test_a_vendored_wheel_left_out_of_the_upload_is_refused(tmp_path: Path) -> None:
    # A deploy never uploads dist/ or build/, even when Git tracks them.
    project = _vendored_project(tmp_path, "dist/cayu-0.16.3-py3-none-any.whl")
    _wheel(project / "dist" / "cayu-0.16.3-py3-none-any.whl", _MODULE_FILE)

    check = run_cloud_deploy_check(project, serves_web=True, web_command=_SERVE)

    assert check.codes == ("SERVE_AUTH_TARGET_UNRESOLVABLE",)
    assert check.blocking[0].message == (
        f"`cayu serve` cannot load its auth target {_TARGET}: the cayu path source "
        '"dist/cayu-0.16.3-py3-none-any.whl" in [tool.uv.sources] contains '
        "cayu.server.environment_auth, but Cayu Cloud would not receive "
        "dist/cayu-0.16.3-py3-none-any.whl: Git ignores it, or it is under a directory "
        "`cayu cloud deploy` never uploads, such as dist or build."
    )


def test_a_git_ignored_vendored_wheel_is_refused_until_it_is_tracked(tmp_path: Path) -> None:
    project = _vendored_project(tmp_path, _WHEEL)
    _wheel(project / _WHEEL, _MODULE_FILE)
    subprocess.run(["git", "init", "-q", str(project)], check=True)
    (project / ".gitignore").write_text("*.whl\n")

    check = run_cloud_deploy_check(project, serves_web=True, web_command=_SERVE)

    assert check.codes == ("SERVE_AUTH_TARGET_UNRESOLVABLE",)
    assert f"Cayu Cloud would not receive {_WHEEL}: Git ignores it" in check.blocking[0].message
    subprocess.run(["git", "-C", str(project), "add", "-f", _WHEEL], check=True)
    assert _codes(project) == ("passed", ())


@pytest.mark.parametrize(
    ("ignored", "missing"),
    [
        # The symlink uv installs from is ignored, though the wheel it names is not.
        ("vendor/\n", _WHEEL),
        # The wheel is ignored, though the symlink to it is not.
        ("wheels/\n", "wheels/cayu-0.16.3-py3-none-any.whl"),
    ],
    ids=["symlink-ignored", "wheel-ignored"],
)
def test_every_symlink_and_file_on_the_way_must_be_uploaded(
    tmp_path: Path, ignored: str, missing: str
) -> None:
    project = _vendored_project(tmp_path, _WHEEL)
    _wheel(project / "wheels" / "cayu-0.16.3-py3-none-any.whl", _MODULE_FILE)
    (project / "vendor").mkdir()
    (project / _WHEEL).symlink_to("../wheels/cayu-0.16.3-py3-none-any.whl")
    subprocess.run(["git", "init", "-q", str(project)], check=True)
    (project / ".gitignore").write_text(ignored)

    check = run_cloud_deploy_check(project, serves_web=True, web_command=_SERVE)

    assert check.codes == ("SERVE_AUTH_TARGET_UNRESOLVABLE",)
    assert f"Cayu Cloud would not receive {missing}: Git ignores it" in check.blocking[0].message
    (project / ".gitignore").write_text("")
    assert _codes(project) == ("passed", ())


def test_a_directory_a_symlink_passes_through_must_be_in_the_upload(tmp_path: Path) -> None:
    project = _vendored_project(tmp_path, _WHEEL)
    _wheel(project / "wheels" / "cayu-0.16.3-py3-none-any.whl", _MODULE_FILE)
    (project / "vendor").mkdir()
    (project / _WHEEL).symlink_to("../empty/../wheels/cayu-0.16.3-py3-none-any.whl")
    # An empty directory holds no uploaded file, so Cloud's copy lacks it.
    (project / "empty").mkdir()

    check = run_cloud_deploy_check(project, serves_web=True, web_command=_SERVE)

    assert check.codes == ("SERVE_AUTH_TARGET_UNRESOLVABLE",)
    assert "Cayu Cloud would not receive empty: Git ignores it" in check.blocking[0].message
    (project / "empty" / "README").write_text("")
    assert _codes(project) == ("passed", ())


def test_a_source_tree_whose_module_file_is_not_uploaded_is_refused(tmp_path: Path) -> None:
    project = _vendored_project(tmp_path, "vendor/cayu")
    module = project / "vendor" / "cayu" / "src" / _MODULE_FILE
    module.parent.mkdir(parents=True)
    module.write_text("")
    subprocess.run(["git", "init", "-q", str(project)], check=True)
    (project / ".gitignore").write_text("vendor/cayu/src/\n")

    check = run_cloud_deploy_check(project, serves_web=True, web_command=_SERVE)

    assert check.codes == ("SERVE_AUTH_TARGET_UNRESOLVABLE",)
    assert (
        "Cayu Cloud would not receive vendor/cayu/src/cayu/server/environment_auth.py"
    ) in check.blocking[0].message


def test_a_missing_auth_target_hint_names_the_path_source(tmp_path: Path) -> None:
    project = _vendored_project(tmp_path, _WHEEL)
    pyproject = project / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text().replace(f'\n[tool.cayu.serve]\nauth = "{_TARGET}"\n', "")
    )
    _wheel(project / _WHEEL, "cayu/__init__.py")

    check = run_cloud_deploy_check(project, serves_web=True, web_command=_SERVE)

    assert check.codes == ("SERVE_AUTH_MISSING",)
    assert (
        f'because the cayu path source "{_WHEEL}" in [tool.uv.sources] does not contain '
        "cayu.server.environment_auth"
    ) in (check.blocking[0].hint or "")


def test_a_path_source_outside_the_project_is_refused(tmp_path: Path) -> None:
    project = _vendored_project(tmp_path / "agent", "../cayu")
    module = tmp_path / "cayu" / "src" / _MODULE_FILE
    module.parent.mkdir(parents=True)
    module.write_text("")

    check = run_cloud_deploy_check(project, serves_web=True, web_command=_SERVE)

    assert check.codes == ("SERVE_AUTH_TARGET_UNRESOLVABLE",)
    assert check.blocking[0].message == (
        f"`cayu serve` cannot load its auth target {_TARGET}: the cayu path source "
        '"../cayu" in [tool.uv.sources] leads outside the project directory, and Cayu '
        "Cloud receives only the project's files."
    )


def test_a_source_tree_inside_the_project_still_passes(tmp_path: Path) -> None:
    project = _vendored_project(tmp_path, "vendor/cayu")
    module = project / "vendor" / "cayu" / "src" / _MODULE_FILE
    module.parent.mkdir(parents=True)
    module.write_text("")

    assert _codes(project) == ("passed", ())


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


@pytest.mark.parametrize("dev", [False, True])
def test_reload_mode_blocks_a_cloud_web_process(tmp_path: Path, dev: bool) -> None:
    project = _project(tmp_path)
    command = "cayu serve --reload --dev" if dev else _SERVE + " --reload"

    check = run_cloud_deploy_check(project, serves_web=True, web_command=command)

    assert check.status == "failed"
    assert "SERVE_RELOAD_MODE" in check.codes
    if dev:
        assert "SERVE_DEV_MODE" in check.codes
    finding = next(item for item in check.blocking if item.code == "SERVE_RELOAD_MODE")
    assert finding.path == "cayu-cloud.toml:[web].command"
    assert "Remove --reload" in finding.hint
