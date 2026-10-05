"""`cayu cloud init` makes the `cayu serve` web process it writes able to start."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from cayu.cli import main
from cayu.cli._serve_readiness import (
    GENERATED_AUTH_MODULE,
    declared_runtime_ships_environment_auth,
    environment_auth_upgrade_edits,
    plan_serve_setup,
    requirement_ships_environment_auth,
)
from cayu.cli._targets import load_target
from cayu.cli.project import project_context
from cayu.cli.scaffold import project_files
from cayu.server import BasicAuth

_TARGET = "cayu.server.environment_auth:OPERATOR_BASIC_AUTH"
_GENERATED_TARGET = "server_auth:OPERATOR_BASIC_AUTH"


def _init(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> tuple[int, dict]:
    status = main(["cloud", "init", str(tmp_path)])
    return status, json.loads(capsys.readouterr().out)


def _write_scaffold(root: Path, *, preset: str) -> None:
    for relative, content in project_files("smoke-agent", preset=preset).items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


@pytest.mark.parametrize("preset", ("agent", "coding"))
def test_init_leaves_a_servable_scaffold_unchanged(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    preset: str,
) -> None:
    _write_scaffold(tmp_path, preset=preset)
    original = (tmp_path / "pyproject.toml").read_text()

    status, output = _init(tmp_path, capsys)

    assert status == 0
    serve = output["result"]["serve"]
    assert serve["server_extra"]["status"] == "present"
    assert serve["auth"] == {"status": "kept", "target": _TARGET}
    assert serve["pyproject_changes"] == []
    assert serve["next_steps"] == []
    assert "cayu cloud service credentials" in serve["notes"][0]
    assert (tmp_path / "pyproject.toml").read_text() == original


def test_init_does_not_add_serve_auth_to_a_public_service(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_scaffold(tmp_path, preset="service")
    original = (tmp_path / "pyproject.toml").read_text()

    status, output = _init(tmp_path, capsys)

    assert status == 0
    assert output["result"]["serve"]["auth"] == {"status": "service_factory", "target": None}
    assert (tmp_path / "pyproject.toml").read_text() == original


def _released_scaffold_pyproject(requirement: str = "cayu[postgres]==0.8.1") -> str:
    # The pyproject `cayu new` wrote with Cayu 0.8.1, which lacks
    # cayu.server.environment_auth, including its development pin.
    return f"""[project]
name = "smoke-agent"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["{requirement}"]

[project.optional-dependencies]
dev = ["cayu[postgres,server]==0.8.1", "pytest"]

[tool.cayu]
factory = "app:build_app"
eval_target = "evals.agent:build_eval"

[tool.cayu.session_store]
backend = "sqlite"
path = "data/cayu.db"
"""


def _load_generated_module(
    root: Path, monkeypatch: pytest.MonkeyPatch, **environment: str
) -> object:
    for name in ("CAYU_OPERATOR_USERNAME", "CAYU_OPERATOR_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    with project_context(root):
        return load_target(_GENERATED_TARGET, label="Serve authentication target")


def test_init_adds_server_extra_and_a_project_auth_module_to_a_released_scaffold(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "pyproject.toml").write_text(_released_scaffold_pyproject())

    status, output = _init(tmp_path, capsys)

    assert status == 0
    serve = output["result"]["serve"]
    # 0.8.1 has no cayu.server.environment_auth, so the target is a project module.
    assert serve["pyproject_changes"] == [
        '[project].dependencies: "cayu[postgres]==0.8.1" -> "cayu[postgres,server]==0.8.1"',
        f'[tool.cayu.serve].auth = "{_GENERATED_TARGET}"',
    ]
    assert serve["auth"] == {"status": "added", "target": _GENERATED_TARGET}
    assert serve["auth_module"] == {"path": "server_auth.py", "status": "created"}
    assert serve["next_steps"] == ["uv lock"]
    assert "lack cayu.server.environment_auth" in serve["notes"][1]
    text = (tmp_path / "pyproject.toml").read_text()
    assert text.endswith(f'\n[tool.cayu.serve]\nauth = "{_GENERATED_TARGET}"\n')
    document = tomllib.loads(text)
    assert document["project"]["dependencies"] == ["cayu[postgres,server]==0.8.1"]
    assert document["project"]["optional-dependencies"]["dev"][0] == "cayu[postgres,server]==0.8.1"
    assert document["tool"]["cayu"]["serve"] == {"auth": _GENERATED_TARGET}
    assert document["tool"]["cayu"]["session_store"]["path"] == "data/cayu.db"
    module = (tmp_path / "server_auth.py").read_text()
    assert module == GENERATED_AUTH_MODULE.content
    # Only BasicAuth's constructor, present in every release, is used.
    assert "environment_auth" not in module.split('"""')[2]
    assert "from_environment" not in module

    auth = _load_generated_module(
        tmp_path,
        monkeypatch,
        CAYU_OPERATOR_USERNAME="operator",
        CAYU_OPERATOR_PASSWORD="secret",
    )
    assert isinstance(auth, BasicAuth)
    assert (auth.username, auth.password) == ("operator", "secret")
    with pytest.raises(RuntimeError, match="unset or empty: CAYU_OPERATOR_PASSWORD"):
        _load_generated_module(
            tmp_path,
            monkeypatch,
            CAYU_OPERATOR_USERNAME="operator",
            CAYU_OPERATOR_PASSWORD="  ",
        )

    # Rerunning init keeps the module and the target.
    status, output = main(["cloud", "init", "--force", str(tmp_path)]), None
    assert status == 0
    output = json.loads(capsys.readouterr().out)
    assert output["result"]["serve"]["auth"] == {"status": "kept", "target": _GENERATED_TARGET}
    assert (tmp_path / "server_auth.py").read_text() == module


@pytest.mark.parametrize("requirement", ["cayu>=0.8", "cayu>=0.7,<1", "cayu", "cayu~=0.8.0"])
def test_init_uses_the_project_module_when_the_range_allows_an_older_release(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    requirement: str,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "range-agent"\ndependencies = ["{requirement}"]\n\n'
        '[tool.cayu]\nfactory = "app:build_app"\n'
    )

    status, output = _init(tmp_path, capsys)

    assert status == 0
    assert output["result"]["serve"]["auth"] == {"status": "added", "target": _GENERATED_TARGET}
    assert (tmp_path / "server_auth.py").read_text() == GENERATED_AUTH_MODULE.content


@pytest.mark.parametrize(
    "requirement", ["cayu[postgres,server]==0.8.2", "cayu[server]>0.8.1", "cayu[server]>=0.9"]
)
def test_init_uses_the_ready_made_target_when_the_requirement_guarantees_it(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    requirement: str,
) -> None:
    (tmp_path / "pyproject.toml").write_text(_released_scaffold_pyproject(requirement))

    status, output = _init(tmp_path, capsys)

    assert status == 0
    serve = output["result"]["serve"]
    assert serve["auth"] == {"status": "added", "target": _TARGET}
    assert "auth_module" not in serve
    assert not (tmp_path / "server_auth.py").exists()
    document = tomllib.loads((tmp_path / "pyproject.toml").read_text())
    assert document["tool"]["cayu"]["serve"] == {"auth": _TARGET}


def test_init_refuses_rather_than_replace_a_different_auth_module(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pyproject = _released_scaffold_pyproject("cayu[postgres,server]==0.8.1")
    (tmp_path / "pyproject.toml").write_text(pyproject)
    (tmp_path / "server_auth.py").write_text("AUTH = None\n")

    status, output = _init(tmp_path, capsys)

    assert status == 2
    error = output["error"]
    assert error["category"] == "serve_setup_required"
    assert "server_auth.py already exists" in error["message"]
    assert (
        'In [project].dependencies, change "cayu[postgres,server]==0.8.1" to '
        '"cayu[postgres,server]>0.8.1".'
    ) in error["message"]
    assert (
        'In [project.optional-dependencies].dev, change "cayu[postgres,server]==0.8.1" to '
        '"cayu[postgres,server]>0.8.1".'
    ) in error["message"]
    assert (tmp_path / "server_auth.py").read_text() == "AUTH = None\n"
    assert (tmp_path / "pyproject.toml").read_text() == pyproject
    assert not (tmp_path / "cayu-cloud.toml").exists()


def test_init_warns_when_the_ready_made_target_is_missing_from_the_declared_runtime(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "pinned-agent"\ndependencies = ["cayu[server]==0.8.1"]\n\n'
        f'[tool.cayu]\nfactory = "app:build_app"\n\n[tool.cayu.serve]\nauth = "{_TARGET}"\n'
    )

    status, output = _init(tmp_path, capsys)

    assert status == 0
    serve = output["result"]["serve"]
    assert serve["auth"] == {"status": "kept", "target": _TARGET}
    assert "`cayu serve` would not start" in serve["notes"][0]
    assert '"cayu[server]>0.8.1"' in serve["notes"][0]


def test_init_never_replaces_a_custom_auth_target(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """[project]
name = "oidc-agent"
dependencies = ["cayu[server]>=0.8"]

[tool.cayu]
factory = "app:build_app"

[tool.cayu.serve]
auth = "operator_auth:AUTH"
"""
    )
    original = (tmp_path / "pyproject.toml").read_text()

    status, output = _init(tmp_path, capsys)

    assert status == 0
    serve = output["result"]["serve"]
    assert serve["auth"] == {"status": "kept", "target": "operator_auth:AUTH"}
    assert (
        "Left the existing [tool.cayu.serve].auth target operator_auth:AUTH" in (serve["notes"][0])
    )
    assert (tmp_path / "pyproject.toml").read_text() == original


@pytest.mark.parametrize(
    ("pyproject", "expected_edit"),
    [
        (
            '[project]\nname = "dynamic-agent"\ndynamic = ["dependencies"]\n\n'
            '[tool.cayu]\nfactory = "app:build_app"\n',
            "Declare a static [project].dependencies list",
        ),
        (
            '[project]\nname = "no-cayu-agent"\ndependencies = ["httpx"]\n\n'
            '[tool.cayu]\nfactory = "app:build_app"\n',
            'Add "cayu[server]"',
        ),
        (
            # The same literal appears twice, so a text edit could change the wrong one.
            '[project]\nname = "twice-agent"\ndependencies = ["cayu==0.8.1"]\n'
            '# pinned: "cayu==0.8.1"\n\n[tool.cayu]\nfactory = "app:build_app"\n',
            'change "cayu==0.8.1" to "cayu[server]==0.8.1"',
        ),
    ],
)
def test_init_refuses_with_the_exact_edit_when_it_cannot_edit_safely(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    pyproject: str,
    expected_edit: str,
) -> None:
    (tmp_path / "pyproject.toml").write_text(pyproject)

    status, output = _init(tmp_path, capsys)

    assert status == 2
    error = output["error"]
    assert error["category"] == "serve_setup_required"
    assert expected_edit in error["message"]
    assert f'auth = "{_TARGET}"' in error["message"]
    assert "uv lock" in error["message"]
    assert (tmp_path / "pyproject.toml").read_text() == pyproject
    assert not (tmp_path / "cayu-cloud.toml").exists()


def test_serve_setup_inserts_auth_into_an_existing_serve_table() -> None:
    plan = plan_serve_setup(
        '[project]\nname = "a"\ndependencies = [\n  "cayu[server]>=0.9",\n]\n\n'
        '[tool.cayu]\nfactory = "app:build_app"\n\n'
        '[tool.cayu.serve]  # recovery settings\nstartup_recovery_statuses = ["pending"]\n'
        "recovery_inactive_after_seconds = 60\n"
    )

    assert plan.manual_edits == ()
    assert plan.updated_text is not None
    assert f'[tool.cayu.serve]  # recovery settings\nauth = "{_TARGET}"\n' in plan.updated_text


def test_serve_setup_refuses_a_serve_table_it_cannot_extend() -> None:
    plan = plan_serve_setup(
        '[project]\nname = "a"\ndependencies = ["cayu[server]"]\n\n'
        '[tool.cayu]\nfactory = "app:build_app"\n'
        'serve.startup_recovery_statuses = ["pending"]\n'
    )

    assert plan.updated_text is None
    assert plan.changes == ()
    assert len(plan.manual_edits) == 1
    assert "Set [tool.cayu.serve].auth" in plan.manual_edits[0]


@pytest.mark.parametrize(
    ("requirement", "replacement"),
    [
        ("cayu", "cayu[server]"),
        ("Cayu[postgres] >=0.8; python_version >= '3.11'", None),
        ("cayu [ postgres , aws ]==0.8.1", "cayu[postgres,aws,server]==0.8.1"),
    ],
)
def test_serve_setup_adds_the_server_extra_to_the_cayu_requirement(
    requirement: str,
    replacement: str | None,
) -> None:
    text = (
        f"[project]\nname = 'a'\ndependencies = [{json.dumps(requirement)}]\n\n"
        "[tool.cayu]\nfactory = 'app:build_app'\n\n[tool.cayu.serve]\nauth = 'x:AUTH'\n"
    )

    plan = plan_serve_setup(text)

    assert plan.manual_edits == ()
    assert plan.updated_text is not None
    dependencies = tomllib.loads(plan.updated_text)["project"]["dependencies"]
    expected = replacement or "Cayu[postgres,server] >=0.8; python_version >= '3.11'"
    assert dependencies == [expected]


@pytest.mark.parametrize(
    ("requirement", "ships"),
    [
        ("cayu[postgres,server]==0.8.1", False),
        ("cayu>=0.8", False),
        ("cayu>=0.8,<0.9", False),
        ("cayu[server]", False),
        ("cayu==0.8.*", False),
        ("cayu>0.8.1.post1", False),
        ("cayu>=0.8.1.post1", False),
        ("cayu>=1!0.1", False),
        ("cayu @ git+https://example.com/cayu.git", False),
        ("cayu[postgres,server]==0.8.2", True),
        ("cayu>0.8.1", True),
        ("cayu>=0.9", True),
        ("cayu~=0.9.0", True),
        ("cayu==0.9.*", True),
        ("cayu (>=0.9)", True),
        ("cayu>=0.9,<1; python_version >= '3.11'", True),
    ],
)
def test_requirement_ships_environment_auth_only_when_it_excludes_0_8_1(
    requirement: str, ships: bool
) -> None:
    assert requirement_ships_environment_auth(requirement) is ships


def test_declared_runtime_reads_uv_overrides_and_sources(tmp_path: Path) -> None:
    def document(extra: str, requirement: str = "cayu[server]>=0.9") -> dict:
        return tomllib.loads(f'[project]\nname = "a"\ndependencies = ["{requirement}"]\n{extra}')

    assert declared_runtime_ships_environment_auth(document(""))
    # An override replaces the declared requirement.
    overridden = document('[tool.uv]\noverride-dependencies = ["cayu==0.8.1"]\n')
    assert not declared_runtime_ships_environment_auth(overridden)
    assert environment_auth_upgrade_edits(overridden) == (
        'In [tool.uv].override-dependencies, change "cayu==0.8.1" to "cayu>0.8.1".',
    )
    # A local checkout counts only when it contains the module.
    sourced = document(
        '[tool.uv.sources]\ncayu = { path = "checkout" }\n', requirement="cayu==0.8.1"
    )
    assert not declared_runtime_ships_environment_auth(sourced, root=tmp_path)
    module = tmp_path / "checkout" / "src" / "cayu" / "server" / "environment_auth.py"
    module.parent.mkdir(parents=True)
    module.write_text("")
    assert declared_runtime_ships_environment_auth(sourced, root=tmp_path)
    assert not declared_runtime_ships_environment_auth(sourced)
    git = document('[tool.uv.sources]\ncayu = { git = "https://example.com/cayu.git" }\n')
    assert not declared_runtime_ships_environment_auth(git, root=tmp_path)


def test_upgrade_edits_list_development_pins_that_hold_the_lock_back() -> None:
    edits = environment_auth_upgrade_edits(
        tomllib.loads(
            '[project]\nname = "a"\ndependencies = ["cayu[server]>=0.8"]\n'
            '[project.optional-dependencies]\ndev = ["cayu[postgres,server]==0.8.1"]\n'
            '[dependency-groups]\ntest = ["cayu<0.8.2", "cayu<1"]\n'
            '[tool.uv]\ndev-dependencies = ["cayu~=0.7.0"]\n'
        )
    )

    assert edits == (
        'In [project].dependencies, change "cayu[server]>=0.8" to "cayu[server]>0.8.1".',
        'In [project.optional-dependencies].dev, change "cayu[postgres,server]==0.8.1" to '
        '"cayu[postgres,server]>0.8.1".',
        'In [dependency-groups].test, change "cayu<0.8.2" to "cayu>0.8.1".',
        'In [tool.uv].dev-dependencies, change "cayu~=0.7.0" to "cayu>0.8.1".',
    )
