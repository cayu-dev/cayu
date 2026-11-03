"""`cayu cloud init` makes the `cayu serve` web process it writes able to start."""

from __future__ import annotations

import gzip
import json
import tarfile
import tomllib
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest

from cayu.cli import main, scaffold
from cayu.cli._serve_readiness import (
    GENERATED_AUTH_MODULE,
    environment_auth_unavailable_reason,
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
_MODULE_FILE = "cayu/server/environment_auth.py"
_WHEEL = "vendor/cayu-0.16.3-py3-none-any.whl"


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
    monkeypatch: pytest.MonkeyPatch,
    preset: str,
) -> None:
    # A release install pins the release it scaffolds with.
    monkeypatch.setattr(scaffold, "_cayu_source_checkout", lambda: None)
    monkeypatch.setattr(scaffold, "_installed_cayu_version", lambda: "0.16.3")
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

    assert environment_auth_unavailable_reason(document("")) is None
    # An override replaces the declared requirement.
    overridden = document('[tool.uv]\noverride-dependencies = ["cayu==0.8.1"]\n')
    assert environment_auth_unavailable_reason(overridden) == (
        "the declared cayu requirement allows cayu 0.8.1 or older, which lack "
        "cayu.server.environment_auth"
    )
    assert environment_auth_upgrade_edits(overridden) == (
        'In [tool.uv].override-dependencies, change "cayu==0.8.1" to "cayu>0.8.1".',
    )
    # A local checkout counts only when it contains the module.
    sourced = document(
        '[tool.uv.sources]\ncayu = { path = "checkout" }\n', requirement="cayu==0.8.1"
    )
    assert environment_auth_unavailable_reason(sourced, root=tmp_path) == (
        'the cayu path source "checkout" in [tool.uv.sources] does not contain '
        "cayu.server.environment_auth"
    )
    module = tmp_path / "checkout" / "src" / "cayu" / "server" / "environment_auth.py"
    module.parent.mkdir(parents=True)
    module.write_text("")
    assert environment_auth_unavailable_reason(sourced, root=tmp_path) is None
    assert environment_auth_unavailable_reason(sourced) is not None
    git = document('[tool.uv.sources]\ncayu = { git = "https://example.com/cayu.git" }\n')
    assert "not locked in a readable uv.lock" in (
        environment_auth_unavailable_reason(git, root=tmp_path) or ""
    )
    assert environment_auth_upgrade_edits(git) == (
        "Remove the cayu entry from [tool.uv.sources] so a released cayu is installed.",
    )
    assert environment_auth_unavailable_reason(document("", requirement="httpx")) == (
        "[project].dependencies does not declare cayu"
    )


def _wheel(path: Path, *members: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as wheel:
        for member in members:
            wheel.writestr(member, "")


def _sdist(path: Path, *members: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as sdist:
        for member in members:
            sdist.addfile(tarfile.TarInfo(member))


def _sourced(path: str) -> dict:
    return tomllib.loads(
        '[project]\nname = "a"\ndependencies = ["cayu[postgres,server]>=0.16.1"]\n'
        f"[tool.uv.sources]\ncayu = {{ path = {json.dumps(path)} }}\n"
    )


def test_a_git_source_counts_when_uv_lock_locks_a_commit_that_ships_the_module(
    tmp_path: Path,
) -> None:
    pin = "c0aa269ee30816581e30a5254c15c327354193ed"
    url = "https://example.com/cayu.git"
    document = tomllib.loads(
        '[project]\nname = "a"\ndependencies = ["cayu[server]>=0.9"]\n'
        f'[tool.uv.sources]\ncayu = {{ git = "{url}", rev = "{pin}" }}\n'
    )

    def lock(version: str, fragment: str = pin) -> None:
        (tmp_path / "uv.lock").write_text(
            f'version = 1\n[[package]]\nname = "cayu"\nversion = "{version}"\n'
            f'source = {{ git = "{url}?rev={pin}#{fragment}" }}\n'
        )

    # Without the project, there is no lock to read.
    assert environment_auth_unavailable_reason(document) == (
        "the cayu Git source in [tool.uv.sources] was not checked for cayu.server.environment_auth"
    )
    assert "readable uv.lock" in (
        environment_auth_unavailable_reason(document, root=tmp_path) or ""
    )
    # Cayu Cloud installs the locked commit, whose own version says it has the module.
    lock("0.17.0")
    assert environment_auth_unavailable_reason(document, root=tmp_path) is None
    lock("0.8.1")
    assert environment_auth_unavailable_reason(document, root=tmp_path) == (
        "uv.lock locks cayu 0.8.1, and releases up to 0.8.1 lack cayu.server.environment_auth"
    )
    lock("0.17.0", fragment="main")
    assert "does not record the commit" in (
        environment_auth_unavailable_reason(document, root=tmp_path) or ""
    )


@pytest.mark.parametrize("scheme", ["file", "git+file"])
def test_git_readiness_rejects_local_repositories(tmp_path: Path, scheme: str) -> None:
    pin = "a" * 40
    url = f"{scheme}:///home/example/cayu"
    document = tomllib.loads(
        '[project]\nname = "a"\ndependencies = ["cayu[server]"]\n'
        f'[tool.uv.sources]\ncayu = {{ git = "{url}", rev = "{pin}" }}\n'
    )
    (tmp_path / "uv.lock").write_text(
        'version = 1\n[[package]]\nname = "cayu"\nversion = "0.17.0"\n'
        f'source = {{ git = "{url}?rev={pin}#{pin}" }}\n'
    )

    assert "local Git repository that Cayu Cloud cannot fetch" in (
        environment_auth_unavailable_reason(document, root=tmp_path) or ""
    )


@pytest.mark.parametrize(
    ("other_source", "other_version", "ready"),
    [
        ('{ registry = "https://pypi.org/simple" }', "0.8.1", False),
        ('{ registry = "https://pypi.org/simple" }', "0.17.0", True),
        ('{ git = "https://example.com/cayu.git#' + "b" * 40 + '" }', "0.8.1", False),
        ('{ git = "https://example.com/cayu.git#main" }', "0.17.0", False),
        ('{ directory = "vendor/cayu" }', "0.17.0", False),
        ("{}", "0.17.0", False),
    ],
)
def test_git_readiness_checks_every_marker_qualified_lock_entry(
    tmp_path: Path, other_source: str, other_version: str, ready: bool
) -> None:
    pin = "a" * 40
    document = tomllib.loads(
        '[project]\nname = "a"\ndependencies = ["cayu[server]"]\n'
        '[tool.uv.sources]\ncayu = { git = "https://example.com/cayu.git", '
        f'rev = "{pin}", marker = "sys_platform == \'darwin\'" }}\n'
    )
    (tmp_path / "uv.lock").write_text(
        'version = 1\n[[package]]\nname = "cayu"\nversion = "0.17.0"\n'
        f'source = {{ git = "https://example.com/cayu.git#{pin}" }}\n'
        "resolution-markers = [\"sys_platform == 'darwin'\"]\n"
        f'[[package]]\nname = "cayu"\nversion = "{other_version}"\n'
        f"source = {other_source}\n"
        "resolution-markers = [\"sys_platform != 'darwin'\"]\n"
    )

    reason = environment_auth_unavailable_reason(document, root=tmp_path)
    assert (reason is None) is ready


@pytest.mark.parametrize(
    ("path", "build", "members", "ships"),
    [
        (_WHEEL, _wheel, (_MODULE_FILE,), True),
        (_WHEEL, _wheel, ("cayu/server/app.py",), False),
        # A directory entry with the module's name is not the module.
        (_WHEEL, _wheel, (f"{_MODULE_FILE}/",), False),
        ("vendor/cayu-0.16.3.tar.gz", _sdist, (f"cayu-0.16.3/src/{_MODULE_FILE}",), True),
        ("vendor/cayu-0.16.3.tar.gz", _sdist, (f"cayu-0.16.3/{_MODULE_FILE}",), True),
        # In an sdist the source tree sits under one top directory.
        ("vendor/cayu-0.16.3.tar.gz", _sdist, (_MODULE_FILE,), False),
        ("vendor/cayu-0.16.3.tar.gz", _sdist, ("cayu-0.16.3/src/cayu/server/app.py",), False),
    ],
)
def test_a_vendored_cayu_archive_counts_only_when_it_contains_the_module(
    tmp_path: Path,
    path: str,
    build: Callable[..., None],
    members: tuple[str, ...],
    ships: bool,
) -> None:
    build(tmp_path / path, *members)

    reason = environment_auth_unavailable_reason(_sourced(path), root=tmp_path)

    if ships:
        assert reason is None
    else:
        assert reason == (
            f'the cayu path source "{path}" in [tool.uv.sources] does not contain '
            "cayu.server.environment_auth"
        )


def _sdist_with_a_corrupt_body() -> bytes:
    # A valid first header, then a member body whose compressed data is damaged, so
    # zlib fails while tarfile seeks past the body to the next header.
    header = tarfile.TarInfo("cayu-0.16.3/README")
    header.size = 32768
    return (
        gzip.compress(header.tobuf() + b"x" * 16384)
        + bytes.fromhex("1f8b08000000000002ff07")
        + b"\0" * 16
    )


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("cayu-0.16.3-py3-none-any.whl", b"not an archive"),
        ("cayu-0.16.3.tar.gz", b"not an archive"),
        ("cayu-0.16.3.tar.gz", _sdist_with_a_corrupt_body()),
    ],
    ids=["wheel", "sdist", "sdist-with-a-corrupt-body"],
)
def test_a_damaged_vendored_archive_does_not_count(
    tmp_path: Path, name: str, content: bytes
) -> None:
    (tmp_path / name).write_bytes(content)

    reason = environment_auth_unavailable_reason(_sourced(name), root=tmp_path)

    assert reason is not None
    assert "does not contain cayu.server.environment_auth" in reason


def test_a_path_source_symlink_loop_does_not_count(tmp_path: Path) -> None:
    (tmp_path / "loop").symlink_to("loop")
    (tmp_path / "tree" / "src" / "cayu" / "server").mkdir(parents=True)
    (tmp_path / "tree" / "src" / _MODULE_FILE).symlink_to("environment_auth.py")

    for path in ("loop", "loop/cayu.whl"):
        assert environment_auth_unavailable_reason(_sourced(path), root=tmp_path) == (
            f"the cayu path source {json.dumps(path)} in [tool.uv.sources] goes through a "
            "symlink loop"
        )
    assert environment_auth_unavailable_reason(_sourced("tree"), root=tmp_path) == (
        'the cayu path source "tree" in [tool.uv.sources] does not contain '
        "cayu.server.environment_auth"
    )


def test_a_path_source_counts_only_inside_the_project(tmp_path: Path) -> None:
    project = tmp_path / "project"
    outside = tmp_path / "checkout"
    (outside / "src" / "cayu" / "server").mkdir(parents=True)
    (outside / "src" / _MODULE_FILE).write_text("")
    _wheel(tmp_path / "cayu-0.16.3-py3-none-any.whl", _MODULE_FILE)
    _wheel(project / _WHEEL, _MODULE_FILE)

    def reason(path: str) -> str | None:
        return environment_auth_unavailable_reason(_sourced(path), root=project)

    def outside_reason(path: str) -> str:
        return (
            f"the cayu path source {json.dumps(path)} in [tool.uv.sources] leads outside the "
            "project directory, and Cayu Cloud receives only the project's files"
        )

    assert reason(_WHEEL) is None
    # Cayu Cloud's copy of the project has another directory name and location.
    for path in ("../checkout", "../cayu-0.16.3-py3-none-any.whl", f"../project/{_WHEEL}"):
        assert reason(path) == outside_reason(path)
    for path in (str(outside), str(project / _WHEEL)):
        assert reason(path) == (
            f"the cayu path source {json.dumps(path)} in [tool.uv.sources] is an absolute "
            "path, which exists only on this machine"
        )
    # A symlink that leads out of the project, or is absolute, does not count either.
    (project / "linked").symlink_to("../checkout", target_is_directory=True)
    (project / "linked.whl").symlink_to("../cayu-0.16.3-py3-none-any.whl")
    (project / "absolute.whl").symlink_to(project / _WHEEL)
    for path in ("linked", "linked.whl", "absolute.whl"):
        assert reason(path) == outside_reason(path)
    # Nor does a checkout inside the project whose module file links out of it.
    module = project / "tree" / "src" / _MODULE_FILE
    module.parent.mkdir(parents=True)
    module.symlink_to("../../../../../checkout/src/cayu/server/environment_auth.py")
    assert reason("tree") == (
        'the cayu path source "tree" in [tool.uv.sources] does not contain '
        "cayu.server.environment_auth"
    )
    # Relative symlinks that stay inside the project are followed.
    module.unlink()
    (project / "tree" / "src" / "cayu" / "server" / "real.py").write_text("")
    module.symlink_to("real.py")
    (project / "vendor" / "tree").symlink_to("../tree", target_is_directory=True)
    (project / "vendor" / "cayu.whl").symlink_to(Path(_WHEEL).name)
    for path in ("tree", "vendor/tree", "vendor/cayu.whl", "vendor/../vendor/cayu.whl"):
        assert reason(path) is None


def test_every_directory_on_the_way_must_exist(tmp_path: Path) -> None:
    # `missing/..` cancels out on paper, but the lookup fails when missing/ is absent.
    _wheel(tmp_path / "wheels" / "cayu-0.16.3-py3-none-any.whl", _MODULE_FILE)
    (tmp_path / "vendor").mkdir()
    (tmp_path / _WHEEL).symlink_to("../missing/../wheels/cayu-0.16.3-py3-none-any.whl")
    (tmp_path / "file").write_text("")

    assert environment_auth_unavailable_reason(_sourced(_WHEEL), root=tmp_path) == (
        f'the cayu path source "{_WHEEL}" in [tool.uv.sources] goes through missing, '
        "which is not a directory"
    )
    assert environment_auth_unavailable_reason(_sourced(f"file/../{_WHEEL}"), root=tmp_path) == (
        f'the cayu path source "file/../{_WHEEL}" in [tool.uv.sources] goes through file, '
        "which is not a directory"
    )
    (tmp_path / "missing").mkdir()
    assert environment_auth_unavailable_reason(_sourced(_WHEEL), root=tmp_path) is None


def test_the_declared_file_name_says_which_kind_of_archive_it_is(tmp_path: Path) -> None:
    # uv reads `vendor/cayu-...whl` as a wheel even when it links to a file named otherwise.
    _wheel(tmp_path / "store" / "blob", _MODULE_FILE)
    (tmp_path / "vendor").mkdir()
    (tmp_path / _WHEEL).symlink_to("../store/blob")
    (tmp_path / "vendor" / "blob").symlink_to(Path(_WHEEL).name)

    assert environment_auth_unavailable_reason(_sourced(_WHEEL), root=tmp_path) is None
    assert environment_auth_unavailable_reason(_sourced("vendor/blob"), root=tmp_path) == (
        'the cayu path source "vendor/blob" in [tool.uv.sources] does not contain '
        "cayu.server.environment_auth"
    )


def test_upgrade_edits_point_a_path_source_at_a_cayu_that_ships_the_module() -> None:
    assert environment_auth_upgrade_edits(_sourced(_WHEEL)) == (
        "Point the cayu path source in [tool.uv.sources] at a cayu wheel, source "
        "distribution, or source tree inside the project that contains "
        "cayu.server.environment_auth (any release newer than 0.8.1), or remove the cayu "
        "entry so a released cayu is installed.",
    )


@pytest.mark.parametrize(
    ("path", "member", "reason"),
    [
        (_WHEEL, _MODULE_FILE, None),
        (
            _WHEEL,
            "cayu/server/app.py",
            f'the cayu path source "{_WHEEL}" in [tool.uv.sources] does not contain '
            "cayu.server.environment_auth",
        ),
        # A deploy never uploads dist/, so Cloud would not have the wheel.
        (
            "dist/cayu-0.16.3-py3-none-any.whl",
            _MODULE_FILE,
            "Cayu Cloud would not receive dist/cayu-0.16.3-py3-none-any.whl",
        ),
    ],
)
def test_init_uses_a_vendored_wheel_that_ships_the_module(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    path: str,
    member: str,
    reason: str | None,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "vendored-agent"\ndependencies = ["cayu[postgres,server]>=0.16.1"]\n\n'
        f'[tool.uv.sources]\ncayu = {{ path = "{path}" }}\n\n'
        '[tool.cayu]\nfactory = "app:build_app"\n'
    )
    _wheel(tmp_path / path, member)

    status, output = _init(tmp_path, capsys)

    assert status == 0
    serve = output["result"]["serve"]
    if reason is None:
        assert serve["auth"] == {"status": "added", "target": _TARGET}
        assert not (tmp_path / "server_auth.py").exists()
    else:
        assert serve["auth"] == {"status": "added", "target": _GENERATED_TARGET}
        assert reason in serve["notes"][0]
        assert "0.8.1 or older" not in serve["notes"][0]


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
