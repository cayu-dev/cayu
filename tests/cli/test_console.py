from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import cayu
from cayu import CayuApp
from cayu.cli import _version, main


def test_console_discovers_project_and_opens_booted_namespace(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    project = tmp_path / "project"
    nested = project / "agents" / "reviewer"
    nested.mkdir(parents=True)
    (project / "pyproject.toml").write_text(
        '[tool.cayu]\nfactory = "console_project_app:build_app"\n',
        encoding="utf-8",
    )
    (project / "console_project_app.py").write_text(
        """from pathlib import Path

from cayu import AgentSpec, CayuApp, Environment, EnvironmentSpec

build_count = 0


def build_app():
    global build_count
    build_count += 1
    app = CayuApp(enable_logging=False)
    app.build_count = build_count
    app.factory_cwd = Path.cwd()
    app.secret_token = "do-not-print-this"
    app.register_agent(AgentSpec(name="reviewer", model="fake-model"))
    app.register_environment(Environment(EnvironmentSpec(name="local")))
    return app
""",
        encoding="utf-8",
    )

    launch: dict[str, Any] = {}
    fake_ipython = ModuleType("IPython")

    def start_ipython(*, argv: list[str], user_ns: dict[str, Any]) -> None:
        launch["argv"] = argv
        launch["user_ns"] = user_ns
        launch["cwd"] = Path.cwd()

    fake_ipython.start_ipython = start_ipython  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "IPython", fake_ipython)
    monkeypatch.chdir(nested)
    sys.modules.pop("console_project_app", None)

    assert main(["console"]) == 0

    namespace = launch["user_ns"]
    assert set(namespace) == {"app", "cayu", "knowledge", "sessions", "tasks"}
    assert isinstance(namespace["app"], CayuApp)
    assert namespace["sessions"] is namespace["app"].session_store
    assert namespace["tasks"] is None
    assert namespace["knowledge"] is None
    assert namespace["app"].factory_cwd == project
    assert launch["cwd"] == project
    assert launch["argv"] == []
    assert namespace["app"].build_count == 1
    assert "console_project_app" not in sys.modules

    output = capsys.readouterr().out
    assert f"Cayu {_version()} console" in output
    assert f"Project: {project}" in output
    assert "Factory: console_project_app:build_app" in output
    assert "Agents: reviewer" in output
    assert "Providers: none" in output
    assert "Environments: local" in output
    assert "Session store: InMemorySessionStore" in output
    assert "live, writable application console" in output
    assert "do-not-print-this" not in output


def test_console_explicit_target_uses_current_directory_and_restores_process_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "explicit_console_app.py").write_text(
        """from cayu import CayuApp


class Factories:
    @staticmethod
    def build_app():
        return CayuApp(enable_logging=False)


factories = Factories()
""",
        encoding="utf-8",
    )
    launch: dict[str, Any] = {}
    fake_ipython = ModuleType("IPython")

    def start_ipython(*, argv: list[str], user_ns: dict[str, Any]) -> None:
        launch["cwd"] = Path.cwd()
        launch["path"] = sys.path[0]
        launch["app"] = user_ns["app"]

    fake_ipython.start_ipython = start_ipython  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "IPython", fake_ipython)
    monkeypatch.chdir(tmp_path)
    original_path = list(sys.path)
    sys.modules.pop("explicit_console_app", None)

    assert main(["console", "explicit_console_app:factories.build_app"]) == 0

    assert launch["cwd"] == tmp_path
    assert launch["path"] == str(tmp_path)
    assert isinstance(launch["app"], CayuApp)
    assert Path.cwd() == tmp_path
    assert sys.path == original_path


def test_console_reports_when_no_project_can_be_discovered(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)

    assert main(["console"]) == 1

    error = capsys.readouterr().err
    assert "No Cayu project found" in error
    assert '[tool.cayu] factory = "module:build_app"' in error
    assert "cayu console module:build_app" in error


@pytest.mark.parametrize(
    ("pyproject", "message"),
    [
        ('[tool.cayu]\nfactory = ""\n', "must be a non-empty string"),
        ("[tool.cayu\n", "Could not read"),
    ],
)
def test_console_reports_invalid_project_configuration(
    pyproject: str,
    message: str,
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    (tmp_path / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert main(["console"]) == 1

    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("invalid_console_app:app", "must be a callable, not an application object"),
        ("invalid_console_app:required", "must not require arguments"),
        ("invalid_console_app:async_factory", "must be synchronous"),
        ("invalid_console_app:awaitable_result", "returned an awaitable"),
        ("invalid_console_app:wrong_type", "must return a CayuApp"),
    ],
)
def test_console_rejects_invalid_factory_contracts(
    target: str,
    message: str,
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    (tmp_path / "invalid_console_app.py").write_text(
        """from cayu import CayuApp

app = CayuApp(enable_logging=False)


def required(value):
    return CayuApp(enable_logging=False)


async def async_factory():
    return CayuApp(enable_logging=False)


def awaitable_result():
    return async_factory()


def wrong_type():
    return object()
""",
        encoding="utf-8",
    )
    fake_ipython = ModuleType("IPython")
    fake_ipython.start_ipython = lambda **_kwargs: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "IPython", fake_ipython)
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("invalid_console_app", None)

    assert main(["console", target]) == 1

    assert message in capsys.readouterr().err


def test_console_missing_ipython_has_actionable_optional_dependency_error(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    (tmp_path / "missing_ipython_app.py").write_text(
        "from cayu import CayuApp\n\ndef build_app():\n    return CayuApp(enable_logging=False)\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "IPython", None)
    sys.modules.pop("missing_ipython_app", None)

    assert main(["console", "missing_ipython_app:build_app"]) == 1

    assert (
        'Cayu console requires IPython. Install it with: pip install "cayu[console]"'
        in capsys.readouterr().err
    )
    assert "missing_ipython_app" not in sys.modules


@pytest.mark.parametrize(
    ("raised", "expected_status"),
    [(EOFError(), 0), (KeyboardInterrupt(), 130)],
)
def test_console_converts_shell_exit_paths_to_status_codes(
    raised: BaseException,
    expected_status: int,
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "exit_console_app.py").write_text(
        "from cayu import CayuApp\n\ndef build_app():\n    return CayuApp(enable_logging=False)\n",
        encoding="utf-8",
    )
    fake_ipython = ModuleType("IPython")

    def start_ipython(*, argv: list[str], user_ns: dict[str, Any]) -> None:
        raise raised

    fake_ipython.start_ipython = start_ipython  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "IPython", fake_ipython)
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("exit_console_app", None)

    assert main(["console", "exit_console_app:build_app"]) == expected_status


@pytest.mark.parametrize(
    ("module_source", "message"),
    [
        ('raise RuntimeError("import exploded")\n', "import exploded"),
        (
            'def build_app():\n    raise RuntimeError("factory exploded")\n',
            "factory exploded",
        ),
    ],
)
def test_console_does_not_swallow_application_boot_errors(
    module_source: str,
    message: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "broken_console_app.py").write_text(module_source, encoding="utf-8")
    fake_ipython = ModuleType("IPython")
    fake_ipython.start_ipython = lambda **_kwargs: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "IPython", fake_ipython)
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("broken_console_app", None)

    with pytest.raises(RuntimeError, match=message):
        main(["console", "broken_console_app:build_app"])


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("not-a-target", "must use module:attribute syntax"),
        ("missing_console_module:build_app", "module was not found"),
    ],
)
def test_console_normalizes_target_resolution_errors(
    target: str,
    message: str,
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    fake_ipython = ModuleType("IPython")
    fake_ipython.start_ipython = lambda **_kwargs: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "IPython", fake_ipython)
    monkeypatch.chdir(tmp_path)

    assert main(["console", target]) == 1

    assert message in capsys.readouterr().err


def test_other_cli_commands_work_without_ipython(monkeypatch, capsys) -> None:
    monkeypatch.setitem(sys.modules, "IPython", None)

    assert main(["version"]) == 0

    assert capsys.readouterr().out == f"cayu {_version()}\n"


def test_package_version_attribute_and_root_version_flag(capsys) -> None:
    assert cayu.__version__ == _version()

    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])

    assert excinfo.value.code == 0
    assert capsys.readouterr().out == f"cayu {_version()}\n"
