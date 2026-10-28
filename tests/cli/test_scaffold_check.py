from __future__ import annotations

import ast
import importlib
import json
import sys
from pathlib import Path

import pytest

from cayu.cli import main
from cayu.cli.project import project_context
from cayu.cli.scaffold_check import (
    application_not_imported_diagnostic,
    check_declared_scaffold,
    check_declared_scaffold_source,
)
from cayu.runtime.checks import DiagnosticSeverity, ProjectDiagnostic
from cayu.runtime.manifest import RegistrationProvenance


def _generated_manifest(project: Path):
    with project_context(project):
        app = importlib.import_module("app").build_app()
        return app.describe(project_root=project)


def test_complete_generated_project_has_no_scaffold_findings(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"

    assert check_declared_scaffold(project, _generated_manifest(project)) == ()


@pytest.mark.parametrize(
    "exclusions",
    (
        ("evals",),
        ("memory", "knowledge"),
    ),
)
def test_supported_capability_opt_outs_pass_strict_scaffold_check(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    exclusions: tuple[str, ...],
) -> None:
    command = ["new", "reduced", "--dir", str(tmp_path)]
    for capability in exclusions:
        command.extend(("--without", capability))
    assert main(command) == 0
    capsys.readouterr()
    monkeypatch.chdir(tmp_path / "reduced")

    assert main(["check", "--fail-on", "warning", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["diagnostics"] == []


def test_declared_layout_reports_missing_home_and_composition_collapse(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    manifest = _generated_manifest(project)
    (project / "memory/context.py").unlink()
    app_path = project / "app.py"
    app_path.write_text(
        app_path.read_text(encoding="utf-8") + "\nclass ProductClient:\n    pass\n",
        encoding="utf-8",
    )

    findings = check_declared_scaffold(project, manifest)
    by_code = {item.code: item for item in findings}

    assert by_code["SCAFFOLD_LAYOUT_PATH_MISSING"].path == "memory/context.py"
    assert by_code["SCAFFOLD_APP_COMPOSITION_DRIFT"].path == "app.py"
    assert "explicit custom-layout migration" in by_code["SCAFFOLD_LAYOUT_PATH_MISSING"].hint


def test_async_implementation_in_the_composition_root_is_reported(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    app_path = project / "app.py"
    app_path.write_text(
        app_path.read_text(encoding="utf-8") + "\nasync def hidden_worker():\n    pass\n",
        encoding="utf-8",
    )

    finding = next(
        item
        for item in check_declared_scaffold_source(project)
        if item.code == "SCAFFOLD_APP_COMPOSITION_DRIFT"
    )

    assert finding.parameters["unexpected_functions"] == ("hidden_worker",)


def test_registration_provenance_drift_is_actionable(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    manifest = _generated_manifest(project)
    agent = manifest.agents[0].model_copy(
        update={
            "registration_provenance": RegistrationProvenance(
                kind="project",
                symbol="build_app",
                location="app.py",
            )
        }
    )
    drifted = manifest.model_copy(update={"agents": (agent,)})

    findings = check_declared_scaffold(project, drifted)
    finding = next(
        item for item in findings if item.code == "SCAFFOLD_REGISTRATION_PROVENANCE_DRIFT"
    )
    assert finding.parameters["observed_location"] == "app.py"
    assert "agents/registration.py" in finding.hint


def test_freeform_project_is_not_subject_to_generated_layout_checks(tmp_path: Path) -> None:
    project = tmp_path / "custom"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[tool.cayu]\nfactory = "app:build_app"\n',
        encoding="utf-8",
    )
    generated = tmp_path / "generated"
    assert main(["new", "generated", "--dir", str(tmp_path)]) == 0
    manifest = _generated_manifest(generated)

    assert check_declared_scaffold(project, manifest) == ()


def test_cli_check_emits_stable_json_for_declared_drift(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    (project / "knowledge/retrieval.py").unlink()
    monkeypatch.chdir(project)

    assert main(["check", "--fail-on", "warning", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    finding = next(
        item for item in report["diagnostics"] if item["code"] == "SCAFFOLD_LAYOUT_PATH_MISSING"
    )
    assert finding["path"] == "knowledge/retrieval.py"
    assert finding["verification_command"] == "cayu check --fail-on warning --json"


def test_cli_check_applies_tag_selection_before_source_only_gating(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    (project / "operations/watchers.py").unlink()
    monkeypatch.chdir(project)

    assert main(["check", "--tag", "security", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)

    assert report["manifest_fingerprint"] != "unavailable"
    assert all(item["code"] != "SCAFFOLD_LAYOUT_PATH_MISSING" for item in report["diagnostics"])


def _blocking_finding(path: str) -> ProjectDiagnostic:
    return ProjectDiagnostic(
        code="SCAFFOLD_IMPORT_SIDE_EFFECT",
        severity=DiagnosticSeverity.ERROR,
        subject="scaffold",
        path=path,
        message="unproven",
        verification_command="cayu check",
    )


@pytest.mark.parametrize(("blocking", "phrase"), [(1, "1 finding leaves"), (2, "2 findings leave")])
def test_not_imported_diagnostic_counts_its_blocking_findings(blocking: int, phrase: str) -> None:
    findings = tuple(_blocking_finding(f"app.py:{line}") for line in range(1, blocking + 1))

    assert phrase in application_not_imported_diagnostic(findings).message


def test_not_imported_diagnostic_names_the_findings_that_stopped_the_import() -> None:
    findings = tuple(_blocking_finding(f"app.py:{line}") for line in range(1, 6))

    diagnostic = application_not_imported_diagnostic(findings[:1])
    assert "(SCAFFOLD_IMPORT_SIDE_EFFECT at app.py:1)" in diagnostic.message
    assert "did not run" in diagnostic.message

    diagnostic = application_not_imported_diagnostic(findings)
    assert (
        "(SCAFFOLD_IMPORT_SIDE_EFFECT at app.py:1; SCAFFOLD_IMPORT_SIDE_EFFECT at app.py:2; "
        "SCAFFOLD_IMPORT_SIDE_EFFECT at app.py:3; and 2 more)"
    ) in diagnostic.message
    assert list(diagnostic.parameters["blocking_locations"]) == [
        f"SCAFFOLD_IMPORT_SIDE_EFFECT at app.py:{line}" for line in range(1, 6)
    ]


_PYTEST_FIXTURE = """import pytest


@pytest.fixture(autouse=True)
def _private_state_root(tmp_path, monkeypatch):
    monkeypatch.setenv("PRIVATE_STATE_ROOT", str(tmp_path))
"""


def _payload(marker: Path) -> str:
    return f"from pathlib import Path\n\nPath({str(marker)!r}).write_text('ran')\n"


def test_cli_check_does_not_treat_the_root_conftest_as_an_application_import(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An autouse fixture in the root conftest.py must not skip every application
    # check. pytest imports conftest modules; the app does not.
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    (project / "conftest.py").write_text(_PYTEST_FIXTURE, encoding="utf-8")
    monkeypatch.chdir(project)

    assert main(["check", "--fail-on", "warning", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)

    assert report["manifest_fingerprint"] != "unavailable"
    assert report["diagnostics"] == []


@pytest.mark.parametrize(
    ("module", "statement"),
    (
        ("conftest.py", "import conftest"),
        ("conftest.py", "from conftest import _private_state_root"),
        ("conftest.py", "importlib.import_module('conftest')"),
        # Only the root conftest.py is left out; other modules are application code.
        ("tools/test_support.py", "from tools import test_support"),
    ),
)
def test_cli_check_scans_a_module_the_factory_imports_inside_a_function(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    module: str,
    statement: str,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    marker = tmp_path / "lazy-module-ran"
    (project / module).write_text(_payload(marker), encoding="utf-8")
    with (project / "app.py").open("a", encoding="utf-8") as source:
        source.write(
            "\n\ndef build_lazy_app():\n"
            "    import importlib  # noqa: F401\n\n"
            f"    {statement}  # noqa: F401\n"
            "    return build_app()\n"
        )
    monkeypatch.chdir(project)

    assert main(["check", "--json", "app:build_lazy_app"]) == 1
    report = json.loads(capsys.readouterr().out)

    assert not marker.exists()
    assert report["manifest_fingerprint"] == "unavailable"
    assert ("SCAFFOLD_IMPORT_SIDE_EFFECT", f"{module}:3") in [
        (item["code"], item["path"]) for item in report["diagnostics"]
    ]


@pytest.mark.parametrize("helper_import", ("module", "function"))
def test_cli_check_scans_a_conftest_a_local_helper_imports_inside_a_function(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    helper_import: str,
) -> None:
    # The helper lives outside the application packages, so only following
    # every local module reaches the conftest it imports.
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    marker = tmp_path / "helper-conftest-ran"
    (project / "conftest.py").write_text(_payload(marker), encoding="utf-8")
    (project / "lib").mkdir()
    (project / "lib" / "helpers.py").write_text(
        "def build():\n    import conftest  # noqa: F401\n", encoding="utf-8"
    )
    lazy = "    from lib import helpers\n\n" if helper_import == "function" else ""
    with (project / "app.py").open("a", encoding="utf-8") as source:
        if helper_import == "module":
            source.write("\nfrom lib import helpers  # noqa: E402\n")
        source.write(
            f"\n\ndef build_lazy_app():\n{lazy}    helpers.build()\n    return build_app()\n"
        )
    monkeypatch.chdir(project)

    assert main(["check", "--json", "app:build_lazy_app"]) == 1
    report = json.loads(capsys.readouterr().out)

    assert not marker.exists()
    assert ("SCAFFOLD_IMPORT_SIDE_EFFECT", "conftest.py:3") in [
        (item["code"], item["path"]) for item in report["diagnostics"]
    ]


@pytest.mark.parametrize(
    "statement",
    ("import lib.nested.helpers", "from lib.nested.helpers import build", "import lib.nested"),
)
@pytest.mark.parametrize("initializer", ("lib/__init__.py", "lib/nested/__init__.py"))
def test_cli_check_scans_conftest_imported_by_parent_package(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    statement: str,
    initializer: str,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    marker = tmp_path / "package-initializer-conftest-ran"
    (project / "conftest.py").write_text(_payload(marker), encoding="utf-8")
    (project / "lib" / "nested").mkdir(parents=True)
    (project / initializer).write_text("import conftest\n", encoding="utf-8")
    (project / "lib" / "nested" / "helpers.py").write_text(
        "def build():\n    pass\n", encoding="utf-8"
    )
    with (project / "app.py").open("a", encoding="utf-8") as source:
        source.write(f"\n\ndef build_lazy_app():\n    {statement}\n    return build_app()\n")
    monkeypatch.chdir(project)

    assert main(["check", "--json", "app:build_lazy_app"]) == 1
    report = json.loads(capsys.readouterr().out)

    assert not marker.exists()
    assert ("SCAFFOLD_IMPORT_SIDE_EFFECT", "conftest.py:3") in [
        (item["code"], item["path"]) for item in report["diagnostics"]
    ]


@pytest.mark.parametrize("module, relative", (("app.py", "."), ("tools/helper.py", "..")))
def test_cli_check_keeps_parent_initializer_walk_inside_project(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    module: str,
    relative: str,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    (tmp_path / "__init__.py").write_text("from .sibling import value\n", encoding="utf-8")
    (project / "conftest.py").write_text(_PYTEST_FIXTURE, encoding="utf-8")
    with (project / module).open("a", encoding="utf-8") as source:
        source.write(f"\n\ndef unused():\n    from {relative} import thing\n")
    monkeypatch.chdir(project)

    assert main(["check", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["manifest_fingerprint"] != "unavailable"


@pytest.mark.parametrize("explicit", (False, True))
def test_cli_check_scans_a_factory_module_named_conftest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    explicit: bool,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    marker = tmp_path / "factory-module-ran"
    (project / "conftest.py").write_text(
        _payload(marker) + "from app import build_app  # noqa: E402,F401\n",
        encoding="utf-8",
    )
    command = ["check", "--json"]
    if explicit:
        command.append("conftest:build_app")
    else:
        pyproject = project / "pyproject.toml"
        pyproject.write_text(
            pyproject.read_text(encoding="utf-8").replace(
                'factory = "app:build_app"', 'factory = "conftest:build_app"'
            ),
            encoding="utf-8",
        )
    monkeypatch.chdir(project)

    assert main(command) == 1
    report = json.loads(capsys.readouterr().out)

    assert not marker.exists()
    assert report["manifest_fingerprint"] == "unavailable"
    assert [(item["code"], item["path"]) for item in report["diagnostics"]] == [
        ("SCAFFOLD_IMPORT_SIDE_EFFECT", "conftest.py:3"),
        ("SCAFFOLD_APPLICATION_NOT_IMPORTED", "app.py"),
    ]


def test_cli_check_keeps_source_findings_from_an_explicit_factory_module(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The checks that run after the import cover the same modules as the scan
    # before it, so a warning in the loaded module still counts.
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    (project / "conftest.py").write_text(
        "from app import build_app  # noqa: F401\n"
        "from cayu.tasks import TaskCreate, TaskQuery, run_task_worker\n\n"
        "TIMEOUT = 30\n\n\n"
        "async def worker(app, handler):\n"
        "    await run_task_worker(app, app.task_store, handler, worker_id='w',\n"
        "                          query=TaskQuery(type='job'), execution_timeout_s=TIMEOUT)\n\n\n"
        "async def submit(app):\n"
        "    await app.create_task(TaskCreate(type='job'))\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    assert main(["check", "conftest:build_app", "--fail-on", "warning", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)

    assert report["manifest_fingerprint"] != "unavailable"
    assert [(item["code"], item["path"]) for item in report["diagnostics"]] == [
        ("TASK_EVENT_RETRY_POLICY_UNDECLARED", "conftest.py:13"),
    ]


def test_cli_check_still_scans_a_root_conftest_the_application_imports(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    (project / "conftest.py").write_text(_PYTEST_FIXTURE, encoding="utf-8")
    with (project / "tools" / "registration.py").open("a", encoding="utf-8") as source:
        source.write("\nimport conftest  # noqa: E402,F401\n")
    monkeypatch.chdir(project)

    assert main(["check", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)

    assert report["manifest_fingerprint"] == "unavailable"
    assert [(item["code"], item["path"]) for item in report["diagnostics"]] == [
        ("SCAFFOLD_IMPORT_SIDE_EFFECT", "conftest.py:4"),
        ("SCAFFOLD_APPLICATION_NOT_IMPORTED", "app.py"),
    ]
    assert "SCAFFOLD_IMPORT_SIDE_EFFECT at conftest.py:4" in report["diagnostics"][1]["message"]


def test_cli_check_scans_a_conftest_inside_an_application_package(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Package auto-discovery imports every module without naming any of them.
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    marker = tmp_path / "package-conftest-ran"
    (project / "tools" / "conftest.py").write_text(_payload(marker), encoding="utf-8")
    with (project / "app.py").open("a", encoding="utf-8") as source:
        source.write(
            "\n\ndef build_discovering_app():\n"
            "    import importlib\n"
            "    import pkgutil\n\n"
            "    import tools\n\n"
            "    for info in pkgutil.iter_modules(tools.__path__):\n"
            "        importlib.import_module(f'tools.{info.name}')\n"
            "    return build_app()\n"
        )
    monkeypatch.chdir(project)

    assert main(["check", "--json", "app:build_discovering_app"]) == 1
    report = json.loads(capsys.readouterr().out)

    assert not marker.exists()
    assert [(item["code"], item["path"]) for item in report["diagnostics"]] == [
        ("SCAFFOLD_IMPORT_SIDE_EFFECT", "tools/conftest.py:3"),
        ("SCAFFOLD_APPLICATION_NOT_IMPORTED", "app.py"),
    ]


def test_cli_check_tag_selection_never_imports_an_unproven_application(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    pyproject = project / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace("convention = 1", "convention = 2"),
        encoding="utf-8",
    )
    marker = tmp_path / "imported"
    tools_init = project / "tools/__init__.py"
    tools_init.write_text(
        tools_init.read_text(encoding="utf-8")
        + f"\nfrom pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    # A tagged CI gate must not pass having checked nothing.
    assert main(["check", "--tag", "providers", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)

    assert not marker.exists()
    assert report["manifest_fingerprint"] == "unavailable"
    codes = [item["code"] for item in report["diagnostics"]]
    assert "SCAFFOLD_CONVENTION_UNSUPPORTED" in codes
    (not_imported,) = (
        item
        for item in report["diagnostics"]
        if item["code"] == "SCAFFOLD_APPLICATION_NOT_IMPORTED"
    )
    assert not_imported["parameters"]["blocking_findings_outside_selection"] >= 1
    assert "listed anyway" in not_imported["message"]


def test_cli_check_reports_a_factory_failure_with_only_the_selected_source_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    (project / "operations/watchers.py").unlink()
    app_module = project / "app.py"
    app_module.write_text(
        app_module.read_text(encoding="utf-8").replace(
            "def build_app(\n",
            "def build_app(*_args, **_kwargs):\n    raise RuntimeError('boom')\n\n\ndef _unused(\n",
            1,
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    # Unfiltered: the source error explains the factory failure next to it.
    assert main(["check", "--json"]) == 1
    codes = {item["code"] for item in json.loads(capsys.readouterr().out)["diagnostics"]}
    assert {"SCAFFOLD_LAYOUT_PATH_MISSING", "SCAFFOLD_APPLICATION_LOAD_FAILED"} <= codes

    # The source error is outside the tag, so the factory failure is reported
    # as such instead of a report whose findings would not explain it.
    assert main(["check", "--tag", "security", "--json"]) == 2
    assert "Application factory failed (RuntimeError)" in capsys.readouterr().out


@pytest.mark.parametrize("replacement", ("missing", "directory"))
def test_cli_check_reports_an_invalid_registration_seam_before_import(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    replacement: str,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    (project / "agents/registration.py").unlink()
    if replacement == "directory":
        (project / "agents/registration.py").mkdir()
    monkeypatch.chdir(project)

    assert main(["check", "--fail-on", "warning", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)

    assert report["manifest_fingerprint"] == "unavailable"
    registration_import = next(
        node
        for node in ast.walk(ast.parse((project / "app.py").read_text()))
        if isinstance(node, ast.ImportFrom) and node.module == "agents.registration"
    )
    assert [(item["code"], item["path"]) for item in report["diagnostics"]] == [
        ("SCAFFOLD_IMPORT_SIDE_EFFECT", f"app.py:{registration_import.lineno}"),
        ("SCAFFOLD_LAYOUT_PATH_MISSING", "agents/registration.py"),
        # The report says which checks did not run instead of stopping silently.
        ("SCAFFOLD_APPLICATION_NOT_IMPORTED", "app.py"),
    ]
    assert report["diagnostics"][0]["parameters"]["expression_kind"] == "ImportFrom"


def test_cli_check_rejects_a_symlinked_application_package_before_import(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    agents = project / "agents"
    external_agents = tmp_path / "external-agents"
    agents.rename(external_agents)
    registration = external_agents / "registration.py"
    registration.write_text(
        registration.read_text(encoding="utf-8").replace(
            "from cayu import AgentSpec, CayuApp\nfrom cayu.providers import ModelProvider\n",
            (
                "from pathlib import Path\n\n"
                'Path("symlink-side-effect.txt").write_text('
                '"executed\\n", encoding="utf-8")\n\n'
                "from cayu import AgentSpec, CayuApp\nfrom cayu.providers import ModelProvider\n"
            ),
        ),
        encoding="utf-8",
    )
    agents.symlink_to(external_agents, target_is_directory=True)
    monkeypatch.chdir(project)

    assert main(["check", "--fail-on", "warning", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)

    assert report["manifest_fingerprint"] == "unavailable"
    assert any(
        item["code"] == "SCAFFOLD_LAYOUT_PATH_MISSING" and item["path"] == "agents/registration.py"
        for item in report["diagnostics"]
    )
    assert not (project / "symlink-side-effect.txt").exists()


@pytest.mark.parametrize(
    "statement",
    (
        'Path("import-side-effect.txt").write_text("executed\\n", encoding="utf-8")',
        (
            'IMPORT_RESULT = Path("import-side-effect.txt").write_text('
            '"executed\\n", encoding="utf-8")'
        ),
        (
            '@Path("import-side-effect.txt").write_text("executed\\n", encoding="utf-8")\n'
            "def imported() -> None:\n"
            "    pass"
        ),
        (
            'def imported(value=Path("import-side-effect.txt").write_text('
            '"executed\\n", encoding="utf-8")) -> None:\n'
            "    pass"
        ),
        (
            "if True:\n"
            '    Path("import-side-effect.txt").write_text('
            '"executed\\n", encoding="utf-8")'
        ),
        (
            'def imported(value: Path("import-side-effect.txt").write_text('
            '"executed\\n", encoding="utf-8")) -> None:\n'
            "    pass"
        ),
        (
            'IMPORT_RESULT: Path("import-side-effect.txt").write_text('
            '"executed\\n", encoding="utf-8") = None'
        ),
        (
            "def import_decorator(function):\n"
            '    Path("import-side-effect.txt").write_text("executed\\n", encoding="utf-8")\n'
            "    return function\n\n"
            "@import_decorator\n"
            "def imported() -> None:\n"
            "    pass"
        ),
        (
            "class ImportMeta(type):\n"
            "    def __new__(cls, name, bases, namespace):\n"
            '        Path("import-side-effect.txt").write_text("executed\\n", encoding="utf-8")\n'
            "        return super().__new__(cls, name, bases, namespace)\n\n"
            "class Imported(metaclass=ImportMeta):\n"
            "    pass"
        ),
        (
            "class ImportBase:\n"
            "    def __init_subclass__(cls):\n"
            '        Path("import-side-effect.txt").write_text("executed\\n", encoding="utf-8")\n\n'
            "class Imported(ImportBase):\n"
            "    pass"
        ),
        (
            "def import_decorator(function):\n"
            '    Path("import-side-effect.txt").write_text("executed\\n", encoding="utf-8")\n'
            "    return function\n\n"
            "class Decorated:\n"
            "    property = import_decorator\n\n"
            "    @property\n"
            "    def value(self):\n"
            "        return None"
        ),
        (
            "class ImportBase:\n"
            "    def __init_subclass__(cls):\n"
            '        Path("import-side-effect.txt").write_text("executed\\n", encoding="utf-8")\n\n'
            "Alias = ImportBase\n\n"
            "class Imported(Alias):\n"
            "    pass"
        ),
        (
            "from dataclasses import dataclass\n\n"
            "def import_decorator():\n"
            '    Path("import-side-effect.txt").write_text("executed\\n", encoding="utf-8")\n'
            "    return lambda function: function\n\n"
            "class Decorated:\n"
            "    dataclass = import_decorator\n\n"
            "    @dataclass()\n"
            "    def value(self):\n"
            "        return None"
        ),
        (
            "class Outer:\n"
            "    class ImportBase:\n"
            "        def __init_subclass__(cls):\n"
            '            Path("import-side-effect.txt").write_text("executed\\n", encoding="utf-8")\n\n'
            "    Alias = ImportBase\n\n"
            "    class Imported(Alias):\n"
            "        pass"
        ),
        (
            "class AnnotationHook:\n"
            "    @classmethod\n"
            "    def __class_getitem__(cls, item):\n"
            '        Path("import-side-effect.txt").write_text("executed\\n", encoding="utf-8")\n'
            "        return item\n\n"
            "def imported(value: AnnotationHook[int]) -> None:\n"
            "    pass"
        ),
        (
            "class Outer:\n"
            "    class AnnotationHook:\n"
            "        @classmethod\n"
            "        def __class_getitem__(cls, item):\n"
            '            Path("import-side-effect.txt").write_text("executed\\n", encoding="utf-8")\n'
            "            return item\n\n"
            "    Alias = AnnotationHook\n\n"
            "    def imported(self, value: Alias[int]) -> None:\n"
            "        pass"
        ),
        (
            "from dataclasses import dataclass\n\n"
            "def import_call():\n"
            '    Path("import-side-effect.txt").write_text("executed\\n", encoding="utf-8")\n\n'
            "class Outer:\n"
            "    dataclass = import_call\n"
            "    RESULT = dataclass()"
        ),
        (
            "class AnnotationHook:\n"
            "    @classmethod\n"
            "    def __class_getitem__(cls, item):\n"
            '        Path("import-side-effect.txt").write_text("executed\\n", encoding="utf-8")\n'
            '        return "prompt"\n\n'
            "_SYSTEM_PROMPT_PARTS = []\n"
            "_SYSTEM_PROMPT_PARTS.append(AnnotationHook[int])"
        ),
    ),
    ids=(
        "expression",
        "assignment",
        "decorator",
        "default",
        "control-flow",
        "parameter-annotation",
        "annotated-assignment",
        "bare-decorator",
        "metaclass",
        "init-subclass-hook",
        "class-local-decorator-shadow",
        "init-subclass-hook-alias",
        "called-class-local-decorator-shadow",
        "nested-init-subclass-hook-alias",
        "annotation-subscript-hook",
        "nested-annotation-subscript-hook",
        "class-local-call-shadow",
        "generated-collection-subscript-hook",
    ),
)
def test_cli_check_rejects_import_side_effects_without_executing_them(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    statement: str,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    registration = project / "agents/registration.py"
    registration.write_text(
        registration.read_text(encoding="utf-8").replace(
            "from cayu import AgentSpec, CayuApp\nfrom cayu.providers import ModelProvider\n",
            (
                "from pathlib import Path\n\n"
                f"{statement}\n\n"
                "from cayu import AgentSpec, CayuApp\nfrom cayu.providers import ModelProvider\n"
            ),
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    assert main(["check", "--fail-on", "warning", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)

    assert report["manifest_fingerprint"] == "unavailable"
    finding = report["diagnostics"][0]
    assert finding["code"] == "SCAFFOLD_IMPORT_SIDE_EFFECT"
    assert finding["path"].startswith("agents/registration.py:")
    assert not (project / "import-side-effect.txt").exists()


@pytest.mark.parametrize("package", ("domain", "vendor"))
def test_cli_check_rejects_project_local_imported_class_hooks_before_import(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    package: str,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    package_path = project / package
    package_path.mkdir(exist_ok=True)
    (package_path / "__init__.py").touch()
    (package_path / "import_hook.py").write_text(
        "from pathlib import Path\n\n"
        "class ImportBase:\n"
        "    def __init_subclass__(cls):\n"
        '        Path("import-side-effect.txt").write_text("executed\\n", encoding="utf-8")\n',
        encoding="utf-8",
    )
    registration = project / "agents/registration.py"
    registration.write_text(
        registration.read_text(encoding="utf-8").replace(
            "from cayu import AgentSpec, CayuApp\nfrom cayu.providers import ModelProvider\n",
            (
                f"from {package}.import_hook import ImportBase\n\n"
                "class Imported(ImportBase):\n"
                "    pass\n\n"
                "from cayu import AgentSpec, CayuApp\nfrom cayu.providers import ModelProvider\n"
            ),
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    assert main(["check", "--fail-on", "warning", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)

    assert report["manifest_fingerprint"] == "unavailable"
    assert any(item["code"] == "SCAFFOLD_IMPORT_SIDE_EFFECT" for item in report["diagnostics"])
    assert not (project / "import-side-effect.txt").exists()


def test_cli_check_tracks_decorator_imports_inside_class_control_flow(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    vendor = project / "vendor"
    vendor.mkdir()
    (vendor / "__init__.py").touch()
    (vendor / "class_decorator.py").write_text(
        "from pathlib import Path\n\n"
        "def import_decorator(function):\n"
        '    Path("import-side-effect.txt").write_text("executed\\n", encoding="utf-8")\n'
        "    return function\n",
        encoding="utf-8",
    )
    registration = project / "agents/registration.py"
    registration.write_text(
        registration.read_text(encoding="utf-8").replace(
            "from cayu import AgentSpec, CayuApp\nfrom cayu.providers import ModelProvider\n",
            (
                "class Decorated:\n"
                "    if True:\n"
                "        from vendor.class_decorator import import_decorator as property\n\n"
                "    @property\n"
                "    def value(self):\n"
                "        return None\n\n"
                "from cayu import AgentSpec, CayuApp\nfrom cayu.providers import ModelProvider\n"
            ),
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    assert main(["check", "--fail-on", "warning", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)

    assert report["manifest_fingerprint"] == "unavailable"
    assert any(item["code"] == "SCAFFOLD_IMPORT_SIDE_EFFECT" for item in report["diagnostics"])
    assert not (project / "import-side-effect.txt").exists()


def test_cli_check_tracks_decorator_imports_inside_module_control_flow(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    vendor = project / "vendor"
    vendor.mkdir()
    (vendor / "__init__.py").touch()
    (vendor / "module_decorator.py").write_text(
        "from pathlib import Path\n\n"
        "def import_decorator(function):\n"
        '    Path("import-side-effect.txt").write_text("executed\\n", encoding="utf-8")\n'
        "    return function\n",
        encoding="utf-8",
    )
    registration = project / "agents/registration.py"
    registration.write_text(
        registration.read_text(encoding="utf-8").replace(
            "from cayu import AgentSpec, CayuApp\nfrom cayu.providers import ModelProvider\n",
            (
                "if True:\n"
                "    from vendor.module_decorator import import_decorator as property\n\n"
                "@property\n"
                "def value():\n"
                "    return None\n\n"
                "from cayu import AgentSpec, CayuApp\nfrom cayu.providers import ModelProvider\n"
            ),
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    assert main(["check", "--fail-on", "warning", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)

    assert report["manifest_fingerprint"] == "unavailable"
    assert any(item["code"] == "SCAFFOLD_IMPORT_SIDE_EFFECT" for item in report["diagnostics"])
    assert not (project / "import-side-effect.txt").exists()


def test_cli_check_requires_literal_generated_collection_receivers(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    (project / "domain/effect_collection.py").write_text(
        "from pathlib import Path\n\n"
        "def append(value):\n"
        "    del value\n"
        '    Path("import-side-effect.txt").write_text("executed\\n", encoding="utf-8")\n',
        encoding="utf-8",
    )
    registration = project / "agents/registration.py"
    registration.write_text(
        registration.read_text(encoding="utf-8").replace(
            "from cayu import AgentSpec, CayuApp\nfrom cayu.providers import ModelProvider\n",
            (
                "import domain.effect_collection as _SYSTEM_PROMPT_PARTS\n\n"
                '_SYSTEM_PROMPT_PARTS.append("safe-looking")\n\n'
                "from cayu import AgentSpec, CayuApp\nfrom cayu.providers import ModelProvider\n"
            ),
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    assert main(["check", "--fail-on", "warning", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)

    assert report["manifest_fingerprint"] == "unavailable"
    assert any(item["code"] == "SCAFFOLD_IMPORT_SIDE_EFFECT" for item in report["diagnostics"])
    assert not (project / "import-side-effect.txt").exists()


def test_cli_check_rejects_executable_generated_collection_arguments(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    (project / "domain/effect_attribute.py").write_text(
        "from pathlib import Path\n\n"
        "def __getattr__(name):\n"
        "    del name\n"
        '    Path("import-side-effect.txt").write_text("executed\\n", encoding="utf-8")\n'
        '    return "unsafe"\n',
        encoding="utf-8",
    )
    registration = project / "agents/registration.py"
    registration.write_text(
        registration.read_text(encoding="utf-8").replace(
            "from cayu import AgentSpec, CayuApp\nfrom cayu.providers import ModelProvider\n",
            (
                "import domain.effect_attribute as effect_attribute\n\n"
                "_SYSTEM_PROMPT_PARTS: list[str] = []\n"
                "_SYSTEM_PROMPT_PARTS.append(effect_attribute.trigger)\n\n"
                "from cayu import AgentSpec, CayuApp\nfrom cayu.providers import ModelProvider\n"
            ),
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    assert main(["check", "--fail-on", "warning", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)

    assert report["manifest_fingerprint"] == "unavailable"
    assert any(item["code"] == "SCAFFOLD_IMPORT_SIDE_EFFECT" for item in report["diagnostics"])
    assert not (project / "import-side-effect.txt").exists()


def test_cli_check_rejects_project_local_module_attribute_access_before_import(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    (project / "domain/effect_attribute.py").write_text(
        "from pathlib import Path\n\n"
        "def __getattr__(name):\n"
        "    del name\n"
        '    Path("import-side-effect.txt").write_text("executed\\n", encoding="utf-8")\n'
        '    return "unsafe"\n',
        encoding="utf-8",
    )
    registration = project / "agents/registration.py"
    registration.write_text(
        registration.read_text(encoding="utf-8").replace(
            "from cayu import AgentSpec, CayuApp\nfrom cayu.providers import ModelProvider\n",
            (
                "import domain.effect_attribute as effect_attribute\n\n"
                "IMPORT_RESULT = effect_attribute.trigger\n\n"
                "from cayu import AgentSpec, CayuApp\nfrom cayu.providers import ModelProvider\n"
            ),
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    assert main(["check", "--fail-on", "warning", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)

    assert report["manifest_fingerprint"] == "unavailable"
    assert any(item["code"] == "SCAFFOLD_IMPORT_SIDE_EFFECT" for item in report["diagnostics"])
    assert not (project / "import-side-effect.txt").exists()


@pytest.mark.parametrize(
    "hook_source",
    (
        (
            "def __getattr__(name: str) -> object:\n"
            "    del name\n"
            '    Path("import-side-effect.txt").write_text("executed\\n", encoding="utf-8")\n'
            "    return object()\n"
        ),
        (
            "if True:\n"
            "    def __getattr__(name: str) -> object:\n"
            "        del name\n"
            '        Path("import-side-effect.txt").write_text("executed\\n", encoding="utf-8")\n'
            "        return object()\n"
        ),
        (
            "def import_hook(name: str) -> object:\n"
            "    del name\n"
            '    Path("import-side-effect.txt").write_text("executed\\n", encoding="utf-8")\n'
            "    return object()\n\n"
            "if (__getattr__ := import_hook):\n"
            "    pass\n"
        ),
        (
            "trigger: object\n\n"
            "def __getattr__(name: str) -> object:\n"
            "    del name\n"
            '    Path("import-side-effect.txt").write_text("executed\\n", encoding="utf-8")\n'
            "    return object()\n"
        ),
        (
            "trigger = 1\n"
            "del trigger\n\n"
            "def __getattr__(name: str) -> object:\n"
            "    del name\n"
            '    Path("import-side-effect.txt").write_text("executed\\n", encoding="utf-8")\n'
            "    return object()\n"
        ),
    ),
    ids=(
        "direct",
        "module-control-flow",
        "assignment-expression",
        "annotation-only-import",
        "deleted-import",
    ),
)
def test_cli_check_rejects_project_local_import_from_hooks_before_import(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    hook_source: str,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    (project / "domain/effect_from.py").write_text(
        f"from pathlib import Path\n\n{hook_source}",
        encoding="utf-8",
    )
    registration = project / "agents/registration.py"
    registration.write_text(
        registration.read_text(encoding="utf-8").replace(
            "from cayu import AgentSpec, CayuApp\nfrom cayu.providers import ModelProvider\n",
            (
                "from domain.effect_from import trigger\n\n"
                "from cayu import AgentSpec, CayuApp\nfrom cayu.providers import ModelProvider\n"
            ),
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    assert main(["check", "--fail-on", "warning", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)

    assert report["manifest_fingerprint"] == "unavailable"
    finding = next(
        item for item in report["diagnostics"] if item["code"] == "SCAFFOLD_IMPORT_SIDE_EFFECT"
    )
    assert finding["parameters"]["expression_kind"] == "ImportFrom"
    assert not (project / "import-side-effect.txt").exists()


def test_cli_check_rejects_import_time_assignment_targets_before_import(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    (project / "domain/effect_from.py").write_text(
        "from pathlib import Path\n\n"
        "def import_hook(name: str) -> object:\n"
        "    del name\n"
        '    Path("import-side-effect.txt").write_text("executed\\n", encoding="utf-8")\n'
        "    return object()\n\n"
        'globals()["__getattr__"] = import_hook\n',
        encoding="utf-8",
    )
    registration = project / "agents/registration.py"
    registration.write_text(
        registration.read_text(encoding="utf-8").replace(
            "from cayu import AgentSpec, CayuApp\nfrom cayu.providers import ModelProvider\n",
            (
                "from domain.effect_from import trigger\n\n"
                "from cayu import AgentSpec, CayuApp\nfrom cayu.providers import ModelProvider\n"
            ),
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    assert main(["check", "--fail-on", "warning", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)

    assert report["manifest_fingerprint"] == "unavailable"
    finding = next(
        item for item in report["diagnostics"] if item["code"] == "SCAFFOLD_IMPORT_SIDE_EFFECT"
    )
    assert finding["path"].startswith("domain/effect_from.py:")
    assert not (project / "import-side-effect.txt").exists()


def test_cli_check_rejects_metadata_only_preset_and_database_changes_before_import(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    pyproject = project / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace(
            'preset = "agent"', 'preset = "coding"\ndatabase = "mysql"'
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    assert main(["check", "--fail-on", "warning", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)

    assert report["manifest_fingerprint"] == "unavailable"
    assert {item["code"] for item in report["diagnostics"]} == {
        "SCAFFOLD_CONTRACT_INVALID",
        "SCAFFOLD_APPLICATION_NOT_IMPORTED",
    }


@pytest.mark.parametrize(
    ("replacements", "field"),
    (
        ((('path = "data/cayu.db"', 'path = "data/other.db"'),), "storage"),
        (
            (
                (
                    'backend = "sqlite"\npath = "data/cayu.db"',
                    'backend = "postgres"\nenv = "DB_URL"',
                ),
            ),
            "storage",
        ),
        ((('provider = "neutral"', 'provider = "openai"'),), "provider"),
        (
            (
                (
                    'capabilities = ["approvals", "artifacts", "evals", '
                    '"human-input", "knowledge", "memory", "observability", '
                    '"recovery", "tasks"]',
                    'capabilities = ["approvals", "artifacts", "evals", '
                    '"human-input", "knowledge", "memory", "recovery", "tasks"]',
                ),
            ),
            "capabilities",
        ),
        (
            (
                ('preset = "agent"', 'preset = "service"'),
                (
                    'capabilities = ["approvals", "artifacts", "evals", '
                    '"human-input", "knowledge", "memory", "observability", '
                    '"recovery", "tasks"]',
                    'capabilities = ["approvals", "evals", "observability", "tasks"]',
                ),
            ),
            "preset",
        ),
        (
            (
                ('preset = "agent"', 'preset = "coding"'),
                ('execution = "none"', 'execution = "docker"'),
                (
                    'capabilities = ["approvals", "artifacts", "evals", '
                    '"human-input", "knowledge", "memory", "observability", '
                    '"recovery", "tasks"]',
                    'capabilities = ["artifacts", "delegation", "evals", "human-input", "knowledge", "tasks"]',
                ),
            ),
            "execution",
        ),
    ),
)
def test_source_check_validates_every_selected_plan_axis(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    replacements: tuple[tuple[str, str], ...],
    field: str,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    pyproject = project / "pyproject.toml"
    source = pyproject.read_text(encoding="utf-8")
    for before, after in replacements:
        source = source.replace(before, after)
    pyproject.write_text(source, encoding="utf-8")

    findings = check_declared_scaffold_source(project)

    assert any(
        item.code == "SCAFFOLD_PLAN_DRIFT" and item.parameters["field"] == field
        for item in findings
    )


_LEGACY_SQLITE_STORAGE = """from cayu.storage import SQLiteSessionStore, SQLiteTaskStore


def build_stores(**kwargs):
    return SQLiteSessionStore("data/cayu.db"), SQLiteTaskStore("data/cayu.db")
"""


@pytest.mark.parametrize("storage", ("legacy", "handwritten", "current"))
def test_projects_from_the_retired_database_choice_report_fixed_storage(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    storage: str,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    pyproject = project / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace(
            'preset = "agent"', 'preset = "agent"\ndatabase = "sqlite"'
        ),
        encoding="utf-8",
    )
    storage_module = project / "configuration/storage.py"
    if storage == "legacy":
        storage_module.write_text(_LEGACY_SQLITE_STORAGE, encoding="utf-8")
    elif storage == "handwritten":
        storage_module.write_text("def build_stores(**kwargs):\n    return None\n")

    findings = [
        item
        for item in check_declared_scaffold_source(project)
        if item.code == "SCAFFOLD_PLAN_DRIFT" and item.parameters["field"] == "storage"
    ]

    if storage == "current":
        # The retired contract key alone is accepted; the storage module decides.
        assert findings == []
        return
    assert len(findings) == 1
    finding = findings[0]
    assert finding.path == "configuration/storage.py"
    assert finding.parameters["expected"] == "open_application_stores"
    assert finding.parameters["observed"] == ("SQLiteSessionStore" if storage == "legacy" else None)
    assert "CAYU_DATABASE_URL" in finding.hint


def test_coding_projects_keep_local_tooling_on_the_protected_state_database(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["new", "coder", "--preset", "coding", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "coder"
    pyproject = project / "pyproject.toml"
    assert check_declared_scaffold_source(project) == ()

    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace(
            'path = ".cayu/runtime/cayu.db"', 'path = "data/cayu.db"'
        ),
        encoding="utf-8",
    )
    (finding,) = check_declared_scaffold_source(project)
    assert finding.code == "SCAFFOLD_PLAN_DRIFT"
    assert finding.parameters["field"] == "storage"
    assert finding.path == "pyproject.toml:[tool.cayu.session_store]"
    assert finding.parameters["expected"] == {
        "backend": "sqlite",
        "path": ".cayu/runtime/cayu.db",
    }


_ORDINARY_MODULES = {
    "module_logger": "import logging\n\nlogger = logging.getLogger(__name__)\n",
    "method_named_list": ("class Store:\n    def list(self) -> list[dict]:\n        return []\n"),
    "list_annotation_before_list_method": (
        "class Store:\n"
        "    def items(self) -> list[dict]:\n"
        "        return []\n\n"
        "    def list(self) -> None:\n"
        "        return None\n"
    ),
    "postponed_annotation_after_list_method": (
        "from __future__ import annotations\n\n\n"
        "class Store:\n"
        "    def list(self) -> None:\n"
        "        return None\n\n"
        "    def later(self) -> list[int]:\n"
        "        return []\n"
    ),
}
_IMPORT_TIME_MODULES = {
    # Evaluated eagerly on Python 3.11-3.13, where ``list`` is now the method.
    "annotation_after_list_method": (
        "class Store:\n"
        "    def list(self) -> None:\n"
        "        return None\n\n"
        "    def later(self) -> list[int]:\n"
        "        return []\n"
    ),
    "logger_named_from_environment": (
        'import logging\nimport os\n\nlogger = logging.getLogger(os.environ["NAME"])\n'
    ),
    "environment_read": 'import os\n\nNAME = os.environ.get("NAME")\n',
}


def _import_side_effects(project: Path, source: str) -> list[dict]:
    (project / "domain/probe.py").write_text(source, encoding="utf-8")
    return [
        diagnostic.model_dump(mode="json")
        for diagnostic in check_declared_scaffold_source(project)
        if diagnostic.code == "SCAFFOLD_IMPORT_SIDE_EFFECT"
    ]


@pytest.mark.parametrize("case", sorted(_ORDINARY_MODULES))
def test_import_checker_accepts_ordinary_module_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], case: str
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()

    assert _import_side_effects(tmp_path / "project", _ORDINARY_MODULES[case]) == []


@pytest.mark.parametrize("case", sorted(_IMPORT_TIME_MODULES))
def test_import_checker_still_rejects_real_import_time_work(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], case: str
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()

    findings = _import_side_effects(tmp_path / "project", _IMPORT_TIME_MODULES[case])
    assert [finding["path"].split(":")[0] for finding in findings] == ["domain/probe.py"]


def test_cli_check_reports_every_source_finding_when_import_is_unsafe(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    (project / "tools" / "probe.py").write_text(
        "import subprocess\n\nsubprocess.run(['true'])\n", encoding="utf-8"
    )
    with (project / "app.py").open("a", encoding="utf-8") as app_source:
        app_source.write("\n\ndef helper() -> int:\n    return 1\n")
    monkeypatch.chdir(project)

    assert main(["check", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)

    # The unsafe module is never imported, but lower-severity source findings
    # are no longer hidden behind it, and the report says what did not run.
    assert report["manifest_fingerprint"] == "unavailable"
    codes = [item["code"] for item in report["diagnostics"]]
    assert codes == [
        "SCAFFOLD_IMPORT_SIDE_EFFECT",
        "SCAFFOLD_APPLICATION_NOT_IMPORTED",
        "SCAFFOLD_APP_COMPOSITION_DRIFT",
    ]
    assert report["diagnostics"][0]["parameters"]["reason"] == "import_time_effect"


def test_cli_check_never_constructs_a_zone_that_can_import_project_tzdata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    marker = tmp_path / "tzdata-payload-ran"
    # zoneinfo falls back to importing `tzdata.zoneinfo` for a key missing from
    # the system database, and the project root is first on sys.path.
    (project / "tzdata").mkdir()
    (project / "tzdata" / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n",
        encoding="utf-8",
    )
    (project / "workflows" / "settings.py").write_text(
        'from zoneinfo import ZoneInfo\n\nZONE = ZoneInfo("Cayu/Not_A_Zone")\n',
        encoding="utf-8",
    )
    # Reached from the factory, so accepting the call would run the payload.
    with (project / "agents" / "agent.py").open("a", encoding="utf-8") as agent_source:
        agent_source.write("\nfrom workflows.settings import ZONE  # noqa: E402,F401\n")
    for name in [name for name in sys.modules if name == "tzdata" or name.startswith("tzdata.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.chdir(project)

    assert main(["check", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)

    assert [(item["code"], item["path"]) for item in report["diagnostics"]] == [
        ("SCAFFOLD_IMPORT_SIDE_EFFECT", "workflows/settings.py:3"),
        ("SCAFFOLD_APPLICATION_NOT_IMPORTED", "app.py"),
    ]
    assert not marker.exists()


def test_cli_check_still_loads_the_app_after_a_non_import_source_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    # An agent project with a Docker sandbox config is plan drift, not unsafe source.
    (project / "docker-sandbox.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.chdir(project)

    assert main(["check", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)

    assert report["manifest_fingerprint"] != "unavailable"
    assert [item["code"] for item in report["diagnostics"]] == ["SCAFFOLD_PLAN_DRIFT"]


def test_cli_check_folds_an_application_load_failure_into_the_source_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    (project / "docker-sandbox.json").write_text("{}\n", encoding="utf-8")
    runtime = project / "configuration" / "runtime.py"
    source = runtime.read_text(encoding="utf-8")
    marker = "def build_runtime_options() -> RuntimeOptions:\n"
    assert marker in source
    runtime.write_text(
        source.replace(marker, marker + '    raise RuntimeError("factory failed")\n'),
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    assert main(["check", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)

    assert [item["code"] for item in report["diagnostics"]] == [
        "SCAFFOLD_APPLICATION_LOAD_FAILED",
        "SCAFFOLD_PLAN_DRIFT",
    ]
