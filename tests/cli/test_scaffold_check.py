from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from cayu.cli import main
from cayu.cli.project import project_context
from cayu.cli.scaffold_check import check_declared_scaffold, check_declared_scaffold_source
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
    assert [(item["code"], item["path"]) for item in report["diagnostics"]] == [
        ("SCAFFOLD_LAYOUT_PATH_MISSING", "agents/registration.py")
    ]


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
            "from cayu import AgentSpec, CayuApp, ModelProvider\n",
            (
                "from pathlib import Path\n\n"
                'Path("symlink-side-effect.txt").write_text('
                '"executed\\n", encoding="utf-8")\n\n'
                "from cayu import AgentSpec, CayuApp, ModelProvider\n"
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
            "from cayu import AgentSpec, CayuApp, ModelProvider\n",
            (
                "from pathlib import Path\n\n"
                f"{statement}\n\n"
                "from cayu import AgentSpec, CayuApp, ModelProvider\n"
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
            "from cayu import AgentSpec, CayuApp, ModelProvider\n",
            (
                f"from {package}.import_hook import ImportBase\n\n"
                "class Imported(ImportBase):\n"
                "    pass\n\n"
                "from cayu import AgentSpec, CayuApp, ModelProvider\n"
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
            "from cayu import AgentSpec, CayuApp, ModelProvider\n",
            (
                "class Decorated:\n"
                "    if True:\n"
                "        from vendor.class_decorator import import_decorator as property\n\n"
                "    @property\n"
                "    def value(self):\n"
                "        return None\n\n"
                "from cayu import AgentSpec, CayuApp, ModelProvider\n"
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
            "from cayu import AgentSpec, CayuApp, ModelProvider\n",
            (
                "if True:\n"
                "    from vendor.module_decorator import import_decorator as property\n\n"
                "@property\n"
                "def value():\n"
                "    return None\n\n"
                "from cayu import AgentSpec, CayuApp, ModelProvider\n"
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
            "from cayu import AgentSpec, CayuApp, ModelProvider\n",
            (
                "import domain.effect_collection as _SYSTEM_PROMPT_PARTS\n\n"
                '_SYSTEM_PROMPT_PARTS.append("safe-looking")\n\n'
                "from cayu import AgentSpec, CayuApp, ModelProvider\n"
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
            "from cayu import AgentSpec, CayuApp, ModelProvider\n",
            (
                "import domain.effect_attribute as effect_attribute\n\n"
                "_SYSTEM_PROMPT_PARTS: list[str] = []\n"
                "_SYSTEM_PROMPT_PARTS.append(effect_attribute.trigger)\n\n"
                "from cayu import AgentSpec, CayuApp, ModelProvider\n"
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
            "from cayu import AgentSpec, CayuApp, ModelProvider\n",
            (
                "import domain.effect_attribute as effect_attribute\n\n"
                "IMPORT_RESULT = effect_attribute.trigger\n\n"
                "from cayu import AgentSpec, CayuApp, ModelProvider\n"
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
            "from cayu import AgentSpec, CayuApp, ModelProvider\n",
            (
                "from domain.effect_from import trigger\n\n"
                "from cayu import AgentSpec, CayuApp, ModelProvider\n"
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
            "from cayu import AgentSpec, CayuApp, ModelProvider\n",
            (
                "from domain.effect_from import trigger\n\n"
                "from cayu import AgentSpec, CayuApp, ModelProvider\n"
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
        pyproject.read_text(encoding="utf-8")
        .replace('preset = "agent"', 'preset = "coding"')
        .replace('database = "sqlite"', 'database = "postgres"'),
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    assert main(["check", "--fail-on", "warning", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)

    assert report["manifest_fingerprint"] == "unavailable"
    assert {item["code"] for item in report["diagnostics"]} == {"SCAFFOLD_CONTRACT_INVALID"}


@pytest.mark.parametrize(
    ("replacements", "field"),
    (
        ((('database = "sqlite"', 'database = "postgres"'),), "database"),
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
                    'capabilities = ["delegation", "evals", "human-input", "knowledge", "tasks"]',
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
