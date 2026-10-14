"""Old application imports get repairs before application code can run."""

from pathlib import Path

import pytest

from cayu.cli import main
from cayu.cli.scaffold_check import _check_import_inertness, check_declared_scaffold_source


@pytest.mark.parametrize(
    ("source", "replacement"),
    [
        (
            "from cayu import ExecutionProfileBehaviorIdentity\n"
            "IDENTITY = ExecutionProfileBehaviorIdentity(name='local', version='1')\n",
            "from cayu.runtime import ExecutionProfileBehaviorIdentity",
        ),
        ("from cayu.runtime import RetryPolicy\n", "from cayu.providers import RetryPolicy"),
        (
            "from cayu import ModelProvider as Provider\n",
            "from cayu.providers import ModelProvider as Provider",
        ),
        (
            "def create():\n    from cayu import ModelProvider\n",
            "from cayu.providers import ModelProvider",
        ),
    ],
)
def test_removed_imports_have_exact_repairs_instead_of_side_effect_errors(
    tmp_path, source, replacement
):
    (tmp_path / "app.py").write_text(source)
    findings = _check_import_inertness(tmp_path)
    assert [item.code for item in findings] == ["SCAFFOLD_IMPORT_REMOVED"]
    assert replacement in findings[0].hint
    assert findings[0].path.startswith("app.py:")


def test_multiline_imports_report_each_removed_name_at_its_line(tmp_path):
    (tmp_path / "app.py").write_text(
        "from cayu import (\n    AgentSpec,\n    ModelProvider,\n    RetryPolicy as Policy,\n)\n"
    )
    findings = _check_import_inertness(tmp_path)
    assert [item.path for item in findings] == ["app.py:3", "app.py:4"]
    assert "from cayu.providers import RetryPolicy as Policy" in findings[1].hint


def test_explicit_local_helper_gets_its_own_migration_finding(tmp_path):
    (tmp_path / "support").mkdir()
    (tmp_path / "support/__init__.py").write_text("")
    (tmp_path / "support/declarations.py").write_text("from cayu import ModelProvider\n")
    (tmp_path / "app.py").write_text("from support.declarations import ModelProvider\n")
    findings = _check_import_inertness(tmp_path)
    migrations = [item for item in findings if item.code == "SCAFFOLD_IMPORT_REMOVED"]
    assert [item.path for item in migrations] == ["support/declarations.py:1"]


@pytest.mark.parametrize(
    "source", ["from .cayu import ModelProvider\n", "from cayu import ModelProvider\n"]
)
def test_local_modules_are_not_mistaken_for_removed_framework_imports(tmp_path, source):
    (tmp_path / "cayu.py").write_text("ModelProvider = 1\n")
    (tmp_path / "app.py").write_text(source)
    assert not any(
        item.code == "SCAFFOLD_IMPORT_REMOVED" for item in _check_import_inertness(tmp_path)
    )


@pytest.mark.parametrize("selection", [[], ["--tag", "security"], ["--deploy"]])
def test_old_generated_import_blocks_loading_under_check_filters(
    tmp_path, monkeypatch, capsys, selection
):
    assert main(["new", "probe", "--dir", str(tmp_path.resolve())]) == 0
    project = tmp_path / "probe"
    module = project / "environments/local.py"
    source = module.read_text()
    old = source.replace(
        "from cayu.runtime import ExecutionProfileBehaviorIdentity",
        "from cayu import ExecutionProfileBehaviorIdentity",
    )
    assert source != old
    module.write_text(old)
    marker = project / "loaded.txt"
    app = project / "app.py"
    app_source = app.read_text()
    body = "    knowledge_scope = build_knowledge_scope()"
    assert body in app_source
    app.write_text(app_source.replace(body, f"    open({str(marker)!r}, 'w').close()\n" + body))
    monkeypatch.chdir(project)
    capsys.readouterr()
    assert main(["check", "--json", *selection]) == 1
    output = capsys.readouterr().out
    assert "SCAFFOLD_IMPORT_REMOVED" in output
    assert "from cayu.runtime import ExecutionProfileBehaviorIdentity" in output
    assert "SCAFFOLD_APPLICATION_NOT_IMPORTED" in output
    assert not marker.exists()
    module.write_text(source)
    assert not any(
        item.code == "SCAFFOLD_IMPORT_REMOVED" for item in check_declared_scaffold_source(project)
    )


@pytest.mark.parametrize("argument", ["max_attempts=3", "**unknown", "max_attempts=construct()"])
def test_retry_policy_package_import_retains_declaration_safety(tmp_path: Path, argument: str):
    (tmp_path / "app.py").write_text(
        f"from cayu.providers import RetryPolicy\npolicy = RetryPolicy({argument})\n"
    )
    findings = _check_import_inertness(tmp_path)
    if argument == "max_attempts=3":
        assert findings == ()
    else:
        assert [item.code for item in findings] == ["SCAFFOLD_IMPORT_SIDE_EFFECT"]
