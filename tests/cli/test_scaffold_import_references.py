"""Reads of names from other project modules, and scripts the application never imports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cayu.cli import main
from cayu.cli.scaffold_check import check_declared_scaffold_source

_RUNS = 'def default_run_key() -> str:\n    return "k"\n\n\nLIMIT = 3\n'
_DEFAULT_ARGUMENT = (
    "\n\ndef build(run_key=runs.default_run_key, limit=runs.LIMIT):\n    return run_key()\n"
)
_QUESTION = (
    "from dataclasses import dataclass\n\n\n@dataclass(frozen=True)\n"
    'class Question:\n    text: str\n    answer: str = ""\n'
)
_SCRIPT = (
    'import json\n\nDATA = json.loads("[1]")\n\n\nif __name__ == "__main__":\n    print(DATA)\n'
)


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("import-references")
    assert main(["new", "probe", "--dir", str(root)]) == 0
    return root / "probe"


def _findings(project: Path, files: dict[str, str]) -> list:
    written = []
    try:
        for relative, source in files.items():
            path = project / relative
            path.write_text(source, encoding="utf-8")
            written.append(path)
        return [
            diagnostic
            for diagnostic in check_declared_scaffold_source(project)
            if diagnostic.code.startswith("SCAFFOLD_")
        ]
    finally:
        for path in written:
            path.unlink()


@pytest.mark.parametrize(
    "binding",
    (
        "from tools import runs",
        "import tools.runs as runs",
        "from . import runs",
    ),
)
def test_a_project_module_attribute_is_accepted_as_a_default(project: Path, binding: str) -> None:
    # A default argument such as `run_key=task_runs.default_run_key`.
    findings = _findings(
        project,
        {"tools/runs.py": _RUNS, "tools/candidate.py": binding + "\n" + _DEFAULT_ARGUMENT},
    )

    assert findings == []


@pytest.mark.parametrize(
    ("runs", "read", "construct"),
    (
        (_RUNS, "runs.__dict__", "runs.__dict__"),
        (_RUNS, "runs.missing", "runs.missing"),
        (_RUNS, "runs.LIMIT.bit_length", "runs.LIMIT.bit_length"),
        (_RUNS, "runs.default_run_key()", "runs.default_run_key(...)"),
    ),
)
def test_other_reads_of_a_project_module_stay_refused(
    project: Path, runs: str, read: str, construct: str
) -> None:
    findings = _findings(
        project,
        {
            "tools/runs.py": runs,
            "tools/candidate.py": f"from tools import runs\n\nVALUE = {read}\n",
        },
    )

    assert [(item.code, item.parameters["construct"]) for item in findings] == [
        ("SCAFFOLD_IMPORT_SIDE_EFFECT", construct)
    ]


def test_a_module_getattr_hook_keeps_attribute_reads_refused(project: Path) -> None:
    findings = _findings(
        project,
        {
            "tools/runs.py": "def __getattr__(name):\n    print(name)\n    return 1\n",
            "tools/candidate.py": "from tools import runs\n\nVALUE = runs.anything\n",
        },
    )

    assert {item.path.split(":")[0] for item in findings} == {"tools/runs.py", "tools/candidate.py"}


@pytest.mark.parametrize("binding", ("from tools import runs", "import tools.runs as runs"))
def test_a_package_rebinding_the_submodule_name_is_not_the_module(
    project: Path, binding: str
) -> None:
    # Both forms return the package's attribute, which its initializer can rebind
    # after importing the submodule.
    init = project / "tools" / "__init__.py"
    original = init.read_text(encoding="utf-8")
    init.write_text(
        original + "\nfrom tools import runs as _real  # noqa: E402\n\nruns = 1\n", encoding="utf-8"
    )
    try:
        findings = _findings(
            project,
            {"tools/runs.py": _RUNS, "tools/candidate.py": f"{binding}\n\nVALUE = runs.LIMIT\n"},
        )
    finally:
        init.write_text(original, encoding="utf-8")

    assert "tools/candidate.py:3" in [item.path for item in findings]


def test_project_constants_and_frozen_dataclasses_can_be_imported(project: Path) -> None:
    # ToolSpec patterns from another module's constants,
    # and demo data built from a frozen dataclass.
    findings = _findings(
        project,
        {
            "tools/patterns.py": (
                'REPOSITORY_PATTERN = r"^[a-z]+/[a-z]+$"\nSHA_PATTERN = r"^[0-9a-f]{40}$"\n'
            ),
            "tools/question.py": _QUESTION,
            "tools/candidate.py": (
                "from cayu import Tool, ToolEffect, ToolSpec\n\n"
                "from tools.patterns import REPOSITORY_PATTERN, SHA_PATTERN\n"
                "from tools.question import Question\n\n"
                'QUESTIONS = (Question("a", answer="b"),)\n\n\n'
                "class Probe(Tool):\n    spec = ToolSpec(\n        name='probe',\n"
                "        description='d',\n        input_schema={'type': 'object', 'properties': {\n"
                "            'repository': {'type': 'string', 'pattern': REPOSITORY_PATTERN},\n"
                "            'head_sha': {'type': 'string', 'pattern': SHA_PATTERN},\n"
                "        }},\n        effect=ToolEffect.NONE,\n    )\n"
            ),
        },
    )

    assert findings == []


def test_a_script_nothing_imports_is_not_scanned(project: Path) -> None:
    # A script run as `python -m operations.demo_conversations`.
    assert _findings(project, {"operations/demo_conversations.py": _SCRIPT}) == []


@pytest.mark.parametrize(
    ("files", "reason"),
    (
        ({"operations/demo_conversations.py": _SCRIPT.split("\n\nif")[0] + "\n"}, "no main guard"),
        (
            {
                "operations/demo_conversations.py": _SCRIPT,
                "tools/candidate.py": (
                    "def load():\n    from operations import demo_conversations\n\n"
                    "    return demo_conversations\n"
                ),
            },
            "imported inside a function",
        ),
        (
            {
                "operations/demo_conversations.py": _SCRIPT,
                "tools/candidate.py": 'TARGET = "operations.demo_conversations:main"\n',
            },
            "named in a string",
        ),
        (
            {
                "operations/demo_conversations.py": _SCRIPT,
                "tools/candidate.py": (
                    "import importlib\n\n\ndef load(name):\n"
                    "    return importlib.import_module(f'operations.{name}')\n"
                ),
            },
            "dynamic import",
        ),
    ),
)
def test_a_script_is_scanned_when_the_application_may_import_it(
    project: Path, files: dict[str, str], reason: str
) -> None:
    findings = _findings(project, files)

    assert ("SCAFFOLD_IMPORT_SIDE_EFFECT", "operations/demo_conversations.py:3") in [
        (item.code, item.path) for item in findings
    ], reason


def test_a_script_named_in_pyproject_is_scanned(project: Path) -> None:
    pyproject = project / "pyproject.toml"
    original = pyproject.read_text(encoding="utf-8")
    pyproject.write_text(
        original + '\n[tool.cayu.workers]\ndemo = "operations.demo_conversations:main"\n',
        encoding="utf-8",
    )
    try:
        findings = _findings(project, {"operations/demo_conversations.py": _SCRIPT})
    finally:
        pyproject.write_text(original, encoding="utf-8")

    assert "operations/demo_conversations.py:3" in [item.path for item in findings]


def test_cli_check_imports_the_application_next_to_an_unimported_script(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    marker = tmp_path / "script-ran"
    (project / "operations" / "demo_conversations.py").write_text(
        f"from pathlib import Path\n\nPath({str(marker)!r}).write_text('ran')\n\n\n"
        'if __name__ == "__main__":\n    print("demo")\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    main(["check", "--json"])
    report = json.loads(capsys.readouterr().out)

    assert not marker.exists()
    assert report["manifest_fingerprint"] != "unavailable"
    assert [item for item in report["diagnostics"] if item["code"].startswith("SCAFFOLD_")] == []
