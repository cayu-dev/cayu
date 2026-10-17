from __future__ import annotations

import json
import runpy
import sys
import zipfile
from pathlib import Path

import cayu

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts/report_distribution_size.py"
_REPORT = runpy.run_path(str(_SCRIPT))
main = _REPORT["main"]
physical_lines = _REPORT["physical_lines"]
run_closure_probe = _REPORT["run_closure_probe"]
summarize_closure = _REPORT["summarize_closure"]
ROOT_MODULES = _REPORT["ROOT_MODULES"]


def _report(capsys, *args: str) -> dict:
    assert main([*args, "--no-closure", "--json"]) == 0
    return json.loads(capsys.readouterr().out)


def test_physical_lines_match_python_line_endings() -> None:
    assert physical_lines(b"") == 0
    assert physical_lines(b"a\nb\n") == 2
    assert physical_lines(b"a\r\nb") == 2
    assert physical_lines(b"a\rb\r") == 2


def test_wheel_inventory_counts_only_cayu_files_by_subpackage(tmp_path, capsys) -> None:
    wheel = tmp_path / "cayu-0.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("cayu/__init__.py", "import os\n")
        archive.writestr("cayu/applications.py", "a = 1\nb = 2\n")
        archive.writestr("cayu/runtime/__init__.py", "")
        archive.writestr("cayu/runtime/engine.py", "x = 1\ny = 2\nz = 3\n")
        archive.writestr("cayu/data/catalog.json", "{}")
        archive.writestr("cayu-0.0.0.dist-info/METADATA", "Name: cayu\n")

    report = _report(capsys, "--wheel", str(wheel))

    assert report["archive_bytes"] == wheel.stat().st_size
    assert report["totals"] == {"bytes": 42, "files": 5, "python_files": 4, "python_lines": 6}
    assert report["subpackages"] == {
        ROOT_MODULES: {"bytes": 22, "files": 2, "python_files": 2, "python_lines": 3},
        "runtime": {"bytes": 18, "files": 2, "python_files": 2, "python_lines": 3},
        "data": {"bytes": 2, "files": 1, "python_files": 0, "python_lines": 0},
    }
    assert report["closure"] is None


def test_source_inventory_skips_bytecode(tmp_path, capsys) -> None:
    package = tmp_path / "cayu"
    (package / "tools" / "__pycache__").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "tools" / "search.py").write_text("def search():\n    pass\n")
    (package / "tools" / "__pycache__" / "search.cpython-311.pyc").write_bytes(b"\0" * 64)

    report = _report(capsys, "--source", str(package))

    assert report["archive_bytes"] is None
    assert report["subpackages"]["tools"]["python_lines"] == 2
    assert report["totals"]["files"] == 2


def test_closure_summary_attributes_loaded_files_to_subpackages(tmp_path) -> None:
    root = tmp_path / "cayu"
    (root / "runtime").mkdir(parents=True)
    (root / "__init__.py").write_text("one\n")
    (root / "runtime" / "engine.py").write_text("one\ntwo\n")
    probe = {
        "package_root": str(root),
        "stages": [
            {"name": "import cayu", "modules": {"cayu": str(root / "__init__.py")}},
            {
                "name": "minimal app",
                "modules": {
                    "cayu": str(root / "__init__.py"),
                    "cayu.runtime.engine": str(root / "runtime" / "engine.py"),
                    "cayu.namespace": None,
                },
            },
        ],
        "third_party_top_level_modules": ["pydantic"],
    }

    closure = summarize_closure(probe)

    assert [(stage["modules"], stage["lines"]) for stage in closure["stages"]] == [(1, 1), (2, 3)]
    assert closure["stages"][1]["subpackages"] == {
        ROOT_MODULES: {"modules": 1, "lines": 1},
        "runtime": {"modules": 1, "lines": 2},
    }


def test_closure_probe_measures_the_installed_package_without_credentials() -> None:
    closure = summarize_closure(run_closure_probe(sys.executable))

    assert Path(closure["package_root"]) == Path(cayu.__file__).parent
    stages = closure["stages"]
    assert [stage["name"] for stage in stages] == ["import cayu", "minimal app", "one stub turn"]
    assert 0 < stages[0]["lines"] < stages[1]["lines"] <= stages[2]["lines"]
    assert stages[1]["subpackages"]["storage"]["modules"] > 0
