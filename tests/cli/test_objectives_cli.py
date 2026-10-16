from __future__ import annotations

import json
import shutil
from pathlib import Path

from cayu.cli import main
from cayu.evals import load_project_improvement

_EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "improvement_objectives"


def test_objectives_validate_prints_the_fingerprint_of_a_valid_project(capsys) -> None:
    declaration = load_project_improvement(_EXAMPLE)

    assert main(["objectives", "validate", str(_EXAMPLE / "prompts")]) == 0
    assert capsys.readouterr().out == (
        f"Valid improvement declaration {declaration.declaration_fingerprint}\n"
        f"Objectives {declaration.objectives_fingerprint} (cayu-objectives.toml)\n"
    )


def test_objectives_validate_json_describes_the_declaration(tmp_path: Path, capsys) -> None:
    output = tmp_path / "declaration.json"

    assert main(["objectives", "validate", str(_EXAMPLE), "--json", "--output", str(output)]) == 0
    document = json.loads(output.read_text(encoding="utf-8"))

    assert document["objectives_fingerprint"].startswith("sha256:")
    assert document["declaration_fingerprint"] == (
        load_project_improvement(_EXAMPLE).declaration_fingerprint
    )
    assert document["improvement"]["corpora"] == ["evals/improvement.json"]
    assert document["improvement"]["evaluator_only_paths"] == ["evals/reference"]
    assert document["objectives"]["sampled_review_rate_bps"] == 500
    assert document["objectives"]["mandatory_controls"][1] == {
        "tool_name": "issue_refund",
        "approval_gate": "refund-over-limit",
    }
    assert document["problems"] == []
    assert capsys.readouterr().out == ""


def test_objectives_validate_exits_two_with_every_problem(tmp_path: Path, capsys) -> None:
    root = tmp_path / "project"
    shutil.copytree(_EXAMPLE, root)
    shutil.rmtree(root / "evals" / "reference")

    assert main(["objectives", "validate", str(root)]) == 2
    assert "- evaluator-only path evals/reference does not exist." in capsys.readouterr().out

    assert main(["objectives", "validate", str(root), "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["problems"] == [
        "evaluator-only path evals/reference does not exist."
    ]


def test_objectives_validate_reports_a_project_without_a_declaration(
    tmp_path: Path, capsys
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.cayu]\nfactory = "app:build_app"\n', encoding="utf-8"
    )

    assert main(["objectives", "validate", str(tmp_path)]) == 2
    assert "[tool.cayu.improvement]" in capsys.readouterr().err

    assert main(["objectives", "validate", str(tmp_path), "--json"]) == 2
    document = json.loads(capsys.readouterr().out)
    assert document["schema_version"] == 1
    assert document["error"]["code"] == "OBJECTIVES_COMMAND_FAILED"
    assert "[tool.cayu.improvement]" in document["error"]["message"]


def test_objectives_validate_rejects_a_project_dir_that_does_not_exist(capsys) -> None:
    missing = _EXAMPLE / "does" / "not" / "exist"

    assert main(["objectives", "validate", str(missing)]) == 2
    assert f"{missing} is not a directory." in capsys.readouterr().err


def test_objectives_validate_rewrites_the_output_file_when_invalid(tmp_path: Path) -> None:
    root = tmp_path / "project"
    shutil.copytree(_EXAMPLE, root)
    report = tmp_path / "report.txt"

    assert main(["objectives", "validate", str(root), "--output", str(report)]) == 0
    assert report.read_text(encoding="utf-8").startswith("Valid improvement declaration")

    shutil.rmtree(root / "evals" / "reference")
    assert main(["objectives", "validate", str(root), "--output", str(report)]) == 2
    assert "- evaluator-only path evals/reference does not exist." in report.read_text(
        encoding="utf-8"
    )


def test_objectives_validate_refuses_to_overwrite_an_input(tmp_path: Path, capsys) -> None:
    root = tmp_path / "project"
    shutil.copytree(_EXAMPLE, root)
    objectives = root / "cayu-objectives.toml"
    reference = root / "evals" / "reference" / "expected_states.json"
    original = {path: path.read_bytes() for path in (objectives, reference)}

    for output in (objectives, reference):
        assert main(["objectives", "validate", str(root), "--output", str(output)]) == 2
        assert "--output must not overwrite" in capsys.readouterr().err

    assert {path: path.read_bytes() for path in original} == original

    bare = tmp_path / "bare"
    bare.mkdir()
    pyproject = bare / "pyproject.toml"
    pyproject.write_text('[tool.cayu]\nfactory = "app:build_app"\n', encoding="utf-8")
    assert main(["objectives", "validate", str(bare), "--json", "--output", str(pyproject)]) == 2
    assert pyproject.read_text(encoding="utf-8") == '[tool.cayu]\nfactory = "app:build_app"\n'
    assert "--output must not overwrite" in json.loads(capsys.readouterr().out)["error"]["message"]
