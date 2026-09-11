"""Complete staged input identity, including generated guidance and binary assets."""

import json
from types import SimpleNamespace

import pytest
from scripts import run_runtime_qualification as runner

from tests.qualification.registry import Scenario

INPUTS = {
    "tests/case.py": b"assert True\n",
    "tests/deployment.md": b"Reviewed deployment guide\n",
    "tests/corpus.bin": b"\x00\xff\x01",
    "pyproject.toml": b"[project]\nname='fixture'\n",
    "scripts/run_runtime_qualification.py": b"# staged runner\n",
}


def stage(root, *, reverse=False):
    for name in sorted(INPUTS, reverse=reverse):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(INPUTS[name])
    return root


@pytest.mark.parametrize("name", INPUTS)
def test_every_consumed_input_changes_identity_without_binding_temp_root(tmp_path, name):
    first = stage(tmp_path / "first")
    second = stage(tmp_path / "second", reverse=True)
    original = runner.fixture_fingerprint(first)
    assert runner.fixture_fingerprint(second) == original
    (second / name).write_bytes(INPUTS[name] + b"changed")
    assert runner.fixture_fingerprint(second) != original


def test_new_removed_and_renamed_assets_are_bound(tmp_path):
    root = stage(tmp_path)
    original = runner.fixture_fingerprint(root)
    extra = root / "tests/other.unknown"
    extra.write_bytes(b"same content")
    added = runner.fixture_fingerprint(root)
    assert added != original
    renamed = extra.with_name("renamed.unknown")
    extra.rename(renamed)
    assert runner.fixture_fingerprint(root) not in {original, added}
    renamed.unlink()
    assert runner.fixture_fingerprint(root) == original


def test_main_publishes_identity_for_the_staged_guide_it_consumes(tmp_path, monkeypatch):
    source = stage(tmp_path / "source")
    monkeypatch.setattr(runner, "ROOT", source)
    monkeypatch.setattr(
        runner, "SCENARIOS", (Scenario("fixture", "contract", "stage", ("tests/case.py",)),)
    )
    monkeypatch.setattr(runner, "POSTGRES_SCENARIOS", ())
    monkeypatch.setattr(runner, "DOCKER_SCENARIOS", ())
    build = {"availability": "available", "origin": "wheel_record", "fingerprint": "fixture"}
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            returncode=0, stdout=json.dumps({"package": "fixture", "build": build})
        ),
    )
    consumed = []

    def execute(command, *, cwd, env, timeout, cleanup):
        del command, timeout, cleanup
        consumed.append(
            ((cwd / "tests/deployment.md").read_bytes(), runner.fixture_fingerprint(cwd))
        )
        result = {
            "build": build,
            "cases": {"fixture": {"phases": {"call": "passed"}}},
            "resources": {
                "subprocesses_retained": 0,
                "postgres_databases_retained": 0,
                "subprocess_groups_remaining": 0,
                "subprocesses_remaining": 0,
            },
        }
        from pathlib import Path

        Path(env["CAYU_QUALIFICATION_RESULT"]).write_text(json.dumps(result))
        return 0

    monkeypatch.setattr(runner, "run_bounded", execute)
    fingerprints = []
    for i, content in enumerate((b"first guide", b"second guide")):
        (source / "tests/deployment.md").write_bytes(content)
        report = tmp_path / f"report-{i}.json"
        monkeypatch.setattr("sys.argv", ["qualification", "--repeat", "1", "--report", str(report)])
        assert runner.main() == 0
        value = json.loads(report.read_text())
        assert value["status"] == "passed" and value["fixtures_recipe"] == runner.FIXTURES_RECIPE
        assert consumed[-1] == (content, value["fixtures_sha256"])
        fingerprints.append(value["fixtures_sha256"])
    assert len(set(fingerprints)) == 2
