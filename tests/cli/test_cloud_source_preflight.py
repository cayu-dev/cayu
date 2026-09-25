from __future__ import annotations

import io
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from cayu.cli import _cloud_project as project
from cayu.cli import main


@pytest.fixture
def source(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "preflight-agent"\nversion = "1.0"\n'
    )
    (tmp_path / "uv.lock").write_text("version = 1\n")
    project.initialize_project(tmp_path)
    return tmp_path


@pytest.mark.parametrize("missing", ["uv.lock", "pyproject.toml"])
def test_missing_input_rejected_before_authentication_or_upload(
    source: Path,
    missing: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (source / missing).unlink()
    assert main(["cloud", "--evidence-dir", str(source / "evidence"), "deploy", str(source)]) == 2
    output = json.loads(capsys.readouterr().out)
    assert output["error"]["category"] == "source_build_inputs_invalid"
    assert output["error"]["path"] == missing
    assert output["error"]["reason"] == "missing"
    assert "uv lock" in output["error"]["hint"]
    assert not (source / missing).exists()


def test_ignored_untracked_lock_is_rejected_but_tracked_lock_is_included(source: Path) -> None:
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    (source / ".gitignore").write_text("uv.lock\n")
    with pytest.raises(project.CloudSourceInputsError, match="excluded from the upload"):
        project.resolve_project(str(source), manifest_path=None, revision=None)
    subprocess.run(["git", "-C", str(source), "add", "-f", "uv.lock"], check=True)
    bundle = project.resolve_project(str(source), manifest_path=None, revision=None).bundle
    assert bundle is not None
    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as archive:
        assert archive.getmember("source/uv.lock").isfile()


@pytest.mark.parametrize("target", ["missing", "uv.lock", "../outside", "/etc/hosts"])
def test_unusable_lock_symlink_does_not_satisfy_input_requirement(
    source: Path, target: str
) -> None:
    (source / "uv.lock").unlink()
    (source / "uv.lock").symlink_to(target)
    with pytest.raises(project.CloudSourceInputsError, match="not a usable file"):
        project.resolve_project(str(source), manifest_path=None, revision=None)


def test_included_symlink_target_is_accepted_and_deterministic(source: Path) -> None:
    (source / "uv.lock").rename(source / "dependency.lock")
    (source / "uv.lock").symlink_to("dependency.lock")
    first = project.resolve_project(str(source), manifest_path=None, revision=None)
    second = project.resolve_project(str(source), manifest_path=None, revision=None)
    assert first.bundle == second.bundle
    assert first.content_digest == second.content_digest


def test_nested_lock_does_not_satisfy_root_requirement(source: Path) -> None:
    (source / "nested").mkdir()
    (source / "uv.lock").rename(source / "nested/uv.lock")
    with pytest.raises(project.CloudSourceInputsError, match="uv.lock is missing"):
        project.resolve_project(str(source), manifest_path=None, revision=None)


def test_empty_input_is_rejected(source: Path) -> None:
    (source / "uv.lock").write_bytes(b"")
    with pytest.raises(project.CloudSourceInputsError, match="not a usable file"):
        project.resolve_project(str(source), manifest_path=None, revision=None)
