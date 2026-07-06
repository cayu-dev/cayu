"""Tests for ``cayu new`` (the project scaffold)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from cayu.cli import main


def test_cayu_new_creates_a_valid_importable_project(tmp_path: Path) -> None:
    assert main(["new", "myproj", "--dir", str(tmp_path)]) == 0
    proj = tmp_path / "myproj"
    for filename in ("app.py", "pyproject.toml", "README.md", ".gitignore"):
        assert (proj / filename).exists()
    for dirname in (
        "agents",
        "tools",
        "workflows",
        "prompts",
        "memory",
        "evals",
        "config",
        "tests",
        "data",
    ):
        assert (proj / dirname).is_dir()

    # The generated app.py must import cleanly: every cayu export in the template
    # exists and the syntax is valid. build_app() is not called at import, so no
    # API key is needed here.
    app_py = (proj / "app.py").read_text(encoding="utf-8")
    assert "# GitRepositoryBinding" in app_py
    assert "binding=GitRepositoryBinding" in app_py
    spec = importlib.util.spec_from_file_location("scaffolded_app", proj / "app.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "build_app")


def test_cayu_new_refuses_a_nonempty_directory(tmp_path: Path) -> None:
    proj = tmp_path / "myproj"
    proj.mkdir()
    (proj / "existing.txt").write_text("keep me")
    assert main(["new", "myproj", "--dir", str(tmp_path)]) == 1
    assert (proj / "existing.txt").read_text() == "keep me"


def test_cayu_new_refuses_an_existing_file(tmp_path: Path) -> None:
    proj = tmp_path / "myproj"
    proj.write_text("keep me")

    assert main(["new", "myproj", "--dir", str(tmp_path)]) == 1
    assert proj.read_text() == "keep me"


def test_cayu_new_rejects_invalid_names(tmp_path: Path) -> None:
    assert main(["new", "../escape", "--dir", str(tmp_path)]) == 1
    assert main(["new", "has space", "--dir", str(tmp_path)]) == 1
