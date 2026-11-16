"""The Knowledge area `cayu new` gives an Agent with knowledge (`scaffold_app_knowledge.py`).

Which projects get it, and how the web app composes it, is `test_scaffold_app_web.py`'s. With
organizations, the generated project's own tests (`tests/test_knowledge_area.py` among them)
run on PostgreSQL in `test_scaffold_organizations.py`. Here a generated operator project runs
its knowledge routes' tests: every role reads and searches what the Agent knows, and `admin`
and `developer` correct and forget it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from cayu.cli import scaffold_app_knowledge
from cayu.cli.scaffold import project_files

REPOSITORY = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("preset", ["agent", "coding"])
def test_a_generated_operator_projects_knowledge_routes_pass_their_tests(
    tmp_path: Path, preset: str
) -> None:
    files = project_files("acme-desk", preset=preset)
    router = files["integrations/http/__init__.py"]
    assert "operator_knowledge(scope=scope, namespace=KNOWLEDGE_NAMESPACE)" in router
    assert "member_knowledge" not in router
    for path, text in files.items():
        # The routes' tests read no part of the web app.
        if path.startswith("web/"):
            continue
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(text, encoding="utf-8")
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("CAYU_DATABASE_", "CAYU_TEST_POSTGRES", "CAYU_REQUIRE_POSTGRES"))
    }
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(REPOSITORY / "src"), environment.get("PYTHONPATH")))
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-v",
            "-p",
            "no:cacheprovider",
            scaffold_app_knowledge.KNOWLEDGE_TEST,
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "3 passed" in result.stdout


def test_the_routes_go_only_into_the_router_they_were_written_for() -> None:
    for wire in (
        scaffold_app_knowledge.with_operator_knowledge,
        scaffold_app_knowledge.with_member_knowledge,
    ):
        with pytest.raises(RuntimeError, match="isn't the web app's router"):
            wire('"""Someone else\'s router."""\n')
