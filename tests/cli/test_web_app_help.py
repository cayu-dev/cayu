"""What `cayu new`'s help and the guides say about the web app and the b2b preset.

Every preset but service comes with the web app; b2b is that app plus organizations, for
people from separate organizations sharing one deployment. Help that tied the app to b2b
sent authors who wanted an app to b2b, and authors who didn't want organizations went
without one.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from cayu.cli import main
from cayu.cli.scaffold import project_files
from cayu.cli.scaffold_plan import PRESETS, WEB_APP, normalize_application_plan

if TYPE_CHECKING:
    import pytest

ROOT = Path(__file__).resolve().parents[2]
GUIDES = ("authoring", "applications", "app-ui", "b2b")
# Where a sentence that names b2b and the web app may do so: about organizations, or saying
# every preset has the app.
_ALLOWED = ("organization", "every preset")


def _output(capsys: pytest.CaptureFixture[str], *arguments: str) -> str:
    try:
        code = main(list(arguments))
    except SystemExit as exit_:  # --help
        code = exit_.code
    assert code == 0, arguments
    return capsys.readouterr().out


def _texts(capsys: pytest.CaptureFixture[str]) -> dict[str, str]:
    texts = {
        "cayu new --help": _output(capsys, "new", "--help"),
        "cayu new --list-presets": _output(capsys, "new", "--list-presets"),
        "cayu new --list-capabilities": _output(capsys, "new", "--list-capabilities"),
        "cayu new --explain web-app": _output(capsys, "new", "--explain", "web-app"),
        "cayu new --explain organizations": _output(capsys, "new", "--explain", "organizations"),
        "cayu guide": _output(capsys, "guide"),
        "README.md": (ROOT / "README.md").read_text(encoding="utf-8"),
    }
    for topic in GUIDES:
        texts[f"cayu guide {topic}"] = _output(capsys, "guide", topic)
    for preset in ("agent", "coding", "long-running"):
        texts[f"{preset} AGENTS.md"] = project_files("acme", preset=preset)["AGENTS.md"]
    return texts


def _sentences(text: str) -> list[str]:
    """Sentences, with each table row, heading and list item starting a new one."""

    chunks: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        starts = not stripped or re.match(r"(\||#|[-*] |\d+\. |```)", stripped)
        if starts or not chunks:
            chunks.append(stripped)
        else:
            chunks[-1] += " " + stripped
    return [
        sentence
        for chunk in chunks
        for sentence in re.split(r"(?<=[.;!?])\s+", chunk)
        if sentence.strip()
    ]


def test_no_help_or_guide_text_ties_the_web_app_to_b2b(
    capsys: pytest.CaptureFixture[str],
) -> None:
    for name, text in _texts(capsys).items():
        for sentence in _sentences(text):
            lowered = sentence.casefold()
            if "b2b" in lowered and re.search(r"web (app|interface|ui)", lowered):
                assert any(word in lowered for word in _ALLOWED), (name, sentence)
        assert "preset with a web app" not in text.casefold(), name


def test_the_presets_with_the_web_app_say_so(capsys: pytest.CaptureFixture[str]) -> None:
    listing = _output(capsys, "new", "--list-presets").splitlines()
    for spec in PRESETS:
        (line,) = [line for line in listing if line.split()[:1] == [spec.name]]
        assert "web app" in line, line
        assert ("no web app" in line) is (WEB_APP not in spec.default_capabilities), line

    preset_help = " ".join(_output(capsys, "new", "--help").split())
    assert "Every preset but service comes with a web app" in preset_help
    assert "--without web-app leaves it out" in preset_help

    explained = _output(capsys, "new", "--explain", "web-app")
    on_by_default = [spec.name for spec in PRESETS if WEB_APP in spec.default_capabilities]
    assert "On by default in: " + ", ".join(on_by_default) in explained


def test_a_projects_readme_and_agents_md_point_at_the_app_ui_guide() -> None:
    projects = {
        spec.name: project_files("acme", preset=spec.name)
        for spec in PRESETS
        if WEB_APP in spec.default_capabilities
    }
    assert {"agent", "coding", "long-running", "b2b"} <= projects.keys()
    projects["agent with organizations"] = project_files(
        "acme",
        application_plan=normalize_application_plan(
            name="acme",
            agent_name="acme",
            with_capabilities=("organizations",),
            without_capabilities=("knowledge", "memory"),
        ),
    )
    for name, files in projects.items():
        assert "web/package.json" in files, name
        for document in ("README.md", "AGENTS.md"):
            assert "cayu guide app-ui" in files[document], (name, document)
