"""docs/cayu-cloud.md states the Cloud limits and upload rules the CLI enforces."""

from __future__ import annotations

import re
from pathlib import Path

from cayu.cli import _cloud_project as project

_DOCS = Path(__file__).resolve().parents[2] / "docs" / "cayu-cloud.md"


def _section(heading: str) -> str:
    """The section's text up to its next heading, with whitespace collapsed."""

    text = _DOCS.read_text(encoding="utf-8")
    start = text.index(f"\n{heading}\n") + len(heading) + 2
    end = re.search(r"\n#{2,3} ", text[start:])
    body = text[start:] if end is None else text[start : start + end.start()]
    return " ".join(body.split())


def test_upload_rules_match_what_the_cli_uploads() -> None:
    section = _section("### What a deploy uploads")
    for name in sorted(project._EXCLUDED_DIRECTORIES | project._SAFE_ENV_TEMPLATES):
        assert f"`{name}`" in section, name
    megabytes = project.SOURCE_BUNDLE_MAX_BYTES // 1024 // 1024
    assert f"at most {megabytes} MiB" in section
    assert "`source_too_large`" in section


def test_process_limits_match_the_manifest_checks() -> None:
    section = _section("## Agent processes on Cayu Cloud")
    assert f"at most {project.SCHEDULES_LIMIT} schedules" in section
    # `[a-z0-9][a-z0-9-]{0,62}`: at most 63 characters.
    assert project._PROCESS_NAME.pattern.endswith("{0,62}\\Z")
    assert "interior hyphens (at most 63 characters)" in section


def test_schedule_input_limit_matches_the_manifest_check() -> None:
    section = _section("### Schedules")
    assert f"at most {project.SCHEDULE_INPUT_MAX_BYTES} bytes" in section
    assert "`CAYU_SCHEDULE_INPUT`" in section
