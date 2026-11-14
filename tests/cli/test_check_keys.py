"""`cayu check` without a memory evidence key, and a review without a keyring."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from cayu.applications import CayuApp
from cayu.approvals import HumanReviewDisclosure, HumanReviewPolicy, ToolArgumentReview
from cayu.cli import main
from cayu.cli.check import _key_findings
from cayu.cli.scaffold_convention import _SETTINGS_PY
from cayu.runtime.public_authority import (
    PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID_ENV,
    PUBLIC_AUTHORITY_ALIAS_KEYS_ENV,
)

_REVIEW_MODULE = '''"""Build the merge review when the app has a keyring."""

from cayu.approvals import ToolArgumentReview

from configuration.settings import configured_public_authority_alias_codec


def merge_review() -> ToolArgumentReview | None:
    if configured_public_authority_alias_codec() is None:
        return None
    return ToolArgumentReview({"merge": ("repository",)})
'''


@pytest.fixture
def project(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> Path:
    for name in (
        "CAYU_MEMORY_EVIDENCE_KEY",
        PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID_ENV,
        PUBLIC_AUTHORITY_ALIAS_KEYS_ENV,
        "CAYU_DATABASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    root = tmp_path / "project"
    monkeypatch.chdir(root)
    return root


def _check(capsys: pytest.CaptureFixture[str], *arguments: str) -> dict:
    main(["check", "--json", *arguments])
    return json.loads(capsys.readouterr().out)


def _codes(report: dict) -> list[str]:
    return [item["code"] for item in report["diagnostics"]]


def test_check_loads_the_app_without_a_memory_evidence_key(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A CI checkout: the git-ignored key file isn't there and nothing sets the variable.
    (project / "data" / "memory-evidence.key").unlink()

    report = _check(capsys)

    assert report["manifest_fingerprint"] != "unavailable"
    (finding,) = [
        item for item in report["diagnostics"] if item["code"] == "MEMORY_EVIDENCE_KEY_UNSET"
    ]
    assert finding["severity"] == "info"
    assert "in CI, set CAYU_MEMORY_EVIDENCE_KEY" in finding["hint"]
    assert "conftest.py" in finding["hint"]
    assert "Cayu Cloud" not in finding["hint"]
    # The throwaway key was only lent while the factory ran.
    assert "CAYU_MEMORY_EVIDENCE_KEY" not in os.environ
    assert not (project / "data" / "memory-evidence.key").exists()

    deploy = _check(capsys, "--deploy")
    (finding,) = [
        item for item in deploy["diagnostics"] if item["code"] == "MEMORY_EVIDENCE_KEY_UNSET"
    ]
    assert "Cayu Cloud sets CAYU_MEMORY_EVIDENCE_KEY" in finding["hint"]


def test_check_lends_no_key_when_the_project_has_one(
    project: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert "MEMORY_EVIDENCE_KEY_UNSET" not in _codes(_check(capsys))

    (project / "data" / "memory-evidence.key").unlink()
    monkeypatch.setenv("CAYU_MEMORY_EVIDENCE_KEY", "k" * 48)
    assert "MEMORY_EVIDENCE_KEY_UNSET" not in _codes(_check(capsys))
    assert os.environ["CAYU_MEMORY_EVIDENCE_KEY"] == "k" * 48


def test_check_says_a_review_has_no_keyring_in_this_run(
    project: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert "HUMAN_REVIEW_KEYRING_MISSING" not in _codes(_check(capsys))

    (project / "workflows" / "review.py").write_text(_REVIEW_MODULE, encoding="utf-8")
    report = _check(capsys)
    (finding,) = [
        item for item in report["diagnostics"] if item["code"] == "HUMAN_REVIEW_KEYRING_MISSING"
    ]
    assert finding["severity"] == "info"
    assert finding["path"] == "workflows/review.py"
    assert "no_review_policy" in finding["message"]
    assert "cayu keys generate --env-file .env" in finding["hint"]

    # With a keyring the app has one to derive the review key from: no finding.
    assert main(["keys", "generate", "--env-file", str(project / ".env")]) == 0
    capsys.readouterr()
    for line in (project / ".env").read_text(encoding="utf-8").splitlines():
        name, _, value = line.partition("=")
        monkeypatch.setenv(name, value.strip("'"))
    assert "HUMAN_REVIEW_KEYRING_MISSING" not in _codes(_check(capsys))


def test_the_generated_memory_key_error_says_what_to_do_in_ci() -> None:
    assert "a fresh checkout such as a CI job has none" in _SETTINGS_PY
    assert "token_urlsafe(48)" in _SETTINGS_PY


class _PurposelessReview(HumanReviewPolicy):
    version = "purposeless-v1"

    @property
    def binding_key(self) -> bytes:
        return b"k" * 32

    def authorize(self, context, *, session_id, session_metadata, action) -> bool:
        return True

    def project(self, context, source) -> HumanReviewDisclosure:
        return HumanReviewDisclosure(status="unavailable")


def test_check_warns_when_the_review_policy_names_no_purpose(tmp_path: Path) -> None:
    def codes(app: CayuApp) -> list[str]:
        findings = _key_findings(tmp_path, app, lent_memory_key=False, deploy=False)
        return [item.code for item in findings if item.code == "HUMAN_REVIEW_PURPOSE_MISSING"]

    [finding] = [
        item
        for item in _key_findings(
            tmp_path,
            CayuApp(human_review_policy=_PurposelessReview()),
            lent_memory_key=False,
            deploy=False,
        )
        if item.code == "HUMAN_REVIEW_PURPOSE_MISSING"
    ]
    assert finding.severity.value == "warning"
    assert "/api/app" in finding.message
    assert finding.documentation_anchor == "cayu guide diagnostics#human-review-purpose-missing"
    assert codes(CayuApp()) == []
    named = ToolArgumentReview({"merge": ("repository",)}, binding_key=b"k" * 32, purpose="merges")
    assert codes(CayuApp(human_review_policy=named)) == []
