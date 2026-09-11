"""Host Git authority parsing through the emitted consumer, without remote writes."""

import importlib
import json
import traceback
import warnings

import pytest

from cayu import remote_git_broker_behavior_fingerprint
from cayu.cli.project import project_context
from tests.qualification.test_repository_maintenance_application import project as project


def authority():
    return {
        "repository": {
            "repository_id": "target",
            "broker_repository_id": "delivery",
            "remote_alias": "origin",
            "remote_identity": "disposable-github",
            "base_ref": "refs/heads/main",
            "expected_base_commit": "a" * 40,
            "destination_ref": "refs/heads/cayu/fix",
        },
        "commit": {
            "author_name": "Maintainer",
            "author_email": "maintainer@example.invalid",
            "committer_name": "Maintainer",
            "committer_email": "maintainer@example.invalid",
            "authored_at": "2026-09-10T00:00:00+00:00",
            "title": "Repair inclusive endpoint",
        },
        "security": {
            "broker_behavior_fingerprint": remote_git_broker_behavior_fingerprint(),
            "credential_profile_id": "maintenance-git",
            "egress_profile_id": "github-only",
            "policy_fingerprint": "sha256:" + "b" * 64,
            "approval_policy_fingerprint": "sha256:" + "c" * 64,
            "redaction_profile_fingerprint": "sha256:" + "d" * 64,
        },
        "limits": {"timeout_seconds": 120, "max_paths": 3, "max_file_bytes": 16384},
    }


def test_emitted_git_authority_is_explicit_and_fresh(project, monkeypatch):
    value = authority()
    monkeypatch.setenv("CAYU_MAINTENANCE_GIT_JSON", json.dumps(value))
    with project_context(project):
        module = importlib.import_module("configuration.maintenance")
        first = module.configured_maintenance_git_authority()
        second = module.configured_maintenance_git_authority()
        assert first == second
        assert all(left is not right for left, right in zip(first, second, strict=True))
        repository, commit, security, limits = first
        assert repository.expected_destination_commit is None
        assert commit.authored_at == value["commit"]["authored_at"]
        assert security.credential_profile_id == "maintenance-git"
        assert limits.timeout_seconds == 120
        assert limits.max_paths == 3
        value["commit"]["authored_at"] = "2026-09-11T00:00:00+00:00"
        monkeypatch.setenv("CAYU_MAINTENANCE_GIT_JSON", json.dumps(value))
        changed = module.configured_maintenance_git_authority()
        assert changed[1] != commit
        assert first == second


@pytest.mark.parametrize(
    "invalid",
    ["missing", "duplicate", "oversize", "extra", "section", "timestamp", "bool", "nested", "nan"],
)
def test_git_configuration_rejects_without_diagnostic_disclosure(
    project, monkeypatch, caplog, capsys, invalid
):
    value = authority()
    canary = "private-delivery-configuration-canary"
    value["commit"]["body"] = canary
    if invalid == "extra":
        value["approval"] = canary
    elif invalid == "section":
        value["repository"] = canary
    elif invalid == "timestamp":
        del value["commit"]["authored_at"]
    elif invalid == "bool":
        value["limits"]["timeout_seconds"] = True
    elif invalid == "nested":
        value["repository"]["token"] = canary
    raw = json.dumps(value)
    if invalid == "duplicate":
        raw = '{"repository":"' + canary + '","repository":{}}'
    elif invalid == "oversize":
        raw = canary * 3000
    elif invalid == "nan":
        raw = '{"repository":NaN,"commit":"' + canary + '"}'
    if invalid == "missing":
        monkeypatch.delenv("CAYU_MAINTENANCE_GIT_JSON", raising=False)
    else:
        monkeypatch.setenv("CAYU_MAINTENANCE_GIT_JSON", raw)
    with project_context(project), warnings.catch_warnings(record=True) as recorded:
        module = importlib.import_module("configuration.maintenance")
        with pytest.raises(ValueError, match="maintenance Git configuration") as caught:
            module.configured_maintenance_git_authority()
    assert canary not in "".join(traceback.format_exception(caught.value))
    assert not recorded
    output = capsys.readouterr()
    assert canary not in caplog.text + output.out + output.err
