"""`[operator_check]` in cayu-cloud.toml: the one route Cayu Cloud calls with the Agent's operator
login for `cayu cloud service check`."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from cayu.cli import _cloud_project as cloud_project
from cayu.cli._cloud_api import CloudApiError
from cayu.cli._cloud_project import CloudOperatorCheck, CloudProjectManifest

_BASE = """
schema_version = 2
application = "closing-agent"
name = "Closing Agent"
version = "1.0.0"
entrypoint = "cayu serve --host 0.0.0.0 --port 8000"
capabilities = ["model.generate"]
cpu_millis = 1000
memory_mb = 2048
timeout_seconds = 900
environment = "python"
compatibility = "cayu>=0.1,<1"
policy_version = "cayu-egress-v1"
"""
_WEB = '\n[web]\ncommand = "cayu serve --host 0.0.0.0 --port 8000"\nport = 8000\n'


def _check(table: str) -> str:
    return f"\n[operator_check]\n{table}"


def _load(tables: str) -> CloudProjectManifest:
    return CloudProjectManifest.loads(_BASE + _WEB + tables)


def _refused(tables: str, *, schema: int = 2) -> str:
    content = _BASE.replace("schema_version = 2", f"schema_version = {schema}")
    with pytest.raises(CloudApiError) as raised:
        CloudProjectManifest.loads(content + (_WEB if schema == 2 else "") + tables)
    assert raised.value.category == "manifest_invalid"
    return str(raised.value)


def test_an_agent_without_an_operator_check_declares_none() -> None:
    assert _load("").operator_check is None


def test_a_check_takes_post_and_thirty_seconds_unless_it_says() -> None:
    assert _load(_check('path = "/operator-check"\n')).operator_check == CloudOperatorCheck(
        path="/operator-check", method="POST", timeout_seconds=30
    )
    declared = _load(_check('path = "/ops/check"\nmethod = "GET"\ntimeout_seconds = 45\n'))
    assert declared.operator_check == CloudOperatorCheck(
        path="/ops/check", method="GET", timeout_seconds=45
    )
    declared.validate()


@pytest.mark.parametrize(
    ("table", "message"),
    [
        ('method = "POST"\n', "[operator_check] path must be a path on the Agent's own host"),
        ('path = "https://x.test/check"\n', "[operator_check] path must be a path"),
        ('path = "/a/../check"\n', "[operator_check] path must be a path"),
        ('path = "/.cayu/check"\n', '[operator_check] path can\'t be under "/.cayu/"'),
        ('path = "/%2Ecayu/sign-out"\n', '[operator_check] path can\'t be under "/.cayu/"'),
        ('path = "/check"\nmethod = "PUT"\n', '[operator_check] method must be "POST" or "GET".'),
        ('path = "/check"\nmethod = "post"\n', "[operator_check] method must be"),
        ('path = "/check"\ntimeout_seconds = 0\n', "[operator_check] timeout_seconds must be"),
        ('path = "/check"\ntimeout_seconds = 46\n', "from 1 to 45."),
        ('path = "/check"\ntimeout_seconds = 30.0\n', "[operator_check] timeout_seconds"),
        ('path = "/check"\ntimeout_seconds = true\n', "[operator_check] timeout_seconds"),
        ('path = "/check"\nurl = "/x"\n', "operator_check contains unsupported fields: url."),
    ],
)
def test_refuses_what_cayu_cloud_refuses(table: str, message: str) -> None:
    assert message in _refused(_check(table))


def test_requires_schema_version_2() -> None:
    assert _refused(_check('path = "/check"\n'), schema=1) == (
        "[operator_check] requires cayu-cloud.toml schema_version 2."
    )


def test_is_not_sent_in_the_deployment_request() -> None:
    def payload(manifest: CloudProjectManifest) -> dict[str, object]:
        return manifest.deployment_payload(
            repository="https://github.com/example/agent", revision="a" * 40
        )

    assert payload(_load(_check('path = "/check"\n'))) == payload(_load(""))


def _write_source(root: Path, manifest: str, *, name: str = "cayu-cloud.toml") -> None:
    (root / "pyproject.toml").write_text('[project]\nname = "closing-agent"\nversion = "1.0"\n')
    (root / "uv.lock").write_text("version = 1\n")
    (root / name).parent.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(manifest)


def test_root_manifest_carries_the_check_in_the_upload(tmp_path: Path) -> None:
    _write_source(tmp_path, _BASE + _WEB + _check('path = "/operator-check"\n'))

    project = cloud_project.resolve_project(str(tmp_path), manifest_path=None, revision=None)

    assert project.manifest.operator_check == CloudOperatorCheck(path="/operator-check")
    assert (
        cloud_project._uploaded_root_manifest(project.bundle or b"")
        == (tmp_path / "cayu-cloud.toml").read_text()
    )


def test_custom_manifest_path_refuses_a_check(tmp_path: Path) -> None:
    _write_source(tmp_path, _BASE + _WEB + _check('path = "/check"\n'), name="deploy/own.toml")

    with pytest.raises(CloudApiError, match=re.escape("cayu-cloud.toml at the deployment root")):
        cloud_project.resolve_project(
            str(tmp_path), manifest_path=Path("deploy/own.toml"), revision=None
        )


def test_custom_manifest_path_refuses_a_root_manifest_that_declares_a_check(
    tmp_path: Path,
) -> None:
    # Cloud reads the check only from the uploaded root file.
    _write_source(tmp_path, _BASE + _WEB + _check('path = "/check"\n'))
    _write_source(tmp_path, _BASE + _WEB, name="deploy/own.toml")

    with pytest.raises(CloudApiError) as raised:
        cloud_project.resolve_project(
            str(tmp_path), manifest_path=Path("deploy/own.toml"), revision=None
        )
    assert "[operator_check]" in str(raised.value)
    assert "deploy/own.toml" in str(raised.value)


@pytest.mark.parametrize(
    ("paths", "path"),
    [
        ('["/ops/"]', "/ops/check"),
        ('["/"]', "/operator-check"),
        ('["/hooks"]', "/hooks/check"),
        ('["/ops/"]', "/./ops/check"),
        ('["/ops/"]', "/%6Fps/check"),
    ],
)
def test_refuses_a_check_under_a_webhook_prefix(paths: str, path: str) -> None:
    # The router would keep the check, with the operator login, and deliver it later.
    tables = f"\n[webhooks]\npaths = {paths}\n" + _check(f'path = "{path}"\nmethod = "GET"\n')

    assert "can't be under one of the [webhooks] paths" in _refused(tables)


def test_a_check_beside_a_webhook_prefix_is_fine() -> None:
    tables = '\n[webhooks]\npaths = ["/hooks"]\n' + _check('path = "/hooksy"\n')

    assert _load(tables).operator_check == CloudOperatorCheck(path="/hooksy")
