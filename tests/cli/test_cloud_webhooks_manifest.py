"""`[webhooks]` in cayu-cloud.toml: path prefixes Cayu Cloud keeps requests for while the Agent
sleeps."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from cayu.cli import _cloud_project as cloud_project
from cayu.cli._cloud_api import CloudApiError
from cayu.cli._cloud_project import CloudProjectManifest, CloudWebhooks

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
_PATHS_ERROR = "[webhooks] paths must be an array of 1 to 20 different path prefixes."
_PATH_ERROR = "Each [webhooks] paths entry must be a path on the Agent's own host"


def _webhooks(paths: str) -> str:
    return f"\n[webhooks]\npaths = {paths}\n"


def _load(tables: str) -> CloudProjectManifest:
    return CloudProjectManifest.loads(_BASE + _WEB + tables)


def _refused(tables: str, *, schema: int = 2) -> str:
    content = _BASE.replace("schema_version = 2", f"schema_version = {schema}")
    with pytest.raises(CloudApiError) as raised:
        CloudProjectManifest.loads(content + (_WEB if schema == 2 else "") + tables)
    assert raised.value.category == "manifest_invalid"
    return str(raised.value)


def test_an_agent_without_webhooks_declares_none() -> None:
    assert _load("").webhooks is None


def test_webhooks_carry_their_prefixes() -> None:
    manifest = _load(_webhooks('["/webhooks/", "/github", "/a/./b", "/%7Euser"]'))

    assert manifest.webhooks == CloudWebhooks(paths=("/webhooks/", "/github", "/a/./b", "/%7Euser"))
    manifest.validate()


def test_webhooks_accept_twenty_prefixes() -> None:
    paths = ", ".join(f'"/hooks/{index}/"' for index in range(20))

    assert len(_load(_webhooks(f"[{paths}]")).webhooks.paths) == 20  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "paths",
    [
        "[]",
        '"/webhooks/"',
        '["/webhooks/", "/webhooks/"]',
        "[" + ", ".join(f'"/hooks/{index}/"' for index in range(21)) + "]",
    ],
    ids=["empty", "string", "duplicate", "twenty-one"],
)
def test_webhooks_need_one_to_twenty_different_prefixes(paths: str) -> None:
    assert _refused(_webhooks(paths)) == _PATHS_ERROR


def test_webhooks_need_paths() -> None:
    assert _refused("\n[webhooks]\n") == _PATHS_ERROR


@pytest.mark.parametrize(
    "path",
    [
        "",
        "webhooks",
        "https://example.com/webhooks",
        "//example.com/webhooks",
        "/webhooks?x=1",
        "/webhooks#top",
        "/a/../b",
        "/a/%2e%2e/b",
        "/%5Cexample.com",
        "/web hooks",
        "/wébhooks",
        "/" + "a" * 1024,
    ],
)
def test_webhook_paths_refuse_anything_but_a_same_host_path(path: str) -> None:
    assert _refused(_webhooks(f'["{path}"]')).startswith(_PATH_ERROR)


def test_webhook_paths_must_be_strings() -> None:
    assert _refused(_webhooks("[1]")).startswith(_PATH_ERROR)


def test_webhooks_refuse_unknown_fields_like_other_tables() -> None:
    message = _refused('\n[webhooks]\npaths = ["/webhooks/"]\npath = "/github"\n')

    assert message == "webhooks contains unsupported fields: path."


def test_webhooks_must_be_a_table() -> None:
    with pytest.raises(CloudApiError) as raised:
        CloudProjectManifest.loads(_BASE + 'webhooks = "/webhooks/"\n' + _WEB)

    assert str(raised.value) == (
        "cayu-cloud.toml is missing a required field or contains the wrong type."
    )


def test_webhooks_require_schema_version_2() -> None:
    message = _refused(_webhooks('["/webhooks/"]'), schema=1)

    assert message == "[webhooks] requires cayu-cloud.toml schema_version 2."


def test_webhooks_need_not_match_public_paths_or_cloud_sign_in() -> None:
    manifest = _load('\n[sign_in]\nmode = "own"\n' + _webhooks('["/webhooks/"]'))

    assert manifest.webhooks == CloudWebhooks(paths=("/webhooks/",))
    assert manifest.sign_in.public_paths == ()


def test_webhooks_are_not_sent_in_the_deployment_request() -> None:
    def payload(manifest: CloudProjectManifest) -> dict[str, object]:
        return manifest.deployment_payload(
            repository="https://github.com/example/agent", revision="a" * 40
        )

    assert payload(_load(_webhooks('["/webhooks/"]'))) == payload(_load(""))


def _write_source(root: Path, manifest: str, *, name: str = "cayu-cloud.toml") -> None:
    (root / "pyproject.toml").write_text('[project]\nname = "closing-agent"\nversion = "1.0"\n')
    (root / "uv.lock").write_text("version = 1\n")
    (root / name).parent.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(manifest)


def test_root_manifest_carries_webhooks_in_the_upload(tmp_path: Path) -> None:
    _write_source(tmp_path, _BASE + _WEB + _webhooks('["/webhooks/"]'))

    project = cloud_project.resolve_project(str(tmp_path), manifest_path=None, revision=None)

    assert project.manifest.webhooks == CloudWebhooks(paths=("/webhooks/",))
    assert (
        cloud_project._uploaded_root_manifest(project.bundle or b"")
        == (tmp_path / "cayu-cloud.toml").read_text()
    )


def test_custom_manifest_path_refuses_webhooks(tmp_path: Path) -> None:
    _write_source(tmp_path, _BASE + _WEB + _webhooks('["/webhooks/"]'), name="deploy/own.toml")

    with pytest.raises(CloudApiError, match=re.escape("cayu-cloud.toml at the deployment root")):
        cloud_project.resolve_project(
            str(tmp_path), manifest_path=Path("deploy/own.toml"), revision=None
        )


def test_custom_manifest_path_refuses_a_root_manifest_that_declares_webhooks(
    tmp_path: Path,
) -> None:
    # Cloud reads webhooks only from the uploaded root file, so deploying the custom
    # manifest would silently keep the root file's webhooks.
    _write_source(tmp_path, _BASE + _WEB + _webhooks('["/webhooks/"]'))
    _write_source(tmp_path, _BASE + _WEB, name="deploy/own.toml")

    with pytest.raises(CloudApiError) as raised:
        cloud_project.resolve_project(
            str(tmp_path), manifest_path=Path("deploy/own.toml"), revision=None
        )
    assert raised.value.category == "manifest_invalid"
    assert "[webhooks]" in str(raised.value)
    assert "deploy/own.toml" in str(raised.value)


def test_init_writes_no_webhooks(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "mail-agent"\nversion = "0.4.0"\n\n'
        '[project.scripts]\nmail-agent = "mail_agent.cli:main"\n'
    )

    cloud_project.initialize_project(tmp_path)

    assert "webhooks" not in (tmp_path / "cayu-cloud.toml").read_text()
