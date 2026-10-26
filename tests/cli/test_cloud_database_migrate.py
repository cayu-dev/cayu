"""`[database] migrate` in cayu-cloud.toml: the application's own migration command."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from cayu.cli import _cloud_project as cloud_project
from cayu.cli import cloud as cloud_cli
from cayu.cli._cloud_api import CloudApiError
from cayu.cli._cloud_evidence import EvidenceRecorder
from cayu.cli._cloud_project import CloudDatabase, CloudProjectManifest

_BASE = """
schema_version = 2
application = "orders-agent"
name = "Orders Agent"
version = "1.0.0"
entrypoint = "cayu serve --host 0.0.0.0 --port 8000"
capabilities = ["model.generate"]
cpu_millis = 1000
memory_mb = 2048
timeout_seconds = 900
environment = "python"
compatibility = "cayu>=0.1,<1"
policy_version = "cayu-egress-v1"

[web]
command = "cayu serve --host 0.0.0.0 --port 8000"
port = 8000
"""
_MIGRATE = '\n[database]\nmigrate = "alembic upgrade head"\n'


def _refused(content: str) -> str:
    with pytest.raises(CloudApiError) as raised:
        CloudProjectManifest.loads(content)
    assert raised.value.category == "manifest_invalid"
    return str(raised.value)


def _payload(manifest: CloudProjectManifest) -> dict[str, Any]:
    return manifest.deployment_payload(
        repository="https://github.com/example/orders-agent", revision="a" * 40
    )


def test_a_declared_migration_is_read_and_sent_with_the_deployment() -> None:
    manifest = CloudProjectManifest.loads(_BASE + _MIGRATE)

    assert manifest.database == CloudDatabase(migrate="alembic upgrade head")
    assert _payload(manifest)["manifest"]["database"] == {"migrate": "alembic upgrade head"}


def test_without_the_declaration_the_deployment_request_is_unchanged() -> None:
    manifest = CloudProjectManifest.loads(_BASE)

    assert manifest.database is None
    assert "database" not in _payload(manifest)["manifest"]


@pytest.mark.parametrize(
    "table",
    [
        "\n[database]\n",
        '\n[database]\nmigrate = ""\n',
        "\n[database]\nmigrate = 1\n",
        '\n[database]\nmigrate = ["alembic", "upgrade", "head"]\n',
        f'\n[database]\nmigrate = "{"x" * 2049}"\n',
        '\n[database]\nmigrate = "alembic\\u0000upgrade"\n',
    ],
)
def test_the_migration_must_be_a_command(table: str) -> None:
    assert "[database] migrate" in _refused(_BASE + table)


def test_database_refuses_unknown_fields_like_other_tables() -> None:
    message = _refused(_BASE + _MIGRATE + 'url = "postgresql://example"\n')

    assert message == "database contains unsupported fields: url."


def test_database_must_be_a_table() -> None:
    _refused(_BASE.replace("\n[web]", 'database = "alembic upgrade head"\n\n[web]'))


def test_database_requires_schema_version_2() -> None:
    content = _BASE.replace("schema_version = 2", "schema_version = 1").split("\n[web]")[0]

    assert _refused(content + _MIGRATE) == "[database] requires cayu-cloud.toml schema_version 2."


def _write_pyproject(root: Path, capabilities: list[str]) -> None:
    (root / "pyproject.toml").write_text(
        '[project]\nname = "orders-agent"\nversion = "1.0.0"\n\n'
        '[project.scripts]\norders-agent = "orders_agent.cli:main"\n\n'
        f"[tool.cayu.scaffold]\ncapabilities = {capabilities!r}\n"
    )


def test_init_declares_the_migration_for_an_app_database_project(tmp_path: Path) -> None:
    _write_pyproject(tmp_path, ["app-database", "memory"])

    initialized = cloud_project.initialize_project(tmp_path)

    content = (tmp_path / "cayu-cloud.toml").read_text()
    assert content.endswith(_MIGRATE)
    assert CloudProjectManifest.loads(content).database == CloudDatabase(
        migrate="alembic upgrade head"
    )
    assert initialized.database_migrate == "alembic upgrade head"


@pytest.mark.parametrize("capabilities", [[], ["memory"]])
def test_init_declares_no_migration_without_the_capability(
    tmp_path: Path, capabilities: list[str]
) -> None:
    _write_pyproject(tmp_path, capabilities)

    initialized = cloud_project.initialize_project(tmp_path)

    assert "[database]" not in (tmp_path / "cayu-cloud.toml").read_text()
    assert initialized.database_migrate is None


class RejectingCloud:
    """Cloud that rejects the deployment create with HTTP 422 naming ``fields``."""

    def __init__(self, fields: tuple[tuple[str, str], ...]) -> None:
        self.fields = fields

    def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        if path == "/v1/applications":
            return {
                "items": [
                    {"current_deployment_id": None, "id": "orders-agent", "name": "Orders Agent"}
                ]
            }
        if method == "POST" and path == "/v1/applications/orders-agent/deployments":
            raise CloudApiError(
                "api_request_rejected",
                "Cayu Cloud API returned HTTP 422.",
                status_code=422,
                rejected_fields=self.fields,
            )
        raise AssertionError(f"unexpected Cloud request: {method} {path}")


def _deploy(client: RejectingCloud, tmp_path: Path, manifest: str) -> dict[str, Any]:
    arguments = cloud_cli._build_parser().parse_args(["deploy", "."])
    cloud_cli._preflight_deploy_wait(arguments)
    return cloud_cli._deploy(
        arguments,
        client=client,  # type: ignore[arg-type]
        recorder=EvidenceRecorder(tmp_path / "evidence"),
        project=cloud_project.ResolvedCloudProject(
            root=None,
            manifest_path=tmp_path / "cayu-cloud.toml",
            manifest=CloudProjectManifest.loads(manifest),
            repository="https://github.com/example/orders-agent",
            revision="c" * 40,
        ),
        sleep=lambda _: None,
        monotonic=lambda: 0.0,
    )


def test_a_cloud_that_cannot_run_the_migration_is_named_as_the_reason(tmp_path: Path) -> None:
    unsupported = RejectingCloud((("body.manifest.database", "extra_forbidden"),))

    with pytest.raises(cloud_cli.CloudCommandError) as raised:
        _deploy(unsupported, tmp_path, _BASE + _MIGRATE)

    assert raised.value.category == "database_migration_unsupported"
    assert "nothing was deployed" in str(raised.value)
    assert "[database]" in str(raised.value)

    # Any other rejection is reported as it was.
    other = RejectingCloud((("body.manifest.entrypoint", "string_too_short"),))
    with pytest.raises(CloudApiError) as rejected:
        _deploy(other, tmp_path, _BASE + _MIGRATE)
    assert rejected.value.status_code == 422
