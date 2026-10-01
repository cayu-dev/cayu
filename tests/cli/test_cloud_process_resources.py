from __future__ import annotations

import pytest

from cayu.cli._cloud_api import CloudApiError
from cayu.cli._cloud_project import CloudProjectManifest

_BASE = """
schema_version = 2
application = "sized-agent"
name = "Sized Agent"
version = "1.0.0"
entrypoint = "python -m agent"
capabilities = ["model.generate"]
cpu_millis = 4000
memory_mb = 8192
timeout_seconds = 900
environment = "python"
compatibility = "cayu>=0.1"
policy_version = "v1"
"""


def test_cloud_process_resources_are_optional_and_preserved_in_deployment_payload():
    manifest = CloudProjectManifest.loads(
        _BASE
        + """
[web]
command = "web"
port = 8000
cpu_millis = 4000
memory_mb = 8192
[worker]
command = "worker"
cpu_millis = 250
memory_mb = 512
[[schedules]]
name = "small"
command = "tick"
expression = "rate(1 hour)"
cpu_millis = 250
memory_mb = 512
[[schedules]]
name = "default"
command = "tick"
expression = "rate(1 hour)"
"""
    )
    payload = manifest.deployment_payload(
        repository="https://github.com/example/agent", revision="a" * 40
    )
    assert payload["manifest"]["resources"] == {"cpu_millis": 4000, "memory_mb": 8192}
    runtime = payload["manifest"]["runtime"]
    assert runtime["web"]["cpu_millis"] == 4000
    assert runtime["worker"]["memory_mb"] == 512
    assert runtime["schedules"][0]["cpu_millis"] == 250
    assert "cpu_millis" not in runtime["schedules"][1]
    assert "memory_mb" not in runtime["schedules"][1]


@pytest.mark.parametrize("process", ["web", "worker", "schedules"])
@pytest.mark.parametrize(
    "resources", ["cpu_millis = true", "cpu_millis = 16001", "memory_mb = 127"]
)
def test_cloud_process_resources_reject_invalid_values(process, resources):
    section = {
        "web": '[web]\ncommand = "web"\nport = 8000\n',
        "worker": '[worker]\ncommand = "worker"\n',
        "schedules": '[[schedules]]\nname = "tick"\ncommand = "tick"\nexpression = "rate(1 hour)"\n',
    }[process]
    with pytest.raises(CloudApiError) as raised:
        CloudProjectManifest.loads(_BASE + section + resources)
    assert raised.value.category == "manifest_invalid"


@pytest.mark.parametrize(
    "table",
    [
        '[web]\ncommand = "web"\nport = 8000\n',
        '[worker]\ncommand = "worker"\n',
        '[[schedules]]\nname = "tick"\ncommand = "tick"\nexpression = "rate(1 hour)"\n',
    ],
)
def test_unknown_process_keys_are_actionable_errors(table):
    with pytest.raises(CloudApiError, match="cpu_milis"):
        CloudProjectManifest.loads(_BASE + table + "cpu_milis = 250\n")


@pytest.mark.parametrize("value", ['"4000"', "true", "4000.0"])
@pytest.mark.parametrize("field", ["cpu_millis", "memory_mb"])
def test_top_level_resources_require_integers_like_process_overrides(field, value):
    old = f"{field} = " + ("4000" if field == "cpu_millis" else "8192")
    with pytest.raises(CloudApiError):
        CloudProjectManifest.loads(_BASE.replace(old, f"{field} = {value}"))


def _rejected_deployment_message(monkeypatch, detail):
    import httpx

    from cayu.cli._cloud_api import CloudApiClient

    real_client = httpx.Client
    transport = httpx.MockTransport(lambda request: httpx.Response(422, json={"detail": detail}))
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: real_client(transport=transport, **kwargs)
    )
    with pytest.raises(CloudApiError) as raised:
        CloudApiClient(api_url="https://cloud.example", api_key="test").request(
            "POST", "/deployments"
        )
    return str(raised.value)


def test_cloud_api_renders_integer_resource_suggestions_without_arbitrary_error_text(monkeypatch):
    message = _rejected_deployment_message(
        monkeypatch,
        {
            "code": "manifest_invalid",
            "message": "private-provider-error-canary",
            "valid_pairs": [
                {"cpu_millis": 4000, "memory_mb": 8192},
                {"cpu_millis": 2000, "memory_mb": 4096},
            ],
        },
    )
    assert "cpu_millis = 4000, memory_mb = 8192" in message
    assert "private-provider-error-canary" not in message


def test_cloud_api_renders_the_cloud_ceiling_rejection_body(monkeypatch):
    # The 422 detail Cayu Cloud returns for cpu_millis = 4096, memory_mb = 8192
    # under its default 4 vCPU / 16 GiB ceiling.
    message = _rejected_deployment_message(
        monkeypatch,
        {
            "code": "manifest_invalid",
            "message": (
                "Agent resources require 8192 CPU units / 16384 MiB, above the environment "
                "ceiling of 4096 CPU units / 16384 MiB. Nearest valid manifest pairs: "
                "cpu_millis = 4000, memory_mb = 8192; cpu_millis = 4000, memory_mb = 9216; "
                "cpu_millis = 4000, memory_mb = 10240."
            ),
            "valid_pairs": [
                {"cpu_millis": 4000, "memory_mb": 8192},
                {"cpu_millis": 4000, "memory_mb": 9216},
                {"cpu_millis": 4000, "memory_mb": 10240},
            ],
        },
    )
    assert message == (
        "Cayu Cloud API returned HTTP 422: Agent resources exceed supported sizes. "
        "Valid manifest pairs: cpu_millis = 4000, memory_mb = 8192; "
        "cpu_millis = 4000, memory_mb = 9216; cpu_millis = 4000, memory_mb = 10240."
    )


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"valid_pairs": []},
        {"valid_pairs": [{"cpu_millis": 4096, "memory_mb": 8192}]},
        {"valid_pairs": [{"cpu_millis": 4000, "memory_mb": 8192, "note": "canary"}]},
    ],
)
def test_cloud_resource_rejection_without_usable_pairs_has_a_static_message(monkeypatch, extra):
    message = _rejected_deployment_message(
        monkeypatch,
        {"code": "manifest_invalid", "message": "private-provider-error-canary", **extra},
    )
    assert message == (
        "Cayu Cloud API returned HTTP 422: Cayu Cloud rejected the manifest resources."
    )


@pytest.mark.parametrize("value", ['"900"', "900.9", "true"])
def test_timeout_seconds_requires_an_integer(value):
    with pytest.raises(CloudApiError, match="timeout_seconds"):
        CloudProjectManifest.loads(
            _BASE.replace("timeout_seconds = 900", f"timeout_seconds = {value}")
        )


@pytest.mark.parametrize("value", ['"8000"', "8000.9", "true"])
@pytest.mark.parametrize("field", ["port", "idle_timeout_seconds"])
def test_web_port_and_idle_timeout_require_integers(field, value):
    web = '[web]\ncommand = "web"\n'
    web += f"{field} = {value}\n" + ("port = 8000\n" if field != "port" else "")
    with pytest.raises(CloudApiError, match=field):
        CloudProjectManifest.loads(_BASE + web)


@pytest.mark.parametrize(
    ("schedules", "label"),
    [
        (
            '[[schedules]]\nname = "tick"\ncommand = "tick"\nexpression = "rate(1 hour)"\n'
            '[[schedules]]\nname = "nightly"\ncommand = "report"\n'
            'expression = "rate(1 day)"\ncpu_milis = 250\n',
            'schedule "nightly"',
        ),
        (
            '[[schedules]]\nname = "tick"\ncommand = "tick"\nexpression = "rate(1 hour)"\n'
            '[[schedules]]\ncommand = "report"\nexpression = "rate(1 day)"\ncpu_milis = 250\n',
            "schedules[1]",
        ),
    ],
)
def test_unknown_schedule_keys_name_the_schedule(schedules, label):
    with pytest.raises(CloudApiError) as raised:
        CloudProjectManifest.loads(_BASE + schedules)
    assert str(raised.value) == f"{label} contains unsupported fields: cpu_milis."
