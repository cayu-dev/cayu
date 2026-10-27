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


@pytest.mark.parametrize("value", ["0", "59", "86401"])
def test_an_idle_timeout_out_of_range_names_the_range_and_the_worker(value):
    web = f'[web]\ncommand = "web"\nport = 8000\nidle_timeout_seconds = {value}\n'
    with pytest.raises(CloudApiError) as raised:
        CloudProjectManifest.loads(_BASE + web)
    message = str(raised.value)
    assert "from 60 to 86400" in message
    assert "can't turn sleep off" in message
    assert "[worker]" in message


def _schedules(count: int) -> str:
    return "".join(
        f'[[schedules]]\nname = "tick-{index}"\ncommand = "tick"\nexpression = "rate(1 hour)"\n'
        for index in range(count)
    )


def test_an_agent_declares_as_many_schedules_as_cayu_cloud_accepts():
    # Cayu Cloud accepts 100 schedules per Agent (cayu-cloud MAX_SCHEDULES).
    manifest = CloudProjectManifest.loads(_BASE + _schedules(100))
    payload = manifest.deployment_payload(
        repository="https://github.com/example/agent", revision="a" * 40
    )
    assert len(payload["manifest"]["runtime"]["schedules"]) == 100
    with pytest.raises(CloudApiError) as raised:
        CloudProjectManifest.loads(_BASE + _schedules(101))
    assert raised.value.category == "manifest_invalid"
    assert str(raised.value) == "A manifest declares at most 100 schedules."


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


_WORKER = '[worker]\ncommand = "cayu worker scout"\n'


@pytest.mark.parametrize("seconds", [10, 120])
def test_worker_stop_grace_is_forwarded_in_the_runtime_payload(seconds):
    manifest = CloudProjectManifest.loads(_BASE + _WORKER + f"stop_grace_seconds = {seconds}\n")
    assert manifest.worker is not None and manifest.worker.stop_grace_seconds == seconds
    runtime = manifest.deployment_payload(
        repository="https://github.com/example/agent", revision="a" * 40
    )["manifest"]["runtime"]
    assert runtime["worker"] == {"command": "cayu worker scout", "stop_grace_seconds": seconds}


def test_an_undeclared_worker_stop_grace_is_not_sent():
    manifest = CloudProjectManifest.loads(_BASE + _WORKER)
    assert manifest.worker is not None and manifest.worker.stop_grace_seconds is None
    assert manifest.runtime_payload()["worker"] == {"command": "cayu worker scout"}


@pytest.mark.parametrize("value", ["9", "121", "0", "30.0", '"30"', "true"])
def test_worker_stop_grace_outside_cloud_bounds_is_refused(value):
    with pytest.raises(CloudApiError, match="stop_grace_seconds must be a whole number") as raised:
        CloudProjectManifest.loads(_BASE + _WORKER + f"stop_grace_seconds = {value}\n")
    assert raised.value.category == "manifest_invalid"


@pytest.mark.parametrize(
    "table",
    [
        '[web]\ncommand = "web"\nport = 8000\n',
        '[[schedules]]\nname = "tick"\ncommand = "tick"\nexpression = "rate(1 hour)"\n',
    ],
)
def test_only_the_worker_takes_a_stop_grace(table):
    with pytest.raises(CloudApiError, match="unsupported fields: stop_grace_seconds"):
        CloudProjectManifest.loads(_BASE + table + "stop_grace_seconds = 60\n")
