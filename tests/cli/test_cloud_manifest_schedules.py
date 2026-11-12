"""`[[schedules]]` time zones and Cloud's schedule rules, `cayu cloud init --schedule`, what
`init --force` keeps, `cayu cloud init --check` and `cayu cloud schedule list`."""

from __future__ import annotations

import json
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from cayu.cli import _cloud_api, main
from cayu.cli import cloud as cloud_cli
from cayu.cli._cloud_api import CloudApiClient, CloudApiError
from cayu.cli._cloud_project import CloudProjectManifest, _toml_value
from cayu.cli._cloud_schedules import check_schedule_expression

_BASE = """
schema_version = 2
application = "herald-agent"
name = "Herald Agent"
version = "0.1.0"
entrypoint = "python -m herald"
capabilities = ["model.generate"]
cpu_millis = 1000
memory_mb = 2048
timeout_seconds = 900
environment = "python"
compatibility = "cayu>=0.1,<1"
policy_version = "cayu-egress-v1"

[worker]
command = "python -m herald"
"""


def _manifest(schedule: str) -> CloudProjectManifest:
    return CloudProjectManifest.loads(_BASE + "\n[[schedules]]\n" + schedule)


def _schedule(expression: str, extra: str = "") -> str:
    return (
        f'name = "digest"\ncommand = "python -m herald digest"\nexpression = "{expression}"\n'
        + (extra)
    )


def test_a_schedule_with_a_time_zone_is_sent_with_it() -> None:
    manifest = _manifest(_schedule("cron(0 8 * * ? *)", 'timezone = "America/Los_Angeles"\n'))

    assert manifest.schedules[0].timezone == "America/Los_Angeles"
    runtime = manifest.runtime_payload()
    assert runtime is not None
    assert runtime["schedules"] == [
        {
            "command": "python -m herald digest",
            "expression": "cron(0 8 * * ? *)",
            "name": "digest",
            "timezone": "America/Los_Angeles",
        }
    ]


@pytest.mark.parametrize("extra", ["", 'timezone = "UTC"\n', 'timezone = "Etc/UTC"\n'])
def test_a_utc_schedule_is_sent_as_before(extra: str) -> None:
    # Without a zone other than UTC the payload is unchanged, so a Cloud that predates time
    # zones takes it and its manifest digest stays the same.
    runtime = _manifest(_schedule("cron(0 8 * * ? *)", extra)).runtime_payload()

    assert runtime is not None
    assert "timezone" not in runtime["schedules"][0]


@pytest.mark.parametrize(
    ("timezone", "message"),
    [
        ('"Mars/Olympus"', "is not an IANA time zone name"),
        ('"localtime"', "is not an IANA time zone name"),
        ('"../../etc/passwd"', "is not an IANA time zone name"),
        ('""', "must be an IANA time zone name"),
        ("7", "must be an IANA time zone name"),
    ],
)
def test_a_schedule_time_zone_must_be_an_iana_name(timezone: str, message: str) -> None:
    with pytest.raises(CloudApiError, match=message) as raised:
        _manifest(_schedule("cron(0 8 * * ? *)", f"timezone = {timezone}\n"))

    assert raised.value.category == "manifest_invalid"


def test_a_rate_schedule_takes_no_time_zone() -> None:
    with pytest.raises(CloudApiError, match=r"rate\(\) expression counts an interval"):
        _manifest(_schedule("rate(1 hour)", 'timezone = "America/Los_Angeles"\n'))
    # UTC is the default, so naming it is harmless.
    assert _manifest(_schedule("rate(1 hour)", 'timezone = "UTC"\n')).schedules


@pytest.mark.parametrize(
    ("expression", "zone"),
    [
        ("cron(30 1 * * ? *)", "America/Los_Angeles"),
        ("cron(0 2 ? * SUN *)", "America/Los_Angeles"),
        ("cron(0 * * * ? *)", "America/Los_Angeles"),
        ("cron(0 0-3 * * ? *)", "America/Los_Angeles"),
        ("cron(15 2 * * ? *)", "Europe/Paris"),
        # An hours field it can't read counts as every hour.
        ("cron(0 */0 * * ? *)", "America/Los_Angeles"),
        ("cron(0 1/x * * ? *)", "America/Los_Angeles"),
    ],
)
def test_a_zoned_schedule_never_runs_in_an_hour_a_clock_change_skips_or_repeats(
    expression: str, zone: str
) -> None:
    # EventBridge runs a repeated local time once and a CronJob twice; Cloud refuses both.
    with pytest.raises(CloudApiError, match="daylight-saving change skips or repeats"):
        _manifest(_schedule(expression, f'timezone = "{zone}"\n'))


@pytest.mark.parametrize(
    ("expression", "zone"),
    [
        ("cron(0 8 ? * MON-FRI *)", "America/Los_Angeles"),
        ("cron(0 */6 * * ? *)", "America/Los_Angeles"),
        ("cron(0 3,17 * * ? *)", "America/Los_Angeles"),
        ("cron(30 1 * * ? *)", "Asia/Tokyo"),
        ("cron(30 1 * * ? *)", "UTC"),
    ],
)
def test_a_zoned_schedule_outside_clock_changes_is_kept(expression: str, zone: str) -> None:
    assert _manifest(_schedule(expression, f'timezone = "{zone}"\n')).schedules[0].timezone == zone


@pytest.mark.parametrize(
    "expression",
    [
        "rate(1 minute)",
        "rate(15 minutes)",
        "rate(1 hour)",
        "rate(6 hours)",
        "rate(1 day)",
        "cron(0 3 * * ? *)",
        "cron(0 3 ? * MON-FRI *)",
        "cron(0 3 ? * 2-6 ?)",
        "cron(*/5 * ? * SUN *)",
    ],
)
def test_expressions_cloud_runs_on_every_platform_pass(expression: str) -> None:
    check_schedule_expression(expression)


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("cron(0 3 * * * *)", "exactly one of the day-of-month and day-of-week"),
        ("cron(0 3 ? * ? *)", "exactly one of the day-of-month and day-of-week"),
        ("cron(0 3 * * ?)", "six cron fields"),
        ("cron(0 3 * * ? 2027)", "limits the year"),
        ("cron(0 3 L * ? *)", "L, W or #"),
        ("cron(0 3 ? * 2#1 *)", "L, W or #"),
        ("cron(0 3 ? * 1/2 *)", "steps the day of the week"),
        ("cron(0 3 ? * 8 *)", "outside 1-7"),
        ("rate(7 minutes)", "exact interval"),
        ("rate(5 hours)", "exact interval"),
        ("rate(2 days)", "exact interval"),
        ("rate(1 minutes)", "not a valid rate"),
        ("at(2026-11-12T08:00:00)", "can't run on Cayu Cloud"),
    ],
)
def test_the_loader_refuses_what_cloud_refuses(expression: str, message: str) -> None:
    with pytest.raises(CloudApiError, match=message):
        _manifest(_schedule(expression))


@pytest.mark.parametrize(
    "value",
    [
        "plain",
        'a "quoted" \\ value',
        "line\nbreak\ttab",
        "del \x7f and \x00",
        "ünïcödé ✓",
        1,
        -2.5,
        1e300,
        float("inf"),
        True,
        [],
        [1, "two", [3.0]],
        {},
        {"a": 1, "b c": {"d": [True]}, "": "empty key"},
    ],
)
def test_kept_values_are_written_back_unchanged(value: object) -> None:
    assert tomllib.loads(f"value = {_toml_value(value)}\n")["value"] == value


def _project(root: Path, manifest: str | None = None) -> Path:
    (root / "pyproject.toml").write_text(
        '[project]\nname = "herald-agent"\nversion = "0.1.0"\ndependencies = []\n'
    )
    (root / "uv.lock").write_text("version = 1\n")
    if manifest is not None:
        (root / "cayu-cloud.toml").write_text(manifest)
    return root


def _cli(arguments: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, dict[str, Any]]:
    code = main(["cloud", *arguments])
    return code, json.loads(capsys.readouterr().out)


def test_init_writes_schedules_in_a_time_zone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _project(tmp_path)

    code, output = _cli(
        [
            "init",
            str(tmp_path),
            "--schedule",
            "morning=cron(0 8 ? * MON-FRI *)=python -m herald digest --slot morning",
            "--schedule",
            "evening=cron(0 17 ? * MON-FRI *)=python -m herald digest --slot evening",
            "--schedule-timezone",
            "America/Los_Angeles",
        ],
        capsys,
    )

    assert code == 0
    assert output["result"]["schedules"] == ["morning", "evening"]
    written = tomllib.loads((tmp_path / "cayu-cloud.toml").read_text())
    assert written["schedules"] == [
        {
            "name": "morning",
            "expression": "cron(0 8 ? * MON-FRI *)",
            "command": "python -m herald digest --slot morning",
            "timezone": "America/Los_Angeles",
        },
        {
            "name": "evening",
            "expression": "cron(0 17 ? * MON-FRI *)",
            "command": "python -m herald digest --slot evening",
            "timezone": "America/Los_Angeles",
        },
    ]


@pytest.mark.parametrize(
    ("arguments", "category", "message"),
    [
        (["--schedule", "digest"], "invalid_input", "NAME=EXPRESSION=COMMAND"),
        (["--schedule", "digest=rate(1 hour)="], "invalid_input", "NAME=EXPRESSION=COMMAND"),
        (
            ["--schedule", "a=rate(1 hour)=x", "--schedule", "a=rate(1 day)=y"],
            "invalid_input",
            "name of its own",
        ),
        (["--schedule-timezone", "UTC"], "invalid_input", "applies to --schedule"),
        (["--schedule", "digest=rate(7 minutes)=x"], "manifest_invalid", "exact interval"),
        (
            ["--schedule", "digest=rate(1 hour)=x", "--schedule-timezone", "Europe/Paris"],
            "manifest_invalid",
            "takes no time zone",
        ),
    ],
)
def test_init_refuses_a_schedule_the_deploy_would_refuse(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    category: str,
    message: str,
) -> None:
    _project(tmp_path)

    code, output = _cli(["init", str(tmp_path), *arguments], capsys)

    assert code == 2
    assert output["error"]["category"] == category
    assert message in output["error"]["message"]
    assert not (tmp_path / "cayu-cloud.toml").exists()


_HAND_EDITED = (
    _BASE.replace("cpu_millis = 1000", "cpu_millis = 2000").replace(
        'policy_version = "cayu-egress-v1"\n',
        'policy_version = "cayu-egress-v1"\nhome = "/app/"\n',
    )
    + """
[sign_in]
mode = "cloud"

[env]
LOG_LEVEL = "debug"
"ODD.NAME" = "a \\"quoted\\" value"

[storage]
local_files = [{ path = "/data/cache", reason = "Rebuilt from GitHub on start." }]

[[schedules]]
name = "digest"
command = "python -m herald digest"
expression = "cron(0 16 * * ? *)"

[[schedules]]
name = "weekly"
command = "python -m herald weekly"
expression = "cron(0 9 ? * MON *)"
timezone = "Europe/Paris"
input = { scope = "all", repos = ["cayu", "cayu-cloud"], depth = 2, nested = { on = true } }
"""
)


def test_init_force_keeps_what_it_does_not_generate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _project(tmp_path, _HAND_EDITED)
    before = tomllib.loads(_HAND_EDITED)

    code, output = _cli(["init", str(tmp_path), "--force"], capsys)

    assert code == 0
    result = output["result"]
    assert result["kept"] == ["home", "[env]", "[storage]", "[[schedules]]"]
    assert result["replaced"] == ["entrypoint", "cpu_millis", "worker.command"]
    assert any("Kept from the replaced" in note for note in result["notes"])
    assert any("cpu_millis" in note for note in result["notes"])
    after = tomllib.loads((tmp_path / "cayu-cloud.toml").read_text())
    for key in ("home", "env", "storage", "schedules"):
        assert after[key] == before[key]
    assert after["cpu_millis"] == 1000
    CloudProjectManifest.load(tmp_path / "cayu-cloud.toml")


def test_init_force_with_a_schedule_of_a_kept_name_replaces_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _project(tmp_path, _HAND_EDITED)

    code, output = _cli(
        [
            "init",
            str(tmp_path),
            "--force",
            "--schedule",
            "digest=cron(0 8 * * ? *)=python -m herald digest",
            "--schedule-timezone",
            "America/Los_Angeles",
        ],
        capsys,
    )

    assert code == 0
    assert output["result"]["schedules"] == ["weekly", "digest"]
    schedules = tomllib.loads((tmp_path / "cayu-cloud.toml").read_text())["schedules"]
    assert schedules[0]["name"] == "weekly" and schedules[0]["timezone"] == "Europe/Paris"
    assert schedules[1] == {
        "name": "digest",
        "expression": "cron(0 8 * * ? *)",
        "command": "python -m herald digest",
        "timezone": "America/Los_Angeles",
    }


def test_init_force_says_when_it_cannot_keep_a_value(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    original = _BASE + "\n[env]\nSTARTED = 2026-11-12\n"
    _project(tmp_path, original)

    code, output = _cli(["init", str(tmp_path), "--force"], capsys)

    assert code == 2
    assert output["error"]["category"] == "manifest_invalid"
    assert "[env]" in output["error"]["message"]
    assert (tmp_path / "cayu-cloud.toml").read_text() == original


def test_init_check_reads_the_manifest_as_the_deploy_does(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _project(tmp_path, _HAND_EDITED)
    original = (tmp_path / "cayu-cloud.toml").read_text()

    code, output = _cli(["init", str(tmp_path), "--check"], capsys)

    assert code == 0
    assert output["operation"] == "init.check"
    result = output["result"]
    assert result["valid"] is True
    assert result["processes"] == ["worker"]
    assert [(item["name"], item["timezone"]) for item in result["schedules"]] == [
        ("digest", "UTC"),
        ("weekly", "Europe/Paris"),
    ]
    assert (tmp_path / "cayu-cloud.toml").read_text() == original


@pytest.mark.parametrize(
    ("change", "category", "message"),
    [
        (("cron(0 16 * * ? *)", "cron(0 16 * * * *)"), "manifest_invalid", "day-of-week"),
        (("Europe/Paris", "Europe/Nowhere"), "manifest_invalid", "IANA"),
        (('[env]\nLOG_LEVEL = "debug"', "[env]\nLOG_LEVEL = 3"), "manifest_invalid", "wrong type"),
        (("[storage]", "[unknown]"), "manifest_invalid", "supported fields"),
    ],
)
def test_init_check_refuses_what_the_deploy_refuses(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    change: tuple[str, str],
    category: str,
    message: str,
) -> None:
    _project(tmp_path, _HAND_EDITED.replace(*change))

    code, output = _cli(["init", str(tmp_path), "--check"], capsys)

    assert code == 2
    assert output["error"]["category"] == category
    assert message in output["error"]["message"]


def test_init_check_needs_a_manifest_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _project(tmp_path)

    code, output = _cli(["init", str(tmp_path), "--check"], capsys)
    assert code == 2
    assert output["error"]["category"] == "manifest_unavailable"

    code, output = _cli(["init", str(tmp_path), "--check", "--force"], capsys)
    assert code == 2
    assert output["error"]["category"] == "invalid_input"
    assert not (tmp_path / "cayu-cloud.toml").exists()


def test_deploy_explains_a_cloud_without_schedule_time_zones() -> None:
    refused = CloudApiError(
        "api_request_rejected",
        "Cayu Cloud API returned HTTP 422.",
        status_code=422,
        rejected_fields=(("body.manifest.runtime.schedules.1.timezone", "extra_forbidden"),),
    )

    with pytest.raises(cloud_cli.CloudCommandError) as raised:
        cloud_cli._raise_unsupported_schedule_timezone(refused)

    assert raised.value.category == "schedule_timezone_unsupported"
    # Any other 422 passes through.
    cloud_cli._raise_unsupported_schedule_timezone(
        CloudApiError("api_request_rejected", "x", rejected_fields=(("body.version", "x"),))
    )


_APP = "/v1/applications/herald-agent"


@pytest.fixture
def cloud(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Callable[[dict[tuple[str, str], httpx.Response]], None]:
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "cloud-config.json"))
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)

    def install(routes: dict[tuple[str, str], httpx.Response]) -> None:
        listed = httpx.Response(200, json={"items": [{"id": "herald-agent", "name": "Herald"}]})
        answers = {("GET", "/v1/applications?lifecycle=all"): listed, **routes}
        real_client = httpx.Client
        monkeypatch.setattr(
            _cloud_api.httpx,
            "Client",
            lambda **kwargs: real_client(
                transport=httpx.MockTransport(
                    lambda request: answers[(request.method, request.url.raw_path.decode())]
                ),
                **kwargs,
            ),
        )
        monkeypatch.setattr(
            cloud_cli,
            "_cloud_client",
            lambda *_args, **_kwargs: CloudApiClient(
                api_url="https://cloud.example", api_key="login-token"
            ),
        )

    return install


def test_schedule_list_shows_each_schedules_time_zone(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    release = {
        "id": "dep_1",
        "manifest": {
            "runtime": {
                "schedules": [
                    {"name": "digest", "expression": "cron(0 8 * * ? *)", "command": "a"},
                    {
                        "name": "evening",
                        "expression": "cron(0 17 * * ? *)",
                        "command": "b",
                        "timezone": "America/Los_Angeles",
                        "input": {"slot": "evening"},
                    },
                ]
            }
        },
    }
    cloud(
        {
            ("GET", _APP): httpx.Response(200, json={"current_deployment_id": "dep_1"}),
            ("GET", f"{_APP}/deployments/dep_1"): httpx.Response(200, json=release),
        }
    )

    code, output = _cli(["schedule", "list", "--application", "herald-agent"], capsys)

    assert code == 0
    assert output["operation"] == "schedule.list"
    assert output["result"] == {
        "application": "herald-agent",
        "release": "dep_1",
        "items": [
            {
                "name": "digest",
                "expression": "cron(0 8 * * ? *)",
                "timezone": "UTC",
                "command": "a",
            },
            {
                "name": "evening",
                "expression": "cron(0 17 * * ? *)",
                "timezone": "America/Los_Angeles",
                "command": "b",
                "input": {"slot": "evening"},
            },
        ],
    }


def test_schedule_list_without_a_release_is_empty(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud({("GET", _APP): httpx.Response(200, json={"current_deployment_id": None})})

    code, output = _cli(["schedule", "list", "--application", "herald-agent"], capsys)

    assert code == 0
    assert output["result"] == {"application": "herald-agent", "release": None, "items": []}


def test_init_force_regenerates_a_manifest_that_declares_no_schedules(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _project(tmp_path, _BASE.replace("[worker]", "schedules = []\n\n[worker]"))

    code, output = _cli(["init", str(tmp_path), "--force"], capsys)

    assert code == 0, output
    assert "schedules" not in tomllib.loads((tmp_path / "cayu-cloud.toml").read_text())


def test_init_check_never_boots_the_project(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Booting a public service opens (and can create) its stores, and --check writes nothing.
    from cayu.cli import _cloud_deploy_check

    def booted(_root: Path) -> None:
        raise AssertionError("init --check booted the project")

    monkeypatch.setattr(_cloud_deploy_check, "_run_public_service_check", booted)
    _project(
        tmp_path,
        _BASE.replace(
            "[worker]", '[web]\ncommand = "python -m herald.web"\nport = 8000\n\n[worker]'
        ),
    )
    with (tmp_path / "pyproject.toml").open("a") as pyproject:
        pyproject.write('\n[tool.cayu]\nfactory = "herald:app"\nservice_factory = "herald:svc"\n')

    code, output = _cli(["init", str(tmp_path), "--check"], capsys)

    assert code == 0, output
    check = output["result"]["deploy_check"]
    assert check["status"] == "skipped"
    assert "cayu check --deploy" in check["reason"]
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "cayu-cloud.toml",
        "pyproject.toml",
        "uv.lock",
    ]
