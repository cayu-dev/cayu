"""`cayu cloud compass` calls Cloud's Compass routes with the customer login."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from cayu.cli import _cloud_api, main
from cayu.cli import cloud as cloud_cli
from cayu.cli._cloud_api import CloudApiClient

_AGENT = "/v1/compass/agents/cayu-scout"


class _Cloud:
    """Answers each request from ``routes`` and keeps what the CLI sent."""

    def __init__(self, routes: dict[tuple[str, str], list[httpx.Response]]) -> None:
        self.routes = routes
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        target = request.url.raw_path.decode()
        answers = self.routes[(request.method, target)]
        return answers.pop(0) if len(answers) > 1 else answers[0]

    def bodies(self) -> list[object]:
        return [
            json.loads(request.content) if request.content else None for request in self.requests
        ]


@pytest.fixture
def cloud(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Callable[[dict[tuple[str, str], list[httpx.Response]]], _Cloud]:
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "cloud-config.json"))
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)

    def install(routes: dict[tuple[str, str], list[httpx.Response]]) -> _Cloud:
        fake = _Cloud(routes)
        real_client = httpx.Client

        def client(**kwargs: Any) -> httpx.Client:
            return real_client(transport=httpx.MockTransport(fake), **kwargs)

        monkeypatch.setattr(_cloud_api.httpx, "Client", client)
        monkeypatch.setattr(
            cloud_cli,
            "_cloud_client",
            lambda *_args, **_kwargs: CloudApiClient(
                api_url="https://cloud.example", api_key="login-token"
            ),
        )
        return fake

    return install


def _run(arguments: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, dict[str, Any]]:
    code = main(["cloud", "compass", *arguments])
    return code, json.loads(capsys.readouterr().out)


def test_cloud_help_lists_compass(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["cloud", "compass", "--help"])

    assert raised.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    for command in ("agents", "show", "enroll", "disable", "verify", "pass", "job", "missions"):
        assert command in help_text
    assert "findings" in help_text


def test_enroll_sends_only_the_given_options_with_the_login_and_prints_the_key(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    enrollment = {"application_id": "cayu-scout", "enrollment": {"state": "enrolled"}}
    fake = cloud({("PUT", _AGENT): [httpx.Response(200, json=enrollment)]})

    code, output = _run(
        [
            "enroll",
            "cayu-scout",
            "--agent",
            "scout",
            "--agent",
            "filer",
            "--allow-model-runs",
            "--fork-budget-usd",
            "2.50",
            "--lookback-days",
            "7",
            "--release",
            "dep_one",
            "--api-path",
            "/cayu/api",
        ],
        capsys,
    )

    assert code == 0
    assert output["operation"] == "compass.enroll"
    assert output["result"] == enrollment
    key = output["idempotency_key"]
    assert key.startswith("compass-enroll-")
    (request,) = fake.requests
    assert request.headers["Authorization"] == "Bearer login-token"
    assert request.headers["Idempotency-Key"] == key
    assert fake.bodies() == [
        {
            "agents": ["scout", "filer"],
            "allow_model_runs": True,
            "api_path": "/cayu/api",
            "fork_budget_usd": "2.50",
            "lookback_days": 7,
            "release_id": "dep_one",
        }
    ]


# What a Cloud that predates several agents answers for `agents`: FastAPI's validation.
_SINGLE_AGENT_CLOUD_REFUSAL = httpx.Response(
    422,
    json={
        "detail": [
            {"type": "missing", "loc": ["body", "agent"], "msg": "Field required"},
            {"type": "extra_forbidden", "loc": ["body", "agents"], "msg": "Extra inputs"},
        ]
    },
)


def test_enroll_falls_back_to_one_agent_on_a_cloud_without_several_agents(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    enrollment = {"application_id": "cayu-scout", "enrollment": {"agent": "scout"}}
    fake = cloud(
        {("PUT", _AGENT): [_SINGLE_AGENT_CLOUD_REFUSAL, httpx.Response(200, json=enrollment)]}
    )

    code, output = _run(["enroll", "cayu-scout", "--agent", "scout", "--min-runs", "3"], capsys)

    assert code == 0
    assert output["result"] == enrollment
    assert fake.bodies() == [
        {"agents": ["scout"], "min_runs": 3},
        {"agent": "scout", "min_runs": 3},
    ]
    keys = {request.headers["Idempotency-Key"] for request in fake.requests}
    assert keys == {output["idempotency_key"]}


def test_enroll_of_several_agents_on_a_cloud_without_them_says_so(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud({("PUT", _AGENT): [_SINGLE_AGENT_CLOUD_REFUSAL]})

    code, output = _run(
        ["enroll", "cayu-scout", "--agent", "a1", "--agent", "a2", "--idempotency-key", "e-2"],
        capsys,
    )

    assert code == 2
    assert output["error"] == {
        "category": "api_request_rejected",
        "idempotency_key": "e-2",
        "message": "This Cayu Cloud lets Compass watch one agent per Agent; pass --agent once.",
    }
    assert len(fake.requests) == 1


def test_enroll_reports_agents_required(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    message = "Agent cayu-scout watches several agents; name every one to keep in agents."
    detail = {"code": "agents_required", "message": message}
    cloud({("PUT", _AGENT): [httpx.Response(409, json={"detail": detail})]})

    code, output = _run(["enroll", "cayu-scout", "--agent", "scout"], capsys)

    assert code == 2
    assert output["error"]["code"] == "agents_required"
    assert output["error"]["message"] == f"Cayu Cloud API returned HTTP 409: {message}"


def test_enroll_refusal_reports_clouds_code_message_and_the_key(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    message = (
        "Release dep_one declares no objectives for Compass to judge findings by: its "
        "pyproject.toml declares no [tool.cayu.improvement]. Declare them under "
        "[tool.cayu.improvement] first (cayu objectives validate)."
    )
    cloud(
        {
            ("PUT", _AGENT): [
                httpx.Response(422, json={"detail": {"code": "no_objectives", "message": message}})
            ]
        }
    )

    code, output = _run(
        ["enroll", "cayu-scout", "--agent", "scout", "--idempotency-key", "enroll-1"], capsys
    )

    assert code == 2
    assert output["error"] == {
        "category": "api_request_rejected",
        "code": "no_objectives",
        "idempotency_key": "enroll-1",
        "message": f"Cayu Cloud API returned HTTP 422: {message}",
    }


def test_an_undocumented_refusal_keeps_its_message_private(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud(
        {
            ("GET", _AGENT): [
                httpx.Response(422, json={"detail": {"code": "other", "message": "canary"}})
            ]
        }
    )

    code, output = _run(["show", "cayu-scout"], capsys)

    assert code == 2
    assert output["error"] == {
        "category": "api_request_rejected",
        "message": "Cayu Cloud API returned HTTP 422.",
    }


def test_reads_call_the_compass_routes(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    fake = cloud(
        {
            ("GET", "/v1/compass/agents"): [httpx.Response(200, json={"items": []})],
            ("GET", _AGENT): [httpx.Response(200, json={"application_id": "cayu-scout"})],
            ("GET", f"{_AGENT}/jobs/job%2F1"): [httpx.Response(200, json={"job_id": "job/1"})],
            ("GET", f"{_AGENT}/missions?status=ready"): [
                httpx.Response(200, json={"items": [], "start_problem": None})
            ],
            ("GET", f"{_AGENT}/findings?status=proposed&agent=filer"): [
                httpx.Response(200, json={"items": []})
            ],
            ("GET", f"{_AGENT}/findings/fnd_1"): [httpx.Response(200, json={"id": "fnd_1"})],
            ("DELETE", _AGENT): [httpx.Response(200, json={"state": "disabled"})],
        }
    )

    operations = []
    for arguments in (
        ["agents"],
        ["show", "cayu-scout"],
        ["job", "cayu-scout", "job/1"],
        ["missions", "cayu-scout", "--status", "ready"],
        ["findings", "list", "cayu-scout", "--status", "proposed", "--agent", "filer", "--json"],
        ["findings", "show", "cayu-scout", "fnd_1"],
        ["disable", "cayu-scout"],
    ):
        code, output = _run(arguments, capsys)
        assert code == 0
        assert "idempotency_key" not in output
        operations.append(output["operation"])

    assert operations == [
        "compass.agents",
        "compass.show",
        "compass.job",
        "compass.missions",
        "compass.findings.list",
        "compass.findings.show",
        "compass.disable",
    ]
    assert all("Idempotency-Key" not in request.headers for request in fake.requests)


def test_findings_accept_and_dismiss_send_the_review(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {
            ("POST", f"{_AGENT}/findings/fnd_1/accept"): [
                httpx.Response(200, json={"status": "accepted"})
            ],
            ("POST", f"{_AGENT}/findings/fnd_2/dismiss"): [
                httpx.Response(200, json={"status": "dismissed"})
            ],
        }
    )

    accept_code, accepted = _run(
        [
            "findings",
            "accept",
            "cayu-scout",
            "fnd_1",
            "--note",
            "Worth trying",
            "--expected-effect",
            "Fewer retries",
            "--anatomy",
            "prompts/scout.md",
            "--anatomy",
            "tools/search.py",
            "--case",
            "case-1",
            "--idempotency-key",
            "accept-1",
        ],
        capsys,
    )
    dismiss_code, dismissed = _run(
        ["findings", "dismiss", "cayu-scout", "fnd_2", "--note", "Expected behavior", "--json"],
        capsys,
    )

    assert (accept_code, dismiss_code) == (0, 0)
    assert accepted["operation"] == "compass.findings.accept"
    assert accepted["idempotency_key"] == "accept-1"
    assert dismissed["idempotency_key"].startswith("compass-findings.dismiss-")
    assert fake.bodies() == [
        {
            "affected_anatomy": ["prompts/scout.md", "tools/search.py"],
            "development_cases": ["case-1"],
            "expected_effect": "Fewer retries",
            "note": "Worth trying",
        },
        {"note": "Expected behavior"},
    ]


def test_pass_waits_for_the_job_and_rides_out_a_brief_outage(
    cloud, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("cayu.cli._cloud_compass.time.sleep", lambda _seconds: None)
    fake = cloud(
        {
            ("POST", f"{_AGENT}/passes"): [
                httpx.Response(202, json={"job_id": "job_1", "status": "queued"})
            ],
            ("GET", f"{_AGENT}/jobs/job_1"): [
                httpx.Response(200, json={"job_id": "job_1", "status": "running"}),
                httpx.Response(503, json={"detail": "busy"}),
                httpx.Response(200, json={"job_id": "job_1", "status": "succeeded"}),
            ],
        }
    )

    code, output = _run(["pass", "cayu-scout", "--wait", "--poll-seconds", "1"], capsys)

    assert code == 0
    assert output["operation"] == "compass.pass"
    assert output["result"] == {"job_id": "job_1", "status": "succeeded"}
    assert [request.method for request in fake.requests] == ["POST", "GET", "GET", "GET"]


def test_verify_for_one_agent_waits_for_every_queued_job(
    cloud, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("cayu.cli._cloud_compass.time.sleep", lambda _seconds: None)
    queued = [
        {"job_id": "job_a", "agent": "scout", "status": "queued"},
        {"job_id": "job_b", "agent": "filer", "status": "succeeded"},
    ]
    failed = {"job_id": "job_a", "agent": "scout", "status": "failed"}
    fake = cloud(
        {
            ("POST", f"{_AGENT}/verify?agent=scout"): [
                httpx.Response(202, json={**queued[0], "jobs": queued})
            ],
            ("GET", f"{_AGENT}/jobs/job_a"): [httpx.Response(200, json=failed)],
        }
    )

    code, output = _run(["verify", "cayu-scout", "--agent", "scout", "--wait"], capsys)

    assert code == 0
    assert output["result"] == {**failed, "jobs": [failed, queued[1]]}
    assert [request.url.raw_path.decode() for request in fake.requests] == [
        f"{_AGENT}/verify?agent=scout",
        f"{_AGENT}/jobs/job_a",
    ]


def test_verify_wait_times_out_with_the_job_command(
    cloud, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    clock = iter(range(0, 1000, 10))
    monkeypatch.setattr("cayu.cli._cloud_compass.time.sleep", lambda _seconds: None)
    monkeypatch.setattr("cayu.cli._cloud_compass.time.monotonic", lambda: float(next(clock)))
    cloud(
        {
            ("POST", f"{_AGENT}/verify"): [
                httpx.Response(202, json={"job_id": "job_2", "status": "queued"})
            ],
            ("GET", f"{_AGENT}/jobs/job_2"): [
                httpx.Response(200, json={"job_id": "job_2", "status": "running"})
            ],
        }
    )

    code, output = _run(
        [
            "verify",
            "cayu-scout",
            "--wait",
            "--poll-seconds",
            "5",
            "--wait-seconds",
            "25",
            "--idempotency-key",
            "verify-1",
        ],
        capsys,
    )

    assert code == 2
    assert output["error"] == {
        "category": "wait_timeout",
        # The job is queued, so the key to repeat the command with and its id come back too.
        "idempotency_key": "verify-1",
        "job_ids": ["job_2"],
        "message": (
            "Job job_2 didn't finish within 25 seconds; read it later with "
            "`cayu cloud compass job cayu-scout JOB_ID`, or repeat the command with its "
            "idempotency key."
        ),
    }


def test_a_wait_that_runs_out_of_retries_keeps_the_key_of_the_queued_pass(
    cloud, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("cayu.cli._cloud_compass.time.sleep", lambda _seconds: None)
    cloud(
        {
            ("POST", f"{_AGENT}/passes"): [
                httpx.Response(202, json={"job_id": "job_3", "status": "queued"})
            ],
            ("GET", f"{_AGENT}/jobs/job_3"): [httpx.Response(503, json={"detail": "busy"})],
        }
    )

    code, output = _run(["pass", "cayu-scout", "--wait", "--poll-seconds", "1"], capsys)

    assert code == 2
    assert output["error"]["message"] == "Cayu Cloud API returned HTTP 503."
    assert output["error"]["idempotency_key"].startswith("compass-pass-")
    assert output["error"]["job_ids"] == ["job_3"]


def test_waiting_for_several_jobs_sleeps_once_per_round(
    cloud, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    now = [0.0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    monkeypatch.setattr("cayu.cli._cloud_compass.time.sleep", sleep)
    monkeypatch.setattr("cayu.cli._cloud_compass.time.monotonic", lambda: now[0])
    queued = [{"job_id": f"job_{index}", "status": "queued"} for index in range(3)]
    finished = [{**job, "status": "succeeded"} for job in queued]
    fake = cloud(
        {
            ("POST", f"{_AGENT}/passes"): [httpx.Response(202, json={**queued[0], "jobs": queued})],
            **{
                ("GET", f"{_AGENT}/jobs/{job['job_id']}"): [httpx.Response(200, json=job)]
                for job in finished
            },
        }
    )

    code, output = _run(
        ["pass", "cayu-scout", "--wait", "--poll-seconds", "5", "--wait-seconds", "10"], capsys
    )

    assert code == 0
    assert output["result"] == {**finished[0], "jobs": finished}
    assert now[0] == 5.0
    assert len(fake.requests) == 4


def test_compass_commands_refuse_an_invalid_agent_slug(capsys: pytest.CaptureFixture[str]) -> None:
    code, output = _run(["show", "Not A Slug"], capsys)

    assert code == 2
    assert output["error"]["category"] == "invalid_input"


# Narrowing, counting and dismissing findings in bulk ------------------------------------


def _finding(
    finding_id: str,
    *,
    rule: str = "repeated_tool",
    dimension: str = "time",
    score: int = 5,
    status: str = "proposed",
    tool: str | None = "search",
    step: str = "tool",
) -> dict[str, Any]:
    return {
        "finding_id": finding_id,
        "rule": rule,
        "dimension": dimension,
        "score": score,
        "status": status,
        "agent": "scout",
        "ledger_rows": [{"kind": step, "name": tool, "runs": 9, "sessions": 3}],
    }


_FINDINGS = [
    _finding("f-1", score=9),
    _finding("f-2", score=4),
    _finding("f-3", rule="failing_tool", dimension="correctness", score=3, tool="fetch"),
    _finding("f-4", rule="edited_gate", dimension="attention", step="approval_gate"),
    _finding("f-5", rule="costliest_model_calls", dimension="cost", tool="gpt", step="model_call"),
    _finding("f-6", status="dismissed"),
    _finding("f-7", status="accepted", tool="fetch"),
]


def test_findings_list_sends_the_filters_and_prints_what_cloud_matched(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {
            (
                "GET",
                f"{_AGENT}/findings?status=proposed&rule=repeated_tool&dimension=time"
                "&min_score=4&tool=search&limit=1",
            ): [httpx.Response(200, json={"items": [_FINDINGS[0]], "total": 2})]
        }
    )

    code = main(
        [
            "cloud",
            "compass",
            "findings",
            "list",
            "cayu-scout",
            "--status",
            "proposed",
            "--rule",
            "repeated_tool",
            "--dimension",
            "time",
            "--min-score",
            "4",
            "--tool",
            "search",
            "--limit",
            "1",
        ]
    )

    out = capsys.readouterr().out
    assert code == 0
    assert len(fake.requests) == 1
    assert out.splitlines()[0].split() == [
        "FINDING",
        "STATUS",
        "DIMENSION",
        "SCORE",
        "RULE",
        "TOOL",
        "AGENT",
    ]
    assert out.splitlines()[1].split() == [
        "f-1",
        "proposed",
        "time",
        "9",
        "repeated_tool",
        "search",
        "scout",
    ]
    assert out.splitlines()[-1] == "1 of 2 findings."


def test_findings_list_filters_itself_on_a_cloud_that_answers_every_finding(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    every = {"items": _FINDINGS}
    cloud(
        {
            ("GET", f"{_AGENT}/findings?tool=search"): [httpx.Response(200, json=every)],
            ("GET", f"{_AGENT}/findings?rule=repeated_tool&min_score=5"): [
                httpx.Response(200, json=every)
            ],
            ("GET", f"{_AGENT}/findings?limit=3"): [httpx.Response(200, json=every)],
            ("GET", f"{_AGENT}/findings?dimension=correctness"): [httpx.Response(200, json=every)],
        }
    )

    def listed(*options: str) -> tuple[list[str], int]:
        code, output = _run(["findings", "list", "cayu-scout", *options, "--json"], capsys)
        assert code == 0
        result = output["result"]
        return [item["finding_id"] for item in result["items"]], result["total"]

    # A tool's calls, and the approval gate on it; never a model's name.
    assert listed("--tool", "search") == (["f-1", "f-2", "f-4", "f-6"], 4)
    assert listed("--rule", "repeated_tool", "--min-score", "5") == (["f-1", "f-6", "f-7"], 3)
    # The strongest, ranked as Compass ranks them: correctness, cost, time, attention.
    assert listed("--limit", "3") == (["f-3", "f-5", "f-1"], 7)
    assert listed("--dimension", "correctness") == (["f-3"], 1)


def test_findings_summary_counts_by_rule_and_tool(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud({("GET", f"{_AGENT}/findings"): [httpx.Response(200, json={"items": _FINDINGS})]})

    code, output = _run(["findings", "list", "cayu-scout", "--summary", "--json"], capsys)

    assert code == 0
    assert output["operation"] == "compass.findings.summary"
    assert output["result"]["total"] == 7
    groups = output["result"]["groups"]
    assert groups[0] == {
        "rule": "repeated_tool",
        "tool": "search",
        "dimension": "time",
        "findings": 3,
        "proposed": 2,
        "accepted": 0,
        "dismissed": 1,
        "top_score": 9,
    }
    assert [(group["rule"], group["tool"]) for group in groups[1:]] == [
        ("costliest_model_calls", None),
        ("edited_gate", "search"),
        ("failing_tool", "fetch"),
        ("repeated_tool", "fetch"),
    ]

    assert main(["cloud", "compass", "findings", "list", "cayu-scout", "--summary"]) == 0
    table = capsys.readouterr().out.splitlines()
    assert table[0].split()[:3] == ["RULE", "TOOL", "DIMENSION"]
    assert table[1].split() == ["repeated_tool", "search", "time", "3", "2", "0", "1", "9"]
    assert table[-1] == "7 findings."

    code, output = _run(
        ["findings", "list", "cayu-scout", "--summary", "--limit", "2", "--json"], capsys
    )
    assert (code, output["error"]["category"]) == (2, "invalid_input")


def test_bulk_dismiss_sends_a_rules_proposed_findings_in_batches_under_one_key(
    cloud, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("cayu.cli._cloud_compass_findings.DISMISS_BATCH", 2)
    rule = [_finding(f"r-{index}") for index in range(3)]
    rule += [_finding("r-done", status="dismissed"), _finding("r-kept", status="accepted")]
    fake = cloud(
        {
            ("GET", f"{_AGENT}/findings?rule=repeated_tool"): [
                httpx.Response(200, json={"items": rule, "total": 5})
            ],
            ("POST", f"{_AGENT}/findings/dismiss"): [
                httpx.Response(
                    200,
                    json={
                        "dismissed": [{"finding_id": "r-0"}, {"finding_id": "r-1"}],
                        "skipped": [],
                        "missing": [],
                        "refused": [],
                    },
                ),
                httpx.Response(
                    200,
                    json={
                        "dismissed": [],
                        "skipped": [{"finding_id": "r-2", "status": "dismissed"}],
                        "missing": [],
                        "refused": [],
                    },
                ),
            ],
        }
    )

    code, output = _run(
        [
            "findings",
            "dismiss",
            "cayu-scout",
            "--rule",
            "repeated_tool",
            "--note",
            "The rule counts paging as repeats.",
            "--yes",
            "--idempotency-key",
            "bulk-1",
            "--json",
        ],
        capsys,
    )

    assert code == 0
    result = output["result"]
    assert output["idempotency_key"] == "bulk-1"
    assert result["dismissed"] == ["r-0", "r-1"]
    assert result["matched"] == 5
    assert sorted(item["finding_id"] for item in result["skipped"]) == ["r-2", "r-done", "r-kept"]
    posts = [request for request in fake.requests if request.method == "POST"]
    assert [request.headers["Idempotency-Key"] for request in posts] == ["bulk-1", "bulk-1"]
    assert fake.bodies()[1:] == [
        {"finding_ids": ["r-0", "r-1"], "note": "The rule counts paging as repeats."},
        {"finding_ids": ["r-2"], "note": "The rule counts paging as repeats."},
    ]


def test_bulk_dismiss_prints_how_many_it_dismissed_and_repeats_as_a_no_op(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    listing = [_finding("r-0"), _finding("r-1", status="dismissed")]
    fake = cloud(
        {
            ("GET", f"{_AGENT}/findings?rule=repeated_tool"): [
                httpx.Response(200, json={"items": listing, "total": 2}),
                httpx.Response(
                    200,
                    json={"items": [{**listing[0], "status": "dismissed"}, listing[1]], "total": 2},
                ),
            ],
            ("POST", f"{_AGENT}/findings/dismiss"): [
                httpx.Response(200, json={"dismissed": [{"finding_id": "r-0"}]})
            ],
        }
    )
    command = [
        "cloud",
        "compass",
        "findings",
        "dismiss",
        "cayu-scout",
        "--rule",
        "repeated_tool",
        "--note",
        "Noise.",
        "--yes",
        "--idempotency-key",
        "bulk-2",
    ]

    assert main(command) == 0
    first = capsys.readouterr().out.splitlines()
    assert main(command) == 0
    second = capsys.readouterr().out.splitlines()

    assert first == [
        "Dismissed 1 of the 2 matching findings of rule repeated_tool.",
        "Left as they were, already reviewed: 1 dismissed.",
        "Idempotency key: bulk-2",
    ]
    assert second[0] == "Dismissed 0 of the 2 matching findings of rule repeated_tool."
    assert [request.method for request in fake.requests] == ["GET", "POST", "GET"]


def test_bulk_dismiss_falls_back_to_one_at_a_time_on_a_cloud_without_the_bulk_route(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {
            ("GET", f"{_AGENT}/findings?rule=repeated_tool"): [
                httpx.Response(
                    200, json={"items": [_finding("r-0"), _finding("r-1"), _finding("r-2")]}
                )
            ],
            ("POST", f"{_AGENT}/findings/dismiss"): [
                httpx.Response(405, json={"detail": "Method Not Allowed"})
            ],
            ("GET", f"{_AGENT}/findings/r-0"): [httpx.Response(200, json=_finding("r-0"))],
            ("GET", f"{_AGENT}/findings/r-1"): [httpx.Response(200, json=_finding("r-1"))],
            # Accepted by someone else after the listing: read again, it is left alone.
            ("GET", f"{_AGENT}/findings/r-2"): [
                httpx.Response(200, json=_finding("r-2", status="accepted"))
            ],
            ("POST", f"{_AGENT}/findings/r-0/dismiss"): [
                httpx.Response(200, json={"finding_id": "r-0", "status": "dismissed"})
            ],
            ("POST", f"{_AGENT}/findings/r-1/dismiss"): [
                httpx.Response(
                    409, json={"detail": {"code": "refused", "message": "In a mission."}}
                )
            ],
        }
    )

    code = main(
        [
            "cloud",
            "compass",
            "findings",
            "dismiss",
            "cayu-scout",
            "--rule",
            "repeated_tool",
            "--note",
            "Noise.",
            "--yes",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert code == 0
    assert "no bulk dismiss yet; dismissing the 3 findings one at a time" in captured.err
    assert output["result"]["dismissed"] == ["r-0"]
    assert [item["finding_id"] for item in output["result"]["refused"]] == ["r-1"]
    assert output["result"]["skipped"] == [{"finding_id": "r-2", "status": "accepted"}]
    posts = [request for request in fake.requests[2:] if request.method == "POST"]
    assert [request.url.path for request in posts] == [
        f"{_AGENT}/findings/r-0/dismiss",
        f"{_AGENT}/findings/r-1/dismiss",
    ]
    keys = [request.headers["Idempotency-Key"] for request in posts]
    assert len(set(keys)) == 2
    assert all(key.startswith("compass-") and len(key) <= 128 for key in keys)


def test_bulk_dismiss_asks_first_and_dismisses_nothing_without_a_yes(
    cloud, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {
            ("GET", f"{_AGENT}/findings?rule=repeated_tool"): [
                httpx.Response(200, json={"items": [_finding("r-0")], "total": 1})
            ],
            ("POST", f"{_AGENT}/findings/dismiss"): [
                httpx.Response(200, json={"dismissed": [{"finding_id": "r-0"}]})
            ],
        }
    )
    command = ["dismiss", "cayu-scout", "--rule", "repeated_tool", "--note", "Noise.", "--json"]

    class _Terminal:
        def __init__(self, answer: str, *, tty: bool = True) -> None:
            self.answer, self.tty = answer, tty

        def isatty(self) -> bool:
            return self.tty

        def readline(self) -> str:
            return self.answer

    monkeypatch.setattr("sys.stdin", _Terminal("", tty=False))
    code, output = _run(["findings", *command], capsys)
    assert (code, output["error"]["category"]) == (2, "confirmation_required")
    assert "idempotency_key" not in output["error"]

    monkeypatch.setattr("sys.stdin", _Terminal("n\n"))
    code = main(["cloud", "compass", "findings", *command])
    captured = capsys.readouterr()
    assert "1 proposed findings of Agent cayu-scout to dismiss:" in captured.err
    assert "Dismiss these 1 findings? [y/N]" in captured.err
    assert (code, json.loads(captured.out)["error"]["category"]) == (2, "not_confirmed")
    assert all(request.method == "GET" for request in fake.requests)

    monkeypatch.setattr("sys.stdin", _Terminal("yes\n"))
    code, output = _run(["findings", *command], capsys)
    assert (code, output["result"]["dismissed"]) == (0, ["r-0"])


def test_bulk_dismiss_that_fails_partway_says_how_many_it_dismissed(
    cloud, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("cayu.cli._cloud_compass_findings.DISMISS_BATCH", 1)
    cloud(
        {
            ("GET", f"{_AGENT}/findings?rule=repeated_tool"): [
                httpx.Response(200, json={"items": [_finding("r-0"), _finding("r-1")], "total": 2})
            ],
            ("POST", f"{_AGENT}/findings/dismiss"): [
                httpx.Response(200, json={"dismissed": [{"finding_id": "r-0"}]}),
                httpx.Response(503, json={"detail": "busy"}),
            ],
        }
    )

    code = main(
        [
            "cloud",
            "compass",
            "findings",
            "dismiss",
            "cayu-scout",
            "--rule",
            "repeated_tool",
            "--note",
            "Noise.",
            "--yes",
            "--idempotency-key",
            "bulk-3",
        ]
    )

    assert code == 2
    assert capsys.readouterr().err.splitlines() == [
        "Error: Cayu Cloud API returned HTTP 503. Dismissed 1 findings before that; repeating "
        "the command with the same --idempotency-key dismisses the rest.",
        "Idempotency key: bulk-3",
    ]


def test_bulk_dismiss_keeps_its_count_and_key_when_the_login_fails_to_refresh_partway(
    cloud, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from cayu.cli._cloud_auth import CloudAuthError

    monkeypatch.setattr("cayu.cli._cloud_compass_findings.DISMISS_BATCH", 1)
    cloud(
        {
            ("GET", f"{_AGENT}/findings?rule=repeated_tool"): [
                httpx.Response(200, json={"items": [_finding("r-0"), _finding("r-1")], "total": 2})
            ],
            ("POST", f"{_AGENT}/findings/dismiss"): [
                httpx.Response(200, json={"dismissed": [{"finding_id": "r-0"}]})
            ],
        }
    )
    calls = [0]

    def token() -> str:
        calls[0] += 1
        if calls[0] > 2:
            raise CloudAuthError("login_required", "Run `cayu cloud login` again.")
        return "login-token"

    monkeypatch.setattr(
        cloud_cli,
        "_cloud_client",
        lambda *_args, **_kwargs: CloudApiClient(
            api_url="https://cloud.example", api_key="login-token", api_key_provider=token
        ),
    )

    code, output = _run(
        [
            "findings",
            "dismiss",
            "cayu-scout",
            "--rule",
            "repeated_tool",
            "--note",
            "Noise.",
            "--yes",
            "--idempotency-key",
            "bulk-4",
            "--json",
        ],
        capsys,
    )

    assert code == 2
    error = output["error"]
    assert (error["category"], error["idempotency_key"], error["dismissed"]) == (
        "login_required",
        "bulk-4",
        1,
    )
    assert error["message"].startswith("Run `cayu cloud login` again. Dismissed 1 findings")


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (["f-1", "--rule", "repeated_tool"], "not both (--rule)"),
        (["--tool", "search"], "or --rule to dismiss"),
    ],
)
def test_dismiss_takes_a_finding_id_or_a_rule(
    cloud, options: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud({})
    code, output = _run(
        ["findings", "dismiss", "cayu-scout", *options, "--note", "x", "--json"], capsys
    )

    assert (code, output["error"]["category"]) == (2, "invalid_input")
    assert message in output["error"]["message"]
