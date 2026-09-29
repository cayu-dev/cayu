from __future__ import annotations

# ruff: noqa: E402
import asyncio
import json
import socket
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

pytest.importorskip("fastapi")
uvicorn = pytest.importorskip("uvicorn")

import cayu.cli.doctor as doctor_cli
from cayu.applications import CayuApp
from cayu.cli import main
from cayu.cli.diagnostics import request_cost_endpoint
from cayu.runtime.checks import check_manifest
from cayu.runtime.request_costs import RequestCostRoute, RequestCostSummary
from cayu.server import ServerConfig, create_server
from cayu.support_bundles import (
    DEFAULT_SUPPORT_BUNDLE_LIMITS,
    CollectorDisposition,
    RequestCostSource,
    SupportBundleContext,
    SupportBundleOutcome,
    SupportBundleReport,
    builtin_support_collectors,
    collect_support_bundle,
    encode_support_bundle,
    validate_support_bundle_archive,
)


def _summary(*, cpu_seconds: float, enabled: bool = True) -> RequestCostSummary:
    window = 60.0
    return RequestCostSummary(
        observed_at=datetime(2026, 9, 28, tzinfo=UTC),
        enabled=enabled,
        requested_window_seconds=300.0,
        window_seconds=window,
        truncated=False,
        buffer_capacity=10_000,
        vcpu=0.5,
        requests=30,
        requests_per_minute=30.0,
        cpu_seconds=cpu_seconds,
        cpu_seconds_per_minute=cpu_seconds,
        vcpu_share=cpu_seconds / window / 0.5,
        route_count=2,
        routes=(
            RequestCostRoute(
                method="GET",
                route="/api/state",
                requests=29,
                requests_per_minute=29.0,
                streaming_requests=0,
                status_2xx=29,
                status_3xx=0,
                status_4xx=0,
                status_5xx=0,
                wall_ms_p50=280.0,
                wall_ms_p95=350.0,
                wall_ms_max=400.0,
                cpu_seconds=cpu_seconds * 0.9,
                cpu_seconds_per_minute=cpu_seconds * 0.9,
                vcpu_share=cpu_seconds * 0.9 / window / 0.5,
                response_bytes=29_000,
                clients=1,
            ),
            RequestCostRoute(
                method="GET",
                route="/api/health",
                requests=1,
                requests_per_minute=1.0,
                streaming_requests=0,
                status_2xx=1,
                status_3xx=0,
                status_4xx=0,
                status_5xx=0,
                wall_ms_p50=1.0,
                wall_ms_p95=1.0,
                wall_ms_max=1.0,
                cpu_seconds=cpu_seconds * 0.1,
                cpu_seconds_per_minute=cpu_seconds * 0.1,
                vcpu_share=cpu_seconds * 0.1 / window / 0.5,
                response_bytes=11,
                clients=1,
            ),
        ),
    )


@pytest.fixture
def served_summary(monkeypatch) -> dict:
    """Answer the CLI's HTTP request with a controlled summary."""

    state: dict = {"summary": _summary(cpu_seconds=0.6), "status": 200, "requests": []}
    real_client = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        state["requests"].append(request)
        if state["status"] != 200:
            return httpx.Response(state["status"], json={"detail": "denied"})
        return httpx.Response(200, content=state["summary"].model_dump_json())

    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    return state


def test_requests_reports_rate_latency_cpu_and_vcpu_share(served_summary, capsys) -> None:
    assert main(["diagnostics", "requests", "--since", "5m", "--vcpu", "0.5"]) == 0

    output = capsys.readouterr().out
    assert output == (
        "Requests in the last 5m (covered 1m): 30 requests, 30.0/min\n"
        "Request CPU: 0.600 s, 0.600 s/min, 2.00% of 0.5 vCPU\n"
        "\n"
        "Top routes by CPU:\n"
        "METHOD  ROUTE        REQ/MIN  P50 MS  P95 MS  CPU S/MIN  VCPU %\n"
        "GET     /api/state      29.0   280.0   350.0      0.540    1.80\n"
        "GET     /api/health      1.0     1.0     1.0      0.060    0.20\n"
    )
    (request,) = served_summary["requests"]
    assert str(request.url) == (
        "http://127.0.0.1:8000/api/diagnostics/requests?since_seconds=300.0&vcpu=0.5"
    )
    assert "authorization" not in request.headers


def test_budget_exits_non_zero_only_when_exceeded(served_summary, capsys, monkeypatch) -> None:
    budget = ["diagnostics", "requests", "--budget-idle-cpu", "0.01", "--vcpu", "0.5"]

    assert main(budget) == 1
    assert "Budget: 1.00% of 0.5 vCPU: EXCEEDED" in capsys.readouterr().out

    served_summary["summary"] = _summary(cpu_seconds=0.12)
    assert main([*budget, "--json"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["budget"] == {
        "exceeded": False,
        "idle_cpu": 0.01,
        "vcpu": 0.5,
        "vcpu_share": pytest.approx(0.004),
    }

    # The operator credential comes from the environment and is never printed.
    monkeypatch.setenv("CAYU_API_AUTHORIZATION", "Bearer operator-secret")
    served_summary["status"] = 401
    assert main(budget) == 2
    captured = capsys.readouterr()
    assert "HTTP 401" in captured.err
    assert "operator-secret" not in captured.err + captured.out
    assert served_summary["requests"][-1].headers["authorization"] == "Bearer operator-secret"

    served_summary["status"] = 200
    served_summary["summary"] = _summary(cpu_seconds=0.0, enabled=False)
    assert main(budget) == 2
    assert "request timing is off" in capsys.readouterr().err


def test_requests_rejects_plain_http_outside_loopback(capsys) -> None:
    assert main(["diagnostics", "requests", "--server-url", "http://example.com/cayu"]) == 2
    assert "HTTPS outside loopback" in capsys.readouterr().err


@contextmanager
def _running_server(app) -> Iterator[str]:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(app, lifespan="off", ws="none", log_level="warning", access_log=False)
    )
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        # Readiness wait for the listener; no assertion depends on elapsed time.
        while not server.started:
            assert thread.is_alive() and time.monotonic() < deadline
            time.sleep(0.01)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        listener.close()


def _record_traffic(url: str) -> None:
    for _ in range(5):
        assert httpx.get(f"{url}/api/health").status_code == 200
    httpx.get(f"{url}/api/sessions/session-4c1d9e", params={"cursor": "query-secret"})


def test_requests_command_reads_a_running_server(capsys) -> None:
    server = create_server(
        CayuApp(enable_logging=False),
        config=ServerConfig.local_development(),
    )
    with _running_server(server) as url:
        _record_traffic(url)
        assert main(["diagnostics", "requests", "--server-url", url, "--json"]) == 0

    document = json.loads(capsys.readouterr().out)
    routes = {route["route"]: route for route in document["summary"]["routes"]}
    assert routes["/api/health"]["requests"] == 5
    assert "/api/sessions/{session_id}" in routes
    assert "/api/diagnostics/requests" not in routes


def test_doctor_bundle_includes_the_running_server_request_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "project.py").write_text(
        "from cayu import CayuApp\n\n\ndef build_app():\n    return CayuApp(enable_logging=False)\n",
        encoding="utf-8",
    )
    sys.modules.pop("project", None)
    monkeypatch.chdir(tmp_path)
    server = create_server(
        CayuApp(enable_logging=False),
        config=ServerConfig.local_development(),
    )

    with _running_server(server) as url:
        _record_traffic(url)
        report = doctor_cli._collect_project_report(
            target="project:build_app",
            sessions=(),
            request_costs=RequestCostSource(
                endpoint=request_cost_endpoint(url),
                since_seconds=300,
            ),
        )

    validate_support_bundle_archive(encode_support_bundle(report))
    collectors = {item.name: item for item in report.collectors}
    result = collectors["request_costs"]
    assert result.disposition is CollectorDisposition.COLLECTED
    evidence = result.evidence.model_dump(mode="json")
    assert evidence["kind"] == "request_costs"
    bundled_routes = {route["route"]: route for route in evidence["routes"]}
    assert bundled_routes["api/health"]["requests"] == 5
    assert "api/sessions/{session_id}" in bundled_routes
    serialized = report.model_dump_json()
    for raw in ("session-4c1d9e", "query-secret", "127.0.0.1", "http://"):
        assert raw not in serialized


def test_doctor_requests_from_passes_the_source_to_the_worker(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    captured: dict = {}

    def run_worker(*_args, **kwargs) -> SupportBundleReport:
        captured.update(kwargs)
        return SupportBundleReport.from_results(
            generated_at=datetime.now(UTC),
            outcome=SupportBundleOutcome.CLEAN,
            limits=DEFAULT_SUPPORT_BUNDLE_LIMITS,
            collectors=(),
            collection_duration_ms=0,
        )

    monkeypatch.setattr(doctor_cli, "_run_bounded_worker", run_worker)
    monkeypatch.setattr(doctor_cli, "_run_bounded_publisher", lambda *_args, **_kwargs: True)
    monkeypatch.setenv("CAYU_API_AUTHORIZATION", "Bearer operator-secret")
    bundle = str(tmp_path / "support.zip")
    arguments = ["doctor", "--bundle", bundle, "--requests-from", "http://127.0.0.1:8000/cayu"]

    assert main([*arguments, "--requests-since", "10m", "--json"]) == 0

    source = captured["request_costs"]
    assert source.endpoint == "http://127.0.0.1:8000/cayu/api/diagnostics/requests"
    assert source.since_seconds == 600
    assert source.authorization == "Bearer operator-secret"
    assert "operator-secret" not in repr(source) + capsys.readouterr().out

    main(["doctor", "--bundle", bundle, "--json"])
    assert captured["request_costs"] is None

    with pytest.raises(SystemExit):
        main(["doctor", "--bundle", bundle, "--requests-from", "http://example.com/cayu"])


def test_bundle_collector_reports_an_unreachable_server_as_unavailable() -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.close()
    app = CayuApp(enable_logging=False)
    manifest = app.describe()
    context = SupportBundleContext(
        app=app,
        manifest=manifest,
        check_report=check_manifest(manifest),
        service_manifest=None,
        project_id=None,
        application_release_id=f"manifest-{manifest.fingerprint}",
        eval_backend=None,
        eval_source=None,
    )
    source = RequestCostSource(
        endpoint=f"http://127.0.0.1:{port}/api/diagnostics/requests",
        since_seconds=300,
        authorization="Bearer operator-secret",
    )
    collectors = [
        collector
        for collector in builtin_support_collectors(request_costs=source)
        if collector.name == "request_costs"
    ]

    report = asyncio.run(collect_support_bundle(context, collectors))

    (result,) = report.collectors
    assert result.disposition is CollectorDisposition.UNAVAILABLE
    assert result.reason_code == "request_summary_unavailable"
    assert "operator-secret" not in repr(source)
    validate_support_bundle_archive(encode_support_bundle(report))
