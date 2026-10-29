from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from cayu.cli import main
from cayu.cli import worker as worker_cli
from cayu.cli._worker_drain import DrainRequest, drain_notice, watch_cloud_drain

if TYPE_CHECKING:
    import pytest

CLOUD = {
    "CAYU_WORKER_DRAIN_URL": "https://api.cloud.example.test/v1/agent-worker/drain",
    "CAYU_ATTENTION_HANDOFF_TOKEN": "cayu_attention_token",
    "CAYU_RELEASE_ID": "dep_old",
}
DRAINING = {
    "application_id": "scout-agent",
    "release_id": "dep_old",
    "draining": True,
    "replaced_by": "dep_new",
    "requested_at": "2026-10-29T02:57:00+00:00",
    "deadline_at": "2026-10-29T05:57:00+00:00",
}


def _watch(answers: list[httpx.Response | Exception], environ=CLOUD):
    requests: list[httpx.Request] = []
    sleeps: list[float] = []
    drains: list[DrainRequest] = []

    def answer(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        reply = answers.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    asyncio.run(
        watch_cloud_drain(
            drains.append,
            environ=environ,
            transport=httpx.MockTransport(answer),
            sleep=sleep,
        )
    )
    return requests, sleeps, drains


def test_the_worker_asks_until_its_release_is_draining_then_stops_asking() -> None:
    not_draining = {**DRAINING, "draining": False, "replaced_by": None, "deadline_at": None}

    requests, sleeps, drains = _watch(
        [httpx.Response(200, json=not_draining), httpx.Response(200, json=DRAINING)]
    )

    assert drains == [
        DrainRequest(
            release_id="dep_old", replaced_by="dep_new", deadline_at="2026-10-29T05:57:00+00:00"
        )
    ]
    assert sleeps == [30.0]
    assert [str(request.url) for request in requests] == [
        "https://api.cloud.example.test/v1/agent-worker/drain?release_id=dep_old"
    ] * 2
    assert requests[0].headers["authorization"] == "Bearer cayu_attention_token"


def test_failures_are_retried_quietly_and_retry_after_is_kept_within_bounds() -> None:
    _, sleeps, drains = _watch(
        [
            httpx.ConnectError("down"),
            httpx.Response(503, headers={"Retry-After": "3600"}),
            httpx.Response(401, json={"detail": {"code": "agent_credential_invalid"}}),
            httpx.Response(200, text="not json"),
            httpx.Response(200, json={**DRAINING, "release_id": "dep_other"}),
            httpx.Response(200, json=DRAINING),
        ]
    )

    assert sleeps == [30.0, 300.0, 30.0, 30.0, 30.0]
    assert len(drains) == 1


def test_nothing_is_asked_outside_cayu_cloud_or_of_a_cloud_without_the_question() -> None:
    for missing in CLOUD:
        environ = {key: value for key, value in CLOUD.items() if key != missing}
        requests, _, drains = _watch([], environ=environ)
        assert (requests, drains) == ([], [])

    requests, sleeps, drains = _watch([httpx.Response(404, json={"detail": "Not Found"})])
    assert (len(requests), sleeps, drains) == (1, [], [])


def test_the_drain_notice_names_the_replacement_and_the_deadline() -> None:
    notice = drain_notice(
        DrainRequest(release_id="dep_old", replaced_by="dep_new", deadline_at="05:57")
    )

    assert notice == (
        "Release dep_old was replaced by Release dep_new, so this worker is draining: it claims "
        "no new tasks and exits once the ones it holds finish. Cayu Cloud stops it at 05:57 if "
        "they haven't finished by then."
    )


def test_a_drained_worker_stops_claiming_finishes_and_exits_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        """[tool.cayu]
factory = "drained_worker:build_app"

[tool.cayu.workers]
drain = "drained_worker:run"
""",
        encoding="utf-8",
    )
    (project / "drained_worker.py").write_text(
        """from pathlib import Path

from cayu import CayuApp


def build_app():
    return CayuApp(enable_logging=False)


async def run(app, stop):
    await stop.wait()
    Path("finished.txt").write_text("finished\\n", encoding="utf-8")
""",
        encoding="utf-8",
    )

    async def drained_at_once(on_drain, **_options):
        on_drain(DrainRequest(release_id="dep_old", replaced_by="dep_new", deadline_at=None))

    monkeypatch.setattr(worker_cli, "watch_cloud_drain", drained_at_once)
    monkeypatch.chdir(project)
    sys.modules.pop("drained_worker", None)

    assert main(["worker", "drain"]) == 0

    assert (project / "finished.txt").read_text(encoding="utf-8") == "finished\n"
    assert "this worker is draining" in capsys.readouterr().err


def test_a_failing_drain_watch_never_fails_the_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        """[tool.cayu]
factory = "plain_worker:build_app"

[tool.cayu.workers]
once = "plain_worker:run"
""",
        encoding="utf-8",
    )
    (project / "plain_worker.py").write_text(
        """from cayu import CayuApp


def build_app():
    return CayuApp(enable_logging=False)


async def run(app, stop):
    return None
""",
        encoding="utf-8",
    )

    async def broken(on_drain, **_options):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(worker_cli, "watch_cloud_drain", broken)
    monkeypatch.chdir(project)
    sys.modules.pop("plain_worker", None)

    assert main(["worker", "once"]) == 0
