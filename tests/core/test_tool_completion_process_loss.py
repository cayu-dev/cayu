"""Real process loss after final-tool success, including terminal-event repair."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from tests.core.test_tool_completion import CrashStore, FinalTool, app_for, call, request
from tests.core.test_tool_round_publication_failure_matrix import _TwoCallProvider

import cayu
from cayu import EventType, IncompleteSessionRecoveryRequest, SessionStatus, SQLiteSessionStore


@pytest.mark.parametrize(
    "backend", ["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)]
)
@pytest.mark.parametrize("boundary", ["tool-event", "tool-publication", "completion"])
def test_final_tool_success_survives_sigkill(backend, boundary, tmp_path, request):
    configuration = {
        "cayu_origin": str(Path(cayu.__file__).resolve()),
        "backend": backend,
        "boundary": boundary,
        "directory": str(tmp_path),
        "session_id": f"final-tool-{uuid4()}",
        "database": request.getfixturevalue("postgres_dsn")
        if backend == "postgres"
        else str(tmp_path / "sessions.sqlite"),
    }

    def launch(mode):
        return subprocess.run(
            [sys.executable, "-m", "tests.core.test_tool_completion_process_loss", mode],
            input=json.dumps(configuration),
            capture_output=True,
            text=True,
            timeout=60,
            cwd=Path(__file__).resolve().parents[2],
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join(
                    (
                        str(Path(cayu.__file__).resolve().parent.parent),
                        str(Path(__file__).resolve().parents[2]),
                        os.environ.get("PYTHONPATH", ""),
                    )
                ),
            },
            check=False,
        )

    crashed = launch("crash")
    assert crashed.returncode == -signal.SIGKILL, crashed.stdout + crashed.stderr
    assert (tmp_path / "crash-boundary").read_text() == boundary
    for mode in ("recover", "retry"):
        recovered = launch(mode)
        assert recovered.returncode == 0, recovered.stdout + recovered.stderr
    first = json.loads((tmp_path / "recover.json").read_text())
    second = json.loads((tmp_path / "retry.json").read_text())
    assert first == second
    assert first["completion"]["reason"] == "host_rendered_tool"
    assert first["completion"]["tool_completion"]["effect"] == "idempotent"
    assert first["roles"] == ["user", "assistant", "tool"]
    assert (tmp_path / "provider.calls").read_text() == "dispatch\n"
    assert (tmp_path / "tool.calls").read_text() == "execute\n"


def record(path: Path, value: str) -> None:
    with path.open("a") as output:
        output.write(value + "\n")
        output.flush()
        os.fsync(output.fileno())


async def worker(configuration, mode):
    assert str(Path(cayu.__file__).resolve()) == configuration["cayu_origin"]
    directory = Path(configuration["directory"])

    class ProcessLoss(CrashStore):
        invocation_lifecycle_command_version = 1

        def lose_process(self, message):
            (directory / "crash-boundary").write_text(self.boundary)
            os.kill(os.getpid(), signal.SIGKILL)
            raise AssertionError("SIGKILL unexpectedly returned")

    if configuration["backend"] == "postgres":
        from cayu.storage.migrations import SchemaMode
        from cayu.storage.postgres import PostgresSessionStore

        backend_type = PostgresSessionStore
        options = {"schema_mode": SchemaMode.CREATE, "min_size": 1, "max_size": 2}
    else:
        backend_type = SQLiteSessionStore
        options = {}

    class Store(ProcessLoss, backend_type):
        invocation_lifecycle_command_version = 1

    class Provider(_TwoCallProvider):
        async def stream(self, request):
            record(directory / "provider.calls", "dispatch")
            assert mode == "crash", "Recovery must not dispatch a provider."
            async for event in super().stream(request):
                yield event

    class RecordedTool(FinalTool):
        async def run(self, ctx, args):
            record(directory / "tool.calls", "execute")
            assert mode == "crash", "Recovery must not execute the successful tool."
            return await super().run(ctx, args)

    store = Store(
        configuration["database"],
        boundary=configuration["boundary"] if mode == "crash" else "none",
        **options,
    )
    app = app_for(store, Provider([call()] if mode == "crash" else []), RecordedTool())
    session_id = configuration["session_id"]
    try:
        if mode == "crash":
            [
                e
                async for e in app.run(
                    request(
                        session_id=session_id,
                        max_steps=1,
                        tool_completion={"tool_names": ["ask_customer"]},
                    )
                )
            ]
            raise AssertionError("The configured crash boundary was not reached.")
        result = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id, inactive_for_seconds=0)
        )
        assert result.status == SessionStatus.COMPLETED, result
        events = await store.load_events(session_id)
        completions = [e for e in events if e.type == EventType.SESSION_COMPLETED]
        assert len(completions) == 1
        transcript = await store.load_transcript(session_id)
        snapshot = {
            "completion": completions[0].payload,
            "roles": [m.role for m in transcript],
            "transcript": [m.model_dump(mode="json") for m in transcript],
        }
        (directory / f"{mode}.json").write_text(json.dumps(snapshot))
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(worker(json.loads(sys.stdin.read()), sys.argv[1]))
