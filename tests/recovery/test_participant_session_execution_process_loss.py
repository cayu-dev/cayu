"""Participant-root execution recovery after a real owner SIGKILL."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from pathlib import Path
from uuid import uuid4

import pytest

pytestmark = [
    pytest.mark.process,
    pytest.mark.sigkill_recovery,
    pytest.mark.skipif(os.name != "posix", reason="requires SIGKILL"),
]


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_participant_session_execution_recovers_after_sigkill(request, tmp_path, backend):
    root = Path(__file__).resolve().parents[2]
    address = (
        str(tmp_path / "participant-session.sqlite")
        if backend == "sqlite"
        else request.getfixturevalue("postgres_dsn")
    )
    scope = uuid4().hex
    command = [
        sys.executable,
        "-m",
        "tests.recovery.participant_session_execution_worker",
        backend,
        address,
        address,
        scope,
    ]
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join((str(root / "src"), str(root)))}

    async def exercise():
        worker = await asyncio.create_subprocess_exec(
            *command,
            "admission",
            cwd=root,
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            assert worker.stdout is not None
            line = await asyncio.wait_for(worker.stdout.readline(), 30)
            if not line:
                _, errors = await worker.communicate()
                pytest.fail(errors.decode())
            payload = json.loads(line)
            worker.send_signal(signal.SIGKILL)
            await asyncio.wait_for(worker.wait(), 10)
            assert worker.returncode == -signal.SIGKILL
        finally:
            if worker.returncode is None:
                worker.kill()
            await worker.communicate()

        recovery = await asyncio.create_subprocess_exec(
            *command,
            "replay",
            cwd=root,
            env=environment,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            output, errors = await asyncio.wait_for(
                recovery.communicate(json.dumps(payload).encode()), 30
            )
            assert recovery.returncode == 0, errors.decode()
            assert json.loads(output)["status"] == "completed"
        finally:
            if recovery.returncode is None:
                recovery.kill()
            await recovery.communicate()

    asyncio.run(exercise())
