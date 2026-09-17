"""Fresh-process lifecycle reconstruction after commit and SIGKILL."""

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
@pytest.mark.parametrize("phase", ["disable", "settlement", "retirement", "pruning"])
def test_lifecycle_commit_survives_owner_loss(request, tmp_path, backend, phase):
    address = (
        str(tmp_path / "lifecycle.sqlite")
        if backend == "sqlite"
        else request.getfixturevalue("postgres_dsn")
    )
    root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        "-m",
        "tests.recovery.collaboration_lifecycle_worker",
        backend,
        address,
        uuid4().hex,
    ]
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join((str(root / "src"), str(root)))}

    async def exercise():
        worker = await asyncio.create_subprocess_exec(
            *command,
            phase,
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
            committed = json.loads(line)
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
            output, errors = await asyncio.wait_for(recovery.communicate(line), 30)
            assert recovery.returncode == 0, errors.decode()
            assert json.loads(output) == committed
        finally:
            if recovery.returncode is None:
                recovery.kill()
            await recovery.communicate()

    asyncio.run(exercise())
