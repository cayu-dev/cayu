"""Process-loss recovery settles original producer ownership without redispatch."""

import asyncio
import json
import signal
import sys

import pytest


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize(
    ("before_output", "recovery_mode", "human_gate"),
    [
        (False, "complete", False),
        (True, "complete", False),
        (True, "plan", False),
        (True, "host", False),
        (True, "continue", False),
        (True, "cancel", False),
        (True, "queue", False),
        (True, "complete", True),
    ],
)
async def test_sigkill_producer_native_recovery_settles_original_invocation(
    backend, before_output, recovery_mode, human_gate, tmp_path, request
):
    address = (
        str(tmp_path / "collaboration.sqlite")
        if backend == "sqlite"
        else request.getfixturevalue("postgres_dsn")
    )
    material = {
        "backend": backend,
        "address": address,
        "native_recovery": True,
        "before_output": before_output,
        "recovery_mode": recovery_mode,
        "human_gate": human_gate,
    }
    producer = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "tests.recovery.producer_output_process_worker",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=256 * 1024,
    )
    try:
        producer.stdin.write(json.dumps(material).encode())
        await producer.stdin.drain()
        producer.stdin.close()
        committed = await asyncio.wait_for(producer.stdout.readline(), 90)
        if not committed:
            _, error = await asyncio.wait_for(producer.communicate(), 10)
            pytest.fail(error.decode())
        snapshot = json.loads(committed)
        assert snapshot["provider_calls"] == (2 if human_gate else 1)
        producer.kill()
        await asyncio.wait_for(producer.wait(), 10)
        assert producer.returncode == -signal.SIGKILL
    finally:
        if producer.returncode is None:
            producer.kill()
            await producer.wait()
    material["expected"] = snapshot["expected"]
    recovery = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "tests.recovery.producer_native_recovery_worker",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    communication = asyncio.create_task(recovery.communicate(json.dumps(material).encode()))
    try:
        try:
            output, error = await asyncio.wait_for(asyncio.shield(communication), 90)
        except TimeoutError:
            recovery.kill()
            output, error = await asyncio.wait_for(communication, 10)
            pytest.fail("Recovery worker exceeded its deadline:\n" + error.decode())
        assert recovery.returncode == 0, error.decode()
        result = json.loads(output)
        assert result["provider_calls"] == 0
        source_epoch = (
            snapshot["source_run_epoch"] if before_output else snapshot["output"]["run_epoch"]
        )
        interaction = (
            snapshot["interaction_id"] if before_output else snapshot["output"]["interaction_id"]
        )
        assert result["release"]["run_epoch"] > source_epoch
        assert result["release"]["interaction_id"] == interaction
        assert result["accounting"]["reservation_count"] == len(snapshot["reservations"])
        assert result["finalized"]["native_receipt"]["run_epoch"] == result["release"]["run_epoch"]
    finally:
        if recovery.returncode is None:
            recovery.kill()
            await recovery.wait()
