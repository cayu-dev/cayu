"""SIGKILL qualification of the composed producer/delivery/wait host journey."""

import asyncio
import json
import signal
import sys

import pytest


@pytest.mark.anyio
@pytest.mark.qualification
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize(
    "boundary", ["before-delivery", "after-delivery", "after-continuation-admission"]
)
async def test_host_journey_recovers_delivery_and_continuation_in_fresh_process(
    backend, boundary, tmp_path, request
):
    address = (
        str(tmp_path / "collaboration.sqlite")
        if backend == "sqlite"
        else request.getfixturevalue("postgres_dsn")
    )
    material = {"backend": backend, "address": address, "boundary": boundary}

    async def launch():
        return await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "tests.recovery.collaboration_host_journey_worker",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=1024 * 1024,
        )

    producer = await launch()
    errors = asyncio.create_task(producer.stderr.read())
    try:
        producer.stdin.write(json.dumps(material).encode())
        await producer.stdin.drain()
        producer.stdin.close()
        # Real public setup includes two serialized provider calls, independent
        # ownership stores, retained output, and authenticated disclosure.
        try:
            snapshot = await asyncio.wait_for(producer.stdout.readline(), 900)
        except TimeoutError:
            producer.kill()
            await asyncio.wait_for(producer.wait(), 15)
            pytest.fail(
                "Producer did not reach its process-loss barrier:\n" + (await errors).decode()
            )
        if not snapshot:
            await producer.wait()
            pytest.fail((await errors).decode())
        retained = json.loads(snapshot)
        (tmp_path / "recovery-material.json").write_text(snapshot.decode(), encoding="utf-8")
        assert retained["provider_calls"] == 2
        producer.kill()
        await asyncio.wait_for(producer.wait(), 15)
        assert producer.returncode == -signal.SIGKILL
    finally:
        if producer.returncode is None:
            producer.kill()
            await producer.wait()
        await errors

    recovery = await launch()
    communication = asyncio.create_task(recovery.communicate(json.dumps(retained).encode()))
    try:
        try:
            output, error = await asyncio.wait_for(asyncio.shield(communication), 600)
        except TimeoutError:
            recovery.kill()
            _, error = await asyncio.wait_for(communication, 15)
            pytest.fail("Fresh-process recovery exceeded its bound:\n" + error.decode())
        assert recovery.returncode == 0, error.decode()
        assert json.loads(output) == {
            "provider_calls": 0 if boundary == "after-continuation-admission" else 1,
            "continuation_status": (
                "interrupted" if boundary == "after-continuation-admission" else "completed"
            ),
            "producer_redispatched": False,
            "continuation": "consumed",
            "delivery": "appended",
            "cleanup": "settled",
        }
    finally:
        if recovery.returncode is None:
            recovery.kill()
            await recovery.wait()
        await communication
