"""Native cleanup ACK survives SIGKILL, native deletion and fresh-process recovery."""

import asyncio
import json
import signal
import sys

import pytest

from cayu.collaboration._producer_cleanup_finalization import ProducerCleanupFinalized
from cayu.runtime._producer_cleanup_receipt import NativeProducerCleanupReceipt


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
async def test_cleanup_sigkill_after_native_delete_recovers_without_dispatch(
    backend, tmp_path, request
):
    address = (
        str(tmp_path / "collaboration.sqlite")
        if backend == "sqlite"
        else request.getfixturevalue("postgres_dsn")
    )
    material = {"backend": backend, "address": address, "crash": False, "cleanup_crash": True}
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
        committed = await asyncio.wait_for(producer.stdout.readline(), 240)
        if not committed:
            _, error = await asyncio.wait_for(producer.communicate(), 10)
            pytest.fail(error.decode())
        result = json.loads(committed)
        assert result["provider_calls"] == 1
        native = NativeProducerCleanupReceipt.model_validate(result["native"])
        assert "private-process-loss-canary" not in committed.decode()
        producer.kill()
        await asyncio.wait_for(producer.wait(), 10)
        assert producer.returncode == -signal.SIGKILL
    finally:
        if producer.returncode is None:
            producer.kill()
            await producer.wait()
    material = {
        "backend": backend,
        "address": address,
        "expected": result["expected"],
        "cleanup": True,
    }
    first = None
    for _ in range(2):
        reader = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "tests.recovery.producer_completion_reader_worker",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            output, error = await asyncio.wait_for(
                reader.communicate(json.dumps(material).encode()), 90
            )
            assert reader.returncode == 0, error.decode()
            final = ProducerCleanupFinalized.model_validate_json(output)
            assert final.native_receipt == native
            assert final.delivery == "excluded"
            if first is not None:
                assert final == first
            first = final
        finally:
            if reader.returncode is None:
                reader.kill()
                await reader.wait()
