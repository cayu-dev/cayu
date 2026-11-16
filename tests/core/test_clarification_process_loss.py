"""SIGKILL before/after native service return preserves exact responsibility."""

import asyncio
import json
import signal
import sys

import pytest
from tests.core._clarification_recovery_flow import cleanup_in_fresh_process

from cayu.collaboration.base import CollaborationInitialization


async def _create_schema(dsn: str) -> None:
    """Migrate the module's fresh database before the journey under test starts.

    Otherwise the child's first collaboration call creates the whole schema inside
    its bounded acknowledgement wait, and a slow runner reports that wait as
    unavailable before the journey reaches the process-loss barrier.
    """

    from cayu.storage.collaboration_postgres import PostgresCollaborationStore
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresSessionStore

    for store in (
        PostgresCollaborationStore(dsn, schema_mode=SchemaMode.CREATE),
        PostgresSessionStore(dsn, schema_mode=SchemaMode.CREATE),
    ):
        try:
            await store.ensure_schema()
        finally:
            await store.close()


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize("phase", ["released", "dispatched"])
async def test_public_service_process_loss_reconciles_without_redispatch(
    backend, tmp_path, request, phase
):
    if sys.platform == "win32":
        pytest.skip("SIGKILL qualification requires a POSIX process boundary.")
    descriptor = {"backend": backend, "directory": str(tmp_path), "phase": phase}
    if backend == "postgres":
        descriptor["dsn"] = request.getfixturevalue("postgres_dsn")
        await _create_schema(descriptor["dsn"])
    child = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "tests.core._clarification_process_loss_worker",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert child.stdin is not None and child.stdout is not None and child.stderr is not None
    try:
        child.stdin.write(json.dumps(descriptor).encode())
        await child.stdin.drain()
        child.stdin.close()
        line = await asyncio.wait_for(child.stdout.readline(), 240)
        if not line:
            error = await child.stderr.read()
            pytest.fail("Public journey did not reach process-loss barrier: " + error.decode())
        ready = json.loads(line)
        assert ready["pid"] == child.pid and child.returncode is None
        initial = CollaborationInitialization.model_validate(ready["initial"])
        child.kill()
        assert await asyncio.wait_for(child.wait(), 30) == -signal.SIGKILL
        # A second OS process has no original provider, payload, projector or
        # in-memory owner. Existing public discovery/reconciliation authenticates
        # native admission/release and clears exact retained responsibility.
        await cleanup_in_fresh_process(
            initial,
            backend=backend,
            tmp_path=tmp_path,
            request=request,
            kind="unresolved_service" if phase == "dispatched" else "service",
        )
    finally:
        if child.returncode is None:
            child.kill()
        await child.wait()
