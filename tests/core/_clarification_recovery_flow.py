"""Real observer cancellation while receiving settlement remains owned."""

import asyncio
import json
import os
import sys

import pytest

from cayu.collaboration._request_store import operation_key
from cayu.collaboration.participants import CollaborationUnavailable


async def cleanup_in_fresh_process(
    initial, *, backend, tmp_path, request, kind="service", expiry=None
):
    value = {
        "parent_pid": os.getpid(),
        "backend": backend,
        "initial": initial.model_dump(mode="json"),
        "kind": kind,
    }
    if expiry is not None:
        value["expiry"] = expiry.model_dump(mode="json")
    if backend == "sqlite":
        value.update(
            source_path=str(tmp_path / "prepared-collaboration.sqlite"),
            session_path=str(tmp_path / "creation-fence.sqlite"),
        )
    else:
        value["dsn"] = request.getfixturevalue("postgres_dsn")
    child = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "tests.core._clarification_public_recovery_process",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(child.communicate(json.dumps(value).encode()), 90)
        assert child.returncode == 0, stderr.decode()
        assert stdout == b""
    finally:
        if child.returncode is None:
            child.kill()
            await child.wait()


async def cancel_cleanup_observer(app, collaboration, recovery, *, context, monkeypatch):
    entered = asyncio.Event()
    release = asyncio.Event()
    original = type(collaboration)._settle_permit

    async def held_ack(self, initialized, expected, **kwargs):
        result = await original(self, initialized, expected, **kwargs)
        if expected.intent.request.effect_scope == "clarification_service":
            entered.set()
            await release.wait()
        return result

    with monkeypatch.context() as patch:
        patch.setattr(type(collaboration), "_settle_permit", held_ack)
        observer = asyncio.create_task(
            app.reconcile_clarification_service(recovery, context=context)
        )
        ready = asyncio.create_task(entered.wait())
        try:
            await asyncio.wait_for(
                asyncio.wait((observer, ready), return_when=asyncio.FIRST_COMPLETED), 30
            )
            if not entered.is_set():
                await observer
                pytest.fail("Reconciliation did not reach the committed permit boundary.")
            observer.cancel()
            observer.cancel()
            assert observer.cancelling() == 2
            with pytest.raises(asyncio.CancelledError):
                await observer
            assert observer.cancelled() and observer.cancelling() == 2
            async with collaboration._transaction(
                recovery.operation.application_scope, write=False
            ) as tx:
                debt = await tx.get("clarification_services", operation_key(recovery.operation))
                assert debt["state"] == "pending"
            # Cancellation ended observation, not retained mutation ownership.
            # A competing exact retry may join/reconcile, never re-dispatch.
            release.set()
            async with asyncio.timeout(120):
                while True:
                    try:
                        result = await app.reconcile_clarification_service(
                            recovery, context=context
                        )
                        break
                    except CollaborationUnavailable:
                        await asyncio.sleep(0.05)
            assert result.state == "returned"
        finally:
            release.set()
            ready.cancel()
            if not observer.done():
                observer.cancel()
            await asyncio.gather(observer, ready, return_exceptions=True)
