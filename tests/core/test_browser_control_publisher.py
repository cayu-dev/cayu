"""Owned browser-control publication with actual persistent-store boundaries."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import pytest
from tests.core._execution_profile_fixtures import create_admitted_session
from tests.core.test_browser_control import identity

from cayu import RunRequest, SQLiteSessionStore
from cayu.runtime import InMemorySessionStore
from cayu.runtime._browser_control_checkpoint import BrowserControlCheckpointMutation
from cayu.runtime._browser_control_publication import BrowserControlPublication
from cayu.runtime._browser_control_publisher import (
    BrowserControlPublicationPending,
    BrowserControlPublisher,
)
from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
from cayu.runtime.browser_control import (
    BrowserControlCheckpoint,
    BrowserControlConflict,
    BrowserControlRecord,
)
from cayu.runtime.sessions import SessionOperationPublication


@asynccontextmanager
async def publication_fixture(backend, tmp_path, *, raw_store=None):
    """Seed admission; an explicitly supplied store remains caller-owned."""
    now = datetime(1970, 1, 1, 0, 0, 1, tzinfo=UTC)
    raw = (
        raw_store
        if raw_store is not None
        else (
            InMemorySessionStore(ownership_clock=lambda: now)
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "publication.sqlite", ownership_clock=lambda: now)
        )
    )
    try:
        admitted = await create_admitted_session(
            raw,
            request=RunRequest(
                agent_name="agent", session_id="session", environment_name="browser", messages=[]
            ),
            provider_name="provider",
            model="model",
        )
        exact = identity().model_copy(
            update={
                "session_instance_id": admitted.session.instance_id,
                "interaction_id": admitted.active_invocation_profile.interaction_id,
                "execution_profile_fingerprint": admitted.active_invocation_profile.profile.fingerprint,
            }
        )
        command = BrowserControlPublication(
            BrowserControlCheckpointMutation(
                "session",
                None,
                BrowserControlCheckpoint(records=(BrowserControlRecord(identity=exact),)),
            )
        )
        yield runtime_checkpoint_session_store(raw), command
    finally:
        if raw_store is None and isinstance(raw, SQLiteSessionStore):
            await raw.close()


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_commit_then_raise_reconciles_without_second_write(backend, tmp_path, monkeypatch):
    async def scenario():
        async with publication_fixture(backend, tmp_path) as (store, command):
            original = store.publish_session_operation_guarded_with_store_time
            writes = []

            async def lost_ack(*args, **kwargs):
                writes.append(1)
                await original(*args, **kwargs)
                raise OSError("commit acknowledgement lost")

            monkeypatch.setattr(
                store, "publish_session_operation_guarded_with_store_time", lost_ack
            )
            publisher = BrowserControlPublisher(store)
            assert await publisher.publish(command) == command.changed_record
            # Reconstruct the owner: no in-process success cache proves replay.
            restarted = BrowserControlPublisher(store)
            assert await restarted.publish(command) == command.changed_record
            assert writes == [1]
            assert publisher.pending_count == 0
            assert await publisher.drain()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_cancel_after_commit_retains_write_until_acknowledgement(backend, tmp_path, monkeypatch):
    async def scenario():
        async with publication_fixture(backend, tmp_path) as (store, command):
            original = store.publish_session_operation_guarded_with_store_time
            committed = asyncio.Event()
            release = asyncio.Event()
            writes = []

            async def delayed_ack(*args, **kwargs):
                writes.append(1)
                result = await original(*args, **kwargs)
                committed.set()
                await release.wait()
                return result

            monkeypatch.setattr(
                store, "publish_session_operation_guarded_with_store_time", delayed_ack
            )
            publisher = BrowserControlPublisher(store)
            task = asyncio.create_task(publisher.publish(command, timeout_s=1))
            await committed.wait()
            assert task.cancel("operator disconnected")
            with pytest.raises(asyncio.CancelledError):
                await task
            assert task.cancelled()
            assert task.cancelling() == 1
            assert publisher.pending_count == 1
            assert not await publisher.drain(timeout_s=0.01)
            with pytest.raises(BrowserControlPublicationPending):
                await publisher.publish(command, timeout_s=0.01)
            assert writes == [1]
            release.set()
            assert await publisher.publish(command) == command.changed_record
            assert publisher.pending_count == 0
            assert await publisher.drain()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_publication_and_readback_errors_are_preserved(backend, tmp_path, monkeypatch):
    async def scenario():
        async with publication_fixture(backend, tmp_path) as (store, command):
            original_read = store.load_session_operation
            primary = OSError("write failed")
            secondary = OSError("readback failed")
            failed = False

            async def write_failure(*args, **kwargs):
                nonlocal failed
                failed = True
                raise primary

            async def read_failure(*args, **kwargs):
                if failed:
                    raise secondary
                return await original_read(*args, **kwargs)

            monkeypatch.setattr(
                store, "publish_session_operation_guarded_with_store_time", write_failure
            )
            monkeypatch.setattr(store, "load_session_operation", read_failure)
            publisher = BrowserControlPublisher(store)
            with pytest.raises(ExceptionGroup) as failure:
                await publisher.publish(command)
            assert failure.value.exceptions == (primary, secondary)
            assert publisher.pending_count == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("use_reserved_key", [False, True])
def test_ordinary_publication_cannot_forge_browser_receipt(backend, use_reserved_key, tmp_path):
    async def scenario():
        async with publication_fixture(backend, tmp_path) as (store, command):

            def forge(_session, checkpoint, _previous):
                return SessionOperationPublication(
                    checkpoint=checkpoint or {},
                    operation_records={command.storage_key: command.receipt()},
                )

            with pytest.raises((BrowserControlConflict, ValueError)):
                await store.publish_session_operation(
                    "session",
                    idempotency_key=command.storage_key if use_reserved_key else "application",
                    operation_transform=forge,
                    events=[],
                )
            assert await store.load_session_operation("session", command.storage_key) is None
            assert await BrowserControlPublisher(store).publish(command) == command.changed_record

    asyncio.run(scenario())


def test_exact_scope_cannot_change_its_receipt_payload(tmp_path):
    async def scenario():
        async with publication_fixture("memory", tmp_path) as (_store, command):
            altered = {**command.receipt(), "schema_version": True}
            with command.scope(), pytest.raises((BrowserControlConflict, ValueError)):
                SessionOperationPublication(
                    checkpoint={}, operation_records={command.storage_key: altered}
                )

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_invocation_rebind_compares_and_preserves_private_browser_root(backend, tmp_path):
    from tests.core._execution_profile_fixtures import rebind_test_invocation

    async def scenario():
        async with publication_fixture(backend, tmp_path) as (store, bootstrap):
            publisher = BrowserControlPublisher(store)
            original = await publisher.publish(bootstrap)
            rebound = await rebind_test_invocation(store, original.identity.session_id)
            assert rebound.run_epoch == original.identity.run_epoch + 1
            checkpoint = await store.load_checkpoint(original.identity.session_id)
            assert checkpoint["browser_controls"] == bootstrap.mutation.desired.model_dump(
                mode="json"
            )
            # Advancing the invocation does not itself grant a new browser owner.
            assert (
                checkpoint["browser_controls"]["records"][0]["identity"]["run_epoch"]
                == original.identity.run_epoch
            )
            assert await publisher.drain()

    asyncio.run(scenario())
