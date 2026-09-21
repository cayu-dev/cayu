from __future__ import annotations

import asyncio
import warnings
from contextlib import asynccontextmanager

import pytest
from tests.artifacts.test_resources import make_store, registered_resource, registered_transfer

from cayu.artifacts.resources import ResourceOwnerUnavailable
from cayu.vaults.redaction import SecretRedactor


@pytest.mark.parametrize("cancel", [False, True, "race"])
@pytest.mark.parametrize(
    "phase",
    [
        "authorize",
        "acquire",
        "acquire_registration",
        "transfer",
        "transfer_registration",
        "acquire_reconcile",
        "transfer_reconcile",
    ],
)
def test_stopped_preparation_cannot_dispatch_after_late_authority(
    tmp_path, monkeypatch, phase, cancel
):
    import cayu.artifacts.resources as resources

    store, artifact = make_store(tmp_path)

    async def run():
        (
            source,
            command,
            permit,
            resolver,
            collaboration,
            initialized,
            participant,
        ) = await registered_resource(tmp_path / "source", store, artifact)
        destination_store = None
        source_receipt = None
        resource = source
        try:
            preparation = None
            if phase != "authorize":
                preparation = await source.authorize(command, permit=permit)
            if phase.startswith("transfer"):
                source_receipt = await source.acquire(command, preparation=preparation)
                (
                    resource,
                    command,
                    _,
                    resolver,
                    destination_store,
                    initialized,
                    participant,
                ) = await registered_transfer(
                    tmp_path / "destination", store, artifact, source, source_receipt
                )
            entered, finish = asyncio.Event(), asyncio.Event()
            ledger = destination_store or collaboration
            if phase.endswith("reconcile"):
                original_register = ledger._register_permit

                async def fail_after_register(*args, **kwargs):
                    await original_register(*args, **kwargs)
                    raise OSError("registration acknowledgement lost")

                with monkeypatch.context() as setup_patch:
                    setup_patch.setattr(ledger, "_register_permit", fail_after_register)
                    with pytest.raises(OSError):
                        if phase.startswith("transfer"):
                            await resource.accept_transfer(command, source_owner=source)
                        else:
                            await resource.acquire(command, preparation=preparation)
            original_timeout = resources.RESOURCE_FOREGROUND_TIMEOUT_S
            original_pin = store._pin_resource
            pins = []

            async def pin(*args, **kwargs):
                pins.append(args)
                return await original_pin(*args, **kwargs)

            monkeypatch.setattr(store, "_pin_resource", pin)
            if phase.endswith("registration"):
                original_register = ledger._register_permit

                async def register(*args, **kwargs):
                    result = await original_register(*args, **kwargs)
                    entered.set()
                    await finish.wait()
                    return result

                monkeypatch.setattr(ledger, "_register_permit", register)
            else:
                original_resolve = resolver.acquire

                @asynccontextmanager
                async def resolve(context):
                    entered.set()
                    await finish.wait()
                    async with original_resolve(context) as resolution:
                        yield resolution

                monkeypatch.setattr(resolver, "acquire", resolve)
            monkeypatch.setattr(resources, "RESOURCE_FOREGROUND_TIMEOUT_S", 5.0 if cancel else 1.0)
            if phase.endswith("reconcile"):
                operation = resource.reconcile(
                    source_owners=(source,) if phase.startswith("transfer") else ()
                )
            elif phase == "authorize":
                operation = resource.authorize(command, permit=permit)
            elif phase.startswith("transfer"):
                operation = resource.accept_transfer(command, source_owner=source)
            else:
                operation = resource.acquire(command, preparation=preparation)
            task = asyncio.create_task(operation)
            try:
                async with asyncio.timeout(10):
                    await entered.wait()
                    if cancel:
                        task.cancel("first cancellation")
                        task.cancel("second cancellation")
                        if cancel == "race":
                            finish.set()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                        else:
                            pytest.fail("Caller cancellation was lost")
                        assert task.cancelled()
                        assert task.cancelling() == 2
                    else:
                        with pytest.raises(ResourceOwnerUnavailable):
                            await task
                        assert not task.cancelled()
                        assert task.cancelling() == 0
                    if cancel != "race":
                        assert resource._workers
                        with pytest.raises(ResourceOwnerUnavailable):
                            await resource.reconcile()
                    assert not pins
                    monkeypatch.setattr(
                        resources, "RESOURCE_FOREGROUND_TIMEOUT_S", original_timeout
                    )
                    finish.set()
                    if cancel == "race":
                        try:
                            await resource.drain()
                        except ResourceOwnerUnavailable as stopped:
                            assert "observer stopped" in str(stopped)
                    elif phase.endswith("reconcile"):
                        await resource.drain()
                    else:
                        with pytest.raises(ResourceOwnerUnavailable, match="observer stopped"):
                            await resource.drain()
                assert not pins
                assert not resource._workers
                _, obligations = await ledger.scan_obligations(
                    initialized,
                    participant,
                    after=0,
                    limit=64,
                    pending_only=True,
                    retention_revision=None,
                    redactor=SecretRedactor(),
                )
                assert not obligations
                with resource._journal.locked() as journal:
                    records = journal["transfers" if phase.startswith("transfer") else "operations"]
                    if phase.endswith(("registration", "reconcile")):
                        assert len(records) == 1
                        record = next(iter(records.values()))
                        assert record["stage"] == "released"
                        assert record["responsibility_settled"] is True
                    else:
                        assert not records
                    if phase == "authorize":
                        assert not journal["authorizations"]
                if source_receipt is not None:
                    await source.release(source_receipt)
                await store.delete(artifact.id)
            finally:
                finish.set()
                monkeypatch.setattr(resources, "RESOURCE_FOREGROUND_TIMEOUT_S", original_timeout)
                if not task.done():
                    await task
        finally:
            await collaboration.close()
            if destination_store is not None:
                await destination_store.close()

    asyncio.run(run())


@pytest.mark.parametrize("diagnostic_failure", [False, True])
@pytest.mark.parametrize("cleanup_failure", [False, True])
@pytest.mark.parametrize("drain_cancel", [False, True])
def test_cancelled_observer_preserves_late_primary_diagnostic_and_cleanup_failures(
    tmp_path, monkeypatch, capsys, caplog, diagnostic_failure, cleanup_failure, drain_cancel
):
    store, artifact = make_store(tmp_path)
    primary = OSError("private-primary-canary")
    diagnostic = RuntimeError("private-diagnostic-canary")
    cleanup = OSError("private-cleanup-canary")

    def leaves(error):
        if isinstance(error, BaseExceptionGroup):
            return [leaf for child in error.exceptions for leaf in leaves(child)]
        return [error]

    async def run():
        (
            resource,
            command,
            permit,
            resolver,
            collaboration,
            initialized,
            participant,
        ) = await registered_resource(tmp_path, store, artifact)
        try:
            preparation = await resource.authorize(command, permit=permit)
            entered, finish = asyncio.Event(), asyncio.Event()
            cleanup_entered, cleanup_finish = asyncio.Event(), asyncio.Event()
            original_read = store.read_bytes
            original_diagnostic = resource._set_uncertain
            original_release = store._release_resource_pin

            async def read(*args, **kwargs):
                await original_read(*args, **kwargs)
                entered.set()
                await finish.wait()
                raise primary

            async def fail_diagnostic(*args, **kwargs):
                raise diagnostic

            async def release(*args, **kwargs):
                await original_release(*args, **kwargs)
                if drain_cancel:
                    cleanup_entered.set()
                    await cleanup_finish.wait()
                if cleanup_failure:
                    raise cleanup  # Real unpin succeeded; the acknowledgement was lost.

            monkeypatch.setattr(store, "read_bytes", read)
            if diagnostic_failure:
                monkeypatch.setattr(resource, "_set_uncertain", fail_diagnostic)
            if cleanup_failure or drain_cancel:
                monkeypatch.setattr(store, "_release_resource_pin", release)
            task = asyncio.create_task(resource.acquire(command, preparation=preparation))
            try:
                async with asyncio.timeout(10):
                    await entered.wait()
                    task.cancel("first")
                    task.cancel("second")
                    with pytest.raises(asyncio.CancelledError):
                        await task
                    assert task.cancelled()
                    assert task.cancelling() == 2
                    with pytest.raises(ValueError):
                        await store.delete(artifact.id)
                    with pytest.raises(ResourceOwnerUnavailable):
                        await resource.reconcile()
                    finish.set()
                    if drain_cancel:
                        await cleanup_entered.wait()
                        waiter = asyncio.create_task(resource.drain())
                        await asyncio.sleep(0)
                        waiter.cancel("new cancellation during cleanup")
                        with pytest.raises(asyncio.CancelledError):
                            await waiter
                        assert waiter.cancelled()
                        assert waiter.cancelling() == 1
                        with pytest.raises(ResourceOwnerUnavailable):
                            await resource.reconcile()
                        cleanup_finish.set()
                    with pytest.raises((OSError, ExceptionGroup)) as raised:
                        await resource.drain()
                assert leaves(raised.value) == (
                    [primary]
                    + ([diagnostic] if diagnostic_failure else [])
                    + ([cleanup] if cleanup_failure else [])
                )
                assert not isinstance(raised.value, asyncio.CancelledError)
                with resource._journal.locked() as journal:
                    record = next(iter(journal["operations"].values()))
                    assert record["stage"] == ("cleaning" if cleanup_failure else "released")
                journal_text = resource._journal.path.read_text()
                assert "private-" not in journal_text
                monkeypatch.setattr(store, "read_bytes", original_read)
                monkeypatch.setattr(resource, "_set_uncertain", original_diagnostic)
                monkeypatch.setattr(store, "_release_resource_pin", original_release)
                resolver.revoked = True
                await resource.reconcile()
                _, obligations = await collaboration.scan_obligations(
                    initialized,
                    participant,
                    after=0,
                    limit=64,
                    pending_only=True,
                    retention_revision=None,
                    redactor=SecretRedactor(),
                )
                assert not obligations
                await store.delete(artifact.id)
            finally:
                finish.set()
                cleanup_finish.set()
                if not task.done():
                    await task
        finally:
            await collaboration.close()

    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        asyncio.run(run())
    assert not emitted
    assert not caplog.records
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


@pytest.mark.parametrize("entrance", ["authorize", "acquire", "release"])
@pytest.mark.parametrize("representation", ["raw", "mutated"])
def test_public_resource_rejection_does_not_render_unsafe_input(
    tmp_path, capsys, caplog, entrance, representation
):
    from cayu.collaboration._contracts import CollaborationContractError

    store, artifact = make_store(tmp_path)
    canary = "private-resource-input-canary"
    rendered = []

    class Unsafe:
        def __repr__(self):
            rendered.append("repr")
            return canary

        def __str__(self):
            rendered.append("str")
            return canary

    async def run():
        resource, command, permit, _, collaboration, *_ = await registered_resource(
            tmp_path, store, artifact
        )
        try:
            prep = await resource.authorize(command, permit=permit)
            receipt = None
            if entrance == "release":
                receipt = await resource.acquire(command, preparation=prep)
            value = receipt if receipt is not None else command
            if representation == "raw":
                invalid = value.model_dump(mode="json")
                nested = invalid["command"] if receipt is not None else invalid
                nested["intent"]["max_total_bytes"] = True
                nested["initiator"]["principal"] = canary
            else:
                bad_command = command.model_copy(
                    update={
                        "intent": command.intent.model_copy(update={"max_total_bytes": Unsafe()})
                    }
                )
                invalid = (
                    receipt.model_copy(update={"command": bad_command})
                    if receipt is not None
                    else bad_command
                )
            before = resource._journal.path.read_bytes()
            with pytest.raises(CollaborationContractError) as rejected:
                if entrance == "authorize":
                    await resource.authorize(invalid, permit=permit)
                elif entrance == "acquire":
                    await resource.acquire(invalid, preparation=prep)
                else:
                    await resource.release(invalid)
            assert canary not in str(rejected.value)
            assert canary not in repr(rejected.value)
            assert not rendered
            assert resource._journal.path.read_bytes() == before
            # Rejection must neither alter the original command nor release retention.
            if receipt is None:
                receipt = await resource.acquire(command, preparation=prep)
            with pytest.raises(ValueError):
                await store.delete(artifact.id)
            await resource.release(receipt)
            await store.delete(artifact.id)
        finally:
            await collaboration.close()

    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        asyncio.run(run())
    assert not emitted
    assert not caplog.records
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_previously_handled_cancellation_does_not_stop_new_preparation(tmp_path):
    store, artifact = make_store(tmp_path)

    async def run():
        resource, command, permit, _, collaboration, *_ = await registered_resource(
            tmp_path, store, artifact
        )
        try:
            current = asyncio.current_task()
            current.cancel("previously handled request")
            with pytest.raises(asyncio.CancelledError):
                await asyncio.sleep(0)
            assert current.cancelling() == 1
            preparation = await resource.authorize(command, permit=permit)
            receipt = await resource.acquire(command, preparation=preparation)
            assert current.cancelling() == 1
            await resource.release(receipt)
            await store.delete(artifact.id)
        finally:
            await collaboration.close()

    asyncio.run(run())


@pytest.mark.parametrize("representation", ["serialized", "typed"])
@pytest.mark.parametrize("phase", ["registration", "pin"])
@pytest.mark.parametrize("termination", ["success", "cancel", "timeout"])
def test_transfer_cleanup_uses_detached_command(
    tmp_path, monkeypatch, representation, phase, termination
):
    import cayu.artifacts.resources as resources

    store, artifact = make_store(tmp_path)

    async def run():
        source, command, permit, _, collaboration, *_ = await registered_resource(
            tmp_path / "source", store, artifact
        )
        destination_store = None
        try:
            receipt = await source.acquire(
                command, preparation=await source.authorize(command, permit=permit)
            )
            (
                destination,
                transfer,
                _,
                _,
                destination_store,
                initialized,
                participant,
            ) = await registered_transfer(
                tmp_path / "destination", store, artifact, source, receipt
            )
            payload = (
                transfer.model_dump(mode="json")
                if representation == "serialized"
                else transfer.model_copy(deep=True)
            )
            entered, finish = asyncio.Event(), asyncio.Event()
            pins = []
            original_pin = store._pin_resource
            original_register = destination_store._register_permit

            async def pin(*args, **kwargs):
                await original_pin(*args, **kwargs)
                pins.append(args)
                if phase == "pin":
                    entered.set()
                    await finish.wait()

            async def register(*args, **kwargs):
                result = await original_register(*args, **kwargs)
                if phase == "registration":
                    entered.set()
                    await finish.wait()
                return result

            monkeypatch.setattr(store, "_pin_resource", pin)
            monkeypatch.setattr(destination_store, "_register_permit", register)
            original_timeout = resources.RESOURCE_FOREGROUND_TIMEOUT_S
            monkeypatch.setattr(
                resources, "RESOURCE_FOREGROUND_TIMEOUT_S", 1.0 if termination == "timeout" else 5.0
            )
            task = asyncio.create_task(destination.accept_transfer(payload, source_owner=source))
            try:
                async with asyncio.timeout(10):
                    await entered.wait()
                    # Mutation happens after validation and real registration/pin.
                    # Neither dispatch nor cleanup may return to this raw input.
                    if representation == "serialized":
                        payload["operation"]["caller_key"] = "unrelated-operation"
                        payload["intent"]["acceptance_generation"] += 1
                    else:
                        object.__setattr__(
                            payload,
                            "operation",
                            payload.operation.model_copy(
                                update={"caller_key": "unrelated-operation"}
                            ),
                        )
                        object.__setattr__(
                            payload.intent,
                            "acceptance_generation",
                            payload.intent.acceptance_generation + 1,
                        )
                    if termination == "cancel":
                        task.cancel("first")
                        task.cancel("second")
                        with pytest.raises(asyncio.CancelledError):
                            await task
                        assert task.cancelled()
                        assert task.cancelling() == 2
                    elif termination == "timeout":
                        with pytest.raises(ResourceOwnerUnavailable):
                            await task
                        assert not task.cancelled()
                    if termination != "success":
                        with pytest.raises(ResourceOwnerUnavailable):
                            await destination.reconcile(source_owners=(source,))
                    monkeypatch.setattr(
                        resources, "RESOURCE_FOREGROUND_TIMEOUT_S", original_timeout
                    )
                    finish.set()
                    if termination == "success":
                        accepted = await task
                        assert accepted.command == transfer
                        assert (await destination.read_transfer(transfer)).receipt == accepted
                        await destination.release_transfer(accepted)
                    else:
                        with pytest.raises(ResourceOwnerUnavailable, match="observer stopped"):
                            await destination.drain()
                    with destination._journal.locked() as journal:
                        assert set(journal["transfers"]) == {
                            resources.resource_operation_digest(transfer)
                        }
                        record = next(iter(journal["transfers"].values()))
                        assert record["stage"] == "released"
                        assert record["responsibility_settled"] is True
                    assert len(pins) == (1 if phase == "pin" or termination == "success" else 0)
                    _, obligations = await destination_store.scan_obligations(
                        initialized,
                        participant,
                        after=0,
                        limit=64,
                        pending_only=True,
                        retention_revision=None,
                        redactor=SecretRedactor(),
                    )
                    assert not obligations
                    with pytest.raises(ValueError):
                        await store.delete(artifact.id)  # Source retention remains intact.
                    await source.release(receipt)
                    await store.delete(artifact.id)
            finally:
                finish.set()
                monkeypatch.setattr(resources, "RESOURCE_FOREGROUND_TIMEOUT_S", original_timeout)
                if not task.done():
                    await task
        finally:
            await collaboration.close()
            if destination_store is not None:
                await destination_store.close()

    asyncio.run(run())
