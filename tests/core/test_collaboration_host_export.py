"""Nested export ownership must outlive a cancelled host observer."""

import asyncio

import pytest
from tests.core.producer_export_scenario import completed_export_scenario
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_prepared_admission_public import native_stores as native_stores

from cayu.collaboration._host import (
    CollaborationHost,
    _HostRegistration,
    _ProducerMaintenanceRule,
    _ProducerSource,
)
from cayu.collaboration._host_ownership import HostOwnershipLimits
from cayu.collaboration._host_producer_maintenance import HostProducerMaintenance
from cayu.collaboration._session_export_store import ExportRecord

pytestmark = pytest.mark.anyio


async def test_host_export_retains_nested_owner_after_publication(native_stores, monkeypatch):
    values, access = await completed_export_scenario(native_stores, monkeypatch)
    application, _, _, provider, _, _, command, _ = values
    participant = command.admission.prepared.recipient
    page = await application.pending_producer_outputs(participant, context=CONTEXT)
    token = next(
        item.recovery for item in page.items if item.recovery.registration == command.operation
    )
    destination = command.destinations[0].operation
    exports = application._session_export_coordinator
    publish = exports.publish
    entered, release = asyncio.Event(), asyncio.Event()
    commits = 0

    async def blocked_ack(session, before, after, key, record, *args, **kwargs):
        nonlocal commits
        result = await publish(session, before, after, key, record, *args, **kwargs)
        if type(record) is dict and "payload_json" in record and not commits:
            ExportRecord.model_validate(record)
            commits += 1
            entered.set()
            await release.wait()
        return result

    monkeypatch.setattr(exports, "publish", blocked_ack)
    # Both independent observation limits must be bypassed by retained host
    # ownership. Fixing only the outer request owner would still lose the ACK.
    monkeypatch.setattr(exports.owners, "observation_timeout", 0.02)
    from cayu.collaboration import _host_producer_maintenance

    export = _host_producer_maintenance.export_producer_output

    async def short_outer_observation(*args, **kwargs):
        owners = application._request_coordinator._owners
        previous = owners.observation_timeout
        owners.observation_timeout = 0.02
        try:
            return await export(*args, **kwargs)
        finally:
            owners.observation_timeout = previous

    monkeypatch.setattr(
        _host_producer_maintenance, "export_producer_output", short_outer_observation
    )
    host = CollaborationHost(
        application,
        _HostRegistration(
            limits=HostOwnershipLimits(1, 1, 2, 262144),
            producer_sources=(_ProducerSource(participant, CONTEXT),),
            producer_rules=(
                _ProducerMaintenanceRule(
                    HostProducerMaintenance(
                        recovery=token, action="export", destination=destination
                    ),
                    CONTEXT,
                    access,
                ),
            ),
            observation_timeout_s=60,
            shutdown_timeout_s=0.01,
        ),
    )
    running = asyncio.create_task(host.run())
    try:
        try:
            async with asyncio.timeout(120):
                while not entered.is_set():
                    if running.done():
                        await running
                    for outcome in host._owned.inspect().completed:
                        if outcome.error is not None:
                            raise outcome.error
                    await asyncio.sleep(0.05)
        except TimeoutError as error:
            error.add_note(f"Host export state: {host.inspect()}")
            for pending in host._owned._operations.values():
                coroutine = pending.task.get_coro()
                path = []
                while coroutine is not None:
                    code = getattr(coroutine, "cr_code", None)
                    if code is not None:
                        path.append(code.co_qualname)
                    coroutine = getattr(coroutine, "cr_await", None)
                error.add_note("Retained await path: " + " -> ".join(path))
            for owner in (application._request_coordinator._owners, exports.owners):
                for pending in tuple(owner.pending):
                    coroutine = pending.get_coro()
                    path = []
                    while coroutine is not None:
                        code = getattr(coroutine, "cr_code", None)
                        if code is not None:
                            path.append(code.co_qualname)
                        coroutine = getattr(coroutine, "cr_await", None)
                    error.add_note("Native await path: " + " -> ".join(path))
            raise
        # The host's idle wait may have a timeout-owned cancellation pending.
        # Count our requests separately; that timeout must remove its own request
        # when the observer unwinds, retaining both external cancellations.
        prior_cancellations = running.cancelling()
        assert running.cancel()
        assert running.cancel()
        assert running.cancelling() == prior_cancellations + 2
        with pytest.raises(asyncio.CancelledError):
            await running
        assert running.cancelled()
        assert running.cancelling() == 2
        await asyncio.sleep(0.05)
        state = await host.aclose()
        assert state.uncertain == 1 and state.failed == 0
        assert len(provider.requests) == 1
    finally:
        release.set()
        async with asyncio.timeout(90):
            while (await host.aclose()).pending:
                for outcome in host._owned.inspect().completed:
                    if outcome.error is not None:
                        raise outcome.error
                await asyncio.sleep(0.01)
        if not running.done():
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
    monkeypatch.setattr(exports.owners, "observation_timeout", 60)
    receipt = await application.export_producer_output(command, destination, context=access)
    assert await application.export_producer_output(command, destination, context=access) == receipt
    assert commits == 1
    assert len(provider.requests) == 1
    assert host.inspect().uncertain == 0
