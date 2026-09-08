"""Actual worker termination after durable input admission, before acknowledgement.

Native delivery is deliberately unknown: no input receipt or quiescence is invented.
"""

import asyncio
import multiprocessing
from datetime import UTC, datetime

import pytest
from tests.core.test_browser_control_authorization import Policy
from tests.core.test_browser_control_coordinator import coordinator, intent_for
from tests.core.test_browser_control_publisher import publication_fixture

from cayu import SQLiteSessionStore
from cayu.runtime._browser_control_checkpoint import browser_control_checkpoint_read_scope
from cayu.runtime._browser_control_model import browser_model_control_admission
from cayu.runtime._browser_control_publisher import BrowserControlPublisher
from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
from cayu.runtime.browser_control import (
    BrowserControlAllocation,
    BrowserControlConflict,
    BrowserControlPrincipal,
    BrowserControlRecord,
    BrowserSensitiveEntryIntent,
    BrowserTextInputIntent,
)


def _pending_worker(directory, connection):
    async def scenario():
        async with publication_fixture("sqlite", directory) as (store, bootstrap):
            await BrowserControlPublisher(store).publish(bootstrap)
            owner = coordinator(store, Policy(True))
            principal = BrowserControlPrincipal(subject="operator")
            pending = await owner.request_takeover(
                principal=principal,
                operator_session_id="server-session",
                intent=intent_for(bootstrap),
            )
            acquired = await owner._publish_guest_acquisition(expected=pending, lease_until_ms=2000)
            sensitive = await owner.request_sensitive_entry(
                principal=principal,
                operator_session_id="server-session",
                intent=BrowserSensitiveEntryIntent(
                    identity=acquired.identity,
                    request_id=acquired.request.request_id,
                    expected_record_revision=acquired.revision,
                    expected_control_epoch=acquired.control_epoch,
                ),
            )
            acquired = await owner._publish_guest_sensitive_entry(expected=sensitive)
            intent = BrowserTextInputIntent(
                identity=acquired.identity,
                request_id=acquired.request.request_id,
                expected_record_revision=acquired.revision,
                expected_control_epoch=acquired.control_epoch,
                input_sequence=1,
                page=acquired.request.pages[0],
                input_kind="tab",
            )
            reserved = await owner._admit_text_input(
                principal=principal, operator_session_id="server-session", intent=intent
            )
            connection.send((reserved.model_dump(mode="json"), intent.model_dump(mode="json")))
            await asyncio.Event().wait()

    asyncio.run(scenario())


def test_killed_input_owner_stays_fenced_after_sqlite_reconstruction(tmp_path):
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    worker = context.Process(target=_pending_worker, args=(tmp_path, sender))
    worker.start()
    sender.close()
    try:
        assert receiver.poll(30), "Worker did not reach durable input admission."
        raw_record, raw_intent = receiver.recv()
        assert worker.is_alive()
        worker.kill()
        worker.join(10)
        assert not worker.is_alive() and worker.exitcode != 0
    finally:
        if worker.is_alive():
            worker.kill()
            worker.join(10)
        receiver.close()

    async def recover():
        raw = SQLiteSessionStore(
            tmp_path / "publication.sqlite",
            ownership_clock=lambda: datetime(1970, 1, 1, 0, 0, 1, tzinfo=UTC),
        )
        try:
            store = runtime_checkpoint_session_store(raw)
            owner = coordinator(store, Policy(True))
            expected = BrowserControlRecord.model_validate(raw_record)
            intent = BrowserTextInputIntent.model_validate(raw_intent)
            assert (await owner._load(expected.identity))[1] == expected
            assert expected.pending_input_sequence == 1 and expected.settled_input_sequence == 0
            assert expected.operator_page_operations[0].operations == 1
            with pytest.raises(BrowserControlConflict):
                await owner._admit_text_input(
                    principal=BrowserControlPrincipal(subject="operator"),
                    operator_session_id="server-session",
                    intent=intent,
                )
            with browser_control_checkpoint_read_scope(expected.identity.session_id):
                checkpoint = await store.load_checkpoint(expected.identity.session_id)
            allocation = BrowserControlAllocation.model_validate(
                expected.identity.model_dump(exclude={"worker_instance_id"})
            )
            with pytest.raises(BrowserControlConflict):
                browser_model_control_admission(
                    checkpoint, allocation=allocation, operation_name="observe"
                )
            fenced = await owner.mark_channel_uncertain(expected=expected)
            assert fenced.pending_input_sequence == 1
            assert fenced.operator_page_operations == expected.operator_page_operations
            assert fenced.state != "agent_controlled"
        finally:
            await raw.close()

    asyncio.run(recover())
