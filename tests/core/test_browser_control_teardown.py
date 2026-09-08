"""Cleanup-only publication cannot restore ended browser authority."""

import asyncio

import pytest
from tests.core.test_browser_control_authorization import Policy
from tests.core.test_browser_control_coordinator import coordinator, intent_for
from tests.core.test_browser_control_publisher import publication_fixture

from cayu.runtime._browser_control_checkpoint import BrowserControlCheckpointMutation
from cayu.runtime._browser_control_publication import (
    BrowserControlFencePublication,
    BrowserControlPublication,
)
from cayu.runtime._browser_control_publisher import BrowserControlPublisher
from cayu.runtime.browser_control import (
    BrowserControlConflict,
    BrowserControlPrincipal,
    BrowserSensitiveEntryIntent,
    BrowserTextInputIntent,
)
from cayu.runtime.sessions import SessionStatus


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize(
    "status", [SessionStatus.COMPLETED, SessionStatus.FAILED, SessionStatus.INTERRUPTED]
)
@pytest.mark.parametrize("pending_input", [False, True])
def test_terminal_cleanup_preserves_exact_pending_evidence(
    tmp_path, monkeypatch, backend, status, pending_input
):
    async def scenario():
        async with publication_fixture(backend, tmp_path) as (store, bootstrap):
            await BrowserControlPublisher(store).publish(bootstrap)
            owner = coordinator(store, Policy(True))
            expected = bootstrap.changed_record
            principal = BrowserControlPrincipal(subject="operator")
            if pending_input:
                pending = await owner.request_takeover(
                    principal=principal,
                    operator_session_id="server-session",
                    intent=intent_for(bootstrap),
                )
                acquired = await owner._publish_guest_acquisition(
                    expected=pending, lease_until_ms=2000
                )
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
                settled = await owner._publish_guest_sensitive_entry(expected=sensitive)
                expected = await owner._admit_text_input(
                    principal=principal,
                    operator_session_id="server-session",
                    intent=BrowserTextInputIntent(
                        identity=settled.identity,
                        request_id=settled.request.request_id,
                        expected_record_revision=settled.revision,
                        expected_control_epoch=settled.control_epoch,
                        input_sequence=1,
                        page=settled.request.pages[0],
                    ),
                )
            controls, _ = await owner._load(expected.identity)
            desired = expected.model_copy(
                update={"revision": expected.revision + 1, "state": "control_uncertain"}
            )
            mutation = BrowserControlCheckpointMutation(
                "session", controls, controls.replace_record(expected=expected, desired=desired)
            )
            await store.update_status("session", status)
            with pytest.raises(Exception):
                await BrowserControlPublisher(store).publish(BrowserControlPublication(mutation))
            original = store.publish_session_operation_guarded_with_store_time
            writes = []

            async def lost_ack(*args, **kwargs):
                writes.append(1)
                await original(*args, **kwargs)
                raise OSError("teardown acknowledgement lost")

            monkeypatch.setattr(
                store, "publish_session_operation_guarded_with_store_time", lost_ack
            )
            assert await owner.mark_channel_uncertain(expected=expected) == desired
            assert (
                await coordinator(store, Policy(True)).mark_channel_uncertain(expected=expected)
                == desired
            )
            assert writes == [1]
            assert desired.pending_input_sequence == expected.pending_input_sequence
            assert desired.operator_page_operations == expected.operator_page_operations
            with pytest.raises(BrowserControlConflict):
                await owner._load(expected.identity)
            with pytest.raises(BrowserControlConflict):
                await owner.authorize_view(
                    principal=principal,
                    operator_session_id="server-session",
                    identity=expected.identity,
                    expected_record_revision=desired.revision,
                )
            if pending_input:
                with pytest.raises(BrowserControlConflict):
                    await owner.mark_channel_uncertain(expected=bootstrap.changed_record)
            assert await owner.drain()

    asyncio.run(scenario())


def test_cleanup_publication_cannot_change_control_epoch():
    from tests.core.test_browser_control import identity, request

    from cayu.runtime.browser_control import BrowserControlCheckpoint, BrowserControlRecord

    source = BrowserControlRecord(
        identity=identity(), request=request(), revision=2, state="takeover_requested"
    )
    before = BrowserControlCheckpoint(records=(source,))
    desired = source.model_copy(
        update={
            "revision": source.revision + 1,
            "state": "control_uncertain",
            "control_epoch": source.control_epoch + 1,
        }
    )
    with pytest.raises(BrowserControlConflict):
        BrowserControlFencePublication(
            BrowserControlCheckpointMutation(
                "session", before, before.replace_record(expected=source, desired=desired)
            )
        )
