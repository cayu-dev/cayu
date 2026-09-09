"""Queue terminal outcomes agree with the existing session inspection summary."""

import asyncio
from datetime import UTC, datetime

import pytest
from tests.core.test_session_message_lifecycle_stores import _action, _request, _session
from tests.core.test_session_store_shared_conformance import _close_store, _open_store

from cayu.runtime.session_message_lifecycle import (
    SessionMessageConditions,
    SessionMessageQuery,
    SessionMessageTarget,
)
from cayu.runtime.sessions import SessionMessageDeliveryMode, SessionStatus


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_terminal_messages_are_not_reported_as_outstanding(backend, tmp_path):
    async def run():
        store = await _open_store((backend, tmp_path, None))
        try:
            session = await _session(store)
            for action in ("withdraw", "quarantine"):
                accepted = await store.enqueue_session_message(_request(session.id, action))
                page = await store.inspect_session_messages(
                    SessionMessageQuery(session_id=session.id)
                )
                record = next(
                    row for row in page.records if row.queue_id == accepted.message.queue_id
                )
                await store.apply_session_message_action(_action(page, record, action))
            await store.enqueue_session_message(
                _request(
                    session.id,
                    "expired",
                    SessionMessageConditions(expires_at=datetime(2000, 1, 1, tzinfo=UTC)),
                )
            )
            await store.enqueue_session_message(
                _request(
                    session.id,
                    "stale",
                    SessionMessageConditions(
                        target=SessionMessageTarget(
                            session_instance_id=session.instance_id,
                            run_epoch=session.run_epoch,
                            transcript_cursor=999,
                        )
                    ),
                )
            )
            await store.enqueue_session_message(_request(session.id, "delivered"))
            await store.enqueue_session_message(
                _request(session.id, "pending").model_copy(
                    update={"delivery_mode": SessionMessageDeliveryMode.ON_IDLE}
                )
            )
            await store.transition_status(
                session.id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            await store.deliver_queued_session_messages(session.id, include_on_idle=False)
            page = await store.inspect_session_messages(SessionMessageQuery(session_id=session.id))
            assert {row.status for row in page.records} == {
                "queued",
                "delivered",
                "withdrawn",
                "quarantined",
                "stale",
                "expired",
            }
            summary = await store.inspect_summary(session.id)
            assert summary.queued_message_count == 6
            assert summary.delivered_message_count == 1
            assert summary.outstanding_message_count == 1
        finally:
            await _close_store(store)

    asyncio.run(run())
