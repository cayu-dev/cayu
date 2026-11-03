"""Unsettled upstream work must retain both kinds of broker capacity."""

import asyncio
import contextlib

import pytest
from tests.egress.test_broker_pipeline import _mint, _request
from tests.egress.test_egress_buffer_budget import _SMALL, _broker

from cayu.egress import CapturedResponse, EgressUpstreamOperation


def test_failed_cancellation_retains_buffer_reservation_until_settlement_retry():
    async def run():
        started = asyncio.Event()
        stopped = asyncio.Event()
        cancellation_calls = 0

        class Upstream:
            def prepare(self, request, *, limits):
                async def send():
                    assert limits.response_buffer is not None
                    limits.response_buffer.reserve_to(700)
                    body = bytearray(b"x" * 700)
                    started.set()
                    try:
                        await asyncio.Event().wait()
                        return CapturedResponse(status_code=200, body=bytes(body))
                    finally:
                        stopped.set()

                async def cancel_and_wait(task):
                    nonlocal cancellation_calls
                    cancellation_calls += 1
                    if cancellation_calls == 1:
                        # An external cancellation owner failed to stop its
                        # work; the broker retains it for a settlement retry.
                        raise RuntimeError("termination temporarily failed")
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

                return EgressUpstreamOperation(send, cancel_and_wait=cancel_and_wait)

        broker, registry = _broker(Upstream(), response_buffer_budget_bytes=1000, **_SMALL)
        grant = _mint(registry)
        request_task = asyncio.create_task(
            broker.handle_request(_request(grant.presented_value, "/v1/customers"))
        )
        try:
            await asyncio.wait_for(started.wait(), 2)
            assert broker._response_buffer_budget.used_bytes == 700
            request_task.cancel("client disconnected")
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(request_task, 2)
            assert not stopped.is_set()
            assert len(broker._active_upstream_operations) == 1
            held_while_unsettled = broker._response_buffer_budget.used_bytes
        finally:
            await broker.settle_active_upstream_operations()
            await asyncio.gather(request_task, return_exceptions=True)

        assert stopped.is_set()
        assert broker._response_buffer_budget.used_bytes == 0
        assert held_while_unsettled == 700, (
            "The upstream still holds its 700-byte body, but the shared budget "
            f"accounts for only {held_while_unsettled} bytes"
        )

    asyncio.run(run())
