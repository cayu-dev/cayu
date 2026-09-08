"""Multi-allocation discovery cannot publish mixed authorization generations."""

import asyncio

import pytest
from tests.core.test_browser_control_authorization import Policy
from tests.core.test_browser_control_coordinator import coordinator
from tests.core.test_browser_control_publisher import publication_fixture

from cayu.runtime._browser_control_checkpoint import BrowserControlCheckpointMutation
from cayu.runtime._browser_control_publication import BrowserControlPublication
from cayu.runtime._browser_control_publisher import BrowserControlPublisher
from cayu.runtime.browser_control import (
    BrowserControlConflict,
    BrowserControlPrincipal,
    BrowserControlRecord,
)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("change_first", [False, True])
def test_discovery_rechecks_earlier_allocation_after_later_policy_await(
    tmp_path, backend, change_first
):
    async def scenario():
        async with publication_fixture(backend, tmp_path) as (store, initial):
            publisher = BrowserControlPublisher(store)
            first = await publisher.publish(initial)
            second = BrowserControlRecord(
                identity=first.identity.model_copy(update={"browser_session_id": "bs_second"})
            )
            controls = initial.mutation.desired
            await publisher.publish(
                BrowserControlPublication(
                    BrowserControlCheckpointMutation(
                        first.identity.session_id,
                        controls,
                        controls.replace_record(expected=None, desired=second),
                    )
                )
            )
            entered, release = asyncio.Event(), asyncio.Event()

            class BlockingPolicy(Policy):
                async def decide(self, request):
                    if request.identity == second.identity:
                        entered.set()
                        await release.wait()
                    return await super().decide(request)

            owner = coordinator(store, BlockingPolicy(True))
            discovery = asyncio.create_task(
                owner.inspect_authorized_browsers(
                    session_id=first.identity.session_id,
                    principal=BrowserControlPrincipal(subject="operator"),
                    operator_session_id="operator-continuity",
                )
            )
            await entered.wait()
            if change_first:
                await owner.mark_channel_uncertain(expected=first)
            release.set()
            if change_first:
                with pytest.raises(BrowserControlConflict, match="discovery"):
                    await discovery
            else:
                assert await discovery == (first, second)
            assert await owner.drain()

    asyncio.run(scenario())
