"""Private page descriptors contain no page content and share mutation locking."""

import asyncio

import pytest

from cayu.tools._browser_control_guest import GuestControlFailure
from cayu.tools._browser_guest import _InteractiveDaemon, _InteractivePage


def test_page_descriptors_wait_for_page_owner_and_omit_content():
    async def scenario():
        daemon = _InteractiveDaemon("bs_pages")
        daemon.context = object()
        daemon.claim_operator_channel("channel")
        await daemon.bind_operator_control("a" * 64)
        page = _InteractivePage(
            page=object(),
            session_id="bs_pages",
            page_id="page",
            lifecycle="active",
            revision="before",
            title="credential-canary",
        )
        daemon.pages["page"] = page
        daemon.active_page_id = "page"
        async with daemon.lock:
            reader = asyncio.create_task(daemon.operator_page_descriptors(epoch=1))
            await asyncio.sleep(0)
            assert not reader.done()
            page.revision = "after"
        result = await reader
        assert result["pages"] == [{"page_id": "page", "revision": "after", "control_epoch": 1}]
        assert result["active_page_id"] == "page"
        assert "credential-canary" not in repr(result)
        for epoch in (True, 0, 2):
            with pytest.raises(GuestControlFailure):
                await daemon.operator_page_descriptors(epoch=epoch)
        page.lifecycle = "uncertain"
        with pytest.raises(GuestControlFailure):
            await daemon.operator_page_descriptors(epoch=1)

    asyncio.run(scenario())
