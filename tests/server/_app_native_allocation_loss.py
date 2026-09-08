"""Actual owned Chromium process loss during authenticated operator takeover."""

import asyncio
import os
import signal

from cayu.tools._browser_control_guest import GuestControlAllocationLost


async def lose_browser_allocation(race, browser, coordinator, identity, *, racing_poll=False):
    await race.acquire()
    poll_entered = asyncio.Event()
    release_poll = asyncio.Event()
    poll = None
    pages = race.daemon.operator_page_descriptors

    async def held_pages(*, epoch):
        poll_entered.set()
        await release_poll.wait()
        return await pages(epoch=epoch)

    async def request_pages():
        return await race.winner.post(
            race.root + "/pages",
            json={
                "identity": race.acquired["identity"],
                "expected_record_revision": race.acquired["revision"],
            },
        )

    if racing_poll:
        race.monkeypatch.setattr(race.daemon, "operator_page_descriptors", held_pages)
        poll = asyncio.create_task(request_pages())

        async def settle_poll():
            release_poll.set()
            await asyncio.gather(poll, return_exceptions=True)

        race.stack.push_async_callback(settle_poll)
        await asyncio.wait_for(poll_entered.wait(), 5)
    disconnected = asyncio.Event()
    browser.on("disconnected", disconnected.set)
    inspector = await browser.new_browser_cdp_session()
    processes = await inspector.send("SystemInfo.getProcessInfo")
    browser_pids = [item["id"] for item in processes["processInfo"] if item["type"] == "browser"]
    assert len(browser_pids) == 1
    pid = browser_pids[0]
    assert type(pid) is int and pid > 1 and pid != os.getpid()
    # This PID comes from the isolated browser launched by this test, not a
    # machine-wide process search or the user's normal Chrome instance.
    os.kill(pid, signal.SIGKILL)
    await asyncio.wait_for(disconnected.wait(), 5)
    assert not browser.is_connected()
    if poll is not None:
        # The host admitted this read while live; no status exchange can
        # overtake its sole-reader command. Deliver death before guest validation.
        release_poll.set()
        response = await asyncio.wait_for(poll, 5)
        assert response.status_code in {403, 503}, response.text
        channel = race.daemon._operator_bootstrap_task
        assert channel is not None
        failures = await asyncio.wait_for(asyncio.shield(channel), 5)
        assert len(failures) == 1
        assert type(failures[0]) is GuestControlAllocationLost
        assert race.daemon.control.state == "control_uncertain"

    async with asyncio.timeout(5):
        while True:
            _, current = await coordinator._load(identity)
            if current.state == "control_uncertain":
                break
            await asyncio.sleep(0.01)
    response = await race.winner.post(
        race.root + "/pages",
        json={
            "identity": race.acquired["identity"],
            "expected_record_revision": race.acquired["revision"],
        },
    )
    assert response.status_code in {403, 503}, response.text
    assert current.settled_input_sequence == 0
    assert current.handback_audit is None
    for path, payload in (
        (
            "/input-ticket",
            {
                **race.intent(race.acquired),
                "page": race.pages[0],
                "input_sequence": 1,
                "input_kind": "tab",
            },
        ),
        ("/handback", race.intent(race.acquired)),
    ):
        response = await race.winner.post(race.root + path, json=payload)
        assert response.status_code in {403, 409}, response.text
    assert race.daemon._operator_input_task is None
    assert race.dispatched == ["open", "held-model"]
