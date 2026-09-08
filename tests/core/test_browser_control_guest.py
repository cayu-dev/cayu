"""Native daemon control boundaries; transport and real Chromium are tested separately."""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest
from tests.core.test_browser_session import _interactive_raw_request, _interactive_request

from cayu.tools import _browser_guest as guest
from cayu.tools._browser_control_guest import GuestControlFailure, GuestControlFence
from cayu.tools._browser_visual_guest import VisualPageOwner

_BINDING = "a" * 64
_REQUEST = "bt_" + "1" * 32


def takeover_material(daemon=None) -> dict:
    now_ms = int(time.time() * 1000)
    return {
        "binding_sha256": _BINDING,
        "request_id": _REQUEST,
        "request_sha256": "b" * 64,
        "expected_epoch": 1,
        "expires_at_ms": now_ms + 30_000,
        "maximum_until_ms": now_ms + 60_000,
        "lease_until_ms": now_ms + 30_000,
        "pages": []
        if daemon is None
        else [
            {
                "page_id": page.page_id,
                "revision": daemon._operator_page_revision(page),
                "control_epoch": page.control_epoch,
            }
            for page in daemon.pages.values()
            if page.lifecycle not in {"closed", "crashed"}
        ],
    }


def test_daemon_waits_for_dispatched_model_before_grant(monkeypatch) -> None:
    asyncio.run(_assert_model_settlement(monkeypatch, cancel=False))


@pytest.mark.parametrize("changed", [False, True])
def test_takeover_owns_page_expectations_and_checks_after_model_settlement(monkeypatch, changed):
    async def scenario():
        dispatched = asyncio.Event()
        release = asyncio.Event()
        daemon = guest._InteractiveDaemon("bs_test")
        daemon.context = object()
        page = guest._InteractivePage(
            page=object(), session_id="bs_test", page_id="page", lifecycle="active", revision="old"
        )
        daemon.pages[page.page_id] = page
        await daemon.bind_operator_control(_BINDING)

        async def configured(_request):
            return None

        async def native_operation(_request):
            dispatched.set()
            await release.wait()
            if changed:
                page.revision = "new"
            return {"kind": "success", "observation": {"revision": page.revision}}

        monkeypatch.setattr(daemon, "_ensure_configuration", configured)
        monkeypatch.setattr(daemon, "_execute_locked", native_operation)
        model = asyncio.create_task(
            daemon.execute(replace(_interactive_request("observe"), invocation_control_epoch=1))
        )
        await dispatched.wait()
        material = takeover_material(daemon)
        takeover = asyncio.create_task(daemon.acquire_operator_control(**material))
        await asyncio.sleep(0)
        assert daemon.control.state == "takeover_requested" and not takeover.done()
        # Mutating the original cannot alter consent across the lock await.
        material["pages"][0]["revision"] = "new"
        release.set()
        await model
        if changed:
            with pytest.raises(GuestControlFailure):
                await takeover
            assert daemon.control.state == "control_uncertain"
            assert daemon.control.epoch == 1
        else:
            assert (await takeover)["state"] == "operator_controlled"

    asyncio.run(scenario())


def test_cancelled_takeover_does_not_cancel_model_or_release_fence(monkeypatch) -> None:
    asyncio.run(_assert_model_settlement(monkeypatch, cancel=True))


def test_takeover_deadline_does_not_cancel_model_or_grant_input(monkeypatch) -> None:
    asyncio.run(_assert_model_settlement(monkeypatch, cancel=False, expire=True))


async def _assert_model_settlement(monkeypatch, *, cancel: bool, expire: bool = False) -> None:
    dispatched = asyncio.Event()
    release = asyncio.Event()
    effects = []
    daemon = guest._InteractiveDaemon("bs_test")
    daemon.context = object()
    await daemon.bind_operator_control(_BINDING)

    async def configured(_request):
        return None

    async def native_operation(_request):
        dispatched.set()
        await release.wait()
        effects.append("settled")
        return {"kind": "success", "observation": {"revision": "new"}}

    monkeypatch.setattr(daemon, "_ensure_configuration", configured)
    monkeypatch.setattr(daemon, "_execute_locked", native_operation)
    request = replace(_interactive_request("observe"), invocation_control_epoch=1)
    model = asyncio.create_task(daemon.execute(request))
    await dispatched.wait()
    material = takeover_material()
    if expire:
        material["expires_at_ms"] = int(time.time() * 1000) + 100
    takeover = asyncio.create_task(daemon.acquire_operator_control(**material))
    await asyncio.sleep(0)
    assert daemon.control.state == "takeover_requested"
    assert not takeover.done()
    if cancel:
        takeover.cancel()
        assert takeover.cancelling() == 1
        with pytest.raises(asyncio.CancelledError):
            await takeover
        assert takeover.cancelled()
        assert daemon.control.state == "control_uncertain"
    elif expire:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(takeover, timeout=1.0)
        assert takeover.cancelling() == 0 and not takeover.cancelled()
        assert daemon.control.state == "control_uncertain"
    assert model.cancelling() == 0
    assert not model.done()
    release.set()
    await model
    if not cancel and not expire:
        result = await takeover
        assert result["state"] == "operator_controlled"
        assert result["control_epoch"] == 2
    with pytest.raises(guest._GuestFailure, match="policy_denied"):
        await daemon.execute(replace(request, operation_id="competing-model"))
    assert effects == ["settled"]


@pytest.mark.parametrize("epoch", [None, True, False, 0, -1, 2**53])
def test_guest_rejects_malformed_supplied_control_epoch(epoch) -> None:
    raw = _interactive_raw_request("navigate")
    with pytest.raises(guest._GuestFailure):
        guest._interactive_request_from_json({**raw, "invocation_control_epoch": epoch})


def test_guest_epoch_is_optional_only_before_control_binding() -> None:
    raw = _interactive_raw_request("navigate")
    assert guest._interactive_request_from_json(raw).invocation_control_epoch is None
    assert (
        guest._interactive_request_from_json(
            {**raw, "invocation_control_epoch": 1}
        ).invocation_control_epoch
        == 1
    )
    fence = GuestControlFence(worker_instance="worker")
    fence.check_model(None, "navigate")
    fence.bind(_BINDING)
    with pytest.raises(GuestControlFailure):
        fence.check_model(None, "navigate")
    fence.check_model(1, "navigate")


@pytest.mark.parametrize("cancel", [False, True])
def test_takeover_waits_for_retained_observation_owner(cancel: bool) -> None:
    async def scenario():
        release = asyncio.Event()

        async def restore():
            await release.wait()
            return ()

        daemon = guest._InteractiveDaemon("bs_test")
        daemon.context = object()
        await daemon.bind_operator_control(_BINDING)
        restoration = asyncio.create_task(restore())
        page = guest._InteractivePage(
            page=object(),
            session_id="bs_test",
            page_id="bp_test",
            lifecycle="active",
            observation_cleanup_task=restoration,
            revision="observed",
        )
        daemon.pages[page.page_id] = page
        takeover = asyncio.create_task(daemon.acquire_operator_control(**takeover_material(daemon)))
        await asyncio.sleep(0)
        assert daemon.control.state == "takeover_requested"
        assert not takeover.done()
        if cancel:
            takeover.cancel()
            with pytest.raises(asyncio.CancelledError):
                await takeover
            assert takeover.cancelling() == 1 and takeover.cancelled()
            assert restoration.cancelling() == 0 and not restoration.done()
            assert daemon.control.state == "control_uncertain"
        release.set()
        await restoration
        if not cancel:
            assert (await takeover)["state"] == "operator_controlled"

    asyncio.run(scenario())


def test_duplicate_acquire_and_handback_do_not_advance_again() -> None:
    async def scenario():
        daemon = guest._InteractiveDaemon("bs_test")
        daemon.context = object()
        await daemon.bind_operator_control(_BINDING)
        for index in range(2):
            page = guest._InteractivePage(
                page=SimpleNamespace(url="https://before.test/private-path?token=hidden"),
                session_id="bs_test",
                page_id=f"bp_{index}",
                lifecycle="active" if index == 0 else "background",
                revision="old",
                refs={"old-ref": "node"},
            )
            page.visual_owner = VisualPageOwner(object(), daemon.visual_worker_instance)
            page.visual_owner.evidence = {"old": "evidence"}
            daemon.pages[page.page_id] = page
        material = takeover_material(daemon)
        acquired = await daemon.acquire_operator_control(**material)
        assert {item["origin"] for item in acquired["audit"]["locations"]} == {
            "https://before.test"
        }
        for state in daemon.pages.values():
            state.page.url = "https://after.test/"
        assert await daemon.acquire_operator_control(**material) == acquired
        handed_back = await daemon.handback_operator_control(request_id=_REQUEST, epoch=2)
        assert {item["origin"] for item in handed_back["audit"]["locations"]} == {
            "https://after.test"
        }
        for state in daemon.pages.values():
            state.page.url = "https://later.test/"
        assert handed_back["control_epoch"] == 3
        assert handed_back["fresh_observation_required"] is True
        handed_back_activity = daemon.last_activity
        assert handed_back_activity > 0
        epochs = [page.control_epoch for page in daemon.pages.values()]
        assert await daemon.handback_operator_control(request_id=_REQUEST, epoch=2) == handed_back
        assert daemon.last_activity == handed_back_activity
        assert [page.control_epoch for page in daemon.pages.values()] == epochs
        for page in daemon.pages.values():
            assert not page.refs and page.revision is None
            assert page.visual_owner is not None
            assert page.visual_owner.evidence is None
        with pytest.raises(GuestControlFailure):
            daemon.control.check_model(1, "observe")
        with pytest.raises(GuestControlFailure):
            daemon.control.check_model(3, "click")
        daemon.control.check_model(3, "observe")

    asyncio.run(scenario())


def test_live_takeover_defers_idle_retirement_until_bounded_expiry():
    async def scenario():
        daemon = guest._InteractiveDaemon("bs_test")
        daemon.context = object()
        daemon.last_activity = asyncio.get_running_loop().time() - daemon.idle_timeout_seconds - 1
        await daemon.bind_operator_control(_BINDING)
        material = takeover_material()
        now = int(time.time() * 1000)
        material.update(
            expires_at_ms=now + 100, lease_until_ms=now + 100, maximum_until_ms=now + 300
        )
        await daemon.acquire_operator_control(**material)
        shutdown = asyncio.create_task(guest._wait_for_interactive_shutdown(daemon))
        await asyncio.sleep(0)
        assert not shutdown.done() and not daemon.close_requested.is_set()
        await asyncio.sleep(0.15)
        # Lease expiration refuses input, while the allocation remains available
        # for bounded explicit settlement until the original request maximum.
        with pytest.raises(GuestControlFailure):
            daemon.control.check_model(2, "observe")
        assert daemon.control.state == "control_uncertain"
        assert not shutdown.done()
        async with asyncio.timeout(2):
            await shutdown
        assert daemon.idle_expired and daemon.close_requested.is_set()
        assert daemon.control.state == "control_uncertain"

    asyncio.run(scenario())


def test_settled_handback_restores_the_normal_idle_window():
    async def scenario():
        daemon = guest._InteractiveDaemon("bs_test")
        daemon.context = object()
        daemon.last_activity = asyncio.get_running_loop().time() - daemon.idle_timeout_seconds - 1
        await daemon.bind_operator_control(_BINDING)
        material = takeover_material()
        await daemon.acquire_operator_control(**material)
        await daemon.handback_operator_control(request_id=_REQUEST, epoch=2)
        shutdown = asyncio.create_task(guest._wait_for_interactive_shutdown(daemon))
        await asyncio.sleep(0)
        assert not shutdown.done() and not daemon.close_requested.is_set()
        assert daemon.control.fresh_observation_required
        daemon.close_requested.set()
        async with asyncio.timeout(2):
            await shutdown

    asyncio.run(scenario())


def test_late_input_settlement_cannot_undo_expiry() -> None:
    clock = [100.0]
    fence = GuestControlFence(
        worker_instance="worker", monotonic=lambda: clock[0], wall_clock=lambda: clock[0]
    )
    fence.bind(_BINDING)
    fence.request(
        binding_sha256=_BINDING,
        request_id=_REQUEST,
        request_sha256="b" * 64,
        expected_epoch=1,
        expires_at_ms=120_000,
        maximum_until_ms=160_000,
    )
    fence.grant(request_id=_REQUEST, lease_until_ms=110_000)
    fence.begin_input(request_id=_REQUEST, epoch=2, sequence=1)
    clock[0] = 111.0
    fence.expire()
    fence.settle_input(1)
    assert fence.state == "control_uncertain"
    assert fence.settled_sequence == 1 and fence.pending_sequence is None
    with pytest.raises(GuestControlFailure):
        fence.begin_input(request_id=_REQUEST, epoch=2, sequence=1)
    with pytest.raises(GuestControlFailure):
        fence.check_model(2, "observe")


def test_takeover_joins_cleanup_registered_during_settlement() -> None:
    async def scenario():
        first_release = asyncio.Event()
        second_release = asyncio.Event()
        spawned = asyncio.Event()
        daemon = guest._InteractiveDaemon("bs_test")
        daemon.context = object()
        await daemon.bind_operator_control(_BINDING)
        page = guest._InteractivePage(
            page=object(), session_id="bs_test", page_id="bp_test", lifecycle="active"
        )
        page.visual_owner = VisualPageOwner(object(), daemon.visual_worker_instance)
        daemon.pages[page.page_id] = page

        async def native_action():
            await second_release.wait()

        async def restoration():
            await first_release.wait()
            assert page.visual_owner is not None
            page.visual_owner.action_task = asyncio.create_task(native_action())
            spawned.set()
            return ()

        page.observation_cleanup_task = asyncio.create_task(restoration())
        page.revision = "observed"
        takeover = asyncio.create_task(daemon.acquire_operator_control(**takeover_material(daemon)))
        await asyncio.sleep(0)
        first_release.set()
        await spawned.wait()
        done, _ = await asyncio.wait({takeover}, timeout=0.02)
        assert not done
        assert daemon.control.state == "takeover_requested"
        assert page.visual_owner.action_task is not None
        assert not page.visual_owner.action_task.done()
        second_release.set()
        assert (await takeover)["state"] == "operator_controlled"

    asyncio.run(scenario())
