"""Visual browser configuration and public request contracts."""

from __future__ import annotations

import asyncio
from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
from tests.core.test_browser_session import (
    _configure_interactive_daemon_for_test,
    _interactive_request,
)

from cayu import BrowserSessionTool, BrowserVisualPolicy, ToolContext
from cayu.tools._browser_visual_guest import visual_policy_from_json
from cayu.tools.browser_session import _validated_request
from cayu.tools.browser_visual import BrowserVisualGeometry, canonical_visual_point


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_visual_action_cancellation_owns_guard_cleanup(cleanup_fails: bool) -> None:
    from cayu.tools import _browser_guest as guest

    async def scenario() -> None:
        dispatched = asyncio.Event()
        cleanup_started = asyncio.Event()
        release_cleanup = asyncio.Event()
        release_input = asyncio.Event()
        closed = False

        class Owner:
            action_task: asyncio.Task[None] | None = None
            disarm_task: asyncio.Task[None] | None = None

            async def click(self, _policy: object, _request: object) -> None:
                dispatched.set()
                await release_input.wait()

            def invalidate(self) -> None:
                pass

            async def disarm(self, _policy: object) -> None:
                raise AssertionError("Unsettled native input must not lose its guard.")

        async def close() -> bool:
            nonlocal closed
            cleanup_started.set()
            await release_cleanup.wait()
            release_input.set()
            if state.visual_owner.action_task is not None:
                await state.visual_owner.action_task
            if cleanup_fails:
                raise RuntimeError("browser cleanup failed")
            closed = True
            return True

        daemon = SimpleNamespace(close=close, closing=False, close_requested=asyncio.Event())
        state = SimpleNamespace(visual_owner=Owner())
        # Use the real request dataclass because the dispatch owner serializes it.
        request = guest._InteractiveRequest(
            operation="click_visual_target",
            session_id="session",
            page_id="page",
            expected_revision="revision",
            expected_control_epoch=1,
            ref=None,
            operation_id="operation",
            url=None,
            value=None,
            key=None,
            wait_ms=None,
            full_page=False,
            multi_page=False,
            limits=guest._InteractiveLimits(
                **{item.name: 100 for item in fields(guest._InteractiveLimits)}
            ),
            popup_policy=guest._InteractivePopupPolicy("deny", (), (), ()),
            visual_policy=policy().model_dump(mode="json"),
            visual_revision="vr_" + "1" * 32,
            visual_ref="vt_" + "2" * 32,
        )
        task = asyncio.create_task(
            guest._InteractiveDaemon._execute_visual_action(daemon, state, request)
        )
        await dispatched.wait()
        task.cancel()
        await cleanup_started.wait()
        assert task.cancelling() == 1 and not task.done()
        assert state.visual_owner.action_task.cancelling() == 0
        assert not state.visual_owner.action_task.done()
        task.cancel()
        await asyncio.sleep(0)
        assert task.cancelling() == 2 and not task.done()
        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError) as raised:
            await task
        assert task.cancelled() and task.cancelling() == 2
        assert closed is not cleanup_fails
        if cleanup_fails:
            assert isinstance(raised.value.__cause__, RuntimeError)
            assert str(raised.value.__cause__) == "browser cleanup failed"

    asyncio.run(scenario())


@pytest.mark.parametrize("termination", ["cancel", "deadline", "guard_failure"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_visual_retirement_survives_outer_popup_owner(termination, cleanup_fails) -> None:
    from cayu.tools import _browser_guest as guest

    async def scenario() -> None:
        dispatched = asyncio.Event()
        release_input = asyncio.Event()
        cleanup_started = asyncio.Event()
        release_cleanup = asyncio.Event()
        evaluations = 0
        closes = 0

        class Page:
            url = "https://example.test"

            async def evaluate(self, *_):
                nonlocal evaluations
                evaluations += 1
                if evaluations > 1:
                    raise AssertionError("Must not query a retired visual page")
                return {"blocked": 0, "urls": []}

        class Owner:
            action_task = None
            disarm_task = None

            async def click(self, *_):
                dispatched.set()
                if termination != "guard_failure":
                    await release_input.wait()

            async def disarm(self, *_):
                raise RuntimeError("visual guard failed")

            def invalidate(self):
                pass

        daemon = guest._InteractiveDaemon("bs_test")
        daemon.context = SimpleNamespace()
        request = replace(
            _interactive_request("click_visual_target"),
            expected_revision="br_test",
            expected_control_epoch=1,
            multi_page=True,
            visual_policy=policy(max_processing_ms=20).model_dump(mode="json"),
            visual_revision="vr_" + "1" * 32,
            visual_ref="vt_" + "2" * 32,
            popup_policy=guest._InteractivePopupPolicy(
                "same_origin", ("click_visual_target",), (), ()
            ),
        )
        state = guest._InteractivePage(
            page=Page(),
            session_id="bs_test",
            page_id="bp_test",
            lifecycle="active",
            revision="br_test",
            visual_owner=Owner(),
            public_url=Page.url,
        )
        daemon.pages[state.page_id] = state
        await _configure_interactive_daemon_for_test(daemon, request)

        async def close(**_):
            nonlocal closes
            closes += 1
            cleanup_started.set()
            await release_cleanup.wait()
            release_input.set()
            if state.visual_owner.action_task is not None:
                await state.visual_owner.action_task
            if cleanup_fails:
                raise RuntimeError("browser cleanup failed")
            return True

        daemon.close = close
        task = asyncio.create_task(daemon.execute(request))
        await asyncio.wait_for(dispatched.wait(), 1)
        if termination == "cancel":
            task.cancel()
        await asyncio.wait_for(cleanup_started.wait(), 1)
        assert not task.done() and daemon.closing
        assert state.visual_action_cleanup_disposition == "uncertain"
        competing = asyncio.create_task(daemon.execute(replace(request, operation_id="competing")))
        await asyncio.sleep(0)
        assert not competing.done()
        if termination == "cancel":
            assert task.cancelling() == 1
            task.cancel()
            await asyncio.sleep(0)
            assert task.cancelling() == 2 and not task.done()
        release_cleanup.set()
        if termination == "cancel":
            with pytest.raises(asyncio.CancelledError) as caught:
                await task
            assert task.cancelled() and task.cancelling() == 2
            assert not daemon.operations
            if cleanup_fails:
                assert str(caught.value.__cause__) == "browser cleanup failed"
        else:
            result = await asyncio.wait_for(task, 1)
            assert result["error"] == ("timeout" if termination == "deadline" else "cleanup_failed")
            assert result["allocation_disposition"] == ("uncertain" if cleanup_fails else "retired")
            assert await daemon.execute(request) == result
            assert not task.cancelled() and task.cancelling() == 0
        assert closes == 1 and evaluations == 1
        with pytest.raises(guest._GuestFailure, match="session_closed"):
            await competing
        assert state.visual_owner.action_task is None or state.visual_owner.action_task.done()
        assert daemon.active_request is None and daemon.active_delta is None

    asyncio.run(scenario())


@pytest.mark.parametrize("signal", ["cancel", "deadline"])
@pytest.mark.parametrize("popup_fails,close_fails", [(False, False), (True, False), (True, True)])
def test_successful_visual_disarm_preserves_signal_through_popup_cleanup(
    signal, popup_fails, close_fails
) -> None:
    from cayu.tools import _browser_guest as guest

    async def scenario():
        disarming = asyncio.Event()
        release = asyncio.Event()
        popup_error = RuntimeError("popup evaluation failed")
        evaluations = 0
        closes = 0

        class Page:
            url = "https://example.test"

            async def evaluate(self, *_):
                nonlocal evaluations
                evaluations += 1
                if evaluations == 2 and popup_fails:
                    raise popup_error
                return {"blocked": 0, "urls": []}

        class Owner:
            action_task = None
            disarm_task = None

            async def click(self, *_):
                return None

            async def disarm(self, *_):
                disarming.set()
                await release.wait()

            def invalidate(self):
                pass

        daemon = guest._InteractiveDaemon("bs_test")
        daemon.context = SimpleNamespace()
        request = replace(
            _interactive_request("click_visual_target"),
            expected_revision="br_test",
            expected_control_epoch=1,
            multi_page=True,
            visual_policy=policy().model_dump(mode="json"),
            popup_policy=guest._InteractivePopupPolicy(
                "same_origin", ("click_visual_target",), (), ()
            ),
        )
        state = guest._InteractivePage(
            page=Page(),
            session_id="bs_test",
            page_id="bp_test",
            lifecycle="active",
            revision="br_test",
            visual_owner=Owner(),
            public_url=Page.url,
        )
        daemon.pages[state.page_id] = state
        await _configure_interactive_daemon_for_test(daemon, request)

        async def close(**_):
            nonlocal closes
            closes += 1
            return not close_fails

        daemon.close = close

        async def execute():
            async with asyncio.timeout(0.02 if signal == "deadline" else None):
                return await daemon.execute(request)

        task = asyncio.create_task(execute())
        await asyncio.wait_for(disarming.wait(), 1)
        if signal == "cancel":
            task.cancel()
        else:
            # Observe actual timeout delivery while the disarm owner is blocked.
            async with asyncio.timeout(1):
                while not task.cancelling():
                    await asyncio.sleep(0.001)
        await asyncio.sleep(0)
        assert task.cancelling() == 1 and not task.done()
        assert state.visual_owner.disarm_task.cancelling() == 0
        assert state.visual_action_cleanup_disposition is None
        release.set()
        with pytest.raises(
            asyncio.CancelledError if signal == "cancel" else TimeoutError
        ) as caught:
            await task
        assert task.cancelled() is (signal == "cancel")
        assert task.cancelling() == (1 if signal == "cancel" else 0)
        assert not daemon.operations
        assert state.visual_owner.action_task is None and state.visual_owner.disarm_task is None
        assert evaluations == 2 and closes == int(popup_fails)
        if popup_fails:

            def evidence(error):
                yield error
                if isinstance(error, BaseExceptionGroup):
                    for child in error.exceptions:
                        yield from evidence(child)
                if error.__cause__ is not None:
                    yield from evidence(error.__cause__)

            chain = list(evidence(caught.value))
            assert sum(error is popup_error for error in chain) == 1
            assert any(
                isinstance(error, guest._GuestFailure)
                and str(error) == ("cleanup_failed" if close_fails else "resource_exhausted")
                for error in chain
            )
        assert daemon.active_request is None and daemon.active_delta is None

    asyncio.run(scenario())


def policy(**updates: object) -> BrowserVisualPolicy:
    return BrowserVisualPolicy.model_validate(
        {
            "artifact_store_id": "screenshots",
            "allowed_origins": ["https://visual.browser.test/"],
            "publish_to_model": True,
            "retention": "application_managed",
            **updates,
        }
    )


def test_visual_default_and_explicit_policy() -> None:
    assert BrowserSessionTool().visual_policy is None
    configured = policy()
    tool = BrowserSessionTool(visual_policy=configured)
    assert tool.visual_policy == configured
    assert tool.visual_policy is not configured
    assert configured.allowed_origins == ("https://visual.browser.test",)
    assert visual_policy_from_json(configured.model_dump(mode="json")) == configured.model_dump(
        mode="json"
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"publish_to_model": 1},
        {"max_targets": True},
        {"max_targets": 257},
        {"max_captures": 0},
        {"max_frame_depth": -1},
        {"max_width": 4097},
        {"max_pixels": 1},
        {"max_hit_tests": 1},
        {"retention": "forever-guaranteed"},
        {"allowed_origins": ["http://visual.browser.test"]},
        {"allowed_origins": ["https://user:password@visual.browser.test"]},
        {"allowed_origins": ["https://visual.browser.test/private"]},
    ],
)
def test_visual_policy_rejects_unsafe_authority(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        policy(**changes)


@pytest.mark.parametrize("value", [True, False, "0.5", None, -1, 1, float("nan"), float("inf")])
def test_visual_points_reject_noncanonical_outside_values(value: object) -> None:
    with pytest.raises(ValueError):
        canonical_visual_point(value)


def test_visual_point_equivalence_and_strict_geometry() -> None:
    assert canonical_visual_point(0.50000001) == canonical_visual_point(0.5)
    with pytest.raises(ValueError):
        BrowserVisualGeometry(x=True, y=0, width=0.5, height=0.5)
    with pytest.raises(ValueError):
        BrowserVisualGeometry(x=0.75, y=0, width=0.5, height=0.5)


def test_visual_request_is_exact_and_has_no_free_form_escape() -> None:
    request = {
        "operation": "click_visual_point",
        "session_id": "bs_test",
        "page_id": "bp_test",
        "expected_revision": "br_test",
        "expected_control_epoch": 1,
        "operation_id": "click-1",
        "visual_revision": "vr_" + "a" * 32,
        "screenshot_sha256": "b" * 64,
        "x": 0.50000001,
        "y": 0.25,
    }
    assert _validated_request(request, max_wait_ms=1000)["x"] == 0.5
    for name in ("selector", "javascript", "screenshot", "absolute_x"):
        with pytest.raises(ValueError):
            _validated_request({**request, name: "forbidden"}, max_wait_ms=1000)
    for name in ("visual_revision", "screenshot_sha256", "expected_control_epoch"):
        with pytest.raises(ValueError):
            _validated_request(
                {key: value for key, value in request.items() if key != name}, max_wait_ms=1000
            )


@pytest.mark.parametrize("field", ["max_targets", "publish_to_model", "allowed_origins"])
def test_mutated_visual_policy_is_rejected_without_diagnostic_side_channels(
    field: str,
    caplog: pytest.LogCaptureFixture,
    capfd: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import warnings

    canary = "visual-policy-private-canary"

    class HostileValue:
        def __repr__(self) -> str:
            print(canary)
            return canary

        __str__ = __repr__

    tool = BrowserSessionTool(visual_policy=policy())
    assert tool.visual_policy is not None
    object.__setattr__(tool.visual_policy, field, HostileValue())

    async def no_dispatch(*args: object, **kwargs: object) -> None:
        raise AssertionError("Invalid pixel authority must fail before backend preflight.")

    monkeypatch.setattr(tool._backend, "preflight", no_dispatch)
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        result = asyncio.run(
            tool.run(
                ToolContext(session_id="visual-policy-test"),
                {
                    "operation": "observe_visual",
                    "operation_id": "capture",
                    "session_id": "bs_test",
                    "page_id": "bp_test",
                },
            )
        )
    assert result.structured["error"] == "visual_publication_denied"
    assert not emitted
    captured = capfd.readouterr()
    assert canary not in captured.out + captured.err + caplog.text + repr(result)
