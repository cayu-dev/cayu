from __future__ import annotations

import asyncio
import json
import types
from dataclasses import replace
from pathlib import Path

import pytest
from tests.core.test_browser_session import (
    _configure_interactive_daemon_for_test,
    _durable_context,
    _FakeBrowserBackend,
    _interactive_limits,
    _interactive_request,
)

from cayu.tools import _browser_guest as guest
from cayu.tools import browser_session as host


@pytest.mark.parametrize("cleanup", ["release", "timeout", "failure"])
def test_guard_failure_retains_durable_capacity_until_whole_close(tmp_path: Path, cleanup: str):
    async def scenario():
        started = asyncio.Event()
        release = asyncio.Event()

        class Page:
            url = "https://example.test/form"

            async def evaluate(self, *_):
                raise RuntimeError("guard evaluation failed")

            def locator(self, *_):
                return self

            async def element_handle(self):
                return self

            async def close(self):
                started.set()
                await release.wait()
                if cleanup == "failure":
                    raise RuntimeError("page cleanup failed")

        class Context:
            async def close(self):
                pass

        class Backend(_FakeBrowserBackend):
            daemon = None

            async def execute(self, ctx, request):
                if request["operation"] == "navigate":
                    return await super().execute(ctx, request)
                self.daemon = daemon = guest._InteractiveDaemon(request["session_id"])
                daemon.context = Context()
                parsed = replace(
                    _interactive_request("click"),
                    session_id=request["session_id"],
                    page_id=request["page_id"],
                    operation_id=request["operation_id"],
                    expected_revision=request["expected_revision"],
                    expected_control_epoch=request["expected_control_epoch"],
                    ref=request["ref"],
                    multi_page=True,
                    popup_policy=guest._InteractivePopupPolicy("same_origin", ("click",), (), ()),
                )
                page = guest._InteractivePage(
                    page=Page(),
                    session_id=parsed.session_id,
                    page_id=parsed.page_id,
                    lifecycle="active",
                    configured=True,
                    revision=parsed.expected_revision,
                    control_epoch=parsed.expected_control_epoch,
                    refs={parsed.ref: "internal"},
                    public_url=Page.url,
                    last_observation_revision=parsed.expected_revision,
                    operation_count=1,
                    observation_count=1,
                    ref_count=2,
                )
                daemon.pages[page.page_id] = page
                daemon.active_page_id = page.page_id
                daemon.total_page_creations = daemon.total_operations = 1
                daemon.total_observations = 1
                daemon.total_refs = 2
                await _configure_interactive_daemon_for_test(daemon, parsed)
                response = await daemon.execute(parsed)
                if cleanup != "release":
                    assert response["error"] == "cleanup_failed", response
                    host.BrowserPageSetState.model_validate_json(json.dumps(response["page_set"]))
                return host._parse_runner_response(
                    json.dumps(response),
                    max_artifact_bytes=1024,
                    max_page_records=1,
                    max_page_creations_per_operation=1,
                )

        backend = Backend()
        records = {}
        tool = host.BrowserSessionTool(max_sessions=1, _backend=backend)

        async def run(args, *, instance=tool):
            return await instance.run(
                _durable_context(
                    tmp_path, args=args, records=records, tool_call_id=args["operation_id"]
                ),
                args,
            )

        opened = await run({"operation": "navigate", "url": Page.url, "operation_id": "open"})
        assert not opened.is_error
        state = opened.structured
        action = asyncio.create_task(
            run(
                {
                    "operation": "click",
                    "session_id": state["session_id"],
                    "page_id": state["page_id"],
                    "expected_revision": state["revision"],
                    "expected_control_epoch": state["control_epoch"],
                    "ref": "ref_save",
                    "operation_id": "click",
                }
            )
        )
        await asyncio.wait_for(started.wait(), 2)
        assert not action.done()
        parent = records[host._DURABLE_BROWSER_PARENT_KEY]
        assert parent["live_session_ids"] == [state["session_id"]]
        if cleanup != "timeout":
            release.set()
        result = await asyncio.wait_for(action, 3)
        expected = "retired" if cleanup == "release" else "uncertain"
        assert result.structured["allocation_disposition"] == expected, result
        assert records[host._DURABLE_BROWSER_PARENT_KEY]["live_session_ids"] == (
            [] if cleanup == "release" else [state["session_id"]]
        )
        if cleanup != "release":
            assert result.structured["error"] == "cleanup_failed"
        replacement = await run(
            {"operation": "navigate", "url": Page.url, "operation_id": "competing"},
            instance=host.BrowserSessionTool(max_sessions=1, _backend=backend),
        )
        assert replacement.is_error == (cleanup != "release")
        release.set()
        await backend.daemon.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_early_frame_request_is_bound_to_its_exact_popup(method: str):
    async def scenario():
        daemon = guest._InteractiveDaemon("bs_test")
        request = replace(
            _interactive_request("click"),
            multi_page=True,
            limits=_interactive_limits(max_pages=3, max_total_page_creations=3),
            popup_policy=guest._InteractivePopupPolicy("same_origin", ("click",), (), ()),
        )
        daemon.active_request = request
        daemon.active_delta = guest._InteractivePageDelta()
        daemon.configuration_multi_page = True
        daemon.configuration_popup_policy = request.popup_policy
        opener = guest._InteractivePage(
            page=object(),
            session_id="bs_test",
            page_id="bp_root",
            lifecycle="active",
            public_url="https://example.test/",
        )
        daemon.pages[opener.page_id] = opener
        daemon.active_page_id = daemon.popup_effect_opener_page_id = opener.page_id
        daemon.popup_effect_opener_origin = "https://example.test/"
        exact_frame = object()

        class Request:
            _impl_obj = types.SimpleNamespace(
                _initializer={"frame": types.SimpleNamespace(_object=exact_frame)}
            )
            url = "https://example.test/Upper?q=Case"

            @property
            def frame(self):
                raise RuntimeError("Frame not initialized")

            def is_navigation_request(self):
                return True

        incoming = Request()
        incoming.method = method

        class Route:
            aborted = False

            async def abort(self, _):
                self.aborted = True

        route = Route()
        await daemon._route_interactive_request(route, incoming)
        assert route.aborted
        popup = types.SimpleNamespace(main_frame=types.SimpleNamespace(_impl_obj=exact_frame))
        state = daemon._register_popup_candidate(opener, popup)
        assert state is not None
        assert not daemon.active_delta.staged_frames
        assert state.staged_initial_url == (incoming.url if method == "GET" else None)
        assert state.denied_code == ("policy_denied" if method == "POST" else None)

    asyncio.run(scenario())
