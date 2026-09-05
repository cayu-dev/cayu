"""Visual publication and observation settlement through their real owners."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import pytest
from tests.core.test_browser_session import _context, _durable_context, _FakeBrowserBackend

from cayu import BrowserSessionTool, BrowserVisualPolicy, LocalArtifactStore
from cayu.runtime._invocation_secrets import InvocationSecretTracker
from cayu.tools import _browser_guest as guest
from cayu.tools.browser_session import BrowserArtifactPayload
from cayu.tools.browser_visual import BrowserVisualObservation
from cayu.vaults import SecretRedactor


class _VisualBackend(_FakeBrowserBackend):
    async def execute(self, ctx, request):
        response = await super().execute(
            ctx,
            {**request, "operation": "observe"}
            if request["operation"] == "observe_visual"
            else request,
        )
        if request["operation"] != "observe_visual":
            return response
        self.calls[-1]["operation"] = "observe_visual"
        pixels = b"\x89PNG\r\n\x1a\nvisual-fixture"
        observation = response.observation
        assert observation is not None and response.page_set is not None
        visual = BrowserVisualObservation(
            session_id=observation.session_id,
            page_id=observation.page_id,
            page_revision=observation.revision,
            control_epoch=observation.control_epoch,
            worker_instance="vw_" + "1" * 32,
            visual_revision="vr_" + "2" * 32,
            screenshot_sha256=hashlib.sha256(pixels).hexdigest(),
            viewport_width=1280,
            viewport_height=720,
            device_scale=1,
            scroll_x=0,
            scroll_y=0,
            targets=(),
        )
        self.artifact_count += 1
        page = response.page_set.pages[0].model_copy(update={"artifact_count": self.artifact_count})
        return replace(
            response,
            observation=observation.model_copy(update={"visual": visual}),
            page_set=response.page_set.model_copy(
                update={"pages": (page,), "total_artifacts": self.artifact_count}
            ),
            artifacts=(
                BrowserArtifactPayload(
                    kind="screenshot",
                    filename="visual.png",
                    content_type="image/png",
                    content=pixels,
                ),
            ),
        )


@pytest.mark.parametrize("phase", ["write", "readback", "pending", "unowned", "static_change"])
def test_visual_publication_seals_secret_scope_before_artifact_await(
    tmp_path: Path, phase: str
) -> None:
    async def scenario() -> None:
        tracker = InvocationSecretTracker(SecretRedactor())
        entered = asyncio.Event()
        release = asyncio.Event()

        class Store(LocalArtifactStore):
            async def put_bytes(self, *args: Any, **kwargs: Any):
                value = await super().put_bytes(*args, **kwargs)
                if phase == "readback":
                    raise ConnectionError("lost write acknowledgement")
                entered.set()
                await release.wait()
                return value

            async def read_bytes(self, *args: Any, **kwargs: Any):
                value = await super().read_bytes(*args, **kwargs)
                if phase == "readback":
                    entered.set()
                    await release.wait()
                return value

        store = Store(tmp_path / "artifacts", store_id="browser-artifacts")
        backend = _VisualBackend()
        tool = BrowserSessionTool(
            _backend=backend,
            visual_policy=BrowserVisualPolicy(
                artifact_store_id=store.id,
                allowed_origins=("https://example.test",),
                retention="application_managed",
                publish_to_model=True,
            ),
        )
        records: dict[str, dict[str, Any]] = {}
        opened = await tool.run(
            _durable_context(
                tmp_path,
                args={
                    "operation": "navigate",
                    "url": "https://example.test",
                    "operation_id": "open",
                },
                records=records,
                tool_call_id="open",
            ),
            {
                "operation": "navigate",
                "url": "https://example.test",
                "operation_id": "open",
            },
        )
        assert not opened.is_error
        request = {
            "operation": "observe_visual",
            "operation_id": "capture",
            "session_id": opened.structured["session_id"],
            "page_id": opened.structured["page_id"],
        }
        if phase in {"unowned", "static_change"}:
            # A separate non-durable tool exercises the supported direct entrance.
            tool = BrowserSessionTool(_backend=_VisualBackend(), visual_policy=tool.visual_policy)
            ctx = _context(tmp_path, artifact_store=store).model_copy(
                update={
                    "invocation_secret_snapshot_provider": tracker.snapshot
                    if phase == "unowned"
                    else None,
                    "invocation_secret_redactor": lambda: tracker.redactor,
                }
            )
            opened = await tool.run(
                ctx,
                {"operation": "navigate", "url": "https://example.test", "operation_id": "open"},
            )
            request.update(
                session_id=opened.structured["session_id"], page_id=opened.structured["page_id"]
            )
        else:
            ctx = _durable_context(
                tmp_path,
                args=request,
                records=records,
                tool_call_id="capture",
                secret_tracker=tracker,
            )
            # Replace the store before binding would discard runtime authority;
            # the context's existing LocalArtifactStore instance shares this path.
            object.__setattr__(ctx, "artifact_store", store)
        if phase == "pending":
            tracker.begin_resolution()
        call = asyncio.create_task(tool.run(ctx, request))
        if phase in {"pending", "unowned"}:
            result = await asyncio.wait_for(call, 1)
            assert result.is_error and result.structured["error"] == "policy_denied"
            assert not entered.is_set()
            assert not (await store.list(session_id=ctx.session_id)).artifacts
        else:
            await asyncio.wait_for(entered.wait(), 1)
            if phase == "static_change":
                tracker._redactor = SecretRedactor("visual-late-secret-canary")
            else:
                with pytest.raises(RuntimeError, match="after tool publication"):
                    tracker.begin_resolution()
            release.set()
            result = await asyncio.wait_for(call, 1)
            if phase == "static_change":
                assert result.is_error and result.structured["error"] == "policy_denied"
                assert not result.artifacts
            else:
                assert not result.is_error, result.model_dump_json()
                assert len(result.artifacts) == 1
                assert (await tool.run(ctx, request)).model_dump() == result.model_dump()
                # A later incompatible caller registry must not receive cached pixels.
                tracker._redactor = SecretRedactor("visual-late-secret-canary")
                replay = await tool.run(ctx, request)
                assert replay.structured["error"] == "policy_denied" and not replay.artifacts
        assert "visual-late-secret-canary" not in json.dumps(records)
        if result.is_error:
            assert not result.artifacts
            assert all(not value.get("result", {}).get("artifacts") for value in records.values())

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "termination", ["restoration_timeout", "capture_timeout", "cancel", "primary_failure", "fatal"]
)
@pytest.mark.parametrize("close_fails", [False, True])
def test_visual_observation_retains_stalled_guard_until_browser_close(
    monkeypatch: pytest.MonkeyPatch,
    termination: str,
    close_fails: bool,
) -> None:
    async def scenario() -> None:
        restoring = asyncio.Event()
        release = asyncio.Event()

        async def no_census(*args: Any, **kwargs: Any) -> None:
            return None

        monkeypatch.setattr(guest, "_admit_interactive_snapshot_materialization", no_census)
        monkeypatch.setattr(guest, "_OBSERVATION_GUARD_SETTLEMENT_SECONDS", 0.02)

        class CDP:
            async def send(self, name, arguments):
                if name == "Animation.setPlaybackRate" and arguments["playbackRate"] == 1:
                    restoring.set()
                    await release.wait()
                return {}

        class Page:
            url = "https://example.test/"

            def locator(self, _name):
                return self

            async def aria_snapshot(self, **kwargs):
                if termination == "capture_timeout":
                    await asyncio.Event().wait()
                if termination == "primary_failure":
                    raise guest.VisualGuestFailure("unstable_visual_observation")
                if termination == "fatal":
                    raise GeneratorExit("stop observation")
                return ""

            async def title(self):
                return "Visual fixture"

            async def close(self):
                if close_fails:
                    raise RuntimeError("page close failed")

            def is_closed(self):
                return False

        class Browser:
            async def close(self):
                if close_fails:
                    raise RuntimeError("browser close failed")
                release.set()

            def is_connected(self):
                return True

        daemon = guest._InteractiveDaemon("session")
        original_close = daemon.close

        async def bounded_close(*, timeout_seconds=5):
            return await original_close(timeout_seconds=0.05)

        daemon.close = bounded_close
        daemon.browser = Browser()
        state = guest._InteractivePage(
            page=Page(), session_id="session", page_id="page", cdp=CDP(), lifecycle="active"
        )
        daemon.pages["page"] = state
        daemon.active_page_id = "page"
        limits = guest._InteractiveLimits(
            **{field.name: 100 for field in fields(guest._InteractiveLimits)}
        )

        # Enter through the real daemon lock and operation owner; the test seam
        # places the same visual processing deadline around real observation.
        async def configuration(_request):
            return None

        async def page_operation(_state, _request):
            async with asyncio.timeout(0.01 if termination == "capture_timeout" else 1):
                return await daemon._observe_page(state, limits)

        daemon._ensure_configuration = configuration
        daemon._execute_page = page_operation
        request = guest._InteractiveRequest(
            operation="observe_visual",
            session_id="session",
            page_id="page",
            operation_id="observe",
            expected_revision=None,
            expected_control_epoch=None,
            ref=None,
            url=None,
            value=None,
            key=None,
            wait_ms=None,
            full_page=False,
            limits=limits,
            multi_page=False,
            popup_policy=guest._InteractivePopupPolicy("deny", (), (), ()),
            visual_policy={"max_processing_ms": 10},
        )
        call = asyncio.create_task(daemon.execute(request))
        await asyncio.wait_for(restoring.wait(), 1)
        retained = state.observation_cleanup_task
        assert retained is not None and not retained.done()
        if termination == "cancel":
            call.cancel()
            await asyncio.sleep(0)
            call.cancel()
        try:
            if termination in {"cancel", "fatal"}:
                with pytest.raises(
                    asyncio.CancelledError if termination == "cancel" else GeneratorExit
                ):
                    await asyncio.wait_for(call, 0.5)
                if termination == "cancel":
                    assert call.cancelled() and call.cancelling() == 2
            else:
                result = await asyncio.wait_for(call, 0.5)
                assert result["allocation_disposition"] == (
                    "uncertain" if close_fails else "retired"
                )
                assert not result.get("observation")
            assert daemon.closing and daemon.close_requested.is_set()
            assert retained.cancelling() == 0
            if close_fails:
                assert state.observation_cleanup_task is retained and not retained.done()
            else:
                assert retained.done() and state.observation_cleanup_task is None
            with pytest.raises(guest._GuestFailure, match="session_closed"):
                await daemon.execute(
                    guest._InteractiveRequest(**{**vars(request), "operation_id": "retry"})
                )
        finally:
            release.set()
            await retained

    asyncio.run(scenario())
