"""Reusable interactive browser guest request and daemon test support."""

from __future__ import annotations

from typing import Any

from cayu.tools import _browser_guest


def _interactive_limits(**updates: int | None) -> _browser_guest._InteractiveLimits:
    values = {
        "max_snapshot_bytes": 1024,
        "max_dom_nodes": 100,
        "max_refs": 8,
        "max_artifact_bytes": 1024,
        "max_page_width": 1280,
        "max_page_height": 720,
        "max_page_pixels": 921600,
        "max_wait_ms": 1000,
        "idle_timeout_seconds": 60,
        "max_redirects": 2,
        "max_requests": 8,
        "max_response_bytes": 4096,
        "max_operations": 32,
        "max_pages": 1,
        "max_provisional_pages": 1,
        "max_page_creations_per_operation": 1,
        "max_total_page_creations": 1,
        "max_background_lifetime_seconds": 60,
        "max_operations_per_page": 32,
        "max_observations_per_page": 32,
        "max_total_observations": 32,
        "max_refs_per_page": 128,
        "max_total_refs": 256,
        "max_total_requests": 256,
        "max_artifacts_per_page": 8,
        "max_total_artifacts": 8,
        "max_page_cleanup_operations": 8,
        "max_scroll_repeats": 4,
        "max_upload_files": 4,
        "max_upload_file_bytes": 1024,
        "max_upload_total_bytes": 2048,
        "max_upload_filename_bytes": 255,
        "max_upload_materialization_ms": 1000,
    }
    values.update(updates)
    return _browser_guest._InteractiveLimits(**values)


class _BoundedSnapshotCdp:
    def __init__(self) -> None:
        self.scripts_disabled = False
        self.script_execution_transitions: list[bool] = []
        self.animation_playback_rates: list[int] = []
        self.runtime_evaluate_expressions: list[str] = []

    async def send(
        self,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if method == "Emulation.setScriptExecutionDisabled":
            assert params is not None
            value = params.get("value")
            assert type(value) is bool
            if not hasattr(self, "script_execution_transitions"):
                self.script_execution_transitions = []
            self.scripts_disabled = value
            self.script_execution_transitions.append(value)
            return {}
        if method == "Animation.setPlaybackRate":
            assert params is not None
            playback_rate = params.get("playbackRate")
            assert type(playback_rate) is int
            if not hasattr(self, "animation_playback_rates"):
                self.animation_playback_rates = []
            self.animation_playback_rates.append(playback_rate)
            return {}
        if method == "Page.getFrameTree":
            return {"frameTree": {"frame": {"id": "frame-main"}}}
        if method == "Fetch.enable":
            assert params == {
                "patterns": [
                    {
                        "urlPattern": "*",
                        "resourceType": "Document",
                        "requestStage": "Response",
                    }
                ]
            }
            return {}
        if method == "Page.createIsolatedWorld":
            return {"executionContextId": 1}
        if method == "Runtime.evaluate":
            assert params is not None
            expression = params.get("expression")
            assert type(expression) is str
            if not hasattr(self, "runtime_evaluate_expressions"):
                self.runtime_evaluate_expressions = []
            self.runtime_evaluate_expressions.append(expression)
            return {
                "result": {
                    "type": "object",
                    "value": {
                        "node_count": 1,
                        "source_bytes": 8,
                        "limit_exceeded": False,
                    },
                }
            }
        raise AssertionError(f"Unexpected snapshot CDP method: {method}")


def _interactive_request(
    operation: str,
    *,
    full_page: bool = False,
    limits: _browser_guest._InteractiveLimits | None = None,
) -> _browser_guest._InteractiveRequest:
    return _browser_guest._InteractiveRequest(
        operation=operation,
        session_id="bs_test",
        page_id=None if operation == "close" else "bp_test",
        expected_revision=None,
        expected_control_epoch=None,
        ref=None,
        operation_id=f"{operation}-1",
        url="https://example.test" if operation == "navigate" else None,
        value=None,
        key=None,
        wait_ms=None,
        full_page=full_page,
        direction=None,
        amount=None,
        repeat_count=None,
        upload_files=(),
        limits=limits or _interactive_limits(),
        multi_page=False,
        popup_policy=_browser_guest._InteractivePopupPolicy(
            mode="deny",
            allowed_operations=(),
            allowed_opener_origins=(),
            allowed_destination_origins=(),
        ),
    )


async def _configure_interactive_daemon_for_test(
    daemon: _browser_guest._InteractiveDaemon,
    request: _browser_guest._InteractiveRequest,
) -> None:
    """Establish the production daemon configuration around minimal test doubles."""

    context = daemon.context
    assert context is not None
    if not callable(getattr(context, "add_init_script", None)):

        async def add_init_script(_script: str) -> None:
            return None

        context.add_init_script = add_init_script
    if not callable(getattr(context, "route", None)):

        async def route(_pattern: str, _callback: Any) -> None:
            return None

        context.route = route
    await daemon._ensure_configuration(request)
    if daemon.pages and daemon.active_page_id is None:
        active = next(iter(daemon.pages.values()))
        daemon.active_page_id = active.page_id
        if active.lifecycle == "provisional":
            active.lifecycle = "active"


def _interactive_raw_request(operation: str) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "visual_policy": None,
        "protocol_version": "cayu.browser-session.v4",
        "worker_version": "19",
        "expected_playwright_version": "1.62.0",
        "operation": operation,
        "session_id": "bs_test",
        "limits": {
            "max_snapshot_bytes": 1024,
            "max_dom_nodes": 100,
            "max_refs": 8,
            "max_artifact_bytes": 1024,
            "max_page_width": 1280,
            "max_page_height": 720,
            "max_page_pixels": 921600,
            "max_wait_ms": 1000,
            "idle_timeout_seconds": 60,
            "max_redirects": 2,
            "max_requests": 8,
            "max_response_bytes": 4096,
            "max_operations": 32,
            "max_scroll_repeats": 4,
            "max_upload_files": 4,
            "max_upload_file_bytes": 1024,
            "max_upload_total_bytes": 2048,
            "max_upload_filename_bytes": 255,
            "max_upload_materialization_ms": 1000,
            "max_pages": 1,
            "max_provisional_pages": 1,
            "max_page_creations_per_operation": 1,
            "max_total_page_creations": 1,
            "max_background_lifetime_seconds": 60,
            "max_operations_per_page": 32,
            "max_observations_per_page": 32,
            "max_total_observations": 32,
            "max_refs_per_page": 128,
            "max_total_refs": 256,
            "max_total_requests": 256,
            "max_artifacts_per_page": 8,
            "max_total_artifacts": 8,
            "max_page_cleanup_operations": 8,
        },
        "page_policy": {
            "multi_page": False,
            "popup": {
                "mode": "deny",
                "allowed_operations": [],
                "allowed_opener_origins": [],
                "allowed_destination_origins": [],
            },
        },
    }
    raw["operation_id"] = f"{operation}-1"
    if operation != "close":
        raw["page_id"] = "bp_test"
    if operation == "navigate":
        raw["url"] = "https://example.test"
    return raw
