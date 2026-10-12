"""Browser-session durable reconnect, receipt and authority recovery contracts."""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from tests.core.browser_session_support import (
    _durable_context,
    _FakeBrowserBackend,
    _recover_durable_browser_result,
    _tool,
)

import cayu.tools.browser_session as browser_session_module
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import RunRequest
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.base import ToolContext
from cayu.tools.browser_session import (
    BrowserBackendFailure,
    BrowserBackendResponse,
    BrowserSessionTool,
)
from cayu.vaults.redaction import SecretRedactor


def test_browser_session_reconnects_fresh_tool_to_exact_live_allocation(tmp_path: Path) -> None:
    async def scenario() -> None:
        backend = _FakeBrowserBackend()
        records: dict[str, dict[str, Any]] = {}
        navigate_args = {
            "operation": "navigate",
            "url": "https://example.test/form",
            "operation_id": "durable-navigate-1",
        }
        first = await _tool(backend).run(
            _durable_context(
                tmp_path,
                args=navigate_args,
                records=records,
                tool_call_id="navigate-call",
            ),
            navigate_args,
        )
        assert first.is_error is False
        browser_session_id = first.structured["session_id"]
        page_id = first.structured["page_id"]

        observe_args = {
            "operation": "observe",
            "session_id": browser_session_id,
            "page_id": page_id,
            "operation_id": "durable-observe-1",
        }
        recovered = await _tool(backend).run(
            _durable_context(
                tmp_path,
                args=observe_args,
                records=records,
                tool_call_id="observe-call",
            ),
            observe_args,
        )

        assert recovered.is_error is False
        assert recovered.structured["session_id"] == browser_session_id
        assert recovered.structured["revision"] == "br_revision_2"
        assert len(backend.calls) == 2

    asyncio.run(scenario())


def test_browser_session_durable_retirement_releases_replacement_capacity(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        backend = _FakeBrowserBackend()
        records: dict[str, dict[str, Any]] = {}

        def bounded_tool() -> BrowserSessionTool:
            return BrowserSessionTool(max_sessions=1, _backend=backend)

        navigate_args = {
            "operation": "navigate",
            "url": "https://example.test/one",
            "operation_id": "durable-retirement-navigate",
        }
        opened = await bounded_tool().run(
            _durable_context(
                tmp_path,
                args=navigate_args,
                records=records,
                tool_call_id="durable-retirement-navigate-call",
            ),
            navigate_args,
        )
        assert opened.is_error is False

        backend.failure = BrowserBackendFailure("session_closed")
        backend.failure_disposition = "retired"
        observe_args = {
            "operation": "observe",
            "session_id": opened.structured["session_id"],
            "page_id": opened.structured["page_id"],
            "operation_id": "durable-retirement-observe",
        }
        retired = await bounded_tool().run(
            _durable_context(
                tmp_path,
                args=observe_args,
                records=records,
                tool_call_id="durable-retirement-observe-call",
            ),
            observe_args,
        )
        assert retired.structured["error"] == "allocation_lost"

        backend.failure = None
        backend.failure_disposition = "uncertain"
        replacement_args = {
            "operation": "navigate",
            "url": "https://example.test/two",
            "operation_id": "durable-retirement-replacement",
        }
        replacement = await bounded_tool().run(
            _durable_context(
                tmp_path,
                args=replacement_args,
                records=records,
                tool_call_id="durable-retirement-replacement-call",
            ),
            replacement_args,
        )

        assert replacement.is_error is False
        assert [call["operation"] for call in backend.calls] == [
            "navigate",
            "observe",
            "navigate",
        ]

    asyncio.run(scenario())


def test_browser_session_assigns_random_capabilities_and_replays_recorded_assignment(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        args = {
            "operation": "navigate",
            "url": "https://example.test/form",
            "operation_id": "random-browser-capability",
        }
        first_records: dict[str, dict[str, Any]] = {}
        second_records: dict[str, dict[str, Any]] = {}
        first_backend = _FakeBrowserBackend()
        second_backend = _FakeBrowserBackend()

        first = await _tool(first_backend).run(
            _durable_context(tmp_path, args=args, records=first_records),
            args,
        )
        second = await _tool(second_backend).run(
            _durable_context(tmp_path, args=args, records=second_records),
            args,
        )
        replay = await _recover_durable_browser_result(
            _tool(first_backend),
            args=args,
            records=first_records,
        )

        assert first.structured["session_id"] != second.structured["session_id"]
        assert first.structured["page_id"] != second.structured["page_id"]
        assert replay.structured["session_id"] == first.structured["session_id"]
        assert replay.structured["page_id"] == first.structured["page_id"]
        assert len(first_backend.calls) == 1

    asyncio.run(scenario())


def test_browser_session_reconciliation_publishes_positive_allocation_loss(
    tmp_path: Path,
) -> None:
    class _LostAllocationBackend(_FakeBrowserBackend):
        async def reconcile(
            self,
            ctx: ToolContext,
            request: dict[str, Any],
        ) -> BrowserBackendResponse:
            del ctx, request
            return BrowserBackendResponse(
                failure=BrowserBackendFailure("allocation_lost"),
                allocation_disposition="retired",
            )

    async def scenario() -> None:
        records: dict[str, dict[str, Any]] = {}
        args = {
            "operation": "navigate",
            "url": "https://example.test/form",
            "operation_id": "lost-allocation-reconciliation",
        }
        ambiguous = await _tool(_FakeBrowserBackend()).run(
            _durable_context(
                tmp_path,
                args=args,
                records=records,
                fail_before_state="terminal",
            ),
            args,
        )
        lost = await _tool(_LostAllocationBackend()).run(
            _durable_context(tmp_path, args=args, records=records),
            args,
        )

        assert ambiguous.structured["error"] == "outcome_ambiguous"
        assert lost.structured["error"] == "allocation_lost"
        assert lost.structured["allocation_disposition"] == "retired"
        operation_record = next(
            record
            for record in records.values()
            if record.get("record_type") == "cayu.browser-operation"
        )
        session_record = next(
            record
            for record in records.values()
            if record.get("record_type") == "cayu.browser-session"
        )
        parent_record = next(
            record
            for record in records.values()
            if record.get("record_type") == "cayu.browser-parent"
        )
        assert operation_record["state"] == "terminal"
        assert session_record["state"] == "closed"
        assert parent_record["live_session_ids"] == []

    asyncio.run(scenario())


def test_browser_session_durable_limits_survive_fresh_tool_processes(tmp_path: Path) -> None:
    async def scenario() -> None:
        backend = _FakeBrowserBackend()
        records: dict[str, dict[str, Any]] = {}

        def bounded_tool(*, max_operations: int = 4) -> BrowserSessionTool:
            return BrowserSessionTool(
                max_sessions=1,
                max_operations=max_operations,
                _backend=backend,
            )

        first_args = {
            "operation": "navigate",
            "url": "https://example.test/one",
            "operation_id": "durable-quota-navigate-one",
        }
        first = await bounded_tool().run(
            _durable_context(
                tmp_path,
                args=first_args,
                records=records,
                tool_call_id="quota-navigate-one",
            ),
            first_args,
        )
        first_session_id = first.structured["session_id"]
        call_count = len(backend.calls)

        second_args = {
            "operation": "navigate",
            "url": "https://example.test/two",
            "operation_id": "durable-quota-navigate-two",
        }
        session_exhausted = await bounded_tool().run(
            _durable_context(
                tmp_path,
                args=second_args,
                records=records,
                tool_call_id="quota-navigate-two",
            ),
            second_args,
        )
        assert session_exhausted.structured["error"] == "resource_exhausted"
        assert len(backend.calls) == call_count

        close_args = {
            "operation": "close",
            "session_id": first_session_id,
            "operation_id": "durable-quota-close-one",
        }
        closed = await bounded_tool().run(
            _durable_context(
                tmp_path,
                args=close_args,
                records=records,
                tool_call_id="quota-close-one",
            ),
            close_args,
        )
        assert closed.is_error is False

        third_args = {
            "operation": "navigate",
            "url": "https://example.test/three",
            "operation_id": "durable-quota-navigate-three",
        }
        third = await bounded_tool().run(
            _durable_context(
                tmp_path,
                args=third_args,
                records=records,
                tool_call_id="quota-navigate-three",
            ),
            third_args,
        )
        assert third.is_error is False
        third_close_args = {
            "operation": "close",
            "session_id": third.structured["session_id"],
            "operation_id": "durable-quota-close-three",
        }
        third_closed = await bounded_tool().run(
            _durable_context(
                tmp_path,
                args=third_close_args,
                records=records,
                tool_call_id="quota-close-three",
            ),
            third_close_args,
        )
        assert third_closed.is_error is False

        operation_records: dict[str, dict[str, Any]] = {}
        operation_args = {
            "operation": "navigate",
            "url": "https://example.test/limited",
            "operation_id": "durable-operation-one",
        }
        operation_first = await bounded_tool(max_operations=1).run(
            _durable_context(
                tmp_path,
                args=operation_args,
                records=operation_records,
                tool_call_id="operation-one",
            ),
            operation_args,
        )
        observe_args = {
            "operation": "observe",
            "session_id": operation_first.structured["session_id"],
            "page_id": operation_first.structured["page_id"],
            "operation_id": "durable-operation-two",
        }
        operation_exhausted = await bounded_tool(max_operations=1).run(
            _durable_context(
                tmp_path,
                args=observe_args,
                records=operation_records,
                tool_call_id="operation-two",
            ),
            observe_args,
        )
        cleanup_args = {
            "operation": "close",
            "session_id": operation_first.structured["session_id"],
            "operation_id": "durable-operation-cleanup",
        }
        cleanup = await bounded_tool(max_operations=1).run(
            _durable_context(
                tmp_path,
                args=cleanup_args,
                records=operation_records,
                tool_call_id="operation-cleanup",
            ),
            cleanup_args,
        )

        assert operation_exhausted.structured["error"] == "resource_exhausted"
        assert cleanup.is_error is False

    asyncio.run(scenario())


def test_browser_session_uncertain_reconnect_invalidates_refs_until_observe(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        backend = _FakeBrowserBackend()
        records: dict[str, dict[str, Any]] = {}
        navigate_args = {
            "operation": "navigate",
            "url": "https://example.test/form",
            "operation_id": "durable-navigate-uncertain",
        }
        first = await _tool(backend).run(
            _durable_context(tmp_path, args=navigate_args, records=records),
            navigate_args,
        )
        session_record = next(
            record
            for record in records.values()
            if record.get("record_type") == "cayu.browser-session"
        )
        session_record["state"] = "uncertain"
        session_record["refs_valid"] = False
        fresh_tool = _tool(backend)
        click_args = {
            "operation": "click",
            "session_id": first.structured["session_id"],
            "page_id": first.structured["page_id"],
            "expected_revision": first.structured["revision"],
            "expected_control_epoch": first.structured["control_epoch"],
            "ref": "ref_name",
            "operation_id": "uncertain-click",
        }

        refused = await fresh_tool.run(
            _durable_context(
                tmp_path,
                args=click_args,
                records=records,
                tool_call_id="uncertain-click-call",
            ),
            click_args,
        )
        observe_args = {
            "operation": "observe",
            "session_id": first.structured["session_id"],
            "page_id": first.structured["page_id"],
            "operation_id": "uncertain-observe",
        }
        observed = await fresh_tool.run(
            _durable_context(
                tmp_path,
                args=observe_args,
                records=records,
                tool_call_id="uncertain-observe-call",
            ),
            observe_args,
        )

        assert refused.structured["error"] == "stale_observation"
        assert observed.is_error is False
        assert observed.structured["revision"] == "br_revision_2"
        assert len(backend.calls) == 2

    asyncio.run(scenario())


def test_browser_session_reconnect_rejects_changed_allocation_before_dispatch(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        backend = _FakeBrowserBackend()
        records: dict[str, dict[str, Any]] = {}
        navigate_args = {
            "operation": "navigate",
            "url": "https://example.test/form",
            "operation_id": "durable-navigate-1",
        }
        first = await _tool(backend).run(
            _durable_context(tmp_path, args=navigate_args, records=records),
            navigate_args,
        )
        call_count = len(backend.calls)
        observe_args = {
            "operation": "observe",
            "session_id": first.structured["session_id"],
            "page_id": first.structured["page_id"],
            "operation_id": "durable-observe-1",
        }

        refused = await _tool(backend).run(
            _durable_context(
                tmp_path,
                args=observe_args,
                records=records,
                allocation_fingerprint="c" * 64,
                tool_call_id="observe-call",
            ),
            observe_args,
        )

        assert refused.structured["error"] == "allocation_lost"
        assert len(backend.calls) == call_count

    asyncio.run(scenario())


def test_browser_session_live_state_rejects_changed_allocation_before_preflight(
    tmp_path: Path,
) -> None:
    class _PreflightTrackingBackend(_FakeBrowserBackend):
        def __init__(self) -> None:
            super().__init__()
            self.preflight_calls = 0

        async def preflight(
            self,
            ctx: ToolContext,
            request: dict[str, Any],
        ) -> BrowserBackendFailure | None:
            del ctx, request
            self.preflight_calls += 1
            return None

    async def scenario() -> None:
        backend = _PreflightTrackingBackend()
        tool = _tool(backend)
        records: dict[str, dict[str, Any]] = {}
        navigate_args = {
            "operation": "navigate",
            "url": "https://example.test/form",
            "operation_id": "live-allocation-navigate",
        }
        first = await tool.run(
            _durable_context(tmp_path, args=navigate_args, records=records),
            navigate_args,
        )
        replay_with_replaced_allocation = await tool.run(
            _durable_context(
                tmp_path,
                args=navigate_args,
                records=records,
                allocation_fingerprint="c" * 64,
                tool_call_id="live-allocation-navigate-replay-call",
            ),
            navigate_args,
        )
        replay_with_replaced_profile = await tool.run(
            _durable_context(
                tmp_path,
                args=navigate_args,
                records=records,
                execution_profile_fingerprint="d" * 64,
                tool_call_id="live-allocation-navigate-profile-replay-call",
            ),
            navigate_args,
        )
        replay_with_replaced_invocation = await tool.run(
            _durable_context(
                tmp_path,
                args=navigate_args,
                records=records,
                tool_call_id="live-allocation-navigate-other-call",
            ),
            navigate_args,
        )
        observe_args = {
            "operation": "observe",
            "session_id": first.structured["session_id"],
            "page_id": first.structured["page_id"],
            "operation_id": "live-allocation-observe",
        }

        refused = await tool.run(
            _durable_context(
                tmp_path,
                args=observe_args,
                records=records,
                allocation_fingerprint="c" * 64,
                tool_call_id="live-allocation-observe-call",
            ),
            observe_args,
        )

        assert replay_with_replaced_allocation.structured["error"] == "allocation_lost"
        assert replay_with_replaced_allocation.structured["execution"]["dispatch"] == (
            "not_started"
        )
        assert replay_with_replaced_profile.structured["error"] == "incompatible_profile"
        assert replay_with_replaced_profile.structured["execution"]["dispatch"] == ("not_started")
        assert replay_with_replaced_invocation.structured["error"] == "operation_conflict"
        assert replay_with_replaced_invocation.structured["execution"]["dispatch"] == (
            "not_started"
        )
        assert refused.structured["error"] == "allocation_lost"
        assert refused.structured["execution"]["dispatch"] == "not_started"
        assert backend.preflight_calls == 1
        assert len(backend.calls) == 1

    asyncio.run(scenario())


def test_browser_session_reconnect_requires_exact_profile_and_live_allocation(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        backend = _FakeBrowserBackend()
        records: dict[str, dict[str, Any]] = {}
        navigate_args = {
            "operation": "navigate",
            "url": "https://example.test/form",
            "operation_id": "durable-navigate-identity",
        }
        first = await _tool(backend).run(
            _durable_context(tmp_path, args=navigate_args, records=records),
            navigate_args,
        )
        observe_args = {
            "operation": "observe",
            "session_id": first.structured["session_id"],
            "page_id": first.structured["page_id"],
            "operation_id": "durable-observe-identity",
        }
        call_count = len(backend.calls)

        wrong_profile = await _tool(backend).run(
            _durable_context(
                tmp_path,
                args=observe_args,
                records=records,
                execution_profile_fingerprint="d" * 64,
                tool_call_id="observe-profile-call",
            ),
            observe_args,
        )
        no_live_allocation = await _tool(backend).run(
            _durable_context(
                tmp_path,
                args=observe_args,
                records=records,
                allocation_fingerprint=None,
                tool_call_id="observe-restoration-call",
            ),
            observe_args,
        )
        wrong_run_epoch = await _recover_durable_browser_result(
            _tool(backend),
            args=navigate_args,
            records=records,
            parent_run_epoch=2,
        )

        assert wrong_profile.structured["error"] == "incompatible_profile"
        assert "original execution profile" in wrong_profile.structured["guidance"]
        assert no_live_allocation.structured["error"] == "restoration_required"
        assert "Start a new browser session" in no_live_allocation.structured["guidance"]
        assert wrong_run_epoch.structured["error"] == "authority_expired"
        assert "current runtime authority" in wrong_run_epoch.structured["guidance"]
        assert len(backend.calls) == call_count

    asyncio.run(scenario())


def test_browser_session_seals_page_metadata_in_every_durable_record(tmp_path: Path) -> None:
    async def scenario() -> None:
        secret = "DURABLE_BROWSER_SECRET_CANARY"
        redactor = SecretRedactor(secret)
        backend = _FakeBrowserBackend(title=f"Account {secret}")
        records: dict[str, dict[str, Any]] = {}
        navigate_args = {
            "operation": "navigate",
            "url": f"https://example.test/form?token={secret}",
            "operation_id": "navigate-secret-page-metadata",
        }
        opened = await _tool(backend).run(
            _durable_context(
                tmp_path,
                args=navigate_args,
                records=records,
                secret_redactor=redactor,
            ),
            navigate_args,
        )

        assert opened.is_error is False
        assert secret not in json.dumps(records)
        session_record = next(
            record
            for record in records.values()
            if record.get("record_type") == "cayu.browser-session"
        )
        page = session_record["page_set"]["pages"][0]
        assert secret not in page["url"]
        assert secret not in page["title"]

        observe_args = {
            "operation": "observe",
            "session_id": opened.structured["session_id"],
            "page_id": opened.structured["page_id"],
            "operation_id": "observe-after-secret-page-metadata",
        }
        observed = await _tool(backend).run(
            _durable_context(
                tmp_path,
                args=observe_args,
                records=records,
                tool_call_id="tool-call-2",
                secret_redactor=redactor,
            ),
            observe_args,
        )

        assert observed.is_error is False
        assert secret not in json.dumps(records)

    asyncio.run(scenario())


def test_browser_session_terminal_receipt_survives_later_allocation_loss(tmp_path: Path) -> None:
    async def scenario() -> None:
        backend = _FakeBrowserBackend()
        records: dict[str, dict[str, Any]] = {}
        args = {
            "operation": "navigate",
            "url": "https://example.test/form",
            "operation_id": "terminal-before-allocation-loss",
        }
        tool = _tool(backend)
        terminal = await tool.run(
            _durable_context(tmp_path, args=args, records=records),
            args,
        )

        without_allocation = await _recover_durable_browser_result(
            tool,
            args=args,
            records=records,
            allocation_fingerprint=None,
        )
        replacement_allocation = await _recover_durable_browser_result(
            tool,
            args=args,
            records=records,
            allocation_fingerprint="c" * 64,
        )

        assert without_allocation == terminal
        assert replacement_allocation == terminal
        assert len(backend.calls) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("identity_field", ["model_step_id", "model_attempt_id"])
def test_browser_session_recovery_authenticates_complete_model_attempt_identity(
    tmp_path: Path,
    identity_field: str,
) -> None:
    async def scenario() -> None:
        backend = _FakeBrowserBackend()
        records: dict[str, dict[str, Any]] = {}
        args = {
            "operation": "navigate",
            "url": "https://example.test/form",
            "operation_id": f"tampered-{identity_field}",
        }
        tool = _tool(backend)
        await tool.run(_durable_context(tmp_path, args=args, records=records), args)
        operation_record = next(
            record
            for record in records.values()
            if record.get("record_type") == "cayu.browser-operation"
            and record.get("state") == "terminal"
        )
        operation_record[identity_field] = f"wrong-{identity_field}"

        recovered = await _recover_durable_browser_result(
            tool,
            args=args,
            records=records,
        )

        assert recovered.structured["error"] == "authority_expired"
        assert len(backend.calls) == 1

    asyncio.run(scenario())


def test_browser_session_recovery_finds_hook_effective_operation_identity(tmp_path: Path) -> None:
    async def scenario() -> None:
        backend = _FakeBrowserBackend()
        records: dict[str, dict[str, Any]] = {}
        original_args = {
            "operation": "navigate",
            "url": "https://example.test/original",
            "operation_id": "operation-before-hook",
        }
        effective_args = {
            "operation": "navigate",
            "url": "https://example.test/effective",
            "operation_id": "operation-after-hook",
        }
        tool = _tool(backend)
        terminal = await tool.run(
            _durable_context(tmp_path, args=effective_args, records=records),
            effective_args,
        )

        recovered = await _recover_durable_browser_result(
            tool,
            args=original_args,
            records=records,
        )
        locator = next(
            record
            for record in records.values()
            if record.get("record_type") == "cayu.browser-operation-locator"
        )

        assert recovered == terminal
        assert len(backend.calls) == 1
        assert "arguments" not in locator
        assert "operation_id" not in locator
        assert effective_args["url"] not in json.dumps(locator)

    asyncio.run(scenario())


def test_browser_session_operation_conflict_fails_before_runner_preflight(tmp_path: Path) -> None:
    async def scenario() -> None:
        backend = _FakeBrowserBackend()
        records: dict[str, dict[str, Any]] = {}
        original = {
            "operation": "navigate",
            "url": "https://example.test/original",
            "operation_id": "preflight-conflict",
        }
        conflicting = {
            **original,
            "url": "https://example.test/conflicting",
        }
        await _tool(backend).run(
            _durable_context(tmp_path, args=original, records=records),
            original,
        )

        refused = await _tool(backend).run(
            _durable_context(tmp_path, args=conflicting, records=records),
            conflicting,
        )

        assert refused.structured["error"] == "operation_conflict"
        assert len(backend.preflight_calls) == 1
        assert len(backend.calls) == 1

    asyncio.run(scenario())


def test_browser_session_large_default_observation_survives_durable_recovery(
    tmp_path: Path,
) -> None:
    snapshot = '- paragraph "' + "é" * (512 * 1024) + '"'

    class LargeBackend(_FakeBrowserBackend):
        async def execute(self, ctx, request):
            response = await super().execute(ctx, request)
            return replace(
                response,
                observation=response.observation.model_copy(update={"snapshot": snapshot}),
            )

    async def scenario() -> None:
        backend = LargeBackend()
        records: dict[str, dict[str, Any]] = {}
        args = {
            "operation": "navigate",
            "url": "https://example.test/large-document",
            "operation_id": "large-document",
        }
        tool = _tool(backend)
        assert tool.max_snapshot_bytes is None
        terminal = await tool.run(_durable_context(tmp_path, args=args, records=records), args)
        assert terminal.is_error is False
        assert terminal.structured["snapshot"] == snapshot

        recovered = await _recover_durable_browser_result(tool, args=args, records=records)

        assert recovered == terminal
        assert len(backend.calls) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("character", "size", "refused"),
    [
        ("x", 6 * 1024 * 1024, False),
        ("x", 8 * 1024 * 1024, True),
        ("\x01", 2 * 1024 * 1024, True),
        ("x", 17 * 1024 * 1024, True),
    ],
    ids=["large-success", "duplicated-result", "json-escaping", "structured-result"],
)
def test_browser_session_observation_durable_size_settles_in_sqlite(
    tmp_path: Path, character: str, size: int, refused: bool
) -> None:
    snapshot = "[Bounded rendered text; main document only; element references unavailable]\n"
    snapshot += character * size

    class LargeBackend(_FakeBrowserBackend):
        async def execute(self, ctx, request):
            response = await super().execute(ctx, request)
            return replace(
                response,
                observation=response.observation.model_copy(update={"snapshot": snapshot}),
            )

    async def scenario() -> None:
        path = tmp_path / "browser-observation.sqlite"
        store = SQLiteSessionStore(path)
        backend = LargeBackend()
        records: dict[str, dict[str, Any]] = {}
        args = {
            "operation": "navigate",
            "url": "https://example.test/large-document",
            "operation_id": "large-document",
        }
        try:
            await store.create(
                RunRequest(session_id="parent-session", agent_name="assistant", messages=[]),
                identity=SessionIdentity(provider_name="test", model="test"),
                interaction_started_event=Event(
                    id="large-browser-started",
                    type=EventType.INTERACTION_STARTED,
                    session_id="parent-session",
                    interaction_id="large-browser-interaction",
                    agent_name="assistant",
                ),
                interaction_source_messages=[],
            )
            terminal = await _tool(backend).run(
                _durable_context(tmp_path, args=args, records=records, session_store=store), args
            )
            assert terminal.is_error is refused
            if refused:
                assert terminal.structured["error"] == "oversized_snapshot"
                assert terminal.structured["execution"]["dispatch"] == "completed"
                limit = terminal.structured["limit"]
                assert limit["identifier"] == "durable_observation_bytes"
                assert limit["bound"] == 16 * 1024 * 1024
                assert limit["observed"] > limit["bound"]
                assert limit["measurement"] == "lower_bound"
            else:
                assert terminal.structured["snapshot"] == snapshot
            operation = next(
                value
                for value in records.values()
                if value.get("record_type") == "cayu.browser-operation"
            )
            assert operation["state"] == "terminal"
        finally:
            await store.close()
        reopened = SQLiteSessionStore(path)
        try:
            recovered = await _recover_durable_browser_result(
                _tool(backend),
                args=args,
                records={},
                session_store=reopened,
                expected_disposition="confirmed",
            )
            assert recovered == terminal
            retried = await _tool(backend).run(
                _durable_context(tmp_path, args=args, records=records, session_store=reopened), args
            )
            assert retried == terminal
            assert len(backend.calls) == 1
        finally:
            await reopened.close()

    asyncio.run(scenario())


def test_browser_session_recovery_rejects_oversized_terminal_observation(tmp_path: Path) -> None:
    async def scenario() -> None:
        backend = _FakeBrowserBackend()
        records: dict[str, dict[str, Any]] = {}
        args = {
            "operation": "navigate",
            "url": "https://example.test/form",
            "operation_id": "bounded-terminal-observation",
        }
        tool = BrowserSessionTool(max_snapshot_bytes=128, _backend=backend)
        await tool.run(_durable_context(tmp_path, args=args, records=records), args)
        operation_record = next(
            record
            for record in records.values()
            if record.get("record_type") == "cayu.browser-operation"
            and record.get("state") == "terminal"
        )
        operation_record["result"]["structured"]["snapshot"] = "x" * 129

        recovered = await _recover_durable_browser_result(
            tool,
            args=args,
            records=records,
        )

        assert recovered.structured["error"] == "authority_expired"
        assert len(backend.calls) == 1

    asyncio.run(scenario())


def test_browser_session_reconnect_rejects_unbounded_session_refs_before_dispatch(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        backend = _FakeBrowserBackend()
        records: dict[str, dict[str, Any]] = {}
        navigate_args = {
            "operation": "navigate",
            "url": "https://example.test/form",
            "operation_id": "bounded-session-record",
        }
        tool = BrowserSessionTool(max_refs=2, _backend=backend)
        opened = await tool.run(
            _durable_context(tmp_path, args=navigate_args, records=records),
            navigate_args,
        )
        session_record = next(
            record
            for record in records.values()
            if record.get("record_type") == "cayu.browser-session"
        )
        page_authority = next(
            item
            for item in session_record["page_authorities"]
            if item["page_id"] == opened.structured["page_id"]
        )
        page_authority["refs"] = ["ref_one", "ref_two", "ref_three"]
        observe_args = {
            "operation": "observe",
            "session_id": opened.structured["session_id"],
            "page_id": opened.structured["page_id"],
            "operation_id": "bounded-session-observe",
        }

        refused = await BrowserSessionTool(max_refs=2, _backend=backend).run(
            _durable_context(
                tmp_path,
                args=observe_args,
                records=records,
                tool_call_id="bounded-session-observe-call",
            ),
            observe_args,
        )

        assert refused.structured["error"] == "restoration_required"
        assert len(backend.calls) == 1

    asyncio.run(scenario())


def test_browser_session_reconnect_rejects_per_page_ref_overflow_before_dispatch(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        backend = _FakeBrowserBackend()
        records: dict[str, dict[str, Any]] = {}
        navigate_args = {
            "operation": "navigate",
            "url": "https://example.test/form",
            "operation_id": "per-page-session-record",
        }
        configuration: dict[str, Any] = {
            "max_refs": 2,
            "max_refs_per_page": 2,
            "max_total_refs": 8,
            "_backend": backend,
        }
        opened = await BrowserSessionTool(**configuration).run(
            _durable_context(tmp_path, args=navigate_args, records=records),
            navigate_args,
        )
        session_record = next(
            record
            for record in records.values()
            if record.get("record_type") == "cayu.browser-session"
        )
        session_record["page_set"]["pages"][0]["ref_count"] = 3
        session_record["page_set"]["total_refs"] = 3
        observe_args = {
            "operation": "observe",
            "session_id": opened.structured["session_id"],
            "page_id": opened.structured["page_id"],
            "operation_id": "per-page-session-observe",
        }

        refused = await BrowserSessionTool(**configuration).run(
            _durable_context(
                tmp_path,
                args=observe_args,
                records=records,
                tool_call_id="per-page-session-observe-call",
            ),
            observe_args,
        )

        assert refused.structured["error"] == "restoration_required"
        assert len(backend.calls) == 1

    asyncio.run(scenario())


def test_browser_session_recovery_rejects_terminal_per_page_ref_overflow(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        backend = _FakeBrowserBackend()
        records: dict[str, dict[str, Any]] = {}
        args = {
            "operation": "navigate",
            "url": "https://example.test/form",
            "operation_id": "terminal-per-page-ref-overflow",
        }
        tool = BrowserSessionTool(
            max_refs=2,
            max_refs_per_page=2,
            max_total_refs=8,
            _backend=backend,
        )
        await tool.run(_durable_context(tmp_path, args=args, records=records), args)
        terminal = next(
            record
            for record in records.values()
            if record.get("record_type") == "cayu.browser-operation"
            and record.get("state") == "terminal"
        )
        structured = terminal["result"]["structured"]
        structured["page_set"]["pages"][0]["ref_count"] = 3
        structured["page_set"]["total_refs"] = 3
        structured["pages"][0]["ref_count"] = 3
        structured["portable_result_evidence"] = (
            browser_session_module._browser_portable_result_evidence(structured)
        )

        recovered = await _recover_durable_browser_result(
            tool,
            args=args,
            records=records,
        )

        assert recovered.structured["error"] == "authority_expired"
        assert len(backend.calls) == 1

    asyncio.run(scenario())


def test_browser_session_pending_recovery_does_not_convert_store_failure_to_receipt() -> None:
    async def scenario() -> None:
        tool = _tool(_FakeBrowserBackend())

        async def fail_load(_key: str) -> dict[str, Any] | None:
            raise ConnectionError("durable operation store unavailable")

        with pytest.raises(ConnectionError, match="store unavailable"):
            await tool.reconcile_durable_tool_call(
                parent_session_id="parent-session",
                parent_run_epoch=1,
                execution_profile_fingerprint="b" * 64,
                environment_name="browser",
                environment_allocation_fingerprint="a" * 64,
                model_step_id="model-step-1",
                model_attempt_id="model-attempt-1",
                tool_round_id="tool-round-1",
                tool_call_id="tool-call-1",
                idempotency_key="tool-key-tool-call-1",
                arguments={
                    "operation": "navigate",
                    "url": "https://example.test/form",
                    "operation_id": "recovery-store-failure",
                },
                started=True,
                load_operation=fail_load,
            )

    asyncio.run(scenario())


@pytest.mark.parametrize("worker", ["same", "fresh", "sqlite"])
def test_completed_allocation_generation_preserves_old_browser_receipts(tmp_path, worker):
    async def scenario(stack):
        backend = _FakeBrowserBackend()
        records = {}
        store = None
        store_path = tmp_path / "browser-generation.sqlite"
        if worker == "sqlite":
            store = SQLiteSessionStore(store_path)
            stack.push_async_callback(store.close)
            await store.create(
                RunRequest(
                    session_id="parent-session",
                    agent_name="assistant",
                    messages=[Message.text("user", "open the page")],
                ),
                identity=SessionIdentity(provider_name="test", model="test"),
                interaction_started_event=Event(
                    id="browser-generation-started",
                    type=EventType.INTERACTION_STARTED,
                    session_id="parent-session",
                    interaction_id="browser-generation-interaction",
                    agent_name="assistant",
                ),
                interaction_source_messages=[Message.text("user", "open the page")],
            )

        def context(**kwargs):
            return _durable_context(tmp_path, records=records, session_store=store, **kwargs)

        tool = BrowserSessionTool(max_sessions=1, _backend=backend)
        first_args = {
            "operation": "navigate",
            "url": "https://example.test/form",
            "operation_id": "first",
        }
        first = await tool.run(context(args=first_args), first_args)
        assert not first.is_error
        original_records = json.loads(json.dumps(records))
        if worker == "sqlite":
            await store.close()
            store = SQLiteSessionStore(store_path)
            stack.push_async_callback(store.close)
        if worker != "same":
            tool = BrowserSessionTool(max_sessions=1, _backend=backend)
        next_args = {**first_args, "operation_id": "next"}
        refused = await tool.run(
            context(
                args=next_args,
                allocation_fingerprint="c" * 64,
                tool_call_id="next-call",
            ),
            next_args,
        )
        assert refused.structured["error"] in {"allocation_lost", "resource_exhausted"}
        assert refused.structured["execution"]["dispatch"] == "not_started"
        assert len(backend.calls) == 1
        generation = "d" * 32
        # Runtime supplies this namespace only after exact positive disposal.
        fresh = await tool.run(
            context(
                args=next_args,
                allocation_fingerprint="c" * 64,
                allocation_generation=generation,
                tool_call_id="next-call",
            ),
            next_args,
        )
        assert not fresh.is_error, fresh.structured
        assert len(backend.calls) == 2
        assert fresh.structured["session_id"] != first.structured["session_id"]
        assert all(records[key] == value for key, value in original_records.items())
        assert f"browser-parent:v2:{generation}" in records
        extra_args = {**first_args, "operation_id": "extra"}
        full = await tool.run(
            context(
                args=extra_args,
                allocation_fingerprint="c" * 64,
                allocation_generation=generation,
                tool_call_id="extra-call",
            ),
            extra_args,
        )
        assert full.structured["error"] == "resource_exhausted"
        assert full.structured["execution"]["dispatch"] == "not_started"
        assert len(backend.calls) == 2
        observe = {
            "operation": "observe",
            "session_id": first.structured["session_id"],
            "page_id": first.structured["page_id"],
            "operation_id": "old-observation",
        }
        old = await tool.run(
            context(
                args=observe,
                allocation_fingerprint="c" * 64,
                allocation_generation=generation,
                tool_call_id="old-call",
            ),
            observe,
        )
        assert old.structured["error"] == "allocation_lost"
        replay = await tool.run(
            context(
                args=first_args,
                allocation_fingerprint="c" * 64,
                allocation_generation=generation,
            ),
            first_args,
        )
        assert replay.structured["error"] in {"allocation_lost", "operation_conflict"}
        assert len(backend.calls) == 2
        assert all(records[key] == value for key, value in original_records.items())
        if store is not None:
            for key, value in records.items():
                assert await store.load_session_operation("parent-session", key) == value

    async def run():
        async with contextlib.AsyncExitStack() as stack:
            await scenario(stack)

    asyncio.run(run())
