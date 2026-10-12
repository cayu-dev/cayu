"""Browser-session profile persistence, checkpoint and recovery contracts."""

from __future__ import annotations

import asyncio
import json
import traceback
import warnings
from pathlib import Path
from typing import Any

import pytest
from tests.core.browser_guest_support import (
    _configure_interactive_daemon_for_test,
    _interactive_limits,
    _interactive_request,
)
from tests.core.browser_profile_support import _browser_profile_binding, _ProfileWireRunner
from tests.core.browser_session_support import _durable_context
from tests.provider_traceback_assertions import is_cayu_source_filename

import cayu.tools.browser_session as browser_session_module
from cayu.browser_profiles import (
    BrowserProfileBinding,
    BrowserProfileCheckpointPolicy,
    BrowserProfileCookie,
    BrowserProfileOriginStorage,
    BrowserProfileStateV1,
    BrowserProfileStorageEntry,
    BrowserProfileTerminalOutcome,
    InMemoryBrowserProfileStore,
    SQLiteBrowserProfileStore,
)
from cayu.runners.base import ExecCommand, ExecResult, RunnerExecutionError
from cayu.tools import _browser_guest
from cayu.tools.browser_session import BrowserSessionTool


class _ChildCancelledProfileWireRunner(_ProfileWireRunner):
    def __init__(self, cancelled_operation: str) -> None:
        super().__init__()
        self.cancelled_operation = cancelled_operation

    async def _exec_private_browser_profile(
        self, command: ExecCommand, **kwargs: Any
    ) -> ExecResult:
        request = json.loads(kwargs["stdin"])
        if request["operation"] == self.cancelled_operation:
            self.operations.append(request["operation"])
            raise asyncio.CancelledError("private-profile-child-cancellation-canary")
        return await super()._exec_private_browser_profile(command, **kwargs)


class _BlockingProfileWireRunner(_ProfileWireRunner):
    def __init__(self, blocked_operation: str) -> None:
        super().__init__()
        self.blocked_operation = blocked_operation
        self.started = asyncio.Event()

    async def _exec_private_browser_profile(
        self, command: ExecCommand, **kwargs: Any
    ) -> ExecResult:
        request = json.loads(kwargs["stdin"])
        if request["operation"] == self.blocked_operation:
            self.operations.append(request["operation"])
            self.started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")
        return await super()._exec_private_browser_profile(command, **kwargs)


class _FailedProfileCheckpointWireRunner(_ProfileWireRunner):
    def __init__(self) -> None:
        super().__init__()
        self.fail_checkpoint = True

    async def _exec_private_browser_profile(
        self, command: ExecCommand, **kwargs: Any
    ) -> ExecResult:
        request = json.loads(kwargs["stdin"])
        if request["operation"] == "profile_checkpoint" and self.fail_checkpoint:
            self.operations.append(request["operation"])
            raise RunnerExecutionError("profile checkpoint dispatch failed")
        return await super()._exec_private_browser_profile(command, **kwargs)


class _CheckpointSettlementProcessControlStore(InMemoryBrowserProfileStore):
    async def fail_checkpoint(self, reservation, *, outcome, error_code):
        del reservation, outcome, error_code
        raise GeneratorExit("browser-profile checkpoint settlement stopped")


async def _seed_browser_profile_state(
    binding: BrowserProfileBinding,
    state: BrowserProfileStateV1,
) -> None:
    material = await binding.prepare_restore(
        operation_id="bpo_seed_restore",
        execution_profile_fingerprint="b" * 64,
        allocation_fingerprint="a" * 64,
        browser_session_id="bs_seed_profile",
    )
    await binding.complete_restore(
        material,
        outcome=BrowserProfileTerminalOutcome.SUCCEEDED,
    )
    plan = await binding.reserve_checkpoint(
        material=material,
        operation_id="bpo_seed_checkpoint",
        source_revision="br_seed_profile",
        source_operation_receipt_id="bor_seed_profile",
        source_operation_fingerprint="c" * 64,
        ambiguous_lineage=False,
    )
    await binding.publish_checkpoint(plan, state)
    await binding.release_writer(material)


def test_browser_session_profile_binding_cannot_change_after_composition() -> None:
    binding = _browser_profile_binding(
        InMemoryBrowserProfileStore(store_id="browser-session-profile-store")
    )
    replacement = _browser_profile_binding(
        InMemoryBrowserProfileStore(store_id="replacement-browser-profile-store")
    )
    tool = BrowserSessionTool(
        expected_runner_candidate="wire-browser",
        browser_profile=binding,
        max_sessions=1,
        max_wait_ms=1_000,
        idle_timeout_seconds=60,
    )
    admitted_material = tool._execution_profile_material()

    with pytest.raises(AttributeError):
        tool.browser_profile = replacement

    assert tool.browser_profile is binding
    assert tool._execution_profile_material() == admitted_material


def test_browser_profile_checkpoints_on_close_and_restores_before_navigation(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store = InMemoryBrowserProfileStore(store_id="browser-session-profile-store")
        binding = _browser_profile_binding(store)
        await binding.initialize()
        records: dict[str, dict[str, Any]] = {}
        first_runner = _ProfileWireRunner()
        first_tool = BrowserSessionTool(
            expected_runner_candidate="wire-browser",
            browser_profile=binding,
            max_sessions=1,
            max_wait_ms=1_000,
            idle_timeout_seconds=60,
        )
        first_args = {
            "operation": "navigate",
            "url": "https://example.test/login",
            "operation_id": "profile-navigate-one",
        }
        first = await first_tool.run(
            _durable_context(
                tmp_path,
                args=first_args,
                records=records,
                tool_call_id="profile-navigate-one-call",
                runner=first_runner,
            ),
            first_args,
        )
        assert first.is_error is False
        assert first_runner.operations == ["profile_restore", "navigate"]
        assert first_runner.restored_state == {"cookies": [], "origins": []}

        credential = "credential-only-in-private-profile"
        first_runner.checkpoint_state = {
            "cookies": [
                {
                    "name": "session",
                    "value": credential,
                    "domain": "example.test",
                    "path": "/",
                    "expires": -1.0,
                    "httpOnly": True,
                    "secure": True,
                    "sameSite": "Lax",
                }
            ],
            "origins": [
                {
                    "origin": "https://example.test",
                    "localStorage": [{"name": "session", "value": credential}],
                }
            ],
        }
        close_args = {
            "operation": "close",
            "session_id": first.structured["session_id"],
            "operation_id": "profile-close-one",
        }
        closed = await first_tool.run(
            _durable_context(
                tmp_path,
                args=close_args,
                records=records,
                tool_call_id="profile-close-one-call",
                runner=first_runner,
            ),
            close_args,
        )
        assert closed.is_error is False
        assert first_runner.operations[-2:] == ["profile_checkpoint", "close"]
        inspection = await store.inspect_profile(binding.access)
        assert inspection.generation == 1
        assert inspection.active_writer is False

        second_runner = _ProfileWireRunner(revision="br_profile_revision_two")
        second_tool = BrowserSessionTool(
            expected_runner_candidate="wire-browser",
            browser_profile=binding,
            max_sessions=1,
            max_wait_ms=1_000,
            idle_timeout_seconds=60,
        )
        second_args = {
            "operation": "navigate",
            "url": "https://example.test/account",
            "operation_id": "profile-navigate-two",
        }
        second = await second_tool.run(
            _durable_context(
                tmp_path,
                args=second_args,
                records=records,
                tool_call_id="profile-navigate-two-call",
                runner=second_runner,
            ),
            second_args,
        )

        assert second.is_error is False
        assert second_runner.operations == ["profile_restore", "navigate"]
        assert second_runner.restored_state == first_runner.checkpoint_state
        assert second.structured["session_id"] != first.structured["session_id"]
        assert second.structured["page_id"] != first.structured["page_id"]
        assert second.structured["revision"] != first.structured["revision"]
        assert credential not in json.dumps(records, sort_keys=True)
        assert credential not in json.dumps(second.model_dump(mode="json"), sort_keys=True)

    asyncio.run(scenario())


def test_browser_profile_on_close_failure_retains_writer_and_allows_new_retry(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store = InMemoryBrowserProfileStore(store_id="browser-session-profile-store")
        binding = _browser_profile_binding(store)
        await binding.initialize()
        records: dict[str, dict[str, Any]] = {}
        runner = _FailedProfileCheckpointWireRunner()
        tool = BrowserSessionTool(
            expected_runner_candidate="wire-browser",
            browser_profile=binding,
            max_sessions=1,
            max_wait_ms=1_000,
            idle_timeout_seconds=60,
        )
        navigate_args = {
            "operation": "navigate",
            "url": "https://example.test/login",
            "operation_id": "profile-close-failure-navigate",
        }
        navigated = await tool.run(
            _durable_context(
                tmp_path,
                args=navigate_args,
                records=records,
                tool_call_id="profile-close-failure-navigate-call",
                runner=runner,
            ),
            navigate_args,
        )
        close_args = {
            "operation": "close",
            "session_id": navigated.structured["session_id"],
            "operation_id": "profile-close-failure-one",
        }

        failed = await tool.run(
            _durable_context(
                tmp_path,
                args=close_args,
                records=records,
                tool_call_id="profile-close-failure-one-call",
                runner=runner,
            ),
            close_args,
        )

        assert failed.structured["error"] == "profile_checkpoint_failed"
        assert failed.structured["execution"]["dispatch"] == "not_started"
        assert failed.structured["allocation_disposition"] == "live"
        assert runner.operations == ["profile_restore", "navigate", "profile_checkpoint"]
        failed_inspection = await store.inspect_profile(binding.access)
        assert failed_inspection.active_writer is True
        assert failed_inspection.generation == 0
        assert failed_inspection.safe_error_code == "checkpoint_failed"

        runner.fail_checkpoint = False
        retry_args = {
            **close_args,
            "operation_id": "profile-close-failure-retry",
        }
        closed = await tool.run(
            _durable_context(
                tmp_path,
                args=retry_args,
                records=records,
                tool_call_id="profile-close-failure-retry-call",
                runner=runner,
            ),
            retry_args,
        )

        assert closed.is_error is False
        assert runner.operations[-2:] == ["profile_checkpoint", "close"]
        final_inspection = await store.inspect_profile(binding.access)
        assert final_inspection.active_writer is False
        assert final_inspection.generation == 1

    asyncio.run(scenario())


def test_malformed_profile_export_never_hides_settlement_process_control_or_plaintext(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    canary = "profile-private-diagnostic-canary"

    async def scenario() -> BaseExceptionGroup:
        store = _CheckpointSettlementProcessControlStore(store_id="browser-session-profile-store")
        binding = _browser_profile_binding(store)
        await binding.initialize()
        records: dict[str, dict[str, Any]] = {}
        runner = _ProfileWireRunner()
        tool = BrowserSessionTool(
            expected_runner_candidate="wire-browser",
            browser_profile=binding,
            max_sessions=1,
            max_wait_ms=1_000,
            idle_timeout_seconds=60,
        )
        navigate_args = {
            "operation": "navigate",
            "url": "https://example.test/login",
            "operation_id": "profile-private-diagnostic-navigate",
        }
        navigated = await tool.run(
            _durable_context(
                tmp_path,
                args=navigate_args,
                records=records,
                tool_call_id="profile-private-diagnostic-navigate-call",
                runner=runner,
            ),
            navigate_args,
        )
        runner.checkpoint_state = {
            "cookies": [
                {
                    "name": "session",
                    "value": canary,
                    "domain": "example.test",
                    "path": "/",
                    "expires": -1,
                    "httpOnly": True,
                    "secure": True,
                    "sameSite": "Lax",
                    "unexpected": True,
                }
            ],
            "origins": [],
        }
        close_args = {
            "operation": "close",
            "session_id": navigated.structured["session_id"],
            "operation_id": "profile-private-diagnostic-close",
        }
        with pytest.raises(BaseExceptionGroup) as raised:
            await tool.run(
                _durable_context(
                    tmp_path,
                    args=close_args,
                    records=records,
                    tool_call_id="profile-private-diagnostic-close-call",
                    runner=runner,
                ),
                close_args,
            )
        return raised.value

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        failure = asyncio.run(scenario())

    pending = [failure]
    seen: set[int] = set()
    rendered: list[str] = []
    generator_exits = 0
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        generator_exits += type(current) is GeneratorExit
        rendered.extend((str(current), repr(current), repr(getattr(current, "__notes__", ()))))
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
        for frame, _line_number in traceback.walk_tb(current.__traceback__):
            if is_cayu_source_filename(frame.f_code.co_filename):
                rendered.append(repr(frame.f_locals))
    captured = capsys.readouterr()
    rendered.extend((captured.out, captured.err))
    rendered.extend(str(item.message) for item in captured_warnings)
    rendered.extend(record.getMessage() for record in caplog.records)

    assert generator_exits == 1
    assert canary not in "\n".join(rendered)


def test_browser_profile_rejected_import_is_definite_and_prevents_navigation(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store = InMemoryBrowserProfileStore(store_id="browser-session-profile-store")
        binding = _browser_profile_binding(store)
        await binding.initialize()
        runner = _ProfileWireRunner()
        runner.reject_restore = True
        args = {
            "operation": "navigate",
            "url": "https://example.test/login",
            "operation_id": "profile-rejected-restore",
        }
        result = await BrowserSessionTool(
            expected_runner_candidate="wire-browser",
            browser_profile=binding,
            max_sessions=1,
            max_wait_ms=1_000,
            idle_timeout_seconds=60,
        ).run(
            _durable_context(
                tmp_path,
                args=args,
                records={},
                tool_call_id="profile-rejected-restore-call",
                runner=runner,
            ),
            args,
        )

        assert result.structured["error"] == "restoration_required"
        assert result.structured["execution"]["dispatch"] == "completed"
        assert result.structured["allocation_disposition"] == "retired"
        assert runner.operations == ["profile_restore"]
        inspection = await store.inspect_profile(binding.access)
        assert inspection.active_writer is False
        assert inspection.safe_error_code == "restore_rejected"

    asyncio.run(scenario())


def test_browser_profile_writer_refusal_retires_unstarted_browser_allocation(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store = InMemoryBrowserProfileStore(store_id="browser-session-profile-store")
        binding = _browser_profile_binding(store)
        await binding.initialize()
        held = await binding.prepare_restore(
            operation_id="held-profile-restore",
            execution_profile_fingerprint="1" * 64,
            allocation_fingerprint="2" * 64,
            browser_session_id="held-browser-session",
        )
        runner = _ProfileWireRunner()
        records: dict[str, dict[str, Any]] = {}
        args = {
            "operation": "navigate",
            "url": "https://example.test/login",
            "operation_id": "profile-writer-refused",
        }

        result = await BrowserSessionTool(
            expected_runner_candidate="wire-browser",
            browser_profile=binding,
            max_sessions=1,
            max_wait_ms=1_000,
            idle_timeout_seconds=60,
        ).run(
            _durable_context(
                tmp_path,
                args=args,
                records=records,
                tool_call_id="profile-writer-refused-call",
                runner=runner,
            ),
            args,
        )

        assert result.structured["error"] == "restoration_required"
        assert result.structured["execution"]["dispatch"] == "completed"
        assert result.structured["allocation_disposition"] == "retired"
        assert runner.operations == []
        session_records = [
            record
            for record in records.values()
            if record.get("record_type") == "cayu.browser-session"
        ]
        parent_records = [
            record
            for record in records.values()
            if record.get("record_type") == "cayu.browser-parent"
        ]
        assert len(session_records) == 1
        assert session_records[0]["state"] == "closed"
        assert len(parent_records) == 1
        assert parent_records[0]["live_session_ids"] == []
        inspection = await store.inspect_profile(binding.access)
        assert inspection.active_writer is True
        assert inspection.active_allocation_fingerprint == "2" * 64
        await binding.complete_restore(
            held,
            outcome=BrowserProfileTerminalOutcome.FAILED,
            error_code="restore_rejected",
        )

    asyncio.run(scenario())


def test_browser_profile_child_restore_cancellation_does_not_cancel_owner(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store = InMemoryBrowserProfileStore(store_id="browser-session-profile-store")
        binding = _browser_profile_binding(store)
        await binding.initialize()
        records: dict[str, dict[str, Any]] = {}
        runner = _ChildCancelledProfileWireRunner("profile_restore")
        args = {
            "operation": "navigate",
            "url": "https://example.test/login",
            "operation_id": "profile-child-cancelled-restore",
        }
        owner = asyncio.current_task()
        assert owner is not None
        cancellation_requests = owner.cancelling()
        result = await BrowserSessionTool(
            expected_runner_candidate="wire-browser",
            browser_profile=binding,
            max_sessions=1,
            max_wait_ms=1_000,
            idle_timeout_seconds=60,
        ).run(
            _durable_context(
                tmp_path,
                args=args,
                records=records,
                tool_call_id="profile-child-cancelled-restore-call",
                runner=runner,
            ),
            args,
        )

        assert owner.cancelling() == cancellation_requests
        assert result.structured["error"] == "outcome_ambiguous"
        assert result.structured["execution"]["dispatch"] == "acknowledgement_lost"
        assert runner.operations == ["profile_restore"]
        inspection = await store.inspect_profile(binding.access)
        assert inspection.active_writer is True
        assert inspection.safe_error_code == "restore_outcome_unknown"
        published = json.dumps(
            {"records": records, "result": result.model_dump(mode="json")},
            sort_keys=True,
        )
        assert "private-profile-child-cancellation-canary" not in published

    asyncio.run(scenario())


def test_browser_profile_owner_cancellation_during_restore_is_delivered_once(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store = InMemoryBrowserProfileStore(store_id="browser-session-profile-store")
        binding = _browser_profile_binding(store)
        await binding.initialize()
        records: dict[str, dict[str, Any]] = {}
        runner = _BlockingProfileWireRunner("profile_restore")
        args = {
            "operation": "navigate",
            "url": "https://example.test/login",
            "operation_id": "profile-owner-cancelled-restore",
        }
        ctx = _durable_context(
            tmp_path,
            args=args,
            records=records,
            tool_call_id="profile-owner-cancelled-restore-call",
            runner=runner,
        )
        tool = BrowserSessionTool(
            expected_runner_candidate="wire-browser",
            browser_profile=binding,
            max_sessions=1,
            max_wait_ms=1_000,
            idle_timeout_seconds=60,
        )

        owner = asyncio.create_task(tool.run(ctx, args))
        await runner.started.wait()
        owner.cancel("cancel browser profile restore")
        assert owner.cancelling() == 1
        with pytest.raises(asyncio.CancelledError, match="cancel browser profile restore"):
            await owner

        assert owner.cancelled() is True
        assert owner.cancelling() == 1
        inspection = await store.inspect_profile(binding.access)
        assert inspection.active_writer is True
        assert inspection.safe_error_code == "restore_outcome_unknown"
        replay = await tool.run(ctx, args)
        assert replay.structured["error"] == "outcome_ambiguous"
        assert replay.structured["execution"]["dispatch"] == "acknowledgement_lost"
        assert runner.operations == ["profile_restore"]

    asyncio.run(scenario())


def test_browser_profile_child_checkpoint_cancellation_does_not_cancel_owner(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store = InMemoryBrowserProfileStore(store_id="browser-session-profile-store")
        binding = _browser_profile_binding(
            store,
            checkpoint_policy=BrowserProfileCheckpointPolicy.AFTER_TERMINAL_OPERATION,
        )
        await binding.initialize()
        records: dict[str, dict[str, Any]] = {}
        runner = _ChildCancelledProfileWireRunner("profile_checkpoint")
        args = {
            "operation": "navigate",
            "url": "https://example.test/login",
            "operation_id": "profile-child-cancelled-checkpoint",
        }
        owner = asyncio.current_task()
        assert owner is not None
        cancellation_requests = owner.cancelling()
        tool = BrowserSessionTool(
            expected_runner_candidate="wire-browser",
            browser_profile=binding,
            max_sessions=1,
            max_wait_ms=1_000,
            idle_timeout_seconds=60,
        )
        result = await tool.run(
            _durable_context(
                tmp_path,
                args=args,
                records=records,
                tool_call_id="profile-child-cancelled-checkpoint-call",
                runner=runner,
            ),
            args,
        )

        assert owner.cancelling() == cancellation_requests
        assert result.is_error is False
        assert runner.operations == ["profile_restore", "navigate", "profile_checkpoint"]
        inspection = await store.inspect_profile(binding.access)
        assert inspection.active_writer is True
        assert inspection.safe_error_code == "checkpoint_outcome_unknown"
        published = json.dumps(
            {"records": records, "result": result.model_dump(mode="json")},
            sort_keys=True,
        )
        assert "private-profile-child-cancellation-canary" not in published

        close_args = {
            "operation": "close",
            "session_id": result.structured["session_id"],
            "operation_id": "profile-close-after-child-checkpoint-cancellation",
        }
        closed = await tool.run(
            _durable_context(
                tmp_path,
                args=close_args,
                records=records,
                tool_call_id="profile-close-after-child-checkpoint-cancellation-call",
                runner=runner,
            ),
            close_args,
        )
        assert closed.is_error is False
        assert (await store.inspect_profile(binding.access)).active_writer is False

    asyncio.run(scenario())


def test_browser_profile_owner_cancellation_during_checkpoint_is_delivered_once(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store = InMemoryBrowserProfileStore(store_id="browser-session-profile-store")
        binding = _browser_profile_binding(
            store,
            checkpoint_policy=BrowserProfileCheckpointPolicy.AFTER_TERMINAL_OPERATION,
        )
        await binding.initialize()
        records: dict[str, dict[str, Any]] = {}
        runner = _BlockingProfileWireRunner("profile_checkpoint")
        args = {
            "operation": "navigate",
            "url": "https://example.test/login",
            "operation_id": "profile-owner-cancelled-checkpoint",
        }
        ctx = _durable_context(
            tmp_path,
            args=args,
            records=records,
            tool_call_id="profile-owner-cancelled-checkpoint-call",
            runner=runner,
        )
        tool = BrowserSessionTool(
            expected_runner_candidate="wire-browser",
            browser_profile=binding,
            max_sessions=1,
            max_wait_ms=1_000,
            idle_timeout_seconds=60,
        )

        owner = asyncio.create_task(tool.run(ctx, args))
        await runner.started.wait()
        owner.cancel("cancel browser profile checkpoint")
        assert owner.cancelling() == 1
        with pytest.raises(asyncio.CancelledError, match="cancel browser profile checkpoint"):
            await owner

        assert owner.cancelled() is True
        assert owner.cancelling() == 1
        inspection = await store.inspect_profile(binding.access)
        assert inspection.active_writer is True
        assert inspection.safe_error_code == "checkpoint_outcome_unknown"
        replay = await tool.run(ctx, args)
        assert replay.is_error is False
        assert runner.operations == ["profile_restore", "navigate", "profile_checkpoint"]

    asyncio.run(scenario())


class _CrashBeforeRestoreReceiptStore(SQLiteBrowserProfileStore):
    def __init__(self, path: Path) -> None:
        super().__init__(path, store_id="browser-session-profile-store")
        self.fail_before_restore_receipt = True

    async def complete_restore(self, request, *, outcome, error_code=None):
        if self.fail_before_restore_receipt and outcome is BrowserProfileTerminalOutcome.SUCCEEDED:
            self.fail_before_restore_receipt = False
            raise KeyboardInterrupt
        return await super().complete_restore(
            request,
            outcome=outcome,
            error_code=error_code,
        )


def test_browser_profile_restore_replays_after_process_loss_before_receipt(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        profile_path = tmp_path / "restore-replay.sqlite"
        store = _CrashBeforeRestoreReceiptStore(profile_path)
        binding = _browser_profile_binding(store)
        await binding.initialize()
        records: dict[str, dict[str, Any]] = {}
        runner = _ProfileWireRunner()
        first_args = {
            "operation": "navigate",
            "url": "https://example.test/login",
            "operation_id": "profile-restore-before-receipt",
        }
        with pytest.raises(KeyboardInterrupt):
            await BrowserSessionTool(
                expected_runner_candidate="wire-browser",
                browser_profile=binding,
                max_sessions=1,
                max_wait_ms=1_000,
                idle_timeout_seconds=60,
            ).run(
                _durable_context(
                    tmp_path,
                    args=first_args,
                    records=records,
                    tool_call_id="profile-restore-before-receipt-call",
                    runner=runner,
                ),
                first_args,
            )
        assert runner.operations == ["profile_restore"]
        browser_session_id = next(
            value["browser_session_id"]
            for value in records.values()
            if value.get("record_type") == "cayu.browser-session"
        )
        authority = binding.authority
        key_authority = binding.key_authority
        await store.close()

        reopened = SQLiteBrowserProfileStore(
            profile_path,
            store_id="browser-session-profile-store",
        )
        recovered_binding = BrowserProfileBinding(
            authority=authority,
            store=reopened,
            key_authority=key_authority,
            checkpoint_policy=BrowserProfileCheckpointPolicy.ON_CLOSE,
            lease_seconds=120,
        )
        close_args = {
            "operation": "close",
            "session_id": browser_session_id,
            "operation_id": "profile-close-after-restore-replay",
        }
        closed = await BrowserSessionTool(
            expected_runner_candidate="wire-browser",
            browser_profile=recovered_binding,
            max_sessions=1,
            max_wait_ms=1_000,
            idle_timeout_seconds=60,
        ).run(
            _durable_context(
                tmp_path,
                args=close_args,
                records=records,
                tool_call_id="profile-close-after-restore-replay-call",
                runner=runner,
            ),
            close_args,
        )

        assert closed.is_error is False
        assert runner.operations == ["profile_restore", "profile_restore", "close"]
        inspection = await reopened.inspect_profile(recovered_binding.access)
        assert inspection.active_writer is False
        await reopened.close()

    asyncio.run(scenario())


def test_browser_profile_rejects_binary_capture_before_runner_dispatch(tmp_path: Path) -> None:
    from cayu.tools.browser_visual import BrowserVisualPolicy

    async def scenario() -> None:
        store = InMemoryBrowserProfileStore(store_id="browser-session-profile-store")
        binding = _browser_profile_binding(store)
        await binding.initialize()
        runner = _ProfileWireRunner()
        tool = BrowserSessionTool(
            expected_runner_candidate="wire-browser",
            browser_profile=binding,
            visual_policy=BrowserVisualPolicy(
                artifact_store_id="browser-artifacts",
                allowed_origins=("https://example.test",),
                retention="application_managed",
                publish_to_model=True,
            ),
            max_sessions=1,
            max_wait_ms=1_000,
            idle_timeout_seconds=60,
        )

        for operation, fields in (
            ("screenshot", {"full_page": False}),
            ("download", {"ref": "ref_download"}),
            ("observe_visual", {}),
        ):
            args = {
                "operation": operation,
                "session_id": "bs_profile_private",
                "page_id": "bp_profile_private",
                "expected_revision": "br_profile_private",
                "expected_control_epoch": 1,
                "operation_id": f"profile-{operation}-denied",
                **fields,
            }
            if operation == "observe_visual":
                args.pop("expected_revision")
                args.pop("expected_control_epoch")
            result = await tool.run(
                _durable_context(
                    tmp_path,
                    args=args,
                    records={},
                    tool_call_id=f"profile-{operation}-denied-call",
                    runner=runner,
                ),
                args,
            )

            assert result.structured["error"] == "policy_denied"
            assert result.structured["execution"]["dispatch"] == "not_started"
        assert runner.operations == []

    asyncio.run(scenario())


@pytest.mark.parametrize("profile_output_protected", [False, True])
@pytest.mark.parametrize(
    "private_output",
    ["profile-only-private-value", "private-cookie-name-canary"],
    ids=["value", "name"],
)
def test_browser_profile_requires_guarded_nonsecret_observation(
    tmp_path: Path,
    profile_output_protected: bool,
    private_output: str,
) -> None:
    async def scenario() -> None:
        private_value = "profile-only-private-value"
        private_name = "private-cookie-name-canary"
        store = InMemoryBrowserProfileStore(store_id="browser-session-profile-store")
        binding = _browser_profile_binding(store)
        await binding.initialize()
        await _seed_browser_profile_state(
            binding,
            BrowserProfileStateV1(
                cookies=(
                    BrowserProfileCookie(
                        name=private_name,
                        value=private_value,
                        domain="example.test",
                        http_only=True,
                    ),
                ),
                origins=(
                    BrowserProfileOriginStorage(
                        origin="https://example.test",
                        local_storage=(
                            BrowserProfileStorageEntry(
                                name="private-storage-name-canary",
                                value=private_value,
                            ),
                        ),
                    ),
                ),
            ),
        )
        runner = _ProfileWireRunner()
        runner.observation_snapshot = f"private page: {private_output}"
        runner.profile_output_protected = profile_output_protected
        records: dict[str, dict[str, Any]] = {}
        args = {
            "operation": "navigate",
            "url": "https://example.test/account",
            "operation_id": (
                f"profile-output-guard-{profile_output_protected}-"
                f"{'name' if private_output == private_name else 'value'}"
            ),
        }

        result = await BrowserSessionTool(
            expected_runner_candidate="wire-browser",
            browser_profile=binding,
            max_sessions=1,
            max_wait_ms=1_000,
            idle_timeout_seconds=60,
        ).run(
            _durable_context(
                tmp_path,
                args=args,
                records=records,
                tool_call_id=f"profile-output-guard-{profile_output_protected}-call",
                runner=runner,
            ),
            args,
        )

        assert result.structured["error"] == "policy_denied"
        assert result.structured["execution"]["dispatch"] == "completed"
        assert private_value not in json.dumps(result.model_dump(mode="json"), sort_keys=True)
        assert private_value not in json.dumps(records, sort_keys=True)
        assert private_name not in json.dumps(result.model_dump(mode="json"), sort_keys=True)
        assert private_name not in json.dumps(records, sort_keys=True)
        assert runner.operations == ["profile_restore", "navigate"]

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["navigate", "observe", "switch_page", "list_pages", "error"])
@pytest.mark.parametrize("reflection", ["clean", "title", "url"])
@pytest.mark.parametrize("with_refs", [False, True])
def test_profile_guest_response_protects_page_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    operation: str,
    reflection: str,
    with_refs: bool,
) -> None:
    canary = "new-profile-token-canary"

    class Page:
        url = "https://example.test/"

        def is_closed(self):
            return False

        async def goto(self, url, **kwargs):
            self.url = url

        async def close(self):
            pass

    class Context:
        async def new_page(self):
            return Page()

        async def close(self):
            pass

        async def route_web_socket(self, pattern, callback):
            del pattern, callback

        async def storage_state(self, *, indexed_db):
            assert not indexed_db
            return {
                "cookies": [],
                "origins": [
                    {
                        "origin": "https://example.test",
                        "localStorage": [{"name": "token", "value": canary}],
                    }
                ],
            }

    async def observation(
        state, limits, *, browser_version, visual_policy=None, visual_artifacts=None
    ):
        del limits, browser_version
        assert visual_policy is None and visual_artifacts is None
        if operation == "error":
            raise _browser_guest._GuestFailure("fetch_failed")
        state.revision = "br_observed"
        state.last_observation_revision = state.revision
        state.refs = {"opaque_ref": "private-locator"} if with_refs else {}
        return {
            "session_id": state.session_id,
            "page_id": state.page_id,
            "revision": state.revision,
            "creation_epoch": state.creation_epoch,
            "control_epoch": state.control_epoch,
            "url": "https://example.test/" + (canary if reflection == "url" else ""),
            "title": canary if reflection == "title" else "Safe page",
            "snapshot": '- button "Continue" [ref=opaque_ref]',
            "refs": [{"ref": "opaque_ref", "role": "button", "name": "Continue"}]
            if with_refs
            else [],
            "load_state": "loaded",
            "access_state": "available",
            "access": None,
            "idle_timeout_seconds": 60,
            "truncation_reasons": [],
            "backend_identity": {
                "backend": "playwright",
                "backend_version": "1.62.0",
                "browser": "chromium",
                "browser_version": "test-chromium",
                "worker_protocol": "cayu.browser-session.v4",
                "worker_version": "19",
            },
        }

    monkeypatch.setattr(_browser_guest, "_interactive_observation", observation)

    async def configure_page(self, state, limits):
        # Replace only Playwright/CDP setup; observation protection, accounting,
        # response publication, and host parsing remain the production path.
        del self, limits
        state.configured = True

    monkeypatch.setattr(_browser_guest._InteractiveDaemon, "_configure_page", configure_page)

    async def scenario():
        limits = _interactive_limits(max_pages=2, max_total_page_creations=2)
        request = _interactive_request(
            "observe" if operation == "error" else operation, limits=limits
        )
        daemon = _browser_guest._InteractiveDaemon("bs_test")
        daemon.context = Context()
        daemon.profile_output_values = ()
        daemon.profile_allowed_origins = ("https://example.test",)
        daemon.profile_plaintext_limit = 4096
        daemon.profile_timeout_seconds = 1.0
        state = _browser_guest._InteractivePage(
            page=Page(),
            session_id="bs_test",
            page_id="bp_test",
            configured=True,
            lifecycle="background" if operation == "switch_page" else "active",
            public_url="https://example.test/" + canary,
            title=canary,
        )
        daemon.pages[state.page_id] = state
        daemon.active_page_id = state.page_id
        daemon.total_page_creations = 1
        if operation == "navigate":
            daemon.pages.clear()
            daemon.active_page_id = None
            daemon.total_page_creations = 0
        if operation == "switch_page":
            other = _browser_guest._InteractivePage(
                page=Page(),
                session_id="bs_test",
                page_id="bp_other",
                creation_epoch=2,
                configured=True,
                lifecycle="active",
                title=canary,
            )
            daemon.pages[other.page_id] = other
            daemon.active_page_id = other.page_id
            daemon.total_page_creations = 2
        await _configure_interactive_daemon_for_test(daemon, request)
        result = await daemon.execute(request)
        if operation == "navigate" and reflection != "url":
            state = daemon.pages["bp_test"]
        rendered = json.dumps(result)
        assert canary not in rendered
        if operation == "navigate" and reflection == "url":
            assert result["allocation_disposition"] == "retired"
            assert "page_set" not in result
        else:
            assert all(
                page["url"] is None and page["title"] is None
                for page in result["page_set"]["pages"]
            )
        response = browser_session_module._parse_runner_response(
            rendered,
            max_artifact_bytes=1024,
            max_page_records=2,
            max_page_creations_per_operation=1,
        )
        if operation == "error" or (reflection == "url" and operation != "list_pages"):
            assert response.failure is not None
            assert response.failure.code == (
                "fetch_failed" if operation == "error" else "policy_denied"
            )
        else:
            assert response.failure is None
            if operation != "list_pages":
                assert response.profile_output_protected
                assert response.observation is not None
                expected_refs = 0 if reflection == "title" or not with_refs else 1
                assert (
                    len(response.observation.refs)
                    == state.ref_count
                    == daemon.total_refs
                    == expected_refs
                )
                assert len(state.refs) == expected_refs
        assert await daemon.execute(request) == result

        if operation == "navigate":

            class GuestRunner(_ProfileWireRunner):
                async def exec(self, command, **kwargs):
                    raw = json.loads(kwargs["stdin"])
                    if raw["operation"] != "navigate":
                        return await super().exec(command, **kwargs)
                    self.operations.append("navigate")
                    guest_request = _browser_guest._interactive_request_from_json(raw)
                    guest = _browser_guest._InteractiveDaemon(guest_request.session_id)
                    guest.context = Context()
                    guest.profile_output_values = ()
                    guest.profile_allowed_origins = ("https://example.test",)
                    guest.profile_plaintext_limit = 4096
                    guest.profile_timeout_seconds = 1.0
                    await _configure_interactive_daemon_for_test(guest, guest_request)
                    return ExecResult(stdout=json.dumps(await guest.execute(guest_request)))

            binding = _browser_profile_binding(
                InMemoryBrowserProfileStore(store_id="browser-session-profile-store")
            )
            await binding.initialize()
            tool = BrowserSessionTool(
                expected_runner_candidate="wire-browser",
                browser_profile=binding,
                max_sessions=1,
                max_wait_ms=1000,
                idle_timeout_seconds=60,
            )
            runner = GuestRunner()
            args = {
                "operation": "navigate",
                "url": "https://example.test/",
                "operation_id": "guest-profile-output",
            }
            records = {}
            ctx = _durable_context(tmp_path, args=args, records=records, runner=runner)
            published = await tool.run(ctx, args)
            assert published.is_error == (reflection == "url")
            if published.is_error:
                assert published.structured["error"] == "policy_denied"
            assert canary not in json.dumps(published.model_dump(mode="json"))
            assert canary not in json.dumps(records)
            assert await tool.run(ctx, args) == published
            assert runner.operations == ["profile_restore", "navigate"]

    asyncio.run(scenario())


@pytest.mark.parametrize("shape", ["observation", "listing", "error"])
@pytest.mark.parametrize("field", ["title", "url"])
def test_profile_summary_rejection_precedes_durable_publication(
    tmp_path: Path,
    shape: str,
    field: str,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    recwarn: pytest.WarningsRecorder,
) -> None:
    canary = "profile-summary-private-canary"

    class Runner(_ProfileWireRunner):
        async def exec(self, command, **kwargs):
            request = json.loads(kwargs["stdin"])
            operation = request["operation"]
            if operation == "list_pages":
                request["page_id"] = self.page_id
                kwargs = {**kwargs, "stdin": json.dumps(request)}
            result = await super().exec(command, **kwargs)
            payload = json.loads(result.stdout)
            if "page_set" not in payload or (shape == "listing" and operation == "navigate"):
                return result
            payload["page_set"]["pages"][0][field] = (
                "https://example.test/" + canary if field == "url" else canary
            )
            if shape != "observation":
                payload.pop("observation", None)
                payload["profile_output_protected"] = False
                if shape == "error":
                    payload["kind"] = "error"
                    payload["error"] = "fetch_failed"
            return result.model_copy(update={"stdout": json.dumps(payload)})

    async def scenario():
        binding = _browser_profile_binding(
            InMemoryBrowserProfileStore(store_id="browser-session-profile-store")
        )
        await binding.initialize()
        tool = BrowserSessionTool(
            expected_runner_candidate="wire-browser",
            browser_profile=binding,
            max_sessions=1,
            max_wait_ms=1000,
            idle_timeout_seconds=60,
            multi_page=True,
        )
        runner = Runner()
        args = {
            "operation": "navigate",
            "url": "https://example.test/",
            "operation_id": "summary-guard",
        }
        records = {}
        ctx = _durable_context(tmp_path, args=args, records=records, runner=runner)
        result = await tool.run(ctx, args)
        if shape == "listing":
            assert not result.is_error
            args = {
                "operation": "list_pages",
                "session_id": result.structured["session_id"],
                "operation_id": "summary-list",
            }
            ctx = _durable_context(
                tmp_path,
                args=args,
                records=records,
                runner=runner,
                tool_call_id="summary-list-call",
            )
            result = await tool.run(ctx, args)
        assert result.is_error
        assert result.structured["error"] == "policy_denied"
        assert canary not in json.dumps(result.model_dump(mode="json"))
        assert canary not in json.dumps(records)
        replay = await tool.run(ctx, args)
        assert replay == result
        assert runner.operations == ["profile_restore", "navigate"] + (
            ["list_pages"] if shape == "listing" else []
        )

    asyncio.run(scenario())
    captured = capsys.readouterr()
    assert canary not in captured.out + captured.err + caplog.text
    assert all(canary not in str(warning.message) for warning in recwarn)


def test_browser_profile_reconciles_post_terminal_checkpoint_after_worker_loss(
    tmp_path: Path,
) -> None:
    class _StopBeforeCheckpointTool(BrowserSessionTool):
        async def _checkpoint_browser_profile(self, ctx, session) -> None:
            del ctx, session
            raise KeyboardInterrupt

    async def scenario() -> None:
        store = InMemoryBrowserProfileStore(store_id="browser-session-profile-store")
        binding = _browser_profile_binding(
            store,
            checkpoint_policy=BrowserProfileCheckpointPolicy.AFTER_TERMINAL_OPERATION,
        )
        await binding.initialize()
        records: dict[str, dict[str, Any]] = {}
        runner = _ProfileWireRunner()
        runner.checkpoint_state = {
            "cookies": [
                {
                    "name": "session",
                    "value": "private-checkpoint-value",
                    "domain": "example.test",
                    "path": "/",
                    "expires": -1.0,
                    "httpOnly": True,
                    "secure": True,
                    "sameSite": "Lax",
                }
            ],
            "origins": [],
        }
        first_args = {
            "operation": "navigate",
            "url": "https://example.test/login",
            "operation_id": "profile-terminal-before-checkpoint",
        }
        first_tool = _StopBeforeCheckpointTool(
            expected_runner_candidate="wire-browser",
            browser_profile=binding,
            max_sessions=1,
            max_wait_ms=1_000,
            idle_timeout_seconds=60,
        )
        with pytest.raises(KeyboardInterrupt):
            await first_tool.run(
                _durable_context(
                    tmp_path,
                    args=first_args,
                    records=records,
                    tool_call_id="profile-terminal-before-checkpoint-call",
                    runner=runner,
                ),
                first_args,
            )
        assert (await store.inspect_profile(binding.access)).generation == 0

        browser_session_id = next(
            value["browser_session_id"]
            for value in records.values()
            if value.get("record_type") == "cayu.browser-session"
        )
        page_id = next(
            value["page_set"]["active_page_id"]
            for value in records.values()
            if value.get("record_type") == "cayu.browser-session"
        )
        observe_args = {
            "operation": "observe",
            "session_id": browser_session_id,
            "page_id": page_id,
            "operation_id": "profile-after-worker-loss-observe",
        }
        recovered = await BrowserSessionTool(
            expected_runner_candidate="wire-browser",
            browser_profile=binding,
            max_sessions=1,
            max_wait_ms=1_000,
            idle_timeout_seconds=60,
        ).run(
            _durable_context(
                tmp_path,
                args=observe_args,
                records=records,
                tool_call_id="profile-after-worker-loss-observe-call",
                runner=runner,
            ),
            observe_args,
        )

        assert recovered.is_error is False
        assert runner.operations[:4] == [
            "profile_restore",
            "navigate",
            "profile_checkpoint",
            "observe",
        ]
        assert (await store.inspect_profile(binding.access)).generation == 2

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("profile_output_protected", "expected_error", "expected_generation"),
    [(True, None, 1), (False, "policy_denied", 0)],
)
def test_browser_profile_reconciles_dispatched_navigation_with_profile_authority(
    tmp_path: Path,
    profile_output_protected: bool,
    expected_error: str | None,
    expected_generation: int,
) -> None:
    async def scenario() -> None:
        store = InMemoryBrowserProfileStore(store_id="browser-session-profile-store")
        binding = _browser_profile_binding(
            store,
            checkpoint_policy=BrowserProfileCheckpointPolicy.AFTER_TERMINAL_OPERATION,
        )
        await binding.initialize()
        records: dict[str, dict[str, Any]] = {}
        runner = _ProfileWireRunner()
        runner.profile_output_protected = profile_output_protected
        args = {
            "operation": "navigate",
            "url": "https://example.test/login",
            "operation_id": "profile-dispatched-before-terminal",
        }
        tool_call_id = "profile-dispatched-before-terminal-call"
        interrupted = await BrowserSessionTool(
            expected_runner_candidate="wire-browser",
            browser_profile=binding,
            max_sessions=1,
            max_wait_ms=1_000,
            idle_timeout_seconds=60,
        ).run(
            _durable_context(
                tmp_path,
                args=args,
                records=records,
                tool_call_id=tool_call_id,
                runner=runner,
                fail_before_state="terminal",
            ),
            args,
        )
        assert interrupted.structured["error"] == "outcome_ambiguous"
        assert (await store.inspect_profile(binding.access)).generation == 0

        recovered = await BrowserSessionTool(
            expected_runner_candidate="wire-browser",
            browser_profile=binding,
            max_sessions=1,
            max_wait_ms=1_000,
            idle_timeout_seconds=60,
        ).run(
            _durable_context(
                tmp_path,
                args=args,
                records=records,
                tool_call_id=tool_call_id,
                runner=runner,
            ),
            args,
        )

        if expected_error is None:
            assert recovered.is_error is False
        else:
            assert recovered.structured["error"] == expected_error
        assert runner.operations == [
            "profile_restore",
            "navigate",
            "navigate",
            *(("profile_checkpoint",) if expected_error is None else ()),
        ]
        inspection = await store.inspect_profile(binding.access)
        assert inspection.generation == expected_generation
        assert any(
            record.get("state") == "terminal"
            for record in records.values()
            if record.get("record_type") == "cayu.browser-operation"
        )

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "failure_mode",
    [
        "stage_ack",
        "stage_ack_adoption_failure",
        "cancel",
        "cancel_adoption_failure",
        "cancel_repeated_adoption_failure",
    ],
)
@pytest.mark.parametrize("initial_generation", [0, 1])
@pytest.mark.parametrize("persistent", [False, True], ids=["memory", "sqlite"])
def test_profile_checkpoint_recovery_adopts_complete_material(
    tmp_path: Path,
    failure_mode: str,
    initial_generation: int,
    persistent: bool,
) -> None:
    cancel_publication = failure_mode.startswith("cancel")
    base = SQLiteBrowserProfileStore if persistent else InMemoryBrowserProfileStore

    class Store(base):
        def __init__(self):
            super().__init__(
                **({"path": tmp_path / "profile.db"} if persistent else {}),
                store_id="browser-session-profile-store",
            )
            self.entered = asyncio.Event()
            self.release = asyncio.Event()
            self.once = True
            self.adoption_failures = (
                2 if "repeated" in failure_mode else 1 if "adoption_failure" in failure_mode else 0
            )

        async def stage_checkpoint(self, *args, **kwargs):
            result = await super().stage_checkpoint(*args, **kwargs)
            if self.once and not cancel_publication:
                self.once = False
                raise ConnectionError("staging acknowledgement lost")
            return result

        async def publish_checkpoint(self, *args, **kwargs):
            if self.once and cancel_publication:
                self.once = False
                self.entered.set()
                await self.release.wait()
            return await super().publish_checkpoint(*args, **kwargs)

        async def resume_writer(self, *args, **kwargs):
            if self.adoption_failures:
                self.adoption_failures -= 1
                raise ConnectionError("committed writer readback unavailable")
            return await super().resume_writer(*args, **kwargs)

    store = Store()

    async def scenario():
        binding = _browser_profile_binding(
            store,
            checkpoint_policy=BrowserProfileCheckpointPolicy.AFTER_TERMINAL_OPERATION,
        )
        await binding.initialize()
        if initial_generation:
            store.once = False
            await _seed_browser_profile_state(binding, BrowserProfileStateV1())
            store.once = True
        runner = _ProfileWireRunner()
        records = {}
        tool = BrowserSessionTool(
            expected_runner_candidate="wire-browser",
            browser_profile=binding,
            max_sessions=1,
            max_wait_ms=1000,
            idle_timeout_seconds=60,
        )
        args = {
            "operation": "navigate",
            "url": "https://example.test/login",
            "operation_id": "adopt-first",
        }
        ctx = _durable_context(tmp_path, args=args, records=records, runner=runner)
        owner = asyncio.create_task(tool.run(ctx, args))
        if cancel_publication:
            await store.entered.wait()
            owner.cancel("checkpoint publication cancelled")
            store.release.set()
            with pytest.raises(asyncio.CancelledError) as cancelled:
                await owner
            assert owner.cancelled()
            assert owner.cancelling() == 1
            pending = [cancelled.value]
            seen = {}
            while pending:
                signal = pending.pop()
                if id(signal) in seen:
                    continue
                seen[id(signal)] = signal
                if isinstance(signal, BaseExceptionGroup):
                    pending.extend(signal.exceptions)
                if signal.__cause__ is not None:
                    pending.append(signal.__cause__)
                elif not signal.__suppress_context__ and signal.__context__ is not None:
                    pending.append(signal.__context__)
            assert sum(isinstance(signal, asyncio.CancelledError) for signal in seen.values()) == 1
            first = await tool.run(ctx, args)
        else:
            first = await owner
        assert not first.is_error
        assert (await store.inspect_profile(binding.access)).generation == initial_generation + 1
        live = tool._states[ctx.session_id].sessions[first.structured["session_id"]]
        if "adoption_failure" in failure_mode:
            assert live.pending_profile_checkpoint_id is not None
            assert live.profile_material.preparation.profile_ref.generation == initial_generation
        followup = {
            "operation": "observe",
            "session_id": first.structured["session_id"],
            "page_id": first.structured["page_id"],
            "operation_id": "adopt-second",
        }
        followup_ctx = _durable_context(
            tmp_path,
            args=followup,
            records=records,
            runner=runner,
            tool_call_id="adopt-second-call",
        )
        if store.adoption_failures:
            operations = tuple(runner.operations)
            refused = await tool.run(followup_ctx, followup)
            assert refused.structured["error"] == "authority_expired"
            assert tuple(runner.operations) == operations
            assert live.pending_profile_checkpoint_id is not None
            assert live.profile_material.preparation.profile_ref.generation == initial_generation
        second = await tool.run(followup_ctx, followup)
        assert not second.is_error
        assert (await store.inspect_profile(binding.access)).generation == initial_generation + 2
        assert runner.operations.count("profile_checkpoint") == 2
        assert live.pending_profile_checkpoint_id is None

    try:
        asyncio.run(scenario())
    finally:
        if persistent:
            asyncio.run(store.close())
