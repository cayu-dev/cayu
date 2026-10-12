"""Browser-session dispatch and process-loss recovery contracts."""

from __future__ import annotations

import asyncio
import hashlib
import json
import multiprocessing
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from tests.core.browser_session_support import (
    _IDENTITY,
    _context,
    _durable_context,
    _FakeBrowserBackend,
    _Issue1275BrowserBackend,
    _issue_1275_operation_args,
    _put_issue_1275_upload_artifact,
    _recover_durable_browser_result,
    _tool,
)

from cayu.artifacts.local import LocalArtifactStore
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.sessions._operation_records import SessionOperationPublication
from cayu.sessions.records import SessionIdentity, SessionStatus
from cayu.sessions.requests import RunRequest
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.base import ToolContext, ToolResult, _bind_runtime_tool_invocation_authority
from cayu.tools.browser_session import (
    BrowserArtifactPayload,
    BrowserBackendObservation,
    BrowserBackendResponse,
    BrowserElementRef,
    BrowserOperationEvidence,
    BrowserPageSetDelta,
    BrowserPageSetState,
    BrowserPageSummary,
    BrowserSessionBackend,
    BrowserSessionTool,
)


class _ProcessBoundaryArtifactStore(LocalArtifactStore):
    def __init__(self, root: Path, *, phase: str) -> None:
        super().__init__(root, store_id="browser-artifacts")
        self.phase = phase

    async def put_bytes(self, *args: Any, **kwargs: Any):
        artifact = await super().put_bytes(*args, **kwargs)
        if self.phase == "after_artifact_publication":
            os._exit(_PROCESS_LOSS_EXIT_CODE)
        return artifact


_PROCESS_LOSS_EXIT_CODE = 86


@dataclass
class _ProcessBoundaryBrowserBackend(BrowserSessionBackend):
    phase: str
    calls_path: str
    target_operation: str = "navigate"

    async def execute(self, ctx: ToolContext, request: dict[str, Any]) -> BrowserBackendResponse:
        del ctx
        path = Path(self.calls_path)
        calls = [] if not path.exists() else json.loads(path.read_text(encoding="utf-8"))
        calls.append(
            {
                key: (
                    [item.durable_evidence() for item in value]
                    if key == "_upload_payloads"
                    else value
                )
                for key, value in request.items()
            }
        )
        path.write_text(json.dumps(calls), encoding="utf-8")
        if self.phase == "after_dispatch" and request["operation"] == self.target_operation:
            os._exit(_PROCESS_LOSS_EXIT_CODE)
        artifacts = (
            (
                BrowserArtifactPayload(
                    kind="screenshot",
                    filename="process-loss.png",
                    content_type="image/png",
                    content=b"process-loss-artifact",
                ),
            )
            if self.phase == "after_artifact_publication"
            else ()
        )
        observation = BrowserBackendObservation(
            session_id=request["session_id"],
            page_id=request["page_id"],
            revision="br_process_revision_1",
            creation_epoch=1,
            control_epoch=1,
            url=request["url"],
            title="Process recovery fixture",
            snapshot='- button "Action" [ref=ref_action]\n- filechooser "Upload" [ref=ref_upload_multiple]',
            refs=(
                BrowserElementRef(ref="ref_action", role="button", name="Action"),
                BrowserElementRef(
                    ref="ref_upload_multiple",
                    role="button",
                    name="Upload",
                    element_type="file_input",
                    allows_multiple_files=True,
                ),
            ),
            load_state="loaded",
            access_state="available",
            idle_timeout_seconds=900,
            truncation_reasons=(),
            backend_identity=_IDENTITY,
        )
        operation = request["operation"]
        operation_evidence = None
        if operation == "scroll":
            operation_evidence = BrowserOperationEvidence(
                operation="scroll",
                moved=True,
                edge="none",
            )
        elif operation == "upload":
            operation_evidence = BrowserOperationEvidence(
                operation="upload",
                selected_file_count=len(request["artifact_ids"]),
                selection_state="selected",
            )
        return BrowserBackendResponse(
            observation=observation,
            page_set=BrowserPageSetState(
                session_id=request["session_id"],
                active_page_id=request["page_id"],
                pages=(
                    BrowserPageSummary(
                        page_id=request["page_id"],
                        lifecycle="active",
                        creation_epoch=1,
                        control_epoch=1,
                        revision=observation.revision,
                        url=observation.url,
                        title=observation.title,
                        load_state="loaded",
                        access_state="available",
                        last_observation_revision=observation.revision,
                        last_operation_id_sha256=hashlib.sha256(
                            request["operation_id"].encode("utf-8")
                        ).hexdigest(),
                        operation_count=1,
                        observation_count=1,
                        ref_count=len(observation.refs),
                        request_count=0,
                        artifact_count=len(artifacts),
                    ),
                ),
                total_page_creations=1,
                total_operations=1,
                total_observations=1,
                total_refs=len(observation.refs),
                total_requests=0,
                total_artifacts=len(artifacts),
                cleanup_operation_count=0,
            ),
            page_delta=BrowserPageSetDelta(
                created_page_ids=(request["page_id"],),
                admitted_page_ids=(request["page_id"],),
            ),
            artifacts=artifacts,
            operation_evidence=operation_evidence,
        )


def _run_crashing_cayu_browser_worker(
    session_path: str,
    artifact_path: str,
    phase: str,
    calls_path: str,
    operation: str = "navigate",
    arguments_path: str | None = None,
) -> None:
    async def run() -> None:
        store = SQLiteSessionStore(session_path)
        artifact_store = _ProcessBoundaryArtifactStore(
            Path(artifact_path) / "artifacts",
            phase=phase,
        )

        async def load(storage_key: str) -> dict[str, Any] | None:
            return await store.load_session_operation("parent-session", storage_key)

        async def compare_and_set(
            storage_key: str,
            expected: dict[str, Any] | None,
            desired: dict[str, Any],
            secondary: dict[str, dict[str, Any]],
        ) -> dict[str, Any]:
            desired_copy = json.loads(json.dumps(desired))
            secondary_copy = json.loads(json.dumps(secondary))
            states = {
                value.get("state")
                for value in (desired_copy, *secondary_copy.values())
                if value.get("record_type") == "cayu.browser-operation"
            }
            if phase == "before_dispatch" and "dispatched" in states:
                os._exit(_PROCESS_LOSS_EXIT_CODE)
            if phase == "after_browser_completion" and "terminal" in states:
                os._exit(_PROCESS_LOSS_EXIT_CODE)

            def publish(current_session, checkpoint, current):
                if current_session.run_epoch != 1 or current != expected:
                    raise RuntimeError("Process recovery fixture lost durable authority.")
                return SessionOperationPublication(
                    checkpoint={} if checkpoint is None else checkpoint,
                    operation_records={storage_key: desired_copy, **secondary_copy},
                )

            await store.publish_session_operation(
                "parent-session",
                idempotency_key=storage_key,
                operation_transform=publish,
                events=[],
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=1,
            )
            if phase == "before_receipt_publication" and "terminal" in states:
                os._exit(_PROCESS_LOSS_EXIT_CODE)
            return desired_copy

        try:
            backend = _ProcessBoundaryBrowserBackend(
                phase=phase,
                calls_path=calls_path,
                target_operation=operation,
            )
            tool = BrowserSessionTool._from_backend_for_testing(backend)

            async def invoke(args: dict[str, Any], *, tool_call_id: str) -> ToolResult:
                ctx = _context(
                    Path(artifact_path),
                    artifact_store=artifact_store,
                ).model_copy(update={"idempotency_key": f"tool-key-{tool_call_id}"})
                _bind_runtime_tool_invocation_authority(
                    ctx,
                    parent_task_id=None,
                    parent_run_epoch=1,
                    model_step_id="model-step-1",
                    model_attempt_id="model-attempt-1",
                    tool_round_id="tool-round-1",
                    tool_call_id=tool_call_id,
                    tool_name="browser_session",
                    idempotency_key=f"tool-key-{tool_call_id}",
                    effective_arguments=args,
                    execution_profile_fingerprint="b" * 64,
                    environment_allocation_fingerprint="a" * 64,
                    load_durable_operation=load,
                    compare_and_set_durable_operation=compare_and_set,
                    seal_durable_output=lambda value: json.loads(json.dumps(value)),
                    secret_publication_sealer=lambda: None,
                )
                return await tool.run(ctx, args)

            if operation == "navigate":
                args = {
                    "operation": "navigate",
                    "url": "https://example.test/form",
                    "operation_id": f"process-{phase}",
                }
                await invoke(args, tool_call_id="tool-call-1")
            else:
                navigate_args = {
                    "operation": "navigate",
                    "url": "https://example.test/upload",
                    "operation_id": f"process-prerequisite-{operation}",
                }
                opened = await invoke(navigate_args, tool_call_id="navigate-call")
                artifact_ids = None
                if operation == "upload":
                    upload_ctx = _context(Path(artifact_path), artifact_store=artifact_store)
                    artifact_ids = [await _put_issue_1275_upload_artifact(upload_ctx)]
                args = _issue_1275_operation_args(
                    operation,
                    dict(opened.structured or {}),
                    artifact_ids=artifact_ids,
                    operation_id=f"process-{phase}-{operation}",
                )
                if arguments_path is None:
                    raise RuntimeError("Process operation recovery requires an arguments path.")
                Path(arguments_path).write_text(json.dumps(args), encoding="utf-8")
                await invoke(args, tool_call_id="tool-call-1")
        finally:
            await store.close()

    asyncio.run(run())


def _run_fresh_cayu_browser_recovery(
    session_path: str,
    phase: str,
    result_path: str,
    arguments_path: str | None = None,
) -> None:
    async def run() -> None:
        store = SQLiteSessionStore(session_path)

        async def load(storage_key: str) -> dict[str, Any] | None:
            return await store.load_session_operation("parent-session", storage_key)

        try:
            arguments = (
                {
                    "operation": "navigate",
                    "url": "https://example.test/form",
                    "operation_id": f"process-{phase}",
                }
                if arguments_path is None
                else json.loads(Path(arguments_path).read_text(encoding="utf-8"))
            )
            result = await BrowserSessionTool._from_backend_for_testing(
                _FakeBrowserBackend()
            ).reconcile_durable_tool_call(
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
                arguments=arguments,
                started=True,
                load_operation=load,
            )
            assert result is not None
            result = result.result
            Path(result_path).write_text(
                json.dumps(result.model_dump(mode="json")),
                encoding="utf-8",
            )
        finally:
            await store.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "operation",
    ["back", "forward", "reload", "scroll", "hover", "upload"],
)
@pytest.mark.parametrize(
    ("durable_boundary", "fail_before_state", "fail_after_state", "expected_error"),
    [
        ("intent", "dispatched", None, "operation_not_dispatched"),
        ("dispatched", "terminal", None, "outcome_ambiguous"),
        ("terminal", None, "terminal", None),
    ],
)
def test_issue_1275_operations_recover_without_redispatch_at_each_durable_boundary(
    tmp_path: Path,
    operation: str,
    durable_boundary: str,
    fail_before_state: str | None,
    fail_after_state: str | None,
    expected_error: str | None,
) -> None:
    async def scenario() -> None:
        case_root = tmp_path / f"{operation}-{durable_boundary}"
        store = LocalArtifactStore(
            case_root / "artifacts",
            store_id="browser-artifacts",
        )
        artifact_ctx = _context(case_root, artifact_store=store)
        artifact_ids = (
            [await _put_issue_1275_upload_artifact(artifact_ctx)] if operation == "upload" else None
        )
        backend = _Issue1275BrowserBackend()
        tool = BrowserSessionTool(_backend=backend)
        records: dict[str, dict[str, Any]] = {}
        navigate_args = {
            "operation": "navigate",
            "url": "https://example.test/upload",
            "operation_id": f"navigate-{operation}-{durable_boundary}",
        }
        opened = await tool.run(
            _durable_context(
                case_root,
                args=navigate_args,
                records=records,
                artifact_store=store,
                tool_call_id=f"navigate-{operation}-{durable_boundary}-call",
            ),
            navigate_args,
        )
        operation_args = _issue_1275_operation_args(
            operation,
            dict(opened.structured or {}),
            artifact_ids=artifact_ids,
            operation_id=f"recover-{operation}-{durable_boundary}",
        )
        operation_call_id = f"recover-{operation}-{durable_boundary}-call"

        interrupted = await tool.run(
            _durable_context(
                case_root,
                args=operation_args,
                records=records,
                artifact_store=store,
                tool_call_id=operation_call_id,
                fail_before_state=fail_before_state,
                fail_after_state=fail_after_state,
            ),
            operation_args,
        )
        calls_before_recovery = len(backend.calls)
        recovered = await _recover_durable_browser_result(
            tool,
            args=operation_args,
            records=records,
            tool_call_id=operation_call_id,
        )

        assert len(backend.calls) == calls_before_recovery
        assert calls_before_recovery == 1 + int(durable_boundary != "intent")
        operation_record = next(
            record
            for record in records.values()
            if record.get("record_type") == "cayu.browser-operation"
            and record.get("tool_call_id") == operation_call_id
        )
        assert operation_record["state"] == durable_boundary
        if expected_error is None:
            # Exact terminal readback reconciles the lost acknowledgement immediately.
            assert interrupted.is_error is False
            assert recovered.is_error is False
            assert interrupted == recovered
            if operation in {"scroll", "upload"}:
                assert recovered.structured["operation_evidence"]["operation"] == operation
        else:
            assert interrupted.structured["error"] == expected_error
            assert recovered.structured["error"] == expected_error

    asyncio.run(scenario())


def test_browser_session_worker_loss_before_dispatch_reconciles_intent(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        backend = _FakeBrowserBackend()
        records: dict[str, dict[str, Any]] = {}
        args = {
            "operation": "navigate",
            "url": "https://example.test/form",
            "operation_id": "before-dispatch",
        }
        tool = _tool(backend)

        stopped = await tool.run(
            _durable_context(
                tmp_path,
                args=args,
                records=records,
                fail_before_state="dispatched",
            ),
            args,
        )
        recovered = await _recover_durable_browser_result(
            tool,
            args=args,
            records=records,
        )

        assert stopped.structured["error"] == "operation_not_dispatched"
        assert recovered.structured["error"] == "operation_not_dispatched"
        assert [
            record["state"]
            for record in records.values()
            if record.get("record_type") == "cayu.browser-operation"
        ] == ["intent"]
        assert backend.calls == []

    asyncio.run(scenario())


def test_browser_session_worker_loss_after_dispatch_records_terminal_ambiguity(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        backend = _FakeBrowserBackend(failure=ConnectionError("worker lost after dispatch"))
        records: dict[str, dict[str, Any]] = {}
        args = {
            "operation": "navigate",
            "url": "https://example.test/form",
            "operation_id": "after-dispatch",
        }
        tool = _tool(backend)

        stopped = await tool.run(
            _durable_context(tmp_path, args=args, records=records),
            args,
        )
        recovered = await _recover_durable_browser_result(
            tool,
            args=args,
            records=records,
        )

        assert stopped.structured["error"] == "outcome_ambiguous"
        assert recovered == stopped
        assert len(backend.calls) == 1
        operation_records = [
            record
            for record in records.values()
            if record.get("record_type") == "cayu.browser-operation"
        ]
        assert len(operation_records) == 1
        assert operation_records[0]["state"] == "terminal"

    asyncio.run(scenario())


def test_browser_session_worker_loss_after_completion_before_receipt_is_not_replayed(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        backend = _FakeBrowserBackend()
        records: dict[str, dict[str, Any]] = {}
        args = {
            "operation": "navigate",
            "url": "https://example.test/form",
            "operation_id": "before-terminal-receipt",
        }
        tool = _tool(backend)

        stopped = await tool.run(
            _durable_context(
                tmp_path,
                args=args,
                records=records,
                fail_before_state="terminal",
            ),
            args,
        )
        recovered = await _recover_durable_browser_result(
            tool,
            args=args,
            records=records,
        )

        assert stopped.structured["error"] == "outcome_ambiguous"
        assert recovered.structured["error"] == "outcome_ambiguous"
        assert recovered.structured["browser_session_id"] == backend.calls[0]["session_id"]
        assert len(backend.calls) == 1
        assert any(record.get("state") == "dispatched" for record in records.values())

    asyncio.run(scenario())


def test_browser_session_lost_terminal_acknowledgement_replays_exact_receipt(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        backend = _FakeBrowserBackend()
        records: dict[str, dict[str, Any]] = {}
        args = {
            "operation": "navigate",
            "url": "https://example.test/form",
            "operation_id": "after-terminal-receipt",
        }
        tool = _tool(backend)

        acknowledgement_lost = await tool.run(
            _durable_context(
                tmp_path,
                args=args,
                records=records,
                fail_after_state="terminal",
            ),
            args,
        )
        recovered = await _recover_durable_browser_result(
            tool,
            args=args,
            records=records,
        )

        # Complete publication readback settles the lost acknowledgement
        # immediately, without waiting for a fresh worker or repeating input.
        assert acknowledgement_lost.is_error is False
        assert acknowledgement_lost == recovered
        assert recovered.is_error is False
        assert recovered.structured["session_id"] == backend.calls[0]["session_id"]
        assert len(backend.calls) == 1
        assert any(record.get("state") == "terminal" for record in records.values())

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("phase", "expected_error", "expected_backend_calls"),
    [
        ("before_dispatch", "operation_not_dispatched", 0),
        ("after_dispatch", "outcome_ambiguous", 1),
        ("after_browser_completion", "outcome_ambiguous", 1),
        ("after_artifact_publication", "outcome_ambiguous", 1),
        ("before_receipt_publication", None, 1),
    ],
)
def test_fresh_process_reconciles_each_browser_worker_loss_window(
    tmp_path: Path,
    phase: str,
    expected_error: str | None,
    expected_backend_calls: int,
) -> None:
    session_path = tmp_path / f"browser-process-{phase}.sqlite"

    async def prepare_parent() -> None:
        store = SQLiteSessionStore(session_path)
        try:
            await store.create(
                RunRequest(
                    session_id="parent-session",
                    agent_name="assistant",
                    messages=[Message.text("user", "open the page")],
                ),
                identity=SessionIdentity(
                    provider_name="process-fixture",
                    model="process-fixture-model",
                ),
                interaction_started_event=Event(
                    id="process-browser-interaction-started",
                    type=EventType.INTERACTION_STARTED,
                    session_id="parent-session",
                    interaction_id="process-browser-interaction",
                    agent_name="assistant",
                ),
                interaction_source_messages=[Message.text("user", "open the page")],
            )
        finally:
            await store.close()

    asyncio.run(prepare_parent())
    spawn = multiprocessing.get_context("spawn")
    calls_path = tmp_path / f"browser-process-{phase}-calls.json"
    result_path = tmp_path / f"browser-process-{phase}-result.json"
    worker = spawn.Process(
        target=_run_crashing_cayu_browser_worker,
        args=(str(session_path), str(tmp_path), phase, str(calls_path)),
    )
    worker.start()
    worker.join(timeout=20)
    if worker.is_alive():
        worker.terminate()
        worker.join(timeout=5)
        raise AssertionError("The crashing Cayu browser worker did not terminate.")
    assert worker.exitcode == _PROCESS_LOSS_EXIT_CODE
    calls = [] if not calls_path.exists() else json.loads(calls_path.read_text(encoding="utf-8"))
    assert len(calls) == expected_backend_calls

    recovery = spawn.Process(
        target=_run_fresh_cayu_browser_recovery,
        args=(str(session_path), phase, str(result_path)),
    )
    recovery.start()
    recovery.join(timeout=20)
    if recovery.is_alive():
        recovery.terminate()
        recovery.join(timeout=5)
        raise AssertionError("The fresh Cayu browser recovery process did not terminate.")
    assert recovery.exitcode == 0
    result = ToolResult.model_validate(json.loads(result_path.read_text(encoding="utf-8")))

    if expected_error is None:
        assert result.is_error is False
        assert result.structured["revision"] == "br_process_revision_1"
    else:
        assert result.is_error is True
        assert result.structured["error"] == expected_error


@pytest.mark.parametrize(
    "operation",
    ["back", "forward", "reload", "scroll", "hover", "upload"],
)
def test_issue_1275_fresh_process_reconciles_each_dispatched_operation_without_replay(
    tmp_path: Path,
    operation: str,
) -> None:
    session_path = tmp_path / f"browser-process-{operation}.sqlite"

    async def prepare_parent() -> None:
        store = SQLiteSessionStore(session_path)
        try:
            await store.create(
                RunRequest(
                    session_id="parent-session",
                    agent_name="assistant",
                    messages=[Message.text("user", "exercise browser operation")],
                ),
                identity=SessionIdentity(
                    provider_name="process-fixture",
                    model="process-fixture-model",
                ),
                interaction_started_event=Event(
                    id=f"process-{operation}-interaction-started",
                    type=EventType.INTERACTION_STARTED,
                    session_id="parent-session",
                    interaction_id=f"process-{operation}-interaction",
                    agent_name="assistant",
                ),
                interaction_source_messages=[Message.text("user", "exercise browser operation")],
            )
        finally:
            await store.close()

    asyncio.run(prepare_parent())
    calls_path = tmp_path / f"browser-process-{operation}-calls.json"
    arguments_path = tmp_path / f"browser-process-{operation}-arguments.json"
    result_path = tmp_path / f"browser-process-{operation}-result.json"
    spawn = multiprocessing.get_context("spawn")
    worker = spawn.Process(
        target=_run_crashing_cayu_browser_worker,
        args=(
            str(session_path),
            str(tmp_path),
            "after_dispatch",
            str(calls_path),
            operation,
            str(arguments_path),
        ),
    )
    worker.start()
    worker.join(timeout=20)
    if worker.is_alive():
        worker.terminate()
        worker.join(timeout=5)
        raise AssertionError("The browser operation worker did not terminate.")
    assert worker.exitcode == _PROCESS_LOSS_EXIT_CODE
    calls = json.loads(calls_path.read_text(encoding="utf-8"))
    assert [call["operation"] for call in calls] == ["navigate", operation]

    recovery = spawn.Process(
        target=_run_fresh_cayu_browser_recovery,
        args=(
            str(session_path),
            "after_dispatch",
            str(result_path),
            str(arguments_path),
        ),
    )
    recovery.start()
    recovery.join(timeout=20)
    if recovery.is_alive():
        recovery.terminate()
        recovery.join(timeout=5)
        raise AssertionError("The browser operation recovery process did not terminate.")
    assert recovery.exitcode == 0
    result = ToolResult.model_validate(json.loads(result_path.read_text(encoding="utf-8")))
    assert result.structured["error"] == "outcome_ambiguous"
    assert [call["operation"] for call in json.loads(calls_path.read_text())] == [
        "navigate",
        operation,
    ]


def test_browser_session_pending_recovery_reads_receipt_without_dispatch(tmp_path: Path) -> None:
    async def scenario() -> None:
        backend = _FakeBrowserBackend()
        records: dict[str, dict[str, Any]] = {}
        args = {
            "operation": "navigate",
            "url": "https://example.test/form",
            "operation_id": "DURABLE_OPERATION_SECRET_CANARY",
        }
        tool = _tool(backend)
        result = await tool.run(
            _durable_context(tmp_path, args=args, records=records),
            args,
        )
        operation_key = next(
            key
            for key, record in records.items()
            if record.get("record_type") == "cayu.browser-operation"
        )

        recovered = await _recover_durable_browser_result(
            tool,
            args=args,
            records=records,
            expected_disposition="confirmed",
        )
        assert recovered == result
        assert len(backend.calls) == 1
        assert "DURABLE_OPERATION_SECRET_CANARY" not in json.dumps(records)

        records[operation_key] = {
            key: value for key, value in records[operation_key].items() if key != "result"
        }
        records[operation_key]["state"] = "dispatched"
        ambiguous = await _recover_durable_browser_result(
            tool,
            args=args,
            records=records,
            expected_disposition="unresolved",
        )
        assert ambiguous.structured["error"] == "outcome_ambiguous"
        assert ambiguous.structured["browser_session_id"] == result.structured["session_id"]
        assert len(backend.calls) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "execution",
    [
        None,
        {},
        {"terminal": "settled"},
        {"dispatch": "completed"},
        {"terminal": "future", "dispatch": "completed"},
        {"terminal": "settled", "dispatch": "acknowledgement_lost"},
    ],
)
def test_browser_native_recovery_needs_positive_terminal_execution_evidence(
    tmp_path: Path, execution: dict[str, str] | None
) -> None:
    async def scenario() -> None:
        backend = _FakeBrowserBackend()
        tool = _tool(backend)
        records: dict[str, dict[str, Any]] = {}
        args = {
            "operation": "navigate",
            "url": "https://example.test/form",
            "operation_id": "native-terminal-proof",
        }
        original = await tool.run(_durable_context(tmp_path, args=args, records=records), args)
        assert not original.is_error
        operation = next(
            record
            for record in records.values()
            if record.get("record_type") == "cayu.browser-operation"
        )
        assert operation["state"] == "terminal"
        original_execution = operation["result"]["structured"].pop("execution")
        if execution is not None:
            operation["result"]["structured"]["execution"] = execution
        await _recover_durable_browser_result(
            tool, args=args, records=records, expected_disposition="unresolved"
        )
        operation["result"]["structured"]["execution"] = original_execution
        recovered = await _recover_durable_browser_result(
            tool, args=args, records=records, expected_disposition="confirmed"
        )
        assert recovered == original
        assert len(backend.calls) == 1

    asyncio.run(scenario())
