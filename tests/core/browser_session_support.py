"""Reusable browser-session backend, runner and durable-context test support."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from cayu.artifacts.base import ArtifactScope
from cayu.artifacts.local import LocalArtifactStore
from cayu.environments.admission import (
    ExecutionAdmissionCandidate,
    ExecutionCapabilityClaim,
    ExecutionCapabilityEvidence,
)
from cayu.runners.base import ExecCommand, ExecResult
from cayu.runners.workloads import PINNED_BROWSER_SESSION_WORKLOAD
from cayu.sessions._operation_records import SessionOperationPublication
from cayu.sessions.records import SessionStatus
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.base import ToolContext, ToolResult, _bind_runtime_tool_invocation_authority
from cayu.tools.browser_session import (
    BrowserArtifactPayload,
    BrowserBackendFailure,
    BrowserBackendIdentity,
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
from cayu.vaults.redaction import SecretRedactor

_IDENTITY = BrowserBackendIdentity(
    backend="playwright",
    backend_version="1.62.0",
    browser="chromium",
    browser_version="test-chromium",
    worker_protocol="cayu.browser-session.v4",
    worker_version="19",
)


@dataclass
class _FakeBrowserBackend(BrowserSessionBackend):
    calls: list[dict[str, Any]] = field(default_factory=list)
    preflight_calls: list[dict[str, Any]] = field(default_factory=list)
    failure: BrowserBackendFailure | BaseException | None = None
    failure_disposition: Literal["live", "retired", "uncertain"] = "uncertain"
    revision_number: int = 0
    title: str = "Example form"
    session_id: str | None = None
    page_id: str | None = None
    control_epoch: int = 1
    operation_count: int = 0
    observation_count: int = 0
    ref_count: int = 0
    artifact_count: int = 0
    last_operation_id_sha256: str | None = None

    async def preflight(
        self,
        ctx: ToolContext,
        request: dict[str, Any],
    ) -> BrowserBackendFailure | None:
        del ctx
        self.preflight_calls.append(dict(request))
        return None

    async def execute(self, ctx: ToolContext, request: dict[str, Any]) -> BrowserBackendResponse:
        del ctx
        self.calls.append(dict(request))
        if isinstance(self.failure, BaseException):
            raise self.failure
        if self.failure is not None:
            return BrowserBackendResponse(
                failure=self.failure,
                allocation_disposition=self.failure_disposition,
            )
        self.revision_number += 1
        operation = request["operation"]
        artifacts: tuple[BrowserArtifactPayload, ...] = ()
        if operation == "screenshot":
            artifacts = (
                BrowserArtifactPayload(
                    kind="screenshot",
                    filename="page.png",
                    content_type="image/png",
                    content=b"\x89PNG\r\n\x1a\nfixture",
                ),
            )
        elif operation == "download":
            artifacts = (
                BrowserArtifactPayload(
                    kind="download",
                    filename="report.txt",
                    content_type="text/plain",
                    content=b"downloaded",
                ),
            )
        if operation == "close":
            return BrowserBackendResponse(closed=True)
        session_id = request.get("session_id") or "bs_test_session"
        page_id = request.get("page_id") or "bp_test_page"
        if operation == "navigate":
            self.session_id = session_id
            self.page_id = page_id
            self.control_epoch = 1
            self.operation_count = 0
            self.observation_count = 0
            self.ref_count = 0
            self.artifact_count = 0
            self.last_operation_id_sha256 = None
        assert self.session_id == session_id
        assert self.page_id == page_id
        self.operation_count += 1
        if operation not in {"navigate", "observe"}:
            self.control_epoch += 1
        self.observation_count += 1
        self.ref_count += 2
        if artifacts:
            self.artifact_count += len(artifacts)
        self.last_operation_id_sha256 = hashlib.sha256(
            request["operation_id"].encode("utf-8")
        ).hexdigest()
        revision = f"br_revision_{self.revision_number}"
        observation = BrowserBackendObservation(
            session_id=session_id,
            page_id=page_id,
            revision=revision,
            creation_epoch=1,
            control_epoch=self.control_epoch,
            url=request.get("url", "https://example.test/form"),
            title=self.title,
            snapshot='- textbox "Name" [ref=ref_name]\n- button "Save" [ref=ref_save]',
            refs=(
                BrowserElementRef(ref="ref_name", role="textbox", name="Name"),
                BrowserElementRef(ref="ref_save", role="button", name="Save"),
            ),
            load_state="loaded",
            access_state="available",
            idle_timeout_seconds=900,
            truncation_reasons=(),
            backend_identity=_IDENTITY,
        )
        page_set = BrowserPageSetState(
            session_id=session_id,
            active_page_id=page_id,
            pages=(
                BrowserPageSummary(
                    page_id=page_id,
                    lifecycle="active",
                    creation_epoch=1,
                    control_epoch=self.control_epoch,
                    revision=revision,
                    url=observation.url,
                    title=observation.title,
                    load_state="loaded",
                    access_state="available",
                    last_observation_revision=revision,
                    last_operation_id_sha256=self.last_operation_id_sha256,
                    operation_count=self.operation_count,
                    observation_count=self.observation_count,
                    ref_count=self.ref_count,
                    request_count=0,
                    artifact_count=self.artifact_count,
                ),
            ),
            total_page_creations=1,
            total_operations=self.operation_count,
            total_observations=self.observation_count,
            total_refs=self.ref_count,
            total_requests=0,
            total_artifacts=self.artifact_count,
            cleanup_operation_count=0,
        )
        return BrowserBackendResponse(
            observation=observation,
            page_set=page_set,
            page_delta=(
                BrowserPageSetDelta(
                    created_page_ids=(page_id,),
                    admitted_page_ids=(page_id,),
                )
                if operation == "navigate"
                else BrowserPageSetDelta()
            ),
            artifacts=artifacts,
        )


@dataclass
class _Issue1275BrowserBackend(_FakeBrowserBackend):
    failure_disposition: Literal["live", "retired", "uncertain"] = "live"

    async def execute(self, ctx: ToolContext, request: dict[str, Any]) -> BrowserBackendResponse:
        response = await super().execute(ctx, request)
        if response.observation is None:
            return response
        assert response.page_set is not None
        operation = request["operation"]
        evidence = None
        if operation == "scroll":
            evidence = BrowserOperationEvidence(operation="scroll", moved=True, edge="none")
        elif operation == "upload":
            evidence = BrowserOperationEvidence(
                operation="upload",
                selected_file_count=len(request["artifact_ids"]),
                selection_state="selected",
            )
        observation = response.observation.model_copy(
            update={
                "snapshot": '- button "Action" [ref=ref_action]\n- filechooser "Single file" [ref=ref_upload_single]\n- filechooser "Multiple files" [ref=ref_upload_multiple]',
                "refs": (
                    BrowserElementRef(ref="ref_action", role="button", name="Action"),
                    BrowserElementRef(
                        ref="ref_upload_single",
                        role="button",
                        name="Single file",
                        element_type="file_input",
                        allows_multiple_files=False,
                    ),
                    BrowserElementRef(
                        ref="ref_upload_multiple",
                        role="button",
                        name="Multiple files",
                        element_type="file_input",
                        allows_multiple_files=True,
                    ),
                ),
            }
        )
        page_set = response.page_set.model_copy(
            update={
                "total_refs": 3 * self.observation_count,
                "pages": tuple(
                    page.model_copy(update={"ref_count": 3 * self.observation_count})
                    for page in response.page_set.pages
                ),
            }
        )
        return BrowserBackendResponse(
            observation=observation,
            page_set=page_set,
            page_delta=response.page_delta,
            operation_evidence=evidence,
            artifacts=response.artifacts,
        )


class _WireRunner:
    def execution_admission_candidate(self) -> ExecutionAdmissionCandidate:
        return ExecutionAdmissionCandidate(
            candidate="wire-browser",
            evidence=ExecutionCapabilityEvidence(
                subject="wire-browser",
                claims=tuple(
                    ExecutionCapabilityClaim.available(name)
                    for name in (
                        "deny_by_default_network",
                        "brokered_egress",
                        "confirmed_cancellation",
                        "confirmed_cleanup",
                    )
                ),
            ),
        )

    def workload_authority(self, name: str):
        if name == PINNED_BROWSER_SESSION_WORKLOAD.name:
            return PINNED_BROWSER_SESSION_WORKLOAD
        return None

    def output_secret_values_present(self) -> bool:
        return False

    async def preflight_exec(self, command: ExecCommand, **kwargs: Any) -> None:
        assert command.argv == list(PINNED_BROWSER_SESSION_WORKLOAD.command)
        assert json.loads(kwargs["stdin"])["operation"] in {
            "navigate",
            "observe",
            "click",
            "fill",
            "select",
            "press",
            "wait",
            "back",
            "forward",
            "reload",
            "scroll",
            "hover",
            "upload",
            "screenshot",
            "download",
            "close",
            "profile_restore",
            "profile_checkpoint",
            "list_pages",
            "switch_page",
            "close_page",
        }

    async def exec(self, command: ExecCommand, **kwargs: Any) -> ExecResult:
        assert command.argv == list(PINNED_BROWSER_SESSION_WORKLOAD.command)
        request = json.loads(kwargs["stdin"])
        return ExecResult(
            stdout=json.dumps(
                {
                    "protocol_version": "cayu.browser-session.v4",
                    "worker_version": "19",
                    "playwright_version": "1.62.0",
                    "kind": "success",
                    "allocation_disposition": "live",
                    "observation": {
                        "session_id": request["session_id"],
                        "page_id": request["page_id"],
                        "revision": "br_wire_revision",
                        "creation_epoch": 1,
                        "control_epoch": 1,
                        "url": request["url"],
                        "title": "Wire browser",
                        "snapshot": "- document",
                        "refs": [],
                        "load_state": "loaded",
                        "access_state": "available",
                        "idle_timeout_seconds": 900,
                        "truncation_reasons": [],
                        "backend_identity": {
                            "backend": "playwright",
                            "backend_version": "1.62.0",
                            "browser": "chromium",
                            "browser_version": "test-chromium",
                            "worker_protocol": "cayu.browser-session.v4",
                            "worker_version": "19",
                        },
                    },
                    "page_set": {
                        "session_id": request["session_id"],
                        "active_page_id": request["page_id"],
                        "pages": [
                            {
                                "page_id": request["page_id"],
                                "lifecycle": "active",
                                "creation_epoch": 1,
                                "control_epoch": 1,
                                "opener_page_id": None,
                                "creating_operation_id_sha256": None,
                                "revision": "br_wire_revision",
                                "url": request["url"],
                                "title": "Wire browser",
                                "load_state": "loaded",
                                "access_state": "available",
                                "last_observation_revision": "br_wire_revision",
                                "last_operation_id_sha256": hashlib.sha256(
                                    request["operation_id"].encode("utf-8")
                                ).hexdigest(),
                                "terminal_reason": None,
                                "operation_count": 1,
                                "observation_count": 1,
                                "ref_count": 0,
                                "request_count": 0,
                                "artifact_count": 0,
                            }
                        ],
                        "total_page_creations": 1,
                        "total_operations": 1,
                        "total_observations": 1,
                        "total_refs": 0,
                        "total_requests": 0,
                        "total_artifacts": 0,
                        "cleanup_operation_count": 0,
                    },
                    "page_delta": {
                        "created_page_ids": [request["page_id"]],
                        "admitted_page_ids": [request["page_id"]],
                        "closed_page_ids": [],
                        "crashed_page_ids": [],
                        "refused": [],
                    },
                    "artifacts": [],
                }
            )
        )


def _context(
    tmp_path: Path,
    *,
    artifact_store: LocalArtifactStore | None = None,
) -> ToolContext:
    store = artifact_store or LocalArtifactStore(
        tmp_path / "artifacts",
        store_id="browser-artifacts",
    )
    return ToolContext(
        session_id="parent-session",
        agent_name="assistant",
        environment_name="browser",
        artifact_store_id=store.id,
        artifact_store=store,
        idempotency_key="tool-call-1",
    )


async def _put_issue_1275_upload_artifact(
    ctx: ToolContext,
    *,
    content: bytes = b"issue-1275-upload",
    artifact_id: str | None = None,
    filename: str = "upload.txt",
    content_type: str = "text/plain",
    scope: ArtifactScope = ArtifactScope.SESSION,
    session_id: str | None = "parent-session",
    environment_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    assert ctx.artifact_store is not None
    artifact = await ctx.artifact_store.put_bytes(
        content,
        artifact_id=artifact_id,
        filename=filename,
        content_type=content_type,
        scope=scope,
        session_id=session_id,
        agent_name=ctx.agent_name,
        environment_name=environment_name,
        metadata=metadata,
    )
    return artifact.id


def _issue_1275_operation_args(
    operation: str,
    state: dict[str, Any],
    *,
    artifact_ids: list[str] | None = None,
    operation_id: str | None = None,
) -> dict[str, Any]:
    args: dict[str, Any] = {
        "operation": operation,
        "session_id": state["session_id"],
        "page_id": state["page_id"],
        "expected_revision": state["revision"],
        "operation_id": operation_id or f"{operation}-issue-1275",
        "expected_control_epoch": state["control_epoch"],
    }
    if operation == "scroll":
        args.update(direction="down", amount="page", repeat_count=2)
    elif operation == "hover":
        args["ref"] = "ref_action"
    elif operation == "upload":
        args["ref"] = "ref_upload_multiple"
        args["artifact_ids"] = list(artifact_ids or [])
    return args


def _durable_context(
    tmp_path: Path,
    *,
    args: dict[str, Any],
    records: dict[str, dict[str, Any]],
    runner: Any | None = None,
    artifact_store: LocalArtifactStore | None = None,
    allocation_fingerprint: str | None = "a" * 64,
    allocation_generation: str | None = None,
    execution_profile_fingerprint: str = "b" * 64,
    tool_call_id: str = "tool-call-1",
    fail_before_state: str | None = None,
    fail_after_state: str | None = None,
    secret_redactor: SecretRedactor | None = None,
    secret_tracker: Any | None = None,
    browser_control_epoch: Any | None = None,
    session_store: SQLiteSessionStore | None = None,
) -> ToolContext:
    from cayu.tools.base import _RuntimeBrowserAllocationAuthority

    ctx = _context(tmp_path, artifact_store=artifact_store).model_copy(
        update={
            "idempotency_key": f"tool-key-{tool_call_id}",
            "runner": runner,
        }
    )
    if secret_tracker is not None:
        ctx = ctx.model_copy(
            update={"invocation_secret_snapshot_provider": secret_tracker.snapshot}
        )

    async def load(key: str) -> dict[str, Any] | None:
        if session_store is not None:
            return await session_store.load_session_operation(ctx.session_id, key)
        record = records.get(key)
        return None if record is None else json.loads(json.dumps(record))

    async def compare_and_set(
        key: str,
        expected: dict[str, Any] | None,
        desired: dict[str, Any],
        secondary: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        assert records.get(key) == expected
        states = {
            value.get("state")
            for value in (desired, *secondary.values())
            if value.get("record_type") == "cayu.browser-operation"
        }
        if fail_before_state in states:
            raise ConnectionError(f"worker stopped before {fail_before_state} publication")
        if session_store is not None:

            def publish(current_session, checkpoint, current):
                assert current_session.run_epoch == 1
                assert current == expected
                return SessionOperationPublication(
                    checkpoint={} if checkpoint is None else checkpoint,
                    operation_records={key: desired, **secondary},
                )

            await session_store.publish_session_operation(
                ctx.session_id,
                idempotency_key=key,
                operation_transform=publish,
                events=[],
                expected_statuses={SessionStatus.RUNNING},
                expected_run_epoch=1,
            )
        records[key] = json.loads(json.dumps(desired))
        records.update(json.loads(json.dumps(secondary)))
        if fail_after_state in states:
            raise ConnectionError(f"worker stopped after {fail_after_state} publication")
        return json.loads(json.dumps(desired))

    def seal_durable_output(value: dict[str, Any]) -> dict[str, Any]:
        copied = json.loads(json.dumps(value))
        if secret_redactor is None:
            return copied
        redacted = secret_redactor.redact_json_values(copied)
        assert type(redacted) is dict
        return redacted

    async def control_admission(browser_session_id, operation):
        from cayu.tools.base import _RuntimeBrowserControlAdmission

        value = await browser_control_epoch(browser_session_id, operation)
        return _RuntimeBrowserControlAdmission(value) if type(value) is int else value

    _bind_runtime_tool_invocation_authority(
        ctx,
        parent_task_id=None,
        parent_run_epoch=1,
        model_step_id="model-step-1",
        model_attempt_id="model-attempt-1",
        tool_round_id="tool-round-1",
        tool_call_id=tool_call_id,
        tool_name="browser_session",
        idempotency_key=ctx.idempotency_key or "",
        effective_arguments=args,
        execution_profile_fingerprint=execution_profile_fingerprint,
        environment_allocation_fingerprint=allocation_fingerprint,
        environment_allocation_generation=allocation_generation,
        load_durable_operation=load,
        compare_and_set_durable_operation=compare_and_set,
        seal_durable_output=seal_durable_output,
        secret_publication_sealer=(
            secret_tracker.seal_for_publication if secret_tracker is not None else lambda: None
        ),
        browser_control_admission=control_admission if browser_control_epoch is not None else None,
        browser_allocation=(
            _RuntimeBrowserAllocationAuthority(
                session_id=ctx.session_id,
                session_instance_id="test-incarnation",
                run_epoch=1,
                interaction_id="test-interaction",
                execution_profile_fingerprint=execution_profile_fingerprint,
                environment_name=ctx.environment_name or "browser",
                allocation_fingerprint=allocation_fingerprint or "a" * 64,
            )
            if browser_control_epoch is not None
            else None
        ),
    )
    return ctx


async def _recover_durable_browser_result(
    tool: BrowserSessionTool,
    *,
    args: dict[str, Any],
    records: dict[str, dict[str, Any]],
    tool_call_id: str = "tool-call-1",
    parent_run_epoch: int = 1,
    allocation_fingerprint: str | None = "a" * 64,
    expected_disposition: str | None = None,
    session_store: SQLiteSessionStore | None = None,
) -> ToolResult:
    async def load(key: str) -> dict[str, Any] | None:
        if session_store is not None:
            return await session_store.load_session_operation("parent-session", key)
        record = records.get(key)
        return None if record is None else json.loads(json.dumps(record))

    result = await tool.reconcile_durable_tool_call(
        parent_session_id="parent-session",
        parent_run_epoch=parent_run_epoch,
        execution_profile_fingerprint="b" * 64,
        environment_name="browser",
        environment_allocation_fingerprint=allocation_fingerprint,
        model_step_id="model-step-1",
        model_attempt_id="model-attempt-1",
        tool_round_id="tool-round-1",
        tool_call_id=tool_call_id,
        idempotency_key=f"tool-key-{tool_call_id}",
        arguments=args,
        started=True,
        load_operation=load,
    )
    assert result is not None
    if expected_disposition is not None:
        assert result.disposition == expected_disposition
    return result.result


def _tool(backend: _FakeBrowserBackend) -> BrowserSessionTool:
    return BrowserSessionTool._from_backend_for_testing(backend)
