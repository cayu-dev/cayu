"""Browser-session operation admission, uploads, replay and cancellation contracts."""

from __future__ import annotations

import asyncio
import hashlib
import json
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from tests.core.browser_guest_support import _interactive_raw_request
from tests.core.browser_session_support import (
    _context,
    _durable_context,
    _Issue1275BrowserBackend,
    _issue_1275_operation_args,
    _put_issue_1275_upload_artifact,
    _recover_durable_browser_result,
)

import cayu.tools.browser_session as browser_session_module
from cayu.artifacts.base import ArtifactReadResult, ArtifactScope
from cayu.artifacts.local import LocalArtifactStore
from cayu.tools import _browser_guest
from cayu.tools._redaction import InvocationRedactorSnapshot
from cayu.tools.base import ToolContext, ToolResult
from cayu.tools.browser_session import (
    BrowserBackendFailure,
    BrowserBackendResponse,
    BrowserElementRef,
    BrowserOperationEvidence,
    BrowserSessionTool,
)
from cayu.tools.policy import TAINT_LABELS_METADATA_KEY
from cayu.vaults.redaction import SecretRedactor


class _MutableReadArtifactStore(LocalArtifactStore):
    replacement_content: bytes | None = None

    async def read_bytes(
        self,
        artifact_id: str,
        *,
        max_bytes: int | None = None,
    ) -> ArtifactReadResult:
        result = await super().read_bytes(artifact_id, max_bytes=max_bytes)
        replacement = self.replacement_content
        if replacement is None:
            return result
        assert len(replacement) == result.total_bytes
        return ArtifactReadResult(
            metadata=result.metadata,
            content=replacement,
            total_bytes=result.total_bytes,
        )


def test_upload_reconciliation_is_metadata_only_but_new_dispatch_requires_content() -> None:
    metadata = {
        "artifact_id": "art_upload",
        "content_sha256": hashlib.sha256(b"abc").hexdigest(),
        "filename": "upload.txt",
        "content_type": "text/plain",
        "size_bytes": 3,
    }
    raw = {
        **_interactive_raw_request("upload"),
        "expected_revision": "br_initial",
        "expected_control_epoch": 1,
        "ref": "ref_input",
        "artifact_ids": ["art_upload"],
        "upload_files": [{**metadata, "content_base64": "YWJj"}],
    }
    dispatched = _browser_guest._interactive_request_from_json(raw)
    replay_raw = {**raw, "reconcile_only": True, "upload_files": [metadata]}
    replay = _browser_guest._interactive_request_from_json(replay_raw)
    assert dispatched.upload_files[0].content == b"abc"
    assert replay.upload_files[0].content == b""
    fingerprint = _browser_guest._interactive_operation_fingerprint
    assert fingerprint(dispatched) == fingerprint(replay)
    with pytest.raises(_browser_guest._GuestFailure):
        _browser_guest._interactive_request_from_json({**replay_raw, "reconcile_only": False})
    for field_name, value in (
        ("content_sha256", "0" * 64),
        ("filename", "other.txt"),
        ("size_bytes", 4),
        ("content_type", "application/pdf"),
    ):
        changed = _browser_guest._interactive_request_from_json(
            {**replay_raw, "upload_files": [{**metadata, field_name: value}]}
        )
        assert fingerprint(changed) != fingerprint(dispatched)


@pytest.mark.parametrize(
    "configuration",
    [
        {"max_scroll_repeats": 5},
        {"max_upload_files": 5},
        {"max_upload_file_bytes": 9 * 1024 * 1024},
        {"max_upload_total_bytes": 17 * 1024 * 1024},
        {"max_upload_filename_bytes": 254},
        {"max_upload_materialization_ms": 31_000},
        {"allowed_upload_content_types": ["application/json", "text/plain"]},
        {"allowed_upload_taint_labels": ["private"]},
        {"expected_artifact_store_id": "another-artifact-store"},
    ],
)
def test_issue_1275_configuration_participates_in_execution_profile_identity(
    configuration: dict[str, Any],
) -> None:
    baseline = BrowserSessionTool()._execution_profile_material()
    changed = BrowserSessionTool(**configuration)._execution_profile_material()

    assert baseline is not None
    assert changed is not None
    assert changed != baseline


@pytest.mark.parametrize(
    "operation",
    ["back", "forward", "reload", "scroll", "hover", "upload"],
)
def test_issue_1275_operations_are_admitted_through_the_public_tool_contract(
    tmp_path: Path,
    operation: str,
) -> None:
    async def scenario() -> None:
        backend = _Issue1275BrowserBackend()
        tool = BrowserSessionTool._from_backend_for_testing(backend)
        ctx = _context(tmp_path)
        artifact_ids = (
            [await _put_issue_1275_upload_artifact(ctx)] if operation == "upload" else None
        )
        opened = await tool.run(
            ctx,
            {
                "operation": "navigate",
                "url": "https://example.test/upload",
                "operation_id": f"navigate-before-{operation}",
            },
        )

        result = await tool.run(
            ctx,
            _issue_1275_operation_args(
                operation,
                dict(opened.structured or {}),
                artifact_ids=artifact_ids,
            ),
        )

        assert result.is_error is False
        assert result.structured["revision"] != opened.structured["revision"]
        assert [call["operation"] for call in backend.calls] == ["navigate", operation]
        if operation == "scroll":
            assert result.structured["operation_evidence"] == {
                "operation": "scroll",
                "moved": True,
                "edge": "none",
                "selected_file_count": None,
                "selection_state": None,
            }
        elif operation == "upload":
            assert result.structured["operation_evidence"] == {
                "operation": "upload",
                "moved": None,
                "edge": None,
                "selected_file_count": 1,
                "selection_state": "selected",
            }
            assert backend.calls[-1]["artifact_ids"] == artifact_ids
            assert len(backend.calls[-1]["_upload_payloads"]) == 1
        else:
            assert "operation_evidence" not in result.structured

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("operation", "invalid_update", "remove_field"),
    [
        ("back", {"ref": "ref_action"}, None),
        ("forward", {"url": "https://example.test/escape"}, None),
        ("reload", {"expected_revision": True}, None),
        ("scroll", {"direction": "diagonal"}, None),
        ("scroll", {"repeat_count": True}, None),
        ("hover", {}, "ref"),
        ("upload", {"artifact_ids": ["art_same", "art_same"]}, None),
        ("upload", {"path": "/tmp/model-controlled"}, None),
    ],
)
def test_issue_1275_operation_validation_fails_closed_before_backend_dispatch(
    tmp_path: Path,
    operation: str,
    invalid_update: dict[str, Any],
    remove_field: str | None,
) -> None:
    async def scenario() -> None:
        backend = _Issue1275BrowserBackend()
        tool = BrowserSessionTool._from_backend_for_testing(backend)
        ctx = _context(tmp_path)
        opened = await tool.run(
            ctx,
            {
                "operation": "navigate",
                "url": "https://example.test/upload",
                "operation_id": f"navigate-invalid-{operation}",
            },
        )
        args = _issue_1275_operation_args(
            operation,
            dict(opened.structured or {}),
            artifact_ids=["art_upload"],
            operation_id=f"invalid-{operation}",
        )
        args.update(invalid_update)
        if remove_field is not None:
            args.pop(remove_field)

        refused = await tool.run(ctx, args)

        assert refused.structured["error"] == "invalid_arguments"
        assert refused.structured["execution"]["dispatch"] == "not_started"
        assert [call["operation"] for call in backend.calls] == ["navigate"]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "operation",
    ["back", "forward", "reload", "scroll", "hover", "upload"],
)
@pytest.mark.parametrize("backend_fails", [False, True])
def test_issue_1275_operations_invalidate_prior_revision_and_refs_once_dispatched(
    tmp_path: Path,
    operation: str,
    backend_fails: bool,
) -> None:
    async def scenario() -> None:
        backend = _Issue1275BrowserBackend()
        tool = BrowserSessionTool._from_backend_for_testing(backend)
        ctx = _context(tmp_path)
        artifact_ids = (
            [await _put_issue_1275_upload_artifact(ctx)] if operation == "upload" else None
        )
        opened = await tool.run(
            ctx,
            {
                "operation": "navigate",
                "url": "https://example.test/upload",
                "operation_id": f"navigate-stale-{operation}-{backend_fails}",
            },
        )
        old_state = dict(opened.structured or {})
        if backend_fails:
            backend.failure = BrowserBackendFailure("browser_crash")
        await tool.run(
            ctx,
            _issue_1275_operation_args(
                operation,
                old_state,
                artifact_ids=artifact_ids,
                operation_id=f"invalidate-{operation}-{backend_fails}",
            ),
        )
        backend.failure = None
        dispatch_count = len(backend.calls)

        stale = await tool.run(
            ctx,
            {
                "operation": "hover",
                "session_id": old_state["session_id"],
                "page_id": old_state["page_id"],
                "expected_revision": old_state["revision"],
                "expected_control_epoch": old_state["control_epoch"],
                "ref": "ref_action",
                "operation_id": f"stale-after-{operation}-{backend_fails}",
            },
        )

        assert stale.structured["error"] == "stale_observation"
        assert len(backend.calls) == dispatch_count

    asyncio.run(scenario())


def test_issue_1275_upload_requires_current_file_input_proof_before_dispatch(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        backend = _Issue1275BrowserBackend()
        tool = BrowserSessionTool._from_backend_for_testing(backend)
        ctx = _context(tmp_path)
        artifacts = [
            await _put_issue_1275_upload_artifact(
                ctx,
                content=f"file-{index}".encode(),
                filename=f"upload-{index}.txt",
            )
            for index in range(2)
        ]
        opened = await tool.run(
            ctx,
            {
                "operation": "navigate",
                "url": "https://example.test/upload",
                "operation_id": "navigate-file-input-proof",
            },
        )
        state = dict(opened.structured or {})

        ordinary_element = _issue_1275_operation_args(
            "upload",
            state,
            artifact_ids=[artifacts[0]],
            operation_id="upload-ordinary-element",
        )
        ordinary_element["ref"] = "ref_action"
        refused_ordinary = await tool.run(ctx, ordinary_element)
        single_input = _issue_1275_operation_args(
            "upload",
            state,
            artifact_ids=artifacts,
            operation_id="upload-multiple-to-single",
        )
        single_input["ref"] = "ref_upload_single"
        refused_multiple = await tool.run(ctx, single_input)

        assert refused_ordinary.structured["error"] == "incompatible_upload_target"
        assert refused_multiple.structured["error"] == "incompatible_upload_target"
        assert [call["operation"] for call in backend.calls] == ["navigate"]

    asyncio.run(scenario())


def test_issue_1275_upload_restores_durable_file_input_proof_before_dispatch(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store = LocalArtifactStore(
            tmp_path / "artifacts",
            store_id="browser-artifacts",
        )
        artifact_ctx = _context(tmp_path, artifact_store=store)
        artifact_id = await _put_issue_1275_upload_artifact(artifact_ctx)
        records: dict[str, dict[str, Any]] = {}
        first_backend = _Issue1275BrowserBackend()
        navigate_args = {
            "operation": "navigate",
            "url": "https://example.test/upload",
            "operation_id": "navigate-before-restored-upload",
        }
        opened = await BrowserSessionTool(_backend=first_backend).run(
            _durable_context(
                tmp_path,
                args=navigate_args,
                records=records,
                artifact_store=store,
                tool_call_id="navigate-before-restored-upload-call",
            ),
            navigate_args,
        )
        upload_args = _issue_1275_operation_args(
            "upload",
            dict(opened.structured or {}),
            artifact_ids=[artifact_id],
            operation_id="restored-upload",
        )
        replacement_backend = replace(first_backend, calls=[], preflight_calls=[])

        uploaded = await BrowserSessionTool(_backend=replacement_backend).run(
            _durable_context(
                tmp_path,
                args=upload_args,
                records=records,
                artifact_store=store,
                tool_call_id="restored-upload-call",
            ),
            upload_args,
        )

        assert uploaded.is_error is False
        assert [call["operation"] for call in replacement_backend.calls] == ["upload"]
        session_record = next(
            record
            for record in records.values()
            if record.get("record_type") == "cayu.browser-session"
        )
        assert session_record["page_authorities"][0]["file_input_refs"] == [
            {"ref": "ref_upload_multiple", "allows_multiple_files": True},
            {"ref": "ref_upload_single", "allows_multiple_files": False},
        ]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("case", "expected_error"),
    [
        ("wrong_session", "artifact_refused"),
        ("environment_scope", "artifact_refused"),
        ("file_count", "invalid_arguments"),
        ("file_size", "upload_too_large"),
        ("content_type", "artifact_refused"),
        ("taint", "artifact_refused"),
        ("secret", "artifact_refused"),
    ],
)
def test_issue_1275_upload_rejects_artifacts_outside_host_owned_bounds(
    tmp_path: Path,
    case: str,
    expected_error: str,
) -> None:
    async def scenario() -> None:
        case_root = tmp_path / case
        backend = _Issue1275BrowserBackend()
        tool_kwargs: dict[str, Any] = {}
        if case == "file_count":
            tool_kwargs["max_upload_files"] = 1
        elif case == "file_size":
            tool_kwargs.update(max_upload_file_bytes=4, max_upload_total_bytes=4)
        tool = BrowserSessionTool(_backend=backend, **tool_kwargs)
        ctx = _context(case_root)
        if case == "secret":
            secret = "ISSUE_1275_UPLOAD_SECRET_CANARY"
            ctx = ctx.model_copy(
                update={"invocation_secret_redactor": lambda: SecretRedactor(secret)}
            )
        artifact_kwargs: dict[str, Any] = {}
        if case == "wrong_session":
            artifact_kwargs["session_id"] = "different-parent-session"
        elif case == "environment_scope":
            artifact_kwargs.update(
                scope=ArtifactScope.ENVIRONMENT,
                session_id=None,
                environment_name="browser",
            )
        elif case == "file_size":
            artifact_kwargs["content"] = b"12345"
        elif case == "content_type":
            artifact_kwargs["content_type"] = "application/x-not-approved"
        elif case == "taint":
            artifact_kwargs["metadata"] = {TAINT_LABELS_METADATA_KEY: ["private"]}
        elif case == "secret":
            artifact_kwargs["content"] = b"ISSUE_1275_UPLOAD_SECRET_CANARY"
        artifact_ids = [
            await _put_issue_1275_upload_artifact(
                ctx,
                filename=f"{case}-one.txt",
                **artifact_kwargs,
            )
        ]
        if case == "file_count":
            artifact_ids.append(
                await _put_issue_1275_upload_artifact(
                    ctx,
                    content=b"second",
                    filename="file-count-two.txt",
                )
            )
        opened = await tool.run(
            ctx,
            {
                "operation": "navigate",
                "url": "https://example.test/upload",
                "operation_id": f"navigate-artifact-{case}",
            },
        )

        refused = await tool.run(
            ctx,
            _issue_1275_operation_args(
                "upload",
                dict(opened.structured or {}),
                artifact_ids=artifact_ids,
                operation_id=f"upload-artifact-{case}",
            ),
        )

        assert refused.structured["error"] == expected_error
        assert refused.structured["execution"]["dispatch"] == "not_started"
        assert [call["operation"] for call in backend.calls] == ["navigate"]

    asyncio.run(scenario())


@pytest.mark.parametrize("total_bytes", [5, 6, 7])
def test_upload_aggregate_limit_counts_individually_admissible_files(
    tmp_path: Path, total_bytes: int
) -> None:
    async def scenario() -> None:
        backend = _Issue1275BrowserBackend()
        tool = BrowserSessionTool(
            _backend=backend, max_upload_file_bytes=4, max_upload_total_bytes=6
        )
        ctx = _context(tmp_path)
        artifacts = [
            await _put_issue_1275_upload_artifact(ctx, content=b"abc", filename="first.txt"),
            await _put_issue_1275_upload_artifact(
                ctx, content=b"x" * (total_bytes - 3), filename="second.txt"
            ),
        ]
        opened = await tool.run(
            ctx,
            {"operation": "navigate", "url": "https://example.test/upload", "operation_id": "open"},
        )
        result = await tool.run(
            ctx,
            _issue_1275_operation_args("upload", dict(opened.structured), artifact_ids=artifacts),
        )
        if total_bytes > 6:
            assert result.structured["error"] == "upload_too_large"
            assert result.structured["execution"]["dispatch"] == "not_started"
            assert [call["operation"] for call in backend.calls] == ["navigate"]
        else:
            assert result.is_error is False
            assert result.structured["operation_evidence"]["selected_file_count"] == 2
            assert [call["operation"] for call in backend.calls] == ["navigate", "upload"]

    asyncio.run(scenario())


@pytest.mark.parametrize("operator_control", [False, True])
def test_issue_1275_upload_replay_is_content_bound_and_durable_evidence_is_byte_free(
    tmp_path: Path,
    operator_control: bool,
) -> None:
    async def scenario() -> None:
        async def control_epoch(browser_session_id: str, operation: str) -> int:
            return 3

        store = _MutableReadArtifactStore(
            tmp_path / "upload-store",
            store_id="browser-artifacts",
        )
        backend = _Issue1275BrowserBackend()
        tool = BrowserSessionTool(_backend=backend)
        records: dict[str, dict[str, Any]] = {}
        artifact_ctx = _context(tmp_path, artifact_store=store)
        artifact_id = await _put_issue_1275_upload_artifact(
            artifact_ctx,
            content=b"alpha",
            artifact_id="art_0123456789abcdef0123456789abcdef",
            filename="content-bound.txt",
        )
        navigate_args = {
            "operation": "navigate",
            "url": "https://example.test/upload",
            "operation_id": "navigate-content-bound",
        }
        opened = await tool.run(
            _durable_context(
                tmp_path,
                args=navigate_args,
                records=records,
                artifact_store=store,
                tool_call_id="navigate-content-bound-call",
                browser_control_epoch=control_epoch if operator_control else None,
            ),
            navigate_args,
        )
        upload_args = _issue_1275_operation_args(
            "upload",
            dict(opened.structured or {}),
            artifact_ids=[artifact_id],
            operation_id="upload-content-bound",
        )
        upload_ctx = _durable_context(
            tmp_path,
            args=upload_args,
            records=records,
            artifact_store=store,
            tool_call_id="upload-content-bound-call",
            browser_control_epoch=control_epoch if operator_control else None,
        )

        first = await tool.run(upload_ctx, upload_args)
        exact_replay = await tool.run(upload_ctx, upload_args)
        store.replacement_content = b"bravo"
        conflicting_replay = await tool.run(upload_ctx, upload_args)

        assert first.is_error is False
        assert exact_replay == first
        assert conflicting_replay.structured["error"] == "operation_conflict"
        assert [call["operation"] for call in backend.calls] == ["navigate", "upload"]
        durable_json = json.dumps(records, sort_keys=True)
        result_json = json.dumps(first.model_dump(mode="json"), sort_keys=True)
        for forbidden in (
            "alpha",
            "bravo",
            str(store.root),
            "_upload_payloads",
            "content_base64",
        ):
            assert forbidden not in durable_json
            assert forbidden not in result_json
        upload_record = next(
            record
            for record in records.values()
            if record.get("record_type") == "cayu.browser-operation"
            and record.get("upload_artifacts") is not None
        )
        if operator_control:
            assert upload_record["invocation_control_epoch"] == 3
            assert backend.calls[-1]["invocation_control_epoch"] == 3
        else:
            assert "invocation_control_epoch" not in upload_record
        assert upload_record["upload_artifacts"] == [
            {
                "artifact_id": artifact_id,
                "content_sha256": (
                    "8ed3f6ad685b959ead7022518e1af76cd816f8e8ec7ccdda1ed4018e8f2223f8"
                ),
                "filename": "content-bound.txt",
                "content_type": "text/plain",
                "size_bytes": 5,
            }
        ]

    asyncio.run(scenario())


def test_issue_1275_upload_rejects_wrong_selected_count_from_backend(tmp_path: Path) -> None:
    class _WrongCountBackend(_Issue1275BrowserBackend):
        async def execute(
            self,
            ctx: ToolContext,
            request: dict[str, Any],
        ) -> BrowserBackendResponse:
            response = await super().execute(ctx, request)
            if request["operation"] != "upload":
                return response
            return BrowserBackendResponse(
                observation=response.observation,
                operation_evidence=BrowserOperationEvidence(
                    operation="upload",
                    selected_file_count=2,
                    selection_state="selected",
                ),
            )

    async def scenario() -> None:
        backend = _WrongCountBackend()
        tool = BrowserSessionTool(_backend=backend)
        ctx = _context(tmp_path)
        artifact_id = await _put_issue_1275_upload_artifact(ctx)
        opened = await tool.run(
            ctx,
            {
                "operation": "navigate",
                "url": "https://example.test/upload",
                "operation_id": "navigate-before-wrong-count",
            },
        )

        result = await tool.run(
            ctx,
            _issue_1275_operation_args(
                "upload",
                dict(opened.structured or {}),
                artifact_ids=[artifact_id],
                operation_id="upload-wrong-count",
            ),
        )

        assert result.structured["error"] == "browser_crash"
        assert result.structured["execution"]["dispatch"] == "completed"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "operation_evidence",
    [
        {
            "operation": "upload",
            "selected_file_count": True,
            "selection_state": "selected",
        },
        {"operation": "scroll", "moved": 1, "edge": "none"},
    ],
)
def test_issue_1275_runner_rejects_coercible_operation_evidence(
    tmp_path: Path,
    operation_evidence: dict[str, Any],
) -> None:
    async def scenario() -> None:
        backend = _Issue1275BrowserBackend()
        response = await backend.execute(
            _context(tmp_path),
            {
                "operation": "navigate",
                "operation_id": "open-evidence",
                "session_id": "bs_test",
                "page_id": "bp_test",
                "url": "https://example.test/",
            },
        )
        assert response.observation is not None and response.page_set is not None
        payload = {
            "protocol_version": browser_session_module.BROWSER_SESSION_PROTOCOL_VERSION,
            "worker_version": browser_session_module.BROWSER_SESSION_WORKER_VERSION,
            "playwright_version": "1.62.0",
            "kind": "success",
            "allocation_disposition": "live",
            "observation": response.observation.model_dump(mode="json"),
            "page_set": response.page_set.model_dump(mode="json"),
            "page_delta": response.page_delta.model_dump(mode="json"),
            "artifacts": [],
        }

        def parse() -> BrowserBackendResponse:
            return browser_session_module._parse_runner_response(
                json.dumps(payload),
                max_artifact_bytes=1024,
                max_page_records=1,
                max_page_creations_per_operation=1,
            )

        assert parse().failure is None
        payload["operation_evidence"] = operation_evidence
        assert parse().failure == BrowserBackendFailure("browser_crash")

    asyncio.run(scenario())


def test_issue_1275_runner_rejects_coercible_file_input_authority() -> None:
    with pytest.raises(ValueError):
        BrowserElementRef(
            ref="ref_upload",
            role="filechooser",
            element_type="file_input",
            allows_multiple_files="false",
        )


def test_issue_1275_recovery_enforces_profile_upload_bounds(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = LocalArtifactStore(tmp_path / "artifacts", store_id="browser-artifacts")
        artifact_ctx = _context(tmp_path, artifact_store=store)
        artifact_id = await _put_issue_1275_upload_artifact(
            artifact_ctx,
            content=b"1234",
        )
        records: dict[str, dict[str, Any]] = {}
        backend = _Issue1275BrowserBackend()
        tool = BrowserSessionTool(
            max_upload_file_bytes=4,
            max_upload_total_bytes=4,
            _backend=backend,
        )
        navigate_args = {
            "operation": "navigate",
            "url": "https://example.test/upload",
            "operation_id": "navigate-before-bounded-recovery",
        }
        opened = await tool.run(
            _durable_context(
                tmp_path,
                args=navigate_args,
                records=records,
                artifact_store=store,
                tool_call_id="navigate-before-bounded-recovery-call",
            ),
            navigate_args,
        )
        upload_args = _issue_1275_operation_args(
            "upload",
            dict(opened.structured or {}),
            artifact_ids=[artifact_id],
            operation_id="bounded-upload-recovery",
        )
        await tool.run(
            _durable_context(
                tmp_path,
                args=upload_args,
                records=records,
                artifact_store=store,
                tool_call_id="bounded-upload-recovery-call",
            ),
            upload_args,
        )
        operation = next(
            record
            for record in records.values()
            if record.get("record_type") == "cayu.browser-operation"
            and record.get("operation") == "upload"
        )
        locator = next(
            record
            for record in records.values()
            if record.get("record_type") == "cayu.browser-operation-locator"
            and record.get("operation") == "upload"
        )
        operation["upload_artifacts"][0]["size_bytes"] = 5
        changed_digest = browser_session_module._upload_artifacts_sha256(
            {"upload_artifacts": operation["upload_artifacts"]}
        )
        operation["upload_artifacts_sha256"] = changed_digest
        locator["upload_artifacts_sha256"] = changed_digest

        recovered = await _recover_durable_browser_result(
            tool,
            args=upload_args,
            records=records,
            tool_call_id="bounded-upload-recovery-call",
        )

        assert recovered.structured["error"] == "authority_expired"
        assert [call["operation"] for call in backend.calls] == ["navigate", "upload"]

    asyncio.run(scenario())


def test_issue_1275_upload_rechecks_invocation_secret_scope_before_dispatch(
    tmp_path: Path,
) -> None:
    class _Tracker:
        revision = 0
        redactor = SecretRedactor()

        def snapshot(self) -> InvocationRedactorSnapshot:
            return InvocationRedactorSnapshot(self.revision, self.redactor)

        def reveal_upload_secret(self) -> None:
            self.revision += 1
            self.redactor = SecretRedactor("upload-content")

    class _ChangingPreflightBackend(_Issue1275BrowserBackend):
        async def preflight(
            self,
            ctx: ToolContext,
            request: dict[str, Any],
        ) -> BrowserBackendFailure | None:
            result = await super().preflight(ctx, request)
            if request["operation"] == "upload":
                tracker.reveal_upload_secret()
            return result

    async def scenario() -> None:
        nonlocal tracker
        tracker = _Tracker()
        backend = _ChangingPreflightBackend()
        tool = BrowserSessionTool._from_backend_for_testing(backend)
        ctx = _context(tmp_path).model_copy(
            update={"invocation_secret_snapshot_provider": tracker.snapshot}
        )
        artifact_id = await _put_issue_1275_upload_artifact(
            ctx,
            content=b"upload-content",
        )
        opened = await tool.run(
            ctx,
            {
                "operation": "navigate",
                "url": "https://example.test/upload",
                "operation_id": "navigate-before-secret-scope-change",
            },
        )

        refused = await tool.run(
            ctx,
            _issue_1275_operation_args(
                "upload",
                dict(opened.structured or {}),
                artifact_ids=[artifact_id],
                operation_id="upload-secret-scope-change",
            ),
        )

        assert refused.structured["error"] == "artifact_refused"
        assert refused.structured["execution"]["dispatch"] == "not_started"
        assert [call["operation"] for call in backend.calls] == ["navigate"]

    tracker: _Tracker
    asyncio.run(scenario())


def test_issue_1275_upload_rejects_mutated_store_evidence_without_diagnostic_leakage(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    canary = "ISSUE_1275_MUTATED_ARTIFACT_SECRET"

    class _Canary:
        def __repr__(self) -> str:
            return canary

        __str__ = __repr__

    class _MutatingStore(LocalArtifactStore):
        async def read_bytes(
            self,
            artifact_id: str,
            *,
            max_bytes: int | None = None,
        ) -> ArtifactReadResult:
            result = await super().read_bytes(artifact_id, max_bytes=max_bytes)
            object.__setattr__(result.metadata, "filename", _Canary())
            return result

    async def scenario() -> ToolResult:
        store = _MutatingStore(tmp_path / "mutating-store", store_id="browser-artifacts")
        ctx = _context(tmp_path, artifact_store=store)
        artifact_id = await _put_issue_1275_upload_artifact(ctx)
        backend = _Issue1275BrowserBackend()
        tool = BrowserSessionTool._from_backend_for_testing(backend)
        opened = await tool.run(
            ctx,
            {
                "operation": "navigate",
                "url": "https://example.test/upload",
                "operation_id": "navigate-mutated-artifact",
            },
        )
        result = await tool.run(
            ctx,
            _issue_1275_operation_args(
                "upload",
                dict(opened.structured or {}),
                artifact_ids=[artifact_id],
                operation_id="upload-mutated-artifact",
            ),
        )
        assert [call["operation"] for call in backend.calls] == ["navigate"]
        return result

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        result = asyncio.run(scenario())

    captured = capsys.readouterr()
    assert result.structured["error"] == "artifact_unavailable"
    assert canary not in repr(captured_warnings)
    assert canary not in caplog.text
    assert canary not in captured.out
    assert canary not in captured.err
    assert canary not in repr(result)


def test_issue_1275_upload_distinguishes_dependency_and_owner_cancellation(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    canary = "ISSUE_1275_ARTIFACT_CHILD_CANCELLATION_SECRET"

    class _CancellingStore(LocalArtifactStore):
        child_only = True
        started: asyncio.Event

        async def read_bytes(
            self,
            artifact_id: str,
            *,
            max_bytes: int | None = None,
        ) -> ArtifactReadResult:
            if self.child_only:
                raise asyncio.CancelledError(canary)
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                # Model a cancellation-opaque dependency that consumes the
                # delivery and completes its read anyway.
                return await super().read_bytes(artifact_id, max_bytes=max_bytes)
            raise AssertionError("unreachable")

    async def scenario() -> tuple[ToolResult, asyncio.Task[ToolResult], list[dict[str, Any]]]:
        store = _CancellingStore(tmp_path / "cancelling-store", store_id="browser-artifacts")
        store.started = asyncio.Event()
        ctx = _context(tmp_path, artifact_store=store)
        artifact_id = await _put_issue_1275_upload_artifact(ctx)
        backend = _Issue1275BrowserBackend()
        tool = BrowserSessionTool._from_backend_for_testing(backend)
        opened = await tool.run(
            ctx,
            {
                "operation": "navigate",
                "url": "https://example.test/upload",
                "operation_id": "navigate-artifact-cancellation",
            },
        )
        arguments = _issue_1275_operation_args(
            "upload",
            dict(opened.structured or {}),
            artifact_ids=[artifact_id],
            operation_id="upload-artifact-cancellation",
        )
        dependency_failure = await tool.run(ctx, arguments)

        store.child_only = False
        owner = asyncio.create_task(
            tool.run(ctx, {**arguments, "operation_id": "upload-owner-cancellation"})
        )
        await store.started.wait()
        owner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await owner
        return dependency_failure, owner, backend.calls

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        dependency_failure, owner, calls = asyncio.run(scenario())

    captured = capsys.readouterr()
    assert dependency_failure.structured["error"] == "artifact_unavailable"
    assert dependency_failure.structured["execution"]["dispatch"] == "not_started"
    assert owner.cancelled() is True
    assert owner.cancelling() == 1
    assert [call["operation"] for call in calls] == ["navigate"]
    assert canary not in repr(captured_warnings)
    assert canary not in caplog.text
    assert canary not in captured.out
    assert canary not in captured.err
    assert canary not in repr(dependency_failure)


def test_issue_1275_rejects_mutated_backend_evidence_without_diagnostic_leakage(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    canary = "ISSUE_1275_MUTATED_BACKEND_EVIDENCE_SECRET"

    class _Canary:
        def __repr__(self) -> str:
            return canary

        __str__ = __repr__

    class _MutatingBackend(_Issue1275BrowserBackend):
        async def execute(
            self,
            ctx: ToolContext,
            request: dict[str, Any],
        ) -> BrowserBackendResponse:
            response = await super().execute(ctx, request)
            if request["operation"] == "scroll":
                assert response.operation_evidence is not None
                object.__setattr__(response.operation_evidence, "moved", _Canary())
            return response

    async def scenario() -> ToolResult:
        backend = _MutatingBackend()
        tool = BrowserSessionTool._from_backend_for_testing(backend)
        ctx = _context(tmp_path)
        opened = await tool.run(
            ctx,
            {
                "operation": "navigate",
                "url": "https://example.test/scroll",
                "operation_id": "navigate-mutated-operation-evidence",
            },
        )
        return await tool.run(
            ctx,
            _issue_1275_operation_args(
                "scroll",
                dict(opened.structured or {}),
                operation_id="scroll-mutated-operation-evidence",
            ),
        )

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        result = asyncio.run(scenario())

    captured = capsys.readouterr()
    assert result.structured["error"] == "browser_crash"
    assert canary not in repr(captured_warnings)
    assert canary not in caplog.text
    assert canary not in captured.out
    assert canary not in captured.err
    assert canary not in repr(result)


@pytest.mark.parametrize(
    "operation",
    ["back", "forward", "reload", "scroll", "hover", "upload"],
)
def test_issue_1275_operations_replay_exactly_and_reject_conflicting_duplicates(
    tmp_path: Path,
    operation: str,
) -> None:
    async def scenario() -> None:
        backend = _Issue1275BrowserBackend()
        tool = BrowserSessionTool._from_backend_for_testing(backend)
        ctx = _context(tmp_path)
        artifact_ids = (
            [await _put_issue_1275_upload_artifact(ctx)] if operation == "upload" else None
        )
        opened = await tool.run(
            ctx,
            {
                "operation": "navigate",
                "url": "https://example.test/upload",
                "operation_id": f"navigate-deduplicate-{operation}",
            },
        )
        args = _issue_1275_operation_args(
            operation,
            dict(opened.structured or {}),
            artifact_ids=artifact_ids,
            operation_id=f"deduplicate-{operation}",
        )

        first = await tool.run(ctx, args)
        replay = await tool.run(ctx, args)
        conflict = await tool.run(
            ctx,
            {
                **args,
                "expected_revision": f"{args['expected_revision']}-different",
            },
        )

        assert replay == first
        assert conflict.structured["error"] == "operation_conflict"
        assert [call["operation"] for call in backend.calls] == ["navigate", operation]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "operation",
    ["back", "forward", "reload", "scroll", "hover", "upload"],
)
def test_issue_1275_operations_preserve_owner_cancellation_and_do_not_redispatch(
    tmp_path: Path,
    operation: str,
) -> None:
    class _BlockingBackend(_Issue1275BrowserBackend):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def execute(
            self,
            ctx: ToolContext,
            request: dict[str, Any],
        ) -> BrowserBackendResponse:
            if request["operation"] == operation:
                self.calls.append(dict(request))
                self.started.set()
                await self.release.wait()
                raise AssertionError("cancelled browser operation continued")
            return await super().execute(ctx, request)

    async def scenario() -> None:
        backend = _BlockingBackend()
        tool = BrowserSessionTool._from_backend_for_testing(backend)
        ctx = _context(tmp_path)
        artifact_ids = (
            [await _put_issue_1275_upload_artifact(ctx)] if operation == "upload" else None
        )
        opened = await tool.run(
            ctx,
            {
                "operation": "navigate",
                "url": "https://example.test/upload",
                "operation_id": f"navigate-cancel-{operation}",
            },
        )
        args = _issue_1275_operation_args(
            operation,
            dict(opened.structured or {}),
            artifact_ids=artifact_ids,
            operation_id=f"cancel-{operation}",
        )
        owner = asyncio.create_task(tool.run(ctx, args))
        await backend.started.wait()

        owner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await owner
        replay = await tool.run(ctx, args)

        assert owner.cancelled() is True
        assert owner.cancelling() == 1
        assert replay.structured["error"] == "outcome_ambiguous"
        assert [call["operation"] for call in backend.calls] == ["navigate", operation]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("operation", "failure_code"),
    [
        ("back", "navigation_timeout"),
        ("forward", "navigation_timeout"),
        ("reload", "navigation_timeout"),
        ("scroll", "timeout"),
        ("hover", "actionability_failed"),
        ("upload", "upload_materialization_failed"),
    ],
)
@pytest.mark.parametrize("failure_kind", ["timeout", "browser_crash", "cleanup_failed"])
def test_issue_1275_operations_report_stable_timeout_crash_and_cleanup_failures(
    tmp_path: Path,
    operation: str,
    failure_code: str,
    failure_kind: str,
) -> None:
    async def scenario() -> None:
        backend = _Issue1275BrowserBackend()
        tool = BrowserSessionTool._from_backend_for_testing(backend)
        ctx = _context(tmp_path)
        artifact_ids = (
            [await _put_issue_1275_upload_artifact(ctx)] if operation == "upload" else None
        )
        opened = await tool.run(
            ctx,
            {
                "operation": "navigate",
                "url": "https://example.test/upload",
                "operation_id": f"navigate-{failure_kind}-{operation}",
            },
        )
        expected = failure_code if failure_kind == "timeout" else failure_kind
        backend.failure = BrowserBackendFailure(expected)
        args = _issue_1275_operation_args(
            operation,
            dict(opened.structured or {}),
            artifact_ids=artifact_ids,
            operation_id=f"{failure_kind}-{operation}",
        )

        failed = await tool.run(ctx, args)
        replay = await tool.run(ctx, args)

        assert failed.structured["error"] == expected
        assert replay == failed
        assert [call["operation"] for call in backend.calls] == ["navigate", operation]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "operation",
    ["back", "forward", "reload", "scroll", "hover", "upload"],
)
def test_issue_1275_operations_are_bounded_by_shared_operation_capacity(
    tmp_path: Path,
    operation: str,
) -> None:
    async def scenario() -> None:
        backend = _Issue1275BrowserBackend()
        tool = BrowserSessionTool(max_operations=1, _backend=backend)
        ctx = _context(tmp_path)
        artifact_ids = (
            [await _put_issue_1275_upload_artifact(ctx)] if operation == "upload" else None
        )
        opened = await tool.run(
            ctx,
            {
                "operation": "navigate",
                "url": "https://example.test/upload",
                "operation_id": f"navigate-capacity-{operation}",
            },
        )

        refused = await tool.run(
            ctx,
            _issue_1275_operation_args(
                operation,
                dict(opened.structured or {}),
                artifact_ids=artifact_ids,
                operation_id=f"capacity-{operation}",
            ),
        )

        assert refused.structured["error"] == "resource_exhausted"
        assert refused.structured["execution"]["dispatch"] == "not_started"
        assert [call["operation"] for call in backend.calls] == ["navigate"]

    asyncio.run(scenario())
