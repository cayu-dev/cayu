"""Environment binding contracts."""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

import pytest
from tests.runtime.runtime_environment_support import (
    MemoryWorkspace,
    ProvisionedTestRunner,
    RecordingEnvironmentFactory,
    RecordingWorkspaceBinding,
    WorkspaceIdTool,
)
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_recovery_support import FailingTerminalToolEventStore
from tests.runtime.runtime_session_support import (
    _SETTLED_DRAIN_TIMEOUT_S,
    _private_record_for_public_event,
    collect_events,
    collect_tool_approval_events,
    tool_round_identity_payload,
)
from tests.runtime.runtime_tool_support import RequireApprovalPolicy, SideEffectTool

import cayu.runtime._environment_lifecycle as environment_lifecycle_module
import cayu.runtime.execution_profiles as execution_profiles_module
from cayu._validation import canonical_durable_json_bytes
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.tools import ToolApprovalDecision, ToolApprovalRequest
from cayu.configuration import CayuConfig, OperationsConfig
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.environments.bindings import SyncBinding, WorkspaceBinding, WorkspaceSnapshot
from cayu.environments.factory import (
    EnvironmentFactory,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
)
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.runtime._binding_cleanup import binding_cleanup_status, record_binding_cleanup_failure
from cayu.runtime.execution_profiles import EXECUTION_PROFILE_FINGERPRINT_FIELD
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import Session, SessionStatus
from cayu.sessions.requests import RunRequest
from cayu.tools.base import ToolResult
from cayu.workspaces.local import LocalWorkspace


def test_cayu_app_binds_environment_for_session_tools_and_finalize(tmp_path):
    async def run():
        store = InMemorySessionStore()
        configured_root = tmp_path / "configured"
        configured_root.mkdir()
        bound_root = tmp_path / "bound"
        bound_root.mkdir()
        configured_workspace = LocalWorkspace(configured_root, workspace_id="configured")
        bound_workspace = LocalWorkspace(bound_root, workspace_id="bound")
        binding = RecordingWorkspaceBinding(bound_workspace=bound_workspace)
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_1",
                        name="workspace_id",
                        arguments={},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                workspace=configured_workspace,
                binding=binding,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[WorkspaceIdTool()],
        )

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_binding",
                messages=[Message.text("user", "run tool")],
            ),
        )
        records = await store.query_events(EventQuery(session_id="sess_binding"))
        private_events = [
            _private_record_for_public_event(records, event).event for event in events
        ]
        return events, private_events, binding, configured_workspace, bound_workspace

    events, private_events, binding, configured_workspace, bound_workspace = asyncio.run(run())
    binding_generation_id = private_events[0].payload["binding_generation_id"]
    assert binding_generation_id.startswith("wbind_")

    assert [event.type for event in events[:3]] == [
        EventType.ENVIRONMENT_BINDING_STARTED,
        EventType.ENVIRONMENT_BINDING_COMPLETED,
        EventType.SESSION_STARTED,
    ]
    assert private_events[0].payload == {
        "binding_type": "RecordingWorkspaceBinding",
        "binding_generation_id": binding_generation_id,
        "configured_workspace_id": configured_workspace.id,
        "has_configured_runner": False,
        EXECUTION_PROFILE_FINGERPRINT_FIELD: private_events[0].payload[
            EXECUTION_PROFILE_FINGERPRINT_FIELD
        ],
    }
    assert private_events[1].payload == {
        "binding_type": "RecordingWorkspaceBinding",
        "binding_generation_id": binding_generation_id,
        "configured_workspace_id": configured_workspace.id,
        "has_configured_runner": False,
        "source_workspace_id": configured_workspace.id,
        "bound_workspace_id": bound_workspace.id,
        "bound_path": "/bound",
        "bound_metadata": {"binding": "recording"},
        "bound_snapshot": None,
        "has_bound_runner": False,
        EXECUTION_PROFILE_FINGERPRINT_FIELD: private_events[0].payload[
            EXECUTION_PROFILE_FINGERPRINT_FIELD
        ],
    }
    assert [event.type for event in events[-3:]] == [
        EventType.ENVIRONMENT_BINDING_FINALIZE_STARTED,
        EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert private_events[-3].payload["configured_workspace_id"] == configured_workspace.id
    assert private_events[-3].payload["binding_generation_id"] == binding_generation_id
    assert private_events[-3].payload["source_workspace_id"] == configured_workspace.id
    assert private_events[-3].payload["bound_workspace_id"] == bound_workspace.id
    assert private_events[-3].payload["outcome"] == "completed"
    assert private_events[-3].payload["bound_snapshot"] is None
    assert private_events[-2].payload["outcome"] == "completed"
    assert private_events[-2].payload["binding_generation_id"] == binding_generation_id
    assert private_events[-2].payload["final_snapshot"] is None
    assert events[-1].type == EventType.SESSION_COMPLETED
    assert len(binding.bind_calls) == 1
    assert binding.bind_calls[0]["workspace"] is configured_workspace
    assert binding.bind_calls[0]["session_id"] == "sess_binding"
    assert binding.bind_calls[0]["agent_name"] == "assistant"
    assert binding.bind_calls[0]["environment_name"] == "local"
    assert len(binding.finalize_calls) == 1
    assert binding.finalize_calls[0]["outcome"] == "completed"
    assert binding.finalize_calls[0]["bound"].workspace is bound_workspace

    tool_events = [event for event in events if event.type == EventType.TOOL_CALL_COMPLETED]
    assert tool_events[0].payload["result"]["structured"] == {"workspace_id": bound_workspace.id}


def test_environment_binding_completion_payload_is_isolated_from_started_event():
    async def run():
        configured_workspace = MemoryWorkspace("configured")
        binding = RecordingWorkspaceBinding()
        app = CayuApp(enable_logging=False)
        app.register_provider(
            FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]),
            default=True,
        )
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                workspace=configured_workspace,
                binding=binding,
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        stream = app.run(
            RunRequest(
                agent_name="assistant",
                session_id="sess_binding_payload_isolation",
                messages=[Message.text("user", "run")],
            )
        )
        interaction_started = await anext(stream)
        assert interaction_started.type == EventType.INTERACTION_STARTED
        started = None
        async for event in stream:
            if event.type is EventType.ENVIRONMENT_BINDING_STARTED:
                started = event
                break
        assert started is not None
        started.payload["binding_type"] = "consumer-spoof"
        started.payload["consumer_only"] = True
        remaining = [event async for event in stream]
        return started, remaining

    started, remaining = asyncio.run(run())

    assert started.type == EventType.ENVIRONMENT_BINDING_STARTED
    completed = next(
        event for event in remaining if event.type is EventType.ENVIRONMENT_BINDING_COMPLETED
    )
    assert completed.type == EventType.ENVIRONMENT_BINDING_COMPLETED
    assert completed.payload["binding_type"] == "RecordingWorkspaceBinding"
    assert "consumer_only" not in completed.payload


def test_cayu_app_binding_events_include_workspace_snapshots(tmp_path):
    async def run():
        store = InMemorySessionStore()
        configured_root = tmp_path / "configured"
        configured_root.mkdir()
        bound_root = tmp_path / "bound"
        bound_root.mkdir()
        configured_workspace = LocalWorkspace(configured_root, workspace_id="configured")
        bound_workspace = LocalWorkspace(bound_root, workspace_id="bound")
        bind_snapshot = WorkspaceSnapshot(
            snapshot_id="snap_bind",
            workspace_id=bound_workspace.id,
            version="commit-a",
            source="git",
            metadata={"branch": "main"},
        )
        final_snapshot = WorkspaceSnapshot(
            snapshot_id="snap_final",
            workspace_id=bound_workspace.id,
            version="commit-b",
            source="git",
            metadata={"dirty": False},
        )
        binding = RecordingWorkspaceBinding(
            bound_workspace=bound_workspace,
            snapshot=bind_snapshot,
            final_snapshot=final_snapshot,
        )
        provider = FakeProvider(
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                workspace=configured_workspace,
                binding=binding,
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_binding_snapshot",
                messages=[Message.text("user", "run")],
            ),
        )
        session = await store.load("sess_binding_snapshot")
        assert session is not None
        profile_fingerprint = execution_profiles_module.execution_profile_from_session_metadata(
            session.metadata
        ).fingerprint
        return events, binding, bind_snapshot, final_snapshot, profile_fingerprint

    events, binding, bind_snapshot, final_snapshot, profile_fingerprint = asyncio.run(run())

    binding_completed = next(
        event for event in events if event.type == EventType.ENVIRONMENT_BINDING_COMPLETED
    )
    finalize_started = next(
        event for event in events if event.type == EventType.ENVIRONMENT_BINDING_FINALIZE_STARTED
    )
    finalize_completed = next(
        event for event in events if event.type == EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED
    )
    binding_events = [
        event
        for event in events
        if event.type
        in {
            EventType.ENVIRONMENT_BINDING_STARTED,
            EventType.ENVIRONMENT_BINDING_COMPLETED,
            EventType.ENVIRONMENT_BINDING_FINALIZE_STARTED,
            EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED,
        }
    ]
    assert {event.payload.get("execution_profile_fingerprint") for event in binding_events} == {
        profile_fingerprint
    }
    assert binding_completed.payload["bound_snapshot"] == {
        "snapshot_id": bind_snapshot.snapshot_id,
        "workspace_id": bind_snapshot.workspace_id,
        "version": bind_snapshot.version,
        "source": bind_snapshot.source,
        "metadata": bind_snapshot.metadata,
    }
    assert finalize_started.payload["bound_snapshot"] == binding_completed.payload["bound_snapshot"]
    assert finalize_completed.payload["final_snapshot"] == {
        "snapshot_id": final_snapshot.snapshot_id,
        "workspace_id": final_snapshot.workspace_id,
        "version": final_snapshot.version,
        "source": final_snapshot.source,
        "metadata": final_snapshot.metadata,
    }
    assert binding.finalize_calls[0]["bound"].snapshot == bind_snapshot


def test_source_publication_receipt_is_integrity_linked_to_exact_sync_workspaces():
    source = MemoryWorkspace("source")
    target = MemoryWorkspace("target")
    binding = SyncBinding(
        target_workspace=target,
        max_file_bytes=1024,
        max_total_bytes=4096,
        source_conflict_policy="require_revision",
    )
    snapshot = WorkspaceSnapshot(
        snapshot_id="snapshot-publication",
        workspace_id=source.id,
        version="revision-final",
        source="sync",
        metadata={
            "target_workspace_id": target.id,
            "outcome": "completed",
            "source_conflict_policy": "require_revision",
            "sync_back": "always",
            "delete_missing": True,
            "copied_files": 2,
            "copied_bytes": 32,
            "deleted_files": 1,
        },
    )

    receipt = environment_lifecycle_module._source_publication_receipt_payload(
        snapshot,
        binding=binding,
        destination_workspace_id=source.id,
        workload_workspace_id=target.id,
        expected_destination_workspace_id=source.id,
        expected_workload_workspace_id=target.id,
    )

    assert receipt is not None
    material = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    assert (
        receipt["receipt_sha256"]
        == "sha256:"
        + hashlib.sha256(
            canonical_durable_json_bytes(material, "source_publication_receipt")
        ).hexdigest()
    )
    snapshot_material = {
        "schema": "cayu.source_publication_snapshot.v1",
        "destination_workspace_id": source.id,
        "workload_workspace_id": target.id,
        "source": "sync",
        "outcome": "completed",
        "source_conflict_policy": "require_revision",
        "sync_back": "always",
        "delete_missing": True,
        "copied_files": 2,
        "copied_bytes": 32,
        "deleted_files": 1,
    }
    assert (
        receipt["snapshot_sha256"]
        == "sha256:"
        + hashlib.sha256(
            canonical_durable_json_bytes(
                snapshot_material,
                "source_publication_snapshot",
            )
        ).hexdigest()
    )
    assert receipt["destination_workspace_id"] == source.id
    assert receipt["workload_workspace_id"] == target.id
    assert (
        environment_lifecycle_module._source_publication_receipt_payload(
            WorkspaceSnapshot(
                snapshot_id=snapshot.snapshot_id,
                workspace_id="other-source",
                version=snapshot.version,
                source=snapshot.source,
                metadata=snapshot.metadata,
            ),
            binding=binding,
            destination_workspace_id=source.id,
            workload_workspace_id=target.id,
            expected_destination_workspace_id=source.id,
            expected_workload_workspace_id=target.id,
        )
        is None
    )


def test_final_git_receipt_rejects_non_docker_sync_binding_authority():
    source = MemoryWorkspace("source")
    target = MemoryWorkspace("target")

    class ForgingSyncBinding(SyncBinding):
        pass

    binding = ForgingSyncBinding(
        target_workspace=target,
        max_file_bytes=1024,
        max_total_bytes=4096,
        source_conflict_policy="require_revision",
    )
    base_result = {
        "scope": "all",
        "changes": [],
        "returned": 0,
        "offset": 0,
        "limit": 200,
        "truncated": False,
        "truncation_reasons": [],
        "next_offset": None,
    }
    snapshot = WorkspaceSnapshot(
        snapshot_id="forged-final-git",
        workspace_id=source.id,
        version="sha256:" + "c" * 64,
        source="sync",
        metadata={
            "target_workspace_id": target.id,
            "final_git_evidence": {
                "request_fingerprint": "sha256:" + "a" * 64,
                "source_workspace_id": source.id,
                "baseline_revision": "sha256:" + "b" * 64,
                "workspace_revision": "sha256:" + "c" * 64,
                "status": {"structured": {**base_result, "mode": "status"}},
                "summary": {"structured": {**base_result, "mode": "summary"}},
                "diff": {
                    "content": "No textual diff for the selected changes.",
                    "structured": {
                        **base_result,
                        "mode": "diff",
                        "diff_offset": 0,
                        "next_diff_offset": None,
                        "binary_omitted": False,
                    },
                },
            },
        },
    )

    assert (
        environment_lifecycle_module._final_git_receipt_payload(
            snapshot,
            binding=binding,
            destination_workspace_id=source.id,
            workload_workspace_id=target.id,
            expected_destination_workspace_id=source.id,
            expected_workload_workspace_id=target.id,
        )
        is None
    )


def test_cayu_app_binding_failure_fails_session_before_start_event():
    async def run():
        store = InMemorySessionStore()
        binding = RecordingWorkspaceBinding(fail_bind=True)
        provider = FakeProvider(
            [
                ModelStreamEvent.text_delta("unreached"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), binding=binding),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_binding_fail",
                messages=[Message.text("user", "run")],
            ),
        )
        session = await store.load("sess_binding_fail")
        return events, session, binding, provider

    events, session, binding, provider = asyncio.run(run())

    assert session is not None
    assert session.status == SessionStatus.FAILED
    assert [event.type for event in events] == [
        EventType.ENVIRONMENT_BINDING_STARTED,
        EventType.ENVIRONMENT_BINDING_FAILED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert events[1].payload["error"] == "bind failed"
    assert events[1].payload["error_type"] == "RuntimeError"
    assert events[3].payload["error"] == "bind failed"
    assert events[3].payload["error_type"] == "RuntimeError"
    assert len(binding.bind_calls) == 1
    assert binding.finalize_calls == []
    assert provider.requests == []


def test_binding_failure_retains_timed_out_factory_release_owner(tmp_path):
    async def run():
        release_started = asyncio.Event()
        finish_release = asyncio.Event()
        release_completed = asyncio.Event()
        release_actions: list[EnvironmentFactoryReleaseAction] = []

        async def release(action: EnvironmentFactoryReleaseAction) -> None:
            release_actions.append(action)
            release_started.set()
            await finish_release.wait()
            release_completed.set()

        workspace_root = tmp_path / "binding-release-timeout"
        workspace_root.mkdir()
        binding = RecordingWorkspaceBinding(fail_bind=True)
        factory = RecordingEnvironmentFactory(
            Environment(
                EnvironmentSpec(name="dynamic"),
                workspace=LocalWorkspace(workspace_root),
                binding=binding,
            ),
            release=release,
            release_timeout_s=0.01,
        )
        app = CayuApp(
            enable_logging=False,
            config=CayuConfig(operations=OperationsConfig(max_environment_lifecycle_owners=1)),
        )
        app.register_provider(
            FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]),
            default=True,
        )
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        first = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="binding-release-owner",
                messages=[Message.text("user", "run")],
            ),
        )
        assert release_started.is_set()
        assert not release_completed.is_set()
        assert "binding-release-owner" in (
            app._environment_lifecycle._deferred_factory_cleanup_tasks
        )
        assert "binding-release-owner" in (
            app._environment_lifecycle._pending_environment_owner_admissions
        )
        assert "binding-release-owner" not in (
            app._environment_lifecycle._active_environment_setups
        )

        contender = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="binding-release-contender",
                messages=[Message.text("user", "run")],
            ),
        )
        assert len(factory.requests) == 1

        finish_release.set()
        assert await app.drain_environment_cleanups(timeout_s=_SETTLED_DRAIN_TIMEOUT_S) is True
        return first, contender, release_actions, release_completed.is_set(), app

    first, contender, release_actions, release_completed, app = asyncio.run(run())

    assert first[-1].type is EventType.SESSION_FAILED
    assert first[-1].payload["error"] == "bind failed"
    assert contender[-1].type is EventType.SESSION_FAILED
    assert "EnvironmentCapacityError" in str(contender[-1].payload)
    assert release_actions == [EnvironmentFactoryReleaseAction.PRESERVE]
    assert release_completed is True
    assert app._environment_lifecycle._deferred_factory_cleanup_tasks == {}
    assert app._environment_lifecycle._pending_environment_owner_admissions == set()


@pytest.mark.parametrize("retry_fails", [False, True])
def test_cayu_app_reports_and_retries_incomplete_binding_cleanup(retry_fails: bool):
    class CleanupRetryRunner(ProvisionedTestRunner):
        def __init__(self) -> None:
            self.cleanup_retry_calls = 0

        async def close(self) -> None:
            raise AssertionError("runtime must use the binding-owned cleanup retry")

    class CleanupFailingBinding(WorkspaceBinding):
        async def bind(self, workspace, runner, **kwargs):  # type: ignore[no-untyped-def]
            bind_error = RuntimeError("bind failed")
            cleanup_error = RuntimeError("binding rollback failed")

            async def retry() -> None:
                runner.cleanup_retry_calls += 1
                if retry_fails:
                    raise RuntimeError("runner cleanup retry failed")

            record_binding_cleanup_failure(bind_error, cleanup_error, retry=retry)
            raise bind_error

        async def finalize(self, bound, *, outcome=None, metadata=None):  # type: ignore[no-untyped-def]
            raise AssertionError("finalize should not run")

    async def run():
        store = InMemorySessionStore()
        runner = CleanupRetryRunner()
        provider = FakeProvider(
            [
                ModelStreamEvent.text_delta("unreached"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                runner=runner,
                binding=CleanupFailingBinding(),
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=f"sess_binding_cleanup_retry_{retry_fails}",
                messages=[Message.text("user", "run")],
            ),
        )
        return events, runner, provider

    events, runner, provider = asyncio.run(run())

    expected_cleanup: dict[str, Any] = {
        "incomplete": retry_fails,
        "initial_error": "binding rollback failed",
        "initial_error_type": "RuntimeError",
        "retry_attempted": True,
        "retry_completed": not retry_fails,
    }
    if retry_fails:
        expected_cleanup.update(
            {
                "retry_error": "runner cleanup retry failed",
                "retry_error_type": "RuntimeError",
            }
        )
    binding_failed = next(
        event for event in events if event.type is EventType.ENVIRONMENT_BINDING_FAILED
    )
    session_failed = next(event for event in events if event.type is EventType.SESSION_FAILED)
    assert binding_failed.payload["binding_cleanup"] == expected_cleanup
    assert session_failed.payload["binding_cleanup"] == expected_cleanup
    assert binding_failed.payload["error"] == "bind failed"
    assert session_failed.payload["error"] == "bind failed"
    assert runner.cleanup_retry_calls == 1
    assert provider.requests == []


def test_cayu_app_persists_bind_failure_before_reraising_cancelled_cleanup_retry():
    bind_error = RuntimeError("bind failed")
    initial_cleanup_error = RuntimeError("binding rollback failed")
    initial_cancellation = asyncio.CancelledError("bind cancelled")
    retry_cleanup_error = RuntimeError("cleanup retry failed")
    retry_cancellation = asyncio.CancelledError("retry cancelled")

    class CancelledCleanupRetryBinding(WorkspaceBinding):
        async def bind(self, workspace, runner, **kwargs):  # type: ignore[no-untyped-def]
            async def retry() -> None:
                raise BaseExceptionGroup(
                    "cleanup retry cancelled and failed",
                    [retry_cancellation, retry_cleanup_error],
                )

            failure = BaseExceptionGroup(
                "bind rollback cancelled and failed",
                [bind_error, initial_cleanup_error, initial_cancellation],
            )
            record_binding_cleanup_failure(
                failure,
                initial_cleanup_error,
                retry=retry,
            )
            raise failure

        async def finalize(self, bound, *, outcome=None, metadata=None):  # type: ignore[no-untyped-def]
            raise AssertionError("finalize should not run")

    async def run() -> tuple[BaseExceptionGroup, list[Event], Session]:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([]), default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                binding=CancelledCleanupRetryBinding(),
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        with pytest.raises(BaseExceptionGroup) as exc_info:
            await collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_cancelled_binding_cleanup_retry",
                    messages=[Message.text("user", "run")],
                ),
            )
        events = await store.load_events("sess_cancelled_binding_cleanup_retry")
        session = await store.load("sess_cancelled_binding_cleanup_retry")
        assert session is not None
        return exc_info.value, events, session

    failure, events, session = asyncio.run(run())

    assert failure.exceptions[0].exceptions == (
        bind_error,
        initial_cleanup_error,
        initial_cancellation,
    )
    assert failure.exceptions[1].exceptions == (
        retry_cancellation,
        retry_cleanup_error,
    )
    cleanup_status = binding_cleanup_status(failure)
    assert cleanup_status is not None
    assert cleanup_status.retry_attempted is True
    assert cleanup_status.retry_error is failure.exceptions[1]
    binding_failed = next(
        event for event in events if event.type is EventType.ENVIRONMENT_BINDING_FAILED
    )
    assert binding_failed.payload["binding_cleanup"] == {
        "incomplete": True,
        "initial_error": "binding rollback failed",
        "initial_error_type": "RuntimeError",
        "retry_attempted": True,
        "retry_completed": False,
        "retry_error": "cleanup retry cancelled and failed (2 sub-exceptions)",
        "retry_error_type": "BaseExceptionGroup",
    }
    assert session.status is SessionStatus.INTERRUPTED
    assert any(event.type is EventType.SESSION_INTERRUPTED for event in events)


def test_cayu_app_propagates_real_cancellation_during_ordinary_bind_retry():
    retry_started = asyncio.Event()
    bind_error = RuntimeError("bind failed")
    cleanup_error = RuntimeError("binding rollback failed")

    class OrdinaryBindRetryBinding(WorkspaceBinding):
        async def bind(self, workspace, runner, **kwargs):  # type: ignore[no-untyped-def]
            async def retry() -> None:
                retry_started.set()
                await asyncio.Event().wait()

            record_binding_cleanup_failure(bind_error, cleanup_error, retry=retry)
            raise bind_error

        async def finalize(self, bound, *, outcome=None, metadata=None):  # type: ignore[no-untyped-def]
            raise AssertionError("finalize should not run")

    async def run() -> tuple[BaseExceptionGroup, list[Event]]:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([]), default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                binding=OrdinaryBindRetryBinding(),
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_ordinary_bind_retry_cancelled",
                    messages=[Message.text("user", "run")],
                ),
            )
        )
        await retry_started.wait()
        task.cancel()
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await task
        return exc_info.value, await store.load_events("sess_ordinary_bind_retry_cancelled")

    failure, events = asyncio.run(run())
    assert isinstance(failure.exceptions[0], RuntimeError)
    assert isinstance(failure.exceptions[1], asyncio.CancelledError)
    assert any(event.type is EventType.ENVIRONMENT_BINDING_FAILED for event in events)


def test_cayu_app_persists_factory_failure_before_grouped_cancellation_cleanup():
    factory_error = RuntimeError("factory setup failed")
    cleanup_error = RuntimeError("factory rollback failed")
    cancellation = asyncio.CancelledError("factory rollback cancelled")

    class FailingFactory(EnvironmentFactory):
        async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
            raise BaseExceptionGroup(
                "factory rollback cancelled and failed",
                [factory_error, cleanup_error, cancellation],
            )

    async def run() -> tuple[BaseExceptionGroup, list[Event], Session]:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([]), default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="factory-env"),
            FailingFactory(),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_grouped_factory_cleanup",
                    messages=[Message.text("user", "run")],
                ),
            )
        events = await store.load_events("sess_grouped_factory_cleanup")
        session = await store.load("sess_grouped_factory_cleanup")
        assert session is not None
        return exc_info.value, events, session

    failure, events, session = asyncio.run(run())

    assert failure.exceptions == (factory_error, cleanup_error, cancellation)
    factory_failed = next(
        event for event in events if event.type is EventType.ENVIRONMENT_FACTORY_FAILED
    )
    assert factory_failed.payload["error_type"] == "BaseExceptionGroup"
    assert session.status is SessionStatus.INTERRUPTED
    assert any(event.type is EventType.SESSION_INTERRUPTED for event in events)


def test_cayu_app_preserves_bind_signals_when_failure_publication_fails():
    bind_error = RuntimeError("bind failed")
    cleanup_error = RuntimeError("rollback failed")
    cancellation = asyncio.CancelledError("bind cancelled")
    publication_error = RuntimeError("binding failure event unavailable")

    class FailingBinding(WorkspaceBinding):
        async def bind(self, workspace, runner, **kwargs):  # type: ignore[no-untyped-def]
            failure = BaseExceptionGroup(
                "bind rollback cancelled and failed",
                [bind_error, cleanup_error, cancellation],
            )

            async def retry() -> None:
                return None

            record_binding_cleanup_failure(failure, cleanup_error, retry=retry)
            raise failure

        async def finalize(self, bound, *, outcome=None, metadata=None):  # type: ignore[no-untyped-def]
            raise AssertionError("finalize should not run")

    class FailingStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def append_event(self, session_id: str, event: Event) -> None:
            if event.type is EventType.ENVIRONMENT_BINDING_FAILED:
                raise publication_error
            await super().append_event(session_id, event)

    async def run() -> BaseExceptionGroup:
        app = CayuApp(session_store=FailingStore(), enable_logging=False)
        app.register_provider(FakeProvider([]), default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), binding=FailingBinding()),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_bind_failure_publication_fails",
                    messages=[Message.text("user", "run")],
                ),
            )
        return exc_info.value

    failure = asyncio.run(run())

    assert failure.exceptions[0].exceptions == (bind_error, cleanup_error, cancellation)
    assert failure.exceptions[1] is publication_error
    assert failure.__cause__ is cancellation
    cleanup_status = binding_cleanup_status(failure)
    assert cleanup_status is not None
    assert cleanup_status.retry_attempted is True
    assert cleanup_status.retry_error is None


def test_cayu_app_does_not_publish_fatal_only_bind_group_as_ordinary_failure():
    fatal_signal = GeneratorExit("binding aborted")
    failure_event_attempted = False

    class FatalBinding(WorkspaceBinding):
        async def bind(self, workspace, runner, **kwargs):  # type: ignore[no-untyped-def]
            raise BaseExceptionGroup("fatal bind failure", [fatal_signal])

        async def finalize(self, bound, *, outcome=None, metadata=None):  # type: ignore[no-untyped-def]
            raise AssertionError("finalize should not run")

    class RecordingStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def append_event(self, session_id: str, event: Event) -> None:
            nonlocal failure_event_attempted
            if event.type is EventType.ENVIRONMENT_BINDING_FAILED:
                failure_event_attempted = True
            await super().append_event(session_id, event)

    async def run() -> BaseExceptionGroup:
        app = CayuApp(session_store=RecordingStore(), enable_logging=False)
        app.register_provider(FakeProvider([]), default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), binding=FatalBinding()),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_fatal_only_bind_group",
                    messages=[Message.text("user", "run")],
                ),
            )
        return exc_info.value

    failure = asyncio.run(run())

    assert failure.exceptions == (fatal_signal,)
    assert failure_event_attempted is False


def test_cayu_app_binding_finalize_failure_is_reported_on_terminal_event():
    from cayu.runtime._binding_cleanup import BINDING_FINALIZE_ERROR_TEXT_MAX_BYTES
    from cayu.vaults.redaction import REDACTED_SECRET, SecretRedactor

    secret = "binding-finalize-secret"
    raw_message = f"finalize failed: {secret}: {'界' * 300}"
    finalize_error = RuntimeError(raw_message)

    async def run():
        store = InMemorySessionStore()
        binding = RecordingWorkspaceBinding(
            fail_finalize=True,
            finalize_error=finalize_error,
        )
        provider = FakeProvider(
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), binding=binding),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_binding_finalize_fail",
                messages=[Message.text("user", "run")],
            ),
        )
        session = await store.load("sess_binding_finalize_fail")
        return events, session, binding

    events, session, binding = asyncio.run(run())

    assert session is not None
    assert session.status == SessionStatus.COMPLETED
    assert [event.type for event in events[-3:]] == [
        EventType.ENVIRONMENT_BINDING_FINALIZE_STARTED,
        EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED,
        EventType.SESSION_COMPLETED,
    ]
    persisted_error = events[-2].payload["error"]
    assert REDACTED_SECRET in persisted_error
    assert secret not in persisted_error
    assert persisted_error.endswith("... [truncated]")
    assert len(persisted_error.encode("utf-8")) <= BINDING_FINALIZE_ERROR_TEXT_MAX_BYTES
    assert events[-2].payload["error_type"] == "RuntimeError"
    assert events[-2].payload["outcome"] == "completed"
    assert events[-2].payload["final_revision"] == {
        "workspace_id": "workspace-unavailable",
        "observer": "RecordingWorkspaceBinding",
        "status": "unsupported",
        "revision": None,
        "head_revision": None,
        "branch": None,
        "path_scope": "complete",
        "total_paths": 0,
        "detail_code": "revision_observation_unsupported",
        "finalization_delta": {
            "attribution_confidence": "unattributed_finalization_change",
            "status": "incomplete",
            "before_revision": None,
            "after_revision": None,
            "paths": [],
            "retained_paths": 0,
            "total_paths": 0,
            "truncated": False,
            "head_changed": False,
            "branch_changed": False,
            "detail_code": "finalization_baseline_unavailable",
        },
    }
    assert events[-1].type == EventType.SESSION_COMPLETED
    assert len(binding.finalize_calls) == 1
    assert events[-1].payload["binding_finalize_error"] == {
        "error": persisted_error,
        "error_type": "RuntimeError",
        "outcome": "completed",
        "failures": [
            {
                "phase": "workspace_finalize",
                "error": persisted_error,
                "error_type": "RuntimeError",
            }
        ],
    }
    assert secret not in str(events[-1].payload["binding_finalize_error"])
    assert events[-1].payload["final_revision"] == events[-2].payload["final_revision"]
    assert str(finalize_error) == raw_message


def test_cayu_app_invalid_binding_finalize_snapshot_is_reported_on_terminal_event():
    async def run():
        store = InMemorySessionStore()
        binding = RecordingWorkspaceBinding()
        binding.final_snapshot = object()  # type: ignore[assignment]
        provider = FakeProvider(
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), binding=binding),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_binding_invalid_finalize_snapshot",
                messages=[Message.text("user", "run")],
            ),
        )
        session = await store.load("sess_binding_invalid_finalize_snapshot")
        return events, session

    events, session = asyncio.run(run())

    assert session is not None
    assert session.status == SessionStatus.COMPLETED
    assert events[-2].type == EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED
    assert events[-2].payload["error"] == (
        "Workspace snapshot copies require a WorkspaceSnapshot or None."
    )
    assert events[-2].payload["error_type"] == "TypeError"
    assert events[-2].payload["final_revision"]["status"] == "unsupported"
    assert events[-1].type == EventType.SESSION_COMPLETED
    assert events[-1].payload["binding_finalize_error"] == {
        "error": "Workspace snapshot copies require a WorkspaceSnapshot or None.",
        "error_type": "TypeError",
        "outcome": "completed",
        "failures": [
            {
                "phase": "workspace_finalize",
                "error": "Workspace snapshot copies require a WorkspaceSnapshot or None.",
                "error_type": "TypeError",
            }
        ],
    }
    assert events[-1].payload["final_revision"] == events[-2].payload["final_revision"]


def test_cayu_app_binds_environment_for_approved_tool_continuation(tmp_path):
    async def run():
        store = InMemorySessionStore()
        configured_root = tmp_path / "configured"
        configured_root.mkdir()
        bound_root = tmp_path / "bound"
        bound_root.mkdir()
        configured_workspace = LocalWorkspace(configured_root, workspace_id="configured")
        bound_workspace = LocalWorkspace(bound_root, workspace_id="bound")
        binding = RecordingWorkspaceBinding(bound_workspace=bound_workspace)
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_1",
                        name="workspace_id",
                        arguments={},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                workspace=configured_workspace,
                binding=binding,
            ),
            default=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[WorkspaceIdTool()],
            tool_policy=RequireApprovalPolicy(),
        )

        first_events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                environment_name="local",
                session_id="sess_binding_approval",
                messages=[Message.text("user", "run tool")],
            ),
        )
        approval_event = next(
            event for event in first_events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        approval_id = approval_event.payload["approval"]["approval_id"]
        approved_events = await collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_binding_approval",
                approval_id=approval_id,
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id=approval_event.payload["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
        return first_events, approved_events, binding, bound_workspace

    first_events, approved_events, binding, bound_workspace = asyncio.run(run())

    assert first_events[0].type == EventType.ENVIRONMENT_BINDING_STARTED
    assert first_events[1].type == EventType.ENVIRONMENT_BINDING_COMPLETED
    assert first_events[-3].type == EventType.ENVIRONMENT_BINDING_FINALIZE_STARTED
    assert first_events[-2].type == EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED
    assert first_events[-1].type == EventType.SESSION_INTERRUPTED
    assert approved_events[0].type == EventType.SESSION_RESUMED
    assert approved_events[1].type == EventType.ENVIRONMENT_BINDING_STARTED
    assert approved_events[2].type == EventType.ENVIRONMENT_BINDING_COMPLETED
    assert approved_events[-3].type == EventType.ENVIRONMENT_BINDING_FINALIZE_STARTED
    assert approved_events[-2].type == EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED
    assert approved_events[-1].type == EventType.SESSION_COMPLETED
    assert [call["outcome"] for call in binding.finalize_calls] == [
        "interrupted",
        "completed",
    ]
    assert len(binding.bind_calls) == 2
    tool_events = [
        event for event in approved_events if event.type == EventType.TOOL_CALL_COMPLETED
    ]
    assert tool_events[0].payload["result"]["structured"] == {"workspace_id": bound_workspace.id}


def test_cayu_app_validates_tool_approval_retry_before_binding(tmp_path):
    async def run():
        store = InMemorySessionStore()
        configured_root = tmp_path / "configured"
        configured_root.mkdir()
        binding = RecordingWorkspaceBinding()
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_1",
                        name="workspace_id",
                        arguments={},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                workspace=LocalWorkspace(configured_root, workspace_id="configured"),
                binding=binding,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[WorkspaceIdTool()],
            tool_policy=RequireApprovalPolicy(),
        )

        first_events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_binding_invalid_approval_retry",
                messages=[Message.text("user", "run tool")],
            ),
        )
        approval_event = next(
            event for event in first_events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        public_approval_id = approval_event.payload["approval"]["approval_id"]
        private_approval = _private_record_for_public_event(
            await store.query_events(EventQuery(session_id="sess_binding_invalid_approval_retry")),
            approval_event,
        ).event
        private_approval_id = private_approval.payload["approval"]["approval_id"]
        bind_calls_after_interrupt = len(binding.bind_calls)
        finalize_calls_after_interrupt = len(binding.finalize_calls)

        await store.append_event(
            "sess_binding_invalid_approval_retry",
            Event(
                type=EventType.TOOL_CALL_APPROVAL_DENIED,
                session_id="sess_binding_invalid_approval_retry",
                agent_name="assistant",
                environment_name="local",
                tool_name="workspace_id",
                payload={
                    **tool_round_identity_payload(private_approval),
                    "approval_id": private_approval_id,
                    "tool_call_id": private_approval.payload["tool_call_id"],
                    "result": ToolResult(content="denied").model_dump(),
                },
            ),
        )

        retry_events = [
            event
            async for event in app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id="sess_binding_invalid_approval_retry",
                    approval_id=public_approval_id,
                    tool_round_id=approval_event.payload["tool_round_id"],
                    tool_call_id=approval_event.payload["tool_call_id"],
                    decision=ToolApprovalDecision.APPROVE,
                )
            )
        ]
        session = await store.load("sess_binding_invalid_approval_retry")
        return (
            retry_events,
            session,
            binding,
            bind_calls_after_interrupt,
            finalize_calls_after_interrupt,
        )

    (
        retry_events,
        session,
        binding,
        bind_calls_after_interrupt,
        finalize_calls_after_interrupt,
    ) = asyncio.run(run())

    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED
    assert [event.type for event in retry_events] == [
        EventType.INTERACTION_RESUMED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert "already denied" in retry_events[-1].payload["error"]
    assert len(binding.bind_calls) == bind_calls_after_interrupt
    assert len(binding.finalize_calls) == finalize_calls_after_interrupt


def test_cayu_app_validates_tool_approval_retry_before_factory_resolution(tmp_path):
    async def run():
        store = InMemorySessionStore()
        configured_root = tmp_path / "configured"
        configured_root.mkdir()
        factory = RecordingEnvironmentFactory(
            Environment(
                EnvironmentSpec(name="dynamic"),
                workspace=LocalWorkspace(configured_root, workspace_id="configured"),
            )
        )
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_1",
                        name="workspace_id",
                        arguments={},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[WorkspaceIdTool()],
            tool_policy=RequireApprovalPolicy(),
        )

        first_events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_factory_invalid_approval_retry",
                messages=[Message.text("user", "run tool")],
            ),
        )
        approval_event = next(
            event for event in first_events if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        public_approval_id = approval_event.payload["approval"]["approval_id"]
        private_approval = _private_record_for_public_event(
            await store.query_events(EventQuery(session_id="sess_factory_invalid_approval_retry")),
            approval_event,
        ).event
        private_approval_id = private_approval.payload["approval"]["approval_id"]
        factory_calls_after_interrupt = len(factory.requests)

        await store.append_event(
            "sess_factory_invalid_approval_retry",
            Event(
                type=EventType.TOOL_CALL_APPROVAL_DENIED,
                session_id="sess_factory_invalid_approval_retry",
                agent_name="assistant",
                environment_name="dynamic",
                tool_name="workspace_id",
                payload={
                    **tool_round_identity_payload(private_approval),
                    "approval_id": private_approval_id,
                    "tool_call_id": private_approval.payload["tool_call_id"],
                    "result": ToolResult(content="denied").model_dump(),
                },
            ),
        )

        retry_events = [
            event
            async for event in app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id="sess_factory_invalid_approval_retry",
                    approval_id=public_approval_id,
                    tool_round_id=approval_event.payload["tool_round_id"],
                    tool_call_id=approval_event.payload["tool_call_id"],
                    decision=ToolApprovalDecision.APPROVE,
                )
            )
        ]
        session = await store.load("sess_factory_invalid_approval_retry")
        return retry_events, session, factory, factory_calls_after_interrupt

    retry_events, session, factory, factory_calls_after_interrupt = asyncio.run(run())

    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED
    assert [event.type for event in retry_events] == [
        EventType.INTERACTION_RESUMED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert "already denied" in retry_events[-1].payload["error"]
    assert len(factory.requests) == factory_calls_after_interrupt


def test_cayu_app_recovery_resolves_factory_before_resume_events(tmp_path):
    async def run():
        store = FailingTerminalToolEventStore()
        workspace_root = tmp_path / "factory"
        workspace_root.mkdir()
        factory = RecordingEnvironmentFactory(
            Environment(
                EnvironmentSpec(name="dynamic"),
                workspace=LocalWorkspace(workspace_root, workspace_id="factory-workspace"),
            ),
            reconnect_metadata={"sandbox_id": "sandbox_recovery"},
        )
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_1",
                        name="side_effect",
                        arguments={"value": "secret"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        tool = SideEffectTool()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
            tool_policy=RequireApprovalPolicy(),
        )

        interrupt_events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                environment_name="dynamic",
                session_id="sess_factory_recovery",
                messages=[Message.text("user", "use the tool")],
            ),
        )
        approval_event = next(
            event
            for event in interrupt_events
            if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        approval_id = approval_event.payload["approval"]["approval_id"]

        retry_events = await collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_factory_recovery",
                approval_id=approval_id,
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id=approval_event.payload["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
        factory.reconnect_metadata = {"sandbox_id": "sandbox_recovery", "generation": 2}
        # The external result is already selected. Resume the original approval
        # instead of replacing durable evidence with an operator-supplied result.
        recovery_events = await collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_factory_recovery",
                approval_id=approval_id,
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id="call_1",
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
        assert tool.calls == [{"value": "secret"}]
        checkpoint = await store.load_checkpoint("sess_factory_recovery")
        return retry_events, recovery_events, factory, checkpoint

    retry_events, recovery_events, factory, checkpoint = asyncio.run(run())

    assert [event.type for event in retry_events] == [
        EventType.ENVIRONMENT_FACTORY_STARTED,
        EventType.ENVIRONMENT_FACTORY_COMPLETED,
        EventType.SESSION_RESUMED,
        EventType.TOOL_CALL_APPROVED,
        EventType.TOOL_CALL_STARTED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert [event.type for event in recovery_events[:3]] == [
        EventType.ENVIRONMENT_FACTORY_STARTED,
        EventType.ENVIRONMENT_FACTORY_COMPLETED,
        EventType.SESSION_RESUMED,
    ]
    assert EventType.TOOL_CALL_COMPLETED in [event.type for event in recovery_events]
    assert len(factory.requests) == 3
    assert [request.reconnect_metadata for request in factory.requests] == [
        {},
        {"sandbox_id": "sandbox_recovery"},
        {"sandbox_id": "sandbox_recovery"},
    ]
    assert checkpoint is not None
    assert checkpoint["environment_factory_reconnect"] == {
        "dynamic": {"sandbox_id": "sandbox_recovery", "generation": 2}
    }


def test_cayu_app_recovery_factory_failure_returns_to_interrupted_before_resume(tmp_path):
    async def run():
        store = FailingTerminalToolEventStore()
        workspace_root = tmp_path / "factory"
        workspace_root.mkdir()
        factory = RecordingEnvironmentFactory(
            Environment(
                EnvironmentSpec(name="dynamic"),
                workspace=LocalWorkspace(workspace_root, workspace_id="factory-workspace"),
            )
        )
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_1",
                        name="side_effect",
                        arguments={"value": "secret"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
            ]
        )
        tool = SideEffectTool()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
            tool_policy=RequireApprovalPolicy(),
        )

        interrupt_events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_factory_recovery_failure",
                messages=[Message.text("user", "use the tool")],
            ),
        )
        approval_event = next(
            event
            for event in interrupt_events
            if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        approval_id = approval_event.payload["approval"]["approval_id"]

        retry_events = await collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_factory_recovery_failure",
                approval_id=approval_id,
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id=approval_event.payload["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
        before_recovery = await store.load_checkpoint("sess_factory_recovery_failure")
        assert before_recovery is not None
        original_approval = before_recovery["pending_tool_approval"]
        factory.create_error = RuntimeError("factory failed\u0000workload-secret")
        recovery_events = await collect_tool_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_factory_recovery_failure",
                approval_id=approval_id,
                tool_round_id=approval_event.payload["tool_round_id"],
                tool_call_id="call_1",
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
        assert tool.calls == [{"value": "secret"}]
        session = await store.load("sess_factory_recovery_failure")
        after_recovery = await store.load_checkpoint("sess_factory_recovery_failure")
        assert after_recovery is not None
        retained_approval = after_recovery["pending_tool_approval"]
        assert retained_approval["approval_id"] == original_approval["approval_id"]
        assert retained_approval["tool_call_id"] == original_approval["tool_call_id"]
        return retry_events, recovery_events, session

    retry_events, recovery_events, session = asyncio.run(run())

    assert [event.type for event in retry_events] == [
        EventType.ENVIRONMENT_FACTORY_STARTED,
        EventType.ENVIRONMENT_FACTORY_COMPLETED,
        EventType.SESSION_RESUMED,
        EventType.TOOL_CALL_APPROVED,
        EventType.TOOL_CALL_STARTED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert [event.type for event in recovery_events] == [
        EventType.ENVIRONMENT_FACTORY_STARTED,
        EventType.ENVIRONMENT_FACTORY_FAILED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert recovery_events[-1].payload["error"] == (
        "Operation failed with a non-portable diagnostic."
    )
    assert recovery_events[-1].payload["error_type"] == "RuntimeError"
    assert recovery_events[-1].payload["durable_value_error_code"] == "nul_character"
    assert recovery_events[-1].payload["durable_value_error_path"] == "$"
    assert "workload-secret" not in json.dumps(
        [event.model_dump(mode="json") for event in recovery_events]
    )
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED
