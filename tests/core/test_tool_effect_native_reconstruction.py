from __future__ import annotations

import asyncio

import pytest
from tests.core.test_structured_commands import _AdmittedRunner, _profile
from tests.core.test_tool_effect_runtime_dispatch import _ObservingSQLiteStore, _ObservingStore
from tests.core.test_tool_round_execution_identities import _SequencedProvider

from cayu import (
    AgentSpec,
    ApplyPatchTool,
    CayuApp,
    ExecutionProfileBehaviorIdentity,
    Message,
    ResumeRequest,
    RunRequest,
)
from cayu.environments import Environment, EnvironmentSpec
from cayu.providers import ModelStreamEvent
from cayu.runners import ExecResult, Runner
from cayu.runtime._tool_effect_state import ToolEffectRecord
from cayu.runtime.checkpoints import WORKSPACE_OBSERVATIONS_CHECKPOINT_KEY
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.structured_commands import RunCommandTool
from cayu.workspaces import LocalWorkspace


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("cleanup_uncertain", [False, True], ids=["settled", "uncertain"])
def test_public_native_command_recovery_requires_unambiguous_result(
    tmp_path, monkeypatch, backend, cleanup_uncertain
):
    async def scenario(store):
        workspace_root = tmp_path / "command-workspace"
        workspace_root.mkdir()
        (workspace_root / "uv.lock").write_bytes(b"locked\n")
        profile = _profile()
        calls = []
        original_run = RunCommandTool.run

        async def lose_return(tool, context, arguments):
            result = await original_run(tool, context, arguments)
            calls.append(result)
            assert result.structured["status"] == (
                "ambiguous" if cleanup_uncertain else "succeeded"
            ), result.structured
            raise ConnectionError("lost command tool acknowledgement")

        monkeypatch.setattr(RunCommandTool, "run", lose_return)

        class CommandRunner(_AdmittedRunner, Runner):
            @property
            def execution_profile_identity(self):
                return ExecutionProfileBehaviorIdentity(
                    name="tests:native-command-runner",
                    behavior_version="1",
                    implementation_version="1",
                )

            def output_secret_values_present(self):
                return False

        runner = CommandRunner(
            profile,
            result=ExecResult(
                exit_code=0,
                stdout="finished",
                artifacts=[
                    {
                        "type": "cayu.runner_cleanup.v1",
                        "adapter": "docker",
                        "action": "kill_command",
                        "status": "failed",
                        "timeout_s": 1.0,
                    }
                ]
                if cleanup_uncertain
                else [],
            ),
        )

        class Provider(_SequencedProvider):
            @property
            def execution_profile_identity(self):
                return ExecutionProfileBehaviorIdentity(
                    name="tests:native-command-provider",
                    behavior_version="1",
                    implementation_version="1",
                )

        provider = Provider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="command-call",
                        name="run_command",
                        arguments={"selector": "focused-test", "args": ["tests/test_unit.py"]},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )

        def build_app():
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_environment(
                Environment(
                    EnvironmentSpec(
                        name="command-workspace",
                        execution_profile_identity=ExecutionProfileBehaviorIdentity(
                            name="tests:native-command-environment",
                            behavior_version="1",
                            implementation_version="1",
                        ),
                    ),
                    workspace=LocalWorkspace(workspace_root, workspace_id="command-workspace"),
                    runner=runner,
                ),
                default=True,
            )
            app.register_agent(
                AgentSpec(name="agent", model="test"),
                tools=[RunCommandTool(toolchain_profile=profile)],
            )
            return app

        app = build_app()
        initial = [
            event
            async for event in app.run(
                RunRequest(
                    session_id="native-command-recovery",
                    agent_name="agent",
                    messages=[Message.text("user", "run the test")],
                )
            )
        ]
        assert initial[-1].type.value == "session.interrupted"
        assert len(calls) == len(runner.commands) == 1
        assert calls[0].structured["status"] == (
            "ambiguous" if cleanup_uncertain else "succeeded"
        ), calls[0].structured
        key = store.effect_keys[0]
        before = ToolEffectRecord.model_validate(
            await store.load_session_operation("native-command-recovery", key)
        )
        assert before.state == "outcome_unknown"
        app = build_app()
        recovered = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id="native-command-recovery",
                    messages=[Message.text("user", "continue from durable evidence")],
                )
            )
        ]
        record = ToolEffectRecord.model_validate(
            await store.load_session_operation("native-command-recovery", key)
        )
        assert record.state == ("outcome_unknown" if cleanup_uncertain else "completed")
        assert recovered[-1].type.value == (
            "session.interrupted" if cleanup_uncertain else "session.completed"
        )
        assert len(provider.requests) == (1 if cleanup_uncertain else 2)
        assert len(calls) == len(runner.commands) == 1
        assert record.intent == before.intent
        assert (record.terminal is None) is cleanup_uncertain

    async def run():
        store = (
            _ObservingStore()
            if backend == "memory"
            else _ObservingSQLiteStore(str(tmp_path / "native-command.db"))
        )
        try:
            await scenario(store)
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("publication_fault", [None, "acknowledgement", "readback"])
def test_public_resume_recovers_real_patch_journal_without_second_mutation(
    tmp_path, monkeypatch, backend, publication_fault
):
    async def scenario(store):
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        calls = []
        journal_reads = []
        original_run = ApplyPatchTool.run
        original_reconcile = ApplyPatchTool.reconcile_durable_tool_call

        async def commit_then_lose_return(tool, context, arguments):
            calls.append(context.idempotency_key)
            result = await original_run(tool, context, arguments)
            assert not result.is_error, result.structured
            assert (workspace_root / "created.txt").read_text() == "one mutation\n"
            raise ConnectionError("lost acknowledgement after durable patch completion")

        monkeypatch.setattr(ApplyPatchTool, "run", commit_then_lose_return)

        async def observe_journal_recovery(tool, **kwargs):
            journal_reads.append(kwargs["idempotency_key"])
            return await original_reconcile(tool, **kwargs)

        monkeypatch.setattr(ApplyPatchTool, "reconcile_durable_tool_call", observe_journal_recovery)

        class Provider(_SequencedProvider):
            @property
            def execution_profile_identity(self):
                return ExecutionProfileBehaviorIdentity(
                    name="tests:native-patch-provider",
                    behavior_version="1",
                    implementation_version="1",
                )

        provider = Provider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="patch-call",
                        name="apply_patch",
                        arguments={
                            "operations": [
                                {
                                    "type": "create",
                                    "path": "created.txt",
                                    "content": "one mutation\n",
                                }
                            ]
                        },
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )

        def build_app():
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_environment(
                Environment(
                    EnvironmentSpec(
                        name="patch-workspace",
                        execution_profile_identity=ExecutionProfileBehaviorIdentity(
                            name="tests:native-patch-environment",
                            behavior_version="1",
                            implementation_version="1",
                        ),
                    ),
                    workspace=LocalWorkspace(workspace_root, workspace_id="patch-workspace"),
                ),
                default=True,
            )
            app.register_agent(AgentSpec(name="agent", model="test"), tools=[ApplyPatchTool()])
            return app

        app = build_app()
        initial = [
            event
            async for event in app.run(
                RunRequest(
                    session_id="native-patch-reconstruction",
                    agent_name="agent",
                    messages=[Message.text("user", "create the file")],
                )
            )
        ]
        assert initial[-1].type.value == "session.interrupted"
        assert len(calls) == 1
        key = store.effect_keys[0]
        before = ToolEffectRecord.model_validate(
            await store.load_session_operation("native-patch-reconstruction", key)
        )
        assert before.state == "outcome_unknown"
        assert (workspace_root / "created.txt").read_text() == "one mutation\n"
        original_publish = store.publish_session_operation
        original_load = store.load_session_operation
        injected = False
        fail_readback = False

        async def lose_terminal_acknowledgement(session_id, **kwargs):
            nonlocal injected, fail_readback
            result = await original_publish(session_id, **kwargs)
            if (
                publication_fault is not None
                and not injected
                and kwargs["idempotency_key"] == key
                and any(event.type.value == "tool.call.completed" for event in kwargs["events"])
            ):
                injected = True
                fail_readback = publication_fault == "readback"
                raise ConnectionError("native terminal committed; acknowledgement lost")
            return result

        async def fail_exact_readback(session_id, idempotency_key, *, checkpoint_root_guard=None):
            nonlocal fail_readback
            if fail_readback and idempotency_key == key:
                fail_readback = False
                raise OSError("native terminal exact readback unavailable")
            return await original_load(
                session_id, idempotency_key, checkpoint_root_guard=checkpoint_root_guard
            )

        monkeypatch.setattr(store, "publish_session_operation", lose_terminal_acknowledgement)
        monkeypatch.setattr(store, "load_session_operation", fail_exact_readback)
        app = build_app()
        resumed = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id="native-patch-reconstruction",
                    messages=[Message.text("user", "continue from the durable receipt")],
                )
            )
        ]
        if publication_fault == "readback":
            assert injected
            assert resumed[-1].type.value == "session.failed"
            evidence = resumed[-1].payload["failure_evidence"]
            assert evidence["classification"] == "failure"
            for name in ("ExceptionGroup", "ConnectionError", "OSError"):
                assert evidence["exception_types"].count(name) == 1
            assert len(calls) == 1
            assert len(provider.requests) == 1
            committed = ToolEffectRecord.model_validate(
                await original_load("native-patch-reconstruction", key)
            )
            assert committed.state == "completed"
            app = build_app()
            resumed = [
                event
                async for event in app.resume(
                    ResumeRequest(
                        session_id="native-patch-reconstruction",
                        messages=[Message.text("user", "continue the selected result")],
                    )
                )
            ]
        assert injected is (publication_fault is not None)
        assert resumed[-1].type.value == "session.completed", [
            (event.type.value, event.payload)
            for event in resumed
            if event.type.value in {"session.failed", "session.interrupted"}
        ]
        selected = ToolEffectRecord.model_validate(
            await store.load_session_operation("native-patch-reconstruction", key)
        )
        assert selected.state == "completed"
        assert selected.intent == before.intent
        assert len(calls) == 1
        assert len(provider.requests) == 2
        assert journal_reads == calls
        assert (workspace_root / "created.txt").read_text() == "one mutation\n"
        checkpoint = await store.load_checkpoint("native-patch-reconstruction")
        assert not (checkpoint or {}).get(WORKSPACE_OBSERVATIONS_CHECKPOINT_KEY)
        events = await store.load_events("native-patch-reconstruction")
        finalized = [e for e in events if e.type.value == "workspace.observation.finalized"]
        assert len(finalized) == 1
        terminals = [
            event
            for event in events
            if event.type.value in {"tool.call.completed", "tool.call.failed"}
        ]
        assert len(terminals) == 1
        assert terminals[0].id == selected.terminal.event_id
        assert terminals[0].payload["result"]["is_error"] is False
        assert events.index(finalized[0]) < events.index(terminals[0])

    async def run_backend():
        store = (
            _ObservingStore()
            if backend == "memory"
            else _ObservingSQLiteStore(str(tmp_path / "native-patch.db"))
        )
        try:
            await scenario(store)
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(run_backend())
