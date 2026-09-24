from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from tests.core.test_runtime import _valid_png_bytes
from tests.core.test_tool_effect_reconciliation_registration import _spec
from tests.core.test_tool_effect_runtime_dispatch import _ObservingSQLiteStore, _ObservingStore
from tests.core.test_tool_round_execution_identities import _SequencedProvider

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.artifacts.attachments import RESOLVED_FILE_ATTACHMENTS_OPTION
from cayu.artifacts.local import LocalArtifactStore
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.runtime._tool_effect_state import ToolEffectRecord
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.tool_effects import (
    ToolEffectReceipt,
    ToolEffectReconciliationRegistration,
    ToolEffectReconciliationRequest,
    ToolEffectReconciliationResult,
)
from cayu.sessions.base import RunRequest
from cayu.tools.base import Tool, ToolEffect, ToolSpec


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("image_only", [False, True])
def test_partial_round_receipt_selection_does_not_replay_siblings(
    backend, reverse, image_only, tmp_path
):
    artifacts = []

    async def scenario():
        store = (
            _ObservingStore()
            if backend == "memory"
            else _ObservingSQLiteStore(str(tmp_path / "multi.db"))
        )
        calls, lookups = [], []
        both_entered = asyncio.Event()

        class External(Tool):
            spec = ToolSpec(
                name="record",
                effect=ToolEffect.EXTERNAL,
                execution_profile_identity=ExecutionProfileBehaviorIdentity(
                    name="tests:multicall-effect", behavior_version="1", implementation_version="1"
                ),
            )

            async def run(self, ctx, args):
                calls.append(ctx.idempotency_key)
                if len(calls) == 2:
                    both_entered.set()
                await asyncio.wait_for(both_entered.wait(), timeout=10)
                raise ConnectionError("lost external acknowledgement")

        class Reconciler:
            async def reconcile(self, *, context, receipt):
                assert receipt is None
                assert context.idempotency_key in calls
                lookups.append(context.tool_call_id)
                return ToolEffectReconciliationResult(
                    outcome="completed",
                    observation="sent",
                    receipt=ToolEffectReceipt(
                        receipt_id=f"receipt-{context.tool_call_id}",
                        receipt_schema="deployment",
                        receipt_schema_version=1,
                        tool_call_id=context.tool_call_id,
                        tool_name=context.tool_name,
                        idempotency_key=context.idempotency_key,
                        outcome="completed",
                        message="" if image_only else f"verified {context.tool_call_id}",
                        artifacts=artifacts,
                        source="reconciler",
                        observed_at=datetime(2026, 9, 9, tzinfo=UTC),
                    ),
                )

        class Provider(_SequencedProvider):
            @property
            def execution_profile_identity(self):
                return ExecutionProfileBehaviorIdentity(
                    name="tests:multicall-provider",
                    behavior_version="1",
                    implementation_version="1",
                )

        provider = Provider(
            [
                [
                    ModelStreamEvent.tool_call(id=name, name="record", arguments={})
                    for name in ("first", "second")
                ]
                + [ModelStreamEvent.completed({"finish_reason": "tool_calls"})],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )

        def build_app():
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_environment(
                Environment(
                    EnvironmentSpec(
                        name="local",
                        execution_profile_identity=ExecutionProfileBehaviorIdentity(
                            name="tests:receipt-artifacts",
                            behavior_version="1",
                            implementation_version="1",
                        ),
                    ),
                    artifact_store=LocalArtifactStore(tmp_path / "artifacts"),
                ),
                default=True,
            )
            app.register_provider(provider, default=True)
            app.register_agent(
                AgentSpec(name="agent", model="test"),
                tools=[External()],
                tool_effect_reconcilers={
                    "record": ToolEffectReconciliationRegistration(
                        reconciler=Reconciler(), spec=_spec(supports_lookup=True)
                    )
                },
            )
            return app

        async def records():
            result = {}
            for key in set(store.effect_keys):
                record = ToolEffectRecord.model_validate(
                    await store.load_session_operation("multi", key)
                )
                result[record.intent.tool_call_id] = record
            return result

        try:
            app = build_app()
            if image_only:
                part = await app.attach_file(
                    _valid_png_bytes(), filename="screenshot.png", kind="image", session_id="multi"
                )
                artifacts.append(part.attachment)
            initial = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="multi",
                        agent_name="agent",
                        messages=[Message.text("user", "run both")],
                    )
                )
            ]
            assert initial[-1].type.value == "session.interrupted"
            original = await records()
            assert len(original) == len(set(calls)) == 2
            assert {record.state for record in original.values()} == {"outcome_unknown"}
            order = sorted(original, reverse=reverse)
            requests = []
            for index, call_id in enumerate(order):
                app = build_app()
                current = (await records())[call_id]
                session = await store.load("multi")
                request = ToolEffectReconciliationRequest(
                    **{
                        name: getattr(current.intent, name)
                        for name in (
                            "session_id",
                            "session_instance_id",
                            "tool_round_id",
                            "tool_call_id",
                            "tool_name",
                            "idempotency_key",
                        )
                    },
                    expected_run_epoch=session.run_epoch,
                    expected_revision=current.revision,
                    lookup=True,
                )
                requests.append(request)
                events = [event async for event in app.reconcile_tool_effect(request)]
                current_records = await records()
                assert current_records[call_id].state == "reconciled_completed"
                assert current_records[call_id].intent == original[call_id].intent
                assert len(calls) == 2
                if index == 0:
                    assert events[-1].type.value == "session.interrupted"
                    assert current_records[order[1]].state == "outcome_unknown"
                    assert current_records[order[1]].intent == original[order[1]].intent
                    assert len(provider.requests) == 1
                    partial_replay = [event async for event in app.reconcile_tool_effect(request)]
                    assert partial_replay[-1].type.value == "session.interrupted"
                    assert lookups == [call_id]
                    assert len(calls) == 2
                    assert len(provider.requests) == 1
                    assert (await records())[order[1]].state == "outcome_unknown"
                else:
                    assert events[-1].type.value == "session.completed", events[-1].payload
                    assert len(provider.requests) == 2
                    if image_only:
                        resolved = provider.requests[-1].options[RESOLVED_FILE_ATTACHMENTS_OPTION]
                        assert set(resolved) == {artifacts[0]["artifact_id"]}
            before_events = await store.load_events("multi")
            for request in requests:
                assert [event async for event in app.reconcile_tool_effect(request)]
            assert await store.load_events("multi") == before_events
            assert lookups == order
            assert len(calls) == 2
            assert len(provider.requests) == 2
            terminals = [
                event for event in before_events if event.type.value == "tool.call.completed"
            ]
            assert len(terminals) == 2
            for event in terminals:
                assert event.payload["result"]["artifacts"] == artifacts
                if image_only:
                    assert event.payload["result"]["content"] == ""
            transcript = await store.load_transcript("multi")
            tool_results = [
                block
                for message in transcript
                for block in message.content
                if block.type == "tool_result"
            ]
            assert len(tool_results) == 2
            for result in tool_results:
                assert result.model_dump(mode="json")["artifacts"] == artifacts
                if image_only:
                    assert result.content == ""
        finally:
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())
