"""A reconstructed human wait cannot adopt changed credential authority."""

import asyncio

import pytest
from tests.core.test_execution_profiles import IdentityConfiguredEgressEnvironmentFactory
from tests.core.test_foreground_child_resolution_contention import _app, _ContendedTool
from tests.core.test_foreground_subagent_recovery import _identity, _Provider

from cayu import (
    EnvironmentSpec,
    ExecutionProfileMismatchError,
    InMemorySessionStore,
    Message,
    RunRequest,
    SessionQuery,
    SQLiteSessionStore,
    ToolApprovalDecision,
)
from cayu.providers import ModelStreamEvent
from cayu.runtime import ToolApprovalRequest, UserInputResponse


class _StaticCredentialIdentityFactory(IdentityConfiguredEgressEnvironmentFactory):
    # This identity-only fixture creates no vault or credential proxy. Declare
    # that fact so a foreground result need not retain unknown secret scope.
    @property
    def secret_resolution_scope(self):
        return "static"


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("action", ["approval", "input"])
def test_nested_resolution_rejects_changed_credential_generation(tmp_path, backend, action):
    async def scenario():
        path = tmp_path / "authority.sqlite"
        store = InMemorySessionStore() if backend == "memory" else SQLiteSessionStore(path)

        def build(provider, generation):
            tool = _ContendedTool()
            app = _app(store, provider, tool, action)
            factory = _StaticCredentialIdentityFactory(generation=generation, allow_post=False)
            app.register_environment_factory(
                EnvironmentSpec(
                    name="egress", execution_profile_identity=_identity("nested-credential-env")
                ),
                factory,
                default=True,
            )
            return app, factory, tool

        try:
            provider = _Provider(
                [
                    [
                        ModelStreamEvent.tool_call(
                            id="spawn",
                            name="subagent",
                            arguments={"agent": "child", "task": "work"},
                        ),
                        ModelStreamEvent.completed(),
                    ],
                    [
                        ModelStreamEvent.tool_call(
                            id="action",
                            name="ask_user" if action == "input" else "record",
                            arguments={"question": "Continue?"}
                            if action == "input"
                            else {"value": 7},
                        ),
                        ModelStreamEvent.completed(),
                    ],
                ]
            )
            app, _, _ = build(provider, 1)
            _ = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="parent",
                        agent_name="parent",
                        messages=[Message.text("user", "go")],
                    )
                )
            ]
            assert await app.drain_background_interruptions(timeout_s=10)
            child = (await store.list_sessions(SessionQuery(parent_session_id="parent"))).sessions[
                0
            ]
            events = await store.load_events(child.id)
            if action == "input":
                pending = next(
                    event for event in events if event.type == "session.awaiting_user_input"
                )
                request = UserInputResponse(
                    session_id=child.id, input_id=pending.payload["input_id"], answer="yes"
                )
            else:
                pending = next(
                    event for event in events if event.type == "tool.call.approval_requested"
                )
                request = ToolApprovalRequest(
                    session_id=child.id,
                    approval_id=pending.payload["approval"]["approval_id"],
                    tool_round_id=pending.payload["tool_round_id"],
                    tool_call_id=pending.payload["tool_call_id"],
                    decision=ToolApprovalDecision.APPROVE,
                )
            if backend == "sqlite":
                await store.close()
                store = SQLiteSessionStore(path)
            checkpoint = await store.load_checkpoint(child.id)
            transcript = await store.load_transcript(child.id)
            replacement_provider = _Provider([])
            changed, factory, tool = build(replacement_provider, 2)
            with pytest.raises(ExecutionProfileMismatchError):
                stream = (
                    changed.resolve_user_input(request)
                    if action == "input"
                    else changed.resolve_tool_approval(request)
                )
                _ = [event async for event in stream]
            assert replacement_provider.requests == [] and tool.values == []
            assert factory.create_calls == 0
            assert await store.load_checkpoint(child.id) == checkpoint
            assert await store.load_transcript(child.id) == transcript
            compatible_provider = _Provider(
                [
                    [ModelStreamEvent.text_delta("child complete"), ModelStreamEvent.completed()],
                    [ModelStreamEvent.text_delta("parent complete"), ModelStreamEvent.completed()],
                ]
            )
            compatible, _, tool = build(compatible_provider, 1)
            stream = (
                compatible.resolve_user_input(request)
                if action == "input"
                else compatible.resolve_tool_approval(request)
            )
            _ = [event async for event in stream]
            assert await compatible.drain_background_interruptions(timeout_s=10)
            if len(compatible_provider.requests) != 2:
                raise AssertionError(
                    str(
                        [
                            (event.type, event.payload)
                            for event in (await store.load_events("parent"))[-4:]
                        ]
                    )
                )
            assert tool.values == ([] if action == "input" else [7])
        finally:
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())
