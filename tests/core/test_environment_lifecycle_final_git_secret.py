"""Terminal Git evidence must respect invocation-local secret quarantine."""

import asyncio
import json
import shutil

import pytest
from tests.core.test_workspace_mutation_receipts import (
    _portable_environment_spec,
    _SingleToolProvider,
)
from tests.environments.test_docker_coding import _coding_product_test_binding

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.environments import Environment, EnvironmentFactory, EnvironmentFactoryResult
from cayu.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.vaults import SecretRef, StaticVault


class _WriteResolvedSecret(Tool):
    spec = ToolSpec(
        name="write_resolved_secret",
        workspace_mutation=True,
        parallel_safe=False,
        max_terminal_payload_bytes=1024,
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        assert ctx.workspace is not None
        if ctx.vault is None:
            value = "public-value"
        else:
            secret = await ctx.vault.resolve(SecretRef(name="token"))
            value = secret.value.get_secret_value()
        await ctx.workspace.write_bytes("code.py", ("token = '" + value + "'\n").encode())
        return ToolResult(content="written")


@pytest.mark.parametrize("dynamic_scope", [False, True], ids=["static-control", "vault-secret"])
def test_final_git_receipt_does_not_publish_invocation_scoped_secret(tmp_path, dynamic_scope):
    if shutil.which("git") is None:
        pytest.skip("git executable is required")
    source_root, target_root = tmp_path / "source", tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "code.py").write_text("token = None\n")
    runner, source, binding = _coding_product_test_binding(source_root, target_root)
    secret = "PRIVATE_FINAL_GIT_RECEIPT_TOKEN_CANARY"

    class Factory(EnvironmentFactory):
        @property
        def secret_resolution_scope(self):
            return "dynamic" if dynamic_scope else "static"

        async def create(self, request):
            async def release(action):
                await runner.close()

            return EnvironmentFactoryResult(
                environment=Environment(
                    _portable_environment_spec(request.environment_name),
                    workspace=source,
                    runner=runner,
                    binding=binding,
                    vault=StaticVault({"token": secret}) if dynamic_scope else None,
                ),
                reconnect_metadata={"allocation": "exact-test-container"},
                release=release,
            )

        async def recover_finalization_disposal(self, request, state):
            await runner.close()

    async def scenario():
        app = CayuApp(enable_logging=False)
        app.register_provider(
            _SingleToolProvider(tool_name="write_resolved_secret", arguments={}), default=True
        )
        app.register_environment_factory(
            _portable_environment_spec("coding"), Factory(), default=True
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"), tools=[_WriteResolvedSecret()]
        )
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="final-git-secret",
                    messages=[Message.text("user", "write the token")],
                )
            )
        ]
        assert events[-1].type is EventType.SESSION_COMPLETED, events[-1].payload
        completed = next(
            event
            for event in events
            if event.type is EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED
        )
        if not dynamic_scope:
            assert "public-value" in completed.payload["final_git_receipt"]["diff"]["content"]
            return
        assert completed.payload["final_snapshot"] is None
        assert (
            completed.payload["final_revision"]["detail_code"]
            == "final_revision_secret_scope_unavailable"
        )
        # The invocation's tool publication has already sealed and redacted
        # its secret scope. The finalization event must not reintroduce it.
        tool_events = [event for event in events if event.type is EventType.TOOL_CALL_COMPLETED]
        assert tool_events
        assert secret not in repr(tool_events)
        durable = await app.session_store.load_events("final-git-secret")
        serialized = json.dumps([event.model_dump(mode="json") for event in events + durable])
        assert secret not in serialized, (
            "Invocation-local secret escaped through final_git_receipt despite terminal quarantine",
            completed.payload.get("final_git_receipt"),
        )

    asyncio.run(scenario())
