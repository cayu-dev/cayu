from __future__ import annotations

import asyncio
import warnings
from datetime import UTC, datetime

import pytest
from pydantic import SecretStr
from tests.core.test_mcp import FakeMcpSession, _fake_server_spec, _fake_tool_definitions
from tests.core.test_tool_effect_reconciliation_registration import _spec
from tests.core.test_tool_effect_runtime_dispatch import _ObservingSQLiteStore, _ObservingStore
from tests.core.test_tool_round_execution_identities import _SequencedProvider

from cayu import AgentSpec, CayuApp, ExecutionProfileBehaviorIdentity, Message, RunRequest
from cayu.mcp.tools import McpToolAdapter, McpToolset
from cayu.providers import ModelStreamEvent
from cayu.runtime._tool_effect_state import ToolEffectRecord
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.runtime.tool_effects import (
    ToolEffectReceipt,
    ToolEffectReconciliationRegistration,
    ToolEffectReconciliationRequest,
    ToolEffectReconciliationResult,
)
from cayu.vaults import REDACTED_SECRET, SecretRedactor


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("reject_redacted_result", [False, True])
def test_public_mcp_receipt_redaction_and_reconstruction(
    backend, reject_redacted_result, tmp_path, caplog, capsys
):
    secret = "mcp-session-only-credential-canary"

    async def scenario():
        tool_name = "mcp__local-mcp__echo"
        calls = []
        lookups = []
        codec = PublicAuthorityAliasCodec(
            PublicAuthorityAliasKeyring(active_key_id="test", keys={"test": SecretStr("A" * 43)})
        )
        path = str(tmp_path / "mcp-receipts.db")
        store = (
            _ObservingStore(public_authority_alias_codec=codec)
            if backend == "memory"
            else _ObservingSQLiteStore(path, public_authority_alias_codec=codec)
        )

        class Session(FakeMcpSession):
            @property
            def secret_redactor(self):
                # The transport-owned registry is deliberately absent from the app.
                return SecretRedactor(secret)

            async def call_tool(self, name, arguments):
                calls.append((name, arguments))
                raise RuntimeError("external acknowledgement lost")

        class Provider(_SequencedProvider):
            @property
            def execution_profile_identity(self):
                return ExecutionProfileBehaviorIdentity(
                    name="tests:mcp-receipt-provider",
                    behavior_version="1",
                    implementation_version="1",
                )

        class Reconciler:
            async def reconcile(self, *, context, receipt):
                assert receipt is None
                lookups.append(context.idempotency_key)
                return ToolEffectReconciliationResult(
                    outcome="completed",
                    observation="sent",
                    resource_versions={"part-1": secret},
                    receipt=ToolEffectReceipt(
                        receipt_id="mcp-receipt",
                        receipt_schema="deployment",
                        receipt_schema_version=1,
                        tool_call_id=context.tool_call_id,
                        tool_name=context.tool_name,
                        idempotency_key=context.idempotency_key,
                        outcome="completed",
                        message=secret,
                        structured={"credential": secret},
                        integrity={"signature": secret},
                        resource_versions={"part-2": secret},
                        source="reconciler",
                        observed_at=datetime(2026, 9, 8, tzinfo=UTC),
                    ),
                )

        def build_app(*, initial=False):
            app = CayuApp(
                session_store=store, enable_logging=False, secret_redactor=SecretRedactor()
            )
            responses = [[ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({})]]
            if initial:
                responses.insert(
                    0,
                    [
                        ModelStreamEvent.tool_call(id="call", name=tool_name, arguments={}),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ],
                )
            app.register_provider(Provider(responses), default=True)
            definitions = _fake_tool_definitions("echo")
            toolset = McpToolset(
                server=_fake_server_spec().model_copy(update={"connection_id": "local-mcp"}),
                session=Session(definitions=definitions),
                definitions=definitions,
            )
            adapter = McpToolAdapter(toolset=toolset, definition=definitions[0])
            # Reconstructed opaque adapters need an application-declared identity.
            adapter.spec = adapter.spec.model_copy(
                update={
                    "execution_profile_identity": ExecutionProfileBehaviorIdentity(
                        name="tests:mcp-receipt-adapter",
                        behavior_version="1",
                        implementation_version="1",
                    )
                }
            )
            app.register_agent(
                AgentSpec(name="agent", model="test"),
                tools=[adapter],
                tool_effect_reconcilers={
                    tool_name: ToolEffectReconciliationRegistration(
                        reconciler=Reconciler(),
                        spec=_spec(
                            supports_lookup=True,
                            result_schema={
                                "type": "object",
                                "properties": {
                                    "credential": {"const": secret}
                                    if reject_redacted_result
                                    else {"type": "string"}
                                },
                                "additionalProperties": False,
                            },
                        ),
                    )
                },
            )
            return app

        try:
            app = build_app(initial=True)
            initial = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="mcp-receipt",
                        agent_name="agent",
                        messages=[Message.text("user", "go")],
                    )
                )
            ]
            assert initial[-1].type.value == "session.interrupted"
            key = store.effect_keys[0]

            async def load_record():
                return ToolEffectRecord.model_validate(
                    await store.load_session_operation("mcp-receipt", key)
                )

            assert (await load_record()).state == "outcome_unknown"
            started = next(event for event in initial if event.type.value == "tool.call.started")
            if backend == "sqlite":
                await store.close()
                store = _ObservingSQLiteStore(path, public_authority_alias_codec=codec)
            app = build_app()
            target = await app.inspect_tool_effect(
                "mcp-receipt",
                tool_round_id=started.payload["tool_round_id"],
                tool_call_id=started.payload["tool_call_id"],
            )
            request = ToolEffectReconciliationRequest(**target.model_dump(), lookup=True)
            if reject_redacted_result:
                with pytest.raises(ValueError, match="result violates its registered schema"):
                    _ = [event async for event in app.reconcile_tool_effect(request)]
                record = await load_record()
                assert record.state == "outcome_unknown"
                assert record.terminal is None
                events = await store.load_events("mcp-receipt")
                assert not any(
                    event.type.value == "tool.effect.receipt.validated" for event in events
                )
                assert secret not in "".join(event.model_dump_json() for event in events)
                assert len(calls) == len(lookups) == 1
                return

            output = [event async for event in app.reconcile_tool_effect(request)]
            record = await load_record()
            assert record.state == "reconciled_completed"
            assert record.terminal.receipt.message == REDACTED_SECRET
            assert record.terminal.receipt.structured == {"credential": REDACTED_SECRET}
            assert record.terminal.receipt.integrity == {"signature": REDACTED_SECRET}
            assert record.terminal.receipt.resource_versions == {
                "part-1": REDACTED_SECRET,
                "part-2": REDACTED_SECRET,
            }
            events = await store.load_events("mcp-receipt")
            if backend == "sqlite":
                await store.close()
                store = _ObservingSQLiteStore(path, public_authority_alias_codec=codec)
            replay = [event async for event in build_app().reconcile_tool_effect(request)]
            assert await load_record() == record
            assert await store.load_events("mcp-receipt") == events
            assert len(calls) == len(lookups) == 1
            for serialized in [
                record.model_dump_json(),
                *(event.model_dump_json() for event in events + output + replay),
            ]:
                assert secret not in serialized
            assert any(event.type.value == "tool.call.completed" for event in output)
            assert output[-1].type.value == "session.completed"
        finally:
            if backend == "sqlite":
                await store.close()

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        asyncio.run(scenario())
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err + caplog.text
    assert all(secret not in str(warning.message) for warning in captured_warnings)
