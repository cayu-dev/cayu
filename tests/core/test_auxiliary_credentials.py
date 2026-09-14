from __future__ import annotations

import asyncio
import base64
import json
import warnings

import pytest
from pydantic import SecretStr

from cayu import (
    AgentSpec,
    AuxiliaryInferencePolicy,
    CayuApp,
    ChatCompletionsProvider,
    EventType,
    InferenceLimits,
    Message,
    ModelRequest,
    RetryPolicy,
    RunRequest,
    Tool,
    ToolResult,
    ToolSpec,
)
from cayu.providers import ModelProviderError
from cayu.runtime import (
    InMemoryEventSink,
    InMemorySessionStore,
    PublicAuthorityAliasCodec,
    PublicAuthorityAliasKeyring,
)
from cayu.runtime.evidence import RuntimeEvidenceRequest, runtime_evidence
from cayu.storage import SQLiteSessionStore
from cayu.vaults import SecretRedactor


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("outcome", ["success", "error"])
@pytest.mark.parametrize("redact_attribution", [False, True])
def test_managed_inference_keeps_model_credentials_out_of_tool_and_evidence(
    sqlite_resources, caplog, capsys, backend, outcome, redact_attribution
):
    credential = "auxiliary-model-key-canary-0123456789"
    header_credential = "auxiliary-model-header-canary-0123456789"
    limits = InferenceLimits(max_input_tokens=10, max_output_tokens=10, timeout_seconds=10)
    observed = []

    class Transport:
        def __init__(self):
            self.outer = 0
            self.nested = 0

        async def stream_chat_completions(self, *, headers, payload, **kwargs):
            # Positive control: model credentials reached only their transport.
            assert headers["Authorization"] == f"Bearer {credential}"
            assert headers["x-model-auth"] == header_credential
            if payload.get("max_completion_tokens") == 10:
                self.nested += 1
                assert not payload.get("tools")
                if outcome == "error":
                    raise RuntimeError(f"transport rejected {dict(headers)}")
                delta = {"content": "summary"}
                finish = "stop"
            else:
                self.outer += 1
                delta = (
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "parent",
                                "type": "function",
                                "function": {"name": "summarize", "arguments": "{}"},
                            }
                        ]
                    }
                    if self.outer == 1
                    else {"content": "done"}
                )
                finish = "tool_calls" if self.outer == 1 else "stop"
            yield {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
            yield {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}
            yield {
                "choices": [],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            }

    class Summarize(Tool):
        spec = ToolSpec(
            name="summarize",
            description="Managed call without workload credentials",
            input_schema={"type": "object", "properties": {}},
            auxiliary_inference=AuxiliaryInferencePolicy(limits=limits, purposes=("tool.summary",)),
        )

        async def run(self, ctx, args):
            assert ctx.vault is None
            assert ctx.inference is not None
            for field in ("client", "api_key", "provider", "registry", "app"):
                assert not hasattr(ctx.inference, field)
            assert credential not in ctx.model_dump_json()
            assert header_credential not in ctx.model_dump_json()
            try:
                response = await ctx.inference.invoke(
                    ModelRequest(model="test-model", messages=[Message.text("user", "summarize")]),
                    purpose="tool.summary",
                    limits=limits,
                )
            except ModelProviderError as exc:
                observed.append(str(exc) + repr(vars(exc)))
                assert outcome == "error"
                return ToolResult(content="Managed model failed", is_error=True)
            assert outcome == "success"
            observed.append(response.model_dump_json())
            return ToolResult(content=response.text)

    async def scenario():
        path = sqlite_resources.path("credentials.sqlite")
        keyring = PublicAuthorityAliasKeyring(
            active_key_id="test",
            keys={
                "test": SecretStr(
                    base64.urlsafe_b64encode(bytes(range(32))).decode("ascii").rstrip("=")
                )
            },
        )
        codec = PublicAuthorityAliasCodec(keyring)
        store = (
            sqlite_resources.own(SQLiteSessionStore(path, public_authority_alias_codec=codec))
            if backend == "sqlite"
            else InMemorySessionStore(public_authority_alias_codec=codec)
        )
        sink = InMemoryEventSink()
        transport = Transport()
        redactor = (
            SecretRedactor(("aux_", "auxiliary_inference", "tool_call_id"))
            if redact_attribution
            else SecretRedactor()
        )
        app = CayuApp(
            session_store=store,
            event_sinks=[sink],
            enable_logging=True,
            secret_redactor=redactor,
            public_authority_alias_keyring=keyring,
        )
        app.register_provider(
            ChatCompletionsProvider(
                api_key=credential,
                extra_headers={"x-model-auth": header_credential},
                transport=transport,
            ),
            default=True,
        )
        assert app._secret_redactor.has_values is redact_attribution
        assert app._secret_redactor.redact_text(credential) == credential
        app.register_agent(AgentSpec(name="assistant", model="test-model"), tools=[Summarize()])
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="credential-boundary",
                    messages=[Message.text("user", "go")],
                    retry_policy=RetryPolicy(max_attempts=1),
                )
            )
        ]
        assert transport.nested == 1
        assert transport.outer == 2
        assert len(observed) == 1
        assert app._secret_redactor.has_values is redact_attribution
        assert app._secret_redactor.redact_text(credential) == credential
        stored = await store.load_events("credential-boundary")
        terminal = [e for e in stored if e.type == EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED]
        assert len(terminal) == 1
        assert terminal[0].payload["auxiliary_outcome"] == (
            "completed" if outcome == "success" else "failed"
        )
        for event in events + sink.events + stored:
            observed.append(event.model_dump_json())
        observed.append(json.dumps(await store.load_checkpoint("credential-boundary")))
        request = RuntimeEvidenceRequest(
            root_session_id="credential-boundary", max_sessions=10, max_events=1000
        )
        report = await runtime_evidence(app, request)
        observed.append(report.model_dump_json())
        if backend == "sqlite":
            await store.close()
            reopened = sqlite_resources.own(
                SQLiteSessionStore(path, public_authority_alias_codec=codec)
            )
            try:
                reconstructed = await runtime_evidence(
                    CayuApp(session_store=reopened, enable_logging=False), request
                )
                assert reconstructed == report
                observed.append(reconstructed.model_dump_json())
            finally:
                await reopened.close()

    async def run():
        async with sqlite_resources:
            await scenario()

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        asyncio.run(run())
    output = capsys.readouterr()
    surfaces = "\n".join(observed) + caplog.text + output.out + output.err + repr(captured)
    assert credential not in surfaces
    assert header_credential not in surfaces
