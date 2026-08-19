from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import ValidationError
from tests._session_provenance import fixture_session_invocation

from cayu.core import AgentSpec, Message, TextPart
from cayu.embeddings import (
    TextEmbedding,
    TextEmbeddingProvider,
    TextEmbeddingRequest,
    TextEmbeddingResult,
    TextEmbeddingUsage,
)
from cayu.evals import ScriptedModelProvider
from cayu.providers import (
    AnthropicProvider,
    CacheBreakpoint,
    CachePolicy,
    ModelCompletion,
    ModelFinishReason,
    ModelRequest,
    ModelStreamEvent,
    ModelStreamEventType,
)
from cayu.proxies import AllowlistProxy
from cayu.runners import ExecCommand
from cayu.runtime import (
    CompactionRequest,
    ContextRequest,
    DispatchRequest,
    ResumeRequest,
    RunLimits,
    RunRequest,
    Session,
)
from cayu.storage import (
    InMemoryEmbeddingKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeEntry,
)
from cayu.tools import CommandRequest
from cayu.vaults import SecretRef, StaticVault


@pytest.mark.parametrize("bad_text", ["bad\x00text", "bad\ud800text"])
@pytest.mark.parametrize("field", ["request_model", "request_text", "result_model"])
def test_embedding_identity_and_input_text_require_portable_text(
    field: str,
    bad_text: str,
) -> None:
    with pytest.raises(ValidationError):
        if field == "request_model":
            TextEmbeddingRequest(model=bad_text, texts=["portable"])
        elif field == "request_text":
            TextEmbeddingRequest(model="embedding-model", texts=["portable", bad_text])
        else:
            TextEmbeddingResult(
                model=bad_text,
                embeddings=[TextEmbedding(index=0, vector=[0.1])],
            )


def test_embedding_result_detaches_multiple_vectors_and_usage_evidence() -> None:
    first_vector = [0.1, 0.2]
    second_vector = [0.3, 0.4]
    first = TextEmbedding(index=0, vector=first_vector)
    second = TextEmbedding(index=1, vector=second_vector)
    usage = TextEmbeddingUsage(
        input_tokens=7,
        total_tokens=7,
        metadata={"provider": {"request_id": "request-1"}},
    )

    result = TextEmbeddingResult(
        model="embedding-model",
        embeddings=[first, second],
        usage=usage,
    )
    original_dump = result.model_dump(mode="json")

    first_vector[0] = 8.8
    second_vector[0] = 8.8
    first.vector[0] = 9.9
    second.index = 9
    usage.input_tokens = 99
    usage.metadata["provider"]["request_id"] = "mutated"

    assert result.model_dump(mode="json") == original_dump
    assert result.embeddings[0] is not first
    assert result.embeddings[0].vector is not first.vector
    assert result.embeddings[1] is not second
    assert result.usage is not usage


def test_embedding_contract_preserves_ordinary_unicode_and_vector_order() -> None:
    request = TextEmbeddingRequest(
        model="埋め込み-модель",
        texts=["こんにちは", "مرحبا", "emoji 🧭"],
    )
    result = TextEmbeddingResult(
        model=request.model,
        embeddings=[
            TextEmbedding(index=0, vector=[0.3, -0.1]),
            TextEmbedding(index=1, vector=[0.2, 0.4]),
        ],
        usage=TextEmbeddingUsage(input_tokens=3, total_tokens=3),
    )

    assert request.texts == ["こんにちは", "مرحبا", "emoji 🧭"]
    assert [embedding.index for embedding in result.embeddings] == [0, 1]
    assert [embedding.vector for embedding in result.embeddings] == [
        [0.3, -0.1],
        [0.2, 0.4],
    ]
    assert result.usage is not None
    assert result.usage.total_tokens == 3


async def _collect(
    provider: ScriptedModelProvider,
    request: ModelRequest,
) -> list[ModelStreamEvent]:
    return [event async for event in provider.stream(request)]


class _UnvalidatedEmbeddingProvider(TextEmbeddingProvider):
    name = "unvalidated"

    async def embed_texts(self, request: TextEmbeddingRequest) -> TextEmbeddingResult:
        return TextEmbeddingResult.model_construct(
            model=request.model,
            embeddings=[TextEmbedding.model_construct(index=0, vector=[float("nan")])],
        )


def test_embedding_store_revalidates_provider_results() -> None:
    provider = _UnvalidatedEmbeddingProvider()
    store = InMemoryEmbeddingKnowledgeStore(
        access_scope=KnowledgeAccessScope.privileged(),
        embedding_provider=provider,
        embedding_model="embedding-model",
        embedding_dimensions=1,
    )

    with pytest.raises(ValidationError, match="finite"):
        asyncio.run(store.create_entry(KnowledgeEntry(id="entry-1", text="portable text")))


def _session() -> Session:
    return Session(
        id="session-1",
        agent_name="agent",
        provider_name="provider",
        model="model",
        causal_budget_id="budget-1",
        invocation=fixture_session_invocation("session-1"),
        metadata={"nested": {"value": "original"}},
    )


def _agent() -> AgentSpec:
    return AgentSpec(
        name="agent",
        model="model",
        metadata={"nested": {"value": "original"}},
    )


def _completion() -> ModelCompletion:
    return ModelCompletion(
        finish_reason=ModelFinishReason.STOP,
        raw_finish_reason="end_turn",
        status="completed",
        end_turn=True,
    )


def _event_from_completion(completion: ModelCompletion) -> ModelStreamEvent:
    return ModelStreamEvent(
        type=ModelStreamEventType.COMPLETED,
        payload={"finish_reason": "stop"},
        completion=completion,
    )


def _completion_status(event: ModelStreamEvent) -> str | None:
    return None if event.completion is None else event.completion.status


def _anthropic_config() -> tuple[CachePolicy, SecretRef, AllowlistProxy]:
    return (
        CachePolicy(
            breakpoints=(CacheBreakpoint.SYSTEM_PROMPT,),
            ttl="standard",
        ),
        SecretRef(
            name="anthropic-key",
            handle="vault://anthropic-key",
            metadata={"scope": {"tenant": "alpha"}},
        ),
        AllowlistProxy(
            vault=StaticVault({"anthropic-key": "secret-value"}),
            allowed_destinations=["api.anthropic.com"],
        ),
    )


def _anthropic_provider(
    config: tuple[CachePolicy, SecretRef, AllowlistProxy],
) -> AnthropicProvider:
    policy, ref, proxy = config
    return AnthropicProvider(
        api_key_ref=ref,
        credential_proxy=proxy,
        cache_policy=policy,
    )


def _mutate_anthropic_config(
    config: tuple[CachePolicy, SecretRef, AllowlistProxy],
) -> None:
    policy, ref, _ = config
    policy.breakpoints = (CacheBreakpoint.TOOL_DEFINITIONS,)
    ref.name = "mutated"
    ref.metadata["scope"]["tenant"] = "mutated"


def _observe_anthropic_provider(provider: AnthropicProvider) -> object:
    assert provider.cache_policy is not None
    assert provider.api_key_ref is not None
    assert isinstance(provider.credential_proxy, AllowlistProxy)
    return (
        provider.cache_policy.breakpoints,
        provider.api_key_ref.name,
        provider.api_key_ref.metadata,
        provider.credential_proxy.allowed_destinations,
    )


def _script_event() -> ModelStreamEvent:
    return ModelStreamEvent(
        type=ModelStreamEventType.COMPLETED,
        payload={"finish_reason": "stop"},
        completion=_completion(),
    )


def _mutate_script_event(event: ModelStreamEvent) -> None:
    event.payload["finish_reason"] = "mutated"
    assert event.completion is not None
    event.completion.status = "mutated"


def _observe_script(provider: ScriptedModelProvider) -> object:
    streamed = asyncio.run(
        _collect(
            provider,
            ModelRequest(model="model", messages=[Message.text("user", "request")]),
        )
    )
    completion = streamed[0].completion
    return streamed[0].payload, None if completion is None else completion.status


def _model_request() -> ModelRequest:
    return ModelRequest(
        model="model",
        messages=[Message.text("user", "original")],
        options={"nested": {"value": "original"}},
    )


def _provider_with_captured_request(request: ModelRequest) -> ScriptedModelProvider:
    provider = ScriptedModelProvider([ModelStreamEvent.completed({"finish_reason": "stop"})])
    asyncio.run(_collect(provider, request))
    return provider


def _mutate_model_request(request: ModelRequest) -> None:
    request.model = "mutated"
    request.messages[0] = Message.text("user", "mutated")
    request.options["nested"]["value"] = "mutated"


def _observe_captured_request(provider: ScriptedModelProvider) -> object:
    request = provider.requests[0]
    part = request.messages[0].content[0]
    assert isinstance(part, TextPart)
    return request.model, part.text, request.options


@dataclass(frozen=True)
class _InvocationDetachmentCase:
    name: str
    source: Callable[[], Any]
    construct: Callable[[Any], Any]
    mutate: Callable[[Any], None]
    observe: Callable[[Any], Any]
    expected: Any


_INVOCATION_DETACHMENT_CASES = (
    _InvocationDetachmentCase(
        name="model_stream_event_completion",
        source=_completion,
        construct=_event_from_completion,
        mutate=lambda value: setattr(value, "status", "mutated"),
        observe=_completion_status,
        expected="completed",
    ),
    _InvocationDetachmentCase(
        name="anthropic_configuration",
        source=_anthropic_config,
        construct=_anthropic_provider,
        mutate=_mutate_anthropic_config,
        observe=_observe_anthropic_provider,
        expected=(
            (CacheBreakpoint.SYSTEM_PROMPT,),
            "anthropic-key",
            {"scope": {"tenant": "alpha"}},
            ("api.anthropic.com",),
        ),
    ),
    _InvocationDetachmentCase(
        name="scripted_event_batches",
        source=_script_event,
        construct=lambda value: ScriptedModelProvider([value]),
        mutate=_mutate_script_event,
        observe=_observe_script,
        expected=({"finish_reason": "stop"}, "completed"),
    ),
    _InvocationDetachmentCase(
        name="scripted_request_history",
        source=_model_request,
        construct=_provider_with_captured_request,
        mutate=_mutate_model_request,
        observe=_observe_captured_request,
        expected=("model", "original", {"nested": {"value": "original"}}),
    ),
    _InvocationDetachmentCase(
        name="command",
        source=lambda: ExecCommand.process("git", "status"),
        construct=lambda value: CommandRequest(command=value, timeout_s=60),
        mutate=lambda value: value.argv.__setitem__(1, "mutated"),
        observe=lambda request: request.command.argv,
        expected=["git", "status"],
    ),
    _InvocationDetachmentCase(
        name="dispatch_limits",
        source=lambda: RunLimits(max_total_tokens=10),
        construct=lambda value: DispatchRequest(
            session_id="session-1",
            messages=[Message.text("user", "hello")],
            limits=value,
        ),
        mutate=lambda value: setattr(value, "max_total_tokens", 99),
        observe=lambda request: request.limits.max_total_tokens,
        expected=10,
    ),
    _InvocationDetachmentCase(
        name="run_limits",
        source=lambda: RunLimits(max_total_tokens=10),
        construct=lambda value: RunRequest(
            agent_name="agent",
            messages=[Message.text("user", "hello")],
            limits=value,
        ),
        mutate=lambda value: setattr(value, "max_total_tokens", 99),
        observe=lambda request: request.limits.max_total_tokens,
        expected=10,
    ),
    _InvocationDetachmentCase(
        name="resume_limits",
        source=lambda: RunLimits(max_total_tokens=10),
        construct=lambda value: ResumeRequest(
            session_id="session-1",
            messages=[Message.text("user", "hello")],
            limits=value,
        ),
        mutate=lambda value: setattr(value, "max_total_tokens", 99),
        observe=lambda request: request.limits.max_total_tokens,
        expected=10,
    ),
    _InvocationDetachmentCase(
        name="context_session",
        source=_session,
        construct=lambda value: ContextRequest(
            session=value,
            agent=_agent(),
            messages=[Message.text("user", "hello")],
            step=1,
        ),
        mutate=lambda value: value.metadata["nested"].__setitem__("value", "mutated"),
        observe=lambda request: request.session.metadata,
        expected={"nested": {"value": "original"}},
    ),
    _InvocationDetachmentCase(
        name="context_agent",
        source=_agent,
        construct=lambda value: ContextRequest(
            session=_session(),
            agent=value,
            messages=[Message.text("user", "hello")],
            step=1,
        ),
        mutate=lambda value: value.metadata["nested"].__setitem__("value", "mutated"),
        observe=lambda request: request.agent.metadata,
        expected={"nested": {"value": "original"}},
    ),
    _InvocationDetachmentCase(
        name="compaction_session",
        source=_session,
        construct=lambda value: CompactionRequest(
            session=value,
            agent=_agent(),
            messages=[Message.text("user", "hello")],
        ),
        mutate=lambda value: value.metadata["nested"].__setitem__("value", "mutated"),
        observe=lambda request: request.session.metadata,
        expected={"nested": {"value": "original"}},
    ),
    _InvocationDetachmentCase(
        name="compaction_agent",
        source=_agent,
        construct=lambda value: CompactionRequest(
            session=_session(),
            agent=value,
            messages=[Message.text("user", "hello")],
        ),
        mutate=lambda value: value.metadata["nested"].__setitem__("value", "mutated"),
        observe=lambda request: request.agent.metadata,
        expected={"nested": {"value": "original"}},
    ),
)


@pytest.mark.parametrize(
    "case",
    _INVOCATION_DETACHMENT_CASES,
    ids=lambda case: case.name,
)
def test_invocation_boundaries_detach_nested_contracts(
    case: _InvocationDetachmentCase,
) -> None:
    source = case.source()
    request = case.construct(source)

    case.mutate(source)

    assert case.observe(request) == case.expected


@dataclass(frozen=True)
class _InvocationRevalidationCase:
    name: str
    source: Callable[[], Any]
    corrupt: Callable[[Any], None]
    construct: Callable[[Any], Any]


def _corrupt_anthropic_policy(
    config: tuple[CachePolicy, SecretRef, AllowlistProxy],
) -> None:
    policy, _, _ = config
    policy.conversation_prefix_n = 0


def _corrupt_session_or_agent(value: Session | AgentSpec) -> None:
    value.metadata["nested"]["value"] = float("nan")


_INVOCATION_REVALIDATION_CASES = (
    _InvocationRevalidationCase(
        name="model_stream_event_completion",
        source=_completion,
        corrupt=lambda value: setattr(value, "status", "bad\x00status"),
        construct=_event_from_completion,
    ),
    _InvocationRevalidationCase(
        name="anthropic_configuration",
        source=_anthropic_config,
        corrupt=_corrupt_anthropic_policy,
        construct=_anthropic_provider,
    ),
    _InvocationRevalidationCase(
        name="scripted_event_batches",
        source=_script_event,
        corrupt=lambda value: setattr(value.completion, "status", "bad\x00status"),
        construct=lambda value: ScriptedModelProvider([value]),
    ),
    _InvocationRevalidationCase(
        name="scripted_request_history",
        source=_model_request,
        corrupt=lambda value: setattr(value, "model", "bad\x00model"),
        construct=_provider_with_captured_request,
    ),
    _InvocationRevalidationCase(
        name="command",
        source=lambda: ExecCommand.process("git", "status"),
        corrupt=lambda value: value.argv.clear(),
        construct=lambda value: CommandRequest(command=value, timeout_s=60),
    ),
    *(
        _InvocationRevalidationCase(
            name=name,
            source=lambda: RunLimits(max_total_tokens=10),
            corrupt=lambda value: setattr(value, "max_total_tokens", 0),
            construct=construct,
        )
        for name, construct in (
            (
                "dispatch_limits",
                lambda value: DispatchRequest(
                    session_id="session-1",
                    messages=[Message.text("user", "hello")],
                    limits=value,
                ),
            ),
            (
                "run_limits",
                lambda value: RunRequest(
                    agent_name="agent",
                    messages=[Message.text("user", "hello")],
                    limits=value,
                ),
            ),
            (
                "resume_limits",
                lambda value: ResumeRequest(
                    session_id="session-1",
                    messages=[Message.text("user", "hello")],
                    limits=value,
                ),
            ),
        )
    ),
    *(
        _InvocationRevalidationCase(
            name=name,
            source=source,
            corrupt=_corrupt_session_or_agent,
            construct=construct,
        )
        for name, source, construct in (
            (
                "context_session",
                _session,
                lambda value: ContextRequest(
                    session=value,
                    agent=_agent(),
                    messages=[Message.text("user", "hello")],
                    step=1,
                ),
            ),
            (
                "context_agent",
                _agent,
                lambda value: ContextRequest(
                    session=_session(),
                    agent=value,
                    messages=[Message.text("user", "hello")],
                    step=1,
                ),
            ),
            (
                "compaction_session",
                _session,
                lambda value: CompactionRequest(
                    session=value,
                    agent=_agent(),
                    messages=[Message.text("user", "hello")],
                ),
            ),
            (
                "compaction_agent",
                _agent,
                lambda value: CompactionRequest(
                    session=_session(),
                    agent=value,
                    messages=[Message.text("user", "hello")],
                ),
            ),
        )
    ),
)


@pytest.mark.parametrize(
    "case",
    _INVOCATION_REVALIDATION_CASES,
    ids=lambda case: case.name,
)
def test_invocation_boundaries_revalidate_nested_contracts(
    case: _InvocationRevalidationCase,
) -> None:
    source = case.source()
    case.corrupt(source)

    with pytest.raises(ValidationError):
        case.construct(source)
