"""Runtime prompt-cache affinity keys follow the fork lineage (#2289)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from typing import Any

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    ForkSessionRequest,
    Message,
    ResumeRequest,
    RunRequest,
)
from cayu.providers import OpenAIProvider
from cayu.runtime._cache_affinity import cache_affinity_key_for_lineage_root
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions.base import InMemorySessionStore

_MODEL = "gpt-affinity"


class _RecordingResponsesTransport:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    async def stream_response_events(
        self, *, payload: Mapping[str, Any], **_: Any
    ) -> AsyncIterator[Mapping[str, Any]]:
        self.payloads.append(dict(payload))
        yield {"type": "response.created", "response": {"id": "resp-affinity"}}
        yield {"type": "response.output_text.delta", "delta": "done"}
        yield {
            "type": "response.completed",
            "response": {
                "id": "resp-affinity",
                "model": _MODEL,
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "id": "msg-affinity",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "done", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 9, "output_tokens": 1, "total_tokens": 10},
            },
        }

    async def aclose(self) -> None:
        return None


class _RestartStableOpenAIProvider(OpenAIProvider):
    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="test.affinity-openai-provider",
            behavior_version="1",
            implementation_version="1",
        )


def _app(
    store: InMemorySessionStore,
    transport: _RecordingResponsesTransport,
    *,
    provider_options: dict[str, Any] | None = None,
) -> CayuApp:
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(
        _RestartStableOpenAIProvider(api_key="affinity-key", transport=transport),
        default=True,
    )
    app.register_agent(
        AgentSpec(
            name="assistant",
            model=_MODEL,
            system_prompt="Stable rules.",
            provider_options=provider_options or {},
        )
    )
    return app


async def _drain(stream) -> list:
    return [event async for event in stream]


async def _run(app: CayuApp, session_id: str, text: str, **request: Any) -> None:
    events = await _drain(
        app.run(
            RunRequest(
                agent_name=request.pop("agent_name", "assistant"),
                session_id=session_id,
                messages=[Message.text("user", text)],
                **request,
            )
        )
    )
    assert events[-1].type is EventType.SESSION_COMPLETED, events[-1]


async def _fork(app: CayuApp, source: str, destination: str, **request: Any) -> None:
    events = await _drain(
        app.fork_session(
            ForkSessionRequest(source_session_id=source, session_id=destination, **request)
        )
    )
    assert events[-1].type is EventType.SESSION_FORKED, events[-1]


async def _resume(app: CayuApp, session_id: str, text: str) -> None:
    events = await _drain(
        app.resume(ResumeRequest(session_id=session_id, messages=[Message.text("user", text)]))
    )
    assert events[-1].type is EventType.SESSION_COMPLETED, events[-1]


def test_forks_share_the_lineage_key_and_children_get_their_own() -> None:
    store = InMemorySessionStore()
    transport = _RecordingResponsesTransport()
    app = _app(store, transport)
    source_key = cache_affinity_key_for_lineage_root("affinity-source")

    async def scenario() -> None:
        await _run(app, "affinity-source", "Investigate the outage.")
        await _fork(app, "affinity-source", "affinity-branch-a")
        await _fork(app, "affinity-source", "affinity-branch-b")
        await _resume(app, "affinity-branch-a", "Try hypothesis one.")
        await _resume(app, "affinity-branch-b", "Try hypothesis one.")
        await _fork(app, "affinity-branch-a", "affinity-branch-a-1")
        await _resume(app, "affinity-branch-a-1", "Go deeper.")
        await _run(
            app,
            "affinity-delegated-child",
            "Check one log file.",
            parent_session_id="affinity-source",
        )

    asyncio.run(scenario())

    source, branch_a, branch_b, fork_of_fork, child = transport.payloads
    for payload in (source, branch_a, branch_b, fork_of_fork):
        assert payload["prompt_cache_key"] == source_key
    # Sibling forks send byte-identical requests, so they share the cached prefix.
    assert branch_a == branch_b
    assert fork_of_fork["input"][: len(branch_a["input"])] == branch_a["input"]
    child_key = cache_affinity_key_for_lineage_root("affinity-delegated-child")
    assert child["prompt_cache_key"] == child_key != source_key
    # The wire key is a digest, never a raw session identifier.
    assert all("affinity-" not in payload["prompt_cache_key"] for payload in transport.payloads)
    assert source_key.startswith("cayu-") and len(source_key) <= 64


def test_lineage_key_is_rebuilt_from_durable_state_after_restart() -> None:
    store = InMemorySessionStore()
    transport = _RecordingResponsesTransport()
    first = _app(store, transport)

    async def before_restart() -> None:
        await _run(first, "restart-source", "Start.")
        await _fork(first, "restart-source", "restart-fork")
        await _fork(first, "restart-fork", "restart-fork-of-fork")

    asyncio.run(before_restart())
    # A fresh app has a fresh runtime with no in-process key cache.
    restarted = _app(store, transport)
    asyncio.run(_resume(restarted, "restart-fork-of-fork", "Continue."))

    expected = cache_affinity_key_for_lineage_root("restart-source")
    assert [payload["prompt_cache_key"] for payload in transport.payloads] == [expected] * 2


def test_caller_prompt_cache_key_overrides_the_lineage_key() -> None:
    transport = _RecordingResponsesTransport()
    app = _app(
        InMemorySessionStore(),
        transport,
        provider_options={"openai": {"prompt_cache_key": "tenant-a-agent"}},
    )

    asyncio.run(_run(app, "override-source", "Start."))

    assert [payload["prompt_cache_key"] for payload in transport.payloads] == ["tenant-a-agent"]
