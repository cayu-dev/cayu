"""Credential-free controls; passing fixtures do not establish upstream origin."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from cayu import Message
from cayu.providers.base import (
    ModelRequest,
    ModelStreamDeadlineError,
    ModelStreamEvent,
    ModelStreamEventType,
    _normalized_provider_progress_kind,
)
from cayu.providers.deadlines import (
    ProviderDeadlineKind,
    ProviderProgressKind,
    ProviderStreamDeadlineController,
    ProviderStreamDeadlines,
)
from cayu.providers.openai import HttpxOpenAITransport, OpenAIProvider
from cayu.runtime._model_errors import model_provider_error_from_payload


def frame(event):
    return b"data: " + json.dumps(event).encode() + b"\n\n"


@pytest.mark.anyio
@pytest.mark.parametrize("chunk_size", [1, 17, 65536])
@pytest.mark.parametrize("whitespace", [" ", "\n", "\t", "\u2003"])
@pytest.mark.parametrize("complete", [False, True])
async def test_http_adapter_preserves_whitespace_and_closes_locally(
    chunk_size, whitespace, complete
):
    prefix = '{"value":'
    expected = [prefix]
    received = []
    structural = {
        "transport_chunks": 0,
        "transport_bytes": 0,
        "text_events": 0,
        "text_bytes": 0,
        "source_text_events": 0,
        "source_text_bytes": 0,
    }

    class Stream(httpx.AsyncByteStream):
        closed = 0

        async def __aiter__(self):
            async def chunks(event):
                if event["type"] == "response.output_text.delta":
                    structural["source_text_events"] += 1
                    structural["source_text_bytes"] += len(event["delta"].encode())
                wire = frame(event)
                for offset in range(0, len(wire), chunk_size):
                    part = wire[offset : offset + chunk_size]
                    structural["transport_chunks"] += 1
                    structural["transport_bytes"] += len(part)
                    yield part

            async for part in chunks({"type": "response.output_text.delta", "delta": prefix}):
                yield part
            for _ in range(3 if complete else 100):
                await asyncio.sleep(0.01)
                delta = whitespace * 8
                expected.append(delta)
                async for part in chunks({"type": "response.output_text.delta", "delta": delta}):
                    yield part
            if complete:
                expected.append("1}")
                async for part in chunks({"type": "response.output_text.delta", "delta": "1}"}):
                    yield part
                async for part in chunks(
                    {
                        "type": "response.completed",
                        "response": {
                            "id": "synthetic",
                            "status": "completed",
                            "output": [],
                            "usage": {},
                        },
                    }
                ):
                    yield part

        async def aclose(self):
            self.closed += 1

    stream = Stream()
    transport = HttpxOpenAITransport()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=stream
            )
        )
    ) as client:
        transport._client._client = client
        provider = OpenAIProvider(
            api_key="synthetic",
            transport=transport,
            stream_deadlines=ProviderStreamDeadlines(
                # Completion controls verify byte preservation, including one-byte
                # HTTP chunks; only timeout controls need a short semantic budget.
                semantic_progress_timeout_s=5 if complete else 0.15,
                absolute_stream_timeout_s=10,
            ),
        )
        events = []
        try:
            async for event in provider.runtime_stream(
                ModelRequest(model="synthetic", messages=[Message.text("user", "fixture")])
            ):
                events.append(event)
                if event.type is ModelStreamEventType.TEXT_DELTA:
                    received.append(event.delta)
                    assert _normalized_provider_progress_kind(event) is (
                        None if event.delta.isspace() else ProviderProgressKind.CONTENT
                    )
                    structural["text_events"] += 1
                    structural["text_bytes"] += len(event.delta.encode())
                    # Exercise excluded consumer time in both completion and timeout controls.
                    await asyncio.sleep(0.02)
        except ModelStreamDeadlineError as exc:
            assert not complete
            events.append(ModelStreamEvent.error(str(exc), cause=exc))

    assert stream.closed == 1
    assert structural["text_events"] == len(received)
    assert structural["text_bytes"] == len("".join(received).encode())
    assert structural["transport_bytes"] > structural["text_bytes"]
    # Timeout may interrupt one frame; it cannot duplicate, transform, or reorder accepted deltas.
    assert received == expected[: len(received)]
    assert "".join(received).encode() == "".join(expected[: len(received)]).encode()
    if complete:
        assert received == expected
        assert structural["source_text_events"] == structural["text_events"]
        assert structural["source_text_bytes"] == structural["text_bytes"]
        assert events[-1].type is ModelStreamEventType.COMPLETED
        if whitespace != "\u2003":
            assert json.loads("".join(received)) == {"value": 1}
    else:
        assert events[-1].type is ModelStreamEventType.ERROR
        payload = events[-1].payload
        assert payload["provider_deadline_kind"] == "semantic_idle"
        assert payload["provider_semantic_idle_elapsed_s"] >= 0.15
        assert payload["provider_excluded_semantic_pause_s"] > 0
        assert payload["provider_effect_outcome"] == "unknown"
        assert payload["provider_recovery_disposition"] == "manual_settlement_required"
        restored = model_provider_error_from_payload(payload, fallback_provider="openai")
        assert (
            restored.error_payload_fields()["provider_semantic_idle_elapsed_s"]
            == payload["provider_semantic_idle_elapsed_s"]
        )


@pytest.mark.anyio
async def test_semantic_timing_separates_consumer_pause_and_resets(monkeypatch):
    loop = asyncio.get_running_loop()
    now = 100.0
    monkeypatch.setattr(loop, "time", lambda: now)
    controller = ProviderStreamDeadlineController(
        ProviderStreamDeadlines(semantic_progress_timeout_s=4)
    )
    try:
        controller.observe_semantic(ProviderProgressKind.CONTENT)
        now += 1
        pause = controller.idle_pause_started()
        now += 20
        controller.exclude_idle_pause(pause, kinds=[ProviderDeadlineKind.SEMANTIC_IDLE])
        now += 3
        controller.observe_text(" \n\t")
        evidence = controller.evidence([ProviderDeadlineKind.SEMANTIC_IDLE])
        assert evidence.elapsed_s == 24
        assert evidence.semantic_idle_elapsed_s == 4
        assert evidence.excluded_semantic_pause_s == 20
        assert evidence.last_progress_elapsed_s == 0
        controller.observe_semantic(ProviderProgressKind.CONTENT)
        evidence = controller.evidence([ProviderDeadlineKind.SEMANTIC_IDLE])
        assert evidence.semantic_idle_elapsed_s == 0
        assert evidence.excluded_semantic_pause_s == 0
    finally:
        controller.close()


@pytest.mark.anyio
async def test_semantic_timing_survives_sqlite_readback(tmp_path):
    from cayu import AgentSpec, CayuApp, EventType, RetryPolicy, RunRequest, SQLiteSessionStore
    from cayu.providers.base import ModelProvider

    class Provider(ModelProvider):
        name = "synthetic"

        @property
        def stream_deadlines(self):
            return ProviderStreamDeadlines(semantic_progress_timeout_s=0.1)

        async def stream(self, request):
            yield ModelStreamEvent.text_delta('{"value":')
            while True:
                await asyncio.sleep(0.01)
                yield ModelStreamEvent.text_delta(" \n\t")

    database = tmp_path / "timing.sqlite"
    app = CayuApp(session_store=SQLiteSessionStore(database), enable_logging=False)
    app.register_provider(Provider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="synthetic"))
    observed = []
    with pytest.raises(ModelStreamDeadlineError):
        async for event in app.run(
            RunRequest(
                session_id="timing",
                agent_name="assistant",
                messages=[Message.text("user", "fixture")],
                retry_policy=RetryPolicy(max_attempts=1),
            )
        ):
            observed.append(event)
    stored = await SQLiteSessionStore(database).load_events("timing")
    for collection in (observed, stored):
        errors = [event.payload for event in collection if event.type is EventType.MODEL_ERROR]
        assert errors
        payload = errors[-1]
        assert payload["provider_semantic_idle_elapsed_s"] >= 0.1
        assert payload["provider_excluded_semantic_pause_s"] >= 0
        assert payload["provider_effect_outcome"] == "unknown"
        assert payload["retryable"] is False


@pytest.mark.parametrize(
    "elapsed,pause",
    [(True, 0), (-1, 0), (float("inf"), 0), (0, float("nan")), (None, 0), (0, None)],
)
def test_invalid_semantic_timing_is_rejected(elapsed, pause):
    from cayu.providers.deadlines import ProviderStreamDeadlineEvidence

    with pytest.raises(ValueError):
        ProviderStreamDeadlineEvidence(
            deadline_kind=ProviderDeadlineKind.SEMANTIC_IDLE,
            configured_timeout_s=1,
            elapsed_s=1,
            last_progress_kind=None,
            last_progress_elapsed_s=None,
            last_progress_at=None,
            semantic_idle_elapsed_s=elapsed,
            excluded_semantic_pause_s=pause,
        )
