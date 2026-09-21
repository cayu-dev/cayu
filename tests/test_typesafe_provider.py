import asyncio
import json

import httpx
import pytest

from cayu import Message
from cayu.experimental.typesafe import TypeSafeProvider, validate_response
from cayu.providers import ModelRequest


def test_native_batch_and_governed_stream():
    questions = {
        "truth": {"type": "noul", "instructions": "True?"},
        "route": {
            "type": "choice",
            "instructions": "Where?",
            "criteria": {"a": "Alpha", "b": "Beta"},
        },
        "quality": {"type": "score", "instructions": "Quality?", "criteria": ["low", "high"]},
    }
    body = {
        "model": "jev-test",
        "answers": {
            "truth": {"type": "noul", "noul": 0},
            "route": {
                "type": "choice",
                "choice": "b",
                "probabilities": {"a": 0.2, "b": 0.8},
                "confidence": 0.6,
            },
            "quality": {
                "type": "score",
                "score": 1,
                "probabilities": {"0": 0, "1": 1},
                "confidence": 1,
            },
        },
        "usage": {"input_tokens": 20, "output_tokens": 2},
    }

    def handler(request):
        assert request.url == "https://api.typesafe.ai/v1/systemone"
        assert request.headers["Authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        assert payload["questions"] == questions
        assert payload["state"] == "user: A statement"
        return httpx.Response(200, json=body)

    provider = TypeSafeProvider(api_key="test-key", transport=httpx.MockTransport(handler))

    async def run():
        return [
            e
            async for e in provider.runtime_stream(
                ModelRequest(
                    model="jev-latest",
                    messages=[Message.text("user", "A statement")],
                    options={"typesafe": {"questions": questions}},
                )
            )
        ]

    events = asyncio.run(run())
    assert json.loads(events[0].delta)["answers"]["truth"]["noul"] == 0
    assert str(events[-1].type) == "completed"
    assert events[-1].payload["usage"] == body["usage"]


@pytest.mark.parametrize("probability", [True, "0.5", -1, 1.1, float("nan"), float("inf")])
def test_rejects_invalid_probabilities(probability):
    with pytest.raises(ValueError):
        validate_response(
            {"model": "jev", "answers": {"truth": {"type": "noul", "noul": probability}}},
            {"truth": {"type": "noul", "instructions": "True?"}},
        )


def test_error_does_not_expose_credentials_or_body():
    provider = TypeSafeProvider(
        api_key="secret-key",
        transport=httpx.MockTransport(
            lambda r: httpx.Response(401, text="secret-key private error body")
        ),
    )

    async def run():
        return [
            e
            async for e in provider.runtime_stream(
                ModelRequest(
                    model="jev",
                    messages=[Message.text("user", "x")],
                    options={
                        "typesafe": {
                            "questions": {"truth": {"type": "noul", "instructions": "True?"}}
                        }
                    },
                )
            )
        ]

    events = asyncio.run(run())
    assert str(events[-1].type) == "error"
    assert "secret-key" not in str(events)
    assert "private error" not in str(events)


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        (httpx.ConnectError, "connection"),
        (httpx.ReadTimeout, "timeout"),
        (httpx.RemoteProtocolError, "connection"),
        (401, None),
        (429, "http_status"),
        (529, "http_status"),
        (200, None),
    ],
)
def test_sanitized_failures_preserve_retry_classification(failure, reason):
    from cayu.runtime._model_errors import model_provider_error_from_payload
    from cayu.runtime.retry_policy import RetryPolicy, classify_retryable_error

    def handler(request):
        if isinstance(failure, int):
            return httpx.Response(failure, text="secret-key private error body")
        raise failure("secret-key private transport detail", request=request)

    provider = TypeSafeProvider(api_key="secret-key", transport=httpx.MockTransport(handler))

    async def run():
        return [
            event
            async for event in provider.runtime_stream(
                ModelRequest(
                    model="jev",
                    messages=[Message.text("user", "x")],
                    options={
                        "typesafe": {
                            "questions": {"truth": {"type": "noul", "instructions": "True?"}}
                        }
                    },
                )
            )
        ]

    events = asyncio.run(run())
    assert len(events) == 1
    assert str(events[0].type) == "error"
    assert "secret-key" not in str(events)
    assert "private" not in str(events)
    error = model_provider_error_from_payload(events[0].payload, fallback_provider="typesafe")
    assert error is not None
    assert error.__context__ is None
    assert error.__cause__ is None
    actual, status = classify_retryable_error(
        policy=RetryPolicy(),
        error=str(error),
        status_code=error.status_code,
        retryable=error.retryable,
    )
    assert actual == reason
    assert status == (failure if isinstance(failure, int) and failure != 200 else None)
    if failure == 200:
        assert error.retryable is False


@pytest.mark.parametrize("native", [None, [], "questions", {}, {"questions": {}, "extra": True}])
def test_invalid_configuration_fails_before_dispatch(native):
    def handler(request):
        pytest.fail("Invalid configuration must not reach the network")

    provider = TypeSafeProvider(api_key="test-key", transport=httpx.MockTransport(handler))
    request = ModelRequest(model="jev", messages=[], options={"typesafe": native})
    with pytest.raises(ValueError, match="typesafe.questions"):
        provider.request_fingerprint_options(request)

    async def run():
        return [event async for event in provider.runtime_stream(request)]

    with pytest.raises(ValueError, match="typesafe.questions"):
        asyncio.run(run())


def test_decision_is_persisted_by_cayu(sqlite_resources):
    from cayu import AgentSpec, CayuApp, RunRequest, SQLiteSessionStore

    async def run():
        async with sqlite_resources as resources:
            store = resources.own(SQLiteSessionStore(resources.path()))
            provider = TypeSafeProvider(
                api_key="test-key",
                transport=httpx.MockTransport(
                    lambda r: httpx.Response(
                        200,
                        json={
                            "model": "jev-test",
                            "answers": {"truth": {"type": "noul", "noul": 0.2}},
                            "usage": {"input_tokens": 10, "output_tokens": 1},
                        },
                    )
                ),
            )
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(
                AgentSpec(
                    name="decision",
                    model="jev-latest",
                    provider_options={
                        "typesafe": {
                            "questions": {
                                "truth": {"type": "noul", "instructions": "True?"},
                            }
                        }
                    },
                )
            )
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="decision",
                        messages=[Message.text("user", "A statement")],
                    )
                )
            ]
            assert events[-1].type == "session.completed"
            assert any(event.type == "model.completed" for event in events)
            saved = await store.load_events(events[-1].session_id)
            deltas = [event.payload["delta"] for event in saved if event.type == "model.text.delta"]
            assert json.loads("".join(deltas))["answers"]["truth"]["noul"] == 0.2

    asyncio.run(run())
