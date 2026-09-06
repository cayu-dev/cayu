from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    Message,
    ModelProvider,
    ModelStreamEvent,
    RetryPolicy,
    RetrySuppression,
    RunRequest,
    SQLiteSessionStore,
    retry_decision,
)
from cayu.providers.base import ModelProviderError
from cayu.runtime._model_errors import copy_provider_exception_control
from cayu.runtime._model_step_executor import (
    ModelAttemptFailed,
    _attempt_retry_suppression,
    _typed_retry_fields,
)


@pytest.mark.parametrize(
    "fields,attempt,expected,retry,effective",
    [
        ({"status_code": 500}, 1, "retry_scheduled", True, 10),
        ({"status_code": 500}, 10, "configured_attempt_exhaustion", False, 10),
        ({"retryable": False}, 1, "explicit_nonretryable", False, 1),
        ({"retryable": True}, 1, "retry_scheduled", True, 10),
        ({"unknown_provider_error": True}, 1, "retry_scheduled", True, 2),
        ({"unknown_provider_error": True}, 2, "unknown_provider_attempt_cap", False, 2),
        ({}, 1, "classification_unavailable", False, 1),
        ({"error": "insufficient_quota"}, 1, "permanent_provider_error", False, 1),
        ({"status_code": 401}, 1, "policy_disallowed", False, 1),
        ({"error": "timeout"}, 1, "retry_scheduled", True, 10),
        ({"error": "connection reset"}, 1, "retry_scheduled", True, 10),
        ({"error": "rate limit"}, 1, "retry_scheduled", True, 10),
    ],
)
def test_retry_disposition(fields, attempt, expected, retry, effective):
    decision = retry_decision(
        policy=RetryPolicy(max_attempts=10, initial_delay_s=0.0),
        attempt=attempt,
        **{"error": "OpenAIAPIError: OpenAI provider failed", **fields},
    )
    assert decision.disposition.value == expected
    assert decision.retry is retry
    assert decision.max_attempts == 10
    assert decision.effective_max_attempts == effective
    assert decision.provider_retryable is fields.get("retryable")


@pytest.mark.parametrize("suppression", list(RetrySuppression))
@pytest.mark.parametrize("retryable", [None, False, True])
def test_suppression_preserves_tristate_and_budget(suppression, retryable):
    decision = retry_decision(
        policy=RetryPolicy(max_attempts=10),
        attempt=1,
        error="redacted",
        retryable=retryable,
        suppression=suppression,
    )
    assert not decision.retry
    assert decision.disposition.value == "suppressed"
    assert decision.suppression is suppression
    assert decision.provider_retryable is retryable
    assert decision.effective_max_attempts == 1
    assert decision.max_attempts == 10


@pytest.mark.parametrize("completion", [False, True])
def test_wrapping_preserves_runtime_authority_and_provider_classification(completion):
    original = ModelProviderError(
        "safe provider failure",
        provider="fake",
        retryable=True,
        response_body="secret-response-body",
    )
    control = copy_provider_exception_control(original)
    failure = ModelAttemptFailed(
        message=control.message,
        payload={},
        emitted_error_event=False,
        cause=control.cause,
        completion_observed=completion,
        automatic_retry_disabled=True,
    )
    status, retryable, delay, unknown = _typed_retry_fields(failure)
    decision = retry_decision(
        policy=RetryPolicy(max_attempts=10),
        attempt=1,
        error=failure.message,
        status_code=status,
        retryable=retryable,
        retry_after_s=delay,
        unknown_provider_error=unknown,
        suppression=_attempt_retry_suppression(failure),
    )
    assert not decision.retry
    assert decision.provider_retryable is True
    assert decision.suppression is (
        RetrySuppression.COMPLETION_OBSERVED
        if completion
        else RetrySuppression.AUTOMATIC_RETRY_DISABLED
    )
    assert control.cause.response_body is None
    assert "secret-response-body" not in decision.model_dump_json()


@pytest.mark.parametrize("stream_error", [False, True])
@pytest.mark.parametrize("retryable", [None, False, True])
def test_durable_retry_explanation_survives_fresh_process(tmp_path, stream_error, retryable):
    class FailingProvider(ModelProvider):
        name = "fake"
        calls = 0

        async def stream(self, request):
            self.calls += 1
            error = ModelProviderError(
                "OpenAIAPIError: OpenAI provider failed",
                provider="fake",
                retryable=retryable,
                response_body="secret-response-body",
            )
            if stream_error:
                event = ModelStreamEvent.error(str(error), cause=error)
                event.payload.update(
                    {
                        "retry": True,
                        "retry_disposition": "forged-upstream-authority",
                        "retry_suppression": "forged-upstream-authority",
                        "provider_retryable": "forged-upstream-authority",
                    }
                )
                yield event
            else:
                raise error

    database = tmp_path / "sessions.sqlite"
    provider = FailingProvider()
    app = CayuApp(session_store=SQLiteSessionStore(database), enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def run():
        return [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="retry-diagnostics",
                    messages=[Message.text("user", "hello")],
                    retry_policy=RetryPolicy(max_attempts=10, initial_delay_s=0.0),
                )
            )
        ]

    events = asyncio.run(run())
    errors = [event for event in events if event.type is EventType.MODEL_ERROR]
    retries = [event for event in events if event.type is EventType.MODEL_RETRY]
    assert len(errors) == provider.calls == {None: 2, False: 1, True: 10}[retryable]
    assert len(retries) == len(errors) - 1
    assert (
        errors[-1].payload["retry_disposition"]
        == {
            None: "unknown_provider_attempt_cap",
            False: "explicit_nonretryable",
            True: "configured_attempt_exhaustion",
        }[retryable]
    )
    keys = [
        "retry",
        "retry_disposition",
        "retry_suppression",
        "provider_retryable",
        "attempt",
        "max_attempts",
        "effective_max_attempts",
    ]
    for error, retry in zip(errors, retries, strict=False):
        assert {k: error.payload[k] for k in keys} == {k: retry.payload[k] for k in keys}
    script = """
import asyncio, json, sys
from cayu import SQLiteSessionStore, EventType
async def read():
    events = await SQLiteSessionStore(sys.argv[1]).load_events("retry-diagnostics")
    print(json.dumps([e.payload for e in events if e.type is EventType.MODEL_ERROR]))
asyncio.run(read())
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(database)],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )
    retained = json.loads(result.stdout)
    assert len(retained) == len(errors)
    for public, stored in zip(errors, retained, strict=True):
        assert {k: public.payload[k] for k in keys} == {k: stored[k] for k in keys}
    assert "secret-response-body" not in result.stdout
    assert "forged-upstream-authority" not in result.stdout


@pytest.mark.parametrize(
    "flag,message,reason",
    [
        ("retry_on_timeout", "timeout", "timeout"),
        ("retry_on_connection_error", "connection reset", "connection"),
        ("retry_on_rate_limit", "rate limit", "rate_limit"),
    ],
)
@pytest.mark.parametrize("enabled", [False, True])
def test_recognized_category_retains_policy_exclusion(flag, message, reason, enabled):
    decision = retry_decision(
        policy=RetryPolicy(max_attempts=10, **{flag: enabled}),
        attempt=1,
        error=message,
    )
    assert decision.disposition.value == ("retry_scheduled" if enabled else "policy_disallowed")
    assert decision.retry is enabled
    assert decision.reason == (reason if enabled else None)
    assert decision.provider_retryable is None
    assert decision.max_attempts == 10
    assert decision.effective_max_attempts == (10 if enabled else 1)


@pytest.mark.parametrize(
    "fields,expected,retry",
    [
        ({"retryable": False}, "explicit_nonretryable", False),
        ({"retryable": True}, "retry_scheduled", True),
        ({"status_code": 500}, "retry_scheduled", True),
        ({"error": "timeout connection reset"}, "retry_scheduled", True),
        ({"error": "timeout insufficient_quota"}, "permanent_provider_error", False),
        ({"suppression": RetrySuppression.COMPLETION_OBSERVED}, "suppressed", False),
        ({"unknown_provider_error": True}, "retry_scheduled", True),
    ],
)
def test_policy_exclusion_diagnostics_preserve_existing_retry_precedence(fields, expected, retry):
    decision = retry_decision(
        policy=RetryPolicy(max_attempts=10, retry_on_timeout=False),
        attempt=1,
        **{"error": "timeout", **fields},
    )
    assert decision.disposition.value == expected
    assert decision.retry is retry
    assert decision.provider_retryable is fields.get("retryable")
    if fields.get("unknown_provider_error"):
        assert decision.reason == "unknown_provider"
        assert decision.effective_max_attempts == 2
