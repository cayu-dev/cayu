"""Failure-path observability is useful AND private; no live API calls."""

import asyncio
import json

import httpx
import pytest
from tests.core.test_openai_subscription_provider import StaticSubscriptionAuth

from cayu import (
    AgentSpec,
    CayuApp,
    CheckpointCompactionContextPolicy,
    EventType,
    Message,
    ModelCompactor,
    RunRequest,
    SQLiteSessionStore,
)
from cayu.providers import (
    HttpxOpenAITransport,
    ModelRequest,
    OpenAIAPIError,
    OpenAIProvider,
    OpenAISubscriptionProvider,
    capture_provider_errors,
)
from cayu.providers.diagnostics import _record_stream_error
from cayu.vaults import SecretRedactor

ERROR = {
    "type": "invalid_request_error",
    "code": "invalid_prompt",
    "message": "Unsupported input item: private-customer-value subscription-access",
    "param": "input[2].content[0]",
}


class Body(httpx.AsyncByteStream):
    def __init__(self, data):
        self.data = data

    async def __aiter__(self):
        yield self.data


@pytest.mark.parametrize("subscription", [False, True])
@pytest.mark.parametrize("kind", ["http", "nested", "flat", "response.failed", "transport"])
def test_real_transport_captures_details_without_changing_public_failures(subscription, kind):
    records = []

    async def run(enabled):
        calls = []

        def handler(request):
            calls.append(request)
            if kind == "transport":
                raise httpx.ConnectError("https://secret-url subscription-access", request=request)
            if kind == "http":
                return httpx.Response(
                    400, json={"error": ERROR}, headers={"x-request-id": "req_test"}
                )
            event = (
                {"type": "error", "error": ERROR}
                if kind == "nested"
                else {"type": "error", **{k: v for k, v in ERROR.items() if k != "type"}}
                if kind == "flat"
                else {"type": "response.failed", "response": {"error": ERROR}}
            )
            return httpx.Response(
                200,
                headers={"x-request-id": "req_test", "x-secret-header": "MUST_NOT_CAPTURE"},
                stream=Body(f"data: {json.dumps(event)}\n\n".encode()),
            )

        transport = HttpxOpenAITransport()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            transport._client._client = client
            provider = (
                OpenAISubscriptionProvider(auth=StaticSubscriptionAuth(), transport=transport)
                if subscription
                else OpenAIProvider(api_key="subscription-access", transport=transport)
            )
            request = ModelRequest(model="test-model", messages=[Message.text("user", "NO_PROMPT")])
            if enabled:
                with capture_provider_errors(
                    records.append, redactor=SecretRedactor("private-customer-value")
                ) as capture:
                    result = [event.payload async for event in provider.stream(request)]
                assert capture.records_written == 1
                assert capture.sink_failures == 0
            else:
                result = [event.payload async for event in provider.stream(request)]
            assert len(calls) == 1
            assert "x-client-request-id" not in calls[0].headers
            return result

    before = asyncio.run(run(False))
    assert not records
    after = asyncio.run(run(True))
    assert before == after
    serialized = json.dumps(records)
    for secret in (
        "private-customer-value",
        "subscription-access",
        "MUST_NOT_CAPTURE",
        "NO_PROMPT",
    ):
        assert secret not in serialized
    assert "Unsupported input" not in json.dumps(after)
    record = records[0]
    if kind == "transport":
        assert record["error"]["type"] == "ConnectError"
        assert record["field_states"]["message"] == "absent"
        assert "secret-url" not in serialized
    else:
        assert record["error"]["message"].startswith("Unsupported input item: ")
        assert record["field_states"]["message"] == "redacted"
        assert record["error"]["code"] == "invalid_prompt"
        assert record["error"]["param"] == "input[2].content[0]"
        assert record["error"]["request_id"] == "req_test"
        assert record["request_id_source"] == "header"
        assert record["http_status_code"] == (400 if kind == "http" else 200)


def test_bounded_fields_redact_before_truncation_and_report_omissions():
    records = []
    secret = "boundary-secret"
    with capture_provider_errors(records.append, redactor=SecretRedactor(secret)) as capture:
        capture.record_error(
            {
                "message": "a" * 4090 + secret + "b" * 200,
                "param": "z" * 65537,
                "code": {"not": "text"},
                "request_id": "req_" + secret,
                "status_code": True,
            },
            boundary="stream_error",
        )
    record = records[0]
    assert len(record["error"]["message"].encode()) <= 4096
    assert "boundary" not in record["error"]["message"]
    assert record["field_states"] == {
        "message": "redacted_truncated",
        "param": "omitted_oversize",
        "request_id": "redacted",
        "type": "absent",
        "code": "invalid_type",
    }
    assert "error_status_code" not in record


def test_malformed_unicode_cannot_reconstruct_a_secret_through_lossy_repair():
    records = []
    secret = "synthetic-password?"
    with capture_provider_errors(records.append, redactor=SecretRedactor(secret)) as capture:
        capture.record_error({"message": "synthetic-password\ud800"}, boundary="stream_error")
    assert records[0]["field_states"]["message"] == "invalid_unicode"
    assert "message" not in records[0]["error"]
    assert secret not in json.dumps(records)


def test_sink_failure_and_record_limit_are_explicit_without_exposing_sink_exception():
    def sink(record):
        raise OSError("secret sink path")

    with capture_provider_errors(sink, redactor=SecretRedactor()) as capture:
        for _ in range(34):
            capture.record_error(ERROR, boundary="stream_error")
    assert capture.records_attempted == capture.sink_failures == 32
    assert capture.records_written == 0
    assert capture.records_dropped == 2
    assert "secret sink path" not in repr(capture)


def test_nested_scopes_are_isolated_and_closed_scopes_do_not_capture():
    outer, inner = [], []
    event = {"type": "error", "message": "one"}
    with capture_provider_errors(outer.append, redactor=SecretRedactor()) as capture:
        _record_stream_error(event, headers={})
        with capture_provider_errors(inner.append, redactor=SecretRedactor()):
            _record_stream_error(event, headers={})
        _record_stream_error(event, headers={})
    _record_stream_error(event, headers={})
    capture.record_error(ERROR, boundary="stream_error")
    assert len(outer) == 2 and len(inner) == 1
    assert outer[0]["capture_id"] != inner[0]["capture_id"]


def test_parallel_requests_use_their_own_header_secret_registry():
    async def run():
        records = []

        async def call(secret):
            with capture_provider_errors(records.append, redactor=SecretRedactor()):
                await asyncio.sleep(0)
                _record_stream_error(
                    {"type": "error", "message": secret},
                    headers={"Authorization": f"Bearer {secret}"},
                )

        await asyncio.gather(call("first-secret"), call("second-secret"))
        return records

    records = asyncio.run(run())
    assert len({r["capture_id"] for r in records}) == 2
    assert "first-secret" not in json.dumps(records)
    assert "second-secret" not in json.dumps(records)


@pytest.mark.parametrize("identifier", ["arbitrary-customer-text", "../bad-path"])
def test_capture_identity_must_be_a_uuid(identifier):
    with (
        pytest.raises(ValueError),
        capture_provider_errors(
            lambda record: None, redactor=SecretRedactor(), capture_id=identifier
        ),
    ):
        pass


def test_async_sink_is_rejected_and_malformed_event_does_not_change_protocol_handling():
    async def sink(record):
        pass

    with (
        pytest.raises(TypeError, match="synchronous"),
        capture_provider_errors(sink, redactor=SecretRedactor()),
    ):
        pass
    records = []
    with capture_provider_errors(records.append, redactor=SecretRedactor()):
        _record_stream_error({"type": []}, headers={})
    assert not records


def test_compactor_failure_is_durable_and_private_details_remain_outside_events(tmp_path):
    records = []
    database = tmp_path / "sessions.sqlite3"

    async def run():
        def handler(request):
            event = {"type": "error", "error": ERROR}
            return httpx.Response(
                200,
                headers={"x-request-id": "req_compaction"},
                stream=Body(f"data: {json.dumps(event)}\n\n".encode()),
            )

        store = SQLiteSessionStore(database)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            transport = HttpxOpenAITransport()
            transport._client._client = client
            provider = OpenAISubscriptionProvider(
                auth=StaticSubscriptionAuth(), transport=transport
            )
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="test-model"),
                context_policy=CheckpointCompactionContextPolicy(
                    compactor=ModelCompactor(provider=provider, model="test-model"),
                    max_user_turns=1,
                    compact_after_messages=2,
                ),
            )
            try:
                with capture_provider_errors(
                    records.append, redactor=SecretRedactor("private-customer-value")
                ) as capture:
                    events = [
                        e
                        async for e in app.run(
                            RunRequest(
                                agent_name="assistant",
                                session_id="compaction-diagnostic",
                                messages=[
                                    Message.text("user", "Old request"),
                                    Message.text("assistant", "Old answer"),
                                    Message.text("user", "New request"),
                                ],
                            )
                        )
                    ]
                assert events[-1].type == EventType.SESSION_FAILED
                assert capture.records_written == 1
            finally:
                await store.close()

        # A fresh connection proves that failure classification was persisted,
        # not merely visible in an in-process callback.
        reopened = SQLiteSessionStore(database)
        try:
            return await reopened.load_events("compaction-diagnostic")
        finally:
            await reopened.close()

    events = asyncio.run(run())
    completions = [e for e in events if e.type == EventType.MODEL_COMPLETED]
    assert len(completions) == 1
    assert completions[0].payload["purpose"] == "context_compaction"
    assert completions[0].payload["compaction_outcome"] == "provider_error"
    assert any(e.type == EventType.CONTEXT_COMPACTION_FAILED for e in events)
    serialized = json.dumps([e.payload for e in events])
    assert "Unsupported input item" not in serialized
    assert "req_compaction" not in serialized
    assert records[0]["error"]["request_id"] == "req_compaction"
    assert records[0]["error"]["param"] == "input[2].content[0]"
    assert records[0]["error"]["message"].startswith("Unsupported input item")


@pytest.mark.parametrize("content", [b"not-json", b"x" * 65537])
def test_unavailable_http_body_still_captures_status_and_request_id(content):
    records = []

    async def run():
        transport = HttpxOpenAITransport()
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    400,
                    headers={"content-type": "application/json", "x-request-id": "req_no_body"},
                    content=content,
                )
            )
        ) as client:
            transport._client._client = client
            with (
                capture_provider_errors(records.append, redactor=SecretRedactor()),
                pytest.raises(OpenAIAPIError),
            ):
                await transport.create_response(
                    url="https://provider.invalid/responses",
                    headers={},
                    payload={},
                    timeout_s=5,
                )

    asyncio.run(run())
    assert records[0]["http_status_code"] == 400
    assert records[0]["error"]["request_id"] == "req_no_body"
    assert records[0]["body_state"] == "unavailable"
