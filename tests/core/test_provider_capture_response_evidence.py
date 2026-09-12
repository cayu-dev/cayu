"""Private diagnostics retain received evidence without changing error policy."""

import asyncio
import json
from contextlib import nullcontext

import httpx
import pytest

from cayu.providers import (
    HttpxOpenAITransport,
    ModelStreamDeadlineError,
    OpenAIAPIError,
    capture_provider_errors,
)
from cayu.providers._sse import SseEventLimitError, SseEventTimeoutError
from cayu.providers.deadlines import (
    ProviderDeadlineKind,
    ProviderStreamDeadlineEvidence,
    ProviderStreamDeadlineExceeded,
)
from cayu.providers.diagnostics import _record_http_error, _record_stream_error
from cayu.vaults import SecretRedactor


def _failure(kind):
    if kind == "deadline":
        return ProviderStreamDeadlineExceeded(
            ProviderStreamDeadlineEvidence(
                deadline_kind=ProviderDeadlineKind.TRANSPORT_IDLE,
                configured_timeout_s=5,
                elapsed_s=5,
                last_progress_kind=None,
                last_progress_elapsed_s=None,
                last_progress_at=None,
            )
        )
    return {
        "connect": httpx.ConnectError,
        "read": httpx.ReadError,
        "timeout": SseEventTimeoutError,
        "limit": SseEventLimitError,
    }[kind]("synthetic private transport explanation")


@pytest.mark.parametrize(
    ("kind", "after_headers"),
    [
        ("connect", False),
        ("read", False),
        ("read", True),
        ("timeout", True),
        ("limit", True),
        ("deadline", False),
        ("deadline", True),
    ],
)
def test_transport_failure_retains_only_received_response_metadata(kind, after_headers):
    records = []

    async def run(enabled):
        calls = []

        class BrokenBody(httpx.AsyncByteStream):
            closed = False

            async def __aiter__(self):
                yield b'data: {"type":"response.created"}\n\n'
                raise _failure(kind)

            async def aclose(self):
                self.closed = True

        body = BrokenBody()

        def handler(request):
            calls.append(request)
            if not after_headers:
                raise _failure(kind)
            return httpx.Response(
                200,
                headers={
                    "x-request-id": "req_request-credential-canary",
                    "x-private-header": "must-not-capture",
                },
                stream=body,
            )

        transport = HttpxOpenAITransport()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            transport._client._client = client
            scope = (
                capture_provider_errors(records.append, redactor=SecretRedactor())
                if enabled
                else nullcontext()
            )
            expected_exception = ModelStreamDeadlineError if kind == "deadline" else OpenAIAPIError
            with scope, pytest.raises(expected_exception) as caught:
                async for _ in transport.stream_response_events(
                    url="https://provider.invalid/responses",
                    headers={"Authorization": "Bearer request-credential-canary"},
                    payload={},
                    timeout_s=10,
                    transport_idle_timeout_s=10,
                    protocol_idle_timeout_s=10,
                    semantic_progress_timeout_s=10,
                    absolute_stream_timeout_s=10,
                ):
                    pass
            assert len(calls) == 1
            if after_headers:
                assert body.closed
            failure = caught.value
            return (
                type(failure),
                getattr(failure, "status_code", None),
                getattr(failure, "retryable", None),
                getattr(failure, "error_type", None),
                getattr(failure, "evidence", None),
            )

    before = asyncio.run(run(False))
    assert not records
    assert asyncio.run(run(True)) == before
    assert len(records) == 1
    record = records[0]
    assert record["boundary"] == "transport_error"
    assert record["body_state"] == "unavailable"
    assert record["field_states"]["message"] == "absent"
    assert record["error_status_codes"] == {}
    assert record["error_status_conflict"] is False
    assert "error_status_code" not in record
    if after_headers:
        assert record["http_status_code"] == 200
        assert record["error"]["request_id"] == "req_[REDACTED_SECRET]"
        assert record["field_states"]["request_id"] == "redacted"
        assert record["request_id_source"] == "header"
    else:
        assert "http_status_code" not in record
        assert record["field_states"]["request_id"] == "absent"
        assert record["request_id_source"] == "unavailable"
    for private in ("request-credential-canary", "must-not-capture", "private transport"):
        assert private not in json.dumps(record)


@pytest.mark.parametrize("boundary", ["http", "error", "response.failed"])
@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"status_code": 429}, {"status_code": 429}),
        ({"status_code": 429, "error": {}}, {"status_code": 429}),
        ({"error": {"status_code": 400}}, {"error.status_code": 400}),
        (
            {"status_code": 429, "error": {"status_code": 429}},
            {"status_code": 429, "error.status_code": 429},
        ),
        (
            {"status_code": 503, "error": {"status_code": 400}},
            {"status_code": 503, "error.status_code": 400},
        ),
        ({"status_code": True, "error": {"status_code": "400"}}, {}),
        ({"status_code": 99, "error": {"status_code": 600}}, {}),
    ],
)
def test_envelope_statuses_keep_fixed_paths_and_do_not_confuse_http_status(
    boundary, body, expected
):
    records = []
    with capture_provider_errors(records.append, redactor=SecretRedactor()):
        if boundary == "http":
            response = httpx.Response(400, json=body)
            _record_http_error(response, headers={}, body_response=response)
        else:
            if boundary == "response.failed":
                event = {"type": boundary, "status_code": 429, "response": body}
                expected = {"status_code": 429, **{f"response.{k}": v for k, v in expected.items()}}
            else:
                event = {"type": boundary, **body}
            _record_stream_error(event, headers={}, response=httpx.Response(200))
    record = records[0]
    assert record["error_status_codes"] == expected
    statuses = set(expected.values())
    assert record["error_status_conflict"] is (len(statuses) > 1)
    if len(statuses) == 1:
        assert record["error_status_code"] == next(iter(statuses))
    else:
        assert "error_status_code" not in record
    assert record["http_status_code"] == (400 if boundary == "http" else 200)


def test_custom_status_fields_cannot_add_arbitrary_keys_or_values():
    records = []
    with capture_provider_errors(records.append, redactor=SecretRedactor()) as capture:
        capture.record_error(
            {},
            boundary="stream_error",
            status_fields={
                "status_code": 400,
                "error.status_code": True,
                "response.status_code": "private-text",
                "response.error.status_code": 600,
                "private-key": 401,
            },
        )
    assert records[0]["error_status_codes"] == {"status_code": 400}
    assert "private" not in json.dumps(records)


def test_custom_error_status_keeps_default_path():
    records = []
    with capture_provider_errors(records.append, redactor=SecretRedactor()) as capture:
        capture.record_error({"status_code": 400}, boundary="stream_error")
    assert records[0]["error_status_codes"] == {"status_code": 400}
    assert records[0]["error_status_code"] == 400
    assert records[0]["error_status_conflict"] is False
