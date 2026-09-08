"""Operator client transport bounds; real native handoff is a separate proof."""

import asyncio
import ssl

import httpx
import pytest
from pydantic import SecretStr

from cayu.evals.internal.browser_acceptance_operator import _request, perform_fixture_handoff


@pytest.mark.parametrize("kind", ["status", "oversized", "malformed", "array"])
def test_operator_error_response_never_formats_server_content(kind, caplog, capsys, recwarn):
    canary = "private-operator-response-canary"

    async def scenario():
        def response(request):
            return httpx.Response(
                403 if kind == "status" else 200,
                content=(
                    (canary * 4096).encode()
                    if kind == "oversized"
                    else b"[]"
                    if kind == "array"
                    else canary.encode()
                ),
            )

        async with httpx.AsyncClient(
            base_url="https://operator.test", transport=httpx.MockTransport(response)
        ) as client:
            with pytest.raises(RuntimeError) as failure:
                await _request(client, "GET", "/status")
            assert canary not in str(failure.value) + repr(failure.value)

    asyncio.run(scenario())
    captured = capsys.readouterr()
    assert canary not in captured.out + captured.err + caplog.text + str(list(recwarn))


@pytest.mark.parametrize(
    "change",
    [
        {"input_endpoint": "wss://other.test/api/browser-control/input"},
        {"input_endpoint": "wss://operator.test:8443/api/browser-control/input"},
        {"input_endpoint": "wss://operator.test/api/browser-control/viewer"},
        {"operator_origin": "http://operator.test"},
        {"operator_origin": "https://operator.test/path"},
        {"private_text": SecretStr("")},
        {"private_text": SecretStr("private\x00canary")},
        {"private_text": SecretStr("private\ud800canary")},
        {"private_text": SecretStr("x" * 4097)},
    ],
)
def test_invalid_operator_inputs_fail_before_http_activity(change):
    async def scenario():
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(500)

        async with httpx.AsyncClient(
            base_url="https://operator.test", transport=httpx.MockTransport(respond)
        ) as client:
            values = {
                "client": client,
                "input_endpoint": "wss://operator.test/api/browser-control/input",
                "tls": ssl.create_default_context(),
                "operator_origin": "https://operator.test",
                "session_id": "session",
                "browser_session_id": "bs_fixture",
                "private_text": SecretStr("private-fixture-input"),
                **change,
            }
            with pytest.raises(ValueError):
                await perform_fixture_handoff(**values)
            assert requests == []

    asyncio.run(scenario())
