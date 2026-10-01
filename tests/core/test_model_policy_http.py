import asyncio

import httpx
import pytest
from tests.core.test_model_policy_runtime import Channel

from cayu.model_policy import HttpPolicyChannel, PolicyResponseError, PolicyTransportUnavailable
from cayu.runtime._policy_wire import canonical, decode

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def channel(monkeypatch, authority, handler):
    monkeypatch.setenv("TEST_POLICY_CREDENTIAL", "private-policy-credential-canary")
    result = HttpPolicyChannel(
        origin="https://cloud.test",
        scope=authority.scope,
        incarnation=authority.incarnation,
        credential_env="TEST_POLICY_CREDENTIAL",
    )
    result._client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return result


async def test_live_transport_correlates_snapshot_and_refusal(monkeypatch):
    authority = Channel()
    paths = []

    async def handler(request):
        assert request.headers["authorization"] == "Bearer private-policy-credential-canary"
        paths.append(request.url.path)
        if request.method == "GET":
            body = await authority.read_snapshot()
        else:
            body = await authority.report(request.content)
        return httpx.Response(200, content=body, headers={"content-type": "application/json"})

    transport = channel(monkeypatch, authority, handler)
    try:
        snapshot = decode(await transport.read_snapshot())
        report = canonical(
            {
                "schema_version": 1,
                "kind": "adoption_refusal",
                "scope": authority.scope,
                "incarnation_id": "incarnation-1",
                "incarnation_epoch": 1,
                "operation_id": "policy-1",
                "decision_seq": 1,
                "target": "application_default",
                "snapshot": {
                    "snapshot_id": snapshot["snapshot_id"],
                    "effective_revision": 1,
                    "config_sha256": snapshot["config_sha256"],
                },
                "reason": "model_unknown",
                "observed_at": snapshot["issued_at"],
            }
        )
        assert decode(await transport.report(report))["report"] == decode(report)
        assert paths[-1].endswith("/refusals")
    finally:
        await transport.aclose()


async def test_error_and_reflected_credential_never_escape(monkeypatch, caplog, capsys):
    authority = Channel()
    responses = [
        httpx.Response(
            403,
            json={
                "schema_version": 1,
                "kind": "policy_error",
                "request_id": "r1",
                "code": "forbidden",
                "retryable": False,
            },
        ),
        httpx.Response(500, text="private-policy-credential-canary"),
    ]
    transport = channel(monkeypatch, authority, lambda request: responses.pop(0))
    try:
        with pytest.raises(PolicyResponseError) as error:
            await transport.read_snapshot()
        assert error.value.code == "forbidden"
        with pytest.raises(PolicyTransportUnavailable) as failure:
            await transport.read_snapshot()
        assert "private-policy" not in str(failure.value)
        captured = capsys.readouterr()
        assert "private-policy" not in captured.out + captured.err + caplog.text
    finally:
        await transport.aclose()


async def test_caller_cancel_during_stream_is_preserved(monkeypatch):
    entered = asyncio.Event()

    class Pending(httpx.AsyncByteStream):
        async def __aiter__(self):
            entered.set()
            await asyncio.Event().wait()
            yield b"{}"

    transport = channel(
        monkeypatch,
        Channel(),
        lambda request: httpx.Response(
            200, headers={"content-type": "application/json"}, stream=Pending()
        ),
    )
    task = asyncio.create_task(transport.read_snapshot())
    try:
        await asyncio.wait_for(entered.wait(), 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled() and task.cancelling() == 1
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await transport.aclose()
