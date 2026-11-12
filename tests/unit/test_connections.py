"""Calls through an application's connection slots: to Cayu Cloud's broker as the Agent, or
straight to a local host."""

from __future__ import annotations

import asyncio
import base64
import json
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import pytest

from cayu.connections import (
    MAX_BODY_BYTES,
    MAX_RETRY_AFTER_SECONDS,
    CloudConnectionBroker,
    ConnectionCallError,
    LocalConnections,
    LocalSlot,
    retry_after_seconds,
)

BROKER = "https://api.example.test/v1/agent-connections"


def test_from_environment_is_inert_outside_cloud_and_refuses_a_url_alone() -> None:
    assert CloudConnectionBroker.from_environment({}) is None
    with pytest.raises(ValueError, match="CAYU_CONNECTIONS_TOKEN"):
        CloudConnectionBroker.from_environment({"CAYU_CONNECTIONS_URL": BROKER})
    broker = CloudConnectionBroker.from_environment(
        {"CAYU_CONNECTIONS_URL": BROKER, "CAYU_CONNECTIONS_TOKEN": "cayu_conn_x"}
    )
    assert broker is not None
    assert "cayu_conn_x" not in repr(broker)


def test_a_request_goes_to_the_slot_as_the_agent_and_answers_the_connectors_response() -> None:
    sent: list[httpx.Request] = []

    def broker(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(
            200,
            json={
                "status": 201,
                "headers": [["content-type", "application/json"]],
                "body": base64.b64encode(b'{"ok":true}').decode(),
            },
        )

    client = CloudConnectionBroker(
        url=BROKER, token="cayu_conn_x", transport=httpx.MockTransport(broker)
    )
    answer = asyncio.run(
        client.request(
            "warehouse",
            "post",
            "/api/v2/statements",
            query={"async": "true", "tag": ["a", "b"]},
            headers={"Content-Type": "application/json"},
            body=b'{"statement":"select 1"}',
        )
    )

    [request] = sent
    assert str(request.url) == f"{BROKER}/warehouse/request"
    assert request.headers["authorization"] == "Bearer cayu_conn_x"
    assert json.loads(request.content) == {
        "method": "POST",
        "path": "/api/v2/statements",
        "query": {"async": "true", "tag": ["a", "b"]},
        "headers": {"content-type": "application/json"},
        "body": base64.b64encode(b'{"statement":"select 1"}').decode(),
    }
    assert (answer.status, answer.header("Content-Type"), answer.body) == (
        201,
        "application/json",
        b'{"ok":true}',
    )


def test_a_broker_refusal_names_its_code() -> None:
    def broker(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={"detail": {"code": "connection_unbound", "message": "The slot crm isn't bound."}},
        )

    client = CloudConnectionBroker(
        url=BROKER, token="cayu_conn_x", transport=httpx.MockTransport(broker)
    )
    with pytest.raises(ConnectionCallError) as caught:
        asyncio.run(client.request("crm", "GET", "/v1/records"))
    assert (caught.value.code, caught.value.status) == ("connection_unbound", 404)
    assert str(caught.value) == "The slot crm isn't bound."


def test_a_broker_refusal_carries_its_retry_after() -> None:
    def refused(retry_after: str | None) -> ConnectionCallError:
        def broker(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                429,
                headers={} if retry_after is None else {"Retry-After": retry_after},
                json={"detail": {"code": "too_many_calls", "message": "Too many calls."}},
            )

        client = CloudConnectionBroker(
            url=BROKER, token="cayu_conn_x", transport=httpx.MockTransport(broker)
        )
        with pytest.raises(ConnectionCallError) as caught:
            asyncio.run(client.request("crm", "GET", "/v1"))
        return caught.value

    error = refused("2")
    assert (error.code, error.status, error.retry_after) == ("too_many_calls", 429, 2.0)
    assert refused(None).retry_after is None
    assert refused("soon").retry_after is None
    assert refused("999999999").retry_after == MAX_RETRY_AFTER_SECONDS
    assert retry_after_seconds("\u00b2") is None
    assert retry_after_seconds("  ") is None
    later = format_datetime(datetime.now(UTC) + timedelta(seconds=90), usegmt=True)
    assert 80 < (refused(later).retry_after or 0) <= 90


def test_local_connections_send_straight_to_the_slots_host() -> None:
    sent: list[httpx.Request] = []

    def host(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(302, headers={"location": "https://elsewhere.test/"})

    local = LocalConnections(
        {"crm": LocalSlot("https://api.acme.test", {"x-acme-key": "dev-key"})},
        transport=httpx.MockTransport(host),
    )
    answer = asyncio.run(
        local.request("crm", "GET", "/v1/records", headers={"X-Acme-Key": "ignored"})
    )

    assert answer.status == 302
    assert sent[0].headers.get_list("x-acme-key") == ["dev-key"]
    assert "dev-key" not in repr(LocalSlot("https://api.acme.test", {"x-acme-key": "dev-key"}))
    with pytest.raises(ConnectionCallError) as caught:
        asyncio.run(local.request("warehouse", "GET", "/"))
    assert caught.value.code == "connection_unbound"


@pytest.mark.parametrize(
    "path",
    [
        "@evil.test/x",
        ".evil.test/x",
        "/v1/../admin",
        "/v1/%2e%2e/admin",
        "/v1/a%2F..%2Fadmin",
        "//evil.test/x",
        "/%2Fevil.test/x",
        "/v1/a%2F%2Fb",
        "/a%5cb",
        "/a\\b",
        "/a%00b",
        "/a?b",
        "/a#b",
    ],
)
def test_a_path_that_could_leave_the_host_is_refused_before_anything_is_sent(path: str) -> None:
    sent: list[httpx.Request] = []

    def host(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200)

    local = LocalConnections(
        {"crm": LocalSlot("https://api.acme.test", {"x-acme-key": "dev-key"})},
        transport=httpx.MockTransport(host),
    )
    broker = CloudConnectionBroker(
        url=BROKER, token="cayu_conn_x", transport=httpx.MockTransport(host)
    )
    for client in (local, broker):
        with pytest.raises(ConnectionCallError) as caught:
            asyncio.run(client.request("crm", "GET", path))
        assert caught.value.code == "path_invalid"
    assert sent == []


@pytest.mark.parametrize(
    "path",
    [
        "/repos/acme/widgets/labels/area%2Fruntime",
        "/repos/acme/widgets/contents/docs/C%23.md",
        "/repos/acme/widgets/contents/why%3F.md",
    ],
)
def test_a_slash_hash_or_question_mark_inside_a_segment_is_sent_still_encoded(path: str) -> None:
    sent: list[httpx.Request] = []

    def host(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        if request.url.host == "api.example.test":
            return httpx.Response(200, json={"status": 204, "headers": [], "body": ""})
        return httpx.Response(204)

    local = LocalConnections(
        {"crm": LocalSlot("https://api.acme.test")}, transport=httpx.MockTransport(host)
    )
    broker = CloudConnectionBroker(
        url=BROKER, token="cayu_conn_x", transport=httpx.MockTransport(host)
    )
    for client in (local, broker):
        assert asyncio.run(client.request("crm", "DELETE", path)).status == 204

    assert sent[0].url.raw_path.decode() == path
    assert json.loads(sent[1].content)["path"] == path


def test_headers_twice_and_an_oversized_body_are_refused_before_sending() -> None:
    def never(_: httpx.Request) -> httpx.Response:
        raise AssertionError("nothing is sent")

    broker = CloudConnectionBroker(
        url=BROKER, token="cayu_conn_x", transport=httpx.MockTransport(never)
    )
    with pytest.raises(ConnectionCallError) as caught:
        asyncio.run(broker.request("crm", "GET", "/v1", headers={"X-Key": "a", "x-key": "b"}))
    assert caught.value.code == "headers_invalid"
    with pytest.raises(ConnectionCallError) as caught:
        asyncio.run(broker.request("crm", "POST", "/v1", body=b"x" * (MAX_BODY_BYTES + 1)))
    assert caught.value.code == "body_too_large"


@pytest.mark.parametrize("slot", ["..", ".", "Warehouse", "a/b", "-x", ""])
def test_a_slot_that_isnt_a_slot_name_never_reaches_a_route(slot: str) -> None:
    def never(_: httpx.Request) -> httpx.Response:
        raise AssertionError("nothing is sent")

    broker = CloudConnectionBroker(
        url=BROKER, token="cayu_conn_x", transport=httpx.MockTransport(never)
    )
    with pytest.raises(ConnectionCallError) as caught:
        asyncio.run(broker.request(slot, "GET", "/x"))
    assert caught.value.code == "slot_invalid"


class _Recorded(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.closed = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        assert not self.closed
        return httpx.Response(200, json={"status": 200, "headers": [], "body": ""})

    async def aclose(self) -> None:
        self.closed = True


def test_an_applications_transport_stays_open_across_calls() -> None:
    transport = _Recorded()
    broker = CloudConnectionBroker(url=BROKER, token="cayu_conn_x", transport=transport)
    local = LocalConnections({"crm": LocalSlot("https://api.acme.test")}, transport=transport)

    async def calls() -> None:
        await asyncio.gather(
            broker.request("crm", "GET", "/x"),
            broker.request("crm", "GET", "/y"),
            local.request("crm", "GET", "/z"),
        )

    asyncio.run(calls())
    assert not transport.closed


def test_a_local_slot_never_takes_the_applications_host_header() -> None:
    sent: list[httpx.Request] = []

    def host(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200)

    local = LocalConnections(
        {"crm": LocalSlot("https://api.acme.test", {"authorization": "Bearer slot-secret"})},
        transport=httpx.MockTransport(host),
    )
    asyncio.run(local.request("crm", "GET", "/", headers={"Host": "attacker.example"}))

    assert sent[0].headers["host"] == "api.acme.test"


@pytest.mark.parametrize(
    ("code", "ambiguous"),
    [
        # Refused before the connector carried the call out.
        ("agent_credential_invalid", False),
        ("connection_unbound", False),
        ("connection_unavailable", False),
        ("connection_not_ready", False),
        ("connector_not_brokered", False),
        ("credential_invalid", False),
        ("scope_not_allowed", False),
        ("path_not_allowed", False),
        ("path_invalid", False),
        ("method_not_allowed", False),
        ("headers_invalid", False),
        ("body_too_large", False),
        ("request_too_large", False),
        ("too_many_calls", False),
        # May follow the connector's carrying it out.
        (None, True),
        ("upstream_failed", True),
        ("upstream_response_too_large", True),
        ("a_code_this_release_doesnt_know", True),
    ],
)
def test_only_a_refusal_made_before_sending_is_unambiguous(
    code: str | None, ambiguous: bool
) -> None:
    assert ConnectionCallError("Failed.", code=code).ambiguous is ambiguous


def test_a_local_answer_over_the_brokers_limit_is_refused_as_too_large() -> None:
    def host(request: httpx.Request) -> httpx.Response:
        size = MAX_BODY_BYTES + (1 if request.url.path == "/large" else 0)
        return httpx.Response(201, content=b"x" * size)

    local = LocalConnections(
        {"crm": LocalSlot("https://api.acme.test")}, transport=httpx.MockTransport(host)
    )
    assert len(asyncio.run(local.request("crm", "POST", "/fits")).body) == MAX_BODY_BYTES
    with pytest.raises(ConnectionCallError) as caught:
        asyncio.run(local.request("crm", "POST", "/large"))
    assert caught.value.code == "upstream_response_too_large"
    assert caught.value.ambiguous is True


def test_a_local_slot_asks_for_an_uncompressed_answer_whatever_the_application_asks() -> None:
    sent: list[httpx.Request] = []

    def host(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200)

    local = LocalConnections(
        {"crm": LocalSlot("https://api.acme.test")}, transport=httpx.MockTransport(host)
    )
    asyncio.run(local.request("crm", "GET", "/v1", headers={"Accept-Encoding": "gzip"}))
    assert sent[0].headers["accept-encoding"] == "identity"
