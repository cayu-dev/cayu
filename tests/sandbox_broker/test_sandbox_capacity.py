"""``SandboxBrokerClient.capacity``: Cloud's ``GET /sandboxes/capacity``, against a fake broker."""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError
from tests.sandbox_broker.fake_broker import API_URL, APPLICATION, TOKEN, FakeBroker

from cayu.sandbox_broker import (
    SandboxBrokerClient,
    SandboxBrokerError,
    SandboxBrokerRefusedError,
    SandboxBrokerSettings,
    SandboxBrokerUnavailableError,
    SandboxNotFoundError,
)
from cayu.sandbox_broker import client as client_module

# The example in Cloud's contract (docs/contracts/sandbox-broker-v1.md, "Capacity"), as
# a Cloud that doesn't report queue positions yet answered it.
CONTRACT_EXAMPLE: dict[str, Any] = {
    "available": 1,
    "limited_by": "install",
    "agent": {"limit": 8, "in_use": 3, "queued": 0, "available": 5},
    "organization": {"id": "org_01J", "limit": 32, "in_use": 5, "queued": 0, "available": 27},
    "install": {"limit": 10, "in_use": 9, "queued": 2, "available": 1},
    "queue": {"available": 8, "max_seconds": 3600},
}


# The same contract's example once Cloud reports where each queued create stands.
QUEUED_EXAMPLE: dict[str, Any] = {
    "available": 0,
    "limited_by": "install",
    "agent": {"limit": 8, "in_use": 3, "queued": 1, "available": 5},
    "organization": {"id": "org_01J", "limit": 32, "in_use": 5, "queued": 1, "available": 27},
    "install": {"limit": 10, "in_use": 10, "queued": 3, "available": 0},
    "queue": {
        "available": 7,
        "max_seconds": 3600,
        "requests": [
            {
                "id": "sbx_7f2c",
                "queued_until": "2026-10-27T20:00:00+00:00",
                "waiting_for": "install",
                "position": 2,
            }
        ],
    },
}

_MISSING = object()


def _with(path: tuple[Any, ...], value: Any, *, base: dict[str, Any] = CONTRACT_EXAMPLE) -> Any:
    body = copy.deepcopy(base)
    target = body
    for key in path[:-1]:
        target = target[key]
    if value is _MISSING:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return body


def test_capacity_reads_clouds_shape(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario() -> None:
        client = broker.client()
        empty = await client.capacity()
        # Cloud's answer for an Agent with nothing running and no install limit.
        assert empty.model_dump() == {
            "available": 8,
            "limited_by": "agent",
            "agent": {"limit": 8, "in_use": 0, "queued": 0, "available": 8},
            "organization": {
                "id": "org_test",
                "limit": 32,
                "in_use": 0,
                "queued": 0,
                "available": 32,
            },
            "install": None,
            "queue": {"available": 8, "max_seconds": 3600, "requests": ()},
        }
        request = broker.requests[-1]
        assert request.method == "GET"
        assert request.url.path == "/v1/applications/compound/sandboxes/capacity"
        assert request.headers["authorization"] == f"Bearer {TOKEN}"

        # Another Organization holds one of the install's three places; this Agent one more.
        broker.install_limit = 3
        broker.install_in_use_elsewhere = 1
        await client.create(
            idempotency_key="a", image="coding-python", egress_mode="none", lease_seconds=60
        )
        busy = await client.capacity()
        assert (busy.available, busy.limited_by) == (1, "install")
        assert busy.install is not None
        assert (busy.install.limit, busy.install.in_use, busy.install.available) == (3, 2, 1)
        assert busy.agent.in_use == 1 and busy.agent.available == 7
        assert busy.scope("install") is busy.install
        with pytest.raises(ValidationError):
            busy.available = 5  # type: ignore[misc]

        # A hosting Agent reads it for an Organization it works for.
        broker.hosted_organizations = {"org_customer"}
        hosted = await client.capacity(organization="org_customer")
        assert hosted.organization.id == "org_customer"
        assert broker.requests[-1].url.params["organization"] == "org_customer"

        # The contract's own example parses as documented.
        broker.capacity_body = QUEUED_EXAMPLE
        queued = await client.capacity()
        listed = queued.queued_request("sbx_7f2c")
        assert listed is not None
        assert (listed.waiting_for, listed.position) == ("install", 2)
        # Both are null when a free place is certainly the create's and the next pass takes it.
        broker.capacity_body = _with(
            ("queue", "requests", 0),
            {**listed.model_dump(mode="json"), "waiting_for": None, "position": None},
            base=QUEUED_EXAMPLE,
        )
        free = (await client.capacity()).queued_request("sbx_7f2c")
        assert free is not None and (free.waiting_for, free.position) == (None, None)

        broker.capacity_body = CONTRACT_EXAMPLE
        example = await client.capacity()
        assert example.organization.id == "org_01J"
        assert example.install is not None and example.install.queued == 2
        assert (example.queue.available, example.queue.max_seconds) == (8, 3600)
        # A Cloud that doesn't report queue positions leaves them out.
        assert example.queue.requests is None
        assert example.queued_request("sbx_1") is None
        await client.aclose()

    asyncio.run(scenario())


def test_a_queued_create_reports_where_it_stands_until_admitted(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")
    broker.agent_limit = 1

    async def scenario() -> None:
        client = broker.client()
        running = await client.create(
            idempotency_key="a", image="coding-python", egress_mode="none", lease_seconds=60
        )
        assert "queue_seconds" not in json.loads(broker.requests[-1].content)
        await client.wait_until_ready(running.id, timeout_s=5, poll_interval_s=0)
        queued = await client.create(
            idempotency_key="b",
            image="coding-python",
            egress_mode="none",
            lease_seconds=60,
            queue_seconds=600,
        )
        assert json.loads(broker.requests[-1].content)["queue_seconds"] == 600
        assert queued.status == "provisioning" and queued.queued_until is not None

        position = await client.queue_position(queued.id)
        assert position is not None
        assert (position.id, position.waiting_for, position.position) == (queued.id, "agent", 1)
        assert position.queued_until == queued.queued_until
        capacity = await client.capacity()
        assert capacity.agent.queued == 1
        assert capacity.queued_request(queued.id) == position
        # A sandbox that isn't waiting has no position.
        assert await client.queue_position(running.id) is None

        # When Cloud's list is cut short, a queued sandbox missing from it isn't reported as
        # having left the queue.
        broker.capacity_body = {
            **capacity.model_dump(mode="json"),
            "queue": {**capacity.queue.model_dump(mode="json"), "requests": []},
        }
        with pytest.raises(SandboxBrokerError, match="position is unknown"):
            await client.queue_position(queued.id)
        broker.capacity_body = None

        # Once the running sandbox ends, the queued one is admitted and leaves the line.
        await client.destroy(running.id)
        broker.expire(running.id)
        ready = await client.wait_until_ready(queued.id, timeout_s=5, poll_interval_s=0)
        assert (ready.status, ready.queued_until) == ("ready", None)
        assert await client.queue_position(queued.id) is None
        await client.aclose()

    asyncio.run(scenario())


def test_queue_position_needs_a_cloud_that_reports_it(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")
    broker.capacity_body = CONTRACT_EXAMPLE

    async def scenario() -> None:
        client = broker.client()
        with pytest.raises(SandboxBrokerError, match="queue positions"):
            await client.queue_position("sbx_1")
        for queue_seconds in (-1, 3601, True, 1.5):
            with pytest.raises(ValueError, match="queue_seconds"):
                await client.create(
                    idempotency_key="q",
                    image="coding-python",
                    egress_mode="none",
                    lease_seconds=60,
                    queue_seconds=queue_seconds,  # ty: ignore[invalid-argument-type]
                )
        await client.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "body",
    [
        pytest.param([CONTRACT_EXAMPLE], id="not-an-object"),
        pytest.param(_with(("install",), _MISSING), id="install-missing"),
        pytest.param(_with(("queue",), _MISSING), id="queue-missing"),
        pytest.param(_with(("agent", "in_use"), -1), id="negative-count"),
        pytest.param(_with(("agent", "limit"), True), id="bool-count"),
        pytest.param(_with(("agent", "limit"), 8.0), id="float-count"),
        pytest.param(_with(("agent", "limit"), "8"), id="string-count"),
        pytest.param(_with(("limited_by",), "region"), id="unknown-scope"),
        pytest.param(_with(("install",), None), id="limited-by-unset-install"),
        pytest.param(_with(("available",), 2), id="available-not-the-tightest"),
        pytest.param(_with(("agent", "available"), 6), id="available-over-the-limits-free-places"),
        pytest.param(_with(("queue", "available"), 9), id="queue-over-the-agent-limit"),
        pytest.param(_with(("organization", "id"), ""), id="blank-organization"),
        pytest.param(_with(("organization", "id"), 17), id="numeric-organization"),
        pytest.param(
            _with(("queue", "requests", 0, "position"), 0, base=QUEUED_EXAMPLE),
            id="position-zero",
        ),
        pytest.param(
            _with(("queue", "requests", 0, "position"), "2", base=QUEUED_EXAMPLE),
            id="string-position",
        ),
        pytest.param(
            _with(("queue", "requests", 0, "waiting_for"), None, base=QUEUED_EXAMPLE),
            id="position-without-a-limit",
        ),
        pytest.param(
            _with(("queue", "requests", 0, "waiting_for"), "region", base=QUEUED_EXAMPLE),
            id="unknown-waiting-for",
        ),
        pytest.param(
            _with(("agent", "queued"), 0, base=QUEUED_EXAMPLE), id="more-requests-than-queued"
        ),
    ],
)
def test_a_malformed_capacity_answer_is_refused(tmp_path: Path, body: Any) -> None:
    broker = FakeBroker(tmp_path / "cloud")
    broker.capacity_body = body

    async def scenario() -> None:
        client = broker.client()
        with pytest.raises(SandboxBrokerError, match="invalid SandboxCapacity") as raised:
            await client.capacity()
        assert type(raised.value) is SandboxBrokerError
        await client.aclose()

    asyncio.run(scenario())
    # A well-formed 200 that doesn't validate is final, not retried.
    assert len(broker.requests) == 1


def test_capacity_for_another_organization_or_over_its_size_is_refused(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario() -> None:
        client = broker.client()
        broker.capacity_body = CONTRACT_EXAMPLE
        with pytest.raises(SandboxBrokerError, match="another Organization"):
            await client.capacity(organization="org_customer")
        broker.capacity_body = {**CONTRACT_EXAMPLE, "padding": "x" * (64 * 1024)}
        with pytest.raises(SandboxBrokerError, match="byte limit"):
            await client.capacity()
        for organization in ("", "x" * 201, "org\nnext", 17):
            with pytest.raises(ValueError, match="organization"):
                await client.capacity(organization=organization)  # ty: ignore[invalid-argument-type]
        await client.aclose()

    asyncio.run(scenario())
    # Neither invalid organization reached the broker.
    assert len(broker.requests) == 2


def test_capacity_errors_map_like_every_other_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(client_module, "_RETRY_DELAYS_SECONDS", (0.0, 0.0, 0.0))
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario() -> None:
        # A revoked credential's 401 is final.
        revoked = SandboxBrokerClient(
            SandboxBrokerSettings(
                api_url=API_URL, application_id=APPLICATION, token="cayu_sbx_revoked"
            ),
            transport=httpx.MockTransport(broker.transport),
        )
        with pytest.raises(SandboxBrokerRefusedError) as raised:
            await revoked.capacity()
        assert (raised.value.status, raised.value.code) == (401, None)
        assert len(broker.requests) == 1
        await revoked.aclose()

        client = broker.client()
        # Asking for an Organization the Agent doesn't host is refused with the contract code.
        before = len(broker.requests)
        with pytest.raises(SandboxBrokerRefusedError) as refused:
            await client.capacity(organization="org_other")
        assert (refused.value.status, refused.value.code) == (403, "on_behalf_of_not_allowed")
        assert len(broker.requests) == before + 1
        # An unknown Agent's 404 is final.
        broker.bare_statuses = [404]
        with pytest.raises(SandboxNotFoundError):
            await client.capacity()
        # 429 and 5xx without a contract code are retried ...
        broker.bare_statuses = [429, 503]
        assert (await client.capacity()).available == 8
        # ... until the retries run out.
        broker.bare_statuses = [503, 503, 503, 503]
        with pytest.raises(SandboxBrokerUnavailableError) as unavailable:
            await client.capacity()
        assert unavailable.value.status == 503
        await client.aclose()

    asyncio.run(scenario())
