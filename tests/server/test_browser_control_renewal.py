"""Authenticated renewal traverses durable admission and the real guest command owner."""

import asyncio
import json
from datetime import UTC, datetime

import httpx
import pytest
from fastapi import FastAPI
from tests.core.test_browser_control_authorization import Policy
from tests.core.test_browser_control_coordinator import coordinator, intent_for
from tests.core.test_browser_control_publisher import publication_fixture

from cayu.runtime._browser_control_channel import BoundBrowserGuest, BrowserGuestCommandOwner
from cayu.runtime._browser_control_publisher import BrowserControlPublisher
from cayu.server._browser_control_routes import (
    BrowserOperatorSessionTokens,
    create_browser_control_router,
)
from cayu.server.auth import BasicAuth
from cayu.tools._browser_control_guest import GuestControlChannel, GuestControlFence
from cayu.tools._browser_guest import _InteractiveDaemon, _InteractivePage


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize(
    "outcome", ["success", "revoked", "expired", "lost_ack", "cancelled", "publication_lost"]
)
def test_http_renewal_requires_native_settlement(tmp_path, backend, outcome):
    async def scenario():
        async with publication_fixture(backend, tmp_path) as (store, bootstrap):
            record = await BrowserControlPublisher(store).publish(bootstrap)
            now = [1.0]
            policy = Policy(True)
            control = coordinator(store, policy)
            control._clock = lambda: datetime.fromtimestamp(now[0], UTC)
            daemon = _InteractiveDaemon(record.identity.browser_session_id)
            daemon.context = object()
            daemon.visual_worker_instance = record.identity.worker_instance_id
            daemon.control = GuestControlFence(
                worker_instance=record.identity.worker_instance_id,
                monotonic=lambda: now[0],
                wall_clock=lambda: now[0],
            )
            channel = GuestControlChannel(daemon, scope_sha256="a" * 64)
            channel._binding = "b" * 64
            daemon.claim_operator_channel(channel._nonce)
            await daemon.bind_operator_control("b" * 64)
            daemon.pages["page"] = _InteractivePage(
                page=object(),
                session_id=daemon.session_id,
                page_id="page",
                lifecycle="active",
                revision="revision",
            )
            daemon.active_page_id = "page"
            queue = asyncio.Queue()
            renewed = asyncio.Event()
            release = asyncio.Event()
            renewal_calls = []

            class Connection:
                async def send(self, raw):
                    message = json.loads(raw)
                    result = await channel._command(message, self)
                    if message["kind"] == "renew":
                        renewal_calls.append(message)
                        renewed.set()
                        if outcome == "lost_ack":
                            raise OSError("lost acknowledgement")
                        if outcome == "cancelled":
                            await release.wait()
                    await queue.put(
                        json.dumps(
                            {
                                "kind": "settled",
                                "channel_id": channel._nonce,
                                "sequence": channel._sequence,
                                **result,
                            }
                        )
                    )

                async def recv(self):
                    return await queue.get()

            owner = BrowserGuestCommandOwner(
                coordinator=control,
                connection=Connection(),
                bound=BoundBrowserGuest(record, channel._nonce, "b" * 64),
            )
            server = FastAPI()
            server.include_router(
                create_browser_control_router(
                    coordinator=control,
                    auth=BasicAuth(username="operator", password="password", tenant="tenant"),
                    sessions=BrowserOperatorSessionTokens(b"k" * 32),
                    allowed_origin="https://operator.test",
                )
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server),
                base_url="https://operator.test",
                auth=httpx.BasicAuth("operator", "password"),
            ) as client:
                token = (await client.post("/browser-control/operator-session")).json()[
                    "operator_session_token"
                ]
                headers = {"X-Cayu-Browser-Operator": token}
                takeover = intent_for(bootstrap).model_copy(update={"maximum_until_ms": 60_000})
                assert (
                    await client.post(
                        "/browser-control/takeover",
                        headers=headers,
                        json=takeover.model_dump(mode="json"),
                    )
                ).status_code == 200
                await owner.step()
                acquired = owner.bound.record
                assert acquired.lease_until_ms == 31_000
                now[0] = 2.0
                payload = {
                    "identity": acquired.identity.model_dump(mode="json"),
                    "expected_record_revision": acquired.revision,
                    "expected_control_epoch": acquired.control_epoch,
                    "request_id": takeover.request_id,
                    "expected_lease_until_ms": 31_000,
                    "lease_until_ms": 32_000,
                }
                # Wrong continuity never changes the lease even for the same user.
                other = (await client.post("/browser-control/operator-session")).json()[
                    "operator_session_token"
                ]
                assert (
                    await client.post(
                        "/browser-control/renew",
                        headers={"X-Cayu-Browser-Operator": other},
                        json=payload,
                    )
                ).status_code == 409
                response = await client.post(
                    "/browser-control/renew", headers=headers, json=payload
                )
                assert response.status_code == 202
                assert response.json()["lease_until_ms"] == 31_000
                assert response.json()["pending_lease_until_ms"] == 32_000
                assert daemon.control.lease_wall_ms == 31_000
                _, pending = await control._load(acquired.identity)
                handback = {
                    key: value
                    for key, value in payload.items()
                    if key not in {"expected_lease_until_ms", "lease_until_ms"}
                }
                handback["expected_record_revision"] = pending.revision
                assert (
                    await client.post("/browser-control/handback", headers=headers, json=handback)
                ).status_code == 409
                if outcome == "revoked":
                    policy.allowed = False
                elif outcome == "expired":
                    now[0] = 31.0
                elif outcome == "publication_lost":
                    publish = control._publish_guest_renewal

                    async def lose_publication_ack(*, expected):
                        await publish(expected=expected)
                        raise OSError("publication acknowledgement lost")

                    control._publish_guest_renewal = lose_publication_ack
                if outcome == "cancelled":
                    task = asyncio.create_task(owner.step())
                    await asyncio.wait_for(renewed.wait(), 2)
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                    assert task.cancelled() and task.cancelling() == 1
                    release.set()
                elif outcome != "success":
                    with pytest.raises(Exception):
                        await owner.step()
                else:
                    await owner.step()
                    settled = owner.bound.record
                    assert settled.lease_until_ms == daemon.control.lease_wall_ms == 32_000
                    assert settled.pending_lease_until_ms is None
                    assert settled.control_epoch == acquired.control_epoch
                    assert settled.revision == acquired.revision + 2
                    # Same old authority cannot dispatch another renewal.
                    assert (
                        await client.post("/browser-control/renew", headers=headers, json=payload)
                    ).status_code == 409
                if outcome != "success":
                    await owner.disconnect()
                    _, fenced = await control._load(acquired.identity)
                    assert fenced.state == "control_uncertain"
                    assert fenced.lease_until_ms == (
                        32_000 if outcome == "publication_lost" else 31_000
                    )
                    assert fenced.pending_lease_until_ms == (
                        None if outcome == "publication_lost" else 32_000
                    )
                assert len(renewal_calls) == (0 if outcome in {"revoked", "expired"} else 1)
            assert await control.drain()

    asyncio.run(scenario())
