"""HTTP handback and two ready input sockets cannot invalidate the input owner."""

import asyncio
import json

import httpx
import pytest
from fastapi import FastAPI
from tests.core.test_browser_control import operator_purpose
from tests.core.test_browser_control_authorization import Policy
from tests.core.test_browser_control_coordinator import coordinator, intent_for
from tests.core.test_browser_control_publisher import publication_fixture

from cayu.runtime._browser_control_channel import BoundBrowserGuest, BrowserGuestCommandOwner
from cayu.runtime._browser_control_input_tickets import BrowserInputTickets
from cayu.runtime._browser_control_publisher import BrowserControlPublisher
from cayu.runtime._browser_control_service import BrowserControlService
from cayu.server._browser_control_routes import (
    BrowserOperatorSessionTokens,
    create_browser_control_router,
)
from cayu.server._browser_input_routes import create_browser_input_router
from cayu.server.auth import BasicAuth


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_ready_input_sockets_and_handback_preserve_exact_settlement(tmp_path, monkeypatch, backend):
    async def scenario():
        async with publication_fixture(backend, tmp_path) as (store, bootstrap):
            await BrowserControlPublisher(store).publish(bootstrap)
            owner = coordinator(store, Policy(True))
            tickets = BrowserInputTickets(owner)
            service = BrowserControlService(
                purpose=operator_purpose(), guest_endpoint="wss://guest.test/control"
            )
            app = FastAPI()
            app.include_router(
                create_browser_control_router(
                    coordinator=owner,
                    auth=BasicAuth(username="operator", password="password"),
                    sessions=BrowserOperatorSessionTokens(b"k" * 32),
                    allowed_origin="https://operator.test",
                    input_tickets=tickets,
                )
            )
            app.include_router(
                create_browser_input_router(
                    coordinator=owner,
                    service=service,
                    tickets=tickets,
                    allowed_origin="https://operator.test",
                )
            )
            dispatched, release = asyncio.Event(), asyncio.Event()
            native_inputs = []
            tasks = []
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="https://operator.test",
                auth=httpx.BasicAuth("operator", "password"),
            ) as client:
                token = (await client.post("/browser-control/operator-session")).json()[
                    "operator_session_token"
                ]
                client.headers["X-Cayu-Browser-Operator"] = token
                response = await client.post(
                    "/browser-control/takeover", json=intent_for(bootstrap).model_dump(mode="json")
                )
                assert response.status_code == 200
                pending = (await owner._load(bootstrap.changed_record.identity))[1]
                assert response.json()["revision"] == pending.revision
                acquired = await owner._publish_guest_acquisition(
                    expected=pending, lease_until_ms=4000
                )

                def authority(record):
                    return {
                        "identity": record.identity.model_dump(mode="json"),
                        "expected_record_revision": record.revision,
                        "expected_control_epoch": record.control_epoch,
                        "request_id": pending.request.request_id,
                    }

                response = await client.post(
                    "/browser-control/sensitive-entry", json=authority(acquired)
                )
                assert response.status_code == 202
                sensitive_pending = (await owner._load(bootstrap.changed_record.identity))[1]
                assert response.json()["revision"] == sensitive_pending.revision
                sensitive = await owner._publish_guest_sensitive_entry(expected=sensitive_pending)

                class Connection:
                    async def send(self, raw):
                        self.message = json.loads(raw)
                        if self.message["kind"] == "text_input":
                            native_inputs.append(self.message.pop("text"))
                            dispatched.set()

                    async def recv(self):
                        message = self.message
                        if message["kind"] == "text_input":
                            await release.wait()
                        record = (await owner._load(sensitive.identity))[1]
                        return json.dumps(
                            {
                                **{
                                    key: message[key]
                                    for key in (
                                        "sequence",
                                        "channel_id",
                                        "worker_instance",
                                        "binding_sha256",
                                    )
                                },
                                "kind": "settled",
                                "state": record.state,
                                "control_epoch": record.control_epoch,
                                "settled_sequence": message.get(
                                    "input_sequence", record.settled_input_sequence
                                ),
                                "pending_sequence": None,
                                "fresh_observation_required": record.fresh_observation_required,
                            }
                        )

                commands = BrowserGuestCommandOwner(
                    coordinator=owner,
                    connection=Connection(),
                    bound=BoundBrowserGuest(sensitive, "channel", "b" * 64),
                )

                def connected(identity):
                    assert identity == sensitive.identity
                    return commands

                monkeypatch.setattr(service, "_connected_commands", connected)

                async def ready_socket():
                    response = await client.post(
                        "/browser-control/input-ticket",
                        json={
                            **authority(sensitive),
                            "input_sequence": 1,
                            "page": sensitive.request.pages[0].model_dump(mode="json"),
                        },
                    )
                    assert response.status_code == 200
                    incoming, outgoing = asyncio.Queue(), asyncio.Queue()
                    await incoming.put({"type": "websocket.connect"})
                    scope = {
                        "type": "websocket",
                        "asgi": {"version": "3.0"},
                        "http_version": "1.1",
                        "scheme": "wss",
                        "path": "/browser-control/input",
                        "raw_path": b"/browser-control/input",
                        "root_path": "",
                        "query_string": b"",
                        "subprotocols": ["cayu.browser-input.v1"],
                        "headers": [(b"origin", b"https://operator.test")],
                        "server": ("operator.test", 443),
                        "client": ("127.0.0.1", 1234),
                    }
                    task = asyncio.create_task(app(scope, incoming.get, outgoing.put))
                    tasks.append(task)
                    assert (await outgoing.get())["type"] == "websocket.accept"
                    await incoming.put(
                        {"type": "websocket.receive", "text": response.json()["ticket"]}
                    )
                    assert (await outgoing.get())["text"] == "ready"
                    return incoming, outgoing, task

                async def queued():
                    async with asyncio.timeout(3):
                        while commands._input is None:
                            await asyncio.sleep(0)

                try:
                    async with asyncio.timeout(10):
                        first, first_out, first_task = await ready_socket()
                        second, second_out, second_task = await ready_socket()
                        await first.put({"type": "websocket.receive", "bytes": b"private-canary"})
                        await queued()
                        drive = asyncio.create_task(commands.step())
                        tasks.append(drive)
                        await dispatched.wait()
                        reserved = (await owner._load(sensitive.identity))[1]
                        assert reserved.pending_input_sequence == 1
                        before = await store.load_checkpoint("session")
                        response = await client.post(
                            "/browser-control/handback", json=authority(reserved)
                        )
                        assert response.status_code == 409
                        assert await store.load_checkpoint("session") == before
                        release.set()
                        await drive
                        assert json.loads((await first_out.get())["text"])["state"] == "settled"
                        await first_task
                        settled = commands.bound.record
                        assert (
                            settled.pending_input_sequence is None
                            and settled.settled_input_sequence == 1
                        )
                        before = await store.load_checkpoint("session")
                        await second.put({"type": "websocket.receive", "bytes": b"stale-canary"})
                        await queued()
                        entry = commands._input
                        await commands.step()
                        await second_task
                        assert (await second_out.get())["type"] == "websocket.close"
                        assert entry.payload == bytearray()
                        assert native_inputs == ["private-canary"]
                        assert not commands._closed and commands.publication_transition is None
                        assert await store.load_checkpoint("session") == before
                        await commands.step()  # Shared status exchange remains usable.
                        response = await client.post(
                            "/browser-control/handback", json=authority(settled)
                        )
                        assert (
                            response.status_code == 200
                            and response.json()["state"] == "handback_pending"
                        )
                finally:
                    release.set()
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    assert await owner.drain()

    asyncio.run(scenario())
