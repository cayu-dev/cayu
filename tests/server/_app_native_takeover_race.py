"""Two authenticated operators racing an actually dispatched app browser call."""

import asyncio
import json
import time
from contextlib import AsyncExitStack

import httpx
from fastapi import HTTPException
from tests.core.test_browser_control_authorization import Policy
from websockets.asyncio.client import connect
from websockets.typing import Origin, Subprotocol

from cayu.server import BasicAuth
from cayu.server._browser_input_routes import OPERATOR_INPUT_SUBPROTOCOL
from cayu.tools._browser_control_transport import _private_transport_logger
from cayu.tools.browser_session import _durable_browser_operation_key


class NativeTakeoverRace(Policy):
    def __init__(self, *, tls, private_text):
        super().__init__(True)
        self.tls = tls
        self.private_text = private_text
        self.barrier = asyncio.Barrier(2)
        self.contenders = []
        self.dispatched = []
        self.model_entered = asyncio.Event()
        self.release_model = asyncio.Event()
        self.acquisition_ready = asyncio.Event()
        self.release_acquisition = asyncio.Event()
        self.fresh_status = asyncio.Event()
        self.fresh_in_progress = False
        self.stack = AsyncExitStack()
        self.task = None
        self.clients = []
        self.root = "/api/browser-control"

    async def authenticate(self, request):
        for name in ("operator-a", "operator-b"):
            try:
                return await BasicAuth(username=name, password="password", tenant="tenant")(request)
            except HTTPException as error:
                if error.status_code != 401:
                    raise
        raise HTTPException(401, "Authentication required.")

    async def decide(self, request):
        if request.action == "takeover" and len(self.contenders) < 2:
            self.contenders.append(request.principal.subject)
            await self.barrier.wait()
        return await super().decide(request)

    def instrument(self, daemon, monkeypatch):
        self.monkeypatch = monkeypatch
        execute = daemon._execute_locked
        settle = daemon._settle_operator_control_owners

        async def held_acquisition():
            await settle()
            if daemon.control.state == "takeover_requested":
                self.acquisition_ready.set()
                await self.release_acquisition.wait()

        async def held(request):
            self.dispatched.append(request.operation_id)
            result = await execute(request)
            if request.operation_id == "held-model":
                assert result["kind"] == "success"
                assert daemon.lock.locked()
                self.model_entered.set()
                await self.release_model.wait()
            return result

        monkeypatch.setattr(daemon, "_execute_locked", held)
        monkeypatch.setattr(daemon, "_settle_operator_control_owners", held_acquisition)

    async def discover(self, client):
        result = await client.get(self.root + "/sessions/" + self.session_id)
        # A concurrent CAS may invalidate read authority. Retry reads only.
        if result.status_code == 403:
            return None
        assert result.status_code == 200, result.text
        records = result.json()["browsers"]
        assert len(records) == 1
        return records[0]

    async def wait_for(self, client, state, *, sensitive=False):
        async with asyncio.timeout(10):
            while True:
                record = await self.discover(client)
                if (
                    record is not None
                    and record["state"] == state
                    and (
                        not sensitive
                        or (record["sensitive_entry"] and not record["sensitive_entry_pending"])
                    )
                ):
                    return record
                await asyncio.sleep(0.01)

    @staticmethod
    def intent(record):
        return {
            "identity": record["identity"],
            "request_id": record["owned_request"]["request_id"],
            "expected_record_revision": record["revision"],
            "expected_control_epoch": record["control_epoch"],
        }

    async def prepare(self, *, endpoint, session_id, daemon, store, commands):
        self.endpoint = endpoint.removesuffix("/guest") + "/input"
        self.session_id = session_id
        self.daemon = daemon
        self.store = store
        step = commands.step

        async def observed_status():
            result = await step()
            if self.fresh_in_progress:
                assert commands.bound.record.fresh_observation_required
                self.fresh_status.set()
            return result

        self.monkeypatch.setattr(commands, "step", observed_status)
        for name in ("operator-a", "operator-b"):
            client = await self.stack.enter_async_context(
                httpx.AsyncClient(
                    base_url=endpoint.split("/api/", 1)[0].replace("wss://", "https://", 1),
                    verify=self.tls,
                    trust_env=False,
                    auth=httpx.BasicAuth(name, "password"),
                )
            )
            response = await client.post(self.root + "/operator-session")
            assert response.status_code == 200, response.text
            client.headers["X-Cayu-Browser-Operator"] = response.json()["operator_session_token"]
            self.clients.append(client)
        self.original = await self.wait_for(self.clients[0], "agent_controlled")
        response = await self.clients[0].post(
            self.root + "/pages",
            json={
                "identity": self.original["identity"],
                "expected_record_revision": self.original["revision"],
            },
        )
        assert response.status_code == 200, response.text
        self.pages = response.json()["pages"]
        assert len(self.pages) == 1
        self.task = asyncio.create_task(self.race(), name="native-takeover-contenders")

    async def before_fresh_publication(self):
        # Native observe has completed, but its actual runner response is still
        # held: the host cannot yet publish the observation's terminal receipt.
        assert not self.daemon.control.fresh_observation_required
        self.fresh_in_progress = True
        try:
            await asyncio.wait_for(self.fresh_status.wait(), 5)
            current = await self.wait_for(self.winner, "agent_controlled")
            assert current["fresh_observation_required"]
            response = await self.winner.post(
                self.root + "/pages",
                json={
                    "identity": current["identity"],
                    "expected_record_revision": current["revision"],
                },
            )
            assert response.status_code == 200, response.text
            assert response.json()["pages"]
            unchanged = await self.wait_for(self.winner, "agent_controlled")
            assert unchanged == current
        finally:
            self.fresh_in_progress = False

    async def race(self):
        try:
            await asyncio.wait_for(self.model_entered.wait(), 10)
            now = time.time_ns() // 1_000_000
            payload = {
                "identity": self.original["identity"],
                "expected_record_revision": self.original["revision"],
                "expected_control_epoch": self.original["control_epoch"],
                "pages": self.pages,
                "purpose_code": self.original["identity"]["operator_purpose"]["code"],
                "requested_at_ms": now,
                "expires_at_ms": now + 30_000,
                "maximum_until_ms": now + 60_000,
                "checkpoint_consent": "deny",
            }
            responses = await asyncio.gather(
                *(
                    client.post(
                        self.root + "/takeover",
                        json={**payload, "request_id": "bt_" + str(index + 1) * 32},
                    )
                    for index, client in enumerate(self.clients)
                )
            )
            assert sorted(response.status_code for response in responses) == [200, 409]
            assert sorted(self.contenders) == ["operator-a", "operator-b"]
            winner = next(
                index for index, response in enumerate(responses) if response.status_code == 200
            )
            self.winner, self.loser = self.clients[winner], self.clients[1 - winner]
            pending = await self.wait_for(self.winner, "takeover_requested")
            async with asyncio.timeout(5):
                while self.daemon.control.state != "takeover_requested":
                    await asyncio.sleep(0.01)
            assert self.daemon.lock.locked() and not self.release_model.is_set()
            held = await self.store.load_session_operation(
                self.session_id, _durable_browser_operation_key("held-model")
            )
            assert held is not None and held["state"] == "dispatched"
            assert held["operation"] == "list_pages"
            assert self.daemon.control.epoch == self.original["control_epoch"]
            assert self.daemon.control.settled_sequence == 0
            assert self.daemon._operator_input_task is None
            response = await self.winner.post(
                self.root + "/input-ticket",
                json={
                    **self.intent(pending),
                    "page": self.pages[0],
                    "input_sequence": 1,
                    "input_kind": "tab",
                },
            )
            assert response.status_code == 409, response.text
            assert self.daemon._operator_input_task is None
            self.release_model.set()
            await asyncio.wait_for(self.acquisition_ready.wait(), 5)
            assert self.daemon.control.state == "takeover_requested"
        finally:
            self.release_model.set()

    async def acquire(self):
        assert self.dispatched == ["open", "held-model"]
        assert (
            await self.store.load_session_operation(
                self.session_id, _durable_browser_operation_key("blocked-model")
            )
            is None
        )
        pending = await self.wait_for(self.winner, "takeover_requested")
        assert pending["control_epoch"] == self.original["control_epoch"]
        assert self.daemon.control.state == "takeover_requested"
        assert self.daemon._operator_input_task is None
        self.release_acquisition.set()
        self.acquired = await self.wait_for(self.winner, "operator_controlled")
        assert self.acquired["control_epoch"] == self.original["control_epoch"] + 1
        losing_view = await self.wait_for(self.loser, "operator_controlled")
        assert losing_view["owned_request"] is None
        response = await self.loser.post(
            self.root + "/sensitive-entry", json=self.intent(self.acquired)
        )
        assert response.status_code == 409, response.text
        assert not self.daemon.control.sensitive_entry

    async def finish(self):
        await self.acquire()
        response = await self.winner.post(
            self.root + "/sensitive-entry", json=self.intent(self.acquired)
        )
        assert response.status_code == 202, response.text
        acquired = await self.wait_for(self.winner, "operator_controlled", sensitive=True)
        for sequence, kind, text in [(1, "tab", "tab"), (2, "text", self.private_text)]:
            response = await self.winner.post(
                self.root + "/pages",
                json={
                    "identity": acquired["identity"],
                    "expected_record_revision": acquired["revision"],
                },
            )
            assert response.status_code == 200, response.text
            response = await self.winner.post(
                self.root + "/input-ticket",
                json={
                    **self.intent(acquired),
                    "page": response.json()["pages"][0],
                    "input_sequence": sequence,
                    "input_kind": kind,
                },
            )
            assert response.status_code == 200, response.text
            async with connect(
                self.endpoint,
                ssl=self.tls,
                origin=Origin("https://operator.test"),
                subprotocols=[Subprotocol(OPERATOR_INPUT_SUBPROTOCOL)],
                proxy=None,
                compression=None,
                max_size=65536,
                logger=_private_transport_logger(),
            ) as channel:
                await channel.send(response.json()["ticket"])
                assert await channel.recv() == "ready"
                await channel.send(text.encode())
                receipt = json.loads(await channel.recv())
                assert receipt["state"] == "settled"
                assert receipt["settled_input_sequence"] == sequence
            acquired = await self.wait_for(self.winner, "operator_controlled", sensitive=True)
        response = await self.winner.post(self.root + "/handback", json=self.intent(acquired))
        assert response.status_code == 200, response.text
        returned = await self.wait_for(self.winner, "agent_controlled")
        assert returned["fresh_observation_required"]
        assert self.daemon.control.fresh_observation_required
        page = self.daemon.pages[self.pages[0]["page_id"]]
        assert await page.page.locator("input").input_value() == self.private_text

    async def close(self):
        self.release_model.set()
        self.release_acquisition.set()
        try:
            if self.task is not None:
                if not self.task.done():
                    self.task.cancel()
                await asyncio.gather(self.task, return_exceptions=True)
        finally:
            await self.stack.aclose()
