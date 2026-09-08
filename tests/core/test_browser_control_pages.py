"""Page discovery traverses the serialized owner and native guest command parser."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from tests.core.test_browser_control import operator_purpose
from tests.core.test_browser_control_authorization import Policy
from tests.core.test_browser_control_coordinator import coordinator
from tests.core.test_browser_control_publisher import publication_fixture

from cayu.runtime._browser_control_channel import BoundBrowserGuest, BrowserGuestCommandOwner
from cayu.runtime._browser_control_publisher import BrowserControlPublisher
from cayu.runtime._browser_control_service import BrowserControlService
from cayu.runtime.browser_control import BrowserControlPrincipal
from cayu.server._browser_control_routes import (
    BrowserOperatorSessionTokens,
    create_browser_control_router,
)
from cayu.server.auth import BasicAuth
from cayu.tools._browser_control_guest import GuestControlChannel
from cayu.tools._browser_guest import _InteractiveDaemon, _InteractivePage


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("denied_first", [False, True])
@pytest.mark.parametrize("revision", ["revision", None])
@pytest.mark.parametrize(
    ("url", "secret", "expected_origin"),
    [
        ("https://site.test/private-canary?token=private-canary", None, "https://site.test"),
        ("https://private-canary.test/", "private-canary", None),
        ("https://site.test/", "private-canary", "https://site.test"),
        ("https://private-canary@site.test/", None, None),
        ("about:blank", None, None),
    ],
)
def test_serialized_page_discovery_uses_guest_authority(
    tmp_path, monkeypatch, backend, denied_first, revision, url, secret, expected_origin
):
    async def scenario():
        async with publication_fixture(backend, tmp_path) as (store, bootstrap):
            record = await BrowserControlPublisher(store).publish(bootstrap)
            daemon = _InteractiveDaemon(record.identity.browser_session_id)
            reads = []

            async def storage_state(*, indexed_db):
                reads.append(indexed_db)
                return {
                    "cookies": [],
                    "origins": [
                        {
                            "origin": "https://site.test",
                            "localStorage": [{"name": "token", "value": secret}],
                        }
                    ],
                }

            daemon.context = SimpleNamespace(storage_state=storage_state)
            if secret is not None:
                daemon.profile_output_values = ()
                daemon.profile_plaintext_limit = 65536
                daemon.profile_timeout_seconds = 5
                daemon.control.capture_restricted = True
            daemon.visual_worker_instance = record.identity.worker_instance_id
            channel = GuestControlChannel(daemon, scope_sha256="a" * 64)
            channel._binding = "b" * 64
            daemon.claim_operator_channel(channel._nonce)
            await daemon.bind_operator_control("b" * 64)
            daemon.pages["page"] = _InteractivePage(
                page=SimpleNamespace(url=url),
                session_id=daemon.session_id,
                page_id="page",
                lifecycle="active",
                revision=revision,
                title="private-canary",
            )
            daemon.active_page_id = "page"
            queue = asyncio.Queue()

            class Connection:
                async def send(self, raw):
                    result = await channel._command(json.loads(raw), self)
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

            policy = Policy(True)
            control = coordinator(store, policy)
            owner = BrowserGuestCommandOwner(
                coordinator=control,
                connection=Connection(),
                bound=BoundBrowserGuest(record, channel._nonce, "b" * 64),
            )
            if denied_first:
                service = BrowserControlService(
                    purpose=operator_purpose(), guest_endpoint="wss://guest.test/control"
                )
                monkeypatch.setattr(service, "_connected_commands", lambda identity: owner)
                app = FastAPI()
                app.include_router(
                    create_browser_control_router(
                        coordinator=control,
                        auth=BasicAuth(username="operator", password="password"),
                        sessions=BrowserOperatorSessionTokens(b"k" * 32),
                        allowed_origin="https://operator.test",
                        service=service,
                    )
                )
                before = await store.load_checkpoint(record.identity.session_id)
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="https://operator.test",
                    auth=httpx.BasicAuth("operator", "password"),
                ) as client:
                    token = (await client.post("/browser-control/operator-session")).json()[
                        "operator_session_token"
                    ]
                    policy.allowed = False
                    denied = asyncio.create_task(
                        client.post(
                            "/browser-control/pages",
                            headers={"X-Cayu-Browser-Operator": token},
                            json={
                                "identity": record.identity.model_dump(mode="json"),
                                "expected_record_revision": record.revision,
                            },
                        )
                    )
                    async with asyncio.timeout(2):
                        while owner._pages is None:
                            await asyncio.sleep(0)
                    await owner.step()
                    assert (await denied).status_code == 403
                assert owner.sequence == 0 and queue.empty()
                assert not owner._closed and owner._pages is None
                assert await store.load_checkpoint(record.identity.session_id) == before
                policy.allowed = True
            caller = asyncio.create_task(
                owner.request_pages(
                    principal=BrowserControlPrincipal(subject="operator"),
                    operator_session_id="continuity",
                )
            )
            await asyncio.sleep(0)
            await owner.step()
            result = await caller
            assert result.active_page_id == "page"
            assert result.pages[0].revision == daemon._operator_page_revision(daemon.pages["page"])
            assert daemon.pages["page"].revision == revision
            assert result.locations[0].page == result.pages[0]
            assert result.locations[0].origin == expected_origin
            assert reads == ([] if secret is None else [False])
            assert "private-canary" not in repr(result)
            assert owner._pages is None
            assert await control.drain()

    asyncio.run(scenario())
