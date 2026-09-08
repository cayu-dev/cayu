"""Authenticated HTTP handoff using only the real application's discovered scope."""

import asyncio
import json
import time

import httpx
from websockets.asyncio.client import connect
from websockets.typing import Origin, Subprotocol

from cayu.server._browser_input_routes import OPERATOR_INPUT_SUBPROTOCOL
from cayu.server._browser_viewer_routes import OPERATOR_VIEW_SUBPROTOCOL
from cayu.tools._browser_control_transport import _private_transport_logger


async def handback_app_browser(
    *, session_id, input_endpoint, tls, private_text, viewer_only=False, checkpoint_consent="deny"
):
    async with httpx.AsyncClient(
        base_url=input_endpoint.split("/api/", 1)[0].replace("wss://", "https://", 1),
        verify=tls,
        trust_env=False,
    ) as client:
        root = "/api/browser-control"
        assert (await client.post(root + "/operator-session")).status_code == 401
        client.auth = httpx.BasicAuth("operator", "password")
        response = await client.post(root + "/operator-session")
        assert response.status_code == 200
        client.headers["X-Cayu-Browser-Operator"] = response.json()["operator_session_token"]

        async def discover(*, transition_pending=False):
            result = await client.get(root + "/sessions/" + session_id)
            # Discovery fails closed if its authority changes across store
            # readback. Poll only while an already accepted transition settles;
            # never retry the mutation or treat this response as authorization.
            if transition_pending and result.status_code == 403:
                return None
            assert result.status_code == 200, result.text
            records = result.json()["browsers"]
            assert len(records) == 1
            return records[0]

        async def wait_for(state, *, sensitive=False):
            async with asyncio.timeout(10):
                while True:
                    record = await discover(transition_pending=True)
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

        original = await discover()
        assert original is not None
        response = await client.post(
            root + "/pages",
            json={
                "identity": original["identity"],
                "expected_record_revision": original["revision"],
            },
        )
        assert response.status_code == 200, response.text
        pages = response.json()["pages"]
        assert len(pages) == 1
        if viewer_only:
            response = await client.post(
                root + "/view-ticket",
                json={
                    "identity": original["identity"],
                    "expected_record_revision": original["revision"],
                    "page": pages[0],
                },
            )
            assert response.status_code == 200, response.text
            async with connect(
                input_endpoint.removesuffix("/input") + "/viewer",
                ssl=tls,
                origin=Origin("https://operator.test"),
                subprotocols=[Subprotocol(OPERATOR_VIEW_SUBPROTOCOL)],
                proxy=None,
                compression=None,
                max_size=2 * 1024 * 1024,
                logger=_private_transport_logger(),
            ) as channel:
                await channel.send(response.json()["ticket"])
                assert await channel.recv() == "ready"
                await channel.send("frame")
                frame = await channel.recv()
                assert isinstance(frame, bytes) and frame.startswith(b"\x89PNG\r\n\x1a\n")
            # Deliberately close without a purge acknowledgement. Invocation end,
            # not this disconnect, must eventually permit bookkeeping retirement.
            return pages[0]["page_id"]
        now = time.time_ns() // 1_000_000
        response = await client.post(
            root + "/takeover",
            json={
                "request_id": "bt_" + "a" * 32,
                "identity": original["identity"],
                "expected_record_revision": original["revision"],
                "expected_control_epoch": original["control_epoch"],
                "pages": pages,
                "purpose_code": original["identity"]["operator_purpose"]["code"],
                "requested_at_ms": now,
                "expires_at_ms": now + 30_000,
                "maximum_until_ms": now + 60_000,
                "checkpoint_consent": checkpoint_consent,
            },
        )
        assert response.status_code == 200, response.text
        acquired = await wait_for("operator_controlled")
        assert acquired["owned_request"]["checkpoint_consent"] == checkpoint_consent

        def intent(record):
            return {
                "identity": record["identity"],
                "request_id": record["owned_request"]["request_id"],
                "expected_record_revision": record["revision"],
                "expected_control_epoch": record["control_epoch"],
            }

        response = await client.post(root + "/sensitive-entry", json=intent(acquired))
        assert response.status_code == 202, response.text
        acquired = await wait_for("operator_controlled", sensitive=True)
        for sequence, kind, value in [(1, "tab", "tab"), (2, "text", private_text)]:
            response = await client.post(
                root + "/pages",
                json={
                    "identity": acquired["identity"],
                    "expected_record_revision": acquired["revision"],
                },
            )
            assert response.status_code == 200, response.text
            response = await client.post(
                root + "/input-ticket",
                json={
                    **intent(acquired),
                    "page": response.json()["pages"][0],
                    "input_sequence": sequence,
                    "input_kind": kind,
                },
            )
            assert response.status_code == 200, response.text
            async with connect(
                input_endpoint,
                ssl=tls,
                origin=Origin("https://operator.test"),
                subprotocols=[Subprotocol(OPERATOR_INPUT_SUBPROTOCOL)],
                proxy=None,
                compression=None,
                max_size=65536,
                logger=_private_transport_logger(),
            ) as channel:
                await channel.send(response.json()["ticket"])
                assert await channel.recv() == "ready"
                await channel.send(value.encode("utf-8"))
                receipt = json.loads(await channel.recv())
                assert receipt["state"] == "settled"
                assert receipt["settled_input_sequence"] == sequence
                assert private_text not in json.dumps(receipt)
            acquired = await wait_for("operator_controlled", sensitive=True)
            assert acquired["owned_request"]["settled_input_sequence"] == sequence
        response = await client.post(
            root + "/handback",
            json=intent(acquired),
        )
        assert response.status_code == 200, response.text
        returned = await wait_for("agent_controlled")
        assert returned["fresh_observation_required"]
        assert returned["control_epoch"] > original["control_epoch"]
        return pages[0]["page_id"]
