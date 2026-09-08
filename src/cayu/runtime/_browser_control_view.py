"""Application-authorized private view exchange on the serialized guest owner."""

from __future__ import annotations

import asyncio
import json
import secrets

from cayu.runtime._browser_control_channel import BrowserGuestCommandOwner, _message
from cayu.runtime._browser_control_evidence import browser_read_evidence_matches
from cayu.runtime._browser_control_frames import (
    BrowserViewUnavailable,
    PrivateBrowserFrame,
    decode_browser_frame,
)
from cayu.runtime.browser_control import (
    BrowserControlConflict,
    BrowserControlPage,
    BrowserControlPrincipal,
)


async def capture_authorized_browser_view(
    owner: BrowserGuestCommandOwner,
    *,
    principal: BrowserControlPrincipal,
    operator_session_id: str,
    page: BrowserControlPage,
    until_ms: int,
) -> PrivateBrowserFrame:
    """Private command-loop entrance, not a concurrent socket reader or public API.

    The enclosing owner serializes this exchange with takeover/input and retains
    teardown on failure. The result remains private until a viewer owner validates
    its lifetime and delivers it; no tool, event or artifact publication occurs.
    """
    page = BrowserControlPage.model_validate(page)
    principal = BrowserControlPrincipal.model_validate(principal)
    record = owner.bound.record
    authorized = await owner.coordinator.authorize_view(
        principal=principal,
        operator_session_id=operator_session_id,
        identity=record.identity,
        expected_record_revision=record.revision,
    )
    _, current = await owner.coordinator._load(record.identity)
    if (
        authorized.record != record
        or current != record
        or record.sensitive_entry
        or record.sensitive_entry_pending
    ):
        raise BrowserViewUnavailable("Browser view lost its authorized generation.")
    view_id = "bv_" + secrets.token_hex(16)

    def common(sequence):
        return {
            "channel_id": owner.bound.channel_id,
            "worker_instance": record.identity.worker_instance_id,
            "binding_sha256": owner.bound.binding_sha256,
            "sequence": sequence,
        }

    def evidence(sequence):
        return {
            **common(sequence),
            "kind": "settled",
            "state": record.state,
            "control_epoch": record.control_epoch,
            "settled_sequence": record.settled_input_sequence,
            "pending_sequence": record.pending_input_sequence,
            "fresh_observation_required": record.fresh_observation_required,
        }

    async with asyncio.timeout(5):
        owner.sequence += 1
        await owner.connection.send(
            json.dumps(
                {
                    **common(owner.sequence),
                    "kind": "view",
                    "view_id": view_id,
                    "epoch": record.control_epoch,
                    "until_ms": until_ms,
                }
            )
        )
        reply = _message(await owner.connection.recv())
        generation = reply.get("generation")
        if type(generation) is not int or not 1 <= generation < 2**53:
            raise BrowserControlConflict("Browser view generation is unavailable.")
        expected = {**evidence(owner.sequence), "view_id": view_id, "generation": generation}
        if not browser_read_evidence_matches(reply, expected):
            raise BrowserControlConflict("Browser view acknowledgement differs.")
        owner.sequence += 1
        await owner.connection.send(
            json.dumps(
                {
                    **common(owner.sequence),
                    "kind": "frame",
                    "view_id": view_id,
                    "epoch": record.control_epoch,
                    "page_id": page.page_id,
                    "page_epoch": page.control_epoch,
                }
            )
        )
        raw = await owner.connection.recv_frame()
        frame = decode_browser_frame(
            raw,
            bound=owner.bound,
            view_id=view_id,
            sequence=owner.sequence,
            expected_page=page,
            expected_generation=generation,
        )
        del raw
        settled = _message(await owner.connection.recv())
        if not browser_read_evidence_matches(settled, evidence(owner.sequence)):
            raise BrowserControlConflict("Browser capture did not settle its exact command.")
    if frame is None:
        raise BrowserViewUnavailable("Browser capture is restricted.")
    # A concurrent durable takeover or sensitive transition invalidates delivery.
    _, current = await owner.coordinator._load(record.identity)
    if current != record:
        raise BrowserViewUnavailable("Browser view changed before delivery.")
    return frame
