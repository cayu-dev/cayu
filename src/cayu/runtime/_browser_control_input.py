"""Transient text ownership for the existing serialized browser guest channel."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cayu._validation import canonical_durable_json_bytes
from cayu.runtime._browser_control_authorization import (
    BrowserControlInputRejected,
    BrowserControlPermissionDenied,
)
from cayu.runtime.browser_control import (
    BrowserControlConflict,
    BrowserControlPrincipal,
    BrowserControlRecord,
    BrowserTextInputIntent,
)

if TYPE_CHECKING:
    from cayu.runtime._browser_control_channel import BrowserGuestCommandOwner


@dataclass(slots=True, repr=False)
class PendingBrowserText:
    principal: BrowserControlPrincipal
    operator_session_id: str
    intent: BrowserTextInputIntent
    payload: bytearray
    result: asyncio.Future[BrowserControlRecord]
    dispatched: bool = False

    def clear(self) -> None:
        self.payload[:] = b"\x00" * len(self.payload)
        self.payload.clear()


async def dispatch_browser_text(owner: BrowserGuestCommandOwner, entry: PendingBrowserText) -> None:
    from cayu.runtime._browser_control_channel import BoundBrowserGuest, _message

    entry.dispatched = True
    encoded = None
    try:
        source = owner.bound.record
        try:
            pending, publication = await owner.coordinator._prepare_text_input_admission(
                principal=entry.principal,
                operator_session_id=entry.operator_session_id,
                intent=entry.intent,
            )
        except (BrowserControlPermissionDenied, BrowserControlInputRejected) as error:
            # Positive pre-publication refusal: no native operation or durable
            # reservation exists. Do not tear down the shared command owner.
            if not entry.result.done():
                entry.result.set_exception(error)
            return
        if pending != owner.coordinator._input_admission_successor(source, entry.intent):
            raise BrowserControlConflict("Browser input preparation changed its owner.")
        owner._retain_publication_transition(source, pending)
        admitted = await owner.coordinator._publish_text_input_admission(publication)
        if admitted != pending:
            raise BrowserControlConflict("Browser input admission changed its owner.")
        owner.expected = pending
        owner._retain_publication_transition(pending, owner.coordinator._input_successor(pending))
        if entry.result.cancelled():
            raise BrowserControlConflict("Browser input caller left before dispatch.")
        owner.sequence += 1
        common = {
            "sequence": owner.sequence,
            "channel_id": owner.bound.channel_id,
            "worker_instance": pending.identity.worker_instance_id,
            "binding_sha256": owner.bound.binding_sha256,
        }
        encoded = json.dumps(
            {
                **common,
                "kind": "text_input" if entry.intent.input_kind == "text" else "key_input",
                "request_id": entry.intent.request_id,
                "epoch": pending.control_epoch,
                "input_sequence": entry.intent.input_sequence,
                "page_id": entry.intent.page.page_id,
                "page_epoch": entry.intent.page.control_epoch,
                **(
                    {"text": entry.payload.decode("utf-8")}
                    if entry.intent.input_kind == "text"
                    else {"key": entry.intent.input_kind}
                ),
            }
        )
        entry.clear()
        async with asyncio.timeout(5):
            await owner.connection.send(encoded)
            encoded = None
            reply = _message(await owner.connection.recv())
        expected = {
            **common,
            "kind": "settled",
            "state": pending.state,
            "control_epoch": pending.control_epoch,
            "settled_sequence": entry.intent.input_sequence,
            "pending_sequence": None,
            "fresh_observation_required": pending.fresh_observation_required,
        }
        if canonical_durable_json_bytes(reply, "browser input") != canonical_durable_json_bytes(
            expected, "browser input"
        ):
            raise BrowserControlConflict("Browser input acknowledgement differs.")
        settled = await owner.coordinator._publish_guest_input(expected=pending)
        owner.bound = BoundBrowserGuest(settled, owner.bound.channel_id, owner.bound.binding_sha256)
        owner.expected = settled
        owner.publication_transition = None
        if not entry.result.done():
            entry.result.set_result(settled)
    except BaseException:
        if not entry.result.done():
            entry.result.set_exception(BrowserControlConflict("Browser input did not settle."))
        raise
    finally:
        encoded = None
        entry.clear()
