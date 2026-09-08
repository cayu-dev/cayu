"""Content-free, policy-authorized page census on the single guest reader."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cayu.runtime._browser_control_authorization import BrowserControlRevisionChanged
from cayu.runtime._browser_control_evidence import browser_read_evidence_matches
from cayu.runtime.browser_control import (
    BrowserControlConflict,
    BrowserControlPage,
    BrowserControlPrincipal,
    BrowserObservedPageLocation,
)

if TYPE_CHECKING:
    from cayu.runtime._browser_control_channel import BrowserGuestCommandOwner


@dataclass(frozen=True, slots=True)
class BrowserPageDescriptors:
    active_page_id: str | None
    pages: tuple[BrowserControlPage, ...]
    locations: tuple[BrowserObservedPageLocation, ...]


async def read_browser_pages(
    owner: BrowserGuestCommandOwner, *, principal: BrowserControlPrincipal, operator_session_id: str
) -> BrowserPageDescriptors:
    from cayu.runtime._browser_control_channel import _message

    record = owner.bound.record
    authorized = await owner.coordinator.authorize_view(
        principal=principal,
        operator_session_id=operator_session_id,
        identity=record.identity,
        expected_record_revision=record.revision,
    )
    _, current = await owner.coordinator._load(record.identity)
    if current.revision != record.revision:
        raise BrowserControlRevisionChanged("Browser pages lost their authorized generation.")
    if current != record or authorized.record != record:
        raise BrowserControlConflict("Browser pages lost their authorized generation.")
    owner.sequence += 1
    common = {
        "sequence": owner.sequence,
        "channel_id": owner.bound.channel_id,
        "worker_instance": record.identity.worker_instance_id,
        "binding_sha256": owner.bound.binding_sha256,
    }
    async with asyncio.timeout(5):
        await owner.connection.send(
            json.dumps({"kind": "pages", "epoch": record.control_epoch, **common})
        )
        reply = _message(await owner.connection.recv())
    raw_pages = reply.pop("pages", None)
    raw_locations = reply.pop("locations", None)
    active = reply.pop("active_page_id", None)
    expected = {
        **common,
        "kind": "settled",
        "control_epoch": record.control_epoch,
        "state": record.state,
        "settled_sequence": record.settled_input_sequence,
        "pending_sequence": record.pending_input_sequence,
        "fresh_observation_required": record.fresh_observation_required,
    }
    if not browser_read_evidence_matches(reply, expected):
        raise BrowserControlConflict("Browser page census changed control authority.")
    if type(raw_pages) is not list or len(raw_pages) > 16:
        raise BrowserControlConflict("Browser page census is malformed.")
    pages = tuple(BrowserControlPage.model_validate(page) for page in raw_pages)
    if type(raw_locations) is not list or len(raw_locations) != len(pages):
        raise BrowserControlConflict("Browser page locations are malformed.")
    locations = tuple(BrowserObservedPageLocation.model_validate(item) for item in raw_locations)
    if tuple(item.page for item in locations) != pages:
        raise BrowserControlConflict("Browser locations belong to different page observations.")
    locations = tuple(
        item.model_copy(
            update={
                "origin": owner.coordinator.protect_page_origin(
                    item.origin, identity=record.identity
                )
            }
        )
        for item in locations
    )
    ids = {page.page_id for page in pages}
    if len(ids) != len(pages) or (
        active is not None and (type(active) is not str or active not in ids)
    ):
        raise BrowserControlConflict("Browser page census has conflicting identities.")
    _, current = await owner.coordinator._load(record.identity)
    if current.revision != record.revision:
        raise BrowserControlRevisionChanged("Browser pages changed during discovery.")
    if current != record:
        raise BrowserControlConflict("Browser pages changed during discovery.")
    return BrowserPageDescriptors(active, pages, locations)
