"""Server-side exact guest handshake after private bootstrap authentication.

The caller must authenticate a capability delivered through the admitted runner;
the guest's self-reported allocation, worker, or digest cannot authenticate itself.
No viewer token or ordinary HTTP body may supply that trusted allocation argument.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING, Any

from cayu._task_wait import consume_pending_task_cancellation, restore_task_cancellation_requests
from cayu._validation import canonical_durable_json_bytes
from cayu.runtime._browser_control_authorization import (
    BrowserControlPermissionDenied,
    BrowserControlRevisionChanged,
)
from cayu.runtime._browser_control_coordinator import BrowserControlCoordinator
from cayu.runtime._browser_control_input import PendingBrowserText, dispatch_browser_text
from cayu.runtime._browser_control_pages import BrowserPageDescriptors, read_browser_pages
from cayu.runtime.browser_control import (
    BrowserControlAllocation,
    BrowserControlConflict,
    BrowserControlIdentity,
    BrowserControlPage,
    BrowserControlPageAudit,
    BrowserControlPrincipal,
    BrowserControlRecord,
    BrowserTextInputIntent,
    closed_browser_control_successor,
)

if TYPE_CHECKING:
    from cayu.runtime._browser_control_frames import PrivateBrowserFrame


def browser_allocation_digest(allocation: BrowserControlAllocation) -> str:
    owned = BrowserControlAllocation.model_validate(allocation)
    return sha256(
        canonical_durable_json_bytes(owned.model_dump(mode="json"), "browser allocation")
    ).hexdigest()


def _message(raw: Any) -> dict[str, Any]:
    if type(raw) is not str or len(raw) > 65536:
        raise BrowserControlConflict("Browser guest handshake is malformed.")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise BrowserControlConflict("Browser guest handshake is malformed.")
            result[key] = value
        return result

    value = json.loads(raw, object_pairs_hook=unique)
    if type(value) is not dict:
        raise BrowserControlConflict("Browser guest handshake is malformed.")
    return value


def _protected_boundary_audit(
    coordinator: BrowserControlCoordinator, raw: Any, *, identity: BrowserControlIdentity
) -> BrowserControlPageAudit:
    audit = BrowserControlPageAudit.model_validate(raw)
    return audit.model_copy(
        update={
            "locations": tuple(
                item.model_copy(
                    update={
                        "origin": coordinator.protect_page_origin(item.origin, identity=identity)
                    }
                )
                for item in audit.locations
            )
        }
    )


@dataclass(frozen=True, slots=True, repr=False)
class BoundBrowserGuest:
    record: BrowserControlRecord
    channel_id: str
    binding_sha256: str


@dataclass(slots=True, repr=False)
class _PendingView:
    principal: BrowserControlPrincipal
    operator_session_id: str
    page: BrowserControlPage
    until_ms: int
    result: asyncio.Future[PrivateBrowserFrame]
    dispatched: bool = False


class BrowserGuestCommandOwner:
    """Serialize control commands/replies on one authenticated guest connection.

    The route owns this object's lifetime. It does not create background readers
    or dispatch another command while publication remains unresolved.
    """

    def __init__(
        self,
        *,
        coordinator: BrowserControlCoordinator,
        connection: Any,
        bound: BoundBrowserGuest,
        suspend_viewer_delivery: Callable[[BrowserControlIdentity], Awaitable[None]] | None = None,
    ):
        self.coordinator = coordinator
        self.connection = connection
        self.bound = bound
        self.sequence = 0
        self.expected = bound.record
        self.acquisition_lease: int | None = None
        self.publication_transition: tuple[BrowserControlRecord, BrowserControlRecord] | None = None
        self._input: PendingBrowserText | None = None
        self._suspend_viewer_delivery = suspend_viewer_delivery
        self._view: _PendingView | None = None
        self._closed = False
        self._views_suspended = False
        self._view_idle = asyncio.Event()
        self._view_idle.set()
        self._pages: (
            tuple[BrowserControlPrincipal, str, asyncio.Future[BrowserPageDescriptors]] | None
        ) = None

    def _retain_publication_transition(
        self, source: BrowserControlRecord, successor: BrowserControlRecord
    ) -> None:
        # Retain the complete known publication, not an ID from which teardown
        # would have to reconstruct receipt fields after acknowledgement loss.
        self.publication_transition = (
            BrowserControlRecord.model_validate(source),
            BrowserControlRecord.model_validate(successor),
        )

    async def request_pages(
        self, *, principal: BrowserControlPrincipal, operator_session_id: str
    ) -> BrowserPageDescriptors:
        if self._closed or self._pages is not None:
            raise BrowserControlConflict("Browser page discovery is unavailable.")
        result = asyncio.get_running_loop().create_future()
        self._pages = (
            BrowserControlPrincipal.model_validate(principal),
            operator_session_id,
            result,
        )
        try:
            async with asyncio.timeout(10):
                return await asyncio.shield(result)
        finally:
            result.cancel()

    async def request_text_input(
        self,
        *,
        principal: BrowserControlPrincipal,
        operator_session_id: str,
        intent: BrowserTextInputIntent,
        text: str,
    ) -> BrowserControlRecord:
        intent = BrowserTextInputIntent.model_validate(intent)
        if (
            self._closed
            or self._input is not None
            or type(text) is not str
            or not 1 <= len(text) <= 4096
            or any(char == "\x00" or 0xD800 <= ord(char) <= 0xDFFF for char in text)
            or (intent.input_kind != "text" and text != intent.input_kind)
        ):
            text = ""
            raise BrowserControlConflict("Browser private input is unavailable.")
        entry = PendingBrowserText(
            BrowserControlPrincipal.model_validate(principal),
            operator_session_id,
            BrowserTextInputIntent.model_validate(intent),
            bytearray(text, "utf-8"),
            asyncio.get_running_loop().create_future(),
        )
        text = ""
        self._input = entry
        try:
            async with asyncio.timeout(10):
                return await asyncio.shield(entry.result)
        finally:
            entry.result.cancel()
            if not entry.dispatched:
                entry.clear()
                if self._input is entry:
                    self._input = None

    async def suspend_views(self) -> None:
        """Stop server capture admission and join its in-flight exchange.

        This does not prove browser-side frame purge or permit sensitive input.
        The sensitive-entry owner must additionally settle every delivery owner.
        """
        self._views_suspended = True
        entry = self._view
        if entry is not None:
            if not entry.result.done():
                entry.result.set_exception(
                    BrowserControlConflict("Browser private capture is suspended.")
                )
            if not entry.dispatched:
                self._view = None
                self._view_idle.set()
        async with asyncio.timeout(5):
            await self._view_idle.wait()

    async def request_view(
        self,
        *,
        principal: BrowserControlPrincipal,
        operator_session_id: str,
        page: BrowserControlPage,
        until_ms: int,
    ) -> PrivateBrowserFrame:
        """One pending/in-flight capture. Caller cancellation never cancels capture."""
        if self._closed or self._views_suspended or self._view is not None:
            raise BrowserControlConflict("Browser private capture capacity is unavailable.")
        entry = _PendingView(
            BrowserControlPrincipal.model_validate(principal),
            operator_session_id,
            BrowserControlPage.model_validate(page),
            until_ms,
            asyncio.get_running_loop().create_future(),
        )
        self._view = entry
        self._view_idle.clear()
        try:
            async with asyncio.timeout(5):
                return await asyncio.shield(entry.result)
        finally:
            # A vanished viewer cannot retain a queued frame or cause replay.
            entry.result.cancel()
            if not entry.dispatched and self._view is entry:
                self._view = None
                self._view_idle.set()

    async def _capture_pending_view(self, entry: _PendingView) -> None:
        from cayu.runtime._browser_control_frames import BrowserViewUnavailable
        from cayu.runtime._browser_control_view import capture_authorized_browser_view

        entry.dispatched = True
        try:
            if entry.result.cancelled():
                return
            frame = await capture_authorized_browser_view(
                self,
                principal=entry.principal,
                operator_session_id=entry.operator_session_id,
                page=entry.page,
                until_ms=entry.until_ms,
            )
            if not entry.result.done():
                entry.result.set_result(frame)
        except (
            BrowserControlPermissionDenied,
            BrowserControlRevisionChanged,
            BrowserViewUnavailable,
        ) as error:
            if not entry.result.done():
                entry.result.set_exception(error)
        except BaseException:
            # Fixed caller error; the channel lifecycle retains the actual failure.
            if not entry.result.done():
                entry.result.set_exception(
                    BrowserControlConflict("Browser private capture did not settle.")
                )
            raise
        finally:
            if self._view is entry:
                self._view = None
                self._view_idle.set()

    async def step(self) -> bool | None:
        """Advance one exchange; False means the exact allocation close is durable."""
        if self._closed:
            raise BrowserControlConflict("Browser guest channel is closed.")
        _, current = await self.coordinator._load_channel_record(self.bound.record.identity)
        if current == closed_browser_control_successor(self.bound.record):
            self.bound = BoundBrowserGuest(
                current, self.bound.channel_id, self.bound.binding_sha256
            )
            self.expected = current
            return False
        if current.state == "closed":
            raise BrowserControlConflict("Browser channel lost its exact close transition.")
        if current != self.bound.record:
            previous = self.bound.record
            if (
                previous.state == "agent_controlled"
                and previous.fresh_observation_required
                and current
                == previous.model_copy(
                    update={
                        "revision": previous.revision + 1,
                        "fresh_observation_required": False,
                    }
                )
            ):
                # The model terminal and this control successor committed together.
                # No guest command or new lease is created by adopting the readback.
                self.bound = BoundBrowserGuest(
                    current, self.bound.channel_id, self.bound.binding_sha256
                )
                self.expected = current
                return
            self.sequence += 1
            if current.pending_lease_until_ms is not None:
                expected_pending = self.bound.record.model_copy(
                    update={
                        "revision": self.bound.record.revision + 1,
                        "pending_lease_until_ms": current.pending_lease_until_ms,
                    }
                )
                if current != expected_pending:
                    raise BrowserControlConflict("Browser renewal changed its source owner.")
                self.expected = current
                self.bound = await renew_browser_guest_control(
                    coordinator=self.coordinator,
                    connection=self.connection,
                    bound=self.bound,
                    pending=current,
                    sequence=self.sequence,
                )
                self.expected = self.bound.record
                return
            if current.sensitive_entry_pending:
                expected_pending = self.bound.record.model_copy(
                    update={
                        "revision": self.bound.record.revision + 1,
                        "sensitive_entry_pending": True,
                        "capture_restricted": True,
                    }
                )
                if current != expected_pending:
                    raise BrowserControlConflict("Browser sensitive entry changed its owner.")
                self.expected = current
                await self.suspend_views()
                if self._suspend_viewer_delivery is None:
                    raise BrowserControlConflict("Browser viewer settlement is unavailable.")
                await self._suspend_viewer_delivery(current.identity)
                self.bound = await enter_sensitive_browser_guest_control(
                    coordinator=self.coordinator,
                    connection=self.connection,
                    bound=self.bound,
                    pending=current,
                    sequence=self.sequence,
                )
                self.expected = self.bound.record
                return
            if (
                current.state == "handback_pending"
                and self.bound.record.state == "operator_controlled"
                and current.request == self.bound.record.request
                and current.revision == self.bound.record.revision + 1
                and current.control_epoch == self.bound.record.control_epoch
            ):
                self.expected = current
                self.bound = await handback_browser_guest_control(
                    coordinator=self.coordinator,
                    connection=self.connection,
                    bound=self.bound,
                    pending=current,
                    sequence=self.sequence,
                    retain_publication=self._retain_publication_transition,
                )
                self.expected = self.bound.record
                self.publication_transition = None
                return
            if (
                current.state != "takeover_requested"
                or current.request is None
                or current.request.expected_record_revision != self.bound.record.revision
            ):
                raise BrowserControlConflict("Browser channel control changed without its owner.")
            # Capture the pending generation before dispatch, so teardown cannot
            # attempt to fence the preceding agent-controlled revision.
            self.expected = current
            self.acquisition_lease = self.coordinator._acquisition_lease_until(current)
            self.bound = await acquire_browser_guest_control(
                coordinator=self.coordinator,
                connection=self.connection,
                bound=self.bound,
                pending=current,
                sequence=self.sequence,
                lease_until_ms=self.acquisition_lease,
                retain_publication=self._retain_publication_transition,
            )
            self.expected = self.bound.record
            self.publication_transition = None
            self.acquisition_lease = None
            return
        if self._input is not None:
            entry = self._input
            try:
                await dispatch_browser_text(self, entry)
            finally:
                self._input = None
            return
        if self._view is not None:
            await self._capture_pending_view(self._view)
            return
        if self._pages is not None:
            principal, operator_session_id, result = self._pages
            try:
                if not result.cancelled():
                    pages = await read_browser_pages(
                        self, principal=principal, operator_session_id=operator_session_id
                    )
                    if not result.done():
                        result.set_result(pages)
            except (BrowserControlPermissionDenied, BrowserControlRevisionChanged) as error:
                # Page permission is checked before the guest exchange. A
                # caller-local denial must not fence the shared command owner.
                if not result.done():
                    result.set_exception(error)
            except BaseException:
                if not result.done():
                    result.set_exception(BrowserControlConflict("Browser page discovery failed."))
                raise
            finally:
                self._pages = None
            return
        self.sequence += 1
        common = {
            "sequence": self.sequence,
            "channel_id": self.bound.channel_id,
            "worker_instance": current.identity.worker_instance_id,
            "binding_sha256": self.bound.binding_sha256,
        }
        async with asyncio.timeout(5):
            await self.connection.send(json.dumps({"kind": "status", **common}))
            reply = _message(await self.connection.recv())
        expected = {
            **common,
            "kind": "settled",
            "control_epoch": current.control_epoch,
            "state": current.state,
            "settled_sequence": current.settled_input_sequence,
            "pending_sequence": current.pending_input_sequence,
            "fresh_observation_required": current.fresh_observation_required,
        }
        from cayu.runtime._browser_control_evidence import browser_read_evidence_matches

        if not browser_read_evidence_matches(reply, expected):
            raise BrowserControlConflict("Browser guest status changed without its owner.")

    async def disconnect(self, *, cancellation: asyncio.CancelledError | None = None) -> None:
        self._closed = True
        if self._pages is not None and not self._pages[2].done():
            self._pages[2].set_exception(
                BrowserControlConflict("Browser page discovery disconnected.")
            )
        if self._input is not None:
            self._input.clear()
            if not self._input.result.done():
                self._input.result.set_exception(
                    BrowserControlConflict("Browser input disconnected.")
                )
        if self._view is not None and not self._view.result.done():
            self._view.result.set_exception(
                BrowserControlConflict("Browser guest channel disconnected.")
            )
        task = asyncio.current_task()
        consumed = 0
        if cancellation is not None and task is not None:
            before = task.cancelling()
            consume_pending_task_cancellation(cancellation)
            consumed = before - task.cancelling()
        try:
            if self.publication_transition is not None:
                source, successor = self.publication_transition
                await self.coordinator._fence_owned_transition(expected=source, successor=successor)
            elif self.acquisition_lease is not None:
                await self.coordinator._fence_guest_acquisition(
                    expected=self.expected, lease_until_ms=self.acquisition_lease
                )
            elif self.expected.sensitive_entry_pending:
                await self.coordinator._fence_guest_sensitive_entry(expected=self.expected)
            elif self.expected.pending_lease_until_ms is not None:
                await self.coordinator._fence_owned_transition(
                    expected=self.expected,
                    successor=self.coordinator._renewal_successor(self.expected),
                )
            elif self.expected.state == "handback_pending":
                await self.coordinator._fence_guest_handback(expected=self.expected)
            else:
                await self.coordinator._fence_idle_channel(expected=self.expected)
        finally:
            if consumed:
                restore_task_cancellation_requests(consumed, cancellation=cancellation)


async def renew_browser_guest_control(
    *,
    coordinator: BrowserControlCoordinator,
    connection: Any,
    bound: BoundBrowserGuest,
    pending: BrowserControlRecord,
    sequence: int,
) -> BoundBrowserGuest:
    pending = BrowserControlRecord.model_validate(pending)
    if (
        pending.request is None
        or bound.record.pending_lease_until_ms is not None
        or pending
        != bound.record.model_copy(
            update={
                "revision": bound.record.revision + 1,
                "pending_lease_until_ms": pending.pending_lease_until_ms,
            }
        )
    ):
        raise BrowserControlConflict("Browser renewal differs from its channel.")
    await coordinator._authorize_pending_renewal(pending)
    common = {
        "sequence": sequence,
        "channel_id": bound.channel_id,
        "worker_instance": pending.identity.worker_instance_id,
        "binding_sha256": bound.binding_sha256,
    }
    async with asyncio.timeout(5):
        await connection.send(
            json.dumps(
                {
                    **common,
                    "kind": "renew",
                    "request_id": pending.request.request_id,
                    "epoch": pending.control_epoch,
                    "lease_until_ms": pending.pending_lease_until_ms,
                }
            )
        )
        reply = _message(await connection.recv())
    expected = {
        **common,
        "kind": "settled",
        "state": pending.state,
        "control_epoch": pending.control_epoch,
        "settled_sequence": pending.settled_input_sequence,
        "pending_sequence": None,
        "fresh_observation_required": pending.fresh_observation_required,
        "lease_until_ms": pending.pending_lease_until_ms,
    }
    if canonical_durable_json_bytes(reply, "browser renewal") != canonical_durable_json_bytes(
        expected, "browser renewal"
    ):
        raise BrowserControlConflict("Browser renewal acknowledgement differs.")
    returned = await coordinator._publish_guest_renewal(expected=pending)
    return BoundBrowserGuest(returned, bound.channel_id, bound.binding_sha256)


async def enter_sensitive_browser_guest_control(
    *,
    coordinator: BrowserControlCoordinator,
    connection: Any,
    bound: BoundBrowserGuest,
    pending: BrowserControlRecord,
    sequence: int,
) -> BoundBrowserGuest:
    pending = BrowserControlRecord.model_validate(pending)
    if (
        pending.request is None
        or pending.identity != bound.record.identity
        or not pending.sensitive_entry_pending
    ):
        raise BrowserControlConflict("Browser sensitive entry differs from its channel.")
    common = {
        "sequence": sequence,
        "channel_id": bound.channel_id,
        "worker_instance": pending.identity.worker_instance_id,
        "binding_sha256": bound.binding_sha256,
    }
    async with asyncio.timeout(5):
        await connection.send(
            json.dumps(
                {
                    **common,
                    "kind": "sensitive",
                    "request_id": pending.request.request_id,
                    "epoch": pending.control_epoch,
                }
            )
        )
        reply = _message(await connection.recv())
    generation = reply.pop("capture_generation", None)
    if type(generation) is not int or not 1 <= generation <= 2**63 - 1:
        raise BrowserControlConflict("Browser capture settlement generation is malformed.")
    expected = {
        **common,
        "kind": "settled",
        "state": pending.state,
        "control_epoch": pending.control_epoch,
        "settled_sequence": pending.settled_input_sequence,
        "pending_sequence": None,
        "fresh_observation_required": pending.fresh_observation_required,
        "sensitive_entry": True,
        "capture_restricted": True,
    }
    if canonical_durable_json_bytes(
        reply, "browser sensitive entry"
    ) != canonical_durable_json_bytes(expected, "browser sensitive entry"):
        raise BrowserControlConflict("Browser sensitive entry acknowledgement differs.")
    returned = await coordinator._publish_guest_sensitive_entry(expected=pending)
    return BoundBrowserGuest(returned, bound.channel_id, bound.binding_sha256)


async def handback_browser_guest_control(
    *,
    coordinator: BrowserControlCoordinator,
    connection: Any,
    bound: BoundBrowserGuest,
    pending: BrowserControlRecord,
    sequence: int,
    retain_publication: Callable[[BrowserControlRecord, BrowserControlRecord], None],
) -> BoundBrowserGuest:
    pending = BrowserControlRecord.model_validate(pending)
    if (
        pending.request is None
        or pending.state != "handback_pending"
        or pending.identity != bound.record.identity
    ):
        raise BrowserControlConflict("Browser handback differs from its channel.")
    common = {
        "sequence": sequence,
        "channel_id": bound.channel_id,
        "worker_instance": pending.identity.worker_instance_id,
        "binding_sha256": bound.binding_sha256,
    }
    async with asyncio.timeout(5):
        await connection.send(
            json.dumps(
                {
                    **common,
                    "kind": "handback",
                    "request_id": pending.request.request_id,
                    "epoch": pending.control_epoch,
                }
            )
        )
        reply = _message(await connection.recv())
    audit = _protected_boundary_audit(
        coordinator, reply.pop("audit", None), identity=pending.identity
    )
    expected = {
        **common,
        "kind": "settled",
        "state": "agent_controlled",
        "control_epoch": pending.control_epoch + 1,
        "settled_sequence": pending.pending_input_sequence or pending.settled_input_sequence,
        "pending_sequence": None,
        "fresh_observation_required": True,
    }
    if canonical_durable_json_bytes(reply, "browser handback") != canonical_durable_json_bytes(
        expected, "browser handback"
    ):
        raise BrowserControlConflict("Browser guest handback acknowledgement differs.")
    retain_publication(pending, coordinator._handback_successor(pending, audit=audit))
    returned = await coordinator._publish_guest_handback(expected=pending, audit=audit)
    return BoundBrowserGuest(returned, bound.channel_id, bound.binding_sha256)


async def acquire_browser_guest_control(
    *,
    coordinator: BrowserControlCoordinator,
    connection: Any,
    bound: BoundBrowserGuest,
    pending: BrowserControlRecord,
    sequence: int,
    lease_until_ms: int,
    retain_publication: Callable[[BrowserControlRecord, BrowserControlRecord], None] | None = None,
) -> BoundBrowserGuest:
    """One serialized channel exchange, never an operator/public input entrance.

    The channel owner must not issue another command until this exchange and its
    durable publication settle. Failure closes/fences the channel rather than
    replaying native work on a second connection.
    """
    pending = BrowserControlRecord.model_validate(pending)
    request = pending.request
    if (
        request is None
        or pending.state != "takeover_requested"
        or pending.identity != bound.record.identity
        or request.expected_record_revision != bound.record.revision
        or pending.control_epoch != bound.record.control_epoch
        or type(sequence) is not int
        or not 1 <= sequence < 2**53
        or type(lease_until_ms) is not int
        or not request.requested_at_ms < lease_until_ms <= request.maximum_until_ms
    ):
        raise BrowserControlConflict("Browser takeover differs from its channel authority.")
    _, current = await coordinator._load(pending.identity)
    if current != pending:
        raise BrowserControlConflict("Browser takeover is no longer pending.")
    digest = sha256(
        canonical_durable_json_bytes(request.model_dump(mode="json"), "browser takeover")
    ).hexdigest()
    common = {
        "channel_id": bound.channel_id,
        "worker_instance": pending.identity.worker_instance_id,
        "binding_sha256": bound.binding_sha256,
        "sequence": sequence,
    }
    audit = None
    try:
        async with asyncio.timeout(5):
            await connection.send(
                json.dumps(
                    {
                        **common,
                        "kind": "takeover",
                        "request_id": request.request_id,
                        "request_sha256": digest,
                        "expected_epoch": pending.control_epoch,
                        "expires_at_ms": request.expires_at_ms,
                        "maximum_until_ms": request.maximum_until_ms,
                        "lease_until_ms": lease_until_ms,
                        "pages": [page.model_dump(mode="json") for page in request.pages],
                    }
                )
            )
            reply = _message(await connection.recv())
        proposed_audit = _protected_boundary_audit(
            coordinator, reply.pop("audit", None), identity=pending.identity
        )
        expected = {
            **common,
            "kind": "settled",
            "request_id": request.request_id,
            "request_sha256": digest,
            "lease_until_ms": lease_until_ms,
            "control_epoch": pending.control_epoch + 1,
            "state": "operator_controlled",
            "settled_sequence": pending.settled_input_sequence,
            "pending_sequence": None,
            "fresh_observation_required": pending.fresh_observation_required,
        }
        if canonical_durable_json_bytes(
            reply, "browser acquisition"
        ) != canonical_durable_json_bytes(expected, "browser acquisition"):
            raise BrowserControlConflict("Browser acquisition acknowledgement differs.")
        successor = coordinator._acquisition_successor(
            pending, lease_until_ms=lease_until_ms, audit=proposed_audit
        )
        audit = proposed_audit
        if retain_publication is not None:
            retain_publication(pending, successor)
        acquired = await coordinator._publish_guest_acquisition(
            expected=pending, lease_until_ms=lease_until_ms, audit=audit
        )
    except BaseException as primary:
        task = asyncio.current_task()
        consumed = 0
        if isinstance(primary, asyncio.CancelledError) and task is not None:
            before = task.cancelling()
            consume_pending_task_cancellation(primary)
            consumed = before - task.cancelling()
        try:
            await coordinator._fence_guest_acquisition(
                expected=pending, lease_until_ms=lease_until_ms, audit=audit
            )
        except BaseException as cleanup:
            if isinstance(primary, asyncio.CancelledError):
                raise primary from cleanup
            raise BaseExceptionGroup(
                "Browser acquisition and fencing failed.", [primary, cleanup]
            ) from None
        finally:
            if consumed and isinstance(primary, asyncio.CancelledError):
                restore_task_cancellation_requests(consumed, cancellation=primary)
        raise
    return BoundBrowserGuest(acquired, bound.channel_id, bound.binding_sha256)


async def bind_browser_guest_channel(
    *, coordinator: BrowserControlCoordinator, allocation: BrowserControlAllocation, connection: Any
) -> BoundBrowserGuest:
    allocation = BrowserControlAllocation.model_validate(allocation)
    scope_digest = browser_allocation_digest(allocation)
    async with asyncio.timeout(5):
        hello = _message(await connection.recv())
    if (
        set(hello)
        != {
            "kind",
            "schema_version",
            "scope_sha256",
            "channel_id",
            "browser_session_id",
            "worker_instance",
        }
        or hello["kind"] != "hello"
        or type(hello["schema_version"]) is not int
        or hello["schema_version"] != 1
        or hello["scope_sha256"] != scope_digest
        or hello["browser_session_id"] != allocation.browser_session_id
        or type(hello["channel_id"]) is not str
        or re.fullmatch(r"bc_[0-9a-f]{32}", hello["channel_id"]) is None
        or type(hello["worker_instance"]) is not str
        or re.fullmatch(r"vw_[0-9a-f]{32}", hello["worker_instance"]) is None
    ):
        raise BrowserControlConflict("Browser guest differs from its authenticated allocation.")
    record = await coordinator.bind_guest(
        allocation=allocation, worker_instance_id=hello["worker_instance"]
    )
    binding = sha256(
        canonical_durable_json_bytes(record.identity.model_dump(mode="json"), "browser binding")
    ).hexdigest()
    expected = {
        "kind": "bound",
        "channel_id": hello["channel_id"],
        "worker_instance": record.identity.worker_instance_id,
        "binding_sha256": binding,
        "control_epoch": record.control_epoch,
        "state": record.state,
        "settled_sequence": record.settled_input_sequence,
        "pending_sequence": record.pending_input_sequence,
        "fresh_observation_required": record.fresh_observation_required,
    }
    try:
        async with asyncio.timeout(5):
            await connection.send(
                json.dumps(
                    {
                        "kind": "bind",
                        "schema_version": 1,
                        "scope_sha256": scope_digest,
                        "channel_id": hello["channel_id"],
                        "worker_instance": hello["worker_instance"],
                        "binding_sha256": binding,
                    }
                )
            )
            reply = _message(await connection.recv())
        if canonical_durable_json_bytes(reply, "browser handshake") != canonical_durable_json_bytes(
            expected, "browser handshake"
        ):
            raise BrowserControlConflict(
                "Browser guest did not acknowledge the exact durable fence."
            )
    except BaseException as primary:
        try:
            await coordinator._fence_idle_channel(expected=record)
        except BaseException as cleanup:
            if isinstance(primary, asyncio.CancelledError):
                raise primary from cleanup
            raise BaseExceptionGroup(
                "Browser binding acknowledgement and fencing failed.", [primary, cleanup]
            ) from None
        raise
    return BoundBrowserGuest(record, hello["channel_id"], binding)
