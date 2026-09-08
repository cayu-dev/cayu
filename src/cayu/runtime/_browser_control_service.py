"""Finite owner joining private runner delivery to authenticated guest binding."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from cayu.core.tools import ToolContext, _runtime_tool_invocation_authority
from cayu.runtime._browser_control_bootstrap import BrowserGuestBootstrap
from cayu.runtime._browser_control_channel import (
    BoundBrowserGuest,
    BrowserGuestCommandOwner,
    browser_allocation_digest,
)
from cayu.runtime._browser_control_channels import BrowserGuestChannels
from cayu.runtime._browser_control_coordinator import BrowserControlCoordinator
from cayu.runtime._browser_control_frames import PrivateBrowserFrame
from cayu.runtime._browser_control_pages import BrowserPageDescriptors
from cayu.runtime._browser_viewer_delivery import BrowserViewerDelivery
from cayu.runtime._invocation_secrets import InvocationPublicationSnapshot
from cayu.runtime.browser_control import (
    BrowserControlAllocation,
    BrowserControlConflict,
    BrowserControlIdentity,
    BrowserControlPage,
    BrowserControlPrincipal,
    BrowserControlRecord,
    BrowserOperatorPurpose,
    BrowserPagesIntent,
    BrowserTextInputIntent,
)
from cayu.tools._browser_control_transport import validate_control_endpoint
from cayu.tools.browser_session import _RunnerBrowserSessionBackend
from cayu.vaults.redaction import SecretRedactor


@dataclass(slots=True, repr=False)
class _BootstrapRetirement:
    coordinator: BrowserControlCoordinator
    identity: BrowserControlIdentity
    task: asyncio.Task[BrowserControlRecord | None] | None = None


@dataclass(slots=True, repr=False)
class _BootstrapOwner:
    allocation: BrowserControlAllocation
    backend: _RunnerBrowserSessionBackend
    bound: asyncio.Future[BoundBrowserGuest]
    delivery: asyncio.Task[tuple[BaseException, ...]]
    origin_redactor: SecretRedactor | None = None
    commands: BrowserGuestCommandOwner | None = None
    retirement: _BootstrapRetirement | None = None


class BrowserControlBootstrapPending(RuntimeError):
    """The exact retained owner has not completed; never dispatch a replacement."""


class BrowserControlService:
    def __init__(self, *, guest_endpoint: str, purpose: BrowserOperatorPurpose) -> None:
        self._endpoint = validate_control_endpoint(guest_endpoint)
        self._purpose = BrowserOperatorPurpose.model_validate(purpose)
        self._capabilities = BrowserGuestBootstrap()
        self._owners: dict[str, _BootstrapOwner] = {}
        self._closed = False
        self.channels = BrowserGuestChannels()
        self._viewers: set[BrowserViewerDelivery] = set()
        self._suspended_viewer_identities: list[BrowserControlIdentity] = []

    def invocation_origin_redactor(self, identity: BrowserControlIdentity) -> SecretRedactor:
        allocation = BrowserControlAllocation.model_validate(
            identity.model_dump(mode="python", exclude={"worker_instance_id"})
        )
        owner = self._owners.get(browser_allocation_digest(allocation))
        if (
            owner is None
            or owner.allocation != allocation
            or not owner.bound.done()
            or owner.bound.cancelled()
            or owner.bound.exception() is not None
            or owner.bound.result().record.identity != identity
            or owner.origin_redactor is None
        ):
            raise BrowserControlConflict("Browser invocation redaction authority is unavailable.")
        return owner.origin_redactor

    @property
    def purpose(self) -> BrowserOperatorPurpose:
        return BrowserOperatorPurpose.model_validate(self._purpose)

    def register_viewer(self, identity: BrowserControlIdentity) -> BrowserViewerDelivery:
        identity = BrowserControlIdentity.model_validate(identity)
        if (
            self._closed
            or len(self._viewers) >= 32
            or identity in self._suspended_viewer_identities
        ):
            raise BrowserControlConflict("Browser viewer capacity is unavailable.")
        viewer = BrowserViewerDelivery(identity)
        self._viewers.add(viewer)
        return viewer

    def retire_viewer(self, viewer: BrowserViewerDelivery) -> None:
        viewer.closed = True
        if not viewer.may_hold_frames and (viewer.purge_token is None or viewer.purge_settled):
            self._viewers.discard(viewer)
        for key, owner in tuple(self._owners.items()):
            retirement = owner.retirement
            if retirement is None or retirement.identity != viewer.identity:
                continue
            task = retirement.task
            if (
                task is not None
                and task.done()
                and not task.cancelled()
                and task.exception() is None
                and task.result() is not None
            ):
                self._retire_owner_after_delivery(key, owner, retirement.identity)

    async def suspend_viewer_delivery(self, identity: BrowserControlIdentity) -> None:
        identity = BrowserControlIdentity.model_validate(identity)
        if identity not in self._suspended_viewer_identities:
            if len(self._suspended_viewer_identities) >= 32:
                raise BrowserControlConflict("Browser viewer suspension capacity is unavailable.")
            self._suspended_viewer_identities.append(identity)
        viewers = tuple(viewer for viewer in self._viewers if viewer.identity == identity)
        for viewer in viewers:
            viewer.request_purge()
        # No coroutine cancellation can stand in for a viewer acknowledgement.
        # A closed unacknowledged viewer remains a bounded retained owner.
        await asyncio.gather(*(viewer.wait_purged() for viewer in viewers))

    def authenticate_guest(self, credential: str) -> BrowserControlAllocation:
        return self._capabilities.consume(credential)

    def attach_commands(self, commands: BrowserGuestCommandOwner) -> None:
        allocation = BrowserControlAllocation.model_validate(
            commands.bound.record.identity.model_dump(exclude={"worker_instance_id"})
        )
        owner = self._owners.get(browser_allocation_digest(allocation))
        if (
            self._closed
            or owner is None
            or owner.commands is not None
            or not owner.bound.done()
            or owner.bound.cancelled()
            or owner.bound.result() != commands.bound
        ):
            raise BrowserControlConflict("Browser guest command owner differs from its bootstrap.")
        owner.commands = commands

    def detach_commands(self, commands: BrowserGuestCommandOwner) -> None:
        # The channel route calls this only after its durable fence and socket
        # cleanup settle. Disconnect alone does not prove browser retirement.
        for key, owner in tuple(self._owners.items()):
            if owner.commands is commands:
                owner.commands = None
                owner.retirement = _BootstrapRetirement(
                    commands.coordinator, commands.bound.record.identity
                )
                self._start_retirement(key, owner)

    def _start_retirement(self, key: str, owner: _BootstrapOwner) -> None:
        retirement = owner.retirement
        if retirement is None or (retirement.task is not None and not retirement.task.done()):
            return
        if retirement.task is not None and not retirement.task.cancelled():
            failure = retirement.task.exception()
            if failure is not None and not isinstance(failure, Exception):
                raise failure
        if (
            retirement.task is not None
            and not retirement.task.cancelled()
            and retirement.task.exception() is None
            and retirement.task.result() is not None
        ):
            # Positive proof is already owned while opaque delivery settles.
            return
        task = asyncio.create_task(
            retirement.coordinator.bootstrap_retirement_record(retirement.identity),
            name="cayu-browser-bootstrap-retirement",
        )
        retirement.task = task

        def completed(result: asyncio.Task[BrowserControlRecord | None]) -> None:
            # Observe failures but retain their task and exact owner for retry.
            # This read cannot revoke the channel's already-settled teardown.
            if result.cancelled() or result.exception() is not None:
                return
            record = result.result()
            if record is not None and retirement.task is result:
                self._retire_owner_after_delivery(key, owner, record.identity)

        task.add_done_callback(completed)

    async def settle_bootstrap_retirements(self, *, timeout_s: float = 5.0) -> bool:
        """Retry retained bookkeeping, without releasing uncertain native authority."""
        if type(timeout_s) not in {int, float} or not 0 < timeout_s <= 30:
            raise ValueError("Browser retirement timeout must be positive and bounded.")
        tasks = set()
        for key, owner in tuple(self._owners.items()):
            if owner.retirement is not None:
                self._start_retirement(key, owner)
                assert owner.retirement.task is not None
                tasks.add(owner.retirement.task)
        if not tasks:
            return True
        done, pending = await asyncio.wait(tasks, timeout=timeout_s)
        settled = not pending
        for task in done:
            if task.cancelled():
                settled = False
            else:
                failure = task.exception()
                if failure is not None:
                    if not isinstance(failure, Exception):
                        raise failure
                    settled = False
        return settled

    def _retire_owner_after_delivery(
        self, key: str, owner: _BootstrapOwner, identity: BrowserControlIdentity
    ) -> None:
        if not owner.delivery.done():
            owner.delivery.add_done_callback(
                lambda _: self._retire_owner_after_delivery(key, owner, identity)
            )
            return
        if self._owners.get(key) is owner and owner.commands is None:
            viewers = tuple(viewer for viewer in self._viewers if viewer.identity == identity)
            if any(not viewer.closed for viewer in viewers):
                # Keep the exact positive retirement proof until every viewer
                # route has finished. Its bounded idle/lifetime owner still runs.
                return
            # This invocation can no longer accept input or capture. Retiring its
            # closed viewer bookkeeping does not acknowledge pixels, permit a
            # sensitive transition, or release the durable uncertain browser.
            for viewer in viewers:
                self._viewers.discard(viewer)
            del self._owners[key]
            # This exact browser is closed or its invocation ended while fenced,
            # and bootstrap delivery has settled. No future capture can be
            # admitted by this command owner. Removing this obsolete admission
            # marker does not acknowledge viewer pixels.
            if identity in self._suspended_viewer_identities:
                self._suspended_viewer_identities.remove(identity)

    async def discover_pages(
        self,
        *,
        principal: BrowserControlPrincipal,
        operator_session_id: str,
        intent: BrowserPagesIntent,
    ) -> BrowserPageDescriptors:
        intent = BrowserPagesIntent.model_validate(intent)
        commands = self._connected_commands(intent.identity)
        expected = commands.bound.record
        if expected.revision != intent.expected_record_revision:
            raise BrowserControlConflict("Browser page discovery has a stale revision.")
        pages = await commands.request_pages(
            principal=principal, operator_session_id=operator_session_id
        )
        _, current = await commands.coordinator._load(intent.identity)
        if current != expected:
            raise BrowserControlConflict("Browser page discovery changed its revision.")
        return pages

    async def capture_view(
        self,
        *,
        identity: BrowserControlIdentity,
        principal: BrowserControlPrincipal,
        operator_session_id: str,
        page: BrowserControlPage,
        until_ms: int,
    ) -> PrivateBrowserFrame:
        """Authenticated operator transport supplies principal; policy is not bypassed."""
        commands = self._connected_commands(identity)
        return await commands.request_view(
            principal=principal,
            operator_session_id=operator_session_id,
            page=page,
            until_ms=until_ms,
        )

    async def submit_text_input(
        self,
        *,
        principal: BrowserControlPrincipal,
        operator_session_id: str,
        intent: BrowserTextInputIntent,
        text: str,
    ) -> BrowserControlRecord:
        """Private operator transport only; native admission still checks policy."""
        try:
            intent = BrowserTextInputIntent.model_validate(intent)
            commands = self._connected_commands(intent.identity)
            return await commands.request_text_input(
                principal=principal,
                operator_session_id=operator_session_id,
                intent=intent,
                text=text,
            )
        finally:
            text = ""

    def _connected_commands(self, identity: BrowserControlIdentity) -> BrowserGuestCommandOwner:
        identity = BrowserControlIdentity.model_validate(identity)
        allocation = BrowserControlAllocation.model_validate(
            identity.model_dump(exclude={"worker_instance_id"})
        )
        owner = self._owners.get(browser_allocation_digest(allocation))
        if (
            self._closed
            or owner is None
            or owner.commands is None
            or owner.commands.bound.record.identity != identity
        ):
            raise BrowserControlConflict("Browser guest is not connected to this control service.")
        return owner.commands

    def confirm_guest(self, bound: BoundBrowserGuest) -> None:
        allocation = BrowserControlAllocation.model_validate(
            bound.record.identity.model_dump(exclude={"worker_instance_id"})
        )
        owner = self._owners.get(browser_allocation_digest(allocation))
        if self._closed or owner is None or owner.allocation != allocation or owner.bound.done():
            raise BrowserControlConflict("Browser guest has no pending bootstrap owner.")
        owner.bound.set_result(bound)

    async def bootstrap(
        self,
        context: ToolContext,
        *,
        backend: _RunnerBrowserSessionBackend,
        browser_session_id: str,
        arguments: dict,
        timeout_s: float = 15.0,
    ) -> BoundBrowserGuest:
        if type(timeout_s) not in {int, float} or not 0 < timeout_s <= 30:
            raise ValueError("Browser bootstrap timeout must be positive and at most 30 seconds.")
        if self._closed:
            raise BrowserControlBootstrapPending("Browser bootstrap owner is closing.")
        if type(backend) is not _RunnerBrowserSessionBackend:
            raise BrowserControlConflict("Browser control requires the built-in runner backend.")
        allocation = self._capabilities.allocation_for_invocation(
            context,
            browser_session_id=browser_session_id,
            arguments=arguments,
            purpose=self.purpose,
        )
        authority = _runtime_tool_invocation_authority(context)
        if authority is None:
            raise BrowserControlConflict("Browser invocation redaction authority is unavailable.")
        snapshot = authority.secret_publication_sealer()
        if (
            type(snapshot) is not InvocationPublicationSnapshot
            or snapshot.unsafe_output is not False
            or snapshot.secret_scope_incomplete is not False
            or not isinstance(snapshot.redactor, SecretRedactor)
        ):
            raise BrowserControlConflict("Browser invocation redaction scope is incomplete.")
        key = browser_allocation_digest(allocation)
        owner = self._owners.get(key)
        if owner is not None:
            if (
                owner.allocation != allocation
                or owner.backend is not backend
                or owner.origin_redactor is None
            ):
                raise BrowserControlConflict("Browser bootstrap conflicts with its original owner.")
            # Preserve prior browser-call secrets and adopt this sealed scope
            # before any suspension permits a control-plane origin projection.
            owner.origin_redactor = owner.origin_redactor.merged_with(snapshot.redactor)
        await self.settle_bootstrap_retirements(timeout_s=min(timeout_s, 5.0))
        owner = self._owners.get(key)
        if owner is None:
            if self._closed or len(self._owners) >= 32:
                raise BrowserControlBootstrapPending("Browser bootstrap capacity is unavailable.")
            credential = self._capabilities.issue(allocation)

            async def deliver(token: str) -> tuple[BaseException, ...]:
                try:
                    await backend.bootstrap_control(
                        context,
                        browser_session_id=browser_session_id,
                        endpoint=self._endpoint,
                        credential=token,
                        scope_sha256=key,
                    )
                    return ()
                except BaseException as failure:
                    return (failure,)
                finally:
                    token = ""

            owner = _BootstrapOwner(
                allocation,
                backend,
                asyncio.get_running_loop().create_future(),
                asyncio.create_task(deliver(credential), name="cayu-browser-bootstrap-delivery"),
                origin_redactor=snapshot.redactor,
            )
            self._owners[key] = owner
            credential = ""
        elif (
            owner.allocation != allocation
            or owner.backend is not backend
            or owner.origin_redactor is None
        ):
            raise BrowserControlConflict("Browser bootstrap conflicts with its original owner.")
        else:
            # Another bootstrap may have installed this exact owner during
            # retirement settlement. Its scope cannot replace this caller's.
            owner.origin_redactor = owner.origin_redactor.merged_with(snapshot.redactor)
        # Neither caller timeout nor cancellation cancels an opaque runner call.
        try:
            async with asyncio.timeout(timeout_s):
                await asyncio.shield(owner.delivery)
                return await asyncio.shield(owner.bound)
        except TimeoutError:
            raise BrowserControlBootstrapPending("Browser bootstrap remains owned.") from None

    async def drain_bootstrap_deliveries(self, *, timeout_s: float = 5.0) -> bool:
        """Stop issuance and join delivery only; this does not close guest channels."""
        self._closed = True
        self._capabilities.close()
        pending = {owner.delivery for owner in self._owners.values() if not owner.delivery.done()}
        if pending:
            _, pending = await asyncio.wait(pending, timeout=timeout_s)
        return not pending

    async def drain(self, *, timeout_s: float = 5.0) -> bool:
        """Stop issuance, join bootstrap delivery and fence live channel owners."""
        self._closed = True
        self._capabilities.close()
        channels_settled = await self.channels.drain(timeout_s=timeout_s)
        deliveries_settled = await self.drain_bootstrap_deliveries(timeout_s=timeout_s)
        retirements_settled = await self.settle_bootstrap_retirements(timeout_s=timeout_s)
        return channels_settled and deliveries_settled and retirements_settled
