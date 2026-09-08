"""Join authenticated application permission to exact durable control admission.

The server supplies authenticated principal provenance, not caller body identity.
This owner does not manufacture ToolContext or dispatch native input. A committed
takeover request only fences model admission; guest quiescence is still required.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from cayu.runtime._browser_control_authorization import (
    AuthorizedBrowserControl,
    BrowserControlPermissionDenied,
    BrowserControlRevisionChanged,
    authorize_browser_control,
)
from cayu.runtime._browser_control_checkpoint import (
    BrowserControlCheckpointMutation,
    browser_control_checkpoint_read_scope,
)
from cayu.runtime._browser_control_publication import (
    BrowserControlFencePublication,
    BrowserControlPublication,
    validate_browser_control_invocation,
)
from cayu.runtime._browser_control_publisher import BrowserControlPublisher
from cayu.runtime._checkpoint_store import (
    load_runtime_session_checkpoint_snapshot,
    runtime_checkpoint_session_store,
)
from cayu.runtime.browser_control import (
    BrowserControlAllocation,
    BrowserControlCheckpoint,
    BrowserControlConflict,
    BrowserControlIdentity,
    BrowserControlPageAudit,
    BrowserControlPolicy,
    BrowserControlPrincipal,
    BrowserControlRecord,
    BrowserHandbackIntent,
    BrowserOperatorPageOperations,
    BrowserOperatorPurpose,
    BrowserRenewIntent,
    BrowserSensitiveEntryIntent,
    BrowserTakeoverIntent,
    BrowserTakeoverRequest,
    BrowserTextInputIntent,
    closed_browser_control_successor,
    request_browser_takeover,
)
from cayu.runtime.checkpoints import BROWSER_CONTROLS_CHECKPOINT_KEY
from cayu.runtime.sessions import SessionStatus, SessionStore
from cayu.vaults.redaction import SecretRedactor


class BrowserControlInvocationEnded(BrowserControlConflict):
    """Exact retained channel outlived its invocation; only teardown is allowed."""


class BrowserControlCoordinator:
    def __init__(
        self,
        *,
        store: SessionStore,
        policy: BrowserControlPolicy | None,
        purpose: BrowserOperatorPurpose,
        redactor: SecretRedactor,
        clock: Callable[[], datetime],
        origin_redactor: Callable[[BrowserControlIdentity], SecretRedactor] | None = None,
    ) -> None:
        self._store = runtime_checkpoint_session_store(store)
        self._policy = policy
        self._purpose = BrowserOperatorPurpose.model_validate(purpose)
        self._redactor = redactor
        self._origin_redactor = origin_redactor
        if any(
            redactor.redact_text(value) != value
            for value in (self._purpose.code, *self._purpose.expected_origins)
        ):
            raise BrowserControlPermissionDenied()
        self._clock = clock
        self._publisher = BrowserControlPublisher(self._store)

    async def bind_guest(
        self, *, allocation: BrowserControlAllocation, worker_instance_id: str
    ) -> BrowserControlRecord:
        """Private bootstrap after allocation-capability authentication, not HTTP identity.

        A changed worker cannot replace an existing live allocation. This is a
        durable binding only: the guest must still acknowledge its exact fence.
        """
        if self._policy is None:
            raise BrowserControlPermissionDenied()
        allocation = BrowserControlAllocation.model_validate(allocation)
        if allocation.operator_purpose != self._purpose:
            raise BrowserControlPermissionDenied()
        identity = BrowserControlIdentity(
            **allocation.model_dump(mode="python"), worker_instance_id=worker_instance_id
        )
        with browser_control_checkpoint_read_scope(identity.session_id):
            session, raw = await load_runtime_session_checkpoint_snapshot(
                self._store, identity.session_id
            )
        validate_browser_control_invocation(session, raw, identity)
        original = (
            None
            if raw is None or BROWSER_CONTROLS_CHECKPOINT_KEY not in raw
            else BrowserControlCheckpoint.model_validate(raw[BROWSER_CONTROLS_CHECKPOINT_KEY])
        )
        controls = original or BrowserControlCheckpoint()
        current = next(
            (
                record
                for record in controls.records
                if record.identity.browser_session_id == identity.browser_session_id
            ),
            None,
        )
        if current is not None:
            if current.identity != identity or current.state in {"closed", "allocation_lost"}:
                raise BrowserControlConflict(
                    "Browser bootstrap conflicts with its admitted worker."
                )
            return current
        desired = BrowserControlRecord(identity=identity)
        return await self._publisher.publish(
            BrowserControlPublication(
                BrowserControlCheckpointMutation(
                    identity.session_id,
                    original,
                    controls.replace_record(expected=None, desired=desired),
                )
            )
        )

    async def mark_channel_uncertain(
        self, *, expected: BrowserControlRecord
    ) -> BrowserControlRecord:
        """Fence an exact channel owner; never infer guest quiescence from disconnect.

        The channel owner retains this publication until it settles. A conflicting
        revision requires reconciliation by that owner, not blind replay against
        newer control authority.
        """
        expected = BrowserControlRecord.model_validate(expected)
        _, _, controls, current = await self._read_record(expected.identity)
        if current == closed_browser_control_successor(expected) or (
            current.state == "closed" and current == expected
        ):
            return current
        if current.state in {"closed", "allocation_lost"}:
            raise BrowserControlConflict("Browser teardown lost its exact terminal generation.")
        desired = expected.model_copy(
            update={"revision": expected.revision + 1, "state": "control_uncertain"}
        )
        if current == desired:
            return current
        if current != expected:
            raise BrowserControlConflict("Browser channel lost its exact control revision.")
        if current.state == "control_uncertain":
            return current
        return await self._publisher.publish(
            BrowserControlFencePublication(
                BrowserControlCheckpointMutation(
                    expected.identity.session_id,
                    controls,
                    controls.replace_record(expected=current, desired=desired),
                )
            )
        )

    async def _fence_idle_channel(self, *, expected: BrowserControlRecord) -> BrowserControlRecord:
        """Private teardown: include exact successors not yet seen by the channel.

        HTTP admission or model observation publication may commit while idle.
        Only exact successors of that owner's complete record may be
        fenced here; no native settlement or permission is inferred from them.
        Keep mark_channel_uncertain's strict contract for all other callers.
        """
        expected = BrowserControlRecord.model_validate(expected)
        _, _, _, current = await self._read_record(expected.identity)
        candidates = [expected]
        if expected.state == "agent_controlled" and expected.fresh_observation_required:
            candidates.append(
                expected.model_copy(
                    update={
                        "revision": expected.revision + 1,
                        "fresh_observation_required": False,
                    }
                )
            )
        if expected.state == "agent_controlled" and current.request is not None:
            request = current.request
            if request.expected_record_revision == expected.revision:
                candidates.append(
                    request_browser_takeover(expected, request, now_ms=request.requested_at_ms)
                )
        elif expected.state == "operator_controlled":
            if not expected.sensitive_entry_pending and expected.pending_lease_until_ms is None:
                candidates.append(
                    expected.model_copy(
                        update={"revision": expected.revision + 1, "state": "handback_pending"}
                    )
                )
                if expected.pending_input_sequence is None:
                    if not expected.sensitive_entry:
                        candidates.append(
                            expected.model_copy(
                                update={
                                    "revision": expected.revision + 1,
                                    "sensitive_entry_pending": True,
                                    "capture_restricted": True,
                                }
                            )
                        )
                    if current.pending_lease_until_ms is not None:
                        candidates.append(
                            expected.model_copy(
                                update={
                                    "revision": expected.revision + 1,
                                    "pending_lease_until_ms": current.pending_lease_until_ms,
                                }
                            )
                        )
        for candidate in candidates:
            fenced = candidate.model_copy(
                update={"revision": candidate.revision + 1, "state": "control_uncertain"}
            )
            if current in (candidate, fenced):
                return await self.mark_channel_uncertain(expected=candidate)
        # Exact closed successors are handled here; unrelated generations still
        # fail and remain with the channel's retained cleanup owner.
        return await self.mark_channel_uncertain(expected=expected)

    async def _publish_guest_acquisition(
        self,
        *,
        expected: BrowserControlRecord,
        lease_until_ms: int,
        audit: BrowserControlPageAudit | None = None,
    ) -> BrowserControlRecord:
        """Private channel owner only, after exact native quiescence acknowledgement.

        A public request or a stored pending record is not evidence of acquisition.
        The channel validates its authenticated reply before entering this seam.
        """
        expected = BrowserControlRecord.model_validate(expected)
        if expected.state != "takeover_requested" or expected.request is None:
            raise BrowserControlConflict("Browser acquisition requires its pending request.")
        now_ms = int(self._clock().timestamp() * 1000)
        if (
            type(lease_until_ms) is not int
            or not now_ms < lease_until_ms <= now_ms + 60_000
            or lease_until_ms > expected.request.maximum_until_ms
            or now_ms >= expected.request.expires_at_ms
        ):
            raise BrowserControlConflict("Browser acquisition lease is unavailable.")
        controls, current = await self._load(expected.identity)
        desired = self._acquisition_successor(expected, lease_until_ms=lease_until_ms, audit=audit)
        if current == desired:
            return current
        if current != expected:
            raise BrowserControlConflict("Browser acquisition lost its exact pending revision.")
        return await self._publisher.publish(
            BrowserControlPublication(
                BrowserControlCheckpointMutation(
                    expected.identity.session_id,
                    controls,
                    controls.replace_record(expected=current, desired=desired),
                )
            )
        )

    def _acquisition_lease_until(self, pending: BrowserControlRecord) -> int:
        pending = BrowserControlRecord.model_validate(pending)
        if pending.state != "takeover_requested" or pending.request is None:
            raise BrowserControlConflict("Browser acquisition requires its pending request.")
        now_ms = int(self._clock().timestamp() * 1000)
        if now_ms >= pending.request.expires_at_ms:
            raise BrowserControlConflict("Browser takeover request expired before acquisition.")
        return min(now_ms + 30_000, pending.request.maximum_until_ms)

    @staticmethod
    def _acquisition_successor(
        expected: BrowserControlRecord,
        *,
        lease_until_ms: int,
        audit: BrowserControlPageAudit | None = None,
    ) -> BrowserControlRecord:
        expected = BrowserControlRecord.model_validate(expected)
        if expected.state != "takeover_requested" or expected.request is None:
            raise BrowserControlConflict("Browser acquisition has no pending owner.")
        return expected.model_copy(
            update={
                "revision": expected.revision + 1,
                "control_epoch": expected.control_epoch + 1,
                "state": "operator_controlled",
                "lease_until_ms": lease_until_ms,
                "acquisition_audit": audit,
            }
        )

    async def _fence_guest_acquisition(
        self,
        *,
        expected: BrowserControlRecord,
        lease_until_ms: int,
        audit: BrowserControlPageAudit | None = None,
    ) -> BrowserControlRecord:
        """Reconcile only this channel's pending/acquired publication on teardown.

        A cancelled writer may commit after its waiter stops. CAS fencing of the
        pending record either wins first or observes the exact acquired successor;
        it must never adopt unrelated newer authority merely by matching an ID.
        """
        expected = BrowserControlRecord.model_validate(expected)
        acquired = self._acquisition_successor(expected, lease_until_ms=lease_until_ms, audit=audit)
        return await self._fence_owned_transition(expected=expected, successor=acquired)

    async def _fence_owned_transition(
        self, *, expected: BrowserControlRecord, successor: BrowserControlRecord
    ) -> BrowserControlRecord:
        candidates = (expected, successor)
        fenced = tuple(
            record.model_copy(
                update={"revision": record.revision + 1, "state": "control_uncertain"}
            )
            for record in candidates
        )
        for attempt in range(2):
            _, _, _, current = await self._read_record(expected.identity)
            if current in tuple(closed_browser_control_successor(item) for item in candidates):
                return current
            if current in fenced:
                return current
            if current not in candidates:
                raise BrowserControlConflict("Browser control teardown lost its exact generation.")
            try:
                return await self.mark_channel_uncertain(expected=current)
            except BrowserControlConflict:
                if attempt:
                    raise
        raise AssertionError("Browser control teardown did not settle.")

    async def _load(
        self, identity: BrowserControlIdentity
    ) -> tuple[BrowserControlCheckpoint, BrowserControlRecord]:
        controls, record = await self._load_record(identity)
        if record.state in {"closed", "allocation_lost"}:
            raise BrowserControlConflict("The live browser allocation has ended.")
        return controls, record

    async def _load_record(
        self, identity: BrowserControlIdentity
    ) -> tuple[BrowserControlCheckpoint, BrowserControlRecord]:
        session, raw, controls, record = await self._read_record(identity)
        if identity.operator_purpose != self._purpose:
            raise BrowserControlPermissionDenied()
        validate_browser_control_invocation(session, raw, identity)
        return controls, record

    async def _load_channel_record(
        self, identity: BrowserControlIdentity
    ) -> tuple[BrowserControlCheckpoint, BrowserControlRecord]:
        """Private channel cleanup may read an exact close after the run ends.

        This grants no live action or publication authority. Every non-closed
        record still requires the current running invocation.
        """
        session, raw, controls, record = await self._read_record(identity)
        if record.state != "closed":
            if session.run_epoch > identity.run_epoch or session.status in {
                SessionStatus.COMPLETED,
                SessionStatus.FAILED,
                SessionStatus.INTERRUPTED,
            }:
                raise BrowserControlInvocationEnded("Browser channel invocation has ended.")
            validate_browser_control_invocation(session, raw, identity)
        return controls, record

    async def bootstrap_retirement_record(
        self, identity: BrowserControlIdentity
    ) -> BrowserControlRecord | None:
        """Prove an ended invocation's fenced bookkeeping is no longer runnable.

        This is not native quiescence evidence and does not release allocation,
        pending input, or viewer ownership. Their durable controls are retained.
        """
        session, _, _, record = await self._read_record(identity)
        if record.state == "closed":
            return record
        if record.state == "control_uncertain" and (
            session.run_epoch > identity.run_epoch
            or session.status
            in {
                SessionStatus.COMPLETED,
                SessionStatus.FAILED,
                SessionStatus.INTERRUPTED,
            }
        ):
            return record
        return None

    async def _read_record(self, identity: BrowserControlIdentity):
        if self._policy is None:
            raise BrowserControlPermissionDenied()
        with browser_control_checkpoint_read_scope(identity.session_id):
            session, raw = await load_runtime_session_checkpoint_snapshot(
                self._store, identity.session_id
            )
        if (
            session.id != identity.session_id
            or session.instance_id != identity.session_instance_id
            or session.run_epoch < identity.run_epoch
        ):
            raise BrowserControlConflict("Browser control lost its owning session generation.")
        if raw is None or BROWSER_CONTROLS_CHECKPOINT_KEY not in raw:
            raise BrowserControlConflict(
                "No admitted live browser control allocation is available."
            )
        controls = BrowserControlCheckpoint.model_validate(raw[BROWSER_CONTROLS_CHECKPOINT_KEY])
        record = next((item for item in controls.records if item.identity == identity), None)
        if record is None:
            raise BrowserControlConflict("The expected live browser allocation is unavailable.")
        return session, raw, controls, record

    def protect_page_origin(
        self, origin: str | None, *, identity: BrowserControlIdentity
    ) -> str | None:
        """Omit an observed origin if workload redaction would change its identity."""
        if origin is not None and self._redactor.redact_text(origin) != origin:
            return None
        if origin is not None and self._origin_redactor is not None:
            redactor = self._origin_redactor(identity)
            if not isinstance(redactor, SecretRedactor):
                raise BrowserControlConflict(
                    "Browser invocation redaction authority is unavailable."
                )
            if redactor.redact_text(origin) != origin:
                return None
        return origin

    async def authorize_view(
        self,
        *,
        principal: BrowserControlPrincipal,
        operator_session_id: str,
        identity: BrowserControlIdentity,
        expected_record_revision: int,
    ) -> AuthorizedBrowserControl:
        principal = BrowserControlPrincipal.model_validate(principal)
        identity = BrowserControlIdentity.model_validate(identity)
        _, record = await self._load(identity)
        if type(expected_record_revision) is not int:
            raise BrowserControlConflict("Browser control changed before view authorization.")
        if record.revision != expected_record_revision:
            raise BrowserControlRevisionChanged(
                "Browser control changed before view authorization."
            )
        return await authorize_browser_control(
            policy=self._policy,
            principal=principal,
            record=record,
            operator_session_id=operator_session_id,
            action="view",
        )

    async def inspect_authorized_browsers(
        self, *, session_id: str, principal: BrowserControlPrincipal, operator_session_id: str
    ) -> tuple[BrowserControlRecord, ...]:
        """Discover exact controls only after per-allocation application permission."""
        principal = BrowserControlPrincipal.model_validate(principal)
        if self._policy is None:
            raise BrowserControlPermissionDenied()
        with browser_control_checkpoint_read_scope(session_id):
            _, raw = await load_runtime_session_checkpoint_snapshot(self._store, session_id)
        if raw is None or BROWSER_CONTROLS_CHECKPOINT_KEY not in raw:
            raise BrowserControlPermissionDenied()
        controls = BrowserControlCheckpoint.model_validate(raw[BROWSER_CONTROLS_CHECKPOINT_KEY])
        visible = []
        for record in controls.records:
            if record.state in {"closed", "allocation_lost"}:
                continue
            try:
                await self.authorize_view(
                    principal=principal,
                    operator_session_id=operator_session_id,
                    identity=record.identity,
                    expected_record_revision=record.revision,
                )
            except BrowserControlPermissionDenied:
                continue
            _, current = await self._load(record.identity)
            if current != record:
                raise BrowserControlConflict("Browser control changed during discovery.")
            visible.append(current)
        if not visible:
            raise BrowserControlPermissionDenied()
        # One final snapshot covers the whole result. Per-item readback alone
        # lets a later policy await invalidate an earlier visible allocation.
        with browser_control_checkpoint_read_scope(session_id):
            session, final_raw = await load_runtime_session_checkpoint_snapshot(
                self._store, session_id
            )
        if final_raw is None or BROWSER_CONTROLS_CHECKPOINT_KEY not in final_raw:
            raise BrowserControlConflict("Browser control changed during discovery.")
        final = BrowserControlCheckpoint.model_validate(final_raw[BROWSER_CONTROLS_CHECKPOINT_KEY])
        for record in visible:
            validate_browser_control_invocation(session, final_raw, record.identity)
            if record not in final.records:
                raise BrowserControlConflict("Browser control changed during discovery.")
        return tuple(visible)

    def _view_grant_until(self) -> int:
        return int(self._clock().timestamp() * 1000) + 10_000

    async def request_takeover(
        self,
        *,
        principal: BrowserControlPrincipal,
        operator_session_id: str,
        intent: BrowserTakeoverIntent,
    ) -> BrowserControlRecord:
        principal = BrowserControlPrincipal.model_validate(principal)
        intent = BrowserTakeoverIntent.model_validate(intent)
        if intent.purpose_code != self._purpose.code:
            raise BrowserControlPermissionDenied()
        controls, record = await self._load(intent.identity)
        authorized = await authorize_browser_control(
            policy=self._policy,
            principal=principal,
            record=record,
            operator_session_id=operator_session_id,
            action="takeover",
        )
        # All values were defensively validated before serialization. Actor
        # authority is constructed here, never copied from the HTTP body.
        request = BrowserTakeoverRequest(
            **intent.model_dump(mode="python"), operator=authorized.operator
        )
        untrusted_text = (
            request.purpose_code,
            request.operator.subject,
            request.operator.tenant,
            request.operator.operator_session_id,
            *(page.page_id for page in request.pages),
            *(page.revision for page in request.pages),
        )
        if any(
            value is not None and self._redactor.redact_text(value) != value
            for value in untrusted_text
        ):
            raise BrowserControlPermissionDenied()
        now = self._clock()
        if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
            raise BrowserControlConflict("Browser control requires an aware application clock.")
        delta = now.astimezone(UTC) - datetime(1970, 1, 1, tzinfo=UTC)
        now_ms = (delta.days * 86400 + delta.seconds) * 1000 + delta.microseconds // 1000
        desired = request_browser_takeover(record, request, now_ms=now_ms)
        if desired == record:
            # Historical request readback does not dispatch, renew, or grant.
            return record
        command = BrowserControlPublication(
            BrowserControlCheckpointMutation(
                intent.identity.session_id,
                controls,
                controls.replace_record(expected=record, desired=desired),
            )
        )
        return await self._publisher.publish(command)

    async def drain(self, *, timeout_s: float = 5.0) -> bool:
        return await self._publisher.drain(timeout_s=timeout_s)

    async def request_renewal(
        self,
        *,
        principal: BrowserControlPrincipal,
        operator_session_id: str,
        intent: BrowserRenewIntent,
    ) -> BrowserControlRecord:
        intent = BrowserRenewIntent.model_validate(intent)
        controls, record = await self._load(intent.identity)
        authorized = await authorize_browser_control(
            policy=self._policy,
            principal=principal,
            record=record,
            operator_session_id=operator_session_id,
            action="renew",
        )
        now_ms = int(self._clock().timestamp() * 1000)
        if (
            record.request is None
            or record.request.request_id != intent.request_id
            or record.request.operator != authorized.operator
            or record.revision != intent.expected_record_revision
            or record.control_epoch != intent.expected_control_epoch
            or record.state != "operator_controlled"
            or record.pending_lease_until_ms is not None
            or record.pending_input_sequence is not None
            or record.sensitive_entry_pending
            or record.lease_until_ms != intent.expected_lease_until_ms
            or not now_ms < intent.expected_lease_until_ms < intent.lease_until_ms
            or intent.lease_until_ms > min(now_ms + 60_000, record.request.maximum_until_ms)
        ):
            raise BrowserControlConflict("Browser renewal differs from its live lease authority.")
        desired = record.model_copy(
            update={
                "revision": record.revision + 1,
                "pending_lease_until_ms": intent.lease_until_ms,
            }
        )
        return await self._publisher.publish(
            BrowserControlPublication(
                BrowserControlCheckpointMutation(
                    intent.identity.session_id,
                    controls,
                    controls.replace_record(expected=record, desired=desired),
                )
            )
        )

    @staticmethod
    def _renewal_successor(expected: BrowserControlRecord) -> BrowserControlRecord:
        expected = BrowserControlRecord.model_validate(expected)
        if expected.state != "operator_controlled" or expected.pending_lease_until_ms is None:
            raise BrowserControlConflict("Browser renewal has no pending owner.")
        return expected.model_copy(
            update={
                "revision": expected.revision + 1,
                "lease_until_ms": expected.pending_lease_until_ms,
                "pending_lease_until_ms": None,
            }
        )

    async def _authorize_pending_renewal(self, expected: BrowserControlRecord) -> None:
        expected = BrowserControlRecord.model_validate(expected)
        self._renewal_successor(expected)
        if expected.request is None or expected.lease_until_ms is None:
            raise BrowserControlConflict("Browser renewal has no owner.")
        operator = expected.request.operator
        authorized = await authorize_browser_control(
            policy=self._policy,
            principal=BrowserControlPrincipal(subject=operator.subject, tenant=operator.tenant),
            record=expected,
            operator_session_id=operator.operator_session_id,
            action="renew",
        )
        _, current = await self._load(expected.identity)
        if (
            authorized.operator != operator
            or current != expected
            or expected.lease_until_ms <= int(self._clock().timestamp() * 1000)
        ):
            raise BrowserControlConflict("Browser renewal lost its live authorized source.")

    async def _publish_guest_renewal(
        self, *, expected: BrowserControlRecord
    ) -> BrowserControlRecord:
        desired = self._renewal_successor(expected)
        controls, current = await self._load(expected.identity)
        if current == desired:
            return current
        if current != expected:
            raise BrowserControlConflict("Browser renewal lost its pending revision.")
        return await self._publisher.publish(
            BrowserControlPublication(
                BrowserControlCheckpointMutation(
                    expected.identity.session_id,
                    controls,
                    controls.replace_record(expected=current, desired=desired),
                )
            )
        )

    async def request_handback(
        self,
        *,
        principal: BrowserControlPrincipal,
        operator_session_id: str,
        intent: BrowserHandbackIntent,
    ) -> BrowserControlRecord:
        intent = BrowserHandbackIntent.model_validate(intent)
        controls, record = await self._load(intent.identity)
        authorized = await authorize_browser_control(
            policy=self._policy,
            principal=principal,
            record=record,
            operator_session_id=operator_session_id,
            action="handback",
        )
        if (
            record.request is None
            or record.request.request_id != intent.request_id
            or record.request.operator != authorized.operator
        ):
            raise BrowserControlConflict("Browser handback differs from its operator authority.")
        if (
            record.state == "handback_pending"
            and record.revision == intent.expected_record_revision + 1
            and record.control_epoch == intent.expected_control_epoch
        ) or (
            record.state == "agent_controlled"
            and record.revision == intent.expected_record_revision + 2
            and record.control_epoch == intent.expected_control_epoch + 1
            and record.fresh_observation_required
        ):
            return record  # Exact readback never dispatches or renews control.
        if (
            record.control_epoch != intent.expected_control_epoch
            or record.revision != intent.expected_record_revision
            or record.state != "operator_controlled"
            or record.sensitive_entry_pending
            or record.pending_lease_until_ms is not None
        ):
            raise BrowserControlConflict("Browser handback differs from its operator authority.")
        desired = record.model_copy(
            update={"revision": record.revision + 1, "state": "handback_pending"}
        )
        return await self._publisher.publish(
            BrowserControlPublication(
                BrowserControlCheckpointMutation(
                    intent.identity.session_id,
                    controls,
                    controls.replace_record(expected=record, desired=desired),
                )
            )
        )

    async def request_sensitive_entry(
        self,
        *,
        principal: BrowserControlPrincipal,
        operator_session_id: str,
        intent: BrowserSensitiveEntryIntent,
    ) -> BrowserControlRecord:
        """Fence capture durably; only subsequent positive settlement grants input."""
        intent = BrowserSensitiveEntryIntent.model_validate(intent)
        controls, record = await self._load(intent.identity)
        authorized = await authorize_browser_control(
            policy=self._policy,
            principal=principal,
            record=record,
            operator_session_id=operator_session_id,
            action="sensitive_entry",
        )
        if (
            record.request is None
            or record.request.request_id != intent.request_id
            or record.request.operator != authorized.operator
            or record.control_epoch != intent.expected_control_epoch
            or record.state != "operator_controlled"
        ):
            raise BrowserControlConflict("Browser sensitive entry differs from its owner.")
        if (
            record.sensitive_entry_pending
            and record.revision == intent.expected_record_revision + 1
        ):
            return record  # Readback is not positive capture or input settlement.
        if (
            record.revision != intent.expected_record_revision
            or record.sensitive_entry_pending
            or record.sensitive_entry
            or record.pending_lease_until_ms is not None
            or record.pending_input_sequence is not None
            or record.lease_until_ms is None
            or record.lease_until_ms <= int(self._clock().timestamp() * 1000)
        ):
            raise BrowserControlConflict("Browser sensitive entry is unavailable.")
        desired = record.model_copy(
            update={
                "revision": record.revision + 1,
                "sensitive_entry_pending": True,
                "capture_restricted": True,
            }
        )
        return await self._publisher.publish(
            BrowserControlPublication(
                BrowserControlCheckpointMutation(
                    intent.identity.session_id,
                    controls,
                    controls.replace_record(expected=record, desired=desired),
                )
            )
        )

    async def _prepare_text_input(
        self,
        *,
        principal: BrowserControlPrincipal,
        operator_session_id: str,
        intent: BrowserTextInputIntent,
    ) -> tuple[BrowserControlCheckpoint, BrowserControlRecord, AuthorizedBrowserControl]:
        """One current policy/authority check shared by tickets and actual admission."""
        intent = BrowserTextInputIntent.model_validate(intent)
        controls, record = await self._load(intent.identity)
        authorized = await authorize_browser_control(
            policy=self._policy,
            principal=principal,
            record=record,
            operator_session_id=operator_session_id,
            action="text_input" if intent.input_kind == "text" else "key_input",
        )
        if (
            record.request is None
            or record.request.request_id != intent.request_id
            or record.request.operator != authorized.operator
            or record.revision != intent.expected_record_revision
            or record.control_epoch != intent.expected_control_epoch
            or record.state != "operator_controlled"
            or not record.sensitive_entry
            or record.pending_lease_until_ms is not None
            or record.sensitive_entry_pending
            or record.pending_input_sequence is not None
            or intent.input_sequence != record.settled_input_sequence + 1
            or record.lease_until_ms is None
            or record.lease_until_ms <= int(self._clock().timestamp() * 1000)
        ):
            raise BrowserControlConflict("Browser text input differs from its live authority.")
        return controls, record, authorized

    async def _admit_text_input(
        self,
        *,
        principal: BrowserControlPrincipal,
        operator_session_id: str,
        intent: BrowserTextInputIntent,
    ) -> BrowserControlRecord:
        """Reserve once before private dispatch; never persist text or its digest."""
        _, publication = await self._prepare_text_input_admission(
            principal=principal, operator_session_id=operator_session_id, intent=intent
        )
        return await self._publish_text_input_admission(publication)

    async def _prepare_text_input_admission(
        self,
        *,
        principal: BrowserControlPrincipal,
        operator_session_id: str,
        intent: BrowserTextInputIntent,
    ) -> tuple[BrowserControlRecord, BrowserControlPublication]:
        """Final authorization and immutable CAS preparation; no publication."""
        intent = BrowserTextInputIntent.model_validate(intent)
        controls, record, _ = await self._prepare_text_input(
            principal=principal, operator_session_id=operator_session_id, intent=intent
        )
        desired = self._input_admission_successor(record, intent)
        return desired, BrowserControlPublication(
            BrowserControlCheckpointMutation(
                intent.identity.session_id,
                controls,
                controls.replace_record(expected=record, desired=desired),
            )
        )

    async def _publish_text_input_admission(
        self, publication: BrowserControlPublication
    ) -> BrowserControlRecord:
        return await self._publisher.publish(publication)

    @staticmethod
    def _input_admission_successor(
        record: BrowserControlRecord, intent: BrowserTextInputIntent
    ) -> BrowserControlRecord:
        """Exact shape shared by admission and its cancellation reconciliation owner."""
        record = BrowserControlRecord.model_validate(record)
        intent = BrowserTextInputIntent.model_validate(intent)
        if (
            record.identity != intent.identity
            or record.state != "operator_controlled"
            or record.request is None
            or record.request.request_id != intent.request_id
            or record.revision != intent.expected_record_revision
            or record.control_epoch != intent.expected_control_epoch
            or record.pending_input_sequence is not None
            or intent.input_sequence != record.settled_input_sequence + 1
        ):
            raise BrowserControlConflict("Browser input accounting differs from its owner.")
        counts = {item.page_id: item.operations for item in record.operator_page_operations}
        counts[intent.page.page_id] = counts.get(intent.page.page_id, 0) + 1
        return record.model_copy(
            update={
                "revision": record.revision + 1,
                "pending_input_sequence": intent.input_sequence,
                "pending_input_page": intent.page,
                "pending_input_kind": intent.input_kind,
                "manual_mutation_uncertain": True,
                "operator_page_operations": tuple(
                    BrowserOperatorPageOperations(page_id=page_id, operations=count)
                    for page_id, count in sorted(counts.items())
                ),
            }
        )

    @staticmethod
    def _input_successor(expected: BrowserControlRecord) -> BrowserControlRecord:
        expected = BrowserControlRecord.model_validate(expected)
        if (
            expected.state != "operator_controlled"
            or expected.pending_input_sequence is None
            or expected.pending_input_page is None
        ):
            raise BrowserControlConflict("Browser input settlement has no exact owner.")
        return expected.model_copy(
            update={
                "revision": expected.revision + 1,
                "settled_input_sequence": expected.pending_input_sequence,
                "pending_input_sequence": None,
                "pending_input_page": None,
                "pending_input_kind": None,
                "manual_mutation_uncertain": False,
            }
        )

    async def _publish_guest_input(self, *, expected: BrowserControlRecord) -> BrowserControlRecord:
        """Private serialized input owner after its exact native acknowledgement."""
        desired = self._input_successor(expected)
        controls, current = await self._load(expected.identity)
        if current == desired:
            return current
        if current != expected:
            raise BrowserControlConflict("Browser input lost its pending revision.")
        return await self._publisher.publish(
            BrowserControlPublication(
                BrowserControlCheckpointMutation(
                    expected.identity.session_id,
                    controls,
                    controls.replace_record(expected=current, desired=desired),
                )
            )
        )

    @staticmethod
    def _sensitive_successor(expected: BrowserControlRecord) -> BrowserControlRecord:
        expected = BrowserControlRecord.model_validate(expected)
        if expected.state != "operator_controlled" or not expected.sensitive_entry_pending:
            raise BrowserControlConflict("Browser sensitive entry has no pending owner.")
        return expected.model_copy(
            update={
                "revision": expected.revision + 1,
                "sensitive_entry_pending": False,
                "sensitive_entry": True,
            }
        )

    async def _publish_guest_sensitive_entry(
        self, *, expected: BrowserControlRecord
    ) -> BrowserControlRecord:
        """Private owner after positive server, viewer and guest capture settlement."""
        desired = self._sensitive_successor(expected)
        controls, current = await self._load(expected.identity)
        if current == desired:
            return current
        if current != expected:
            raise BrowserControlConflict("Browser sensitive entry lost its pending revision.")
        return await self._publisher.publish(
            BrowserControlPublication(
                BrowserControlCheckpointMutation(
                    expected.identity.session_id,
                    controls,
                    controls.replace_record(expected=current, desired=desired),
                )
            )
        )

    async def _fence_guest_sensitive_entry(
        self, *, expected: BrowserControlRecord
    ) -> BrowserControlRecord:
        return await self._fence_owned_transition(
            expected=expected, successor=self._sensitive_successor(expected)
        )

    @staticmethod
    def _handback_successor(
        expected: BrowserControlRecord, *, audit: BrowserControlPageAudit | None = None
    ) -> BrowserControlRecord:
        expected = BrowserControlRecord.model_validate(expected)
        if expected.state != "handback_pending" or expected.request is None:
            raise BrowserControlConflict("Browser handback has no pending owner.")
        return expected.model_copy(
            update={
                "revision": expected.revision + 1,
                "state": "agent_controlled",
                "control_epoch": expected.control_epoch + 1,
                "lease_until_ms": None,
                "pending_input_sequence": None,
                "pending_input_page": None,
                "pending_input_kind": None,
                "sensitive_entry": False,
                "settled_input_sequence": expected.pending_input_sequence
                or expected.settled_input_sequence,
                "fresh_observation_required": True,
                "handback_audit": audit,
            }
        )

    async def _fence_guest_handback(
        self, *, expected: BrowserControlRecord
    ) -> BrowserControlRecord:
        expected = BrowserControlRecord.model_validate(expected)
        return await self._fence_owned_transition(
            expected=expected, successor=self._handback_successor(expected)
        )

    async def _publish_guest_handback(
        self, *, expected: BrowserControlRecord, audit: BrowserControlPageAudit | None = None
    ) -> BrowserControlRecord:
        """Private channel owner after positive input settlement and invalidation."""
        expected = BrowserControlRecord.model_validate(expected)
        desired = self._handback_successor(expected, audit=audit)
        controls, current = await self._load(expected.identity)
        if current == desired:
            return current
        if current != expected:
            raise BrowserControlConflict("Browser handback lost its exact pending revision.")
        return await self._publisher.publish(
            BrowserControlPublication(
                BrowserControlCheckpointMutation(
                    expected.identity.session_id,
                    controls,
                    controls.replace_record(expected=current, desired=desired),
                )
            )
        )
