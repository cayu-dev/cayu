"""Browser-control publication inside the existing session transaction.

The server/model owner supplies already-authorized, content-free material. This
boundary binds it to the live invocation and atomically publishes the control
root and its acknowledgement receipt. A receipt is readback evidence, never an
input capability or proof that guest work has stopped.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from cayu.runtime._browser_control_checkpoint import (
    BrowserControlCheckpointMutation,
    browser_control_checkpoint_mutation_scope,
    browser_control_receipt,
    browser_control_receipt_key,
)
from cayu.runtime.browser_control import (
    BrowserControlConflict,
    BrowserControlIdentity,
    BrowserControlRecord,
)
from cayu.runtime.checkpoints import (
    BROWSER_CONTROLS_CHECKPOINT_KEY,
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
)
from cayu.runtime.execution_profiles import active_invocation_execution_profile_from_checkpoint
from cayu.runtime.sessions import (
    Session,
    SessionOperationPublication,
    SessionStatus,
    _invocation_lifecycle_authority_read_scope,
)


def validate_browser_control_invocation(
    session: Session, checkpoint: dict[str, Any] | None, identity: BrowserControlIdentity
) -> None:
    active = active_invocation_execution_profile_from_checkpoint(checkpoint)
    if (
        session.id != identity.session_id
        or session.instance_id != identity.session_instance_id
        or session.run_epoch != identity.run_epoch
        or session.environment_name != identity.environment_name
        or session.status is not SessionStatus.RUNNING
        or active is None
        or active.session_id != identity.session_id
        or active.run_epoch != identity.run_epoch
        or active.interaction_id != identity.interaction_id
        or active.profile.fingerprint != identity.execution_profile_fingerprint
    ):
        raise BrowserControlConflict("Browser control lost its active invocation authority.")


@dataclass(frozen=True)
class BrowserControlPublication:
    mutation: BrowserControlCheckpointMutation

    def __post_init__(self) -> None:
        mutation = self.mutation
        object.__setattr__(
            self,
            "mutation",
            BrowserControlCheckpointMutation(
                mutation.session_id, mutation.expected, mutation.desired
            ),
        )

    @property
    def owned_mutation(self) -> BrowserControlCheckpointMutation:
        mutation = self.mutation
        return BrowserControlCheckpointMutation(
            mutation.session_id, mutation.expected, mutation.desired
        )

    @property
    def changed_record(self) -> BrowserControlRecord:
        mutation = self.owned_mutation
        old = () if mutation.expected is None else mutation.expected.records
        return next(record for record in mutation.desired.records if record not in old)

    @property
    def storage_key(self) -> str:
        return browser_control_receipt_key(self.owned_mutation)

    def receipt(self) -> dict[str, Any]:
        return browser_control_receipt(self.owned_mutation)

    @contextmanager
    def scope(self) -> Iterator[None]:
        with (
            browser_control_checkpoint_mutation_scope(self.mutation),
            _invocation_lifecycle_authority_read_scope(),
        ):
            yield

    def validate_commit_time(self, store_now: datetime) -> None:
        if store_now.tzinfo is None or store_now.utcoffset() is None:
            raise BrowserControlConflict("Browser control requires an aware store clock.")
        delta = store_now.astimezone(UTC) - datetime(1970, 1, 1, tzinfo=UTC)
        now_ms = (delta.days * 86400 + delta.seconds) * 1000 + delta.microseconds // 1000
        if now_ms < 0:
            raise BrowserControlConflict("Browser control requires a valid store clock.")
        record = self.changed_record
        request = record.request
        if request is not None and record.state in {"takeover_requested", "operator_controlled"}:
            until = (
                request.expires_at_ms
                if record.state == "takeover_requested"
                else record.lease_until_ms
            )
            if until is None or not request.requested_at_ms <= now_ms < until:
                raise BrowserControlConflict("Browser control authority expired before commit.")

    def validate_invocation(self, session: Session, checkpoint: dict[str, Any] | None) -> None:
        validate_browser_control_invocation(session, checkpoint, self.changed_record.identity)

    @property
    def expected_statuses(self) -> set[SessionStatus] | None:
        return {SessionStatus.RUNNING}

    @property
    def expected_run_epoch(self) -> int | None:
        return self.changed_record.identity.run_epoch

    def transform(
        self,
        session: Session,
        checkpoint: dict[str, Any] | None,
        current_receipt: dict[str, Any] | None,
        store_now: datetime,
    ) -> SessionOperationPublication:
        self.validate_invocation(session, checkpoint)
        if current_receipt is not None:
            # Readback is separate and read-only. Never replay a historical
            # control command into a newer generation, even with a matching ID.
            raise BrowserControlConflict("Browser control publication already has a receipt.")
        self.validate_commit_time(store_now)
        result = {} if checkpoint is None else dict(checkpoint)
        result[CHECKPOINT_SCHEMA_VERSION_KEY] = CURRENT_CHECKPOINT_SCHEMA_VERSION
        result[BROWSER_CONTROLS_CHECKPOINT_KEY] = self.owned_mutation.desired.model_dump(
            mode="json"
        )
        return SessionOperationPublication(
            checkpoint=result, operation_records={self.storage_key: self.receipt()}
        )


@dataclass(frozen=True)
class BrowserControlFencePublication(BrowserControlPublication):
    """Exact teardown only, including after its invocation has ended.

    Never grants control or proves native quiescence. The checkpoint mutation
    scope still requires the complete expected root at the storage transaction.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        mutation = self.owned_mutation
        desired = self.changed_record
        source = next(
            (
                record
                for record in (() if mutation.expected is None else mutation.expected.records)
                if record.identity == desired.identity
            ),
            None,
        )
        if (
            source is None
            or source.state in {"closed", "allocation_lost", "control_uncertain"}
            or desired
            != source.model_copy(
                update={"revision": source.revision + 1, "state": "control_uncertain"}
            )
        ):
            raise BrowserControlConflict(
                "Browser teardown requires an exact uncertainty transition."
            )

    def validate_invocation(self, session: Session, checkpoint: dict[str, Any] | None) -> None:
        identity = self.changed_record.identity
        if (
            session.id != identity.session_id
            or session.instance_id != identity.session_instance_id
            or session.run_epoch < identity.run_epoch
        ):
            raise BrowserControlConflict("Browser teardown lost its owning session generation.")

    @property
    def expected_statuses(self) -> set[SessionStatus] | None:
        return None

    @property
    def expected_run_epoch(self) -> int | None:
        return None
