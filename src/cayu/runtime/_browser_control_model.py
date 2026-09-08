"""Model admission against operator control in the browser publication transaction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from cayu._validation import canonical_durable_json_bytes
from cayu.browser_profiles import BrowserProfileCheckpointConsentDenied
from cayu.core.tools import _RuntimeBrowserControlAdmission
from cayu.runtime._browser_control_checkpoint import BrowserControlCheckpointMutation
from cayu.runtime.browser_control import (
    BrowserControlAllocation,
    BrowserControlCheckpoint,
    BrowserControlConflict,
    closed_browser_control_successor,
)
from cayu.runtime.checkpoints import BROWSER_CONTROLS_CHECKPOINT_KEY

if TYPE_CHECKING:
    from cayu.runtime.sessions import Session


def browser_terminal_checkpoint_mutation(
    checkpoint: dict[str, Any] | None,
    *,
    session_id: str,
    operation_records: Mapping[str, dict[str, Any]],
    operation_name: object,
) -> BrowserControlCheckpointMutation | None:
    """Prepare an exact mutation for the built-in's terminal publication owner.

    This is not a public result-adoption seam. The caller must publish the same
    records and validate invocation authority inside the existing store transaction.
    """
    if operation_name not in {"observe", "close"} or checkpoint is None:
        return None
    if BROWSER_CONTROLS_CHECKPOINT_KEY not in checkpoint:
        return None
    operations = [
        r for r in operation_records.values() if r.get("record_type") == "cayu.browser-operation"
    ]
    if len(operations) != 1:
        raise BrowserControlConflict("Browser terminal lacks exact operation evidence.")
    operation = operations[0]
    if operation.get("state") != "terminal" or operation.get("operation") != operation_name:
        return None
    controls = BrowserControlCheckpoint.model_validate(checkpoint[BROWSER_CONTROLS_CHECKPOINT_KEY])
    record = next(
        (
            r
            for r in controls.records
            if r.identity.browser_session_id == operation.get("browser_session_id")
        ),
        None,
    )
    if record is None or record.state not in {"agent_controlled", "takeover_requested"}:
        return None
    if (
        type(operation.get("invocation_control_epoch")) is not int
        or operation["invocation_control_epoch"] != record.control_epoch
    ):
        return None
    result = operation.get("result")
    if type(result) is not dict or result.get("is_error") is not False:
        return None
    structured = result.get("structured")
    if operation_name == "close":
        if (
            type(structured) is not dict
            or structured.get("session_id") != record.identity.browser_session_id
            or structured.get("closed") is not True
            or structured.get("allocation_disposition") != "retired"
            or operation.get("close_confirmed") is not True
        ):
            return None
        desired = closed_browser_control_successor(record)
        if desired is None:
            return None
        return BrowserControlCheckpointMutation(
            session_id, controls, controls.replace_record(expected=record, desired=desired)
        )
    if record.state != "agent_controlled" or not record.fresh_observation_required:
        return None
    if (
        type(structured) is not dict
        or structured.get("session_id") != record.identity.browser_session_id
        or type(structured.get("revision")) is not str
        or not structured["revision"]
        or structured.get("allocation_disposition") != "live"
        or operation.get("observation_confirmed") is not True
        or (record.capture_restricted and operation.get("observation_protected") is not True)
    ):
        return None
    desired = record.model_copy(
        update={"revision": record.revision + 1, "fresh_observation_required": False}
    )
    return BrowserControlCheckpointMutation(
        session_id, controls, controls.replace_record(expected=record, desired=desired)
    )


def browser_model_control_admission(
    checkpoint: dict[str, Any] | None,
    *,
    allocation: BrowserControlAllocation,
    operation_name: str,
) -> _RuntimeBrowserControlAdmission | None:
    epoch = browser_model_control_epoch(
        checkpoint, allocation=allocation, operation_name=operation_name
    )
    if epoch is None:
        return None
    assert checkpoint is not None
    controls = BrowserControlCheckpoint.model_validate(checkpoint[BROWSER_CONTROLS_CHECKPOINT_KEY])
    record = next(
        item
        for item in controls.records
        if item.identity.browser_session_id == allocation.browser_session_id
    )
    return _RuntimeBrowserControlAdmission(
        epoch, tuple((item.page_id, item.operations) for item in record.operator_page_operations)
    )


def browser_model_control_epoch(
    checkpoint: dict[str, Any] | None,
    *,
    allocation: BrowserControlAllocation,
    operation_name: str,
) -> int | None:
    """Snapshot for guest delivery; the publication transaction rechecks it."""
    allocation = BrowserControlAllocation.model_validate(allocation)
    if checkpoint is None or BROWSER_CONTROLS_CHECKPOINT_KEY not in checkpoint:
        return None
    controls = BrowserControlCheckpoint.model_validate(checkpoint[BROWSER_CONTROLS_CHECKPOINT_KEY])
    record = next(
        (
            item
            for item in controls.records
            if item.identity.browser_session_id == allocation.browser_session_id
        ),
        None,
    )
    if record is None:
        return None
    actual = BrowserControlAllocation.model_validate(
        record.identity.model_dump(exclude={"worker_instance_id"})
    )
    if actual != allocation or record.state != "agent_controlled":
        raise BrowserControlConflict("Browser model dispatch lost control authority.")
    if operation_name == "profile_checkpoint" and (
        record.fresh_observation_required
        or (record.request is not None and record.checkpoint_consent != "allow")
    ):
        raise BrowserProfileCheckpointConsentDenied(
            "Browser profile capture requires consent and a fresh protected observation."
        )
    if record.fresh_observation_required and operation_name not in {"observe", "close"}:
        raise BrowserControlConflict("Browser handback requires a fresh protected observation.")
    return record.control_epoch


def validate_browser_model_publication(
    checkpoint: dict[str, Any] | None,
    *,
    session: Session,
    operation_records: Mapping[str, dict[str, Any]],
    operation_name: object,
) -> None:
    """Called only for the exact registered built-in, never result lookalikes.

    Terminal receipts must remain publishable while takeover waits for settlement.
    New intent and dispatch both require current agent control; an earlier local
    preflight is not authority. Guest delivery independently checks the epoch.
    """

    if checkpoint is None or BROWSER_CONTROLS_CHECKPOINT_KEY not in checkpoint:
        # No control handshake has been published for this invocation. This
        # preserves normal model-only browser operation; it grants no operator
        # authority and cannot be used to reconnect an operator channel.
        return
    controls = BrowserControlCheckpoint.model_validate(checkpoint[BROWSER_CONTROLS_CHECKPOINT_KEY])
    operations = [
        record
        for record in operation_records.values()
        if record.get("record_type") == "cayu.browser-operation"
    ]
    if len(operations) != 1:
        raise BrowserControlConflict("Browser publication lacks exact operation evidence.")
    operation = operations[0]
    if (
        type(operation.get("schema_version")) is not int
        or operation["schema_version"] != 1
        or operation.get("state") not in {"intent", "dispatched", "terminal"}
        or type(operation.get("browser_session_id")) is not str
    ):
        raise BrowserControlConflict("Browser publication contains malformed operation evidence.")
    record = next(
        (
            record
            for record in controls.records
            if record.identity.browser_session_id == operation["browser_session_id"]
        ),
        None,
    )
    if record is None:
        # A new allocation has no operator channel until its guest handshake is
        # independently published. Other controlled allocations remain fenced.
        return
    identity = record.identity
    if (
        identity.session_id != session.id
        or identity.session_instance_id != session.instance_id
        or identity.run_epoch != session.run_epoch
        or operation.get("parent_session_id") != identity.session_id
        or type(operation.get("parent_run_epoch")) is not int
        or operation["parent_run_epoch"] != identity.run_epoch
        or operation.get("execution_profile_fingerprint") != identity.execution_profile_fingerprint
        or operation.get("environment_name") != identity.environment_name
        or operation.get("allocation_fingerprint") != identity.allocation_fingerprint
    ):
        raise BrowserControlConflict("Browser publication lost its exact control generation.")
    if operation["state"] == "terminal":
        return
    if record.state != "agent_controlled":
        raise BrowserControlConflict("Browser input is fenced for operator control.")
    if (
        type(operation.get("invocation_control_epoch")) is not int
        or operation["invocation_control_epoch"] != record.control_epoch
    ):
        raise BrowserControlConflict("Browser dispatch carries a stale control epoch.")
    expected_accounting = [
        [item.page_id, item.operations] for item in record.operator_page_operations
    ]
    if canonical_durable_json_bytes(
        operation.get("operator_page_operations", []), "browser accounting"
    ) != canonical_durable_json_bytes(expected_accounting, "browser accounting"):
        raise BrowserControlConflict("Browser dispatch carries different operator accounting.")
    if record.fresh_observation_required and operation_name not in {"observe", "close"}:
        raise BrowserControlConflict("Browser handback requires a fresh protected observation.")
