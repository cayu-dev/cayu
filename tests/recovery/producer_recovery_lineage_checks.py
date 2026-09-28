"""Adversarial native-readback fault injection through public producer settlement."""

from unittest.mock import patch

import pytest
from tests.core.test_participant_identity import CONTEXT

from cayu.collaboration._producer_store import read_output_registration
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.runtime._invocation_lifecycle import (
    InvocationLifecycleCommandKind,
    _invocation_lifecycle_receipt_ledger_from_checkpoint,
    _InvocationLifecycleReceiptLedger,
)
from cayu.sessions.checkpoints import INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY
from cayu.storage import _producer_observation


async def require_missing_rebind_refusal(application, command):
    store, initialized = application._participant_coordinator._ready()

    async def retained():
        async with store._transaction(initialized.owner.application_scope, write=False) as tx:
            return await read_output_registration(
                tx, command, redactor=application._secret_redactor
            )

    before = await retained()
    assert before.cleanup is None
    project = _producer_observation._project
    observations = []

    def without_rebind(expected, kind, session, checkpoint, attachment, *, attachment_only=False):
        if attachment_only:
            return project(expected, kind, session, checkpoint, attachment, attachment_only=True)
        ledger = _invocation_lifecycle_receipt_ledger_from_checkpoint(checkpoint)
        assert any(item.kind is InvocationLifecycleCommandKind.REBIND for item in ledger.receipts)
        receipts = tuple(
            item for item in ledger.receipts if item.kind is InvocationLifecycleCommandKind.RELEASE
        )
        assert len(receipts) < len(ledger.receipts)
        assert any(item.kind is InvocationLifecycleCommandKind.RELEASE for item in receipts)
        # A release-only ledger remains structurally valid. It must not substitute
        # for the admission/rebind lineage joining original and recovered owners.
        incomplete = _InvocationLifecycleReceiptLedger(receipts=receipts)
        snapshot = {
            **checkpoint,
            INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY: incomplete.model_dump(mode="json"),
        }
        observations.append(True)
        return project(expected, kind, session, snapshot, attachment)

    with (
        patch.object(_producer_observation, "_project", without_rebind),
        pytest.raises(CollaborationUnavailable) as caught,
    ):
        await application.settle_producer_output(command, context=CONTEXT)
    if not observations:
        caught.value.add_note("Settlement refused before the injected native lineage read.")
        raise caught.value
    assert await retained() == before
