"""Exact recovery selection is restrictive input, never a serialized grant."""

import asyncio
import warnings

import pytest

from cayu import CayuApp, ProducerRecoveryExpectation
from cayu.collaboration._contracts import CollaborationContractError
from cayu.collaboration.access import CollaborationAccessContext
from cayu.sessions import SessionStatus
from cayu.sessions.recovery import (
    RecoveryExecutionRequest,
    RecoveryPlanBounds,
    RecoveryPlanRequest,
    RecoveryPlanSelection,
)
from cayu.vaults.redaction import SecretRedactor


@pytest.mark.parametrize("entrance", ["plan", "execute"])
@pytest.mark.parametrize(
    "field",
    [
        "producer",
        "participant_context",
        "continuation",
        "selection",
        "selection-statuses",
        "selection-cursor",
        "selection-inactive_for_seconds",
        "bounds",
        "bounds-item_limit",
    ],
)
def test_nested_recovery_selection_rejects_before_copy_or_serialization(
    entrance, field, capsys, caplog
):
    canary = "private-recovery-selection-canary-94612"
    copies = []

    class Unsafe:
        def __repr__(self):
            return canary

        def __deepcopy__(self, memo):
            copies.append(True)
            raise RuntimeError(canary)

    async def scenario():
        app = CayuApp(enable_logging=False, secret_redactor=SecretRedactor(canary))
        baseline = await app.plan_recovery(
            RecoveryPlanRequest(
                selection=RecoveryPlanSelection(statuses=frozenset({SessionStatus.RUNNING}))
            )
        )
        producer = ProducerRecoveryExpectation(
            session_instance_id="original-incarnation",
            attachment_operation_key="original-attachment",
            attachment_commitment="sha256:" + "a" * 64,
        )
        context = CollaborationAccessContext(principal="operator")
        request = RecoveryPlanRequest(
            selection=RecoveryPlanSelection(session_ids=("original-session",)),
            producer=producer,
            participant_context=context,
        )
        malformed = (
            producer.model_copy(update={"session_instance_id": Unsafe()})
            if field == "producer"
            else request.selection.model_copy(update={"session_ids": (Unsafe(),)})
            if field == "selection"
            else Unsafe()
            if field in {"continuation", "bounds"}
            else context.model_copy(update={"principal": Unsafe()})
        )
        if "-" in field:
            container, member = field.split("-", 1)
            value = (Unsafe(),) if member == "statuses" else Unsafe()
            malformed = getattr(request, container).model_copy(update={member: value})
            field_name = container
        else:
            field_name = field
        request = request.model_copy(update={field_name: malformed})
        execution = RecoveryExecutionRequest(plan=baseline, execution_id="test-selection")
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            with pytest.raises(CollaborationContractError) as caught:
                if entrance == "plan":
                    await app.plan_recovery(request)
                else:
                    await app.execute_recovery(
                        execution.model_copy(
                            update={"plan": baseline.model_copy(update={"request": request})}
                        )
                    )
        assert not observed
        assert not copies
        assert canary not in str(caught.value)
        assert canary not in repr(caught.value)
        assert caught.value.__cause__ is None and caught.value.__context__ is None

    asyncio.run(scenario())
    captured = capsys.readouterr()
    assert canary not in captured.out + captured.err + caplog.text


def test_recovery_snapshot_preserves_large_valid_selection():
    """Recovery's documented page bounds are larger than collaboration envelopes."""
    from cayu.runtime._producer_recovery_selection import prepare_recovery_selection

    identities = tuple(f"session-{index}-" + "x" * 400 for index in range(1000))
    request = RecoveryPlanRequest(
        selection=RecoveryPlanSelection(session_ids=identities),
        bounds=RecoveryPlanBounds(item_limit=1000, inspection_limit=10000),
    )
    snapshot = prepare_recovery_selection(request, redactor=SecretRedactor())
    assert snapshot == request
    assert snapshot is not request
    assert snapshot.selection is not request.selection
    assert snapshot.bounds is not request.bounds
