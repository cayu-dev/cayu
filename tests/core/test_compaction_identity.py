"""The two compaction entrances must preserve distinct dispatch authority."""

from __future__ import annotations

import pytest
from tests.core.test_recovery_composition import _run_without_owners

from cayu.context.base import _COMPACTION_ATTEMPT_ID_KEY
from cayu.execution_units import new_model_step_identity
from cayu.runtime._compaction.identity import _CompactionExecutionIdentityLedger


def test_compaction_parts_import_without_execution_owners() -> None:
    _run_without_owners(
        (
            "cayu.applications",
            "cayu.runtime._session_engine",
            "cayu.runtime._model_step_executor",
            "cayu.runtime._model_completion_recovery",
        ),
        """
from cayu.runtime._compaction.automatic import AutomaticCompaction, AutomaticCompactionRun
from cayu.runtime._compaction.explicit import SessionCompaction
from cayu.runtime._compaction.identity import _CompactionExecutionIdentityLedger
from cayu.runtime._compaction.recovery import reconcile_completed_stage
from cayu.execution_units import new_model_step_identity

identity = new_model_step_identity().new_attempt()
ledger = _CompactionExecutionIdentityLedger(new_model_step_identity())
assert ledger.begin_dispatch(identity) == identity
ledger.end_dispatch(identity)
""",
    )


@pytest.mark.parametrize("independent_step", [False, True], ids=["explicit", "automatic"])
def test_compaction_completion_preserves_issued_step_and_replays_after_dispatch(
    independent_step: bool,
) -> None:
    parent = new_model_step_identity()
    dispatch = (new_model_step_identity() if independent_step else parent).new_attempt()
    ledger = _CompactionExecutionIdentityLedger(parent)
    assert ledger.begin_dispatch(dispatch) == dispatch
    original = {_COMPACTION_ATTEMPT_ID_KEY: "compaction-call"}
    identified = ledger.identify_payloads([original])[0]
    assert identified == {**original, **dispatch.payload()}
    assert original == {_COMPACTION_ATTEMPT_ID_KEY: "compaction-call"}
    ledger.end_dispatch(dispatch)

    assert ledger.identify_payloads([identified]) == [identified]
    assert (identified["model_step_id"] != parent.model_step_id) is independent_step


def test_compaction_rejects_unissued_completion_identity() -> None:
    ledger = _CompactionExecutionIdentityLedger(new_model_step_identity())
    issued = ledger.begin_dispatch(new_model_step_identity().new_attempt())
    forged = {
        _COMPACTION_ATTEMPT_ID_KEY: "compaction-call",
        **new_model_step_identity().new_attempt().payload(),
    }
    with pytest.raises(ValueError, match="was not issued"):
        ledger.identify_payloads([forged])
    ledger.end_dispatch(issued)


def test_compaction_rejects_conflicting_completion_for_one_dispatch() -> None:
    ledger = _CompactionExecutionIdentityLedger(new_model_step_identity())
    issued = ledger.begin_dispatch(new_model_step_identity().new_attempt())
    ledger.identify_payloads([{_COMPACTION_ATTEMPT_ID_KEY: "first-call"}])
    with pytest.raises(ValueError, match="conflicting completion identities"):
        ledger.identify_payloads([{_COMPACTION_ATTEMPT_ID_KEY: "second-call"}])
    ledger.end_dispatch(issued)


def test_compaction_cannot_rebind_a_completion_to_another_dispatch() -> None:
    ledger = _CompactionExecutionIdentityLedger(new_model_step_identity())
    first = ledger.begin_dispatch(new_model_step_identity().new_attempt())
    payload = {_COMPACTION_ATTEMPT_ID_KEY: "compaction-call"}
    ledger.identify_payloads([payload])
    ledger.end_dispatch(first)
    second = ledger.begin_dispatch(new_model_step_identity().new_attempt())
    with pytest.raises(ValueError, match="conflicts with its provider dispatch"):
        ledger.identify_payloads([payload])
    ledger.end_dispatch(second)
