"""Close ambiguous execution slots without replaying candidate or scoring work."""

from datetime import UTC, datetime

from cayu.evals.memory_attribution import (
    EvalMemoryAttributionEvidenceV1,
    EvalMemoryEvidenceLimitation,
)
from cayu.evals.models import EvalOutcome, EvalStatus, EvalTrialResult
from cayu.evals.result_contract import (
    EvalTrialDiagnosticCode,
    EvalTrialOutputPreviewV1,
    _EvalTrialPublicData,
)
from cayu.evals.runner import _blocked_assertion_results


def block_uncheckpointed_trials(compiled, completed_trials):
    """Retain exact completed checkpoints; every other slot stays unscored."""

    retained = dict(completed_trials)
    now = datetime.now(UTC)
    message = "Recovery requires a new, explicitly authorized execution attempt."
    memory = EvalMemoryAttributionEvidenceV1.unavailable(EvalMemoryEvidenceLimitation.MISSING)
    for case in compiled.suite.cases:
        for trial_number in range(1, compiled.trials + 1):
            key = (case.id, trial_number)
            if key in retained:
                continue
            retained[key] = (
                EvalTrialResult(
                    trial_number=trial_number,
                    status=EvalStatus.UNAVAILABLE,
                    assertions=_blocked_assertion_results(
                        case.assertions,
                        EvalOutcome.UNAVAILABLE,
                        message,
                        memory_attribution_evidence=memory,
                    ),
                    unavailable_reason=message,
                    started_at=now,
                    completed_at=now,
                ),
                _EvalTrialPublicData(
                    diagnostic_code=EvalTrialDiagnosticCode.RECOVERY_REEXECUTION_BLOCKED,
                    output=EvalTrialOutputPreviewV1(
                        evidence_state="unavailable", retained_chars=0, retained_bytes=0
                    ),
                ),
            )
    return retained
