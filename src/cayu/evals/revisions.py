"""Content identities shared by eval execution and publication."""

from __future__ import annotations

from hashlib import sha256

from cayu._validation import canonical_durable_json_bytes
from cayu.evals.models import EvalTrialResult


def eval_trial_result_revision(result: EvalTrialResult) -> str:
    """Return the exact durable identity of one lossless trial result."""

    if type(result) is not EvalTrialResult:
        raise TypeError("result must be an exact EvalTrialResult.")
    document = result.model_dump(mode="json", round_trip=True, warnings="none")
    return sha256(canonical_durable_json_bytes(document, "eval trial result")).hexdigest()


__all__ = ["eval_trial_result_revision"]
