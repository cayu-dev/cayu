# Workflow capture policy and saved-attempt recovery

Workflow execution, evidence capture, and assertion scoring are distinct outcomes.
A workflow that completes can have an unavailable score because its evidence exceeds
capture limits. Increasing a capture bound does not grant more execution tokens,
tool calls, time, or model budget.

## Configure capture before dispatch

`WorkflowEvalTarget.capture_bounds` owns the live workflow child-tree policy. There
is no plan or CLI override with competing precedence. The same target is used by
`run_workflow_eval_suite`, `EvalPlan`, corpus compilation, CLI plans, and registered
server targets. Set GAIA's target to:

```python
from cayu import SessionTrajectoryBounds, WorkflowEvalTarget

workflow_target = WorkflowEvalTarget(
    # ... existing target configuration ...
    capture_bounds=SessionTrajectoryBounds(max_events=50_000),
)
```

The setting is aggregate across admitted children, including descendants. It is
not a separate allowance for every child. Workflow journal records retain their
separate existing 10,000-record limit. `None` is not unlimited; omit the field to
use defaults. Unknown fields, booleans in integer fields, zero event allowances,
and values above safety ceilings are rejected before candidate dispatch.

| Bound | Default | Hard maximum |
| --- | ---: | ---: |
| Child sessions | 100 | 500 |
| Tree depth, counting workflow root as depth 1 | 32 | 32 |
| Child events | 10,000 | 100,000 |
| Child transcript records | 5,000 | 50,000 |
| Individual record bytes | 1 MiB | 8 MiB |
| Total retained child evidence bytes | 32 MiB | 128 MiB |

Lineage enumeration also retains its independent 500-candidate ceiling.
`memory_attribution_bounds` is the existing standalone trajectory projection
setting; workflow targets and recovery reject nondefault values for that field.
Workflow Evals memory projection continues to use its separate Evals memory policy. Raising the event bound changes none of these other limits.

Nondefault capture bounds participate in the workflow target revision. Default
bounds retain the previous target revision, allowing original saved root IDs to
be resolved. Every new workflow trial records effective `capture_bounds`; public
corpus trials retain the policy too. Model and tool budgets remain independent.

## Capture failures and reports

The direct trial records `execution_status="completed"` after validating the
workflow completion. A typed child capture/revalidation rejection produces:

- `status="unavailable"`, `score=null`, incomplete evidence, and unavailable assertions;
- `capture_diagnostic` with stage, trajectory code, terminal rejection code, affected
  session, configured global bounds, consumed event/transcript/byte budgets, and
  the rejecting read's limit and observed lower-bound witness;
- an exact `workflow_attempt` anchor binding the original run/suite/case/trial,
  target and projector revisions, input/output digests, root session, attempt,
  completion event and sequence, and complete root-record digest.

For an event rejection, the rejecting read's `limit` is generally the remaining
allowance: `bounds.max_events - consumed_events`. An `observed_lower_bound` of
remaining allowance plus one proves overflow; it is not the child's full size.
An already exhausted budget can be rejected before another read. Record-size,
transcript, byte, depth, lineage, missing-record, and inconsistent-evidence reasons
remain distinct. Diagnostics contain identifiers and counters, never event payloads.

Public corpus/server reports use `workflow_capture_failed`, preserve the typed
rejection, and present runtime completion separately from unavailable scoring.
Direct HTML reports explicitly show workflow completion and the diagnostic.
Assertions never run on an incomplete child tree.

These are additive optional fields in the current EvalRun v11 and published result
contracts. Updated readers accept older documents without these fields; their
execution/capture state remains unknown. Older strict readers can reject new
fields or the new diagnostic code and must be upgraded to consume these reports.
No old report is silently converted from error to successful capture.

## Recapture and score a saved attempt

Use the original target configuration and an app connected to the **original
saved session store**. Per-trial factories may have used a different store from
the profile-probe app: explicitly reconnect that store for recovery. The recovery
functions do not invoke the factory, projector, workflow, providers, tools, or
model judges. Supply the original messages and explicitly recovered projected
output; their hashes must match the recorded anchor.

```python
from pathlib import Path
from cayu import (
    SessionTrajectoryBounds, WorkflowEvalResult,
    capture_workflow_eval_attempt, score_workflow_eval_capture,
)
from cayu.evals.corpus import FinalOutputEqualsAssertionSpec

# source_trial comes from the original saved EvalRun, which remains unchanged.
# original_target preserves its original capture_bounds and behavior revisions.
capture = await capture_workflow_eval_attempt(
    original_target,
    source_trial,
    messages=tuple(original_case.request.messages),
    output=WorkflowEvalResult(
        final_output=original_projected_text,
        structured_output=original_projected_structured_output,
    ),
    bounds=SessionTrajectoryBounds(max_events=50_000),
)
with Path("recovered-capture.json").open("x") as stream:
    stream.write(capture.model_dump_json())

score = await score_workflow_eval_capture(
    original_target,
    capture,
    (FinalOutputEqualsAssertionSpec(id="answer", expected=expected_answer),),
)
with Path("recovered-score.json").open("x") as stream:
    stream.write(score.model_dump_json())
```

Capture documents contain private evidence, including transcript text. Store them
with the original private evidence; they are not public report projections.
`SavedWorkflowEvalCapture.model_validate_json` loads a saved capture document.
Capture and score documents have independent IDs, timestamps and schema version 1.
The score links its capture and evidence digest, records assertion revisions and
evidence/pricing identities, and records zero new model calls. Missing evidence
needed by an assertion produces an unavailable score, not an incorrect answer.

Scoring freshly recaptures the exact store evidence and requires the saved digest.
It evaluates the freshly validated trajectory, not arbitrary caller-edited child
records. A changed or deleted child, changed root payload, different attempt,
input/output/target mismatch, or inconsistent lineage fails closed. Every captured
child must start after the workflow attempt marker and finish before the selected
workflow completion. This bounded recovery API rejects multi-attempt workflow
journals; it does not combine prior-attempt children or choose another completion.
For another capture under different bounds, pass the earlier capture's
`evidence_sha256` as `expected_evidence_sha256`.

Only deterministic portable assertion specs are accepted by the scoring API.
Both model-judge spec types are rejected before scoring; no judge is implicitly
called during recapture. Workspace/artifact probes absent from the saved evidence
are unavailable and are not reconstructed by running tools. If a project needs a
model judge, it must separately and explicitly use its model-scoring surface with
that surface's normal usage accounting.

## Import reports written before attempt anchors

The two retained GAIA smoke reports predate the anchor field. Use
`import_workflow_eval_attempt` explicitly, providing the original `EvalRun`, target,
case/trial slot, input, projected result, and the **exact known** attempt and
completion event IDs read from the saved journal. This validates the deterministic
root ID and exact current completion; it never selects the latest unrelated run.

```python
from cayu import import_workflow_eval_attempt

source_trial = await import_workflow_eval_attempt(
    original_target,
    original_run,
    case_id=original_case_id,
    trial_number=original_trial_number,
    attempt_id=original_attempt_id,
    completion_event_id=original_completion_event_id,
    messages=original_messages,
    output=explicitly_projected_result,
)
# Now call capture_workflow_eval_attempt and deterministic scoring as above.
```

The returned trial preserves the original scoring/error fields and adds an anchor
marked `origin="saved_store_import"` with the original report's digest. It does
not overwrite the report or replace an existing execution-time anchor.

A first capture seals the child evidence visible in the saved store at that time.
The original failed capture did not retain a complete child digest, so neither
this import nor any recovery can prove that unseen bytes were never changed
before that first seal. Preserve an immutable original store backup and report
for historical attestation. Fresh reads still validate terminal and lineage
consistency, and later recovery/scoring rejects mutations against the seal.

Operational acceptance for GAIA requires separately recovering both retained
smoke stores, applying their actual deterministic assertions, and recording their
actual scores. Synthetic tests prove the API and no-dispatch behavior; they do not
establish the smoke answers' correctness. Deployments or GAIA dependency updates
must record the exact Runtime revision used for those results.
