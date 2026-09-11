# Explicit maintenance coding-stage evaluation

`evals/agent.py` remains the credential-free scaffold smoke. It is not maintenance
acceptance. The opt-in `evals/maintenance.py` supplies a context-managed native
workflow target and a fixed two-trial corpus. Invoking it requires the admitted
application image/toolchain, prepared PostgreSQL schemas, shared durable budget,
configured provider and explicit spending authority. It does not create schemas,
approve delivery, push Git, open a PR or merge.

Review the original failure's private source and input before constructing the
corpus. Use the existing `EvaluationSourceIdentityV1` from that reviewed source;
do not manufacture a source identity from arbitrary uploaded JSON. Keep raw
reports, captures and prompts in the authorized private evidence location. The
corpus helper neither redacts nor authenticates caller-supplied input.

```python
from cayu import run_eval_plan, compare_eval_results
from evals.maintenance import maintenance_coding_corpus, maintenance_eval_plan

# reviewed_source and reviewed_task come from the separately reviewed capture.
# Freeze this corpus once and use the same document for both application versions.
corpus = maintenance_coding_corpus(
    source=reviewed_source,
    reviewed_instruction=reviewed_task.instruction,
)

async def evaluate_installed_version(scope):
    # The caller retains scope if startup/evaluation/cleanup fails.
    async with scope as plan:
        result = await run_eval_plan(
            plan, corpus=corpus, suite_id="maintenance-coding",
        )
        # Save the report through your authorized evidence owner inside the context.
        return result

# Run separately in each pinned application's configured environment, serially.
# baseline_scope = maintenance_eval_plan(
#     source_directory=private_source_directory, task=reviewed_task, **baseline_pins,
# )
# baseline = await evaluate_installed_version(baseline_scope)
# current_scope = maintenance_eval_plan(
#     source_directory=private_source_directory, task=reviewed_task, **current_pins,
# )
# current = await evaluate_installed_version(current_scope)
# comparison = compare_eval_results(baseline, current)
```

The pins require `application_release_id`, `implementation_revision`,
`result_projector_revision` and `execution_scope_revision`. Supply actual reviewed
release and SHA-256 commitments; there are no placeholder defaults. The application
release pins the candidate version. The three workflow revisions identify the
workflow implementation, projection and execution-scope contracts—not arbitrary
labels for a model result. Native comparison deliberately rejects changed workflow
contracts. Compare candidate prompt/model changes under unchanged verification,
projection, scope, environment and corpus rules; never keep a workflow revision
unchanged after actually changing its corresponding code or behavior.

The corpus compiler enforces one case, two sequential trials and a 180-second
maximum per trial. Do not run overlapping evaluations or delivery workers against
the same admitted source. Each trial receives a fresh application/workflow and an
exact native workflow root/idempotency key. The existing reservation owner retains
the accepted request, IDs and original deadline; later configuration differences
reject before coding. All deployments use the configured shared budget history
and ledger; constructing another trial does not reset spending authority.

Supply an existing private `source_directory` for retained trial repositories.
The scope creates a new randomly named child for the reference application and
each trial, then materializes the fixed seed using isolated, bounded Git commands.
It never resets, reuses or deletes an operator repository or previous trial output.
`scope.sources` retains each root and preparation outcome, even when preparation
fails or the caller cancels. `source.prepared` means preparation completed with the
expected seed commit; it does not prove the source is still unchanged after coding.
`await source.wait_prepared()` observes the owned preparation without abandoning
an in-flight writer. The native product still validates the clean baseline and
source-copy authority before execution. Only the internally prepared root may
differ in the accepted configuration; all sibling fields must remain equal.
Keep these directories as private input/output evidence under the caller's
retention policy; do not publish their raw host paths in portable reports.

The retained `scope.attempts` tuple records the native run, suite, case, trial,
workflow-root and idempotency identities whenever the trial factory is entered,
including rejected or interrupted construction. It contains no prompt or source
path. Preserve it privately alongside the original report before leaving the
scope. These in-process observations are not durable receipts, authentication,
recovery authority or proof that a provider call occurred. Native trials that
never reached the factory have no such observation; do not omit those slots from
the campaign roster or invent a zero charge for them.

For same-execution host correlation, pass your configured, not-yet-entered scope
and fixed corpus to the emitted helper:

```python
from evals.maintenance import maintenance_corpus_execution

async with maintenance_corpus_execution(scope, corpus) as observation:
    result = observation.result
    attempts = observation.attempts
    # Inspect costs and retain the private evidence before scope exit.
```

This enters the existing scope and directly awaits native `run_eval_plan`; it
does not schedule retries or create another evaluation engine. It retains a
bounded immutable result JSON snapshot and copied attempt observations from that
execution. Each `observation.result` access reconstructs a fresh native result.
Dependencies remain open inside the context and close through the existing scope
on exit. Keep `scope` after a failure to inspect its retained cleanup owners.
Cancellation or failure before the native result is available yields no observation.
This association is private host evidence, not authenticated upload data, a
lossless `EvalRun` capture, recovery authority, or proof of verified PR completion.
The portable result has no private run ID; do not join unrelated uploaded results
and attempt lists merely because their case and trial numbers match. This helper
does not persist evidence: retain it privately under the campaign retention policy.

For recorded-cost readback inside that context, use
`await observation.inspect_costs(reader_app, corpus, pinned_price_book)` with one
registered, open application reading the shared campaign store and alias namespace.
The helper validates the corpus/result roster, then uses the native
`project_causal_budget_id_for_exposure` / `get_causal_budget_cost` entrances for
every observed root, including failed trials. Runtime discovers attributable child
sessions and cost events. Identical repeated roots and session summaries count once;
conflicting bindings or summaries reject the report. Unobserved trials, missing
roots, empty roots, unpriced lines and unknown hosted calls remain explicit.
Validation errors and read failures return no partial report; cancellation propagates.

The known sum is a recorded estimate, not an invoice or an atomic snapshot across
roots. Stop campaign activity through its owners before reading; the reader does
not stop workers. Billing completeness remains `not_established`, even with no
reported gaps. Monetary evidence is limited to 128 digits and exponent magnitude
128; out-of-bound evidence is rejected instead of rounded. No private root mapping
is included in the report. A coding-stage report does
not supply the complete campaign's verified-PR denominator. That denominator also
needs independent final-delivery and cleanup evidence. A new application's local
drain cannot prove that another worker has stopped, and a durable completion
record alone does not prove current cleanup.

The context owns the profile-reference deployment and every trial deployment,
including partially initialized trials. Trial quiescence uses the deployment's
existing Runtime drains without closing evidence dependencies. Evals then performs
its final profile/evidence reads. Context exit closes retained deployments in
reverse order through the existing shutdown owner, preserving failures and
cancellation. Await all native evaluation work and evidence writes within the
context; do not return its plan, start detached work, or use its resources afterward.
Retain the scope handle on startup or exit failure: `scope.deployments` exposes
the exact native owners even when startup failed before yielding a plan. Inspect
their existing settlement/cleanup state; this handle does not authorize blind
retries, store reset or new execution. A successful close does not require keeping
those closed resources alive indefinitely.
Unknown cleanup is not successful qualification. The process supervisor remains
responsible for the deployed stop bound; killing it is not proof of quiescence.

This target's `verified` verdict means independently verified **coding-stage**
output. It is not a verified PR or a full issue-to-PR success count. Published
corpus results are not private `EvalRun` capture documents. Saved-attempt capture
and deterministic rescoring use the existing workflow recovery APIs with the
original target, original private report and original saved store; do not replay
coding to reconstruct missing capture evidence.

Controlled tests exercise the emitted target with native workflow children,
schema/quiescence failures, cancellation, compiler limits and a candidate-output
regression. A separate generated-product integration executes both coding trials
through the emitted target using SQLite and a local runner harness: each reproduces
the bug, repairs a distinct fresh source and retains its output; the operator's
source remains unchanged. A source changed after preparation is rejected before
provider dispatch, while the subsequent clean trial still succeeds. These tests
also read the cohort through one shared SQLite application before scope exit,
including the rejected trial's empty root and both successful roots' child costs.
Scripted usage with deliberately unmatched pricing remains explicitly unpriced;
it is not measured provider billing. Memory/SQLite reporting regressions separately
cover absent and empty roots, unknown hosted-search outcomes, conflicting overlap,
invalid trial/monetary evidence and cancellation during readback.
These tests
do not establish admitted Docker execution
or production model quality. Reviewed live failure, supported PostgreSQL/Docker
trials and independently retained
two-version results remain separate obligations. Total attributable cohort cost,
recovery, intervention and verified PR completion reporting are also separate;
neither a stage score nor the paired accounting API's verified-only aggregate
can replace those measurements.

The coding product can synchronize a repair back to its authoritative source.
Constructing a fresh application against the same source path would not recreate
the seeded bug. The scope therefore prepares independent sources as described
above; do not replace that preparation with an operator-repository reset or relax
the native baseline/profile checks to make another trial pass.
