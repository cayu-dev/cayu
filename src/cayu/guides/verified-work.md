# Verified work

Use verified work when a task may only count as done after an independent check
of durable evidence. The application freezes a `WorkContract`; a
`VerifiedTaskWorker` runs bounded attempts; a registered verifier decides each
proposal; and only an applied, accepted decision completes the task.

The credential-free reference is
[`examples/durable_file_workflow/verified.py`](https://github.com/cayu-dev/cayu/blob/main/examples/durable_file_workflow/verified.py).
Its sibling `demo.py` shows the lighter alternative: application code checks the
artifact itself and completes an ordinary task. Choose verified work when the
check, its evidence, and the rejection history must be durable and attributable.
For the general observe/propose/act-once lifecycle, see
`cayu guide durable-operations`.

## Shortest deterministic journey

```text
create contract -> create bound task -> attempt 1 -> proposal -> verifier rejects with a gap
  -> continuation on the same session -> attempt 2 -> proposal -> verifier accepts
  -> resolver rebuilds the result -> decision applied -> task completed -> binding retired
```

From a repository checkout:

```sh
uv run python -m examples.durable_file_workflow.verified
uv run python -m examples.durable_file_workflow.verified --store sqlite
uv run pytest tests/examples/test_verified_file_workflow.py
```

The scripted model's first program forgets the trailing newline. The verifier
rejects that attempt with the gap `artifact.missing_trailing_newline`, the model
receives exactly that gap, repairs the program, and the second attempt is
accepted. The run also loses one acknowledgement after a durable write and
recovers without rerunning the model or the program. A worker that rewrites the
workspace copy of its input to match a forged output is rejected, because the
verifier reads the expected input from the task record instead.

## Five layers, five owners

Keep these separate. Each one proves less than people often assume.

| Layer | Owner | Proves | Does not prove |
| --- | --- | --- | --- |
| Structured-output validation | provider request and schema | a response has the expected shape | that any work happened or is correct |
| Completion proposal | `VerifiedTaskHandler.propose` | the worker claims one digest-bound result, with optional evidence | success; final model prose is never a proposal or a verdict |
| Verifier decision | the registered `DeterministicCompletionVerifier` | a verdict for every criterion and constraint, with cited evidence and gaps | anything about other attempts; it never mutates state |
| Result reconstruction | the registered `CompletionResultResolver` | the accepted result can be rebuilt from application evidence | correctness beyond the decision; Cayu only accepts a result whose digest equals the proposal's |
| Task ownership | Cayu decision application | the task completes once, through a durable receipt | that external side effects happened exactly once |

While a contract binding owns the session, ordinary `complete`, `fail`, and
`cancel` operations on the task are fenced, and queued steering is rejected.
Before its first attempt is admitted, a bound task can still be failed or
cancelled through the ordinary task APIs, but never completed; only an applied,
accepted decision completes it.
Acceptance retires the binding (`retired_contract_binding` on the lifecycle
receipt), after which the session behaves like any other.

## Contract anatomy

`CayuApp.create_work_contract(WorkContractDraft(...))` publishes an immutable,
fingerprinted contract. It carries:

- `objective`: what the work is for.
- `criteria`: ordered `WorkCriterion` entries (`ordinal` 1..n) that a verifier
  must judge individually.
- `constraints`: `WorkConstraint` entries that must hold for any acceptance.
- `evidence_requirements`: `WorkEvidenceRequirement` kinds that satisfied
  criteria and constraints must cite.
- `verifier` and `result_resolver`: exact references (id, version, and
  configuration fingerprint; the verifier reference also names its kind). Register the matching adapters with
  `register_completion_verifier` and `register_completion_result_resolver`
  before any worker runs.
- `continuation_policy`: what a rejection means, and the attempt and
  repeated-gap ceilings.

Create the task with `CayuApp.create_task(TaskCreate(task_id=..., work_contract=contract.reference()))`.
A bound task needs a caller-stable `task_id`. Ordinary task workers never run
bound tasks; they park one as `needs_attention` with
`verified_work_contract_runner_required`.

The handler is two read-only, retry-safe callbacks. `prepare` returns the
`RunRequest` for the first attempt; the worker assigns task, session, and lease.
`propose` reads the attempt's evidence and returns a `CompletionProposalCreate`
with the supplied `proposal_id` and `attempt_id`. In an uninterrupted run,
`prepare` runs once per task and `propose` once per attempt, but either may run
again during recovery, so neither may perform the domain operation.

## Deterministic verifiers first

Start with a deterministic verifier: plain code that reads application-owned
evidence and returns a `CompletionVerifierDecision`. It must be side-effect free,
declare a stable `execution_profile_identity`, and cover every criterion and
constraint exactly once, in contract order. Declare each decision-bearing
dependency that is not part of the adapter, such as where expected values come
from, as a `CompletionVerifierProfileComponentDeclaration` in
`execution_profile_components`. Cayu fingerprints them, so a dependency that
changes after registration, or between attempts and restarts, fails closed
before the verifier runs instead of silently changing how work is judged.

- Read expected values from application-owned state, such as the task record or
  your own database, not from the worker's workspace. The worker's copy of an
  input is evidence to check, not the reference to check against.
- That boundary holds only if the worker's execution cannot reach the
  application's state either. The reference example runs tools locally with no
  sandbox, so in its SQLite mode a worker program could edit the task database
  directly. In production, run worker tools in a sandboxed environment that
  cannot reach your stores or the verifier's inputs.
- Read worker-written files with bounded reads. The example uses
  `LocalWorkspace.read_bytes(path, max_bytes=...)`, which runs off the event loop
  and refuses non-regular files, so a pipe or an oversized file cannot stall the
  verifier.
- An accepted decision marks every outcome `satisfied` and has no gaps.
- A satisfied outcome whose subject declares evidence requirements uses
  `satisfaction_basis=evidence` and cites an available reference for every one.
- A non-accepted decision carries at least one `CompletionGap` for every
  unresolved outcome and none for satisfied ones. Gaps must be unique and sorted:
  criterion gaps before constraint gaps, then by subject ID, code, and evidence
  requirement IDs. This is not contract order, so sort them explicitly.
- Gap `code` values are stable machine codes; the `summary` is for people and
  does not affect repetition detection.

Provider-backed judging is optional, governed work, not a shortcut around this
contract. The `provider` verifier kind is reserved: a contract may name it, but
registering or running such a verifier is rejected today. When a
model judgment is genuinely needed, run it as its own budgeted, attributable
step, persist its output as evidence, and let a deterministic verifier decide
whether that evidence satisfies the criteria. Calibrate such judges with
`cayu guide evals-ai-quality`.

## Attempts, repeated gaps, and budgets

A rejected attempt is neither a failed task nor a completed one. It is a durable
decision about one proposal. What happens next is frozen in the contract. The
common outcomes are:

| Outcome | Task status | Status reason |
| --- | --- | --- |
| Accepted | `completed` | (none) |
| Rejected, `rejection_action=continue`, below every ceiling | stays `running` | a continuation attempt starts on the same session |
| Rejected, `rejection_action=interrupt` | `paused` | `work_contract_rejected` |
| Rejected at the `max_attempts` attempt | `needs_attention` | `work_contract_attempt_limit` |
| The same gaps recur `max_repeated_gap_count` times after their first rejection, before the attempt limit | `needs_attention` | `work_contract_repeated_gap_limit` |
| Verifier verdict `blocked` | `blocked` | `work_contract_blocked` |
| Verifier verdict `needs_review` | `needs_attention` | `work_contract_needs_review` |
| Elapsed-time or budget limit reached | `needs_attention` | `work_contract_elapsed_limit` or `work_contract_budget_limit` |
| Run paused for approval or user input, or stopped by a run limit | `needs_attention` | `work_contract_execution_interrupted` |

Handler, preparation, and execution failures, and cancellation through a task
group, have their own status reasons; none of them completes the task.

Budget and step ceilings come from the `RunRequest` that `prepare` returns: its
`limits`, `budget_limits`, and causal budget are frozen with the first attempt and
reused unchanged by every continuation. Each continuation is a new run, so what
the ceiling covers depends on its scope. `RunLimits` defaults to `scope="run"`,
which resets for every attempt: a four-tool-call run limit lets each attempt make
four calls. Use `scope="session"` for a ceiling across all attempts, and session,
causal, or app-scoped budget limits for spend across the whole task. The
elapsed-time ceiling is the worker's `max_elapsed_seconds`, measured from when
the task's preparation starts and tightened by any `RunRequest.execution_deadline`;
it is separate from `RunLimits.max_elapsed_seconds`. A verifier that cannot judge at all, for example because
the application's own input is missing, should return `blocked` rather than
`rejected`, because another attempt by the worker cannot fix it.

A continuation never edits the contract. The model's next user message is a JSON
document of type `cayu.verified-task-continuation.v1` carrying the rejected
`decision_id` and its gaps, so the next attempt acts on the verifier's
authoritative findings rather than on its own earlier prose. The repeated-gap
ceiling compares each rejection's gap fingerprint (its gaps' subjects, codes,
and evidence requirement IDs, plus the unresolved outcomes' statuses and reason
codes) with every earlier rejection of the same task. Summaries are excluded, so
rewording a gap does not reset the count. The example sets
`max_repeated_gap_count=1`, so the same gaps twice stop the task. No ceiling ever
fabricates success.

## Recovery and lost acknowledgements

Contracts, attempts, proposals, decisions, and receipts live in the task store.
A replacement process rebuilds `CayuApp` over the same stores, registers the
same verifier and resolver references, and starts a new `VerifiedTaskWorker`.
The worker discovers unfinished attempts from durable evidence:

- A committed proposal is not executed again; the model and its tools do not
  rerun.
- A proposal receives at most one committed decision. A stored decision is
  reused, but if a verifier claim expires before its decision is recorded, the
  verifier runs again, so it must be deterministic and side-effect free.
- An accepted chain settles once. The resolver may likewise rerun after a lost
  claim, and Cayu accepts only a result matching the proposal's digest.

The reference example simulates a reply lost after attempt 2's proposal commits.
After the restart, the model call count and the program's external-effect count
are unchanged, the verifier runs once for that attempt, the resolver runs once,
and exactly one `task.completion_result.resolved` event exists. Every proposal,
decision, and application receipt recorded before the restart is equal
afterwards, and calling `resolve_completion_result` again with the
decision application receipt's idempotency key returns the stored task without running the
resolver again.

Recovery does not make external effects exactly-once. Make the tools inside an
attempt idempotent or act-once (`cayu guide tool-effects`), and treat `propose`,
the verifier, and the resolver as read-only.

For durable stores, give `SQLiteSessionStore` a stable
`public_authority_alias_codec` (for example from
`public_authority_alias_codec_from_environment()`) so a restarted process can
verify workspace aliases written before it. The worker never closes stores. Use
it as an async context manager; if closing reports `VerifiedTaskWorkerDraining`,
alone or inside an exception group, keep the worker and its stores and retry
`aclose()` until it returns. A cancellation is reported as a cancellation, so
treat a cancelled worker's stores the same way the application treats any
cancelled operation that may still have work in flight.

## The same concepts in other domains

| Domain | Objective | Criteria | Evidence | Deterministic verifier | Resolved result |
| --- | --- | --- | --- | --- | --- |
| Terms negotiation | agree terms with a counterparty and send them | terms within the principal's preferences; every required fact present; the correct approval recorded | the negotiated terms revision, its approval decision, and the durable send receipt | checks the revision against the preferences, and the approval and receipt against that exact revision; a draft or a model's claim is not completion | reference to the sent, approved terms |
| Utility-bill reconciliation | reconcile one billing period | every meter read matched; totals within tolerance | the bill version and the ledger entries it cites | recomputes totals from the cited versions | the reconciliation record |
| Bid readiness | publish a complete bid package | every required section present; required approval recorded | the immutable package version and the approval decision | checks sections and the approval against the package digest | the published package reference |
| Revision-bound code review | approve one exact revision | checks pass; required findings addressed | the commit digest, check results, and review notes for that commit | rejects when evidence names a different revision | the approved revision reference |

In each row the worker proposes, the verifier decides from evidence, the
resolver rebuilds the result from that same evidence, and Cayu owns the task
transition. Larger workflows that must apply an exact set of business actions
once each need a dedicated recipe on top of this foundation; this guide covers
one contract-bound task.
