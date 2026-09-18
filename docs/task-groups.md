# Durable task groups

A task group records whether an immutable subset of a new task graph has met an
explicit completion policy. Tasks still execute through the existing workers;
groups do not introduce another scheduler, queue, or session orchestration layer.

## Submit a group

```python
from cayu import (
    TaskCreate, TaskGraphCreate, TaskGraphNode, TaskGroupCreate, TaskGroupPolicy,
)

request = TaskGroupCreate(
    group_id="report-review",
    graph=TaskGraphCreate(
        graph_id="report-work",
        nodes=(
            TaskGraphNode(task=TaskCreate(task_id="prepare", type="prepare")),
            *(
                TaskGraphNode(
                    task=TaskCreate(task_id=name, type="review"),
                    prerequisite_task_ids=("prepare",),
                )
                for name in ("review-a", "review-b", "review-c")
            ),
        ),
    ),
    member_task_ids=("review-a", "review-b", "review-c"),
    policy=TaskGroupPolicy(kind="quorum", k=2),
)
receipt = await app.create_task_group(request)
snapshot = await app.load_task_group(receipt.group_id)
events = await app.list_task_group_events(receipt.group_id)
```

The graph, all tasks, group authority, and admission events commit atomically.
Only the three selected reviewers count toward the quorum. Completing preparation
does not count as a group success. If preparation fails, dependency propagation
skips the reviewers and records the impossible group outcome in the same
transaction.

Membership must be nonempty, unique, and restricted to submitted tasks. One group
owns one newly submitted graph; existing graphs cannot acquire groups. Neither
dependencies nor group membership/policy can be edited after admission.
Task and graph limits still apply: at most 128 tasks, 1,024 edges, 256-byte
identities, and 1 MiB for the complete canonical group admission authority.

## Completion policies

| Policy | Succeeds when | Fails when |
| --- | --- | --- |
| `all` | Every selected member completes successfully | Any selected member cannot succeed |
| `first_success` | One selected member completes successfully | No selected member can still succeed |
| `quorum`, `k` | At least `k` selected members complete successfully | Fewer than `k` successes remain possible |

Quorum requires an integer threshold from one through the selected member count;
other policies do not accept a threshold. Duplicate membership and boolean
thresholds are rejected, not normalized into different authority.

Only authoritative task `completed` state contributes success. A proposed or
unaccepted verified result does not. Failed, cancelled, and dependency-skipped
members cannot contribute success. A hold, interruption, expired worker lease,
or cancellation request alone is not a terminal member outcome.

Groups bind exact task IDs, not retry series. A retry successor does not replace
a failed member or change an already decided group.

The group status is `pending`, `succeeded`, or `failed`. A decision retains its
timestamp, exact contributing successes, observed unsuccessful members, and the
stable failure reason `completion_policy_impossible` when applicable. Decisions
follow durable observation order, not worker-reported completion timestamps.
The decision never changes; later member outcomes remain observable separately.

## Durability, replay, and evidence

Member state, dependency propagation, group decision, and their events commit
together. A process restart reads the committed decision; no separate aggregation
worker is needed to finish a missed event.

Retry an identical creation request after lost acknowledgement. The receipt binds
membership, policy, threshold, every task request and dependency, and the resolved
admission authority. Replay returns the original receipt without resetting tasks
or rebinding a replacement parent/session. Different content under the same group
identity conflicts.

Group events are task-store-owned, not session events:

- `task.group_created`
- `task.group_member_terminal`
- `task.group_policy_satisfied` or `task.group_policy_impossible`
- `task.group_succeeded` or `task.group_failed`

Each member terminal outcome is recorded once, including dependency skips. One
policy event and one terminal group event record the decision. Events have
contiguous sequence numbers; paginate with `after_sequence` and a limit from one
through 1,000. Unknown group inspection returns `None`; event listing raises
`KeyError`.

Inspection returns the immutable receipt, current selected-member states,
decision, and last sequence. Minimal member evidence and the decision survive
permitted deletion of terminal task records. The existing graph retention rules
still prevent deletion while graph work remains nonterminal.

## Execution is separate from the group result

**Group success alone does not prove remaining effects have stopped.**
First-success and quorum groups can be decided while other members are running or
have not started. Without an explicit `quiescence` policy, groups remain
outcome-only: ordinary graph dependencies continue to govern those tasks and
the group does not cancel losers or release a finalizer.

## Opt in to quiescence and a finalizer

```python
from cayu import TaskGroupQuiescencePolicy

request = TaskGroupCreate(
    group_id="first-review",
    graph=TaskGraphCreate(
        graph_id="review-race",
        nodes=tuple(
            TaskGraphNode(task=TaskCreate(task_id=name, type=kind))
            for name, kind in (
                ("review-a", "review"),
                ("review-b", "review"),
                ("publish", "publish"),
            )
        ),
    ),
    member_task_ids=("review-a", "review-b"),
    policy=TaskGroupPolicy(kind="first_success"),
    quiescence=TaskGroupQuiescencePolicy(timeout_seconds=60),
    finalizer_task_id="publish",
)
receipt = await app.create_task_group(request)
```

The policy requires a finite positive timeout of at most 86,400 seconds. It and
the exact precreated finalizer identity are part of immutable admission authority.
The finalizer is not a voting member and initially has `waiting_group` status.
It can have independent prerequisites, but neither a selected member nor the
finalizer may transitively depend on the other. User holds and independent
prerequisites remain additional gates after group release.
If a hold outlives group release, resuming the ready finalizer records its first
graph readiness event atomically, even without independent prerequisites. Later
hold/resume cycles do not duplicate that event.

The decisive task transition atomically freezes the contributing members,
fences noncontributors, requests cancellation for active losing work, and starts
the store-timed barrier. Never-started losers are cancelled without dispatch.
Store-owned automatic retry descendants are fenced and drained too; they do not
replace exact voting members or change the frozen group result. Independent tasks
and unrelated graph branches are not cancelled.

Outcome, quiescence, and finalizer lifecycle are separate:

| State | Meaning |
| --- | --- |
| `waiting_decision` | No completion policy decision yet |
| `draining` | A decision exists, but losing execution or effect obligations remain |
| `attention_required` | The original deadline passed without a released barrier |
| `quiescent` | Positive settlement proof covers every losing obligation |

An execution obligation survives terminal task publication and lease clearing.
Existing workers retain their handlers until settlement; cancelling an asyncio
waiter is not proof that a thread, provider, or external mutation stopped.
Verified work uses its exact admission, release, and lifecycle receipts rather
than ordinary task terminalization. An ownerless linked session needs both its
execution owner's return and its exact invocation cleanup receipt. Releasing a
nested session cannot acknowledge an outer worker's later callback work.
If publishing owner return fails, the invocation raises
`TaskGroupInvocationSettlementPending`. Retain the exception's `settlement`
handle and await `settlement.retry()` after storage recovers. Retry publishes
only the exact acknowledgement; it does not rerun model, tool, or cleanup work.
Observation is bounded, and retries join a still-running publication rather
than dispatching another. Cancellation preserves the retry handle in its cause
chain. A lost acknowledgement after a successful commit does not prevent
ordinary group reconciliation from using that durable proof. Process loss
before owner return is durably recorded still leaves the barrier fenced; a
release receipt alone does not recreate that proof.
Preparation records execution ownership before calling the handler. If entry
acknowledgement is lost, the worker reconciles the exact claim before settling
proven non-dispatch. Failed reconciliation remains owned by that worker; retain
it and retry `aclose()` to finish settlement. Resuming a sessionless task,
including unadmitted preparation, requires its prior execution to have settled
and clears its dispatch marker.
Further hold/resume cycles before dispatch preserve that settled evidence and
leave the marker clear.
The next marker strictly advances past its predecessor, even if store timestamps
repeat, so an earlier acknowledgement cannot settle the resumed callback. This
identity ordering does not grant lease authority.
If a winner prevents admission after preparation returns, the worker retains
the exact preparation hold until it can confirm non-admission and publish the
hold atomically. This includes election observed before admission is attempted
and while draining caller cancellation; retrying `aclose()` does not rerun
preparation. An admission that did commit retains its own recovery ownership
instead; failed acknowledgement is never treated as proof of non-admission.
If election refuses execution after admission, the returned worker settles its
exact pre-entry owner through the same transaction that fences execution entry.
This cleanup-only proof binds the admission and claim generation, tolerates lease
renewal, and is rejected if execution entered or ownership changed. It does not
invent invocation-release evidence. Failed publication retains an acknowledgement
owner retried by `run()` or `aclose()`, without rerunning preparation or execution.
Before publishing that task receipt, the cleanup owner closes the admitted
session interaction, atomically publishes its complete admitted initial transcript
(or materializes deferred continuation input), and obtains the real
invocation-release receipt. A retry reconciles the exact interruption marker
before release; a conflicting marker remains fenced. No model, tool, or terminal hook is dispatched on
this cleanup-only path. Session publication and release failures remain
retryable; the admission stays discoverable until cleanup completes. After owner
loss, an expired, group-cancelled admission with no execution entry can be
reconciled by a fresh verified worker without starting execution.
That recovery also requires positive return evidence for the admission's
`interaction.started` delivery. A completed first delivery proves that no
older sink callback remains in flight. An exact failed first delivery can instead
be atomically dead-lettered, preserving its failure while preventing future
delivery retries. A competing retry prevents retirement. Pending or multiply
attempted deliveries retain the barrier and can reach attention-required; lease expiry and invocation
release do not settle that separate callback. Recovery does not redeliver the
event to manufacture quiescence proof.
All built-in SessionStores support exact first-failure retirement and
`claim_first_persisted_event_side_effect()`. Custom stores that leave either
operation unsupported return no proof or claim; recovery stays fenced instead
of silently accepting failure or falling back to an ordinary retry claim.
The same return proof is required for the exact `interaction.interrupted` and
`session.interrupted` deliveries produced by pre-entry cleanup. A competing
cleanup worker cannot release the invocation merely because terminal state has
committed while another worker is still delivering those events. Recovery can
dispatch a never-attempted terminal delivery only by comparing its complete
pending snapshot atomically with the claim. If another publisher advanced it,
recovery rereads the result without dispatching a retry. First-attempt settlement
is still required before releasing the barrier; leased or multiply attempted
delivery does not prove quiescence.
The worker retains a store-confirmed initial-entry refusal before renewal or
readback. Failed admission, receipt, or cancellation lookups remain retryable;
cleanup does not require renewing an execution lease. Cancellation stops waiting
for this retained lookup/publication, not the owned operation itself.
Result resolution additionally records its exact decision and callback owner in the
group barrier before dispatch. A verifier decision alone cannot settle that
callback, and neither claim expiry nor a competing recovery worker can release
its fence. The retained callback acknowledges only after natural return; failed
acknowledgements remain owned by the original runtime for exact retry.
The same retained acknowledgement applies when preparation drains before any
winner is elected: failed election readback or settlement does not discard its
proof, and retrying the worker's `aclose()` never reruns the callback.
Successful preparation also retains this proof while acquiring the handoff lock
and checking cancellation before admission. A failed read or cancellation there
can be settled through `aclose()` without another preparation call. Admission
dispatch keeps the handoff owned until the original call settles and its exact
result is established. Cancellation stops observation, not that call. A later
`aclose()` joins it without another admission dispatch: confirmed non-admission
can settle preparation; a matching durable admission transfers responsibility
to admitted-attempt recovery. This match binds the immutable original admission
intent, even if recovery has already advanced its execution claim. Failed or
conflicting readback, or disappearance of an acknowledged admission, retains the
fence; preparation-only proof cannot settle potentially admitted execution.

Ordinary task workers and dispatchers preserve a naturally returned outcome
separately from execution-settlement writes. They also acknowledge proven
non-dispatch when execution-entry acknowledgement consumes the local deadline.
Busy-session dispatch requeue authenticates non-admission for the exact queued
operation before acknowledging its execution. The next dequeue receives a new
execution identity. A session-conflict exception alone does not establish
non-admission, and failed acknowledgement retains the same retry handle.
For busy-session and generic dispatch failures, the owner retains the exact
admission/release lookup before awaiting storage. A
failed or cancelled lookup leaves an acknowledgement-only retry handle; retry
joins a pending lookup or repeats a failed one without dispatching the invocation.
Unknown admission remains fenced without requeue. After acknowledgement retry, ordinary claim release
or reclamation can make the queue item available again.
Positive admission evidence instead enters exact invocation recovery/requeue;
it never acknowledges non-dispatch or starts another invocation. If that returned
dispatch belongs to a quiescent group, its acknowledgement owner remains retained
until exact terminal evidence proves invocation ownership released. Retry then
settles the original execution identity without another provider or tool call.
If entry acknowledgement fails or its caller is cancelled, the retry owner joins
the original start publication and reconciles its exact claim without starting
the handler. An uncertain readback remains fenced. Verified preparation freezes
its lease during retained entry reconciliation so renewal cannot change that
claim's authority.
Each acknowledgement retry observes at most three writes for at most one second
each. A timed-out write stays owned; a later retry joins that same write rather
than cancelling or duplicating it. If acknowledgement remains pending after
normal task disposition (or failed entry before dispatch),
`TaskExecutionSettlementPending` exposes
`settlement.retry()` and the retained `settlement.result`. Retain that handle and
retry only acknowledgement, never the handler. When cancellation or another
failure is primary, the handle is attached to its exception cause chain instead
of replacing the original signal. Process loss without durable acknowledgement
keeps the barrier fenced; it does not infer callback settlement.
The long-running dispatcher propagates errors carrying this retry owner rather
than logging and discarding it; the caller retains and retries the same handle.

Expiry can settle an undispatched, sessionless claim. It cannot settle dispatched
work, a detached process, or an unknown external outcome. Independent local
execution receipts remain part of the barrier even after a task becomes terminal.
Missing positive proof retains the fence; this API does not kill arbitrary
processes or infer quiescence from elapsed time.

On successful group outcome and positive quiescence, the existing finalizer task
becomes eligible once. A failed group still drains its losers, but marks its
finalizer ineligible. Completing a finalizer is an ordinary task operation;
reconciliation never creates a new task or reruns a completed finalizer. External
effects with lost acknowledgements still need their ordinary idempotency or
reconciliation contract; this is not universal exactly-once external execution.

## Timeout, recovery, and barrier evidence

`load_task_group()` remains read-only. Existing task-worker, verified-worker, and
dispatcher loops perform bounded maintenance scans. An application can also call
`await app.reconcile_task_group(group_id)` to observe exact invocation release and
refresh the barrier. Reopening a store preserves the original deadline, decision,
execution obligations, and finalizer identity.

Timeout latches `attention_required`; late cleanup does not silently authorize
the finalizer. After all losing work has positively settled, inspect the current
snapshot and explicitly resolve that exact barrier:

```python
from cayu import TaskGroupQuiescenceResolution

snapshot = await app.reconcile_task_group("first-review")
resolved = await app.resolve_task_group_quiescence(TaskGroupQuiescenceResolution(
    group_id=snapshot.receipt.group_id,
    request_sha256=snapshot.receipt.request_sha256,
    expected_sequence=snapshot.last_sequence,
    idempotency_key="review-cleanup-confirmed",
))
```

Resolution rejects outstanding obligations or changed authority. Exact replay
returns the recorded resolution without duplicate events. A resolution request
is not permission to discard uncertain execution evidence.

Barrier events share the durable group journal and contiguous sequence cursor:

- `task.group_cancellation_requested`
- `task.group_draining`
- `task.group_quiescent`
- `task.group_quiescence_timeout`
- `task.group_quiescence_resolved`
- `task.group_finalizer_released`, `task.group_finalizer_ineligible`, and
  `task.group_finalizer_settled`

Events contain bounded barrier summaries; inspection retains the exact execution
obligations. Repeated heartbeats and unchanged reconciliation do not emit another
progress event. Required unsettled evidence is retained across task deletion.
Once members are terminal and the quiescence barrier and finalizer are settled,
ordinary reconciliation uses retained evidence even after permitted task deletion;
it neither recreates tasks nor republishes release events.
Acknowledgement-only execution settlement retries also use that retained proof,
but only for the same task, worker, and execution-start identity already settled
in the group. A different execution cannot inherit the retained acknowledgement.
Ownerless invocation retries likewise authenticate the retained task, session and
session instance, interaction, run epoch, and execution-profile fingerprint. The
owner-return acknowledgement may omit the later release digest, but cannot change
it or restore active ownership after settlement.
Nested sessions do not settle their outer worker: the worker acknowledges only
after its handler returns. Deletion retains graph records needed by outstanding
execution acknowledgements, including winners, even when loser quiescence has
already released the finalizer.

All three built-in task stores support groups. Custom stores must explicitly
advertise `supports_task_groups` and implement the atomic graph/group contract,
including runtime-prepared provenance and exact replay. The Python SDK refuses
unsupported stores before admission. There are no dedicated group HTTP routes.

Quiescence additionally requires `supports_task_group_quiescence` and the complete
barrier, discovery, resolution, cancellation-observation, and internal execution
settlement contract. Implementations must atomically publish task, graph, group,
retry-lineage, and event changes; wrappers must not silently drop ownership
observations, including result-resolver entry and settlement. Internal settlement
methods consume runtime-owned evidence, not caller assertions that cleanup
probably finished. All three built-in stores
implement this contract in their existing TaskStore database.

This contract does not add group-level retries, weighted voting, hedging,
compensation, semantic evaluation, or a general workflow language.
