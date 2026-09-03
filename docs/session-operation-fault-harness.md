# SessionStore publication fault harness

Cayu's repository tests use
`tests.core._session_operation_fault_harness.SessionOperationFaultHarness` to
exercise durable `SessionStore.publish_session_operation()` failures without
replacing the store's transform, compare-and-set, validation, or transaction.
The optional `PublicationBoundary.EVENT_APPEND` target wraps the same store's
`append_events()` entry point for runtime event publications that do not carry a
session-operation transform. The default remains `SESSION_OPERATION`.
The helper is repository-private test infrastructure. It is not part of the
installed `cayu.testing` API and must not be imported by production code.

## Fault vocabulary

Every rule combines a bounded semantic selector with an ordered action schedule:

- `FailBeforeTransform` raises before the store sees the transform. The transform
  does not run and durable state cannot change.
- `FailBeforeCommit` delegates through the store's real guarded publication,
  reaches its native pre-commit guard after transform validation and staging, and
  raises so the backend rolls the transaction back.
- `CommitThenRaise` waits for the native publication to return, then raises a
  content-free `ConnectionError`. Durable state committed but acknowledgement did
  not reach the workflow.
- `Delegate` invokes the native publication normally.
- `PauseBeforeTransform`, `PauseBeforeCommit`, and `PauseAfterCommit` expose
  explicit one-shot barriers at those same boundaries. Their release disposition
  decides whether the call delegates, commits/returns, or raises.

For `EVENT_APPEND`, `FailBeforeTransform` and `PauseBeforeTransform` mean before
the store receives the append batch. `CommitThenRaise` and `PauseAfterCommit`
retain their ordinary acknowledgement-loss meaning. The harness rejects
`FailBeforeCommit` and `PauseBeforeCommit` for this target because the public
append contract does not expose a safe seam inside the backend transaction.

Failure and delegate actions accept a positive bounded `count`. A rule's
`on_exhausted` policy determines what happens after those occurrences. Use
`MatchPolicy.DELEGATE` for “fail the first N, then run normally”; the default is
fail-closed. Calls not selected by any rule delegate by default. Two matching
rules are always an error rather than an ordering decision.

## Selectors and labels

`SessionOperationSelector` ANDs every supplied criterion:

- exact session ID;
- exact idempotency key or key prefix;
- required event types; and
- a caller-supplied test label.

At least one criterion is required. Prefer stable operation keys and event types.
Use `harness.label("safe-test-label")` when a workflow's key is deliberately
private or when concurrent calls otherwise have identical metadata. The label is
held in a harness-local `ContextVar`, follows tasks created inside the label
scope, and is never persisted or added to the publication. Selectors never
inspect event payloads, checkpoints, transcripts, or operation records.

```python
rule = SessionOperationFaultRule(
    rule_id="terminal-ack-loss",
    selector=SessionOperationSelector(
        session_id=session.id,
        label="terminal-publication",
    ),
    actions=(CommitThenRaise(),),
)

async with SessionOperationFaultHarness(store, rules=(rule,)) as faults:
    with faults.label("terminal-publication"), pytest.raises(ConnectionError):
        await publish_terminal_result()

assert faults.trace[0].committed is CommitEvidence.YES
assert faults.trace[0].acknowledgement_returned is False
```

Rule IDs and labels are short safe identifiers because they appear in bounded
diagnostics. Raw selector values do not appear in exceptions or traces.

## Barrier lifecycle and cancellation

Create a distinct `PublicationBarrier` for every pause action. Start the selected
publication in a task, await `barrier.wait_until_entered()`, perform the competing
operation or deliver real cancellation with `Task.cancel()`, and release the
barrier in a `finally` block. The harness context also releases every barrier,
restores the exact prior store method, and waits boundedly for intercepted calls
to settle if the test body fails.

If cancellation preempts a scheduled action before its fault point is reached,
`CancelledError` remains the top-level signal and the typed unsatisfied-schedule
failure is retained as its explicit cause. Tests that expect owner cancellation
must inspect that cause as well as task cancellation state.

Use the boundary that represents the intended ownership state:

| Barrier | Store ownership at entry | Appropriate use |
| --- | --- | --- |
| Before transform | No store lock or transaction | Let a successor publish first, then prove the stale transform sees its durable state |
| Before commit | Transform and staging occurred inside the store transaction | Prove rollback, cancellation settlement, or ownership retention while mutation is pending |
| After commit | The native call returned to the harness | Prove acknowledgement loss and durable readback/replay |

A before-commit barrier holds the backend's session lock or database transaction.
It cannot be used to let a same-session successor commit before release. Blocking
inside the synchronous transform is also forbidden because memory and PostgreSQL
execute that transform on the event-loop thread. The harness instead uses Cayu's
owned off-thread commit guard.

Barrier timeouts are test watchdogs, not correctness synchronization. Do not use
sleep duration or call order as evidence that a boundary was reached.

## Trace and diagnostics

`harness.trace` returns an immutable bounded tuple. Each record contains only its
local sequence, safe rule ID, action/outcome enums, transform flags, tri-state
commit evidence, and caller acknowledgement status. `UNKNOWN` means the selected
boundary did not positively prove whether a native failing call committed; read
durable state before retrying. The trace never retains publication arguments,
events, payloads, records, results, or exception objects/messages. Entries beyond
the configured bound increment `dropped_trace_entries`. The configured limit must
reserve one entry for every expanded scheduled action. Those scheduled entries
are always retained; only unmatched, exhausted, or rejected incidental calls use
the remaining capacity and may be dropped. This keeps the trace bounded without
allowing unrelated calls to displace evidence for a scheduled fault.

On normal context exit, every scheduled occurrence must have reached its intended
fault point. A transform or validation error before a scheduled pre-commit guard
therefore leaves that action unsatisfied. Unexpected extra matches can delegate
only when the rule explicitly selects that policy.

## Adding a durable-operation scenario

1. Identify the complete production operation tuple and the exact durable record,
   event, and checkpoint evidence that distinguishes commit from rollback.
2. Choose the `SESSION_OPERATION` or `EVENT_APPEND` publication target, then a
   semantic selector and one explicit fault boundary. Never select by a
   process-global call ordinal.
3. Run the real workflow or store transform. Assert both the raised/returned shape
   and all durable representations before retry.
4. For ownership races, pause the stale call before transform, commit the successor,
   release the stale call, and assert that the real transform fences it.
5. Deliver cancellation with `Task.cancel()` and assert `Task.cancelling()`, final
   `Task.cancelled()`, durable settlement, and ordinary `CancelledError` handling.
6. Add the scenario to the shared conformance helper when every built-in backend
   must agree. Backend setup and cleanup remain in existing memory, SQLite, and
   PostgreSQL fixtures.

7. Assert the trace without rendering workload data, and keep a `finally` release
   even though harness teardown is a second safety net.

Current workflow adopters include shared-artifact receipt rejoin after lost
acknowledgement and public provider-operation resolution. The provider workflow
keeps full pre-publication evidence across repeated native rollbacks, preserves
its semantic failure terminalization on retry, replays the exact durable
resolution without duplicate events, and fences a paused stale epoch after a
successor publishes.

The harness does not simulate process death, arbitrary database corruption, or
exactly-once external effects. Long-running and fresh-process qualification may
use this vocabulary while retaining its own process and recovery owner.
