# Choosing a ToolEffect

`ToolEffect` tells Cayu what replay can do to externally meaningful durable
state. Classify the operation's behavior, not its implementation, transport,
price, or name.

- `NONE`: replay creates no externally meaningful durable mutation and needs no
  downstream idempotency identity.
- `IDEMPOTENT`: the operation may mutate state, but a stable downstream
  idempotency contract or key collapses repeated execution.
- `EXTERNAL`: the operation has a non-idempotent or outcome-ambiguous external
  effect that generic retry must never assume is safe to repeat.

## Canonical decisions

| Operation behavior | ToolEffect | Why |
| --- | --- | --- |
| Pure computation | `NONE` | Replay creates no externally meaningful durable mutation. |
| Stable local file, database, search, or public HTTP read, including a paid or logged read | `NONE` | Transport, billing, and observability do not turn a read into a mutation. |
| Consuming read, dequeue, or one-time-token invalidation without a stable replay contract | `EXTERNAL` | The call mutates durable state and replay may consume again. |
| Read that creates a durable snapshot or artifact without a stable replay contract | `EXTERNAL` | Snapshot creation is a durable mutation and replay may create another snapshot. |
| Write or other mutating operation with a stable downstream idempotency key or equivalent replay contract | `IDEMPOTENT` | The downstream system collapses repeated execution through a stable operation identity or equivalent idempotency contract. |
| Ordinary file write, email, payment, or other mutation without such a contract | `EXTERNAL` | Replay may repeat the mutation. |
| Mutating request whose outcome is unknown after a timeout or disconnect and has no stable replay contract | `EXTERNAL` | The caller cannot prove whether the first mutation committed or safely collapse replay. |

An ordinary public HTTP GET used for remote research is therefore `NONE`. It
does not become `IDEMPOTENT` merely because it crosses a network, is logged, is
rate-limited, or costs money. Conversely, a method or tool named "read" is not
necessarily `NONE`: dequeueing work, consuming a one-time token, or creating a
durable artifact snapshot mutates state.

When behavior is uncertain, use `EXTERNAL` until the application can name and
test a stable downstream replay contract.

The classification describes replay safety; it does not mean Cayu will
automatically retry the tool.

## Act-once recovery

Use this focused protocol with the canonical `cayu guide durable-operations`
lifecycle when an external mutation may commit before its success acknowledgement
reaches Cayu. A timeout, cancellation, disconnect, worker crash, or
commit-then-raise failure can leave the local outcome ambiguous. A missing local
receipt is not proof that the external mutation did not commit.

Keep one stable operation identity through every transition:

```text
proposed -> authorized -> intent_recorded -> completed
                                  \-> outcome_unknown
                                      -> completed | safe_to_retry | manual_review
```

- `proposed` identifies one bounded mutation and target.
- `authorized` records authority for that exact proposal. Authority must come
  from policy and an authenticated product boundary, not from this state name.
- `intent_recorded` is a durable, atomic record written before dispatch. It is
  evidence that dispatch may occur, not evidence that the effect completed.
- `completed` requires a trustworthy terminal receipt tied to the same operation
  identity, plus separate fresh post-action verification.
- `outcome_unknown` means dispatch may have committed but no trustworthy terminal
  receipt is locally durable. Stop and reconcile before any retry.
- `safe_to_retry` is valid only when an authoritative downstream query proves the
  original mutation did not commit. Recheck authorization before redispatch.
- `manual_review` is required when reconciliation is unavailable, inconclusive,
  contradictory, or cannot bind its evidence to the same operation identity.

When a runtime-classified tool timeout becomes an unknown external outcome, the
`tool.effect.outcome_unknown` event retains bounded `failure_evidence` with a
`timeout` classification. This diagnostic contains neither tool arguments nor
result content and does not authorize retry or prove the external operation stopped.

An invalid tool result is not a reconciliation receipt. Its raw response fields,
including a field named `receipt_id`, are not copied into the default unknown-effect
event. Reconcile the retained logical call using the registered adapter's lookup
or a separately supplied receipt accepted by that adapter's validator.

Interruption after the runtime persists a preparation but before dispatch consumes
it is different from an unknown external outcome. Recovery atomically settles the
exact unconsumed preparation as `tool.call.failed`, with `executed=false` and
`outcome_unknown=false` in the structured result. The effect record retains no
dispatch identity or external receipt. This settlement competes with dispatch on
the same durable comparison; it never reruns the tool. After
`recover_incomplete_session`, use normal resume or the existing approval/user-input
resolution entrance to continue. Approval and answer requirements are unchanged.

A terminal receipt and fresh verification answer different questions. The
receipt is downstream evidence about the attempted operation. Verification is a
new read of the resulting state. Neither can be replaced by the durable intent,
and narration by the model is neither receipt nor verification.

The following credential-free contract uses fake process-local maps only to make
the crash window deterministic. Production intent, outcomes, and reconciliation
evidence must be durable and tenant-qualified. The fake downstream system commits
the mutation and then raises before returning its success acknowledgement. The
recovery path queries that system by the same operation identity and never calls
`dispatch` again.

```python
class CommitThenRaiseSystem:
    """Fake downstream system that loses one success acknowledgement."""

    def __init__(self) -> None:
        self.committed: dict[str, str] = {}
        self.dispatch_count = 0
        self.operation_ids_seen: list[str] = []

    def dispatch(self, operation_id: str, *, target: str) -> dict[str, str]:
        self.operation_ids_seen.append(operation_id)
        self.dispatch_count += 1
        if operation_id in self.committed:
            raise AssertionError("logical operation was dispatched twice")
        self.committed[operation_id] = target
        raise ConnectionError("lost success acknowledgement after commit")

    def reconcile(self, operation_id: str) -> dict[str, str] | None:
        self.operation_ids_seen.append(operation_id)
        target = self.committed.get(operation_id)
        if target is None:
            return None
        return {
            "operation_id": operation_id,
            "target": target,
            "status": "committed",
        }

    def verify(self, operation_id: str) -> dict[str, object]:
        self.operation_ids_seen.append(operation_id)
        target = self.committed.get(operation_id)
        return {
            "operation_id": operation_id,
            "target": target,
            "verified": target is not None,
        }


def run_commit_then_raise_contract() -> tuple[dict, CommitThenRaiseSystem, list]:
    operation_id = "change-0001"
    timeline = [
        ("proposed", operation_id),
        ("authorized", operation_id),
        ("intent_recorded", operation_id),  # persist before dispatch
    ]
    record = {
        "operation_id": operation_id,
        "target": "demo",
        "state": "intent_recorded",
        "receipt": None,
        "verification": None,
    }
    external_system = CommitThenRaiseSystem()
    try:
        external_system.dispatch(operation_id, target=record["target"])
    except ConnectionError:
        record["state"] = "outcome_unknown"
        timeline.append(("outcome_unknown", operation_id))

    # A replacement worker loads this record and reconciles before any retry.
    receipt = external_system.reconcile(operation_id)
    if receipt is None:
        record["state"] = "manual_review"
        timeline.append(("manual_review", operation_id))
        return record, external_system, timeline
    record["receipt"] = receipt
    if (
        receipt.get("operation_id") != operation_id
        or receipt.get("target") != record["target"]
        or receipt.get("status") != "committed"
    ):
        record["state"] = "manual_review"
        timeline.append(("manual_review", operation_id))
        return record, external_system, timeline
    verification = external_system.verify(operation_id)
    record["verification"] = verification
    if (
        verification.get("operation_id") != operation_id
        or verification.get("target") != record["target"]
        or verification.get("verified") is not True
    ):
        record["state"] = "manual_review"
        timeline.append(("manual_review", operation_id))
        return record, external_system, timeline
    record["state"] = "completed"
    timeline.append(("completed", operation_id))
    return record, external_system, timeline
```

For a real `safe_to_retry` result, require positive downstream evidence that the
first mutation did not commit; absence from an eventually consistent read is not
enough. Retain the original operation identity and durable uncertainty even when
an operator must finish reconciliation manually.

### Fault checklist

- **Duplicate delivery:** the same operation identity cannot describe different
  targets, and recovery does not redispatch a merely uncertain operation.
- **Restart after intent:** a replacement worker sees `intent_recorded` or
  `outcome_unknown` and reconciles before doing more work.
- **Concurrent workers:** use one transactional compare-and-set owner for each
  transition; a process-local lock is not durable fencing.
- **Timeout or cancellation:** if it can occur after dispatch, record
  `outcome_unknown`; do not translate it into failure-safe retry.
- **Commit-then-raise:** assume the mutation may have committed even though the
  tool returned an exception.
- **Lost success acknowledgement:** query the downstream system using the same
  operation identity instead of creating a new attempt identity.
- **Unavailable reconciliation:** retain uncertainty and require `manual_review`.

This is an act-once-or-stop application protocol. It does not create generic
exactly-once execution for arbitrary external systems. `ToolEffect` classifies
replay risk; it does not authorize execution. Policy, authenticated approval,
durable ownership, and downstream reconciliation remain separate controls.

## Inspect a receipt-reconciliation target

For an external call interrupted while resolving a user-input pause, include the
original `UserInputResponse` as `ToolEffectReconciliationRequest.user_input_response`.
The runtime compares the answer and continuation digests with the retained pause
authority before admitting recovery. Re-supply the same answer, structured data,
artifacts, metadata, actor, and review reference; a receipt does not authorize a
different answer. The response's session and task identities must match the receipt
request. Ordinary and approval-owned calls do not accept this field. Continuation
retains the paused invocation's frozen configuration; an explicit receipt
`max_steps` must agree with it. The existing user-input closure receipt proves
consumption, so identical reconciliation replay does not run the model again.


Incomplete-session recovery can report `pending_tool_effect`. It retains the
unknown call and finishes any pending session interruption without invoking the
tool or receipt validator. Inspect that call and submit an explicit receipt or
lookup request to continue; this recovery action is not a completed tool result.

After receipt selection, an abandoned reconciliation stream still needs its
explicit continuation. For ordinary tool rounds, recovery plans report
`tool_effect_continuation_required`
and do not offer generic repair or manual outcome replacement for that pending
round. Incomplete-session recovery retains the selected receipt and pending round;
replay the original reconciliation request to continue without revalidating the
receipt or executing the tool again.

For an existing external call, `CayuApp.inspect_tool_effect` supplies the exact
identity and version fields needed by `ToolEffectReconciliationRequest`. Use the
round and call identifiers from public runtime events:

```python
from cayu import ToolEffectReconciliationRequest

target = await app.inspect_tool_effect(
    session_id,
    tool_round_id=tool_event.payload["tool_round_id"],
    tool_call_id=tool_event.payload["tool_call_id"],
)
request = ToolEffectReconciliationRequest(**target.model_dump(), lookup=True)
async for event in app.reconcile_tool_effect(request):
    handle_event(event)
```

Inspection requires a public-authority alias codec shared by the app and store.
The memory store supplies an ephemeral keyring by default. Persistent deployments
must configure their store's keyring and keep it available across reconstruction;
inspection fails read-only when no codec is available.
The target uses field- and session-scoped aliases for the private idempotency key
and session incarnation. These are public references, not downstream keys to send
to an external system. A registered lookup receives the original stable key.

Inspection is read-only: it does not claim the session, invoke a reconciler,
reconnect an environment, or authorize execution. It can describe an already
terminal call, but only an eligible unresolved call may receive a new settlement.
Its versions may become stale; submission checks them rather than refreshing them.
Keep the original request for exact replay after acknowledgement loss. A newly
inspected target is not a replacement for that original request.

Once durable consumption is proven, exact replay returns the selected terminal
event without renewing task execution authority, even if continuation completed
the linked task. The original worker and handoff identities remain part of request
equality; changing them is a conflict, not a new authorization. Any recovery that
still requires execution retains the normal live task-authority checks.

For supplied receipts, request and receipt identity fields must use matching
representations. Resolving an alias does not authenticate external evidence:
the registered application validator must still verify the receipt. Never treat
a raw third-party response or an operator assertion as a validated receipt.

The selected `tool.call.completed` or `tool.call.failed` event includes a
versioned `receipt_evidence` envelope: receipt/schema identities, outcome, source,
observation time, content digest, and the registered integrity/resource fields.
The digest identifies the validated, redacted receipt stored by Cayu, not the raw
third-party response. This envelope is audit evidence, not an authorization token
or a replacement for application receipt validation. It is committed atomically
with the receipt-bound terminal outcome and retained on exact replay.

`tool.effect.reconciliation.started` is versioned attempt evidence, emitted after
exact request preflight and before application reconciliation. It records the
logical call, dispatch identity, intent/request hashes, expected versions, and
lookup mode, not the supplied receipt. It is neither receipt authentication nor
proof of an external mutation. Replaying an already selected reconciliation does
not emit another start; a new explicit attempt after interruption can do so.
The SDK stream yields this event before entering the reconciliation callback.
Closing the stream at that point leaves the effect unresolved without invoking
the callback; a later attempt still requires an explicit reconciliation request.
Admission stores the start event and its bounded attempt identity atomically in
the existing effect record, advancing that record's revision. Cancellation,
timeout, or stream closure does not erase this admission evidence. Inspect the
current target before making a new explicit request after an interrupted attempt.
An older observation cannot regain replay authority once a newer request has
been admitted.

If a new request or a native terminal outcome supersedes an admitted attempt,
that successor transaction also commits `tool.effect.reconciliation.conflict`
with `kind=reconciliation_superseded`. This event identifies the losing attempt
and successor using bounded digests; it does not assert that the old callback
returned or validated a receipt. The audit therefore does not depend on the old
caller remaining alive. Selecting an observation or receipt for the admitted
request itself consumes admission without creating a supersession conflict.

`tool.effect.receipt.validated` contains the bounded receipt evidence envelope,
without the result body or raw external response. Cayu commits it atomically with
the receipt-bound terminal result, then yields it before the terminal event.
Closing the stream at this validation event preserves the selected receipt and
resource versions. Resubmitting the exact original request resumes from that
selection without invoking the validator or protected tool again.

An application validator's `conflict` outcome emits a version1
`tool.effect.reconciliation.conflict` event with `kind=validator_rejected`.
It is committed with the unresolved observation, not a terminal tool result.
The effect stays `outcome_unknown`, retaining accepted partial resource evidence.
Exact request replay returns that observation and raises `ToolEffectConflict`
without another validator call. `not_found` and `unsupported` instead use
`tool.effect.reconciliation.observed` and also leave the effect unresolved.

## Native durable recovery evidence

Tools with an existing operation journal can implement the narrow
`DurableToolRecovery.reconcile_durable_tool_call` extension from
`cayu.core.tools`. It returns `DurableToolRecoveryEvidence` or `None`, not a
bare `ToolResult`. The evidence pairs a diagnostic result with one explicit
disposition:

- `confirmed`: the journal owner validated the complete call identity and has
  positive terminal evidence. Ordinary session recovery can select that result
  atomically with the external-effect record without calling the tool again.
- `not_started`: validated preparation evidence proves this journal's dispatch
  did not start. This is not a generic retry authorization.
- `unresolved`: evidence is incomplete, invalid, conflicting, or ambiguous.
  Its diagnostic result must not be interpreted as a terminal external outcome.

`None` means there is no usable journal evidence; it does not prove the effect
was absent. For an unresolved `EXTERNAL` call, only `confirmed` native evidence
may select a terminal result. The other dispositions retain the unresolved
call. Existing `NONE` and `IDEMPOTENT` recovery still use their diagnostic
result semantics. A caller-supplied result with identical text does not become
trusted journal evidence.

The registered journal implementation owns identity validation and disposition.
Recovery provides bounded observation and fenced journal settlement authority;
the extension must not dispatch the protected mutation. A stored state label
alone is insufficient when its result still describes an ambiguous operation.

## Check a `NONE` declaration before deployment

`verify_tool_effect(...)` is an explicit deployment-readiness test seam. It
invokes one registered tool against a bounded temporary Cayu workspace, then
reports the declared effect and any created, updated, or deleted paths:

```python
from cayu.testing import ToolEffectVerificationStatus, verify_tool_effect

evidence = await verify_tool_effect(
    app,
    agent_name="reporter",
    tool_name="calculate_report_total",
    arguments={"source": "input.json"},
    workspace_files={"input.json": b'{"total": 42}'},
    unobserved_systems=("reporting_database",),
)
assert evidence.status is ToolEffectVerificationStatus.CONSISTENT
```

For `NONE`, an unchanged workspace is `consistent`; any observed create,
update, or delete is a `mismatch`. This is scoped evidence, not proof that the
tool is universally pure. This first observer compares regular-file paths and
content only. Empty directories, symlinks, other non-regular entries,
permissions, timestamps, and filesystem metadata are outside its mutation
evidence, although every traversed entry counts toward the observation limit.
The result always names systems outside the boundary, including network
services, databases outside the workspace, artifact stores, runner execution,
process state, and host paths outside the temporary workspace. Add
application-specific systems through `unobserved_systems`.

The verifier supplies no runner, artifact store, vault, proxy, or knowledge
store and runs the tool directly without policy, approvals, hooks, events, or
the model loop. It rejects `ProcessIsolatedTool`; those tools must enter through
Cayu's runtime-owned process boundary so their hard deadline, containment, and
recovery contracts remain active. Build a fresh application with controlled
adapters for this test: the current Python process is not a security sandbox,
and tool-instance state is not observed. One cooperative asyncio deadline
covers workspace seeding, tool execution, both snapshots, and cleanup checks.
If it expires, the helper raises `TimeoutError` and returns no verdict because
observation did not complete. A tool or filesystem operation that blocks the
event loop can delay that failure; enforcing a hard wall-clock stop requires a
killable process boundary. Snapshots stop at configured traversed-entry and
regular-file caps, and bound per-file and total content bytes. Deadline and
observation-limit failures therefore fail closed.

`IDEMPOTENT` and `EXTERNAL` declarations require the explicit
`allow_effectful_execution=True` opt-in. They execute once and return `observed`,
which records workspace changes but does not claim replay safety. Use a
domain-specific test for the downstream idempotency or reconciliation contract.
`cayu check` remains structural and never invokes this verifier or application
tools.

## Keep other controls separate

`ToolEffect` does not authorize execution. Authorization and approval belong in
tool, command, and network policy. Billing, budgets, quotas, and rate limits are
cost-governance controls. Taint tracks information flow. Events and telemetry
provide observability and audit evidence.

A `NONE` tool can still be expensive, sensitive, denied by policy, or heavily
audited. An `IDEMPOTENT` tool is not automatically authorized. Classify replay
semantics here, then configure those orthogonal controls explicitly.
