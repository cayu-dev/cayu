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
