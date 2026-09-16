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

**Group success does not stop remaining tasks or prove their effects have stopped.**
First-success and quorum groups can be decided while other members are running or
have not started. Ordinary graph dependencies continue to govern those tasks.
This API does not automatically cancel losers or release a finalizer.

All three built-in task stores support groups. Custom stores must explicitly
advertise `supports_task_groups` and implement the atomic graph/group contract,
including runtime-prepared provenance and exact replay. The Python SDK refuses
unsupported stores before admission. There are no dedicated group HTTP routes.

This contract does not add group-level retries, weighted voting, hedging,
compensation, semantic evaluation, or a general workflow language.
