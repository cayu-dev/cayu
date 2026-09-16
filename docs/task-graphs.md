# Durable task dependencies

A task graph atomically submits a bounded set of tasks and their execution
prerequisites. The task store owns readiness and terminal propagation; an
application does not need a separate loop to release joins.

## Submit and inspect

```python
from cayu import TaskCreate, TaskGraphCreate, TaskGraphNode

request = TaskGraphCreate(
    graph_id="report-42",
    nodes=(
        TaskGraphNode(
            task=TaskCreate(task_id="report-42-publish", type="publish"),
            prerequisite_task_ids=("report-42-review", "report-42-check"),
        ),
        TaskGraphNode(task=TaskCreate(task_id="report-42-review", type="review")),
        TaskGraphNode(task=TaskCreate(task_id="report-42-check", type="check")),
    ),
)
receipt = await app.create_task_graph(request)
snapshot = await app.load_task_graph(receipt.graph_id)
events = await app.list_task_graph_events(receipt.graph_id, after_sequence=0, limit=100)
```

Every member needs an explicit task ID. Submission order is irrelevant: forward
references are accepted. All prerequisites must name members of this submission.
Cycles, self-dependencies, duplicate identities and occupied task IDs reject the
whole admission. Bounds are 128 tasks, 1,024 prerequisite edges and 1 MiB of
canonical admission authority, including the task requests. Graph/member IDs are
bounded to 256 UTF-8 bytes. Task payloads retain their existing independent bounds.
Contract-bound members also reserve snapshot capacity for dependency-skip
diagnostics, in addition to ordinary lifecycle headroom. Admission rejects a graph
whose members cannot retain those diagnostics within their JSON-value and byte limits.

Graph identity binds the complete canonical request. Retry the same request after
acknowledgement loss; its receipt does not recreate tasks or reset their current
states. Reusing the graph ID with different content conflicts. Dependencies cannot
be edited after admission. Use graph inspection for current state rather than
interpreting the creation receipt as a current task snapshot.
The receipt separates the original submission digest from the resolved admission
digest. Admission binds resolved parent and session provenance; exact replay returns
that original receipt without resolving or rebinding a replacement parent/session.

## Readiness and outcomes

Tasks with unresolved prerequisites have status `waiting_dependencies`. They are
not claimable. When every exact prerequisite task is `completed`, the store
releases the dependent to `pending` in the same transaction as the prerequisite
outcome. Normal queue eligibility still applies: holds, delayed availability,
scheduling, ownership and work-contract gates are not bypassed.

A failed, cancelled or dependency-skipped prerequisite makes the required success
condition impossible. Its unresolved dependents become terminal
`dependency_skipped`, transitively, with bounded failed-prerequisite identities.
No handler is dispatched for skipped work. A retry successor has a different task
ID and does not substitute for a failed prerequisite; its later success cannot
revive a skipped dependent.

An independent hold remains a hold when dependencies succeed. Resuming that task
does not bypass unresolved prerequisites. `parent_task_id` remains invocation
lineage, not an execution dependency: use `prerequisite_task_ids` for sequencing.
If prerequisites complete during a hold, explicit resume publishes the member's
first readiness event atomically with becoming pending. Later hold/resume cycles
do not duplicate that dependency-satisfaction evidence.
Ordinary tasks outside a graph keep their existing behavior.

Use the existing `run_task_worker` to execute ready members. Workers can compete
normally; graph propagation does not create a second dispatch owner. Reopening a
SQLite or PostgreSQL store reconstructs graph state and readiness from durable
records, without re-running already-completed prerequisites.

## Evidence and lifetime

Graph events are task-store-owned, not session events. Creation, waiting, readiness,
terminal source outcomes and dependency skips have ordered sequence numbers.
Read the next page using the last returned sequence. Page limits are 1–1,000;
an unknown graph returns `None` from inspection and raises `KeyError` from event
listing. Operational task counts and existing task status filters distinguish
dependency waiting and dependency skipping from pending and failed work.
Server contract 45 adds these task statuses and aggregate fields; the bundled
dashboard uses the same contract.

A nonterminal graph retains all of its member tasks. Once the whole graph is
terminal, existing task/session deletion rules may remove live task records;
minimal terminal member evidence and graph identity remain available. Retained
member IDs cannot be reused for new work.

The Python SDK and all three built-in task stores support graph operations.
Custom task stores must implement the graph contract and explicitly expose
`supports_task_graphs`; the base implementation refuses graph operations.
There are no dedicated graph HTTP routes. This API does not introduce task groups,
quorum, compensation, session orchestration or a general workflow language.
