# Application-authorized resource access

A shared Cayu application can enforce application-defined access to independently
classified resources. The application authenticates users and evaluates membership,
roles, departments, organizations, and sharing. Runtime enforces opaque subject
references and bounded label predicates. `AuthContext.tenant` remains provenance.

Use `await app.access(verified_subject)` in trusted host code. Never construct a
handle or choose its subject from model arguments, an unverified header, or a
serialized execution binding. Raw `CayuApp` and store objects remain operator
interfaces. This boundary is not a Python sandbox, a database privilege boundary,
or automatic isolation for arbitrary application tools with their own credentials.

## Admission and policy

```python
from cayu import (
    CayuApp, ResourceAccessPolicy, ResourceAccessGrant, ResourceAccessDecision,
    SessionAccessScope, SessionAccessRule, SessionAccessSelector,
)


def rule(**labels):
    return SessionAccessRule(selectors=tuple(
        SessionAccessSelector(key=key, values=(value,))
        for key, value in labels.items()
    ))


class ProductPolicy(ResourceAccessPolicy):
    authority = "product-access-v1"

    async def resolve(self, subject):
        # Host-owned lookup of CURRENT verified membership, including revocation.
        member = await product_memberships.load_active(subject)
        if member is None:
            return ResourceAccessDecision(ResourceAccessGrant(), revision=0)
        private = rule(organization=member.organization, owner=subject)
        department = rule(organization=member.organization,
                          department=member.department, sharing="department")
        organization = rule(organization=member.organization, sharing="organization")
        scope = SessionAccessScope(
            read=(private, department, organization),
            create=(private,), modify=(private,), delete=(private,),
            execute=(private, department, organization),
            update_labels=(private,),
            protected_label_keys=("organization", "owner", "department", "sharing"),
        )
        # Each family is independent. Omitted families deny access.
        return ResourceAccessDecision(
            ResourceAccessGrant(sessions=scope, tasks=scope,
                                artifacts=scope, knowledge=scope),
            revision=member.policy_revision,
            valid_until=member.decision_expires_at,
        )


app = CayuApp(resource_access_policy=ProductPolicy())
access = await app.access(verified_subject)
page = await access.sessions.list_sessions()
```

`product_memberships` and `verified_subject` are application-owned symbols. The
same registered agent can serve private users, departments, organization-wide
resources, or separate organizations by changing classification and policy. No
membership tables or `tenant_id` columns are added by Runtime.

Each action permits at most 16 alternative rules. Selectors inside a rule are
ANDed; rules are ORed. Effective authority is **admitted maximum AND current
authority AND caller filters**. Empty action grants deny. An empty rule is invalid
unless `allow_all=True` is explicit. Unsupported arbitrary ACL callbacks are not
post-filtered after pagination: the host must compile a supported bounded
predicate or use a separate enforcement integration.

`SessionAccessScope` is the reusable action-predicate value for all four resource
families. Supplying it directly as a policy result explicitly grants the same
bounds to every family; prefer `ResourceAccessGrant` when permissions differ.
`inspect_state` additionally gates raw checkpoint reads. Tools that inspect child
checkpoints need that action. `relabel` is separate from ordinary label updates.
Task creation requires `create`; worker dispatch also requires `execute`.

## Scoped API and ownership matrix

Handles have explicit methods and no fallback forwarding to raw stores.

| Surface / operation | Authoritative owner and action | Enforcement and denial |
| --- | --- | --- |
| `sessions.load`, `list_sessions` | Current session labels; read | Memory lock or native SQL predicate before pagination/count; absent/foreign load has the same denial |
| `sessions.create` | Validated new labels and any parent; read + create | Native create transaction; parent must be readable and execution binding cannot switch |
| `sessions.update_metadata`, `update_labels`, `delete_session` | Current session; modify, update_labels, delete | Checks and mutation share the write lock/transaction; normal lifecycle/closure fences remain |
| `sessions.read_records` | Session owns events/transcript/checkpoint/relabel audit | Bounded page and bytes under one read snapshot; checkpoint also requires inspect_state |
| `sessions.events`, `usage`, `costs` | Current session owns historical events | Policy predicate precedes export, aggregation and grouping; historical event label snapshots do not grant access |
| `tasks.load`, `list`, `create`, `cancel`, `pause`, `resume` | Immutable `TaskInvocation.access_labels`; corresponding read/create/modify | Native filtering and mutation checks; standalone tasks have their own durable classification |
| `tasks.create_graph`, `create_group`, `graph`, `group`, graph/group events | Collection receipt retains the uniform member classification and binding | Admission checks all members; replay must match classification; receipts retain ownership after member deletion |
| `artifacts(store, environment_name=...)` put/read/range/list/delete | Immutable artifact labels, plus existing artifact/environment/session ownership | Local and S3 metadata checked before bytes; inventory filters before counts; immutable publication identity prevents relabel-by-ID-reuse |
| `knowledge(access_scope=...)` get/list/search/create/revise/delete/chunks/evidence | Knowledge entry labels AND existing KnowledgeAccessScope | Native predicates before search/list/count; caller namespace/visibility/source constraints cannot replace host constraints |
| Automatic recall and model knowledge tools | Execution's knowledge grant AND existing knowledge scope | Same native intersection; protected label changes require relabel authority |
| Scoped message enqueue/inspect/action/source snapshot | Target session; source session for provenance links | Existing SessionMessageAccessPolicy remains required; source read and target read/modify checks occur in native transaction/snapshot |
| Scoped peer-content append | Source and resolved target sessions | Existing collaboration/export authority remains required; native source-read and target-modify checks precede replay or mutation |
| `run`, `resume`, `fork`, `recover` | Stored execution binding and current session classification | Fresh admission; resume/recovery keep original maximum; aliases resolve before authorization; source/child checks at fork admission |
| Native task workers | Stored task invocation binding | Trusted resolver reconstructed at dispatch and after process replacement; denial never invokes handler; lease/outcome settlement remains Runtime-owned |
| Foreground children and incomplete recovery | Inherited/stored binding | Continuations retain binding; new provider/tool dispatch revalidates; recovery may settle already-known outcomes without admitting new work |
| HTTP resource router | Host-authenticated subject | Only scoped endpoints mounted; body/provenance cannot grant authority; absent and foreign resources return the same 404 |
| SSE delivery | Original execution maximum and fresh current decision | Checked before producer advance and delivery, including buffered first HTTP event |
| Other store/control-plane operations | Trusted operator | Not forwarded by scoped handles or product router. Unsupported native methods reject model-controlled calls; raw operator use remains available |

The last row includes administrative leases, global reconciliation, schema work,
closure/pinning journals, provider-disposition commands, collaboration participant
administration, and unexposed topology/maintenance operations. Adding a product
endpoint requires an explicit scoped implementation; mounting the existing
packaged control plane does not make those operations product-safe.

Memory, SQLite, and Postgres session/task/knowledge stores implement the native
contract. Local and S3 artifact stores implement immutable resource classification.
Native in-memory embedding and Postgres pgvector stores use the same predicates
for semantic candidates and index coverage. Custom stores and unadapted subclasses
are rejected at scoped admission or use; inheriting a capability marker is not
sufficient. Additional integrations require native conformance before they can
be used in scoped mode.

## Persistence, transactions, and sharing

Existing session label tables remain authoritative. Non-secret execution bindings
are optional fields in persisted session/task invocation JSON: authority name,
opaque subject, canonical admitted predicates, and admission revision. Standalone
task labels live in invocation JSON; collection receipts retain that invocation.
Artifact metadata carries immutable labels. Knowledge uses its existing normalized
label/index structures. No historical migration is rewritten and no tenant schema
is introduced. Empty optional fields are omitted to preserve legacy commitments.
Task-label SQL predicates operate before pagination; deployments should measure
query plans for their workload before adding application-specific expression indexes.

SQLite scoped reads pin a snapshot; Postgres uses repeatable-read where multiple
reads form one result. Native mutations check classification inside their existing
write transactions or locks. Protected session-label replacement validates both
old and new classifications, requires relabel permission, and writes audit evidence
in the same transaction. Audit includes subject/authority when supplied by an
application handle, old/new labels, time, and admitted-scope digest. Existing
knowledge revision evidence remains the record of knowledge changes.

Task and artifact classification is immutable in this API. Sharing is expressed
by labels selected at creation and overlapping read grants. Do not attach a
foreign session's artifact merely because its ID is known: existing session and
environment artifact ownership checks compose with label access. Derived child
sessions and delegated tasks inherit the binding; they cannot replace it with a
broader subject or current default. A different subject's access to shared data
does not automatically transfer authority to resume that subject's execution.

Legacy unbound executions are not adopted implicitly by scoped resume. Operators
must make an explicit migration/adoption decision. A trusted explicit `allow_all`
grant can read otherwise unclassified data; applications that require a label
should use positive selectors requiring that label.

## Revocation and cleanup

Runtime does not cache policy decisions. Every protected operation samples current
policy; provider/tool dispatch, worker dispatch, resume, and subsequent stream
delivery revalidate. Permission expansion never enlarges an admitted maximum.
Resolver failure, unknown authority, expired decisions, and revisions older than
the persisted admission revision fail closed. The resolver is responsible for
returning current decisions and preventing rollback of its own policy history;
Runtime's persisted revision is an admission floor, not a distributed high-water
mark for every observation. Use monotonic application revisions and bounded expiry.

The linearization point for an external decision is the operation's successful
policy resolution. A revocation arriving afterward cannot retroactively cancel
that admitted database operation. Classification checks and writes are atomic
within the store, but no atomic transaction spans an unrelated external policy
service. Already-dispatched effects are not undone and already-delivered bytes
cannot be recalled. Recovery retains enough internal authority to settle known
outcomes, record denial, and release leases; that authority cannot dispatch a new
business effect or return protected data to a scoped caller.

## Product HTTP integration

```python
from cayu.server.resource_access import create_resource_router

# authenticate verifies credentials and returns the application's opaque subject.
server.include_router(create_resource_router(app, authenticate=authenticate))
```

The router exposes session queries/records/mutations, tasks, graph/group reads,
event export, usage, run/resume/fork streams, and configured artifact operations.
Supply `knowledge_scope` to enable knowledge read/search routes. Artifact store
aliases and environments are host configuration, never client-selected backend
credentials. Keep the packaged operator server behind separate operator access.
Application membership/RBAC and HTTP ingress rate limits are separate concerns.

Conformance coverage lives in `test_session_access.py`, `test_task_access.py`,
`test_knowledge_resource_access.py`, `test_resource_execution_access.py`,
`test_resource_access_http.py`, and `tests/artifacts/test_access.py`. It exercises
native backends, known foreign IDs, predicate intersection, export/usage, protected
label races, durable classification, dispatch denial/settlement, and stream bounds.
