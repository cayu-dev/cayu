# Recipe: server authentication and tenant isolation

**Goal:** embed one Cayu-backed application in a multi-tenant product without
mistaking authenticated tenant provenance for data authorization.

## The contract

`create_server(..., auth=...)` and `mount_cayu(..., auth=...)` authenticate
access to Cayu's protected server surfaces. An auth dependency may return:

```python
AuthContext(subject="user-123", tenant="org-456", claims={"role": "member"})
```

The `tenant` value is useful operator provenance. Cayu stamps verified identity
into durable evidence for approvals, interruptions, recovery, user-input
resolution, session compaction, and message enqueue. It does **not** scope
Cayu's stores or authorize access to records.

In particular, the built-in routes do not automatically filter:

- sessions, events, or transcripts;
- tasks or usage;
- knowledge or artifacts; or
- the packaged dashboard/control plane.

If callers from two tenants can reach those routes and one caller knows the
other tenant's raw record identifier, authentication alone does not prevent the
lookup.

## Safe application-owned boundary

Expose product-owned routes to end users and authorize a product resource before
calling Cayu. Keep the association between the public product id, tenant, opaque
Cayu session id, and durable task id in trusted application storage. Make every
identifier column non-null, and enforce uniqueness for `(tenant_id, public_id)`,
`(tenant_id, idempotency_key)`, the Cayu session id, and the task id. Persist a
fingerprint of the accepted creation request with the reservation. The same
idempotency key and fingerprint return the original row; the same key with a
different fingerprint is a conflict. A missing Cayu session id must fail closed;
passing `None` to an optional Cayu query filter means that filter is not applied.

The central query must include the authenticated tenant as a criterion, for
example:

```sql
SELECT public_id, cayu_session_id, task_id
FROM product_runs
WHERE public_id = :public_id AND tenant_id = :authenticated_tenant
```

A missing row is denied or reported as not found. Do not load by `public_id`
first and compare a caller-supplied tenant afterward.

The following application sketch shows where the boundary belongs.
`StartRunBody`, `Principal`, `ProductRunStore`, `IdempotencyConflict`,
`require_product_principal`, and `require_operator` are application-owned
symbols; they are intentionally not Cayu abstractions.
`ProductRunStore.reserve(...)` uses an atomic insert-or-load under the unique
`(tenant_id, idempotency_key)` constraint. On the first call it allocates
non-null `public_id`, `cayu_session_id`, and `task_id` values and persists the
request fingerprint. A retry returns the same row only when its fingerprint
matches; otherwise it raises `IdempotencyConflict`.
`ProductRunStore.load_owned(...)` performs the tenant-qualified lookup and
returns `None` when no row matches:

```python
import json
from dataclasses import dataclass
from hashlib import sha256

from fastapi import APIRouter, Depends, HTTPException

from cayu import CayuApp, EventQuery, TaskCreate

router = APIRouter(prefix="/product")
cayu_app = CayuApp(...)  # process-scoped runtime and stores
product_runs = ProductRunStore(...)  # tenant-owned product records


@dataclass(frozen=True)
class Principal:
    subject: str
    tenant: str


def product_run_request_fingerprint(body: StartRunBody) -> str:
    canonical = json.dumps(
        {"agent_name": "assistant", "prompt": body.prompt, "schema": 1},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(canonical).hexdigest()


async def require_owned_run(*, public_id: str, tenant: str):
    owned = await product_runs.load_owned(public_id=public_id, tenant=tenant)
    if owned is None or owned.cayu_session_id is None or owned.task_id is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    return owned


def task_matches_product_run(
    task,
    request: TaskCreate,
    *,
    cayu_session_id: str,
) -> bool:
    return (
        task.id == request.task_id
        and task.type == request.type
        and task.title == request.title
        and task.description == request.description
        and task.parent_task_id == request.parent_task_id
        and task.assigned_agent_name == request.assigned_agent_name
        and task.input == request.input
        and task.metadata == request.metadata
        and task.session_id in {None, cayu_session_id}
    )


async def ensure_product_run_task(*, owned, prompt: str):
    request = TaskCreate(
        task_id=owned.task_id,
        type="product_run",
        title="Run product agent",
        assigned_agent_name="assistant",
        input={"product_run_id": owned.public_id, "prompt": prompt},
        metadata={"product_run_id": owned.public_id},
    )
    task = await cayu_app.task_store.load_task(owned.task_id)
    if task is None:
        try:
            return await cayu_app.create_task(request)
        except ValueError:
            # Another request may have won the create race. Only accept its task
            # after loading and validating it; unrelated ValueErrors are re-raised.
            task = await cayu_app.task_store.load_task(owned.task_id)
            if task is None:
                raise
    if not task_matches_product_run(
        task,
        request,
        cayu_session_id=owned.cayu_session_id,
    ):
        raise RuntimeError("Reserved task id belongs to different work.")
    return task


@router.post("/runs", status_code=202)
async def start_run(
    body: StartRunBody,
    principal: Principal = Depends(require_product_principal),
):
    # The store atomically allocates all three ids and persists their ownership.
    try:
        owned = await product_runs.reserve(
            tenant=principal.tenant,
            created_by=principal.subject,
            idempotency_key=body.idempotency_key,
            request_fingerprint=product_run_request_fingerprint(body),
        )
    except IdempotencyConflict as exc:
        raise HTTPException(
            status_code=409,
            detail="Idempotency key was already used for a different request.",
        ) from exc
    await ensure_product_run_task(owned=owned, prompt=body.prompt)
    return {"id": owned.public_id, "status": "accepted"}


@router.get("/runs/{public_id}/events")
async def list_run_events(
    public_id: str,
    principal: Principal = Depends(require_product_principal),
):
    # Enforce ownership and a complete id mapping before constructing a Cayu query.
    owned = await require_owned_run(
        public_id=public_id,
        tenant=principal.tenant,
    )
    records = await cayu_app.session_store.query_events(
        EventQuery(session_id=owned.cayu_session_id, limit=100)
    )
    # Return an application-owned projection, never raw Cayu events or payloads.
    return {
        "events": [
            {
                "type": str(record.event.type),
                "occurred_at": record.event.timestamp,
            }
            for record in records
        ]
    }
```

The projection above deliberately omits the opaque Cayu session/event ids and
the complete event payload. Add fields only through an application-owned
allowlist and redact secrets or customer data before returning them. Apply the
same rule to transcripts, artifacts, errors, tool arguments/results, model
content, and metadata; successful ownership authorization does not make raw
runtime records safe for end-user disclosure.

Use a durable `TaskStore` and a `run_task_worker(...)` loop for the
`product_run` task type. The trusted worker resolves `task.id` back through
`ProductRunStore` before starting `RunRequest(session_id=owned.cayu_session_id,
task_id=task.id, task_worker_id=worker_id, ...)`. This keeps execution and lease
heartbeats outside the HTTP request, avoids accumulating a complete event stream
in request memory, and preserves tenant ownership in application storage. The
load/create/reload sequence above handles a lost HTTP response and concurrent
task-creation race without weakening Cayu's duplicate-id rejection: an existing
task is accepted only after its immutable creation fields and session mapping
match the reservation. Reconcile an application reservation if durable task
creation ultimately fails. See [Triggering runs](../triggering-runs.md) for the
complete worker contract.

Apply the same `require_owned_run(...)` step before resume, interruption,
approval, recovery, transcript, artifact, task, usage, and deletion operations.
For tenant-scoped knowledge or aggregate queries, use an application route and
store/wrapper that enforces the authenticated scope in the underlying query;
do not expose the raw Cayu query as the authorization check.

For background work, persist the authorized product resource or trusted tenant
scope with the job. Reload it from application storage in the worker. Do not
let a queued payload or model-selected argument choose the tenant whose records
will be read.

## Treat the inspector as an operator surface

The simplest safe hosted layout keeps Cayu's complete control plane on an
operator-only boundary:

```python
mount_cayu(
    server,
    cayu_app,
    path="/internal/cayu",
    auth=require_operator,  # rejects ordinary product members
)
```

Network policy, a separate operator host, or an authenticated gateway can add
another boundary. Do not expose the mount to product tenants unless every route
and backing store has been reviewed for end-to-end tenant scoping. The auth
dependency runs before protected routes, but returning different
`AuthContext.tenant` values does not rewrite their store queries.

## What does not establish authorization

These values can improve traceability or make accidental collisions less
likely, but none is sufficient authorization by itself:

- session labels or metadata containing a tenant id;
- a tenant query filter supplied by the caller;
- namespaced, prefixed, hashed, or hard-to-guess record identifiers; or
- tenant claims copied into an event or task payload.

Use them only after an application-owned authorization boundary or a store-level
policy such as correctly configured row-level security has established access.
Test cross-tenant reads and mutations for every product route, including
background jobs and identifier-enumeration attempts.

Native tenant-aware storage is not currently part of Cayu's server contract.
Any future core implementation requires separate design, migrations, and
conformance tests; it must not be inferred from `AuthContext.tenant`.
