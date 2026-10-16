# Tool authorization grants

A tool authorization grant lets a person approve a call once and have Cayu allow
later calls of the same kind without asking again. The grant is durable policy
state recorded next to one approval. It is not a new approval outcome: the
pending-approval primitive stays binary (approve or deny), and the call that
created the grant is still approved by its own approval.

## Four kinds of authority

| Kind | Who decides | Lives where | Cayu guarantees |
| --- | --- | --- | --- |
| One-shot approval | A resolver, for one pending call | The session checkpoint and approval events | Exactly-once resolution and execution of that call |
| Reusable authorization (this page) | A resolver, for a class of later calls | A `ToolAuthorizationGrantStore` | Exact matching, scope and boundary isolation, expiry, revocation, durable use evidence |
| Application-owned policy | Your `ToolPolicy`, with your own store | Your application | Only what the policy returns; Cayu enforces ALLOW, DENY, and REQUIRE_APPROVAL |
| Runtime-enforced controls | Cayu | The runtime | REQUIRE_APPROVAL always pauses; mandatory controls are never granted; a grant never overrides DENY |

Applications that prefer their own grant store and a custom `ToolPolicy` can
keep doing that. Every part below is usable on its own.

## Parts

- **Records and matchers** (`cayu.approvals`): `ToolAuthorizationGrant` is a
  typed, versioned (`schema_version: 1`) record with the policy and tool
  identity, approval gate, matcher, scope (`kind` and `id`), authorization
  boundary, originating approval identity, creator and resolver provenance,
  creation time, optional expiry, sampled-review rate, justification and
  evidence, and lifecycle state.
- **Stores**: `InMemoryToolAuthorizationGrantStore`,
  `cayu.storage.SQLiteToolAuthorizationGrantStore`, and
  `cayu.storage.PostgresToolAuthorizationGrantStore` share one set of
  transition rules, so matching, idempotency, expiry, revocation, ordering, and
  reconstruction agree. The SQL tables join storage revision 119; run
  `cayu storage migrate` on an existing database.
- **Policy**: `ToolAuthorizationGrantPolicy` wraps any `ToolPolicy` and only
  ever turns its REQUIRE_APPROVAL into ALLOW. A wrapped `TaintAwareToolPolicy`,
  directly or inside `GuardedToolPolicy`, keeps its protection: the runtime
  still finds it and supplies its taint labels.
- **Service**: `ToolAuthorizationGrantService` creates grants from the
  runtime's pending-approval snapshot, settles, lists, and revokes them.
- **Server routes**: `cayu.server.authorization_grants.ToolAuthorizationGrantServer`
  exposes the service over authenticated HTTP.

## Matching

A grant applies to a call only when all of these hold:

- The grant is `active` and has not reached `expires_at`. Expiry is observed
  from the clock (`effective_status(now)` returns `expired`); it is never a write.
- The policy id, tool name, and approval gate are equal. The gate is the string
  a gating policy puts in its REQUIRE_APPROVAL metadata under
  `cayu:approval_gate`. It names the class of decision.
- The scope applies. A `session` grant applies only in its originating session
  incarnation: a new session that reuses a deleted session's id does not
  inherit it.
  A `workspace` grant applies in any session running in the same workspace id,
  and only when the application named that workspace (`LocalWorkspace(...,
  workspace_id="repo-1")` and the other workspaces' `workspace_id=` argument).
  A derived default id, such as a path or a runner description, can be shared
  by unrelated workspaces, so it never carries a workspace grant; session
  grants still work there.
- The boundary is equal: agent name, environment name, workspace id, the
  session's resource-access principal (the `authority` and `subject` trusted
  access control assigned; an absent principal matches only an absent one), and
  the values of the session labels named in `boundary_labels`. A session missing
  one of those labels is never grantable. Grants therefore never cross
  principal, environment, or workspace boundaries. Clients can set session
  labels, so `boundary_labels` narrow a grant but are not an authority
  boundary; bind tenants through resource access.
- The matcher accepts the arguments.

There are two matchers, and the policy author chooses one per tool in
`grantable`. Clients cannot choose a matcher or send pattern text.

- `process_command`: the exact process command. `argv`, `cwd`, `env`, and
  `stdin` must be equal; `timeout_s` and `max_output_bytes` are ignored. The
  grant keeps `argv` and `cwd`, and only a digest of `env` and `stdin`, which
  often carry credentials. Shell
  text, unknown argument keys, prefixes, and longer or shorter argument vectors
  never match.
- `tool_call`: any call to the tool under the same gate (a class of
  decisions), optionally with `pinned_argument_paths`. Each pinned path must
  equal, with type-strict JSON equality, the value it had on the originating
  approval. Pinned values are copied from the snapshot, not from the request.

When several grants apply, session grants are checked before workspace grants,
each ordered by `(created_at, grant_id)`, and the first is used.

## Sampled review

A grant carries `sampled_review_rate_bps` (0 to 10000 basis points). For each
granted call Cayu computes a sample bucket from the SHA-256 of the grant id,
session id, and tool call id (`tool_authorization_grant_sample_bucket`). If the
bucket is below the rate, the call still goes to a person: the policy returns
REQUIRE_APPROVAL with the grant evidence in its metadata. The bucket depends
only on durable identity, so replay, recovery, and audit reach the same
decision. A sampled-review approval is never itself grantable.

The service's `default_sampled_review_rate_bps` applies when a request leaves
the rate out.

## Mandatory controls

A mandatory control is a `ToolAuthorizationGrantControl(tool_name, approval_gate)`.
`approval_gate=None` covers every gate on the tool. Pass them as
`mandatory_controls` to the policy, which enforces them when each call is
authorized. A gating policy can also mark a single REQUIRE_APPROVAL decision
with `cayu:mandatory_approval: true`.

A call covered by a mandatory control always requires approval, even when the
downstream policy would allow it and even for a tool that is not grantable; a
downstream DENY stays a denial. Existing grants are never applied to it, and the
pending approval records that it is not grantable, so the service cannot create
a grant from it. A project objectives document is
the expected source of these controls and of the default sampled-review rate;
the runtime takes them as plain inputs.

## Evidence

- The REQUIRE_APPROVAL result carries a grant offer under
  `cayu:authorization_grant_offer`. The runtime persists policy metadata on the
  pending-approval checkpoint, so grant creation reads the tool, gate, matcher
  kind, and boundary from that immutable snapshot.
- Before returning ALLOW or a sampled REQUIRE_APPROVAL, the policy writes a
  `ToolAuthorizationGrantUse` record (grant id, scope, session and its
  incarnation, tool round, tool call, arguments digest, decision, sample bucket,
  rate). The round is part of the record's identity, so a provider that reuses a
  call id in a later round still leaves one record per execution, while a retry
  of the same execution does not add one. If that write fails, the
  policy requires approval instead. `list_uses(grant_id)` reconstructs every
  granted decision after a restart, a page at a time: pass
  `after=ToolAuthorizationGrantUseCursor.after(page[-1])` to continue. Over HTTP,
  `/uses` returns `next_cursor`; pass it back as `cursor`.
- The ALLOW result and the approval request that created the grant both carry
  `cayu:authorization_grant` with the grant id and scope.
- The grant records `created_by` (the actor that asked), `resolved_by` (the
  actor on the approval event), its `justification`, and bounded `evidence`
  (for example the history of unchanged approvals that justified it).
- Only the grant policy writes `cayu:authorization_grant` and
  `cayu:authorization_grant_offer`; the policy drops both keys from its
  downstream `then` result, so another policy cannot forge evidence or offers.
- `arguments_public` records whether the originating approval allowed its
  arguments to be published with a static secret scope (arguments that resolve
  secrets at run time are never public). When it is `False`, the HTTP views show argument
  values copied into the matcher (`argv`, `cwd`, pinned values) as
  `[REDACTED_SECRET]`.

## Creation, idempotency, and crash behavior

`approve_and_grant(approval, grant)` is "remember and approve". It owns the
runtime's resolution stream: closing it, or a failure while reconciling, still
waits for the runtime's own cleanup before returning, so the session is never
left running behind a closed stream:

1. Load the pending approval from the checkpoint and check the offer. Refuse an
   approval that is resolved, expired, not offered, or a mandatory control, an
   offer whose principal differs from the session's, and a task continuation
   (a worker relays an approval; it cannot create standing authority).
2. Ask the `ToolAuthorizationGrantAuthority`. A refusal raises
   `ToolAuthorizationGrantDenied`; nothing is written and the approval stays
   pending.
3. Write the grant as `staged`. The grant id is derived from the session and
   approval ids, so there is one grant per approval. The same request again
   returns the same grant; different terms raise
   `ToolAuthorizationGrantConflict` and change nothing. A failed write raises
   before the approval is touched.
4. Resolve the approval with a request that carries the grant marker. Its
   digest was recorded on the staged grant.
5. Activate the grant once the runtime has durably approved the call with one
   of the grant's request digests.

A staged grant never authorizes anything. If the process stops after step 3,
the approval is still pending and retrying the same call continues from the
staged grant. If it stops after step 4, the call already ran exactly once and
`reconcile(grant_id)` or `reconcile_staged()` (for example at startup)
activates the grant. If the approval was denied, expired, or approved by a
request without the marker, reconciliation marks the grant `abandoned`.
Reconciliation reads the approval's closure receipt before and after its other
evidence; if the approval was being published meanwhile, the grant stays
`staged` until the next reconciliation instead of being judged from a partial
view. Abandonment also requires the grant to still carry exactly the approval
requests that were classified, so a request added concurrently by `prepare` is
classified again rather than lost. A retry
reuses the runtime's exactly-once approval resolution, so the protected call
cannot run twice.

`prepare(approval, grant)` does steps 1 to 3 and returns the staged grant and
the exact `approval_request` to submit through any resolution path; call
`reconcile` afterward.

## Listing, expiry, and revocation

- `list_grants(ToolAuthorizationGrantQuery(...))` filters exactly by policy id,
  tool, scope kind and id, originating session, and stored status.
- `list_applicable(session_id=..., workspace_id=...)` returns active, unexpired
  grants scoped to that session or workspace.
- `revoke(grant_id, actor=..., reason=...)` requires the authority's approval.
  Revoking again returns the first revocation.
- Listing pages by `(created_at, grant_id)`: pass `after_created_at` and
  `after_grant_id` from the last record, or use `query.next_page(page)`. Over
  HTTP, `GET /api/tool-authorization-grants` returns `next`; pass its fields back
  as query parameters. The policy, `list_applicable` and `reconcile_staged` read
  every page, so older or expired grants never hide a newer one. A store must
  return only records after the cursor; a page that does not advance is refused,
  and the policy then requires approval.
- A grant is standalone policy state. Deleting its originating session does
  not delete or revoke it, and retention does not prune grants; revoke a grant
  explicitly when its authority should end.

## HTTP

Pass a server to `create_server(app, config=ServerConfig.protected(auth),
tool_authorization_grants=ToolAuthorizationGrantServer(service, authorize_read=...))`.
The routes are mounted only when configured and require authenticated access:

| Route | Purpose |
| --- | --- |
| `POST /api/tool-authorization-grants/approve` | Remember and approve; streams the resolution like `/api/tool-approvals/resolve` |
| `GET /api/tool-authorization-grants` | List grants the caller may read |
| `GET /api/tool-authorization-grants/{grant_id}` | One grant with its expiry-aware status |
| `GET /api/tool-authorization-grants/{grant_id}/uses` | Use evidence |
| `POST /api/tool-authorization-grants/{grant_id}/revoke` | Revoke |
| `POST /api/tool-authorization-grants/{grant_id}/reconcile` | Settle a staged grant |

The approve body has no `resolved_by`, tool, argument, or decision field. The
actor is always the authenticated caller, and the call is always the pending
snapshot. The router can also be mounted in your own FastAPI application with
`server.router(auth=...)`.

## Minimal consumer example

```python
from cayu import AgentSpec, CayuApp
from cayu.approvals import (
    AllowedActorsGrantAuthority,
    ResolutionActor,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolAuthorizationGrantControl,
    ToolAuthorizationGrantPolicy,
    ToolAuthorizationGrantRequest,
    ToolAuthorizationGrantScopeKind,
    ToolAuthorizationGrantService,
)
from cayu.storage import SQLiteToolAuthorizationGrantStore
from cayu.tools import AlwaysRequireApprovalToolPolicy, ExecCommandTool

grants = SQLiteToolAuthorizationGrantStore("grants.sqlite")
# Calls under the "commands" gate are grantable; a tool that must always ask
# would be listed here, for example ToolAuthorizationGrantControl(tool_name="deploy").
mandatory = [ToolAuthorizationGrantControl(tool_name="deploy")]

app = CayuApp()
app.register_agent(
    AgentSpec(name="coder", model="..."),
    tools=[ExecCommandTool()],
    tool_policy=ToolAuthorizationGrantPolicy(
        policy_id="coder-commands",
        store=grants,
        grantable={"exec_command": "process_command"},
        then=AlwaysRequireApprovalToolPolicy(tools=["exec_command"], approval_gate="commands"),
        mandatory_controls=mandatory,
    ),
)
service = ToolAuthorizationGrantService(
    app,
    store=grants,
    authority=AllowedActorsGrantAuthority(creators=["alice"]),
    default_sampled_review_rate_bps=500,
)


async def remember_and_approve(pending) -> None:
    approval = ToolApprovalRequest(
        session_id=pending.session_id,
        approval_id=pending.approval_id,
        tool_round_id=pending.tool_round_id,
        tool_call_id=pending.tool_call_id,
        decision=ToolApprovalDecision.APPROVE,
        resolved_by=ResolutionActor(subject="alice"),
    )
    grant = ToolAuthorizationGrantRequest(
        scope=ToolAuthorizationGrantScopeKind.WORKSPACE,
        expires_in_seconds=7 * 86_400,
        justification="Approved unchanged in every run this month.",
    )
    async for event in service.approve_and_grant(approval, grant):
        print(event.type)
```

`pending` is the approval identity from the `tool.call.approval_requested`
event. The next identical `argv` in any session of the same workspace runs
without a pause, except for the roughly 5% of calls chosen for sampled review.
