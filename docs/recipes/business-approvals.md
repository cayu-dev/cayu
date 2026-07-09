# Recipe: multi-tier business approvals

**Goal:** gate an agent's high-stakes tool call behind a real business
approval — routed to a computed tier (`area → national → corporate`), resolved
with one of three outcomes (**approve / condition / decline**), and audited so
the business outcome sits next to the runtime decision.

Cayu's approval primitive is deliberately small: one durable pending approval
per round, resolved with a binary approve/deny. This recipe composes the
business semantics on top of that primitive with the
`cayu.runtime.business_approvals` adapter — the primitive stays untouched, and
everything below rides on public surface (policy metadata, resolution
metadata, and the event log). The complete runnable no-key demo is
[`examples/business_approval_tiers.py`](../../examples/business_approval_tiers.py):

```bash
PYTHONPATH=src python examples/business_approval_tiers.py
```

## The shape

```
model calls route_package(...)          product computes the routing tier
        │                                        │
        ▼                                        ▼
TieredApprovalPolicy ── REQUIRE_APPROVAL + BusinessApprovalRouting carrier
        │
        ▼
durable pause: pending_tool_approval checkpoint
tool.call.approval_requested (routing visible to the approval UI)
        │
        ▼
resolve_business_approval(approver_tier=..., outcome=approve|condition|decline)
        │ tier gate: at-or-above required_tier in the chain
        │ condition  -> APPROVE + condition_text (required)
        │ decline    -> DENY   + self-explaining reason
        ▼
tool.call.approved / approval_denied carry the stamped business record
        │
        ▼
business_approval_audit(events) -> BusinessApprovalRecord per approval
```

## 1. Compute the routing in product code

Tier computation is business logic — package value, cost center, region — so
the adapter never guesses it. You provide a callback (sync or async; it may
hit your database) and `TieredApprovalPolicy` standardizes the carrier:

```python
from cayu import BusinessApprovalRouting, TieredApprovalPolicy

async def compute_routing(request):
    if request.tool_name != "route_package":
        return None                      # no approval needed -> ALLOW
    amount = request.arguments.get("amount_usd", 0)
    tier = "corporate" if amount > 100_000 else "national" if amount > 10_000 else "area"
    return BusinessApprovalRouting(
        required_tier=tier,
        chain=("area", "national", "corporate"),   # ordered lowest -> highest
        metadata={"package_id": request.arguments.get("package_id")},
    )

app.register_agent(spec, tools=[...], tool_policy=TieredApprovalPolicy(
    compute_routing=compute_routing,
    reason="Package routing requires business approval.",
    expires_in_seconds=24 * 3600,   # optional: bound how long the demand stays valid
))
```

Already have a `ToolPolicy`? Merge `business_approval_routing_metadata(...)`
into your own `ToolPolicyResult(decision=REQUIRE_APPROVAL, metadata=...)` —
`TieredApprovalPolicy` is just that pattern packaged.

## 2. The pause, and what the approval UI reads

The pause is Cayu's ordinary durable approval interrupt: a
`pending_tool_approval` checkpoint, `tool.call.approval_requested`, and
`session.interrupted` with `interruption_type="tool_approval_required"`. The
routing rides along — an approval inbox reads it straight off the event:

```python
from cayu import PendingToolApproval, business_approval_routing

pending = PendingToolApproval.from_event(approval_requested_event)
routing = business_approval_routing(pending)   # required_tier, chain, product metadata
```

Your product decides which humans hold which tier; the routing tells it which
tier this approval needs.

## 3. Resolve with the tier gate and three outcomes

```python
from cayu import BusinessApprovalOutcome, resolve_business_approval

events = await resolve_business_approval(
    app,
    session_id=session_id,
    approval_id=pending.approval_id,
    approver_id="maria.k",
    approver_tier="national",
    outcome=BusinessApprovalOutcome.CONDITIONED,
    condition_text="Ship only after payment clears.",
)
async for event in events:
    ...
```

- **Tier gate** — `approver_tier` must sit *at or above* `required_tier` in the
  ordered chain (a corporate approver can resolve a national item); anything
  else raises `BusinessApprovalTierMismatch`. Approvals paused without the
  routing carrier raise `BusinessApprovalRoutingMissing` — the gate never
  silently degrades into an ungated resolve.
- **`APPROVED`** resumes as a plain approve.
- **`CONDITIONED`** resumes as approve with `condition_text` **required** —
  the business said yes-with-strings; the runtime runs the tool.
- **`DECLINED`** resumes as deny; when you pass no `reason`, the adapter
  synthesizes `"Business approval declined by {approver_id} ({approver_tier})."`
  so the model's blocked result explains itself.
- The call is a coroutine returning the event stream: gate failures raise at
  `await` time, before anything resumes — `events = await resolve_business_approval(...)`,
  then iterate (same streaming contract as `resolve_tool_approval`).

## 4. What the tool and the model see

On approve-path outcomes the stamped business record reaches the **tool**
through `ToolContext.metadata["cayu:business_approval"]` — a tool can enforce
or log its condition:

```python
async def run(self, ctx, args):
    record = ctx.metadata.get("cayu:business_approval") or {}
    if record.get("outcome") == "conditioned":
        attach_condition(args["package_id"], record["condition_text"])
    ...
```

In a round with several tool calls, every tool in the round receives the
stamped record (resolution metadata is round-level, matching the primitive's
one-approval-per-round shape).

The **model** does not see approve-path metadata — it only sees the tool's
result. If a condition must steer the model's subsequent behavior, surface it
in the tool's `ToolResult` (or send a follow-up message on the resumed
session). On `DECLINED`, the reason and metadata do reach the model inside the
blocked tool result.

## 5. Audit: business outcome next to the runtime decision

The stamped record lands durably on the `tool.call.approved` /
`tool.call.approval_denied` event payloads, so the audit trail is a pure
projection of the event log — no side table to keep consistent:

```python
from cayu import business_approval_audit

records = business_approval_audit(await store.load_events(session_id))
for r in records:
    print(r.approval_id, r.decision, r.outcome, r.condition_text,
          r.approver_id, r.approver_tier, r.requested_at, r.resolved_at)
```

Each `BusinessApprovalRecord` pairs the request with its resolution:
`decision` is the raw Cayu decision, `outcome` the business outcome. A record
with `decision` set but `outcome=None` means someone resolved through the raw
primitive without the stamp — visible instead of hidden. Note the limit: the
projection records what callers *asserted*, it does not authenticate them; a
caller at the same trust level can write the stamp by hand through the raw
primitive (see the trust section). Events may arrive in any order and from
any number of sessions — records pair per session; for cross-session audits,
feed it `query_events(EventQuery(event_types=(...)))` output.

## 6. Trust boundaries — read this one

The tier gate runs **in the calling process**. It is an application
affordance, not a security boundary:

- `approver_id` / `approver_tier` are asserted by the caller. Authenticate the
  human and map them to a tier in *your* API layer — exactly where this gate
  lives in product code today — then call `resolve_business_approval` from
  your (server-side) handler.
- Anyone with direct access to `CayuApp.resolve_tool_approval` or
  `POST /api/tool-approvals/resolve` bypasses the adapter legitimately. An
  unstamped resolution shows up as `outcome=None` in the audit — but a caller
  at that trust level can also write the `cayu:business_approval` stamp by
  hand, indistinguishably from an adapter resolution. The audit records
  assertions; authenticating who may assert them is your product's job — gate
  access to the raw resolve surface accordingly.
- Pair the assertion with typed provenance: pass
  `resolved_by=ResolutionActor(subject=..., source=...)` from your auth layer
  (see the runtime-contracts provenance section) and the runtime stamps it on
  the resolution events; `business_approval_audit` surfaces it as
  `record.resolved_by`, next to the asserted `approver_id`/`approver_tier`.

**Expiry composes decision-first:** if the policy set `expires_in_seconds` and
the window closes before resolution, the runtime coerces *any* resolution —
including an adapter-approved one — into a denial (`tool.call.approval_expired`
plus denial events with `expired: true`; no typed error from the adapter). The
audit record then shows `expired=True` with `decision=DENY` next to the
asserted business outcome: "approved too late", preserved for the trail.

## v1 limits

- **One routing per round.** The pending approval carries the gating call's
  policy metadata; if several calls in one round need approval, the round
  resolves under that one routing (per-call policy metadata stays visible on
  `pending.tool_calls[*].metadata`).
- **Resolve-only.** Crash recovery (`recover_tool_approval`) supplies a tool's
  terminal result, not a business decision — the business outcome was already
  recorded at approval time.
- **Single resolution.** The chain is an escalation ladder for routing one
  decision, not a sequential multi-signature workflow; sequential sign-off
  remains product-side orchestration (one Cayu approval per stage works).
