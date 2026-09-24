# A restartable support agent with external approval

This installed application investigates a damaged order, asks which item was
broken, resumes the conversation in another process, and holds a replacement
call for an externally issued approval. It uses the same public contracts as
`cayu guide durable-service-tools` and `cayu guide durable-operations`; read those
companions for identity declarations and ambiguous-outcome reconciliation.

The canonical executable source ships as `cayu.examples.order_support`. There is
no checkout dependency or second copy to keep in sync. Inspect it with:

```sh
python -c 'import inspect, cayu.examples.order_support as example; print(inspect.getsource(example))'
python -m cayu.examples.order_support --help
```

## Shortest deterministic journey

`investigate -> propose -> wait for approval -> execute -> verify`

In this example, creating a bounded replacement proposal does **not** require
representative approval. After clarifying the damaged item, create the proposal
and request execution of that exact proposal; Cayu holds the execution tool call
until verified approval arrives. Approval gates execution, not proposal creation.
Customer prose is not an approval receipt, and resuming a conversation does not
approve a protected tool call. Other applications may choose different proposal
authorization rules.

The application authenticates the external receipt and checks business authority.
Cayu owns the pending tool approval and execution evidence; the downstream service
owns its idempotency and effect contract. These responsibilities are separate even
when the demo stores live in the same state directory.

With Cayu installed, run these commands from an empty working directory. Every
command starts and exits a separate OS process. No network or provider credentials
are needed. Use the same interpreter throughout.

```sh
python -m cayu.examples.order_support init demo-state demo-operator
python -m cayu.examples.order_support start demo-state
python -m cayu.examples.order_support clarify demo-state --item mug
python -m cayu.examples.order_support review demo-state review.json
python -m cayu.examples.order_support issue demo-state demo-operator review.json receipt.json approve
python -m cayu.examples.order_support deliver demo-state receipt.json
python -m cayu.examples.order_support verify demo-state
python -m cayu.examples.order_support deliver demo-state receipt.json
```

`start` reads the order, tracking, and policy, then asks whether the mug or plate
was damaged. `clarify` resumes the same session, persists one immutable proposal,
and exits with a native pending approval. No replacement exists yet. `review`
exports the proposal and the persisted native approval/round/call identities.
`issue` runs the separate representative fixture and signs its decision; it does
not execute a replacement. `deliver` verifies that receipt and constructs a
`ToolApprovalRequest`. Cayu executes the protected tool and the agent makes a
separate verification read. `verify` performs another read in a new conversation
turn. Delivering the identical receipt again reuses the completed resolution and
creates no additional replacement. Commands print Runtime events, not only a
canned success message.

The independently persisted databases are `demo-state/sessions.sqlite` (Cayu)
and `demo-state/service.sqlite` (orders, proposals, issued receipts, effects).
The representative's private signing key is under `demo-operator`, created with
owner-only permissions and never passed to a model or tool. Possession of that
file represents representative authority **only in this local teaching fixture**.
Production needs authenticated operators, protected signing infrastructure,
real business authorization, key rotation/revocation, and appropriate expiry.
Do not deploy this local fixture as an authentication service.

To repeat from fresh state, remove only these generated fixture files/directories:

```sh
rm -rf demo-state demo-operator review.json receipt.json
```

## Rejection and invalid delivery

Run `init`, `start`, `clarify`, and `review` again, then issue `deny` instead of
`approve`. Deliver and verify it: the replacement list stays empty. Re-delivery of
that denial also has no effect.

```sh
python -m cayu.examples.order_support issue demo-state demo-operator review.json receipt.json deny
python -m cayu.examples.order_support deliver demo-state receipt.json
python -m cayu.examples.order_support verify demo-state
```

With a pending approval, a receipt delivered with `--session another-session`
is rejected. Editing the receipt's proposal, decision, native IDs, or receipt ID
invalidates its signature. An unregistered receipt fails verification, even with
a valid signature. A legitimately issued receipt for different native IDs cannot
resolve the pending action. None of these paths calls the protected service tool.
Changing `--version 2` on delivery demonstrates profile rejection; the example
never auto-adopts changes or disables checks.

## Receipt adapter and state ownership

Read `receipt_to_native_request` in the canonical source for the complete adapter.
It verifies the pinned public-key signature, issued-receipt digest, conversation,
proposal contents/hash, bounded action, and approver authority. It then compares
the native identities with the pending action before constructing the typed
request with actor provenance. A repeated delivery still goes through Cayu's
completed-resolution contract; the adapter never calls the replacement service.
`ResolutionActorSource.REQUEST` accurately describes this trusted SDK adapter's
identity assertion; it does not claim server HTTP authentication.

Pending-action arguments are quarantined. The application therefore stores one
bounded immutable proposal per conversation. Its tool policy checks the exact
proposal ID against the protected call **before** requiring approval. The
representative reviews that service record, not inferred redacted arguments.
External receipt IDs, service action IDs, and Cayu approval/round/call IDs are
separate namespaces and are never substituted for each other.

| Owner | Records and responsibilities |
| --- | --- |
| Cayu | Conversation, pending approval, tool-round evidence, admission, supported resume/recovery, completed-resolution replay |
| Application/service | Operator authentication and business authorization, immutable proposal, receipt signature/binding verification, stable downstream action key and receipt, reconciliation |

This avoids an application conversation log, an application pending-tool state
machine, and a direct post-approval replacement handler. It intentionally retains
business proposals, representative receipts, and downstream idempotency records.
`review.json` is a transport view, not another authority for pending Runtime state.
An application-owned approval workflow remains reasonable when business approvals
span multiple systems or precede a Runtime session; use native tool approval when
holding this exact pending tool call is the useful boundary.

## Optional live provider

Set `OPENAI_API_KEY` and `SUPPORT_MODEL` in the environment, using a model available
to your account. Do not put provider secrets in files or command arguments. Start
with fresh state and add `--live` to **every** start/clarify/review/deliver/verify
command (init and issue are provider-independent). Never switch an existing
scripted session to live mode: that changes its execution profile.

For example, after setting those two environment variables:

```sh
python -m cayu.examples.order_support init live-state live-operator
python -m cayu.examples.order_support start live-state --live
python -m cayu.examples.order_support clarify live-state --item mug --live
python -m cayu.examples.order_support review live-state live-review.json --live
python -m cayu.examples.order_support issue live-state live-operator live-review.json live-receipt.json approve
python -m cayu.examples.order_support deliver live-state live-receipt.json --live
python -m cayu.examples.order_support verify live-state --live
```

The live model chooses its investigative calls and clarification using the same
tools, policy, stores, proposal and approval adapter. It runs no authoring CLI.
The deterministic script proves Runtime/service boundaries; it does not establish
live-model judgment or behavior. Live mode is optional and is validated separately.

## Troubleshooting and limits

- Duplicate session: `run` creates, `resume(ResumeRequest(...))` continues an
  ordinary conversation. The CLI's `clarify` and `verify` use resume. Inspect a
  running or pending session before choosing approval/input resolution or recovery.
- Execution-profile mismatch: keep the same provider/model and declared versions
  across processes. Process-local opaque identities need stable declarations from
  the first run; adding one does not repair a persisted opaque baseline. Digests
  may support only class-level diagnosis. See `cayu guide durable-service-tools`.
- No pending proposal: the live model may still be asking a question, or the
  approval may already be resolved. Inspect events; never fabricate native IDs.
- Signature/binding error: obtain the receipt for this exact reviewed proposal and
  conversation from the trusted representative. Do not rewrite signed fields.

The fixture supports process exits at completed conversation and pending-approval
boundaries. It does not prove arbitrary crash recovery, multi-host coordination,
power-loss durability, or generic exactly-once external effects. The local service
uses an atomic SQLite insert with a stable unique action key. A real remote service
must durably deduplicate that key and reconcile an ambiguous external outcome
before retrying. Never blindly replay an unknown effect: follow the existing
protocol in `cayu guide durable-operations` and `cayu guide tool-effects`.
