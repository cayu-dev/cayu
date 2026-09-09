# Protected session-message lifecycle

Durable steering uses the existing session-message queue. Inspection, withdrawal,
quarantine, source provenance, and target freshness do not create another workflow
or give a caller permission to execute a tool. Observe-only applications retain
their findings outside the queue and do not call enqueue.

## Application authorization

Configure `CayuApp(session_message_access_policy=policy)` with a
`SessionMessageAccessPolicy`. Its synchronous, side-effect-free `authorize`
method receives `SessionMessageAccessContext(subject, tenant)`, the resolved
`session_id`, `session_instance_id`, and one action: `inspect`, `enqueue`,
`source`, `withdraw`, or `quarantine`. Only the literal result `True` permits
access. The policy must consult trusted application ownership and permission
data. An identical tenant string in session metadata, a known session ID, or a
successful public-alias lookup is not an authorization grant.

Trusted Python application code supplies `context=` separately from the request.
Do not construct it from unverified request JSON, model output, or queue metadata.
Source access is authorized independently from target enqueue access. Possessing
a source snapshot does not grant either permission.

For scoped SDK calls, omit `requested_by`: the runtime derives subject and tenant
from context and stamps `source=request`. Supplying an actor is rejected even
when subject and tenant match. HTTP uses a separate trusted server entrance that
stamps `source=http_auth`; no public request field selects that entrance. Both
paths construct the actor without caller claims. Ordinary unscoped SDK enqueue
may carry caller identity, but it is re-stamped `source=request` and its claims
are not accepted as authority. The durable actor tuple is exactly subject,
tenant, and source; claims are not part of that tuple.

An explicit negative/non-boolean policy result is denied. Missing or invalid
session identities also fail closed. A failed store read or policy evaluation
is an operational failure, not a negative permission decision: the SDK preserves
store failures and reports policy-evaluation failure separately; HTTP returns a
controlled error without the underlying diagnostic text.

Ordinary Python SDK enqueue without source provenance remains available without
a policy. If a policy is configured, a source condition is supplied, or the
caller supplies a context, enqueue requires both a policy and verified context.
Inspection, source snapshots, withdrawal, and quarantine always require them.

Eval scenario queued-input execution uses a runtime-only enqueue entrance that
derives the fixed `cayu:eval-scenario` system actor, rather than trusting a
scenario-supplied actor. It retains the same scoped admission checks as public
enqueue. A target configured with `SessionMessageAccessPolicy`
requires future explicit application-owned context integration; without it,
scenario queued input fails closed. Scenario data and launch authorization do
not supply session-message context or bypass the policy.

All HTTP session-message operations require authentication **and** this scoped
policy, including ordinary enqueue. This is an intentional pre-release change:
an authenticated operator mount or local-development access alone no longer
enables the message endpoint. It does not change the authorization contract of
unrelated Cayu routes or introduce native tenant-partitioned storage.

HTTP derives context and action actors from `AuthContext`; request-body context,
metadata, or actor overrides cannot supply authorization. Enqueue rejects a
body `requested_by` even if it matches the authenticated actor. Terminal-action
bodies have no actor or context field. Protected responses, including errors,
are private/no-store and use bounded, non-disclosing errors.

## Python API

```python
from cayu import (
    EnqueueSessionMessageRequest,
    SessionMessageAccessContext,
    SessionMessageActionRequest,
    SessionMessageConditions,
    SessionMessageQuery,
)

# principal has already been authenticated by the application.
context = SessionMessageAccessContext(
    subject=principal.subject,
    tenant=principal.tenant,
)
source = await app.snapshot_session_message_source(
    source_session_id,
    context=context,
    include_transcript_digest=True,
    include_checkpoint_digest=True,
)
accepted = await app.enqueue_session_message(
    EnqueueSessionMessageRequest(
        session_id=target_session_id,
        idempotency_key="finding-42",
        content="Please recheck the deployment assumptions.",
        delivery_mode="next_turn",
        conditions=SessionMessageConditions(source=source),
    ),
    context=context,
)
page = await app.inspect_session_messages(
    SessionMessageQuery(session_id=target_session_id, limit=50),
    context=context,
)
record = next(item for item in page.records if item.queue_id == accepted.message.queue_id)
withdrawn = await app.apply_session_message_action(
    SessionMessageActionRequest(
        session_id=target_session_id,
        session_instance_id=page.session_instance_id,
        queue_id=record.queue_id,
        expected_revision=record.revision,
        idempotency_key="withdraw-finding-42",
        action="withdraw",
    ),
    context=context,
)
```

Pages contain at most 100 records in delivery priority order: `next_turn`
(priority 0), then `on_idle` (1), then unknown delivery modes (2), with FIFO
ordering within each priority. Terminal and unreadable rows are included.
The first page captures an acceptance high-water mark. Continue by passing its
`next_cursor` as `SessionMessageQuery.cursor`; `None` means pagination is complete.
The cursor binds the session incarnation, high-water mark, and last priority/FIFO
position. Messages admitted later do not enter that pagination pass; begin a new
query without a cursor to see them. Deleting/recreating the session invalidates
the old cursor. A cursor is a position, not an access grant: every page requires
fresh authorization. An unreadable record has `message=None`; use its row identity
and revision for quarantine, not a repaired or reserialized content payload.

`SessionMessageConditions` accepts optional `source`, `target`, and `expires_at`.
A `SessionMessageSource` binds session ID, incarnation, run epoch, transcript
cursor, and optional transcript/checkpoint SHA-256 digests. A
`SessionMessageTarget` requires the exact target incarnation, epoch, and cursor.
These fields are exact authority: secret-bearing identities or digests are
rejected, not redacted into different values. Ordinary message content still
passes through workload-secret redaction before durable admission.

A source session's public alias is resolved independently from the target and
still requires source authorization. Its exact private resolution is carried
only through the internal enqueue boundary, separately from request data. A raw
secret-bearing source ID does not gain that provenance by matching a stored
session. The existing prohibition on identities equal to an entire workload
secret remains in force even for aliases.

Source snapshots are protected observations, not executable capabilities.
Every derivative enqueue retry reauthorizes both sessions. The source must still
exist in the originally observed incarnation. Deleting it denies replay;
recreating the same ID does not restore authority, even if the caller receives
access to the new incarnation. An already accepted target record remains durable
and can still be inspected under target inspection permission. This intentionally
does not expose an old acceptance merely because its idempotency key is known.
Within the same authorized source incarnation, accepted retries retain the
original historical observation rather than recomputing its epoch or digests.
Target mismatch and expiry are delivery-time decisions owned by the store.
Withdrawal/quarantine use the inspected revision and a separate action
idempotency key. Retry the same action request after an acknowledgement loss;
do not substitute a new revision while reusing its key.

## Store extension guarantee

The base `SessionStore.session_message_lifecycle_version` defaults to `None`.
The in-memory, SQLite, and PostgreSQL implementations attest version `1`.
Conditional enqueue (any source, target, or expiry condition) checks both the
enqueue and delivery implementation owners before admission. Scoped enqueue,
even without conditions, checks the enqueue owner's v1 guarantee and carries
the policy-authorized target incarnation as a private admission fence. The
store must compare that incarnation atomically before replay or writes; it is
not a request JSON field and does not add epoch/cursor freshness conditions.
Inspection and source snapshots carry a separate private
`expected_authorized_session_instance_id` from the policy-authorized session.
Stores compare it under the read lock/transaction before queue-content hydration
or transcript/checkpoint digest access. A cursor never supplies this authority.
Inspection, terminal actions, and source snapshots check their respective implementation
owners. The declared value must be the integer `1`, not `True` or an unknown
future version. Missing support raises `NotImplementedError` before the queue
operation; HTTP reports unavailable support without accepting the message.

Unchanged inherited methods retain their owner's guarantee. A class overriding
any of these methods must explicitly redeclare
`session_message_lifecycle_version = 1` on the class defining the override and
implement the complete v1 contract, including atomic freshness/expiry checks,
terminal compare-and-set, exact replay, bounded unreadable-row inspection, and
source snapshot consistency. Merely inheriting the version attribute does not
attest an override. This is a current extension guarantee, not a compatibility
fallback for older queue implementations. Ordinary unscoped, unconditioned SDK
enqueue does not require the new capability and passes no admission-fence keyword
to the store.

## HTTP API

| Operation | Endpoint | Request |
| --- | --- | --- |
| Enqueue | `POST /api/sessions/{session_id}/messages` | Existing enqueue body plus optional `conditions`; SSE acceptance |
| Inspect | `GET /api/sessions/{session_id}/messages` | `limit` plus the four optional cursor fields below |
| Source snapshot | `POST /api/sessions/{session_id}/messages/source-snapshot` | Boolean `include_transcript_digest` / `include_checkpoint_digest` |
| Withdraw | `POST /api/sessions/{session_id}/messages/{queue_id}/withdraw` | `session_instance_id`, `idempotency_key`, `expected_revision` |
| Quarantine | `POST /api/sessions/{session_id}/messages/{queue_id}/quarantine` | Same terminal-action body |

Inspection and source snapshots return typed JSON. Terminal actions return a
typed record, the content-free lifecycle event, and `replayed`. That protected
record may contain valid message content; the event must not. Do not copy the
whole action response into public event streams or logs.

For the first HTTP inspection page, omit all cursor parameters. For each next
page, map every field of `next_cursor` to its prefixed query parameter:
`cursor_session_instance_id`, `cursor_through_ordering_key`,
`cursor_after_priority`, and `cursor_after_ordering_key`. Supply all four together;
partial, duplicate, unknown, or invalid query fields are rejected. `limit` is
1–100 (default 50); priority is 0–2 and the after-ordering key cannot exceed the
high-water mark. There is no standalone `after_ordering_key` pagination parameter.

Message enqueue rejects `Last-Event-ID`; it never replays general session
history. To recover a lost acceptance response, repeat the identical request
body, including its idempotency key, without that header. A matching retained
acceptance is returned after authorization is checked again.
This also applies after delivery or terminal rejection: SDK replay preserves the
message's actual status and verifies its acceptance/delivery references, while
HTTP returns the original acceptance event. Inspection and terminal-action
responses additionally require verified terminal evidence.
Queue record event references use the same public aliases as returned events,
using retained events that match the exact session, queue, event ID, and event
type. Side-effect sequence receipts alone do not prove that tuple. Inspection
revisions remain exact raw-store revisions, not hashes of the projected response.
SQLite pruning retains canonical queue-linked lifecycle events while the queue
exists, including terminal and unreadable records. Inspection and exact terminal
action replay therefore retain their public aliases after pruning and app restart;
unrelated history remains eligible for pruning. Missing or conflicting
terminal event evidence fails closed instead of exposing a raw ID or guessing its
public alias. A completed lookup proving an invalid acceptance pointer makes
the public record unreadable (`message=None`), preserving its raw revision so
it can be quarantined. Quarantine and replay retain that damaged pointer in
storage; their public terminal reference still requires exact verified event
linkage. Operational lookup failures propagate, rather than being classified
as unreadable content.

Inspection and terminal actions do not claim a run, call a provider, answer
`ask_user`, or resolve pending approvals. Those pauses retain their dedicated
resolution APIs.

## Store deployment

SQL schema revision 83 adds message conditions, terminal-action receipts, and
the rejection-only delivery mode. Stop workers sharing the store before applying
the migration, then restart them on the updated implementation; do not mix
workers that understand these constraints with workers that can ignore them.

Custom stores participating in protected lifecycle operations must explicitly
attest `session_message_lifecycle_version = 1` on the class implementing those
operations. The attestation covers atomic source validation, delivery-time target
and expiry checks, bounded inspection, and exact terminal-action replay. Merely
accepting the request fields is not sufficient. An overridden operation does
not inherit another implementation's attestation.
