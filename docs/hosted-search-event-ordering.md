# Hosted-search event ordering investigation

The hosted-search ordering investigation remains an investigation. Its historical
`web_search_lifecycle_arrived_before_output_item_added` diagnostic proves only
that the native adapter had no pending search at the received output index.
It cannot identify the original wire order or assign responsibility to a provider,
proxy, or Runtime. No observed incident wire fixture is available here.

## Synthetic controls and state transitions

`tests/core/test_openai_search_ordering.py` sends credential-free synthetic SSE
through `HttpxOpenAITransport`, the native adapter, Runtime dispatch/retries, and
SQLite durable event readback. JSON frames are split across byte chunks.
`A(i)` is `response.output_item.added` with an in-progress `web_search_call`,
`L(i)` is a hosted-search lifecycle event, and `D(i)` is
`response.output_item.done` with completed action/source evidence.

| Sequence | Expected and verified behavior |
| --- | --- |
| A(0), L(0), D(0), response.completed | Register, report progress, settle with source evidence, emit one model completion |
| A(0), A(1), L(1), L(0), D(1), D(0), response.completed | Independent pending entries; both searches settle, one model completion |
| L(0) | No registration: existing `web_search_lifecycle_arrived_before_output_item_added` reason |
| A(0), L(1) | No registration at index 1: same reason; trace shows index discrepancy |
| A(0), D(0), L(0) | Registration already removed: `web_search_lifecycle_arrived_after_output_item_done` |
| A(0), D(0), D(0) | `web_search_call_output_item_done_was_repeated` |
| A(0), A(0) | Existing repeated-add rejection |
| A(0), L(0) with different item_id | Existing identity-mismatch rejection; trace records `differs` |
| A(0), L(0), EOF | Existing early-stream-end rejection and unknown hosted outcome |
| Failed attempt, valid attempt using the same synthetic IDs | New registration and trace state; one successful model completion |

Repeated lifecycle progress remains idempotent. An omitted lifecycle `item_id`
retains existing compatibility behavior; this change does not make it required.
`response.completed` seals semantic state. The adapter validates subsequent
transport input and rejects post-terminal events before they can produce an
executable tool call. Completion already accepted before a tail failure remains
accounting evidence; it does not authorize a successful continuation after that
failure. Invalid pre-terminal streams do not produce model completion. The
late-lifecycle control can have an already settled hosted call before its model
attempt fails; that evidence is not a successful model completion.

## Bounded diagnostic contract

Ordering/identity guards retain the existing scalar fields:

- `provider_protocol_stream_boundary`: `native_adapter`.
- `provider_protocol_stream_trace`: compact JSON with at most 16 rows.
- `provider_protocol_stream_trace_truncated`: 0 or 1, indicating dropped history.

Each row contains, in order:

1. Received event ordinal, starting at 1 for each parser invocation and saturating
   at 1,000,000. This is arrival order, not the upstream `sequence_number`.
2. Allowlisted event type; unrecognized types become `other`.
3. Output index, or -1 when absent/invalid/larger than 1,000,000.
4. Search registration state immediately **before** processing this event:
   `pending`, `completed`, or `absent`.
5. Item identity relation to the pending/completed search at that index:
   `matches`, `differs`, `unregistered`, `missing`, or `invalid`.
6. Response identity relation to the response registered by `response.created`,
   using the same vocabulary. Events without a response identity say `missing`.

The ring is constant-size, contains no raw IDs, hashes, query/source/body content,
headers, credentials, endpoint identity, or exception text, and serializes to
less than 4 KiB. Error projection revalidates every field. The existing output
item map supplies completed-state lookup; diagnostics add no unbounded identity
registry. Trace rows do not enforce new response/item identity requirements.
The trace is published on ordering/identity failures and other protocol failures
after search activity, including malformed lifecycle identities and repeated
identities rejected by the outer settlement guard. Successful responses do not
publish traces. Retry attempts retain separate
error receipts and start with empty trace/registration state.

## Cross-index identity and boundary structure

`provider_protocol_stream_identities` adds rows of `[ordinal, relation, matched_index]`
aligned with the existing trace. Relations are `missing`, `invalid`, `unknown`,
`pending_here`, `pending_elsewhere`, `completed_here`, `completed_elsewhere`, or
`ambiguous`. A unique match has its registered index (or -1 above 1,000,000);
other relations use -1. These relationships inspect the parser's current pending
and completed search maps before processing the event, including matches outside
the trace window. Multiple registrations with the same identity are ambiguous;
no registration is selected. Missing identities include omitted/null IDs; invalid
identities include non-string or blank values. Item comparison uses the adapter's
whitespace normalization. Existing reason codes and stream acceptance stay unchanged.

`provider_protocol_stream_item_types` adds aligned
`[ordinal, incoming_item_type, registered_item_type]` rows. Item types are allowlisted;
missing types become `missing` and unrecognized values become `other`.

`provider_protocol_native_structure` and `provider_protocol_transport_structure`
contain independently captured JSON rows:

```
[ordinal, event_type, index_status, index, incoming_item_type,
 sequence_status, sequence_number, identity_status, identity_alias]
```

Index and upstream sequence statuses are `present`, `missing`, `invalid`, or
`out_of_range`. Explicit nulls, booleans, negative numbers and non-integers are
invalid; values above 1,000,000 are out of range. Unavailable numbers use -1.
Runtime ordinals start at 1 and saturate at 1,000,000; supplied upstream sequences
are never inferred from arrival order.

Each boundary gives the first 128 distinct nonblank item identities local integer
aliases starting at 1. Identity status is `present`, `missing` (omitted/null),
`invalid`, `oversized` (over 1024 characters), or `exhausted`; unavailable aliases
use 0. Aliases never evict or reuse slots, and known aliases remain available after
exhaustion. Only keyed per-attempt digests are held internally for alias lookup;
raw IDs, digests and keys are never projected. All other content is excluded.
Each boundary retains the last 16 rows and reports separate
`provider_protocol_{native,transport}_structure_truncated` and
`provider_protocol_{native,transport}_aliases_exhausted` flags (0 or 1).

HTTP metadata travels outside decoded JSON in a private typed envelope. Custom
transports without this envelope omit transport evidence. Both boundaries start
fresh for each invocation/attempt. Aliases are comparable within an attempt when
the observed identity order agrees; they are not global identity identifiers.
An event inserted, removed or changed between boundaries can change alias
assignment and ordinals. Compare full structure, not isolated alias numbers.
Every new field is revalidated before API/subscription projection and durable
`model.error` persistence. Invalid diagnostic objects are omitted.

`tests/core/test_openai_search_identity_diagnostics.py` covers shifted matching
versus unrelated IDs, interleaving, completed and duplicate identities, missing
and malformed fields, bounded capture/exhaustion, hostile projection values,
retry isolation, SQLite readback, and a synthetic change between boundaries.
These controls are diagnostic evidence, not permission to reconcile identities.

## Remaining evidence and fault ownership

These synthetic invalid inputs establish the rejection location and eliminate
ambiguity in future diagnostics. They do not establish the cause of the original
incident. Runtime compares decoded HTTP transport and native adapter input; this does
not instrument an external upstream or proxy.

For a fresh occurrence, compare bounded structural metadata at upstream egress,
proxy ingress/egress (if present), and native adapter ingress for the **same
attempt**, using an authorized external attempt correlation. Use event order,
type, output index, and local identity-equality relationships; retain no raw
response dump or workload identifiers. If the relevant registration has fallen
out of the 16-event window, the trace explicitly remains incomplete evidence.
Matching malformed sequences across boundaries would locate where the sequence
first appeared; a valid sequence arriving intact but misregistered would support
an adapter defect. Neither outcome is yet established.

The rejection policy, retry caps, and hosted-effect accounting are unchanged.
No buffering or permissive orphan handling is justified by these controls.
