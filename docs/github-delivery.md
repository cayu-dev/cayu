# GitHub pull-request, checks, and review delivery

Cayu's optional GitHub connector consumes the exact `next_commit` and
`next_ref` produced by [approved remote Git delivery](remote-git-delivery.md).
It can create or update one bound pull request, observe required checks and
bounded review feedback, and produce provenance-labelled input for a new coding
run. It never pushes Git objects and never merges.

Generate both delivery layers with:

```bash
cayu new mycoder --preset coding --execution docker --with github-delivery
```

`github-delivery` implies `remote-git-delivery`. The generated
`integrations/github.py` configures the host-only repository/installation
mapping, `SecretRef`, resolver, API origin, egress identity, and fixed-operation
transport. Neither the token, API network, owner/name mapping, nor connector is
given to the coding agent or Docker environment.

## Immutable request and approval

`github_pull_request_delivery_request()` binds one request to the immutable
coding-product artifact and pushed remote-Git result. It also binds:

- configured repository alias, repository/installation/account identities;
- exact base and head refs and commits;
- create mode or one exact existing pull-request number for update;
- bounded title, body, labels, reviewers, teams, and draft state;
- an application-declared required-check set and permitted conclusions;
- explicit approvers and minimum approval count, follow-up policy, and finite
  polling/response/feedback bounds; and
- allowed operations plus connector, credential, egress, policy, approval, and
  redaction identities.

Unknown fields fail closed. Merge is a literal `false`; raw endpoints,
repository owner/name, tokens, arbitrary operations, requests, refs, commits,
merge methods, workflow dispatch, releases, and branch deletion are not request
fields. Every requested metadata effect needs its named operation, and
`approve_github_delivery()` binds the complete request, metadata, policy, and
operation set before mutation.

## Fixed host-side operations

`GitHubRestTransport` implements only connector-owned ref observation, exact PR
find/get/create/update, labels, reviewer requests, one fixed draft-to-ready
GraphQL mutation, and bounded check/review reads. It rejects redirects, bounds
response bytes and time, classifies malformed JSON, rate, permission, timeout,
transport, and provider failures, and resolves its token from the configured
vault only at the HTTP boundary. Enterprise REST origins ending in `/api/v3`
derive the corresponding `/api/graphql` endpoint; origins must be explicitly
configured as one credential-free HTTPS origin/path.
Ready-to-draft conversion is unsupported: a request requiring draft state for
an already-ready PR is denied before any metadata mutation.

`GitHubConnectorTransport` is the semantic transport seam. An application may
implement it with a hardened `gh` process broker instead of REST, but it must
preserve the same fixed methods, authority, redaction, bounds, and recovery
contract. A raw `gh` argv or generic API-call tool is not equivalent.

## One durable observation per call

`GitHubPullRequestConnector.run()` never sleeps in an unbounded polling loop.
Calls for the same connector-run identity are single-flight within one connector
instance; an overlapping call is rejected until the owner settles, rather than
sharing a result without validating its own supplied receipts.
A distributed scheduler must likewise maintain one active owner per
run; durable reconciliation handles ownership loss and restart, not concurrent
multi-owner mutation.

Each call:

1. revalidates the exact upstream artifacts and configured identities;
2. freshly verifies the provider base and head refs;
3. reconciles the exact PR before any mutation retry;
4. requires durable approval for needed mutations;
5. freshly re-reads and verifies the PR binding; and
6. performs one bounded exact-head check and review observation.

A pending result includes `next_poll_after_seconds`; the application invokes
the same request identity again. Its durable `next_poll_at` prevents another
provider poll before that cadence, including after connector restart.
Durable poll count and elapsed-time authority
produce a truthful timeout. A previously persisted approval is reconstructed
and retained across restart and later observations; callers do not have to
re-supply credential-free approval evidence on every poll. Lost create/update,
label, and ready-state acknowledgements are reconciled from exact provider
state. A reviewer-request acknowledgement that cannot be observed remains
terminally ambiguous, so it is not repeated merely because its acknowledgement
was lost.

Every provider write has a durable intent before dispatch and a separate
settlement afterward. A crash between those publications leaves uncertainty,
not permission to repeat the write. When acknowledgement reconciliation succeeds,
the call can return `pr_updated` progress; another call resumes any remaining
approved effects before check/review settlement.
Reconciled `pr_updated` progress includes both polling fields so the same
scheduler resumes that unfinished work. Unresolved operation evidence remains
authoritative even if a later recovery read publishes a transient provider error.
The configured API target, owner/name mapping and credential reference are bound
durably to the request.

Cancellation during a provider mutation is also recorded as ambiguous, not as
proof that no effect occurred. Create, update, label, and ready-state recovery
re-observes provider state to prove the original effect; absence alone does not
authorize repeating a possibly in-flight write, and unresolved effects stay ambiguous;
an unobservable reviewer request remains non-repeatable. Durable request,
approval, lifecycle, and result reads verify artifact identity, session owner,
filename, content type, byte bound, content digest, and lifecycle continuity.

The public call's total wait is bounded by `timeout_seconds`. Cancellation or
deadline expiry does not cancel opaque dispatched work: a retained owner settles
it, prevents reuse of that run while unsettled, and starts no further provider
effects once the caller has stopped waiting. At most 64 run owners are retained
per connector. Artifact settlement observations keep a finished coroutine from
being mistaken for a finished background artifact mutation. Applications must
keep the connector and its event loop alive while owners drain; replacing a live
owner is not crash recovery. Durable effect intents support recovery after an
actual process restart without blindly repeating uncertain writes.

Transport failures cross a bounded, content-free diagnostic boundary, including
response-close failures. Process-control signals remain catchable by their
ordinary Python handlers; background ownership does not turn them into an
unhandled worker-task exit. Unclassified mutation failures remain ambiguous.

## Independent settlement

The result reports pull-request state, draft/merged/mergeability fields, exact
check state, review state, provider operations, and bounded feedback separately.
`checks_passed` does not mean approved, mergeable, or merged. `approved` reports
retained review approval and can coexist with pending checks; callers must read
`checks_state` independently. It never means merged. Missing, pending, failed,
cancelled, timed-out, superseded, ambiguous, rate-limited, denied, closed, and
permission failures remain explicit.

Check and review observations are exact-head bound. Feedback is deduplicated by
provider ID, secret-redacted, byte-bounded, and labelled as untrusted provider
evidence. Provider-set and body bounds are exposed through `checks_truncated`
and `feedback_truncated`; truncated settlement is `partial` and cannot overclaim
success. Provider pagination is treated as explicit truncation instead of
guessing from page length, and historical commit statuses are reduced to the
provider's latest observation per context. A same-named check run and commit
status are independent required signals; both must pass. Feedback cannot grant
tools, credentials,
network, paths, delivery, approval, or merge authority.

Only reviews carrying the exact admitted commit and from the application-declared approver set count toward required
approval, and the configured minimum must settle. An approval from any other
GitHub user remains visible as bounded untrusted feedback but cannot satisfy the
application review gate.

Transient rate limits and read-side provider outages retain a bounded retryable
state and consume poll authority; they do not become permanent success or an
unbounded retry loop. Mutation timeouts remain ambiguous because the provider
may have applied the external effect.

`github_follow_up_coding_input()` turns an application-selected retained
feedback set into provenance-labelled messages and new caller-owned product,
session, and task IDs. It enforces the request-bound allow/deny decision plus
maximum evidence items and follow-up iterations. The application must pass that
through a new ordinary maintained coding-product run. Updating the PR then
requires a new approved remote-Git delivery and a new GitHub request bound to
its exact commit. The connector never edits files or runs code directly.
