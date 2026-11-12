# ADR 0005: GitHub pull-request delivery through connections

- Status: proposed.
- Date: 2026-11-12.
- Context: the GitHub connector work: moving pull-request delivery onto the connection contract
  of [ADR 0003](0003-connections-runtime-contract.md).

## Context

`cayu.delivery.github` delivers one approved commit as one pull request. It keeps a durable
request, approval, and an intent and settlement record for every effect (create, update, labels,
reviewer requests, mark ready), observes checks and reviews once per call, and never merges. It
reaches GitHub through `GitHubConnectorTransport`, a fixed set of ten operations, whose only
implementation, `GitHubRestTransport`, opens its own HTTP client and resolves a GitHub token from a
`SecretRef` for every call.

`cayu.connectors.github` reaches GitHub through a connection slot: Cayu Cloud's broker, which holds
the GitHub credential so the Agent never does, or, off Cloud, a token or a GitHub App key.
Keeping both would mean two ways to call GitHub and two credential models, and pull-request
delivery could never run on Cayu Cloud without a GitHub secret in the Agent.

The delivery contract has no known defect and its behaviour is pinned by
`github_connector_behavior_fingerprint()`, which every durable request records. Today, any failed
write is treated as possibly carried out (ambiguous): delivery reconciles it from GitHub's state and
never repeats an effect it hasn't observed.

## Decision

1. **The contract stays.** `GitHubConnectorTransport`, the request, approval and result schema, the
   error codes and the behaviour fingerprint are unchanged.
2. **One way to reach GitHub.** `GitHubRestTransport` sends every call through a
   `ConnectionClient` slot and has no HTTP client of its own. A configured repository's credential
   is one of:
   - `GitHubCredentials` (a `SecretRef`), resolved for each call and scoped to the repository, as
     today, behind a private connection;
   - a connection slot: Cayu Cloud's broker, or a local token or GitHub App connection.
3. **Delivery keeps its own reading of GitHub's answers.** It sends each call once and classifies
   the status, headers and body itself, as it does today. It doesn't take on the general GitHub
   client's waits, resends or error mapping: that policy serves agents and can change, while
   delivery's behaviour is fixed by its fingerprint. It uses the connector's API version, media
   type and repository header.
4. **Which failures are ambiguous.**
   - A write that GitHub answered with any failure, an unreadable or oversized answer, a lost
     answer, a timeout or a cancellation is ambiguous, as today.
   - A connection failure on a write is ambiguous unless the connection says it refused the call
     before sending (`ConnectionCallError.ambiguous` is false, ADR 0003). Those fail cleanly:
     `rate_limited` (retryable) for too many calls, `provider_unavailable` (retryable) when the
     call's credential couldn't be had, and a new code, `connection_refused`, for the rest.
   - Reads fail as today; an answer over the connection's 5 MiB limit is
     `provider_response_truncated`.
5. **Bounds.** Each call stays within `timeout_seconds`; through Cayu Cloud the broker also ends a
   call at 25 seconds. Answers are bounded at 5 MiB by the connection, then checked against
   `max_response_bytes`.
6. **Durable binding.** A `SecretRef` configuration binds exactly the same credential reference as
   before, so deliveries in progress continue across the upgrade. A slot credential binds a
   separately tagged document. Changing an existing delivery's credential form fails closed: the new
   form applies to new deliveries.
7. **Identity and redaction with a slot.** The connection, not the application, picks GitHub's host
   and the App installation, so a slot configuration's API base, installation and account are
   recorded as declared, not verified. There is no secret in the process to redact.
8. **GitHub Enterprise Server.** Local token and App connections send GraphQL to `/api/graphql`
   when their API base ends in `/api/v3`. Cayu Cloud's GitHub connector serves github.com.
9. **Git pushes are out of scope.** Pushing a commit uses its own remote Git credentials; removing
   that secret is a separate decision.
10. **Order.** The change is a follow-up and lands when an application needs pull-request delivery
    on Cayu Cloud, together with Cayu Cloud's GitHub connector.

## Consequences

- Cayu has one credential model for GitHub; on Cayu Cloud a delivering Agent holds no GitHub API
  credential.
- Delivery's classifier (about 70 lines) remains beside the general client's. That is deliberate:
  it is the durable policy the fingerprint names, stricter than the agent client's.
- A busy Agent's broker limit, or a slot refused before sending, fails a write cleanly and is retried
  or reported, rather than leaving the delivery ambiguous. This relies on ADR 0003's rule that an
  upstream never refuses a write after sending it with a before-sending code; Cayu Cloud's GitHub
  connector is tested against it.
- Through Cayu Cloud each call is slower (one more hop, and a token minted on first use). A run's
  calls share one `timeout_seconds` budget, so a slot configuration may need a larger one.
- `GitHubCredentials` remains a supported credential form; nothing is deprecated.

## Alternatives considered

- **A second transport on the general GitHub client, beside the current one.** Leaves two HTTP and
  credential paths inside delivery, permanently.
- **Delivery on the general GitHub client as its only layer.** Removes the most code, but maps the
  client's typed errors back to delivery's codes, and the client treats a refused write (401, 403,
  404, 422, 429, a redirect) as not carried out where delivery treats it as ambiguous. That would
  change recovery behind an unchanged fingerprint, and future client changes would keep doing so.
- **Keep delivery separate, with the boundary written down.** Free, but keeps two credential models
  and no way to deliver on Cayu Cloud without a GitHub secret in the Agent.
- **Re-express delivery on the connector's pull-request reader and write-once markers.** Markers make
  comments and issues idempotent; they don't provide approval binding, per-effect intent and
  settlement, or exact-head check settlement. The reader waits for mergeability inside a call, and a
  new fingerprint would strand deliveries in progress.
