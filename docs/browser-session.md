# Stateful browser sessions

`BrowserSessionTool` is Cayu's closed, provider-neutral interface for stateful
browser interaction. Applications opt in through an admitted browser
environment; the model cannot choose a browser backend, runner, image, proxy,
credential, header, launch argument, selector, script, CDP command, filesystem
path, or safety limit.

Use the interactive WebBridge profile at application setup:

```python
from cayu import DEFAULT_WEBBRIDGE_INTERACTIVE_BROWSER_IMAGE, WebBridge

browser = WebBridge.sandboxed_browser(
    environment=browser_environment_factory,
    browser_image=DEFAULT_WEBBRIDGE_INTERACTIVE_BROWSER_IMAGE,
    interactive=True,
)
for tool in browser.tools:
    app.register_tool(tool)
```

The environment or factory must prove the exact
`cayu-browser-fetch:6-playwright-1.62.0` image, the
`cayu.browser-session.v2` protocol and worker version 6, brokered deny-by-default egress,
confirmed cancellation and cleanup, and one stable ArtifactStore. Construction
is side-effect-free for factories; the same candidate, workload, and artifact
authorities are checked again after materialization. There is no fallback to
host Playwright, host HTTP, another provider, a CLI, or MCP.

## Model contract

One ordinary `browser_session` tool exposes only `navigate`, `observe`,
`click`, `fill`, `select`, `press`, bounded `wait`, `screenshot`, `download`,
and `close`. The first navigation creates Cayu-owned opaque `session_id` and
`page_id` values. Every observation returns:

- an opaque page `revision` and revision-bound Cayu element refs;
- a byte- and ref-bounded Playwright AI-mode ARIA snapshot;
- canonical URL, bounded title, load/access state, and truncation reasons; and
- exact worker, Playwright, Chromium, and protocol identity.

Playwright `aria-ref` values remain private inside the guest worker. Every
operation requires a stable `operation_id`. Cayu
replaces them with random opaque refs and resolves them only through strict
Playwright locators. Ref actions require the matching Cayu session, page,
revision, ref, and a stable `operation_id`. Cayu rejects a stale revision or
unknown ref before runner dispatch. Once an action is admitted, the old refs
are invalid even if the action fails; observe again before interacting.

The worker keeps one bounded live Playwright allocation with one page/tab
inside the selected runner, so page state, cookies, web storage, tab identity,
and navigation state survive ordinary model turns for that allocation's
lifetime. A pre-document guard denies explicit and inherited browsing-context
targets and both ordinary and prototype `window.open` calls; the context popup
observer remains a fail-safe that retires the allocation if an extra page still
appears. The default idle lifetime is 900 seconds and
applications may configure `idle_timeout_seconds` from 1 through 3,600. The
deadline resets only after an operation has produced its response; expiration
waits for an already-admitted operation and rejects newly queued work before
cleanup, so it cannot close Chromium halfway through an action. Every positively
settled daemon exit, including startup failure and idle cleanup, records a
bounded guest-owned retirement marker. The launcher retains a separate bounded
startup-cleanup settlement window so a marker published just after the ordinary
connection deadline can release parent capacity; a missing marker remains
outcome-ambiguous and capacity-bearing.

Requests, redirect hops, response bytes, URL/title bytes, observation bytes,
DOM nodes, snapshot depth, refs, operation wait, artifact bytes, screenshot
width, height, and pixel count, live allocations, parent-session state, and
operation identities have independent application-owned ceilings. DOM-node and
accessibility-source admission share one script-and-animation-frozen page window before
Playwright materializes the depth-bounded AI snapshot. Cayu also applies a
conservative aggregate expansion ceiling across nodes, source-derived names,
serialization escaping, and computed pseudo-element content. The source ceiling
leaves bounded room for ordinary output truncation without admitting an
unbounded accessible scalar or repeated-name amplification.
Full-page geometry is measured and admitted before Chromium captures the
raster. Downloads are cancelled as soon as request policy or the response-byte
ceiling fails and are read only after a bounded regular-file result settles.
`close` remains admissible after the
normal operation-id table is full and owns a separately bounded idempotency
receipt. It settles browser/context/driver cleanup before acknowledging;
cleanup failure returns `cleanup_failed`. Environment teardown remains the
outer cleanup fence.

## Effects, evidence, and failure

The tool follows Cayu's ordinary policy, approval, taint, execution-profile,
effect, budget, hook, cancellation, event, projection, and transcript paths.
Its structured result distinguishes admission, dispatch, observation
publication, and terminal classification. A runner failure after dispatch is
`outcome_ambiguous`; the operation result is bound to its `operation_id` and is
never automatically replayed. Caller cancellation remains authoritative, but
the in-process operation record is also sealed as ambiguous before cancellation
escapes so an immediate retry cannot repeat the action. Reusing an operation ID
with different arguments fails before dispatch.

Screenshots and downloads are independently byte-bounded and published to the
active ArtifactStore. Model-visible results contain artifact references only,
never raw bytes, base64, or guest paths. Binary artifact capture fails closed
with `policy_denied` when the invocation already owns resolved credentials, the
runner declares virtual-egress/output secret values, or the invocation registry
changes during dispatch; Cayu does not attempt to redact secrets from rendered
pixels or downloaded bytes. Browser exceptions and stderr are not published;
callers receive stable bounded codes such as `destination_denied`,
`fetch_failed`, `stale_observation`, `unknown_element`, `actionability_failed`,
`navigation_timeout`, `download_failed`, `browser_crash`, `cleanup_failed`, or
`outcome_ambiguous`.

Credentials remain application-owned credential/egress authority. Runner
handles expose only a tri-state secret-presence declaration for this admission
decision; no credential value crosses into the tool. Do not place credentials
in model arguments or URLs. Textual worker output passes through
the invocation runner's evolving secret-redaction boundary, and the complete
URL/title/snapshot projection is wrapped and escaped as untrusted evidence.
Page snapshots, titles, URLs, and download names remain untrusted even when
browser execution itself is admitted.

## Worker-loss recovery

Durable recovery is available only when the runtime has both an exact execution
profile and a reconnectable environment-allocation receipt. Before runner
dispatch, Cayu binds the browser operation to the parent session and run epoch,
model attempt, tool round and call, idempotency key, execution profile,
environment name, and opaque allocation fingerprint. It publishes one durable
intent, advances it to `dispatched`, and permits at most one terminal receipt.
The guest worker independently binds the same `operation_id` to one exact
request and returns its retained response for an exact duplicate. A conflicting
request is `operation_conflict` and is never executed.
The same fenced parent record carries the bounded normal-operation count,
cleanup-operation count, and live browser-session identities. Those ceilings
therefore do not reset when a fresh Cayu process reconnects, while `close`
retains its separate bounded cleanup allowance.

A fresh Cayu process may reconnect only to the exact still-live allocation
identified by that durable receipt. It reconstructs the last terminal browser
session/page revision and opaque refs, then revalidates the materialized runner
and worker before dispatch. A recovered `observe` uses a new operation identity,
advances the revision, and publishes fresh refs. Evidence marked uncertain does
not authorize pre-loss refs; observe again before an action. The durable
continuity/session records contain only opaque identities, bounded status,
revisions, and refs—not cookies, local/session storage, profile files,
credentials, page content, or artifact bytes. A sealed terminal operation
receipt necessarily retains its bounded `ToolResult`, including bounded URL,
title, snapshot, and refs needed for exact replay; it never retains raw binary
artifact bytes or browser-profile contents.

Recovery does not automatically redispatch any browser operation. An intent
that never reached dispatch becomes `operation_not_dispatched`. A terminal
receipt is replayed exactly. A dispatched operation without a terminal receipt
becomes `outcome_ambiguous`, with the known browser session/page identities and
guidance to avoid replay. The other recovery categories carry bounded guidance
for explicit restart, matching-profile resume, or outer cleanup. This applies
to observations too: although a new
observation is safe under the same admitted live allocation, pending-round
recovery itself remains read-only and never calls the browser. Effectful clicks,
fills, submits, key presses, and downloads are never retried after ambiguity.

The failure categories are deliberately distinct:

- `allocation_lost` means the durable session names a different or unavailable
  live allocation;
- `incompatible_profile` means the execution profile changed;
- `authority_expired` means the parent/tool authority or durable record no
  longer matches;
- `restoration_required` means no exact live-allocation continuity exists;
- `outcome_ambiguous` means dispatch may have produced an external effect; and
- `cleanup_failed` means explicit browser cleanup did not settle.

These terms are not interchangeable. A **live-allocation reconnect** continues
the same admitted browser process and its surviving page, cookies, storage, and
navigation state. **Browser-profile restoration** would recreate only a
separately declared persisted subset; Cayu does not currently offer that mode.
A **page reload** is a new navigation and may change external state. An
**operation replay** repeats an old request and is forbidden after ambiguity. A
**full execution-environment snapshot** would preserve process/VM state; this
contract makes no such claim. JavaScript heap objects, open sockets, in-flight
downloads, and arbitrary process state are not reconstructible if the live
allocation itself is lost.

Access-block classification and explicit fallback routing belong to a separate
browser-access contract; this recovery boundary does not infer or select a
fallback. A blocked main-document response publishes only typed transport
evidence with an origin-only URL and an empty title/snapshot/ref set. Denial-page
text never executes or becomes fallback evidence: the guest intercepts the
classified response before body execution, while broker egress denial remains
authoritative. See
[WebBridge access routing](web-fetch.md#classified-access-barriers-and-explicit-routing).
