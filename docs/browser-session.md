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
`cayu-browser-fetch:7-playwright-1.62.0` image, the
`cayu.browser-session.v3` protocol and worker version 7, brokered deny-by-default egress,
confirmed cancellation and cleanup, and one stable ArtifactStore. Construction
is side-effect-free for factories; the same candidate, workload, and artifact
authorities are checked again after materialization. There is no fallback to
host Playwright, host HTTP, another provider, a CLI, or MCP.

## Model contract

One ordinary `browser_session` tool exposes only `navigate`, `observe`,
`click`, `fill`, `select`, `press`, bounded `wait`, `screenshot`, `download`,
`list_pages`, `switch_page`, `close_page`, and `close`. The first navigation
creates Cayu-owned opaque `session_id` and `page_id` values. Cayu does not
currently expose `new_page`: additional pages can arise only as a
policy-admitted effect of an action on the active page. Every observation
returns:

- an opaque page `revision` and revision-bound Cayu element refs;
- a byte- and ref-bounded Playwright AI-mode ARIA snapshot;
- canonical URL, bounded title, load/access state, and truncation reasons; and
- exact worker, Playwright, Chromium, and protocol identity.

Playwright `aria-ref` values remain private inside the guest worker. Every
operation requires a stable `operation_id`. Cayu
replaces them with random opaque refs and resolves them only through strict
Playwright locators. Ref actions require the matching Cayu session, page,
revision, control epoch, ref, and stable `operation_id`. Cayu rejects a stale
revision, stale control epoch, cross-page ref, or unknown ref before runner
dispatch. Once an action is admitted, the old refs are invalid even if the
action fails; observe again before interacting. Switching pages invalidates
both the prior and selected page namespaces and returns a fresh observation for
the selected page. Closing a page invalidates that page and deterministically
selects the earliest surviving admitted page when possible.

The default remains single-page mode. Its pre-document guard denies explicit
and inherited browsing-context targets and both ordinary and prototype
`window.open` calls; the page popup observer and context-wide request guard
remain fail-safes that retire the allocation if an extra page still appears.
Applications opt into bounded multi-page behavior with `multi_page=True`, an application-owned
`BrowserPopupPolicy`, and finite page limits. The model cannot relax the policy
or select a new context, profile, proxy, credential, header, extension,
provider, or runner in an action. Popup authority can be granted only around an
explicit post-navigation `click`, `fill`, `select`, `press`, or `wait`; initial
`navigate` never receives popup authority, so document-load scripts start with
the pre-document guard closed. For example:

```python
from cayu import (
    DEFAULT_WEBBRIDGE_INTERACTIVE_BROWSER_IMAGE,
    BrowserPopupPolicy,
    WebBridge,
)

browser = WebBridge.sandboxed_browser(
    environment=browser_environment_factory,
    browser_image=DEFAULT_WEBBRIDGE_INTERACTIVE_BROWSER_IMAGE,
    interactive=True,
    interactive_options={
        "multi_page": True,
        "popup_policy": BrowserPopupPolicy(
            mode="destination_policy",
            allowed_operations=("click",),
            allowed_opener_origins=("https://app.example/",),
            allowed_destination_origins=(
                "https://app.example/",
                "https://login.example/",
            ),
        ),
        "max_pages": 3,
        "max_provisional_pages": 1,
        "max_page_creations_per_operation": 1,
        "max_total_page_creations": 8,
    },
)
```

`list_pages` returns only the bounded page registry: opaque IDs, lifecycle and
lineage, revision/control identities, bounded canonical URL/title/access/load
state, terminal reason, and counters. It never returns DOM, history, cookies,
storage, screenshots, Chromium targets, CDP sessions, or window handles.
`switch_page` and `close_page` accept only Cayu page IDs. All pages remain in
one browser context, so they intentionally share cookies, web storage, selected
profile authority, credential routing, and egress policy. They are tabs in one
security boundary—not independent browser profiles or independent security
boundaries.

A popup begins as an untrusted provisional effect. A token-gated context init
guard is installed before the first page is created, defaults closed in every
new document, and is armed only for the exact bounded application-admitted
action. The context-wide route guard applies the same brokered
egress, destination/redirect/access checks, response/request limits, download
policy, and credential isolation throughout `about:blank`, opener inheritance,
immediate redirect, and self-navigation. Model-visible admission occurs only
after that transition settles inside the configured popup policy. A denied or
over-capacity popup is closed through bounded cleanup and appears only as a
bounded refusal. A popup burst cannot grow the registry, cleanup queue, event
stream, or diagnostics beyond the configured limits.
Initial popup document requests support GET. Non-GET requests, including POST
forms targeting a new page, are refused before network dispatch; they are never
converted into GET requests. Provisional requests follow their exact browser
frame identity independently of URL changes or callback order.
Downloads are admitted only for the exact active-page `download` operation;
an automatic or popup-initiated download is cancelled and its page is
quarantined through the same bounded cleanup owner.

One action that creates pages is still one operation. Its terminal result
contains a bounded `page_delta` and complete bounded `page_set`. An
acknowledgement lost after popup creation is `outcome_ambiguous`; Cayu never
re-clicks to recreate a page. An exact duplicate may recover the original delta
only from the same live allocation's exact guest receipt. The default idle
lifetime is 900 seconds and
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
width, height, and pixel count, live and provisional pages, page creations per
operation and per allocation, background lifetime, per-page and aggregate
operations/observations, per-observation and cumulative per-page/aggregate
refs, per-page/aggregate requests and artifacts, page cleanup, live
allocations, parent-session state, and operation identities have independent
application-owned ceilings. Construction rejects unbounded or inconsistent
settings. `max_refs` bounds one observation, while `max_refs_per_page` and
`max_total_refs` independently bound cumulative allocation consumption.
Per-page and page-set `ref_count` values are cumulative allocation
consumption counters rather than the number of refs still actionable; only refs
from the active page's exact latest returned observation carry action authority.
DOM-node and
accessibility-source admission share one script-and-animation-frozen page window before
Playwright materializes the depth-bounded AI snapshot. Cayu also applies a
conservative aggregate expansion ceiling across nodes, source-derived names,
serialization escaping, and computed pseudo-element content. The source ceiling
leaves bounded room for ordinary output truncation without admitting an
unbounded accessible scalar or repeated-name amplification.
Full-page geometry is measured and admitted before Chromium captures the
raster. Downloads are cancelled as soon as request policy or the response-byte
ceiling fails and are read only after a bounded regular-file result settles.
`close_page` and `close` remain admissible after the normal operation-id table
is full and own separate bounded idempotency/cleanup capacity. Whole-session
close settles every admitted or provisional page, pending cleanup task,
context, browser, driver, and temporary profile before acknowledging. One page
failure cannot be hidden by another resource closing successfully; failure is
reported as bounded `cleanup_failed`/uncertain evidence. Environment teardown
remains the outer cleanup fence.

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
identified by that durable receipt. It reconstructs the exact bounded surviving
page registry, counters, lifecycles, revisions, control epochs, and ref
authority, then revalidates the materialized runner and worker before dispatch.
Closed or crashed targets are reconciled from guest evidence; uncertain pages
do not authorize pre-loss refs and require a new observation before an action.
Pending recovery itself never lists, switches, closes, creates, navigates, or
otherwise operates a page. A recovered `observe` is a new admitted operation
with a new identity. The durable continuity/session records contain only opaque
identities, bounded safe status and counters, revisions, and refs—not cookies,
local/session storage, profile files, credentials, page content, history,
screenshots, downloads, or Chromium identifiers. A sealed terminal operation
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
the same admitted browser process and its exact surviving page set, cookies,
storage, and navigation state. Losing that allocation loses the entire page set;
Cayu never rebuilds tabs from stored URLs or history. **Browser-profile
restoration** creates a fresh allocation and page set and imports only an
application-declared cookie and origin-local-storage snapshot. A **page reload**
is a new navigation inside one allocation and may change
external state. An **operation replay** repeats an old request and is forbidden
after ambiguity. A **full execution-environment snapshot** would preserve
process/VM state; this contract makes no such claim. Profile restoration never
reconstructs prior pages, URLs, revisions, refs, operation results, JavaScript
heap objects, sockets, in-flight downloads, or arbitrary process state.

## Application-owned browser profiles

Browser-profile restoration is an explicit composition option for the pinned
interactive worker. The application creates a dedicated profile store and key
authority, defines ownership and sharing scope, fixes the maximum destination
policy, and passes the resulting immutable binding to `WebBridge`. None of
those values appears in the `browser_session` tool schema, so the model cannot
select or switch profiles, stores, keys, tenants, generations, merge behavior,
or checkpoint policy.

```python
from cayu import (
    AESGCMBrowserProfileKeyAuthority,
    BrowserProfileBinding,
    BrowserProfileCheckpointPolicy,
    BrowserProfileDestinationPolicy,
    BrowserProfileScope,
    DEFAULT_WEBBRIDGE_INTERACTIVE_BROWSER_IMAGE,
    SQLiteBrowserProfileStore,
    WebBridge,
)
from cayu.tools.browser_session import (
    BROWSER_SESSION_PROTOCOL_VERSION,
    BROWSER_SESSION_WORKER_VERSION,
)

profile_store = SQLiteBrowserProfileStore(
    ".cayu/browser-profiles.sqlite",
    store_id="production-browser-profiles",
)
profile_key = AESGCMBrowserProfileKeyAuthority(
    authority_id="browser-profile-key-2026-01",
    key=key_loaded_from_application_secret_storage,
)
profile = BrowserProfileBinding.build(
    scope=BrowserProfileScope.build(
        application_id="support-console",
        tenant_id="tenant-42",
        sharing_scope="support-agent-release-7",
    ),
    destination_policy=BrowserProfileDestinationPolicy.build(
        ("https://accounts.example.com",)
    ),
    browser_protocol=BROWSER_SESSION_PROTOCOL_VERSION,
    browser_worker_version=BROWSER_SESSION_WORKER_VERSION,
    store=profile_store,
    key_authority=profile_key,
    profile_id=browser_profile_id_from_control_config,
    created_at=browser_profile_created_at_from_control_config,
    checkpoint_policy=BrowserProfileCheckpointPolicy.ON_CLOSE,
)
await profile.initialize()

browser = WebBridge.sandboxed_browser(
    environment=browser_environment_factory,
    browser_image=DEFAULT_WEBBRIDGE_INTERACTIVE_BROWSER_IMAGE,
    interactive=True,
    browser_profile=profile,
)
```

The store is separate from `SessionStore`, `ArtifactStore`, `Workspace`, and
environment-variable storage. `InMemoryBrowserProfileStore` is process-local;
`SQLiteBrowserProfileStore` is the durable local implementation and should be
closed during application shutdown. The AES-GCM key value remains with the
application-owned key authority. Only a bounded encrypted envelope and
content-free authority, lease, receipt, count, timestamp, and status metadata
reach the profile store. Profile plaintext is private runner transport to the
pinned worker; it is not a tool argument, transcript item, ordinary event,
artifact, workspace file, diagnostic, or support-bundle field.

Custom stores implement only the protected create, load, list, and atomic
mutation primitives on `BrowserProfileStore`. The mutation primitive must run
the supplied callback exactly once under one store write boundary, using the
store-owned UTC observation passed to that callback, and must commit the
returned record together with its returned result. It must not override the
public lifecycle methods. Custom key authorities keep key material private and
implement the same AES-256-GCM operation selected by the binding; their async
calls are treated as opaque external work and remain owned until they return or
the configured import/export timeout proves the local wait has ended. The
binding defensively validates store and key-authority identities and every
returned authority-bearing model, but those checks do not turn a contract-
violating extension into a supported implementation.

For durable use, persist the immutable `BrowserProfileAuthority` in authenticated
application configuration, or reconstruct it with the same explicit `profile_id`,
`created_at`, scope, policy, protocol, worker, store, and key-authority identities.
Calling `BrowserProfileBinding.build()` with generated identity defaults after a
restart creates a different profile; it does not discover or adopt an existing
one by store contents alone. Reinitializing the exact same immutable authority is
idempotent and returns the store's current generation rather than recreating or
rolling it back.

For a profile-bound interactive tool, the pinned worker rechecks current cookie
and local-storage values before every textual observation and omits content that
would reflect them. Cayu rejects screenshot and download operations before
dispatch because binary output cannot be redacted soundly. Use a separate,
credential-free browser tool when an application needs those capture operations.
Page summaries omit titles and URLs for profile-bound sessions, including listing,
background-page, and error responses. Only a protected observation exposes page
text and its checked URL; omitted refs do not consume published ref authority.
The worker retains prior credential strings only within the configured plaintext
byte ceiling and one complete profile's category-count ceiling. If rotating site
state exhausts that anti-reflection history, later textual observations fail
closed with `resource_exhausted`; `close` remains available.

Schema v1 persists only bounded Secure cookies with exact host domains and
bounded `localStorage` entries for explicitly admitted HTTPS origins. It names
all omitted categories and rejects unknown fields or categories. It does not
persist session storage, profile directories, passwords, TOTP seeds,
extensions, cache, service-worker bodies, history, downloads, screenshots,
traces, open pages, refs, heap state, sockets, renderer state, or in-flight
requests. Origin, cookie, storage-entry, name, value, aggregate plaintext,
aggregate ciphertext, import-time, and export-time limits are independent.

The authority fingerprint binds application/tenant ownership, declared sharing,
the maximum origin policy, protocol/worker/state versions, key-authority and
store identities, creation, and optional expiry. The current policy may be a
strict subset of the recorded policy, but never a superset. Restoration rejects
the complete generation if any cookie or local-storage origin falls outside the
current policy; it does not silently discard security-significant state and
report success. `expected_ref` may pin an exact generation and encrypted-content
fingerprint when the application needs an exact restore.

Before import, the profile store acquires one bounded renewable writer lease for
the exact generation, execution profile, materialized allocation, and new
browser-session identity. Every writer admission, renewal, and profile-publication
mutation loads the record, samples the store clock, validates the lease, and
commits while holding one store write boundary. Direct failed settlement requires
that exact lease to remain live; exact release and revocation can still close or
fence authority without pretending an expired lease is live. A later writer
admission records an unsettled expired operation as `outcome_unknown` before
reusing the generation. Restart reconstruction atomically renews the exact live
writer before replaying or importing state into its allocation. A second writable
allocation is rejected. The worker creates its browser context from the validated
storage state before it creates a page or executes an untrusted document. The
subsequent explicit `navigate` creates fresh page, revision, and ref identities. A
restore receipt proves only that this state was imported; it does not prove the
website still considers the session authenticated. An expired site cookie can
therefore restore correctly while the first navigation honestly observes a
signed-out page.

The current profile-origin policy also travels through the private pinned-worker
protocol and governs every HTTPS document and subresource request plus every
secure WebSocket connection. Plain HTTP and WebSocket destinations remain
denied. This remains necessary when the selected virtual-egress environment has
a broader application allow-list: authenticated page state never widens that
environment into profile authority.

Checkpointing is runtime policy: `DISABLED`, `AFTER_TERMINAL_OPERATION`, or
`ON_CLOSE`. The model has no checkpoint operation. Cayu reserves the store's
configured ciphertext capacity before asking the browser to export plaintext,
validates the complete exported state, encrypts with a fresh AES-256-GCM nonce,
and authenticates profile, scope, destination, schema, generation, store/key,
and length metadata as AAD. It stages one complete envelope and receipt, then
publishes by compare-and-swap against the generation held by that writer. The
old generation remains authoritative until publication commits. Exact readback
adopts a committed replacement after acknowledgement loss without recapturing
state. Each receipt binds the latest settled browser revision and source
operation identity; an ambiguous earlier effect remains marked in its lineage.
With `ON_CLOSE`, a definite checkpoint failure prevents the close dispatch and
retains the live writer; a new close operation identity may retry checkpointing
before the allocation is retired.

Durable restore/checkpoint settlement resists caller cancellation long enough to
reach or reconcile its exact terminal record. If a Cayu process disappears after
reserving a restore but before its receipt commits, exact durable allocation and
operation authority may replay the same private import against the still-live
pinned worker; the worker's operation ledger returns the original result rather
than importing twice. If that allocation cannot be reconstructed, the finite
writer lease prevents immediate concurrent reuse. A later authorized restore
fences the expired writer and records any unsettled import or staged checkpoint
as `outcome_unknown`; an unpublished staged envelope never replaces the prior
generation. Successful historical checkpoints retain explicit publication
evidence even after a newer generation becomes current. Revocation and profile
expiry reject new restores, renewals, and checkpoints once observed, but do not
claim to erase credentials already loaded in a live remote allocation. The
application must separately close or quarantine that allocation.

`inspect_profile()` and scope-filtered `list_profiles()` return only opaque
identity and fingerprints, compatibility identity, generation, category counts,
byte totals, timestamps, active-writer state, receipt IDs, status, and fixed safe
error codes. They never return names, values, ciphertext, nonces, or key
material. Keep the `BrowserProfileAccess` capability and its owner/sharing scope
inside authenticated application or operator control code; browser tools do not
expose listing or inspection to model sessions.

Access-block classification and explicit fallback routing belong to a separate
browser-access contract; this recovery boundary does not infer or select a
fallback. A blocked main-document response publishes only typed transport
evidence with an origin-only URL and an empty title/snapshot/ref set. Denial-page
text never executes or becomes fallback evidence: the guest intercepts the
classified response before body execution, while broker egress denial remains
authoritative. See
[WebBridge access routing](web-fetch.md#classified-access-barriers-and-explicit-routing).
