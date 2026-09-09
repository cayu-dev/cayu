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
`cayu-browser-fetch:13-playwright-1.62.0` image, the
`cayu.browser-session.v4` protocol and worker version 9, brokered deny-by-default egress,
confirmed cancellation and cleanup, and one stable ArtifactStore. Construction
is side-effect-free for factories; the same candidate, workload, and artifact
authorities are checked again after materialization. There is no fallback to
host Playwright, host HTTP, another provider, a CLI, or MCP.

## Private operator view and login handoff

Browser operator control is an explicit application and server opt-in. It uses
the admitted browser allocation; it does not launch a second browser or expose
CDP, arbitrary JavaScript, selectors, shell commands, clipboard access, or a
general remote desktop. Existing destination, popup, egress, resource and
permission restrictions still apply.

Configure `CayuApp(browser_control=BrowserControlConfig(...))` with an
application-owned `BrowserControlPolicy` and a browser-reachable `wss://` guest
endpoint. Configure the server separately with `BrowserControlServerConfig`;
see [protected transport setup](server-configuration.md#browser-operator-transport).
Enabling only authentication, session inspection, or model screenshot permission
does not enable operator access.

The policy implements a stable, versioned, non-secret `identity` and an async
`decide(request)` returning `BrowserControlPolicyResult(allowed=...)`. Denial is
the default. Each request supplies the authenticated principal, exact browser
identity, operator session, action, record revision, control epoch and state.
The application must check access to that session and allocation for the requested
action; a matching `tenant` string alone is not proof of ownership. The action
set is `view`, `takeover`, `renew`, `handback`, `checkpoint`, `sensitive_entry`,
`text_input`, and `key_input`. Decisions do not receive page content or typed
credentials and are not reusable grants for later control generations.

### Operator workflow

In the session dashboard's private browser operator panel:

1. Discover the admitted browser and its current pages. Viewing is separately
   authorized and may be unavailable for a protected profile or sensitive page.
   Native private viewing currently supports the active page only; requesting a
   background or stale page is refused without capturing another page or closing
   shared browser control.
2. Choose checkpoint consent for the next takeover: undecided, deny, or allow
   configured profile checkpointing. This does not create a profile store.
3. Request exclusive takeover. A committed request blocks new model browser
   dispatch, but input is not granted until the native browser's existing work
   has settled. Refresh control state to observe the result.
4. Before entering login or MFA text, select **Prepare sensitive entry** while
   any private viewer remains connected. The transition waits for native capture
   settlement and acknowledgement that outstanding viewer frames were purged.
   Closing a viewer is not a substitute for that acknowledgement.
   The dashboard's **Close private view** and viewer replacement perform a bounded
   purge acknowledgement before closing; abrupt disconnection remains fail-closed.
5. Use the private text field and the limited native keys: Tab, Shift+Tab, Enter,
   Escape and Backspace. Refresh page discovery after an input changes authority.
   Do not send credentials as ordinary tool arguments, chat messages or metadata.
6. Return control to the agent. Handback settles pending input, advances control
   authority and invalidates old page targets. The agent must obtain a fresh
   protected observation before taking another browser action.

The operator panel displays the application's declared intervention purpose and
expected origins separately from page observations. These come from
`BrowserControlConfig.purpose`, not page content or operator input, and are bound
to the browser-control allocation identity. They do not override egress policy.

Page discovery displays the selected browser session, environment and allocation
fingerprint, plus origin-only observations for the discovered pages. Paths, query
strings, fragments and page titles are not part of these location observations.
An origin that cannot be safely represented is shown as unavailable or withheld.
These are observations, not application instructions, proof of successful login,
or permission to deliver input. While the selected panel is visible and idle,
origin observations refresh every two seconds for up to five minutes. A failed
read or expired refresh lifetime clears the observations and asks for explicit
discovery. Automatic observation does not refresh action targets: refresh page
discovery after navigation before requesting input or takeover authority.

Acquisition and handback control publications include bounded origin-only page
snapshots from their native boundaries, tied to the request and control epoch.
The guest omits origins containing known browser-private values; the host also
applies workload-secret protection, including the sealed browser-tool invocation
scopes retained by that allocation's bootstrap owner, before durable publication. Receipt replay
retains the original boundary snapshot rather than substituting a later page
location. These snapshots do not contain page titles, paths or entered values,
and do not establish that login succeeded.

Takeover has a finite lease and request maximum. Renewal extends only the current
live lease within that maximum; it does not start an unlimited new grant. A lost
input acknowledgement must not be retried as a new input: the original input may
already have reached the page. Disconnect, expired authority, worker loss and
uncertain cleanup do not automatically restore agent control. Refreshing the UI
does not clear a durable `control_uncertain` state or prove browser quiescence.

Disconnect cleanup can fence an exact takeover, renewal, sensitive-entry or
handback request committed while the guest owner was idle, or the exact fresh-
observation publication it had not yet observed. This only settles teardown;
it does not grant input, renew authority or treat transport loss as native
quiescence. Unrelated control generations are never adopted during cleanup.

### Privacy and profile consent

Operator frames and input use private, bounded, transient WebSocket channels,
not model attachments, artifacts, ordinary runner output or tool-result payloads.
This is separate from `BrowserVisualPolicy`: permission to view privately does
not authorize publishing screenshots to the model. Authenticated-profile and
sensitive-entry capture restrictions continue to apply, including after login
inside an initially temporary profile. Sensitive entry prevents prohibited
capture rather than attempting to redact pixels afterward.

Allowing checkpoint consent permits only the application's configured encrypted
profile checkpoint policy after handback and a fresh observation. Deny or
undecided does not authorize saving newly entered state. Consent does not enable
disabled checkpointing, change its destination, or prove that login succeeded.
See the [profile configuration below](#application-owned-browser-profiles) for the
separate application-owned profile setup.

### Local login/MFA acceptance

The opt-in designated-account test uses a disposable HTTPS password/TOTP service,
native Chromium, the protected Cayu server and the compiled operator dashboard.
It needs no external account or model API key. With the browser test dependencies,
Chromium and dashboard build already available, run from the repository root:

```bash
CAYU_BROWSER_CONTROL_LIVE=1 \
CAYU_BROWSER_CONTROL_CHROMIUM=/absolute/path/to/chromium \
CAYU_BROWSER_DASHBOARD_BUILD="$PWD/dashboard/dist" \
python -m pytest tests/server/test_browser_operator_mfa_live.py -q
```

Both Memory and SQLite cases exercise private password entry, rejected and
accepted TOTP entry, handback, and authenticated fresh observation. They check
generated credentials against model requests, durable events/checkpoints,
SQLite files, captured logs/warnings/stdout/stderr and artifact output. The model
and allocation adapter are test doubles; remote egress, profile-checkpoint
consent and worker-loss proofs remain separate tests. This journey does not save
an authenticated browser profile.

## Revision-bound visual controls

Prefer the accessibility snapshot and semantic refs for ordinary controls. Use
visual observation only when a canvas, image, or custom surface lacks usable
accessibility semantics. Pixels and labels are untrusted website content, not
instructions or authority to change policy.

Enable pixel publication explicitly at application setup:

```python
from cayu import BrowserVisualPolicy, WebBridge

browser = WebBridge.sandboxed_browser(
    environment=browser_environment_factory,
    interactive=True,
    interactive_options={
        "visual_policy": BrowserVisualPolicy(
            artifact_store_id="browser-artifacts",
            allowed_origins=("https://app.example",),
            retention="application_managed",
            publish_to_model=True,
            allow_coordinate_fallback=False,
            max_captures=8,
        ),
    },
)
```

The named store must be the admitted environment's artifact store. Images are
session-scoped; the application owns their deletion. `publish_to_model=True`
authorizes image attachments to the models selected by the application's
execution profile, through normal attachment admission and request accounting.
Images consume provider capacity and may increase cost. Capture has independent
dimension, pixel, byte, target, label, hit-test, frame-depth, processing-time and
capture-count bounds. Pixel permission does not bypass origin, resolved-secret
or output-secret restrictions. Cayu does not use OCR, blurring or string redaction
to make otherwise forbidden pixels safe.

The admitted profile context is currently `fresh_temporary`: the worker must own
its fresh temporary profile. Restored authenticated profiles and sensitive-entry
takeover are not supported capture contexts; their policy flags accept only
`False`. Origin permission is still explicit consent to the page's pixels, not a
claim that a temporary profile makes those pixels public or non-sensitive.

`observe_visual` returns a PNG artifact and a bounded `visual` bundle from the
same frozen observation window as the ARIA snapshot. It binds page revision,
control epoch, visual revision, image digest, viewport, device scale, scroll,
worker instance and opaque `vt_...` target refs. DOM nodes, browser targets and
CDP identifiers remain guest-private.

Before writing a visual artifact, runtime invocation secret discovery is sealed;
an incomplete or nonempty secret scope refuses publication. Artifact write/readback
and exact attachment replay also check the current secret scope. A dynamic direct
context without the runtime publication seal cannot publish visual artifacts.

Observation guard restoration has a separate one-second settlement allowance.
If restoration fails or remains pending, the allocation is fenced and its exact
restoration task is retained through bounded browser closure (up to five seconds).
Only successful cleanup reports retirement; otherwise allocation ownership remains
uncertain. The visual processing deadline does not cancel restoration as proof
that browser work has stopped.

`click_visual_target` requires the exact session, page, expected revision,
expected control epoch, visual revision, visual ref and stable operation ID.
The guest rechecks the retained node, document, viewport, geometry, actionability
and hit-test correspondence at native input delivery. It never searches by label,
selector, accessibility ref or nearby point. Observe again after any action,
page switch, navigation or expired evidence. Exact operation replay returns the
original result; it never repeats a click to resolve an uncertain outcome.

If separately enabled, `click_visual_point` takes normalized `x` and `y` in
`[0, 1)`, rounded to six decimal places, plus the exact screenshot SHA-256 and
visual revision. It accepts no desktop or host coordinates. Its result says
`coordinate_directed`; neither point delivery nor opaque-target delivery proves
semantic task success. Verify that separately from a subsequent observation or
application-owned oracle. Embedded-frame captures and unprovable hit surfaces
are refused rather than presenting child-renderer pixels as covered by the main
renderer freeze. This does not add typing, dragging, arbitrary scripts,
selectors, CAPTCHA solving or a general computer-use interface.

Point admission currently requires an exact capture-time hit-test sample (the
retained target's sampled center after conversion to viewport pixels). It never
snaps an unsampled point to that center, and a fresh hit test cannot create new
capture authority. Unsampled points return `unsupported_visual_surface`.
The native guard rejects DOM changes before the first admitted input event;
changes caused by that admitted event are not treated as pre-input staleness.

Visual delivery supports provable HTML canvas, image, button, link and permitted
input surfaces, plus SVG surfaces other than shadow-backed `use` instances.
Shadow-capable receiving hosts (including
bare `div` and custom-element hosts) report `opaque_surface` and are not actionable:
the host could conceal a closed-shadow password or file control. Open-shadow
content is usable only when the actual receiving element is independently
supported. This restriction does not change the accessibility-first path.
Visual retirement is carried through the outer popup owner: no popup JavaScript
is evaluated on a retired page, and cancellation or timeout remains authoritative.

For a reproducible opt-in real-model demonstration, see
[`examples/browser_fetch/visual_live.py`](../examples/browser_fetch/visual_live.py)
and its [execution instructions](../examples/browser_fetch/README.md#live-visual-acceptance).

The deterministic browser corpus includes semantic-first, canvas, image-only,
custom-widget, hostile-label, popup, stale-evidence, embedded-frame, movement,
overlay, secret-refusal and viewport-scroll cases. Movement cases release a fixture response
only after visual capture, then attempt the original target. The opt-in local
Chromium regressions additionally check exact replay, independent authority
conflicts, and real process/acknowledgement loss without repeating a click:

```bash
CAYU_RUN_VISUAL_BROWSER_ACCEPTANCE=1 pytest tests/egress/test_browser_visual_docker_e2e.py
```

These tests require the pinned image and Docker egress setup. They use a
deterministic model and local fixture, not a paid provider. The fixture's effect
counter is the semantic oracle; successful input delivery alone is not a pass.
Process-loss cases use a separate quarantine safety oracle when the dynamic
invocation secret scope cannot be reconstructed. It verifies that the exact
browser call is durably retained and no longer resumable, and reports an
ambiguous outcome. It does not fabricate the private call into the transcript or
claim successful task completion. Acknowledgement-loss cases independently
verify the committed result without another click.

## Model contract

The provider-facing schema is a closed object without top-level conditional
combinators, so OpenAI can accept the tool definition. Operation-specific required
and forbidden fields remain enforced by Cayu before durable browser admission or
browser dispatch; the tool description explains those requirements.

One ordinary `browser_session` tool exposes only `navigate`, `observe`,
`click`, `fill`, `select`, `press`, bounded `wait`, `back`, `forward`, safe `reload`, semantic `scroll`,
strict `hover`, artifact-backed `upload`, `screenshot`, `download`,
`list_pages`, `switch_page`, `close_page`, and `close`. Visual operations
(`observe_visual`, `click_visual_target`, and `click_visual_point`) require a
separate application policy; they are refused by default. The first navigation
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

`back` and `forward` traverse Chromium's existing history rather than
synthesizing a URL navigation. `reload` is admitted only when the current main
document is positively known to have used `GET` or `HEAD`, so Cayu does not
silently resubmit a form. Method evidence is bound to Chromium's committed
main-frame loader, not merely an observed response: a noncommitting response
such as HTTP 204 cannot relabel the retained document. Reload dispatch is bound
to that validated loader; Chromium refuses it if another document commits first.
Dialogs are dismissed but make history and reload fail
closed, and navigation is not retried after an ambiguous acknowledgement. Every
main document and subresource continues through the same deny-by-default broker
policy.

`scroll` accepts only `up`, `down`, `left`, or `right`, a `line` or `page`
amount, and an application-bounded repeat count. Its result reports only whether
the selected axis moved and whether the requested edge was reached; it does not
claim that the whole document was observed. `hover` resolves only the exact
current revision-bound ARIA ref and has no selector, coordinate, script, or
fallback entrance.

`upload` accepts only a current file-input ref and a bounded list of opaque
ArtifactStore IDs. Before browser dispatch, Cayu requires exact session scope,
compatible optional agent/environment ownership, an allowed media type and
taint set, bounded file and aggregate sizes, one safe basename, a stable
secret-redaction revision, and secret-free bytes. The durable operation identity
includes artifact IDs and content fingerprints. Raw bytes and private guest
paths are never durable or model-visible. The guest revalidates content and the
file-input/multiple contract, then transfers bounded in-memory file payloads to
Chromium with the admitted filename and MIME type. No temporary upload files
are created or removed on the guest event loop. Selected contents remain available
for later page reads and submission until the browser releases them; outer
allocation teardown remains the process-loss cleanup boundary. Success proves
browser file selection only,
not remote acceptance or form submission.

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
Upload file count, per-file and aggregate bytes, filename bytes, materialization
time, media types, taint labels, and scroll repeats are also application-owned
and execution-profile-bound. DOM-node and
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
`navigation_timeout`, `history_unavailable`, `unsafe_reload`, `invalid_scroll`,
`incompatible_upload_target`, `artifact_refused`, `artifact_unavailable`,
`upload_too_large`, `upload_materialization_failed`, `upload_cleanup_failed`,
`download_failed`, `browser_crash`, `cleanup_failed`, or `outcome_ambiguous`.

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

Docker virtual-egress continuity is opt-in through
`DockerEgressAdapter(reconnect_state_dir=...)`, using the pinned worker v11 and the
same application execution-profile identity. See
[Docker retained-allocation reconnect](virtual-egress.md#docker-retained-allocation-reconnect)
for local-host ownership, fencing, CA rotation and the real process-loss tests.
Fresh CA trust invalidates previous page refs/control epochs; obtain a fresh
observation before subsequent actions. This does not add profile restoration or
mutation replay.

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
screenshots, downloads, or Chromium identifiers. Upload operation records retain only bounded artifact identity,
basename/media metadata, size, and content fingerprint. A sealed terminal operation
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

### Close response settlement

Worker 12 keeps the admitted explicit-close response handler alive through bounded
response drain and socket closure before allowing daemon shutdown. Queued requests
and receipt replays cannot release that handler's shutdown signal. Native cleanup
failures remain failures, and missing acknowledgements remain ambiguous; a guest
WebSocket disconnect alone never proves that a model action succeeded. Deploy the
pinned worker-13 image with this Runtime revision. Existing worker-12 allocation
authority does not become compatible merely by retagging an image.

## Optional video recording

Applications can opt into [Docker browser recording](browser-recording.md) for
private post-run playback. Recording is separate from live viewing, visual model
attachments and browser-profile checkpoint consent.
