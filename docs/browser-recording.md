# Docker browser recordings

Browser recording is **off by default**. An application can authorize a finite
recording scope for one exact Runtime session and selected HTTPS origins. Model
instructions, live-view permissions and browser-profile checkpoint consent do
not enable recording. Recording grants no input or takeover authority.

Use `BrowserRecordingPolicy`, `BrowserRecordingStore.authorize_capture`, and
`BrowserSessionTool(recording=...)`. Add a `BrowserRecordingServer` with a
separate application access callback to `create_server(browser_recordings=...)`.
The runnable [example](../examples/browser_recording/README.md) uses these public
contracts without a live viewer or external provider.

## Backend and capture semantics

Version 1 supports the admitted Docker browser worker **13**, a POSIX application
worker with `cayu[recording]`, and FFmpeg with the VP8 encoder. Probe the backend
contract with `browser_recording_capability("docker")`; other backend names report
`unsupported_backend`. There is no host-browser or dashboard-canvas fallback.
The recording store checks encoder/dependency availability before issuing consent.

The guest starts a bounded sampling task before its first page is created. It
samples the **active page viewport**, at up to the configured rate (default two
frames per second). Navigation changes the document within that browser; switching
pages changes the captured page, identified on every segment. Background pages,
popups that have not become the active admitted page, audio, and full-page capture
are excluded. Recording continues independently of viewer connections.

Each admitted sample becomes a separately validated WebM segment. Finalization
combines settled segments into one silent WebM video and decodes it to verify its
frame count. The video compresses capture gaps; the manifest retains original
elapsed times, page identities, ordered segment hashes, gap intervals, expiry,
finalization status, and the final video's SHA-256 and byte count. It is sampled
coverage, not evidence of every visual change between samples.

Capture is excluded while browser operations own the lifecycle lock, on denied
origins, in imported profiles, after sensitive takeover, and when cookies, local
storage or session storage are present. Child frames, shadow roots, password/file
controls, and credential/payment autocomplete controls are refused before a
screenshot. Version 1 also stops future capture before model text, key or file
entry (`fill`, `press`, `upload`), since authentication need not use cookies.
Secret-bearing runner environments cannot enable this capture contract.
Applications must allow only origins appropriate for their recording consent;
Runtime cannot infer the business sensitivity of arbitrary page text.

Admission pauses the debugger and animations, requires a fully loaded document,
and inspects the native document. It then rechecks sensitive DOM content, document
readiness and identity, navigation epoch and restrictions after the in-memory
screenshot. Script suspension alone does not stop HTML parsing. Refused candidates
never reach the encoder or a temporary file. Pending application callbacks resume
after capture; recording never disables script execution or discards queued work.
Failures to restore a frozen page retire the uncertain browser instead of letting
business operations run against unproved browser state.

## Lifetime, interruption and recovery

Normal Docker allocation disposal and explicit browser close request finalization.
Sensitive-entry pauses exclude capture. Other human/approval pauses retain the
recording within its duration bound. A retained Docker allocation keeps the guest
sampler and its sequence counter. Application-server reconnection claims a new
publication owner, fences stale writers, and preserves previously committed
segments. Disconnects and missed samples produce partial coverage.

A completed invocation's next allocation has a different allocation/browser
identity and a different recording ID, even within the same session. It cannot
overwrite the earlier recording. Cancellation, deadline/worker loss, browser
crashes, and encoder/storage failures never authorize replaying browser actions,
model calls or business mutations to fill missing footage.

Statuses are `recording`, `complete`, `partial`, `unavailable`, and `failed`.
`complete` requires a normal finalization receipt and no known coverage gap;
ordinary sampled runs may legitimately be `partial`. No footage yields
`unavailable`; a storage failure without footage yields `failed`. A lost final
acknowledgement cannot overwrite a settled result. SQLite transactions publish
media and metadata atomically. Recovery remuxes only retained, validated media.
If final-video publication exhausts storage, the recording settles as `partial`
with reason `storage_failure`. Its committed segments remain available through
the protected segment endpoints, even when no combined video is available.

The recording server periodically seals abandoned recordings after their maximum
duration and inactivity window. A disconnected recording can remain `recording`
until that bounded recovery point. Explicit `recover_abandoned()` uses the same
bounds; it does not infer that a disconnected, paused browser is finished.

## Access and retention

The capture credential is separate from operator login and is sent only through
Runtime's private runner transport. Keep it and the recording database outside
model-accessible workspaces. Never pass either through tool arguments. Recording
bytes and references are not model attachments, transcripts or ordinary artifacts.

The application access callback is invoked for every manifest/media request.
`view` and `download` are separate decisions; neither follows from live-view
permission. Retrieval serves only settled media, verifies integrity, uses
`Cache-Control: no-store`, and supports bounded byte ranges. Permission to view
necessarily delivers video bytes to that viewer; this is not a DRM boundary.
The dashboard session page offers playback, coverage gaps and download when the
server advertises the `browser_recordings` capability. Unconfigured deployments
do not poll recording endpoints.

`max_duration_seconds`, `max_bytes`, resolution and frame-rate bounds stop capture
without retrying business operations. The store's `max_storage_bytes` includes a
reserve for database journals and encoder staging; usable media capacity is lower
than that ceiling. Anonymous local `/tmp` staging is serialized across workers,
contains only admitted media, and disappears on process loss. The guest creates
no recording files. Disabled runs publish no video segments or outputs.

Retention is an absolute deadline from authorization and never extends on reconnect.
Expired reads are denied immediately; server maintenance deletes expired media on
its next 30-second pass. Outside the server, applications must schedule
`purge_expired()`. `delete(recording_id)` removes stored media and leaves a bounded
tombstone until expiry so a retained guest cannot resurrect that recording.
Applications own storage placement, access decisions, backup deletion, and any
retention obligations for copies they download.
