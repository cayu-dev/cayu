# Diagnostic support bundles

Use `cayu doctor` when an operator or maintainer needs a portable snapshot of a
Cayu project without running an agent or making a live provider request:

```bash
cayu doctor --bundle cayu-support.zip
```

The command discovers the same project and maintained-service factories as
`cayu check`, selects the production bootstrap profile, and constructs the
application once in a bounded worker process. During that bootstrap, a
process-scoped diagnostic owner forces file-backed CayuApp and project
control-plane stores backed by Cayu's built-in SQLite and PostgreSQL
implementations to validate and read through database-enforced read-only
connections. Static SQLite files use immutable inspection and are revalidated
before publication; an active WAL uses SQLite's locking-aware read path so
committed frames remain visible, while an incomplete WAL/SHM pair fails closed.
A missing configured SQLite session/task source or project eval source is
represented by an empty query-only shadow inside the disposable worker so a
fresh scaffold can still be inspected. The configured stores remain classified
as durable, while schema readiness and operational state report `unavailable`;
an appearance of that source during collection fails closed. A SQLite `:memory:`
store likewise initializes only an empty schema in the disposable worker, remains
classified as development storage, and is only read by collectors. No database
file is created and no external schema is migrated. A PostgreSQL store supplied
with a caller-owned
pool fails closed because the diagnostic owner cannot establish ownership of the
pool's connection lifecycle. This boundary does not sandbox arbitrary project
factory code or custom store implementations, which remain responsible for
side-effect-free construction. An incompatible existing built-in store produces
a minimal `boot_failed` bundle. The command does not start a session, run a
model or tool, probe a provider, invoke recovery, repair data, or upload the
result.

Diagnostic read-only access is distinct from configured durability. Store
descriptors and the embedded maintained-service checks retain the deployment
durability declared by the project while database access remains read-only.

The ZIP contains exactly two files:

- `report.json`, a typed schema-version-1 report with command version,
  fresh bundle identity, collection duration, result counts, and evidence bytes;
  and
- `summary.txt`, a human-readable rendering derived from that report.

The archive is staged beside the destination, validated before publication,
written with mode `0600` on POSIX, fsynced, and atomically replaced. On POSIX,
the existing parent is resolved so stable platform aliases such as macOS `/tmp`
remain usable, then every canonical parent component is opened directory-relative
with no symlink following; the requested path is resolved again and its pinned
parent identity is checked before success. The staging descriptor remains open
through replacement, and the reopened destination must match its file identity
and exact validated archive bytes. Windows publication
rejects junction/reparse traversal, holds a native namespace fence on the
validated parent, creates the staging file with a protected DACL for the owner,
system, and administrators, and replaces with write-through semantics. The
published Windows file is rechecked for regular-file identity, size, and a
protected DACL before success. The canonical report, empty ZIP metadata, regular
member types, and member permissions are also checked. An existing symlink or
non-regular destination is rejected. A write failure does not expose a partial
staging file; failure while confirming directory durability or parent identity
can still leave a complete validated replacement at an earlier pinned path.

## Collected evidence

Built-in collectors report content-free operational evidence:

- Cayu, Python, operating-system, and machine identity;
- redacted project/release identity and the application manifest fingerprint;
- the complete structured `cayu check` report;
- bounded agent/provider resolution, environment component and MCP server
  counts, MCP policy presence, capability summaries, and typed workspace branch
  capabilities and attached lifecycle status;
- an actor-free portable projection of the existing protected system-diagnostics
  response when a maintained service is selected, including access posture and
  capability readiness but no request actor or artifact-store identity;
- the configured recovery-cleanup deadlines and capacity together with bounded,
  content-free process-local supervision counters and retained-operation
  identities;
- configured store implementations, declared durability, bounded-event-read
  support, safe eval-store backend/source category, and per-store schema
  readiness;
- exact store-native session/task status aggregates when supported;
- artifact-store availability and registration count, without identities,
  fingerprints, or listing or reading artifacts; and
- installed/not-installed status and versions for Cayu's optional dependency
  families.

Built-in file-backed SQLite stores report `validated_compatible` only after
their constructor-owned validation succeeds. Built-in PostgreSQL stores perform
their lazy schema validation through diagnostic read-only transactions before
reporting the same status. Process-private in-memory stores and wrappers without
their own schema report `not_applicable`; custom stores report `unavailable`.
Validation failure remains content-free. Doctor never creates or migrates a
schema.

Recovery-cleanup evidence comes from the current application's typed
`recovery_cleanup_status()` owner and its manifest policy. Retained-operation
details are limited by the bundle item bound while preserving the exact total
and truncation state.

Doctor's application context does not own a selected durable-worker metrics
cohort, an authoritative lease-health projection, persisted event-side-effect
health, general session handoff/recovery health, or environment health, and
Doctor does not run a live provider probe. Those domains are not registered as
requested collectors until an owner is available through that context;
otherwise their unconditional placeholders would make a clean result
unreachable. Existing manifest and control-plane evidence still reports the
authoritative configured capabilities without fabricating live health.

Every result records its monotonic duration and the exact serialized bytes of
accepted evidence. The report records collected and omitted result counts, total
accepted evidence bytes, and whether evidence is complete. Each collector ends
with one disposition:

| Disposition | Meaning |
| --- | --- |
| `collected` | Typed evidence passed size and redaction validation. |
| `unavailable` | The selected project/backend has no authoritative typed evidence. |
| `skipped` | A framework deadline prevented the collector from starting. |
| `timed_out` | The collector exceeded its real async deadline. |
| `failed` | Collection or typed reconstruction failed; raw exceptions are discarded. |
| `redacted` | Application redaction changed or could not safely validate the evidence. |

For a successfully booted collection, every non-`collected` disposition produces
a `partial` bundle while preserving every successful collector. Evidence-byte
exhaustion skips the collector that
would cross the aggregate bound and every remaining collector without running
them.

## Optional event tails

No session history is read by default. Request a session explicitly:

```bash
cayu doctor --bundle cayu-support.zip --session SESSION_ID
```

Repeat `--session` for several sessions, up to the command limit. Each selector
must be clean Unicode text of at most 2,048 UTF-8 bytes, is used only inside the
worker, and is never written to the archive. Public session aliases are resolved
through the application's store-owned authority boundary. The collector uses
`SessionStore.query_events_bounded`; a custom store without that operation
reports `unavailable`.

Event evidence carries an explicit `redacted_envelope_only` projection and
contains only durable sequence, a built-in event type, and timestamp. Payloads,
event IDs, session IDs, interaction IDs, agent/environment names, transcript or
model text, tool arguments/results, and arbitrary metadata are omitted.
Non-built-in event types collapse to `custom.redacted`. The report includes
first/last sequence and timestamp bounds, an explicit completeness flag, the
returned count, and either an exact zero omitted count or a lower bound when
more events exist; it never performs an unbounded count scan.

## Exclusions and redaction

The bundle does not contain:

- prompts, messages, transcripts, model text, or reasoning;
- tool arguments, tool results, artifact bytes, or artifact metadata;
- raw exceptions, tracebacks, project stdout/stderr, logs, or warnings;
- credentials, tokens, DSNs, environment maps, or live provider responses;
- raw private session/event/interaction identifiers; or
- absolute filesystem paths.

Artifact-store identities are also omitted rather than hashed. Built-in default
identities can contain resolved local paths or S3 bucket/prefix names, and a
stable unkeyed digest would let a bundle recipient test guessed values offline.

Collectors build allow-listed typed projections rather than dumping manifests,
objects, or events. The application redactor must leave each projection
unchanged, and the projection must independently pass the archive's forbidden
content rules; otherwise only that collector is discarded as `redacted`. The
parent process then repeats validation over canonical typed report
reconstruction, the exact ZIP member set and metadata, member and archive byte
limits, summary correspondence, forbidden keys, URIs/DSNs, and absolute paths
before publishing the output.

A support bundle is deliberately narrow, but operators should still handle it
as operational data and inspect it before sharing.

## Bounds

Schema version 1 uses fixed framework limits:

| Boundary | Limit |
| --- | ---: |
| Complete command, including owned teardown | 40 seconds |
| Project boot and complete worker | 20 seconds |
| Atomic archive publication | 10 seconds |
| Failed-publication staging reconciliation | 5 seconds |
| All collectors inside a booted worker | 15 seconds |
| One collector | 2 seconds |
| Items per manifest inventory | 100 |
| Explicit sessions | 10 |
| One explicit session selector | 2,048 UTF-8 bytes |
| Events returned per session | 50 |
| One store-native event read | 256 KiB |
| One serialized collector | 256 KiB |
| All accepted collector evidence | 1 MiB |
| Complete ZIP | 2 MiB |

A hanging project factory is terminated by the parent deadline and produces a
minimal `boot_failed` bundle. The worker also owns an independent lifetime guard,
which remains armed through interpreter shutdown, so parent death or a
project-created non-daemon thread cannot turn a bounded factory into an
indefinitely orphaned process. SIGTERM is translated into bounded owned cleanup
before the CLI exits. Receiving a worker payload does not transfer process
ownership: the parent accepts it only after the child settles normally within
the worker deadline and rejects it if forced teardown is required.
Read-only collectors are expected to cooperate with async cancellation; the
per-collector deadline remains authoritative through typed serialization,
application redaction, forbidden-content validation, and byte accounting.
Framework-owned synchronous runtime/package probes and evidence preparation run
in disposable daemon threads, so one blocked synchronous step is abandoned at
the collector deadline while later collectors retain their own time. A collector
that suppresses deadline cancellation but eventually returns is still reported
as `timed_out`. The process deadline remains the final containment boundary for
collector code that never cooperates with async cancellation. Archive publication
runs in a separately owned process under the remaining command deadline. A
stalled write is terminated, and its exact staging name is reconciled by a
second independently bounded process. Publication, both owned-process teardown
sequences, and reconciliation share the complete command's absolute deadline;
later phases can receive less than their individual maximum when earlier phases
consume that budget. If the filesystem also prevents cleanup,
that reconciler is terminated and the staging leaf can remain for manual
recovery; the command still reports `output_write_failed` and never reports
success without a settled publisher. Collection-deadline or aggregate evidence-byte exhaustion
produces a validated `partial` bundle whose skipped results make the omissions
explicit.

## Related diagnostic boundaries

The bundle embeds the exact structured `cayu check` result; it does not copy or
rename that command's diagnostic codes or severities. For a maintained service,
it also projects the exact existing
[`SystemDiagnosticsResponse` from #276](https://github.com/cayu-tech/cayu/issues/276)
using the route-owned capability snapshot and current bounded artifact
registrations; no request actor is invented. A direct-app project reports that
collector as `unavailable` with `maintained_service_not_selected`. Discovery
metadata is never treated as authorization or live readiness.

[AWS preflight #252](https://github.com/cayu-tech/cayu/issues/252) remains the
owner of AWS-specific configuration and permission probes.
[Live provider verification #470](https://github.com/cayu-tech/cayu/issues/470)
remains the owner of an explicitly requested bounded provider call. `cayu
doctor` neither runs those probes nor turns their absence into fabricated
health.

## Outcomes and automation

Use `--json` for the stable command result:

```bash
cayu doctor --bundle cayu-support.zip --json
```

| Exit | Outcome | Meaning |
| ---: | --- | --- |
| `0` | `clean` | Every requested collector produced typed evidence. |
| `1` | `partial` | Safe successes were retained alongside unavailable or otherwise non-collected evidence. |
| `2` | `boot_failed` | Discovery, source validation, project boot, or the worker deadline failed; a minimal safe bundle was written. |
| `3` | `validation_failed` | Request or final bundle validation failed; a minimal safe bundle was written when possible. |
| `4` | `output_write_failed` | The destination could not be safely published. |

Without `--json`, the command prints the same stable outcome token in one
human-readable line; output-write failure is written to stderr. Neither form
includes the destination path or an exception message. `report.json` carries
fixed reason codes for collector and bootstrap outcomes.
