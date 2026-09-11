# Repository-maintenance qualification contract

This extends the existing application-convention benchmark for the repository-maintenance qualification. It is a
fixed acceptance contract, **not a passing production qualification report**.
The generated application, independent adoption, and live delivery still need
their own retained evidence. Do not count the corpus tests below as those gates.

## Baseline and fixture ownership

The inspected Runtime baseline is
`a815c7b2348313f79de911c69b47bb93efafc549`. Qualification must record the final
candidate commit, wheel digest, installed build identity, application revision,
toolchain/image identity, configuration digest, and corpus fingerprint. An
uncommitted source checkout is not a pinned installed-build trial.

The application starts from the ordinary `coding` preset with Docker execution,
the Python toolchain, PostgreSQL stores, and the `github-delivery` capability
(including its remote-Git prerequisite). Declaring these host integrations does
not configure credentials or authorize publication. Do not combine it with the incompatible
`service + postgres` preset. Application-specific API/worker deployment must retain
the canonical generated module ownership and public Cayu imports.

The fixed repository case is owned by
`tests/qualification/repository_maintenance_case.py`. The existing qualification
runner stages and fingerprints this Python fixture with the rest of `tests/`.
`materialize_seed_repository()` creates a new disposable repository and refuses
to reuse an existing directory. Git metadata is fixed, and no remote is configured.

| Identity | Pinned value |
| --- | --- |
| Case | `closed-integer-range-v1` |
| Corpus SHA-256 | `95933d96773bcd2cd28f080cf8e8d97b5b42278ed400ba72535ea58342e0a242` |
| Base commit | `219ebbfb37f09b0a08e9ca98ad4474bf4640c06f` |
| Base tree | `325cea918a90d6596e35a5bd95a75d962ef31cc5` |

The corpus fingerprint binds exact seeded file bytes, permitted change paths,
ordered probe inputs, and expected responses. Changing any of these starts a new
reviewed corpus; comparisons cannot silently switch corpora between versions.

## Bounded bug and independent oracle

`range_ops.in_closed_range(value, lower, upper)` must include both endpoints.
The seeded implementation excludes the upper endpoint. The project contains a
reproducible failing endpoint test and a passing outside-range test.

The declared domain contains 105 probes: integer values from -3 through 3 and
all bounds satisfying `-2 <= lower <= upper <= 2`. It includes degenerate ranges,
both endpoints, interior values, and values outside each range. The independent
oracle requires one actual Boolean response for every probe, in order. Missing,
extra, wrong-type, or incorrect responses fail. Integer `1` is not Boolean `true`.

Permitted edits are `range_ops.py` and `tests/test_range_ops.py` only. Each changed
file is limited to 16 KiB. The project configuration, independent oracle, probe
inputs, and acceptance rules are outside agent mutation authority. Project tests
may improve, but weakening or deleting their assertions cannot change the oracle.

The generated application policy checks mutation paths before tool dispatch.
`write_file` and `edit_file` require an explicit integer `max_bytes` from 1 through
16384; the file tools enforce the resulting size. Patch calls allow at most two
operations, check every path (both endpoints for moves), and use the existing
patch tool's 16-KiB file bound. Noncanonical path spellings and out-of-scope paths
are rejected rather than normalized into authority. A task cannot replace the
fixed required-check set with a weaker one; rejection precedes admission.

The application must execute candidate code only through admitted no-network
Docker execution. The trusted verifier must compare the returned bounded data
outside the candidate interpreter. Importing candidate code into the verifier
would allow it to terminate or modify the verifier. A zero command exit code,
agent-written success marker, or green project test is not an oracle result.
The fixed probe declaration and parent-side decoder live in
`tests/qualification/repository_maintenance_probe.py`. The image-owned program
fingerprint is
`sha256:ef64277091932bb39fee728aaa04e4c0def4940e36449e4f81a04eae9458db94`.
Its `independent-range-probe` named check runs the admitted Python executable with
`-I -B`, the fixed probe program, and `/workspace/range_ops.py`; the expected
answers remain outside the candidate interpreter. Install the program at
`/opt/cayu-acceptance/range_probe.py` in the immutable image and include that path
in its read-only support authority before running the application.

Use the declared 256-byte model-preview bound for `RunCheckTool`. Every complete
valid response exceeds that preview, so the normal named-check owner retains the
full output artifact. The verifier must read the complete digest-bound artifact,
not parse the truncated model preview. Check identity, profile, artifact digest,
output completeness, and the final workspace revision remain mandatory. The
fixture tests prove decoding and named-check output transport, not those complete
production boundaries. The generated fixture includes the image installation
declaration. Local public-workflow integration exercises retained output and the
verifier together; actual image build and live Docker admission remain unverified.

Behavioral responses alone do not authorize publication. The complete result must
also bind the admitted repository/base, final source revision, permitted path
delta, complete required checks, exact source-publication evidence, and oracle
identity. Revalidate the same final revision before delivery. Any intervening
mutation invalidates the result rather than moving verification to a newer tree.

The generated consumer adds `CodingProductApplication.verify(task, publication)`
as a separate acceptance step. Its application-owned domain implementation reads
the sealed result through `CodingProductArtifactRepository`, reuses the existing
coding-product completion decision, checks exact declared check profiles, and
compares the retained probe responses. Initial file hashes must match the fixed
seed; only permitted paths may change. Bounded current-source reads must match
the retained final manifest before and after verification. The returned record
binds the result, request, final revision, corpus and probe-output digest; it is
not human approval, a lock, or authorization for a later Git/GitHub write.

The qualification tests cover both the domain boundary and the generated public
`run → verify → replay` path with real local artifacts, Git, source copy-back,
and Runtime tool execution. The integration uses a test-owned Docker-shaped local
factory and controlled check outputs; it does not establish live Docker isolation,
model task quality, or delivery safety. Correct probe responses are accepted and
incorrect responses rejected even when the probe command exits successfully.
The persistent variant uses the generated SQLite store configuration, closes and
reopens the stores, and rebuilds the application before replay and verification.
It checks durable reconstruction without provider redispatch; it is not evidence
for the declared PostgreSQL production deployment or abrupt process loss.

`workflows/maintenance_delivery.py` owns the verified Git handoff:
`prepare_verified_git_delivery(...)` and `run_verified_git_delivery(...)` each
revalidate independent acceptance before calling the existing broker. Their inputs
must come from trusted application lookup, not raw product-route data. They do not
create approval. The broker owns exact-tree preparation, approval validation,
commit/push, reconciliation and cleanup; its result states remain distinct from
independent acceptance and later GitHub PR/check/review states. A local bare-Git
fixture tests this handoff without granting authority to publish to GitHub.
At `approval_required`, the broker intentionally retains its prepared private
repository; `cleanup_settled` is false because terminal removal has not occurred.
The local fixture reconstructs a broker after normal return and reads back the
exact prepared intent and approval-required result before granting approval.
This does not authenticate a human or establish process-restart acceptance.
The accepted-push/lost-acknowledgement variant observes the real local destination
before injecting the failure, then requires reconciliation and exact replay with
one push dispatch and one repair commit. This is application-path local Git proof,
not live GitHub acceptance, process-loss recovery, or PostgreSQL qualification.

`run_verified_github_delivery(...)` likewise revalidates independent acceptance
before one call to the existing GitHub connector. It does not grant approval or
own a replacement scheduler. Its caller must retain connector lifetime across
timeout/cancellation and observe quiescence before releasing worker ownership.
The local generated-consumer fixture uses the real REST adapter with controlled
HTTP responses, real pushed Git refs, and separate fixture approval. This is not
authorization to contact GitHub or evidence of a live production deployment.

The maintenance consumer specializes `integrations/remote_git.py` to accept
optional native `RemoteGitHttpCredentials` and an explicit `egress_profile_id`.
Supply `SecretRef` values and a supported resolver in the delivery process only;
the factory does not resolve them. Native remote configuration rejects credentials
on non-HTTPS transports and rejects URLs containing credentials. Without credentials
the local fixture retains the `none` credential profile. Keep these objects and
the secret environment out of coding workers and containers. An egress profile
name records authority; it does not enforce a network policy. The deployment must
enforce that boundary, and authenticated live transport remains unqualified here.
Its lost-response variant accepts one PR creation, loses the response, and makes
the first reconciliation read unavailable. The connector must retain ambiguity.
Only after local quiescence does the fixture reconstruct a connector, recover the
durable approval, and reconcile the same PR without another create. This is
durable local-artifact reconstruction, not abrupt process-loss qualification.

## Trial envelope

The emitted `operations/maintenance_requests.py` captures a private typed request
before reservation. It binds the original instruction, configured repository
root (the workspace ID alone is not a path identity), source/destination and
artifact-store IDs, execution/toolchain profiles, seed base, corpus/probe, and
default settlement policy. The maintenance workflow compares that stored request
with current read-only inspection before product dispatch. It rejects custom
settlement/review overrides; delivery approval remains a separate host operation.
This record is neither authentication nor proof that the actual source passed
admission. Keep it out of public response projections. The local worker, Evals,
and deadline fixtures use this reservation-backed boundary. The emitted
`operations/maintenance_worker.py` handler resolves the coding claim through the
reservation's trusted task index, checks its immutable request and current worker
identity, and reconstructs the task and original deadline from saved data. It
never recreates a missing claimed task. Runtime retains the lease, heartbeat,
cancellation, and terminal-mutation authority; readback is not a new lease.
Only independently verified publication reaches `complete_managed_task`.
Ordinary rejection and pre-dispatch expiry use Runtime's failure terminalization.
The emitted `operations/maintenance_results.py` reconstructs downstream coding
inputs from the same reservation and the exact completed task's product/digest
result. It compares current accepted configuration, reloads the content-addressed
publication, and repeats independent verification. Missing, incomplete, failed,
conflicting, or malformed completion cannot trigger coding or manufacture a
publication. This host-only readback is not caller authentication or delivery
approval; the Git/GitHub boundaries still require their own fresh checks and
exact approval. The managed-worker fixtures now use this reconstructed result
for their subsequent delivery journey. Production delivery role/approval wiring
and native deployment qualification remain required.

`operations/maintenance_github.py` observes the exact approved GitHub request
within one enclosing worker lifetime. It follows only the connector's advertised
next-poll delay and uses the original request time plus elapsed limit as a portable
deadline. Each observation repeats the existing verified handoff. Ending polling
does not imply success: the returned native state still governs the result.
Cancellation seals future effects and retains the connector until positive local
closure; it does not cancel opaque in-flight work. The caller must keep its task
lease, transport, and stores alive for this entire await. An unresolved close stays
owned; the process supervisor's hard stop is failed cleanup, not proof of abort.
The generated SQLite/local-Git/controlled-HTTP journey exercises pending-to-current
checks and fresh-connector replay after closure. Production queue ownership and
real remote/process-loss cleanup remain separate required qualification.
The emitted `operations/maintenance_http.py` constructs an application-owned host
with bounded `POST /runs` intake and tenant-qualified `GET /runs/{public_id}`.
`integrations/maintenance_auth.py` requires distinct configured product/operator
bearer credentials and has no open or development-header fallback. The operator
credential protects the Cayu mount at `/internal/cayu`; product credentials cannot
use it. Product responses contain only `id` and `coding_task_status`, not raw
Runtime records or a claim of approved PR delivery. A missing queued task is
`intake_pending`, and reads never create it. Intake failures may follow a commit;
retry the same idempotency key rather than inventing a replacement operation.
Host construction and new intake require one priced, reserved, all-time app
budget in USD with a cap no greater than USD 1. The emitted domain guard reuses
Runtime validation; it does not create accounting state or certify provider token
bounds, live prices, or shared ledger deployment. Removing the policy stops new
intake without disabling read-only lookup of previously accepted work.
The in-process HTTP tests do not qualify listener/TLS configuration, startup and
shutdown of production resources, or a deployed shared monetary cap. The host
accepts an application-owned `lifespan` that surrounds mounted Runtime startup and
drains. A completed mount shutdown is not positive quiescence evidence: the
dependency owner must check for unresolved work before closing its stores.
Complete deployment and remote-outcome recovery/cleanup qualification remain
separate work.

Operators can resolve an accepted run without querying private tables:
`GET /operator/runs/{public_id}?tenant={tenant}` requires the operator credential,
not a product credential. Its `allocated_references` maps the reservation to the
product, coding/workflow sessions, and four phase tasks. Session identifiers use
Runtime's public exposure projection. Follow the coding task at
`/internal/cayu/api/tasks/{task_id}` and sessions at
`/internal/cayu/api/sessions/{session_id}` with the same operator authentication.
The operator credential has deployment-wide scope; the explicit tenant selects
the business record and does not grant product-route access. The response omits
the private accepted request, instructions, and subject.

These are allocated references, not evidence that the referenced resource exists
or that execution, delivery, or cleanup succeeded. A session may still return 404
before coding starts. The task status is an observation, not recovery permission;
use current Runtime inspection and exact recovery plans for actions. This lookup
does not implement delivery approval, product recovery, or the complete incident
exercise, which remain required qualification work.

`GET /operator/runs/{public_id}/tasks?tenant={tenant}` uses the same operator and
tenant boundary and reads only the four reserved task IDs. Each phase independently
reports `absent`, `conflicting`, `unavailable`, or `recorded` evidence. Recorded
observations include task status, update/observation times, recorded owner, lease
expiry, and a cancellation marker. Only the named roles' opaque worker IDs are
shown; other recorded owners are `present_unprojected`. Arbitrary task reasons,
errors, results, requests, and status payloads are not echoed. One unavailable or
conflicting phase does not erase the other observations. These sequential reads
are not an atomic snapshot. An expired lease or absent cancellation marker does
not grant recovery permission or prove quiescence. Native effect and cleanup
evidence explicitly remain `not_inspected` by this task-only view; a completed task
must not be interpreted as a successful PR or completed cleanup.

`GET /operator/runs/{public_id}/delivery?tenant={tenant}` reads historical native
Git/GitHub evidence through the same operator authorization boundary. It compares
saved preparation/delivery requests and reconstructs native lifecycle/result
artifacts. Each delivery reports independent evidence; missing or conflicting
records never become success. Git lifecycle-only observations do not claim a result
or cleanup. Result-bearing observations expose the recorded cleanup flag. GitHub
reports native state, exact result/head identifiers, PR number, check/review state,
and native next-poll time, but no general cleanup flag exists in its result schema.
Task receipt agreement is reported separately: a receipt for an older result does
not authenticate the newest effect. A matching GitHub task receipt is evidence of
the maintained handler's local settlement contract, not proof of remote success.
Provider URLs, titles/bodies, arbitrary reason text, and raw task payloads are
omitted. These sequential historical reads neither contact the provider nor grant
retry/recovery permission, and do not constitute the final business-success report.

The emitted `operations.maintenance_github_intake.load_verified_github_result`
provides read-only final-delivery reconstruction after the caller authorizes a
reservation. It joins independently verified coding and settled approved Git
publication with the exact saved GitHub request, consent, native result and
completed owner-cleared task receipt. Passed checks are required independently of
review approval: the native `approved` state can still have pending checks.
Required review approval, open/unmerged exact-head PR, complete observations and
absence of a pending poll are also required. Failed, partial, uncertain or stale
evidence cannot pass. This is recorded delivery evidence, not a new provider
observation or permission to retry; the final HTTP result projection remains
separate from this readback boundary.

`GET /operator/runs/{public_id}/cost?tenant={tenant}` is a read-only recorded-cost
view under the same operator authorization. Runtime's causal accounting includes
the workflow root and its causal descendants, not unrelated runs. The response
identifies the current configured pricing by fingerprint and returns its recorded
USD estimate; a later inspection can reprice history if configuration changes.
Unpriced line items remain visible, including standalone hosted-resource evidence
without a completed model step. Missing cohort evidence returns a null estimate,
not a claim of zero spend. An empty recorded cohort has no cost observations.
Every response marks billing completeness `not_established`: this is not an
invoice, proof of quiescence, remaining-budget calculation, final business outcome,
or the repeated-trial cohort report. Provider details and raw pricing data are not
projected. Invalid or unavailable accounting returns a fixed 503 diagnostic.

`GET /operator/runs/{public_id}/result?tenant={tenant}` joins verified coding,
approved Git publication, and the recorded GitHub result through the readback
boundary above. A successful response says `recorded_verified_delivery` and
`observation: durable_history`: it is not a fresh GitHub query, permission to
merge, a final invoice, or proof that all five qualification gates passed. It
includes the exact commit/tree/workspace revision, PR link, result digests, and
operator-authorized acceptance/cost links. Rejected readback returns 409;
unavailable or invalid native evidence/configuration returns a fixed 503 without
partial success output. Use the task and delivery views to diagnose non-success
states; this endpoint does not derive retry authority from them.

Configure `CAYU_MAINTENANCE_GITHUB_WEB_ORIGIN` explicitly, for example
`https://github.com`, alongside the existing GitHub host configuration. The API
origin is not inferred to be the web origin. This initial presentation contract
accepts canonical lowercase ASCII HTTPS DNS origins with an optional canonical
port; no credentials, path, trailing slash, query, fragment, Unicode hostname,
or IPv6 literal. Use a canonical DNS hostname for enterprise installations.
The recorded PR URL must exactly match that origin, configured owner/repository,
and PR number. A mismatched URL is not silently rewritten or published.

`GET /operator/runs/{public_id}/acceptance?tenant={tenant}` reconstructs the sealed
coding result and repeats application-owned verification. It returns only bound
result/request/revision/corpus/probe SHA-256 fingerprints, not raw artifact
contents, provider output, or a local artifact-store path. This remains available
for a verified coding result before PR delivery. Both new routes require operator
authentication and tenant-qualified lookup; product credentials do not grant
access. All these reads leave delivery and task state unchanged.

`tests/qualification/test_repository_maintenance_coding_loss.py` exercises a real
SIGKILL after model dispatch through the emitted application's intake, workflow,
and worker handler. A fresh process reconstructs the same reservation, observes
the expired task fence through the operator route, and uses registered Runtime
recovery before exact settled-product readback. Native task cancellation
reconciliation remains separate: it produces `cancelled`, never a successful
coding result or permission to deliver. The test checks receipt replay, stale
worker rejection, zero model redispatch, unchanged source, artifact retention,
and process reaping. Its stable-identity provider and local runner have no remote
allocations; this does not prove live Docker/provider cleanup or continuation of
unfinished coding work. The isolated local target is explicitly retained until
fixture cleanup, not presented as a disposed production container.

The source qualification case `sqlite-approval-process-restart` in
`tests/qualification/test_repository_maintenance_application.py` runs preparation
through the emitted worker handler in a separate process. After the native pending
approval and completed preparation task are durable, the existing recovery harness
SIGKILLs and reaps that process. A fresh generated application reads the same
request/tree/result and reservation, refuses a stale tree, and queues/replays the
exact approval through the authenticated operator route. The rest of that journey
checks native Git delivery, one controlled-service PR, and settled cleanup. Coding
and Docker admission remain controlled fixtures; this is not production Docker,
PostgreSQL, live-provider, or installed-adopter evidence. No task is reset or given
new execution authority simply because a process was lost.

The `sqlite-managed-lost-push-ack` and `sqlite-managed-lost-github-ack` cases use
that same queued application path. They lose the response only after the local
Git remote or controlled GitHub service accepts the write. The GitHub case also
makes reconciliation temporarily unavailable, verifies the durable ambiguous result
while the original task owner remains held, and follows the native connector's
bounded poll permission to matching evidence. Assertions require one commit, one
PR, exact saved consent and task receipts, current-head checks, and cleanup. The
separate direct `sqlite-lost-github-ack` case covers fresh-connector readback after
positive close. These cases do not establish the complete operator recovery
experience for an indefinite outage; that remains separate coverage.

`operations/maintenance_deployment.py` constructs the PostgreSQL deployment
dependencies without replacing generated session/knowledge/artifact wiring. It
requires explicit database and persistent public-authority alias configuration and
a validated budget policy, and supplies a PostgreSQL budget ledger using that same
database. `validate_startup_schema()` invokes Runtime validation and application
reservation readiness; it never initializes or migrates schemas. It is not a
periodic database-health probe. Before starting application roles, run the normal
`cayu storage status --postgres DSN` and explicit `cayu storage migrate --postgres
DSN` deployment procedure with the required backup authority and exact breaking
revision acknowledgements. Do not automatically waive backups or acknowledge
revisions.

The PostgreSQL consumer also emits `compose.yaml`, `Dockerfile.application`,
`Dockerfile.host-tools`, their context allow-lists, and `deployment/README.md`.
The guide specifies immutable offline application inputs, separate role env files,
explicit migrations, persistent mounts, stop controls and coordinated restore.
Structural asset tests are not native Compose, image-build or deployment evidence;
those checks remain required on an authorized host.

After Runtime migration, from the generated project with the same
`CAYU_DATABASE_URL`, run `python -m operations.maintenance_schema initialize` and
`python -m operations.maintenance_schema check`. These commands touch only the
application reservation schema through its existing store owner; they do not
construct providers, inspect Docker, or migrate Runtime schemas. Run bootstrap
under one operator with application writers stopped. A failure may follow a
successful commit: inspect/check before retrying, and do not assume rollback.
Initialization refuses partial/conflicting reservation schemas rather than
repairing them. These source-tested commands do not establish live PostgreSQL
deployment readiness. Startup validation is not an automatic resource closer. The caller retains these
dependencies after partial startup failure until they can be safely closed.

The declared zero-argument factory is `app:build_maintenance_app`. Before any
application resources are constructed it reads `CAYU_MAINTENANCE_BUDGET_JSON`, a
complete serialized public `BudgetPolicy` with pinned pricing and reservations.
The value is required, limited to 64 KiB of UTF-8, and rejects duplicate keys,
non-finite JSON constants, invalid policy shape, or the maintenance budget guard's
unsupported settings. Configuration errors do not echo the submitted policy.
There is no default price book or unbudgeted fallback. All deployed roles and
compared versions must use the same pinned policy, database, and retained budget
history; changing those can change the accounting identity. Loading this JSON
does not establish current provider prices, dispatch bounds, or live bill totals.
The generic generated constructors remain explicit fixture/construction seams,
not the declared production factory.

`configuration/maintenance.py` also exposes `configured_maintenance_access()` for
the application-owned API host. It requires `CAYU_MAINTENANCE_ACCESS_JSON`, bounded
to 64 KiB, with exactly `operator_token` and `product_tokens`. The latter maps
credentials to objects containing exactly `tenant_id` and `subject_id`; one to
64 product credentials are supported. The existing access adapter validates and
copies these principals and rejects sharing an operator token with a product
credential. Missing, duplicate-key, malformed, or oversized configuration has no
open-access fallback and is not echoed in errors. Supply credentials only to the
host process, never to the coding workspace. This loader does not itself start or
qualify an HTTP listener.

`configured_maintenance_git_authority()` reads required
`CAYU_MAINTENANCE_GIT_JSON` through the same 64 KiB, duplicate-key/non-finite-value
rejecting parser. Its four object sections are exactly `repository`, `commit`,
`security`, and `limits`, validated as the public `RemoteGitRepositoryAuthority`,
`RemoteGitCommitAuthority`, `RemoteGitSecurityAuthority`, and
`RemoteGitDeliveryLimits` contracts. Each call returns fresh native models.
Commit `authored_at` is explicit, not generated from the current retry time.
Configuration failures use a fixed error without echoing the document.

These settings are host data, not human approval or evidence that credentials,
source, or remote refs are valid. Retain the full native request at first phase
admission; reloading mutable configuration cannot reconstruct original approved
authority. The native broker still owns fresh validation and exact publication.
The local generated delivery journey exercises this loader; production phase
intake and restart integration remain unqualified. Do not supply credential
values in these authority sections or expose them to the coding workspace.

The host-only `operations.maintenance_git_intake.ensure_git_preparation_task`
connects verified completed coding readback to a reserved Runtime preparation
task. It captures the configured native request as bounded canonical JSON, with
the original coding session and allocated delivery identity, and records an
explicit host-asserted operator origin. The caller must authenticate that operator
and authorize the tenant-qualified run first. A matching identity or actor string
does not authenticate a caller. Exact replay preserves the existing task; changed
configuration or actor conflicts instead of resetting it. Enqueue performs no
commit, push, or approval.

`operations.maintenance_git_worker.handle_git_preparation_task` validates the
store-owned claim and complete saved request, rechecks current configuration and
verified coding evidence, then uses the existing verified native broker handoff
without approval. The supported handler requires the native broker and local
artifact store. It retains the call when its waiter is cancelled, preserving task
ownership until settlement; enclosing roles must retain shared dependencies for
that entire await. Runtime owns the task lease and final task transition. The
result contains the native request fingerprint and result digest: task completion
alone is not delivery success. The generated SQLite/local-Git journey exercises
this handler through `run_task_worker` and reads back `approval_required`, with
the private repository intentionally retained pending approval. Named Git roles
and host configuration are described below; production restart qualification
remains unfinished.

The host-only `load_git_approval_request` joins completed preparation task evidence
to the native pending result and prepared intent, including exact request, tree,
and artifact digest. It revalidates coding and current configuration; its return
is historical pending evidence, not a promise that remote state has not changed.
`ensure_git_delivery_task` requires explicit expected request fingerprint, tree,
stable approval ID, and authenticated operator subject. It retains the full native
request and affirmative approval in the reserved delivery task. The caller must
authenticate and authorize the run before invoking either helper.

A first enqueue requires the latest native pending receipt to match the selected
result. Exact task replay remains possible after native progress, preserving
response-loss recovery without allocating another task. Changed approval or actor
conflicts. Neither helper writes native approval artifacts, commits, or pushes;
the eventual broker call must still revalidate current source and destination.
`handle_git_delivery_task` reconstructs the exact store-owned claim, request and
saved approval, then passes that approval explicitly through the same retained
native owner used for preparation. It rechecks current configuration and coding
evidence before dispatch. The broker still owns exact tree/destination validation
and reconciliation; a task result fingerprint/digest is not itself push success.
The local generated journey executes the approved task through `run_task_worker`,
reads its native result, and requires one exact pushed commit and settled cleanup.

`load_verified_git_result` reconstructs the downstream handoff from the
tenant-qualified coding result and reserved completed Git task. It checks saved
request/approval authority, current configuration, the exact native result digest,
and the final `pushed` lifecycle receipt, including matching approval and tree.
Non-success, incomplete cleanup and conflicting evidence cannot feed PR delivery.
This is historical verified evidence, not a current remote-head assertion or new
publication permission. The GitHub connector still owns those checks. The generated
SQLite/local-Git journey feeds this reconstructed result into its controlled GitHub
flow rather than relying solely on objects retained by the fixture.

The authenticated operator host also provides these routes, each with exactly one
`tenant` query selector and the deployment operator bearer credential:

- `POST /operator/runs/{id}/git/preparation` with `{}` queues preparation.
- `GET /operator/runs/{id}/git/approval` returns the recorded pending request,
  fingerprint, prepared tree and result digest for review. `recorded_state` is
  historical evidence, not a claim about current remote eligibility.
- `POST /operator/runs/{id}/git/approval` accepts exactly `request_fingerprint`,
  `prepared_tree`, and a stable `approval_id`. Actor identity comes from the
  authenticated operator, never the request body.

POST responses are `202` with `id`, `phase`, and stored `task_status`; they do not
claim delivery succeeded. Product credentials cannot access these routes. Operator
authentication and tenant-qualified lookup precede body parsing and action. Bodies
are limited to 8192 bytes; extra/duplicate fields, malformed fingerprints/trees,
and actor/configuration overrides are rejected. A conflicting action returns
`409`; uncertain or unavailable execution returns a fixed `503` without claiming
rollback. Retry the same approval identity and expected values after response loss.
The native request projection is operator-only review data, not a product response.
Dedicated ASGI tests cover these adapters with actual queues and controlled native
evidence reads. The managed generated SQLite journey additionally enters through
product `POST /runs`, runs coding, preparation and approved Git delivery through Runtime workers, and
uses the operator routes against real local artifacts for approval. Its initial
and reopened application use a pinned synthetic price book and SQLite budget
ledger; every scripted response reports explicit usage. Tests require one settled
reservation per model step and retain those amounts after reopening. These are
fixture prices and usage, not actual provider charges or model-quality evidence.
The managed SQLite GitHub stage now reviews and approves through operator HTTP,
then runs the queued task through Runtime with the native connector and controlled
REST responses. It requires one PR creation, checks at the exact pushed head,
positive connector close, and exact approval replay without a new task. Named
GitHub process configuration is described below. Registered recovery now has a
same-application controlled coding-loss test; full production recovery remains
unqualified. The emitted `operations/maintenance-incidents.md` describes supported
diagnosis, exact native plan execution, approval restart and acknowledgement-loss
observations. Its presence does not establish a completed operator trial.
Local worker cancellation tests retain the task fence until the
dispatched native call settles, including across lease expiry.
ASGI tests do not establish TLS, a running listener, human consent, or the complete
Docker/PostgreSQL production journey.

For a named worker that already received the configured factory's `CayuApp`,
`bind_maintenance_deployment(app, agent_name=...)` creates the coding front door
from that app's public native Docker factory registration. It retains the original
workspace, artifact store, provider, and Runtime stores rather than calling the
factory again. Bind once per role and retain that bundle as the sole shutdown
owner. This host-only adapter is not an authentication boundary for arbitrary
injected apps, nor evidence that independently configured stores share a database.

After permanently stopping top-level dispatch, the caller can observe
`MaintenanceDeployment.aclose(timeout_s=30)`. One retained shutdown task joins
background subagent registries and the five Runtime cleanup drains before closing
the provider and native store pools. Every drain must return `True`. A timeout
returns `False`; cancelling an observer does not cancel the shutdown task. Keep
the deployment reachable and observe it again rather than dropping its resources.
Only a completed drain refusal, before any dependency close, permits a fresh
drain attempt. A close failure remains an error on later observation, including
after an earlier store closed successfully. The wait is bounded, not necessarily
the underlying close operation; neither process termination nor a timed-out wait
proves cleanup. Do not reuse the application after shutdown starts. Executable
service roles still need to prove the dispatch-stop and retained-owner conditions.

The consumer registers `coding`, `git_preparation`, `git_delivery`, and
`github_delivery` entrypoints
for `cayu worker <name> --shutdown-grace-seconds 30`. Each binds the CLI-created
app once, validates Runtime and reservation schemas, and delegates only its
corresponding `maintenance.<name>` tasks to the existing Runtime worker and
claimed-task handler. All four use one retained lifecycle. They do not elect the
store-wide interrupted-task scanner. Recovery uses the registered Runtime entrance
and phase-specific owners; no generic replacement-worker recovery role is implied.
Run one process per role for this single-repository deployment. The stop event is
cooperative: active work keeps its Runtime claim until the handler settles.
Cleanup refusal retains the process and its resources instead of returning a
clean result. A caller cancellation is propagated after that owned lifetime ends,
with original cancellation bookkeeping preserved. The CLI's signal grace supplies
the hard process bound; exit 124 is failed shutdown, not workspace/container cleanup
proof. Without an operator stop signal, unresolved cleanup deliberately remains
resident. CLI subprocess regressions send real SIGTERM and verify cooperative exit
143 versus blocked-close exit 124 using controlled dependency boundaries. Native
PostgreSQL/Docker qualification remains required; these process tests do not prove
database or external-container cleanup.

Git roles additionally require host-only `CAYU_MAINTENANCE_GIT_HOST_JSON`, containing
exactly `broker_root`, `git_executable`, and `remote_url` as nonempty strings; both
paths must be absolute. This is separate from `CAYU_MAINTENANCE_GIT_JSON` request
authority. The supported remote alias is `origin`. Credential profile `none`
uses no credentials; `maintenance-git-https` uses native secret references mapped
to `CAYU_MAINTENANCE_GIT_USER` and `CAYU_MAINTENANCE_GIT_TOKEN`. Secrets are not
resolved at construction. Native transport checks reject credentials on non-HTTPS
remotes and reject embedded URL credentials before creating the broker directory.
Do not supply these secret variables to coding processes or containers. The egress
profile is an identity, not a network firewall: the deployment must enforce its
admitted destination.

GitHub review/intake helpers now retain separate PR consent in the reserved
`maintenance.github_delivery` task. `load_github_approval_request` derives a native
request from verified durable Git/coding evidence. `ensure_github_delivery_task`
requires an authenticated operator subject, the exact reviewed request fingerprint,
and a stable GitHub approval ID; it does not reuse Git push consent or contact
GitHub. Exact retries preserve the task; changed authority conflicts.

Required `CAYU_MAINTENANCE_GITHUB_JSON` contains exactly `requested_at`,
`repository_alias`, `installation_id`, `account_id`, `mode`,
`existing_pull_request_number`, `metadata`, `checks`, `reviews`, `security`, and
`limits`. Native schemas validate the final request. `requested_at` is explicit
host-selected configuration, canonicalized by the native schema, never renewed
by retry. Set it before review/intake and retain it with that operation. The
saved task contains the complete request and approval; claim reconstruction
validates both against the store-owned invocation and reserved identities.
The same operator-only host exposes `GET /operator/runs/{id}/github/approval`
with one `tenant` query selector. It returns `id`, `phase`, `request_fingerprint`
and the complete reviewed native `request`; it does not assert current remote
state or pending consent. `POST` at that path accepts exactly
`request_fingerprint` and `approval_id`, deriving the actor from authentication.
The shared 8192-byte parser rejects extra fields, Git-tree or actor overrides,
duplicate keys and malformed values. `202` reports the queued task's status,
not PR success; conflicts are `409`, uncertainty/unavailability is a fixed `503`.
Exact retries retain the same task after acknowledgement loss. Product credentials
cannot use these routes. `handle_github_delivery_task` reloads the store-owned
claim and verified upstream evidence, compares current request configuration,
then obtains a fresh connector from a host-owned synchronous factory. The existing
observer owns run/poll/seal/close; only after it settles may the handler complete
the task with the native request fingerprint and result digest. Non-success native
outcomes are not promoted to PR success. A connector cannot be shared across tasks
after the observer seals it. The `github_delivery` role validates a host factory
after schema readiness and before task claiming. Local controlled REST does not
qualify live delivery or the complete production deployment.

`CAYU_MAINTENANCE_GITHUB_HOST_JSON` contains exactly `owner`, `repository_name`,
and `api_base_url`. The native configuration requires a canonical account/repository
and credential-free HTTPS API base. The supported generated connector uses alias
`github`, connector ID `maintenance-app-github`, credential profile
`github-installation-token`, egress profile `github-api-only`, and the installed
connector behavior fingerprint; conflicting request configuration is rejected.
Repository ID comes from the Git authority; installation/account IDs come from
GitHub request authority. Token reference `github-token` resolves only through
`CAYU_MAINTENANCE_GITHUB_TOKEN` in the host process. Keep it out of coding processes
and containers. Factory setup validates and captures configuration but neither
resolves secrets nor creates a connector; every task obtains a fresh native one.
The deployment must enforce the declared egress destination. Process regressions
cover real SIGTERM and grace expiry with controlled dependencies, not production
database, network or container cleanup.

`operations/maintenance_asgi.py` supplies the host's request-settlement wrapper.
Admission opens only after the wrapped FastAPI lifespan successfully reports
startup. Shutdown permanently seals admission (HTTP 503, WebSocket 1013), then
waits for every admitted ASGI call to finish before forwarding shutdown to Runtime.
A cancelled request that continues working remains included in that wait.
Cancellation of the shutdown observer does not cancel request completion handles;
the original signal is propagated after teardown. The wrapper cannot restart its
lifespan. This establishes call settlement, not the quiescence of untracked work
inside arbitrary extensions; deployment cleanup drains remain necessary. It has
no independent hard-kill mechanism, and the executable host/Compose qualification
must still establish the process bound.

The executable ASGI factory is `operations.maintenance_api:build_api`. It validates
access configuration before constructing one canonical app, binds that app's
deployment once, and wraps the existing product/operator host with `MaintenanceASGI`.
Readiness runs before mounted Runtime startup; dependency cleanup follows request
settlement and Runtime drains, including when startup fails. API and coding roles
share cancellation-preserving lifetime observation and ordered error handling.
For a local listener, after explicit schema preparation and all required host
configuration:

```bash
uvicorn operations.maintenance_api:build_api --factory --host 127.0.0.1 --port 8000 --timeout-graceful-shutdown 30
```

Uvicorn's graceful timeout bounds request waiting, not the complete lifespan.
The production process supervisor/Compose stop limit is still required. Hard
termination is failed or uncertain cleanup, not proof that containers disappeared.
Factory and ASGI tests use controlled readiness/close boundaries; they do not
establish native PostgreSQL readiness, real Docker disposal, or deployed TLS.

Use one task/repository at a time, at most eight model steps, a 180-second coding
deadline, and owned cleanup bounded separately to 30 seconds. Required project
checks remain `format`, `lint`, and `test`; the independent oracle is additional.
Probe execution has a 30-second deadline and a 32-KiB response bound. Exhausting a
bound is a non-success outcome, not permission to skip work. These limits must be
wired into the application before a trial is accepted.

The emitted reservation owner allocates the coding expiry once alongside the run
identifiers. An intake retry reads that expiry unchanged, even after it expires;
it does not start another 180-second allowance. Explicit trusted expiry readback
must match the original value. The local managed-worker and workflow-Evals
journeys exercise this boundary; production intake and worker deployment remain
separate qualification obligations. Evals retains its evaluator-owned case expiry
in the reservation rather than allocating a later coding deadline.

The four controlled workflow-Evals journeys also save and reload the private
`EvalRun`, recapture its exact completed attempt with the native workflow capture
API, and rescore the unchanged independent verdict. Report JSON intentionally
omits the in-memory trajectory: child evidence is read back from the original
store. The checks reject changed input or target identity and forbid execution
callbacks during recapture; rejected repairs remain failed after scoring. This
proves a saved coding-stage capture path, not a reviewed live failure, an emitted
production eval target, repeated-version comparison, or verified PR completion.

The emitted `evals/maintenance.py` now provides the opt-in, context-managed native
target and fixed two-trial corpus; see emitted `evals/maintenance.md` for ownership
and revision requirements. Its corpus-only plan uses native concurrency/time/trial
ceilings. The starter eval command remains separate. Controlled target tests are
not the missing production paired-version or reviewed-failure evidence.

The authorized live spending cap is **USD 1 total**, not per trial. Plan two pinned
application versions with two trials each, at most USD 0.15 per trial; reserve at
most USD 0.25 for source-blind adoption and USD 0.15 for remaining attributable
work. These allocations do not authorize additional spending or an extra cohort.
All provider calls, failed attempts, repairs, children, and any judges consume the
same total. Do not dispatch when the remaining bounded cost cannot be established.
No paid trial starts before its provider access, disposable GitHub target/account,
and private Cayu Brain evidence location are authorized and configured.

The paired versions use the same seeded repository, probes, image, and acceptance
rules. A deliberately regressed application version must fail the same independent
acceptance; changing the oracle to create a favorable comparison is prohibited.
Report every attempt, including non-success and unknown cost. Cost per verified
completion is undefined when there are no verified completions.

## Non-compensating proof obligations

The qualification declaration in
`tests/qualification/repository_maintenance_toolchain.py` separates the target's
pinned `pyproject.toml` from the application's immutable build context. It exposes
the fixed format, lint, project-test and independent-probe checks, plus the
`python-version` structured command. The probe lives at
`/opt/cayu-acceptance/range_probe.py`, outside the writable target; its exact bytes
are checked by a Docker admission probe. Image construction must include that
file in the protected build inputs. The generated application's default profile
pins its own repository and must not be reused unchanged for this separate target.
Declaration/dependency tests do not prove that the image has been built or admitted.

1. **Normal path:** reproduce the seed failure, make a scoped change, run complete
   checks and independent verification, settle the coding product, approve the
   exact change/destination, and reconcile its commit and PR. No merge.
2. **Recovery:** kill the coding worker, restart around durable approval and
   publication, reject stale owners/duplicates, and prove cleanup for each case.
   Settled-result reconstruction does not establish unfinished-work continuation.
3. **Operations:** diagnose and recover seeded incidents through supported
   operator surfaces. No private SQL or manual state edits.
4. **Adoption:** freeze fresh source-blind submissions before independent grading;
   retain prerequisites, elapsed time, and every intervention. Keep private trial
   prompts and raw evidence in the authorized Cayu Brain location.
5. **Evaluation:** capture a reviewed workflow failure through existing Evals,
   compare the two pinned versions with repeated trials, and report attributable
   costs and verified completions separately from Runtime correctness.

Use `scripts/run_runtime_qualification.py` and its registry for installed-build
correctness. Maintenance contract modules and the eighteen complete journey cases
have separate `repository-maintenance-*` scenario identities so each runs under
the existing per-scenario time and cleanup bounds. `repository-maintenance-corpus`
selects only seed/probe/toolchain checks, not the full application. Run all of the
maintenance scenarios for application correctness; a collection regression checks
that their selectors cover every maintenance case exactly once. Scenario splitting
does not establish installed-wheel success or production acceptance.
Real Docker/PostgreSQL and separately authorized real-model/GitHub
evidence remain required. Corpus unit tests, generated-workflow tests, and existing
subsystem results cannot substitute for the complete application journey.
