# Maintained coding product

The maintained coding product is Cayu's GitHub-independent path from an
application-owned coding request to a durable `patch_ready_for_delivery` result.
Generate the complete editable application with:

```bash
cayu new mycoder --preset coding --execution docker \
  --coding-toolchain python --coding-command-authority structured
```

Docker execution is for repositories the application has already classified as
trusted. It is not hostile-code isolation. The runtime has no network or
workload credentials and the core product never commits, pushes, opens a pull
request, waits for CI, or merges.

## Generated ownership

The scaffold deliberately gives coding agents an architecture before they edit:

- `domain/coding_product.py` owns caller input, stable recovery IDs, required
  checks, and explicit review or human gates.
- `workflows/coding_product.py` owns product admission, execution, evidence
  compilation, verification, publication, and recovery.
- `operations/coding.py` owns the concrete Cayu app, source workspace, Docker
  environment factory, toolchain, stores, tools, policies, and agents.
- `agents/`, `prompts/`, `tools/`, `policies/`, `environments/`, and `knowledge/`
  own their named concerns. Root `app.py` only exposes construction functions.
- `tests/test_coding_composition.py` is the credential-free structural and
  deterministic execution seam. Application-specific tests belong beside it.

The generated required `test` check executes the complete `tests/` tree, so the
maintained composition proof and application-owned regressions participate in
the same patch-ready gate.

`build_app()` remains available for ordinary Cayu sessions.
`build_coding_product_application()` returns the maintained high-level front
door. Its async `run(CodingProductTask(...))` method returns a
`CodingProductPublication` whose candidate contains the terminal product state
and bounded evidence references.

The generated `build_app()` and `build_coding_product_application()` accept
`budget_policy` and `budget_ledger` and pass them to `CayuApp`. The ledger remains
caller-owned. For shared reservation enforcement, supply the same durable ledger,
session history and pinned app-level policy to every participating worker; the
default ledger is in-memory. Supplying these dependencies does not establish live
pricing or provider token bounds by itself.

For shutdown, stop top-level dispatch before observing background reviewers. A
registered `SubagentTool` exposes `background_task_registry`; its bounded `drain()`
waits for child streams without cancelling them. The generated composition uses a
separate registry per app. A false result leaves work owned and dependencies must
remain available. After streams settle, the app's other cleanup drains are still
required before closing provider transports or stores; registry settlement alone
does not establish remote-effect or environment cleanup.

`await application.inspect_execution_profile(task)` inspects the exact runtime
request shape used by `run()` without creating a session or product admission or
dispatching work. It delegates Cayu's read-only run preflights and reflects the
current runtime controls, including budget-policy identity. This fingerprint is
not source verification, authentication, or delivery approval. Execution still
reconstructs and validates current authority; retain and compare the expected
fingerprint when binding an application-owned accepted request.

Use stable, caller-owned `product_run_id`, `session_id`, and `task_id` values.
Reconstructing the application with the same IDs and the same admitted source
and runtime authority recovers an already published result instead of blindly
replaying work. Use new IDs for a new attempt.

## Authority fixed before execution

Admission observes a complete bounded source manifest and binds:

- source origin, workspace, destination, baseline revision, and clean committed
  Git `HEAD`, staged-index, and tracked-file-flag authority;
- the source projection that removes runtime/private directories, credentials,
  package stores, caches, build outputs, and the application-owned
  `docker-coding-image.json` authority receipt before copy-in or copy-back;
- task identity and the digest of the exact input messages;
- explicit parent-session and causal-budget IDs, when supplied; these are bound
  into admission and replay, not inferred from an enclosing Python workflow;
- Docker image, toolchain profile, dependency inputs, and no-network boundary;
- the exact Cayu execution-profile fingerprint for the requested run;
- tool-manifest, tool-policy, approval-policy, and redaction identities; and
- the finite required-check, reviewer, human-approval, and evidence bounds.

`CayuApp.inspect_run_execution_profile()` performs the same bounded read-only
initial preflights used by execution without creating a session or dispatching
provider, tool, hook, or environment-factory work. This lets the application
bind exact runtime authority before exposing the coding surface.

The product publishes a matching `WorkContract`, deterministic completion
verifier, and result resolver. Runtime events remain evidence, not authority:
model prose and tool output cannot waive a check, approve a review, change the
source destination, broaden the toolchain, or claim external delivery.

A new product run rejects a dirty source tree or non-default tracked-file index
flags before admission. Credential and generated-output patterns are outside the
source projection. Any other existing Git-ignored path fails admission, and a
newly created ignored path outside that policy blocks publication instead of
disappearing from final Git evidence. The bounded index observer supports the
complete admitted source-path envelope; Git submodules are rejected because the
flat Docker source projection cannot preserve gitlink semantics.
After copy-in, the Docker binding re-observes the projected guest tree against
the admitted source revision, including each regular file's Git executable mode.
The transfer preserves `100644` and `100755` in both directions, and
revision-checked copy-back rejects concurrent host content or mode drift. A
source change between admission and copy-in therefore fails before provider or
tool execution can use a stale workspace.
Unsettled recovery reuses but freshly validates the previously admitted Git
control authority before dispatch. The application validates it again after
execution and immediately before patch-ready publication, while permitting the
expected copied-back working-tree changes. A changed host `HEAD`, staged index,
or tracked-file flag settles as a source conflict. Already settled historical
results remain recoverable after the run has intentionally changed the working
tree. Host Git observation and revalidation run outside the application event
loop, so bounded subprocess timeouts cannot stall unrelated async work.

## `patch_ready_for_delivery`

This state is accepted only when all of the following are true:

1. Every application-required named check has one settled passing receipt from
   the admitted execution profile, including a measured duration and bound to
   the exact final workspace revision.
2. Every retained structured command has a complete collected and published
   result, an admitted exit code, bounded output identity, measured duration,
   and settled workspace effect. A complete admitted nonzero diagnostic may be
   retained; partial, ambiguous, timed-out, cancelled, or unadmitted command
   evidence does not pass.
3. All retained mutations are settled; partial, stale, or ambiguous effects do
   not pass.
4. Revision-aware source publication completed or proved the source unchanged,
   a complete final source revision is known, and the runtime-owned receipt
   authenticates the destination/workload identities, conflict policy,
   copy/delete counts, and secret-free settlement snapshot digest.
5. Complete bounded final Git status, summary, and diff evidence was retained.
   The Docker binding captures this evidence at finalization, verifies that its
   ephemeral Git `HEAD`, index entries, tracked-file flags, and effective
   configuration were not changed,
   proves that the retained Git entries and counts cover every byte-level path
   change in the copied workspace, and seals the result into the runtime
   finalization receipt only for the exact Docker coding binding. Model-invoked
   Git output is not settlement authority. The diff includes untracked-file
   contents and preserves exact textual whitespace. The binding compares raw and
   Git-filtered object identities at the baseline and final state; any actual
   text, encoding, ident, or clean-filter transformation on a changed path makes
   the diff incomplete. Includes and effective clean, smudge, or process
   filter commands are rejected before filtered hashing. Pagination, runner
   truncation, unresolved projection truncation, redaction, or omitted binary
   evidence does not pass.
6. Configured reviewer and human gates have explicit application-owned
   settlement. A successful delegated tool call is not itself review approval.
7. The admitted source, task, session, agent, toolchain, execution profile, and
   application identities still match.
8. No external delivery effect was performed by the core product.

Any artifact used to settle these gates is read back in full and checked against
its content digest, size, session, agent, environment, media type, and operation
metadata. Missing, substituted, redacted, or truncated artifacts fail closed.

The result says the patch is ready for an optional delivery layer. It does not
say that delivery happened.

## Non-success and recovery states

The lifecycle distinguishes `checks_not_run`, `checks_failed`,
`source_conflict`, `toolchain_rebuild_required`, `review_required`,
`human_input_required`, `blocked`, `denied`, `cancelled`, `failed`, `partial`,
`ambiguous`, and `reconstruction_required`. These are product results or
fail-closed recovery signals, not alternate spellings of success.

Lifecycle receipts and product artifacts are append-only and content-addressed.
Before publishing any terminal candidate, including a non-success result, the
runner records ready-to-publish and publishing receipts bound to the exact
candidate digest. If the result artifact becomes durable before its final
lifecycle receipt, retry reconstructs and returns that exact result without
dispatching the session again. An active or ambiguous prior attempt is not
silently replayed: the runner requires evidence reconstruction. Before initial
execution lifecycle mutation or session dispatch, the runner atomically creates one
deterministic execution-claim artifact. This is a compare-and-swap boundary:
among concurrent callers for the same product run, one wins and every loser
fails without modifying the winner's lifecycle. If the source baseline changed
before execution, admission fails with a source conflict. If Docker or
dependency authority drifted, the toolchain must be rebuilt and explicitly
readopted. Dependency-sensitive checks verify the exact inputs both before
dispatch and after the runner is quiescent, so a repository check cannot modify
its own toolchain authority and retain a passing result.

`CodingProductRunner.recover_settled_execution(request)` reconstructs product
publication for the exact original invocation after that invocation has settled.
It checks the retained initial source observation, original session incarnation
and invocation anchor, a consistent durable event snapshot, and Runtime's
invocation-release evidence. It does not dispatch a provider or remove the
execution claim. Cleanup may advance the run epoch: retained Runtime rebind
receipts must prove the unchanged invocation identity and profile across those
advances. Missing lineage remains reconstruction-required; identical values in a
new admission do not substitute for that proof.
A missing result after `ready_to_publish` or `publishing` is
recompiled with the original lifecycle prefix and must match the already-selected
candidate digest; changed evidence cannot select a replacement candidate. An
ordinary retry's `result_publication_unsettled` diagnostic does not discard that
selection. Malformed published evidence is rejected rather than overwritten.

This entrance does not resume unfinished work or accept a later resumed or
recreated invocation as the original execution. It can reconstruct a non-success
result and does not manufacture Docker publication or check evidence. A recovered
result is historical evidence, not permission to execute or deliver; delivery
retains its separate current-source and destination checks.

Generated `CodingProductApplication` exposes this as `recover_settled(task)`.
Reconstruct the application with the same persistent stores and pass the original
`CodingProductTask`. Missing admission and changed task/runtime authority fail
instead of starting a new run. The workflow delegates reconstruction to Runtime;
it does not parse checkpoints or create its own recovery owner.

## Extending the product

Keep product policy in the generated domain and workflow files. Custom language
profiles remain public Cayu `DockerCodingToolchainProfile` values with finite
named checks and structured command authorities; pass the selected profile into
`CodingProductApplication`. Preserve the exact-profile inspection and all
evidence gates when replacing the generated Python composition.

Application-owned environment adapters must preserve the request-bound source
copy authority through
`DockerCodingEnvironmentFactory.create_workspace_binding(...)`. Generated
projects must not parse Cayu's private authority metadata or import private
`cayu._*` modules.

External delivery is a separate optional composition. It should consume only an
accepted immutable coding-product artifact, establish its own credentials,
idempotency, provider authority, and receipts, and retain commit/push/PR/CI/merge
state independently. Cayu's maintained first layer is the
[approved remote Git delivery](remote-git-delivery.md) broker, selected in a
generated Docker coding application with `--with remote-git-delivery`.
Applications that also need a pull request, exact-head checks, and bounded
review feedback can select the subsequent [GitHub delivery](github-delivery.md)
layer with `--with github-delivery`.
