# Approved remote Git delivery

Cayu's optional remote Git delivery product turns one accepted
`patch_ready_for_delivery` coding-product publication into one exact commit on a
new remote branch. The broker runs on the application host, outside the coding
agent and its environment. It never gives the model a remote URL, credentials,
raw Git command authority, or network authority.

Generate the maintained integration seam with:

```bash
cayu new mycoder --preset coding --execution docker \
  --with remote-git-delivery
```

The capability adds `integrations/remote_git.py`. Its application-owned
`build_remote_git_delivery_broker()` factory is disabled in scaffolds that did
not explicitly select the capability. The generated coding composition remains
credential-free and continues to stop at `patch_ready_for_delivery`.

## Authority and trust boundary

The application constructs a `RemoteGitDeliveryRequest` from the immutable
coding-product publication with `remote_git_delivery_request()`. Before any
commit or push, it fixes:

- the broker-configured remote alias and public remote identity;
- one full expected base commit and canonical base ref;
- one new destination branch under the configured namespace;
- exact author, committer, timestamp, title, and body;
- credential, egress, policy, approval, redaction, and broker behavior
  identities; and
- finite source, output, Git-object, timeout, and retry bounds.

The broker configuration maps the public alias to the actual URL. That mapping,
the broker-owned repository, the Git executable, vault resolvers, and network
access remain host-side application configuration. HTTPS authentication uses
`RemoteGitHttpCredentials` containing `SecretRef` values and a
`SecretResolver`; credentials are resolved only for remote Git subprocesses and
are redacted before output evidence is retained. URLs containing credentials
are rejected.

The built-in broker requires a POSIX application host with file-size resource
limits. Git and its transport/indexing children inherit
`max_git_storage_bytes` as a per-file cap. A shallow
fetch retains packs instead of expanding an unbounded history into loose
objects. Aggregate Git storage and object counts are checked before publishing
a prepared intent. These are broker-process bounds, not quotas on the remote
server or unrelated host processes.

The durable configuration identity also binds the exact executable content,
remote URL, credential references, broker directory and branch namespace.
Reusing public aliases does not authorize changing their configured targets.
Repository-local configuration may contain only inert Git initialization
bookkeeping; URL rewrites, includes, filters and additional transport behavior
fail before another Git command starts.

The built-in transport invokes a fixed Git executable with structured argument
arrays and an isolated environment. It disables inherited system/global Git
configuration, hooks, credential helpers, external diffs, prompts, optional
locks, and submodule recursion. `.git`, `.cayu`, and `.runtime` source roots,
links, submodules, unresolved index entries, and attribute/filter transformations
are rejected. Applications are still responsible for enforcing the configured
egress profile around the broker process.

## Two-phase delivery

Delivery deliberately requires two calls:

1. `broker.prepare(...)` revalidates the durable coding-product result, checks,
   final source manifest, retained diff, remote base, and destination absence.
   It materializes the exact source into a broker-owned isolated repository,
   stages it, verifies every Git blob byte, and publishes a durable
   `RemoteGitPreparedIntent`. It does not create a commit or write remotely.
2. The application obtains an approval specifically bound to the request
   fingerprint, prepared tree, and policy fingerprint. The helper
   `approve_remote_git_delivery()` constructs that typed approval after the
   application has made the decision. Calling `broker.run(...)` with the same
   publication, workspace, request, and approval creates the exact commit and
   attempts the one authorized push.

For an unfinished delivery, calling `run(...)` without an approval returns
`approval_required`. A missing, denied, stale, or mismatched approval never
commits or pushes. An exact terminal replay retrieves its original receipt
without authorizing new execution. Returning a successful receipt still requires
a fresh observation that the remote destination equals its exact commit.

The push is limited to a new `refs/heads/cayu/...` branch by default. It uses an
empty expected-value lease, so a destination created by another actor is a
conflict. The default branch, force-updating an existing ref, ref deletion,
tags, merge, pull-request creation, CI waiting, and merge are outside this
product and cannot be expressed by `RemoteGitDeliveryRequest`.

## Success and recovery

Git discovery is confined to each invocation's owned directory; ancestor checkout
configuration cannot redirect remote observations. The delivery parent must equal
the retained coding baseline. Preparation applies the retained reviewed diff to
that parent and requires its tree to equal the exact materialized source tree.
Authenticated no-change evidence skips patch application only when status,
summary, and diff all describe no changes; the source tree must still equal the
exact parent tree.
Validated manifest paths are staged explicitly, including tracked files matched
by ignore rules; ignore rules cannot silently remove admitted source files.

`pushed` is reported only after a fresh remote observation proves the
destination ref equals the exact local commit and broker cleanup succeeds. The
result exposes `next_commit` and `next_ref` only in that state. Every result
also retains the delivery session, source-result/run/workspace, final revision,
diff/check, broker repository, policy, redaction, credential/egress profile,
behavior, and execution-limit identities needed to audit or reconstruct its
authority. Durable request, prepared intent, approval, lifecycle, step, and
result evidence make retries identity-bound and content-conflict detecting.

Other states are explicit:

- `conflict` means the base, source, or destination no longer matches admitted
  authority;
- `denied` means approval was missing required authority;
- `committed_locally` and `pushing` retain the exact commit for crash or
  cancellation recovery;
- `failed` means a definite push rejection was freshly observed;
- `ambiguous` means the transport outcome could not be proved and may be safely
  reconciled by retrying the same delivery identity;
- `partial` means the remote ref was proved correct but required cleanup did
not settle; and
- `cancelled` or `reconstruction_required` prevent Cayu from inventing success
  when retained state is insufficient.

A lost push acknowledgement is reconciled by observing the remote ref. Recovery
reuses the retained commit and never manufactures a second commit. Exhausted
retry authority or missing broker state requires explicit reconstruction rather
than silently repeating an uncertain effect.

Public preparation and delivery calls use one process-shared owner per delivery
in the broker directory. A competing call fails without deleting or reusing the
active repository. If an interrupted artifact write remains in flight, a durable
fence prevents reuse; the same process can retry after its retained store
observer proves settlement. Another process without that proof requires
reconstruction and cannot clear the fence merely because the caller stopped.

Remote success is durably recorded as `partial` before cleanup removes local
recovery material. Cleanup runs in an owned subprocess with the configured
timeout; a timeout leaves `partial` rather than granting success. An exact retry
can finish cleanup and publish `pushed`, including after terminal-publication
failure when the local repository was already removed. It does not push again.

Definitive `denied`, `conflict`, and `failed` outcomes are persisted before bounded
private-repository cleanup. Their `cleanup_settled` field records whether removal
finished. Exact delivery retry settles pending cleanup without restarting Git
delivery; receipts remain available. Ambiguous work retains its recovery material.

Remote Git delivery is the provider-independent lower layer for later hosting
integrations. A GitHub connector may consume its exact `next_commit` and
`next_ref` to open a pull request and observe checks, but must retain separate
GitHub authority and lifecycle evidence. Neither layer authorizes merge.

## Authenticated transport verification

The deterministic suite pushes only to temporary local bare repositories. An
additional opt-in test exercises the public broker against a dedicated private
HTTPS repository that requires authentication. It deliberately supplies an
impossible expected base, so it stops after authenticated ref observation and
cannot fetch, commit, or push.

With explicit operator approval, provide `CAYU_REMOTE_GIT_TEST_URL`,
`CAYU_REMOTE_GIT_TEST_BASE_COMMIT` (the exact current `refs/heads/main` commit),
`CAYU_REMOTE_GIT_TEST_USERNAME`, and `CAYU_REMOTE_GIT_TEST_PASSWORD` through the
test process environment. Use dedicated test credentials, not production ones.
Then run:

```bash
CAYU_RUN_REMOTE_GIT_AUTH_TEST=1 pytest -q tests/core/test_remote_git_delivery_live.py
```

Without the opt-in flag the test skips before reading credentials or contacting
a service. This probe verifies authenticated transport admission, not a live
service push; full commit/push/recovery proof remains the isolated bare-Git suite.
