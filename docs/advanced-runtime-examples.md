# Advanced Runtime Strategies

Cayu's advanced examples demonstrate what becomes possible when an application
can program durable sessions, checkpoints, forks, policies, budgets, workspaces,
and recovery as one runtime.

The individual primitives are not the claim. Session branching, compaction,
evaluation, and approval flows exist in other systems. The Cayu-specific value
shown here is their composition under explicit runtime control, with durable
lineage and evidence that survives model variation and `CayuApp`
reconstruction around the same store.

For commands, source locations, and contributor guidance, start with the
[advanced example developer index](../examples/ADVANCED_RUNTIME_EXAMPLES.md).

## Five executable product stories

### Refresh a provider cache, then compact only the checkpoint delta

The [prompt-cache compaction example](../examples/prompt_cache_compaction/)
uses a real tool round and two compaction cycles. The first compaction must
extend the exact provider request prefix, including tools and thinking options,
and persist its cache-read usage. A bounded `ModelCompactor` control receives
the same captured compactable source after the candidate session completes;
its comparison-only attempt is reported separately from durable session spend.
Both summaries must preserve a mandatory retention token, and the candidate
must recover it after the second compaction, so lower counters alone cannot
pass the scenario.
The second session compaction must summarize only the previous checkpoint plus
newly compactable messages. Deterministic coverage is PR-safe and uses clearly
labeled fixture prices; the credential-gated Anthropic path additionally
requires a nonzero provider cache-read counter. Live dollar evidence is emitted
only with a caller-supplied, provenance-bearing model catalog.

### Share context once, then explore several futures

The [cache-aware research council](../examples/cache_aware_research_council/)
creates one durable research source, records a cache-window observation,
compacts at a checkpoint, and forks several strategy sessions from the shared
state. A separate evaluator fork identifies a material weakness, and a repair
fork addresses it.

The stable claim is not that every model writes the same report. The result must
prove that:

- all sessions share the intended causal budget and lineage;
- compaction state is persisted before the candidate forks;
- compacted and uncompacted branches use the same source and prompts;
- provider-reported input usage is measured separately for first attempts and
  total attempts, including retries;
- the evaluator finds a weakness; and
- the repair addresses the evaluator's critique.

This turns session forking into a cost and quality strategy rather than a copy
of chat history.

### Compute during approval waits without granting authority

The [counterfactual approval example](../examples/counterfactual_approval/)
forks two read-only futures while a protected action waits for a human:
"approved" and "denied." Both branches may analyze and prepare a decision brief,
but neither can perform the protected mutation.

After the decision, the scenario selects the approved analysis as advisory
continuation context and ignores the denied analysis. Both child sessions remain
durable and auditable; this is selection, not deletion or a runtime-level branch
commit. Cayu then rejects stale external state, revalidates the actual state,
and performs exactly one mutation. The example reconstructs `CayuApp` around the
same store and recovers the external receipt without replaying the side effect.

The product improvement is useful computation during human latency while
approval semantics remain unchanged. The current evidence proves authority and
recovery behavior; it does not yet quantify wall-clock latency saved.

### Tournament repairs and publish only the verified winner

The [repository maintainer tournament](../examples/repo_maintainer_tournament/)
forks three repair strategies, applies their model-produced changes in isolated
workspaces, and runs deterministic tests and diff-policy gates. An evaluator
rejects test weakening and selects the eligible production patch.

The default boundary is a GitHub-shaped loopback HTTP service for deterministic
CI. An explicit real-repository mode additionally:

- loads the actual source PR, head SHA, changed files, and baseline before the
  agents deliberate;
- replays every candidate in a real Git worktree;
- requires the real gates and winner to match the preliminary tournament;
- fails closed before publication if they diverge;
- commits and pushes the winning branch;
- creates one real GitHub PR; and
- reconstructs the client, retries publication, and verifies the same open PR,
  head SHA, base, and changed files.

This is the difference between demonstrating patch-selection logic and proving
the complete repository boundary.

### Preserve trust boundaries across forks and application reconstruction

The [tainted incident response example](../examples/tainted_incident_response/)
reads hostile evidence through a configured untrusted source tool. A generic
session fork derives the durable taint, persists it in child metadata, and
reconstructs the application around the same store.

The quarantine session contains a registered protected mutation tool so the
example exercises the actual runtime gate. Success requires one durable
`tool.call.blocked` event with the inherited label and zero protected
executions. Only a sanitized, provenance-bearing artifact reaches the clean
session, which performs one allowed notification.

The claim is origin-based authority control, not prompt-injection detection.

## Observed live evidence

The following observations were recorded during credentialed verification on
July 11, 2026. They are evidence that the examples exercised real provider and
external boundaries; they are not universal benchmarks or pricing guarantees.

| Scenario | Provider and model | Trials | Observed result |
| --- | --- | ---: | --- |
| Research council | Gemini `gemini-3.1-flash-lite` | 1 | Uncompacted branches reported 25,853 input tokens and compacted branches 11,228: 14,625 fewer, or 56.6%. Both sides completed in three model steps. |
| Research council | OpenAI `gpt-5.4-mini` | 1 | First-attempt input fell from 18,146 to 10,151: 7,995 fewer, or 44.1%. The compacted side required two extra model steps, so total input was 17,487: only 659 fewer, or 3.6%. |
| Counterfactual approval | OpenAI `gpt-5.4-mini` | 1 | All nine assertions passed across eight model requests; one protected mutation and one recovery receipt were recorded. |
| Repository maintainer | OpenAI `gpt-5.4-mini` plus a private disposable GitHub repository | 1 | All twelve fake-and-real boundary assertions passed across five model requests; three real worktrees were gated, one commit was pushed, and one PR was created and recovered idempotently. |
| Tainted incident response | OpenAI `gpt-5.4-mini` | 1 | All six assertions passed across nine model requests; the runtime blocked the protected mutation after `CayuApp` reconstruction, executed it zero times, and sent one sanitized notification. |

Anthropic support is wired through the same scenario and credential-gated
nightly registrations, but this dated observation set does not claim a live
Anthropic result because an authorized key was not available for the run.

### Reading the cost observation correctly

The research example separates two different questions:

1. **Did compaction make the first candidate request smaller?** Both observed
   providers said yes.
2. **Did the whole candidate branch use fewer input tokens after retries?** The
   Gemini observation retained most of the reduction; the OpenAI observation
   retained only 3.6% because extra attempts consumed most of the first-request
   saving.

That distinction is why Cayu records provider-reported total usage instead of
marketing a context-size estimate as realized savings. Dollar savings require a
price book and must include source, evaluator, repair, and retry overhead.
The [cost optimization and governance guide](cost-optimization.md) applies the
same evidence standard across Cayu's optimization and budget-control options.

Before publishing a general benchmark, run several trials per provider and
report the median and range. The registered Gemini checks use five trials when
invoked; OpenAI and Anthropic portability checks currently use one trial to
bound spend.

## What the suite proves

- Runtime-owned session lineage and checkpoint state can be composed into
  branch, evaluator, and repair strategies.
- Stable behavioral assertions can survive provider and prose variation.
- Authority-free speculative work can coexist with a protected human decision.
- Side-effect recovery can reconcile a durable external receipt without replay.
- Origin-based taint remains enforceable across a generic fork and `CayuApp`
  reconstruction around the same store.
- Repository promotion can fail closed until real worktrees reproduce the
  preliminary tournament result.
- Provider usage evidence can distinguish first-attempt opportunity from total
  realized usage.

## What the suite does not prove

- A universal percentage reduction in token usage, latency, or dollars.
- That every model/provider succeeds with the same number of attempts.
- That evaluator quality is correct for every domain.
- That selecting an approval future durably deletes, merges, or commits child
  sessions; the example chooses advisory context while preserving both branches
  for audit.
- That taint labels detect malicious text; applications configure trusted and
  untrusted origins.
- That these scenarios prove process death or recovery in a fresh operating
  system process. Cayu's separate `SIGKILL` verification lane covers that
  boundary with durable stores.
- That the repository maintainer is safe to point at an arbitrary production
  repository. Real mode requires an explicitly configured disposable target and
  scoped credentials.
- That registered nightly checks are already scheduled continuously; the runner
  exists, while scheduled GitHub automation is tracked separately.

## Verification model

Each example has one provider-neutral scenario used by deterministic and live
backends. The result writes named assertions, per-session lineage and usage,
recovery state, authority evidence, and scenario outputs under an ignored
`.cayu-example-results/` directory.

The verification ladder is:

1. deterministic specifications in `tests/advanced_examples/`;
2. credential-gated live provider checks in `scripts/nightly_verification.py`;
3. explicit external-boundary runs such as real GitHub promotion; and
4. the exhaustive invariants in [Runtime contracts](runtime-contracts.md).

A green model response is not enough. `ScenarioResult.require_verified()` makes
the process fail unless every stable assertion passes.
