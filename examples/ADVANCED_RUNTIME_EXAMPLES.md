# Advanced Runtime Examples

Start here when you are evaluating, extending, or asking an agent to work on
Cayu's advanced examples. These applications serve two purposes:

1. exercise Cayu against deterministic providers and real provider APIs; and
2. demonstrate programmable runtime behavior built from durable sessions,
   checkpoints, forks, policies, budgets, workspaces, and recovery.

The assertion contract is the stable behavioral envelope around model output.
Do not assert exact prose.

For the product narrative and dated measurements, see
[Advanced runtime strategies](../docs/advanced-runtime-examples.md). For the
cost strategy and governance map, see
[Cost optimization and governance](../docs/cost-optimization.md).

## Example map

| Example | Product idea | Stable runtime evidence | Entry point |
| --- | --- | --- | --- |
| `prompt_cache_compaction` | Refresh an expensive provider prefix once, compare it with bounded compaction from the same compactable source, then keep later summaries bounded to the checkpoint delta. | Tool and thinking request-shape parity, paired cache/uncached counters, provenance-gated cost evidence, two real compaction cycles, bounded incremental input, and explicit separation of session and comparison-only spend. | [README](prompt_cache_compaction/README.md) · [app](prompt_cache_compaction/app.py) |
| `cache_aware_research_council` | Prepare and compact shared context once, then reduce repeated input while exploring several research strategies independently. | Shared causal budget and lineage, persisted compaction checkpoint, paired provider-token comparison, evaluator weakness, and critique-aware repair. | [README](cache_aware_research_council/README.md) · [app](cache_aware_research_council/app.py) |
| `counterfactual_approval` | Turn approval latency into useful computation without granting speculative branches authority. | Approved and denied futures are authority-free, stale state is rejected, one analysis is selected as advisory continuation context while the other is ignored, exactly one mutation occurs, and its receipt is recovered after `CayuApp` reconstruction. Both child sessions remain durable and auditable. | [README](counterfactual_approval/README.md) · [app](counterfactual_approval/app.py) |
| `repo_maintainer_tournament` | Generate competing repairs, reject reward hacking, and promote only the verified winner. | Candidate isolation, deterministic tests and diff gates, test-weakening rejection, one winner, idempotent PR recovery, and optional real Git worktree/commit/push/PR verification. | [README](repo_maintainer_tournament/README.md) · [app](repo_maintainer_tournament/app.py) |
| `tainted_incident_response` | Preserve trust boundaries even when untrusted context crosses a session fork and `CayuApp` reconstruction. | Durable inherited taint, a real runtime `tool.call.blocked` event, zero protected mutations, restricted outbound authority, and one sanitized handoff. | [README](tainted_incident_response/README.md) · [app](tainted_incident_response/app.py) |

## Where agents and developers should look

Use these files in this order:

1. This index for the suite-level contract and commands.
2. The example's `README.md` for its scenario and prerequisites.
3. The example's `scenario.py` for the provider-neutral orchestration.
4. `deterministic.py` and `live.py` for backend construction.
5. [`_advanced_support/results.py`](_advanced_support/results.py) for the
   durable evidence envelope.
6. [`tests/advanced_examples/`](../tests/advanced_examples/) for deterministic
   behavioral specifications.
7. [`scripts/nightly_verification.py`](../scripts/nightly_verification.py) for
   credential-gated live registration.
8. [`docs/runtime-contracts.md`](../docs/runtime-contracts.md) before changing a
   fork, checkpoint, policy, budget, or recovery invariant.

Shared helpers belong under `_advanced_support/` only when two or more examples
need the same runtime-facing behavior. Scenario-specific domain logic stays with
the scenario.

## Run the suite deterministically

Deterministic runs use `ScriptedModelProvider`, call no external model API, and
must satisfy the same structural assertions as live runs:

```bash
for example in \
  prompt_cache_compaction \
  cache_aware_research_council \
  counterfactual_approval \
  repo_maintainer_tournament \
  tainted_incident_response
do
  uv run python -m "examples.${example}.app"
done
```

Run the deterministic specifications directly with:

```bash
uv run pytest -q tests/advanced_examples
```

## Run with a real provider

Every example supports Gemini, OpenAI, and Anthropic through the same CLI:

```bash
# Gemini through its OpenAI-compatible endpoint
GEMINI_API_KEY=... uv run python -m examples.cache_aware_research_council.app \
  --mode live --provider gemini --trials 1

# OpenAI through Cayu's native Responses API provider
OPENAI_API_KEY=... uv run python -m examples.cache_aware_research_council.app \
  --mode live --provider openai --trials 1

# Anthropic through Cayu's native Messages API provider
ANTHROPIC_API_KEY=... uv run python -m examples.cache_aware_research_council.app \
  --mode live --provider anthropic --trials 1
```

Replace the module name to run another example. Defaults can be overridden with
`CAYU_GEMINI_MODEL`, `CAYU_OPENAI_MODEL`, and `CAYU_ANTHROPIC_MODEL`. The current
defaults are `gemini-3.1-flash-lite`, `gpt-5.4-mini`, and
`claude-sonnet-4-6`.

`prompt_cache_compaction` is intentionally Anthropic-only in live mode because
its assertion reads Anthropic prompt-cache counters; its deterministic mode is
provider-neutral.

### Real repository maintainer boundary

The repository tournament uses the fake GitHub-shaped loopback service by
default, including during ordinary live-provider runs. Opt into real Git and
GitHub side effects explicitly:

```bash
export CAYU_REPO_MAINTAINER_REPOSITORY=your-org/disposable-private-repo
export CAYU_REPO_MAINTAINER_SOURCE_PULL=1
export GITHUB_TOKEN=...

OPENAI_API_KEY=... uv run python -m examples.repo_maintainer_tournament.app \
  --mode live --provider openai --trials 1
```

The token needs pull-request read/write access. Clone and push authentication use
the configured Git remote; set `CAYU_REPO_MAINTAINER_CLONE_URL` when SSH is not
appropriate. Use a disposable repository with a known source pull request, not
a production repository.

Before publication, the live path loads the real source PR, SHA, file list, and
baseline; replays every candidate in an isolated Git worktree; and reruns the
same gates used by the preliminary tournament. Any divergence fails closed
before a branch is pushed. The successful path verifies the remote commit, PR
head SHA, base, changed files, and one-open-PR recovery contract.

## Evidence contract

Every successful run writes ignored JSON under
`.cayu-example-results/<scenario>/<run-id>.json`. Repository workspaces are
written under ignored `.cayu-example-workspaces/` or
`.cayu-example-repositories/` directories.

The shared result contains:

- scenario, mode, provider, model, and run identity;
- named boolean assertions rather than exact model prose;
- per-session parent lineage and causal budget identity;
- provider-reported token usage and model/tool step counts;
- recovery state, taint labels, compaction count, and durable receipt ids; and
- scenario-specific metrics and output evidence.

`ScenarioResult.require_verified()` fails the process if any named assertion is
false or the result is not `verified`. Live success therefore means the runtime
and semantic envelope passed—not merely that the provider returned text.

## Verification layers

| Layer | Cost and prerequisites | Purpose |
| --- | --- | --- |
| Deterministic specifications | No provider key or model spend | PR-safe behavioral coverage for all five scenarios. |
| Primary Gemini checks | `GEMINI_API_KEY`; five trials when the registered nightly checks are invoked | Multi-trial verification of the main live-provider path. |
| OpenAI and Anthropic portability checks | Matching provider key; one trial per registered check | Detect provider-specific tool, structured-output, and usage regressions. |
| Real repository promotion | Provider key, GitHub authority, Git credentials, and a disposable repository; manual opt-in | Verify clone, real worktrees, commit, push, PR creation, and idempotent recovery. |

The nightly runner is available today, but scheduled GitHub automation is still
tracked separately. See [Nightly verification](../docs/nightly-verification.md)
for current invocation and status vocabulary.

## Adding or changing an advanced example

- State the Cayu runtime behavior the example demonstrates.
- Define stable assertions before adding provider prompts.
- Keep deterministic and live modes on the same scenario function.
- Persist evidence for lineage, usage, recovery, authority, and side effects
  relevant to the claim.
- Add deterministic coverage under `tests/advanced_examples/`.
- Register credential-gated live checks in `scripts/nightly_verification.py`.
- Separate first-attempt token effects from total provider usage, including
  retries.
- State what the example does not prove.
- Never publish a universal cost or quality claim from one provider run.
