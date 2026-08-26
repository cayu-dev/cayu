# Project Layout

This is a design/maintainer document for Cayu's production agent runtime. It describes the intended repo and generated-project structure; it is not a complete end-user guide.

The Cayu repository and user-created agent projects should use different structures.

## Cayu Repository

The Cayu repository is horizontal by subsystem:

```text
src/cayu/
  core/
  workflows/
  environments/
  runtime/
  providers/
  runners/
  workspaces/
  storage/
  mcp/
  vaults/
  proxies/
  cli/
  dashboard/
```

This keeps runtime package dependency direction clear and avoids circular imports.

## Example and Verification Surfaces

Runnable product examples and deterministic measurement fixtures are maintained
contributor surfaces, not miscellaneous snippets:

```text
examples/
  ADVANCED_RUNTIME_EXAMPLES.md
  _advanced_support/
  bounded_fork_group/
  cache_aware_research_council/
  counterfactual_approval/
  prompt_cache_compaction/
  repo_maintainer_tournament/
  tainted_incident_response/
  tool_exposure_economics/
tests/advanced_examples/
scripts/nightly_verification.py
```

Agents and developers changing advanced runtime behavior should start with
[`examples/ADVANCED_RUNTIME_EXAMPLES.md`](../examples/ADVANCED_RUNTIME_EXAMPLES.md).
It routes to each scenario, the shared evidence envelope, deterministic
specifications, live-provider registrations, and the relevant runtime contracts.
The product narrative and measured proof boundaries live in
[`docs/advanced-runtime-examples.md`](advanced-runtime-examples.md).

Keep one provider-neutral `scenario.py` per advanced example. Deterministic and,
when supported, live modules construct backends around that scenario rather
than implementing separate behavior. Shared runtime-facing helpers belong under
`examples/_advanced_support/`; domain-specific code remains inside its example.

## Generated User Project

Default user projects should be Rails-like and easy to understand:

The package-shipped [application-anatomy guide](../src/cayu/guides/application-anatomy.md)
is the canonical contract for the generated factory, process-scoped app, shared
durable state, and explicit service ownership described by this layout.

```text
invoice-agent/
  pyproject.toml
  app.py
  run.py
  AGENTS.md
  agents/
  tools/
  evals/
  tests/
```

`app.py` explicitly registers agents, tools, storage, and runtime config. The
default scaffold creates only directories containing working source; add
workflows, prompts, memory, configuration, environments, and domain packages
when the requested behavior actually needs them. `AGENTS.md` is the generated
project-local source of truth for inspection, safe generation, testing, evals,
and evidence reporting.

`cayu new NAME --composition coding` is the maintained opt-in starter for a
repository-coding application. It keeps provider selection in `app.py` and puts
the explicit assembly in `composition.py`. That module registers existing public
APIs for bounded file inspection and mutation, `rg` search, Git change review,
local artifacts, SQLite knowledge with pending writes, a bounded background
reviewer plus result recovery, and human-input pauses. The generator preflights
`git`, `rg`, and the POSIX descriptor-relative primitives required by secure
`LocalWorkspace` path operations, then creates a clean initial Git commit so
change review has a deterministic baseline. Unsupported hosts fail during
generation or application construction instead of a later tool call.

The composition selects concrete implementations; selection is not authority.
The registered exposure policy separately decides which tools are model-visible,
and ordinary tool policy, approval policy, and runtime gates independently
authorize calls.

The coding workspace defaults to the generated project root. An explicit
`CAYU_WORKSPACE_ROOT` may be relative to that project or absolute, but it must
resolve to an existing Git repository root and never a filesystem root. The
generated `LocalWorkspace` and minimal-environment `LocalRunner` are trusted-host
development adapters, not a sandbox. With `inherit_env=False`, the runner still
forwards Cayu's operational allow-list for command resolution, home, locale, and
temporary-directory behavior; it does not inherit arbitrary host variables. The
composition is ordinary editable Python; do not replace it with an implicit agent
kind, registry, permission grant, or post-start mutation.

## Large User Project

Large projects should support vertical domain modules:

```text
support-agent/
  app.py
  domains/
    billing/
      agents.py
      tools.py
      workflows.py
      prompts/
      evals/
    onboarding/
      agents.py
      tools.py
      workflows.py
      prompts/
      evals/
  shared/
    tools/
    memory/
```

Cayu should care about explicit registration, not hardcoded folder names.

```python
from cayu import CayuApp
from domains.billing.agents import billing_agent, billing_tools
from domains.onboarding.agents import onboarding_agent, onboarding_tools

app = CayuApp()
app.register_agent(billing_agent, tools=billing_tools)
app.register_agent(onboarding_agent, tools=onboarding_tools)
```
