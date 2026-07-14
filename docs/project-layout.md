# Project Layout

This is a design/maintainer document for Cayu's runtime framework. It describes the intended repo and generated-project structure; it is not a complete end-user guide.

The framework repo and user-created agent projects should use different structures.

## Framework Repo

The framework repo is horizontal by subsystem:

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

This keeps framework dependency direction clear and avoids circular imports.

## Example and Verification Surfaces

Runnable examples are product and contributor surfaces, not miscellaneous test
fixtures:

```text
examples/
  ADVANCED_RUNTIME_EXAMPLES.md
  _advanced_support/
  cache_aware_research_council/
  counterfactual_approval/
  repo_maintainer_tournament/
  tainted_incident_response/
tests/advanced_examples/
scripts/nightly_verification.py
```

Agents and developers changing advanced runtime behavior should start with
[`examples/ADVANCED_RUNTIME_EXAMPLES.md`](../examples/ADVANCED_RUNTIME_EXAMPLES.md).
It routes to each scenario, the shared evidence envelope, deterministic
specifications, live-provider registrations, and the relevant runtime contracts.
The product narrative and measured proof boundaries live in
[`docs/advanced-runtime-examples.md`](advanced-runtime-examples.md).

Keep one provider-neutral `scenario.py` per advanced example. Deterministic and
live modules construct backends around that scenario rather than implementing
separate behavior. Shared runtime-facing helpers belong under
`examples/_advanced_support/`; domain-specific code remains inside its example.

## Generated User Project

Default user projects should be Rails-like and easy to understand:

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

The framework should care about explicit registration, not hardcoded folder names.

```python
from cayu import CayuApp
from domains.billing.agents import billing_agent, billing_tools
from domains.onboarding.agents import onboarding_agent, onboarding_tools

app = CayuApp()
app.register_agent(billing_agent, tools=billing_tools)
app.register_agent(onboarding_agent, tools=onboarding_tools)
```
