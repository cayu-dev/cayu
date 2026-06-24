# Project Layout

This is a design/maintainer document for the current framework foundation. It describes the intended repo and generated-project structure; it is not a complete end-user guide.

The framework repo and user-created agent projects should use different structures.

## Framework Repo

The framework repo is horizontal by subsystem:

```text
src/cayu/
  core/
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

## Generated User Project

Default user projects should be Rails-like and easy to understand:

```text
invoice-agent/
  pyproject.toml
  app.py
  agents/
  tools/
  workflows/
  prompts/
  memory/
  evals/
  config/
  tests/
```

`app.py` explicitly registers agents, tools, workflows, storage, and runtime config.

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
