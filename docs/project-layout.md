# Project Layout

The framework repo and user-created agent projects should use different structures.

## Framework Repo

The framework repo is horizontal by subsystem:

```text
src/cayu/
  core/
  runtime/
  providers/
  runners/
  workspaces/
  storage/
  mcp/
  vaults/
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
from cayu import AgentApp
from domains.billing import billing_module
from domains.onboarding import onboarding_module

app = AgentApp("support-agent")
app.include(billing_module)
app.include(onboarding_module)
```
