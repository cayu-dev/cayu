# Repo Plan

This is a design/maintainer document for Cayu's runtime framework. It records implementation direction; it is not a public roadmap or complete end-user guide.

## Naming

```text
Repository: cayu
Python package: cayu
Import: import cayu
CLI: cayu
```

## Framework Direction

The repository should preserve stable contracts while expanding runtime capabilities through small vertical slices. Avoid preserving accidental APIs before each public runtime shape is deliberate.

Important framework capabilities:

- event stream shape
- basic agent loop
- agent/environment/session separation
- tool protocol
- path-safe framework-native file tools
- framework-native command execution tool
- Anthropic Messages API and OpenAI Responses API providers
- LocalRunner implementation
- LocalWorkspace implementation
- CLI scaffold ideas

Areas that need deliberate design:

- public API
- provider abstractions
- structured tool results
- runtime/server/session store
- dashboard
- workflow durability
- sandbox lifecycle
- memory/search

## Implementation Rule

Implement contracts first, then one tiny vertical demo:

```text
orchestrator agent
two worker agents
shared state/task table
SQLite event log
local runner/workspace
```

Avoid building polished CLI/dashboard before the contracts are real.
