# Repo Plan

This is a design/maintainer document for the current framework foundation. It records near-term implementation direction; it is not a public roadmap or complete end-user guide.

## Naming

```text
Repository: cayu
Python package: cayu
Import: import cayu
CLI: cayu
```

## Framework Direction

The repository should grow from stable contracts into a usable runtime through small vertical slices. Avoid preserving accidental APIs before the runtime shape is proven.

Important framework capabilities:

- event stream shape
- basic agent loop
- tool protocol
- path-safe file tools
- LocalRunner implementation
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

## Initial Implementation Rule

Implement contracts first, then one tiny vertical demo:

```text
orchestrator agent
two worker agents
shared state/task table
SQLite event log
local runner/workspace
```

Avoid building polished CLI/dashboard before the contracts are real.
