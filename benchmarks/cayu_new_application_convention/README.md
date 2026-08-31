# `cayu new` application-convention acceptance

This benchmark asks a fresh coding agent to extend a project produced by the
installed Cayu distribution without a Cayu source checkout, web documentation,
credentials, or task-specific persistent configuration. It complements the
deterministic scaffold tests; model behavior is not a release-time unit-test
oracle.

The two cases cover different authoring paths:

- `ordinary-tool-transcript.md` adds an approval-gated external-effect tool,
  injected integration, runtime test, and behavioral eval through
  `cayu generate tool`.
- `observability-transcript.md` changes a cross-cutting selectable capability
  through catalog discovery and a disposable `cayu new --dry-run` comparison.

Both sessions started from a committed untouched `--preset agent` scaffold,
used an isolated environment containing the exact candidate wheel and pytest,
ran with network access disabled, and were invoked with Codex user configuration
and rules disabled. The prompts explicitly prohibited web access and inspection
of any Cayu source checkout. The agent could read only the generated repository,
the installed distribution, and its package-shipped guides for task knowledge.

The durable environment and deterministic receipts are recorded in
`results/2026-08-30.md`. Each transcript includes the exact prompt, chronological
commands and results, resulting source tree, and limits of the evidence.
