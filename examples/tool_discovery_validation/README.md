# Provider-neutral tool discovery validation

This credential-free fixture exercises Cayu's real `search_tools` / `call_tool`
path and prints one bounded, versioned report:

```bash
uv run python -m examples.tool_discovery_validation.deterministic
```

The lifecycle proof performs the following sequence through ordinary `CayuApp`
APIs:

```text
parent search -> invoke -> resume -> invoke -> fork
child reject copied parent ref -> search -> invoke
```

It verifies the branch-local tool view through typed inspection and runtime
events. The parent retains one discovery grant across resume. The fork starts
with a new empty generation, rejects the copied parent reference before target
work, and creates a fresh reference only after its own search. Every provider
request retains the same two-definition `search_tools`, `call_tool` core; the
discovered schema remains private transcript content rather than becoming a
new top-level tool definition. The fixture also verifies that neither opaque
reference appears in public events or the content-minimized inspection
projection.

The paired evaluation runs one fixed outcome under two valid application
strategies:

- `direct_catalogue` exposes all 36 fixture tools directly;
- `search_tools` exposes only Cayu's stable two-tool core and discovers the one
  hidden capability needed by the task.

Both sides attempt one invalid call, apply one target effect, produce the exact
`quality-ok` outcome, and report provider requests/model steps, searches,
workload-relative unnecessary searches, invalid-argument rejection, target
invocations, effects, approval requests, provider tool counts, keyed manifest
identity, provider-style token/cache categories, observed end-to-end latency,
and fixture-priced whole-session cost. A six-case bounded corpus separately
measures exact-name, canonical-id, description, and parameter-property ranking,
plus exclusion of a directly exposed tool.

The usage counters and prices are deterministic fixtures, not provider
benchmarks or invoices. Observed latency is local wall-clock evidence and will
vary by machine and load. `unnecessary_searches` means searches above the one
search required by this fixed hidden-tool workload; it is not a general model
quality metric. The report therefore sets
`evidence_scope="deterministic_fixture"` and
`universal_savings_claimed=false`. It demonstrates what to measure, not that
discovery is universally faster, cheaper, or better than direct exposure.
