# Tool-exposure economics

This deterministic paired evaluation compares two valid application strategies:

- `stable_broad` exposes the same two-tool profile for both model steps;
- `changing_narrow` exposes one tool per step and changes profiles between steps.

Both sides execute the same two-step workload, use the same model, scripted output
contract, quality check, and fixture prices, and must produce the exact `quality-ok` outcome.
The report includes request and retry counts, exposure profiles and transitions,
keyed tool-manifest and cache-prefix identities, provider-reported token/cache
categories, the quality result, and Cayu's estimated session cost.

```bash
uv run python -m examples.tool_exposure_economics.deterministic
```

The token counters and price book are deterministic fixtures, not provider
benchmarks or invoices. The report therefore sets
`evidence_scope="deterministic_fixture"` and
`universal_savings_claimed=false`. A narrower tool list can reduce schema input,
while a stable list can improve prompt-cache reuse; applications must measure the
whole workload—including retries and quality—against their actual provider and
pricing contract.
