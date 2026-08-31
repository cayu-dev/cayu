# Ordinary tool/eval transcript

## Environment

- Date: 2026-08-30
- Coding agent: Codex CLI 0.151.0, `gpt-5.6-sol`
- Thread: `01a051f6-4a3a-7960-832f-899a2f429c46`
- Initial generated-project commit:
  `2199652f17c649a1fd2229cddbb39ef2b5149b8f`
- Installed candidate: `cayu-0.4.0-py3-none-any.whl`
- Wheel SHA-256:
  `83a82ac7a87eeb0c0728c667b1992b7eae262bb30f96fb3a8f744d8160259ede`
- Invocation boundary: `codex exec --approve-for-me --ignore-user-config
  --ignore-rules`; workspace-write sandbox; network disabled

## Prompt

> You are in a fresh Cayu-generated repository. Implement a real
> publish_status tool for the existing agent. The tool must call an injectable
> external status-publisher client with one required string field named status,
> declare the correct external ToolEffect, and be protected by an explicit
> approval policy. Prove with a runtime test that execution pauses before the
> external effect, makes zero client calls before approval, and makes exactly
> one call with the unchanged status after approval and resume. Add a focused
> behavioral eval for the approval interruption. Use only this repository and
> the installed Cayu distribution and its package-shipped guides: do not browse
> the web and do not inspect any Cayu source checkout. First discover and follow
> repository instructions, inspect the baseline, and load the relevant installed
> guides. Preview the generator plan before applying it. Put the client protocol,
> tool, policy/registration, tests, eval, and prompt material in their canonical
> owning modules; keep app.py composition-only except for explicit dependency
> injection and final wiring. Run inspect, strict check, focused/full tests,
> focused/default evals, and git diff checks. Do not commit. Finish with exact
> evidence and state any limit of the exactly-once proof.

## Curated chronological transcript

The transcript below retains every architecture or verification decision. Long
guide bodies and complete JSON manifests are summarized by the fields used.

1. The agent listed the repository, read root `AGENTS.md`, the scaffold contract,
   composition root, registration/policy seams, baseline tests, and default eval.
   It identified `integrations/`, `tools/`, `policies/`, `agents/registration.py`,
   and `evals/` as the owning boundaries.
2. It loaded only installed guidance:

   ```console
   uv run --no-sync cayu guide authoring#cayu-map
   uv run --no-sync cayu guide durable-operations
   uv run --no-sync cayu guide tool-effects
   uv run --no-sync cayu guide references#domain-tool
   uv run --no-sync cayu guide references#approvals
   uv run --no-sync cayu guide references#evals
   ```

   It concluded that an ordinary status publication without a downstream replay
   contract is `ToolEffect.EXTERNAL`, and that effect metadata does not authorize
   execution.
3. It ran baseline inspection and the declared-layout check. Inspect showed no
   tools and the strict check returned zero diagnostics.
4. Before applying anything, it previewed the generator:

   ```console
   uv run --no-sync cayu generate tool publish_status \
     --agent tool-project --effect external --dry-run
   ```

   The plan returned `status: ready`, no conflicts, canonical edits in
   `agents/agent.py` and `agents/registration.py`, and new tool/test/eval files.
   It then applied the same command without `--dry-run`.
5. It replaced the tracer bullet with:

   - an async `StatusPublisher` protocol and fail-closed unconfigured adapter in
     `integrations/status_publisher.py`;
   - `PublishStatusTool` in `tools/publish_status.py`, with a closed one-field
     schema, `parallel_safe=False`, and `ToolEffect.EXTERNAL`;
   - explicit injected construction in `agents/registration.py` and the existing
     `AlwaysRequireApprovalToolPolicy` seam;
   - a composition-only `app.py` dependency-injection parameter;
   - prompt material in `prompts/agent.py`;
   - a runtime pause/approve/resume test and an interruption eval.

   No implementation class, prompt body, policy, or external client was placed
   in `app.py`.
6. The first focused test exposed one missing public-API import in the new test:
   `NameError: PendingActionQuery is not defined`. The agent inspected the test
   import block, added the omitted `PendingActionQuery`, `PendingActionKind`, and
   resolution-actor imports, and reran the same focused proof. This was a bounded
   red/green correction; it did not search Cayu source, change architecture, or
   require human guidance.
7. Final proof was sequential:

   ```text
   uv run --no-sync pytest -q tests/test_publish_status.py
   1 passed in 4.42s

   uv run --no-sync cayu inspect --json
   publish_status: effect=external, parallel_safe=false,
   policy_coverage=approval_required,
   registration=agents.registration:register_agents,
   implementation=tools.publish_status:PublishStatusTool,
   tool_policy=AlwaysRequireApprovalToolPolicy

   uv run --no-sync cayu check --fail-on warning --json
   diagnostics=[]

   uv run --no-sync cayu eval run evals.publish_status:build_eval
   status=passed, score=1, approval-request events=1

   uv run --no-sync pytest
   5 passed in 4.88s

   uv run --no-sync cayu eval run
   status=passed, score=1

   git diff --check
   clean
   ```

The runtime test used the exact value `"  Deploy complete — café ✅  "`. It
asserted an interrupted session and zero client calls, resolved the exact durable
pending approval identity, and then asserted one call containing that unchanged
value.

## Resulting source tree

The complete generated convention remained present. The feature delta was:

```text
tool-project/
  agents/
    agent.py                         modified: finished tool contract
    registration.py                  modified: injected tool + approval policy
  app.py                              modified: dependency injection and wiring only
  evals/
    publish_status.py                 added: interruption/approval-event eval
  integrations/
    status_publisher.py               added: protocol + fail-closed adapter
  prompts/
    agent.py                          modified: tool-use guidance
  tests/
    test_publish_status.py            added: pause/approve/resume proof
  tools/
    publish_status.py                 added: external-effect tool
```

`git status --short` contained exactly those eight paths. The scaffold contract,
ownership homes, and default eval remained intact.

## Evidence boundary

The exactly-one assertion proves this uninterrupted approved test trajectory.
It does not claim generic exactly-once delivery if the external system commits
and the process dies before Cayu records terminal evidence. Production needs a
downstream idempotency or reconciliation contract for that failure window.
