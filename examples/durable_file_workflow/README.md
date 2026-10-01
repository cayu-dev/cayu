# Durable file workflow

This credential-free example composes the production seams that smaller
examples introduce separately:

- `InMemoryTaskStore` plus `run_task_worker` for claimed work;
- a fresh `EnvironmentFactory` workspace and runner for every session;
- `WriteFileTool`, `ExecCommandTool`, and `ReadFileTool` behind tool and command
  policies;
- a goal-and-contract prompt rather than a numbered implementation recipe;
- recovery after the first generated program fails; and
- application-owned verification before durable completion.

The run deliberately omits `RunRequest.task_id`. The model-driven session
therefore cannot complete its task merely by reaching `session.completed`.
Trusted worker code reads `result.txt`, checks its exact contract, and only then
calls `complete_task`. Missing source input is held as `blocked` without
starting a session; a verification mismatch is held as `needs_attention`.

Run it from the repository root:

```bash
uv run python -m examples.durable_file_workflow.demo
```

## Contract-verified variant

`verified.py` runs the same transform as a contract-bound task. Instead of
worker code deciding success, the application publishes one immutable
`WorkContract` and a `VerifiedTaskWorker` runs bounded attempts:

1. Attempt 1 writes `result.txt` without its trailing newline. The handler
   proposes that artifact by digest; it never decides the outcome.
2. An independent `DeterministicCompletionVerifier` rejects the proposal with
   the cited gap `artifact.missing_trailing_newline`. It checks the artifact
   against the task record, not against the workspace's `source.txt`, which the
   worker could rewrite; the contract's `source-unmodified` constraint catches
   that. This only protects the expectation while the worker cannot reach the
   task store: tools here run locally without a sandbox, so in SQLite mode a
   worker program could edit the database. Production deployments need a
   sandboxed runner.
3. The contract's `continue` policy starts attempt 2 on the same session. The
   model receives the rejected decision and its gaps and repairs the program,
   and the handler proposes the new artifact.
4. The verifier accepts, a `CompletionResultResolver` rebuilds the task result,
   Cayu applies the decision, the task completes, and the contract binding on
   the session is retired.

The run also drops the acknowledgement for attempt 2's committed proposal. A
fresh `CayuApp` and worker recover the task from durable evidence without
another model call or program run.

```bash
uv run python -m examples.durable_file_workflow.verified
uv run python -m examples.durable_file_workflow.verified --store sqlite
```

Both entry points share `file_worker.py`, the agent and its guarded tools.
Use `demo.py` when trusted application code can check the result directly.
Use `verified.py` when the check, its evidence, and every rejection must be
durable and attributable. `cayu guide verified-work` explains the contract.
