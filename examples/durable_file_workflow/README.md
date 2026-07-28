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
uv run python examples/durable_file_workflow/demo.py
```
