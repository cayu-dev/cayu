# Canonical application example

Create this example with `cayu new`. The maintained generator produces the
application structure used by Cayu's authoring guides, CLI inspection, tests,
and evals. Generating it keeps the example aligned with your installed Cayu
version.

## Generate and verify

Use Python 3.11 or newer and `uv`. Run these commands from a directory where you
want to create a new `myagent` project:

```sh
uvx --from cayu cayu new myagent
cd myagent
uv sync --extra dev
uv run --no-sync cayu guide anatomy
uv run --no-sync cayu inspect --json
uv run --no-sync cayu check --fail-on warning --json
uv run --no-sync pytest
uv run --no-sync cayu eval run
```

The first command runs `cayu new` through `uv` without requiring a global Cayu
installation. If Cayu is already installed, use `cayu new myagent` instead.

No model credentials are needed for these checks. The generated tests and eval
inject a deterministic provider through the application's public factory seam.
They exercise the real session runtime and assert the starter's output; they do
not establish live-model quality for your eventual task.

## Read the generated application

Open the generated `README.md` and `AGENTS.md` for the full layout and capability
configuration. Start with these files:

| Generated path | Responsibility |
| --- | --- |
| `pyproject.toml` | Declares `[tool.cayu] factory = "app:build_app"` and the scaffold plan. |
| `app.py` | Composes a fresh process-scoped application through `build_app()`. |
| `configuration/` | Owns provider selection, settings, storage, and runtime configuration. |
| `agents/agent.py` | Declares the first agent, its model, and prompt composition. |
| `prompts/agent.py` | Owns the agent's system prompt material. |
| `agents/registration.py` | Registers the agent and connects its tools and policies. |
| `tools/` | Owns tool implementations and their registration seam. |
| `tests/test_agent.py` | Proves the agent's behavior with injected test dependencies. |
| `evals/agent.py` | Defines the deterministic output eval. |
| `run.py` | Runs a request against the configured application. |

The default scaffold also supplies homes and explicit wiring for capabilities
such as knowledge, memory, tasks, approvals, and observability. Follow the
generated capability guidance as you extend the project. Keep `app.py` focused
on construction and registration; place behavior in its owning modules.

## Implement the first job

Edit the existing agent in `agents/agent.py` and its prompt material in
`prompts/agent.py`. Update its test and eval to express the requested behavior.
Configure a live provider through
`configuration/settings.py` or `CAYU_PROVIDER`, following
`uv run --no-sync cayu guide providers` for credentials and model selection.
Then run:

```sh
uv run --no-sync python run.py --message "Review this change."
```

Re-run the verification commands after changing the application. For the
construction and lifecycle contract, use `cayu guide anatomy` or read
[application anatomy](../../src/cayu/guides/application-anatomy.md). For the
public API ownership map, see [public concepts](../../docs/public-concepts.md).
The separate [public-import demonstration](../public_concepts/README.md) is a
small script for exploring those imports.
