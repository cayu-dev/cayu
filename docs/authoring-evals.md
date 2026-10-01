# Authoring evals

`scripts/authoring_evals.py` measures whether coding agents (Copilot CLI, Claude
Code, Codex) use Cayu's features when building on a generated project, or
rebuild them. Changes to the generated `AGENTS.md`, the scaffold, guides and
`cayu check` hints should be judged by these results, not by how the wording
reads.

## Tasks

Each task is a product request with no Cayu vocabulary, and a grader for the
feature it should lead to:

| Task | Request (abridged) | Expected feature | Reinvention signals |
| --- | --- | --- | --- |
| `chat` | Follow-up questions that remember the conversation | `ResumeRequest` / `app.resume` | A new `RunRequest` per message |
| `chat-casual` | "Can we add a chat to control the agent?" (a real user's wording) | `ResumeRequest` / `app.resume` | A new `RunRequest` per message |
| `approval` | A person confirms every notification before it is sent | Tool with `approval_required` coverage | Approval code outside the policy seam |
| `structured-output` | Findings as data Python code can use | `StructuredOutputSpec` | JSON parsed out of model text |
| `live-ui` | A page showing the agent's progress live | `client.js` session following | `fetch` on a timer |

Graders read application source only. Tests and evals can mention a feature
without the application using it, so they don't count.

## Running

```bash
python scripts/authoring_evals.py list
python scripts/authoring_evals.py run --agent copilot --task chat --out results/
python scripts/authoring_evals.py run --agent-command '["my-agent", "--yes", "{prompt}"]' --out results/
python scripts/authoring_evals.py grade --task approval --verify path/to/project
```

`run` scaffolds a fresh project against `--cayu-source` (default: this checkout),
pins the project to that source, runs the agent in the project directory, grades
it, and records `cayu check` errors and the project's test result in
`results/<task>.json` and `results/summary.md`. Use `--keep` to inspect the
generated project afterwards.

Runs call a live agent and model, so they cost money and are not part of CI.
Agents run with their permission prompts disabled inside a throwaway temporary
project; do not point `--cayu-source` or the agent at a directory you care about.
`grade` only reads files, plus `cayu inspect` for the approval task.

## Comparing a change

Run the same tasks against two checkouts and compare the summaries:

```bash
python scripts/authoring_evals.py run --task chat --cayu-source ../cayu-main --out results/main
python scripts/authoring_evals.py run --task chat --cayu-source . --out results/branch
```

One run per task is a signal, not a measurement; repeat runs before drawing a
conclusion from a single difference.

## Adding a task

Add a `Task` to `TASKS` with a request phrased the way a user would ask, the
expected feature, and a grader that returns a `Grade`. Prefer graders that check
concrete evidence (an API call, a manifest field) over keyword counts, and add a
hermetic case to `tests/core/test_authoring_evals.py`.
