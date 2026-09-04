# Cayu generated applications

`cayu new` compiles a normalized application plan into ordinary explicit
Python. The generated tree is executable architecture for people and coding
agents; it is not a runtime plugin system, service locator, or authority grant.

## Convention

Normal agent, service, and coding presets share these ownership boundaries:

| Concern | Canonical home |
| --- | --- |
| Agent identity and registration | `agents/` |
| Prompt material | `prompts/` |
| Native model-callable capabilities | `tools/` |
| Exposure, authorization, execution, egress, budgets, and retries | `policies/` |
| Workspaces, runners, artifacts, knowledge bindings, and lifecycle | `environments/` |
| Deterministic orchestration | `workflows/` |
| Tasks, workers, approvals, completion, and recovery | `operations/` |
| Reviewed retrieval and curation | `knowledge/` |
| Context, recall, compaction, and memory attribution | `memory/` |
| Business rules | `domain/` |
| External protocols and MCP adapters | `integrations/` |
| Behavioral evidence | `evals/` |
| Event sinks and tracing | `observability/` |
| Final construction and registration only | `app.py` |

Implement in the owning module first, then connect it through
`agents/registration.py` and the composition root. Explicit imports and
`register_*` calls remain authoritative. Do not infer permission from placement,
prompts, scaffold metadata, or capability selection.

`[tool.cayu.scaffold]` records the convention version, preset, adapters,
capabilities, and minimal choice. It is source-controlled creation intent used
by compatible generators and read-only diagnostics. It never selects runtime
objects. Projects without this declaration remain freeform.

For declared convention projects, `cayu check --fail-on warning --json` reports
missing ownership seams, implementation collapsed into `app.py`, top-level
composition work, and agent registration that no longer originates from the
explicit registration module. Removing the contract is an intentional custom
layout migration, not a supported way to silence a finding.

## Planning

Discover the package-shipped catalog before choosing a plan:

```console
uv run --no-sync cayu new --list-presets --json
uv run --no-sync cayu new --list-capabilities --json
uv run --no-sync cayu new --explain knowledge --json
```

Resolve the full plan without writing:

```console
uv run --no-sync cayu new my_agent --preset agent --database sqlite --provider neutral --dry-run --json
```

Dry-run and apply use the same normalized plan. Presets select a coherent
application shape; database, provider, and execution flags select maintained
adapters; `--with` and `--without` change only package-shipped selectable
capabilities. Extension-only concerns retain their canonical homes but cannot be
claimed active through a flag. Invalid combinations fail before target creation.

`NAME` is always the project directory's basename and `--dir` is always its
existing parent. To inspect a maintained variant while changing an existing
project, create a disposable reference instead of passing a path as `NAME` or
running `cayu new` over the current repository:

```console
reference_parent="$(mktemp -d)"
uv run --no-sync cayu new my_agent_reference --agent-name my_agent --preset agent \
  --database postgres --provider neutral --execution none \
  --dir "$reference_parent" --json
```

The reference is comparison material, not an automatic migration. Review the
owning-file diff and update the existing source plus `[tool.cayu.scaffold]`
explicitly. `cayu new`, `cayu check`, and `cayu inspect` never migrate a project.

All presets use the same application convention. Use `--preset` and `--execution`
to select the application and execution environment.

## Generator compatibility

`cayu generate tool` and `cayu generate slice` inspect the declared scaffold
contract. Convention projects update only delimited regions in
`agents/registration.py` and the narrow agent contract. Legacy generated
projects retain their `app.py` seams. A custom or drifted source shape produces
a reviewable conflict or manual action instead of an arbitrary rewrite.

Use this authoring loop:

```text
understand -> inspect -> plan -> change -> test -> eval -> exercise -> report evidence
```

After a change, run the exact commands emitted by `cayu new`; at minimum:

```console
uv run --no-sync cayu inspect --json
uv run --no-sync cayu check --fail-on warning --json
uv run --no-sync pytest
uv run --no-sync cayu eval run
```

Run application-constructing proof commands sequentially when they share the
generated local SQLite store. A first construction may initialize or migrate
that schema.

Runtime sessions, events, checkpoints, tasks, approvals, receipts, knowledge
entries, usage, artifacts, eval results, and snapshots belong in configured
stores or artifact backends. They do not belong in generated source packages.
