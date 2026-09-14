# Public concepts and source ownership

Import Cayu capabilities by the concept you are working with. The public
packages own their implementations; the former flat and core/runtime module
paths have been removed in this prerelease migration.

To build an application, start with the [canonical application
example](../examples/application/README.md), which uses `cayu new` and the generated
project structure. See [application anatomy](../src/cayu/guides/application-anatomy.md)
for the factory and lifecycle contract.

For a focused demonstration of the public imports below, see the
[public-import demonstration](../examples/public_concepts/README.md):

```python
from cayu.applications import CayuApp
from cayu.agents import AgentSpec
from cayu.messages import Message
from cayu.events import Event, EventType
from cayu.context import DefaultContextPolicy
from cayu.sessions import InMemorySessionStore, RunRequest
from cayu.providers import ModelStreamEvent
from cayu.evals import ScriptedModelProvider
```

The import demonstration runs a real session locally with a deterministic
provider. It needs no provider credentials. Run it from the repository root with
`uv run python -m examples.public_concepts.app`. Its single-file layout illustrates
API imports; use the generated project above as the application authoring example.

## Source map

```text
src/cayu/
    applications.py             # Application composition and execution entry point
    agents.py                   # Agent definitions
    messages.py                 # Messages and content parts
    events.py                   # Events and event identities
    configuration.py            # Application defaults and configuration
    exceptions.py               # Public runtime failure contracts
    tools/                      # Contracts, policy, discovery, execution grants, built-ins
    sessions/                   # Requests, stores, checkpoints, recovery, child sessions
    tasks/                      # Durable tasks, workers, dispatch, work contracts
    workflows/                  # Workflow contracts and authoring
    context/                    # Context policies, counting, compaction, structured output
    memory/                     # Recall, relevance, attribution, evidence, interventions
    knowledge/                  # Curation, enrichment, governance, maintenance
    approvals/                  # Tool approvals, business approvals, review, user input
    budgets/                    # Budgets, billing, pricing, usage, aggregates
    snapshots/                  # Snapshot capture, bundles, container transport
    delivery/                   # Git and GitHub delivery, process helpers
    providers/                  # Model-provider contracts and adapters
    storage/                    # Storage adapters
    environments/               # Environment contracts and adapters
    runners/                    # Execution runners
    workspaces/                 # Workspace adapters, branching, checkpoint lifecycle
    artifacts/                  # Artifact contracts and adapters
    egress/                     # Network authority and runtime transitions
    vaults/                     # Secret storage
    proxies/                    # Proxy support
    mcp/                        # MCP integration
    webhooks/                   # Webhook support
    observability/              # Event sinks, watchers, hooks, instrumentation
    evals/                      # Evaluation and scripted provider support
    testing/                    # Testing helpers and isolated worker fixtures
    server/                     # HTTP control plane
    cli/                        # Command-line interface
    runtime/                    # Execution coordination
```

Inside a capability, `base.py` owns its principal contracts and implementation.
For example, `sessions/base.py` owns `RunRequest` and `SessionStore`,
`tools/base.py` owns `Tool`, and `snapshots/base.py` owns `AgentSnapshot`.
More specific modules own the rest: `tasks/worker.py`, `budgets/pricing.py`,
`approvals/review.py`, `memory/relevance.py`, and `delivery/github.py`.

The workspace branch/checkpoint contracts retain `workspaces/branches.py` and
`workspaces/checkpoints.py`. Their application lifecycle implementations live
separately in `workspaces/branch_lifecycle.py` and `workspaces/checkpoint_lifecycle.py`.

## Retained root-level concepts

A concept with one principal module can remain directly under `cayu`; concept
ownership does not require a directory for every file. In addition to the
application, agent, message, event, configuration and exception entry points
shown above, these named modules intentionally remain at the root:

| Module | Responsibility |
| --- | --- |
| `browser_profiles.py` | Application-owned encrypted browser profile state |
| `browser_recording.py` | Browser recording contracts and storage |
| `build_provenance.py` | Installed distribution and build identity |
| `capabilities.py` | Capability discovery contracts |
| `coding_products.py` | Maintained repository-coding product contracts |
| `credentials.py` | Credential authority contracts |
| `deadlines.py` | Shared execution deadline contracts |
| `embeddings.py` | Embedding contracts |
| `entrypoint.py` | Project application entrypoint loading |
| `failure_evidence.py` | Shared typed execution failure evidence |
| `immutable_inputs.py` | Immutable input capture and authority |
| `project_control_plane.py` | Project control-plane composition |
| `support_bundles.py` | Diagnostic support bundle contracts |
| `work_context.py` | Durable application work-context contracts |

Private root helpers serve shared concerns such as validation, locking, clock,
exception state and cross-cutting authority. The CLI entry point and generated
lazy-export support also stay at the root. These are intentional ownership
boundaries, not unfinished compatibility aliases. Generated applications may
use either supported root exports or concept packages; their imports resolve to
the same canonical implementations.

## Public exports

Twenty-five package initializers use explicit lazy exports. Each package's
`_exports.py` maps a public name to its defining module and lists its `__all__`
entries. The paired `__init__.pyi` declares the same names for static tools. The package
ships a `py.typed` marker so installed-wheel consumers discover these declarations.
The initializer loads and caches only the requested object. Use `dir()` or
`__all__` to discover exports; `vars()` contains only exports already loaded.
There is no filesystem scanning or registration by directory name.

The root `cayu` API provides convenient public exports.
Common imports such as `from cayu import CayuApp, AgentSpec` still work. For
specialized capabilities, prefer the concept package. Importing agent/message
contracts no longer initializes the application runtime, evals, or optional
provider/server adapters.

When adding a public capability:

1. Put its implementation in the owning package and use canonical imports.
2. Add its export to that package's `_exports.py` and `__init__.pyi`.
3. Add it to `PUBLIC_NAMES` when it should be included in `__all__`.
4. Run the public API migration tests to check runtime/static export agreement.

New implementation code must not import legacy aliases. The architecture test
checks this boundary, and the export maps point directly to canonical modules.
Packaged-guide checks reject stale module paths. Release validation requires every
canonical implementation and lazy-export declaration and rejects removed aliases
in both wheels and source distributions. The installed-wheel smoke runs in the
release core lane, independently of the source checkout.

## Prerelease migration boundaries

Ninety-four former module paths map to the locations below. The 92 alias
modules and the former `cayu.core` facade have been removed. `cayu.memory` and
`cayu.testing` are now concept packages with public exports. Import definitions
from their canonical modules or concept packages; executable helpers also use
the relocated entry points. Type declarations describe the current public APIs.

Durable schema names and hashing domain identifiers retain their original
values. Python class `__module__` values and source locations change. Old module
paths and historical pickle references are not supported. Build and execution
profile checks continue to govern recovery; validation uses fresh installations
and same-build restart/recovery, without cross-version upgrade guarantees.

This migration changes ownership and imports. It does not decompose `CayuApp`
or change session algorithms.

## Complete module migration

The machine-readable map is [public-api-migration.json](public-api-migration.json).
The public package list is [public-api-packages.json](public-api-packages.json).

| Former import | Canonical implementation |
| --- | --- |
| `cayu._knowledge_publication_owner` | [`cayu.knowledge._publication`](../src/cayu/knowledge/_publication.py) |
| `cayu._remote_git_cleanup` | [`cayu.delivery._git_cleanup`](../src/cayu/delivery/_git_cleanup.py) |
| `cayu._remote_git_configuration` | [`cayu.delivery._git_configuration`](../src/cayu/delivery/_git_configuration.py) |
| `cayu._remote_git_ownership` | [`cayu.delivery._git_ownership`](../src/cayu/delivery/_git_ownership.py) |
| `cayu._remote_git_process` | [`cayu.delivery._git_process`](../src/cayu/delivery/_git_process.py) |
| `cayu.agent_bundle_containers` | [`cayu.snapshots.containers`](../src/cayu/snapshots/containers.py) |
| `cayu.agent_bundles` | [`cayu.snapshots.bundles`](../src/cayu/snapshots/bundles.py) |
| `cayu.agent_snapshots` | [`cayu.snapshots.base`](../src/cayu/snapshots/base.py) |
| `cayu.core.agents` | [`cayu.agents`](../src/cayu/agents.py) |
| `cayu.core.billing` | [`cayu.budgets.billing`](../src/cayu/budgets/billing.py) |
| `cayu.core.events` | [`cayu.events`](../src/cayu/events.py) |
| `cayu.core.execution_identity` | [`cayu.runtime.execution_identity`](../src/cayu/runtime/execution_identity.py) |
| `cayu.core.isolated_tools` | [`cayu.tools.isolated`](../src/cayu/tools/isolated.py) |
| `cayu.core.messages` | [`cayu.messages`](../src/cayu/messages.py) |
| `cayu.core.runtime_authority` | [`cayu.runtime.authority`](../src/cayu/runtime/authority.py) |
| `cayu.core.thinking` | [`cayu.context.thinking`](../src/cayu/context/thinking.py) |
| `cayu.core.tools` | [`cayu.tools.base`](../src/cayu/tools/base.py) |
| `cayu.core.workflows` | [`cayu.workflows.base`](../src/cayu/workflows/base.py) |
| `cayu.github_delivery` | [`cayu.delivery.github`](../src/cayu/delivery/github.py) |
| `cayu.knowledge_curator` | [`cayu.knowledge.curator`](../src/cayu/knowledge/curator.py) |
| `cayu.knowledge_enrichment` | [`cayu.knowledge.enrichment`](../src/cayu/knowledge/enrichment.py) |
| `cayu.knowledge_governance` | [`cayu.knowledge.governance`](../src/cayu/knowledge/governance.py) |
| `cayu.knowledge_maintenance` | [`cayu.knowledge.maintenance`](../src/cayu/knowledge/maintenance.py) |
| `cayu.knowledge_maintenance_governance` | [`cayu.knowledge.maintenance_governance`](../src/cayu/knowledge/maintenance_governance.py) |
| `cayu.knowledge_maintenance_persistence` | [`cayu.knowledge.maintenance_persistence`](../src/cayu/knowledge/maintenance_persistence.py) |
| `cayu.knowledge_maintenance_planning` | [`cayu.knowledge.maintenance_planning`](../src/cayu/knowledge/maintenance_planning.py) |
| `cayu.knowledge_semantic_watch` | [`cayu.knowledge.semantic_watch`](../src/cayu/knowledge/semantic_watch.py) |
| `cayu.memory` | [`cayu.memory.base`](../src/cayu/memory/base.py) |
| `cayu.memory_attribution` | [`cayu.memory.attribution`](../src/cayu/memory/attribution.py) |
| `cayu.memory_evidence` | [`cayu.memory.evidence`](../src/cayu/memory/evidence.py) |
| `cayu.memory_intervention_execution` | [`cayu.memory.execution`](../src/cayu/memory/execution.py) |
| `cayu.memory_interventions` | [`cayu.memory.interventions`](../src/cayu/memory/interventions.py) |
| `cayu.recall` | [`cayu.memory.recall`](../src/cayu/memory/recall.py) |
| `cayu.recall_processing` | [`cayu.memory.processing`](../src/cayu/memory/processing.py) |
| `cayu.recall_relevance` | [`cayu.memory.relevance`](../src/cayu/memory/relevance.py) |
| `cayu.remote_git_delivery` | [`cayu.delivery.git`](../src/cayu/delivery/git.py) |
| `cayu.retrieval` | [`cayu.memory.retrieval`](../src/cayu/memory/retrieval.py) |
| `cayu.runtime.aggregates` | [`cayu.budgets.aggregates`](../src/cayu/budgets/aggregates.py) |
| `cayu.runtime.app` | [`cayu.applications`](../src/cayu/applications.py) |
| `cayu.runtime.approvals` | [`cayu.approvals.tools`](../src/cayu/approvals/tools.py) |
| `cayu.runtime.browser_control` | [`cayu.tools.browser_control`](../src/cayu/tools/browser_control.py) |
| `cayu.runtime.browser_control_config` | [`cayu.tools.browser_control_config`](../src/cayu/tools/browser_control_config.py) |
| `cayu.runtime.budgets` | [`cayu.budgets.base`](../src/cayu/budgets/base.py) |
| `cayu.runtime.business_approvals` | [`cayu.approvals.business`](../src/cayu/approvals/business.py) |
| `cayu.runtime.checkpoints` | [`cayu.sessions.checkpoints`](../src/cayu/sessions/checkpoints.py) |
| `cayu.runtime.child_session_context` | [`cayu.sessions.child_context`](../src/cayu/sessions/child_context.py) |
| `cayu.runtime.child_session_results` | [`cayu.sessions.child_results`](../src/cayu/sessions/child_results.py) |
| `cayu.runtime.config` | [`cayu.configuration`](../src/cayu/configuration.py) |
| `cayu.runtime.context` | [`cayu.context.base`](../src/cayu/context/base.py) |
| `cayu.runtime.context_counting` | [`cayu.context.counting`](../src/cayu/context/counting.py) |
| `cayu.runtime.cost_quality` | [`cayu.budgets.quality`](../src/cayu/budgets/quality.py) |
| `cayu.runtime.costs` | [`cayu.budgets.pricing`](../src/cayu/budgets/pricing.py) |
| `cayu.runtime.dispatch` | [`cayu.tasks.dispatch`](../src/cayu/tasks/dispatch.py) |
| `cayu.runtime.egress` | [`cayu.egress.runtime`](../src/cayu/egress/runtime.py) |
| `cayu.runtime.egress_authority_transitions` | [`cayu.egress.transitions`](../src/cayu/egress/transitions.py) |
| `cayu.runtime.errors` | [`cayu.exceptions`](../src/cayu/exceptions.py) |
| `cayu.runtime.event_sinks` | [`cayu.observability.events`](../src/cayu/observability/events.py) |
| `cayu.runtime.event_watchers` | [`cayu.observability.watchers`](../src/cayu/observability/watchers.py) |
| `cayu.runtime.exports` | [`cayu.sessions.exports`](../src/cayu/sessions/exports.py) |
| `cayu.runtime.hooks` | [`cayu.observability.hooks`](../src/cayu/observability/hooks.py) |
| `cayu.runtime.human_review` | [`cayu.approvals.review`](../src/cayu/approvals/review.py) |
| `cayu.runtime.interactions` | [`cayu.sessions.interactions`](../src/cayu/sessions/interactions.py) |
| `cayu.runtime.invocation` | [`cayu.sessions.invocation`](../src/cayu/sessions/invocation.py) |
| `cayu.runtime.memory_context` | [`cayu.memory.context`](../src/cayu/memory/context.py) |
| `cayu.runtime.outcomes` | [`cayu.sessions.outcomes`](../src/cayu/sessions/outcomes.py) |
| `cayu.runtime.pending_actions` | [`cayu.sessions.pending_actions`](../src/cayu/sessions/pending_actions.py) |
| `cayu.runtime.recovery_cleanup` | [`cayu.sessions.cleanup`](../src/cayu/sessions/cleanup.py) |
| `cayu.runtime.recovery_plans` | [`cayu.sessions.recovery`](../src/cayu/sessions/recovery.py) |
| `cayu.runtime.request_footprints` | [`cayu.context.footprints`](../src/cayu/context/footprints.py) |
| `cayu.runtime.sessions` | [`cayu.sessions.base`](../src/cayu/sessions/base.py) |
| `cayu.runtime.structured_output` | [`cayu.context.structured_output`](../src/cayu/context/structured_output.py) |
| `cayu.runtime.targeted_tool_projection` | [`cayu.tools.targeted_projection`](../src/cayu/tools/targeted_projection.py) |
| `cayu.runtime.task_worker` | [`cayu.tasks.worker`](../src/cayu/tasks/worker.py) |
| `cayu.runtime.tasks` | [`cayu.tasks.base`](../src/cayu/tasks/base.py) |
| `cayu.runtime.tool_catalogue` | [`cayu.tools.catalogue`](../src/cayu/tools/catalogue.py) |
| `cayu.runtime.tool_discovery` | [`cayu.tools.discovery`](../src/cayu/tools/discovery.py) |
| `cayu.runtime.tool_exposure` | [`cayu.tools.exposure`](../src/cayu/tools/exposure.py) |
| `cayu.runtime.tool_gateway` | [`cayu.tools.gateway`](../src/cayu/tools/gateway.py) |
| `cayu.runtime.tool_grants` | [`cayu.tools.grants`](../src/cayu/tools/grants.py) |
| `cayu.runtime.tool_policy` | [`cayu.tools.policy`](../src/cayu/tools/policy.py) |
| `cayu.runtime.tool_result_projection` | [`cayu.tools.result_projection`](../src/cayu/tools/result_projection.py) |
| `cayu.runtime.tool_rounds` | [`cayu.tools.rounds`](../src/cayu/tools/rounds.py) |
| `cayu.runtime.tool_terminal_publication` | [`cayu.tools.terminal_publication`](../src/cayu/tools/terminal_publication.py) |
| `cayu.runtime.usage` | [`cayu.budgets.usage`](../src/cayu/budgets/usage.py) |
| `cayu.runtime.user_input` | [`cayu.approvals.user_input`](../src/cayu/approvals/user_input.py) |
| `cayu.runtime.work_attempt_admission` | [`cayu.tasks.admission`](../src/cayu/tasks/admission.py) |
| `cayu.runtime.work_contracts` | [`cayu.tasks.contracts`](../src/cayu/tasks/contracts.py) |
| `cayu.runtime.workspace_branches` | [`cayu.workspaces.branch_lifecycle`](../src/cayu/workspaces/branch_lifecycle.py) |
| `cayu.runtime.workspace_checkpoints` | [`cayu.workspaces.checkpoint_lifecycle`](../src/cayu/workspaces/checkpoint_lifecycle.py) |
| `cayu.runtime.workspace_mutation_attribution` | [`cayu.workspaces.mutation_attribution`](../src/cayu/workspaces/mutation_attribution.py) |
| `cayu.runtime.workspace_observation_recovery` | [`cayu.workspaces.observation_recovery`](../src/cayu/workspaces/observation_recovery.py) |
| `cayu.testing` | [`cayu.testing.base`](../src/cayu/testing/base.py) |
| `cayu.testing_isolated_tools` | [`cayu.testing.isolated_tools`](../src/cayu/testing/isolated_tools.py) |
| `cayu.testing_isolated_worker_faults` | [`cayu.testing.isolated_worker_faults`](../src/cayu/testing/isolated_worker_faults.py) |

## Local checks

```sh
uv run ruff check src/ tests/ examples/ scripts/ maintenance/
uv run ruff format --check src/ tests/ examples/ scripts/ maintenance/
uv run ty check src/cayu examples maintenance
uv run pytest tests/test_public_api_migration.py tests/test_public_api_surface.py
uv run python -m examples.public_concepts.app
```

The migration tests check canonical module/class identity, every declared public
export, static declaration agreement, isolated
contract imports, and absence of old import paths in implementation code.
