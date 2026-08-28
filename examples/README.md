# Cayu examples

Start with the smallest example that matches the capability you need. Examples
are references, not a checklist or a required project structure. Files ending
in `_live.py` cross a real provider or infrastructure boundary and require the
corresponding credentials or service.

## Start here

- A model-only project: run `cayu new NAME`; the generated test and eval are
  credential-free.
- [`echo_tool_runtime.py`](echo_tool_runtime.py) — credential-free model/tool loop.
- [`local_environment_runtime.py`](local_environment_runtime.py) — local files and
  commands through an explicit environment.
- [`structured_output_live.py`](structured_output_live.py) — provider-native typed
  output.
- [`evals_judge_calibration_live.py`](evals_judge_calibration_live.py) — a real,
  explicitly configured judge calibrating fixed evidence without candidate execution.
- [`server_example.py`](server_example.py) — authenticated control-plane application.

## Tools and providers

- [`openai_local_tools.py`](openai_local_tools.py) — OpenAI Responses with local tools.
- [`anthropic_local_tools.py`](anthropic_local_tools.py) — Anthropic Messages with local tools.
- [`vertex_local_tools.py`](vertex_local_tools.py) — Anthropic models through Vertex AI.
- [`chat_completions_local_tools.py`](chat_completions_local_tools.py) — OpenAI-compatible chat completions.
- [`thinking.py`](thinking.py) — provider-neutral thinking configuration.
- [`stdio_mcp_runtime.py`](stdio_mcp_runtime.py) — an MCP server through Cayu's tool contract.
- [`custom_runner_tool.py`](custom_runner_tool.py) — a custom tool using the active runner.
- [`credential_proxy_tool.py`](credential_proxy_tool.py) — scoped credentials at a tool boundary.
- [`webbridge/`](webbridge/) — explicit local, hosted, and sandboxed web profiles,
  canonical browse/extract/verify evidence, and external-cron task execution.

## Execution environments

- [`sync_binding_local.py`](sync_binding_local.py) — local workspace synchronization.
- [`docker_interrupt_live.py`](docker_interrupt_live.py) — Docker interruption behavior.
- [`e2b_runner_live.py`](e2b_runner_live.py) and
  [`e2b_workspace_live.py`](e2b_workspace_live.py) — E2B execution and workspaces.
- [`microsandbox_runner_live.py`](microsandbox_runner_live.py) and
  [`microsandbox_workspace_live.py`](microsandbox_workspace_live.py) — local microVM execution.
- [`modal_runner.py`](modal_runner.py) — an application-owned remote runner.
- [`artifact_file_live.py`](artifact_file_live.py) and
  [`artifact_workspace_bridge.py`](artifact_workspace_bridge.py) — durable files and mutable workspaces.
- [`fastapi_stripe_virtual_egress.py`](fastapi_stripe_virtual_egress.py) — virtual credentials and restricted egress.
- [`github_cli_virtual_egress.py`](github_cli_virtual_egress.py) — an unmodified CLI with a brokered token and restricted egress.
- [`aws/`](aws/) — Bedrock, Lambda MicroVM, and AWS environment examples.

## Durable orchestration

- [`durable_file_workflow/`](durable_file_workflow/) — hermetic task worker with
  per-session files and commands, failure recovery, and app-verified outcomes.
- [`task_worker_loop.py`](task_worker_loop.py) — durable task claiming and completion.
- [`task_retry_worker.py`](task_retry_worker.py) — cumulative retry limits across fresh worker processes.
- [`dispatch_worker.py`](dispatch_worker.py) — dispatcher-owned placement.
- [`workflow_helpers.py`](workflow_helpers.py) — deterministic orchestration helpers.
- [`subagent_live.py`](subagent_live.py) and
  [`subagent_parallel_live.py`](subagent_parallel_live.py) — bounded delegated model work.
- [`session_labels_summary.py`](session_labels_summary.py) — session metadata and summaries.
- [`knowledge_remember_local.py`](knowledge_remember_local.py) — local reviewed knowledge.
- [`reviewed_knowledge_curator.py`](reviewed_knowledge_curator.py) — credential-free,
  explicitly invoked curation from completed-run evidence through pending review and
  later recall.
- [`forked_session_knowledge.py`](forked_session_knowledge.py) — one targeted-only
  memory child running concurrently with its continuing parent, with no provider
  credentials.
- [`knowledge_recall_live.py`](knowledge_recall_live.py) and
  [`knowledge_embedding_live.py`](knowledge_embedding_live.py) — provider-backed retrieval.
- [`cross_source_recall.py`](cross_source_recall.py) — credential-free bounded
  knowledge/transcript recall with calibrated focus/offer/silent admission.
- [`agent_snapshot_stateful_evaluation.py`](agent_snapshot_stateful_evaluation.py) —
  API-key-free capture, isolated candidate overlays, exact result lineage, and
  fresh-process recovery from one portable agent snapshot.
- [`postgres_knowledge_embedding.py`](postgres_knowledge_embedding.py) — durable PostgreSQL knowledge.

## Operations and advanced strategies

- [`usage_cost_summary.py`](usage_cost_summary.py) — session usage and cost reporting.
- [`cost_quality_comparison.py`](cost_quality_comparison.py) — deterministic paired
  accounting with `verified`, `measured_unmatched`, `unpriced`, and `unavailable`
  proof statuses.
- [`tool_discovery_validation/`](tool_discovery_validation/) — credential-free
  discovery, resume, fork reset, copied-reference rejection, ranking, and bounded economics
  evidence through the real `search_tools` / `call_tool` path.
- [`openai_client_tool_search/`](openai_client_tool_search/) — credential-free
  native `tool_search_call` / `tool_search_output` projection through the real
  OpenAI adapter and ordinary Cayu tool executor.
- [`openai_hosted_tool_search/`](openai_hosted_tool_search/) — credential-free
  server-executed Tool Search selection, atomic branch-local grant publication,
  replay, and ordinary tool execution through the real OpenAI adapter.
- [`real_spend_budget_live.py`](real_spend_budget_live.py) — live causal budget enforcement.
- [`context_counting_live.py`](context_counting_live.py) and
  [`context_pressure_calibration_live.py`](context_pressure_calibration_live.py) — context limits and calibration.
- [`otel_tracing.py`](otel_tracing.py) — OpenTelemetry runtime events.
- [`dashboard_pending_actions.py`](dashboard_pending_actions.py) and
  [`dashboard_knowledge_review.py`](dashboard_knowledge_review.py) — operator dashboard projections.
- [`business_approval_tiers.py`](business_approval_tiers.py) — application-owned approval routing.
- [`github_pr_reviewer/`](github_pr_reviewer/) — durable cloud PR-review workflow.
- [`ADVANCED_RUNTIME_EXAMPLES.md`](ADVANCED_RUNTIME_EXAMPLES.md) — bounded fork groups,
  caching, compaction, counterfactual approval, repository tournaments, and taint
  isolation with explicit deterministic and live evidence boundaries.
