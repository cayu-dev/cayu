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
- [`aws/`](aws/) — Bedrock, Lambda MicroVM, and AWS environment examples.

## Durable orchestration

- [`task_worker_loop.py`](task_worker_loop.py) — durable task claiming and completion.
- [`dispatch_worker.py`](dispatch_worker.py) — dispatcher-owned placement.
- [`workflow_helpers.py`](workflow_helpers.py) — deterministic orchestration helpers.
- [`subagent_live.py`](subagent_live.py) and
  [`subagent_parallel_live.py`](subagent_parallel_live.py) — bounded delegated model work.
- [`session_labels_summary.py`](session_labels_summary.py) — session metadata and summaries.
- [`knowledge_remember_local.py`](knowledge_remember_local.py) — local reviewed knowledge.
- [`knowledge_recall_live.py`](knowledge_recall_live.py) and
  [`knowledge_embedding_live.py`](knowledge_embedding_live.py) — provider-backed retrieval.
- [`postgres_knowledge_embedding.py`](postgres_knowledge_embedding.py) — durable PostgreSQL knowledge.

## Operations and advanced strategies

- [`usage_cost_summary.py`](usage_cost_summary.py) — session usage and cost reporting.
- [`real_spend_budget_live.py`](real_spend_budget_live.py) — live causal budget enforcement.
- [`context_counting_live.py`](context_counting_live.py) and
  [`context_pressure_calibration_live.py`](context_pressure_calibration_live.py) — context limits and calibration.
- [`otel_tracing.py`](otel_tracing.py) — OpenTelemetry runtime events.
- [`dashboard_pending_actions.py`](dashboard_pending_actions.py) and
  [`dashboard_knowledge_review.py`](dashboard_knowledge_review.py) — operator dashboard projections.
- [`business_approval_tiers.py`](business_approval_tiers.py) — application-owned approval routing.
- [`github_pr_reviewer/`](github_pr_reviewer/) — durable cloud PR-review workflow.
- [`ADVANCED_RUNTIME_EXAMPLES.md`](ADVANCED_RUNTIME_EXAMPLES.md) — caching, compaction,
  counterfactual approval, repository tournaments, and taint isolation with explicit
  deterministic and live evidence boundaries.
