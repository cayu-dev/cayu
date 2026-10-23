/**
 * The server contract a Cayu 0.13.0 server returns for a local agent with SQLite stores,
 * captured from `cayu serve --dev`, with pricing turned on because the seeds carry costs.
 * Today's pages gate themselves on its capabilities.
 */
import type { ServerContract } from "@/lib/api"

export const CONTRACT: ServerContract = {
  api_prefix: "/api",
  contract_version: "49",
  versioning: {
    contract_version: "49",
    compatibility: "additive-with-explicit-breaking-review",
    breaking_change_requires: ["openapi_snapshot_update", "client_regeneration", "migration_note"],
  },
  accounting: {
    usage: {
      supported: true,
      method: "GET",
      path: "/api/sessions/{session_id}/usage",
      response_schema: "SessionUsageSummary",
      etag_header: "ETag",
      conditional_request_header: "If-None-Match",
      not_modified_status: 304,
      session_list_include: "usage",
    },
    cost: {
      supported: true,
      method: "POST",
      path: "/api/sessions/{session_id}/cost",
      request_schema: "SessionCostBody",
      response_schema: "SessionCostSummary",
    },
  },
  sse: {
    content_type: "text/event-stream",
    event_id_format: "session_id:cayu_event_<sequence>",
    replay_header: "Last-Event-ID",
    max_event_id_chars: 512,
    mutation_id_header: "Cayu-Mutation-ID",
    mutation_acceptance_event_type: "server.mutation.accepted",
    replay_start_marker_format: "session_id:",
    unknown_event_marker_behavior: "reject",
    event_data_schema: "SseEventEnvelope",
    error_event_name: "error",
    error_data_schema: "SseErrorEnvelope",
    max_event_data_bytes: 2097152,
    max_error_text_bytes: 4096,
    examples: {
      event_data: {
        id: "event_123",
        type: "session.started",
        session_id: "session-123",
        interaction_id: "interaction-123",
        agent_name: "assistant",
        environment_name: "production",
        workflow_name: null,
        tool_name: null,
        payload: {
          status: "running",
        },
        timestamp: "2026-07-06T00:00:00+00:00",
      },
      error_data: {
        type: "stream.error",
        kind: "runtime",
        code: "runtime_failed",
        error: "Runtime stream failed.",
        error_type: "RuntimeError",
        retryable: false,
        session_id: "session-123",
      },
    },
    session_follow: {
      method: "GET",
      path_template: "/api/sessions/{session_id}/events/stream",
      event_id_format: "session_id:cayu_event_<sequence>",
      start_query_param: "after_sequence",
      resume_header: "Last-Event-ID",
      unknown_event_marker_behavior: "reject",
      filter_query_params: ["event_type", "exclude_event_type", "interaction_id"],
      event_data_schema: "SseEventEnvelope",
      heartbeat_interval_seconds: 15.0,
      end_event_name: "end",
      end_data_schema: "SseSessionFollowEndEnvelope",
      terminal_behavior: "end_after_terminal_event",
      idle_timeout_seconds: 300.0,
      max_streams_per_principal: 16,
      max_streams_per_session: 32,
      limit_exceeded_status: 429,
      end_data_example: {
        type: "session.follow.end",
        session_id: "session-123",
        status: "completed",
      },
    },
  },
  client_generation: {
    openapi_url: "/openapi.json",
    supported_targets: ["typescript", "python"],
    source_of_truth: "openapi",
  },
  client: {
    module_url: "/client.js",
    types_url: "/client.d.ts",
    version: "1",
    guide_topic: "app-ui",
  },
  capabilities: {
    cayu_version: "0.13.0",
    configured_store_roles: ["session", "task", "knowledge", "artifact"],
    actor: null,
    surfaces: {
      dashboard: {
        configured: true,
        read: {
          enabled: true,
          unavailable_reason: null,
        },
        mutate: {
          enabled: false,
          unavailable_reason: "unsupported",
        },
      },
      workflow: {
        configured: true,
        read: {
          enabled: true,
          unavailable_reason: null,
        },
        mutate: {
          enabled: false,
          unavailable_reason: "unsupported",
        },
      },
      browser_recordings: {
        configured: false,
        read: {
          enabled: false,
          unavailable_reason: "not_configured",
        },
        mutate: {
          enabled: false,
          unavailable_reason: "unsupported",
        },
      },
      session_follow: {
        configured: true,
        read: {
          enabled: true,
          unavailable_reason: null,
        },
        mutate: {
          enabled: false,
          unavailable_reason: "unsupported",
        },
      },
      supervision: {
        configured: true,
        read: {
          enabled: true,
          unavailable_reason: null,
        },
        mutate: {
          enabled: false,
          unavailable_reason: "unsupported",
        },
      },
      inbox: {
        configured: true,
        read: {
          enabled: true,
          unavailable_reason: null,
        },
        mutate: {
          enabled: true,
          unavailable_reason: null,
        },
      },
      tasks: {
        configured: true,
        read: {
          enabled: true,
          unavailable_reason: null,
        },
        mutate: {
          enabled: true,
          unavailable_reason: null,
        },
      },
      reviewed_knowledge: {
        configured: true,
        read: {
          enabled: true,
          unavailable_reason: null,
        },
        mutate: {
          enabled: true,
          unavailable_reason: null,
        },
      },
      artifacts: {
        configured: true,
        read: {
          enabled: true,
          unavailable_reason: null,
        },
        mutate: {
          enabled: false,
          unavailable_reason: "unsupported",
        },
      },
      usage: {
        configured: true,
        read: {
          enabled: true,
          unavailable_reason: null,
        },
        mutate: {
          enabled: false,
          unavailable_reason: "unsupported",
        },
      },
      pricing: {
        configured: true,
        read: {
          enabled: true,
          unavailable_reason: null,
        },
        mutate: {
          enabled: false,
          unavailable_reason: "unsupported",
        },
      },
      evaluation_promotion: {
        configured: false,
        read: {
          enabled: false,
          unavailable_reason: "not_configured",
        },
        mutate: {
          enabled: false,
          unavailable_reason: "not_configured",
        },
      },
      evals: {
        configured: true,
        read: {
          enabled: true,
          unavailable_reason: null,
        },
        mutate: {
          enabled: true,
          unavailable_reason: null,
        },
      },
    },
    mutations: {
      session_execution: {
        enabled: true,
        unavailable_reason: null,
      },
      session_interruption: {
        enabled: true,
        unavailable_reason: null,
      },
      provider_operation_resolution: {
        enabled: true,
        unavailable_reason: null,
      },
      pending_action_resolution: {
        enabled: true,
        unavailable_reason: null,
      },
      session_annotations: {
        enabled: true,
        unavailable_reason: null,
      },
      task_lifecycle: {
        enabled: true,
        unavailable_reason: null,
      },
      knowledge_review: {
        enabled: true,
        unavailable_reason: null,
      },
    },
    evals_readiness: {
      captured_evaluation: {
        state: "ready",
        reason_code: null,
      },
      catalog_read: {
        state: "ready",
        reason_code: null,
      },
      catalog_write: {
        state: "ready",
        reason_code: null,
      },
      captured_result_persistence: {
        state: "ready",
        reason_code: null,
      },
      scenario_conversion: {
        state: "ready",
        reason_code: null,
      },
      fresh_launch: {
        state: "ready",
        reason_code: null,
      },
      cancellation: {
        state: "ready",
        reason_code: null,
      },
      comparison: {
        state: "ready",
        reason_code: null,
      },
      reports: {
        state: "ready",
        reason_code: null,
      },
    },
  },
}
