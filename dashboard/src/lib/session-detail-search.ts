import {
  type SessionHistorySearch,
  validateSessionHistorySearch,
} from "./session-history-search.ts"

import { validateWorkflowSearch, type WorkflowSearch } from "./workflow-search.ts"

export type SessionDetailSearch = SessionHistorySearch &
  Omit<WorkflowSearch, "agent_name" | "environment_name"> & {
    workflow_agent_name?: string
    workflow_environment_name?: string
    session_instance_id?: string
    panel?: string
    view?: string
    workflow?: "1"
    span?: string
    event?: string
    log?: string
  }

const text = (value: unknown) =>
  typeof value === "string" && value.trim() ? value.trim() : undefined

/** The conversation shell retains the existing technical history deep links. */
export function validateSessionDetailSearch(search: Record<string, unknown>): SessionDetailSearch {
  const history = validateSessionHistorySearch(search)
  const event = text(search.event) ?? history.event_id
  const panel = text(search.panel)
  return {
    ...history,
    ...workflowSearchPatch(workflowSearchFromSession(search)),
    session_instance_id: text(search.session_instance_id),
    panel: ["overview", "usage", "artifacts", "browser"].includes(panel ?? "") ? panel : undefined,
    view:
      search.view === "conversation"
        ? "conversation"
        : search.view === "trace" || history.event_id
          ? "trace"
          : undefined,
    workflow: search.workflow === "1" ? "1" : undefined,
    span: text(search.span),
    event,
    log:
      search.log === "1" ||
      (history.event_id && search.view !== "trace" && search.view !== "conversation")
        ? "1"
        : undefined,
  }
}

/** Workflow's specialist/environment selectors are separate from exact event filters. */
export function workflowSearchFromSession(search: Record<string, unknown>): WorkflowSearch {
  return validateWorkflowSearch({
    ...search,
    agent_name: search.workflow_agent_name,
    environment_name: search.workflow_environment_name,
  })
}

/** Replacement patch includes undefined keys, including the invalid marker. */
export function workflowSearchPatch(search: WorkflowSearch) {
  const { agent_name, environment_name, ...rest } = validateWorkflowSearch(search)
  return { ...rest, workflow_agent_name: agent_name, workflow_environment_name: environment_name }
}

export function historySearchPatch(search: SessionHistorySearch) {
  const {
    transcript_role: _role,
    include_thinking: _thinking,
    ...events
  } = validateSessionHistorySearch(search)
  return { ...events, event: events.event_id }
}
