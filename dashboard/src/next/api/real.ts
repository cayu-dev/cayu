/**
 * The same calls against the runtime server. Existing endpoints go through src/lib's
 * client; the proposed reads fall back to what today's server can answer, so this can
 * replace the fake API before the backend items land.
 */
import {
  blockTask,
  fetchAgents,
  fetchDashboardIdentity,
  fetchSession,
  fetchSessionEvents,
  fetchSessionState,
  fetchSessionsPage,
  fetchSessionTrace,
  fetchSessionTranscript,
  fetchTasks,
  markTaskNeedsAttention,
  pauseTask,
  resumeTask,
} from "@/lib/api"
import type { InboxSettleResponse } from "@/lib/generated/server-api"
import {
  executeInterruptMutation,
  executeRecoverToolApprovalMutation,
  executeRecoverToolRoundMutation,
  executeRecoverUserInputMutation,
  executeResumeMutation,
  executeRunMutation,
} from "@/lib/mutation-browser"
import type { MutationTransportSnapshot } from "@/lib/mutation-transport"
import { fetchSessionRecovery } from "@/lib/session-recovery"
import { json } from "./http"
import { overviewForDisplay } from "./overview"
import { createDeclaredTask, fetchTaskDependents, fetchTaskIndex, taskQueryString } from "./tasks"
import type { Api, InboxPage, SessionOverview, TaskScheduleEvent } from "./types"

/** Proposed (backend item 14): the suite-centred eval calls have no endpoint yet. */
async function needsEvalReads(): Promise<never> {
  throw new Error("Needs backend item 14 (suite-centred eval reads).")
}

const query = (params: Record<string, string | number | null | undefined>) => {
  const search = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) if (v != null && v !== "") search.set(k, String(v))
  const text = search.toString()
  return text ? `?${text}` : ""
}

export const realApi: Api = {
  app: fetchDashboardIdentity,
  specialists: () => fetchAgents(),
  sessions: (q, signal) => fetchSessionsPage({ include: ["overview"], ...q }, signal),
  session: (id) => fetchSession(id),
  async overviews(ids) {
    const wanted = [...new Set(ids)]
    const result: SessionOverview[] = []
    for (let offset = 0; offset < wanted.length; offset += 200) {
      const page = await fetchSessionsPage({
        ids: wanted.slice(offset, offset + 200),
        include: ["overview"],
      })
      result.push(...(page.overviews ?? []).map(overviewForDisplay))
    }
    return result
  },
  inboxCounts: (q = {}) => json(`/inbox/counts${query(q)}`),
  overviewCounts: (q) => json(`/sessions/overview/counts${query(q)}`),
  transcript: (id, q, signal) => fetchSessionTranscript(id, q, signal),
  trace: (id, q, signal) => fetchSessionTrace(id, q, signal),
  events: (id, q, signal) => fetchSessionEvents(id, q, signal),
  tasks: () => fetchTasks({ limit: 200, order_by: "updated_at_desc" }),
  task: (id, signal) => json(`/tasks/${encodeURIComponent(id)}`, { signal }),
  taskIndex: fetchTaskIndex,
  taskList: (query, signal) => json(`/tasks${taskQueryString(query)}`, { signal }),
  taskDependents: fetchTaskDependents,
  taskScheduleEvents: (id, afterSequence, signal) =>
    json<TaskScheduleEvent[]>(
      `/tasks/${encodeURIComponent(id)}/schedule/events${query({ limit: 100, after_sequence: afterSequence })}`,
      { signal },
    ),
  holdTask: (id, hold, reason) => {
    const body = reason ? { reason } : {}
    if (hold === "pause") return pauseTask(id, body)
    if (hold === "block") return blockTask(id, body)
    return markTaskNeedsAttention(id, body)
  },
  resumeTask: (id) => resumeTask(id),
  inbox: (q = {}, signal) => json<InboxPage>(`/inbox${query(q)}`, { signal }),
  settle: async (id, body) => {
    const result = await json<InboxSettleResponse>(`/inbox/${encodeURIComponent(id)}/settle`, {
      method: "POST",
      body: JSON.stringify(body),
    })
    return result.item
  },
  async start({ agent, prompt }, options) {
    const snapshot = await executeRunMutation({ agent, prompt }, options)
    return { session_id: snapshot.sessionId, snapshot }
  },
  async reply(id, prompt, options) {
    return executeResumeMutation({ session_id: id, prompt }, options)
  },
  /**
   * The session's pending manual recovery, settled through the endpoint its identity
   * names from the exact public pending-action projection.
   */
  async recover(sessionId, body, options) {
    const action = await fetchSessionRecovery(sessionId, body.attention_id, options?.signal)
    if (!action?.tool_call_id)
      throw new Error("This recovery is no longer pending. Refresh the session.")
    const common = {
      session_id: sessionId,
      tool_call_id: action.tool_call_id,
      outcome: body.outcome,
      message: body.message,
    }
    let snapshot: MutationTransportSnapshot
    if (action.input_id) {
      if (!body.answer) throw new Error("Recovering a question needs the person's answer.")
      snapshot = await executeRecoverUserInputMutation(
        {
          ...common,
          input_id: action.input_id,
          answer: body.answer,
        },
        options,
      )
    } else if (action.approval_id) {
      if (!action.round_id)
        throw new Error("Recovering an approval needs the tool round, which wasn't reported.")
      snapshot = await executeRecoverToolApprovalMutation(
        {
          ...common,
          approval_id: action.approval_id,
          tool_round_id: action.round_id,
        },
        options,
      )
    } else if (action.round_id) {
      snapshot = await executeRecoverToolRoundMutation(
        { ...common, round_id: action.round_id },
        options,
      )
    } else {
      throw new Error("Manual recovery needs an approval, question or tool round to settle.")
    }
    return snapshot
  },
  async interrupt(id, reason, options) {
    return executeInterruptMutation(id, reason ? { reason } : {}, options)
  },
  state: (id) => fetchSessionState(id),
  taskTypes: async () => {
    const result = await json<{ task_types: import("@/lib/generated/server-api").TaskTypeSpec[] }>(
      "/task-types",
    )
    return result.task_types.map((type) => ({
      ...type,
      description: type.description ?? "",
      agent_name: type.agent_name ?? null,
    }))
  },
  async createTask(body, operationId) {
    if (!operationId) throw new Error("Task creation requires an operation identity.")
    const result = await createDeclaredTask({ ...body, operation_id: operationId })
    return { task_id: result.task_id, session_id: null }
  },
  /** Direct specialist replay; task replay belongs to the task creation owner. */
  async runAgain(sessionId, options) {
    const session = await fetchSession(sessionId, options?.signal)
    if (
      !["http_run", "sdk_run", "fork", "subagent", "workflow_step"].includes(
        session.invocation.source,
      )
    )
      throw new Error("This session requires its task's replay action.")
    const transcript = await fetchSessionTranscript(
      sessionId,
      { role: "user", offset: 0, limit: 1 },
      options?.signal,
    )
    const first = transcript.messages[0]
    if (
      first?.role !== "user" ||
      !first.content.length ||
      first.content.some((part) => part.type !== "text" || typeof part.text !== "string")
    )
      throw new Error(
        "Run again requires a first user message supported by the text run API. Start a session with explicit input instead.",
      )
    const prompt = first.content.map((part) => part.text as string).join("")
    const snapshot = await executeRunMutation(
      {
        agent: session.agent_name,
        prompt,
        labels: session.labels,
      },
      options,
    )
    return { session_id: snapshot.sessionId, snapshot }
  },
  // Proposed reads: until backend items 11 and 12 land there is nothing to show.
  toolUsage: async () => [],
  knowledge: async () => [],
  async removeKnowledge() {
    throw new Error("Removing knowledge needs the runtime's knowledge API (backend item 12).")
  },
  evalSuites: needsEvalReads,
  evalSuite: needsEvalReads,
  evalRun: needsEvalReads,
  runEvalSuite: needsEvalReads,
  approveBaseline: needsEvalReads,
  createEvalSuite: needsEvalReads,
  addToEvals: needsEvalReads,
  evalSettings: needsEvalReads,
}
