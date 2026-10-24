/**
 * The same calls against the runtime server. Existing endpoints go through src/lib's
 * client; the proposed reads fall back to what today's server can answer, so this can
 * replace the fake API before the backend items land.
 */
import { mutationTransportErrorMessage } from "@/components/dashboard/mutation-transport-status"
import {
  blockTask,
  fetchAgents,
  fetchDashboardIdentity,
  fetchPendingActions,
  fetchSession,
  fetchSessionEvents,
  fetchSessionState,
  fetchSessionsPage,
  fetchSessionTrace,
  fetchSessionTranscript,
  fetchTask,
  fetchTasks,
  markTaskNeedsAttention,
  pauseTask,
  resumeTask,
  throwResponseError,
} from "@/lib/api"
import { apiUrl } from "@/lib/config"
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
import { overviewForDisplay } from "./overview"
import type { Api, InboxPage, SessionOverview, TaskScheduleEvent } from "./types"

/** Proposed (backend item 14): the suite-centred eval calls have no endpoint yet. */
async function needsEvalReads(): Promise<never> {
  throw new Error("Needs backend item 14 (suite-centred eval reads).")
}

async function json<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(apiUrl(path), {
    ...init,
    headers: { "Content-Type": "application/json", ...init.headers },
  })
  if (!response.ok) await throwResponseError(response)
  return (await response.json()) as T
}

/** A mutation that didn't reach its terminal outcome becomes an error the page can show. */
function requireTerminal(snapshot: MutationTransportSnapshot, fallback: string) {
  if (snapshot.phase !== "terminal")
    throw new Error(mutationTransportErrorMessage(snapshot) ?? fallback)
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
  task: (id) => fetchTask(id),
  taskScheduleEvents: (id) =>
    json<TaskScheduleEvent[]>(
      `/tasks/${encodeURIComponent(id)}/schedule/events${query({ limit: 100 })}`,
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
  async reply(id, prompt) {
    await executeResumeMutation({ session_id: id, prompt })
  },
  /**
   * The session's pending manual recovery, settled through the endpoint its identity
   * names, as today's session page does (`handleManualRecovery`).
   */
  async recover(sessionId, body) {
    const page = await fetchPendingActions({ session_id: sessionId, limit: 50 })
    const recoveries = page.actions.filter((a) => a.kind === "manual_recovery" && a.tool_call_id)
    const action =
      recoveries.find((a) => body.attention_id && a.attention_id === body.attention_id) ??
      (body.attention_id ? undefined : recoveries[0])
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
      snapshot = await executeRecoverUserInputMutation({
        ...common,
        input_id: action.input_id,
        answer: body.answer,
      })
    } else if (action.approval_id) {
      if (!action.round_id)
        throw new Error("Recovering an approval needs the tool round, which wasn't reported.")
      snapshot = await executeRecoverToolApprovalMutation({
        ...common,
        approval_id: action.approval_id,
        tool_round_id: action.round_id,
      })
    } else if (action.round_id) {
      snapshot = await executeRecoverToolRoundMutation({ ...common, round_id: action.round_id })
    } else {
      throw new Error("Manual recovery needs an approval, question or tool round to settle.")
    }
    requireTerminal(snapshot, "The recovery outcome is unavailable.")
  },
  async interrupt(id, reason) {
    const snapshot = await executeInterruptMutation(id, reason ? { reason } : {})
    requireTerminal(snapshot, "The interruption outcome is unavailable.")
  },
  state: (id) => fetchSessionState(id),
  // Proposed (backend item 7): no task types or task creation on today's server.
  taskTypes: async () => [],
  async createTask() {
    throw new Error("Creating a task needs the runtime's task API (backend item 7).")
  },
  /** Today: rerun the same specialist with the session's first message and labels. */
  async runAgain(sessionId) {
    const [session, transcript] = await Promise.all([
      fetchSession(sessionId),
      fetchSessionTranscript(sessionId, { limit: 10 }),
    ])
    const first = transcript.messages.find((m) => m.role === "user")
    const prompt = (first?.content ?? [])
      .map((p) => (typeof p.text === "string" ? p.text : ""))
      .join("")
    const snapshot = await executeRunMutation({
      agent: session.agent_name,
      prompt,
      labels: session.labels,
    })
    return { session_id: snapshot.sessionId }
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
