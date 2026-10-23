/**
 * The same calls against the runtime server. Existing endpoints go through src/lib's
 * client; the proposed reads fall back to what today's server can answer, so this can
 * replace the fake API before the backend items land.
 */
import { mutationTransportErrorMessage } from "@/components/dashboard/mutation-transport-status"
import {
  ApiClientError,
  blockTask,
  fetchAgents,
  fetchDashboardIdentity,
  fetchPendingActions,
  fetchSession,
  fetchSessionEvents,
  fetchSessionState,
  fetchSessionsPage,
  fetchSessionTranscript,
  fetchTask,
  fetchTasks,
  markTaskNeedsAttention,
  pauseTask,
  resumeTask,
  type Session as SessionRecord,
  throwResponseError,
} from "@/lib/api"
import { apiUrl } from "@/lib/config"
import type { SessionUsageSummary } from "@/lib/generated/server-api"
import {
  executeInterruptMutation,
  executeRecoverToolApprovalMutation,
  executeRecoverToolRoundMutation,
  executeRecoverUserInputMutation,
  executeResumeMutation,
  executeRunMutation,
} from "@/lib/mutation-browser"
import type { MutationTransportSnapshot } from "@/lib/mutation-transport"
import { isEvalTrialMetadata } from "../app/session-model"
import { filterInboxItems, readAllInbox } from "./paging"
import type {
  Api,
  InboxItemState,
  InboxPage,
  SessionOverview,
  SessionState,
  TaskScheduleEvent,
} from "./types"

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

/** Today's server has no overview; derive what the session record says. */
function fallbackState(status: string): SessionState {
  if (status === "active" || status === "running") return "running"
  if (status === "failed") return "failed"
  if (status === "interrupted") return "interrupted"
  return "completed"
}

/** What today's session record says, in the overview's shape. */
function fallbackOverview(
  s: SessionRecord,
  usage: SessionUsageSummary | undefined,
  metadata: Record<string, unknown> | undefined,
): SessionOverview {
  const tokens = usage?.usage
  const count = (v: unknown) => (v == null ? 0 : Number(v) || 0)
  return {
    session_id: s.id,
    // No titles today: fall back to the labels, as the prototype's switch shows.
    title:
      Object.entries(s.labels)
        .map(([k, v]) => `${k}=${v}`)
        .join(" · ") || null,
    summary: null,
    summary_tone: null,
    state: fallbackState(s.status),
    reason: null,
    started_at: s.created_at,
    ended_at: null,
    last_activity_at: s.updated_at,
    model_steps: usage?.model_steps ?? 0,
    tool_calls: usage?.tool_calls ?? 0,
    tokens: {
      input: count(tokens?.input_tokens),
      output: count(tokens?.output_tokens),
      cached: count(tokens?.cache?.cached_input_tokens),
    },
    cost: null,
    uses: { browser: false, subagents: 0, artifacts: 0 },
    // The session list doesn't return metadata, so only sessions read one by one are known.
    eval_trial: isEvalTrialMetadata(metadata),
  }
}

/** Run `work` over `items`, at most `limit` at a time. */
async function mapLimited<T, R>(items: T[], limit: number, work: (item: T) => Promise<R>) {
  const results: R[] = new Array(items.length)
  let next = 0
  const lanes = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (next < items.length) {
      const i = next++
      results[i] = await work(items[i] as T)
    }
  })
  await Promise.all(lanes)
  return results
}

/** Above this many IDs, overviews reads a page of recent sessions before reading one by one. */
const OVERVIEW_DIRECT_READS = 3

/**
 * Sessions the last list reads returned, so `overviews` right after `sessions` (as every
 * list page does) needs no second request. Entries go stale after a few seconds.
 */
const LISTED_FRESH_MS = 5000
const LISTED_MAX = 2000
const listed = new Map<
  string,
  { session: SessionRecord; usage?: SessionUsageSummary; at: number }
>()
function rememberListed(sessions: SessionRecord[], usage: SessionUsageSummary[]) {
  const at = Date.now()
  const byId = new Map(usage.map((u) => [u.session_id, u]))
  for (const session of sessions) {
    listed.delete(session.id)
    listed.set(session.id, { session, usage: byId.get(session.id), at })
  }
  for (const id of listed.keys()) {
    if (listed.size <= LISTED_MAX) break
    listed.delete(id)
  }
}

export const realApi: Api = {
  app: fetchDashboardIdentity,
  specialists: () => fetchAgents(),
  async sessions(q) {
    const page = await fetchSessionsPage({ include: ["usage"], ...q } as Parameters<
      typeof fetchSessionsPage
    >[0])
    rememberListed(page.sessions, page.usage ?? [])
    return page
  },
  session: (id) => fetchSession(id),
  /**
   * The sessions asked for, not the first N: a page of recent sessions covers most lists,
   * and any ID not on it is read with `fetchSession`. A few IDs are read directly.
   */
  async overviews(ids) {
    const wanted = [...new Set(ids)]
    if (!wanted.length) return []
    const found = new Map<string, SessionOverview>()
    // The list a page just read usually holds every ID it then asks about.
    const fresh = Date.now() - LISTED_FRESH_MS
    for (const id of wanted) {
      const hit = listed.get(id)
      if (hit && hit.at >= fresh) found.set(id, fallbackOverview(hit.session, hit.usage, undefined))
    }
    if (wanted.length - found.size > OVERVIEW_DIRECT_READS) {
      const page = await fetchSessionsPage({
        limit: Math.min(1000, Math.max(wanted.length * 2, 100)),
        include: ["usage"],
      })
      const usage = new Map((page.usage ?? []).map((u) => [u.session_id, u]))
      rememberListed(page.sessions, page.usage ?? [])
      const ask = new Set(wanted)
      for (const s of page.sessions)
        if (ask.has(s.id) && !found.has(s.id))
          found.set(s.id, fallbackOverview(s, usage.get(s.id), undefined))
    }
    const missing = wanted.filter((id) => !found.has(id))
    const details = await mapLimited(missing, 6, (id) =>
      fetchSession(id).catch((error: unknown) => {
        if (error instanceof ApiClientError && error.status === 404) return null
        throw error
      }),
    )
    for (const d of details) if (d) found.set(d.id, fallbackOverview(d, undefined, d.metadata))
    return wanted.flatMap((id) => {
      const o = found.get(id)
      return o ? [o] : []
    })
  },
  transcript: (id, q) => fetchSessionTranscript(id, q),
  events: (id, q) => fetchSessionEvents(id, q),
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
  /**
   * The whole inbox, every page (see `readAllInbox`). The filters are sent for servers
   * that support them, and applied here too until today's server does.
   */
  async inbox(q = {}) {
    const page = await readAllInbox(
      (cursor) =>
        json<InboxPage>(
          `/inbox${query({ limit: q.limit, cursor, session_id: q.session_id, agent_name: q.agent_name, kind: q.kind })}`,
        ),
      { isStaleCursor: (e) => e instanceof ApiClientError && e.status === 422 },
    )
    return { ...page, items: filterInboxItems(page.items, q) }
  },
  settle: (id, body) =>
    json<InboxItemState>(`/inbox/${encodeURIComponent(id)}/settle`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  async start({ agent, prompt }) {
    const snapshot = await executeRunMutation({ agent, prompt })
    return { session_id: snapshot.sessionId }
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
