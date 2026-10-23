/**
 * Every read the pages make, as hooks over TanStack Query. Sessions that are running
 * or waiting keep their queries polling; finished ones don't.
 */
import { useQuery, useQueryClient } from "@tanstack/react-query"
import { useEffect, useRef } from "react"
import {
  EVENT_WAIT_SECONDS,
  type EventHistory,
  loadEventHistory,
  loadEventTail,
  TRANSCRIPT_PAGE_MAX,
} from "../api/paging"
import type { InboxQuery, Session, SessionOverview, SessionTranscriptPage } from "../api/types"
import { isLive as isLiveRun } from "./eval-model"
import { useApi } from "./services"
import { isActive } from "./session-model"

const ACTIVE_POLL_MS = 1500

export type SessionRow = { session: Session; overview: SessionOverview | null }

/** Rows without eval-trial sessions, which counts and attention leave out (as Cloud does). */
export const withoutEvalTrials = (rows: SessionRow[]) => rows.filter((r) => !r.overview?.eval_trial)

export function useApp() {
  const api = useApi()
  return useQuery({
    queryKey: ["app"],
    queryFn: () => api.app(),
    staleTime: Number.POSITIVE_INFINITY,
  })
}

export function useSpecialists() {
  const api = useApi()
  return useQuery({ queryKey: ["specialists"], queryFn: () => api.specialists() })
}

/** The session list joined with each session's overview. */
export function useSessionRows(filter: { agent?: string; q?: string; label?: string } = {}) {
  const api = useApi()
  return useQuery({
    queryKey: ["sessions", filter],
    queryFn: async () => {
      const page = await api.sessions({
        limit: 200,
        agent_name: filter.agent || undefined,
        q: filter.q || undefined,
        label: filter.label ? [filter.label] : undefined,
      })
      const overviews = await api.overviews(page.sessions.map((s) => s.id))
      const byId = new Map(overviews.map((o) => [o.session_id, o]))
      const rows: SessionRow[] = page.sessions.map((session) => ({
        session,
        overview: byId.get(session.id) ?? null,
      }))
      return { rows, total: page.total_count ?? rows.length }
    },
    refetchInterval: (query) =>
      query.state.data?.rows.some((r) => isActive(r.overview?.state)) ? ACTIVE_POLL_MS : false,
  })
}

export function useSessionDetail(id: string) {
  const api = useApi()
  return useQuery({ queryKey: ["session", id], queryFn: () => api.session(id) })
}

export function useOverview(id: string) {
  const api = useApi()
  return useQuery({
    queryKey: ["overview", id],
    queryFn: async () => (await api.overviews([id]))[0] ?? null,
    refetchInterval: (query) => (isActive(query.state.data?.state) ? ACTIVE_POLL_MS : false),
  })
}

/** Transcript pages one read follows; a longer conversation shows its first pages. */
const TRANSCRIPT_MAX_PAGES = 10

/** The whole transcript, in pages of the server's maximum; polled while the session is active. */
export function useTranscript(id: string, active: boolean) {
  const api = useApi()
  return useQuery({
    queryKey: ["transcript", id],
    queryFn: async (): Promise<SessionTranscriptPage> => {
      let page = await api.transcript(id, { limit: TRANSCRIPT_PAGE_MAX })
      const messages = [...page.messages]
      for (let n = 1; page.has_more && n < TRANSCRIPT_MAX_PAGES; n++) {
        page = await api.transcript(id, { limit: TRANSCRIPT_PAGE_MAX, offset: page.next_offset })
        if (!page.messages.length) break
        messages.push(...page.messages)
      }
      return { ...page, offset: 0, messages }
    },
    refetchInterval: active ? ACTIVE_POLL_MS : false,
  })
}

/**
 * A session's events, oldest first. The first read loads the history in pages of at most
 * 1,000, keeping the newest `EVENT_CAP` (`truncated` says older ones weren't loaded).
 * After that each read fetches only events after the newest one; while the session is
 * live it long-polls, so new events show as they happen.
 */
export function useEvents(id: string, enabled: boolean, active = false) {
  const api = useApi()
  const queryClient = useQueryClient()
  const queryKey = ["events", id]
  const query = useQuery({
    queryKey,
    queryFn: async (): Promise<EventHistory> => {
      const read = (q: Parameters<typeof api.events>[1]) => api.events(id, q)
      const cached = queryClient.getQueryData<EventHistory>(queryKey)
      if (!cached) return loadEventHistory(read)
      const started = Date.now()
      const next = await loadEventTail(read, cached, {
        wait: active ? EVENT_WAIT_SECONDS : undefined,
      })
      if (next.events.length === cached.events.length && next.through === cached.through) {
        // Nothing new. The server only holds the read while the session runs (not while it
        // waits on a person), so keep quick empty answers to the usual polling pace.
        const rest = ACTIVE_POLL_MS - (Date.now() - started)
        if (active && rest > 0) await new Promise((resolve) => setTimeout(resolve, rest))
        return cached
      }
      return next
    },
    enabled,
    // The tail read itself waits for new events, so the next one can start right away.
    refetchInterval: active ? 250 : false,
  })
  // When the session stops, read the tail once more for its final events.
  const wasActive = useRef(active)
  const { refetch, data } = query
  useEffect(() => {
    if (wasActive.current && !active && enabled && data) void refetch()
    wasActive.current = active
  }, [active, enabled, data, refetch])
  return query
}

export function useInbox(query: InboxQuery = {}) {
  const api = useApi()
  return useQuery({
    queryKey: ["inbox", query],
    queryFn: () => api.inbox({ limit: 200, ...query }),
    refetchInterval: 5000,
  })
}

export function useTaskTypes() {
  const api = useApi()
  return useQuery({ queryKey: ["task-types"], queryFn: () => api.taskTypes() })
}

export function useSessionState(id: string, enabled: boolean) {
  const api = useApi()
  return useQuery({ queryKey: ["state", id], queryFn: () => api.state(id), enabled })
}

export function useToolUsage() {
  const api = useApi()
  return useQuery({ queryKey: ["tool-usage"], queryFn: () => api.toolUsage() })
}

export function useKnowledge() {
  const api = useApi()
  return useQuery({ queryKey: ["knowledge"], queryFn: () => api.knowledge() })
}

const TASK_POLL_MS = 5000
const LIVE_TASK = new Set([
  "pending",
  "claimed",
  "running",
  "waiting_dependencies",
  "waiting_group",
])

/** Every task; polls while any is live, as today's Tasks page does. */
export function useTasks() {
  const api = useApi()
  return useQuery({
    queryKey: ["tasks"],
    queryFn: () => api.tasks(),
    refetchInterval: (query) =>
      query.state.data?.some((t) => LIVE_TASK.has(t.status)) ? TASK_POLL_MS : false,
  })
}

export function useTask(id: string) {
  const api = useApi()
  return useQuery({
    queryKey: ["tasks", id],
    queryFn: () => api.task(id),
    refetchInterval: (query) =>
      query.state.data && LIVE_TASK.has(query.state.data.status) ? TASK_POLL_MS : false,
  })
}

export function useTaskScheduleEvents(id: string, enabled: boolean) {
  const api = useApi()
  return useQuery({
    queryKey: ["tasks", id, "schedule"],
    queryFn: () => api.taskScheduleEvents(id),
    enabled,
  })
}

const EVAL_POLL_MS = 1500

/** Every suite; polls while a suite's last run is still going. */
export function useEvalSuites() {
  const api = useApi()
  return useQuery({
    queryKey: ["evals"],
    queryFn: () => api.evalSuites(),
    refetchInterval: (query) =>
      query.state.data?.some((s) => s.last_run && isLiveRun(s.last_run.status))
        ? EVAL_POLL_MS
        : false,
  })
}

export function useEvalSuite(id: string) {
  const api = useApi()
  return useQuery({
    queryKey: ["evals", "suite", id],
    queryFn: () => api.evalSuite(id),
    refetchInterval: (query) =>
      query.state.data?.runs.some((r) => isLiveRun(r.status)) ? EVAL_POLL_MS : false,
  })
}

export function useEvalRun(id: string, compareWith?: string) {
  const api = useApi()
  return useQuery({
    queryKey: ["evals", "run", id, compareWith ?? null],
    queryFn: () => api.evalRun(id, compareWith),
    refetchInterval: (query) =>
      query.state.data && isLiveRun(query.state.data.status) ? EVAL_POLL_MS : false,
  })
}

export function useEvalSettings() {
  const api = useApi()
  return useQuery({ queryKey: ["evals", "settings"], queryFn: () => api.evalSettings() })
}
