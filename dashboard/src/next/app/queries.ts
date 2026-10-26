/**
 * Every read the pages make, as hooks over TanStack Query. Sessions that are running
 * or waiting keep their queries polling; finished ones don't.
 */
import { useInfiniteQuery, useQuery, useQueryClient } from "@tanstack/react-query"
import { useEffect, useMemo, useRef, useState } from "react"
import { ApiClientError } from "@/lib/api"
import { activityPollInterval } from "@/lib/polling"
import {
  loadTranscriptWindow,
  type TranscriptHistoryQuery,
  type TranscriptPageParam,
} from "@/lib/session-history"
import { overviewForDisplay } from "../api/overview"
import {
  EVENT_WAIT_SECONDS,
  type EventHistory,
  loadEventHistory,
  loadEventTail,
} from "../api/paging"
import type { InboxQuery, Session, SessionOverview, SessionTranscriptPage } from "../api/types"
import { isLive as isLiveRun } from "./eval-model"
import { useApi, useServices } from "./services"
import { loadSessionList } from "./session-list-loader"
import { isActive } from "./session-model"
import { LIVE_TASK, TASK_POLL_MS } from "./task-polling"

const ACTIVE_POLL_MS = 1500

export type SessionRow = { session: Session; overview: SessionOverview | null; title?: string }

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
  return useQuery({ queryKey: ["specialists"], queryFn: ({ signal }) => api.specialists(signal) })
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

export function useOverview(id: string, poll = true) {
  const api = useApi()
  return useQuery({
    queryKey: ["overview", id],
    queryFn: async () => (await api.overviews([id]))[0] ?? null,
    refetchInterval: (query) =>
      poll && isActive(query.state.data?.state) ? ACTIVE_POLL_MS : false,
  })
}

/** One bounded window, using the same tail owner as the runtime history page. */
export function useTranscript(
  id: string,
  active: boolean,
  page: TranscriptPageParam = null,
  filters: TranscriptHistoryQuery = { include_thinking: true },
) {
  const api = useApi()
  return useQuery({
    queryKey: ["transcript", id, filters, page],
    queryFn: ({ signal }): Promise<SessionTranscriptPage> =>
      loadTranscriptWindow(page, 100, 3, (window) =>
        api.transcript(id, { ...filters, ...window }, signal),
      ),
    refetchInterval: active && page === null ? ACTIVE_POLL_MS : false,
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

export function useInbox(query: InboxQuery = {}, poll = true) {
  const api = useApi()
  const client = useQueryClient()
  const key = useMemo(
    () => [
      "inbox",
      {
        session_id: query.session_id,
        agent_name: query.agent_name,
        q: query.q,
        kind: query.kind,
        limit: query.limit,
      },
    ],
    [query.session_id, query.agent_name, query.q, query.kind, query.limit],
  )
  const identity = JSON.stringify(key)
  const [position, setPosition] = useState({
    identity,
    cursor: null as string | null,
    visited: [] as string[],
    number: 1,
  })
  const current =
    position.identity === identity ? position : { identity, cursor: null, visited: [], number: 1 }
  const result = useQuery({
    queryKey: [...key, current.cursor],
    queryFn: ({ signal }) => api.inbox({ limit: 50, ...query, cursor: current.cursor }, signal),
    retry: false,
    refetchInterval: poll && current.cursor === null ? 5000 : false,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  })
  const page = result.data
  const next = page?.next_cursor
  const cycleError =
    next && current.visited.includes(next)
      ? new Error("The Inbox source did not advance. Refresh the list.")
      : null
  const advance = () => {
    if (next && !cycleError && !result.isFetching)
      setPosition({
        identity,
        cursor: next,
        visited: [...current.visited, next],
        number: current.number + 1,
      })
  }
  useEffect(() => {
    if (
      next &&
      page.items.length === 0 &&
      !page.issues?.length &&
      current.number % 25 !== 0 &&
      !result.isFetching &&
      !result.error &&
      !cycleError
    )
      setPosition({
        identity,
        cursor: next,
        visited: [...current.visited, next],
        number: current.number + 1,
      })
  }, [next, page, current, identity, result.isFetching, result.error, cycleError])
  const restarted = useRef<string | null>(null)
  useEffect(() => {
    if (
      result.error instanceof ApiClientError &&
      result.error.status === 422 &&
      current.cursor &&
      restarted.current !== identity
    ) {
      restarted.current = identity
      setPosition({ identity, cursor: null, visited: [], number: 1 })
      void client.resetQueries({ queryKey: [...key, null], exact: true })
    }
  }, [result.error, current.cursor, identity, client, key])
  const refresh = async () => {
    await client.cancelQueries({ queryKey: key })
    setPosition({ identity, cursor: null, visited: [], number: 1 })
    await client.resetQueries({ queryKey: [...key, null], exact: true })
  }
  return {
    ...result,
    refetch: refresh,
    fetchNextPage: advance,
    isFetchingNextPage: result.isFetching,
    historical: current.cursor !== null,
    error: result.error ?? cycleError,
    isError: result.isError || !!cycleError,
    isSuccess: result.isSuccess && !cycleError,
    hasNextPage: !!next && !cycleError,
    data: page ? { ...page, issues: page.issues ?? [] } : undefined,
  }
}

export function useTaskTypes() {
  const api = useApi()
  return useQuery({ queryKey: ["task-types"], queryFn: () => api.taskTypes() })
}

export function useSessionState(id: string, enabled: boolean) {
  const api = useApi()
  return useQuery({ queryKey: ["state", id], queryFn: () => api.state(id), enabled })
}

export function useToolUsage(enabled = true) {
  const api = useApi()
  return useQuery({
    queryKey: ["tool-usage"],
    enabled,
    queryFn: ({ signal }) => api.toolUsage(signal),
    refetchInterval: TASK_POLL_MS,
    retry: false,
  })
}

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
    queryFn: ({ signal }) => api.task(id, signal),
    retry: (count, error) => count < 2 && !(error instanceof ApiClientError && error.status < 500),
    refetchInterval: (query) =>
      activityPollInterval(
        !!query.state.data && LIVE_TASK.has(query.state.data.status),
        TASK_POLL_MS,
      ),
  })
}

export function useTaskScheduleEvents(id: string, enabled: boolean) {
  const api = useApi()
  return useInfiniteQuery({
    queryKey: ["tasks", id, "schedule"],
    initialPageParam: undefined as number | undefined,
    queryFn: ({ pageParam, signal }) => api.taskScheduleEvents(id, pageParam, signal),
    getNextPageParam: (page, _pages, _param, params) => {
      const next = page.at(-1)?.sequence
      return page.length === 100 && next != null && !params.includes(next) ? next : undefined
    },
    enabled,
    refetchInterval: 20_000,
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

/** Exact counts are separate from list pagination and may deliberately be unavailable. */
export function useInboxCounts(query: Pick<InboxQuery, "session_id" | "agent_name" | "q"> = {}) {
  const api = useApi()
  const { prototype, inboxEnabled } = useServices()
  return useQuery({
    enabled: !!prototype || inboxEnabled === true,
    queryKey: ["inbox-counts", query],
    queryFn: () => api.inboxCounts(query),
    refetchInterval: 30000,
  })
}

/** Home lists use the server's bounded filtered pages; their lengths are never totals. */
export function useHomeSessionRows(scope: "waiting" | "failed" | "recent") {
  const api = useApi()
  const query = useInfiniteQuery({
    queryKey: ["home", "sessions", scope],
    initialPageParam: null as string | null,
    queryFn: async ({ pageParam, signal }) => {
      const result = await loadSessionList(
        {
          include: ["overview"],
          include_eval: false,
          include_forks: false,
          overview_state:
            scope === "waiting"
              ? ["waiting_approval", "waiting_input", "needs_recovery"]
              : scope === "failed"
                ? ["failed"]
                : undefined,
          limit: 20,
          cursor: pageParam,
          order_by: "updated_at_desc",
        },
        (params, signal) => api.sessions(params, signal),
        (id, params, signal) => api.transcript(id, params, signal),
        signal,
      )
      return {
        rows: result.rows.map((row) => ({
          session: row.session,
          overview: overviewForDisplay(row.overview),
          title: row.title,
        })),
        next_cursor: result.page.next_cursor,
        searchPaused: result.searchPaused,
      }
    },
    getNextPageParam: (page) => page.next_cursor ?? undefined,
    refetchInterval: (query) =>
      activityPollInterval(
        query.state.data?.pages.some((page) =>
          page.rows.some((row) => row.overview?.state === "running"),
        ) ?? false,
        5000,
      ),
  })
  return {
    ...query,
    rows: [
      ...new Map(
        (query.data?.pages.flatMap((page) => page.rows) ?? []).map((row) => [row.session.id, row]),
      ).values(),
    ],
  }
}
