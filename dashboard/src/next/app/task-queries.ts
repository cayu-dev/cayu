import { useInfiniteQuery, useQuery } from "@tanstack/react-query"
import type { TaskStatus, TaskStatusCounts } from "@/lib/generated/server-api"
import { activityPollInterval } from "@/lib/polling"
import type { TaskIndexQuery } from "../api/tasks"
import { useApi, useServices } from "./services"
import { taskGroupStatuses } from "./task-index-model"
import { LIVE_TASK, TASK_POLL_MS } from "./task-polling"

// Foreground observation refreshes only the explicitly loaded window. React Query owns
// cancellation, serial cursor reconstruction and overlapping-read suppression.

export type TaskFilters = {
  group?: string
  q?: string
  type?: string
  agent?: string
  assigned_agent_name?: string
  session_id?: string
  cayu?: string
  task_id?: string
}
export function taskFiltersQuery(search: TaskFilters): TaskIndexQuery {
  return {
    q: search.q,
    type: search.type,
    assigned_agent_name: search.assigned_agent_name ?? search.agent,
    session_id: search.session_id,
    include_internal: search.cayu === "1",
    statuses: taskGroupStatuses(search.group) as TaskStatus[],
    limit: 50,
    order_by: "updated_at_desc",
  }
}
export function useTaskIndex(search: TaskFilters) {
  const api = useApi()
  const services = useServices()
  const native = !!services.prototype || services.taskIndexEnabled === true
  const query = taskFiltersQuery(search)
  return useInfiniteQuery({
    queryKey: ["tasks", "index", query, native],
    initialPageParam: undefined as string | undefined,
    queryFn: async ({ pageParam, signal }) => {
      if (native) return api.taskIndex({ ...query, cursor: pageParam }, signal)
      const offset = Number(pageParam ?? 0)
      const items = await api.taskList({ ...query, offset }, signal)
      return {
        items,
        total_count: null,
        counts_by_status: null as TaskStatusCounts | null,
        as_of: null,
        next_cursor: items.length === 50 ? String(offset + 50) : null,
      }
    },
    getNextPageParam: (page, _pages, _param, params) =>
      page.next_cursor && !params.includes(page.next_cursor) ? page.next_cursor : undefined,
    refetchInterval: (query) =>
      activityPollInterval(
        query.state.data?.pages.some((page) =>
          page.items.some((task) => LIVE_TASK.has(task.status)),
        ) ?? false,
        TASK_POLL_MS,
      ),
  })
}
export function useTaskRelationPage(id: string, kind: "children" | "dependents", enabled: boolean) {
  const api = useApi()
  return useInfiniteQuery({
    queryKey: ["tasks", id, kind],
    refetchInterval: TASK_POLL_MS,
    enabled,
    initialPageParam: undefined as string | undefined,
    queryFn: ({ pageParam, signal }) =>
      kind === "dependents"
        ? api.taskDependents(id, pageParam, signal)
        : api.taskIndex(
            { parent_task_id: id, include_internal: true, cursor: pageParam, limit: 50 },
            signal,
          ),
    getNextPageParam: (page, _pages, _param, params) =>
      page.next_cursor && !params.includes(page.next_cursor) ? page.next_cursor : undefined,
  })
}
export function useTaskLinks(ids: string[], enabled = true, observe = false) {
  const api = useApi()
  return useQuery({
    queryKey: ["tasks", "links", ids],
    refetchInterval: observe ? TASK_POLL_MS : false,
    enabled: enabled && ids.length > 0,
    queryFn: async ({ signal }) => {
      const rows = [],
        issues: { id: string; error: unknown }[] = []
      for (const id of [...new Set(ids)]) {
        signal.throwIfAborted()
        try {
          rows.push(await api.task(id, signal))
        } catch (error) {
          signal.throwIfAborted()
          issues.push({ id, error })
        }
      }
      return { rows, issues }
    },
  })
}
export function useTaskAttempts(id: string, enabled: boolean, limit: number) {
  const api = useApi()
  return useQuery({
    queryKey: ["tasks", id, "attempts", limit],
    refetchInterval: TASK_POLL_MS,
    enabled,
    queryFn: async ({ signal }) => {
      const first = await api.task(id, signal)
      const rows = [first],
        seen = new Set([first.id])
      let incomplete = false
      for (const direction of ["predecessor_task_id", "successor_task_id"] as const) {
        let current = first
        while (current.retry_series?.[direction]) {
          const next = current.retry_series[direction]!
          if (seen.has(next)) throw new Error("Retry history contains a repeated task identity.")
          if (rows.length >= limit) {
            incomplete = true
            break
          }
          signal.throwIfAborted()
          current = await api.task(next, signal)
          seen.add(next)
          rows.push(current)
        }
      }
      return {
        rows: rows.sort((a, b) => (a.retry_series?.attempt ?? 0) - (b.retry_series?.attempt ?? 0)),
        incomplete,
      }
    },
  })
}
