import { useInfiniteQuery, useQuery } from "@tanstack/react-query"
import { ApiClientError } from "@/lib/api"
import type { TaskStatus, TaskStatusCounts } from "@/lib/generated/server-api"
import { activityPollInterval } from "@/lib/polling"
import { TASK_BATCH_MAX, type TaskIndexQuery } from "../api/tasks"
import type { Api, TaskInfo } from "../api/types"
import {
  firstOccurrences,
  firstPageOnly,
  singlePageOnly,
  useFirstPageRefresh,
} from "./first-page-refresh"
import { useApi, useServices } from "./services"
import { taskGroupStatuses, taskTypeFilter } from "./task-index-model"
import { LIVE_TASK, TASK_POLL_MS, taskCanChange } from "./task-polling"

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
    ...taskTypeFilter(search.type),
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
  const queryKey = ["tasks", "index", query, native]
  const readPage = async (pageParam: string | undefined, signal: AbortSignal) => {
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
  }
  type TaskPage = Awaited<ReturnType<typeof readPage>>
  const interval = (pages: readonly TaskPage[]) =>
    activityPollInterval(
      firstOccurrences(
        pages.flatMap((page) => page.items),
        (task) => task.id,
      ).some((task) => LIVE_TASK.has(task.status)),
      TASK_POLL_MS,
    )
  const result = useInfiniteQuery({
    queryKey,
    initialPageParam: undefined as string | undefined,
    queryFn: ({ pageParam, signal }) => readPage(pageParam, signal),
    getNextPageParam: (page, _pages, _param, params) =>
      page.next_cursor && !params.includes(page.next_cursor) ? page.next_cursor : undefined,
    // Only page 1 is re-read on the interval; older pages keep what was loaded.
    refetchInterval: singlePageOnly(interval),
  })
  useFirstPageRefresh({
    queryKey,
    data: result.data,
    dataUpdatedAt: result.dataUpdatedAt,
    fetchFirst: (signal) => readPage(undefined, signal),
    interval,
    rowKeys: (page) => page.items.map((task) => task.id),
  })
  return result
}
/** Relations poll while the task, or a related task, hasn't finished. */
export function useTaskRelationPage(
  id: string,
  kind: "children" | "dependents",
  enabled: boolean,
  live = true,
) {
  const api = useApi()
  return useInfiniteQuery({
    queryKey: ["tasks", id, kind],
    refetchInterval: firstPageOnly<Awaited<ReturnType<typeof api.taskIndex>>>((pages) =>
      live || pages.some((page) => page.items.some((task) => taskCanChange(task.status)))
        ? TASK_POLL_MS
        : false,
    ),
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
type TaskLinks = { rows: TaskInfo[]; issues: { id: string; error: unknown }[] }
type TaskAttempts = { rows: TaskInfo[]; incomplete: boolean }

export function useTaskLinks(ids: string[], enabled = true, observe = false) {
  const api = useApi()
  return useQuery({
    queryKey: ["tasks", "links", ids],
    // Observed links stop polling once every linked task has finished; a held one can resume.
    refetchInterval: (query: { state: { data: TaskLinks | undefined } }) =>
      observe &&
      (query.state.data === undefined ||
        query.state.data.issues.length > 0 ||
        query.state.data.rows.some((task) => taskCanChange(task.status)))
        ? TASK_POLL_MS
        : false,
    enabled: enabled && ids.length > 0,
    queryFn: ({ signal }): Promise<TaskLinks> => readTasks(api, ids, signal),
  })
}
export function useTaskAttempts(id: string, enabled: boolean, limit: number) {
  const api = useApi()
  return useQuery({
    queryKey: ["tasks", id, "attempts", limit],
    // A retry series grows only while one of its attempts can still change.
    refetchInterval: (query: { state: { data: TaskAttempts | undefined } }) =>
      query.state.data?.rows.some((task) => taskCanChange(task.status)) === false
        ? false
        : TASK_POLL_MS,
    enabled,
    queryFn: async ({ signal }): Promise<TaskAttempts> => {
      const first = await api.task(id, signal)
      const rows = [first],
        seen = new Set([first.id])
      let incomplete = false
      // Each step reads the next attempt in both directions with one request.
      let ends: Partial<Record<RetryDirection, TaskInfo>> = {
        predecessor_task_id: first,
        successor_task_id: first,
      }
      while (Object.keys(ends).length) {
        const wanted = new Map<string, RetryDirection>()
        for (const [direction, task] of Object.entries(ends) as [RetryDirection, TaskInfo][]) {
          const next = task.retry_series?.[direction]
          if (!next) continue
          if (seen.has(next)) throw new Error("Retry history contains a repeated task identity.")
          wanted.set(next, direction)
        }
        if (!wanted.size) break
        if (rows.length + wanted.size > limit) {
          incomplete = true
          break
        }
        const { rows: found, issues } = await readTasks(api, [...wanted.keys()], signal)
        const [missing] = issues
        if (missing) throw missing.error
        ends = {}
        for (const task of found) {
          seen.add(task.id)
          rows.push(task)
          ends[wanted.get(task.id)!] = task
        }
      }
      return {
        rows: rows.sort((a, b) => (a.retry_series?.attempt ?? 0) - (b.retry_series?.attempt ?? 0)),
        incomplete,
      }
    },
  })
}

type RetryDirection = "predecessor_task_id" | "successor_task_id"

/** Each task in `ids` with one `tasks?ids=` request per 100, and an issue for each missing one. */
export async function readTasks(api: Api, ids: string[], signal: AbortSignal) {
  const wanted = [...new Set(ids)]
  const rows: TaskInfo[] = []
  for (let start = 0; start < wanted.length; start += TASK_BATCH_MAX) {
    signal.throwIfAborted()
    const batch = wanted.slice(start, start + TASK_BATCH_MAX)
    const found = await api.tasksByIds(batch, signal)
    if (found.some((task) => !batch.includes(task.id)))
      return readTasksOneByOne(api, wanted, signal)
    rows.push(...found)
  }
  const byId = new Map(rows.map((task) => [task.id, task]))
  return {
    rows: wanted.flatMap((id) => byId.get(id) ?? []),
    issues: wanted
      .filter((id) => !byId.has(id))
      .map((id) => ({ id, error: new ApiClientError(`Task not found: ${id}`, 404) as unknown })),
  }
}

// A server that predates `ids` ignores it and lists other tasks.
async function readTasksOneByOne(api: Api, ids: string[], signal: AbortSignal) {
  const rows: TaskInfo[] = [],
    issues: { id: string; error: unknown }[] = []
  for (const id of ids) {
    signal.throwIfAborted()
    try {
      rows.push(await api.task(id, signal))
    } catch (error) {
      signal.throwIfAborted()
      issues.push({ id, error })
    }
  }
  return { rows, issues }
}
