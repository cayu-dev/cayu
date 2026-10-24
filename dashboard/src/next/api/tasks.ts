/** Task control-plane contracts; public identifiers never substitute for replay authority. */

import type {
  ApiTaskDetail,
  ApiTaskIndex,
  DeclaredTaskCreateBody,
  TaskCreationResponse,
  TaskReplayAvailability,
  TaskReplayBody,
  TaskStatus,
} from "@/lib/generated/server-api"
import { json as read } from "./http"

export type TaskIndexQuery = {
  q?: string
  type?: string
  assigned_agent_name?: string
  parent_task_id?: string
  session_id?: string
  status?: TaskStatus
  statuses?: TaskStatus[]
  include_internal?: boolean
  order_by?: "updated_at_desc" | "created_at_asc"
  cursor?: string
  limit?: number
}
export function taskQueryString(query: Record<string, unknown>): string {
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(query)) {
    if (value === undefined || value === null || value === "") continue
    for (const item of Array.isArray(value) ? value : [value]) search.append(key, String(item))
  }
  return `?${search}`
}
export const fetchTaskIndex = (query: TaskIndexQuery, signal?: AbortSignal) =>
  read<ApiTaskIndex>(`/tasks${taskQueryString({ ...query, include: "counts" })}`, { signal })
export const fetchTaskDependents = (id: string, cursor?: string, signal?: AbortSignal) =>
  read<ApiTaskIndex>(
    `/tasks/${encodeURIComponent(id)}/dependents${taskQueryString({ limit: 50, cursor })}`,
    { signal },
  )
export const createDeclaredTask = (body: DeclaredTaskCreateBody) =>
  read<TaskCreationResponse>("/tasks", { method: "POST", body: JSON.stringify(body) })
export const readTaskReplay = (id: string, signal?: AbortSignal) =>
  read<TaskReplayAvailability>(`/tasks/${encodeURIComponent(id)}/replay`, { signal })
export const replayDeclaredTask = (id: string, body: TaskReplayBody) =>
  read<TaskCreationResponse>(`/tasks/${encodeURIComponent(id)}/replay`, {
    method: "POST",
    body: JSON.stringify(body),
  })
export const cancelDeclaredTask = (task: ApiTaskDetail, operationId: string) =>
  task.schedule
    ? read<unknown>("/tasks/schedule/cancel", {
        method: "POST",
        body: JSON.stringify({
          task_id: task.id,
          operation_id: operationId,
          expected_revision: task.schedule.revision,
        }),
      })
    : read<ApiTaskDetail>(`/tasks/${encodeURIComponent(task.id)}/cancel`, { method: "POST" })
