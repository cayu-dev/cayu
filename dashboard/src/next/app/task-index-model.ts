import type { ApiTaskIndex } from "../../lib/generated/server-api/index.ts"
import { TASK_GROUPS } from "./task-model.ts"

export function taskGroupCount(counts: ApiTaskIndex["counts_by_status"], group?: string): string {
  const statuses = group ? taskGroupStatuses(group) : Object.keys(counts)
  return statuses
    .reduce((sum, status) => sum + BigInt(counts[status as keyof typeof counts] ?? "0"), 0n)
    .toString()
}
export function taskGroupStatuses(group?: string): string[] {
  return TASK_GROUPS.find((item) => item.id === group)?.statuses ?? []
}
/** The `type` filter: one type, or several separated by commas (`?type=pass,recheck`). */
export function taskTypeFilter(type?: string): { type?: string; types?: string[] } {
  const types = [...new Set((type ?? "").split(",").map((item) => item.trim()))].filter(Boolean)
  if (types.length > 1) return { types }
  return types.length ? { type: types[0] } : {}
}
