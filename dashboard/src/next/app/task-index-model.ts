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
