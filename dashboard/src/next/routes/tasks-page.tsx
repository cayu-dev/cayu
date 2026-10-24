/**
 * The agent's tasks: five plain groups with counts instead of twelve raw statuses, and one
 * line per task saying what it's doing or waiting for. Cayu's own housekeeping tasks
 * (`cayu.*` types) stay hidden unless asked for. A task opens its own page.
 */
import { useSearch } from "@tanstack/react-router"
import { Plus, Search } from "lucide-react"
import { useEffect, useState } from "react"
import { useViewer } from "@/components/dashboard/viewer"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import type { TaskItem } from "../api/types"
import { useSpecialists, useTaskTypes } from "../app/queries"
import { useNow, useServices } from "../app/services"
import { constantColumns, relativeTime } from "../app/session-model"
import { taskGroupCount } from "../app/task-index-model"
import {
  groupOf,
  isCayuTask,
  statusLine,
  TASK_GROUPS,
  type TaskGroup,
  taskName,
} from "../app/task-model"
import { type TaskFilters, useTaskIndex, useTaskLinks } from "../app/task-queries"
import {
  EmptyState,
  ErrorState,
  NavLink,
  PageBody,
  ROW_CLASS,
  ROW_LINK_CLASS,
  TopBar,
  taskPath,
  useGo,
  useRowClick,
} from "./common"
import { StartSession } from "./start-session"

type Filters = TaskFilters

export const GROUP_DOT: Record<TaskGroup, string> = {
  active: "bg-info animate-pulse motion-reduce:animate-none",
  waiting: "border border-muted-foreground/60 bg-transparent",
  held: "bg-warning",
  failed: "bg-destructive",
  done: "bg-muted-foreground/40",
}
export const GROUP_TEXT: Record<TaskGroup, string> = {
  active: "text-foreground",
  waiting: "text-muted-foreground",
  held: "text-warning",
  failed: "text-destructive",
  done: "text-muted-foreground",
}

const COLUMNS: Record<string, (t: TaskItem) => string | null> = {
  Specialist: (t) => t.assigned_agent_name,
}

export function TasksPage() {
  const search = useSearch({ strict: false }) as Filters
  const go = useGo()
  const rowClick = useRowClick()
  const now = useNow()
  const tasks = useTaskIndex(search)
  const services = useServices()
  const viewer = useViewer()
  const specialists = useSpecialists()
  const catalog = useTaskTypes()
  const [creating, setCreating] = useState(false)
  const [draft, setDraft] = useState(search.q ?? "")
  useEffect(() => setDraft(search.q ?? ""), [search.q])
  useEffect(() => {
    if (search.task_id) go(taskPath(search.task_id), undefined, { replace: true })
  }, [search.task_id, go])
  const set = (patch: Partial<Filters>) =>
    go("/tasks", { ...search, ...patch }, { replace: true, resetScroll: false })
  const all = [
    ...new Map(
      (tasks.data?.pages.flatMap((page) => page.items) ?? []).map((task) => [task.id, task]),
    ).values(),
  ]
  const failedDetails = useTaskLinks(
    all.filter((task) => task.status === "failed").map((task) => task.id),
  )
  const failedById = new Map((failedDetails.data?.rows ?? []).map((task) => [task.id, task]))
  const rows = all
  const showCayu = search.cayu === "1"
  const counts = tasks.data?.pages[0]?.counts_by_status
  const total = tasks.data?.pages[0]?.total_count
  const { constant, varying } = constantColumns(rows, COLUMNS)
  const types = [
    ...new Set([
      ...(catalog.data ?? []).map((type) => type.name),
      ...all.map((task) => task.type),
      ...(search.type ? [search.type] : []),
    ]),
  ]
  const agents = [
    ...new Set([
      ...(specialists.data?.agents ?? []).map((agent) => agent.name),
      ...(search.assigned_agent_name || search.agent
        ? [search.assigned_agent_name ?? search.agent!]
        : []),
    ]),
  ]
  const canCreate =
    !!services.prototype || (viewer?.full_access === true && services.taskCreationEnabled === true)
  const cycle =
    !!tasks.data?.pages.at(-1)?.next_cursor &&
    tasks.data.pageParams.includes(tasks.data.pages.at(-1)!.next_cursor!)
  const field =
    "h-8 rounded-lg border border-input bg-background px-2 text-sm outline-none focus:border-ring"

  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <TopBar
        actions={
          canCreate ? (
            <Button
              variant="outline"
              size="sm"
              onClick={() => setCreating(!creating)}
              aria-expanded={creating}
            >
              <Plus />
              Create a task
            </Button>
          ) : undefined
        }
      >
        Tasks
      </TopBar>
      <PageBody width="max-w-6xl">
        {creating && <StartSession initialMode="task" onClose={() => setCreating(false)} />}
        <fieldset className="flex min-w-0 flex-wrap gap-1.5">
          <legend className="sr-only">Filter by status</legend>
          {[{ id: undefined, label: "All" }, ...TASK_GROUPS].map((g) => {
            const n = counts ? taskGroupCount(counts, g.id) : "—"
            const on = search.group === g.id
            return (
              <button
                key={g.label}
                type="button"
                aria-pressed={on}
                onClick={() => set({ group: g.id })}
                className={cn(
                  "inline-flex h-7 items-center gap-1.5 rounded-full border px-3 text-[13px]",
                  on
                    ? "border-foreground bg-foreground text-background"
                    : "border-border hover:bg-muted",
                )}
              >
                {g.id && (
                  <span className={cn("size-2 rounded-full", GROUP_DOT[g.id])} aria-hidden="true" />
                )}
                {g.label}
                {
                  <span
                    className={cn(
                      "rounded-full px-1.5 text-xs font-medium tabular-nums",
                      on ? "bg-background/20" : "bg-muted text-muted-foreground",
                    )}
                  >
                    {n}
                  </span>
                }
              </button>
            )
          })}
        </fieldset>

        <div className="mt-3 flex flex-wrap items-center gap-2">
          <label className="flex h-8 w-full min-w-0 flex-none items-center gap-2 rounded-lg border border-input bg-background px-2.5 text-sm focus-within:border-ring sm:max-w-sm sm:min-w-56 sm:flex-[1_1_14rem]">
            <Search className="size-4 shrink-0 text-muted-foreground" aria-hidden="true" />
            <span className="sr-only">Search tasks</span>
            <input
              id="tasks-search"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") set({ q: e.currentTarget.value || undefined })
              }}
              onBlur={(e) => set({ q: e.currentTarget.value || undefined })}
              placeholder="Search by title, ID, session or reason"
              className="grouped-field-input w-full min-w-0 bg-transparent outline-none placeholder:text-muted-foreground"
            />
          </label>
          {
            <select
              id="tasks-type"
              aria-label="Task type"
              className={field}
              value={search.type ?? ""}
              onChange={(e) => set({ type: e.target.value || undefined })}
            >
              <option value="">All types</option>
              {types.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          }
          {
            <select
              id="tasks-agent"
              aria-label="Specialist"
              className={field}
              value={search.assigned_agent_name ?? search.agent ?? ""}
              onChange={(e) =>
                set({ assigned_agent_name: e.target.value || undefined, agent: undefined })
              }
            >
              <option value="">All specialists</option>
              {agents.map((a) => (
                <option key={a} value={a}>
                  {a}
                </option>
              ))}
            </select>
          }
          {
            <label className="ml-auto flex items-center gap-2 text-[13px] text-muted-foreground">
              <input
                id="tasks-cayu"
                type="checkbox"
                checked={showCayu}
                onChange={(e) => set({ cayu: e.target.checked ? "1" : undefined })}
                className="size-3.5 accent-foreground"
              />
              Show Cayu's own tasks
            </label>
          }
        </div>

        {Object.keys(constant).length > 0 && rows.length > 1 && (
          <p className="pt-3 text-xs text-muted-foreground">
            Every task shown is handled by {Object.values(constant)[0]}.
          </p>
        )}

        <Button
          className="mt-2"
          variant="ghost"
          onClick={() => {
            setDraft("")
            go("/tasks", {}, { replace: true, resetScroll: false })
          }}
        >
          Clear filters
        </Button>
        {!!failedDetails.data?.issues.length && (
          <p role="status">
            Some task details are unavailable; their status lines omit detail-specific information.
          </p>
        )}
        {tasks.error && (
          <div className="mt-3">
            <ErrorState error={tasks.error} />
          </div>
        )}
        {tasks.isSuccess && rows.length === 0 ? (
          <div className="mt-4">
            <EmptyState title="No tasks match these filters">
              Tasks appear when the agent’s code creates them. Try changing the filters.
            </EmptyState>
          </div>
        ) : (
          <div className="mt-3 overflow-x-auto rounded-xl border border-border">
            <table className="w-full min-w-[680px] text-sm">
              <thead>
                <tr className="border-b border-border text-left text-xs text-muted-foreground">
                  <th className="px-3 py-2 font-medium">Task</th>
                  <th className="px-3 py-2 font-medium">Status</th>
                  {varying.map((c) => (
                    <th key={c} className="px-3 py-2 font-medium">
                      {c}
                    </th>
                  ))}
                  <th className="px-3 py-2 text-right font-medium">Updated</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((t) => {
                  const group = groupOf(t.status)
                  return (
                    <tr key={t.id} onClick={rowClick(taskPath(t.id))} className={ROW_CLASS}>
                      <td className="max-w-[26rem] px-3 py-2">
                        <div className="flex min-w-0 items-center gap-2">
                          <span
                            className={cn("size-2 shrink-0 rounded-full", GROUP_DOT[group])}
                            aria-hidden="true"
                          />
                          <NavLink to={taskPath(t.id)} className={cn("truncate", ROW_LINK_CLASS)}>
                            {taskName(t)}
                          </NavLink>
                          {isCayuTask(t) && (
                            <span className="shrink-0 rounded border border-border px-1 text-[10px] text-muted-foreground">
                              Cayu
                            </span>
                          )}
                        </div>
                        <div className="mt-0.5 truncate pl-4 font-mono text-[11px] text-muted-foreground">
                          {t.type}
                        </div>
                      </td>
                      <td className={cn("max-w-[22rem] px-3 py-2", GROUP_TEXT[group])}>
                        <span className="line-clamp-2">
                          {statusLine(failedById.get(t.id) ?? t, all, now(), undefined, false)}
                        </span>
                      </td>
                      {varying.map((c) => (
                        <td key={c} className="px-3 py-2 whitespace-nowrap text-muted-foreground">
                          {COLUMNS[c]?.(t) ?? "–"}
                        </td>
                      ))}
                      <td className="px-3 py-2 text-right whitespace-nowrap tabular-nums text-muted-foreground">
                        {relativeTime(Date.parse(t.updated_at), now())}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
        {tasks.isPending && (
          <p role="status" className="mt-4">
            Loading tasks…
          </p>
        )}
        {tasks.isSuccess && !counts && (
          <p role="status" className="mt-3 text-sm text-muted-foreground">
            Exact counts are unavailable for this task store. This list shows loaded records only.
          </p>
        )}
        {cycle && <p role="alert">Task paging did not advance. Refresh the list to continue.</p>}
        {tasks.hasNextPage && (
          <Button
            className="mt-4"
            variant="outline"
            disabled={tasks.isFetching}
            onClick={() => void tasks.fetchNextPage()}
          >
            Load more tasks
          </Button>
        )}
        {tasks.isSuccess && (
          <p className="mt-3 text-xs text-muted-foreground">
            {rows.length} loaded{total !== null && total !== undefined ? ` of ${total}` : ""}
          </p>
        )}
      </PageBody>
    </div>
  )
}
