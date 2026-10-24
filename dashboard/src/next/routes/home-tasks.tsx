import { useQuery } from "@tanstack/react-query"
import { useDashboardCapability } from "@/components/dashboard/server-contract"
import { fetchTasks, type Task } from "@/lib/api"
import { dashboardCapabilityUnavailableText } from "@/lib/dashboard-capabilities"
import { activityPollInterval, taskStatusIsActive } from "@/lib/polling"
import { useTasks } from "../app/queries"
import { useServices } from "../app/services"
import { ErrorState, NavLink } from "./common"

const ATTENTION = new Set(["blocked", "needs_attention", "failed"])

export function HomeTasks() {
  const { prototype } = useServices()
  return prototype ? <PrototypeTasks /> : <RuntimeTasks />
}

function PrototypeTasks() {
  const tasks = useTasks()
  return (
    <TaskSections
      recent={(tasks.data ?? []).slice(0, 8)}
      attention={(tasks.data ?? []).filter((task) => ATTENTION.has(task.status)).slice(0, 8)}
    />
  )
}

function RuntimeTasks() {
  const capability = useDashboardCapability({ kind: "surface", surface: "tasks" })
  const tasks = useQuery({
    queryKey: ["home", "tasks"],
    enabled: capability.enabled,
    queryFn: async () => {
      const [recent, ...attentionPages] = await Promise.all([
        fetchTasks({ limit: 8, order_by: "updated_at_desc" }),
        ...(["blocked", "needs_attention", "failed"] as const).map((status) =>
          fetchTasks({ status, limit: 8, order_by: "updated_at_desc" }),
        ),
      ])
      return {
        recent,
        attention: attentionPages
          .flat()
          .sort((a, b) => b.updated_at.localeCompare(a.updated_at))
          .slice(0, 8),
      }
    },
    refetchInterval: (query) =>
      activityPollInterval(
        query.state.data?.recent.some((task) => taskStatusIsActive(task.status)) ?? false,
        5000,
      ),
  })
  if (!capability.enabled)
    return (
      <p className="mt-6 text-sm text-muted-foreground">
        Tasks: {dashboardCapabilityUnavailableText(capability)}
      </p>
    )
  if (tasks.error) return <ErrorState error={tasks.error} />
  if (!tasks.data) return <p role="status">Loading tasks…</p>
  return <TaskSections {...tasks.data} />
}

function TaskSections({ recent, attention }: { recent: Task[]; attention: Task[] }) {
  return (
    <>
      {[
        { label: "Tasks needing attention", tasks: attention },
        { label: "Recent tasks", tasks: recent },
      ].map(({ label, tasks }) => {
        return (
          <section key={label} className="mt-6 space-y-2">
            <div className="flex justify-between">
              <h2 className="text-sm font-medium">{label}</h2>
              <NavLink to="/tasks" className="text-xs text-muted-foreground">
                All tasks
              </NavLink>
            </div>
            <div className="divide-y rounded-xl border border-border bg-card">
              {tasks.length === 0 ? (
                <p className="p-3 text-sm text-muted-foreground">No tasks to show.</p>
              ) : (
                tasks.map((task) => (
                  <NavLink
                    key={task.id}
                    to="/tasks"
                    search={{ task_id: task.id }}
                    className="flex gap-3 px-3.5 py-2.5 text-sm hover:bg-muted/60"
                  >
                    <span className="min-w-0 flex-1 truncate">
                      {task.title || task.type || task.id}
                    </span>
                    <span className="text-xs text-muted-foreground">
                      {task.status_reason || task.status.replaceAll("_", " ")}
                    </span>
                  </NavLink>
                ))
              )}
            </div>
            {tasks.length === 8 && (
              <p className="text-xs text-muted-foreground">
                Showing the eight most recently updated tasks. Open All tasks for more.
              </p>
            )}
          </section>
        )
      })}
    </>
  )
}
