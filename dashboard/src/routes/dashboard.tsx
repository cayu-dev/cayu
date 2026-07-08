import { useQuery } from "@tanstack/react-query"
import { Link } from "@tanstack/react-router"
import { Activity, CheckCircle, Clock, XCircle } from "lucide-react"
import { DataCard, Page, PageHeader, StateMessage } from "../components/dashboard/layout"
import { Badge } from "../components/ui/badge"
import { buttonVariants } from "../components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card"
import { fetchSessions, fetchTasks } from "../lib/api"
import { formatDateTime } from "../lib/format"

function StatusBadge({ status }: { status: string }) {
  const variant =
    status === "completed"
      ? "default"
      : status === "running"
        ? "secondary"
        : status === "failed"
          ? "destructive"
          : "outline"
  return <Badge variant={variant}>{status}</Badge>
}

export function DashboardPage() {
  const sessions = useQuery({ queryKey: ["sessions"], queryFn: fetchSessions })
  const tasks = useQuery({ queryKey: ["tasks"], queryFn: fetchTasks })

  const list = sessions.data || []
  const taskList = tasks.data || []
  const sessionsError = sessions.error instanceof Error ? sessions.error.message : null
  const tasksError = tasks.error instanceof Error ? tasks.error.message : null
  const running = list.filter((s) => s.status === "running").length
  const completed = list.filter((s) => s.status === "completed").length
  const failed = list.filter((s) => s.status === "failed").length

  return (
    <Page>
      <PageHeader
        title="Dashboard"
        actions={
          <Link to="/run" className={buttonVariants()}>
            New Run
          </Link>
        }
      />

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {[
          { icon: Activity, label: "Total Sessions", value: list.length },
          { icon: Clock, label: "Running", value: running },
          { icon: CheckCircle, label: "Completed", value: completed },
          { icon: XCircle, label: "Failed", value: failed },
        ].map(({ icon: Icon, label, value }) => (
          <Card key={label}>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">{label}</CardTitle>
              <Icon className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{value}</div>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <DataCard title="Recent Sessions" contentClassName="p-4">
          {sessions.isError ? (
            <StateMessage tone="danger" className="py-6">
              {sessionsError || "Failed to load sessions."}
            </StateMessage>
          ) : list.length === 0 ? (
            <StateMessage className="py-6">
              No sessions yet.{" "}
              <Link to="/run" className="text-primary underline">
                Start a run
              </Link>
            </StateMessage>
          ) : (
            <div className="space-y-2">
              {list.slice(0, 8).map((s) => (
                <Link
                  key={s.id}
                  to="/sessions/$sessionId"
                  params={{ sessionId: s.id }}
                  className="flex items-center justify-between gap-3 rounded-md p-3 text-foreground no-underline transition-colors hover:bg-muted"
                >
                  <div className="min-w-0">
                    <div className="truncate font-mono text-sm">{s.id}</div>
                    <div className="text-xs text-muted-foreground">
                      {formatDateTime(s.created_at)}
                    </div>
                  </div>
                  <div className="shrink-0">
                    <StatusBadge status={s.status} />
                  </div>
                </Link>
              ))}
            </div>
          )}
        </DataCard>

        <DataCard title="Recent Tasks" contentClassName="p-4">
          {tasks.isError ? (
            <StateMessage tone="danger" className="py-6">
              {tasksError || "Failed to load tasks."}
            </StateMessage>
          ) : taskList.length === 0 ? (
            <StateMessage className="py-6">No tasks yet.</StateMessage>
          ) : (
            <div className="space-y-2">
              {taskList.slice(0, 8).map((t) => (
                <div
                  key={t.id}
                  className="flex items-center justify-between gap-3 rounded-md bg-muted/50 p-3"
                >
                  <div className="min-w-0">
                    <div className="truncate text-sm">{t.title || t.type}</div>
                    <div className="text-xs text-muted-foreground">
                      {formatDateTime(t.created_at)}
                    </div>
                  </div>
                  <div className="shrink-0">
                    <StatusBadge status={t.status} />
                  </div>
                </div>
              ))}
            </div>
          )}
        </DataCard>
      </div>
    </Page>
  )
}
