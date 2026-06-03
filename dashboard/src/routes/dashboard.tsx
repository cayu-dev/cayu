import { useQuery } from "@tanstack/react-query"
import { Link } from "@tanstack/react-router"
import { Activity, CheckCircle, Clock, XCircle } from "lucide-react"
import { Badge } from "../components/ui/badge"
import { buttonVariants } from "../components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card"
import { fetchSessions, fetchTasks } from "../lib/api"

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
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
        <Link to="/run" className={buttonVariants()}>
          New Run
        </Link>
      </div>

      <div className="grid grid-cols-4 gap-4">
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

      <div className="grid grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Recent Sessions</CardTitle>
          </CardHeader>
          <CardContent>
            {sessions.isError ? (
              <p className="text-sm text-destructive">
                {sessionsError || "Failed to load sessions."}
              </p>
            ) : list.length === 0 ? (
              <p className="text-sm text-muted-foreground">
                No sessions yet.{" "}
                <Link to="/run" className="text-primary underline">
                  Start a run
                </Link>
              </p>
            ) : (
              <div className="space-y-2">
                {list.slice(0, 8).map((s) => (
                  <Link
                    key={s.id}
                    to="/sessions/$sessionId"
                    params={{ sessionId: s.id }}
                    className="flex items-center justify-between p-3 rounded-md hover:bg-muted transition-colors no-underline text-foreground"
                  >
                    <div>
                      <div className="font-mono text-sm">{s.id}</div>
                      <div className="text-xs text-muted-foreground">
                        {new Date(s.created_at).toLocaleString()}
                      </div>
                    </div>
                    <StatusBadge status={s.status} />
                  </Link>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Recent Tasks</CardTitle>
          </CardHeader>
          <CardContent>
            {tasks.isError ? (
              <p className="text-sm text-destructive">{tasksError || "Failed to load tasks."}</p>
            ) : taskList.length === 0 ? (
              <p className="text-sm text-muted-foreground">No tasks yet.</p>
            ) : (
              <div className="space-y-2">
                {taskList.slice(0, 8).map((t) => (
                  <div
                    key={t.id}
                    className="flex items-center justify-between p-3 rounded-md bg-muted/50"
                  >
                    <div>
                      <div className="text-sm">{t.title || t.type}</div>
                      <div className="text-xs text-muted-foreground">
                        {new Date(t.created_at).toLocaleString()}
                      </div>
                    </div>
                    <StatusBadge status={t.status} />
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
