import { Bot, Play } from "lucide-react"
import { useSessionRows, useSpecialists } from "../app/queries"
import { useNow } from "../app/services"
import { relativeTime } from "../app/session-model"
import { ErrorState, NavLink, PageBody, TopBar } from "./common"

export function SpecialistsPage() {
  const specialists = useSpecialists()
  const sessions = useSessionRows()
  const now = useNow()
  const rows = sessions.data?.rows ?? []

  return (
    <div className="flex min-w-0 flex-1 flex-col">
      <TopBar>Specialists</TopBar>
      <PageBody width="max-w-3xl">
        <p className="text-sm text-muted-foreground">The specialists this agent registers.</p>
        {specialists.error && <ErrorState error={specialists.error} />}
        <div className="mt-4 space-y-3">
          {specialists.data?.agents.map((a) => {
            const mine = rows.filter((r) => r.session.agent_name === a.name)
            const last = mine[0]
            // Proposed (backend item 1): an app-set description, like a session's cayu:title.
            const description =
              typeof a.metadata["cayu:description"] === "string"
                ? a.metadata["cayu:description"]
                : null
            return (
              <div key={a.name} className="rounded-xl border border-border bg-card p-4">
                <div className="flex items-start gap-3">
                  <div className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-muted">
                    <Bot className="size-4" aria-hidden="true" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="font-medium">{a.name}</p>
                    {description && <p className="text-sm text-muted-foreground">{description}</p>}
                    <p className="mt-0.5 text-xs text-muted-foreground">
                      {a.model}
                      {a.provider_name ? ` via ${a.provider_name}` : ""} · {a.tool_count} tools ·{" "}
                      {mine.length
                        ? `${mine.length} recent sessions, last ${relativeTime(Date.parse(last?.overview?.last_activity_at ?? last?.session.updated_at ?? ""), now())}`
                        : "no sessions yet"}
                    </p>
                  </div>
                  <NavLink
                    to="/"
                    search={{ start: "1" }}
                    className="inline-flex h-7 shrink-0 items-center gap-1 rounded-lg border border-border px-2.5 text-[0.8rem] font-medium hover:bg-muted"
                  >
                    <Play className="size-3.5" aria-hidden="true" />
                    Start a session
                  </NavLink>
                </div>
                <div className="mt-3 flex flex-wrap gap-1.5">
                  {a.tools.map((t) => (
                    <span
                      key={t.name}
                      title={t.description}
                      className="rounded-md bg-muted px-2 py-0.5 font-mono text-xs"
                    >
                      {t.name}
                    </span>
                  ))}
                </div>
              </div>
            )
          })}
        </div>
      </PageBody>
    </div>
  )
}
