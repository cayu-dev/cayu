import { useQuery } from "@tanstack/react-query"
import { useSearch } from "@tanstack/react-router"
import { Bot, ChevronDown, Play, Search } from "lucide-react"
import { useState } from "react"
import { useViewer } from "@/components/dashboard/viewer"
import type { AgentSummary } from "@/lib/api"
import { useSpecialists } from "../app/queries"
import { useApi, useNow, useServices } from "../app/services"
import { relativeTime } from "../app/session-model"
import { TASK_POLL_MS } from "../app/task-polling"
import {
  EmptyState,
  ErrorState,
  NavLink,
  PageBody,
  sessionRoute,
  TopBar,
  taskRoute,
  useGo,
} from "./common"

export function SpecialistsPage({
  recentEnabled = true,
  tasksEnabled = true,
}: {
  recentEnabled?: boolean
  tasksEnabled?: boolean
} = {}) {
  const specialists = useSpecialists()
  const search = useSearch({ strict: false }) as { q?: string }
  const go = useGo()
  const q = search.q ?? ""
  const needle = q.trim().toLocaleLowerCase()
  const rows = (specialists.data?.agents ?? []).filter((a) =>
    [
      a.name,
      a.model,
      a.provider_name,
      a.metadata["cayu:description"],
      ...a.tools.flatMap((t) => [t.name, t.effect]),
    ].some((value) => typeof value === "string" && value.toLocaleLowerCase().includes(needle)),
  )
  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <TopBar>Specialists</TopBar>
      <PageBody width="max-w-3xl">
        <p className="text-sm text-muted-foreground">The specialists this agent registers.</p>
        <label className="mt-4 flex h-8 max-w-sm min-w-0 items-center gap-2 rounded-lg border border-input bg-background px-2.5 text-sm focus-within:border-ring">
          <Search className="size-4 shrink-0 text-muted-foreground" aria-hidden="true" />
          <span className="sr-only">Search specialists</span>
          <input
            value={q}
            onChange={(e) =>
              go({
                to: "/specialists",
                search: { q: e.target.value || undefined },
                replace: true,
                resetScroll: false,
              })
            }
            placeholder="Search specialists, models or tools"
            className="grouped-field-input min-w-0 w-full bg-transparent outline-none"
          />
        </label>
        {specialists.isPending && (
          <p role="status" className="mt-4 text-sm text-muted-foreground">
            Loading specialists…
          </p>
        )}
        {specialists.error && <ErrorState error={specialists.error} />}
        {specialists.isSuccess && !rows.length && (
          <EmptyState title={q ? "No specialists match" : "No specialists"} />
        )}
        <div className="mt-4 space-y-3">
          {publicOccurrences(rows).map(({ value: agent, key }) => (
            <SpecialistCard
              key={key}
              agent={agent}
              recentEnabled={recentEnabled}
              tasksEnabled={tasksEnabled}
            />
          ))}
        </div>
      </PageBody>
    </div>
  )
}

function SpecialistCard({
  agent,
  recentEnabled,
  tasksEnabled,
}: {
  agent: AgentSummary
  recentEnabled: boolean
  tasksEnabled: boolean
}) {
  const [open, setOpen] = useState(false)
  const api = useApi()
  const now = useNow()
  const { prototype, executionEnabled } = useServices()
  const viewer = useViewer()
  const recent = useQuery({
    queryKey: ["specialist-recent", agent.name],
    enabled: recentEnabled,
    refetchInterval: TASK_POLL_MS,
    queryFn: ({ signal }) =>
      api.sessions(
        {
          agent_name: agent.name,
          limit: 5,
          order_by: "updated_at_desc",
          include_eval: true,
          include_forks: true,
        },
        signal,
      ),
  })
  const tasks = useQuery({
    queryKey: ["specialist-tasks", agent.name],
    enabled: open && tasksEnabled,
    refetchInterval: TASK_POLL_MS,
    queryFn: ({ signal }) =>
      api.taskList(
        { assigned_agent_name: agent.name, limit: 5, order_by: "updated_at_desc" },
        signal,
      ),
  })
  const last = recent.data?.sessions[0]
  const description = agent.metadata["cayu:description"]
  return (
    <article className="min-w-0 rounded-xl border border-border bg-card p-4 [overflow-wrap:anywhere]">
      <div className="flex flex-wrap items-start gap-3">
        <div className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-muted">
          <Bot className="size-4" aria-hidden="true" />
        </div>
        <div className="min-w-0 flex-[1_1_10rem]">
          <button
            type="button"
            aria-label={`Inspect ${agent.name}`}
            aria-expanded={open}
            onClick={() => setOpen(!open)}
            className="flex max-w-full items-center gap-2 text-left font-medium hover:underline"
          >
            {agent.name}
            <ChevronDown className="size-4 shrink-0" aria-hidden="true" />
          </button>
          {typeof description === "string" && (
            <p className="text-sm text-muted-foreground">{description}</p>
          )}
          <p className="mt-0.5 text-xs text-muted-foreground">
            {agent.model}
            {agent.provider_name ? ` via ${agent.provider_name}` : ""} · {agent.tool_count}{" "}
            {agent.tool_count === 1 ? "tool" : "tools"}
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            {!recentEnabled
              ? "Recent activity unavailable"
              : recent.isPending
                ? "Loading recent activity…"
                : recent.error
                  ? "Recent activity unavailable"
                  : last
                    ? `Last active ${relativeTime(Date.parse(last.updated_at), now())}`
                    : "No recent sessions found"}
          </p>
        </div>
        {(prototype || (viewer?.full_access && executionEnabled)) && (
          <NavLink
            to="/"
            search={{ start: "1" }}
            className="ml-12 inline-flex h-7 shrink-0 items-center gap-1 rounded-lg border border-border px-2.5 text-[0.8rem] font-medium hover:bg-muted sm:ml-0"
          >
            <Play className="size-3.5" aria-hidden="true" /> Start a session
          </NavLink>
        )}
      </div>
      <div className="mt-3 flex flex-wrap gap-1.5">
        {publicOccurrences(agent.tools).map(({ value: tool, key }) => (
          <NavLink
            key={key}
            to="/tools"
            search={{ q: tool.name }}
            className="max-w-full rounded-md bg-muted px-2 py-0.5 font-mono text-xs hover:underline"
          >
            {tool.name}
          </NavLink>
        ))}
      </div>
      {open && (
        <div className="mt-4 space-y-4 border-t border-border pt-4">
          <h2 className="text-sm font-medium">Configuration</h2>
          <dl className="grid grid-cols-[minmax(0,7rem)_minmax(0,1fr)] gap-3 text-sm">
            <dt>Provider</dt>
            <dd>{agent.provider_name ?? "Default provider"}</dd>
            <dt>Model</dt>
            <dd>{agent.model}</dd>
            <dt>System prompt</dt>
            <dd>{agent.has_system_prompt ? "configured" : "Not configured"}</dd>
            <dt>Metadata</dt>
            <dd className="min-w-0">
              <Payload value={agent.metadata} />
            </dd>
            <dt>Provider options</dt>
            <dd className="min-w-0">
              <Payload value={agent.provider_options} />
            </dd>
            <dt>Thinking</dt>
            <dd className="min-w-0">
              {agent.thinking ? <Payload value={agent.thinking} /> : "Not configured"}
            </dd>
          </dl>
          <section>
            <h3 className="text-sm font-medium">Recent sessions</h3>
            <p className="text-xs text-muted-foreground">
              Latest five sessions, including eval trials and forks.
            </p>
            {recent.error && <ErrorState error={recent.error} />}
            {recent.data?.sessions.map((session) => (
              <NavLink
                key={session.id}
                {...sessionRoute(session.id)}
                className="mt-1 block rounded px-1 py-1 text-sm hover:bg-muted"
              >
                {recent.data?.overviews?.find((o) => o.session_id === session.id)?.title ??
                  session.id}
              </NavLink>
            ))}
            <NavLink
              to="/sessions"
              search={{ agent_name: agent.name, include_eval: true, include_forks: true }}
              className="mt-2 inline-block text-sm underline"
            >
              All sessions
            </NavLink>
          </section>
          <section>
            <h3 className="text-sm font-medium">Related tasks</h3>
            {!tasksEnabled ? (
              <p className="text-sm text-muted-foreground">Tasks are unavailable.</p>
            ) : tasks.isPending ? (
              <p role="status">Loading tasks…</p>
            ) : tasks.error ? (
              <ErrorState error={tasks.error} />
            ) : tasks.data?.length ? (
              tasks.data.map((task) => (
                <NavLink
                  key={task.id}
                  {...taskRoute(task.id)}
                  className="mt-1 block rounded px-1 py-1 text-sm hover:bg-muted"
                >
                  {task.title || task.id}
                </NavLink>
              ))
            ) : (
              <p className="text-sm text-muted-foreground">No tasks assigned.</p>
            )}
            {tasksEnabled && (
              <NavLink
                to="/tasks"
                search={{ assigned_agent_name: agent.name }}
                className="mt-2 inline-block text-sm underline"
              >
                All tasks
              </NavLink>
            )}
          </section>
        </div>
      )}
    </article>
  )
}

function Payload({ value }: { value: unknown }) {
  return (
    <pre className="max-h-56 overflow-auto rounded-md bg-muted p-2 font-mono text-xs">
      {JSON.stringify(value, null, 2)}
    </pre>
  )
}

/** Public redaction may collapse names; retain every indistinguishable occurrence. */
function publicOccurrences<T>(values: T[]) {
  const seen = new Map<string, number>()
  return values.map((value) => {
    const identity = JSON.stringify(value)
    const occurrence = seen.get(identity) ?? 0
    seen.set(identity, occurrence + 1)
    return { value, key: `${identity}:${occurrence}` }
  })
}
