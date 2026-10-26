/**
 * Every tool the agent's specialists can call: where it comes from, what it can change,
 * whether it asks a person first, and how it has been used, with links into the calls.
 */
import { useInfiniteQuery, useQuery, useQueryClient } from "@tanstack/react-query"
import { useSearch } from "@tanstack/react-router"
import { ChevronRight, Search } from "lucide-react"
import { Fragment, useMemo, useState } from "react"
import { cn } from "@/lib/utils"
import type { ToolApproval, ToolSource, ToolUsage, ToolWithDisplay } from "../api/types"
import { useSpecialists, useToolUsage } from "../app/queries"
import { useApi, useNow } from "../app/services"
import { relativeTime } from "../app/session-model"
import { TASK_POLL_MS } from "../app/task-polling"
import { EmptyState, ErrorState, NavLink, PageBody, sessionPath, TopBar, useGo } from "./common"

type ToolRow = {
  key: string
  tool: ToolWithDisplay
  specialists: string[]
  usage: ToolUsage | null
}

/** The runtime's declared side effects (cayu.tools.base.ToolEffect). */
const EFFECT: Record<string, { label: string; tone: string; title?: string }> = {
  none: { label: "None", tone: "text-muted-foreground", title: "Doesn't change durable state" },
  idempotent: {
    label: "Idempotent",
    tone: "",
    title: "Changes state, but a replay collapses into one",
  },
  external: {
    label: "External",
    tone: "text-warning",
    title: "Changes state in a way a replay can repeat",
  },
}

const SOURCE: Record<ToolSource["kind"], string> = {
  cayu: "Cayu built-in",
  agent: "Agent code",
  package: "Package",
  mcp: "MCP server",
  provider: "Provider",
  unknown: "Unknown",
}

/** The source's kind, then what names it: the package and version, the server, the provider. */
function sourceText(source: ToolSource | null | undefined): { label: string; detail?: string } {
  if (!source) return { label: "–" }
  const label = SOURCE[source.kind] ?? "Unknown"
  if (source.kind === "package" && source.package)
    return { label, detail: [source.package, source.version].filter(Boolean).join(" ") }
  if (source.kind === "mcp" && source.server) return { label, detail: source.server }
  if (source.kind === "provider" && source.provider) return { label, detail: source.provider }
  return { label }
}

/** Whether the tool asks a person before it runs (Cloud inventory's `approval`). */
const APPROVAL: Record<ToolApproval, { label: string; tone: string; title: string }> = {
  always: { label: "Asks first", tone: "text-warning", title: "Every call waits for a person" },
  sometimes: {
    label: "Sometimes asks",
    tone: "text-warning",
    title: "A policy decides per call whether a person approves it",
  },
  never: { label: "Never asks", tone: "text-muted-foreground", title: "Runs without approval" },
  blocked: { label: "Blocked", tone: "text-destructive", title: "Policy denies every call" },
  unknown: { label: "Unknown", tone: "text-muted-foreground", title: "Not reported" },
}

export function ToolsPage({ usageEnabled = true }: { usageEnabled?: boolean } = {}) {
  const specialists = useSpecialists()
  const usage = useToolUsage(usageEnabled)
  const now = useNow()
  const search = useSearch({ strict: false }) as { q?: string }
  const q = search.q ?? ""
  const go = useGo()
  const setQ = (value: string) =>
    go("/tools", { q: value || undefined }, { replace: true, resetScroll: false })
  const [open, setOpen] = useState<string | null>(null)

  const rows = useMemo(() => {
    const byName = new Map<string, ToolRow>()
    for (const agent of specialists.data?.agents ?? []) {
      for (const tool of agent.tools as ToolWithDisplay[]) {
        const key = JSON.stringify(tool)
        const row = byName.get(key) ?? { key, tool, specialists: [], usage: null }
        row.specialists.push(agent.name)
        byName.set(key, row)
      }
    }
    const usageByName = new Map((usage.data ?? []).map((item) => [item.tool, item]))
    for (const row of byName.values()) row.usage = usageByName.get(row.tool.name) ?? null
    const needle = q.trim().toLowerCase()
    return [...byName.values()]
      .filter(
        (r) =>
          !needle ||
          `${r.tool.name} ${r.tool.description} ${r.specialists.join(" ")}`
            .toLowerCase()
            .includes(needle),
      )
      .sort(
        (a, b) =>
          (b.usage?.calls ?? 0) - (a.usage?.calls ?? 0) || a.tool.name.localeCompare(b.tool.name),
      )
  }, [specialists.data, usage.data, q])

  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <TopBar>Tools</TopBar>
      <PageBody width="max-w-5xl">
        <p className="text-sm text-muted-foreground">
          Every tool this agent's specialists can call.
        </p>
        <label className="mt-4 flex h-8 max-w-sm min-w-0 items-center gap-2 rounded-lg border border-input bg-background px-2.5 text-sm focus-within:border-ring">
          <Search className="size-4 shrink-0 text-muted-foreground" aria-hidden="true" />
          <span className="sr-only">Search tools</span>
          <input
            id="tools-search"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search by name, description or specialist"
            className="grouped-field-input w-full min-w-0 bg-transparent outline-none placeholder:text-muted-foreground"
          />
        </label>
        {specialists.isPending && (
          <p role="status" className="mt-4 text-sm text-muted-foreground">
            Loading tools…
          </p>
        )}
        {!usageEnabled && (
          <p className="mt-3 text-sm text-muted-foreground">
            Tool usage is unavailable for this deployment. Declarations remain inspectable.
          </p>
        )}
        {usageEnabled && usage.isPending && (
          <p role="status" className="mt-3 text-sm text-muted-foreground">
            Loading usage…
          </p>
        )}
        {(specialists.error || usage.error) && (
          <div className="mt-3">
            <ErrorState error={specialists.error ?? usage.error} />
          </div>
        )}
        {specialists.isSuccess && rows.length === 0 ? (
          <div className="mt-4">
            <EmptyState title={q ? "No tools match" : "No tools"} />
          </div>
        ) : (
          <div className="mt-3 overflow-x-auto rounded-xl border border-border">
            <table className="w-full min-w-[820px] text-sm">
              <thead>
                <tr className="border-b border-border text-left text-xs text-muted-foreground">
                  <th className="px-3 py-2 font-medium">Tool</th>
                  <th className="px-3 py-2 font-medium">Source</th>
                  <th className="px-3 py-2 font-medium">Effect</th>
                  <th className="px-3 py-2 font-medium">Approval</th>
                  <th className="px-3 py-2 font-medium">Used by</th>
                  <th className="px-3 py-2 text-right font-medium">Calls</th>
                  <th className="px-3 py-2 text-right font-medium">Failed</th>
                  <th className="px-3 py-2 text-right font-medium">Last used</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => {
                  const expanded = open === r.key
                  const effect = EFFECT[r.tool.effect] ?? { label: r.tool.effect, tone: "" }
                  const source = sourceText(r.tool.source)
                  const approval = r.tool.approval ? APPROVAL[r.tool.approval] : null
                  return (
                    <Fragment key={r.key}>
                      <tr
                        className={cn(
                          "cursor-pointer border-b border-border last:border-0 hover:bg-muted/60 focus-visible:bg-muted/60 focus-visible:outline-none",
                          expanded && "bg-muted/60",
                        )}
                      >
                        <td className="max-w-[18rem] px-3 py-2">
                          <div className="flex items-center gap-1.5">
                            <ChevronRight
                              className={cn(
                                "size-3.5 shrink-0 text-muted-foreground transition-transform",
                                expanded && "rotate-90",
                              )}
                              aria-hidden="true"
                            />
                            <button
                              type="button"
                              aria-label={`Inspect ${r.tool.name}`}
                              aria-expanded={expanded}
                              onClick={() => setOpen(expanded ? null : r.key)}
                              className="min-w-0 text-left font-mono text-[13px] hover:underline [overflow-wrap:anywhere]"
                            >
                              {r.tool.name}
                            </button>
                          </div>
                          <div className="truncate pl-5 text-xs text-muted-foreground">
                            {r.tool.description}
                          </div>
                        </td>
                        <td className="max-w-[12rem] px-3 py-2 text-muted-foreground">
                          <div className="whitespace-nowrap">{source.label}</div>
                          {source.detail && (
                            <div className="truncate font-mono text-[11px]">{source.detail}</div>
                          )}
                        </td>
                        <td
                          className={cn("px-3 py-2 whitespace-nowrap", effect.tone)}
                          title={effect.title}
                        >
                          {effect.label}
                        </td>
                        <td
                          className={cn("px-3 py-2 whitespace-nowrap", approval?.tone)}
                          title={approval?.title}
                        >
                          {approval?.label ?? "–"}
                        </td>
                        <td className="max-w-[12rem] truncate px-3 py-2 text-muted-foreground">
                          {r.specialists.join(", ")}
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums">
                          {r.usage?.calls ?? "–"}
                        </td>
                        <td
                          className={cn(
                            "px-3 py-2 text-right tabular-nums",
                            (r.usage?.failed ?? 0) > 0 && "text-destructive",
                          )}
                        >
                          {r.usage?.failed ?? "–"}
                        </td>
                        <td className="px-3 py-2 text-right whitespace-nowrap text-muted-foreground tabular-nums">
                          {r.usage?.last_used_at
                            ? relativeTime(Date.parse(r.usage.last_used_at), now())
                            : r.usage
                              ? "never"
                              : "–"}
                        </td>
                      </tr>
                      {expanded && (
                        <tr className="border-b border-border bg-muted/30">
                          <td colSpan={8} className="px-4 py-4">
                            <ToolDetail row={r} now={now()} usageEnabled={usageEnabled} />
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </PageBody>
    </div>
  )
}

function ToolDetail({
  row,
  now,
  usageEnabled,
}: {
  row: ToolRow
  usageEnabled: boolean
  now: number
}) {
  const api = useApi()
  const { tool, usage } = row
  const client = useQueryClient()
  const calls = useInfiniteQuery({
    queryKey: ["tool-calls", tool.name],
    enabled: usageEnabled && !!api.toolCalls,
    initialPageParam: undefined as string | undefined,
    queryFn: ({ pageParam, signal }) => api.toolCalls!(tool.name, pageParam, signal),
    getNextPageParam: (page) => page.next_cursor ?? undefined,
    // Only the first window is live. Explicit older pages retain their snapshot.
    refetchInterval: (query) => ((query.state.data?.pages.length ?? 0) <= 1 ? TASK_POLL_MS : false),
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
    refetchOnMount: false,
    retry: false,
  })
  const historical = (calls.data?.pages.length ?? 0) > 1
  const latest = useQuery({
    queryKey: ["tool-calls-latest", tool.name],
    enabled: usageEnabled && !!api.toolCalls && historical,
    queryFn: ({ signal }) => api.toolCalls!(tool.name, undefined, signal),
    refetchInterval: TASK_POLL_MS,
    retry: false,
  })
  const visibleCalls = [
    ...new Map(
      [
        ...(historical ? (latest.data?.calls ?? []) : []),
        ...(calls.data?.pages.flatMap((p) => p.calls) ?? []),
      ].map((call) => [`${call.session_instance_id}:${call.sequence}`, call]),
    ).values(),
  ]
  const approvals = usage?.approvals
  const source = sourceText(tool.source)
  return (
    <div className="grid gap-5 [overflow-wrap:anywhere] lg:grid-cols-2">
      <div className="min-w-0 space-y-3">
        <p className="text-xs text-muted-foreground">
          Source: {source.label}
          {source.detail ? ` · ${source.detail}` : ""}
        </p>
        <p className="text-xs text-muted-foreground">
          Approval policy:{" "}
          {tool.approval ? (APPROVAL[tool.approval]?.label ?? "Unknown") : "Unknown"}
        </p>
        <div>
          <p className="text-xs text-muted-foreground">
            {row.specialists.length} published{" "}
            {row.specialists.length === 1 ? "declaration" : "declarations"} with this configuration.
          </p>
          <p className="text-xs font-medium text-muted-foreground">Description</p>
          <p className="mt-1">{tool.description || "–"}</p>
        </div>
        {tool.display && (
          <p className="text-xs text-muted-foreground">
            Reads in a session as “{tool.display.verb}
            {tool.display.noun ? ` 3 ${tool.display.plural ?? `${tool.display.noun}s`}` : ""}”.
          </p>
        )}
        <div>
          <p className="text-xs font-medium text-muted-foreground">Input schema</p>
          <pre className="mt-1 max-h-56 overflow-auto rounded-md bg-background p-2 font-mono text-[11px] leading-relaxed">
            {JSON.stringify(tool.input_schema, null, 2)}
          </pre>
        </div>
        <p className="text-xs text-muted-foreground">
          Effect <span className="font-mono">{tool.effect}</span>
          {tool.parallel_safe ? " · can run in parallel" : ""}
          {tool.workspace_mutation ? " · changes the workspace" : ""}
        </p>
      </div>
      <div className="min-w-0 space-y-3">
        {approvals && (
          <div>
            <p className="text-xs font-medium text-muted-foreground">Approvals</p>
            <p className="mt-1 tabular-nums">
              {approvals.approved} approved · {approvals.denied} denied
              {approvals.waiting != null ? ` · ${approvals.waiting} waiting` : ""}
            </p>
          </div>
        )}
        <div>
          <p className="text-xs font-medium text-muted-foreground">Recent calls</p>
          <p className="mt-1 text-xs text-muted-foreground">
            Usage is shared by declarations with this public tool name.
          </p>
          {api.toolCalls ? (
            !usageEnabled ? (
              <p className="text-muted-foreground">Recent calls unavailable.</p>
            ) : calls.isPending ? (
              <p role="status">Loading recent calls…</p>
            ) : calls.error ? (
              <div>
                <ErrorState error={calls.error} />
                <button
                  type="button"
                  onClick={() => void calls.refetch()}
                  className="mt-2 rounded-lg border px-3 py-1 text-sm"
                >
                  Refresh calls
                </button>
              </div>
            ) : (
              <>
                {historical && (
                  <div className="mt-2 text-xs text-muted-foreground">
                    Latest calls update automatically. Older loaded calls are a snapshot; newer
                    calls between these windows may not be loaded.
                    <button
                      type="button"
                      className="ml-1 underline"
                      onClick={() =>
                        void client.resetQueries({
                          queryKey: ["tool-calls", tool.name],
                          exact: true,
                        })
                      }
                    >
                      Start from latest calls
                    </button>
                    {latest.error && <ErrorState error={latest.error} />}
                  </div>
                )}
                {visibleCalls.length === 0 && (
                  <p className="mt-1 text-muted-foreground">
                    {calls.hasNextPage ? "No calls in this page of history." : "Not called yet."}
                  </p>
                )}
                <ul className="mt-2 space-y-1">
                  {visibleCalls.map((call) => (
                    <li key={`${call.session_instance_id}:${call.sequence}`}>
                      <NavLink
                        to={sessionPath(call.session_id)}
                        search={{
                          view: "trace",
                          log: "1",
                          event_id: call.event_id,
                          session_instance_id: call.session_instance_id,
                        }}
                        className="block rounded px-1 py-1 text-sm hover:bg-background [overflow-wrap:anywhere]"
                      >
                        {call.session_id} · {call.outcome.replaceAll("_", " ")} ·{" "}
                        {relativeTime(Date.parse(call.timestamp), now)}
                      </NavLink>
                    </li>
                  ))}
                </ul>
                {calls.hasNextPage && (
                  <button
                    type="button"
                    disabled={calls.isFetchingNextPage}
                    onClick={() => void calls.fetchNextPage()}
                    className="mt-2 rounded-lg border border-border px-3 py-1 text-sm"
                  >
                    {calls.isFetchingNextPage ? "Loading…" : "Load older calls"}
                  </button>
                )}
              </>
            )
          ) : !usage || usage.recent.length === 0 ? (
            <p className="mt-1 text-muted-foreground">Not called yet.</p>
          ) : (
            <ul className="mt-1 space-y-0.5">
              {usage.recent.map((c) => (
                <li key={`${c.session_id}-${c.at}`}>
                  <NavLink
                    to={sessionPath(c.session_id)}
                    search={{ view: "trace" }}
                    className="flex items-center gap-2 rounded px-1 py-0.5 hover:bg-background"
                  >
                    <span
                      className={cn(
                        "inline-block size-2 shrink-0 rounded-full",
                        c.status === "error"
                          ? "bg-destructive"
                          : c.status === "waiting"
                            ? "bg-warning"
                            : "bg-success",
                      )}
                      aria-hidden="true"
                    />
                    <span className="min-w-0 flex-1 truncate">{c.session_id}</span>
                    <span className="shrink-0 text-xs text-muted-foreground tabular-nums">
                      {relativeTime(Date.parse(c.at), now)}
                    </span>
                  </NavLink>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  )
}
