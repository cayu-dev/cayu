/**
 * Every tool the agent's specialists can call: where it comes from, what it can change,
 * whether it asks a person first, and how it has been used, with links into the calls.
 */
import { ChevronRight, Search } from "lucide-react"
import { Fragment, useMemo, useState } from "react"
import { cn } from "@/lib/utils"
import type { ToolApproval, ToolSource, ToolUsage, ToolWithDisplay } from "../api/types"
import { useSessionRows, useSpecialists, useToolUsage } from "../app/queries"
import { useNow } from "../app/services"
import { relativeTime } from "../app/session-model"
import { EmptyState, ErrorState, NavLink, PageBody, sessionPath, TopBar } from "./common"

type ToolRow = { tool: ToolWithDisplay; specialists: string[]; usage: ToolUsage | null }

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
const asksPeople = (a: ToolApproval | null | undefined) => a === "always" || a === "sometimes"

export function ToolsPage() {
  const specialists = useSpecialists()
  const usage = useToolUsage()
  const sessions = useSessionRows()
  const now = useNow()
  const [q, setQ] = useState("")
  const [open, setOpen] = useState<string | null>(null)

  const rows = useMemo(() => {
    const byName = new Map<string, ToolRow>()
    for (const agent of specialists.data?.agents ?? []) {
      for (const tool of agent.tools as ToolWithDisplay[]) {
        const row = byName.get(tool.name) ?? { tool, specialists: [], usage: null }
        row.specialists.push(agent.name)
        byName.set(tool.name, row)
      }
    }
    for (const u of usage.data ?? []) {
      const row = byName.get(u.tool)
      if (row) row.usage = u
    }
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
  const titles = new Map(
    (sessions.data?.rows ?? []).map((r) => [r.session.id, r.overview?.title ?? r.session.id]),
  )

  return (
    <div className="flex min-w-0 flex-1 flex-col">
      <TopBar>Tools</TopBar>
      <PageBody width="max-w-5xl">
        <p className="text-sm text-muted-foreground">
          Every tool this agent's specialists can call.
        </p>
        <label className="mt-4 flex h-8 max-w-sm items-center gap-2 rounded-lg border border-input bg-background px-2.5 text-sm focus-within:border-ring">
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
                  const expanded = open === r.tool.name
                  const effect = EFFECT[r.tool.effect] ?? { label: r.tool.effect, tone: "" }
                  const source = sourceText(r.tool.source)
                  const approval = r.tool.approval ? APPROVAL[r.tool.approval] : null
                  return (
                    <Fragment key={r.tool.name}>
                      <tr
                        tabIndex={0}
                        aria-expanded={expanded}
                        onClick={() => setOpen(expanded ? null : r.tool.name)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") setOpen(expanded ? null : r.tool.name)
                        }}
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
                            <span className="font-mono text-[13px]">{r.tool.name}</span>
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
                        <td className="px-3 py-2 text-right tabular-nums">{r.usage?.calls ?? 0}</td>
                        <td
                          className={cn(
                            "px-3 py-2 text-right tabular-nums",
                            (r.usage?.failed ?? 0) > 0 && "text-destructive",
                          )}
                        >
                          {r.usage?.failed ?? 0}
                        </td>
                        <td className="px-3 py-2 text-right whitespace-nowrap text-muted-foreground tabular-nums">
                          {r.usage?.last_used_at
                            ? relativeTime(Date.parse(r.usage.last_used_at), now())
                            : "never"}
                        </td>
                      </tr>
                      {expanded && (
                        <tr className="border-b border-border bg-muted/30">
                          <td colSpan={8} className="px-4 py-4">
                            <ToolDetail row={r} titles={titles} now={now()} />
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
  titles,
  now,
}: {
  row: ToolRow
  titles: Map<string, string>
  now: number
}) {
  const { tool, usage } = row
  const approvals = usage?.approvals
  return (
    <div className="grid gap-5 lg:grid-cols-2">
      <div className="min-w-0 space-y-3">
        <div>
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
          <p className="text-xs font-medium text-muted-foreground">Input</p>
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
        {asksPeople(tool.approval) && approvals && (
          <div>
            <p className="text-xs font-medium text-muted-foreground">Approvals</p>
            <p className="mt-1 tabular-nums">
              {approvals.approved} approved · {approvals.denied} denied · {approvals.waiting}{" "}
              waiting
            </p>
          </div>
        )}
        <div>
          <p className="text-xs font-medium text-muted-foreground">Recent calls</p>
          {!usage || usage.recent.length === 0 ? (
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
                    <span className="min-w-0 flex-1 truncate">
                      {titles.get(c.session_id) ?? c.session_id}
                    </span>
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
