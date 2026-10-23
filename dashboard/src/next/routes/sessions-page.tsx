import { useSearch } from "@tanstack/react-router"
import { Search } from "lucide-react"
import { useMemo, useState } from "react"
import { cn } from "@/lib/utils"
import {
  type SessionRow,
  useSessionRows,
  useSpecialists,
  withoutEvalTrials,
} from "./../app/queries"
import { useNow } from "../app/services"
import {
  constantColumns,
  formatDuration,
  formatTokens,
  isWaiting,
  relativeTime,
} from "../app/session-model"
import {
  EmptyState,
  ErrorState,
  EvalTrialTag,
  LabelChips,
  NavLink,
  PageBody,
  ROW_CLASS,
  ROW_LINK_CLASS,
  StateBadge,
  sessionPath,
  TopBar,
  useGo,
  useRowClick,
} from "./common"
import { StartSession, StartSessionButton } from "./start-session"

type Filters = { q?: string; state?: string; agent?: string; label?: string; trials?: string }

const COLUMNS: Record<string, (r: SessionRow) => string | null> = {
  Specialist: (r) => r.session.agent_name,
  Model: (r) => r.session.model,
  Provider: (r) => r.session.provider_name,
  Environment: (r) => r.session.environment_name,
}

export function SessionsPage() {
  const search = useSearch({ strict: false }) as Filters
  const go = useGo()
  const rowClick = useRowClick()
  const now = useNow()
  const specialists = useSpecialists()
  const sessions = useSessionRows({ q: search.q, agent: search.agent, label: search.label })
  const [starting, setStarting] = useState(false)
  const set = (patch: Partial<Filters>) => go("/sessions", { ...search, ...patch })

  const includeTrials = search.trials === "1"
  const loaded = sessions.data?.rows ?? []
  // Eval trials are the eval runner's sessions; they're hidden unless asked for.
  const trialCount = loaded.filter((r) => r.overview?.eval_trial).length
  const rows = useMemo(() => {
    const all = includeTrials ? loaded : withoutEvalTrials(loaded)
    if (!search.state) return all
    return all.filter((r) =>
      search.state === "waiting"
        ? isWaiting(r.overview?.state)
        : r.overview?.state === search.state,
    )
  }, [loaded, includeTrials, search.state])
  const { constant, varying } = constantColumns(rows, COLUMNS)
  const labelKeys = [...new Set(rows.flatMap((r) => Object.keys(r.session.labels)))]
  const field =
    "h-8 rounded-lg border border-input bg-background px-2 text-sm outline-none focus:border-ring"

  return (
    <div className="flex min-w-0 flex-1 flex-col">
      <TopBar
        actions={<StartSessionButton open={starting} onToggle={() => setStarting(!starting)} />}
      >
        Sessions
      </TopBar>
      <PageBody width="max-w-6xl">
        {starting && <StartSession onClose={() => setStarting(false)} />}
        <div className="flex flex-wrap items-center gap-2">
          <label className="flex h-8 min-w-0 flex-1 items-center gap-2 rounded-lg border border-input bg-background px-2.5 text-sm focus-within:border-ring sm:max-w-sm">
            <Search className="size-4 shrink-0 text-muted-foreground" aria-hidden="true" />
            <span className="sr-only">Search sessions</span>
            <input
              id="sessions-search"
              defaultValue={search.q ?? ""}
              onKeyDown={(e) => {
                if (e.key === "Enter") set({ q: e.currentTarget.value || undefined })
              }}
              onBlur={(e) => set({ q: e.currentTarget.value || undefined })}
              placeholder="Search by title, ID or label"
              className="w-full min-w-0 bg-transparent outline-none placeholder:text-muted-foreground"
            />
          </label>
          <select
            id="sessions-state"
            aria-label="State"
            className={field}
            value={search.state ?? ""}
            onChange={(e) => set({ state: e.target.value || undefined })}
          >
            <option value="">Any state</option>
            <option value="waiting">Waiting for you</option>
            <option value="running">Running</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
            <option value="interrupted">Stopped</option>
          </select>
          {(specialists.data?.agents.length ?? 0) > 1 && (
            <select
              id="sessions-agent"
              aria-label="Specialist"
              className={field}
              value={search.agent ?? ""}
              onChange={(e) => set({ agent: e.target.value || undefined })}
            >
              <option value="">All specialists</option>
              {specialists.data?.agents.map((a) => (
                <option key={a.name} value={a.name}>
                  {a.name}
                </option>
              ))}
            </select>
          )}
          {labelKeys.length > 0 && (
            <input
              id="sessions-label"
              aria-label="Label filter"
              defaultValue={search.label ?? ""}
              placeholder={`Label, e.g. ${labelKeys[0]}=…`}
              onKeyDown={(e) => {
                if (e.key === "Enter") set({ label: e.currentTarget.value || undefined })
              }}
              className={cn(field, "w-44")}
            />
          )}
          <label className="flex h-8 items-center gap-1.5 px-1 text-sm text-muted-foreground">
            <input
              id="sessions-trials"
              type="checkbox"
              checked={includeTrials}
              onChange={(e) => set({ trials: e.target.checked ? "1" : undefined })}
              className="size-3.5 accent-foreground"
            />
            Include eval trials
          </label>
        </div>

        {Object.keys(constant).length > 0 && (
          <p className="pt-3 text-xs text-muted-foreground">
            Every session shown has the same{" "}
            {Object.entries(constant)
              .map(([k, v]) => `${k.toLowerCase()} (${v})`)
              .join(", ")}
            , so {Object.keys(constant).length === 1 ? "that column is" : "those columns are"}{" "}
            hidden.
          </p>
        )}

        {sessions.error && (
          <div className="mt-3">
            <ErrorState error={sessions.error} />
          </div>
        )}
        {sessions.isSuccess && rows.length === 0 ? (
          <div className="mt-4">
            <EmptyState
              title={
                sessions.data.rows.length ? "No sessions match these filters" : "No sessions yet"
              }
            />
          </div>
        ) : (
          <div className="mt-3 overflow-x-auto rounded-xl border border-border">
            <table className="w-full min-w-[760px] text-sm">
              <thead>
                <tr className="border-b border-border text-left text-xs text-muted-foreground">
                  <th className="px-3 py-2 font-medium">Session</th>
                  {varying.map((c) => (
                    <th key={c} className="px-3 py-2 font-medium">
                      {c}
                    </th>
                  ))}
                  <th className="px-3 py-2 font-medium">State</th>
                  <th className="px-3 py-2 text-right font-medium">Duration</th>
                  <th className="px-3 py-2 text-right font-medium">Tool calls</th>
                  <th className="px-3 py-2 text-right font-medium">
                    {rows.some((r) => r.overview?.cost) ? "Cost" : "Tokens"}
                  </th>
                  <th className="px-3 py-2 text-right font-medium">Updated</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => {
                  const o = r.overview
                  const seconds = o?.ended_at
                    ? (Date.parse(o.ended_at) - Date.parse(o.started_at)) / 1000
                    : null
                  return (
                    <tr
                      key={r.session.id}
                      onClick={rowClick(sessionPath(r.session.id))}
                      className={ROW_CLASS}
                    >
                      <td className="max-w-[28rem] px-3 py-2">
                        <div className="flex min-w-0 items-center gap-2">
                          <NavLink
                            to={sessionPath(r.session.id)}
                            className={cn("truncate", ROW_LINK_CLASS)}
                          >
                            {o?.title ?? r.session.id}
                          </NavLink>
                          {o?.eval_trial && <EvalTrialTag />}
                        </div>
                        <div className="mt-0.5 flex min-w-0 items-center gap-2">
                          <span className="truncate font-mono text-[11px] text-muted-foreground">
                            {r.session.id}
                          </span>
                          <LabelChips
                            labels={r.session.labels}
                            className="flex-nowrap overflow-hidden"
                          />
                        </div>
                      </td>
                      {varying.map((c) => (
                        <td key={c} className="px-3 py-2 whitespace-nowrap text-muted-foreground">
                          {COLUMNS[c]?.(r) ?? "–"}
                        </td>
                      ))}
                      <td className="px-3 py-2">
                        <StateBadge overview={o} />
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums">
                        {formatDuration(seconds)}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums">{o?.tool_calls ?? "–"}</td>
                      <td className="px-3 py-2 text-right tabular-nums">
                        {o?.cost
                          ? `$${o.cost.amount.toFixed(2)}`
                          : o
                            ? formatTokens(o.tokens.input + o.tokens.output)
                            : "–"}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">
                        {relativeTime(
                          Date.parse(o?.last_activity_at ?? r.session.updated_at),
                          now(),
                        )}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
        {sessions.data && (
          <p className="pt-2 text-xs text-muted-foreground">
            Showing {rows.length} of {sessions.data.total.toLocaleString()}.
            {!includeTrials && trialCount > 0 && (
              <>
                {" "}
                {trialCount} eval {trialCount === 1 ? "trial" : "trials"} hidden.
              </>
            )}
          </p>
        )}
      </PageBody>
    </div>
  )
}
