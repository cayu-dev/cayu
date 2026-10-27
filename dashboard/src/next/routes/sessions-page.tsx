import { useSearch } from "@tanstack/react-router"
import { useCallback, useState } from "react"
import { Button } from "@/components/ui/button"
import { formatCount, formatCurrencyWithCode, sumCounts } from "@/lib/format"
import {
  DEFAULT_SESSION_ORDER,
  SESSION_INDEX_DEBUG_STATES,
  SESSION_INDEX_ORDERS,
  SESSION_INDEX_STATUSES,
  type SessionIndexFilterPatch,
  type SessionIndexSearch,
  sessionIndexAdvancedFilterKey,
  sessionIndexFilterCount,
  sessionIndexSearchWithCursor,
  sessionIndexSearchWithFilters,
  sessionIndexSearchWithoutFilters,
} from "@/lib/session-index-search"
import { cn } from "@/lib/utils"
import { overviewForDisplay } from "../api/overview"
import { useNow } from "../app/services"
import { useSessionList } from "../app/session-list"
import type { SessionListRow } from "../app/session-list-loader"
import { constantColumns, formatDuration, relativeTime } from "../app/session-model"
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
import {
  AdvancedFilters,
  DEBUG_LABELS,
  ORDER_LABELS,
  SessionSearchInput,
  selectClassName,
} from "./session-filters"
import { StartSession, StartSessionButton } from "./start-session"

const COLUMNS: Record<string, (row: SessionListRow) => string | null> = {
  Specialist: (row) => row.session.agent_name,
  Model: (row) => row.session.model,
  Provider: (row) => row.session.provider_name,
  Environment: (row) => row.session.environment_name,
}

export function SessionsPage({ canStart = true }: { canStart?: boolean } = {}) {
  const search = useSearch({ strict: false }) as SessionIndexSearch
  const go = useGo()
  const rowClick = useRowClick()
  const now = useNow()
  const sessions = useSessionList(search)
  const [searchReset, setSearchReset] = useState(0)
  const [starting, setStarting] = useState(false)
  const [startPending, setStartPending] = useState(false)
  const set = useCallback(
    (patch: SessionIndexFilterPatch) =>
      go("/sessions", sessionIndexSearchWithFilters(search, patch)),
    [go, search],
  )
  const rows = sessions.data?.rows ?? []
  const { constant, varying } = constantColumns(rows, COLUMNS)
  // The canonical server returns a cost object iff a price book is configured.
  const priced = rows.some((row) => row.overview.cost !== null)
  const next = sessions.data?.page.next_cursor
  const paused = sessions.data?.searchPaused
  const hasFilters = sessionIndexFilterCount(search) > 0
  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <TopBar
        actions={
          canStart ? (
            <StartSessionButton
              pending={startPending}
              open={starting}
              onToggle={() => setStarting(!starting)}
            />
          ) : undefined
        }
      >
        Sessions
      </TopBar>
      <PageBody width="max-w-6xl">
        {canStart && starting && (
          <StartSession onPendingChange={setStartPending} onClose={() => setStarting(false)} />
        )}
        <div className="flex flex-wrap items-center gap-2">
          <SessionSearchInput
            value={search.q ?? ""}
            resetVersion={searchReset}
            onCommit={(q) =>
              go("/sessions", sessionIndexSearchWithFilters(search, { q }), {
                replace: true,
                resetScroll: false,
              })
            }
          />
          <select
            aria-label="State"
            className={selectClassName}
            value={search.state ?? ""}
            onChange={(event) =>
              set({ state: (event.target.value || undefined) as SessionIndexSearch["state"] })
            }
          >
            <option value="">Any state</option>
            <option value="waiting">Waiting for you</option>
            <option value="running">Running</option>
            <option value="waiting_approval">Needs your approval</option>
            <option value="waiting_input">Needs your answer</option>
            <option value="needs_recovery">Needs recovery</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
            <option value="interrupted">Stopped</option>
          </select>
          <select
            aria-label="Sort sessions"
            className={selectClassName}
            value={search.order_by ?? DEFAULT_SESSION_ORDER}
            onChange={(event) =>
              set({ order_by: event.target.value as SessionIndexSearch["order_by"] })
            }
          >
            {SESSION_INDEX_ORDERS.map((order) => (
              <option key={order} value={order}>
                {ORDER_LABELS[order]}
              </option>
            ))}
          </select>
          <label className="flex h-8 items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={search.include_eval === true}
              onChange={(event) => set({ include_eval: event.target.checked ? true : undefined })}
            />
            Include eval trials
          </label>
          <label className="flex h-8 items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={search.include_forks === true}
              onChange={(event) => set({ include_forks: event.target.checked ? true : undefined })}
            />
            Include forks
          </label>
          {hasFilters && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                setSearchReset((version) => version + 1)
                go("/sessions", sessionIndexSearchWithoutFilters(search))
              }}
            >
              Clear filters
            </Button>
          )}
          <AdvancedFilters
            key={sessionIndexAdvancedFilterKey(search)}
            search={search}
            onApply={set}
            secondary={
              <>
                <select
                  aria-label="Raw status"
                  className={selectClassName}
                  value={search.status ?? ""}
                  onChange={(event) =>
                    set({
                      status: (event.target.value || undefined) as SessionIndexSearch["status"],
                    })
                  }
                >
                  <option value="">Any raw status</option>
                  {SESSION_INDEX_STATUSES.map((status) => (
                    <option key={status} value={status}>
                      {status}
                    </option>
                  ))}
                </select>
                <select
                  aria-label="Debug state"
                  className={selectClassName}
                  value={search.debug_state ?? ""}
                  onChange={(event) =>
                    set({
                      debug_state: (event.target.value ||
                        undefined) as SessionIndexSearch["debug_state"],
                    })
                  }
                >
                  <option value="">Any debug state</option>
                  {SESSION_INDEX_DEBUG_STATES.map((state) => (
                    <option key={state} value={state}>
                      {DEBUG_LABELS[state]}
                    </option>
                  ))}
                </select>
              </>
            }
          />
        </div>
        {Object.keys(constant).length > 0 && (
          <p className="pt-3 text-xs text-muted-foreground">
            Every session shown has the same{" "}
            {Object.entries(constant)
              .map(([key, value]) => `${key.toLowerCase()} (${value})`)
              .join(", ")}
            , so {Object.keys(constant).length === 1 ? "that column is" : "those columns are"}{" "}
            hidden.
          </p>
        )}
        {sessions.isPending && (
          <p role="status" className="py-6 text-sm text-muted-foreground">
            Searching sessions…
          </p>
        )}
        {sessions.error && (
          <div className="mt-3">
            <ErrorState error={sessions.error} />
            <Button variant="outline" onClick={() => void sessions.refetch()}>
              Retry
            </Button>
            {search.cursor && (
              <Button
                variant="ghost"
                onClick={() => go("/sessions", sessionIndexSearchWithCursor(search, null))}
              >
                Restart search
              </Button>
            )}
          </div>
        )}
        {sessions.isSuccess && rows.length === 0 && (
          <div className="mt-4">
            <EmptyState
              title={
                paused
                  ? "No matches in the pages scanned yet"
                  : search.cursor
                    ? "No more sessions match these filters"
                    : hasFilters
                      ? "No sessions match these filters"
                      : "No sessions yet"
              }
            />
            {paused && (
              <p className="pt-2 text-sm text-muted-foreground">
                Continue searching to scan the next pages.
              </p>
            )}
          </div>
        )}
        {rows.length > 0 && (
          <section
            aria-label="Session results"
            // biome-ignore lint/a11y/noNoninteractiveTabindex: keyboard users must be able to scroll the wide results table.
            tabIndex={0}
            className="mt-3 overflow-x-auto rounded-xl border border-border focus-visible:outline-2 focus-visible:outline-ring"
          >
            <table className="w-full min-w-[760px] text-sm">
              <thead>
                <tr className="border-b border-border text-left text-xs text-muted-foreground">
                  <th className="px-3 py-2 font-medium">Session</th>
                  {varying.map((column) => (
                    <th key={column} className="px-3 py-2 font-medium">
                      {column}
                    </th>
                  ))}
                  <th className="px-3 py-2 font-medium">State</th>
                  <th className="px-3 py-2 text-right font-medium">Duration</th>
                  <th className="px-3 py-2 text-right font-medium">Tool calls</th>
                  <th className="px-3 py-2 text-right font-medium">{priced ? "Cost" : "Tokens"}</th>
                  <th className="px-3 py-2 text-right font-medium">Updated</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => {
                  const o = row.overview
                  const duration = o.ended_at
                    ? (Date.parse(o.ended_at) - Date.parse(o.started_at)) / 1000
                    : null
                  return (
                    <tr
                      key={row.session.id}
                      onClick={rowClick(sessionPath(row.session.id))}
                      className={ROW_CLASS}
                    >
                      <td className="max-w-[28rem] px-3 py-2">
                        <div className="flex min-w-0 items-center gap-2">
                          <NavLink
                            to={sessionPath(row.session.id)}
                            className={cn("truncate", ROW_LINK_CLASS)}
                          >
                            {row.title}
                          </NavLink>
                          {o.eval && <EvalTrialTag />}
                          {o.forked && <span className="text-xs text-muted-foreground">Fork</span>}
                        </div>
                        <div className="mt-0.5 flex min-w-0 items-center gap-2">
                          <span className="truncate font-mono text-[11px] text-muted-foreground">
                            {row.session.id}
                          </span>
                          <LabelChips
                            labels={row.session.labels}
                            className="flex-nowrap overflow-hidden"
                          />
                        </div>
                        {row.titleUnavailable && (
                          <span className="text-xs text-muted-foreground">
                            Message title unavailable
                          </span>
                        )}
                      </td>
                      {varying.map((column) => (
                        <td
                          key={column}
                          className="px-3 py-2 whitespace-nowrap text-muted-foreground"
                        >
                          {COLUMNS[column]?.(row) ?? "–"}
                        </td>
                      ))}
                      <td className="max-w-sm px-3 py-2">
                        <StateBadge overview={overviewForDisplay(o)} />
                        {o.summary && (
                          <p
                            className={cn(
                              "mt-1 max-w-64 truncate text-xs",
                              o.summary_tone === "attention"
                                ? "text-warning"
                                : "text-muted-foreground",
                            )}
                            title={o.summary}
                          >
                            {o.summary}
                          </p>
                        )}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums">
                        {formatDuration(duration)}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums">
                        {formatCount(o.tool_calls)}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums">
                        {priced ? (
                          o.cost ? (
                            <>
                              {formatCurrencyWithCode(o.cost.amount, o.cost.currency)}
                              {!o.cost.complete && (
                                <span className="block text-xs text-muted-foreground">
                                  Partial cost
                                </span>
                              )}
                            </>
                          ) : (
                            "–"
                          )
                        ) : (
                          formatCount(sumCounts(o.tokens.input, o.tokens.output))
                        )}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">
                        {relativeTime(Date.parse(o.last_activity_at), now())}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </section>
        )}
        {sessions.isSuccess && (
          <div className="flex flex-wrap items-center justify-between gap-3 pt-3">
            <p className="text-xs text-muted-foreground">
              {rows.length} {rows.length === 1 ? "session" : "sessions"} loaded on this page.
            </p>
            <div className="flex gap-2">
              {search.cursor && (
                <Button
                  variant="outline"
                  onClick={() => go("/sessions", sessionIndexSearchWithCursor(search, null))}
                >
                  First page
                </Button>
              )}
              {next && (
                <Button
                  variant="outline"
                  onClick={() => go("/sessions", sessionIndexSearchWithCursor(search, next))}
                >
                  {paused ? "Continue searching" : "Next page"}
                </Button>
              )}
            </div>
          </div>
        )}
      </PageBody>
    </div>
  )
}
