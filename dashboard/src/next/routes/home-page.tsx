import { keepPreviousData, useQuery } from "@tanstack/react-query"
/**
 * The developer's overview of the agent: a line about the last day, what needs
 * attention, and the latest sessions. Cayu agents work on their own; end users reach them
 * through the app's own UI. Starting a session here is a way to test.
 */
import { useSearch } from "@tanstack/react-router"
import { type ReactNode, useEffect, useState } from "react"
import { CapabilityRoute } from "@/components/dashboard/server-contract"
import { DASHBOARD_ROUTE_REQUIREMENTS } from "@/lib/dashboard-capabilities"
import { activityPollInterval } from "@/lib/polling"
import { cn } from "@/lib/utils"
import { type SessionRow, useHomeSessionRows, useInboxCounts } from "../app/queries"
import { useApi, useNow, useServices } from "../app/services"
import { isWaiting, relativeTime } from "../app/session-model"
import {
  ErrorState,
  NavLink,
  needsAttention,
  PageBody,
  StateDot,
  sessionRoute,
  stateText,
  TopBar,
} from "./common"
import { HomeTasks } from "./home-tasks"
import { StartSession, StartSessionButton } from "./start-session"

const DAY = 24 * 3600_000
const RECENT = 8
const activity = (r: SessionRow) => Date.parse(r.overview?.last_activity_at ?? r.session.updated_at)
const failed = (r: SessionRow) => r.overview?.state === "failed"

export function HomePage() {
  const { prototype } = useServices()
  const search = useSearch({ strict: false }) as { start?: string }
  const [starting, setStarting] = useState(search.start === "1")
  const [startPending, setStartPending] = useState(false)
  useEffect(() => {
    if (search.start === "1") setStarting(true)
  }, [search.start])
  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <TopBar
        actions={
          <StartSessionButton
            pending={startPending}
            open={starting}
            onToggle={() => setStarting(!starting)}
          />
        }
      >
        Home
      </TopBar>
      <PageBody width="max-w-3xl">
        {starting && (
          <StartSession onPendingChange={setStartPending} onClose={() => setStarting(false)} />
        )}
        {prototype ? (
          <HomeOverview />
        ) : (
          <CapabilityRoute
            inline
            requirement={DASHBOARD_ROUTE_REQUIREMENTS["/"]}
            title="Home overview"
          >
            <HomeOverview />
          </CapabilityRoute>
        )}
      </PageBody>
    </div>
  )
}

function HomeOverview() {
  const now = useNow()
  const sessions = useHomeSessionRows("recent")
  const waiting = useHomeSessionRows("waiting")
  const failures = useHomeSessionRows("failed")
  const inbox = useInboxCounts()
  const api = useApi()
  const [window, setWindow] = useState(() => ({
    since: new Date(now() - DAY).toISOString(),
    until: new Date(now()).toISOString(),
  }))
  useEffect(() => {
    const timer = setInterval(
      () =>
        setWindow({
          since: new Date(now() - DAY).toISOString(),
          until: new Date(now()).toISOString(),
        }),
      60_000,
    )
    return () => clearInterval(timer)
  }, [now])
  const counts = useQuery({
    queryKey: ["home", "counts", window],
    placeholderData: keepPreviousData,
    queryFn: () => api.overviewCounts(window),
    refetchInterval: (query) =>
      activityPollInterval(Number(query.state.data?.running_now ?? 0) > 0, 5000),
  })

  const attention = [...waiting.rows, ...failures.rows.filter((row) => activity(row) > now() - DAY)]
  const recent = sessions.rows
    .filter(
      (row) => !isWaiting(row.overview?.state) && !(failed(row) && activity(row) > now() - DAY),
    )
    .slice(0, RECENT)
  const attentionPending = waiting.isPending || failures.isPending
  const attentionError = waiting.error || failures.error
  const attentionMore = waiting.hasNextPage || failures.hasNextPage
  const attentionEmpty = attentionPending
    ? "Loading sessions needing attention…"
    : attentionError
      ? "Attention could not be fully checked."
      : attentionMore
        ? "No matching attention sessions on the pages checked. Continue scanning."
        : "No sessions need attention."
  const recentEmpty = sessions.isPending
    ? "Loading recent sessions…"
    : sessions.error
      ? "Recent sessions could not be fully checked."
      : sessions.hasNextPage
        ? "No recent sessions outside Needs attention on the pages loaded. Continue scanning."
        : sessions.rows.length > 0
          ? "The loaded recent sessions need attention."
          : "No recent sessions found."
  const knowledge = inbox.data?.complete === true ? (inbox.data.counts?.knowledge ?? 0) : 0
  const summary = counts.data
    ? `${counts.data.total_count} sessions · ${counts.data.failed} failed · ${counts.data.cost ? `${counts.data.cost.amount} ${counts.data.cost.currency}${counts.data.cost.complete ? "" : " (partial cost)"}` : `${BigInt(counts.data.tokens.input) + BigInt(counts.data.tokens.output)} tokens${counts.data.tokens_complete ? "" : " (partial usage)"}`}`
    : null

  return (
    <>
      {sessions.error && <ErrorState error={sessions.error} />}
      {waiting.error && <ErrorState error={waiting.error} />}
      {failures.error && <ErrorState error={failures.error} />}
      {counts.error && <ErrorState error={counts.error} />}
      {counts.isSuccess && (
        <p className="text-sm text-muted-foreground">
          Last 24 hours: <span className="text-foreground tabular-nums">{summary}</span>
        </p>
      )}

      <Section
        title="Needs attention"
        link={
          knowledge ? (
            <NavLink to="/inbox">{knowledge} knowledge to review in the inbox</NavLink>
          ) : null
        }
      >
        {attention.length > 0 && (attentionPending || attentionError) && (
          <p role="status" className="px-3.5 py-2 text-sm text-muted-foreground">
            {attentionError
              ? "Attention could not be fully checked."
              : "Checking more attention sessions…"}
          </p>
        )}
        {attention.length === 0 ? (
          <p role="status" className="px-3.5 py-5 text-sm text-muted-foreground">
            {attentionEmpty}
          </p>
        ) : (
          attention.map((r) => (
            <Row
              key={r.session.id}
              row={r}
              now={now()}
              trace={failed(r)}
              right={
                <span className={cn(failed(r) ? "text-destructive" : "text-warning")}>
                  {failed(r) ? (r.overview?.reason ?? "Failed") : stateText(r.overview)}
                </span>
              }
            />
          ))
        )}
      </Section>

      {(waiting.hasNextPage || failures.hasNextPage) && (
        <button
          type="button"
          className="mt-2 text-sm underline"
          disabled={waiting.isFetching || failures.isFetching}
          onClick={() => {
            if (waiting.hasNextPage) void waiting.fetchNextPage()
            if (failures.hasNextPage) void failures.fetchNextPage()
          }}
        >
          Load more attention sessions
        </button>
      )}

      <Section title="Recent sessions" link={<NavLink to="/sessions">All sessions</NavLink>}>
        {recent.length === 0 ? (
          <p role="status" className="px-3.5 py-5 text-sm text-muted-foreground">
            {recentEmpty}
          </p>
        ) : (
          recent.map((r) => (
            <Row
              key={r.session.id}
              row={r}
              now={now()}
              right={<span className="text-muted-foreground">{stateText(r.overview)}</span>}
            />
          ))
        )}
      </Section>
      {sessions.hasNextPage && recent.length < RECENT && (
        <button
          type="button"
          className="mt-2 text-sm underline"
          disabled={sessions.isFetching}
          onClick={() => void sessions.fetchNextPage()}
        >
          Load more recent sessions
        </button>
      )}
      <HomeTasks />
    </>
  )
}

function Section({
  title,
  link,
  children,
}: {
  title: string
  link?: ReactNode
  children: ReactNode
}) {
  return (
    <section className="mt-6 space-y-2">
      <div className="flex items-baseline justify-between gap-3">
        <h2 className="text-sm font-medium">{title}</h2>
        <span className="truncate text-xs text-muted-foreground [&_a:hover]:text-foreground">
          {link}
        </span>
      </div>
      <div className="divide-y divide-border overflow-hidden rounded-xl border border-border bg-card">
        {children}
      </div>
    </section>
  )
}

function Row({
  row,
  right,
  now,
  trace = false,
}: {
  row: SessionRow
  right: ReactNode
  now: number
  /** Open the session's trace, for failures. */
  trace?: boolean
}) {
  return (
    <NavLink
      {...sessionRoute(row.session.id)}
      search={trace ? { view: "trace", log: "1", event_type: "session.failed" } : undefined}
      className="flex items-center gap-3 px-3.5 py-2.5 hover:bg-muted/60"
    >
      <StateDot state={row.overview?.state} attention={needsAttention(row.overview)} />
      <span className="min-w-0 flex-1">
        <span className="block truncate text-sm">
          {row.title ?? row.overview?.title ?? row.session.id}
        </span>
        <span className="block truncate text-xs text-muted-foreground">
          {row.session.agent_name} · {relativeTime(activity(row), now)}
        </span>
      </span>
      <span className="max-w-[40%] shrink-0 truncate text-right text-xs">{right}</span>
    </NavLink>
  )
}
