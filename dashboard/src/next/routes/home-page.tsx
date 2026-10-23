/**
 * The developer's overview of the agent: a line about the last day, what needs
 * attention, and the latest sessions. Cayu agents work on their own; end users reach them
 * through the app's own UI. Starting a session here is a way to test.
 */
import { useSearch } from "@tanstack/react-router"
import { type ReactNode, useEffect, useState } from "react"
import { cn } from "@/lib/utils"
import { type SessionRow, useInbox, useSessionRows, withoutEvalTrials } from "../app/queries"
import { useNow } from "../app/services"
import { formatTokens, isWaiting, relativeTime } from "../app/session-model"
import {
  ErrorState,
  NavLink,
  needsAttention,
  PageBody,
  StateDot,
  sessionPath,
  stateText,
  TopBar,
} from "./common"
import { StartSession, StartSessionButton } from "./start-session"

const DAY = 24 * 3600_000
const RECENT = 8
const activity = (r: SessionRow) => Date.parse(r.overview?.last_activity_at ?? r.session.updated_at)
const failed = (r: SessionRow) => r.overview?.state === "failed"

export function HomePage() {
  const now = useNow()
  const search = useSearch({ strict: false }) as { start?: string }
  const sessions = useSessionRows()
  const inbox = useInbox()
  const [starting, setStarting] = useState(search.start === "1")
  useEffect(() => {
    if (search.start === "1") setStarting(true)
  }, [search.start])

  // Eval trials are the eval runner's, not the agent's work: Home leaves them out, as Cloud does.
  const rows = withoutEvalTrials(sessions.data?.rows ?? [])
  const day = rows.filter((r) => activity(r) > now() - DAY)
  // What a developer should look at: anything waiting on a person, and failures from the last day.
  const attention = rows.filter(
    (r) => isWaiting(r.overview?.state) || (failed(r) && activity(r) > now() - DAY),
  )
  const recent = rows.filter((r) => !attention.includes(r)).slice(0, RECENT)
  const cost = day.reduce((n, r) => n + (r.overview?.cost?.amount ?? 0), 0)
  const tokens = day.reduce(
    (n, r) => n + (r.overview ? r.overview.tokens.input + r.overview.tokens.output : 0),
    0,
  )
  const knowledge = (inbox.data?.items ?? []).filter((i) => i.kind === "knowledge").length
  const summary = [
    `${day.length} ${day.length === 1 ? "session" : "sessions"}`,
    `${day.filter(failed).length} failed`,
    cost ? `$${cost.toFixed(2)}` : `${formatTokens(tokens)} tokens`,
  ].join(" · ")

  return (
    <div className="flex min-w-0 flex-1 flex-col">
      <TopBar
        actions={<StartSessionButton open={starting} onToggle={() => setStarting(!starting)} />}
      >
        Home
      </TopBar>
      <PageBody width="max-w-3xl">
        {starting && <StartSession onClose={() => setStarting(false)} />}
        {sessions.error && <ErrorState error={sessions.error} />}
        {sessions.isSuccess && (
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
          {attention.length === 0 ? (
            <p className="px-3.5 py-5 text-sm text-muted-foreground">Nothing needs attention.</p>
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

        <Section title="Recent sessions" link={<NavLink to="/sessions">All sessions</NavLink>}>
          {recent.length === 0 ? (
            <p className="px-3.5 py-5 text-sm text-muted-foreground">
              No sessions yet. They appear when the agent runs, from its own UI, a task, Cayu Cloud,
              or Start a session.
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
      </PageBody>
    </div>
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
      to={sessionPath(row.session.id)}
      search={trace ? { view: "trace" } : undefined}
      className="flex items-center gap-3 px-3.5 py-2.5 hover:bg-muted/60"
    >
      <StateDot state={row.overview?.state} attention={needsAttention(row.overview)} />
      <span className="min-w-0 flex-1">
        <span className="block truncate text-sm">{row.overview?.title ?? row.session.id}</span>
        <span className="block truncate text-xs text-muted-foreground">
          {row.session.agent_name} · {relativeTime(activity(row), now)}
        </span>
      </span>
      <span className="max-w-[40%] shrink-0 truncate text-right text-xs">{right}</span>
    </NavLink>
  )
}
