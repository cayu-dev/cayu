import { useQuery } from "@tanstack/react-query"
import { useSearch } from "@tanstack/react-router"
import { useState } from "react"
import { cn } from "@/lib/utils"
import type { InboxItem } from "../api/types"
import { useInbox, useInboxCounts } from "../app/queries"
import { useApi } from "../app/services"
import { ChildActions } from "./child-actions"
import { EmptyState, ErrorState, NavLink, PageBody, sessionPath, TopBar, useGo } from "./common"
import { InboxCard } from "./session/inbox-card"
import { selectClassName } from "./session-filters"

const KINDS: [InboxItem["kind"] | undefined, string][] = [
  [undefined, "All"],
  ["approval", "Approvals"],
  ["question", "Questions"],
  ["recovery", "Recovery"],
  ["knowledge", "Knowledge"],
]

export function InboxPage() {
  const search = useSearch({ strict: false }) as {
    q?: string
    kind?: InboxItem["kind"]
    session_id?: string
    agent_name?: string
  }
  const go = useGo()
  const [pageSize, setPageSize] = useState(50)
  const inbox = useInbox({
    limit: pageSize,
    q: search.q,
    kind: search.kind,
    session_id: search.session_id,
    agent_name: search.agent_name,
  })
  const items = inbox.data?.items ?? []
  const api = useApi()
  const ids = [...new Set(items.flatMap((item) => (item.session_id ? [item.session_id] : [])))]
  const sessions = useQuery({
    queryKey: ["inbox-session-titles", ids],
    queryFn: () => api.overviews(ids),
    enabled: ids.length > 0,
    refetchInterval: 5000,
  })
  const titles = new Map(
    (sessions.data ?? []).map((row) => [row.session_id, row.title ?? row.session_id]),
  )
  const all = useInboxCounts({
    q: search.q,
    session_id: search.session_id,
    agent_name: search.agent_name,
  })
  const countFor = (kind: InboxItem["kind"] | undefined) =>
    all.data?.complete === true ? (kind ? all.data.counts?.[kind] : all.data.total_count) : null

  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <TopBar>Inbox</TopBar>
      <PageBody width="max-w-3xl">
        <p className="text-sm text-muted-foreground">
          Everything in this agent waiting on a person. Deciding here settles the same record in
          Cayu Cloud and in the agent's own UI.
        </p>
        <form
          key={JSON.stringify(search)}
          className="mt-4 flex flex-wrap gap-2"
          onSubmit={(event) => {
            event.preventDefault()
            const values = new FormData(event.currentTarget)
            go("/inbox", {
              ...search,
              q: String(values.get("q") ?? "").trim() || undefined,
              session_id: String(values.get("session_id") ?? "").trim() || undefined,
              agent_name: String(values.get("agent_name") ?? "").trim() || undefined,
            })
          }}
        >
          <label className="text-xs">
            Search Inbox
            <input name="q" defaultValue={search.q} className={cn("ml-2", selectClassName)} />
          </label>
          <button
            type="submit"
            className="h-8 rounded-lg border border-border px-3 text-sm hover:bg-muted focus-visible:outline-ring"
          >
            Filter
          </button>
          <details className="w-full text-sm" open={!!search.session_id || !!search.agent_name}>
            <summary className="cursor-pointer text-muted-foreground">Advanced filters</summary>
            <div className="mt-3 flex flex-wrap items-end gap-3">
              <label className="text-xs">
                Session
                <input
                  name="session_id"
                  defaultValue={search.session_id}
                  className={cn("ml-2", selectClassName)}
                />
              </label>
              <label className="text-xs">
                Specialist
                <input
                  name="agent_name"
                  defaultValue={search.agent_name}
                  className={cn("ml-2", selectClassName)}
                />
              </label>
              <label className="mt-3 block text-xs">
                Inbox page size
                <select
                  aria-label="Inbox page size"
                  value={pageSize}
                  onChange={(event) => setPageSize(Number(event.target.value))}
                  className={cn("ml-2", selectClassName)}
                >
                  {[25, 50, 100].map((size) => (
                    <option key={size} value={size}>
                      {size}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          </details>
        </form>
        <fieldset className="mt-4 flex min-w-0 flex-wrap gap-1.5">
          <legend className="sr-only">Filter by kind</legend>
          {KINDS.map(([kind, label]) => (
            <button
              key={label}
              type="button"
              aria-pressed={search.kind === kind}
              onClick={() => go("/inbox", { ...search, kind })}
              className={cn(
                "inline-flex h-7 items-center gap-1.5 rounded-full border px-3 text-[13px]",
                search.kind === kind
                  ? "border-foreground bg-foreground text-background"
                  : "border-border hover:bg-muted",
              )}
            >
              {label}
              {(countFor(kind) ?? 0) > 0 && (
                <span
                  className={cn(
                    "rounded-full px-1.5 text-xs font-medium tabular-nums",
                    search.kind === kind ? "bg-background/20" : "bg-warning/15 text-warning",
                  )}
                >
                  {countFor(kind)}
                </span>
              )}
            </button>
          ))}
        </fieldset>
        <div className="mt-4 space-y-3">
          {inbox.isPending && (
            <p role="status" className="text-sm text-muted-foreground">
              Loading inbox…
            </p>
          )}
          {inbox.error && <ErrorState error={inbox.error} />}
          {all.error && <ErrorState error={all.error} />}
          {all.data && all.data.complete !== true && (
            <p role="status" className="text-sm text-muted-foreground">
              Counts unavailable:{" "}
              {all.data.unavailable_reason === "access_limited"
                ? "your access is limited"
                : all.data.unavailable_reason === "source_incomplete"
                  ? "a source is incomplete"
                  : "the server could not establish a complete total"}
              .
            </p>
          )}
          {inbox.data?.issues?.map((issue) => (
            <p
              key={`${issue.code}-${issue.session_id ?? ""}-${issue.detail}`}
              role="status"
              className="text-sm text-warning [overflow-wrap:anywhere]"
            >
              {issue.detail}
              {issue.session_id && (
                <NavLink to={sessionPath(issue.session_id)} className="ml-2 underline">
                  Inspect session {issue.session_id}
                </NavLink>
              )}
            </p>
          ))}
          {inbox.isSuccess && items.length === 0 && (
            <EmptyState
              title={
                inbox.hasNextPage
                  ? "No visible items on the pages loaded"
                  : "Nothing is waiting for you"
              }
            >
              Approvals, questions, recovery and knowledge to review show up here as sessions need
              them.
            </EmptyState>
          )}
          {items.map((item) => (
            <InboxCard
              key={item.id}
              item={item}
              context={
                item.session_id
                  ? {
                      sessionId: item.session_id,
                      title: titles.get(item.session_id) ?? item.session_id,
                    }
                  : undefined
              }
            />
          ))}
          {inbox.hasNextPage && (
            <button
              type="button"
              className="rounded border border-border px-3 py-2 text-sm"
              disabled={inbox.isFetchingNextPage}
              onClick={() => void inbox.fetchNextPage()}
            >
              {items.length === 0 ? "Continue scanning" : "Load more"}
            </button>
          )}
        </div>
        <ChildActions
          key={`${search.session_id}-${search.agent_name}`}
          sessionId={search.session_id}
          agentName={search.agent_name}
        />
      </PageBody>
    </div>
  )
}
