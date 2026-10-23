import { useSearch } from "@tanstack/react-router"
import { cn } from "@/lib/utils"
import type { InboxItem } from "../api/types"
import { useInbox, useSessionRows } from "../app/queries"
import { EmptyState, ErrorState, PageBody, TopBar, useGo } from "./common"
import { InboxCard } from "./session/inbox-card"

const KINDS: [InboxItem["kind"] | undefined, string][] = [
  [undefined, "All"],
  ["approval", "Approvals"],
  ["question", "Questions"],
  ["recovery", "Recovery"],
  ["knowledge", "Knowledge"],
]

export function InboxPage() {
  const search = useSearch({ strict: false }) as { kind?: InboxItem["kind"] }
  const go = useGo()
  const inbox = useInbox(search.kind ? { kind: search.kind } : {})
  const sessions = useSessionRows()
  const titles = new Map(
    (sessions.data?.rows ?? []).map((r) => [r.session.id, r.overview?.title ?? r.session.id]),
  )
  const items = inbox.data?.items ?? []
  // Counts per kind come from the unfiltered inbox, the same query the sidebar badge uses.
  const all = useInbox()
  const counts = new Map<string, number>()
  for (const i of all.data?.items ?? []) counts.set(i.kind, (counts.get(i.kind) ?? 0) + 1)
  const countFor = (kind: InboxItem["kind"] | undefined) =>
    kind ? (counts.get(kind) ?? 0) : (all.data?.items.length ?? 0)

  return (
    <div className="flex min-w-0 flex-1 flex-col">
      <TopBar>Inbox</TopBar>
      <PageBody width="max-w-3xl">
        <p className="text-sm text-muted-foreground">
          Everything in this agent waiting on a person. Deciding here settles the same record in
          Cayu Cloud and in the agent's own UI.
        </p>
        <fieldset className="mt-4 flex min-w-0 flex-wrap gap-1.5">
          <legend className="sr-only">Filter by kind</legend>
          {KINDS.map(([kind, label]) => (
            <button
              key={label}
              type="button"
              aria-pressed={search.kind === kind}
              onClick={() => go("/inbox", kind ? { kind } : {})}
              className={cn(
                "inline-flex h-7 items-center gap-1.5 rounded-full border px-3 text-[13px]",
                search.kind === kind
                  ? "border-foreground bg-foreground text-background"
                  : "border-border hover:bg-muted",
              )}
            >
              {label}
              {countFor(kind) > 0 && (
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
          {inbox.error && <ErrorState error={inbox.error} />}
          {inbox.isSuccess && items.length === 0 && (
            <EmptyState title="Nothing is waiting for you">
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
        </div>
      </PageBody>
    </div>
  )
}
